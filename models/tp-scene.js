// TP-only, one P and one D worker. Shapes below describe logical GEMMs, independent of fusion.
function makeTPScene(m, engine) {
  const v = engine === "vLLM",
    H = m.hidden,
    events = [],
    shape = (...dims) => "[" + dims.join(",") + "]";
  const bridge = m.bridge?.[engine] || (v ? "NixlConnector · producer → consumer" : "Mooncake · KV transfer");
  const rows = (...entries) =>
    entries.map(([name, value, detail]) => ({ name, value, detail: detail || value }));
  const service = (phase, section, title, from, via, to) =>
    events.push({
      path: [phase, section],
      title,
      op: via,
      svc: { from, via, to },
      phase: phase === "Decode" ? 17 : -1,
    });
  const data = (phase, path, title, op, blocks, options = {}) =>
    events.push({ phase, path, title, op, blocks, ...options });
  service(
    "Prefill",
    "请求接入",
    "用户发送流式聊天请求",
    "客户端",
    "POST /v1/chat/completions",
    v ? "vllm-router" : "SGLang Model Gateway",
  );
  service(
    "Prefill",
    "请求接入",
    "分配 Prefill 请求",
    v ? "vllm-router" : "SGLang 路由器",
    v ? "round_robin" : "PD 路由",
    "Prefill worker",
  );
  service(
    "Prefill",
    "请求接入",
    "模板、分词与缓存分配",
    "文本消息",
    v ? "vLLM frontend / V1 scheduler" : "TokenizerManager / Scheduler",
    "16 个提示位置、KV 块表",
  );
  for (const side of ["Prefill", "Decode"]) {
    const P = side === "Prefill",
      phase = P ? 2 : 15,
      T = P ? 16 : 1,
      X = shape(T, H);
    const add = (section, title, op, blocks, options = {}) =>
      data(phase, [side, section, ...(options.ffn ? [options.ffn] : [])], title, op, blocks, options);
    const projection = (section, title, input, weight, output, formula, extra = {}) =>
      add(section, title, "GEMM", () => rows(input, weight, output), { formula, ...extra });
    const reduce = (section, title, ffn) =>
      add(section, title, "AllReduce · SUM", () => [], {
        collective: true,
        ffn,
        formula: ["Y=\\sum_{r=0}^{7}Y^{(r)}", "按元素合并八份 TP 部分积。"],
      });
    add("Embedding", "词表分片查表", "Lookup", (r) =>
      rows(
        ["输入 ID", shape(T)],
        [
          "词表行",
          (r * m.vocab) / 8 + "…" + (((r + 1) * m.vocab) / 8 - 1),
          "教学 ID 按位置 j 选择词表分片 j mod 8；Decode 的示例 ID 属于分片 0。",
        ],
        ["局部词向量", X, "命中本卡词表时读取对应行，其余位置贡献零。"],
      ),
    );
    reduce("Embedding", "合并词向量");
    const context = { add, projection, reduce, rows, shape, H, T, X, P };
    if (m.streams)
      add("Embedding", "展开多路残差", "Expand", () =>
        rows(
          ["词向量", X],
          ["复制到各分支", String(m.streams) + " 路"],
          ["层 0 残差", shape(T, m.streams, H)],
        ),
      );
    m.beforeAttention?.(context);
    add(
      "Attention",
      "输入归一化",
      "RMSNorm",
      () => rows(["输入 X", X], ["通道尺度", shape(H)], ["归一化行", X]),
      {
        formula: m.norm || [
          String.raw`\widehat x_i=\gamma_i x_i/\sqrt{H^{-1}\sum_j x_j^2+\epsilon}`,
          "沿隐藏通道归一化。",
        ],
      },
    );
    if (m.attention) m.attention(context);
    else buildMLAAttention(m, context);
    reduce("Attention", "归并输出投影");
    add("Attention", "写回层间残差", m.streams ? "多路残差" : "Residual", () =>
      rows(
        ["部分积已归并", X],
        ["残差通路", m.streams ? shape(T, m.streams, H) : X],
        ["下一子层残差", m.streams ? shape(T, m.streams, H) : X],
      ),
    );
    add("FFN", "读出残差与前馈归一化", "RMSNorm", () =>
      rows(["残差", m.streams ? shape(T, m.streams, H) : X], ["归一化尺度", shape(H)], ["前馈输入", X]),
    );
    for (const ffn of m.dense ? ["Dense", "MoE"] : ["MoE"]) {
      const moe = ffn === "MoE",
        W = (moe ? m.middle : m.dense) / 8,
        N = moe ? "mₑ" : T,
        E = moe ? m.latent || H : H;
      if (moe) {
        const example = Array.from({ length: m.top }, (_, k) => (7 + 13 * k) % m.experts);
        add(
          "FFN",
          "专家路由与行打包",
          "Top-k",
          () =>
            rows(
              ["输入行", X],
              ["专家分数", shape(T, m.experts)],
              [
                "命中专家",
                m.top + " 路 + 1 共享",
                "示例 token A₀ 命中 " +
                  example.map((e) => "E" + e).join("、") +
                  "。教学 ID；各卡均持有这些专家的 TP 分片，按专家收集命中行。",
              ],
            ),
          { ffn },
        );
      }
      if (moe && m.latent)
        add(
          "FFN",
          "投影到路由专家空间",
          "LatentMoE",
          () => rows(["完整输入", X], ["下降投影", shape(H, E)], ["路由空间", shape(T, E)]),
          { ffn },
        );
      add(
        "FFN",
        "Gate / Up 与门控激活",
        "SwiGLU",
        (r) =>
          rows(
            ["输入", shape(N, E)],
            ["Wgate / Wup", shape(E, W), "中间维区间 [" + r * W + "," + (r + 1) * W + ")"],
            ["本地激活", shape(N, W)],
          ),
        {
          ffn,
          formula: m.activation || [
            "M_r=\\operatorname{SiLU}(XW_{g,r})\\odot XW_{u,r}",
            "两组权重切相同的中间通道。",
          ],
        },
      );
      add(
        "FFN",
        "Down 投影",
        "GEMM",
        () => rows(["本地激活", shape(N, W)], ["Wdown", shape(W, E)], ["局部贡献", shape(N, E)]),
        { ffn },
      );
      if (moe) {
        add(
          "FFN",
          "加权还原 token 顺序",
          "Combine",
          () =>
            rows(
              ["专家输出", shape(N, E)],
              ["还原行号 / 系数", shape(T, m.top)],
              ["各 token 部分和", shape(T, E)],
            ),
          { ffn, formula: ["R_t=\\sum_{e\\in I_t}p_{t,e}F_e(x_t)", "按路由权重合并同一行的专家贡献。"] },
        );
        if (m.latent)
          add(
            "FFN",
            "投回主干隐藏维",
            "LatentMoE",
            () => rows(["低维部分和", shape(T, E)], ["上升投影", shape(E, H)], ["主干部分和", X]),
            { ffn },
          );
        add(
          "FFN",
          "计算共享专家",
          "Shared FFN",
          () =>
            rows(
              ["主干输入", X],
              ["Gate / Up 切片", shape(H, (m.sharedMiddle || m.middle) / 8)],
              [
                "Down 局部贡献",
                X,
                m.sharedGate ? "共享专家另乘输入产生的 sigmoid 标量门。" : "共享分支独立于路由 Top-k。",
              ],
            ),
          { ffn },
        );
      }
      reduce("FFN", moe ? "归并专家 TP 部分和" : "归并 Dense 部分积", ffn);
      add(
        "FFN",
        "写回前馈残差",
        "Residual",
        () =>
          rows(
            ["完整分支结果", X],
            ["残差通路", m.streams ? shape(T, m.streams, H) : X],
            ["下一层残差", m.streams ? shape(T, m.streams, H) : X],
          ),
        { ffn },
      );
    }
    add("输出", "取末位置并投影到词表", "LM head", (r) =>
      rows(
        ["末位置 · RMSNorm", shape(1, H)],
        [
          "Whead",
          shape(H, m.vocab / 8),
          "词表行区间 [" + (r * m.vocab) / 8 + "," + ((r + 1) * m.vocab) / 8 + ")",
        ],
        ["局部 logits", shape(1, m.vocab / 8)],
      ),
    );
    add("输出", "汇集词表分片", "AllGather · logits", () => [], { collective: true, vocab: true });
    add("输出", "从完整词表采样", "Sample", () =>
      rows(
        ["完整 logits", shape(1, m.vocab)],
        ["采样分布", "归一化词表分数"],
        ["本轮输出", "1 个 token ID", "各 rank 逻辑上共用同一采样结果。"],
      ),
    );
    if (!P) m.decodeTail?.(context);
    if (P) {
      service(
        "Prefill",
        "KV 交接",
        "匹配缓存源与请求",
        v ? "kv_producer" : "Prefill worker",
        v ? "KV transfer metadata" : "Bootstrap",
        "Decode 目标缓存块",
      );
      data(13, ["Prefill", "KV 交接"], "搬运请求缓存", bridge, (r, g, side) =>
        rows(
          ["请求 A", "TP rank " + r],
          ["缓存对象", m.cache || "MLA KV" + (m.index ? " / 索引" : "")],
          [side === "P" ? "源缓存块" : "目标缓存块", side === "P" ? "可读" : "装入本地"],
        ),
      );
      service(
        "Decode",
        "流式输出",
        "接续首个 token",
        "首个 token 与缓存映射",
        v ? "kv_consumer" : "Decode worker",
        "下一生成位置",
      );
      service("Decode", "流式输出", "返回首个文本增量", "首个生成 ID", "增量解码与 SSE", "客户端输出缓冲");
    }
  }
  service(
    "Decode",
    "返回与清理",
    "增量解码并流式回传",
    "已提交 token ID",
    "Detokenize / SSE",
    "客户端收到文本增量",
  );
  service(
    "Decode",
    "返回与清理",
    "结束输出并释放请求状态",
    "达到示例输出上限",
    "finish_reason · [DONE]",
    "释放缓存引用与请求槽",
  );
  return {
    id: m.id + "-" + engine,
    ranks: 8,
    hidden: m.hidden,
    sharedDecode: true,
    bridge,
    layerClasses: m.layers,
    transfer:
      m.transfer || (() => "MLA latent [16,512]、RoPE key [16,64]" + (m.index ? "，另含 DSA 索引缓存" : "")),
    profile:
      m.profile?.[engine] ||
      (v
        ? "vLLM Recipes · 双 H200 节点 · TP/TP · 普通文本解码"
        : "SGLang · TP8/TP8 教学配置 · Mooncake · 普通文本解码"),
    parallel: {
      hardware: "2 台机器 · 各 8 张卡",
      schemes: m.schemes || [{ id: "tp", label: "1P1D · 双端 TP8" }],
      columns: ["节点", "TP", "DP", "PP", "CP", "注意力 TP", "EP", "单专家 TP"],
      rows: () => ["Prefill", "Decode"].map((p) => [p, 8, 1, 1, 1, 8, 1, 8]),
    },
    stages: events.map((e) => [e.path[1], e.title, e.op, ""]),
    path: (i) => events[i]?.path,
    tocPath: (i) => {
      const e = events[i];
      return [
        e.path[1] === "请求接入" ? "请求准备" : e.path[1] === "KV 交接" ? "PD 交接" : e.path[0],
        e.path.slice(1).join(" · "),
      ];
    },
    chunk: (i) => (events[i].vocab ? { width: m.vocab / 8, axis: "词表", rows: 1 } : null),
    phase: (i) => events[i].phase,
    service: (i) => events[i].svc,
    op: (i) => (events[i].collective ? events[i].op : ""),
    layerAllows: (i, g, l) =>
      !!events[i] &&
      (!events[i].scheme || events[i].scheme === l) &&
      (!events[i].ffn || events[i].ffn === g.ffn) &&
      (!events[i].when || events[i].when(g)),
    layerPresentation: () => null,
    describe(i, r, side, n, l, g) {
      const blocks = events[i].blocks?.(r, g, side) || [];
      const weight = blocks[1];
      if (weight && /^Wₒ|^Wdown/.test(weight.name)) {
        const inputWidth = Number(weight.value.match(/^\[(\d+),/)?.[1]);
        if (inputWidth)
          weight.detail += "。本卡持有输入通道 [" + r * inputWidth + "," + (r + 1) * inputWidth + ")。";
      }
      return blocks;
    },
    equation: (i) => events[i].formula,
    weightView(i, side, r, l, active, g) {
      const e = events[i];
      if (active && e.title === "词表分片查表") return ["Embedding 分片", shape(m.vocab / 8, H)];
      if (active && !m.attention && e.title.includes("展开本卡"))
        return ["Wqᵦ · 8 头", shape(m.qRank, 8 * (m.content + 64))];
      if (active && e.blocks) {
        const b = e.blocks(r, g, side)?.[1];
        if (b && /^(W|上升投影|下降投影|Gate \/ Up 切片|归一化尺度)/.test(b.name)) return [b.name, b.value];
      }
      return ["词表嵌入分片", shape(m.vocab / 8, H)];
    },
  };
}

function buildMLAAttention(m, { add, projection, rows, shape, T, H, X }) {
  projection(
    "Attention",
    "Q 低秩投影",
    ["归一化输入", X],
    ["Wqₐ", shape(H, m.qRank)],
    ["Q latent", shape(T, m.qRank)],
    ["Q_a=XW_{q,a}", "A 投影在每个 TP rank 可用。"],
  );
  const dim = m.content + 64;
  add(
    "Attention",
    "展开本卡查询头",
    "GEMM / RoPE",
    (r) =>
      rows(
        ["Q latent", shape(T, m.qRank)],
        ["Wqᵦ", shape(m.qRank, 8 * dim)],
        ["Q · 头 " + 8 * r + "–" + (8 * r + 7), shape(T, 8, dim)],
      ),
    { formula: ["Q_r=\\operatorname{RMSNorm}(Q_a)W_{q,b}^{(r)}", "64 个查询头均分到 8 卡。"] },
  );
  projection(
    "Attention",
    "写入 MLA 压缩缓存",
    ["隐藏行", X],
    ["Wkvₐ", shape(H, 576)],
    ["latent + RoPE", "512 + 64 / 行"],
    ["[C,K_p]=XW_{kv,a}", "对内容归一化，对位置分量应用 RoPE，然后写入缓存。"],
  );
  add("Attention", "拆分查询并旋转位置通道", "RoPE", () =>
    rows(["查询", shape(T, 8, dim)], ["内容 / 位置", m.content + " + 64"], ["旋转后的位置", shape(T, 8, 64)]),
  );
  if (m.index)
    add("Attention", "DSA 候选位置", "Indexer", (r, g) =>
      rows(
        ["索引头", "32 × 128"],
        ["候选地址", g.index === "full" ? "本层计算" : "读取前层"],
        ["因果 Top-k", "至多 2048"],
      ),
    );
  add("Attention", "把内容查询投影到 KV 潜空间", "MLA", () =>
    rows(
      ["查询内容", shape(T, 8, m.content)],
      ["每头 Wkc", shape(m.content, 512)],
      ["潜空间查询", shape(T, 8, 512)],
    ),
  );
  add(
    "Attention",
    "计算可见位置的分数",
    "QK",
    () => rows(["内容 / 位置查询", "512 + 64"], ["缓存行", "512 + 64"], ["每头分数", shape(T, 8, "L")]),
    {
      formula: [
        "s_{hi}=q^c_h c_i^T+q^R_h(k_i^R)^T",
        "L 为本查询可见的历史位置数，缩放与因果掩码在归一化前应用。",
      ],
    },
  );
  add("Attention", "在潜空间聚合缓存行", "Softmax / AV", () =>
    rows(["每头分数", shape(T, 8, "L")], ["缓存内容", shape("L", 512)], ["加权 latent", shape(T, 8, 512)]),
  );
  add("Attention", "恢复各头 value", "GEMM", () =>
    rows(
      ["加权 latent", shape(T, 8, 512)],
      ["每头 Wv", shape(512, m.value)],
      ["头输出", shape(T, 8, m.value)],
    ),
  );
  const width = 8 * m.value;
  projection(
    "Attention",
    "输出投影与本卡部分积",
    ["本地输出", shape(T, width)],
    ["Wₒ 分片", shape(width, H)],
    ["部分积", X],
    ["Y_r=Z_rW_o^{(r)}", "沿投影的输入维切分。"],
  );
}
