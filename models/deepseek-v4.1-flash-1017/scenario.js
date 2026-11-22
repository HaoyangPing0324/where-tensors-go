{
  const spec = {
    id: "deepseek-v4.1-flash-1017",
    hidden: 5120,
    vocab: 129280,
    qRank: 1280,
    experts: 384,
    top: 6,
    middle: 2304,
    cache: "SWA / 压缩 KV",
    streams: 4,
    activation: [
      "M=\\operatorname{SiLU}(\\min(G,10))\\odot\\operatorname{clip}(U,-10,10)",
      "Gate 限制上界，Up 双侧限幅，随后逐元素相乘。",
    ],
    profile: {
      SGLang: "SGLang PD · 对称 TP8 教学分片 · 1017 参考前向",
      vLLM: "H200 双端 TP8、NIXL 配方 · 图按 1017 公开参考前向展开",
    },
    transfer: () =>
      "各层滑窗 [16,512]；层 2、8、14 的压缩源 [8,512] 和层 20 的源 [16,512]，以及索引键和续算状态（列逻辑形状）。",
    beforeAttention({ add, rows, shape, T, H, X }) {
      add(
        "Attention",
        "查询 Engram 记忆表",
        "n-gram lookup",
        () => rows(["历史 token", "最长 4-gram"], ["哈希头", "8 × 256"], ["门控注入", "四路残差"]),
        { when: (g) => g.engram },
      );
      add("Attention", "四路残差与 mHC 混合", "mHC", () =>
        rows(
          ["残差流", shape(T, 4, H)],
          ["混合矩阵", "4 × 4", "20 次 Sinkhorn 迭代产生混合系数。"],
          ["子层输入", X],
        ),
      );
    },
    attention({ add, projection, rows, shape, T, H, X }) {
      projection(
        "Attention",
        "Q 低秩投影",
        ["归一化输入", X],
        ["Wqₐ", shape(H, 1280)],
        ["Q latent", shape(T, 1280)],
        ["Q_a=XW_{q,a}", "A 投影在每个 TP rank 可用。"],
      );
      const dim = 512;
      add(
        "Attention",
        "展开本卡查询头",
        "GEMM / RoPE",
        (r) =>
          rows(
            ["Q latent", shape(T, 1280)],
            ["Wqᵦ", shape(1280, 8 * dim)],
            ["Q · 头 " + 8 * r + "–" + (8 * r + 7), shape(T, 8, dim)],
          ),
        { formula: ["Q_r=\\operatorname{RMSNorm}(Q_a)W_{q,b}^{(r)}", "64 个查询头均分到 8 卡。"] },
      );
      add("Attention", "滑动窗口与本地缓存", "SWA", () =>
        rows(
          ["当前位置", shape(T, H)],
          ["Wkv", shape(H, 512), "每个位置经归一化、末 64 维 RoPE，写入长度 128 的环形缓存。"],
          ["本步窗口 KV", shape(T, 512)],
        ),
      );
      add(
        "Attention",
        "构造或复用压缩 KV",
        "CSA2",
        (r, g) =>
          rows(
            ["KV 来源", g.mode === "full" ? (g.decoder ? "层19输出 → 层20" : "本层输入") : "最近源层 KV"],
            ["压缩倍率", g.decoder ? "1" : "2"],
            ["主 KV", g.mode === "full" ? "写入缓存" : "复用源层缓存"],
          ),
        { when: (g) => g.mode !== "window" },
      );
      add(
        "Attention",
        "分层稀疏索引",
        "CSA2",
        (r, g) =>
          rows(
            [
              "候选范围",
              g.decoder ? (g.mode === "full" ? "本层建立候选块" : "层20候选块内") : "可见压缩位置",
            ],
            ["候选位置", g.mode === "reuse" ? "复用 Top-k" : "选择 Top-512"],
            ["稀疏分支", "与滑窗共同参与"],
          ),
        { when: (g) => g.mode !== "window" },
      );
      add(
        "Attention",
        "读取可见的 KV 并聚合",
        "Softmax / AV",
        (r, g) =>
          rows(
            ["本卡查询", shape(T, 8, dim)],
            ["有效历史", g.mode === "window" ? "滑窗，含 sink" : "滑窗 + 压缩候选，含 sink"],
            ["每头输出", shape(T, 8, 512)],
          ),
        { formula: ["O=\\operatorname{softmax}(S)V", "按可见位置归一化；展示逻辑运算。"] },
      );
      add("Attention", "逆旋转输出的位置通道", "Inverse RoPE", () =>
        rows(["注意力输出", shape(T, 8, 512)], ["末 64 通道", "逆向 RoPE"], ["分组输入", shape(T, 4096)]),
      );
      projection(
        "Attention",
        "分组低秩输出投影",
        ["本卡 8 头", shape(T, 4096)],
        ["组投影", shape(4096, 1024)],
        ["组内结果", shape(T, 1024)],
        ["Z_r=O_rW_{o,a}^{(r)}", "八个输出组，每卡一组。"],
      );
      const width = 1024;
      projection(
        "Attention",
        "输出投影与本卡部分积",
        ["本地输出", shape(T, width)],
        ["Wₒ 分片", shape(width, H)],
        ["部分积", X],
        ["Y_r=Z_rW_o^{(r)}", "沿投影的输入维切分。"],
      );
    },
    layers: [
      { label: "0 · 编码器 SWA", mode: "window" },
      { label: "1 · 编码器 SWA + Engram", mode: "window", engram: true },
      { label: "2、8 · 编码器 CSA2 Full", mode: "full" },
      { label: "14 · 编码器 CSA2 Full + Engram", mode: "full", engram: true },
      { label: "3–7、9–13、15–19 · 编码器 Reuse", mode: "reuse" },
      { label: "20 · 解码器 CSA2 Full", mode: "full", decoder: true },
      { label: "24、28、32、36 · 解码器 Reindex", mode: "reindex", decoder: true },
      { label: "21–39 其余解码器层 · Reuse", mode: "reuse", decoder: true },
    ].map((group) => ({ ...group, ffn: "MoE" })),
  };
  window.inferenceModels[spec.id] = { vLLM: makeTPScene(spec, "vLLM"), SGLang: makeTPScene(spec, "SGLang") };
}
