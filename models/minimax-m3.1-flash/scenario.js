{
  const spec = {
    id: "minimax-m3.1-flash",
    hidden: 6144,
    vocab: 200064,
    dense: 12288,
    middle: 3072,
    experts: 128,
    top: 4,
    norm: [
      "\\hat x_i=(1+w_i)x_i/\\sqrt{H^{-1}\\sum_j x_j^2+\\epsilon}",
      "Gemma 风格的 RMSNorm 使用 1+w 作为缩放系数。",
    ],
    activation: ["M=g\\,\\sigma(1.702g)\\odot(u+1)", "g 为上限 7 的 gate，u 为裁剪至 [−7,7] 的 up。"],
    cache: "GQA KV / 索引 K",
    schemeLabel: "解码方式",
    schemes: [
      { id: "tp", label: "逐 token 解码" },
      { id: "draft", label: "DSpark 推测解码" },
    ],
    profile: {
      SGLang: "官方支持 PD 与 DSpark；图示对称 TP8 教学分片",
      vLLM: "vLLM Recipes · 2 × 8 B300 · TP/TP、NIXL；DSpark 为额外教学分支",
    },
    transfer: (r) =>
      "K、V 各 [16,1,128]，本卡 KV 头 " +
      Math.floor(r / 2) +
      "（每两卡复制一头），并交接本层 MSA 索引 K。草稿另需对应上下文。",
    layers: [
      { label: "0–2 · MSA + Dense FFN", ffn: "Dense" },
      { label: "3–59 · MSA + MoE", ffn: "MoE" },
    ],
    attention({ add, rows, shape, T, H, X }) {
      add("Attention", "分片投影 Q 与复制 KV 头", "GQA / GEMM", (r) =>
        rows(
          ["归一化输入", X],
          [
            "Wq / Wk / Wv",
            "1024 | 128 | 128",
            "三个矩阵输入维均为 6144。查询头 " +
              8 * r +
              "–" +
              (8 * r + 7) +
              "；KV 头 " +
              Math.floor(r / 2) +
              "。",
          ],
          ["本地 Q / K / V", "8 / 1 / 1 头", "每头 128 维，同一组内两卡持有相同 KV 头。"],
        ),
      );
      add("Attention", "归一化 QK 并追加新 KV", "RoPE", (r) =>
        rows(
          ["头内向量", shape(T, 128)],
          ["旋转通道", "前 64 / 128"],
          [
            "KV 头 " + Math.floor(r / 2),
            shape(T, 1, 128),
            "K 和 V 各一份；Decode 增加当前位置，旧位置留在缓存。",
          ],
        ),
      );
      add("Attention", "生成 MSA 索引键与查询", "Indexer", () =>
        rows(
          ["当前状态", X],
          ["全组索引头", "4 × 128"],
          ["索引缓存", "K only", "每层单独保留索引键，和模型注意力的 K/V 分开。"],
        ),
      );
      add("Attention", "为历史块打分并选择地址", "MSA Top-k", () =>
        rows(
          ["历史 KV 页", "128 位置 / 块"],
          ["块评分", "块内最大值"],
          ["访问列表", "至多 16 个块", "另遵循因果与局部块规则。当前位置仅见一个块。"],
        ),
      );
      add(
        "Attention",
        "读取选中块并计算 GQA",
        "Softmax / AV",
        () =>
          rows(
            ["本卡 Q", shape(T, 8, 128)],
            ["可见 K / V", shape("L", 1, 128)],
            ["本地头输出", shape(T, 8, 128)],
          ),
        {
          formula: [
            "O_h=\\operatorname{softmax}(Q_h K_{I}^T/\\sqrt{128}+M)V_{I}",
            "I 是选中块里的可见位置。8 个本地查询头读取同一 KV 头。",
          ],
        },
      );
      add("Attention", "行并行输出投影", "GEMM", () =>
        rows(["头输出", shape(T, 1024)], ["Wₒ", shape(1024, H)], ["部分积", X]),
      );
    },
    decodeTail({ add, rows, shape }) {
      const branch = { scheme: "draft" };
      add(
        "推测解码",
        "读出主模型五层特征",
        "DSpark",
        () =>
          rows(
            ["目标层", "3、17、31、45、59"],
            ["单位置拼接", shape(1, 5 * 6144)],
            ["草稿条件", "投影到 6144 维"],
          ),
        branch,
      );
      add(
        "推测解码",
        "从锚点提出下一块候选",
        "Block draft",
        () =>
          rows(
            ["锚点与掩码", "训练块宽 7"],
            ["草稿主干", "5 层 Dense"],
            ["候选与置信度", "待主模型验证", "共享目标模型的 embedding 与词表头，以 Markov 头建模块内依赖。"],
          ),
        branch,
      );
      add(
        "推测解码",
        "用主模型验证并提交前缀",
        "Verify",
        () =>
          rows(
            ["提议的 token", "k 个候选"],
            ["目标模型 logits", shape("k+1", 200064)],
            ["已提交 token", "接受前缀与校正 ID", "未接受的临时缓存槽不提交；接受长度依赖本轮验证。"],
          ),
        branch,
      );
    },
  };
  window.inferenceModels[spec.id] = { SGLang: makeTPScene(spec, "SGLang"), vLLM: makeTPScene(spec, "vLLM") };
}
