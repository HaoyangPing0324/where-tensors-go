{
  const spec = {
    id: "qwen3.8-flash-next",
    hidden: 2560,
    vocab: 248320,
    middle: 640,
    experts: 512,
    top: 10,
    streams: 4,
    cache: "GDN / QSA / PLE",
    sharedGate: true,
    profile: {
      SGLang: "SGLang 已实现混合状态 PD 交接；此图采用双端 TP8 教学拓扑",
      vLLM: "vLLM 有 QSA/GDN/PLE 实现与 Ascend PD 示例；GPU 对称 TP8 为教学拓扑",
    },
    transfer: (r, g) =>
      (g.qsa
        ? "所选 QSA 层交接位置 KV 和索引。"
        : g.ple
          ? "所选层还需 PLE 卷积状态。"
          : "所选 GDN 层交接递归与卷积状态。") +
      "36 个 GDN 层分别交接本卡递归状态 [6,128,128] 和短卷积历史。12 个 QSA 层交接 KV 头 " +
      Math.floor(r / 4) +
      " 的 K、V（各 [16,1,256]）与索引状态。第 2 层另需 PLE 卷积状态和 n-gram 历史。",
    layers: [
      { label: "0、1、4–6…44–46（除第 2 层）· Gated DeltaNet", ffn: "MoE" },
      { label: "2 · Gated DeltaNet + n-gram PLE", ple: true, ffn: "MoE" },
      { label: "3、7、11…47 · Qwen Sparse Attention", qsa: true, ffn: "MoE" },
    ],
    beforeAttention({ add, rows, shape, T, X }) {
      add("Attention", "从四路残差读出隐藏态", "Gated Residual", () =>
        rows(["分支状态", shape(T, 4, 2560)], ["低秩读门", "瓶颈 320"], ["逐通道混合", X]),
      );
      add(
        "Attention",
        "读取上下文 n-gram",
        "PLE",
        () =>
          rows(
            ["短历史", "二元 / 三元组"],
            ["哈希表", "每阶 8 头"],
            ["检索特征", shape(T, 2560), "结合当前状态与宽 4 的短卷积。Decode 保留卷积历史。"],
          ),
        { when: (g) => g.ple },
      );
    },
    attention({ add, rows, shape, T, H, X, P }) {
      const gdn = { when: (g) => !g.qsa },
        qsa = { when: (g) => g.qsa };
      add(
        "Attention",
        "分片投影线性注意力头",
        "GDN QKV",
        (r) =>
          rows(
            ["隐藏态", X],
            [
              "Q / K / V 头",
              "2 / 2 / 6",
              "Q、K 使用头 " +
                2 * r +
                "–" +
                (2 * r + 1) +
                "，V 使用头 " +
                6 * r +
                "–" +
                (6 * r + 5) +
                "；每头 128 维。",
            ],
            ["本地打包", shape(T, 1280)],
          ),
        gdn,
      );
      add(
        "Attention",
        "更新短卷积与门控",
        "Causal Conv1d",
        () =>
          rows(["打包投影", shape(T, 1280)], ["卷积核宽", "4 个位置"], ["局部 Q / K / V", "及衰减、更新门"]),
        gdn,
      );
      add(
        "Attention",
        P ? "逐位置扫描递归状态" : "续写上一位置递归状态",
        "Gated DeltaNet",
        () =>
          rows(
            ["上一状态", shape(6, 128, 128)],
            ["新 K / V", "每个 V 头 128 维"],
            ["更新状态", shape(6, 128, 128)],
          ),
        {
          ...gdn,
          formula: [
            "\\bar S_t=\\alpha_t S_{t-1},\\quad S_t=\\bar S_t+\\beta_t k_t(v_t^T-k_t^T\\bar S_t)",
            "单头等价递推。Prefill 可并行扫描，Decode 只推进一个位置。",
          ],
        },
      );
      add(
        "Attention",
        "读出并门控本地 value 头",
        "Gated readout",
        () => rows(["新状态", shape(6, 128, 128)], ["查询读出", "qᵀ × S"], ["本地头输出", shape(T, 6, 128)]),
        gdn,
      );
      add(
        "Attention",
        "生成 QSA 查询与分组 KV",
        "GQA / RoPE",
        (r) =>
          rows(
            ["输入", X],
            [
              "Q / KV 分片",
              "3 Q / 1 KV",
              "Q 头 " +
                3 * r +
                "–" +
                (3 * r + 2) +
                "，KV 头 " +
                Math.floor(r / 4) +
                " 由相邻 4 卡复制。每头旋转 64 个通道。",
            ],
            ["本卡查询", shape(T, 3, 256)],
          ),
        qsa,
      );
      add(
        "Attention",
        "写入 KV 与压缩索引键",
        "QSA cache",
        () =>
          rows(
            ["本卡 K / V", shape(T, 1, 256)],
            ["索引 K", "全组 1 头 × 128"],
            ["压缩索引", "每 4 个位置一组"],
          ),
        qsa,
      );
      add(
        "Attention",
        "选择因果可见微块",
        "QSA indexer",
        () =>
          rows(
            ["全组索引 Q", "4 头 × 128"],
            ["预算", "512 块 × 4 位置"],
            ["候选行", "含未封口块尾", "16 个提示位置都在预算内。"],
          ),
        qsa,
      );
      add(
        "Attention",
        "稀疏聚合与输出门",
        "Gated attention",
        () =>
          rows(
            ["查询", shape(T, 3, 256)],
            ["选中 K / V", shape("L", 1, 256)],
            ["每头输出", shape(T, 3, 256)],
          ),
        qsa,
      );
      add("Attention", "投影本卡头输出", "Row parallel", (r, g) =>
        rows(
          ["展平本卡头", shape(T, 768), g.qsa ? "3 × 256 通道" : "6 × 128 通道"],
          ["Wₒ", shape(768, H)],
          ["部分积", X],
        ),
      );
    },
  };
  window.inferenceModels[spec.id] = { SGLang: makeTPScene(spec, "SGLang"), vLLM: makeTPScene(spec, "vLLM") };
}
