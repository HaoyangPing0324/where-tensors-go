{
  const spec = {
    id: "hy4-flash",
    hidden: 3072,
    vocab: 120832,
    dense: 13824,
    middle: 1536,
    latent: 1536,
    experts: 512,
    top: 16,
    streams: 4,
    cache: "MLA / 滑窗 / 索引",
    profile: {
      SGLang: "官方 Cookbook 提供 PD Playground；此图为双端 TP8、普通解码",
      vLLM: "GPU TP8/TP8 教学对照。官方 Ascend 1P1D 示例使用不同并行度",
    },
    transfer: () =>
      "稀疏层 MLA 内容 [16,512]、位置键 [16,64] 与 IndexPool 状态；滑窗层保存最近至多 512 行。请求携带用于 n-gram 的历史 ID。本例关闭 DFly2。",
    layers: [
      { ffn: "Dense", sparse: true, label: "0 · IndexPool · Dense" },
      { ffn: "MoE", sparse: true, engram: true, label: "1、21 · IndexPool · Engram · MoE" },
      { ffn: "MoE", sparse: true, label: "5、9、13、17、25、29、33、37、41 · IndexPool · MoE" },
      { ffn: "MoE", engram: true, label: "11、31 · 滑动窗口 · Engram · MoE" },
      { ffn: "MoE", label: "其余 28 层 · 滑动窗口 · MoE" },
    ],
    beforeAttention({ add, rows, T, X, shape }) {
      add(
        "Attention",
        "用相邻 token 查 Engram",
        "N-gram",
        () =>
          rows(
            ["连续 ID", "2-gram / 3-gram"],
            ["分头查表", "每阶 8 × 96"],
            ["拼接检索向量", shape(T, 1536), "两阶各贡献 768 维，经门控注入残差流。"],
          ),
        { when: (g) => g.engram },
      );
      add("Attention", "从四路 iHC 残差读取", "iHC", () =>
        rows(["四路残差", shape(T, 4, 3072)], ["分支读权重", "4 路"], ["注意力输入", X]),
      );
    },
    attention({ add, rows, shape, T, H, X }) {
      const heads = (g) => (g.sparse ? 4 : 8);
      add(
        "Attention",
        "计算低秩查询",
        "MLA",
        () => rows(["归一化输入", X], ["Wqₐ", shape(H, 1024)], ["Q latent", shape(T, 1024)]),
        { formula: ["Q_a=\\operatorname{RMSNorm}(XW_{q,a})", "查询中间维 1024。"] },
      );
      add("Attention", "展开查询并施加旋转编码", "GEMM / RoPE", (r, g) =>
        rows(
          ["Q latent", shape(T, 1024)],
          ["Wqᵦ", shape(1024, heads(g) * 256)],
          [
            "Q · 头 " + r * heads(g) + "–" + ((r + 1) * heads(g) - 1),
            shape(T, heads(g), 256),
            "每头 192 个内容通道与 64 个位置通道。",
          ],
        ),
      );
      add("Attention", "构造潜空间键值", "MLA cache", () =>
        rows(["输入", X], ["Wkvₐ", shape(H, 576)], ["内容 / 位置", "512 + 64"]),
      );
      add(
        "Attention",
        "按组筛选历史位置",
        "IndexPool",
        () =>
          rows(
            ["索引头", "16 × 128 全组"],
            ["压缩分组", "4 个位置 / 组"],
            ["候选地址", "≤ 512 组 × 4", "最多 2048 个可见位置；16 个提示位置最多占 4 组。"],
          ),
        { when: (g) => g.sparse },
      );
      add(
        "Attention",
        "截取本层滑动窗口",
        "SWA",
        () =>
          rows(
            ["当前位置", T === 1 ? "位置 16" : "位置 0–15"],
            ["因果窗口", "511 历史 + 当前"],
            ["访问 KV 行", T === 1 ? "17 行" : "1–16 行"],
          ),
        { when: (g) => !g.sparse },
      );
      add("Attention", "计算分数与注意力输出", "Gated MLA", (r, g) =>
        rows(
          ["本卡查询", shape(T, heads(g), 256)],
          ["可见 KV", g.sparse ? "IndexPool 候选" : "最近 512 行"],
          ["门控头输出", shape(T, heads(g), 256), "在候选上计算有 sink 的注意力，经 value 投影和门控输出。"],
        ),
      );
      add("Attention", "投影回残差空间", "Row parallel", (r, g) =>
        rows(["展平本卡头", shape(T, heads(g) * 256)], ["Wₒ", shape(heads(g) * 256, H)], ["部分积", X]),
      );
    },
  };
  window.inferenceModels[spec.id] = { SGLang: makeTPScene(spec, "SGLang"), vLLM: makeTPScene(spec, "vLLM") };
}
