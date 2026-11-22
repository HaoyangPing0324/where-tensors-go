{
  const spec = {
    id: "kimi-k2.6",
    hidden: 7168,
    vocab: 163840,
    qRank: 1536,
    content: 128,
    value: 128,
    experts: 384,
    top: 8,
    middle: 2048,
    dense: 18432,
    transfer: () =>
      "每层请求 A 的 16 个提示位置：压缩内容 [16,512] 与旋转 key [16,64]；8 个目标 rank 分别续算本卡 8 个查询头。",
    profile: {
      SGLang: "官方单机 TP8；这里用双节点 Mooncake 教学拓扑",
      vLLM: "vLLM Recipes · H200 双节点 · 显式选择 TP/TP · 普通解码",
    },
    layers: [
      { ffn: "Dense", label: "第 0 层 · Dense FFN" },
      { ffn: "MoE", label: "第 1–60 层 · MLA + MoE" },
    ],
  };
  window.inferenceModels[spec.id] = { vLLM: makeTPScene(spec, "vLLM"), SGLang: makeTPScene(spec, "SGLang") };
}
