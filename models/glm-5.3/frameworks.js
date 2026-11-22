{
  const original = window.inferenceModels["glm-5.3"];
  const spec = {
    id: original.id,
    hidden: 6144,
    vocab: 154880,
    qRank: 2048,
    content: 192,
    value: 256,
    experts: 256,
    top: 8,
    middle: 2048,
    dense: 12288,
    layers: original.layerClasses,
    index: true,
  };
  const vllm = makeTPScene(spec, "vLLM");
  original.moe = vllm.moe;
  original.transfer = (r, g, l) =>
    l === "cp"
      ? "P" +
        r +
        " 对应交错位置 " +
        r +
        "、" +
        (r + 8) +
        " 的 MLA 内容 [2,512] 与位置键 [2,64]，连同索引键汇入 D0；示意逻辑来源。"
      : "TP 注意力各 rank 可访问相同的压缩内容 [16,512] 和位置键 [16,64]。D0 接管本请求；冗余副本的实际发送 rank 由后端决定。";
  original.bridge = "NIXL / UCX · InfiniBand";
  original.profile = "Dynamo + SGLang · H200 · Prefill EP8 / Decode EP1";
  vllm.profile = "vLLM Recipes · GLM-5.3 / 2 × 8 H200 · 双端 TP8、NIXL";
  window.inferenceModels[spec.id] = { SGLang: original, vLLM: vllm };
}
