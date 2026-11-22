const catalog = [
  {
    id: "glm-5.3",
    name: "GLM-5.2 / GLM-5.3",
    defaultEngine: "SGLang",
    scripts: ["models/glm-5.3/scenario.js", "models/glm-5.3/request.js", "models/glm-5.3/frameworks.js"],
  },
  {
    id: "deepseek-v4.1-flash-1017",
    name: "DeepSeek-V4.1-Flash-1017",
    defaultEngine: "vLLM",
    scripts: ["models/deepseek-v4.1-flash-1017/scenario.js"],
  },
  { id: "kimi-k2.6", name: "Kimi-K2.6", defaultEngine: "vLLM", scripts: ["models/kimi-k2.6/scenario.js"] },
  { id: "hy4-flash", name: "Hy4 Flash", defaultEngine: "SGLang", scripts: ["models/hy4-flash/scenario.js"] },
  {
    id: "minimax-m3.1-flash",
    name: "MiniMax-M3.1-Flash",
    defaultEngine: "SGLang",
    scripts: ["models/minimax-m3.1-flash/scenario.js"],
  },
  {
    id: "qwen3.8-flash-next",
    name: "Qwen3.8-Flash-Next",
    defaultEngine: "SGLang",
    scripts: ["models/qwen3.8-flash-next/scenario.js"],
  },
  { id: "nemotron-3-super", name: "Nemotron 3 Super 120B-A12B", defaultEngine: "vLLM", scripts: ["models/nemotron-3-super/scenario.js"] },
  { id: "qwen3.5-397b-a17b", name: "Qwen3.5-397B-A17B", defaultEngine: "vLLM", scripts: ["models/qwen3.5-397b-a17b/scenario.js"] },
];
window.inferenceModels = {};
