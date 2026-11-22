{
  // NVIDIA checkpoint 2dc98e2: M, * and E are standalone pre-norm residual blocks.
  const pattern = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";
  const runs = [...pattern.matchAll(/M\*?E/g)];
  const spec = {
    id: "nemotron-3-super",
    hidden: 4096,
    vocab: 131072,
    experts: 512,
    top: 22,
    latent: 1024,
    middle: 2688,
    sharedMiddle: 5376,
    relu2: true,
    activation: [String.raw`M_r=\operatorname{ReLU}(XW_{u,r})^{\odot 2}`, "Up 投影后逐元素取正并平方，再乘 Down。"],
    router: "4096 维主干输入乘 [4096,512]。用 sigmoid 分数加选择偏置选 22 路，选中项使用原分数归一化并乘 5。",
    cache: "SSM / Conv / GQA KV",
    profile: {
      vLLM: "BF16 · 双端 TP8 分片示例 · vLLM 的 Nemotron 混合状态 PD 参考使用 TP4 / NIXL",
      SGLang: "BF16 · 双端 TP8 分片示例 · Dynamo 的 SGLang PD 参考使用 FP8、双端 TP2 / NIXL",
    },
    bridge: { vLLM: "NIXL · 混合状态逻辑分片", SGLang: "NIXL · 混合状态逻辑分片" },
    cacheBlocks: r => [
      {name:"SSM 末状态",shape:"[16,64,128]",detail:"40 层各一份，本卡头 "+16*r+"–"+(16*r+15)+"。Decode 从提示末位置的矩阵继续更新。"},
      {name:"卷积历史",shape:"[3,1280]",detail:"40 层各一份，宽 4 的因果卷积下一步需要最后 3 个输入位置。仅计有效历史。"},
      {name:"GQA K / V",shape:"[16,1,128]",detail:"8 个 GQA 层各有 K、V 两份同形张量。本卡 KV 头 "+Math.floor(r/4)+"。接续新位置时追加一行。"},
    ],
    layers: [true, false].map((gqa) => {
      const groups = runs.filter((run) => run[0].includes("*") === gqa);
      const ranges = groups.map((run) => run.index + "–" + (run.index + run[0].length - 1));
      return {
        ffn: "MoE",
        gqa,
        label:
          (gqa ? "Mamba-2 → GQA → LatentMoE" : "Mamba-2 → LatentMoE") +
          " · " +
          ranges[0] +
          " 等 " +
          ranges.length +
          " 组",
        detail: ranges.join("、"),
      };
    }),
    attentionWhen: (g) => g.gqa,
    beforeAttention({ add, reduce, rows, shape, H, T, X, P }) {
      const mix = (title, op, blocks, formula) => add("Mamba-2", title, op, blocks, { formula });
      mix("归一化 Mamba 层输入", "RMSNorm", () => rows(["残差输入", X], ["通道尺度", shape(H)], ["SSM 输入", X]));
      mix("投影门、输入、状态系数与时间步", "Column parallel", (r) =>
        rows(
          ["归一化行", X],
          ["Win 分片", shape(H, 2320), "本卡 16 个头，自 " + 16 * r + " 起；B/C 对应组 " + r + "。"],
          ["z / x / B / C / Δ", "2×1024|2×128|16", "z、x 各 1024 维，B、C 各 128 维，Δ 有 16 项。每项均有 "+T+" 行。"],
        ),
      );
      mix("卷积更新 x、B 和 C", "Causal Conv1d / SiLU", () =>
        rows(
          ["x | B | C", shape(T, 1280)],
          ["卷积核", shape(1280, 4), "每通道独立卷积，带偏置，随后 SiLU。"],
          ["x、B、C", shape(T, 1280), "下次使用末 3 个位置的输入历史。"],
        ),
      );
      mix(
        P ? "扫描提示序列的 SSM 状态" : "从 Prefill 末状态推进一格",
        "Selective SSM",
        (r) =>
          rows(
            [
              "前一状态",
              shape(16, 64, 128),
              P ? "本请求从零状态起步。" : "从 P" + r + " 交接到 D" + r + " 的本层状态起步。",
            ],
            ["Δ、A、D", "每头各一标量", "Δ=softplus(dt+b)，A=−exp(A_log)。本组 B、C 各 128 维。"],
            ["最新 SSM 状态", shape(16, 64, 128), "处理 " + T + " 个位置后保存末状态。"],
          ),
        [
          String.raw`S_t=e^{\Delta_t A} S_{t-1}+\Delta_t x_t B_t^\top,\qquad y_t=S_t C_t+D x_t`,
          "按一个头写出，x 有 64 个通道，B 和 C 各有 128 个状态分量。",
        ],
      );
      mix(
        "门控并进行组内 RMSNorm",
        "Gated RMSNorm",
        () => rows(["读出 y", shape(T, 1024)], ["门 z", shape(T, 1024)], ["本卡混合结果", shape(T, 1024)]),
        [String.raw`u=\operatorname{GroupRMSNorm}(y\odot\operatorname{SiLU}(z))`, "本卡对应一组 1024 个通道。"],
      );
      mix("SSM 输出投影", "Row parallel", () =>
        rows(["本卡混合结果", shape(T, 1024)], ["Wₒ 分片", shape(1024, H)], ["本卡部分积", X]),
      );
      reduce("Mamba-2", "归并 SSM 输出投影");
      mix("写回 Mamba 层残差", "Residual", () => rows(["分支输出", X], ["本层输入", X], ["下一块输入", X]));
    },
    attention({ add, rows, shape, H, T, X, P }) {
      add("Attention", "分片投影 GQA 查询与键值", "GQA", (r) =>
        rows(
          ["本层归一化行", X],
          ["Wq / Wk / Wv", "512 | 128 | 128", "输入维 4096，输出分别对应本卡 4 个 Q 头和 1 个 KV 头。"],
          [
            "本卡 Q",
            shape(T, 4, 128),
            "Q 头 " + 4 * r + "–" + (4 * r + 3) + "，KV 头 " + Math.floor(r / 4) + " 由四卡复制。",
          ],
        ),
      );
      add("Attention", "写入全注意力缓存", "KV write", () =>
        rows(
          ["K / V 各", shape(T, 1, 128)],
          ["写入位置", P ? "0–15" : "16"],
          ["当前 K / V 各", shape(P ? 16 : 17, 1, 128)],
        ),
      );
      add(
        "Attention",
        "按因果位置计算 GQA",
        "Softmax / AV",
        () =>
          rows(["每个查询", shape(4, 128)], ["可见键值", P ? "1–16 行" : "17 行"], ["本卡头输出", shape(T, 4, 128)]),
        {
          formula: [
            String.raw`O_h=\operatorname{softmax}(Q_hK_g^\top/\sqrt{128}+M)V_g`,
            "四个本地查询头读取同一 KV 头。参考前向直接使用投影后的 Q 和 K。",
          ],
        },
      );
      add("Attention", "全注意力输出投影", "Row parallel", () =>
        rows(["头输出", shape(T, 512)], ["Wₒ 分片", shape(512, H)], ["本卡部分积", X]),
      );
    },
  };
  window.inferenceModels[spec.id] = { vLLM: makeTPScene(spec, "vLLM"), SGLang: makeTPScene(spec, "SGLang") };
}
