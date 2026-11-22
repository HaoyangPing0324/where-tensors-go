{
  // Qwen3.5-397B-A17B text_config at 8472618, ordinary text decoding.
  const spec = {
    id: "qwen3.5-397b-a17b",
    hidden: 4096,
    vocab: 248320,
    experts: 512,
    top: 10,
    middle: 1024,
    sharedGate: true,
    norm: [String.raw`\hat x_i=(1+w_i)x_i/\sqrt{H^{-1}\sum_j x_j^2+\epsilon}`, "RMSNorm 使用 1+w 作为通道尺度。"],
    router: "主干行乘 [4096,512]，softmax 后取 Top-10，再按选中分数之和归一化。共享专家单独计算 sigmoid 门。",
    cache: "GDN / Conv / GQA KV",
    profile: {
      vLLM: "vLLM Recipes · 2 × 8 H200 · TP8/TP8 · NIXL · 文本逐 token 解码",
      SGLang: "SGLang · Mooncake 混合状态交接 · 图按两端 TP8 展开",
    },
    cacheBlocks: r => [
      {name:"GDN 末状态",shape:"[8,128,128]",detail:"45 层各一份，本卡头 "+8*r+"–"+(8*r+7)+"。Decode 从提示末位置的矩阵继续更新。"},
      {name:"卷积历史",shape:"[3,1536]",detail:"45 层各一份，宽 4 的因果卷积下一步需要最后 3 个输入位置。仅计有效历史。"},
      {name:"GQA K / V",shape:"[16,1,256]",detail:"15 个 GQA 层各有 K、V 两份同形张量。本卡 KV 头 "+Math.floor(r/4)+"。接续新位置时追加一行。"},
    ],
    layers: [
      { ffn: "MoE", full: false, label: "0–2、4–6、…、56–58 · Gated DeltaNet + MoE" },
      { ffn: "MoE", full: true, label: "3、7、…、59 · 全注意力 GQA + MoE" },
    ],
    attention({ add, rows, shape, H, T, X, P }) {
      const step = (full, title, op, blocks, formula) =>
        add("Attention", title, op, blocks, { when: (g) => g.full === full, formula });
      step(false, "生成 GDN 的 Q、K、V", "Column parallel", (r) =>
        rows(
          ["归一化行", X],
          [
            "Wqkv 分片",
            shape(H, 1536),
            "本卡 Q/K 头 " + 2 * r + "–" + (2 * r + 1) + "，V 头 " + 8 * r + "–" + (8 * r + 7) + "。",
          ],
          ["Q | K | V", "256 | 256 | 1024", "各有 " + T + " 行，单头宽度 128。"],
        ),
      );
      step(
        false,
        "投影输出门与状态更新系数",
        "GDN gates",
        () =>
          rows(
            ["归一化行", X],
            ["Wz / Wa / Wb", "1024 | 8 | 8", "三个投影的输入维均为 4096。"],
            ["z、衰减、写入门", "1024 | 8 | 8", "z 为 "+shape(T,1024)+"，衰减和写入门各为 "+shape(T,8)+"。"],
          ),
        [
          String.raw`\beta=\sigma(b),\quad \log\alpha=-\exp(A_{\log})\operatorname{softplus}(a+b_{\Delta})`,
          "每个 value 头有独立的衰减和更新门。",
        ],
      );
      step(false, "卷积与 QK 归一化", "Causal Conv1d", () =>
        rows(
          ["打包 QKV", shape(T, 1536)],
          ["逐通道卷积核", shape(1536, 4)],
          ["本地 Q、K、V", "2 / 2 / 8 头", "卷积后 SiLU，Q、K 沿头维做 L2 归一化；每个 K 头对应四个 V 头。"],
        ),
      );
      step(
        false,
        P ? "扫描 Gated DeltaNet 状态" : "续写 Gated DeltaNet 状态",
        "Delta rule",
        (r) =>
          rows(
            ["上一状态", shape(8, 128, 128), P ? "初始为零。" : "从 P" + r + " 交接，覆盖此前 16 个提示位置。"],
            ["k、v、α、β", "按 value 头更新"],
            ["最新递归状态", shape(8, 128, 128)],
          ),
        [
          String.raw`\bar S_t=\alpha_t S_{t-1},\qquad S_t=\bar S_t+\beta_t k_t(v_t^\top-k_t^\top\bar S_t)`,
          "每个头各存一张 128 × 128 的矩阵。",
        ],
      );
      step(
        false,
        "状态读出与 SiLU 输出门",
        "Gated RMSNorm",
        () => rows(["缩放查询 × S", shape(T, 8, 128)], ["门 z", shape(T, 8, 128)], ["读出结果", shape(T, 1024)]),
        [
          String.raw`y_t=\operatorname{RMSNorm}(q_t^\top S_t/\sqrt{128})\odot\operatorname{SiLU}(z_t)`,
          "各 value 头分别归一化、门控，然后拼接。",
        ],
      );
      step(true, "投影查询、输出门和 KV", "GQA / GEMM", (r) =>
        rows(
          ["归一化行", X],
          ["Wq+gate", shape(H, 2048), "拆为 4 个 256 维 Q 头及对应门。Wk、Wv 各 [4096,256]。"],
          [
            "Q / KV 头",
            "4 Q / 1 KV",
            "本卡 Q 头 " + 4 * r + "–" + (4 * r + 3) + "；KV 头 " + Math.floor(r / 4) + " 由相邻四卡复制。",
          ],
        ),
      );
      step(true, "头内归一化、RoPE 与缓存写入", "RMSNorm / RoPE", () =>
        rows(
          ["Q / K 每头", shape(T, 256)],
          ["旋转位置通道", "前 64 / 256", "Q 与 K 各自归一化，再对 1/4 头维旋转。"],
          ["K / V 缓存各", shape(P ? 16 : 17, 1, 256)],
        ),
      );
      step(
        true,
        "读因果 KV 并应用输出门",
        "Softmax / AV",
        () => rows(["本卡 Q", shape(T, 4, 256)], ["可见位置", P ? "每行 1–16" : "17"], ["门控输出", shape(T, 1024)]),
        [
          String.raw`O_h=\operatorname{softmax}(Q_hK_g^\top/\sqrt{256}+M)V_g\odot\sigma(Z_h)`,
          "四个本地 Q 头共享一个 KV 头，门由本层输入投影。",
        ],
      );
      add("Attention", "投影本卡输出通道", "Row parallel", (r, g) =>
        rows(
          ["局部输出", shape(T, 1024), g.full ? "4 × 256 个注意力通道。" : "8 × 128 个递归读出通道。"],
          ["Wₒ 分片", shape(1024, H)],
          ["本卡部分积", X],
        ),
      );
    },
  };
  window.inferenceModels[spec.id] = { vLLM: makeTPScene(spec, "vLLM"), SGLang: makeTPScene(spec, "SGLang") };
}
