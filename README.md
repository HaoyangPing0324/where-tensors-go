<p align="center">
  <a href="https://haoyangping0324.github.io/where-tensors-go/"><img src="./assets/wordmark.png" width="420" alt="where-tensors-go"></a>
</p>

<h1 align="center">一次推理，逐卡看</h1>
<p align="center">
  Follow one request across GPUs, from prompt to response.<br>
  <a href="https://haoyangping0324.github.io/where-tensors-go/"><strong>打开交互演示 ↗</strong></a>
</p>

输入一条消息，到屏幕上出现回复，中间的张量去了哪里？**where-tensors-go** 把一次大模型推理展开在两台机器、十六张 GPU 上。每张卡在图里的位置固定。沿请求向前一步，就能看到哪些数据留在原处，哪些参与了矩阵乘法，哪些被送到了另一张卡。

<a href="https://haoyangping0324.github.io/where-tensors-go/"><img src="./assets/preview.png" alt="Prefill 的八张 GPU 正在合并查表结果。AllGather 第五轮结束，每张卡已有六个完整块。"></a>

## 在图上能看到什么？

| 想弄明白的问题 | 在哪里看 |
| --- | --- |
| **这次矩阵乘法，每张卡到底算哪部分？** | 打开 **Prefill → Attention → 展开 Q**，对照本卡的输入、权重切片与输出形状。切换按头或按位置分片，观察同一个投影怎样分到八张卡。 |
| **一条通信连线里传的是什么？** | 打开 **Prefill → Embedding → 合并查表结果**，点击箭头查看源卡、目的卡、张量分块及维度。每过一轮，缓冲区都会更新。 |
| **Prefill 算完以后，Decode 从哪里接着算？** | 打开 **PD 交接 → KV 缓存传输**，看缓存从哪一端交给哪一端。再进入 **Decode → Attention**，看新位置的查询怎样使用之前的 KV。 |

## 跟一个分块走完 AllReduce

八张卡各持一段词表。一个 token ID 只会在其中一张卡上查到词向量，其余七张卡为该位置填零。下图中，**每列是一张 GPU，每行是一个张量分块**。只读第一行 B0，就能跟住同一份数据。

![八卡 AllReduce 的六个时刻，从本地查表、分块求和，到将完整结果复制到每张卡](./assets/embedding-allreduce.png)

在这个 16-token 示例里，B0 装着第 0 和第 8 个位置的词向量，形状为 `[2, 6144]`。**ReduceScatter** 结束时，P0 收齐 B0 的八份贡献。接着 **AllGather** 把求和结果沿环复制：先从 P0 到 P1，第五轮结束已有六张卡持有 B0，第七轮结束八张卡全部拿到。其余七个分块也同时沿环传递，最终每张卡都拼齐 `[16, 6144]` 的输入矩阵。

演示中的模型维度取自公开配置，token、专家命中和通信轮次采用教学示例。下载仓库后，用浏览器打开 `index.html` 即可离线浏览。

[模型配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) · [PD 分离配方](https://github.com/ai-dynamo/dynamo/blob/v1.6.1/recipes/glm-5.2/sglang/disagg-h200-agentic/deploy.yaml) · [通信交互参考 · InferFlux](https://abatom.github.io/inferflux/collective-3d.html)
