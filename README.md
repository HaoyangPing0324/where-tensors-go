<p align="center">
  <a href="https://haoyangping0324.github.io/where-tensors-go/">
    <img src="./assets/wordmark.png" width="420" alt="where-tensors-go">
  </a>
</p>

<h1 align="center">看见张量在 GPU 之间流动</h1>

<p align="center">
  一个交互式大模型推理可视化项目。跟随一次请求，从输入到回复，<br>
  看每张 GPU 存了什么、算了什么，又把结果交给了谁。
</p>

<p align="center">
  <a href="https://haoyangping0324.github.io/where-tensors-go/"><img src="./assets/readme/demo.svg" height="24" alt="打开在线演示"></a>
  &nbsp;<img src="./assets/readme/topology.svg" height="24" alt="两台机器，每台八张 GPU">
  &nbsp;<img src="./assets/readme/phases.svg" height="24" alt="从 Prefill 到 Decode">
</p>

<a href="https://haoyangping0324.github.io/where-tensors-go/">
  <img src="./assets/preview.png" width="100%" alt="演示全景：左侧是请求流程，右侧八张 Prefill 卡正在交换张量分块">
</a>

## 🧭 怎样读这张图

- **沿请求走** · 从目录选择一个阶段，或点「下一步」，跟随同一个请求经过 Prefill、KV 交接和 Decode。
- **往卡里看** · 每张卡的位置固定。点击卡内的数据块，查看本步的张量形状、权重分片和计算结果。
- **顺着线看** · 通信动画逐轮更新各卡的缓冲区。点击连线，就能看到这一跳的发送卡、接收卡和传输内容。
