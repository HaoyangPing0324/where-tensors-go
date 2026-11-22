<p align="center">
  <a href="https://haoyangping0324.github.io/where-tensors-go/">
    <img src="./assets/wordmark.png" width="440" alt="where-tensors-go">
  </a>
</p>

<h1 align="center">大模型推理，逐卡可视化</h1>

<p align="center">
  跟随一次请求，看张量如何穿过模型，在 GPU 之间流动。<br>
  <em>A visual walkthrough of LLM inference, one GPU at a time.</em>
</p>

<p align="center">
  <a href="https://haoyangping0324.github.io/where-tensors-go/"><strong>打开在线演示 ↗</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://github.com/HaoyangPing0324/where-tensors-go/archive/refs/heads/main.zip">下载离线版 ↓</a>
</p>

---

一条消息发出去，直到回复出现在屏幕上，两台机器里的十六张 GPU 各自做了什么？**where-tensors-go** 把这段过程画成一张可以逐步翻看的 GPU 图。每张卡的位置固定，卡内的数据块随计算更新，卡间的连线显示张量的去向。

<p align="center">
  <a href="https://haoyangping0324.github.io/where-tensors-go/">
    <img src="./assets/preview.png" width="100%" alt="演示界面预览：左侧为请求流程目录，右侧按 GPU 展示张量分块与跨卡通信">
  </a>
</p>

<table>
  <tr>
    <td width="33%" valign="top"><strong>请求的全程</strong><br><br>从接入、Prefill 到 KV 交接、Decode 和流式返回，沿着同一个请求前进。</td>
    <td width="33%" valign="top"><strong>每张卡的分工</strong><br><br>并排查看各卡持有的权重与张量，比较不同并行方案怎样分配同一项计算。</td>
    <td width="33%" valign="top"><strong>数据的去向</strong><br><br>让通信逐轮播放，点一条连线，就能看到哪张卡把多大的张量发给了谁。</td>
  </tr>
</table>

<p align="center"><sub>演示可直接在浏览器中运行。离线使用时，解压下载包并打开 index.html 即可。</sub></p>
