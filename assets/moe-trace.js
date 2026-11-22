// One token row, one contiguous expert partition per EP rank. The routing IDs are illustrative.
function makeExpertTrace(spec, count) {
  const perRank = spec.experts / count;
  const placements = [2, 0, 5, 2, 6, 4, 5, 7, 1, 2, 7, 6, 1, 3, 0, 4];
  const owners = Array.from({ length: count }, (_, rank) => ({
    rank,
    first: rank * perRank,
    last: (rank + 1) * perRank - 1,
    ids: [],
  }));
  if (!Number.isInteger(perRank) || count !== 8 || spec.top > placements.length)
    throw Error("缺少此专家拓扑的路由示例");
  for (const rank of placements.slice(0, spec.top)) {
    const owner = owners[rank];
    owner.ids.push(owner.first + ((3 + owner.ids.length * 5) % perRank));
  }
  return owners;
}

const traceSteps = ["Dispatch · 分发", "Expert FFN · 计算", "Combine · 合并"];
let traceStep = 0;
const traceMode = () => $("moePath").value;

function traceEdges(mode, step, owners) {
  if (step === 1) return [];
  const direct = mode === "direct",
    dispatch = step === 0;
  if (direct)
    return owners
      .filter((o) => o.rank && o.ids.length)
      .map((o) => ({
        src: dispatch ? 0 : o.rank,
        dst: dispatch ? o.rank : 0,
        rank: o.rank,
        tick: 1,
      }));
  // Follow only the chunk containing A. Its ReduceScatter owner is rank zero.
  return owners.slice(1).map(({ rank }) => ({
    src: dispatch ? rank - 1 : rank,
    dst: dispatch ? rank : (rank + 1) % owners.length,
    rank,
    tick: rank,
  }));
}

function traceWire(edge, side, step, direct, owners, width, animate) {
  const { src, dst, rank } = edge,
    dispatch = step === 0;
  const ids = owners[rank].ids.map((e) => "E" + e).join("、");
  let payload;
  if (dispatch) {
    payload = direct
      ? "xₐ [1," + width + "]；目的卡路由项 " + ids + "（专家 ID、行号与系数逻辑关联）"
      : "A 的输入行 xₐ [1," + width + "] 及路由信息；复制到下一卡，再筛选本卡专家";
  } else {
    payload = direct
      ? ids + " 对 A 的加权贡献，在本卡相加后回传 s" + src + " [1," + width + "]"
      : "A 的累加行 [1," + width + "]；已含 rank 1…" + src + " 的贡献，接收卡加入本地贡献";
  }
  const info = side + src + " → " + side + dst + " · 请求 A · " + payload;
  // Each remote peer has its own source port and lane. Lines meet only at card boundaries.
  const port = (r) => gx(r) + (r === 0 && direct ? 12 + rank * 13 : 68);
  const sx = port(src),
    dx = port(dst),
    y = 245,
    lane = y + 15 + rank * 13;
  const path = "M" + sx + " " + y + " V" + lane + " H" + dx + " V" + y;
  const color = colors[rank];
  return (
    '<g class="wire" role="button" tabindex="0" aria-label="' +
    esc(info) +
    '" data-info="' +
    esc(info) +
    '"><title>' +
    esc(info) +
    '</title><path class="hit" d="' +
    path +
    '"/><path class="path" d="' +
    path +
    '" stroke="' +
    color +
    '"/><path d="M' +
    (dx - 4) +
    " " +
    (y + 6) +
    ' l4 -6 4 6" stroke="' +
    color +
    '" fill="none"/>' +
    (animate
      ? '<circle class="packet" r="4" fill="' +
        color +
        '"><animateMotion dur="0.65s" path="' +
        path +
        '" fill="freeze"/></circle>'
      : "") +
    "</g>"
  );
}

function traceBoard(side, spec, owners, mode, step, tick) {
  const direct = mode === "direct",
    dispatch = step === 0,
    compute = step === 1;
  const w = spec.width,
    m = spec.middle,
    vec = "[1," + w + "]";
  const edges = traceEdges(mode, step, owners);
  const last = direct ? 2 : 7;
  const complete = tick >= last;
  let cards = "";
  for (const owner of owners) {
    const { rank, ids, first, last: end } = owner,
      x = gx(rank),
      matched = ids.length > 0;
    const hasInput = dispatch
      ? rank === 0 || (direct ? matched && tick >= 1 : tick >= rank)
      : direct
        ? matched || rank === 0
        : true;
    const idsLabel = ids.map((e) => "E" + e).join(" ");
    const box = (y, label, sub, info, col) =>
      slot(x + 6, y, 116, 40, label, col || colors[rank], side + rank + " · " + info, sub);
    cards +=
      '<rect class="device" x="' +
      x +
      '" y="8" width="128" height="237" rx="7"/>' +
      txt(x + 8, 30, side + rank, "rank");
    cards += txt(x + 8, 53, "E" + first + "–" + end + " 常驻", "hint");
    cards += box(
      66,
      matched ? idsLabel : "未命中专家",
      matched ? "命中 " + ids.length + " 个专家" : "A 的贡献为 0",
      "路由结果示例。" +
        (idsLabel || "此卡没有命中的专家") +
        "。本卡持有 E" +
        first +
        "–E" +
        end +
        " 的完整 Gate、Up、Down 权重。",
    );
    if (compute && matched) {
      cards += box(
        113,
        "Gate / Up",
        "各 [1," + m + "]",
        "每个命中专家分别读取 xₐ " + vec + "，乘 Wgate 和 Wup，各 [" + w + "," + m + "]。",
      );
      cards += box(
        158,
        "激活 → Down",
        vec,
        "门控激活 [1," + m + "] 乘 Wdown [" + m + "," + w + "]，得到每个专家输出 " + vec + "。",
      );
      cards += txt(x + 8, 225, "本卡加权求和 s" + rank, "hint");
    } else if (dispatch || compute) {
      cards += box(
        118,
        hasInput ? (rank === 0 ? "源行 xₐ" : "收到 xₐ") : "无输入行",
        hasInput ? vec : "—",
        hasInput ? "当前持有 A 的完整专家输入行 " + vec : "此拍尚无 A 的行。",
      );
      cards += txt(
        x + 8,
        184,
        hasInput && matched ? "按 ID 打包命中行" : hasInput ? "筛选后无计算" : direct && !matched ? "无需接收" : "等待",
        "hint",
      );
      if (rank === 0) cards += txt(x + 8, 218, "A · 原 token 槽", "hint");
    } else {
      cards += box(
        113,
        matched ? "局部贡献 s" + rank : "零贡献",
        vec,
        matched ? "s" + rank + " 为本卡各命中专家输出乘各自路由权重后的和。" : "本卡没有命中 A 的专家，逻辑贡献为零。",
      );
      if (rank === 0) {
        cards += box(
          161,
          complete ? "合并完成 yₐ" : "原 token 槽",
          complete ? vec : "等待远端贡献",
          complete ? "路由分支最终输出 " + vec + "，还原到 A 的原 token 槽。" : "保留本地贡献，等待回传。",
        );
      } else {
        const sent = direct ? matched && tick >= 1 : tick >= rank;
        cards += txt(x + 8, 189, sent ? "已发送贡献" : direct && !matched ? "无回传" : "等待交换", "hint");
      }
    }
  }
  const paths = edges
    .filter((e) => e.tick <= tick)
    .map((e) => traceWire(e, side, step, direct, owners, w, e.tick === tick))
    .join("");
  const h = compute ? 253 : 365;
  return (
    '<div class="scroll" tabindex="0" aria-label="八卡通信图，可横向滚动"><svg class="deck traceDeck" role="group" aria-label="' +
    side +
    ' 节点路由专家通信" viewBox="0 0 1160 ' +
    h +
    '">' +
    cards +
    paths +
    "</svg></div>"
  );
}

function traceFormula(spec, step) {
  if (step === 0)
    return [
      String.raw`x_A\in\mathbb R^{1\times ` +
        spec.width +
        String.raw`},\quad I_A=\{e_1,\ldots,e_{` +
        spec.top +
        String.raw`}\}`,
      "跟踪请求 A 的一个 token。源槽在 rank 0，图中固定命中 " +
        spec.top +
        " 个专家。" +
        (spec.width !== spec.hidden ? "路由分支先将 " + spec.hidden + " 维投影至 " + spec.width + " 维。" : ""),
    ];
  if (step === 1)
    return [
      String.raw`G=x_AW_{g,e},\ U=x_AW_{u,e},\quad f_e(x_A)=\phi(G,U)W_{d,e}`,
      spec.activation?.[1] || "每个命中专家计算 φ(G,U) = SiLU(G) ⊙ U。点矩阵块查看输入、权重和输出形状。",
    ];
  return [
    String.raw`s_r=\sum_{e\in I_A\cap E_r} w_{A,e}\, f_e(x_A),\qquad y_A=\sum_{r=0}^{7}s_r`,
    "每个专家的路由系数只乘一次。路由输出写回原 token 槽" +
      (spec.width !== spec.hidden ? "，随后投回 " + spec.hidden + " 维" : "") +
      "；共享专家走独立分支。",
  ];
}

function drawTraceNode(side, activeSide) {
  const name = side === "P" ? "01 · Prefill" : "02 · Decode";
  if (side !== activeSide) {
    const cards = ranks()
      .map(
        (r) =>
          '<rect class="device" x="' +
          gx(r) +
          '" y="8" width="128" height="60" rx="7"/>' +
          txt(gx(r) + 8, 32, side + r, "rank") +
          txt(gx(r) + 8, 55, "本步未参与", "hint"),
      )
      .join("");
    return (
      '<section class="node"><h2>' +
      name +
      '</h2><div class="scroll"><svg class="deck" viewBox="0 0 1160 76">' +
      cards +
      "</svg></div></section>"
    );
  }
  const spec = scene.moe,
    owners = makeExpertTrace(spec, scene.ranks);
  const [formula, caption] = traceFormula(spec, traceStep);
  const remote = owners.filter((o) => o.rank && o.ids.length).length;
  const counts =
    traceStep === 0
      ? " 本行" + (traceMode() === "direct" ? "发往 " + remote + " 张远端卡" : "复制到其余 7 张卡") + "。"
      : "";
  return (
    '<section class="node"><h2>' +
    name +
    ' <span>路由专家组 · EP 8 · 单专家 TP 1</span></h2><div class="nodeFormula"><div class="mathDisplay">' +
    katex.renderToString(formula, { displayMode: true }) +
    '</div><span class="mathCaption">' +
    esc(caption + counts) +
    "</span></div>" +
    traceBoard(side, spec, owners, traceMode(), traceStep, beat) +
    "</section>"
  );
}

// Backend examples from SGLang's GLM-5.3 configurator and vLLM's per-model recipe YAML.
function traceProfile(model, engine, phase) {
  const scope = "EP8 算子对照 · 固定路由样例";
  if (traceMode() !== "direct") return scope + " · 环形行分块";
  if (model === "glm-5.3" && engine === "SGLang") return scope + " · GLM-5.3 可选 DeepEP";
  if (model === "kimi-k2.6" && engine === "vLLM" && phase === "Decode")
    return scope + " · GB200 DEP 解码配置使用 FlashInfer NVLink";
  if ((model === "deepseek-v4.1-flash-1017" || model === "minimax-m3.1-flash") && engine === "vLLM")
    return scope + " · Blackwell EP 配置可融合通信与计算（MegaMoE）";
  return scope + " · 按模型维度展示逻辑交换";
}

function traceLimit() {
  if (traceStep === 1) return 0;
  return traceMode() === "direct" ? 2 : 7;
}

function startTrace(step) {
  stop();
  traceStep = step;
  beat = 0;
  $("inspect").hidePopover();
  if (traceLimit()) {
    timer = setInterval(() => {
      beat++;
      if (beat >= traceLimit()) stop();
      render();
    }, communicationRoundMs);
  }
  render();
}

function renderTrace() {
  const side = scene.path(stage)[0] === "Prefill" ? "P" : "D";
  const direct = traceMode() === "direct";
  const ops = direct
    ? ["AllToAll · 按目的卡发送", "Grouped GEMM", "AllToAll · 按来源卡返回"]
    : ["AllGather · 复制 token 行", "Grouped GEMM", "ReduceScatter · 求和并归还行"];
  const timing = direct ? ["初始缓冲区", "交换中", "已到达"][beat] : beat ? "环上第 " + beat + " / 7 跳" : "初始缓冲区";
  $("title").textContent = traceSteps[traceStep];
  $("beat").textContent = ops[traceStep] + (traceStep === 1 ? "" : "　·　" + timing);
  $("ringStages").hidden = true;
  $("service").hidden = true;
  $("play").hidden = traceStep === 1;
  $("play").textContent = "重新播放通信";
  $("prev").disabled = false;
  $("next").disabled = false;
  const scrollPositions = [...$("map").querySelectorAll(".node .scroll")].map((el) => el.scrollLeft);
  const steps = traceSteps
    .map(
      (s, i) =>
        '<button data-trace-step="' + i + '"' + (i === traceStep ? ' aria-current="step"' : "") + ">" + s + "</button>",
    )
    .join("");
  $("map").innerHTML =
    '<div class="traceSteps" aria-label="路由专家步骤">' +
    steps +
    "<button data-close-trace>返回当前部署</button></div>" +
    drawTraceNode("P", side) +
    drawTraceNode("D", side);
  $("map")
    .querySelectorAll(".node .scroll")
    .forEach((el, i) => {
      el.scrollLeft = scrollPositions[i] || 0;
    });
}

function closeTrace() {
  $("moePath").value = "";
  tocStage = -1;
  enter(stage);
}
