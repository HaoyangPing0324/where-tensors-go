const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const vm = require("node:vm");
const katex = require("../assets/katex/katex.min.js");

const runtime = vm.createContext({ window: {} });
function load(file) {
  vm.runInContext(readFileSync(resolve(__dirname, "..", file), "utf8"), runtime, { filename: file });
}
load("models/catalog.js");
load("models/tp-scene.js");
const catalog = vm.runInContext("catalog", runtime);
assert.equal(catalog.length, 8);

function output(scene, title, group, rank = 0) {
  const stage = scene.stages.findIndex(
    (step, i) => step[1] === title && scene.path(i)[0] === "Prefill" && scene.layerAllows(i, group, "tp"),
  );
  assert.notEqual(stage, -1, title);
  return scene.describe(stage, rank, "P", 1, "tp", group)[2].value;
}

for (const entry of catalog) {
  entry.scripts.forEach(load);
  const variants = runtime.window.inferenceModels[entry.id];
  assert.deepEqual(Object.keys(variants).sort(), ["SGLang", "vLLM"]);
  const scene = variants.vLLM;
  const group = scene.layerClasses[0];
  switch (entry.id) {
    case "glm-5.3":
      assert.equal(output(scene, "展开本卡查询头", group), "[16,8,256]");
      break;
    case "deepseek-v4.1-flash-1017":
      assert.equal(output(scene, "分组低秩输出投影", group), "[16,1024]");
      break;
    case "nemotron-3-super": {
      const pair = scene.layerClasses.find(g => !g.gqa);
      const attention = scene.stages.findIndex(s => s[1] === "全注意力输出投影");
      assert.equal(scene.layerAllows(attention, pair, "tp"), false);
      assert.equal(output(scene, "扫描提示序列的 SSM 状态", pair), "[16,64,128]");
      assert.equal(output(scene, "全注意力输出投影", group), "[16,4096]");
      assert.equal(output(scene, "Up 投影与平方 ReLU", group), "[mₑ,336]");
      assert.equal(scene.moe.top, 22);
      assert.match(scene.transfer(7), /112–127/);
      break;
    }
    case "qwen3.5-397b-a17b":
      assert.equal(output(scene, "扫描 Gated DeltaNet 状态", group), "[8,128,128]");
      assert.equal(output(scene, "头内归一化、RoPE 与缓存写入", scene.layerClasses.find(g=>g.full)), "[16,1,256]");
      assert.equal(scene.moe.top, 10);
      assert.match(scene.transfer(7), /56–63/);
      break;
    case "kimi-k2.6":
      assert.equal(output(scene, "恢复各头 value", group), "[16,8,128]");
      break;
    case "hy4-flash":
      assert.equal(output(scene, "展开查询并施加旋转编码", group), "[16,4,256]");
      assert.equal(
        output(
          scene,
          "展开查询并施加旋转编码",
          scene.layerClasses.find((g) => !g.sparse),
        ),
        "[16,8,256]",
      );
      break;
    case "minimax-m3.1-flash":
      assert.match(scene.transfer(7), /KV 头 3/);
      const draft = scene.stages.findIndex((s) => s[1].includes("验证并提交"));
      assert.equal(scene.layerAllows(draft, group, "tp"), false);
      assert.equal(scene.layerAllows(draft, group, "draft"), true);
      break;
    case "qwen3.8-flash-next":
      assert.equal(output(scene, "逐位置扫描递归状态", group), "[6,128,128]");
      assert.equal(
        output(
          scene,
          "生成 QSA 查询与分组 KV",
          scene.layerClasses.find((g) => g.qsa),
        ),
        "[16,3,256]",
      );
      break;
  }
  for (const variant of Object.values(variants)) {
    for (const layout of variant.parallel.schemes) {
      const group = variant.layerClasses[0];
      for (const [phase, side] of [["Prefill", "P"], ["Decode", "D"]]) {
        const outputs = variant.stages.flatMap((step, i) =>
          variant.path(i)[0] === phase && variant.path(i)[1] === "输出" &&
          variant.layerAllows(i, group, layout.id) && ["LM head", "LM head / Sample", "AllGather · logits", "Sample"].includes(step[2])
            ? [i] : []);
        assert.ok(outputs.some(i => variant.stages[i][2].includes("LM head")));
        assert.ok(outputs.some(i => variant.stages[i][2].includes("Sample")));
        for (const i of outputs) {
          const [tex, caption] = variant.equation(i, side, 1, layout.id, group);
          assert.ok(caption, variant.stages[i][1]);
          assert.match(katex.renderToString(tex, {throwOnError:true}), /class="katex"/);
        }
      }
      for (const layer of variant.layerClasses) {
        for (const [i] of variant.stages.entries()) {
          if (
            !variant.layerAllows(i, layer, layout.id) ||
            variant.service(i) ||
            variant.op(i, layout.id).match(/AllReduce|ReduceScatter|AllGather/)
          )
            continue;
          const side = variant.phase(i) < 13 ? "P" : "D";
          for (let r=0; r<variant.ranks; r++) {
            const blocks = variant.describe(i, r, side, 1, layout.id, layer);
            assert.equal(blocks.length, 3, [variant.id, i, r].join("/"));
            assert.doesNotMatch(JSON.stringify(blocks), /NaN|undefined|Infinity/);
          }
        }
      }
    }
  }
  console.log("PASS", entry.name);
}

load("assets/moe-trace.js");
const {makeExpertTrace, traceEdges} = runtime;
for (const entry of catalog) {
  for (const scene of Object.values(runtime.window.inferenceModels[entry.id])) {
    const owners = makeExpertTrace(scene.moe, scene.ranks);
    const ids = owners.flatMap(owner => owner.ids);
    assert.equal(new Set(ids).size, scene.moe.top);
    for (const {ids, first, last} of owners) assert.ok(ids.every(id => id >= first && id <= last));
    // Give each routed expert a distinct nonzero contribution and reconstruct A at its source.
    const values = owners.map(owner => owner.ids.reduce((sum,id) => sum + id + 1, 0));
    const expected = ids.reduce((sum,id) => sum + id + 1, 0);
    for (const mode of ["direct","gather"]) {
      const buffers = [...values];
      for(const edge of traceEdges(mode,2,owners)) buffers[edge.dst] += buffers[edge.src];
      assert.equal(buffers[0], expected);
    }
    const outgoing = traceEdges("direct", 0, owners);
    assert.equal(outgoing.length, owners.filter(o => o.rank && o.ids.length).length);
    assert.deepEqual(outgoing.map(e => [e.dst,e.src]), traceEdges("direct", 2, owners).map(e => [e.src,e.dst]));
    for (const step of [0,2]) {
      const hops = traceEdges("gather", step, owners);
      assert.equal(hops.length, 7);
      assert.equal(step === 0 ? hops[0].src : hops[6].dst, 0);
      assert.ok(hops.slice(1).every((edge,i) => edge.src === hops[i].dst));
    }
  }
}
