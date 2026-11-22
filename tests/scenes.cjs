const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const vm = require("node:vm");

const runtime = vm.createContext({ window: {} });
function load(file) {
  vm.runInContext(readFileSync(resolve(__dirname, "..", file), "utf8"), runtime, { filename: file });
}
load("models/catalog.js");
load("models/tp-scene.js");
const catalog = vm.runInContext("catalog", runtime);
assert.equal(catalog.length, 6);

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
      for (const layer of variant.layerClasses) {
        for (const [i] of variant.stages.entries()) {
          if (
            !variant.layerAllows(i, layer, layout.id) ||
            variant.service(i) ||
            variant.op(i, layout.id).match(/AllReduce|ReduceScatter|AllGather/)
          )
            continue;
          const side = variant.phase(i) < 13 ? "P" : "D";
          for (const r of [0, 7]) {
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
