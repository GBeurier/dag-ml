#!/usr/bin/env node
/** Nested grouped selection in WASM with real Methods JS fits in one worker.
 * Usage: node scripts/test_wasm_nested_selection.mjs <dag-ml-wasm-node-pkg> <methods-js-dist>
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const dagPkg = path.resolve(process.argv[2]);
const methods = await import(pathToFileURL(path.join(path.resolve(process.argv[3]), "index.js")).href);
await methods.loadModule();
const dagPackage = JSON.parse(fs.readFileSync(path.join(dagPkg, "package.json"), "utf8"));
const dagMl = dagPackage.type === "module"
  ? await import(pathToFileURL(path.join(dagPkg, "dag_ml_wasm.js")).href)
  : require(path.join(dagPkg, "dag_ml_wasm.js"));
if (dagPackage.type === "module") {
  dagMl.initSync({ module: fs.readFileSync(path.join(dagPkg, "dag_ml_wasm_bg.wasm")) });
}
const { N4mWasmRegressionController } = await import(pathToFileURL(path.join(dagPkg, "n4m_controller.mjs")).href);

const sha = text => createHash("sha256").update(text).digest("hex");
const nodeId = "model:probe";
const method = "n4m:models.pls.pls_regression";
const params = { center_x: true, scale_x: false, center_y: true, scale_y: false };

// 30 samples in 10 groups of 3, two latent directions plus small noise.
let state = 0x2545f4914f6cdd1dn;
const next = () => {
  state = (state * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
  return Number(state >> 11n) / 2 ** 53 - 0.5;
};
const ids = Array.from({ length: 30 }, (_, i) => `sample:${String(i).padStart(2, "0")}`);
const x = new Map(), y0 = new Map();
for (const id of ids) {
  const a = next(), b = next();
  x.set(id, Array.from({ length: 6 }, (_, j) => a * (j + 1) + b * (6 - j) + 0.01 * next()));
  y0.set(id, 2 * a - b + 0.01 * next());
}
const groups = Object.fromEntries(ids.map((id, i) => [id, `group:${Math.floor(i / 3)}`]));

const envelope = {
  schema_version: 1, schema_fingerprint: sha("schema"), plan_fingerprint: sha("plan"),
  relation_fingerprint: null, coordinator_relations: null,
};
const graph = {
  id: "graph:nested", interface: { inputs: [], outputs: [] },
  nodes: [{ id: nodeId, kind: "model", operator: { type: method }, params,
    ports: { inputs: [{ name: "x", kind: "data", representation: "tabular_numeric", cardinality: "one", description: "" }],
      outputs: [{ name: "y_hat", kind: "prediction", representation: null, cardinality: "one", description: "" }] },
    metadata: {}, seed_label: null }],
  edges: [], search_space_fingerprint: null, metadata: {},
};
const campaign = {
  id: "campaign:nested", root_seed: 7,
  data_bindings: { [nodeId]: [{ node_id: nodeId, input_name: "x", request_id: "x",
    schema_fingerprint: envelope.schema_fingerprint, plan_fingerprint: envelope.plan_fingerprint,
    relation_fingerprint: null, output_representation: "tabular_numeric", feature_set_id: "x",
    source_ids: ["nir"], require_relations: false }] },
};
const request = {
  id: "nested", graph, campaign, target_node: nodeId,
  candidates: [1, 2, 3].map(k => ({ candidate_id: `n_components:${String(k).padStart(3, "0")}`,
    params: { n_components: k } })),
  sample_groups: groups, outer: { n_splits: 5 }, inner: { n_splits: 3 }, selection_metric: "rmse",
};
const dense = rows => ({ data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length });

function run(y, metrics) {
  const views = [];
  const controller = new N4mWasmRegressionController({
    methods, operators: { [nodeId]: { type: method } },
    resolveFeatures: ({ view, task }) => {
      views.push({ fold: task.fold_id, partition: view.partition, ids: [...view.sample_ids] });
      return { sampleIds: [...view.sample_ids], matrix: dense(view.sample_ids.map(id => x.get(id))) };
    },
    resolveTargets: ({ sampleIds }) => ({ sampleIds: [...sampleIds], matrix: dense(sampleIds.map(id => [y.get(id)])) }),
  });
  try {
    const manifests = [controller.manifest(dagMl)];
    const outcome = JSON.parse(dagMl.execute_nested_selection_json(
      JSON.stringify(request), JSON.stringify(manifests), JSON.stringify(envelope), controller.callback,
      metrics));
    return { outcome, views };
  } finally { controller.close(); }
}

const { outcome, views } = run(y0);
assert.equal(outcome.scopes.length, 5);
assert.equal(outcome.oof.length, 30);
assert.equal(outcome.n_fits, 5 * (3 * 3 + 1));
for (const fold of outcome.outer_fold_set.folds) {
  const held = new Set(fold.validation_sample_ids.map(id => groups[id]));
  assert.ok(fold.train_sample_ids.every(id => !held.has(groups[id])), "a group crosses an outer fold");
}
// Inner views never leave their outer training pool; fit and validation are disjoint.
for (const view of views) {
  const [outer] = view.fold.split(".inner.");
  if (outer === view.fold) continue;
  const pool = outcome.outer_fold_set.folds.find(f => f.fold_id === outer).train_sample_ids;
  assert.ok(view.ids.every(id => pool.includes(id)), "inner view leaves its outer training pool");
}
for (const scope of outcome.scopes) {
  const fold = outcome.outer_fold_set.folds.find(f => f.fold_id === scope.outer_fold_id);
  assert.equal(scope.outer_fold_id, fold.fold_id);
  const best = scope.candidates.reduce((a, b) => (b.score < a.score ? b : a));
  assert.equal(scope.decision.selected_candidate_id, best.candidate_id);
  // Independent reference: the selected recipe fitted directly on the outer pool.
  const k = Number(best.candidate_id.split(":")[1]);
  const model = methods.RolePipeline.fromSteps([{ class: method, params: { ...params, n_components: k } }]);
  try {
    model.fit(dense(fold.train_sample_ids.map(id => x.get(id))), dense(fold.train_sample_ids.map(id => [y0.get(id)])));
    const expected = model.predict(dense(fold.validation_sample_ids.map(id => x.get(id)))).data;
    fold.validation_sample_ids.forEach((id, i) => {
      const row = outcome.oof.find(r => r.sample_id === id);
      assert.equal(row.outer_fold_id, fold.fold_id);
      assert.equal(row.y_true, y0.get(id));
      assert.ok(Math.abs(row.prediction - expected[i]) <= 1e-12 * (1 + Math.abs(expected[i])));
    });
  } finally { model.dispose(); }
}
assert.ok(outcome.model_metrics.rmse < 0.05 && !("r2" in outcome.model_metrics));

// Host metrics decide the selection (Methods RMSE here); DAG-ML keeps the
// comparison and the tie rule (constant metrics tie to the smallest ID).
if (methods.statsRegressionScores) {
  let calls = 0;
  const viaMethods = run(y0, (name, payload) => {
    calls++;
    const { y_true, y_pred } = JSON.parse(payload);
    const s = methods.statsRegressionScores(Float64Array.from(y_true), Float64Array.from(y_pred));
    return name === "mse" ? s.rmse * s.rmse : name === "rmse" ? s.rmse : s.mae;
  }).outcome;
  assert.equal(calls, 5 * 3 + 3);
  viaMethods.scopes.forEach((scope, i) =>
    assert.equal(scope.decision.selected_candidate_id, outcome.scopes[i].decision.selected_candidate_id));
}
const tied = run(y0, () => 1).outcome;
assert.ok(tied.scopes.every(scope => scope.decision.selected_candidate_id === "n_components:001"));
assert.throws(() => run(y0, () => -1), /finite and nonnegative/);

// Outer validation targets never reach their own inner selection.
const shifted = new Map(y0);
for (const id of outcome.outer_fold_set.folds[0].validation_sample_ids) shifted.set(id, y0.get(id) + 100);
const changed = run(shifted).outcome;
assert.deepEqual(changed.scopes[0].candidates, outcome.scopes[0].candidates);
assert.deepEqual(changed.scopes[0].decision, outcome.scopes[0].decision);

// Infeasible plans are refused before any controller call.
let calls = 0;
const refusal = new N4mWasmRegressionController({ methods, operators: { [nodeId]: { type: method } },
  resolveFeatures: () => { calls++; throw new Error("unexpected"); }, resolveTargets: () => { throw new Error("unexpected"); } });
try {
  assert.throws(() => dagMl.execute_nested_selection_json(
    JSON.stringify({ ...request, outer: { n_splits: 11 } }), JSON.stringify([refusal.manifest(dagMl)]),
    JSON.stringify(envelope), refusal.callback, undefined), /GroupKFold|exceeds/);
} finally { refusal.close(); }
assert.equal(calls, 0);
fs.writeSync(1, "WASM nested selection: 5 outer x 3 inner folds, real Methods fits, parity with direct fits\n");
