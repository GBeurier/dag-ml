#!/usr/bin/env node
/** Four-source nested HPO with real Methods fits and native fold scheduling.
 * Numerical sources below are deterministic test fixtures, not a generator API.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageDir = path.resolve(process.argv[2]);
const methodsDist = path.resolve(process.argv[3]);
const metadata = JSON.parse(fs.readFileSync(path.join(packageDir, "package.json"), "utf8"));
const dagMl = metadata.type === "module" || metadata.module ?
  await import(pathToFileURL(path.join(packageDir, "dag_ml_wasm.js")).href) :
  require(path.join(packageDir, "dag_ml_wasm.js"));
if (metadata.type === "module" || metadata.module)
  dagMl.initSync({ module: fs.readFileSync(path.join(packageDir, "dag_ml_wasm_bg.wasm")) });
const methods = await import(pathToFileURL(path.join(methodsDist, "index.js")).href);
await methods.loadModule();
const { N4mWasmRegressionController } = await import(pathToFileURL(path.join(packageDir, "n4m_controller.mjs")).href);
const { N4mWasmHostOptimizer } = await import(pathToFileURL(path.join(packageDir, "n4m_hpo_optimizer.mjs")).href);
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
const canonical = value => Array.isArray(value) ? value.map(canonical) :
  value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const fingerprint = value => sha256(JSON.stringify(canonical(value)));
const sourceNames = ["nir", "image", "series", "metadata"];
const sampleIds = Array.from({ length: 12 }, (_, i) => "sample-" + String(i).padStart(2, "0"));
const sourceRows = Object.fromEntries(sourceNames.map((name, source) => [name,
  sampleIds.map((_, row) => Array.from({ length: [6, 12, 10, 3][source] }, (_, column) =>
    Math.sin((row + 1) * (column + 1) * 0.13 + source) + row * (source + 1) / 11))]));
const target = sampleIds.map((_, row) => 1.3 * sourceRows.nir[row][0] -
  0.8 * sourceRows.image[row][4] + 0.4 * sourceRows.series[row][2] + 0.1 * sourceRows.metadata[row][0]);
const groups = Object.fromEntries(sampleIds.map((id, i) => [id, "plant-" + Math.floor(i / 2)]));
const folds = {
  id: "folds:four-sources", sample_ids: sampleIds, sample_groups: groups,
  folds: Array.from({ length: 3 }, (_, fold) => ({
    fold_id: "outer:" + fold,
    train_sample_ids: sampleIds.filter((_, i) => Math.floor(i / 2) % 3 !== fold),
    validation_sample_ids: sampleIds.filter((_, i) => Math.floor(i / 2) % 3 === fold),
    metadata: {},
  })),
};
const envelope = JSON.parse(fs.readFileSync(path.join(repo,
  "crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"), "utf8"));
// Describe the numeric projections actually supplied by this fixture. Raw N-D
// encoders are a separate qualification; do not reuse the single-NIR plan hash.
envelope.plan = {
  id: "four-source-numeric-projections", output_representation: "tabular_numeric", issues: [],
  steps: [...sourceNames.map(source => ({
    kind: "materialize", source_id: source, adapter_id: null, input_representation: null,
    output_representation: "tabular_numeric", fit_scope: "stateless", requires_user_choice: false,
    metadata: { output: "src:" + source },
  })), { kind: "join", source_id: null, adapter_id: null,
    input_representation: "tabular_numeric", output_representation: "tabular_numeric",
    fit_scope: "stateless", requires_user_choice: false,
    metadata: { inputs: sourceNames.map(source => "src:" + source), output: "port:X" } }],
};
envelope.plan_fingerprint = fingerprint(envelope.plan);
envelope.schema_fingerprint = fingerprint(sourceNames.map(source => ({
  source_id: source, representation: "tabular_numeric", width: sourceRows[source][0].length,
})));
envelope.coordinator_relations.records = sampleIds.flatMap(sampleId => sourceNames.map(sourceId => ({
  observation_id: "obs:" + sampleId + ":" + sourceId, sample_id: sampleId,
  target_id: "target:" + sampleId, group_id: groups[sampleId], origin_sample_id: null,
  source_id: sourceId, is_augmented: false,
})));
// Native relation identity includes canonical effective units and defaults.
// The frozen identity was produced by the Python binding to the same Rust core;
// check the fixture bytes before using it rather than replacing its algorithm.
const relationIdentity = JSON.parse(fs.readFileSync(path.join(repo,
  "scripts/fixtures/methods_four_source_relation_identity.json"), "utf8"));
assert.equal(fingerprint(envelope.coordinator_relations), relationIdentity.relation_json_sha256);
envelope.relation_fingerprint = relationIdentity.native_relation_fingerprint;
const fitLog = [];
const resolvers = {
  resolveFeatures: ({ view, task }) => {
    assert.equal(view.source_ids.length, 1);
    const source = view.source_ids[0];
    const rows = view.sample_ids.map(id => sourceRows[source][sampleIds.indexOf(id)]);
    assert.ok(rows.every(Boolean), "No unknown or external sample can enter a training view");
    if (view.partition === "fold_train") fitLog.push({
      variant: task.variant_id, node: task.node_plan.node_id, fold: task.fold_id,
      ids: view.sample_ids, source, width: rows[0].length, alpha: task.node_plan.params.alpha,
    });
    return { sampleIds: [...view.sample_ids],
      matrix: { data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length } };
  },
  resolveTargets: ({ sampleIds: ids }) => ({
    sampleIds: [...ids],
    matrix: { data: Float64Array.from(ids.map(id => target[sampleIds.indexOf(id)])),
      rows: ids.length, cols: 1 },
  }),
};
const operators = Object.fromEntries([...sourceNames, "meta"].map(source =>
  ["model:" + source, { type: "n4m:models.regularized.ridge" }]));
const controller = new N4mWasmRegressionController({ methods, operators, ...resolvers, digest: sha256 });
const manifest = controller.manifest(dagMl);
const binding = source => ({
  node_id: "model:" + source, input_name: "x", request_id: envelope.plan.id,
  schema_fingerprint: envelope.schema_fingerprint, plan_fingerprint: envelope.plan_fingerprint,
  relation_fingerprint: envelope.relation_fingerprint, output_representation: "tabular_numeric",
  feature_set_id: "x", source_ids: [source], require_relations: true,
  view_policy: { fit_partition: "fold_train", predict_partition: "fold_validation",
    include_augmented_train: false, include_augmented_validation: false,
    include_excluded: false, require_sample_ids: true }, metadata: {},
});
const policy = { split_unit: "group", forbid_origin_cross_fold: true,
  allow_observation_split_with_shared_target: false, require_group_ids: true, unsafe_flags: [] };
const dsl = {
  id: "dsl:methods.four-sources", input: { name: "x", representation: "tabular_numeric" },
  campaign_id: "campaign:methods.four-sources", root_seed: 19,
  leakage_policy: policy,
  inner_cv: { kind: "group_kfold", n_splits: 2, shuffle: false, seed: 19 },
  split_invocation: { id: "split:four-sources", controller_id: null,
    leakage_policy: policy, params: {}, fold_set: folds },
  data_bindings: sourceNames.map(binding),
  steps: [{ kind: "branch", branches: sourceNames.map(source => ({
    id: source, steps: [{ kind: "model", id: "model:" + source,
      operator: { type: "n4m:models.regularized.ridge" }, params: { alpha: 0.2 } }],
  })) }, {
    kind: "merge_model", id: "model:meta", operator: { type: "n4m:models.regularized.ridge" },
    params: { alpha: 0.2 }, metadata: { stacking_oof_execution: "nested_oof_v1" },
  }],
};
const compiled = JSON.parse(dagMl.compile_pipeline_dsl_artifact_with_controllers_json(
  JSON.stringify(dsl), JSON.stringify([manifest])));
const plan = dagMl.build_execution_plan_json("plan:methods.four-sources",
  JSON.stringify(compiled.graph), JSON.stringify(compiled.campaign_template), JSON.stringify([manifest]));
const request = {
  target_node: "model:meta", trial_budget: 2, metric: "rmse", direction: "minimize",
  fold_score_reduction: "mean", optimizer_descriptor: { owner: "methods.wasm.four-sources" },
  parameter_bindings: Object.fromEntries([...sourceNames, "meta"].map(source =>
    [source + ".alpha", { node_id: "model:" + source, param_path: "alpha" }])),
};
const space = Object.fromEntries([...sourceNames, "meta"].map(source =>
  [source + ".alpha", { kind: "float", low: 0.05, high: 2 }]));
const options = { sampler: "sobol", pruner: "none", direction: "minimize", metric: "rmse", seed: 7 };
const config = { Optimizer: methods.Optimizer, dagMl, space, options,
  objective: { request, plan: JSON.parse(plan).graph_fingerprint,
    data: fingerprint({ sourceRows, target }) } };
let stored;
const persist = snapshot => { stored = structuredClone(snapshot); };
const run = (optimizer, budget, nativeController = controller) => {
  request.trial_budget = budget;
  return JSON.parse(dagMl.host_hpo_search_json(plan, JSON.stringify([manifest]),
    JSON.stringify(envelope), JSON.stringify(request),
    optimizer.committed === null ? undefined : JSON.stringify(optimizer.committed),
    nativeController.callback, optimizer.callback));
};
let optimizer = new N4mWasmHostOptimizer({ ...config, persist });
const first = run(optimizer, 2);
assert.equal(first.status, "completed");
assert.equal(first.trials.length, 2);
assert.ok(first.trials.every(trial => Number.isFinite(trial.score)));
optimizer.close();
optimizer = new N4mWasmHostOptimizer({ ...config, persist, state: stored });
const resumed = run(optimizer, 3);
assert.deepEqual(resumed.trials.slice(0, 2), first.trials);
optimizer.close();
const uninterruptedOptimizer = new N4mWasmHostOptimizer({ ...config, persist: () => {} });
const freshController = new N4mWasmRegressionController({ methods, operators, ...resolvers, digest: sha256 });
const continuous = run(uninterruptedOptimizer, 3, freshController);
assert.deepEqual(resumed.trials, continuous.trials);
assert.deepEqual(resumed.checkpoint.trials, continuous.checkpoint.trials);
uninterruptedOptimizer.close();
freshController.close();
assert.deepEqual([...new Set(fitLog.map(fit => fit.source))].sort(), [...sourceNames].sort());
for (const fit of fitLog) {
  assert.equal(fit.width, { nir: 6, image: 12, series: 10, metadata: 3 }[fit.source]);
  const externalFold = folds.folds.find(fold => fit.fold.startsWith(fold.fold_id));
  if (externalFold) assert.ok(fit.ids.every(id => !externalFold.validation_sample_ids.includes(id)));
}

// Qualify native archive transport with a selected source estimator. Full
// four-source stacking archive replay remains a separate qualification.
const selected = resumed.trials.find(trial => trial.trial_index === resumed.selected_trial_index);
const refitDsl = structuredClone(dsl);
delete refitDsl.split_invocation;
delete refitDsl.inner_cv;
refitDsl.data_bindings = refitDsl.data_bindings.slice(0, 1);
refitDsl.steps = [{ ...dsl.steps[0].branches[0].steps[0], params: { alpha: selected.params["nir.alpha"] } }];
const refitCompiled = JSON.parse(dagMl.compile_pipeline_dsl_artifact_with_controllers_json(
  JSON.stringify(refitDsl), JSON.stringify([manifest])));
const refitPlan = dagMl.build_execution_plan_json("plan:methods.selected-source.refit",
  JSON.stringify(refitCompiled.graph), JSON.stringify(refitCompiled.campaign_template), JSON.stringify([manifest]));
const trainingEnvelope = { ...envelope, data_content_fingerprint: fingerprint(sourceRows),
  target_content_fingerprint: fingerprint(target) };
const captured = JSON.parse(dagMl.execute_initial_full_refit_json(
  refitPlan, JSON.stringify([manifest]), JSON.stringify(trainingEnvelope), JSON.stringify(sampleIds),
  "package:methods.selected-source.refit", "run:methods.selected-source.refit", "19", controller.callback));
const packageJson = captured.initial_full_refit_package_json;
dagMl.validate_initial_full_refit_package_json(packageJson);
assert.equal(captured.initial_full_refit_package.artifacts.length, 1);
controller.close();
const heldoutId = "heldout-0";
const heldoutRow = sourceRows.nir[0].map(value => value + 0.1);
const replayController = new N4mWasmRegressionController({ methods, operators, ...resolvers, digest: sha256,
  resolveFeatures: ({ view }) => {
    assert.deepEqual(view.sample_ids, [heldoutId]);
    assert.deepEqual(view.source_ids, ["nir"]);
    return { sampleIds: [heldoutId], matrix: { data: Float64Array.from(heldoutRow), rows: 1, cols: heldoutRow.length } };
  },
});
const predictEnvelope = dagMl.initial_full_refit_predict_envelope_json(packageJson, JSON.stringify({
  role: "inference", target_names: ["y"], data_content_fingerprint: fingerprint(heldoutRow),
  target_content_fingerprint: null,
  relations: { records: [{ observation_id: "obs.heldout", sample_id: heldoutId, target_id: null,
    group_id: "plant-heldout", source_id: "nir", origin_sample_id: null, is_augmented: false }] },
}));
const replayOperations = [];
const originalFit = methods.RolePipeline.prototype.fit;
methods.RolePipeline.prototype.fit = () => { throw new Error("Native archive replay must not fit"); };
let replay;
try {
  replay = JSON.parse(dagMl.replay_initial_full_refit_json(packageJson, predictEnvelope,
    JSON.stringify([captured.initial_full_refit_package.outputs[0].output_id]), "{}",
    "run:methods.selected-source.replay", (id, json, seed) => {
      const task = JSON.parse(json);
      replayOperations.push(task.operation ?? task.phase);
      return replayController.callback(id, json, seed);
    }));
} finally {
  methods.RolePipeline.prototype.fit = originalFit;
  replayController.close();
}
assert.deepEqual(replayOperations, ["hydrate_artifact_payload", "PREDICT", "release_hydrated_artifact_payload"]);
const selectedPrediction = replay.replay_outcome.outputs[0].prediction;
assert.deepEqual(selectedPrediction.sample_ids, [heldoutId]);
assert.ok(selectedPrediction.values.flat().every(Number.isFinite));
const nativeRefit = { alpha: selected.params["nir.alpha"], heldoutRow,
  prediction: selectedPrediction.values, operations: replayOperations };
if (process.argv[4]) fs.writeFileSync(process.argv[4], JSON.stringify({
  dsl, envelope, manifest, request, sourceRows, sampleIds, target, resumed, fitLog, nativeRefit,
  dagml_version: dagMl.dag_ml_version(), methods_version: methods.version(), methods_abi: methods.abiVersion(),
}, null, 2) + "\n");
controller.close();
console.log("METHODS_MULTIMODAL_HPO_OK", JSON.stringify({
  dagml: dagMl.dag_ml_version(), methods: methods.version(), sources: sourceNames,
  trials: resumed.trials.length, sourceFitViews: fitLog.length, selected: resumed.selected_trial_index,
  selectedSourceNativeArchiveReplay: true,
}));
