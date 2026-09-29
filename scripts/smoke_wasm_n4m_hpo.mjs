#!/usr/bin/env node
// Real Methods JS/WASM proposals over a DAG-ML/WASM Ridge host HPO run.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

import { N4mWasmHostOptimizer } from "../examples/adapters/hpo_n4m_wasm_adapter.mjs";
import { assertSelectedRidgeRefitReplay } from "./wasm_ridge_refit_oracle.mjs";

const require = createRequire(import.meta.url);
const repo = path.resolve(import.meta.dirname, "..");
const dagPkg = path.resolve(process.argv[2] ?? path.join(repo, "target/wasm/dag-ml-wasm"));
const methodsDist = path.resolve(process.argv[3] ?? path.join(repo, "../nirs4all-methods/bindings/js/dist"));
const dagMl = require(path.join(dagPkg, "dag_ml_wasm.js"));
const { loadModule, Optimizer } = await import(pathToFileURL(path.join(methodsDist, "index.js")).href);
await loadModule();
const { ridgeNodeResult, assertRidgeHpoScores } = require("./hpo_ridge_operator.cjs");

const graph = {
  id: "graph:wasm-n4m-hpo", interface: { inputs: [], outputs: [] },
  nodes: [{ id: "model:m", kind: "model", operator: { type: "JSModel" }, params: {},
    ports: { inputs: [], outputs: [{ name: "oof", kind: "prediction", representation: null,
      cardinality: "one", description: "" }] }, metadata: {}, seed_label: null }],
  edges: [], search_space_fingerprint: null, metadata: {},
};
const manifest = JSON.parse(dagMl.derive_controller_manifest_json(JSON.stringify({
  controller_id: "controller:js-hpo-model", controller_version: "1.0.0", operator_kind: "model",
  input_ports: [], output_ports: graph.nodes[0].ports.outputs,
})));
const folds = {
  id: "folds:wasm-n4m-hpo", sample_ids: ["s1", "s2"], sample_groups: {},
  folds: [
    { fold_id: "fold:0", train_sample_ids: ["s2"], validation_sample_ids: ["s1"], metadata: {} },
    { fold_id: "fold:1", train_sample_ids: ["s1"], validation_sample_ids: ["s2"], metadata: {} },
  ],
};
const campaign = {
  id: "campaign:wasm-n4m-hpo", root_seed: 7,
  split_invocation: { id: "split:outer", controller_id: null,
    leakage_policy: { split_unit: "sample", forbid_origin_cross_fold: true,
      allow_observation_split_with_shared_target: false, require_group_ids: false, unsafe_flags: [] },
    params: {}, fold_set: folds },
};
const plan = dagMl.build_execution_plan_json(
  "plan:wasm-n4m-hpo", JSON.stringify(graph), JSON.stringify(campaign), JSON.stringify([manifest]));
const envelope = fs.readFileSync(path.join(repo, "crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"), "utf8");
const request = { target_node: "model:m", trial_budget: 2, metric: "rmse", direction: "minimize",
  optimizer_descriptor: { owner: "methods-js-wasm" } };
const config = { Optimizer, dagMl, space: { offset: { kind: "int", low: 1, high: 3 } },
  options: { sampler: "sobol", pruner: "none", direction: "minimize", metric: "rmse", seed: 19 },
  objective: { target_node: request.target_node, metric: request.metric,
    graph_fingerprint: JSON.parse(plan).graph_fingerprint },
  warmStart: { offset: 1 } };
const controller = (controllerId, taskJson) => ridgeNodeResult(controllerId, taskJson, folds, "js-n4m-hpo");
let stored;
const persist = snapshot => { stored = structuredClone(snapshot); };
const run = (adapter, budget, checkpoint = null, callback = adapter.callback) => {
  request.trial_budget = budget;
  return JSON.parse(dagMl.host_hpo_search_json(plan, JSON.stringify([manifest]), envelope,
    JSON.stringify(request), checkpoint === null ? undefined : JSON.stringify(checkpoint), controller, callback));
};

const firstAdapter = new N4mWasmHostOptimizer({ ...config, persist });
const first = run(firstAdapter, 2);
assert.equal(first.checkpoint.trials.length, 2);
assertRidgeHpoScores(first);
const firstState = structuredClone(stored);
firstAdapter.close();

const resumedAdapter = new N4mWasmHostOptimizer({ ...config, persist, state: stored });
assert.deepEqual(resumedAdapter.committed, first.checkpoint);
const resumed = run(resumedAdapter, 3, resumedAdapter.committed);
assert.equal(resumed.checkpoint.trials.length, 3);
assert.deepEqual(resumed.trials.slice(0, 2), first.trials);
assertRidgeHpoScores(resumed);
resumedAdapter.close();

let continuousState;
const continuous = new N4mWasmHostOptimizer({ ...config, persist: snapshot => { continuousState = snapshot; } });
const continuousResult = run(continuous, 3);
assert.deepEqual(resumed.trials, continuousResult.trials);
assert.deepEqual(resumed.checkpoint.trials, continuousResult.checkpoint.trials);
assert.equal(continuousState.committed.trials.length, 3);
continuous.close();

let crashState;
const interrupted = new N4mWasmHostOptimizer({ ...config, persist: snapshot => { crashState = structuredClone(snapshot); } });
assert.throws(() => run(interrupted, 2, null, (operation, payload) => {
  const reply = interrupted.callback(operation, payload);
  if (operation === "prepare_terminal") throw new Error("simulated crash after prepared checkpoint");
  return reply;
}), /simulated crash|prepare_terminal/);
assert.equal(crashState.committed.trials.length, 0);
assert.equal(crashState.prepared.trials.length, 1);
interrupted.close();
const recovered = new N4mWasmHostOptimizer({ ...config,
  persist: snapshot => { crashState = structuredClone(snapshot); }, state: crashState });
assert.equal(recovered.committed.trials.length, 1);
const afterCrash = run(recovered, 2, recovered.committed);
assert.deepEqual(afterCrash.trials, first.trials);
recovered.close();

let orphanState;
const orphaned = new N4mWasmHostOptimizer({ ...config,
  persist: snapshot => { orphanState = structuredClone(snapshot); } });
assert.throws(() => run(orphaned, 2, null, (operation, payload) => {
  const reply = orphaned.callback(operation, payload);
  if (operation === "ask") throw new Error("simulated crash after Methods ask");
  return reply;
}), /simulated crash|ask/);
assert.equal(orphanState.committed.trials.length, 0);
orphaned.close();
const recoveredOrphan = new N4mWasmHostOptimizer({ ...config,
  persist: snapshot => { orphanState = structuredClone(snapshot); }, state: orphanState });
assert.equal(recoveredOrphan.committed.trials[0].state, "failed");
const afterOrphan = run(recoveredOrphan, 2, recoveredOrphan.committed);
assert.deepEqual(afterOrphan.checkpoint.trials.map(trial => trial.state), ["failed", "complete"]);
recoveredOrphan.close();

// A native pruning decision can change the in-memory Methods trial before
// DAG-ML prepares its terminal. The last durable snapshot must still be
// recoverable if the process stops in that interval.
let pruneState;
let sawPrune = false;
let crashedDuringPrune = false;
const pruneConfig = { ...config, options: { ...config.options, pruner: "median", startupTrials: 1 } };
request.progressive_pruning = true;
const prunedMidReport = new N4mWasmHostOptimizer({ ...pruneConfig,
  persist: snapshot => { pruneState = structuredClone(snapshot); } });
assert.throws(() => run(prunedMidReport, 3, null, (operation, payload) => {
  if (crashedDuringPrune) throw new Error("simulated process unavailable after native prune");
  const reply = prunedMidReport.callback(operation, payload);
  if (operation === "report_intermediate" && JSON.parse(reply).prune) {
    sawPrune = true;
    crashedDuringPrune = true;
    throw new Error("simulated crash after native prune decision");
  }
  return reply;
}));
assert.equal(sawPrune, true);
assert.equal(pruneState.committed.trials.length, 1);
const storedPruner = Optimizer.load(Uint8Array.from(pruneState.n4mopt), pruneConfig.space);
assert.equal(storedPruner.trialRecords().at(-1).status, "running");
storedPruner.dispose();
prunedMidReport.close();
const recoveredPrune = new N4mWasmHostOptimizer({ ...pruneConfig,
  persist: snapshot => { pruneState = structuredClone(snapshot); }, state: pruneState });
assert.equal(recoveredPrune.committed.trials.at(-1).state, "failed");
const afterPruneCrash = run(recoveredPrune, 3, recoveredPrune.committed);
assert.equal(afterPruneCrash.checkpoint.trials.length, 3);
recoveredPrune.close();

let preparedPruneState;
let crashedAfterPrunePrepare = false;
const prunedPrepared = new N4mWasmHostOptimizer({ ...pruneConfig,
  persist: snapshot => { preparedPruneState = structuredClone(snapshot); } });
assert.throws(() => run(prunedPrepared, 3, null, (operation, payload) => {
  if (crashedAfterPrunePrepare) throw new Error("simulated process unavailable after prune prepare");
  const reply = prunedPrepared.callback(operation, payload);
  if (operation === "prepare_terminal" && JSON.parse(payload).checkpoint.trials.at(-1).state === "pruned") {
    crashedAfterPrunePrepare = true;
    throw new Error("simulated crash after pruned terminal preparation");
  }
  return reply;
}));
assert.equal(crashedAfterPrunePrepare, true);
assert.equal(preparedPruneState.prepared.trials.at(-1).state, "pruned");
prunedPrepared.close();
const recoveredPreparedPrune = new N4mWasmHostOptimizer({ ...pruneConfig,
  persist: snapshot => { preparedPruneState = structuredClone(snapshot); }, state: preparedPruneState });
assert.equal(recoveredPreparedPrune.committed.trials.at(-1).state, "pruned");
assert.equal(run(recoveredPreparedPrune, 3, recoveredPreparedPrune.committed).checkpoint.trials.length, 3);
recoveredPreparedPrune.close();
request.progressive_pruning = false;

const damaged = structuredClone(firstState);
damaged.n4mopt[damaged.n4mopt.length - 1] ^= 1;
assert.throws(() => new N4mWasmHostOptimizer({ ...config, persist, state: damaged }));
assert.throws(() => new N4mWasmHostOptimizer({ ...config, options: { ...config.options, seed: 20 },
  persist, state: firstState }), /snapshot contract mismatch/);

let refitQualified = false;
if (typeof dagMl.execute_initial_full_refit_json === "function") {
  const refitFixture = JSON.parse(fs.readFileSync(path.join(repo, "crates/dag-ml-core/tests/fixtures/initial_full_refit/package.json"), "utf8"));
  await assertSelectedRidgeRefitReplay(dagMl, resumed, refitFixture);
  refitQualified = true;
} else if (process.env.DAGML_ALLOW_OLD_WASM !== "1") {
  throw new Error("Current DAG-ML WASM build is required for selected Ridge refit/replay");
}
console.log(`WASM_N4M_HOST_HPO_OK 3 trials, resume, prepared/orphan/prune recovery, tamper refusal, refit/replay=${refitQualified}`);
