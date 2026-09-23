"use strict";

const fs = require("fs");
const path = require("path");
const { ridgeNodeResult, assertRidgeHpoScores } = require("./hpo_ridge_operator.cjs");

module.exports = async function smokeHostHpo(dagMl, repo, pkgDir) {
  const graph = {
    id: "graph:wasm-hpo", interface: { inputs: [], outputs: [] },
    nodes: [{
      id: "model:m", kind: "model", operator: { type: "JSModel" }, params: {},
      ports: { inputs: [], outputs: [{ name: "oof", kind: "prediction", representation: null, cardinality: "one", description: "" }] },
      metadata: {}, seed_label: null,
    }],
    edges: [], search_space_fingerprint: null, metadata: {},
  };
  const manifest = JSON.parse(dagMl.derive_controller_manifest_json(JSON.stringify({
    controller_id: "controller:js-hpo-model", controller_version: "1.0.0", operator_kind: "model",
    input_ports: [], output_ports: graph.nodes[0].ports.outputs,
  })));
  const folds = {
    id: "folds:wasm-hpo", sample_ids: ["s1", "s2"], sample_groups: {},
    folds: [
      { fold_id: "fold:0", train_sample_ids: ["s2"], validation_sample_ids: ["s1"], metadata: {} },
      { fold_id: "fold:1", train_sample_ids: ["s1"], validation_sample_ids: ["s2"], metadata: {} },
    ],
  };
  const campaign = {
    id: "campaign:wasm-hpo", root_seed: 7,
    split_invocation: {
      id: "split:outer", controller_id: null,
      leakage_policy: { split_unit: "sample", forbid_origin_cross_fold: true,
        allow_observation_split_with_shared_target: false, require_group_ids: false, unsafe_flags: [] },
      params: {}, fold_set: folds,
    },
  };
  const plan = dagMl.build_execution_plan_json(
    "plan:wasm-hpo", JSON.stringify(graph), JSON.stringify(campaign), JSON.stringify([manifest]),
  );
  const envelope = fs.readFileSync(path.join(repo, "crates", "dag-ml-core", "tests", "fixtures", "package", "data", "coordinator_data_plan_envelope_sample12.json"), "utf8");
  const request = {
    target_node: "model:m", trial_budget: 2, metric: "rmse", direction: "minimize",
    optimizer_descriptor: { owner: "wasm-smoke" },
  };
  const prepared = [];
  let lastPrepared = null;
  const published = [];
  const told = [];
  const optimizer = (operation, payloadJson) => {
    const payload = JSON.parse(payloadJson);
    if (operation === "ask") return JSON.stringify({ params: { offset: payload.trial_index + 1 } });
    if (operation === "prepare_terminal") {
      prepared.push(payload.checkpoint.trials.length);
      lastPrepared = payload.checkpoint;
      return JSON.stringify({ ok: true });
    }
    if (operation === "checkpoint") {
      published.push(payload.checkpoint.trials.length);
      return JSON.stringify({ continue: true });
    }
    if (operation === "tell") {
      if (prepared.at(-1) !== payload.trial_index + 1) throw new Error("WASM optimizer was told before checkpoint preparation");
      told.push(payload.trial_index);
      return JSON.stringify({ ok: true });
    }
    if (operation === "report_intermediate") return JSON.stringify({ prune: false });
    return JSON.stringify({ ok: true });
  };
  const controller = (controllerId, taskJson) => ridgeNodeResult(controllerId, taskJson, folds);
  const first = JSON.parse(dagMl.host_hpo_search_json(
    plan, JSON.stringify([manifest]), envelope, JSON.stringify(request), undefined, controller, optimizer,
  ));
  if (first.selected_trial_index !== 0 || first.checkpoint.trials.length !== 2
      || prepared.join(",") !== "1,2" || published.join(",") !== "0,1,2" || told.join(",") !== "0,1") {
    throw new Error("WASM HPO did not select and journal two native trials");
  }
  assertRidgeHpoScores(first);
  const recovered = JSON.parse(dagMl.recover_host_hpo_checkpoint_json(
    JSON.stringify(first.checkpoint), "null", "[]",
  ));
  if (recovered.fingerprint !== first.checkpoint.fingerprint) {
    throw new Error("WASM HPO checkpoint recovery changed a completed checkpoint");
  }
  request.trial_budget = 3;
  const resumed = JSON.parse(dagMl.host_hpo_search_json(
    plan, JSON.stringify([manifest]), envelope, JSON.stringify(request),
    JSON.stringify(first.checkpoint), controller, optimizer,
  ));
  if (resumed.selected_trial_index !== 0 || resumed.checkpoint.trials.length !== 3
      || prepared.join(",") !== "1,2,3" || told.join(",") !== "0,1,2") {
    throw new Error("WASM HPO resume replayed historical trials or changed selection");
  }
  assertRidgeHpoScores(resumed);
  const interruptedAfterNativeEvaluation = JSON.parse(dagMl.recover_host_hpo_checkpoint_json(
    JSON.stringify(first.checkpoint), JSON.stringify(lastPrepared), "[]",
  ));
  if (interruptedAfterNativeEvaluation.fingerprint !== resumed.checkpoint.fingerprint) {
    throw new Error("WASM HPO failed to recover a prepared terminal after optimizer transition");
  }

  const { Worker } = require("worker_threads");
  const workers = [0, 1].map(() => new Worker(path.join(__dirname, "smoke_wasm_hpo_worker.cjs"), {
    workerData: { pkgDir, manifests: JSON.stringify([manifest]), envelope, request: JSON.stringify(request), folds },
  }));
  const busy = new Set();
  let maximumInFlight = 0;
  const dispatchedFolds = [];
  const dispatchedComplete = [];
  const dispatch = (taskJson) => new Promise((resolve, reject) => {
    const packet = JSON.parse(taskJson);
    if (packet.kind === "fold") dispatchedFolds.push([packet.task.trial_index, packet.fold_index]);
    if (packet.kind === "complete") dispatchedComplete.push(packet.task.trial_index);
    const worker = workers.find((candidate) => !busy.has(candidate));
    if (!worker) return reject(new Error("native window dispatched beyond worker bound"));
    busy.add(worker);
    maximumInFlight = Math.max(maximumInFlight, busy.size);
    const finish = () => {
      worker.removeListener("message", onMessage);
      worker.removeListener("error", onError);
      busy.delete(worker);
    };
    const onMessage = (message) => {
      finish();
      if (message.error) reject(new Error(message.error));
      else resolve(message.result);
    };
    const onError = (error) => { finish(); reject(error); };
    worker.once("message", onMessage);
    worker.once("error", onError);
    worker.postMessage(taskJson);
  });
  const parallelPrepared = [];
  const parallelTold = [];
  const parallelOptimizer = (operation, payloadJson) => {
    const payload = JSON.parse(payloadJson);
    if (operation === "ask") return JSON.stringify({ params: { offset: payload.trial_index + 1 } });
    if (operation === "prepare_terminal") {
      parallelPrepared.push(payload.checkpoint.trials.length);
      return JSON.stringify({ ok: true });
    }
    if (operation === "tell") {
      if (parallelPrepared[parallelPrepared.length - 1] !== payload.trial_index + 1) {
        throw new Error("worker trial was told before native checkpoint preparation");
      }
      parallelTold.push(payload.trial_index);
      return JSON.stringify({ ok: true });
    }
    if (operation === "checkpoint") return JSON.stringify({ continue: true });
    return JSON.stringify({ ok: true });
  };
  try {
    request.trial_budget = 2;
    const parallel = JSON.parse(await dagMl.host_hpo_search_parallel_json(
      plan, JSON.stringify([manifest]), envelope, JSON.stringify(request), undefined,
      2, dispatch, parallelOptimizer,
    ));
    if (parallel.status !== "completed" || parallel.selected_trial_index !== 0
        || parallel.checkpoint.trials.length !== 2 || maximumInFlight !== 2
        || parallelTold.join(",") !== "0,1") {
      throw new Error("WASM worker HPO did not execute two concurrent candidates with ordered native terminalization");
    }
    assertRidgeHpoScores(parallel);
    request.trial_budget = 3;
    const resumedParallel = JSON.parse(await dagMl.host_hpo_search_parallel_json(
      plan, JSON.stringify([manifest]), envelope, JSON.stringify(request),
      JSON.stringify(parallel.checkpoint), 2, dispatch, parallelOptimizer,
    ));
    if (resumedParallel.checkpoint.trials.length !== 3 || parallelTold.join(",") !== "0,1,2"
        || resumedParallel.selected_trial_index !== 0) {
      throw new Error("WASM worker HPO replayed historical candidates or changed selection on resume");
    }
    assertRidgeHpoScores(resumedParallel);

    request.trial_budget = 2;
    request.progressive_pruning = true;
    const pruningEvents = [];
    const pruningTerminal = [];
    const pruningOptimizer = (operation, payloadJson) => {
      const payload = JSON.parse(payloadJson);
      if (operation === "ask") return JSON.stringify({ params: { offset: payload.trial_index + 1 } });
      if (operation === "report_intermediate") {
        pruningEvents.push([payload.trial_index, payload.step, payload.score]);
        return JSON.stringify({ prune: payload.trial_index === 1 && payload.step === 0 });
      }
      if (operation === "tell" || operation === "pruned") pruningTerminal.push([operation, payload.trial_index]);
      if (operation === "checkpoint") return JSON.stringify({ continue: true });
      return JSON.stringify({ ok: true });
    };
    const pruned = JSON.parse(await dagMl.host_hpo_search_parallel_json(
      plan, JSON.stringify([manifest]), envelope, JSON.stringify(request), undefined,
      2, dispatch, pruningOptimizer,
    ));
    if (pruned.status !== "completed" || pruned.checkpoint.trials.length !== 2
        || pruned.pruned_trials.length !== 1 || pruned.pruned_trials[0].trial_index !== 1
        || pruningEvents.map((event) => event.slice(0, 2).join(":")).join(",") !== "0:0,1:0,0:1"
        || dispatchedFolds.map((event) => event.join(":")).join(",") !== "0:0,1:0,0:1"
        || dispatchedComplete.join(",") !== "0"
        || pruningTerminal.map((event) => event.join(":")).join(",") !== "tell:0,pruned:1") {
      throw new Error("WASM worker pruning did not stop candidate 1 before its second fold");
    }
    assertRidgeHpoScores(pruned);
  } finally {
    await Promise.all(workers.map((worker) => worker.terminate()));
  }
};
