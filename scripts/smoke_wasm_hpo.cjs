"use strict";

const fs = require("fs");
const path = require("path");

module.exports = function smokeHostHpo(dagMl, repo) {
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
  const published = [];
  const told = [];
  const optimizer = (operation, payloadJson) => {
    const payload = JSON.parse(payloadJson);
    if (operation === "ask") return JSON.stringify({ params: { offset: payload.trial_index + 1 } });
    if (operation === "prepare_terminal") {
      prepared.push(payload.checkpoint.trials.length);
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
  const controller = (controllerId, taskJson) => {
    const task = JSON.parse(taskJson);
    const sample = task.fold_id === "fold:0" ? "s1" : "s2";
    const target = sample === "s1" ? 1 : 2;
    return JSON.stringify({
      node_id: task.node_plan.node_id, outputs: {},
      predictions: [{ producer_node: task.node_plan.node_id, partition: "validation",
        fold_id: task.fold_id, sample_ids: [sample], values: [[target + task.node_plan.params.offset]], target_names: ["y"] }],
      regression_targets: [{ level: "sample", unit_ids: [{ level: "sample", id: sample }],
        values: [[target]], target_names: ["y"] }],
      lineage: {
        record_id: `lineage:js-hpo:${task.variant_id}:${task.fold_id}`,
        run_id: task.run_id, node_id: task.node_plan.node_id, phase: task.phase,
        controller_id: controllerId, controller_version: task.node_plan.controller_version,
        variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
        input_lineage: [], artifact_refs: [], params_fingerprint: task.node_plan.params_fingerprint,
        data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null, seed: null,
        unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: [],
      },
    });
  };
  const first = JSON.parse(dagMl.host_hpo_search_json(
    plan, JSON.stringify([manifest]), envelope, JSON.stringify(request), undefined, controller, optimizer,
  ));
  if (first.selected_trial_index !== 0 || first.checkpoint.trials.length !== 2
      || prepared.join(",") !== "1,2" || published.join(",") !== "0,1,2" || told.join(",") !== "0,1") {
    throw new Error("WASM HPO did not select and journal two native trials");
  }
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
};
