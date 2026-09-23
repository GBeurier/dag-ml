"use strict";

const path = require("path");
const { parentPort, workerData } = require("worker_threads");
const dagMl = require(path.join(workerData.pkgDir, "dag_ml_wasm.js"));

const controller = (controllerId, taskJson) => {
  const task = JSON.parse(taskJson);
  const sample = task.fold_id === "fold:0" ? "s1" : "s2";
  const target = sample === "s1" ? 1 : 2;
  return JSON.stringify({
    node_id: task.node_plan.node_id, outputs: {},
    predictions: [{ producer_node: task.node_plan.node_id, partition: "validation",
      fold_id: task.fold_id, sample_ids: [sample],
      values: [[target + task.node_plan.params.offset]], target_names: ["y"] }],
    regression_targets: [{ level: "sample", unit_ids: [{ level: "sample", id: sample }],
      values: [[target]], target_names: ["y"] }],
    lineage: {
      record_id: `lineage:js-hpo-worker:${task.variant_id}:${task.fold_id}`,
      run_id: task.run_id, node_id: task.node_plan.node_id, phase: task.phase,
      controller_id: controllerId, controller_version: task.node_plan.controller_version,
      variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
      input_lineage: [], artifact_refs: [], params_fingerprint: task.node_plan.params_fingerprint,
      data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null,
      seed: null, unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: [],
    },
  });
};

parentPort.on("message", (taskJson) => {
  try {
    const packet = JSON.parse(taskJson);
    const result = packet.kind === "fold"
      ? dagMl.host_hpo_evaluate_worker_fold_json(
        JSON.stringify(packet.task), packet.fold_index, workerData.manifests,
        workerData.envelope, workerData.request, controller,
      )
      : dagMl.host_hpo_evaluate_worker_task_json(
        packet.kind === "complete" ? JSON.stringify(packet.task) : taskJson,
        workerData.manifests, workerData.envelope, workerData.request, controller,
      );
    parentPort.postMessage({ result });
  } catch (error) {
    parentPort.postMessage({ error: String(error) });
  }
});
