import init, * as dagMl from "/pkg/dag_ml_wasm.js";

const ready = init();
self.onmessage = async event => {
  try {
    await ready;
    const {taskJson, manifests, envelope, request} = event.data;
    const result = dagMl.host_hpo_evaluate_worker_task_json(
      taskJson, manifests, envelope, request, (controllerId, nodeTaskJson) => {
        const task = JSON.parse(nodeTaskJson);
        const sample = task.fold_id === "fold:0" ? "s1" : "s2";
        const target = sample === "s1" ? 1 : 2;
        return JSON.stringify({
          node_id: task.node_plan.node_id, outputs: {},
          predictions: [{producer_node: task.node_plan.node_id, partition: "validation",
            fold_id: task.fold_id, sample_ids: [sample],
            values: [[target + task.node_plan.params.offset]], target_names: ["y"]}],
          regression_targets: [{level: "sample", unit_ids: [{level: "sample", id: sample}],
            values: [[target]], target_names: ["y"]}],
          lineage: {
            record_id: "lineage:browser-hpo:" + task.variant_id + ":" + task.fold_id,
            run_id: task.run_id, node_id: task.node_plan.node_id, phase: task.phase,
            controller_id: controllerId, controller_version: task.node_plan.controller_version,
            variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
            input_lineage: [], artifact_refs: [], params_fingerprint: task.node_plan.params_fingerprint,
            data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null, seed: null,
            unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: [],
          },
        });
      });
    self.postMessage({result});
  } catch (error) {
    self.postMessage({error: String(error.stack || error)});
  }
};
