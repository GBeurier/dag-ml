// Browser-local ridge model used by the real Web Worker HPO smoke. The host
// owns features/targets and fits only the fold-train IDs supplied in the plan.
const samples = {
  s1: {x: 1, y: 1},
  s2: {x: 2, y: 2},
};

function fitRidge(trainIds, alpha) {
  if (!Number.isFinite(alpha) || alpha < 0 || !trainIds.length) {
    throw new Error("ridge requires training rows and nonnegative regularization");
  }
  let xy = 0;
  let xx = 0;
  for (const id of trainIds) {
    const row = samples[id];
    if (!row) throw new Error(`unknown train sample ${id}`);
    xy += row.x * row.y;
    xx += row.x * row.x;
  }
  return {weight: xy / (xx + alpha), alpha};
}

export function ridgeNodeResult(controllerId, taskJson, foldSet) {
  const task = JSON.parse(taskJson);
  const fold = foldSet.folds.find(item => item.fold_id === task.fold_id);
  if (task.phase !== "FIT_CV" || !fold) throw new Error("ridge requires a declared FIT_CV fold");
  const trainIds = fold.train_sample_ids;
  const validationIds = fold.validation_sample_ids;
  if (trainIds.some(id => validationIds.includes(id))) throw new Error("validation sample leaked into ridge training");
  const model = fitRidge(trainIds, task.node_plan.params.offset - 1);
  const rows = validationIds.map(id => samples[id]);
  if (rows.some(row => !row)) throw new Error("unknown validation sample");
  return JSON.stringify({
    node_id: task.node_plan.node_id, outputs: {},
    predictions: [{producer_node: task.node_plan.node_id, partition: "validation",
      fold_id: task.fold_id, sample_ids: validationIds,
      values: rows.map(row => [model.weight * row.x]), target_names: ["y"]}],
    regression_targets: [{level: "sample", unit_ids: validationIds.map(id => ({level: "sample", id})),
      values: rows.map(row => [row.y]), target_names: ["y"]}],
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
}

export function assertRidgeHpoScores(result) {
  for (const trial of result.trials) {
    const alpha = trial.params.offset - 1;
    const expected = {"fold:0": alpha / (4 + alpha), "fold:1": 2 * alpha / (1 + alpha)};
    const reports = Object.fromEntries(trial.scores.reports
      .filter(report => report.partition === "validation" && report.fold_id.startsWith("fold:"))
      .map(report => [report.fold_id, report.metrics.rmse]));
    for (const [foldId, score] of Object.entries(expected)) {
      if (Math.abs(reports[foldId] - score) > 1e-9) {
        throw new Error(`ridge ${foldId} score ${reports[foldId]} differs from ${score}`);
      }
    }
    const pooled = Math.hypot(...Object.values(expected)) / Math.sqrt(2);
    if (Math.abs(trial.score - pooled) > 1e-9) {
      throw new Error(`ridge OOF score ${trial.score} differs from ${pooled}`);
    }
  }
  if (result.trials.length && (result.selected_trial_index !== 0 || result.trials[0].score !== 0)) {
    throw new Error("ridge HPO did not select its exact-fit first candidate");
  }
}
