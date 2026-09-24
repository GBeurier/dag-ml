// Complete a real JS Ridge HPO candidate with native no-CV REFIT and replay.
// Used by both Node WASM and actual Chrome WASM smokes.
const rows = Object.freeze({
  "sample:1": {x: 1, y: 1},
  "sample:2": {x: 2, y: 2},
  "sample:heldout:1": {x: 3, y: 3},
});

function idsFor(task, partition) {
  const views = Object.values(task.data_views).filter(view => view.partition === partition);
  if (views.length !== 1 || !views[0].sample_ids?.length) {
    throw new Error(`Ridge ${task.phase} has no unique ${partition} view`);
  }
  return views[0].sample_ids;
}

function nativeStage(label, invoke) {
  try { return invoke(); }
  catch (error) { throw new Error(`${label}: ${String(error)}`); }
}

async function sha256Hex(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

export async function assertSelectedRidgeRefitReplay(dagMl, hpo, fixture) {
  if (hpo.selected_trial_index !== 0 || hpo.trials[0].score !== 0) {
    throw new Error("Ridge REFIT must consume the selected exact-fit HPO candidate");
  }
  const selectedOffset = hpo.trials[0].params.offset;
  const graph = structuredClone(fixture.effective_plan.graph_plan.graph);
  graph.nodes[0].operator = {type: "JsRidge"};
  graph.nodes[0].params = {offset: selectedOffset};
  const campaign = structuredClone(fixture.effective_plan.campaign);
  const manifests = Object.values(fixture.effective_plan.controller_manifests);
  const plan = dagMl.build_execution_plan_json(
    "plan:js-ridge-selected-refit", JSON.stringify(graph),
    JSON.stringify(campaign), JSON.stringify(manifests));
  const artifactId = "artifact:model:initial:js-ridge:refit";
  const trainingIds = fixture.training_sample_ids;
  const alpha = selectedOffset - 1;
  const trainingRows = trainingIds.map(id => rows[id]);
  if (trainingRows.some(row => !row)) throw new Error("unknown full-train Ridge sample");
  const expectedSidecar = {weight: trainingRows.reduce((sum, row) => sum + row.x * row.y, 0) /
    (trainingRows.reduce((sum, row) => sum + row.x * row.x, 0) + alpha),
    alpha, trainIds: [...trainingIds]};
  const artifactFingerprint = await sha256Hex(JSON.stringify(expectedSidecar));
  let sidecar = null;
  let refitCalls = 0;
  let replayCalls = 0;
  const invoke = (controllerId, taskJson, exactSeed) => {
    const task = JSON.parse(taskJson);
    if (!exactSeed || task.node_plan.params.offset !== selectedOffset) {
      throw new Error("Ridge REFIT lost its selected parameter or exact native seed");
    }
    const isRefit = task.phase === "REFIT";
    const ids = idsFor(task, isRefit ? "full_train" : "predict");
    const values = ids.map(id => {
      const row = rows[id];
      if (!row) throw new Error(`Ridge REFIT unknown sample ${id}`);
      return row;
    });
    let artifacts = [];
    let artifactHandles = {};
    if (isRefit) {
      refitCalls++;
      const xy = values.reduce((sum, row) => sum + row.x * row.y, 0);
      const xx = values.reduce((sum, row) => sum + row.x * row.x, 0);
      sidecar = {weight: xy / (xx + selectedOffset - 1),
        alpha: selectedOffset - 1, trainIds: [...ids]};
      if (JSON.stringify(sidecar) !== JSON.stringify(expectedSidecar)) {
        throw new Error("Ridge REFIT was not fitted on the full training cohort");
      }
      artifacts = [{id: artifactId, kind: "js_ridge_model", controller_id: controllerId,
        backend: "json", uri: "artifacts/js-ridge.json",
        content_fingerprint: artifactFingerprint,
        size_bytes: JSON.stringify(sidecar).length,
        plugin: "dagml.js_ridge_oracle", plugin_version: "1.0.0"}];
      artifactHandles = {[artifactId]: {handle: 7001, kind: "model", owner_controller: controllerId}};
    } else if (task.phase === "PREDICT") {
      replayCalls++;
      const inputs = Object.values(task.artifact_inputs);
      const handles = Object.values(task.input_handles).filter(handle => handle.kind === "model");
      if (!sidecar || inputs.length !== 1 || handles.length !== 1 ||
          inputs[0].artifact.id !== artifactId ||
          inputs[0].artifact.content_fingerprint !== artifactFingerprint ||
          sidecar.alpha !== selectedOffset - 1) {
        throw new Error("JS Ridge replay did not receive its exact selected sidecar");
      }
    } else {
      throw new Error(`unexpected JS Ridge phase ${task.phase}`);
    }
    return JSON.stringify({
      node_id: task.node_plan.node_id,
      outputs: {oof: {handle: 7002, kind: "prediction", owner_controller: controllerId}},
      predictions: [{producer_node: task.node_plan.node_id, partition: "final",
        fold_id: null, sample_ids: ids,
        values: values.map(row => [sidecar.weight * row.x]), target_names: ["y"]}],
      artifacts, artifact_handles: artifactHandles,
      lineage: {
        record_id: `lineage:js-ridge:${task.phase}`, run_id: task.run_id,
        node_id: task.node_plan.node_id, phase: task.phase,
        controller_id: controllerId, controller_version: task.node_plan.controller_version,
        variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
        input_lineage: [], artifact_refs: artifacts,
        params_fingerprint: task.node_plan.params_fingerprint,
        data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null,
        seed: null, unsafe_flags: [], metrics: {},
        loss_attestations: [], early_stopping_records: [],
      },
    });
  };
  const captureJson = dagMl.execute_initial_full_refit_json(
    plan, JSON.stringify(manifests), JSON.stringify(fixture.training_envelope),
    JSON.stringify(fixture.training_sample_ids), "package:js-ridge-selected-refit",
    "run:js-ridge.refit", "7", invoke);
  const capture = JSON.parse(captureJson);
  const pkg = capture.initial_full_refit_package;
  const packageJson = capture.initial_full_refit_package_json;
  if (typeof packageJson !== "string") {
    throw new Error("WASM REFIT did not return lossless package JSON");
  }
  nativeStage("validate selected Ridge package", () =>
    dagMl.validate_initial_full_refit_package_json(packageJson));
  if (refitCalls !== 1 || pkg.artifacts.length !== 1 || sidecar.weight !== 1 ||
      JSON.stringify(sidecar.trainIds) !== JSON.stringify(fixture.training_sample_ids)) {
    throw new Error("selected JS Ridge was not fitted once on all training samples");
  }
  if (await sha256Hex(JSON.stringify(sidecar)) !==
      pkg.artifacts[0].record.artifact.content_fingerprint) {
    throw new Error("JS Ridge sidecar content disagrees with the native package");
  }
  const cohort = {
    role: "external_test", relations: {records: [{
      observation_id: "obs.H001", sample_id: "sample:heldout:1",
      target_id: "target:heldout:1", group_id: "group:heldout",
      origin_sample_id: null, source_id: "nir", is_augmented: false,
    }]}, target_names: ["y"], data_content_fingerprint: "a".repeat(64),
    target_content_fingerprint: "b".repeat(64),
  };
  const envelope = nativeStage("make selected Ridge replay envelope", () =>
    dagMl.initial_full_refit_predict_envelope_json(
      packageJson, JSON.stringify(cohort)));
  const replay = JSON.parse(nativeStage("replay selected Ridge package", () =>
    dagMl.replay_initial_full_refit_json(
    packageJson, envelope, JSON.stringify([pkg.outputs[0].output_id]),
    JSON.stringify(capture.node_results[0].artifact_handles),
    "run:js-ridge.predict", invoke)));
  const prediction = replay.replay_outcome.outputs[0].prediction;
  if (replayCalls !== 1 || prediction.sample_ids[0] !== "sample:heldout:1" ||
      prediction.values[0][0] !== 3) {
    throw new Error("selected JS Ridge replay returned the wrong held-out prediction");
  }
  const missingHandles = {};
  try {
    dagMl.replay_initial_full_refit_json(
      packageJson, envelope, JSON.stringify([pkg.outputs[0].output_id]),
      JSON.stringify(missingHandles), "run:js-ridge.missing", invoke);
    throw new Error("JS Ridge replay accepted a missing artifact handle");
  } catch (error) {
    if (!String(error).includes("handles must exactly cover")) throw error;
  }
  if (replayCalls !== 1) throw new Error("JS controller ran after missing handle refusal");
}
