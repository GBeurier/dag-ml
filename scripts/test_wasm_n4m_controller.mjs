#!/usr/bin/env node
/** Real Methods fits: adversarial views, durable states and no-fit replay. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";

const packageDir = path.resolve(process.argv[2]);
const methods = await import(pathToFileURL(path.join(path.resolve(process.argv[3]), "index.js")).href);
await methods.loadModule();
const { N4mWasmRegressionController } = await import(pathToFileURL(path.join(packageDir, "n4m_controller.mjs")).href);
const controllerId = "controller:methods.wasm.regression";
const operators = { "model:nir": { type: "n4m:models.regularized.ridge" } };
const ids = ["s0", "s1", "s2", "s3"];
const rows = [[1, 2], [2, 1], [3, 5], [5, 4]];
const y = [1.5, 0.7, 2.5, 4.2];
const dense = data => ({ data: Float64Array.from(data.flat()), rows: data.length, cols: data[0].length });
const options = {
  methods, operators, digest: bytes => createHash("sha256").update(bytes).digest("hex"),
  resolveFeatures: ({ view }) => ({ sampleIds: [...view.sample_ids], matrix: dense(view.sample_ids.map(id => rows[ids.indexOf(id)])) }),
  resolveTargets: ({ sampleIds }) => ({ sampleIds: [...sampleIds], matrix: dense(sampleIds.map(id => [y[ids.indexOf(id)]])) }),
};
const task = {
  node_plan: { node_id: "model:nir", kind: "model", controller_id: controllerId,
    controller_version: "1.0.0", params: { alpha: 0.2 }, params_fingerprint: "a".repeat(64) },
  run_id: "run:controller-qualification", variant_id: "variant:0", fold_id: "fold:0",
  phase: "FIT_CV", branch_path: [], input_handles: {}, prediction_inputs: {},
  data_views: {
    "data:x": { partition: "fold_train", sample_ids: ids.slice(0, 2), source_ids: ["nir"] },
    "data:x:validation": { partition: "fold_validation", sample_ids: ids.slice(2), source_ids: ["nir"] },
  },
};
const call = (controller, value) => JSON.parse(controller.callback(controllerId, JSON.stringify(value)));
let fitCalls = 0;
const originalFit = methods.RolePipeline.prototype.fit;
methods.RolePipeline.prototype.fit = function (...args) { fitCalls++; return originalFit.apply(this, args); };
let tests = 0;
const checked = fn => { fn(); tests++; };
const controller = new N4mWasmRegressionController(options);
try {
  checked(() => assert.equal(call(controller, task).predictions[0].values.length, 2));
  const before = fitCalls;
  const bad = () => structuredClone(task);
  checked(() => { const value = bad(); value.data_views["data:x:validation"].sample_ids = ["s0"]; assert.throws(() => call(controller, value), /overlap/); });
  checked(() => { const value = bad(); value.data_view_receipts = { x: {} }; assert.throws(() => call(controller, value), /consumption/); });
  checked(() => { const value = bad(); value.required_loss_attestations = [{}]; assert.throws(() => call(controller, value), /specialized/); });
  checked(() => { const value = bad(); value.fit_influence = { mechanism: "native_weights", row_weights: [1, 2] }; assert.throws(() => call(controller, value), /uniform/); });
  checked(() => { const value = bad(); value.node_plan.controller_version = "2"; assert.throws(() => call(controller, value), /identity/); });
  checked(() => { const value = bad(); value.data_views["data:x"].sample_ids = ["s0", "s0"]; assert.throws(() => call(controller, value), /unique/); });
  assert.equal(fitCalls, before);
  const adversarial = new N4mWasmRegressionController({ ...options, resolveFeatures: request => {
    const resolved = options.resolveFeatures(request);
    resolved.sampleIds.reverse();
    return resolved;
  } });
  checked(() => assert.throws(() => call(adversarial, task), /IDs disagree/));
  adversarial.close();
  const asynchronous = new N4mWasmRegressionController({ ...options, resolveTargets: async request => options.resolveTargets(request) });
  checked(() => assert.throws(() => call(asynchronous, task), /synchronously/));
  asynchronous.close();
  const multiple = new N4mWasmRegressionController({ ...options, resolveFeatures: request => {
    const resolved = options.resolveFeatures(request);
    if (request.key.includes("x0")) resolved.matrix.data = resolved.matrix.data.map(value => 2 * value);
    return resolved;
  } });
  const multipleTask = structuredClone(task);
  multipleTask.data_views["data:x0"] = structuredClone(task.data_views["data:x"]);
  multipleTask.data_views["data:x0:validation"] = structuredClone(task.data_views["data:x:validation"]);
  checked(() => {
    const actual = call(multiple, multipleTask).predictions[0].values.flat();
    const reference = methods.RolePipeline.fromSteps([{ class: "n4m:models.regularized.ridge", params: { alpha: 0.2 } }]);
    try {
      const joined = rows.map(row => [...row, ...row.map(value => 2 * value)]);
      reference.fit(dense(joined.slice(0, 2)), dense(y.slice(0, 2).map(value => [value])));
      assert.deepEqual(actual, Array.from(reference.predict(dense(joined.slice(2))).data));
    } finally { reference.dispose(); }
  });
  multiple.close();
  const refit = { ...structuredClone(task), phase: "REFIT", fold_id: null,
    data_views: { "data:x": { partition: "full_train", sample_ids: ids, source_ids: ["nir"] } } };
  const fitted = call(controller, refit);
  const artifact = fitted.artifacts[0];
  const payload = controller.artifactPayload(artifact.id);
  const request = { node_id: "model:nir", controller_id: controllerId,
    params_fingerprint: task.node_plan.params_fingerprint, artifact };
  checked(() => assert.equal(artifact.content_fingerprint, options.digest(payload)));
  const otherRefit = { ...structuredClone(refit), run_id: "run:other-cohort",
    data_views: { "data:x": { partition: "full_train", sample_ids: ids.slice(1), source_ids: ["nir"] } } };
  const otherFitted = call(controller, otherRefit);
  const originalPrediction = { ...structuredClone(task), phase: "PREDICT", fold_id: null,
    data_views: { "data:x": { partition: "predict", sample_ids: ids, source_ids: ["nir"] } },
    artifact_inputs: { model: request }, input_handles: { model: fitted.artifact_handles[artifact.id] } };
  checked(() => assert.deepEqual(call(controller, originalPrediction).predictions[0].values, fitted.predictions[0].values));
  checked(() => {
    const value = structuredClone(originalPrediction);
    value.input_handles.model = otherFitted.artifact_handles[otherFitted.artifacts[0].id];
    assert.throws(() => call(controller, value), /binding mismatch/);
  });
  checked(() => {
    const value = structuredClone(originalPrediction);
    value.artifact_inputs.model.artifact.content_fingerprint = "c".repeat(64);
    assert.throws(() => call(controller, value), /binding mismatch/);
  });
  controller.close();
  const restored = new N4mWasmRegressionController(options);
  const beforeReplay = fitCalls;
  const handle = restored.hydrate(request, payload);
  const prediction = { ...structuredClone(task), phase: "PREDICT", fold_id: null,
    data_views: { "data:x": { partition: "predict", sample_ids: ids, source_ids: ["nir"] } },
    artifact_inputs: { model: { ...request } }, input_handles: { model: handle } };
  checked(() => assert.deepEqual(call(restored, prediction).predictions[0].values, fitted.predictions[0].values));
  checked(() => { const bytes = payload.slice(); bytes[bytes.length - 2] ^= 1; assert.throws(() => restored.hydrate(request, bytes), /fingerprint/); });
  checked(() => assert.throws(() => restored.hydrate({ ...request, node_id: "model:other" }, payload), /Portable Methods payload belongs to another controller profile, node or parameter binding/));
  checked(() => { const value = structuredClone(prediction); value.node_plan.params_fingerprint = "b".repeat(64); assert.throws(() => call(restored, value), /binding mismatch/); });
  checked(() => { const value = structuredClone(prediction); value.input_handles.model.owner_controller = "other"; assert.throws(() => call(restored, value), /owner/); });
  checked(() => {
    const forged = JSON.parse(new TextDecoder().decode(payload));
    forged.feature_names.reverse();
    assert.equal(forged.feature_names.length, JSON.parse(new TextDecoder().decode(payload)).feature_names.length);
    const bytes = new TextEncoder().encode(JSON.stringify(forged));
    const fingerprint = options.digest(bytes);
    const forgedArtifact = { ...artifact, content_fingerprint: fingerprint,
      size_bytes: bytes.length, uri: "artifacts/" + fingerprint + ".json" };
    const fresh = new N4mWasmRegressionController(options);
    let numericalPredictions = 0;
    const originalPredict = methods.RolePipeline.prototype.predict;
    methods.RolePipeline.prototype.predict = function (...args) {
      numericalPredictions++; return originalPredict.apply(this, args);
    };
    try {
      assert.throws(() => {
        const forgedRequest = { ...request, artifact: forgedArtifact };
        const forgedHandle = fresh.hydrate(forgedRequest, bytes);
        call(fresh, { ...structuredClone(prediction), artifact_inputs: { model: forgedRequest },
          input_handles: { model: forgedHandle } });
      }, /feature.*(order|name)|order.*feature/i);
      assert.equal(numericalPredictions, 0, "Re-sealed feature permutations must fail before Methods prediction");
    } finally { methods.RolePipeline.prototype.predict = originalPredict; fresh.close(); }
    assert.equal(fresh.models.size, 0);
  });
  assert.equal(fitCalls, beforeReplay);
  restored.close();
  checked(() => assert.throws(() => call(restored, prediction), /closed/));
  const metaController = new N4mWasmRegressionController(options);
  const block = (partition, shift) => ({ sample_ids: ids, partition, prediction_width: 1,
    values: rows.map(row => [row[0] + shift]), target_names: ["y"], fold_ids: ["fold:0", "fold:1"] });
  const metaTask = { ...structuredClone(refit), data_views: {}, prediction_inputs: {
    "base.oof": block("validation", 0), "base.oof:refit": block("test", 0.3),
  } };
  const fittedMeta = call(metaController, metaTask);
  const metaArtifact = fittedMeta.artifacts[0];
  const metaPrediction = { ...structuredClone(metaTask), phase: "PREDICT",
    prediction_inputs: { "base.oof:predict": block("final", 0.3) },
    artifact_inputs: { model: { node_id: "model:nir", controller_id: controllerId,
      params_fingerprint: task.node_plan.params_fingerprint, artifact: metaArtifact } },
    input_handles: { model: fittedMeta.artifact_handles[metaArtifact.id] } };
  const beforeMetaReplay = fitCalls;
  checked(() => assert.deepEqual(call(metaController, metaPrediction).predictions[0].values,
    fittedMeta.predictions[0].values));
  checked(() => {
    const reference = methods.RolePipeline.fromSteps([{ class: "n4m:models.regularized.ridge", params: { alpha: 0.2 } }]);
    try {
      reference.fit(dense(rows.map(row => [row[0]])), dense(y.map(value => [value])));
      const expected = reference.predict(dense(rows.map(row => [row[0] + 0.3])));
      assert.deepEqual(fittedMeta.predictions[0].values.flat(), Array.from(expected.data));
    } finally { reference.dispose(); }
  });
  assert.equal(fitCalls, beforeMetaReplay + 1, "Only the independent reference fits during meta replay checks");
  metaController.close();
} finally {
  methods.RolePipeline.prototype.fit = originalFit;
  controller.close();
}

// Reserved recipe-axis dispatch uses a recording controller boundary. The
// existing real Methods tests above remain unchanged; Root measures PLS parity.
{
  const seen = [];
  let hydrated = 0;
  const makeModel = steps => {
    assert(steps.every(step => !Object.hasOwn(step.params ?? {}, "pipeline_steps_v1")));
    const n = steps.at(-1).params.n_components;
    return {
      fit() {},
      predict(input) { return { data: new Float64Array(input.rows).fill(n), rows: input.rows, cols: 1 }; },
      exportStates() { return [{ n4me: Uint8Array.of(1) }]; },
      dispose() {},
    };
  };
  const recordingMethods = { RolePipeline: {
    fromSteps(steps) { seen.push(structuredClone(steps)); return makeModel(steps); },
    fromStates(steps) { hydrated++; return makeModel(steps); },
  } };
  const recipeOperators = { "model:nir": { type: "N4mRolePipeline" } };
  const recipeOptions = { ...options, methods: recordingMethods, operators: recipeOperators,
    controllerVersion: "1.1.0" };
  const recipes = [
    [{ class: "n4m:models.pls", params: {} }],
    [{ methodId: "transforms.snv", params: {} }, { class: "n4m:models.pls", params: {} }],
  ];
  const fingerprint = params => options.digest(new TextEncoder().encode(JSON.stringify(params)));
  const recipeTask = (steps, n) => {
    const value = structuredClone(task);
    value.node_plan.controller_version = "1.1.0";
    value.node_plan.params = { pipeline_steps_v1: structuredClone(steps), n_components: n };
    value.node_plan.params_fingerprint = fingerprint(value.node_plan.params);
    return value;
  };
  const dynamic = new N4mWasmRegressionController(recipeOptions);
  const fingerprints = new Set();
  try {
    checked(() => {
      for (const steps of recipes) for (const n of [1, 2]) {
        const value = recipeTask(steps, n);
        fingerprints.add(value.node_plan.params_fingerprint);
        const result = call(dynamic, value);
        assert.deepEqual(result.predictions[0].values, [[n], [n]]);
        assert.equal(result.lineage.params_fingerprint, value.node_plan.params_fingerprint);
        assert.deepEqual(seen.at(-1), steps.map((step, index) => index === steps.length - 1 ?
          { ...step, params: { ...step.params, n_components: n } } : step));
      }
      assert.equal(fingerprints.size, 4);
    });
    checked(() => {
      const value = recipeTask(recipes[0], 1);
      value.node_plan.params.pipeline_steps_v1 = [];
      assert.throws(() => call(dynamic, value), /nonempty recipe/);
      value.node_plan.params.pipeline_steps_v1 = [{ class: "n4m:models.pls", params: [], methodId: "ambiguous" }];
      assert.throws(() => call(dynamic, value), /Recipe steps/);
    });
    checked(() => {
      const conflicting = new N4mWasmRegressionController({ ...recipeOptions,
        operators: { "model:nir": { type: "N4mRolePipeline", steps: recipes[1] } } });
      try { assert.throws(() => call(conflicting, recipeTask(recipes[0], 1)), /recipe disagrees/); }
      finally { conflicting.close(); }
      const legacy = new N4mWasmRegressionController({ ...recipeOptions, controllerVersion: "1.0.0" });
      const value = recipeTask(recipes[0], 1); value.node_plan.controller_version = "1.0.0";
      try { assert.throws(() => call(legacy, value), /requires.*1.1.0/); }
      finally { legacy.close(); }
    });
    const refit = recipeTask(recipes[1], 2);
    refit.phase = "REFIT"; refit.fold_id = null;
    // Deliberately retain an ordered training subset; no all-row refit.
    refit.data_views = { "data:x": { partition: "full_train", sample_ids: ["s1", "s0"], source_ids: ["nir"] } };
    const fitted = call(dynamic, refit);
    const artifact = fitted.artifacts[0];
    const payload = dynamic.artifactPayload(artifact.id);
    const request = { controller_id: controllerId, node_id: "model:nir",
      params_fingerprint: refit.node_plan.params_fingerprint, artifact };
    const restored = new N4mWasmRegressionController(recipeOptions);
    try {
      const handle = restored.hydrate(request, payload);
      const prediction = recipeTask(recipes[1], 2);
      prediction.phase = "PREDICT"; prediction.fold_id = null;
      prediction.data_views = { "data:x": { partition: "predict", sample_ids: ids.slice(2), source_ids: ["nir"] } };
      prediction.artifact_inputs = { model: request }; prediction.input_handles = { model: handle };
      checked(() => assert.deepEqual(call(restored, prediction).predictions[0].values, [[2], [2]]));
      checked(() => {
        const altered = structuredClone(prediction);
        // Even a caller reusing a claimed params hash cannot change the recipe.
        altered.node_plan.params.pipeline_steps_v1 = recipes[0];
        assert.throws(() => call(restored, altered), /recipe disagrees/);
        assert.throws(() => restored.hydrate({ ...request, params_fingerprint: "f".repeat(64) }, payload), /binding/);
      });
      assert.equal(hydrated, 1);
      const saved = JSON.parse(new TextDecoder().decode(payload));
      assert.equal(saved.schema, "dagml.methods.regression.v2");
      assert.equal(saved.recipe_profile, "pipeline_steps.v1");
      assert.equal(saved.controller_version, "1.1.0");
    } finally { restored.close(); }
  } finally { dynamic.close(); }
}

console.log("METHODS_CONTROLLER_TESTS_OK", tests, "fits", fitCalls);
