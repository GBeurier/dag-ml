/** Methods-backed regression controllers for DAG-ML's synchronous WASM ABI.
 * Heavy feature buffers stay in the host. Folds, OOF construction and scoring
 * stay in DAG-ML; fitting, preprocessing and prediction stay in Methods.
 */

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function ids(value, label) {
  requireCondition(Array.isArray(value) && value.length > 0 &&
    value.every(id => typeof id === "string" && id.length > 0) &&
    new Set(value).size === value.length, label + " needs unique nonempty sample IDs");
  return value;
}

function sameIds(actual, expected, label) {
  ids(actual, label);
  requireCondition(actual.length === expected.length &&
    actual.every((id, i) => id === expected[i]), label + " sample IDs disagree with the native task");
}

function matrix(value, sampleIds, label) {
  requireCondition(value && Number.isSafeInteger(value.rows) &&
    Number.isSafeInteger(value.cols) && value.rows === sampleIds.length &&
    value.cols > 0 && value.data instanceof Float64Array &&
    value.data.length === value.rows * value.cols &&
    value.data.every(Number.isFinite), label + " needs a finite, row-major Float64Array matrix");
  return value;
}

function featureNamespace(key) {
  return key.replace(/:(validation|outer|refit|predict|test)$/, "");
}

function orderFeatureEntries(entries) {
  return entries.sort(([a], [b]) => {
    const left = featureNamespace(a), right = featureNamespace(b);
    return left < right ? -1 : left > right ? 1 : a < b ? -1 : a > b ? 1 : 0;
  });
}

function join(blocks) {
  requireCondition(blocks.length > 0, "No feature blocks were supplied");
  const sampleIds = blocks[0].sampleIds;
  let width = 0;
  for (const block of blocks) {
    sameIds(block.sampleIds, sampleIds, "Feature join");
    matrix(block.matrix, sampleIds, "Feature join");
    width += block.matrix.cols;
  }
  const data = new Float64Array(sampleIds.length * width);
  for (let row = 0; row < sampleIds.length; row++) {
    let offset = row * width;
    for (const block of blocks) {
      data.set(block.matrix.data.subarray(row * block.matrix.cols, (row + 1) * block.matrix.cols), offset);
      offset += block.matrix.cols;
    }
  }
  const featureNames = blocks.flatMap(block => block.featureNames);
  return { sampleIds, matrix: { data, rows: sampleIds.length, cols: width }, featureNames };
}

function recipe(node, operator) {
  requireCondition(operator && (typeof operator.type === "string"),
    "A native Methods operator is required");
  let steps;
  if (operator.type === "N4mRolePipeline") {
    requireCondition(Array.isArray(operator.steps) && operator.steps.length > 0,
      "N4mRolePipeline needs a nonempty native recipe");
    steps = structuredClone(operator.steps);
  } else {
    requireCondition(operator.type.startsWith("n4m:"), "Operator must be n4m:<method-id> or N4mRolePipeline");
    steps = [{ class: operator.type, params: {} }];
  }
  const last = steps.at(-1);
  requireCondition(last && typeof last === "object" && !Array.isArray(last) &&
    (typeof last.class === "string" || typeof last.methodId === "string"),
    "Recipe steps use {class, params} or {methodId, params}");
  last.params = { ...last.params, ...node.params };
  return steps;
}

function sameRecipe(actual, expected) {
  const ordered = value => Array.isArray(value) ? value.map(ordered) :
    value && typeof value === "object" ? Object.fromEntries(
      Object.keys(value).sort().map(key => [key, ordered(value[key])])) : value;
  requireCondition(JSON.stringify(ordered(actual)) === JSON.stringify(ordered(expected)),
    "Portable Methods recipe disagrees with the effective planned operator and parameters");
}

/** Construct after await methods.loadModule(). One controller owns its artifacts. */
export class N4mWasmRegressionController {
  constructor({ methods, operators, resolveFeatures, resolveTargets, targetNames = ["y"],
    controllerId = "controller:methods.wasm.regression", controllerVersion = "1.0.0", digest = null }) {
    requireCondition(methods?.RolePipeline && typeof resolveFeatures === "function" &&
      typeof resolveTargets === "function", "Methods and synchronous feature/target resolvers are required");
    requireCondition(operators && typeof operators === "object" && !Array.isArray(operators) &&
      Object.keys(operators).length > 0, "Bind model node IDs to their compiled graph operators");
    ids(targetNames, "Target names");
    requireCondition(typeof controllerId === "string" && controllerId.length > 0 &&
      typeof controllerVersion === "string" && controllerVersion.length > 0,
    "Controller identity must be nonempty");
    requireCondition(digest === null || typeof digest === "function", "digest must be synchronous SHA-256");
    this.methods = methods;
    this.operators = structuredClone(operators);
    this.resolveFeatures = resolveFeatures;
    this.resolveTargets = resolveTargets;
    this.targetNames = [...targetNames];
    this.controllerId = controllerId;
    this.controllerVersion = controllerVersion;
    this.digest = digest;
    this.artifacts = new Map();
    this.models = new Map();
    this.nextHandle = 1;
    this.closed = false;
    this.callback = this.invoke.bind(this);
  }

  manifest(dagMl) {
    return JSON.parse(dagMl.derive_controller_manifest_json(JSON.stringify({
      controller_id: this.controllerId, controller_version: this.controllerVersion,
      operator_kind: "model", added_capabilities: ["consumes_oof_predictions"],
      rng_policy: "externally_deterministic",
    })));
  }

  _hash(bytes) {
    requireCondition(this.digest !== null, "REFIT/portable replay requires a synchronous SHA-256 digest(bytes)");
    const result = this.digest(bytes);
    requireCondition(typeof result === "string" && /^[0-9a-f]{64}$/.test(result),
      "digest(bytes) must return a lowercase SHA-256 string synchronously");
    return result;
  }

  _handle(entry) {
    requireCondition(Number.isSafeInteger(this.nextHandle), "Controller handle space exhausted");
    const handle = this.nextHandle++;
    this.models.set(handle, entry);
    return { handle, kind: "model", owner_controller: this.controllerId };
  }

  _targets(sampleIds, task) {
    const resolved = this.resolveTargets({ sampleIds: [...sampleIds],
      targetNames: [...this.targetNames], task: structuredClone(task) });
    requireCondition(resolved && typeof resolved.then !== "function",
      "Target resolver must return synchronously");
    sameIds(resolved.sampleIds, sampleIds, "Targets");
    matrix(resolved.matrix, sampleIds, "Targets");
    requireCondition(resolved.matrix.cols === this.targetNames.length, "Target width disagrees with target names");
    return resolved.matrix;
  }

  _features(task, partition) {
    const blocks = [];
    for (const [key, view] of orderFeatureEntries(Object.entries(task.data_views ?? {}))) {
      if (view.partition !== partition) continue;
      const expected = ids(view.sample_ids, "Native view");
      const resolved = this.resolveFeatures({ key, view: structuredClone(view), task: structuredClone(task) });
      requireCondition(resolved && typeof resolved.then !== "function",
        "Feature resolver must return synchronously");
      sameIds(resolved.sampleIds, expected, "Features");
      matrix(resolved.matrix, expected, "Features");
      const names = resolved.featureNames ?? Array.from({ length: resolved.matrix.cols }, (_, i) => String(i));
      requireCondition(Array.isArray(names) && names.length === resolved.matrix.cols &&
        names.every(name => typeof name === "string" && name.length > 0) &&
        new Set(names).size === names.length, "Feature names must be unique and match matrix width");
      // Validation suffix is operational, not a different feature namespace.
      const namespace = featureNamespace(key);
      blocks.push({ ...resolved, featureNames: names.map(name => namespace + "/" + name) });
    }
    return join(blocks);
  }

  _predictions(task, outer) {
    const suffix = task.phase === "FIT_CV" ? ":outer" : task.phase === "REFIT" ? ":refit" : ":predict";
    const entries = orderFeatureEntries(Object.entries(task.prediction_inputs ?? {})
      .filter(([key]) => outer ? key.endsWith(suffix) : !/:(outer|refit|predict|test)$/.test(key)));
    return join(entries.map(([key, input]) => {
      const sampleIds = ids(input.sample_ids, "Native prediction inputs");
      if (task.phase === "FIT_CV" || (task.phase === "REFIT" && !outer)) {
        requireCondition(input.partition === "validation", "Meta training must consume native validation OOF");
        if (task.phase === "FIT_CV" && !outer) requireCondition(!(input.fold_ids ?? []).includes(task.fold_id),
          "Outer-fold predictions cannot be used for fitting its meta-model");
      }
      if (outer && task.phase !== "FIT_CV") requireCondition(
        ["test", "final"].includes(input.partition), "Meta prediction needs native off-fold rows");
      requireCondition(Array.isArray(input.values) && input.values.length === sampleIds.length &&
        input.values.every(row => Array.isArray(row) && row.length === input.prediction_width),
      "Native prediction feature width is inconsistent");
      const names = Array.from({ length: input.prediction_width },
        (_, column) => featureNamespace(key) + "/" + column);
      return { sampleIds, featureNames: names, matrix: {
        data: Float64Array.from(input.values.flat()), rows: sampleIds.length, cols: input.prediction_width,
      } };
    }));
  }

  artifactPayload(artifactId) {
    requireCondition(!this.closed, "Controller is closed");
    const entry = this.artifacts.get(artifactId);
    requireCondition(entry, "Unknown controller artifact");
    return entry.payload.slice();
  }

  hydrate(request, payload) {
    requireCondition(!this.closed && payload instanceof Uint8Array &&
      request.controller_id === this.controllerId &&
      typeof request.artifact?.id === "string" && request.artifact.id.length > 0,
    "Invalid portable artifact owner or payload");
    requireCondition(this._hash(payload) === request.artifact.content_fingerprint,
      "Portable Methods payload fingerprint mismatch");
    const saved = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(payload));
    requireCondition(saved.schema === "dagml.methods.regression.v1" &&
      saved.node_id === request.node_id && saved.params_fingerprint === request.params_fingerprint,
    "Portable Methods payload belongs to another node or parameter binding");
    sameIds(saved.target_names, this.targetNames, "Portable target names");
    ids(saved.feature_names, "Portable feature names");
    requireCondition(Array.isArray(saved.states) && saved.states.length > 0 &&
      saved.states.every(state => Array.isArray(state) && state.length > 0 &&
        state.every(byte => Number.isInteger(byte) && byte >= 0 && byte <= 255)),
    "Portable Methods states must be nonempty byte arrays");
    const model = this.methods.RolePipeline.fromStates(saved.steps,
      saved.states.map(state => Uint8Array.from(state)), { featureNames: saved.feature_names });
    return this._handle({ model, nodeId: saved.node_id, paramsFingerprint: saved.params_fingerprint,
      artifactId: request.artifact.id, contentFingerprint: request.artifact.content_fingerprint,
      steps: structuredClone(saved.steps), featureNames: [...saved.feature_names] });
  }

  invoke(controllerId, taskJson) {
    requireCondition(!this.closed && controllerId === this.controllerId, "Wrong or closed Methods controller");
    const task = JSON.parse(taskJson);
    if (task.operation) {
      requireCondition(task.schema_version === 1, "Unsupported portable bridge schema");
      if (task.operation === "export_artifact_payload") return JSON.stringify({
        operation: "exported_artifact_payload", schema_version: 1,
        payload: Array.from(this.artifactPayload(task.artifact_id)),
      });
      if (task.operation === "hydrate_artifact_payload") {
        requireCondition(Array.isArray(task.payload) &&
          task.payload.every(byte => Number.isInteger(byte) && byte >= 0 && byte <= 255),
        "Portable payload must be a byte array");
        return JSON.stringify({ operation: "hydrated_artifact_payload", schema_version: 1,
          handle: this.hydrate(task.request, Uint8Array.from(task.payload)) });
      }
      requireCondition(task.operation === "release_hydrated_artifact_payload" &&
        task.handle.owner_controller === this.controllerId, "Unsupported portable bridge operation");
      const entry = this.models.get(task.handle.handle);
      requireCondition(entry, "Unknown hydrated Methods handle");
      entry.model.dispose();
      this.models.delete(task.handle.handle);
      return JSON.stringify({ operation: "released_hydrated_artifact_payload", schema_version: 1 });
    }
    const node = task.node_plan;
    requireCondition(node.kind === "model" && node.controller_id === this.controllerId &&
      node.controller_version === this.controllerVersion, "Methods controller identity or kind mismatch");
    requireCondition(["FIT_CV", "REFIT", "PREDICT"].includes(task.phase), "Unsupported Methods phase");
    requireCondition(Object.keys(task.data_view_receipts ?? {}).length === 0,
      "Generated views require a consumption-attesting controller");
    requireCondition(!(task.required_loss_attestations ?? []).length &&
      !task.residual_targets, "Custom losses and residual targets require a specialized controller");
    requireCondition(!task.fit_influence ||
      (task.fit_influence.mechanism === "uniform_rows" && !(task.fit_influence.row_weights ?? []).length),
    "This controller requires uniform row influence");
    const meta = Object.keys(task.prediction_inputs ?? {}).length > 0;
    requireCondition(!(task.phase === "FIT_CV" &&
      Object.keys(task.prediction_inputs ?? {}).some(key => key.endsWith(":test"))),
    "Additional CV test streams require a specialized controller");
    const fitting = task.phase !== "PREDICT";
    let model, kept = false;
    let artifacts = [], artifactHandles = {};
    try {
      if (fitting) {
        if (task.phase === "REFIT") this._hash(new Uint8Array());
        const train = meta ? this._predictions(task, false) :
          this._features(task, task.phase === "FIT_CV" ? "fold_train" : "full_train");
        const valid = task.phase === "FIT_CV" ?
          (meta ? this._predictions(task, true) : this._features(task, "fold_validation")) :
          meta && Object.keys(task.prediction_inputs).some(key => key.endsWith(":refit")) ?
            this._predictions(task, true) : train;
        if (task.phase === "FIT_CV") requireCondition(
          train.sampleIds.every(id => !valid.sampleIds.includes(id)), "Fit and validation sample IDs overlap");
        const targets = this._targets(train.sampleIds, task);
        const steps = recipe(node, this.operators[node.node_id]);
        model = this.methods.RolePipeline.fromSteps(steps);
        model.fit(train.matrix, targets, { featureNames: train.featureNames });
        if (task.phase === "REFIT") {
          const payload = new TextEncoder().encode(JSON.stringify({
            schema: "dagml.methods.regression.v1", node_id: node.node_id,
            params_fingerprint: node.params_fingerprint, target_names: this.targetNames,
            steps, feature_names: train.featureNames,
            states: model.exportStates().map(state => Array.from(state.n4me)),
          }));
          const artifactId = ["artifact:methods", task.run_id, node.node_id, task.variant_id ?? "base", "refit"].join(":");
          requireCondition(!this.artifacts.has(artifactId), "Duplicate Methods REFIT artifact");
          const contentFingerprint = this._hash(payload);
          artifacts = [{ id: artifactId, kind: "methods_role_pipeline", controller_id: controllerId,
            backend: "raw", uri: "artifacts/" + contentFingerprint + ".json",
            content_fingerprint: contentFingerprint, size_bytes: payload.length,
            plugin: "dagml.methods.wasm.regression", plugin_version: this.controllerVersion }];
          // Prediction/export must succeed before publishing a retained model.
          const result = JSON.parse(this._result(task, valid, model, artifacts, {}));
          const handle = this._handle({ model, nodeId: node.node_id, paramsFingerprint: node.params_fingerprint,
            artifactId, contentFingerprint: artifacts[0].content_fingerprint,
            steps: structuredClone(steps), featureNames: [...train.featureNames] });
          artifactHandles[artifactId] = handle;
          result.artifact_handles = artifactHandles;
          this.artifacts.set(artifactId, { payload, handle });
          kept = true;
          return JSON.stringify(result);
        }
        return this._result(task, valid, model, artifacts, artifactHandles);
      }
      const inputs = Object.entries(task.artifact_inputs ?? {});
      requireCondition(inputs.length === 1, "PREDICT requires one attested model artifact");
      const [key, input] = inputs[0];
      const handle = task.input_handles?.[key];
      requireCondition(handle?.owner_controller === controllerId && handle.kind === "model",
        "PREDICT model handle has another owner or kind");
      const entry = this.models.get(handle.handle);
      requireCondition(entry && entry.nodeId === node.node_id &&
        entry.paramsFingerprint === node.params_fingerprint &&
        input.params_fingerprint === node.params_fingerprint &&
        input.artifact?.id === entry.artifactId &&
        input.artifact?.content_fingerprint === entry.contentFingerprint,
      "PREDICT model binding mismatch");
      model = entry.model;
      kept = true;
      sameRecipe(entry.steps, recipe(node, this.operators[node.node_id]));
      const features = meta ? this._predictions(task, true) : this._features(task, "predict");
      requireCondition(entry.featureNames.length === features.featureNames.length &&
        entry.featureNames.every((name, index) => name === features.featureNames[index]),
      "Portable Methods feature order disagrees with the current resolved inputs");
      return this._result(task, features, model, [], {});
    } finally {
      if (model && !kept) model.dispose();
    }
  }

  _result(task, features, model, artifacts, artifactHandles) {
    const predicted = matrix(model.predict(features.matrix, features.featureNames), features.sampleIds, "Predictions");
    requireCondition(predicted.cols === this.targetNames.length, "Prediction width disagrees with targets");
    const rows = data => Array.from({ length: data.rows }, (_, row) =>
      Array.from(data.data.subarray(row * data.cols, (row + 1) * data.cols)));
    const node = task.node_plan;
    const targets = task.phase === "FIT_CV" ? [{
      level: "sample", unit_ids: features.sampleIds.map(id => ({ level: "sample", id })),
      values: rows(this._targets(features.sampleIds, task)), target_names: this.targetNames,
    }] : [];
    return JSON.stringify({
      node_id: node.node_id, outputs: {},
      predictions: [{ producer_node: node.node_id,
        partition: task.phase === "FIT_CV" ? "validation" : "final",
        fold_id: task.fold_id, sample_ids: features.sampleIds,
        values: rows(predicted), target_names: this.targetNames }],
      regression_targets: targets, artifacts, artifact_handles: artifactHandles,
      lineage: {
        record_id: ["lineage:methods-wasm", task.run_id, node.node_id, task.phase,
          task.variant_id ?? "base", task.fold_id ?? "full"].join(":"),
        run_id: task.run_id, node_id: node.node_id, phase: task.phase,
        controller_id: this.controllerId, controller_version: this.controllerVersion,
        variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
        input_lineage: [], artifact_refs: artifacts, params_fingerprint: node.params_fingerprint,
        data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null,
        seed: null, unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: [],
      },
    });
  }

  close() {
    if (this.closed) return;
    for (const { model } of this.models.values()) model.dispose();
    this.models.clear();
    this.artifacts.clear();
    this.closed = true;
  }
}
