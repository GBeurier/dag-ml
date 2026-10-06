/** Thin N4ME role bridge. DAG owns task scope/OOF; Methods owns all numerics. */
const requireValue = (ok, message) => { if (!ok) throw new TypeError(message); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const capabilityNames = ['transform', 'predict', 'predict_proba', 'decision_function', 'predict_labels', 'selected_indices', 'apply_mask', 'serializable', 'affine', 'retains_training_rows'];
const scope = task => JSON.stringify([task.run_id, task.phase, task.variant_id, task.fold_id]);
const rows = matrix => Array.from({length: matrix.rows}, (_, i) => Array.from(matrix.data.subarray(i * matrix.cols, (i + 1) * matrix.cols)));
function checkMatrix(matrix, ids) {
  requireValue(matrix?.data instanceof Float64Array && Number.isSafeInteger(matrix.rows) && Number.isSafeInteger(matrix.cols)
    && matrix.rows === ids.length && matrix.rows > 0 && matrix.cols > 0 && matrix.data.length === matrix.rows * matrix.cols
    && matrix.data.every(Number.isFinite), 'Expected a finite row-major matrix aligned to the task IDs');
  return matrix;
}
function checkedParams(info, params) {
  const specs = new Map(info.params.map(item => [item.name, item]));
  for (const [name, value] of Object.entries(params)) {
    const spec = specs.get(name);
    requireValue(spec, 'Unknown native parameter ' + name);
    const integer = item => Number.isSafeInteger(item);
    const finite = item => typeof item === 'number' && Number.isFinite(item);
    const valid = spec.type === 'int' ? integer(value) : spec.type === 'double' ? finite(value)
      : spec.type === 'bool' ? typeof value === 'boolean' : spec.type === 'enum' ? spec.choices.includes(value)
      : spec.type === 'int_array' ? Array.isArray(value) && value.every(integer)
      : spec.type === 'double_array' ? Array.isArray(value) && value.every(finite) : false;
    requireValue(valid, 'Invalid native parameter type ' + name);
  }
  return params;
}
/** Read identity, capabilities and explicit params from native bytes, never JSON declarations. */
function inspectState(methods, payload, info, params) {
  requireValue(payload instanceof Uint8Array && payload.length > 0 && payload.length <= 64 * 1024 * 1024, 'Invalid N4ME bytes');
  const m = methods.getModule(), allocations = [];
  const alloc = count => { const pointer = m._malloc(Math.max(1, count)); requireValue(pointer, 'Native allocation failed'); allocations.push(pointer); return pointer; };
  const call = (name, types, values) => { requireValue(m.ccall(name, 'number', types, values) === 0, 'Native inspection failed: ' + name); };
  const string = value => { const bytes = new TextEncoder().encode(value + '\0'), pointer = alloc(bytes.length); m.HEAPU8.set(bytes, pointer); return pointer; };
  let context = 0, estimator = 0, resolved = 0;
  try {
    const output = alloc(4), bytes = alloc(payload.length);
    m.HEAPU8.set(payload, bytes);
    call('n4m_context_create', ['number'], [output]); context = m.getValue(output, 'i32');
    call('n4m_estimator_import_from_buffer', ['number', 'number', 'number', 'number'], [context, bytes, payload.length, output]);
    estimator = m.getValue(output, 'i32');
    const index = alloc(4), caps = alloc(8), metadata = alloc(72);
    call('n4m_estimator_info', ['number', 'number', 'number'], [estimator, index, caps]);
    m.HEAPU8.fill(0, metadata, metadata + 72); m.setValue(metadata, 72, 'i32');
    call('n4m_method_info_v1', ['number', 'number'], [m.getValue(index, 'i32'), metadata]);
    const id = m.UTF8ToString(m.getValue(metadata + 8, 'i32'));
    requireValue(id === info.method_id, 'N4ME method differs from the planned operator');
    const bits = new DataView(m.HEAPU8.buffer).getBigUint64(caps, true);
    const capabilities = capabilityNames.filter((_, i) => (bits & (1n << BigInt(i))) !== 0n).sort();
    call('n4m_estimator_get_params', ['number', 'number', 'number'], [context, estimator, output]); resolved = m.getValue(output, 'i32');
    const specs = new Map(info.params.map(item => [item.name, item]));
    for (const [name, value] of Object.entries(params)) {
      const spec = specs.get(name), isDouble = spec.type === 'double' || spec.type === 'double_array';
      const expected = spec.type === 'bool' ? [value ? 1 : 0] : spec.type === 'enum' ? [spec.choices.indexOf(value)] : Array.isArray(value) ? value : [value];
      const count = alloc(8), data = alloc(Math.max(1, expected.length) * 8);
      call(isDouble ? 'n4m_params_get_double' : 'n4m_params_get_int', ['number', 'number', 'number', 'i64', 'number'],
        [resolved, string(name), data, BigInt(expected.length), count]);
      requireValue(new DataView(m.HEAPU8.buffer).getBigInt64(count, true) === BigInt(expected.length), 'N4ME parameter length mismatch');
      const actual = expected.map((_, i) => isDouble ? new DataView(m.HEAPU8.buffer).getFloat64(data + i * 8, true)
        : new DataView(m.HEAPU8.buffer).getBigInt64(data + i * 8, true));
      requireValue(actual.every((item, i) => item === (isDouble ? expected[i] : BigInt(expected[i]))), 'N4ME parameter differs: ' + name);
    }
    return {method_id: id, roles: [...info.roles], capabilities};
  } finally {
    if (resolved) m.ccall('n4m_params_destroy', null, ['number'], [resolved]);
    if (estimator) m.ccall('n4m_estimator_destroy', null, ['number'], [estimator]);
    if (context) m.ccall('n4m_context_destroy', null, ['number'], [context]);
    for (const pointer of allocations.reverse()) m._free(pointer);
  }
}
/** Registration shares only invocation-local, identity-keyed transform buffers. */
export class N4mWasmEstimatorControllers {
  constructor({methods, dagMl, digest, resolveFeatures, resolveTargets, targetNames}) {
    requireValue(methods?.NativeEstimator && typeof digest === 'function' && typeof resolveFeatures === 'function' && typeof resolveTargets === 'function', 'Native role runtime/resolvers required');
    this.methods = methods; this.dag = dagMl; this.digest = digest; this.resolveFeatures = resolveFeatures; this.resolveTargets = resolveTargets;
    this.targetNames = [...targetNames]; this.catalog = new Map(methods.manifest().methods.map(item => [item.method_id, item]));
    this.manifests = JSON.parse(dagMl.derive_controller_manifest_list_json(dagMl.n4m_host_controller_specs_json(JSON.stringify(methods.manifest()))));
    this.manifests.sort((a, b) => a.controller_id < b.controller_id ? -1 : a.controller_id > b.controller_id ? 1 : 0);
    this.controllers = new Map(this.manifests.map(item => [item.controller_id, item]));
    this.features = new Map(); this.models = new Map(); this.artifacts = new Map(); this.nextHandle = 1; this.closed = false;
    this.callback = (owner, task) => this.invoke(owner, task);
  }
  handle(owner, kind, map, entry) {
    const handle = this.nextHandle++;
    requireValue(Number.isSafeInteger(handle), 'Handle space exhausted'); map.set(handle, entry);
    return {handle, kind, owner_controller: owner};
  }
  planned(node) {
    const role = node.controller_id.slice('controller:n4m.'.length), id = node.params.method_id, info = this.catalog.get(id);
    requireValue(info?.kind === 'estimator' && info.roles.includes(role), 'Native method/role mismatch');
    requireValue(node.operator_ref === undefined || node.operator_ref === 'n4m:' + id, 'Native operator/method mismatch');
    const params = Object.fromEntries(Object.entries(node.params).filter(([key]) => !['method_id', 'unsafe_flags'].includes(key)));
    // Kernel/local state retention needs a separately qualified opt-in profile.
    requireValue(!(node.params.unsafe_flags?.length) && !info.capabilities.includes('retains_training_rows'), 'This browser profile refuses training-row state retention');
    requireValue(Object.entries(info.inputs).every(([key, requirement]) => requirement === 'none' || ['x', 'y', 'labels'].includes(key)), 'Native method needs an unsupported fit input');
    return {info, params: checkedParams(info, params), role};
  }
  featuresFor(task) {
    let featureSet;
    const direct = task.node_plan.data_bindings?.length ? Object.entries(task.data_views ?? {}) : [];
    if (direct.length) {
      const collect = partition => {
        const views = direct.filter(([, view]) => view.partition === partition);
        if (!views.length) return null;
        requireValue(views.length === 1, 'Generic pipeline expects one raw source');
        const [key, view] = views[0], ids = [...view.sample_ids], resolved = this.resolveFeatures({key, view, task});
        requireValue(resolved && typeof resolved.then !== 'function' && same(resolved.sampleIds, ids), 'Raw feature IDs disagree with the native task');
        return {sampleIds: ids, matrix: checkMatrix(resolved.matrix, ids)};
      };
      const fitting = task.phase === 'FIT_CV' ? 'fold_train' : task.phase === 'REFIT' ? 'full_train' : 'predict';
      featureSet = {fit: collect(fitting), prediction: task.phase === 'FIT_CV' ? collect('fold_validation') : task.phase === 'REFIT' ? collect('test') : null};
    } else {
      const handle = task.input_handles?.['data:x'];
      requireValue(handle?.kind === 'data', 'Missing upstream native data handle');
      const saved = this.features.get(handle.handle);
      requireValue(saved && saved.owner === handle.owner_controller && saved.scope === scope(task), 'Feature handle belongs to another task scope');
      featureSet = saved.features;
      requireValue(same(featureSet.fit.sampleIds, task.data_views?.['data:x']?.sample_ids), 'Upstream fitting rows differ from the native view');
      if (task.phase === 'FIT_CV') requireValue(same(featureSet.prediction?.sampleIds, task.data_views?.['data:x:validation']?.sample_ids), 'Upstream validation rows differ from the native view');
    }
    requireValue(featureSet.fit, 'Missing fitting/prediction features');
    if (task.phase === 'FIT_CV') requireValue(featureSet.prediction && featureSet.fit.sampleIds.every(id => !featureSet.prediction.sampleIds.includes(id)), 'Fit/validation rows overlap');
    return featureSet;
  }
  targets(block, task) {
    const resolved = this.resolveTargets({sampleIds: block.sampleIds, task});
    requireValue(resolved && typeof resolved.then !== 'function' && same(resolved.sampleIds, block.sampleIds), 'Target IDs disagree with the native task');
    return checkMatrix(resolved.matrix, block.sampleIds);
  }
  descriptor(bytes, owner, planned) {
    const native = inspectState(this.methods, bytes, planned.info, planned.params);
    return JSON.parse(this.dag.native_estimator_descriptor_json(JSON.stringify({descriptor_type: 'dagml.native_estimator_descriptor.v1', schema_version: 1,
      artifact_sha256: this.digest(bytes), owner_controller: owner, format: 'N4ME', ...native, descriptor_fingerprint: ''})));
  }
  invoke(owner, taskJson) {
    requireValue(!this.closed, 'Estimator registration is closed');
    const task = JSON.parse(taskJson);
    if (task.operation) return this.bridge(owner, task);
    const node = task.node_plan, manifest = this.controllers.get(owner);
    requireValue(manifest && node.controller_id === owner && node.controller_version === manifest.controller_version, 'Native controller identity mismatch');
    requireValue(['FIT_CV', 'REFIT', 'PREDICT'].includes(task.phase), 'Unsupported phase');
    requireValue(!Object.keys(task.data_view_receipts ?? {}).length && !(task.required_loss_attestations?.length) && !task.residual_targets
      && (!task.fit_influence || task.fit_influence.mechanism === 'uniform_rows' && !(task.fit_influence.row_weights?.length)), 'Unsupported generated view, loss or fitting influence');
    const planned = this.planned(node), features = this.featuresFor(task), artifacts = [], artifactHandles = {}, outputs = {};
    let model;
    try {
      if (task.phase === 'PREDICT') {
        const inputs = Object.entries(task.artifact_inputs ?? {});
        requireValue(inputs.length === 1, 'Replay requires one retained estimator state per node');
        const [key, input] = inputs[0], handle = task.input_handles?.[key], saved = this.models.get(handle?.handle);
        requireValue(handle?.owner_controller === owner && handle.kind === 'model' && saved && saved.owner === owner
          && saved.artifact.id === input.artifact.id && saved.artifact.content_fingerprint === input.artifact.content_fingerprint, 'Replay artifact handle mismatch');
        const descriptor = this.descriptor(saved.bytes, owner, planned);
        requireValue(same(descriptor, input.artifact.native_estimator_descriptor), 'Replay native descriptor mismatch');
        model = this.methods.NativeEstimator.fromN4me(saved.bytes);
      } else {
        model = new (this.methods.methodClass(planned.info.method_id))(planned.params);
        const needY = ['y', 'labels'].some(key => planned.info.inputs[key] && planned.info.inputs[key] !== 'none');
        const target = needY ? this.targets(features.fit, task) : undefined;
        if (planned.role === 'classifier') requireValue(target?.cols === 1, 'Classification requires one target');
        model.fit(features.fit.matrix, planned.role === 'classifier' ? target.data : target);
        if (task.phase === 'REFIT') {
          const bytes = model.toN4me(), descriptor = this.descriptor(bytes, owner, planned);
          const id = 'artifact:n4m:' + node.node_id + ':' + (task.variant_id ?? 'base') + ':refit';
          const artifact = {id, kind: 'n4m_estimator', controller_id: owner, backend: 'raw', uri: 'methods/' + descriptor.artifact_sha256 + '.n4me',
            content_fingerprint: descriptor.artifact_sha256, size_bytes: bytes.length, abi_major: 2, abi_min_minor: 13, native_estimator_descriptor: descriptor};
          requireValue(!this.artifacts.has(id), 'Duplicate refit artifact'); this.artifacts.set(id, bytes);
          artifacts.push(artifact); artifactHandles[id] = this.handle(owner, 'model', this.models, {owner, artifact, bytes});
        }
      }
      const predictions = [], targets = [];
      if (planned.role === 'transformer' || planned.role === 'selector') {
        const apply = block => block ? {...block, matrix: checkMatrix(model.transform(block.matrix), block.sampleIds)} : null;
        outputs.x_out = this.handle(owner, 'data', this.features, {owner, scope: scope(task), features: {fit: apply(features.fit), prediction: apply(features.prediction)}});
      } else {
        requireValue(['regressor', 'classifier'].includes(planned.role), 'Unsupported generic browser role');
        const pool = task.phase === 'FIT_CV' ? {sampleIds: [...features.fit.sampleIds, ...features.prediction.sampleIds], matrix: {
          rows: features.fit.matrix.rows + features.prediction.matrix.rows, cols: features.fit.matrix.cols,
          data: Float64Array.from([...features.fit.matrix.data, ...features.prediction.matrix.data])}} : null;
        const surfaces = task.phase === 'FIT_CV' ? [[features.prediction, 'validation'], [features.fit, 'train'], [pool, 'train_pool']]
          : task.phase === 'REFIT' && features.prediction ? [[features.fit, 'final'], [features.prediction, 'test']] : [[features.fit, 'final']];
        outputs.oof = {handle: this.nextHandle++, kind: 'prediction', owner_controller: owner};
        for (const [block, partition] of surfaces) {
          let predicted;
          if (planned.role === 'classifier') {
            const values = Array.from(model.predictLabels(block.matrix));
            requireValue(values.length === block.sampleIds.length && values.every(Number.isSafeInteger), 'Invalid native class predictions');
            predicted = values.map(value => [value]);
          } else predicted = rows(checkMatrix(model.predict(block.matrix), block.sampleIds));
          requireValue(predicted.every(row => row.length === this.targetNames.length), 'Native prediction target width mismatch');
          predictions.push({producer_node: node.node_id, producer_port: 'oof', partition, fold_id: task.fold_id, sample_ids: block.sampleIds, values: predicted, target_names: this.targetNames});
          if (task.phase !== 'PREDICT') targets.push({level: 'sample', unit_ids: block.sampleIds.map(id => ({level: 'sample', id})), values: rows(this.targets(block, task)), target_names: this.targetNames});
        }
      }
      return JSON.stringify({node_id: node.node_id, outputs, predictions, regression_targets: targets, artifacts, artifact_handles: artifactHandles,
        lineage: {record_id: 'lineage:n4m:' + this.digest(new TextEncoder().encode(JSON.stringify([task.run_id, node.node_id, task.phase, task.variant_id, task.fold_id]))),
          run_id: task.run_id, node_id: node.node_id, phase: task.phase, controller_id: owner, controller_version: manifest.controller_version,
          variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path, input_lineage: [], artifact_refs: artifacts,
          params_fingerprint: node.params_fingerprint, data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null,
          seed: null, unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: []}});
    } finally { model?.dispose(); }
  }
  bridge(owner, task) {
    requireValue(task.schema_version === 1 && this.controllers.has(owner), 'Invalid native bridge owner/schema');
    if (task.operation === 'export_artifact_payload') {
      const bytes = this.artifacts.get(task.artifact_id);
      requireValue(bytes, 'Unknown native artifact');
      return JSON.stringify({operation: 'exported_artifact_payload', schema_version: 1, payload: Array.from(bytes)});
    }
    if (task.operation === 'hydrate_artifact_payload') {
      const request = task.request, values = task.payload;
      requireValue(request.controller_id === owner && request.artifact.controller_id === owner && request.artifact.kind === 'n4m_estimator'
        && Array.isArray(values) && values.length > 0 && values.length <= 64 * 1024 * 1024 && values.every(value => Number.isInteger(value) && value >= 0 && value <= 255), 'Invalid native artifact payload');
      const bytes = Uint8Array.from(values);
      requireValue(this.digest(bytes) === request.artifact.content_fingerprint && bytes.length === request.artifact.size_bytes, 'Native artifact bytes differ from inventory');
      return JSON.stringify({operation: 'hydrated_artifact_payload', schema_version: 1,
        handle: this.handle(owner, 'model', this.models, {owner, artifact: request.artifact, bytes})});
    }
    requireValue(task.operation === 'release_hydrated_artifact_payload' && task.handle.owner_controller === owner && task.handle.kind === 'model', 'Invalid native release');
    const saved = this.models.get(task.handle.handle);
    requireValue(saved?.owner === owner, 'Unknown native model handle'); this.models.delete(task.handle.handle);
    return JSON.stringify({operation: 'released_hydrated_artifact_payload', schema_version: 1});
  }
  close() { this.features.clear(); this.models.clear(); this.artifacts.clear(); this.closed = true; }
}
