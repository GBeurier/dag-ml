/** Native Methods source composition. DAG owns every fold and OOF admission. */
import { N4mWasmRegressionController } from './n4m_controller.mjs';
import { strictJson } from './n4m_multimodal_controller.mjs';

export const N4M_SOURCE_PROFILE = 'source_composition.v1';
export const N4M_SOURCE_CONTROLLER_VERSION = '1.0.0';
export const N4M_SOURCE_RECIPE_PARAM = 'source_composition_v1';
const SCHEMA = 'dagml.methods.source_composition.v1';
const require = (condition, message) => { if (!condition) throw new Error(message); };
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const canonical = value => Array.isArray(value) ? value.map(canonical) : object(value) ?
  Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const ids = (value, label) => {
  require(Array.isArray(value) && value.length > 0 && value.every(id => typeof id === 'string' && id.length > 0) &&
    new Set(value).size === value.length, `${label} requires unique nonempty identities`);
  return value;
};
const matrix = (value, samples) => {
  const rows = typeof samples === 'number' ? samples : samples.length;
  require(value && value.data instanceof Float64Array && Number.isSafeInteger(value.cols) && value.cols > 0 &&
    value.rows === rows && value.data.length === value.rows * value.cols && value.data.every(Number.isFinite),
  'Source composition requires a complete finite Float64 matrix');
  return value;
};
const steps = value => {
  require(Array.isArray(value) && value.every(step => object(step) && typeof step.methodId === 'string' &&
    step.methodId.length > 0 && object(step.params ?? {}) && Object.keys(step).every(key => ['methodId', 'params'].includes(key))),
  'Native source steps require explicit methodId/params');
  return value;
};
const stateBytes = value => {
  require(Array.isArray(value) && value.length > 0 && value.every(byte => Number.isInteger(byte) && byte >= 0 && byte <= 255),
    'Fitted source state requires exact native bytes');
  return Uint8Array.from(value);
};

function operatorRecipe(operator) {
  require(object(operator) && operator.type === 'N4mSourcePipeline' && Array.isArray(operator.sourceOps) &&
    object(operator.sourceSchemas) && typeof operator.hybrid === 'boolean' && steps(operator.steps).length > 0 &&
    Object.keys(operator).every(key => ['type', 'sourceOps', 'sourceSchemas', 'steps', 'hybrid'].includes(key)),
  'An explicit source-composition recipe is required');
  const order = operator.sourceOps.map(source => source.source_id);
  if (order.length) ids(order, 'Source order');
  require(same([...Object.keys(operator.sourceSchemas)].sort(), [...order].sort()), 'Source schema inventory differs');
  for (const source of operator.sourceOps) {
    require(object(source) && Object.keys(source).every(key => ['source_id', 'steps'].includes(key)), 'Invalid source operation');
    steps(source.steps);
    const schema = operator.sourceSchemas[source.source_id];
    require(object(schema) && typeof schema.representation === 'string' && Array.isArray(schema.input_shape) &&
      schema.input_shape.length > 0 && schema.input_shape.every(width => Number.isSafeInteger(width) && width > 0) &&
      typeof schema.identity === 'string' && schema.identity.length > 0 && Array.isArray(schema.feature_names) &&
      schema.feature_names.length === schema.input_shape.reduce((n, width) => n * width, 1), 'Invalid source layout contract');
    ids(schema.feature_names, 'Source feature names');
  }
  require(!operator.hybrid || order.length > 0, 'Hybrid requires explicitly selected source operations');
  return structuredClone(operator);
}

function join(blocks) {
  require(blocks.length > 0, 'No raw or prediction features supplied');
  const sampleIds = ids(blocks[0].sampleIds, 'Joined rows');
  let width = 0;
  for (const block of blocks) {
    require(same(sampleIds, block.sampleIds), 'Raw and OOF identities differ');
    width += matrix(block.matrix, sampleIds).cols;
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
  ids(featureNames, 'Joined feature names');
  return { sampleIds, matrix: { data, rows: sampleIds.length, cols: width }, featureNames,
    blocks: blocks.map(block => block.matrix.cols) };
}

// RolePipeline deliberately requires a terminal predictor. A source chain is
// instead an ordered composition of generic native transform estimators; every
// step owns its parameters, fitted state and numerical operation in Methods.
class NativeSourceChain {
  constructor(methods, declarations) {
    this.methods = methods;
    this.declarations = declarations;
    this.estimators = [];
  }

  fit(X, y, manifest) {
    let current = X;
    for (const step of this.declarations) {
      const declaration = manifest.get(step.methodId);
      require(declaration, `Method ${step.methodId} is absent from the loaded native manifest`);
      const Estimator = this.methods.methodClass(step.methodId);
      const estimator = new Estimator();
      require(estimator instanceof this.methods.NativeEstimator && typeof estimator.transform === 'function',
        `Source method ${step.methodId} requires a native transform role`);
      this.estimators.push(estimator);
      estimator.params = structuredClone(step.params ?? {});
      // Route targets per step, never to target-independent preprocessing.
      estimator.fit(current, declaration.inputs?.y === 'required' ? y : undefined);
      current = estimator.transform(current);
      matrix(current, X.rows);
    }
  }

  transform(X) {
    return this.estimators.reduce((current, estimator) => {
      const transformed = estimator.transform(current);
      return matrix(transformed, X.rows);
    }, X);
  }

  exportStates() { return this.estimators.map(estimator => ({ n4me: estimator.toN4me() })); }
  dispose() { this.estimators.forEach(estimator => estimator.dispose()); this.estimators = []; }

  static fromStates(methods, declarations, states) {
    const chain = new NativeSourceChain(methods, declarations);
    try {
      states.forEach((state, index) => {
        const estimator = methods.NativeEstimator.fromN4me(state);
        chain.estimators.push(estimator);
        require(estimator.methodId === declarations[index].methodId && typeof estimator.transform === 'function',
          'Saved source state does not match its declared native transform method');
      });
      return chain;
    } catch (error) { chain.dispose(); throw error; }
  }
}

/** Separate artifact profile: source chains + ordered blocks + native head. */
export class N4mWasmSourceController extends N4mWasmRegressionController {
  constructor({ methods, operators, resolveSource, resolveTargets, targetNames = ['y'], digest,
    controllerId = 'controller:methods.wasm.sources', onTask = null }) {
    require(typeof resolveSource === 'function' && typeof methods.manifest === 'function', 'Methods manifest and source resolver required');
    const recipes = Object.fromEntries(Object.entries(operators).map(([id, operator]) => [id, operatorRecipe(operator)]));
    super({ methods, operators: recipes, resolveFeatures: () => { throw new Error('Use the declared source resolver'); },
      resolveTargets, targetNames, digest, controllerId, controllerVersion: N4M_SOURCE_CONTROLLER_VERSION });
    this.resolveSource = resolveSource;
    this.methodManifest = new Map(methods.manifest().methods.map(method => [method.method_id, method]));
    this.onTask = onTask;
    this.audit = [];
    this.callback = this.invoke.bind(this);
  }

  manifest(dagMl) {
    const names = ['x', 'x_original'];
    return JSON.parse(dagMl.derive_controller_manifest_json(JSON.stringify({
      controller_id: this.controllerId, controller_version: this.controllerVersion,
      operator_kind: 'model', added_capabilities: ['consumes_oof_predictions'], rng_policy: 'uses_core_seed',
      input_ports: names.map(name => ({ name, kind: 'data', representation: 'tabular_numeric',
        cardinality: 'optional', description: 'Explicit native source-matrix binding' })),
      data_requirements: { schema_version: 1, metadata: {},
        default_fusion: { mode: 'dict_by_source', alignment: 'sample_id', adapter_id: null, params: {} },
        ports: names.map(name => ({ name, accepted_representations: ['tabular_numeric'],
          accepted_types: ['table'], rank: 2, multi_source: true, optional: true, metadata: {} })) },
    })));
  }

  _recipe(task) {
    const recipe = operatorRecipe(this.operators[task.node_plan.node_id]);
    require(object(task.node_plan.params) && Object.keys(task.node_plan.params).length === 1 &&
      same(task.node_plan.params[N4M_SOURCE_RECIPE_PARAM], recipe),
    'Source composition requires its complete immutable recipe in source_composition_v1 node parameters');
    return recipe;
  }

  _raw(task, recipe, partition, requested) {
    return recipe.sourceOps.map(source => {
      const available = Object.values(task.data_views ?? {}).filter(view => view.partition === partition &&
        (view.source_ids ?? []).includes(source.source_id));
      require(available.length > 0, `No native ${partition} view admits source ${source.source_id}`);
      const allowed = new Set(available.flatMap(view => view.sample_ids ?? []));
      const sampleIds = requested ?? available[0].sample_ids;
      ids(sampleIds, 'Native source cohort');
      require(sampleIds.every(id => allowed.has(id)), 'A source resolver request exceeds its native fit/predict scope');
      const result = this.resolveSource({ sourceId: source.source_id, sampleIds: [...sampleIds], task: structuredClone(task) });
      require(result && typeof result.then !== 'function' && same(result.sampleIds, sampleIds), 'Source resolver must preserve native ordered identities');
      matrix(result.matrix, sampleIds);
      const schema = recipe.sourceSchemas[source.source_id];
      require(result.matrix.cols === schema.feature_names.length && same(result.featureNames, schema.feature_names),
        'Resolved source shape/feature names disagree with the declared recipe');
      return { sampleIds: [...sampleIds], matrix: result.matrix, featureNames: [...result.featureNames], sourceId: source.source_id };
    });
  }

  _cohort(task, recipe, outer) {
    const meta = Object.keys(task.prediction_inputs ?? {}).length > 0;
    const prediction = meta ? this._predictions(task, outer) : null;
    require(meta || !recipe.hybrid, 'Hybrid source composition requires native OOF prediction inputs');
    const refitInputs = Object.entries(task.prediction_inputs ?? {}).filter(([key]) => key.endsWith(':refit'));
    const refitPredict = task.phase === 'REFIT' && outer && prediction !== null &&
      refitInputs.some(([, input]) => input.partition === 'test');
    const partition = task.phase === 'FIT_CV' ? (outer ? 'fold_validation' : 'fold_train') :
      task.phase === 'REFIT' && !refitPredict ? 'full_train' : 'predict';
    const raw = recipe.sourceOps.length ? this._raw(task, recipe, partition, prediction?.sampleIds) : [];
    require(!meta || recipe.hybrid || raw.length === 0, 'Prediction-only meta-model must not silently discard raw features');
    require(raw.length > 0 || prediction, 'Source composition needs data or OOF features');
    return { raw, prediction, sampleIds: prediction?.sampleIds ?? raw[0].sampleIds, refitPredict };
  }

  _encoded(cohort, recipe, chains) {
    const blocks = cohort.raw.map((block, index) => {
      const chain = chains[index];
      const transformed = chain ? chain.transform(block.matrix, block.featureNames) : block.matrix;
      matrix(transformed, block.sampleIds);
      const names = transformed.cols === block.featureNames.length ? block.featureNames :
        Array.from({ length: transformed.cols }, (_, column) => `component:${column}`);
      return { ...block, matrix: transformed, featureNames: names.map(name => `source:${block.sourceId}/${name}`) };
    });
    if (cohort.prediction) blocks.push({ ...cohort.prediction,
      featureNames: cohort.prediction.featureNames.map(name => `prediction:${name}`) });
    return join(blocks);
  }

  _fit(task, recipe, train, seed) {
    const chains = [];
    let head;
    try {
      const y = this._targets(train.sampleIds, task);
      for (let index = 0; index < recipe.sourceOps.length; index++) {
        const source = recipe.sourceOps[index], block = train.raw[index];
        if (!source.steps.length) { chains.push(null); continue; }
        const chain = new NativeSourceChain(this.methods, source.steps);
        chains.push(chain);
        chain.fit(block.matrix, y, this.methodManifest);
      }
      const features = this._encoded(train, recipe, chains);
      const effectiveSteps = structuredClone(recipe.steps);
      for (const step of effectiveSteps) if (step.methodId === 'models.neural.multibranch_regressor') {
        require(seed !== null, 'Learned branches require an exact native task seed');
        step.params = { ...step.params, seed: BigInt.asIntN(64, BigInt(seed)).toString() };
        require(Array.isArray(step.params.branch_shapes) && step.params.branch_shapes.length === features.blocks.length * 3,
          'Native multibranch shapes require three integer dimensions per block');
        const shapes = features.blocks.map((_, index) => step.params.branch_shapes.slice(index * 3, index * 3 + 3));
        require(shapes.length === features.blocks.length && shapes.every((shape, index) =>
          shape.every(width => Number.isSafeInteger(width) && width > 0) &&
          shape.reduce((n, width) => n * width, 1) === features.blocks[index]), 'CNN shapes disagree with post-transform block widths');
      }
      head = this.methods.RolePipeline.fromSteps(effectiveSteps);
      const acceptsBlocks = effectiveSteps.some(step => {
        const declaration = this.methodManifest.get(step.methodId);
        require(declaration, `Method ${step.methodId} is absent from the loaded native manifest`);
        return declaration.inputs?.blocks === 'required' || declaration.inputs?.blocks === 'optional';
      });
      head.fit(features.matrix, y, { featureNames: features.featureNames, ...(acceptsBlocks ? { blocks: features.blocks } : {}) });
      this.audit.push({ nodeId: task.node_plan.node_id, phase: task.phase, foldId: task.fold_id,
        sampleIds: [...train.sampleIds], sourceIds: recipe.sourceOps.map(source => source.source_id),
        blockWidths: [...features.blocks], featureNames: [...features.featureNames], seed });
      return { model: head, chains, recipe, effectiveSteps, featureNames: features.featureNames, blockWidths: features.blocks };
    } catch (error) {
      head?.dispose(); chains.forEach(chain => chain?.dispose()); throw error;
    }
  }

  _dispose(entry) { entry.model.dispose(); entry.chains.forEach(chain => chain?.dispose()); }

  hydrate(request, payload) {
    require(!this.closed && request.controller_id === this.controllerId && payload instanceof Uint8Array &&
      this._hash(payload) === request.artifact.content_fingerprint, 'Invalid source-composition payload owner or hash');
    const saved = strictJson(new TextDecoder('utf-8', { fatal: true }).decode(payload));
    require(saved.schema === SCHEMA && saved.profile === N4M_SOURCE_PROFILE &&
      saved.controller_version === this.controllerVersion && saved.node_id === request.node_id &&
      saved.params_fingerprint === request.params_fingerprint && same(saved.target_names, this.targetNames),
    'Source-composition payload belongs to another profile/node/target');
    const recipe = operatorRecipe(saved.recipe);
    require(same(recipe, operatorRecipe(this.operators[request.node_id])),
      'Saved source operations, schemas or head parameters differ from the declared operator');
    ids(saved.feature_names, 'Fitted feature names');
    require(Array.isArray(saved.block_widths) && saved.block_widths.length === recipe.sourceOps.length + (saved.prediction_width ? 1 : 0) &&
      saved.block_widths.every(width => Number.isSafeInteger(width) && width > 0) &&
      saved.block_widths.reduce((n, width) => n + width, 0) === saved.feature_names.length &&
      Array.isArray(saved.source_states) && saved.source_states.length === recipe.sourceOps.length,
    'Incomplete fitted source/block inventory');
    const chains = [];
    let head;
    try {
      recipe.sourceOps.forEach((source, index) => {
        const states = saved.source_states[index];
        require(Array.isArray(states) && states.length === source.steps.length, 'Source chain/state count differs');
        chains.push(source.steps.length ? NativeSourceChain.fromStates(this.methods, source.steps, states.map(stateBytes)) : null);
      });
      require(Array.isArray(saved.head_states) && saved.head_states.length === saved.effective_steps.length, 'Head state/step inventory differs');
      steps(saved.effective_steps);
      const expected = structuredClone(recipe.steps);
      for (const step of expected) if (step.methodId === 'models.neural.multibranch_regressor')
        step.params.seed = BigInt.asIntN(64, BigInt(saved.execution_seed)).toString();
      require(same(expected, saved.effective_steps), 'Effective head recipe differs');
      head = this.methods.RolePipeline.fromStates(saved.effective_steps, saved.head_states.map(stateBytes), { featureNames: saved.feature_names });
      return this._handle({ model: head, chains, recipe, effectiveSteps: saved.effective_steps,
        featureNames: saved.feature_names, blockWidths: saved.block_widths, nodeId: saved.node_id,
        paramsFingerprint: saved.params_fingerprint, artifactId: request.artifact.id,
        contentFingerprint: request.artifact.content_fingerprint });
    } catch (error) { head?.dispose(); chains.forEach(chain => chain?.dispose()); throw error; }
  }

  invoke(controllerId, taskJson, exactSeed = null) {
    require(!this.closed && controllerId === this.controllerId, 'Wrong or closed source controller');
    const task = strictJson(taskJson);
    if (task.operation) {
      // Reuse the existing attested byte transport, but release all source chains.
      if (task.operation === 'release_hydrated_artifact_payload') {
        require(task.schema_version === 1 && task.handle?.owner_controller === this.controllerId, 'Foreign source handle');
        const entry = this.models.get(task.handle.handle);
        require(entry, 'Unknown source handle'); this._dispose(entry); this.models.delete(task.handle.handle);
        return JSON.stringify({ operation: 'released_hydrated_artifact_payload', schema_version: 1 });
      }
      return super.invoke(controllerId, taskJson);
    }
    const node = task.node_plan;
    require(node.kind === 'model' && node.controller_id === this.controllerId && node.controller_version === this.controllerVersion &&
      ['FIT_CV', 'REFIT', 'PREDICT'].includes(task.phase), 'Invalid source controller task');
    require(!Object.keys(task.data_view_receipts ?? {}).length && !(task.required_loss_attestations ?? []).length &&
      !task.residual_targets && (!task.fit_influence || task.fit_influence.mechanism === 'uniform_rows' &&
      !(task.fit_influence.row_weights ?? []).length), 'Source profile does not support generated views/custom losses/weighted influence');
    require(!(task.phase === 'FIT_CV' && Object.keys(task.prediction_inputs ?? {}).some(key => key.endsWith(':test'))),
      'Additional CV test streams require a specialized controller');
    const token = exactSeed ?? strictJson(taskJson, 'seed');
    const seed = token === 'null' || token === null ? null : String(token);
    require(seed === null || /^\d+$/.test(seed) && BigInt(seed) <= 18446744073709551615n, 'Task seed must be an exact decimal u64');
    this.onTask?.({ nodeId: node.node_id, phase: task.phase, foldId: task.fold_id });
    const recipe = this._recipe(task);
    let entry, kept = false;
    try {
      if (task.phase === 'PREDICT') {
        const inputs = Object.entries(task.artifact_inputs ?? {});
        require(inputs.length === 1, 'PREDICT requires one source-composition artifact');
        const [key, input] = inputs[0], handle = task.input_handles?.[key];
        require(handle?.owner_controller === this.controllerId && handle.kind === 'model', 'Foreign fitted source model');
        entry = this.models.get(handle.handle);
        require(entry && entry.nodeId === node.node_id && entry.paramsFingerprint === node.params_fingerprint &&
          input.params_fingerprint === node.params_fingerprint && input.artifact?.id === entry.artifactId &&
          input.artifact?.content_fingerprint === entry.contentFingerprint && same(entry.recipe, recipe), 'Fitted source recipe binding differs');
        kept = true;
      } else {
        const train = this._cohort(task, recipe, false);
        if (task.phase === 'FIT_CV') {
          const valid = this._cohort(task, recipe, true);
          require(train.sampleIds.every(id => !valid.sampleIds.includes(id)), 'Source fit and validation identities overlap');
        }
        entry = this._fit(task, recipe, train, seed);
      }
      const cohort = this._cohort(task, recipe, task.phase !== 'REFIT' ||
        Object.keys(task.prediction_inputs ?? {}).some(key => key.endsWith(':refit')));
      const features = this._encoded(cohort, recipe, entry.chains);
      require(same(features.featureNames, entry.featureNames) && same(features.blocks, entry.blockWidths), 'Fitted feature order or block widths differ');
      if (task.phase === 'REFIT') {
        const state = { schema: SCHEMA, profile: N4M_SOURCE_PROFILE, controller_version: this.controllerVersion,
          node_id: node.node_id, params_fingerprint: node.params_fingerprint, target_names: this.targetNames,
          recipe, execution_seed: seed, effective_steps: entry.effectiveSteps, feature_names: entry.featureNames,
          block_widths: entry.blockWidths, prediction_width: cohort.prediction?.matrix.cols ?? 0,
          source_states: entry.chains.map(chain => chain ? chain.exportStates().map(state => Array.from(state.n4me)) : []),
          head_states: entry.model.exportStates().map(state => Array.from(state.n4me)) };
        const payload = new TextEncoder().encode(JSON.stringify(state)), hash = this._hash(payload);
        const artifactId = `artifact:sources:${this._hash(new TextEncoder().encode(JSON.stringify([task.run_id, node.node_id, task.variant_id])))}`;
        require(!this.artifacts.has(artifactId), 'Duplicate source REFIT artifact');
        const artifact = { id: artifactId, kind: 'methods_source_pipeline', controller_id: this.controllerId,
          backend: 'raw', uri: `artifacts/${hash}.json`, content_fingerprint: hash, size_bytes: payload.length,
          plugin: 'dagml.methods.wasm.sources', plugin_version: this.controllerVersion };
        const result = strictJson(this._result(task, features, entry.model, [artifact], {}));
        if (cohort.refitPredict) result.predictions[0].partition = 'test';
        // A native REFIT task may carry an explicitly admitted test data view.
        // Emit its predictions in addition to the final training-cohort rows;
        // no raw test row is ever used to fit a source chain or the head.
        if (!cohort.prediction && Object.values(task.data_views ?? {}).some(view => view.partition === 'predict')) {
          const testRaw = this._raw(task, recipe, 'predict');
          const testFeatures = this._encoded({ raw: testRaw, prediction: null }, recipe, entry.chains);
          require(same(testFeatures.featureNames, entry.featureNames) && same(testFeatures.blocks, entry.blockWidths),
            'REFIT test feature/block contract differs');
          const testResult = strictJson(this._result(task, testFeatures, entry.model, [], {}));
          testResult.predictions[0].partition = 'test';
          result.predictions.push(...testResult.predictions);
        }
        const handle = this._handle({ ...entry, nodeId: node.node_id, paramsFingerprint: node.params_fingerprint,
          artifactId, contentFingerprint: hash });
        this.artifacts.set(artifactId, { payload, handle }); result.artifact_handles = { [artifactId]: handle }; kept = true;
        return JSON.stringify(result);
      }
      return this._result(task, features, entry.model, [], {});
    } finally { if (entry && !kept) this._dispose(entry); }
  }

  close() {
    if (this.closed) return;
    for (const entry of this.models.values()) this._dispose(entry);
    this.models.clear(); this.artifacts.clear(); this.closed = true;
  }
}
