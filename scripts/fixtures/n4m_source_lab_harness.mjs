/** Native browser-lab harness for N4mWasmSourceController end-to-end tests.
 * It mirrors the multimodal web host: native folds, DSL compilation, signed
 * training, portable package replay. Hosts keep data; DAG-ML keeps every fold,
 * OOF admission, aggregation, score and selection decision.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ?
  Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
export const fingerprint = value => sha256(new TextEncoder().encode(JSON.stringify(canonical(value))));

export async function loadRuntimes(packageDir, methodsDir) {
  const methods = await import(pathToFileURL(path.join(path.resolve(methodsDir), 'index.js')).href);
  await methods.loadModule();
  const pkg = JSON.parse(fs.readFileSync(path.join(packageDir, 'package.json'), 'utf8'));
  const dag = pkg.type === 'module'
    ? await import(pathToFileURL(path.join(packageDir, 'dag_ml_wasm.js')).href)
    : createRequire(import.meta.url)(path.join(packageDir, 'dag_ml_wasm.js'));
  if (pkg.type === 'module') dag.initSync({ module: fs.readFileSync(path.join(packageDir, 'dag_ml_wasm_bg.wasm')) });
  const sourceModule = await import(pathToFileURL(path.join(packageDir, 'n4m_source_controller.mjs')).href);
  return { methods, dag, sourceModule, pkg };
}

/** Deterministic synthetic two-source dataset; class means separate both sources. */
export function syntheticDataset({ samples = 72, classes = ['low', 'mid', 'high'], seed = 11, noiseScale = 1.6 } = {}) {
  let state = seed >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  const noise = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  const sampleIds = Array.from({ length: samples }, (_, i) => `s${String(i).padStart(3, '0')}`);
  const labels = sampleIds.map((_, i) => i % classes.length);
  const make = (width, scale, shift) => {
    const data = new Float64Array(samples * width);
    labels.forEach((label, row) => {
      for (let col = 0; col < width; col++) {
        const signal = label * scale * Math.sin((col + 1) * (label + 1) * 0.7) + shift * Math.cos(col * 0.3);
        data[row * width + col] = signal + noiseScale * noise();
      }
    });
    return { data, width, featureNames: Array.from({ length: width }, (_, c) => `f${c}`) };
  };
  const sources = { nir: make(10, 1.4, 0.5), meta: make(4, 1.0, 0.2) };
  const values = labels.map(label => label + 0.37 * label * label + 0.2 * noise());
  return { sampleIds, labels, classes, sources, regressionTargets: values };
}

export function sourceSchemas(dataset, ids) {
  return Object.fromEntries(ids.map(id => [id, { representation: 'tabular_numeric',
    input_shape: [dataset.sources[id].width], feature_names: [...dataset.sources[id].featureNames],
    identity: `fixture:${id}` }]));
}

export function operator(dataset, sourceIds, steps, { chains = {}, hybrid = false } = {}) {
  return { type: 'N4mSourcePipeline', sourceOps: sourceIds.map(source_id => ({ source_id, steps: chains[source_id] ?? [] })),
    sourceSchemas: sourceSchemas(dataset, sourceIds), steps, hybrid };
}

function envelopeFor(dataset, ids, inference = false) {
  const sources = Object.keys(dataset.sources);
  const plan = { id: 'lab:source-matrices', output_representation: 'tabular_numeric', issues: [], steps: [
    ...sources.map(source => ({ kind: 'materialize', source_id: source, adapter_id: null, input_representation: null,
      output_representation: 'tabular_numeric', fit_scope: 'stateless', requires_user_choice: false,
      metadata: { output: `src:${source}`, original_representation: 'tabular_numeric', shape: [dataset.sources[source].width], layout: 'sample_feature' } })),
    { kind: 'join', source_id: null, adapter_id: null, input_representation: 'tabular_numeric', output_representation: 'tabular_numeric',
      fit_scope: 'stateless', requires_user_choice: false, metadata: { inputs: sources.map(source => `src:${source}`), output: 'port:X' } },
  ] };
  const records = ids.flatMap(sampleId => sources.map(source => ({ observation_id: `observation:${fingerprint([sampleId, source])}`,
    sample_id: sampleId, target_id: inference ? null : `target:${fingerprint([sampleId, 'y'])}`, group_id: null,
    origin_sample_id: null, source_id: source, is_augmented: false })));
  const relations = { records };
  return { relations, envelope: (dag) => ({ schema_version: 1, schema_fingerprint: fingerprint(sources),
    plan_fingerprint: fingerprint(plan), relation_fingerprint: dag.sample_relation_set_fingerprint_json(JSON.stringify(relations)),
    plan, coordinator_relations: relations, data_content_fingerprint: fingerprint({ ids, sources }),
    target_content_fingerprint: inference ? null : fingerprint(ids.map(id => [id, dataset.labels[dataset.sampleIds.indexOf(id)]])),
    metadata: {} }) };
}

function binding(nodeId, inputName, sources, envelope) {
  return { node_id: nodeId, input_name: inputName, request_id: envelope.plan.id, schema_fingerprint: envelope.schema_fingerprint,
    plan_fingerprint: envelope.plan_fingerprint, relation_fingerprint: envelope.relation_fingerprint,
    output_representation: 'tabular_numeric', feature_set_id: 'x', source_ids: sources, require_relations: true,
    view_policy: { fit_partition: 'fold_train', predict_partition: 'fold_validation', include_augmented_train: false,
      include_augmented_validation: false, include_excluded: false, require_sample_ids: true }, metadata: {} };
}

/** Stratified native outer protocol: one stratified fold is the retained test cohort. */
export function protocol(dag, dataset, { outerFolds = 3, seed = 5 } = {}) {
  const strata = Object.fromEntries(dataset.sampleIds.map((id, i) => [id, dataset.classes[dataset.labels[i]]]));
  const holdout = JSON.parse(dag.stratified_kfold_split_json(JSON.stringify({ n_splits: 4, shuffle: true, seed }),
    JSON.stringify(dataset.sampleIds), JSON.stringify(strata), 'lab:holdout')).folds[0];
  const calibrationIds = holdout.train_sample_ids, holdoutIds = holdout.validation_sample_ids;
  const calibrationStrata = Object.fromEntries(calibrationIds.map(id => [id, strata[id]]));
  const folds = JSON.parse(dag.stratified_kfold_split_json(JSON.stringify({ n_splits: outerFolds, shuffle: true, seed: seed + 1 }),
    JSON.stringify(calibrationIds), JSON.stringify(calibrationStrata), 'lab:outer'));
  return { calibrationIds, holdoutIds, folds, strata: calibrationStrata };
}

/**
 * Run one candidate. `description` = {operators, nodes:[{id, sourceIds, inputName}], steps, targetNode, outputPort}.
 * `mode` = {task, classLabels} for classification, or {task:'regression'}.
 */
export function runCandidate({ dag, methods, SourceController, dataset, proto, description, mode, wrap = null,
  controllerOptions = {}, innerFolds = 3 }) {
  const classification = mode.task === 'classification';
  const targetName = classification ? 'quality' : 'y';
  const index = new Map(dataset.sampleIds.map((id, i) => [id, i]));
  const { envelope: makeEnvelope } = envelopeFor(dataset, proto.calibrationIds);
  const trainingEnvelope = makeEnvelope(dag);
  const options = { methods, operators: description.operators, digest: sha256, targetNames: [targetName],
    resolveSource: ({ sourceId, sampleIds }) => {
      const source = dataset.sources[sourceId], data = new Float64Array(sampleIds.length * source.width);
      sampleIds.forEach((id, row) => { const i = index.get(id); data.set(source.data.subarray(i * source.width, (i + 1) * source.width), row * source.width); });
      return { sampleIds: [...sampleIds], matrix: { data, rows: sampleIds.length, cols: source.width }, featureNames: [...source.featureNames] };
    },
    resolveTargets: ({ sampleIds }) => ({ sampleIds: [...sampleIds], matrix: { rows: sampleIds.length, cols: 1,
      data: Float64Array.from(sampleIds, id => classification ? dataset.labels[index.get(id)] : dataset.regressionTargets[index.get(id)]) } }),
    ...(classification ? { task: 'classification', classLabels: mode.classLabels } : {}), ...controllerOptions };
  const controller = new SourceController(options);
  try {
    const manifests = [controller.manifest(dag)];
    // A native fixed reduction needs only an identity manifest; it never calls a host reducer.
    if (description.nativeMerge) manifests.push(JSON.parse(dag.derive_controller_manifest_json(JSON.stringify({
      controller_id: 'controller:native.prediction-merge', controller_version: '1.0.0',
      operator_kind: 'prediction_join', added_capabilities: ['consumes_oof_predictions', 'emits_predictions'] }))));
    const guard = callback => (owner, task, seed) => {
      if (owner === 'controller:native.prediction-merge') throw new Error('The native merge unexpectedly requested a host reducer');
      return callback(owner, task, seed);
    };
    const policy = { split_unit: 'sample', forbid_origin_cross_fold: true, allow_observation_split_with_shared_target: false,
      require_group_ids: false, unsafe_flags: [] };
    const bindings = description.nodes.map(node => binding(node.id, node.inputName ?? 'x', node.sourceIds, trainingEnvelope));
    const dsl = { id: 'dsl:lab', campaign_id: 'campaign:lab', input: { name: 'x', representation: 'tabular_numeric' }, root_seed: 7,
      leakage_policy: policy,
      inner_cv: classification ? { kind: 'stratified_kfold', n_splits: innerFolds, shuffle: true, seed: 7, strata: proto.strata } :
        { kind: 'kfold', n_splits: innerFolds, shuffle: true, seed: 7 },
      split_invocation: { id: 'split:lab.outer', controller_id: null, leakage_policy: policy, params: {}, fold_set: proto.folds },
      data_bindings: bindings, steps: description.steps };
    const compiled = JSON.parse(dag.compile_pipeline_dsl_artifact_with_controllers_json(JSON.stringify(dsl), JSON.stringify(manifests)));
    const labelOutput = classification ? { prediction_kind: 'class_label', class_labels: [[...mode.classLabels]] } :
      { prediction_kind: 'regression_point', class_labels: [[]] };
    const outputs = [{ output_id: 'output:prediction', node_id: description.targetNode, port_name: description.outputPort,
      prediction_level: 'sample', unit_level: 'physical_sample', ...labelOutput, target_names: [targetName], target_units: [null],
      output_order: 'target_order', target_space: 'raw' }];
    if (classification && description.probabilityPort) outputs.push({ output_id: 'output:probabilities', node_id: description.targetNode,
      port_name: description.probabilityPort, prediction_level: 'sample', unit_level: 'physical_sample', prediction_kind: 'class_probability',
      target_names: [targetName], target_units: [null], class_labels: [[...mode.classLabels]], output_order: 'target_major_class_minor', target_space: 'raw' });
    const request = { schema_version: 1, request_id: 'training:lab', plan_id: 'plan:lab', graph: compiled.graph,
      campaign: compiled.campaign_template, controller_manifests: manifests,
      data_identities: bindings.map(value => JSON.parse(dag.training_data_identity_json(JSON.stringify(value), JSON.stringify(trainingEnvelope))))
        .sort((a, b) => a.requirement_key.localeCompare(b.requirement_key)),
      parameter_patches: [], patch_policies: [], influence_requirements: [], request_fingerprint: '0'.repeat(64),
      options: { refit: true, refit_strategy: 'refit_one', seed: 7,
        selection: { id: 'selection:fixed', metric: classification ? { name: 'balanced_accuracy', objective: 'maximize' } :
          { name: 'rmse', objective: 'minimize' }, require_finite: true, evaluation_scope: 'oof' },
        selection_output_id: 'output:prediction', outputs,
        scheduler: { kind: 'sequential', backend: null, workers: 1 }, resources: { cpu_threads: 1, memory_bytes: null, gpu_devices: [], wall_time_ms: null },
        artifacts: { cv_artifacts: 'discard', prediction_caches: 'retain', fitted_artifacts: 'portable_required' } } };
    const signed = dag.sign_training_request_json(JSON.stringify(request));
    const envelopes = Object.fromEntries(bindings.map(value => [`${value.node_id}.${value.input_name}`, trainingEnvelope]));
    const invoke = guard(wrap ? wrap(controller.callback) : controller.callback);
    const capture = JSON.parse(dag.execute_training_json(signed, JSON.stringify(envelopes), JSON.stringify(trainingEnvelope.coordinator_relations),
      'package:lab', 'outcome:lab', 'run:lab', 'bundle:lab', invoke));
    const outcome = JSON.parse(capture.training_outcome_json);
    const payloads = outcome.execution_bundle?.refit_artifacts?.map(record => controller.artifactPayload(record.artifact.id)) ?? [];
    // Retained-test replay through a fresh controller that may never fit.
    const { envelope: makeInference } = envelopeFor(dataset, proto.holdoutIds, true);
    const inference = makeInference(dag);
    const predictEnvelope = JSON.parse(dag.attach_predict_cohort_to_envelope_json(JSON.stringify(trainingEnvelope), JSON.stringify({
      role: 'inference', target_names: [targetName], data_content_fingerprint: inference.data_content_fingerprint,
      target_content_fingerprint: null, relations: inference.coordinator_relations })));
    const predictEnvelopes = Object.fromEntries(Object.keys(envelopes).map(key => [key, predictEnvelope]));
    const packageView = JSON.parse(capture.portable_predictor_package_json);
    const outputIds = outputs.map(output => output.output_id);
    const replayRequest = dag.sign_training_replay_request_json(JSON.stringify({ schema_version: 1, request_id: 'replay:lab',
      source_outcome_fingerprint: packageView.training_outcome.outcome_fingerprint, phase: 'PREDICT',
      data_envelope_keys: Object.keys(predictEnvelopes).sort(), output_binding_ids: outputIds, request_fingerprint: '0'.repeat(64) }));
    const replayController = new SourceController(options);
    const nativeFit = methods.NativeEstimator.prototype.fit, pipelineFit = methods.RolePipeline.prototype.fit;
    methods.NativeEstimator.prototype.fit = () => { throw new Error('Replay must never fit a source step'); };
    methods.RolePipeline.prototype.fit = () => { throw new Error('Replay must never fit a head'); };
    let replay;
    try {
      replay = JSON.parse(dag.replay_training_package_json(capture.portable_predictor_package_json, replayRequest,
        JSON.stringify(predictEnvelopes), JSON.stringify(manifests), 'outcome:lab.test', 'run:lab.test',
        guard(wrap ? wrap(replayController.callback) : replayController.callback)));
    } finally {
      methods.NativeEstimator.prototype.fit = nativeFit; methods.RolePipeline.prototype.fit = pipelineFit; replayController.close();
    }
    return { capture, outcome, packageView, replay, payloads, audit: controller.audit, dsl, compiled, request };
  } finally { controller.close(); }
}
