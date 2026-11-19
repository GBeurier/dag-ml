#!/usr/bin/env node
/** Classification mode of N4mWasmSourceController through the native DAG:
 * FIT_CV -> SELECT (balanced accuracy) -> REFIT -> portable PREDICT replay for
 * single-source, early fusion, probability stacking and native mean fusion,
 * plus the scientific refusals. Usage: <dag-ml-wasm package> <Methods dist>.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const harness = await import(pathToFileURL(path.resolve(path.dirname(new URL(import.meta.url).pathname),
  'fixtures/n4m_source_lab_harness.mjs')).href);
const { loadRuntimes, syntheticDataset, protocol, operator, runCandidate } = harness;
const { methods, dag, sourceModule } = await loadRuntimes(path.resolve(process.argv[2]), path.resolve(process.argv[3]));
const { N4mWasmSourceController } = sourceModule;
const dataset = syntheticDataset();
const proto = protocol(dag, dataset);
const classLabels = [...dataset.classes];
const mode = { task: 'classification', classLabels };
const P = 'source_composition_v1';
const logistic = (n = 3) => ({ methodId: 'models.classification.pls_logistic', params: { n_components: n, max_iter: 300 } });
const qda = { methodId: 'models.classification.pls_qda', params: { n_components: 2 } };
const scale = [{ methodId: 'preprocessing.scaling.standard_scale', params: { with_mean: true, with_std: true, ddof: 0 } }];
const ports = ['probabilities'];
const model = (id, sources, head, chains = {}) => {
  const op = operator(dataset, sources, [head], { chains });
  return { op, step: { kind: 'model', id, operator: op, params: { [P]: op }, prediction_output_ports: ports } };
};
const labelOf = new Map(dataset.sampleIds.map((id, i) => [id, dataset.labels[i]]));
const argmax = row => row.reduce((best, value, index) => value > row[best] ? index : best, 0);
const isDistribution = row => row.length === classLabels.length && row.every(v => v >= 0 && v <= 1) &&
  Math.abs(row.reduce((a, b) => a + b, 0) - 1) <= 1e-8;
const balancedAccuracy = (ids, predicted) => {
  const recall = classLabels.map((_, c) => {
    const members = ids.map((id, i) => [labelOf.get(id), predicted[i]]).filter(([truth]) => truth === c);
    return members.filter(([truth, guess]) => truth === guess).length / members.length;
  });
  return recall.reduce((a, b) => a + b, 0) / recall.length;
};
const sorted = ids => [...ids].sort();

function checkRun(name, run, { targetNode, labelPort, probabilityPort }) {
  const { outcome, replay, payloads } = run;
  assert.equal(outcome.score_set.selection_metric, 'balanced_accuracy', name);
  assert.equal(outcome.outputs[0].binding.prediction_kind, 'class_label', name);
  // Report-grade OOF labels cover the calibration cohort exactly once and are declared ids.
  const averaged = outcome.oof_averages.filter(block => block.predictions.producer_node === targetNode &&
    block.predictions.producer_port === labelPort && block.predictions.fold_id === 'avg');
  assert.equal(averaged.length, 1, `${name}: one report-grade OOF label block`);
  const oof = { ...averaged[0].predictions, sample_ids: averaged[0].predictions.unit_ids.map(unit => unit.id) };
  assert.deepEqual(sorted(oof.sample_ids), sorted(proto.calibrationIds), `${name}: OOF covers calibration identities`);
  assert.ok(oof.values.every(row => row.length === 1 && Number.isInteger(row[0]) && row[0] >= 0 && row[0] < classLabels.length));
  const oofScore = balancedAccuracy(oof.sample_ids, oof.values.map(row => row[0]));
  assert.ok(oofScore > 0.6, `${name}: separable synthetic classes give informative OOF (${oofScore})`);
  const report = outcome.score_set.reports.find(entry => entry.producer_node === targetNode && entry.producer_port === labelPort &&
    entry.fold_id === 'avg');
  assert.ok(report && Math.abs(report.metrics.balanced_accuracy - oofScore) < 1e-12, `${name}: native OOF balanced accuracy`);
  assert.ok(!('rmse' in report.metrics), `${name}: class-label reports carry classification metrics only`);
  // Replay: retained-test labels plus vocabulary-ordered probabilities, with no fitting.
  const labels = replay.outputs.find(output => output.binding.binding_id === 'output:prediction').predictions[0];
  const probabilities = replay.outputs.find(output => output.binding.binding_id === 'output:probabilities').predictions[0];
  assert.deepEqual(sorted(labels.sample_ids), sorted(proto.holdoutIds), `${name}: replay covers the retained test cohort`);
  assert.deepEqual(labels.sample_ids, probabilities.sample_ids, `${name}: labels and probabilities share identities`);
  assert.deepEqual(probabilities.target_names, classLabels.map(label => `quality:${label}`), `${name}: vocabulary-ordered columns`);
  assert.ok(probabilities.values.every(isDistribution), `${name}: replay probabilities are distributions`);
  labels.values.forEach((row, i) => assert.equal(row[0], argmax(probabilities.values[i]), `${name}: label is the probability argmax`));
  const testScore = balancedAccuracy(labels.sample_ids, labels.values.map(row => row[0]));
  assert.ok(testScore > 0.5, `${name}: informative retained-test labels (${testScore})`);
  for (const payload of payloads) {
    const saved = JSON.parse(new TextDecoder().decode(payload));
    assert.equal(saved.task, 'classification');
    assert.deepEqual(saved.class_labels, classLabels);
  }
  console.log(`${name}: OOF balanced accuracy ${oofScore.toFixed(3)}, retained test ${testScore.toFixed(3)}`);
  return { oof, labels, probabilities };
}

const single = (sources, chains) => {
  const m = model('model:result', sources, logistic(), chains);
  return { operators: { 'model:result': m.op }, nodes: [{ id: 'model:result', sourceIds: sources }], steps: [m.step],
    targetNode: 'model:result', outputPort: 'oof', probabilityPort: 'probabilities' };
};
const branches = () => {
  const a = model('model:nir', ['nir'], logistic(), { nir: scale }), b = model('model:meta', ['meta'], qda);
  return { a, b, nodes: [{ id: 'model:nir', sourceIds: ['nir'] }, { id: 'model:meta', sourceIds: ['meta'] }],
    branch: { kind: 'branch', branches: [{ id: 'nir', steps: [a.step] }, { id: 'meta', steps: [b.step] }] },
    sourcePorts: { 'model:nir': 'probabilities', 'model:meta': 'probabilities' } };
};

// (a) single source and (b) early fusion of two concatenated sources.
checkRun('single', runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto, mode,
  description: single(['nir'], { nir: scale }) }), { targetNode: 'model:result', labelPort: 'oof' });
checkRun('early', runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto, mode,
  description: single(['nir', 'meta'], { nir: scale, meta: scale }) }), { targetNode: 'model:result', labelPort: 'oof' });

// (c) late stacking: the meta-classifier sees only native validation OOF probabilities.
{
  const { a, b, nodes, branch, sourcePorts } = branches();
  const metaOp = operator(dataset, [], [logistic(2)]);
  const metaTasks = [];
  const wrap = callback => (owner, taskJson, seed) => {
    const task = JSON.parse(taskJson);
    if (task.node_plan?.node_id === 'model:result') metaTasks.push(task);
    return callback(owner, taskJson, seed);
  };
  const run = runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto, mode, wrap,
    description: { operators: { 'model:nir': a.op, 'model:meta': b.op, 'model:result': metaOp }, nodes,
      steps: [branch, { kind: 'merge_model', id: 'model:result', operator: metaOp, params: { [P]: metaOp }, include_original_data: false,
        source_ports: sourcePorts, prediction_output_ports: ports,
        metadata: { stacking_oof_execution: 'nested_oof_v1', stacking_refit_oof: 'partitioned_inner_v1' } }],
      targetNode: 'model:result', outputPort: 'oof', probabilityPort: 'probabilities' } });
  checkRun('stacking', run, { targetNode: 'model:result', labelPort: 'oof' });
  const fitting = metaTasks.filter(task => task.phase !== 'PREDICT');
  assert.ok(fitting.some(task => task.phase === 'FIT_CV') && fitting.some(task => task.phase === 'REFIT'));
  const outerTrain = new Map(proto.folds.folds.map(fold => [fold.fold_id, new Set(fold.train_sample_ids)]));
  for (const task of metaTasks) {
    assert.deepEqual(Object.keys(task.data_views ?? {}), [], 'A prediction-only meta-model receives no raw rows');
    const inputs = Object.entries(task.prediction_inputs);
    assert.ok(inputs.length >= 2);
    for (const [key, input] of inputs) {
      assert.equal(input.source_port, 'probabilities', `${key}: probabilities only, never labels`);
      assert.equal(input.prediction_width, classLabels.length);
      assert.ok(input.values.every(isDistribution), `${key}: authentic distributions`);
      const trainingInput = !/:(outer|refit|predict|test)$/.test(key);
      if (task.phase !== 'PREDICT' && trainingInput) {
        assert.equal(input.partition, 'validation', `${key}: meta fit rows are validation OOF predictions`);
        if (task.phase === 'FIT_CV') {
          assert.ok(!(input.fold_ids ?? []).includes(task.fold_id), `${key}: no current outer fold in meta fit`);
          // Inner OOF rows come only from the outer-train universe: never outer-validation rows.
          assert.ok(input.sample_ids.every(id => outerTrain.get(task.fold_id).has(id)), `${key}: inner OOF stays in outer train`);
        }
      }
    }
  }
  // A label input is refused by the classification meta-model.
  const probe = new N4mWasmSourceController({ methods, operators: { 'model:result': metaOp }, digest: harness.sha256, targetNames: ['quality'],
    task: 'classification', classLabels, resolveSource: () => { throw new Error('unused'); },
    resolveTargets: ({ sampleIds }) => ({ sampleIds, matrix: { rows: sampleIds.length, cols: 1, data: Float64Array.from(sampleIds, id => labelOf.get(id)) } }) });
  try {
    const task = structuredClone(metaTasks.find(item => item.phase === 'FIT_CV'));
    const [key, input] = Object.entries(task.prediction_inputs).find(([name]) => !name.endsWith(':outer'));
    task.prediction_inputs[key] = { ...input, source_port: 'oof', prediction_width: 1,
      values: input.sample_ids.map(id => [labelOf.get(id)]), target_names: ['quality'] };
    assert.throws(() => probe.invoke(probe.controllerId, JSON.stringify(task)), /never labels/);
  } finally { probe.close(); }
}

// (d) native late mean: fusion_proba_mean_label averages branch probabilities and emits argmax labels.
{
  const { a, b, nodes, branch, sourcePorts } = branches();
  const branchRows = new Map();
  const wrap = callback => (owner, taskJson, seed) => {
    const text = callback(owner, taskJson, seed), result = JSON.parse(text);
    for (const block of result.predictions ?? []) if (block.partition === 'validation' && block.producer_port === 'probabilities')
      block.sample_ids.forEach((id, i) => branchRows.set(`${JSON.parse(taskJson).variant_id}|${block.fold_id}|${block.producer_node}|${id}`, block.values[i]));
    return text;
  };
  const run = runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto, mode, wrap,
    description: { operators: { 'model:nir': a.op, 'model:meta': b.op }, nodes, nativeMerge: true,
      steps: [branch, { kind: 'merge', id: 'merge:mean', merge_mode: 'fusion_proba_mean_label', output_as: 'predictions',
        include_original_data: false, source_ports: sourcePorts, metadata: { refit_prediction_partition: 'final' } }],
      targetNode: 'merge:mean', outputPort: 'prediction', probabilityPort: 'probabilities' } });
  const { oof } = checkRun('mean', run, { targetNode: 'merge:mean', labelPort: 'prediction' });
  // Each fused OOF label is the argmax of the mean of both branches' outer-validation distributions.
  const foldOf = new Map(proto.folds.folds.flatMap(fold => fold.validation_sample_ids.map(id => [id, fold.fold_id])));
  const variant = run.outcome.selected_variant_id;
  oof.sample_ids.forEach((id, i) => {
    const rows = ['model:nir', 'model:meta'].map(node => branchRows.get(`${variant}|${foldOf.get(id)}|${node}|${id}`));
    assert.ok(rows.every(Boolean), `branch OOF distributions for ${id}`);
    assert.equal(oof.values[i][0], argmax(rows[0].map((value, c) => (value + rows[1][c]) / 2)), `mean fusion label for ${id}`);
  });
}

// (e) refusals.
const options = (operators, extra = {}) => ({ methods, operators, digest: harness.sha256, targetNames: ['quality'],
  resolveSource: () => { throw new Error('unused'); }, resolveTargets: () => { throw new Error('unused'); }, ...extra });
const ridge = { methodId: 'models.regularized.ridge', params: { alpha: 1 } };
assert.throws(() => new N4mWasmSourceController(options({ 'model:result': operator(dataset, ['nir'], [ridge]) }, mode)),
  /requires a native classifier head/, 'regressor head in classification');
assert.throws(() => new N4mWasmSourceController(options({ 'model:result': operator(dataset, ['nir'], [logistic()]) })),
  /Regression refuses classifier head/, 'classifier head in regression');
assert.throws(() => new N4mWasmSourceController(options({ 'model:result': operator(dataset, ['nir'], [logistic()],
  { chains: { nir: [{ methodId: 'models.pls.opls', params: {} }] } }) }, mode)), /requires a continuous y/, 'y-supervised source transform');
assert.throws(() => new N4mWasmSourceController(options({ 'model:result': operator(dataset, ['nir'], [logistic()]) },
  { task: 'classification', classLabels: ['low', 'low'] })), /unique nonempty class labels/);
assert.throws(() => new N4mWasmSourceController(options({ 'model:result': operator(dataset, ['nir'], [ridge]) },
  { classLabels })), /only in classification mode/);
// A vocabulary declaring an unobserved class is refused at the first fit scope.
assert.throws(() => runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto,
  mode: { task: 'classification', classLabels: [...classLabels, 'extra'] }, description: single(['nir'], {}) }),
/not the declared vocabulary ids/, 'vocabulary mismatch');
// Holding each class out as one outer fold leaves every fit scope without a class: refused, never padded.
{
  const ids = proto.calibrationIds;
  const folds = { id: 'lab:by-class', sample_ids: [...ids], sample_groups: {}, folds: classLabels.map((_, c) => ({
    fold_id: `fold${c}`, train_sample_ids: ids.filter(id => labelOf.get(id) !== c),
    validation_sample_ids: ids.filter(id => labelOf.get(id) === c), metadata: {} })) };
  assert.throws(() => runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset,
    proto: { ...proto, folds }, mode, description: single(['nir'], {}) }), /a fit scope lacks a class/, 'missing class in a fold');
}
// Payloads bind task and vocabulary in both directions.
{
  const run = runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto, mode, description: single(['nir'], {}) });
  const record = run.outcome.execution_bundle.refit_artifacts[0], payload = run.payloads[0];
  const request = { node_id: record.node_id, controller_id: record.artifact.controller_id, params_fingerprint: record.params_fingerprint,
    artifact: record.artifact };
  const operators = { 'model:result': operator(dataset, ['nir'], [logistic()]) };
  const reordered = new N4mWasmSourceController(options(operators, { task: 'classification', classLabels: [...classLabels].reverse() }));
  assert.throws(() => reordered.hydrate(request, payload), /task or class vocabulary differs/); reordered.close();
  const regression = new N4mWasmSourceController(options({ 'model:result': operator(dataset, ['nir'], [ridge]) }));
  assert.throws(() => regression.hydrate(request, payload), /belongs to another profile|task or class vocabulary differs/); regression.close();
}

// (f) regression mode keeps its payload profile: no classification fields.
{
  const pls = { methodId: 'models.pls.pls_regression', params: { n_components: 2 } };
  const op = operator(dataset, ['nir'], [pls]);
  const run = runCandidate({ dag, methods, SourceController: N4mWasmSourceController, dataset, proto, mode: { task: 'regression' },
    description: { operators: { 'model:result': op }, nodes: [{ id: 'model:result', sourceIds: ['nir'] }],
      steps: [{ kind: 'model', id: 'model:result', operator: op, params: { [P]: op } }], targetNode: 'model:result' } });
  const saved = JSON.parse(new TextDecoder().decode(run.payloads[0]));
  assert.ok(!('task' in saved) && !('class_labels' in saved));
  assert.equal(run.outcome.score_set.selection_metric, 'rmse');
}
console.log('NATIVE_SOURCE_CLASSIFICATION_OK');
