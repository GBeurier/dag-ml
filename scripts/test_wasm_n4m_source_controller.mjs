#!/usr/bin/env node
/** Real native source transforms, fit-scope isolation and no-fit replay. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const packageDir = path.resolve(process.argv[2]);
const methods = await import(pathToFileURL(path.join(path.resolve(process.argv[3]), 'index.js')).href);
await methods.loadModule();
const pkg = JSON.parse(fs.readFileSync(path.join(packageDir, 'package.json'), 'utf8'));
const dagMl = pkg.type === 'module'
  ? await import(pathToFileURL(path.join(packageDir, 'dag_ml_wasm.js')).href)
  : createRequire(import.meta.url)(path.join(packageDir, 'dag_ml_wasm.js'));
if (pkg.type === 'module') dagMl.initSync({ module: fs.readFileSync(path.join(packageDir, 'dag_ml_wasm_bg.wasm')) });
assert.equal(pkg.exports['./n4m-source-controller'].import, './n4m_source_controller.mjs');
const { N4mWasmSourceController } = await import(pathToFileURL(path.join(packageDir, 'n4m_source_controller.mjs')).href);
const sampleIds = ['a', 'b', 'c', 'heldout'];
const rows = { first: [[1, 4], [2, 1], [4, 3], [99, -32]], second: [[5], [2], [7], [-91]] };
const targets = [1, 3, 2, 123];
const dense = values => ({ data: Float64Array.from(values.flat()), rows: values.length, cols: values[0].length });
const recipe = { type: 'N4mSourcePipeline', hybrid: false,
  sourceOps: Object.keys(rows).map(source_id => ({ source_id, steps: [
    { methodId: 'preprocessing.scaling.standard_scale', params: { with_mean: true, with_std: true, ddof: 0 } },
  ] })),
  sourceSchemas: Object.fromEntries(Object.entries(rows).map(([id, values]) => [id, {
    representation: 'tabular_numeric', input_shape: [values[0].length],
    feature_names: values[0].map((_, i) => `feature:${i}`), identity: `fixture:${id}`,
  }])),
  steps: [{ methodId: 'models.regularized.ridge', params: { alpha: 0.2, scale_x: false } }],
};
const options = { methods, operators: { 'model:sources': recipe },
  digest: bytes => createHash('sha256').update(bytes).digest('hex'),
  resolveSource: ({ sourceId, sampleIds: requested }) => ({ sampleIds: requested,
    matrix: dense(requested.map(id => rows[sourceId][sampleIds.indexOf(id)])),
    featureNames: recipe.sourceSchemas[sourceId].feature_names }),
  resolveTargets: ({ sampleIds: requested }) => ({ sampleIds: requested,
    matrix: dense(requested.map(id => [targets[sampleIds.indexOf(id)]])) }),
};
const controller = new N4mWasmSourceController(options);
const manifest = controller.manifest(dagMl);
for (const name of ['x', 'x_original']) {
  const port = manifest.data_requirements.ports.find(port => port.name === name);
  assert.equal(port.multi_source, true);
  assert.equal(port.optional, true);
  assert.deepEqual(port.accepted_representations, ['tabular_numeric']);
}
assert.equal(manifest.data_requirements.default_fusion.mode, 'dict_by_source');
const task = { phase: 'REFIT', seed: null, fold_id: null, run_id: 'run:source-fixture', variant_id: 'variant:0',
  node_plan: { node_id: 'model:sources', kind: 'model', controller_id: controller.controllerId,
    controller_version: controller.controllerVersion, params: { source_composition_v1: recipe }, params_fingerprint: 'a'.repeat(64) },
  prediction_inputs: {}, input_handles: {},
  data_views: { x: { partition: 'full_train', sample_ids: sampleIds.slice(0, 3), source_ids: Object.keys(rows) },
    'x:test': { partition: 'predict', sample_ids: sampleIds.slice(3), source_ids: Object.keys(rows) } },
};
const invoke = (owner, value) => JSON.parse(owner.callback(owner.controllerId, JSON.stringify(value)));
try {
  const fitted = invoke(controller, task);
  assert.equal(fitted.predictions.find(value => value.partition === 'test').values.length, 1);
  const artifact = fitted.artifacts[0], payload = controller.artifactPayload(artifact.id);
  const saved = JSON.parse(new TextDecoder().decode(payload));
  assert.deepEqual(saved.source_states.map(states => states.length), [1, 1]);
  // Counterfactual heldout values cannot alter any fitted source state.
  rows.first[3] = [-991, 3200]; rows.second[3] = [9100];
  const other = new N4mWasmSourceController(options);
  try {
    const refit = invoke(other, { ...task, run_id: 'run:source-fixture-counterfactual' });
    const otherSaved = JSON.parse(new TextDecoder().decode(other.artifactPayload(refit.artifacts[0].id)));
    assert.deepEqual(otherSaved.source_states, saved.source_states);
    assert.deepEqual(otherSaved.head_states, saved.head_states);
    const request = { node_id: task.node_plan.node_id, controller_id: controller.controllerId,
      params_fingerprint: task.node_plan.params_fingerprint, artifact };
    const restored = new N4mWasmSourceController(options);
    const fit = methods.NativeEstimator.prototype.fit;
    methods.NativeEstimator.prototype.fit = () => { throw new Error('Replay must never fit a source'); };
    try {
      const handle = restored.hydrate(request, payload);
      const predicted = invoke(restored, { ...task, phase: 'PREDICT',
        data_views: { x: task.data_views['x:test'] }, artifact_inputs: { model: request }, input_handles: { model: handle } });
      assert.deepEqual(predicted.predictions[0].values, refit.predictions.find(value => value.partition === 'test').values);
    } finally { methods.NativeEstimator.prototype.fit = fit; restored.close(); }
  } finally { other.close(); }
} finally { controller.close(); }
console.log('NATIVE_SOURCE_CONTROLLER_OK');
