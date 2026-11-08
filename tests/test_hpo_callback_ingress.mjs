// Public HPO ingress controls with owner-generated worker JSON and a local host.
// These tests cover generic coordination, not native/IO/scientific qualification.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { strictJson } from '../bindings/js/n4m_multimodal_controller.mjs';

const require = createRequire(import.meta.url);
assert.ok(process.env.DAG_ML_WASM_PKG && path.isAbsolute(process.env.DAG_ML_WASM_PKG), 'DAG_ML_WASM_PKG must be an absolute file path');
assert.ok(statSync(process.env.DAG_ML_WASM_PKG).isFile(), 'DAG_ML_WASM_PKG must be a file');
assert.notEqual(process.env.HPO_INSPECT, '1', 'inspection mode skips the mandatory suite');
const dag = require(process.env.DAG_ML_WASM_PKG);
const { ridgeNodeResult } = require('../scripts/hpo_ridge_operator.cjs');
const graph = {
  id: 'graph:wasm-hpo', interface: { inputs: [], outputs: [] },
  nodes: [{ id: 'model:m', kind: 'model', operator: { type: 'JSModel' }, params: {},
    ports: { inputs: [], outputs: [{ name: 'oof', kind: 'prediction', representation: null, cardinality: 'one', description: '' }] },
    metadata: {}, seed_label: null }],
  edges: [], search_space_fingerprint: null, metadata: {},
};
const manifest = JSON.parse(dag.derive_controller_manifest_json(JSON.stringify({
  controller_id: 'controller:js-hpo-model', controller_version: '1.0.0', operator_kind: 'model',
  input_ports: [], output_ports: graph.nodes[0].ports.outputs,
})));
const folds = { id: 'folds:wasm-hpo', sample_ids: ['s1', 's2'], sample_groups: {}, folds: [
  { fold_id: 'fold:0', train_sample_ids: ['s2'], validation_sample_ids: ['s1'], metadata: {} },
  { fold_id: 'fold:1', train_sample_ids: ['s1'], validation_sample_ids: ['s2'], metadata: {} },
] };
const campaign = { id: 'campaign:wasm-hpo', root_seed: 7, split_invocation: {
  id: 'split:outer', controller_id: null,
  leakage_policy: { split_unit: 'sample', forbid_origin_cross_fold: true,
    allow_observation_split_with_shared_target: false, require_group_ids: false, unsafe_flags: [] },
  params: {}, fold_set: folds,
} };
const plan = dag.build_execution_plan_json('plan:wasm-hpo', JSON.stringify(graph), JSON.stringify(campaign), JSON.stringify([manifest]));
const manifests = JSON.stringify([manifest]);
const envelope = readFileSync(new URL('../crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json', import.meta.url), 'utf8');
const digest = value => createHash('sha256').update(value).digest('hex');
const controller = (owner, task) => ridgeNodeResult(owner, task, folds, 'js-hpo-ingress');
const baseRequest = { target_node: 'model:m', trial_budget: 3, metric: 'rmse', direction: 'minimize', optimizer_descriptor: { owner: 'wasm-ingress-test' } };

async function exercise(options = {}) {
  const activeDag = options.dagModule ?? dag;
  const request = { ...baseRequest, trial_budget: options.budget ?? 3, progressive_pruning: !!options.pruning };
  const requestJson = options.requestRaw ?? JSON.stringify(request);
  const envelopeJson = options.envelopeRaw ?? envelope;
  const events = [], wires = [];
  let lastCommitted = null, lastPrepared = null, invocations = 0;
  const optimizer = (operation, payloadJson) => {
    const payload = JSON.parse(payloadJson);
    const event = { operation, trial_index: payload.trial_index ?? null, payload };
    if (operation === 'prepare_terminal' || operation === 'checkpoint') {
      event.checkpointRaw = strictJson(payloadJson, 'checkpoint');
      if (operation === 'prepare_terminal') lastPrepared = event.checkpointRaw;
      else lastCommitted = event.checkpointRaw;
    }
    events.push(event);
    const custom = options.reply?.(operation, payload, events);
    let response = custom;
    if (response === undefined && operation === 'ask') response = JSON.stringify({ params: { offset: payload.trial_index + 1 } });
    if (response === undefined && operation === 'report_intermediate') response = JSON.stringify({ prune: !!options.pruneTrial && payload.trial_index === 1 && payload.step === 0 });
    if (response === undefined && operation === 'checkpoint') response = '{"continue":true}';
    if (response === undefined) response = '{"ok":true}';
    event.reply = response;
    return response;
  };
  const invoke = (owner, task) => {
    invocations++;
    if ('controllerThrow' in options) throw options.controllerThrow;
    if (options.failController) throw new Error('controlled host failure');
    return controller(owner, task);
  };
  const dispatch = taskJson => {
    const packet = JSON.parse(taskJson);
    const kind = request.progressive_pruning ? packet.kind : 'complete';
    const taskRaw = request.progressive_pruning ? strictJson(taskJson, 'task') : taskJson;
    const task = JSON.parse(taskRaw);
    options.onTask?.(taskRaw, kind, task.trial_index);
    if (task.trial_index === 1 && kind === 'complete' && 'dispatchThrow' in options) throw options.dispatchThrow;
    if (task.trial_index === 1 && kind === 'complete' && 'rejectValue' in options) return Promise.reject(options.rejectValue);
    if (options.rejectWorker && task.trial_index === 1 && kind === 'complete') return Promise.reject(new Error('controlled worker failure'));
    const workerTask = task.trial_index === 1 && options.mutateTask ? options.mutateTask(taskRaw) : taskRaw;
    const workerRequest = task.trial_index === 1 && options.workerRequestRaw ? options.workerRequestRaw : requestJson;
    const workerEnvelope = task.trial_index === 1 && options.workerEnvelopeRaw ? options.workerEnvelopeRaw : envelopeJson;
    let raw = kind === 'fold'
      ? activeDag.host_hpo_evaluate_worker_fold_json(workerTask, packet.fold_index, manifests, workerEnvelope, workerRequest, invoke)
      : activeDag.host_hpo_evaluate_worker_task_json(workerTask, manifests, workerEnvelope, workerRequest, invoke);
    const original = raw;
    if (options.mutateWorker && task.trial_index === 1 && kind === 'complete') {
      raw = options.mutateWorker(raw);
      const originalEvidence = strictJson(original, 'evidence');
      const params = strictJson(originalEvidence, 'params');
      if (!options.positionalFailed) assert.ok(raw.includes(params), 'mutation changed owner worker params');
      if (!options.positionalFailed && originalEvidence.includes('"generated_view_manifest"')) {
        assert.ok(raw.includes(strictJson(originalEvidence, 'generated_view_manifest')), 'mutation changed the owner generated view manifest');
      } else {
        assert.ok(!raw.includes('"generated_view_manifest"'), 'mutation introduced a generated view manifest');
      }
    }
    if (options.mutateFold && task.trial_index === 1 && kind === 'fold' && packet.fold_index === 0) raw = options.mutateFold(raw);
    wires.push({ trial_index: task.trial_index, kind, original, returned: raw });
    if (options.b1) {
      const targetKind = options.b1.route === 'fold' ? 'fold' : 'complete';
      const target = kind === targetKind && (kind !== 'fold' || packet.fold_index === 0);
      if (target && task.trial_index === 0 && options.b1.mode !== 'valid') {
        return new Promise(resolve => setTimeout(() => resolve(raw), 500));
      }
      if (target && task.trial_index === 1) {
        if (options.b1.mode === 'immediate') return Promise.reject(new Error('controlled fast rejection'));
        const promise = Promise.resolve(raw);
        if (options.b1.mode === 'constructor-getter') Object.defineProperty(promise, 'constructor', { get() { throw new Error('hostile constructor'); } });
        if (options.b1.mode === 'then-getter') Object.defineProperty(promise, 'then', { get() { throw new Error('hostile then getter'); } });
        if (options.b1.mode === 'then-method') promise.then = () => { throw new Error('hostile then method'); };
        if (options.b1.mode === 'then-noncallable') promise.then = 5;
        return promise;
      }
    }
    return Promise.resolve(raw);
  };
  let outcome = null, error = null;
  try {
    const json = options.parallel
      ? await activeDag.host_hpo_search_parallel_json(plan, manifests, envelopeJson, requestJson, options.checkpoint, 3, dispatch, optimizer)
      : activeDag.host_hpo_search_json(plan, manifests, envelopeJson, requestJson, options.checkpoint, invoke, optimizer);
    outcome = JSON.parse(json);
  } catch (err) { error = String(err); }
  return { outcome, error, events, wires, lastCommitted, lastPrepared, invocations };
}

// Each B1 case is isolated in a default Node child: no process-level rejection
// handlers may mask an unobserved rejection or an unsettled search.
if (process.env.DAG_ML_B1_CHILD) {
  const config = JSON.parse(process.env.DAG_ML_B1_CHILD);
  const result = await exercise({ parallel: true, pruning: config.route !== 'plain', b1: config });
  assert.equal(result.error, null, result.error);
  const expected = config.mode === 'valid' ? ['complete', 'complete', 'complete'] : ['complete', 'failed', 'complete'];
  assert.deepEqual(result.outcome.checkpoint.trials.map(trial => trial.state), expected);
  assert.equal(result.outcome.status, config.mode === 'valid' ? 'completed' : 'failed');
  const terminals = result.events.filter(event => event.operation === 'tell' || event.operation === 'fail')
    .map(event => `${event.operation}:${event.trial_index}`);
  assert.deepEqual(terminals, config.mode === 'valid' ? ['tell:0', 'tell:1', 'tell:2'] : ['tell:0', 'fail:1', 'tell:2']);
  assert.equal(result.events.filter(event => event.operation === 'fail').length, config.mode === 'valid' ? 0 : 1);
  assert.deepEqual(JSON.parse(result.lastCommitted).trials.map(trial => trial.state), expected);
  const targetKind = config.route === 'fold' ? 'fold' : 'complete';
  assert.deepEqual(result.wires.filter(wire => wire.kind === targetKind).slice(0, 3).map(wire => wire.trial_index), [0, 1, 2]);
  if (config.mode !== 'valid') {
    const failed = result.outcome.checkpoint.trials[1];
    assert.match(failed.error, /worker promise rejected:|worker returned a non-string result/);
    const callback = result.events.find(event => event.operation === 'fail');
    assert.equal(callback.payload.error, failed.error);
  }
  await new Promise(resolve => setTimeout(resolve, 20));
  console.log(JSON.stringify({ route: config.route, mode: config.mode, status: result.outcome.status,
    states: expected, terminals, committed: true, fail_count: config.mode === 'valid' ? 0 : 1 }));
  process.exit(0);
}

if (process.env.HPO_INSPECT === '1') {
  for (const options of [{}, { parallel: true }, { parallel: true, pruning: true }, { failController: true }]) {
    const result = await exercise(options);
    console.log(JSON.stringify({ options, outcome: result.outcome, error: result.error,
      events: result.events.map(e => ({ operation: e.operation, payload: e.payload })),
      wire: result.wires[0]?.original }));
  }
}

// Mutate only raw owner-returned worker text. In particular, neither params
// nor generated_view_manifest is parsed and reserialized or changed.
function evidenceEdit(raw, edit) {
  const evidence = strictJson(raw, 'evidence');
  const params = strictJson(evidence, 'params');
  const hasManifest = evidence.includes('"generated_view_manifest"');
  const manifestToken = hasManifest ? strictJson(evidence, 'generated_view_manifest') : null;
  const replacement = edit(evidence);
  assert.ok(replacement.includes(params), 'worker params changed');
  if (hasManifest) assert.ok(replacement.includes(manifestToken), 'generated view manifest changed');
  const needle = `"evidence":${evidence}`;
  assert.equal(raw.split(needle).length, 2, 'expected one owner evidence member');
  return raw.replace(needle, `"evidence":${replacement}`);
}
const addEvidence = members => raw => evidenceEdit(raw, e => e.replace(/^\{/, `{${members},`));
const sequenceEvidence = raw => evidenceEdit(raw, e => {
  assert.ok(!e.includes('"generated_view_manifest"'), 'sequence control requires absent default manifest');
  return `[${['trial_index', 'params', 'score', 'variant_id', 'scores'].map(k => strictJson(e, k)).join(',')}]`;
});
const duplicateOuter = key => raw => raw.replace(/^\{/, `{"${key}":${strictJson(raw, key)},`);
const outerUnknown = raw => raw.replace(/^\{/, '{"outer_probe":1,');
const alreadyBadFloat = addEvidence('"float_probe":1e400');
const mutations = [
  ['open-oob', addEvidence('"ignored_probe":18446744073709551616'), 'new_refusal'],
  ['open-duplicate', addEvidence('"ignored_probe":1,"ignored_probe":2'), 'new_refusal'],
  ['open-nfc', addEvidence('"\\u00e9":1,"e\\u0301":2'), 'new_refusal'],
  ['struct-sequence', sequenceEvidence, 'new_refusal'],
  ['outer-state-duplicate', duplicateOuter('state'), 'already_refused_stage_only'],
  ['outer-evidence-duplicate', duplicateOuter('evidence'), 'already_refused_stage_only'],
  ['outer-digest-duplicate', duplicateOuter('data_fingerprint'), 'already_refused_stage_only'],
  ['outer-unknown', outerUnknown, 'already_refused_stage_only'],
  ['float-overflow', alreadyBadFloat, 'already_refused_stage_only'],
  ['legal-integers', addEvidence('"i64_min":-9223372036854775808,"u64_max":18446744073709551615'), 'closed_evidence'],
  ['legal-floats', addEvidence('"max_float":1.7976931348623157e308,"small_float":-2.5e-300,"large_float":1e300'), 'closed_evidence'],
  ['legal-negative-zero', addEvidence('"neg_zero":-0,"one_exp":1.0e0'), 'closed_evidence'],
];
const cases = [];
function add(label, site, expected, classification, run, verify) { cases.push({ label, site, expected, classification, run, verify }); }
const completed = result => {
  assert.equal(result.error, null);
  assert.equal(result.outcome.status, 'completed');
  assert.deepEqual(result.outcome.checkpoint.trials.map(t => t.state), ['complete', 'complete', 'complete']);
};
const labeledError = (result, label) => {
  assert.ok(result.error, 'expected search rejection');
  assert.match(result.error, new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(result.error, /strict TCV1 JSON document|external contract boundary/);
};
const indexOf = (events, operation, index) => events.findIndex(e => e.operation === operation && (index === undefined || e.trial_index === index));
function recoverPrepared(result) {
  assert.ok(result.lastCommitted && result.lastPrepared, 'missing committed/prepared checkpoints');
  const committed = JSON.parse(result.lastCommitted), prepared = JSON.parse(result.lastPrepared);
  assert.equal(prepared.trials.length, committed.trials.length + 1);
  const recovered = JSON.parse(dag.recover_host_hpo_checkpoint_json(result.lastCommitted, result.lastPrepared, '[]'));
  assert.equal(recovered.fingerprint, prepared.fingerprint);
  assert.equal(recovered.trials.length, prepared.trials.length);
}
function assertPreparedUnpublished(result, operation, trial, isSync = false) {
  const prepare = result.events.findLastIndex(e => e.operation === 'prepare_terminal');
  const terminal = indexOf(result.events, operation, trial);
  assert.ok(prepare >= 0 && terminal > prepare, 'terminal callback must follow prepare');
  assert.equal(result.events.slice(terminal + 1).filter(e => e.operation === 'checkpoint').length, 0, 'failed terminal must not publish');
  const prepared = JSON.parse(result.lastPrepared);
  const committedLength = result.lastCommitted ? JSON.parse(result.lastCommitted).trials.length : 0;
  assert.equal(prepared.trials.length, committedLength + 1);
  if (isSync) recoverPrepared(result);
}
function assertWorkerFailure(result, label, options) {
  assert.equal(result.error, null, 'worker result must resolve, not reject');
  assert.equal(result.outcome.status, 'failed');
  assert.deepEqual(result.outcome.checkpoint.trials.map(t => t.state), ['complete', 'failed', 'complete']);
  const failed = result.outcome.checkpoint.trials[1];
  const failure = result.events.find(e => e.operation === 'fail' && e.trial_index === 1);
  assert.ok(failure, 'missing fail callback');
  for (const text of [failed.error, failure.payload.error]) {
    if (options.typedError) {
      // Lexically valid unknown outer fields fail typed deserialization. C2
      // requires Serialization here; the existing decoder does not add a label.
      assert.match(text, /^serialization error: unknown field `/);
      assert.ok(!text.includes(label), 'typed Serialization must retain its existing category');
    } else {
      assert.ok(text.includes(label), `missing ${label} in ${text}`);
      assert.match(text, /strict TCV1 JSON document|external contract boundary/);
    }
  }
  const transitions = result.events.filter(e => ['prepare_terminal', 'tell', 'fail'].includes(e.operation))
    .map(e => e.operation === 'prepare_terminal' ? `prepare:${e.payload.checkpoint.trials.length - 1}` : `${e.operation}:${e.trial_index}`);
  assert.deepEqual(transitions, ['prepare:0', 'tell:0', 'prepare:1', 'fail:1', 'prepare:2', 'tell:2']);
  const checkpoints = result.events.filter(e => e.operation === 'checkpoint').map(e => e.payload.checkpoint.trials.map(t => t.state));
  assert.deepEqual(checkpoints, [['complete'], ['complete', 'failed'], ['complete', 'failed', 'complete']]);
  assert.equal(result.wires.filter(w => w.kind === 'complete' && w.trial_index === 1 && w.original !== w.returned).length, 1);
  if (options.pruning) assert.ok(result.wires.filter(w => w.kind === 'fold').every(w => w.original === w.returned), 'pruning completion control changed fold bytes');
}

add('sync-valid', 'optimizer', 'completed', 'legal', () => exercise(), completed);
add('parallel-valid', 'worker-result', 'completed', 'legal', () => exercise({ parallel: true }), completed);
add('parallel-pruning-valid', 'worker-complete-result', 'completed', 'legal', () => exercise({ parallel: true, pruning: true }), completed);
const dupOk = '{"ok":false,"ok":true}';
add('sync-ask-duplicate-offset', 'optimizer:ask', 'strict rejection before evaluation', 'semantic_duplicate',
  () => exercise({ reply: op => op === 'ask' ? '{"params":{"offset":1,"offset":2}}' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.equal(r.invocations, 0); });
add('sync-ask-nfc-extras', 'optimizer:ask', 'strict rejection before evaluation', 'nfc_collision',
  () => exercise({ reply: op => op === 'ask' ? '{"params":{"offset":1},"\\u00e9":1,"e\\u0301":2}' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.equal(r.invocations, 0); });
add('sync-ask-invalid-json', 'optimizer:ask', 'runtime validation lexical refusal', 'invalid_json',
  () => exercise({ reply: op => op === 'ask' ? '{' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.match(r.error, /runtime_validation/); });
add('sync-ask-legal-extra', 'optimizer:ask', 'completed with ignored legal tokens', 'legal',
  () => exercise({ reply: (op, p) => op === 'ask' ? `{"params":{"offset":${p.trial_index + 1}},"probe":{"min":-9223372036854775808,"max":18446744073709551615,"x":1e300,"y":-2.5e-300,"z":-0,"one":1.0e0}}` : undefined }), completed);
const minString = JSON.stringify(-(2 ** 63));
assert.equal(minString, '-9223372036854776000');
add('sync-js-stringify-i64-min', 'optimizer:ask', 'strict rejection of integer-form token', 'js_number_limit',
  () => exercise({ reply: op => op === 'ask' ? `{"params":{"offset":1},"probe":${minString}}` : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.equal(r.invocations, 0); });
add('sync-js-float-window-positive', 'optimizer:ask', 'strict rejection of finite float serialized as integer', 'js_float_window',
  () => exercise({ reply: op => op === 'ask' ? JSON.stringify({ params: { offset: 2e19 } }) : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.equal(r.invocations, 0); });
add('sync-js-float-window-negative', 'optimizer:ask', 'strict rejection of finite float serialized as integer', 'js_float_window',
  () => exercise({ reply: op => op === 'ask' ? JSON.stringify({ params: { offset: 1, float_axis: -1e19 } }) : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.equal(r.invocations, 0); });
for (const operation of ['tell', 'pruned', 'fail']) {
  const options = { pruning: operation === 'pruned', pruneTrial: operation === 'pruned', failController: operation === 'fail',
    reply: op => op === operation ? dupOk : undefined };
  add(`sync-${operation}-duplicate-ok`, `optimizer:${operation}`, 'rejection after prepared terminal', 'semantic_duplicate',
    () => exercise(options), r => {
      labeledError(r, `JS HPO optimizer \`${operation}\` reply`);
      assertPreparedUnpublished(r, operation, operation === 'pruned' ? 1 : 0, operation !== 'fail');
      const prepared = JSON.parse(r.lastPrepared);
      assert.equal(prepared.trials.at(-1).state, operation === 'fail' ? 'failed' : operation === 'pruned' ? 'pruned' : 'complete');
    });
}
add('sync-prepare-duplicate-ok', 'optimizer:prepare_terminal', 'rejection at prepare', 'semantic_duplicate',
  () => exercise({ reply: op => op === 'prepare_terminal' ? dupOk : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `prepare_terminal` reply'); assert.ok(r.lastPrepared); const index = indexOf(r.events, 'prepare_terminal'); assert.equal(indexOf(r.events, 'tell'), -1); assert.ok(r.events.slice(index + 1).every(e => !['checkpoint', 'fail', 'pruned'].includes(e.operation))); });
add('sync-checkpoint-duplicate-continue', 'optimizer:checkpoint', 'rejection at checkpoint', 'semantic_duplicate',
  () => exercise({ reply: op => op === 'checkpoint' ? '{"continue":false,"continue":true}' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `checkpoint` reply'); assert.equal(r.invocations, 0); });
add('sync-checkpoint-duplicate-after-terminal', 'optimizer:checkpoint', 'checkpoint refusal after terminal transition', 'semantic_duplicate',
  () => exercise({ reply: (op, payload) => op === 'checkpoint' && payload.checkpoint.trials.length === 1 ? '{"continue":false,"continue":true}' : undefined }),
  r => {
    labeledError(r, 'JS HPO optimizer `checkpoint` reply');
    const prepared = indexOf(r.events, 'prepare_terminal'), told = indexOf(r.events, 'tell', 0);
    const refused = r.events.findIndex(e => e.operation === 'checkpoint' && e.payload.checkpoint.trials.length === 1);
    assert.ok(prepared >= 0 && told > prepared && refused > told);
    assert.equal(r.events.slice(refused + 1).filter(e => e.operation === 'ask').length, 0);
  });
add('sync-ask-missing-params', 'optimizer:ask', 'existing field-handler refusal', 'existing_field_refusal',
  () => exercise({ reply: op => op === 'ask' ? '{}' : undefined }),
  r => { assert.ok(r.error?.includes('JS HPO ask reply needs `params`')); assert.ok(!r.error.includes('strict TCV1')); });
add('sync-checkpoint-nonboolean', 'optimizer:checkpoint', 'existing field-handler refusal', 'existing_field_refusal',
  () => exercise({ reply: op => op === 'checkpoint' ? '{"continue":"yes"}' : undefined }),
  r => { assert.ok(r.error?.includes('JS HPO checkpoint reply needs boolean `continue`')); assert.ok(!r.error.includes('strict TCV1')); });
add('sync-tell-false-ok', 'optimizer:tell', 'existing field-handler refusal after prepare', 'existing_field_refusal',
  () => exercise({ reply: op => op === 'tell' ? '{"ok":false}' : undefined }),
  r => { assert.ok(r.error?.includes('must reply')); assert.ok(!r.error.includes('strict TCV1')); assertPreparedUnpublished(r, 'tell', 0, true); });
add('sync-intermediate-nonboolean', 'optimizer:report_intermediate', 'existing field-handler failure and checkpoint', 'existing_field_refusal',
  () => exercise({ pruning: true, reply: op => op === 'report_intermediate' ? '{"prune":"no"}' : undefined }),
  r => {
    assert.ok(r.error?.includes('needs boolean `prune`')); assert.ok(!r.error.includes('strict TCV1'));
    const i = indexOf(r.events, 'report_intermediate', 0), p = indexOf(r.events, 'prepare_terminal'), f = indexOf(r.events, 'fail', 0);
    const c = r.events.findIndex((e, n) => n > f && e.operation === 'checkpoint');
    assert.ok(i >= 0 && p > i && f > p && c > f); assert.equal(r.events[p].payload.checkpoint.trials[0].state, 'failed');
  });
add('sync-progressive-intermediate-duplicate-prune', 'optimizer:report_intermediate', 'sealed failed then fail/checkpoint then Err', 'semantic_duplicate',
  () => exercise({ pruning: true, reply: op => op === 'report_intermediate' ? '{"prune":true,"prune":false}' : undefined }),
  r => {
    labeledError(r, 'JS HPO optimizer `report_intermediate` reply');
    const i = indexOf(r.events, 'report_intermediate', 0), p = indexOf(r.events, 'prepare_terminal'), f = indexOf(r.events, 'fail', 0);
    const c = r.events.findIndex((e, n) => n > f && e.operation === 'checkpoint');
    assert.ok(i >= 0 && p > i && f > p && c > f);
    assert.equal(r.events[p].payload.status, 'failed');
    assert.equal(r.events[p].payload.checkpoint.trials[0].state, 'failed');
    assert.ok(r.events[p].payload.checkpoint.fingerprint);
    assert.ok(r.events[f].payload.error.includes('JS HPO optimizer `report_intermediate` reply'));
    assert.equal(r.events[c].payload.checkpoint.trials[0].state, 'failed');
    assert.equal(r.events[c].payload.checkpoint.fingerprint, r.events[p].payload.checkpoint.fingerprint);
    const initial = r.events.find(e => e.operation === 'checkpoint' && e.payload.checkpoint.trials.length === 0);
    assert.ok(initial?.checkpointRaw, 'missing initial committed checkpoint');
    const recovered = JSON.parse(dag.recover_host_hpo_checkpoint_json(initial.checkpointRaw, r.events[p].checkpointRaw, '[]'));
    assert.equal(recovered.fingerprint, r.events[p].payload.checkpoint.fingerprint);
    assert.equal(recovered.trials[0].state, 'failed');
  });
add('parallel-progressive-intermediate-duplicate-prune', 'optimizer:report_intermediate', 'rejected without failed terminal', 'semantic_duplicate',
  () => exercise({ parallel: true, pruning: true, reply: op => op === 'report_intermediate' ? '{"prune":true,"prune":false}' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `report_intermediate` reply'); assert.equal(indexOf(r.events, 'prepare_terminal'), -1); assert.equal(indexOf(r.events, 'fail'), -1); assert.ok(r.events.every(e => e.payload.status !== 'failed' && !e.payload.checkpoint?.trials?.some(t => t.state === 'failed'))); });
add('parallel-ask-duplicate', 'optimizer:ask', 'strict rejection before evaluation', 'semantic_duplicate',
  () => exercise({ parallel: true, reply: op => op === 'ask' ? '{"params":{"offset":1,"offset":2}}' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `ask` reply'); assert.equal(r.invocations, 0); });
add('parallel-prepare-duplicate', 'optimizer:prepare_terminal', 'rejection at prepare', 'semantic_duplicate',
  () => exercise({ parallel: true, reply: op => op === 'prepare_terminal' ? dupOk : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `prepare_terminal` reply'); const p = indexOf(r.events, 'prepare_terminal'); assert.ok(p >= 0); assert.equal(indexOf(r.events, 'tell'), -1); assert.ok(r.events.slice(p + 1).every(e => !['checkpoint', 'fail', 'pruned'].includes(e.operation))); });
add('parallel-checkpoint-duplicate', 'optimizer:checkpoint', 'rejection at checkpoint', 'semantic_duplicate',
  () => exercise({ parallel: true, reply: op => op === 'checkpoint' ? '{"continue":false,"continue":true}' : undefined }),
  r => { labeledError(r, 'JS HPO optimizer `checkpoint` reply'); const c = indexOf(r.events, 'checkpoint'); assert.ok(c >= 0); assert.equal(r.events.slice(c + 1).length, 0); });
add('sync-fail-recover-resume', 'optimizer:fail', 'recover and resume failed trial', 'semantic_duplicate',
  async () => {
    const failed = await exercise({ failController: true, reply: op => op === 'fail' ? dupOk : undefined });
    recoverPrepared(failed);
    const checkpoint = dag.recover_host_hpo_checkpoint_json(failed.lastCommitted, failed.lastPrepared, '[]');
    const resumed = await exercise({ checkpoint });
    return { ...failed, resumed };
  }, r => {
    labeledError(r, 'JS HPO optimizer `fail` reply');
    assert.equal(r.resumed.error, null); assert.equal(r.resumed.outcome.status, 'completed');
    assert.deepEqual(r.resumed.outcome.checkpoint.trials.map(t => t.state), ['failed', 'complete', 'complete']);
    assert.equal(r.resumed.outcome.checkpoint.trials[0].trial_index, 0);
    assert.deepEqual(r.resumed.outcome.checkpoint.trials.map(t => t.trial_index ?? t.evidence.trial_index), [0, 1, 2]);
  });
for (const operation of ['tell', 'pruned', 'fail']) {
  const options = { parallel: true, pruning: operation === 'pruned', pruneTrial: operation === 'pruned', rejectWorker: operation === 'fail',
    reply: (op, payload) => op === operation && payload.trial_index === 1 ? dupOk : undefined };
  add(`parallel-${operation}-duplicate-ok`, `optimizer:${operation}`, 'window terminal rejection after prepare', 'semantic_duplicate',
    () => exercise(options), r => { labeledError(r, `JS HPO optimizer \`${operation}\` reply`); assertPreparedUnpublished(r, operation, 1); });
}
for (const pruning of [false, true]) {
  const label = pruning ? 'worker-complete' : 'worker-result';
  for (const [mutation, mutateWorker, classification] of mutations) {
    const options = { parallel: true, pruning, mutateWorker, typedError: mutation === 'outer-unknown' || classification === 'closed_evidence' };
    add(`${label}-${mutation}`, label, classification === 'legal' ? 'completed' : 'failed checkpoint and fail callback', classification,
      () => exercise(options), r => classification === 'legal' ? completed(r) : assertWorkerFailure(r, pruning ? 'worker complete result' : 'worker result', options));
  }
}
for (const pruning of [false, true]) {
  for (const state of ['complete', 'failed']) {
    const mutateWorker = state === 'complete'
      ? raw => `["complete",${strictJson(raw, 'evidence')},${strictJson(raw, 'data_fingerprint')}]`
      : () => '["failed",1,"x"]';
    const options = { parallel: true, pruning, mutateWorker, positionalFailed: state === 'failed' };
    add(`${pruning ? 'worker-complete' : 'worker-result'}-positional-${state}`, 'worker-result', 'failed checkpoint', 'shape',
      () => exercise(options), r => assertWorkerFailure(r, pruning ? 'worker complete result' : 'worker result', options));
  }
}
add('pruning-fold-strict-existing', 'worker-fold', 'existing strict fold failure', 'already_refused_stage_only',
  () => exercise({ parallel: true, pruning: true, mutateFold: raw => raw.replace(/^\{/, '{"ignored_probe":18446744073709551616,') }),
  r => {
    assert.equal(r.error, null); assert.equal(r.outcome.status, 'failed');
    assert.equal(r.outcome.checkpoint.trials[1].state, 'failed');
    assert.ok(r.outcome.checkpoint.trials[1].error.includes('worker fold'));
    const failure = r.events.find(e => e.operation === 'fail' && e.trial_index === 1);
    assert.ok(failure?.payload.error.includes('worker fold'));
  });


// Insert after JSON.stringify: stringify escapes lone surrogates.
const utf16Samples = [
  ['raw-high', '"' + '\ud800' + '"', 'raw'], ['raw-low', '"' + '\udc00' + '"', 'raw'],
  ['escaped-high', '"\\ud800"', 'escaped'], ['escaped-low', '"\\udc00"', 'escaped'],
  ['emoji', '"😀"', 'valid'], ['normal', '"normal"', 'valid'], ['replacement', '"�"', 'valid'],
];
for (const [name, token, kind] of utf16Samples) {
  add(`utf16-optimizer-${name}`, 'optimizer:ask', kind, 'utf16',
    () => exercise({reply: (op, payload) => op === 'ask' && payload.trial_index === 0
      ? `{"params":{"offset":1},"probe":${token}}` : undefined}), r => {
      if (kind === 'valid') completed(r);
      else {
        assert.ok(r.error);
        if (kind === 'raw') assert.match(r.error, /JS HPO optimizer `ask` reply.*UTF-16/);
        else assert.doesNotMatch(r.error, /not valid UTF-16/);
        assert.equal(r.invocations, 0);
      }
    });
  for (const site of ['worker', 'complete', 'fold']) {
    const field = `"utf16_probe":${token}`;
    const mutate = site === 'fold' ? raw => raw.replace(/^\{/, `{${field},`) : addEvidence(field);
    const options = {parallel: true, pruning: site !== 'worker',
      ...(site === 'fold' ? {mutateFold: mutate} : {mutateWorker: mutate})};
    add(`utf16-${site}-${name}`, site, kind, 'utf16', () => exercise(options), r => {
      assert.equal(r.error, null);
      assert.equal(r.outcome.checkpoint.trials[1].state, 'failed');
      const error = r.outcome.checkpoint.trials[1].error;
      if (kind === 'raw') assert.match(error, /worker returned a result with invalid UTF-16/);
      else assert.doesNotMatch(error, /invalid UTF-16/);
      assert.ok(r.events.some(e => e.operation === 'fail' && e.trial_index === 1));
    });
  }
}
add('utf16-optimizer-fail-recover-resume', 'optimizer:fail', 'recover and resume', 'utf16',
  async () => {
    const failed = await exercise({failController: true, reply: op => op === 'fail'
      ? '{"ok":true,"probe":"' + '\ud800' + '"}' : undefined});
    recoverPrepared(failed);
    const checkpoint = dag.recover_host_hpo_checkpoint_json(failed.lastCommitted, failed.lastPrepared, '[]');
    return {...failed, resumed: await exercise({checkpoint})};
  }, r => {
    assert.match(r.error, /JS HPO optimizer `fail` reply.*UTF-16/);
    assert.equal(r.resumed.error, null);
    assert.deepEqual(r.resumed.outcome.checkpoint.trials.map(t => t.state), ['failed','complete','complete']);
  });


// These cases exercise the raw ingress boundaries with coordinator-owned tasks.
// Qualification supplies the accepted N1 package so successful task and fold
// results can also be compared byte for byte across the two real modules.
const baselinePath = process.env.DAG_ML_HPO_BASELINE_PKG;
if (baselinePath) {
  assert.ok(path.isAbsolute(baselinePath));
  assert.ok(statSync(baselinePath).isFile());
}
const baselineDag = baselinePath ? require(baselinePath) : null;
const requestFor = pruning => JSON.stringify({ ...baseRequest, trial_budget: 2, progressive_pruning: pruning });
async function ingressOwner(pruning = false, token = null) {
  let ownerTask = null;
  const reply = token === null ? undefined : (op, payload) => op === 'ask'
    ? `{"params":{"offset":${payload.trial_index + 1},"probe":${token}}}` : undefined;
  const run = await exercise({ parallel: true, pruning, budget: 2, reply,
    onTask: (raw, kind, index) => { if (index === 0 && !ownerTask && (!pruning || kind === 'fold')) ownerTask = raw; } });
  assert.equal(run.error, null); assert.equal(run.outcome.status, 'completed'); assert.ok(ownerTask);
  return { task: ownerTask, request: requestFor(pruning), envelope, checkpoint:run.lastCommitted };
}
function ingressMutate(family, raw, mutation) {
  if (mutation === 'positional') {
    if (family === 'task') return `[${['trial_index','phase_index','params','candidate_plan'].map(k => strictJson(raw, k)).join(',')}]`;
    if (family === 'request') {
      const x = JSON.parse(raw);
      return JSON.stringify([x.target_node,x.trial_budget,x.metric,x.direction,x.optimizer_descriptor,[],x.progressive_pruning,null,{},null]);
    }
    const x = JSON.parse(raw);
    return JSON.stringify([x.schema_version,x.schema_fingerprint,x.plan_fingerprint,x.relation_fingerprint ?? null,
      x.data_content_fingerprint ?? null,x.target_content_fingerprint ?? null,x.coordinator_relations ?? null,x.predict_cohort ?? null]);
  }
  if (family === 'task') {
    if (mutation === 'typed') return raw.replace('"trial_index":0','"trial_index":0.0');
    if (mutation === 'known-duplicate') return raw.replace('"trial_index":0','"trial_index":0,"trial_index":0');
    const fields = {duplicate:'"offset":1,"offset":1', escaped:'"offset":1,"o\\u0066fset":1',
      nfc:'"offset":1,"é":1,"e\\u0301":2', range:'"offset":1,"probe":18446744073709551616'};
    const value = raw.replace('"offset":1',fields[mutation]); assert.notEqual(value, raw); return value;
  }
  if (family === 'request') {
    if (mutation === 'typed') return raw.replace('"trial_budget":2','"trial_budget":2.0');
    if (mutation === 'known-duplicate') return raw.replace('"trial_budget":2','"trial_budget":2,"trial_budget":2');
    const fields = {duplicate:'"owner":"wasm-ingress-test","x":1,"x":1', escaped:'"owner":"wasm-ingress-test","x":1,"\\u0078":1',
      nfc:'"owner":"wasm-ingress-test","é":1,"e\\u0301":2', range:'"owner":"wasm-ingress-test","probe":18446744073709551616'};
    const value = raw.replace('"owner":"wasm-ingress-test"',fields[mutation]); assert.notEqual(value, raw); return value;
  }
  if (mutation === 'typed') return raw.replace('"schema_version": 1','"schema_version": 1.0');
  if (mutation === 'known-duplicate') return raw.replace('"schema_version": 1','"schema_version": 1,"schema_version": 1');
  const fields = {duplicate:'"x":1,"x":1', escaped:'"x":1,"\\u0078":1', nfc:'"é":1,"e\\u0301":2', range:'"probe":18446744073709551616'};
  const needle = '"metadata": {}', index = raw.lastIndexOf(needle); assert.ok(index >= 0);
  return raw.slice(0,index) + `"metadata": {${fields[mutation]}}` + raw.slice(index + needle.length);
}
const ingressSites = [
  ['task','worker'],['task','fold'],['request','worker'],['request','fold'],['request','search'],['request','parallel'],
  ['envelope','worker'],['envelope','fold'],['envelope','search'],['envelope','parallel'],
];
for (const [family, site] of ingressSites) {
  for (const mutation of ['positional','duplicate','escaped','nfc','range','typed','known-duplicate']) {
    add(`raw-ingress-${family}-${site}-${mutation}`, site, 'labelled refusal before local host callback', 'hpo-argument', async () => {
      const owner = await ingressOwner(false), source = owner[family], raw = ingressMutate(family,source,mutation);
      assert.notEqual(raw,source);
      if (site === 'search' || site === 'parallel') {
        const result = await exercise({ budget:2, parallel:site === 'parallel',
          ...(family === 'request' ? {requestRaw:raw} : {envelopeRaw:raw}) });
        return { ...result, ingressError:result.error };
      }
      let calls = 0, ingressError = null;
      const invoke = (id, payload) => { calls++; return controller(id,payload); };
      const task = family === 'task' ? raw : owner.task, req = family === 'request' ? raw : owner.request;
      const env = family === 'envelope' ? raw : owner.envelope;
      try {
        if (site === 'worker') dag.host_hpo_evaluate_worker_task_json(task,manifests,env,req,invoke);
        else dag.host_hpo_evaluate_worker_fold_json(task,0,manifests,env,req,invoke);
      } catch (error) { ingressError = String(error); }
      return { ingressError, invocations:calls, events:[] };
    }, result => {
      const label = family === 'task' ? 'host HPO worker task' : `host HPO ${family}`;
      assert.ok(result.ingressError?.includes(label), result.ingressError);
      assert.match(result.ingressError, /runtime_validation/);
      if (mutation === 'positional') assert.match(result.ingressError, /must use a JSON object/);
      else if (mutation === 'typed') { assert.match(result.ingressError, /invalid type/); assert.doesNotMatch(result.ingressError, /at line \d+ column \d+/); }
      else assert.match(result.ingressError, /not a strict TCV1 JSON document/);
      assert.equal(result.invocations,0); assert.equal(result.events.length,0);
    });
  }
}
for (const token of ['18446744073709551616','18446744073709551617','18446744073709552000','100000000000000000000','100000000000000000001','-9223372036854776000','1e400']) {
  add(`raw-ingress-refused-number-${token}`, 'request', 'integer range or nonfinite', 'hpo-argument', () => {
    if (token === '18446744073709552000') assert.equal(JSON.stringify(1.8446744073709552e19), token);
    if (token === '-9223372036854776000') assert.equal(JSON.stringify(-(2 ** 63)), token);
    const raw = requestFor(false).replace('"owner":"wasm-ingress-test"', `"owner":"wasm-ingress-test","probe":${token}`);
    return exercise({budget:2,requestRaw:raw});
  }, result => { assert.match(result.error, /host HPO request.*not a strict TCV1 JSON document/); assert.equal(result.invocations,0); assert.equal(result.events.length,0); });
}
for (const token of ['100000000000000000000.0','1e+20','1E20','-0.0','-0','5e-324','1.7976931348623157e308','18446744073709551615','-9223372036854775808','1e+21','-1e+21']) {
  add(`raw-ingress-accepted-number-${token}`, 'request', 'valid numeric spelling', 'hpo-argument', () => {
    const raw = requestFor(false).replace('"owner":"wasm-ingress-test"', `"owner":"wasm-ingress-test","probe":${token}`);
    return exercise({budget:2,requestRaw:raw});
  }, result => { assert.equal(result.error,null); assert.equal(result.outcome.status,'completed'); });
}
add('raw-ingress-adjacent-u64-identities', 'request', 'distinct owner fingerprints', 'hpo-argument', async () => {
  const results=[];
  for (const token of ['18446744073709551614','18446744073709551615']) {
    const raw=requestFor(false).replace('"owner":"wasm-ingress-test"', `"owner":"wasm-ingress-test","probe":${token}`);
    results.push(await exercise({budget:2,requestRaw:raw}));
  }
  return {results};
}, ({results}) => {
  assert.ok(results.every(r => !r.error));
  for (const key of ['request_fingerprint']) assert.notEqual(results[0].outcome[key],results[1].outcome[key]);
  assert.notEqual(results[0].outcome.checkpoint.binding.objective_fingerprint,results[1].outcome.checkpoint.binding.objective_fingerprint);
  assert.notEqual(results[0].outcome.checkpoint.fingerprint,results[1].outcome.checkpoint.fingerprint);
});
for (const pruning of [false,true]) {
  for (const token of [null,'-0.0','5e-324','18446744073709551615','1e+300','-9223372036854775808']) {
    for (const site of ['worker','fold']) {
      add(`raw-owner-${pruning ? 'pruning' : 'plain'}-${token ?? 'ordinary'}-${site}`, site, 'owner task accepted unchanged', 'hpo-argument', async () => {
        const owner=await ingressOwner(pruning,token);let calls=0;const invoke=(id,t)=>{calls++;return controller(id,t)};
        const call = module => site === 'worker'
          ? module.host_hpo_evaluate_worker_task_json(owner.task,manifests,owner.envelope,owner.request,invoke)
          : module.host_hpo_evaluate_worker_fold_json(owner.task,0,manifests,owner.envelope,owner.request,invoke);
        const result=call(dag), baseline=baselineDag ? call(baselineDag) : null;
        return {result,baseline,calls};
      }, result => { assert.ok(result.calls>0); if (baselineDag) assert.equal(result.result,result.baseline); });
    }
  }
}
add('raw-ingress-v2-additive-envelope-owner-behavior', 'envelope', 'preserve current v2 additive behavior', 'hpo-argument', async () => {
  const cohort = {role:'external_test', relations:{records:[{observation_id:'obs:holdout',sample_id:'sample:holdout',target_id:'target:holdout',group_id:'group:holdout',origin_sample_id:null,source_id:'nir',is_augmented:false}]},
    target_names:['y'],data_content_fingerprint:'c'.repeat(64),target_content_fingerprint:'d'.repeat(64)};
  const v2=dag.attach_predict_cohort_to_envelope_json(envelope,JSON.stringify(cohort));
  const parsed=JSON.parse(v2); assert.equal(parsed.schema_version,2);
  const additive=JSON.stringify({...parsed,future_field:{nested:'unchanged'}});
  return {base:await exercise({budget:2,envelopeRaw:v2}),extra:await exercise({budget:2,envelopeRaw:additive})};
}, ({base,extra}) => {
  assert.equal(base.error,null); assert.equal(extra.error,null);
  assert.equal(base.outcome.status,'completed'); assert.equal(extra.outcome.status,'completed');
  assert.equal(base.outcome.checkpoint.binding.data_fingerprint,extra.outcome.checkpoint.binding.data_fingerprint);
});
add('raw-ingress-valid-additive-envelope', 'envelope', 'valid additive metadata', 'hpo-argument', () => {
  const needle='"metadata": {}', index=envelope.lastIndexOf(needle); assert.ok(index>=0);
  const raw=envelope.slice(0,index)+'"metadata": {"singleDecomposed":"e\\u0301","emoji":"😀","nested":{"extra":1}}'+envelope.slice(index+needle.length);
  return exercise({budget:2,envelopeRaw:raw});
}, r => { assert.equal(r.error,null); assert.equal(r.outcome.status,'completed'); });
for (const site of ['search','parallel']) {
  for (const family of ['request','envelope']) {
    add(`raw-ingress-resume-${site}-${family}`, site, 'refuse before resume callbacks', 'hpo-argument', async () => {
      const owner=await ingressOwner(false), raw=ingressMutate(family,owner[family],'duplicate');
      return exercise({budget:2,parallel:site==='parallel',checkpoint:owner.checkpoint,
        ...(family==='request' ? {requestRaw:raw} : {envelopeRaw:raw})});
    }, r => { assert.match(r.error,new RegExp(`host HPO ${family}.*not a strict TCV1 JSON document`)); assert.equal(r.invocations,0); assert.equal(r.events.length,0); });
  }
}
add('raw-ingress-valid-resume-asymmetry', 'resume', 'preserve owner serial and parallel lifecycle', 'hpo-argument', async () => {
  const owner=await ingressOwner(false);
  return {serial:await exercise({budget:2,checkpoint:owner.checkpoint}),parallel:await exercise({budget:2,parallel:true,checkpoint:owner.checkpoint})};
}, ({serial,parallel}) => {
  assert.equal(serial.error,null); assert.equal(parallel.error,null);
  assert.equal(serial.outcome.status,'completed'); assert.equal(parallel.outcome.status,'completed');
  assert.equal(serial.invocations,0); assert.equal(parallel.invocations,0);
  assert.equal(serial.events.filter(e=>e.operation==='checkpoint').length,1);
  assert.equal(parallel.events.length,0);
});
add('raw-ingress-typed-worker-failure-diagnostic-identity', 'worker', 'typed category and failed checkpoint identity', 'hpo-argument', async () => {
  const mutateTask=raw=>raw.replace('"trial_index":1','"trial_index":1.0');
  const current=await exercise({parallel:true,mutateTask});
  const previous=baselineDag ? await exercise({parallel:true,mutateTask,dagModule:baselineDag}) : null;
  return {current,previous};
}, ({current,previous}) => {
  assert.equal(current.error,null); assert.equal(current.outcome.checkpoint.trials[1].state,'failed');
  const error=current.outcome.checkpoint.trials[1].error;
  assert.match(error,/host HPO worker task/); assert.match(error,/invalid type/); assert.doesNotMatch(error,/at line \d+ column \d+/);
  assert.equal(current.events.filter(e=>e.operation==='fail' && e.trial_index===1).length,1);
  if (previous) {
    assert.equal(previous.outcome.checkpoint.trials[1].state,'failed');
    assert.match(previous.outcome.checkpoint.trials[1].error,/serialization error/);
    assert.match(previous.outcome.checkpoint.trials[1].error,/at line \d+ column \d+/);
    assert.notEqual(current.outcome.checkpoint.fingerprint,previous.outcome.checkpoint.fingerprint);
  }
});
add('raw-ingress-worker-failure-recover-resume', 'worker', 'local refusal, fail, sealed resume', 'hpo-argument', async () => {
  const first=await exercise({parallel:true,mutateTask:raw=>{
    const changed=raw.replace('"offset":2','"offset":2,"offset":2');
    assert.notEqual(changed,raw);
    return changed;
  }});
  assert.equal(first.error,null); assert.equal(first.outcome.checkpoint.trials[1].state,'failed');
  assert.equal(first.events.filter(e=>e.operation==='fail' && e.trial_index===1).length,1);
  assert.match(first.outcome.checkpoint.trials[1].error,/host HPO worker task.*not a strict TCV1 JSON document/);
  const recovered=dag.recover_host_hpo_checkpoint_json(first.lastCommitted,'null','[]');
  return {first,resumed:await exercise({parallel:true,checkpoint:recovered})};
}, r => { assert.equal(r.resumed.error,null); assert.deepEqual(r.resumed.outcome.checkpoint.trials.map(t=>t.state),['complete','failed','complete']); });

// Strict raw checkpoint arguments. All documents originate from a real owner
// callback; only the specifically named member is changed by each mutation.
let checkpointFixture;
async function ownerCheckpointFixture() {
  if (!checkpointFixture) {
    const result = await exercise({ budget: 1 });
    assert.equal(result.error, null);
    checkpointFixture = { committed: result.lastCommitted, prepared: result.lastPrepared,
      initial: result.events.find(e => e.operation === 'checkpoint').checkpointRaw };
    assert.ok(checkpointFixture.committed && checkpointFixture.prepared && checkpointFixture.initial);
  }
  return checkpointFixture;
}
const checkpointSites = [
  ['recovery-committed', 'host HPO recovery checkpoint'],
  ['recovery-prepared', 'host HPO recovery prepared checkpoint'],
  ['recovery-interrupted', 'host HPO recovery interrupted trials'],
  ['search', 'host HPO search checkpoint'],
  ['parallel-search', 'host HPO parallel search checkpoint'],
];
const checkpointMutations = ['valid', 'duplicate', 'escaped-duplicate', 'nfc', 'integer', 'positional-outer', 'positional-binding', 'positional-trial', 'typed-invalid', 'recursion', 'bad-seal'];
const interruptedMutations = ['valid', 'duplicate', 'escaped-duplicate', 'nfc', 'integer', 'positional-outer', 'typed-invalid', 'recursion'];
function malformedCheckpoint(raw, mutation, interrupted) {
  if (interrupted) {
    const x = '"trial_index":1,"params":';
    return ({ valid: '[]', duplicate: `[{${x}{"x":1,"x":1}}]`,
      'escaped-duplicate': `[{${x}{"x":1,"\\u0078":1}}]`,
      nfc: `[{${x}{"é":1,"e\\u0301":2}}]`, integer: `[{${x}{"x":18446744073709551616}}]`,
      'positional-outer': '[[1,{"x":1}]]', 'typed-invalid': '[{"trial_index":"bad","params":{}}]',
      recursion: '['.repeat(150) + '0' + ']'.repeat(150) })[mutation];
  }
  if (mutation === 'valid') return raw;
  if (mutation === 'duplicate') return raw.replace('"offset":1', '"offset":1,"offset":1');
  if (mutation === 'escaped-duplicate') return raw.replace('"offset":1', '"offset":1,"\\u006fffset":1');
  if (mutation === 'nfc') return raw.replace('"offset":1', '"offset":1,"é":1,"e\\u0301":2');
  if (mutation === 'integer') return raw.replace('"offset":1', '"offset":18446744073709551616');
  if (mutation === 'typed-invalid') return raw.replace('"schema_version":1', '"schema_version":"bad"');
  if (mutation === 'recursion') return '['.repeat(150) + '0' + ']'.repeat(150);
  const checkpoint = JSON.parse(raw);
  if (mutation === 'bad-seal') return raw.replace(checkpoint.fingerprint, '0'.repeat(64));
  const binding = checkpoint.binding;
  if (mutation === 'positional-binding' || mutation === 'positional-outer') checkpoint.binding = [binding.objective_fingerprint, binding.graph_fingerprint, binding.controller_fingerprint, binding.campaign_fingerprint, binding.fold_set_fingerprint, binding.data_fingerprint];
  if (mutation === 'positional-trial') checkpoint.trials[0].evidence = Object.values(checkpoint.trials[0].evidence);
  if (mutation === 'positional-outer') return JSON.stringify([checkpoint.schema_version, checkpoint.binding, checkpoint.trials, checkpoint.fingerprint]);
  return JSON.stringify(checkpoint);
}
for (const [site, label] of checkpointSites) {
  for (const mutation of site === 'recovery-interrupted' ? interruptedMutations : checkpointMutations) {
    add(`checkpoint-${site}-${mutation}`, site, mutation, 'checkpoint', async () => {
      const fixture = await ownerCheckpointFixture();
      const source = site === 'recovery-prepared' ? fixture.prepared : fixture.committed;
      const changed = malformedCheckpoint(source, mutation, site === 'recovery-interrupted');
      if (site === 'search' || site === 'parallel-search') {
        const run = await exercise({ budget: 2, parallel: site === 'parallel-search', checkpoint: changed });
        return { ...run, checkpointError: run.error };
      }
      let checkpointError = null, output = null;
      try {
        output = dag.recover_host_hpo_checkpoint_json(
          site === 'recovery-committed' ? changed : site === 'recovery-prepared' ? fixture.initial : fixture.committed,
          site === 'recovery-prepared' ? changed : 'null',
          site === 'recovery-interrupted' ? changed : '[]');
      } catch (error) { checkpointError = String(error); }
      return { checkpointError, output, events: [], invocations: 0 };
    }, result => {
      if (mutation === 'valid') { assert.equal(result.checkpointError, null); return; }
      assert.ok(result.checkpointError, `${site} accepted ${mutation}`);
      if (mutation === 'bad-seal') assert.match(result.checkpointError, /fingerprint mismatch/);
      else { assert.ok(result.checkpointError.includes(label), result.checkpointError); assert.match(result.checkpointError, /runtime_validation/); }
      assert.equal(result.invocations, 0); assert.equal(result.events.length, 0);
    });
  }
}
for (const [site, label] of checkpointSites.filter(([site]) => site === 'search' || site === 'parallel-search')) {
  add(`checkpoint-${site}-json-null`, site, 'refusal', 'checkpoint', () => exercise({ budget: 1, parallel: site === 'parallel-search', checkpoint: 'null' }), result => {
    assert.ok(result.error?.includes(label), result.error); assert.equal(result.invocations, 0); assert.equal(result.events.length, 0);
  });
}

// Mutations start with owner-produced complete or pruned checkpoints.
for (const pruned of [false, true]) {
  for (const [site, label] of checkpointSites.filter(([name]) => name !== 'recovery-interrupted')) {
    for (const location of ['evidence', 'scores', 'report']) {
      add(`n1-${pruned ? 'pruned' : 'complete'}-${site}-${location}`, site, 'unknown field', 'evidence', async () => {
        const owner = await exercise({budget: 2, pruning: pruned, pruneTrial: pruned});
        assert.equal(owner.error, null);
        const source = site === 'recovery-prepared' ? owner.lastPrepared : owner.lastCommitted;
        const changed = JSON.parse(source);
        const trial = changed.trials[pruned ? 1 : 0];
        assert.equal(trial.state, pruned ? 'pruned' : 'complete');
        const target = location === 'evidence' ? trial.evidence : location === 'scores' ? trial.evidence.scores : trial.evidence.scores.reports[0];
        target.n1_unknown = true;
        const raw = JSON.stringify(changed);
        if (site === 'search' || site === 'parallel-search') {
          const run = await exercise({budget: 2, parallel: site === 'parallel-search', pruning: pruned, checkpoint: raw});
          return {...run, checkpointError: run.error};
        }
        let checkpointError = null;
        try { dag.recover_host_hpo_checkpoint_json(site === 'recovery-prepared' ? owner.lastCommitted : raw, site === 'recovery-prepared' ? raw : 'null', '[]'); }
        catch (error) { checkpointError = String(error); }
        return {checkpointError, invocations: 0, events: []};
      }, result => {
        assert.match(result.checkpointError, /unknown field/);
        assert.ok(result.checkpointError.includes(label), result.checkpointError);
        assert.match(result.checkpointError, /runtime_validation/);
        assert.equal(result.invocations, 0); assert.equal(result.events.length, 0);
      });
    }
  }
}
add('n1-recovery-interrupted-unknown-control', 'recovery-interrupted', 'closed interrupted field', 'evidence', async () => {
  const owner = await ownerCheckpointFixture();
  assert.throws(() => dag.recover_host_hpo_checkpoint_json(owner.committed, 'null', '[{"trial_index":1,"params":{},"unknown_probe":true}]'), /host HPO recovery interrupted trials.*unknown field/);
  return {};
}, () => {});
for (const site of ['worker', 'complete', 'fold']) {
  for (const location of ['evidence', 'scores', 'report']) {
    if (site === 'fold' && location === 'evidence') continue;
    const mutate = raw => {
      const value = JSON.parse(raw);
      const container = site === 'fold' ? value : value.evidence;
      const target = location === 'evidence' ? container : location === 'scores' ? container.scores : container.scores.reports[0];
      target.n1_unknown = true;
      return JSON.stringify(value);
    };
    const options = {parallel: true, pruning: site !== 'worker', typedError: true,
      ...(site === 'fold' ? {mutateFold: mutate} : {mutateWorker: mutate})};
    add(`n1-${site}-${location}`, site, 'failed terminal', 'evidence', () => exercise(options), r => {
      assert.equal(r.error, null);
      assert.equal(r.outcome.checkpoint.trials[1].state, 'failed');
      assert.match(r.outcome.checkpoint.trials[1].error, /^serialization error: unknown field/);
      assert.equal(r.events.filter(e => e.operation === 'fail' && e.trial_index === 1).length, 1);
      assert.ok(r.lastCommitted);
      assert.doesNotThrow(() => dag.recover_host_hpo_checkpoint_json(r.lastCommitted, 'null', '[]'));
    });
  }
}


for (const [version, filename] of [['v1', '../examples/fixtures/estimator/training_outcome_no_refit.v1.json'], ['v2', '../examples/fixtures/training/replay/training_outcome_port_explicit.v2.json']]) {
  for (const location of ['scores', 'report']) {
    add(`n1-bundle-${version}-${location}`, 'bundle', 'closed nested field', 'evidence', () => {
      const original = JSON.parse(readFileSync(new URL(filename, import.meta.url), 'utf8')).execution_bundle;
      dag.validate_execution_bundle_json(JSON.stringify(original));
      const changed = structuredClone(original);
      const target = location === 'scores' ? changed.scores : changed.scores.reports[0];
      target.n1_unknown = true;
      assert.throws(() => dag.validate_execution_bundle_json(JSON.stringify(changed)), error => { assert.match(String(error), /serialization_error/); assert.match(String(error), /unknown field/); return true; });
      return {};
    }, () => {});
  }
}
for (const kind of ['fold', 'weights', 'producers']) {
  add(`n1-stacking-${kind}`, 'stacking', 'closed report field', 'evidence', () => {
    const report = {producer_node:'model:a', producer_port:'oof', partition:'validation', fold_id:'fold:0', level:'sample', row_count:2, target_width:1, metrics:{rmse:1.5}};
    const request = kind === 'producers'
      ? {producer_nodes:['model:a'], select:'best', metric:'rmse', reports:[report], fold_ids:['fold:0']}
      : {producer_node:'model:a', fold_ids:['fold:0'], metric:'rmse', reports:[report]};
    const call = kind === 'producers' ? dag.select_stacking_producers_json : kind === 'fold' ? dag.select_stacking_fold_json : dag.stacking_fold_weights_json;
    call(JSON.stringify(request));
    report.n1_unknown = true;
    assert.throws(() => call(JSON.stringify(request)), error => { assert.match(String(error), /serialization_error/); assert.match(String(error), /unknown field/); return true; });
    return {};
  }, () => {});
}

add('pruning-fold-scores-raw-surrogate', 'worker-fold', 'invalid UTF-16', 'utf16',
  () => exercise({ parallel: true, pruning: true, mutateFold: raw => raw.replace('"scores":{', '"scores":{"probe":"' + '\ud800' + '",') }), result => {
    assert.equal(result.outcome.checkpoint.trials[1].state, 'failed');
    assert.match(result.outcome.checkpoint.trials[1].error, /invalid UTF-16/);
  });


// Callback exceptions must settle the owner workflow even if inspection of a
// host-provided object would itself throw or encounter a cyclic graph.
function hostileValues() {
  const getter = new Error('message'); Object.defineProperty(getter, 'stack', { get() { throw new Error('hostile stack'); } });
  const tag = {}; Object.defineProperty(tag, Symbol.toStringTag, { get() { throw new Error('hostile tag'); } });
  const revoked = Proxy.revocable({}, {}); revoked.revoke();
  const cycle = []; cycle.push(cycle);
  const doubled = []; doubled.push(doubled, doubled);
  return [['string-high', '\ud800'], ['string-low', '\udc00'], ['replacement', '�'],
    ['normal', 'plain'], ['literal-escape', '\\ud800'], ['emoji', '😀'], ['symbol-high', Symbol('\ud800')], ['symbol-none', Symbol()], ['symbol-empty', Symbol('')], ['number', 42], ['object', { message: 'plain' }], ['error-normal', new Error('plain')], ['custom-tag', Object.assign([], { [Symbol.toStringTag]: 'Custom' })], ['tag-high', { [Symbol.toStringTag]: '\ud800' }], ['tag-low', { [Symbol.toStringTag]: '\udc00' }], ['tag-replacement', { [Symbol.toStringTag]: '�' }], ['tag-literal', { [Symbol.toStringTag]: '\\ud800' }],
    ['error-high', new Error('\ud800')], ['throwing-stack', getter], ['hostile-tag', tag],
    ['proxy', new Proxy({}, { get() { throw new Error('hostile proxy'); } })],
    ['revoked-proxy', revoked.proxy], ['cyclic-array', cycle], ['self-doubled-array', doubled]];
}
const errorSiteNames = ['optimizer', 'controller', 'worker-dispatch', 'worker-reject'];
async function boundedErrorRun(site, value) {
  let options;
  if (site === 'optimizer') options = { budget: 1, reply: op => { if (op === 'ask') throw value; } };
  else if (site === 'controller') options = { budget: 1, controllerThrow: value };
  else if (site === 'worker-dispatch') options = { budget: 2, parallel: true, dispatchThrow: value };
  else options = { budget: 2, parallel: true, rejectValue: value };
  let timer;
  try { return await Promise.race([exercise(options), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('callback error workflow timed out')), 3000); })]); }
  finally { clearTimeout(timer); }
}
for (const site of errorSiteNames) for (const [kind] of hostileValues()) {
  add(`error-render-${site}-${kind}`, site, 'controlled error', 'exception', async () => {
    const value = hostileValues().find(([name]) => name === kind)[1];
    return boundedErrorRun(site, value);
  }, result => {
    const text = site.startsWith('worker-') ? result.outcome?.checkpoint?.trials?.[1]?.error : result.error;
    assert.ok(text, `missing ${site} failure`);
    const rendered = text.startsWith('{') ? JSON.parse(text).message : text;
    assert.match(text, site === 'optimizer' ? /JS HPO optimizer `ask` threw:/ : site === 'controller' ? /JS controller .* threw:/ : site === 'worker-dispatch' ? /worker dispatch threw:/ : /worker promise rejected:/);
    if (kind === 'string-high' || kind === 'error-high') assert.match(text, /ill_formed_utf16/);
    if (kind === 'string-low') assert.match(text, /ill_formed_utf16/);
    if (kind === 'replacement') assert.ok(text.includes('�'));
    if (kind === 'normal') assert.ok(rendered.includes('JsValue(\"plain\")'));
    if (kind === 'number') assert.ok(rendered.includes('JsValue(42)'));
    if (kind === 'object') assert.ok(text.includes('Object('));
    if (kind === 'emoji') assert.ok(text.includes('😀'));
    if (kind === 'symbol-high') assert.match(text, /ill_formed_utf16/);
    if (kind === 'symbol-none') assert.ok(rendered.includes('JsValue(Symbol)'));
    if (kind === 'symbol-empty') assert.ok(rendered.includes('JsValue(Symbol())'));
    if (kind === 'error-normal') assert.match(text, /Error: plain/);
    if (kind === 'custom-tag') assert.ok(text.includes('Custom'));
    if (kind === 'tag-high' || kind === 'tag-low') assert.match(text, /ill_formed_utf16/);
    if (kind === 'tag-replacement') assert.ok(text.includes('�'));
    if (kind === 'hostile-tag' || kind === 'proxy' || kind === 'revoked-proxy') assert.match(text, /unrenderable JS value/);
    if (kind === 'throwing-stack') { assert.ok(rendered.includes('JsValue(Error: message)'), rendered); assert.doesNotMatch(text, /unreadable|hostile stack/); }
    if (kind === 'cyclic-array' || kind === 'self-doubled-array') assert.match(text, /\.\.\./);
    if (site.startsWith('worker-')) { assert.equal(result.error, null); assert.equal(result.outcome.status, 'failed'); assert.ok(result.events.some(e => e.operation === 'fail' && e.trial_index === 1)); }
  });
}
add('error-render-worker-fingerprint-distinct', 'worker-reject', 'distinct checkpoint identities', 'exception', async () => {
  const results = [];
  for (const value of ['\ud800', '\udc00', '�', '\\ud800']) results.push(await boundedErrorRun('worker-reject', value));
  return { results };
}, result => assert.equal(new Set(result.results.map(r => r.outcome.checkpoint.fingerprint)).size, 4));
add('error-render-single-property-read', 'worker-reject', 'one read per property', 'exception', async () => {
  const counts = { name: 0, message: 0, stack: 0, toJSON: 0 };
  const error = new Error();
  for (const key of ['name', 'message', 'stack']) Object.defineProperty(error, key, { get() { counts[key]++; return key; } });
  const object = {}; Object.defineProperty(object, 'toJSON', { get() { counts.toJSON++; return () => ({ value: 1 }); } });
  const errorRun = await boundedErrorRun('worker-reject', error), objectRun = await boundedErrorRun('worker-reject', object);
  return { counts, errorRun, objectRun };
}, result => assert.deepEqual(result.counts, { name: 1, message: 1, stack: 0, toJSON: 1 }));
// An Error renders as `name: message`; its stack is never read. A single-file
// build inlines WASM as data: URLs, so one host stack frame can carry megabytes.
for (const site of errorSiteNames) {
  add(`error-render-${site}-inline-wasm-stack-unread`, site, 'name and message only', 'exception', async () => {
    let stackReads = 0;
    const error = new TypeError('inline refusal');
    Object.defineProperty(error, 'stack', { configurable: true, get() {
      stackReads++;
      return 'TypeError: inline refusal\n    at f (data:application/wasm;base64,AGFzbQEAAAA=:wasm-function[1]:0x1)';
    } });
    const run = await boundedErrorRun(site, error);
    return { run, stackReads };
  }, result => {
    const { run } = result;
    const text = site.startsWith('worker-') ? run.outcome?.checkpoint?.trials?.[1]?.error : run.error;
    assert.ok(text, `missing ${site} failure`);
    const rendered = text.startsWith('{') ? JSON.parse(text).message : text;
    assert.equal(result.stackReads, 0);
    assert.ok(rendered.includes('JsValue(TypeError: inline refusal)'), rendered);
    assert.doesNotMatch(text, /data:application\/wasm|wasm-function/);
    if (site.startsWith('worker-')) { assert.equal(run.error, null); assert.equal(run.outcome.status, 'failed'); }
  });
}


add('error-render-array-getter-budget', 'worker-reject', 'bounded hostile reads', 'exception', async () => {
  let reads = 0;
  const target = [];
  const value = new Proxy(target, { get(object, key, receiver) {
    if (key === 'length') return 1000000000;
    if (typeof key === 'string' && /^\d+$/.test(key)) { reads++; throw new Error('index denied'); }
    return Reflect.get(object, key, receiver);
  } });
  const run = await boundedErrorRun('worker-reject', value);
  return { run, reads };
}, result => { assert.ok(result.reads > 0 && result.reads <= 1024, String(result.reads)); assert.match(result.run.outcome.checkpoint.trials[1].error, /\.\.\./); });
add('error-render-custom-tags-distinct', 'worker-reject', 'distinct malformed tag identities', 'exception', async () => {
  const results = [];
  for (const tag of ['\ud800', '\udc00', '�', '\\ud800']) results.push(await boundedErrorRun('worker-reject', { [Symbol.toStringTag]: tag }));
  return { results };
}, result => assert.equal(new Set(result.results.map(r => r.outcome.checkpoint.fingerprint)).size, 4));


// Symbol-valued Error fields must use the same lossless renderer as a thrown
// Symbol. The stack is not rendered, so only name and message carry identity.
for (const field of ['name', 'message']) {
  add(`error-render-symbol-field-${field}`, 'worker-reject', 'four distinct identities', 'exception', async () => {
    const results = [];
    for (const description of ['\ud800', '\udc00', '�', '\\ud800']) {
      const value = new Error('controlled');
      Object.defineProperty(value, 'stack', { value: 'fixed stack', configurable: true });
      Object.defineProperty(value, field, { value: Symbol(description), configurable: true });
      results.push(await boundedErrorRun('worker-reject', value));
    }
    return { results };
  }, result => {
    const fingerprints = new Set();
    for (const run of result.results) {
      assert.equal(run.error, null); assert.equal(run.outcome.status, 'failed');
      const trial = run.outcome.checkpoint.trials[1];
      assert.match(trial.error, /worker promise rejected: JsValue\(/);
      assert.ok(run.events.some(e => e.operation === 'fail' && e.trial_index === 1));
      fingerprints.add(run.outcome.checkpoint.fingerprint);
    }
    assert.match(result.results[0].outcome.checkpoint.trials[1].error, /ill_formed_utf16/);
    assert.match(result.results[1].outcome.checkpoint.trials[1].error, /ill_formed_utf16/);
    assert.ok(result.results[2].outcome.checkpoint.trials[1].error.includes('�'));
    assert.equal(fingerprints.size, 4);
  });
}


for (const route of ['plain', 'fold', 'complete']) {
  for (const mode of ['valid', 'immediate', 'constructor-getter', 'then-getter', 'then-method', 'then-noncallable']) {
    add(`b1-${route}-${mode}`, 'worker-promise', 'child settles with owner checkpoint', 'promise-adoption', () => {
      const env = { ...process.env, DAG_ML_B1_CHILD: JSON.stringify({ route, mode }) };
      delete env.NODE_OPTIONS;
      const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url)], { env, encoding: 'utf8', timeout: 12000 });
      return { childStatus: child.status, childSignal: child.signal, childError: child.error?.message ?? null,
        childStdout: child.stdout, childStderr: child.stderr };
    }, result => {
      assert.equal(result.childError, null, result.childError);
      assert.equal(result.childSignal, null, result.childSignal);
      assert.equal(result.childStatus, 0, `${result.childStdout}\n${result.childStderr}`);
      const row = JSON.parse(result.childStdout.trim());
      assert.equal(row.route, route); assert.equal(row.mode, mode); assert.equal(row.committed, true);
    });
  }
}

if (process.env.HPO_INSPECT !== '1') {
  let failures = 0;
  assert.equal(cases.length, 428, 'HPO callback ingress case count');
  if (process.env.HPO_CAPTURE_DIR) mkdirSync(process.env.HPO_CAPTURE_DIR, { recursive: true });
  for (const c of cases) {
    let result, assertion = null;
    try { result = await c.run(); c.verify(result); } catch (error) { assertion = String(error?.stack ?? error); failures++; }
    const events = result?.events ?? [];
    const summary = {
      label: c.label, site: c.site, expected: c.expected, classification: c.classification,
      observed_stage: result?.error ? 'search_error' : result?.outcome?.status ?? 'setup_error',
      ok: assertion === null, error: result ? (result.error ?? result.outcome?.checkpoint?.trials?.find(t => t.state === 'failed')?.error ?? null) : assertion,
      assertions: assertion, callbacks: events.map(e => ({ operation: e.operation, trial_index: e.trial_index,
        status: e.payload.status ?? null, checkpoint_states: e.payload.checkpoint?.trials?.map(t => t.state) ?? null,
        error: e.operation === 'fail' ? e.payload.error : null })),
      invocations: result?.invocations ?? 0,
      trial_params: result?.outcome?.checkpoint?.trials?.map(t => t.evidence?.params ?? t.params) ?? null,
      worker_wires: result?.wires?.map(w => ({ trial_index: w.trial_index, kind: w.kind,
        original_sha256: digest(w.original), returned_sha256: digest(w.returned), mutated: w.original !== w.returned })) ?? [],
    };
    if (process.env.HPO_CAPTURE_DIR && result) {
      writeFileSync(`${process.env.HPO_CAPTURE_DIR}/${c.label}.json`, JSON.stringify({ ...summary, events, wires: result.wires, lastCommitted: result.lastCommitted, lastPrepared: result.lastPrepared }, null, 2) + '\n');
    }
    console.log(JSON.stringify(summary));
  }
  console.log(JSON.stringify({ cases: cases.length, passed: cases.length - failures, failed: failures, skipped: 0 }));
  if (failures) process.exitCode = 1;
}
