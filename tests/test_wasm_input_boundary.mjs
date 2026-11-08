// WASM exported input conversion: the owner must see only well-formed JS text.
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const modulePath = process.env.DAG_ML_WASM_PKG;
assert.ok(modulePath && path.isAbsolute(modulePath) && statSync(modulePath).isFile(), 'DAG_ML_WASM_PKG must be an absolute file');
const dag = createRequire(import.meta.url)(modulePath);
const declarations = readFileSync(path.join(path.dirname(modulePath), 'dag_ml_wasm.d.ts'), 'utf8');
const functions = [...declarations.matchAll(/^export function (\w+)\((.*?)\):\s*[^;]+;/gm)]
  .map(([, name, args]) => ({ name, args, receiver: null }));
const registryText = declarations.match(/export class LocalImplementationRegistry \{([\s\S]*?)\n\}/)?.[1];
assert.ok(registryText, 'missing registry declaration');
const methods = [...registryText.matchAll(/^\s*(\w+)\((.*?)\):\s*[^;]+;/gm)]
  .map(([, name, args]) => ({ name, args, receiver: 'registry' }));
const entries = [...functions, ...methods].map(entry => ({ ...entry,
  parameters: entry.args ? entry.args.split(',').map(raw => {
    const m = raw.trim().match(/^(\w+)(\?)?:\s*(.+)$/); assert.ok(m, `unrecognized argument ${raw}`);
    return { name: m[1], optional: !!m[2] || /\bnull\b|\bundefined\b/.test(m[3]), type: m[3], string: /\bstring\b/.test(m[3]) };
  }) : [],
})).filter(entry => entry.parameters.some(parameter => parameter.string));
assert.equal(entries.length, 63, 'exported string input count');
assert.equal(entries.flatMap(entry => entry.parameters.filter(p => p.string)).length, 166, 'string parameter count');
assert.equal(entries.flatMap(entry => entry.parameters.filter(p => p.string && p.optional)).length, 6, 'optional string parameter count');
const cases = [];
function add(label, run) { cases.push({ label, run }); }
for (const entry of entries) for (const [index, parameter] of entry.parameters.entries()) {
  if (!parameter.string) continue;
  for (const [kind, invalid] of [['raw-high', '\ud800'], ['raw-low', '\udc00'], ['number', 17], ['boxed', new String('x')]]) {
    add(`${entry.name}.${parameter.name}.${kind}`, async () => {
      let callbacks = 0;
      const args = entry.parameters.map(p => p.string ? (p.optional ? null : '') : /Function/.test(p.type) ? (() => { callbacks++; }) : 1);
      args[index] = invalid;
      const owner = entry.receiver ? new dag.LocalImplementationRegistry() : dag;
      try {
        let error;
        try { await owner[entry.name](...args); } catch (failure) { error = String(failure); }
        assert.ok(error, 'expected an input refusal');
        assert.ok(error.includes(`argument \`${parameter.name}\``), error);
        assert.match(error, kind.startsWith('raw-') ? /not valid UTF-16/ : /must be a string/);
        assert.equal(callbacks, 0, 'owner callback occurred before input validation');
      } finally { if (entry.receiver) owner.free(); }
    });
  }
  if (parameter.optional) for (const [kind, value] of [['null', null], ['undefined', undefined], ['omitted', undefined], ['empty', '']]) {
    add(`${entry.name}.${parameter.name}.${kind}`, async () => {
      let callbacks = 0;
      const args = entry.parameters.map(p => p.string ? (p.optional ? null : '') : /Function/.test(p.type) ? (() => { callbacks++; }) : 1);
      args[index] = value;
      if (kind === 'omitted' && index === args.length - 1) args.pop();
      const owner = entry.receiver ? new dag.LocalImplementationRegistry() : dag;
      try {
        try { await owner[entry.name](...args); } catch (failure) {
          assert.ok(!String(failure).includes(`argument \`${parameter.name}\``), String(failure));
        }
        assert.equal(callbacks, 0);
      } finally { if (entry.receiver) owner.free(); }
    });
  }
}
const graph = { id: 'graph:input-probe', interface: { inputs: [], outputs: [] }, nodes: [{ id: 'model:m', kind: 'model', operator: { type: 'JSModel' }, params: {}, ports: { inputs: [], outputs: [{ name: 'oof', kind: 'prediction', representation: null, cardinality: 'one', description: '' }] }, metadata: {}, seed_label: null }], edges: [], search_space_fingerprint: null, metadata: {} };
const manifest = JSON.parse(dag.derive_controller_manifest_json(JSON.stringify({ controller_id: 'controller:js-hpo-model', controller_version: '1.0.0', operator_kind: 'model', input_ports: [], output_ports: graph.nodes[0].ports.outputs })));
const foldSet = { id: 'folds:input-probe', sample_ids: ['s1', 's2'], sample_groups: {}, folds: [ { fold_id: 'fold:0', train_sample_ids: ['s2'], validation_sample_ids: ['s1'], metadata: {} }, { fold_id: 'fold:1', train_sample_ids: ['s1'], validation_sample_ids: ['s2'], metadata: {} } ] };
const campaign = { id: 'campaign:input-probe', root_seed: 7, split_invocation: { id: 'split:outer', controller_id: null, leakage_policy: { split_unit: 'sample', forbid_origin_cross_fold: true, allow_observation_split_with_shared_target: false, require_group_ids: false, unsafe_flags: [] }, params: {}, fold_set: foldSet } };
const fingerprint = value => {
  const raw = JSON.stringify(graph).replace('"metadata":{}', `"metadata":{"probe":"${value}"}`);
  return JSON.parse(dag.build_execution_plan_json('plan:input-probe', raw, JSON.stringify(campaign), JSON.stringify([manifest]))).graph_fingerprint;
};
for (const [name, value] of [['high', '\ud800'], ['low', '\udc00']]) add(`graph.${name}`, () => assert.throws(() => fingerprint(value), /argument `graph_json` is not valid UTF-16/));
let validHashes;
add('graph.valid-controls', () => { validHashes = ['�', '😀', 'plain', 'e\u0301'].map(fingerprint); assert.equal(new Set(validHashes).size, 4); });
add('graph.valid-pair', () => assert.equal(fingerprint('\ud83d\ude00'), validHashes[1]));
const EXPECTED_CASES = 692;
assert.equal(cases.length, EXPECTED_CASES, 'input boundary case count');
let failed = 0;
for (const test of cases) { try { await test.run(); } catch (error) { failed++; console.error(`${test.label}: ${String(error?.stack ?? error)}`); } }
console.log(JSON.stringify({ cases: cases.length, passed: cases.length - failed, failed, skipped: 0 }));
if (failed) process.exitCode = 1;
