// Real staged WASM splitter only; no controllers or fits.
import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const modulePath = process.env.DAG_ML_WASM_PKG;
assert.ok(modulePath && path.isAbsolute(modulePath), 'DAG_ML_WASM_PKG must be an absolute file path');
assert.ok(statSync(modulePath).isFile(), 'DAG_ML_WASM_PKG must be a file');
const dag = createRequire(import.meta.url)(modulePath);
const spec = JSON.stringify({ n_splits: 2 });
const observed = ['s0', 's1', 's2'];
const candidates = [...observed, 'missing'];
const groups = { s0: 'g0', s1: 'g0', s2: 'g1' };
const split = (ids, map, missing) => dag.group_kfold_split_json(
  spec, JSON.stringify(ids), JSON.stringify(map), 'outer',
  missing === undefined ? undefined : JSON.stringify(missing),
);
const json = split(candidates, groups, ['missing']);
const folds = JSON.parse(json);
assert.deepEqual(folds.target_missing_sample_ids, ['missing']);
assert.deepEqual(folds.sample_ids, observed);
assert.deepEqual(folds.sample_groups, groups);
const validated = [];
for (const fold of folds.folds) {
  assert.ok(!fold.train_sample_ids.includes('missing'));
  assert.ok(!fold.validation_sample_ids.includes('missing'));
  const trainGroups = new Set(fold.train_sample_ids.map(id => groups[id]));
  assert.ok(fold.validation_sample_ids.every(id => !trainGroups.has(groups[id])));
  validated.push(...fold.validation_sample_ids);
}
assert.deepEqual(validated.sort(), observed);
assert.equal(split(candidates, { ...groups, missing: 'g0' }, ['missing']), json);
const baseline = JSON.parse(split(observed, groups));
assert.deepEqual(folds.folds, baseline.folds);
assert.equal(split(observed, groups, []), split(observed, groups));
console.log('GroupKFold target-missing WASM export regression passed');
