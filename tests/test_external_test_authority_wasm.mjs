// Exported WASM replay authority controls with a synthetic host; no numerical claim.
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const modulePath = process.env.DAG_ML_WASM_PKG;
assert.ok(modulePath && path.isAbsolute(modulePath), 'DAG_ML_WASM_PKG must be an absolute file path');
assert.ok(statSync(modulePath).isFile(), 'DAG_ML_WASM_PKG must be a file');
const dag = createRequire(import.meta.url)(modulePath);
assert.equal(typeof dag.replay_training_package_json, 'function');
const root = new URL('../crates/dag-ml-core/tests/fixtures/multi_output_target_claims/', import.meta.url);
const read = name => JSON.parse(readFileSync(new URL(name, root), 'utf8'));
const packageJson = readFileSync(new URL('explain/package.json', root), 'utf8');
const pkg = JSON.parse(packageJson);
const trusted = read('explain/training-request.json').controller_manifests;
const trainingRelations = read('shared/training-relations.json');
const cases = [];
for (const phase of ['PREDICT', 'EXPLAIN']) {
  for (const mode of ['positive', 'wrong-set', 'wrong-string', 'missing-string']) {
    cases.push({ phase, mode });
  }
}

for (const { phase, mode } of cases) {
  const label = `${phase}:${mode}`;
  const envelopes = read('shared/envelopes.json');
  for (const [key, base] of Object.entries(envelopes)) {
    const oldCohort = base.predict_cohort;
    const requirement = pkg.execution_bundle.data_requirements.find(item => `${item.node_id}.${item.input_name}` === key);
    assert.ok(requirement?.relation_fingerprint, `${label}: missing owner relation fingerprint for ${key}`);
    base.coordinator_relations = structuredClone(trainingRelations);
    base.relation_fingerprint = requirement.relation_fingerprint;
    envelopes[key] = JSON.parse(dag.attach_predict_cohort_to_envelope_json(
      JSON.stringify(base),
      JSON.stringify({
        role: 'external_test',
        relations: oldCohort.relations,
        target_names: phase === 'EXPLAIN' ? ['protein'] : ['protein', 'fat'],
        data_content_fingerprint: oldCohort.data_content_fingerprint,
        target_content_fingerprint: 'd'.repeat(64),
      }),
    ));
    const envelope = envelopes[key];
    if (mode === 'wrong-set') {
      // Retain the owner's string while replacing the underlying relation set.
      envelope.coordinator_relations = { records: [{
        observation_id: 'observation:authority:unrelated',
        sample_id: 'sample:authority:unrelated',
      }] };
      assert.equal(envelope.relation_fingerprint, requirement.relation_fingerprint);
    } else if (mode === 'wrong-string') {
      envelope.relation_fingerprint = 'a'.repeat(64);
      assert.notEqual(envelope.relation_fingerprint, requirement.relation_fingerprint);
    } else if (mode === 'missing-string') {
      delete envelope.relation_fingerprint;
    }
  }
  const request = JSON.parse(dag.sign_training_replay_request_json(JSON.stringify({
    ...read('shared/request.json'),
    phase,
    source_outcome_fingerprint: pkg.training_outcome.outcome_fingerprint,
    output_binding_ids: phase === 'EXPLAIN' ? ['output:a'] : ['output:a', 'output:b'],
  })));
  const counts = { callbacks: 0, invoke: 0, hydrate: 0, release: 0 };
  let next = 1;
  const callback = (owner, json) => {
    counts.callbacks++;
    const task = JSON.parse(json);
    if (task.operation === 'hydrate_artifact_payload') {
      counts.hydrate++;
      return JSON.stringify({
        operation: 'hydrated_artifact_payload', schema_version: 1,
        handle: { handle: next++, kind: 'artifact', owner_controller: owner },
      });
    }
    if (task.operation === 'release_hydrated_artifact_payload') {
      counts.release++;
      return JSON.stringify({ operation: 'released_hydrated_artifact_payload', schema_version: 1 });
    }
    counts.invoke++;
    const node = task.node_plan;
    const model = node.kind === 'model';
    const port = model ? 'oof' : 'x_out';
    const targets = model ? pkg.output_bindings.find(binding => binding.node_id === node.node_id).target_names : [];
    const ids = ['sample:11', 'sample:12', 'sample:13', 'sample:14'];
    return JSON.stringify({
      node_id: node.node_id,
      outputs: { [port]: { handle: next++, kind: model ? 'prediction' : 'data', owner_controller: owner } },
      predictions: model && phase === 'PREDICT' ? [{
        producer_node: node.node_id, producer_port: port, partition: 'final', fold_id: null,
        sample_ids: ids, values: ids.map(() => targets.map(() => 0)), target_names: targets,
      }] : [],
      explanations: model && phase === 'EXPLAIN' ? [{
        producer_node: node.node_id, producer_port: port, method: 'synthetic',
        target_name: targets[0], payload: { importance: 0 },
      }] : [],
      lineage: {
        record_id: `lineage:external-test-authority:${next++}`, run_id: task.run_id,
        node_id: node.node_id, phase, controller_id: owner, controller_version: node.controller_version,
        variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
        input_lineage: [], artifact_refs: [], params_fingerprint: node.params_fingerprint,
        data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null, seed: null,
        unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: [],
      },
    });
  };
  let outcome;
  let error;
  try {
    outcome = JSON.parse(dag.replay_training_package_json(
      packageJson, JSON.stringify(request), JSON.stringify(envelopes), JSON.stringify(trusted),
      `outcome:external-test-authority:${phase}:${mode}`, `run:external-test-authority:${phase}:${mode}`, callback,
    ));
  } catch (failure) {
    error = String(failure);
  }
  if (mode === 'positive') {
    assert.equal(error, undefined, `${label}: ${error}`);
    assert.ok(outcome.result_count > 0, `${label}: no replay results`);
    assert.ok(counts.invoke > 0, `${label}: no controller callback`);
    assert.ok(counts.hydrate > 0, `${label}: no artifact hydration callback`);
    assert.equal(counts.release, counts.hydrate, `${label}: artifact lifecycle`);
    if (phase === 'PREDICT') assert.equal(outcome.outputs.length, 2, label);
    else assert.equal(outcome.explanations.length, 1, label);
  } else {
    assert.ok(error, `${label}: replay unexpectedly succeeded`);
    const descriptor = JSON.parse(error);
    // Missing strings fail earlier in provider_for_replay's binding validation.
    const expected = mode === 'missing-string'
      ? 'data binding `x` on `transform:snv` requires relations but has no relation_fingerprint'
      : 'external_test training relations do not match owner authority';
    assert.equal(descriptor.context.detail, expected, label);
    assert.deepEqual(counts, { callbacks: 0, invoke: 0, hydrate: 0, release: 0 }, label);
  }
  console.log(JSON.stringify({ label, ok: !error, error, ...counts }));
}
console.log(JSON.stringify({ cases: cases.length, passed: cases.length, failed: 0, skipped: 0 }));
