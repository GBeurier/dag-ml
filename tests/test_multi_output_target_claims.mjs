// Actual wasm32/Node public binding replay. SYNTHETIC HOST/MOCK; no Methods/IO qualification.
// Build wasm-bindgen --target nodejs into scratch and set DAG_ML_WASM_PKG to its .js file.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
assert.ok(process.env.DAG_ML_WASM_PKG, 'DAG_ML_WASM_PKG must identify the scratch-built module');
const dag = require(process.env.DAG_ML_WASM_PKG);
const root = new URL('../crates/dag-ml-core/tests/fixtures/multi_output_target_claims/', import.meta.url);
const read = name => JSON.parse(readFileSync(new URL(name, root), 'utf8'));
const packageJson = readFileSync(new URL('explain/package.json', root), 'utf8');
const pkg = JSON.parse(packageJson); // Inspection only: preserve exact signed integer bytes for the owner.
const trusted = read('explain/training-request.json').controller_manifests;
const prototype = read('shared/envelopes.json');
const base = read('shared/request.json');
for (const phase of ['PREDICT', 'EXPLAIN']) {
  for (const names of [['protein', 'fat'], ['fat', 'protein']]) {
    const envelopes = structuredClone(prototype);
    for (const [key, envelope] of Object.entries(envelopes)) {
      const old = envelope.predict_cohort;
      envelopes[key] = JSON.parse(dag.attach_predict_cohort_to_envelope_json(JSON.stringify(envelope), JSON.stringify({
        role: 'inference', relations: old.relations, target_names: names,
        data_content_fingerprint: old.data_content_fingerprint,
      })));
    }
    const request = JSON.parse(dag.sign_training_replay_request_json(JSON.stringify({
      ...base, phase, source_outcome_fingerprint: pkg.training_outcome.outcome_fingerprint,
    })));
    let invokes = 0, hydrate = 0, release = 0, next = 1;
    const handles = new Set();
    const callback = (owner, json) => {
      const task = JSON.parse(json);
      if (task.operation === 'hydrate_artifact_payload') {
        assert.equal(Buffer.from(task.payload).toString(), 'synthetic-host-fitted-state-v1');
        hydrate += 1;
        const handle = { handle: next++, kind: 'artifact', owner_controller: owner };
        handles.add(handle.handle);
        return JSON.stringify({ operation: 'hydrated_artifact_payload', schema_version: 1, handle });
      }
      if (task.operation === 'release_hydrated_artifact_payload') {
        assert.ok(handles.delete(task.handle.handle)); release += 1;
        return JSON.stringify({ operation: 'released_hydrated_artifact_payload', schema_version: 1 });
      }
      invokes += 1;
      const node = task.node_plan, model = node.kind === 'model';
      const port = model ? 'oof' : 'x_out';
      const targets = model ? pkg.output_bindings.find(b => b.node_id === node.node_id).target_names : [];
      if (model) assert.ok(Object.values(task.input_handles).some(h => handles.has(h.handle)));
      const samples = Object.values(envelopes)[0].predict_cohort.physical_sample_ids;
      return JSON.stringify({
        node_id: node.node_id,
        outputs: { [port]: { handle: next++, kind: model ? 'prediction' : 'data', owner_controller: owner } },
        predictions: model && phase === 'PREDICT' ? [{
          producer_node: node.node_id, producer_port: port, partition: 'final', fold_id: null,
          sample_ids: samples, values: samples.map(() => targets.map(() => 0)), target_names: targets,
        }] : [],
        explanations: model && phase === 'EXPLAIN' ? [{
          producer_node: node.node_id, producer_port: port, method: 'synthetic_zero_importance',
          target_name: targets[0], payload: { importance: 0 },
        }] : [],
        lineage: {
          record_id: 'lineage:wasm.synthetic:' + next++, run_id: task.run_id, node_id: node.node_id,
          phase, controller_id: owner, controller_version: node.controller_version,
          variant_id: task.variant_id, fold_id: task.fold_id, branch_path: task.branch_path,
          input_lineage: [], artifact_refs: [], params_fingerprint: node.params_fingerprint,
          data_model_shape_fingerprint: null, aggregation_policy_fingerprint: null,
          seed: null, unsafe_flags: [], metrics: {}, loss_attestations: [], early_stopping_records: [],
        },
      });
    };
    const replay = JSON.parse(dag.replay_training_package_json(packageJson, JSON.stringify(request),
      JSON.stringify(envelopes), JSON.stringify(trusted), 'outcome:wasm.synthetic', 'run:wasm.synthetic', callback));
    if (phase === 'PREDICT') assert.deepEqual(replay.outputs.map(o => o.predictions[0].target_names), [['protein'], ['fat']]);
    else assert.equal(replay.explanations.length, 2);
    assert.equal(invokes, 3); assert.equal(hydrate, 2); assert.equal(release, 2);
    console.log(JSON.stringify({ phase, names, invokes, hydrate, release, outcome_fingerprint: replay.outcome_fingerprint }));
  }
}
