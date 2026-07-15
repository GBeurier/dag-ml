# Public training replay contracts

This page specifies the public contract for replaying a completed DAG-ML
training outcome on a current cohort. The same `TrainingReplayRequest` /
`TrainingReplayOutcome` authority is used by attached replay while the original
`TrainingResult` is alive and by stateless package replay after a host reloads a
`PortablePredictorPackage` with explicit sidecar artifact handles.

The operation has three authorities:

1. a complete, validated source `TrainingOutcome` and its execution bundle;
2. a canonical `TrainingReplayRequest` selecting a phase, input envelopes and output
   bindings;
3. current coordinator data-plan envelopes whose changing cohort identities are
   recorded in the resulting `TrainingReplayOutcome`.

The longer “training replay” name is used by schema filenames, schema IDs,
fixtures and this documentation to distinguish it from the low-level
`ReplayPhaseRequest` / `ReplayExecutionSummary` API and the pre-existing
estimator replay fixture family.

## Request syntax

The canonical JSON form is:

```json
{
  "schema_version": 1,
  "request_id": "replay:request.predict",
  "source_outcome_fingerprint": "<64 lowercase hex characters>",
  "phase": "PREDICT",
  "data_envelope_keys": ["model:ridge.x"],
  "output_binding_ids": ["output:final"],
  "request_fingerprint": "<TCV1 fingerprint>"
}
```

All fields are required and unknown fields are rejected.

| Keyword | Syntax | Effect |
|---|---|---|
| `schema_version` | integer `1` | Selects this public request contract. |
| `request_id` | portable DAG-ML identifier | Gives this authorization a stable identity. |
| `source_outcome_fingerprint` | lowercase SHA-256 | Pins the complete source training outcome; a transplant is rejected. |
| `phase` | `PREDICT` or `EXPLAIN` | Selects forward prediction or explanation. `REFIT` is not replay V1. |
| `data_envelope_keys` | non-empty, strictly sorted unique `node_id.input_name` strings | Must exactly name the source bundle's direct data requirements. |
| `output_binding_ids` | non-empty, strictly sorted unique identifiers | Must exactly select known output bindings. |
| `request_fingerprint` | lowercase SHA-256 | TCV1 of the entire request with this field omitted. |

Lists use increasing UTF-8 string order. Implementations must reject an empty,
duplicate, unsorted or unknown member; they must not silently sort caller input. See
`examples/fixtures/training/replay/training_replay_request_predict.v1.json` and
`training_replay_request_explain.v1.json` for complete examples.

## Current input envelopes

Each request key resolves to one current `CoordinatorDataPlanEnvelope`. Replay divides
its identity fields into frozen and current parts:

| Frozen from training | Re-attested for the current cohort |
|---|---|
| envelope schema fingerprint | relation fingerprint |
| data-plan structure/fingerprint | data-content fingerprint |
| input representation | target-content fingerprint |
| source IDs | composite current identity fingerprint |
| `feature_set_id` | coordinator relation records |

The envelope map has exactly the requested keys. Every current identity is copied into
`input_data_identities`, sorted by `requirement_key`, and cross-checked against its
envelope. Relation metadata is canonicalized recursively as a BTreeMap before its
fingerprint is computed.

D4's cohort guarantee is deliberately narrow and safe: identities emitted in any
prediction family are unique and belong to the union of current coordinator relations.
`excluded: true` only excludes a relation record from training; it remains eligible for
replay prediction. Exact transitive row coverage in multi-source/missing-input cases is
deferred to D4.1 and must not be inferred from a D4 outcome.

## Outcome syntax and effects

`training_replay_outcome.schema.json` requires the following groups:

| Group | Keywords | Required effect |
|---|---|---|
| Identity | `schema_version`, `outcome_id`, `run_id`, `outcome_fingerprint` | Stable operation identity; the fingerprint is TCV1 with itself omitted. |
| Source authority | `source_training_outcome` | Complete `TrainingOutcomeRef`, including outcome, request, plan, bundle, binding, influence and data-identity fingerprints. |
| Request authority | `replay_request_id`, `replay_request_fingerprint`, `phase` | Must match the validated request exactly. |
| Current data | `input_data_identities` | Exact sorted identities derived from the current envelopes. |
| Plan/bundle | `bundle_id`, `plan_id` | Must match the source outcome and effective plan. |
| Counts | `result_count`, `lineage_record_count`, four block counts, `controller_count` | Must equal the actual arrays, blocks and unique controllers. |
| Result | `outputs`, `explanations`, `lineage` | Sorted, cross-linked, phase-valid portable results. |
| Runtime policy | `prediction_cache_store` | `false` in public forward replay V1; replay does not persist a new training cache. |
| Diagnostics | `warnings`, `diagnostics` | Sorted unique non-blank warnings and strict scalar portable diagnostics. |

`PREDICT` requires at least one selected V2 bound output and forbids explanations.
`EXPLAIN` requires at least one explanation; it may also carry selected V2 bound outputs
or use the explanation-only form with zero output block counts. No outcome may contain
runtime handles.

### Stateless portable package replay

`PortablePredictorPackage` replay uses the package's embedded outcome reference,
effective plan, bundle and output bindings as the frozen authority. The request's
`source_outcome_fingerprint` must target `package.training_outcome`, and every
requested binding must exist in `package.output_bindings`.

For `PREDICT`, emitted output bindings must exactly cover
`request.output_binding_ids`. For `EXPLAIN`, emitted outputs are optional but, when
present, must be a subset of the requested package bindings; at least one
explanation block is still required. The package remains process-handle-free:
hosts provide current `HandleRef` sidecars keyed by package artifact id, and
package replay must not emit new fitted artifacts.

### Output binding and aggregation

Each output is `BoundTrainingOutput` V2. Its embedded `OutputBinding` remains V1 and
continues to define:

- `binding_id`, producer `node_id` and selected `port_name`;
- `prediction_level` and relation `unit_level`;
- `prediction_kind`, target order/space, names, units and class vocabularies;
- `prediction_source`, `refit_strategy` and the effective aggregation fingerprint.

The binding fingerprint is recomputed and every emitted block must use the same node and
port. The aggregation fingerprint is the fingerprint of the effective aggregation
policy, not merely a user spelling of it. `partition` is `final` and `fold_id` is null
for public forward replay.

### Port-explicit V2 family

The port is required at every wire boundary that can carry a prediction or explanation:

| V2 root | Port-bearing content | V1 migration behavior |
|---|---|---|
| `node_result.v2` | prediction, observation, aggregate and explanation blocks | Missing port is accepted only by the legacy single-prediction-port adapter. |
| `bound_training_output.v2` | all prediction block families | Public replay always emits V2. |
| aggregation task/result V2 | input/output blocks | V2 controller dispatch is explicit. |
| process-adapter frame V2 | result-side `NodeResult` | Task payload remains `NodeTask` V1. |
| prediction-cache payload set V2 | cached blocks | Uses `dag-ml-json-prediction-blocks-v2`. |
| `score_set.v2` | each metric report | Uniqueness includes the port. |
| `execution_bundle.v2` | V2 caches and scores | All other bundle constraints remain V1-equivalent. |
| `training_outcome.v2` | V2 outputs/cache/bundle/scores | Migration creates a new signed outcome identity. |

V1 writers never emit `producer_port`. A V1 reader must reject a V2 document. A legacy
ingress wrapper may infer a missing port only when the plan exposes exactly one output
port of kind `prediction`; zero, multiple, unknown or non-prediction ports are errors.
There is no portable-package V2 in D4.

The V2 family is tested by dereferencing each schema and proving exact structural
equality with V1 after removing only version/port deltas, V2 references and the cache
marker. Cross-reader tests prove that every committed, non-empty positive V2 root is
rejected by its V1 schema while retained V1 positives still validate. One historical
schema-only exception is explicit: V1 `ScoreSet.schema_version` used `minimum: 1`, so an
empty V2 score set passes that JSON Schema. The V1 Rust reader still rejects it through
its exact version constant; version validators, not the permissive legacy schema, are
the reader authority.

### Score and cache identities

A V2 score report is unique by the complete coordinate:

```text
(producer_node, producer_port, variant_id, partition, fold_id, level)
```

Two reports differing only by `producer_port` are valid siblings. Repeating the full
coordinate is invalid. Selected scores cross-link to the selected variant's validation
aggregate report. A standalone V2 score set may have no reports, preserving Rust V1
semantics; a complete V2 training outcome must still contain the selected
validation/aggregate coordinate.

Prediction cache JSON fingerprints include `producer_port` immediately after
`producer_node` in each canonical block preimage. Existing V1 cache bytes and outcomes
are never rewritten in place. Migration changes the cache marker, content fingerprints,
bundle fingerprint and outcome fingerprint and assigns a new outcome ID. Arrow buffer
layout is unchanged.

## Classification semantics

For `class_probability`, every row concatenates one segment per target, in
binding target order. Segment width is the corresponding `class_labels` vocabulary
size. Each value is a finite binary64 number in `[0, 1]`. The validator sums a segment
sequentially in vocabulary order and accepts only:

```text
abs(sum - 1.0) <= 1e-12
```

The runtime must not renormalize or mutate values. For `class_label`, each
value is a finite binary64 number integral in value and is interpreted as a zero-based
index into that target's vocabulary. Thus `1.0` is valid when the vocabulary has at
least two labels; `1.5`, `-1.0` and an index equal to the vocabulary length are invalid.

## Explanation payloads

An explanation records `producer_node`, `producer_port`, non-blank `method`, optional
`target_name` and a strict JSON `payload`. When present, the target must be one of the
selected binding targets. Object keys are strings and all numbers are finite.

D4 allows controller-specific portable payloads such as `feature_names` and `values`.
It forbids runtime handles and recursively rejects the reserved raw-input keys
`raw_features`, `feature_matrix`, `raw_spectra` and `raw_wavelengths`. That blacklist is
not a semantic proof that arbitrary JSON under a different key contains no feature
data; the producing controller remains responsible for respecting the data-disclosure
boundary. The reserved names target raw input disclosure and do not turn an attribution
vector named `values` into raw feature data. D4 freezes no sample/unit/order attribution
semantics for the opaque payload. Structured attribution, row identity and cohort
completeness are W2-EXPLAIN E3-E6 work.

## Validation and conformance pack

Run from the `dag-ml` repository:

```bash
python scripts/validate_contracts.py
python scripts/validate_training_replay_contracts.py
pytest -q parity/training/tests
```

Regenerate deterministic fixtures and the pack with:

```bash
python parity/training/generate_training_replay_fixtures.py
```

`training_replay_contract_conformance_pack.v1.json` is separate from and contains the
intact 81-artifact `training_contract_conformance_pack.v1.json` authority. It pins that
base pack's ID, file SHA-256, internal checksum and exact artifact list, then adds the
exact transitive D4 closure. Paths are repository-confined regular files: absolute,
traversal and symlink paths are rejected. The pack also freezes the complete positive
and negative case ID sets.

The production validator and `training_replay_oracle.py` implement the semantics
independently and must agree on every positive and negative case. JSON Schema validation
is offline; unresolved or remote-only references fail the gate.

## Delivery roadmap and parallel implementation

The implementation can proceed in parallel only across lanes whose incoming contracts
are already frozen. Every lane lands with unit, negative, golden serialization and
cross-language parity tests, plus keyword/effect documentation in the owning surface.

The canonical non-parallel critical path is:

```text
(D4.0 + D5a-C) -> D5a-R -> D5 -> D4.1 -> D4.2 -> D8
```

This page does not introduce alternate gate IDs. It uses the IDs and dependencies from
the ecosystem implementation roadmap.

| Canonical gate | Lane | Deliverables and exit criteria |
|---|---|---|
| D4.0 + D5a-C / B0 (this change) | Single contract owner | Freeze training `ReplayRequest` / `ReplayOutcome`, port-explicit schemas/migrations, deterministic fixtures, independent validators, exact pack and CI. No runtime claim. |
| D5a-R | Port-provenance runtime | Add, normalize and propagate `producer_port` through blocks, scheduler, stores, OOF, aggregation, caches and parity while retaining the V1 mono-port reader. |
| D5 | Binding/scoring runtime | Close `OutputBinding` end to end and score/select strictly by `(node, port)`, including the adversarial better-score sibling-port case. |
| D4.1 | Attached core replay | Replace the replay summary with an attached real `ReplayOutcome`; implement `PREDICT`/`EXPLAIN`, current-cohort outputs and exact transitive row coverage across multi-source/missingness cases. |
| D4.2 | Owning bindings | Expose attached replay through C ABI and PyO3/Python with ownership/detach, C conformance, public Python smoke and identical errors. |
| D6 -> D7 (parallel after B0 on disjoint files) | Patch/alias lane | Implement `ParameterPatch` value namespaces, then reversible binding aliases. These IDs are not nirs4all replay fanout. |
| D9 preparation (parallel, disjoint) | Influence evidence | Prepare capability-driven runtime evidence; scheduler integration waits for the critical lane's ownership barrier. |
| D8 | Stateless package replay | After D4.2, D5, D6 and D9, activate export/load and stateless replay over the D2 package contract, including stale-artifact and host/native tests. |
| P1-P9 / PC1-PC10 fanout | nirs4all Python/product | After the required D8/runtime gates, add estimator client, syntax/compiler, conformal APIs, typed results, exact keyword/effect docs and cross-repo tests without duplicating DAG-ML logic. |
| B5 (strictly last) | nirs4all-ui then Studio | After APIs, keyword registry, fixtures and capability matrix freeze: shared view models/components, Studio jobs/views, accessibility and browser/Electron E2E. |

After B0, D5a-R remains sequential before D5 and D4.1. Only file-disjoint work such as
D6, W0.7 documentation/keyword-registry preparation and D9 preparation may proceed in
parallel. D4.2 cannot open before the reviewed D4.1 core. For these gates, split bounded
work among high-reasoning implementation agents—for example
Claude Opus or Fable at `xhigh`/`max` effort, and Ultracode for contract-heavy code—so
the permitted file-disjoint lanes can advance in parallel. Each agent receives one
contract-owned lane and its exact gates. An independent
review pass (Codex or a designated maintainer) must inspect every generated patch,
compare it against this conformance pack, run the owning repository's full test suite and
reject cross-layer reimplementation. Studio work begins only after stable Python/binding
APIs exist, because its graphical semantics depend on those final error and result types.

## Deferred and deliberately unsupported in D4

- public Rust, C, Python or WASM replay runtime claims before D4.1/D4.2;
- `REFIT` replay or automatic fine-tuning mutation;
- a V2 portable predictor package or `.n4a` package migration;
- exact current-cohort row coverage in multi-source/missingness cases;
- structured explanation attribution and unit/order guarantees;
- changes to Arrow buffer layout.
