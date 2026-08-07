# Native Training and Portable Predictor Contracts

Status: W1-0 contract freeze, schema version 1; D3 native C and PyO3
execution bindings.

This page defines the native DAG-ML boundary for training requests, tuning
parameter projection, training-data identity, capability-derived influence,
candidate cache isolation and portable fitted predictors. The contracts are
implemented and validated in `dag-ml-core`; controller execution remains behind
the existing scheduler/controller interfaces.

The lifecycle is:

```text
TrainingRequest
  -> validate graph/campaign/controllers/data identities
  -> resolve requested prediction ports
  -> build a separate immutable typed ParameterProjection
  -> derive predictor closure and mandatory influence slots
  -> normal ExecutionPlan (NodePlan still carries only params)
  -> FIT_CV / SELECT / optional REFIT
  -> TrainingOutcome reference + PortablePredictorPackage
```

No contract contains a Python estimator, process pointer, Rust handle or other
process-local object. Host objects are resolved separately when an artifact is
explicitly marked `host_sidecar`.

## TrainingRequest

Schema: `contracts/training_request.schema.json`

Positive examples:

- `../examples/fixtures/training/training_request_refit.v1.json`;
- `../examples/fixtures/training/training_request_no_refit.v1.json`;
- `../examples/fixtures/training/training_request_active_influence.v1.json`;
- `../examples/fixtures/training/training_request_package_refit.v1.json`, the
  signed request cross-linked by the portable package/outcome fixtures.

The top-level keys are all required. Unknown keys fail closed.

| Key | Meaning and effect |
|---|---|
| `schema_version` | Must be `1`; future versions are not guessed. |
| `request_id` | Stable request identifier used in logs and projections. |
| `plan_id` | Identifier assigned to the normal `ExecutionPlan`. |
| `graph` | Complete `GraphSpec`; it is validated before training projection. |
| `campaign` | Folds, root seed, generation, shape and external-data bindings. |
| `controller_manifests` | Canonically sorted controller registry used to resolve every node. |
| `data_identities` | Exact content identity for every external campaign binding. |
| `parameter_patches` | Typed, namespaced candidate or fine-tuning changes. |
| `patch_policies` | Per-node authorization for the namespaces above. |
| `influence_requirements` | Explicit samples used by active early stopping, weighting or internal tuning. |
| `options` | Refit, selection, outputs, scheduler, resources and artifact policy. |
| `request_fingerprint` | TCV1 SHA-256 of the complete request with only this field omitted. |

The strict JSON reader rejects duplicate object members, non-finite numbers and
keys that collide after Unicode NFC normalization. It fingerprints the original
JSON before deserialization and compares it with the canonical Rust
re-serialization. Arrays backed by Rust `BTreeSet` therefore have a canonical
wire order; changing only the order of `capabilities`, `supported_phases` or
`allowed_namespaces` is not accepted as a second encoding of the same request.
Fold sample arrays are different: their order is semantically neutral and is
not required to be lexicographic.

### `options`

| Key/value | Effect |
|---|---|
| `refit: true` | Runs the selected strategy through `REFIT`; `refit_strategy` is required. |
| `refit: false` | Forbids `refit_strategy`; prediction caches must be retained so replay remains possible. |
| `refit_strategy: refit_one` | Refit one selected candidate. |
| `refit_strategy: refit_ensemble` | Refit the selected ensemble policy. |
| `seed` | Must equal `campaign.root_seed` exactly. |
| `selection` | Existing `SelectionPolicy`; its metric and objective control `SELECT`. |
| `selection_output_id` | Required `output_id` whose producer-specific averaged `FIT_CV` reports alone drive ranking. |
| `outputs` | Ordered output requests; producer/port coordinates must be unique. |
| `scheduler` | `sequential` or explicit parallel `threads`/`processes`. |
| `resources` | CPU, optional memory/GPU/wall-time limits; workers cannot exceed CPU threads. |
| `artifacts` | CV retention, prediction-cache retention and fitted-artifact portability. |

A sequential scheduler requires `backend: null` and `workers: 1`. Parallel
threads require every controller in the predictor closure to be `thread_safe`
and reject `needs_python_gil`. Parallel processes require `process_safe`.

`selection_output_id` must resolve exactly one member of `outputs`. Its producer
must support `FIT_CV`. A producer may expose multiple prediction ports only when
the output request resolves an explicit prediction port and all score reports
used for SELECT carry the matching `producer_port`. Its `prediction_level` must always equal
`campaign.aggregation_policy.selection_metric_level`; an optional
`selection.required_metric_level` must equal the same value. V1 ranking accepts
only this explicit matrix:

| Prediction kind | Metric names | Required objective |
|---|---|---|
| `regression_point` | `mse`, `rmse`, `mae` | `minimize` |
| `regression_point` | `r2` | `maximize` |
| `class_label` | `accuracy`, `balanced_accuracy` | `maximize` |
| `class_probability`, `decision_score` | none in V1 | Refused until an explicit probability/score metric is implemented. |

Report ordering never chooses the winner. The outcome reconstructs the complete
ranking from exactly one `fold_id: "avg"` validation report per plan variant for
the selected producer, then verifies candidate order, ranks, selected score,
metric, objective, level and `oof` evaluation scope.

### Output syntax

Each output has these important fields:

```json
{
  "output_id": "output:prediction",
  "node_id": "model:base",
  "prediction_level": "sample",
  "unit_level": "physical_sample",
  "prediction_kind": "regression_point",
  "target_names": ["protein"],
  "target_units": ["percent"],
  "class_labels": [[]],
  "output_order": "target_order",
  "target_space": "raw"
}
```

`port_name` may be omitted only when the node exposes exactly one prediction
output. Zero prediction ports, multiple prediction ports, a non-prediction port
or an absent explicit port is an error. `target_names` and each target's
`class_labels` preserve semantic order; they are unique but are not sorted
lexicographically. Reordering them changes the fingerprint and the meaning.

Prediction kinds are `regression_point`, `class_label`, `class_probability` and
`decision_score`. Class probabilities require
`target_major_class_minor`; other V1 outputs require `target_order`. Regression
keeps one empty class-label array per target.
`class_probability` requires a non-empty vocabulary for every target;
`class_label` and `decision_score` preserve the W0 rule that the vocabulary may
be empty when the backend does not publish it.

Sample-level outputs require `unit_level: "physical_sample"`. Target- and
group-level outputs require `unit_level: null`; these combinations are part of
the fingerprinted contract, not display metadata.

## Fine-tuning and parameter projection

`kind: "finetune"` remains a Pipeline DSL alias for a `tuner` node. A tuner or
host fine-tuning controller performs its search behind the normal controller
boundary. The selected values cross the native boundary as `ParameterPatch`
records; DAG-ML does not mutate an opaque estimator.

The public namespace-to-plan-root mapping is frozen:

| Wire namespace | Projection root | Typical effect |
|---|---|---|
| `operator` | `params` | Declares a constructor/model/preprocessing parameter. |
| `fit` | `fit_params` | Declares a fit-call-only option such as epochs. |
| `control` | `control_params` | Declares controller behavior that does not change graph shape. |
| `structural` | `structural_params` | Declares a shape/topology change and sets `requires_recompile`. |

Example:

```json
{
  "schema_version": 1,
  "node_id": "model:base",
  "namespace": "operator",
  "path": ["regularization", "alpha"],
  "value": 0.1
}
```

Paths are object-key arrays, not JSON Pointer. They are non-empty; blank and
`-` append segments are invalid. Intermediate keys must already exist and be
objects. Only the final key may be new, and arrays are never traversed in V1.
Patches are strictly sorted by `(node_id, namespace enum order, path)`. Duplicate
or parent/child paths are rejected. Policies are sorted by node and must cover
exactly the patched nodes; a listed namespace is authorization, not merely
documentation.

`ParameterProjection` is a deep clone. It never mutates the source plan. Its
`structural_patch_count` is the number of structural patches and
`requires_recompile` is exactly `structural_patch_count > 0`.

These four maps exist on `ParameterProjection.nodes[*]`. In W1-0,
`ExecutionPlan::NodePlan` still has only its historical `params` map. Validation
and projection therefore do not imply that `fit`, `control` or `structural`
values are already consumed by a controller. The training runtime must
explicitly materialize each supported namespace (and recompile when required)
before execution; silently dropping or applying one is forbidden.

## Native runtime V1 acceptance matrix

Projection describes the complete portable contract. The first native runtime
slice deliberately executes a narrower subset and fails closed outside it; it
never treats an accepted projection as proof that every future option is
implemented.

| Request/runtime input | Native V1 effect |
|---|---|
| `parameter_patches: []` | Required. Any patch is refused until its namespace is materialized by the runtime. |
| `resources.cpu_threads == scheduler.workers` | Required together with `memory_bytes: null`, `gpu_devices: []` and `wall_time_ms: null`. |
| `artifacts.cv_artifacts: discard` | Required; retained fold-fitted artifacts are not implemented. |
| `artifacts.fitted_artifacts: allow_host_sidecar` | Required; V1 cannot prove a controller payload is natively portable. Sidecar handles stay process-local. |
| `artifacts.prediction_caches: discard` | Allowed only when no graph edge requires OOF predictions. Stacking graphs must retain caches. |
| Empty input artifact store | Required so the outcome cannot accidentally claim artifacts from an earlier run. |
| Predictor closure equals all executable plan nodes | Required until unrelated-node execution and persistence have an explicit policy. |
| Exactly one prediction port on every relevant producer | Required because runtime prediction blocks do not yet carry `port_name`. |
| Variant parameter overrides | Require a predeclared `hpo_selection` influence entry. |
| Provider identity and relations | Exact content attestation is mandatory before any controller executes. |
| Diagnostics metadata | Keys named `handle`, ending in `_handle`, or ending in `_handles` are refused recursively. |

Scheduler capability checks remain those of the request contract: threads need
thread-safe controllers and cannot run `needs_python_gil`; processes need
process-safe controllers. A rejected row above is a V1 capability boundary,
not an invitation for a binding to discard or rewrite the option.

## Data identity and candidate caches

A `TrainingDataIdentity` binds one `node_id.input_name` requirement to six
identities:

- schema fingerprint;
- data-plan fingerprint;
- sample-relation fingerprint;
- feature/data content fingerprint;
- target content fingerprint;
- TCV1 identity fingerprint over all preceding fields and the requirement key.

The list is sorted by requirement key and exactly covers campaign data bindings.
Schema, plan and relation fingerprints must equal the corresponding campaign
binding, so changing a dataset while reusing a requirement name cannot hit the
same cache.

The native training operation also asks `RuntimeDataProvider` to attest this
identity for every binding before controller execution. A missing attestation,
or any mismatch in schema, plan, relation, feature content, target content or
the derived identity fingerprint, fails closed. The additive provider method
may still return no attestation for legacy phase execution; only
`execute_training` makes it mandatory.

This is an explicit trust boundary. DAG-ML compares the provider's attestation
with the signed request; it does not independently hash raw feature or target
buffers hidden behind the provider interface. Offline validation of a
`TrainingOutcome` or `PortablePredictorPackage` has no dataset access: it proves
the internal cross-links and commitments, but cannot prove that external bytes
produce the claimed content hashes. A provider or ingestion layer must compute
and attest those hashes when training or replaying against actual data.

`CacheNamespace` V1 is deliberately `FIT_CV`-only and fold-scoped. Its identity
contains producer node/source port, consumer node/target port, complete data
identity fingerprint, effective parameter fingerprint, fold, trial and seed.
The prediction key is reconstructed as:

```text
producer.source->consumer.target
```

Changing any coordinate produces a different namespace fingerprint. `REFIT`
does not silently reuse this namespace.

Runtime prediction-cache stores must keep this proof attached after replay
materialization. `PredictionCacheMaterializationRecord` therefore mirrors the
validated `cache_namespace_fingerprints` from the bundle/cache record while the
runtime handle is also derived from the same namespace fingerprints, cache
content and replay request. A store may omit the field only for legacy payloads
where the bundle cache record also has no namespace fingerprints; it must not
silently drop fingerprints from a D10-enriched cache.

Bundle and payload validation also bind the shape of that proof before replay:
when `cache_namespace_fingerprints` is non-empty, its length must match the
cache `block_count`, and the `ExecutionBundle` must carry a concrete
`selected_variant_id`. This keeps D10-enriched caches from entering replay as
anonymous candidate state.

Every materialized record is also validated against the exact
`PredictionCacheMaterializationRequest` that produced it: run, bundle, replay
phase, variant, requirement key, cache id, namespace fingerprints and handle
owner must match. This keeps the audit trail candidate/dataset/fold-specific
even when a store implementation uses an opaque runtime handle internally. A
D10-enriched cache, identified by non-empty namespace fingerprints, must be
materialized with a concrete `variant_id`; legacy namespace-less payloads are
the only variant-less compatibility case.

Persistent stores carry the same invariant before materialization. The
file-backed store includes `cache_namespace_fingerprints` in both
`prediction_cache_manifest.json` entries and the deterministic
`prediction-cache-*.json` payload filename fingerprint. Two payloads with the
same content but different candidate/data/fold namespaces therefore get
distinct files. The columnar in-process store exposes the fingerprints through
`ColumnarPredictionCacheManifest`, so a manifest-only audit can verify the
namespace without reloading the full payload.

## Training influence

Controller capabilities declare behavior that can change the fitted result.
They are active claims, not generic backend support flags.

| Active capability | Required influence kind | Sample rule |
|---|---|---|
| `uses_early_stopping` | `early_stopping` | Non-empty strict subset of the matching training cohort. |
| `uses_training_weights` | `weighting_resampling` | Exactly the complete matching training cohort. |
| `performs_internal_tuning` | `hpo_selection` | Explicit subset inside the matching training cohort, except when the tuner's base kind already captures HPO selection. |
| `trains_aggregation` | `trained_meta_aggregation` base influence | Requires `aggregates_predictions`. |

`uses_training_weights` additionally requires one of
`supports_sample_weights`, `supports_row_resampling` or
`supports_backend_loss_weights`. No active training capability is allowed on a
`stateless` or `inference_only` controller.

For a `fold_train` controller, every active capability creates one mandatory
slot for every `FIT_CV` fold. When refit is enabled and the controller supports
`REFIT`, it creates one more full-training slot. `fold_id` is required for
`FIT_CV` and must be `null` for `REFIT`. An outer-validation sample in a fold
slot is an explicit leakage error. Missing, duplicate, extra or inactive
capability slots are errors.

The final `TrainingInfluenceManifest` records physical samples and the exact
origin/group closure. Multiple base fitting entries per node are valid—for
example fold 0, fold 1 and refit—but every entry for that node must use its one
derived base kind. Scope identifiers are caller-defined identifiers; DAG-ML
does not infer semantics from their spelling.

## TrainingOutcome commitments

Schema: `contracts/training_outcome.schema.json`

A native outcome carries the exact `training_request_fingerprint`, the complete
sorted `data_identities` attested before execution, and the
`selection_output_id` actually ranked. Its own `outcome_fingerprint` commits all
of those fields together with the effective plan, scores, decision, lineage,
influence, artifacts, caches and execution bundle. Standalone validation
reconstructs SELECT numerically as described above; a re-fingerprinted decision
with a forged selected score is still rejected.

`TrainingOutcomeRef` is compact but not id-only. Besides the outcome and request
fingerprints it carries `effective_plan_fingerprint`,
`execution_bundle_fingerprint`, ordered `output_binding_fingerprints`,
`training_influence_fingerprint` and `data_identities_fingerprint`. Therefore a
package cannot swap bundle content or feature/target content while retaining a
valid reference merely by preserving an id.

### Replayable phases

`replayable_phases` is the honest, self-describing answer to "which inference or
re-training entry points can this outcome actually drive?" It is a `uniqueItems`
array over `REFIT`, `PREDICT`, `EXPLAIN`, always written in the canonical order
`[REFIT, PREDICT, EXPLAIN]`. Standalone validation re-derives the vector and
requires exact order **and** set equality — it is never trusted as written and
is never inferred from the `refit` flag alone.

The derivation reads only portable outcome state and rests on three definitions:

- **Effective predictor closure.** Every node reachable upstream from the bound
  output `node_id`s along the *real* graph edges (`node_plans[*].input_nodes`
  rebuilt from `graph.edges`). In V1 this closure must equal the complete set of
  effective plan nodes; a closure that omits an executable node is refused.
- **Retained inference state.** A closure node requires reloadable fitted state
  **exactly when its controller capabilities are `stateful` OR `emits_artifacts`**.
  This is the only rule. The `artifact_policy` value — including
  `replay_required` — and `fit_scope` are deliberately **not** consulted: a
  stateless deterministic operator (for example a `replay_required` transform
  that simply recomputes at inference, or a seeded fold-train augmentation)
  carries no reloadable state, needs no retained artifact, and must not block
  forward replay. A state-retaining node is "satisfied" only when the execution
  bundle holds a retained refit artifact for it.
- **OOF self-containment.** For each `requires_oof` edge whose *source and target
  are both in the closure*, the outcome must carry the exact triple: a bundle
  `prediction_requirement`, a retained `prediction_cache` record, and a portable
  `prediction_cache` payload — all keyed by
  `producer.source_port->consumer.target_port`. Missing any one leg makes that
  edge not self-contained.

Given those, the derivation is:

| Refit status | Emits | Condition |
|---|---|---|
| `completed` | `PREDICT` | every closure node supports `PREDICT` **and** every state-retaining closure node has a retained refit artifact. |
| `completed` | `EXPLAIN` | same rule, for `EXPLAIN`. |
| `completed` | *(never)* `REFIT` | a completed refit never re-advertises `REFIT`. |
| `skipped` | `REFIT` | every closure node supports `REFIT` **and** every in-closure `requires_oof` edge has the full requirement + cache record + portable payload triple. |
| `skipped` | *(never)* `PREDICT`/`EXPLAIN` | a skipped refit never advertises forward inference. |

The empty array `[]` is a valid, first-class answer and is strictly preferable to
advertising a capability the retained state cannot honor. For example: a
completed refit whose closure supports `PREDICT` but is missing one
artifact-emitting node's retained artifact derives `[]`, not `[PREDICT]`. The two
source fixtures pin the canonical happy paths — the refit outcome derives
`["PREDICT"]` and the no-refit outcome derives `["REFIT"]`:

```json
{ "refit": { "requested": true,  "status": "completed", "strategy": "refit_one" },
  "replayable_phases": ["PREDICT"] }

{ "refit": { "requested": false, "status": "skipped",   "strategy": null },
  "replayable_phases": ["REFIT"] }
```

`PREDICT` replay never consumes OOF payloads, so the OOF triple is irrelevant to
the completed-refit forward rules; conversely `REFIT` replay re-fits from data
and OOF caches, so it ignores retained refit artifacts. A closure with no
`requires_oof` edges is vacuously OOF-self-contained, so a no-refit outcome whose
closure fully supports `REFIT` derives `["REFIT"]` with no caches required.

## PortablePredictorPackage

Schema: `contracts/portable_predictor_package.schema.json`

The package cross-links:

- predictor template and effective plan;
- training request and outcome reference;
- effective-plan TCV1 fingerprint and execution bundle id;
- complete execution-bundle TCV1 fingerprint (the id alone is insufficient);
- ordered output-binding fingerprints;
- training-influence fingerprint and predictor closure;
- complete data identities, their ordered-array TCV1 fingerprint and bundle requirements;
- every refit artifact and its explicit load mode.

A portable package is a deployable predictor, so it **independently requires
`PREDICT`-replayability**: its own plan, closure and retained artifacts must
satisfy the completed-refit `PREDICT` rule above (every closure node supports
`PREDICT` and every state-retaining closure node has a retained refit artifact),
proven directly from the package rather than inferred from a non-empty
`replayable_phases` claim in the referenced outcome. A package whose closure
cannot honestly replay `PREDICT` is refused even if the outcome it references is
itself valid. This `PREDICT` guard consults no OOF caches.

`fitted_artifact_mode: portable_required` permits only `native_portable` artifact
bindings. A native artifact needs a backend, content fingerprint and safe
relative URI, and cannot come from a `host_only` controller. With
`allow_host_sidecar`, selected artifacts may be `host_sidecar`; those handles are
resolved into a process-local `LoadedPredictor` and are never serialized back
into the package. Each refit-artifact record's `controller_id`, nested
`artifact.controller_id` and `params_fingerprint` must match the owning
`NodePlan`, so a re-signed package cannot forge artifact provenance. Keys named
`handle`, ending in `_handle`, or ending in `_handles` are refused recursively.

## CLI

```bash
cargo run -p dag-ml-cli -- validate-training-request \
  examples/fixtures/training/training_request_active_influence.v1.json

cargo run -p dag-ml-cli -- project-training-request \
  examples/fixtures/training/training_request_refit.v1.json \
  --output /tmp/training-projection.json

cargo run -p dag-ml-cli -- validate-portable-predictor-package \
  examples/fixtures/training/portable_predictor_package.v1.json

cargo run -p dag-ml-cli -- validate-cache-namespace \
  examples/fixtures/training/cache_namespace_fit_cv.v1.json
```

The commands read the original JSON text, so duplicate/NFC-colliding members
cannot be hidden by a permissive intermediate parser. The generated
`TrainingContractProjection` reader also fails closed on unknown fields at any
nested path (including embedded historical `GraphSpec` and `CampaignSpec`
values), even though the projection itself is not self-fingerprinted.

## C training binding

Header: [`crates/dag-ml-capi/include/dag_ml.h`](../crates/dag-ml-capi/include/dag_ml.h)

The C binding executes the same native operation as the Rust API:
`COMPILE/PLAN -> FIT_CV -> SELECT -> optional REFIT`. It does not add a second
scheduler or scoring implementation. The entry point is:

```c
DagMlStatusCode dagml_training_execute(
    const DagMlTrainingExecuteRequest *request,
    DagMlTrainingResult **out_result,
    DagMlString *error_out);
```

`DagMlTrainingExecuteRequest` is a synchronous call descriptor. Its byte views
need remain valid only until `dagml_training_execute` returns; transferred or
borrowed `user_data` has the longer lifetime described below.

| Field | Required value and effect |
|---|---|
| `request_json` | Required UTF-8, self-fingerprinted `TrainingRequest`. The raw TCV1 fingerprint is checked before typed decoding. |
| `outcome_id` | Required UTF-8 portable identifier copied into the outcome. |
| `run_id` | Required UTF-8 `RunId`; it scopes tasks, lineage and deterministic execution. |
| `bundle_id` | Required UTF-8 `BundleId` assigned to the generated execution bundle and retained cache payloads. |
| `relations_json` | Required strict `SampleRelationSet` JSON. Its fingerprint must agree with the influence manifest, every signed data identity and every envelope relation set. |
| `influence_json` | Required strict `TrainingInfluenceManifest` JSON, including its valid TCV1 `manifest_fingerprint`. |
| `envelopes_json` | Required strict object mapping every rendered data-requirement key to one `ExternalDataPlanEnvelope`; exact coverage is mandatory. |
| `warnings_json` | Optional UTF-8 JSON string array. A null pointer **or** zero length means `[]`. A non-empty view must contain the array, even when it is empty. |
| `diagnostics_json` | Optional UTF-8 JSON object. A null pointer **or** zero length means `{}`. Runtime-handle-shaped keys are refused recursively. |
| `dataset` | Host dataset handle passed unchanged to `data_provider.materialize`. DAG-ML does not inspect it. |
| `data_provider` | Borrowed `DagMlDataVTable`, ABI version at least `DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION`, with `materialize` and `make_view`. |
| `data_owner_controller_id` | Required UTF-8 `ControllerId` recorded as owner of the returned data/data-view handles. |
| `controller_bindings` | Array of `(controller_id, DagMlControllerVTable)` bindings. Every controller referenced by the effective plan must be registered exactly once. |
| `controller_binding_count` | Number of readable entries. A null pointer is valid only when the count is zero. |

All required byte views must have a non-null pointer and valid UTF-8. JSON
payloads are parsed through the strict typed boundary before any data or
controller callback runs: duplicate keys, NFC-colliding keys, malformed JSON,
invalid fingerprints, envelope-key collisions and incomplete envelope coverage
fail during preflight. Unknown fields are governed by the corresponding Rust
contract and published schema.

### C envelope and callback shape

The envelope key is rendered exactly as `node_id + "." + input_name`; it is not
JSON Pointer syntax and neither component is escaped. For example:

```json
{
  "model:base.x": {
    "schema_version": 1,
    "schema_fingerprint": "<64 lowercase hex>",
    "plan_fingerprint": "<64 lowercase hex>",
    "relation_fingerprint": "<64 lowercase hex>",
    "data_content_fingerprint": "<64 lowercase hex>",
    "target_content_fingerprint": "<64 lowercase hex>",
    "coordinator_relations": { "records": [] }
  }
}
```

The map's keys must equal the key set derived from
`request.campaign.data_bindings`: no missing and no extra member is accepted.
Two distinct `(node_id, input_name)` coordinates that render the same V1 string
are refused rather than merged; for example, `(a.b, c)` and `(a, b.c)` would
both render `a.b.c`. The provider registers the complete signed binding
field-for-field, while the envelope's schema, plan and relation
fingerprints must match that binding. All three relation/data/target content
fingerprints are mandatory for training, even though legacy or prediction-only
envelopes may omit the latter two. When `require_relations` is true,
`coordinator_relations` is also mandatory and must fingerprint to
`relations_json`.

The data vtable materializes opaque host data and view handles. The controller
vtable's `invoke` receives canonical [`NodeTask`](contracts/node_task.schema.json)
JSON and returns host-allocated
[`NodeResult`](contracts/node_result.schema.json) JSON. `invoke` and
`release_bytes` are mandatory. Native training additionally requires `release`
on every controller binding during preflight, before any callback can create an
opaque output or fitted-artifact handle; this does not change the legacy generic
replay ABI, where a missing releaser still denotes borrowed handles. The core validates prediction partitions, sample/unit
ids, targets, lineage, artifacts, OOF boundaries and scores after decoding the
result. Callback responses cross the same strict object-only boundary as public
inputs: duplicate or NFC-colliding keys, fields closed by the published schema,
and serde's internal positional-array representation of structs are refused.
A non-zero callback status aborts the operation as a runtime validation error.
Custom trained aggregation reuses the same `invoke` slot with an
`AggregationControllerTask` request and an `AggregationControllerResult`
response; a host that enables it must dispatch the two JSON task shapes.

### C ownership transaction

The controller slice is an ownership-transfer boundary, not ordinary borrowed
input. Bindings must follow the public ABI rule below regardless of the host
language's internal wrapper design:

| Resource | Lifetime rule |
|---|---|
| Data-provider `user_data` and `dataset` | Always borrowed. Keep them valid for the complete synchronous call. DAG-ML releases materialized data/view handles through `data_provider.release` when it is provided, but never calls `data_provider.destroy`; an allocating host must supply `release`. |
| Controller ABI v2 `user_data` | Borrowed. On success it must remain valid until `dagml_training_result_free`, because tracked result handles are released then. On failure it need remain valid only until the call returns. The native training entry point requires a non-null `release` callback on every v2/v3 controller binding. |
| Controller ABI v3 with non-null `destroy` | Owning. After non-null `request` and `out_result` have been accepted, once the pointer/count pair describes a readable slice, the call consumes every distinct owning `user_data`. On success ownership moves into the result; on any later error it is destroyed before return. The caller must not destroy it in either case. |
| Invalid required pointer | A null `request` or `out_result` is rejected before the binding slice is inspected, so the caller retains every controller owner. |
| Unreadable controller slice | When `controller_bindings == NULL` and `controller_binding_count > 0`, no transfer can occur and the caller retains every intended binding. |

Any repeated `user_data` address involving at least one owning binding is
refused, including borrowed/owned aliases: separate controller wrappers could
otherwise use the address after one wrapper destroys it or double-destroy it.
Once the readable-slice boundary has been crossed, that refusal still consumes
and destroys each distinct owner exactly once. On result destruction,
controller-returned handles are released before owned controller `user_data` is
destroyed. A host callback must never unwind or throw across the C boundary.

### C result, errors and frees

On entry, `*out_result` is cleared. `DAG_ML_STATUS_OK` returns one opaque,
owning `DagMlTrainingResult`; every error leaves it null. Read the portable
outcome with:

```c
DagMlStatusCode dagml_training_result_outcome_json(
    const DagMlTrainingResult *result,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
```

Each successful getter call allocates fresh, independent bytes. Those bytes are
not required to be NUL-terminated and must be released exactly once with
`dagml_owned_bytes_free`. An error string is a Rust allocation with an explicit
length; release it exactly once with `dagml_string_free`. The thread-local
`dagml_last_error_json`/`dagml_last_error_code` pair may provide the structured
ADR-11 descriptor, but only the function status determines success.
Zero-initialize `DagMlString` and `DagMlOwnedBytes` outputs before use. A null
`error_out` simply suppresses the immediate message; a null result or output
byte pointer is `DAG_ML_STATUS_INVALID_ARGUMENT`.

```c
DagMlTrainingResult *result = NULL;
DagMlString error = {0};
DagMlStatusCode status = dagml_training_execute(&request, &result, &error);
if (status != DAG_ML_STATUS_OK) {
    if (error.ptr != NULL) {
        fprintf(stderr, "%.*s\n", (int)error.len, error.ptr);
        dagml_string_free(error);
    }
    return status;
}

DagMlOwnedBytes outcome = {0};
status = dagml_training_result_outcome_json(result, &outcome, &error);
if (status == DAG_ML_STATUS_OK) {
    consume_json(outcome.ptr, outcome.len);
    dagml_owned_bytes_free(outcome);
} else if (error.ptr != NULL) {
    fprintf(stderr, "%.*s\n", (int)error.len, error.ptr);
    dagml_string_free(error);
}

dagml_training_result_free(result);
return status;
```

`dagml_training_result_free(NULL)` is a no-op. A non-null result, an owned byte
buffer and an error string must each be freed **at most once**; double-freeing
any of them is undefined behavior. The result is not itself a portable file:
serialize its outcome first, then free it.

## Python and PyO3 training binding

The Python facade delegates validation, planning, scheduling, native scoring,
selection and outcome construction to `dag-ml-core`; it does not reproduce that
logic in Python. Contract-only use remains unchanged:

```python
import dag_ml

request = dag_ml.TrainingRequest.from_path(
    "examples/fixtures/training/training_request_active_influence.v1.json"
)
projection = request.project()

same_projection = dag_ml.project_training_request(request)
package = dag_ml.PortablePredictorPackage.from_path(
    "examples/fixtures/training/portable_predictor_package.v1.json"
)
namespace = dag_ml.CacheNamespace.from_path(
    "examples/fixtures/training/cache_namespace_fit_cv.v1.json"
)
```

### Python entry points and keywords

`execute_training` is the typed facade:

```python
result = dag_ml.execute_training(
    request,
    data_envelopes,
    relations,
    training_influence,
    op_callback,
    outcome_id="outcome:experiment.001",
    run_id="run:experiment.001",
    bundle_id="bundle:experiment.001",
    warnings=[],
    diagnostics={"host": "python"},
)
```

| Argument | Accepted syntax and effect |
|---|---|
| `request` | `TrainingRequest`, path-like object, UTF-8 bytes/string or JSON-compatible object. It must already carry a valid `request_fingerprint`; the binding never silently re-signs it. |
| `data_envelopes` | Exact `requirement_key -> envelope` mapping described above, or its JSON/path/wrapper equivalent. |
| `relations` | Complete `SampleRelationSet` used for leakage and influence identity. |
| `training_influence` | Complete, self-fingerprinted `TrainingInfluenceManifest`. |
| `op_callback` | Callable receiving a Python dictionary structurally identical to a `NodeTask` and returning a dictionary deserializable as the matching `NodeResult`. One callable services all controller ids, so dispatch on `task["node_plan"]["controller_id"]` or `task["node_plan"]["node_id"]`. |
| `outcome_id` | Keyword-only portable outcome identifier. |
| `run_id` | Keyword-only run identifier propagated into every task and lineage record. |
| `bundle_id` | Keyword-only execution-bundle/cache namespace identifier. |
| `warnings` | Optional strictly sorted, unique sequence of non-empty warning strings; default is empty. |
| `diagnostics` | Optional JSON object copied into the outcome after recursive runtime-handle refusal; `None` becomes `{}`. |

Every contract-like argument is passed through the facade's `_coerce_json`
rules: validated wrappers preserve their text; paths are read as UTF-8; bytes
are decoded; strings are treated as JSON text; other objects are compact JSON
serialized. Consequently a filename supplied as plain `str` is **not** opened;
use `pathlib.Path` or a `.from_path(...)` wrapper. The lower-level
`execute_training_json` has the same semantics but
requires its first four inputs and optional warnings/diagnostics to already be
JSON strings; its three identifiers follow the callback as ordinary
positional-or-keyword parameters:

```python
result = dag_ml.execute_training_json(
    request_json,
    data_envelopes_json,
    relations_json,
    training_influence_json,
    op_callback,
    "outcome:experiment.001",
    "run:experiment.001",
    "bundle:experiment.001",
    "[]",
    '{"host":"python"}',
)
```

The callback receives a real dictionary, not a JSON string, and must return a
JSON-compatible Python object, not encoded JSON text. Returned contract structs
must be dictionaries at every nesting level; lists that only happen to match
serde's internal positional field order and unknown schema-closed keys are
refused. All controller ids in the
projected plan are registered against that same callback. `input_handles` are
process-local tokens; the binding does not expose or copy a feature matrix.
Host code normally closes over its dataset/model registry and uses the exact
`data_views` selectors in the task to obtain fold-train, fold-validation or
full-train rows. The selected data must be the content attested by the supplied
envelopes; a handle is not permission to substitute another dataset.
A Python exception raised by the callback aborts training and is surfaced as a
native `DagMlError` carrying its message; callers must not depend on retaining
the original exception class or traceback.

Custom trained aggregation uses the same callback slot. Such a call carries an
[`AggregationControllerTask`](contracts/aggregation_controller_task.schema.json)
and expects an
[`AggregationControllerResult`](contracts/aggregation_controller_result.schema.json),
so a host supporting custom aggregation must dispatch that shape separately
from ordinary `NodeTask -> NodeResult` calls.

### Threads and the Python GIL

The binding releases the calling thread's GIL around the complete native
training operation. Each operator or aggregation callback re-acquires the GIL
for Python-object conversion and invocation, then returns control to the native
scheduler. This has three practical effects:

- `sequential` requires `backend: null`, `workers: 1` and
  `resources.cpu_threads: 1` in native V1;
- `parallel` with `backend: "threads"` requires `workers >= 2`,
  `resources.cpu_threads == workers`, every predictor-closure controller to
  declare `thread_safe`, and no such controller to declare `needs_python_gil`;
- pure Python callback bodies remain serialized by the CPython GIL. Parallel
  speedup is available only where invoked native libraries release the GIL or
  work proceeds outside Python. The `thread_safe` capability remains a host
  promise and must cover callback-owned model/data registries as well.

The D3 native training operation rejects `backend: "processes"`; declaring
`process_safe` does not bypass that runtime limit. It also rejects memory, GPU
and wall-time resource limits in this first slice, as described in the
[native runtime acceptance matrix](#native-runtime-v1-acceptance-matrix).

### `TrainingResult`

Python receives an owning `TrainingResult`, not a bare dictionary:

| Member | Value |
|---|---|
| `is_attached` | Whether callback/controller objects, data-provider records and the artifact-handle store are still retained. |
| `process_local_artifact_count` | Number of retained refit artifact handles, or `None` after detach. |
| `process_local_data_handle_count` | Number of materialized data-handle records, or `None` after detach. |
| `process_local_data_view_count` | Number of materialized data-view records, or `None` after detach. |
| `outcome_fingerprint` | Stable fingerprint of the complete portable outcome. |
| `outcome` / `outcome_json()` | Validated `TrainingOutcome` wrapper / complete JSON. |
| `execution_bundle` / `execution_bundle_json()` | Validated portable execution bundle. |
| `score_set` / `score_set_json()` | Native score reports and selection evidence. |
| `outputs` / `outputs_json()` | Bound portable prediction outputs. |
| `artifacts` / `artifacts_json()` | Portable artifact **records only**, never Python objects or opaque handles. |
| `portable_prediction_caches` / `portable_prediction_caches_json()` | Retained OOF payload set, or `None`. |
| `detach()` | Releases every process-local resource. Returns `True` for the attached-to-detached transition and `False` on every later call. |

Every portable property remains readable after `detach()`. Normal Python object
destruction performs the same resource release if explicit detach was omitted.
Detach is thread-safe and idempotent, but it is irreversible: process-local
Python models, callback references, synthetic data/view handles and artifact
handles are gone. The callback/controller must persist every required host
sidecar before the result is detached.

### Checked fixture-backed example

The executable PyO3 example is the targeted test
`training::tests::owning_training_result_retains_resources_and_detaches_portably`
in [`crates/dag-ml-py/src/training.rs`](../crates/dag-ml-py/src/training.rs).
It starts from
[`training_request_refit.v1.json`](../examples/fixtures/training/training_request_refit.v1.json),
materializes a matching attested envelope/relation/influence set, runs the
thread scheduler through a Python `NodeTask -> NodeResult` callback, validates
the outcome, then proves detach idempotence and portable access. Run it with:

```bash
PYO3_PYTHON=python3.11 cargo test \
  --manifest-path crates/dag-ml-py/Cargo.toml \
  training::tests::owning_training_result_retains_resources_and_detaches_portably \
  -- --exact --nocapture
```

The published JSON request fixtures freeze the wider contract and intentionally
include options outside the narrower native V1 runtime matrix. The checked test
changes those runtime-only options and recomputes the request fingerprint; do
not edit a signed request dictionary in application code without producing a
new valid TCV1 fingerprint.

## D3 limits and D4 replay boundary

D3 makes native training callable and keeps local resources alive; it does not
yet make `TrainingResult` a generic fitted-estimator API. In particular:

- there is no public Python/C resolver that returns an opaque fitted-model
  handle from a `TrainingResult`, and no `result.predict()` method;
- `artifacts` exposes portable metadata only. Host model objects remain in the
  controller/model registry and must be persisted by the host when a sidecar is
  required;
- parameter patches are projected but native execution still refuses a
  non-empty `request.parameter_patches` list;
- the C result retains controller/artifact resources until its single free,
  while the Python result additionally retains in-memory data/view audit
  records until detach or drop;
- direct conformal, robustness and WASM training bindings remain separate
  integration work.

`TrainingOutcome.replayable_phases` is a validated capability statement, not a
method dispatcher. D4 replay will consume the portable outcome/package,
execution bundle, exact envelopes, retained OOF payloads and reloadable
artifacts through the existing replay contracts. A detached Python result or a
serialized C outcome can participate only through that portable path. Any
`host_sidecar` needed after process exit must have an external persistence and
reload implementation; a process-local handle cannot be reconstructed from its
numeric value. `PortablePredictorPackage` independently proves `PREDICT`
replayability and remains the deployable boundary.

## Fingerprints and compatibility

New W1 contracts use DAG-ML TCV1. TCV1 distinguishes an integer JSON token from
an integral binary64 token (`2` and `2.0`), normalizes strings to NFC, sorts
object keys by normalized UTF-8 bytes and rejects duplicate normalized keys.
Historical graph, campaign and plan fingerprints retain their existing profile;
W1 does not silently reinterpret them.

JSON Schema `required` lists describe the canonical producer wire. Serde
`default` annotations are decoder/migration defenses and internal construction
conveniences; they do not make a required canonical key optional. For
self-fingerprinted TCV1 readers, the original JSON fingerprint is checked before
and after typed decoding, so inserting a default cannot create a second valid
encoding. Historical bundle cache records are the deliberate read-compatibility
exception: an omitted `prediction_level` is read as `sample`, while every V1
writer still emits `"prediction_level": "sample"` explicitly to preserve the
published v0.2.7 wire and its containing fingerprints.

All schema-version bumps, namespace changes, capability additions or fingerprint
profile changes require new fixtures and an explicit migration. Unknown fields
and future versions fail closed.

## Conformance gates

The dedicated pack is
`contracts/training_contract_conformance_pack.v1.json`. It hashes the schemas,
positive and re-fingerprinted negative fixtures, independent oracle, generator,
tests and this documentation. The primary gates are:

```bash
python3 parity/training/generate_fixtures.py
python3 -m pytest parity/training/tests/test_training_contracts.py -q
python3 scripts/validate_contracts.py --require-sibling \
  --sibling-root ../dag-ml-data
cargo test -p dag-ml-core training::tests::
cargo test -p dag-ml-cli --test training_contracts
PYO3_PYTHON=python3.11 cargo test \
  --manifest-path crates/dag-ml-py/Cargo.toml
```

The negative corpus re-fingerprints semantic mutations so a test cannot pass
only because a stale outer checksum was detected.
