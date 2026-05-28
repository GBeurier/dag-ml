# Shared Contracts

This directory contains wire-contract artifacts shared with `dag-ml-data`, plus
DAG-ML-specific publication schemas. `dag-ml` remains the consumer and semantic
validator: it checks fingerprints, campaign fold membership, OOF boundaries and
leakage policies before any controller receives a handle.

## Coordinator Data Plan Envelope v1

Schema: `coordinator_data_plan_envelope.schema.json`

Canonical fixture: `examples/fixtures/data/coordinator_data_plan_envelope_nir.json`

Conformance pack: `conformance_pack.v1.json`

Runtime type consumed here: `ExternalDataPlanEnvelope`

Producer type in `dag-ml-data`: `CoordinatorDataPlanEnvelope`

The envelope binds a data plan to stable schema, plan and relation
fingerprints. It may carry coordinator relation records for sample, target,
group, origin, source and augmentation identity. The JSON Schema documents the
portable shape of that envelope; Rust validation enforces the stronger semantic
rules that depend on the active campaign.

Short-term policy: both repositories keep a JSON-identical conformance fixture
for this envelope plus a copy of the v1 schema, and test that the published
artifact declares the Rust-supported version. `scripts/validate_contracts.py`
compares the fixture and schema copies when `DAG_ML_DATA_REPO` points to a
sibling checkout, validates the shared conformance-pack digests, and CI checks
out that peer explicitly. When development moves into a monorepo, this file
should become a single generated or shared contract artifact used by both
crates.

## Coordinator Branch View v1

Schema: `coordinator_branch_view.schema.json`

Mirrors `dag-ml-data`'s `coordinator_branch_view.v1` byte-for-byte except for
the `$id` (each repo declares its own domain). The normalized SHA-256 (with
`$id` stripped) is pinned identically in both repos' `conformance_pack.v1.json`
so the wire contract cannot drift. `BranchViewPlan` records in
`dag-ml`/`crates/dag-ml-core/src/data.rs` accept the same shape and the
in-memory `dag-ml-data` provider executes `by_source` natively.

## Fitted Adapter Ref v1

Schema: `fitted_adapter_ref.schema.json`

Mirrors `dag-ml-data`'s `fitted_adapter_ref.v1` byte-for-byte except for the
`$id`. Same normalized-SHA-256 enforcement in both `conformance_pack.v1.json`
files. The producer type is `FittedAdapterRef` in `dag-ml-data`; `dag-ml`
relies on this schema to validate fitted-adapter records that flow through
the data layer at refit time.

## Feature Fusion Selector v1

Schema: `feature_fusion_selector.schema.json`

Canonical fixture: `examples/fixtures/data/feature_fusion_selector_nir_chem.json`

Runtime shape passed through data-provider `feature_arrow` when the provider
supports `dag-ml-data` multi-source fusion:
`{ schema_version, feature_set_id, sources, alignment, policy? }`, where each
source maps a `source_id` to a provider-owned `feature_set_id` and optional
column subset. This keeps `DagMlDataVTable` ABI-compatible while making feature
fusion explicit.

## GraphSpec v1

Schema: `graph_spec.schema.json`

Canonical fixture: `examples/branch_merge_oof_graph.json`

Runtime type: `GraphSpec`

C ABI: `DAG_ML_GRAPH_SPEC_SCHEMA_VERSION`,
`dagml_graph_spec_contract_json`, `dagml_graph_validate_json`

This is the portable graph object produced by the DSL compiler and consumed by
the execution-plan builder. The schema documents node kinds, ports, edge
contracts, OOF prediction edges and lineage propagation flags so host bindings
can reject malformed graph JSON before controller resolution or scheduling.
Rust validation remains the semantic authority for uniqueness, endpoint checks,
port-kind alignment and cycle refusal.

## Pipeline DSL v1

Schema: `pipeline_dsl.schema.json`

Canonical compatibility fixture: `examples/pipeline_dsl_nirs4all_compat.json`

Runtime parser: `parse_pipeline_dsl_json`

C ABI: `DAG_ML_PIPELINE_DSL_SCHEMA_VERSION`,
`dagml_pipeline_dsl_contract_json`, `dagml_pipeline_dsl_validate_json`,
`dagml_pipeline_dsl_compile_json`,
`dagml_pipeline_dsl_compile_artifact_json`,
`dagml_pipeline_dsl_execution_plan_build_json`

This is the public input contract for both canonical `PipelineDslSpec` JSON and
serialized nirs4all-style list/dict JSON. The schema documents the accepted
portable surface: canonical step kinds plus compatibility keys such as
`pipeline`, `preprocessing`, `model`, `branch`, `merge`, `split`, `sources`,
`_or_`, `_cartesian_`, `_chain_`, `_grid_`, `_range_`, `_log_range_`, `_zip_`
and `_sample_`. The compatibility profile also accepts minimal nirs4all-style
operator aliases: short strings such as `SNV`/`PLSRegression`, plain
`{"class": ...}` / `{"function": ...}` / `{"ref": ...}` objects, and
`{"name": ..., "step": ...}` wrappers. Rust classifies those aliases only far
enough to build the safe plan: splitters become campaign `SplitInvocation`
entries, obvious estimators become model nodes, obvious tuners such as
`OptunaTuner` become tuner nodes, chart aliases become chart nodes, and
everything else remains an external transform for host controller resolution.
When the compiler is given controller manifests (`compile-pipeline-dsl
--controllers` or `build-pipeline-dsl-plan`), selector-only aliases can refine
this default before graph ports are frozen: a custom bare alias such as
`ElasticSpectra` may become a model if exactly one operator kind claims it
through `operator_selectors`. Cross-kind matches are rejected and must use
explicit DSL syntax.
Rust validation remains the semantic authority: it lowers
compatibility JSON into canonical DSL, compiles the graph/campaign/generation
artifact, rejects unsafe augmentation/shape contracts and enforces OOF graph
edges.

External tuner/finetune controllers are canonical operator steps.
`kind: "tuner"` and its alias `kind: "finetune"` compile to
`NodeKind::Tuner`, preserve public tuning metadata and produce fold-aligned OOF
prediction outputs like model nodes; the actual search implementation remains in
the host controller.

Runtime data generators are canonical operator steps, separate from
compile-time search-space generators. `kind: "data_generation"` and its alias
`kind: "generation"` compile to `NodeKind::Generator` and require a public
`shape` contract so synthetic samples/features can be scoped to fold-train
data, audited through origin/group/target inheritance and executed by an
external controller.

## CampaignSpec v1

Schema: `campaign_spec.schema.json`

Canonical fixture: `examples/campaign_oof_generation.json`

Runtime type: `CampaignSpec`

C ABI: `DAG_ML_CAMPAIGN_SPEC_SCHEMA_VERSION`,
`dagml_campaign_spec_contract_json`, `dagml_campaign_validate_json`

This is the portable experimental-plan contract layered beside the graph. It
keeps split invocation, concrete fold sets, leakage-unit policy, repeated-sample
aggregation policy, generation/search dimensions, data/model shape plans and
data bindings outside operator nodes. Selector-driven separation branches are
recorded here as `branch_view_plans`, so source/metadata/tag/filter branch views
can be materialized by data-provider bindings without turning splits or filters
into graph operators. Rust validation remains the semantic authority for fold
membership, leakage guards, generation consistency, shape-plan/key alignment,
branch-view selector sanity and data-binding fingerprint requirements.

## ExecutionPlan v1

Schema: `execution_plan.schema.json`

Canonical fixture:
`examples/fixtures/runtime/execution_plan_branch_merge_executable.json`

Runtime type: `ExecutionPlan`

C ABI: `DAG_ML_EXECUTION_PLAN_SCHEMA_VERSION`,
`dagml_execution_plan_contract_json`,
`dagml_execution_plan_validate_json`

This is the compiled, scheduler-ready DAG contract. It binds the validated
graph, campaign, resolved controller manifests, per-node execution policies,
generation variants, fold set and canonical fingerprints used later by bundles,
replay and provenance exports. The schema documents the portable envelope and
critical coordination fields; Rust validation remains the authority for DAG
topology, controller-policy consistency, OOF capability checks, fold semantics,
shape/data binding checks and fingerprint consistency.

## ModelInputSpec v1

Schema: `model_input_spec.schema.json`

Canonical fixture:
`examples/fixtures/data/model_input_spec_tabular_regressor.json`

Runtime type: `ModelInputSpec`

C ABI: `DAG_ML_MODEL_INPUT_SPEC_SCHEMA_VERSION`,
`dagml_model_input_spec_contract_json`,
`dagml_model_input_spec_validate_json`

This neutral contract is the data/model compatibility request declared by a
controller or binding. It lists required input ports, accepted
representations/types, tensor rank expectations, multi-source support and the
default fusion policy to ask from a data planner.

## DataPlan v1

Schema: `data_plan.schema.json`

Canonical fixture: `examples/fixtures/data/data_plan_tabular_fusion.json`

Runtime type: `DataPlan`

C ABI: `DAG_ML_DATA_PLAN_SCHEMA_VERSION`, `dagml_data_plan_contract_json`,
`dagml_data_plan_validate_json`

This neutral contract is the data-planner answer to a `ModelInputSpec`: a
deterministic sequence of materialize/adapt/align/join/collate steps plus the
named outputs that feed model ports. DAG-ML validates ordering, output
references and refusal metadata before such a plan can become part of an
execution plan or bundle.

## ControllerManifest v1

Schema: `controller_manifest.schema.json`

Canonical fixture:
`examples/fixtures/runtime/controller_manifest_data_aware_model.json`

Runtime type: `ControllerManifest`

C ABI: `DAG_ML_CONTROLLER_MANIFEST_SCHEMA_VERSION`,
`dagml_controller_manifest_contract_json`,
`dagml_controller_manifest_validate_json`,
`dagml_controller_manifest_list_validate_json`

This is the binding-facing contract each external controller registry must
publish. It declares the controller id/version, operator kind, phase support,
ports, deterministic/replay capabilities, fit scope, RNG policy, artifact
policy and optional `ModelInputSpec` data requirements. The schema is the
portable shape; Rust validation remains the authority for registry uniqueness,
phase/fit-scope consistency, capability/port consistency and typed
`data_requirements` semantics.

`operator_selectors` are the minimal-alias bridge used by bindings. A host can
publish a `TransformerMixin` controller that matches aliases such as `SNV`,
plain strings such as `StandardScaler`, a tuner controller that matches
`OptunaTuner`, or class/function/ref/type descriptors; Rust keeps the operator
payload opaque, uses selectors to classify bare aliases when manifests are
available at compile time, and routes the node to the matching controller before
execution.

## NodeTask / NodeResult v1

Schemas: `node_task.schema.json`, `node_result.schema.json`

Canonical fixtures:
`examples/fixtures/runtime/node_task_transform_scale.json`,
`examples/fixtures/runtime/node_result_transform_scale.json`

Runtime types: `NodeTask`, `NodeResult`

C ABI: `DAG_ML_NODE_TASK_SCHEMA_VERSION`,
`DAG_ML_NODE_RESULT_SCHEMA_VERSION`, `dagml_node_task_contract_json`,
`dagml_node_result_contract_json`,
`dagml_node_result_validate_for_task_json`

These are the direct wire contracts between the Rust coordinator and external
operator controllers. `NodeTask` carries the resolved node plan, phase,
variant/fold context, handles, data views, OOF prediction inputs, refit artifact
inputs and deterministic seed. `NodeResult` returns output handles,
sample predictions, optional observation-level predictions, optional aggregated
sample/target/group predictions, shape deltas, artifacts and lineage. Rust validates every result
against the exact task before committing it, including node/run/phase/fold,
variant, controller, seed, params fingerprint, shape fingerprints, output
ownership and artifact handle consistency.

## SelectionPolicy / SelectionDecision v1

Schemas: `selection_policy.schema.json`, `selection_decision.schema.json`

Canonical fixtures: `examples/fixtures/bundle/selection_policy_rmse.json`,
`examples/fixtures/bundle/selection_decision_branch_b0.json`

Runtime types: `SelectionPolicy`, `SelectionDecision`

C ABI: `DAG_ML_SELECTION_POLICY_SCHEMA_VERSION`,
`DAG_ML_SELECTION_DECISION_SCHEMA_VERSION`,
`dagml_selection_policy_contract_json`,
`dagml_selection_decision_contract_json`,
`dagml_selection_policy_validate_json`,
`dagml_selection_decision_validate_json`

These contracts preserve the selection boundary used before refit/replay:
metric name/objective, optional required prediction level
(`observation`/`sample`/`target`/`group`), selected candidate, selected score
and the deterministic ranked candidate list. Rust validation remains the
semantic authority for rank continuity, selected-candidate consistency,
duplicate candidates and finite selected scores.

## DAG-ML OpenLineage Facets v1

Schema: `openlineage_dagml_facets.schema.json`

This is a DAG-ML-specific publication contract, not a shared `dag-ml-data`
wire contract. `export-open-lineage` derives an OpenLineage `RunEvent` from an
already validated research provenance package and uses these custom `dagml_*`
facets to preserve DAG-ML fingerprints, OOF coverage counters, unsafe flags and
bundle/plan identifiers that OpenLineage does not model natively.

## Prediction Cache Tensor Metadata v1

Schema: `prediction_cache_tensor_metadata.schema.json`

This C ABI metadata contract accompanies
`dagml_prediction_cache_payload_f64_tensor_json`. The tensor carries contiguous
row-major F64 prediction values; the metadata carries the stable requirement
key, cache id, prediction level, block offsets, fold ids, sample ids, unit ids
and target names required to interpret rows without hiding traceability inside
the value buffer.

## Prediction Cache Columnar Tensor Metadata v1

Schema: `prediction_cache_columnar_tensor_metadata.schema.json`

This C ABI metadata contract accompanies
`dagml_prediction_cache_payload_f64_columnar_tensor_json`. It keeps the same
traceability fields as the row-major export and adds `layout:
column_major_f64` plus `column_offsets` so host bindings can read each target
column contiguously without guessing buffer order.

## Data Output Provenance v1

Schema: `data_output_provenance.schema.json`

Canonical fixture:
`examples/fixtures/runtime/data_output_provenance_augmented_view.json`

Runtime type: `DataOutputProvenance`

C ABI: `DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION`,
`DAG_ML_DATA_OUTPUT_PROVENANCE_EXTRA_KEY`,
`dagml_data_output_provenance_contract_json`,
`dagml_data_output_provenance_validate_json`

This DAG-ML runtime contract is embedded under the reserved
`DataProviderViewSpec.extra["dag_ml_output"]` key when a data-producing DAG node
emits a downstream data view. It records the producer node/port/phase,
variant/fold scope, shape-plan and aggregation fingerprints, current feature
schema fingerprint and emitted shape deltas. Controllers and host bindings can
discover and validate this metadata without reverse-engineering free-form JSON
or hardcoding Rust-only constants.

## Process Adapter Description v1

Schema: `process_adapter_description.schema.json`

Canonical fixture:
`examples/fixtures/runtime/process_adapter_description_python.json`

Runtime shape returned by process adapters from `--describe`:
`{ schema_version, protocol, adapter_id, supported_modes, capabilities }`.

C ABI: `DAG_ML_PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION`,
`dagml_process_adapter_description_contract_json`

This CLI/runtime contract lets the coordinator reject unsupported process
adapters before any `NodeTask` is sent. Version 1 requires protocol
`dag-ml-process-adapter`, mode declarations for `one_shot`/`jsonl` support and
explicit JSON task/result capabilities. Persistent worker and parallel
scheduler features remain opt-in capabilities layered on the same description
object.

## Process Adapter Frame v1

Schema: `process_adapter_frame.schema.json`

Canonical fixtures:
`examples/fixtures/runtime/process_adapter_frame_init.json`,
`process_adapter_frame_task_transform_scale.json`,
`process_adapter_frame_result_transform_scale.json`,
`process_adapter_frame_ack_initialized.json`,
`process_adapter_frame_error_retryable_timeout.json`,
`process_adapter_frame_close.json`

Runtime shape used by persistent JSONL process adapters:
`init | task | close` coordinator request frames and
`ack | result | error` adapter response frames.

C ABI: `DAG_ML_PROCESS_ADAPTER_FRAME_SCHEMA_VERSION`,
`dagml_process_adapter_frame_contract_json`

This contract is enabled only when the adapter description declares
`control_frames_v1`. It gives host adapters a stable lifecycle and error
surface: `init` carries controller and worker identity, `task` wraps a
published `NodeTask`, `result` wraps a published `NodeResult`, `error` carries
typed retryability, and `close` gives the coordinator a bounded shutdown path.

## Aggregation Controller Task/Result v1

Schemas: `aggregation_controller_task.schema.json` and
`aggregation_controller_result.schema.json`

C ABI: `DAG_ML_AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION`,
`DAG_ML_AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION`,
`dagml_aggregation_controller_task_contract_json`,
`dagml_aggregation_controller_result_contract_json`,
`dagml_aggregation_controller_task_validate_json`,
`dagml_aggregation_controller_result_validate_for_task_json`

These contracts define the leakage-sensitive payloads used when aggregation is
delegated to an external controller through `AggregationMethod::CustomController`.
The task carries the custom aggregation policy, controller id, repeated
observation or sample-to-unit inputs, relation metadata and requested output
order. The result is validated against the exact task so custom reducers cannot
change sample/unit coverage, fold scope, target names or prediction level.

## Research Provenance Package Profile v1

Profile: `research_provenance_package_profile.v1.json`

This publication profile declares the required files, optional files, checksum
rules, PROV JSON-LD sections, RO-Crate file properties, OpenLineage facets and
CLI tests for a DAG-ML research package. It is validated by
`scripts/validate_contracts.py` so the human-facing publication contract stays
aligned with the Rust/CLI validator.

## Data Provider C ABI v2

The shared provider surface is `DagMlDataVTable` guarded by
`DAG_ML_DATA_VTABLE_DEFINED` and versioned by
`DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION == 2`. `scripts/validate_contracts.py`
and the C ABI tests verify that `dag_ml.h` and `dag_ml_data.h` can be included
together in either order when the sibling checkout is available.
