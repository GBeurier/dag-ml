# C ABI

The ABI is designed around opaque host handles. Rust owns the control lifetime;
the host owns the underlying object behind each handle.

## Current Scaffold

`crates/dag-ml-capi/include/dag_ml.h` exposes:

- version and string-free helpers;
- owned byte release helper for JSON outputs returned by Rust;
- owned row-major `DagMlF64Tensor` and column-major
  `DagMlF64ColumnarTensor` release helpers for Rust-allocated prediction
  buffers returned to host bindings;
- `dagml_graph_spec_contract_json` and `dagml_graph_validate_json` for
  GraphSpec contract discovery and graph validation before plan building;
- `dagml_model_input_spec_contract_json`,
  `dagml_model_input_spec_validate_json`, `dagml_data_plan_contract_json` and
  `dagml_data_plan_validate_json` for non-Rust bindings that need to exchange
  neutral data/model compatibility requests and data-planner answers;
- `dagml_controller_manifest_validate_json` and
  `dagml_controller_manifest_list_validate_json` so bindings can preflight
  controller manifests and registry uniqueness before plan building;
- `dagml_data_output_provenance_contract_json` and
  `dagml_data_output_provenance_validate_json` so bindings can discover the
  reserved `DataProviderViewSpec.extra` key and validate propagated data-view
  provenance before using it;
- `dagml_node_result_validate_for_task_json` so host bindings can preflight a
  controller-produced `NodeResult` against the exact `NodeTask` before handing
  the JSON back to the scheduler;
- `dagml_pipeline_dsl_compile_json` for pure compilation of the strict JSON
  `PipelineDslSpec` surface into canonical `GraphSpec` JSON, plus
  `dagml_pipeline_dsl_compile_artifact_json` when bindings also need the
  extracted `GenerationSpec` including coordinated override dimensions,
  validated shape-plan fragments, validated data-binding fragments, a campaign
  template and search-space fingerprint, and
  `dagml_pipeline_dsl_execution_plan_build_json` when bindings want the Rust
  planner to build a validated `ExecutionPlan` directly from DSL and controller
  manifests;
- `dagml_graph_parallel_levels_json` for deterministic node batches that
  bindings can use to prepare parallel schedulers;
- `dagml_execution_plan_build_json` for compiling graph/campaign/controller
  manifests into an `ExecutionPlan` while Rust owns planner validation;
- `dagml_execution_plan_schedule_json` for exporting deterministic
  phase/variant/fold node-level schedules from a compiled `ExecutionPlan`;
- selection policy/decision validation and candidate selection JSON helpers;
- sample-level and target/group aggregated prediction block conversion helpers
  that validate canonical JSON blocks and return contiguous row-major `double`
  buffers with explicit `rows`, `cols`, `len` and `capacity`;
- prediction-cache payload tensor export that validates a cache payload set
  against its bundle before returning contiguous row-major F64 values plus JSON
  metadata for requirement key, prediction level, block offsets, folds and
  sample/unit ids. The metadata JSON is versioned by
  `DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION` and documented in
  `docs/contracts/prediction_cache_tensor_metadata.schema.json`;
- prediction-cache payload columnar tensor export that returns contiguous
  column-major F64 values with explicit `column_offsets`, versioned by
  `DAG_ML_PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_VERSION` and
  documented in
  `docs/contracts/prediction_cache_columnar_tensor_metadata.schema.json`;
- execution bundle validation, replay-envelope validation, replay-request
  validation and prediction-cache payload validation helpers;
- `dagml_research_provenance_export_json` for building the standards-facing
  `ResearchProvenanceExport` JSON over validated execution plans, bundles,
  optional lineage records, replay envelopes, prediction-cache manifests and
  artifact manifests; null pointer plus zero length denotes an omitted
  optional input;
- `dagml_openlineage_run_event_json` for building the same validated provenance
  evidence as an OpenLineage `RunEvent` JSON with explicit namespace and event
  time byte views;
- vtable replay execution helper that composes host controllers, host data
  provider, host artifact store and optional host prediction-cache store while
  Rust owns scheduling and validation;
- replay-request validation can optionally include an OOF prediction-cache
  payload set, which is required for OOF-dependent `REFIT` replay;
- mock replay execution helper that returns a JSON summary while exercising
  Rust-side data handle materialization, data view creation and artifact handle
  materialization;
- Arrow C Data `ArrowArray` and `ArrowSchema` structs for controller
  predictions and data-provider identity/target/feature exports;
- `DagMlControllerVTable` for host operator controllers, including generic
  `invoke` over `NodeTask`/`NodeResult` JSON and explicit returned-byte
  release. Controller vtable ABI v2 keeps borrowed `user_data` semantics; v3 is
  opt-in owned semantics where Rust calls `destroy(user_data)` after releasing
  controller-owned result handles;
- `DagMlDataVTable` for host data providers, including `materialize`,
  `make_view`, `view_identity`, `target_arrow` and `feature_arrow`.
  `feature_arrow` remains ABI-compatible: hosts may receive either a plain
  feature-set id or a JSON fusion selector understood by `dag-ml-data`
  providers. The vtable uses the shared
  `DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION` macro and guarded
  `DagMlDataVTable` definition so `dag_ml.h` and `dag_ml_data.h` can be
  included together by bindings.
- `DagMlArtifactStoreVTable` for host replay artifact stores, returning typed
  `DagMlHandleRef` values for model/artifact handles. Artifact references are
  JSON-level Rust contracts with optional typed backend, URI, content
  fingerprint and plugin/version metadata; the ABI still transports them inside
  owned JSON payloads so C structs do not freeze a storage layout too early.
  Artifact-store vtable ABI v1 is borrowed; v2 is opt-in owned lifecycle with
  `destroy(user_data)` after materialized artifact handles are released.
- `DagMlPredictionCacheVTable` for host prediction-cache stores, including
  `load_blocks`, `materialize` and explicit returned-byte release.
  `load_blocks` is the single JSON load callback for replay prediction
  requirements: sample-level requirements return `PredictionBlock[]`, while
  target/group requirements return `AggregatedPredictionBlock[]` keyed by typed
  `PredictionUnitId` values. Rust selects and validates the expected block
  shape from the bundle requirement before materializing a prediction handle.
  Prediction-cache vtable ABI v1 is borrowed; v2 is opt-in owned lifecycle with
  `destroy(user_data)` after materialized prediction handles are released.
- `docs/contracts/process_adapter_description.schema.json` documents the
  required process-adapter `--describe` JSON used by CLI-managed host adapters.
  The C ABI exposes its schema version and schema id through
  `dagml_process_adapter_description_contract_json`.
- `docs/contracts/process_adapter_frame.schema.json` documents the
  `control_frames_v1` JSONL protocol used by persistent process adapters:
  coordinator `init`/`task`/`close` request frames and adapter
  `ack`/`result`/`error` response frames wrapping the published
  `NodeTask`/`NodeResult` contracts. The C ABI exposes the frame schema version
  and schema id through `dagml_process_adapter_frame_contract_json`.
- `docs/contracts/graph_spec.schema.json` documents the canonical graph JSON
  consumed by the planner and C ABI.
- `docs/contracts/model_input_spec.schema.json` and
  `docs/contracts/data_plan.schema.json` document the neutral data-shape
  contracts exchanged between controller descriptions, data planners and host
  bindings.

The vtables are intentionally small in this scaffold. They establish shape,
ownership and naming before full execution is implemented.

`DagMlStatusCode` is a fixed `uint32_t` ABI value rather than a C/Rust enum
boundary type. Unknown host status codes are treated as runtime validation
errors instead of being decoded as Rust enum discriminants.

Vtable `user_data` lifetime is explicit per ABI surface. Controller vtable v2 is
borrowed for backwards compatibility, while controller vtable v3 opts into
Rust-owned lifecycle and calls `destroy(user_data)` after handle release on drop.
Artifact-store and prediction-cache vtable v1 are also borrowed; their v2
surfaces opt into Rust-owned lifecycle with `destroy(user_data)` after
materialized handles are released. Data-provider vtables remain borrowed in the
current replay API because that ABI is shared with `dag-ml-data`. Rust releases
controller-result, data/view, replay-artifact and prediction-cache handles that
it receives or materializes through the vtables.

## Ownership Rules

| Object | Owner | Release path |
|---|---|---|
| Host data block | Host | `DataVTable.release` |
| Host controller result JSON | Host allocation returned through controller vtable | `ControllerVTable.release_bytes` |
| Host fitted model | Host | `ControllerVTable.release` |
| Host replay artifact handle | Host | `ArtifactStoreVTable.release` |
| Host prediction cache handle | Host | `PredictionCacheVTable.release` |
| Rust error string | Rust allocation returned through ABI | `dagml_string_free` |
| Rust JSON byte output | Rust allocation returned through ABI | `dagml_owned_bytes_free` |
| Rust row-major F64 tensor output | Rust allocation returned through ABI | `dagml_f64_tensor_free` |
| Rust column-major F64 tensor output | Rust allocation returned through ABI | `dagml_f64_columnar_tensor_free` |
| Host JSON byte output | Host allocation returned through prediction-cache vtable | `PredictionCacheVTable.release_bytes` |
| Arrow arrays | Producer of the Arrow array | Arrow C Data Interface release callback |
| JSON blobs | Caller-provided view unless returned as owned bytes | ABI-specific free function |

## ABI Roadmap

1. Freeze `DagMlBytesView`, `DagMlOwnedBytes`, `DagMlF64Tensor`,
   `DagMlF64ColumnarTensor`, handle and status conventions.
2. Add schema coverage for the remaining execution-plan and bundle sub-blobs
   that are still Rust-implicit.
3. Add conformance tests that call the C ABI from a small C program.
4. Add C conformance tests that drive non-mock replay through the vtable
   surface.
5. Keep shared `dag-ml-data` header inclusion and vtable ABI conformance in CI.
6. Add host adapters for Python and native C++ controllers and artifact stores.
