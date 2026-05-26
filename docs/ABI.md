# C ABI

The ABI is designed around opaque host handles. Rust owns the control lifetime;
the host owns the underlying object behind each handle.

## Current Scaffold

`crates/dag-ml-capi/include/dag_ml.h` exposes:

- version and string-free helpers;
- owned byte release helper for JSON outputs returned by Rust;
- `dagml_graph_validate_json` for graph contract checks;
- `dagml_graph_parallel_levels_json` for deterministic node batches that
  bindings can use to prepare parallel schedulers;
- `dagml_execution_plan_build_json` for compiling graph/campaign/controller
  manifests into an `ExecutionPlan` while Rust owns planner validation;
- `dagml_execution_plan_schedule_json` for exporting deterministic
  phase/variant/fold node-level schedules from a compiled `ExecutionPlan`;
- selection policy/decision validation and candidate selection JSON helpers;
- execution bundle validation, replay-envelope validation, replay-request
  validation and prediction-cache payload validation helpers;
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
  release;
- `DagMlDataVTable` for host data providers, including `materialize`,
  `make_view`, `view_identity`, `target_arrow` and `feature_arrow`.
  `feature_arrow` remains ABI-compatible: hosts may receive either a plain
  feature-set id or a JSON fusion selector understood by `dag-ml-data`
  providers. The vtable uses the shared
  `DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION` macro and guarded
  `DagMlDataVTable` definition so `dag_ml.h` and `dag_ml_data.h` can be
  included together by bindings.
- `DagMlArtifactStoreVTable` for host replay artifact stores, returning typed
  `DagMlHandleRef` values for model/artifact handles.
- `DagMlPredictionCacheVTable` for host prediction-cache stores, including
  `load_blocks`, `materialize` and explicit returned-byte release.
  `load_blocks` is the single JSON load callback for replay prediction
  requirements: sample-level requirements return `PredictionBlock[]`, while
  target/group requirements return `AggregatedPredictionBlock[]` keyed by typed
  `PredictionUnitId` values. Rust selects and validates the expected block
  shape from the bundle requirement before materializing a prediction handle.

The vtables are intentionally small in this scaffold. They establish shape,
ownership and naming before full execution is implemented.

`DagMlStatusCode` is a fixed `uint32_t` ABI value rather than a C/Rust enum
boundary type. Unknown host status codes are treated as runtime validation
errors instead of being decoded as Rust enum discriminants.

Vtable `user_data` lifetime remains host-owned in this scaffold. `release` and
`destroy` callbacks define the ownership shape for bindings, but the current
Rust adapters do not claim ownership of the host context.

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
| Host JSON byte output | Host allocation returned through prediction-cache vtable | `PredictionCacheVTable.release_bytes` |
| Arrow arrays | Producer of the Arrow array | Arrow C Data Interface release callback |
| JSON blobs | Caller-provided view unless returned as owned bytes | ABI-specific free function |

## ABI Roadmap

1. Freeze `DagMlBytesView`, `DagMlOwnedBytes`, handle and status conventions.
2. Add canonical JSON schemas for `describe`, `GraphSpec`, `ModelInputSpec` and
   `DataPlan` blobs.
3. Add conformance tests that call the C ABI from a small C program.
4. Add C conformance tests that drive non-mock replay through the vtable
   surface.
5. Keep shared `dag-ml-data` header inclusion and vtable ABI conformance in CI.
6. Add host adapters for Python and native C++ controllers and artifact stores.
