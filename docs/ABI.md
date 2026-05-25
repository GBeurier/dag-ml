# C ABI

The ABI is designed around opaque host handles. Rust owns the control lifetime;
the host owns the underlying object behind each handle.

## Current Scaffold

`crates/dag-ml-capi/include/dag_ml.h` exposes:

- version and string-free helpers;
- `dagml_graph_validate_json` for a first executable ABI check;
- `DagMlControllerVTable` for host operator controllers;
- `DagMlDataVTable` for host data providers.

The vtables are intentionally small in this scaffold. They establish shape,
ownership and naming before full execution is implemented.

## Ownership Rules

| Object | Owner | Release path |
|---|---|---|
| Host data block | Host | `DataVTable.release` |
| Host fitted model | Host | `ControllerVTable.release` |
| Rust error string | Rust allocation returned through ABI | `dagml_string_free` |
| Arrow arrays | Producer of the Arrow array | Arrow C Data Interface release callback |
| JSON blobs | Caller-provided view unless returned as owned bytes | ABI-specific free function |

## ABI Roadmap

1. Freeze `DagMlBytesView`, `DagMlOwnedBytes`, handle and status conventions.
2. Add canonical JSON schemas for `describe`, `GraphSpec`, `ModelInputSpec` and
   `DataPlan` blobs.
3. Replace placeholder Arrow pointers with explicit `struct ArrowArray` and
   `struct ArrowSchema` declarations in the header.
4. Add conformance tests that call the C ABI from a small C program.
5. Add host adapters for Python and native C++ controllers.
