# Design Source Documents

These files were moved from `nirs4all/docs/_internal/lib_ML` during project
bootstrap. They are intentionally kept close to their original form so future
implementation work can be traced back to the design decisions.

| File | Role |
|---|---|
| `source/dag_ml_synthese.md` | Read first: mission, architecture, ABI decisions and roadmap |
| `source/dag_ml_specification_v1.md` | Full execution-engine specification |
| `source/dag_ml_polyglot_core_design.md` | Rust/C ABI and polyglot design deliberation |
| `source/dag_ml_use_cases.md` | Concrete use cases and leakage invariants |
| `source/dag_ml_externalization_from_code.md` | Notes from the current nirs4all runtime |

The companion ML_DATA contract now lives in the `dag-ml-data` repository at
`docs/design/source/ml_data_specification_v1.md`.
