# Design Source Documents

These files were moved from `nirs4all/docs/_internal/lib_ML` during project
bootstrap. They are intentionally kept close to their original form so future
implementation work can be traced back to the design decisions.

The initial specifications are archived and are not current acceptance criteria.
Read [the development map](../DEVELOPMENT.md) and
[the coordinator contract](../COORDINATOR_SPEC.md) before using them.

| File | Role |
|---|---|
| `source/_archive/dag_ml_synthese.md` | Original mission, architecture, ABI decisions and roadmap |
| `source/_archive/dag_ml_specification_v1.md` | Original execution-engine specification |
| `source/_archive/dag_ml_polyglot_core_design.md` | Rust/C ABI and polyglot design deliberation |
| `source/_archive/dag_ml_use_cases.md` | Historical use cases and leakage invariants |
| `source/_archive/dag_ml_externalization_from_code.md` | Notes from the pre-migration nirs4all runtime |
| `DSL_NIRS4ALL_PARITY.md` | Working parity matrix between the strict dag-ml DSL and nirs4all pipeline expressivity |

The companion ML_DATA contract now lives in the `dag-ml-data` repository at
`docs/design/source/ml_data_specification_v1.md`.
