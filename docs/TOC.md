# dag-ml Table Of Contents

Use this as a validation map before development starts.

| Area | File | Purpose | Validate |
|---|---|---|---|
| Entry point | `README.md` | Project scope, layout and quick start | The repo can be understood in under five minutes |
| Agent handoff | `AGENTS.md` | Rules for autonomous implementation work | A new agent knows boundaries and green gate |
| Architecture | `docs/ARCHITECTURE.md` | Runtime layers, crate responsibilities, phase flow | DAG-ML owns control only, not data buffers |
| Coordinator spec | `docs/COORDINATOR_SPEC.md` | Short normative product and coordinator contract | Controllers stay external; Rust owns orchestration and invariants |
| Shared contract schema | `docs/contracts/coordinator_data_plan_envelope.schema.json` | JSON Schema for the external data-plan envelope consumed from `dag-ml-data` | Fixtures and bindings declare the same v1 envelope shape |
| ABI | `docs/ABI.md` | C ABI shape, handles, vtables, ownership | No host object crosses as a Rust-owned value |
| Rationale | `docs/RATIONALE.md` | Technical decisions and non-goals | Rust/C ABI split is justified and scoped |
| MVP acceptance | `docs/MVP_ACCEPTANCE.md` | First executable target and no-leakage gates | UC6 succeeds and UC11 fails for the right reason |
| Capability matrix | `docs/CAPABILITY_MATRIX.md` | Full nirs4all replacement surface | Every feature maps to an owner and an invariant |
| OOF fixtures | `docs/OOF_FIXTURES.md` | Shared tiny campaign fixtures | UC6 joins and UC11 refuses leakage from file-backed tests |
| Roadmap | `docs/ROADMAP.md` | Sequenced delivery phases | Every phase has an observable definition of done |
| Status | `docs/STATUS.md` | Current scaffold state and next actions | No hidden implementation claims |
| Test plan | `docs/TEST_PLAN.md` | Invariant, ABI and conformance tests | OOF/leakage tests are first-class |
| Supported surface | `docs/SUPPORTED.md` | Release support matrix and public-signature policy | Production, conformance, experimental and backlog surfaces are separated |
| Aggregation interop | `docs/AGGREGATION_INTEROP.md` | Mapping between coordinator aggregation policies and data-side reducers | Cross-repo reducer drift is explicit |
| Performance probes | `docs/PERFORMANCE.md` | Private 0.2.0 performance sanity probes | Critical control-path regressions are measurable |
| Final release audit | `docs/FINAL_RELEASE_AUDIT.md` | Production-readiness audit for `dag-ml` plus sibling `dag-ml-data` | Release blockers and final gates are explicit |
| Source design | `docs/design/source/dag_ml_synthese.md` | Original synthesis and roadmap | Still readable after the move |
| Source design | `docs/design/source/dag_ml_specification_v1.md` | Full DAG-ML contract | Used as implementation source of truth |
| Source design | `docs/design/source/dag_ml_polyglot_core_design.md` | Rust/C ABI design deliberation | ABI skeleton reflects the ownership rules |
| Source design | `docs/design/source/dag_ml_use_cases.md` | Twelve concrete use cases | Test fixtures can be derived from UC6/UC11 first |
| Source design | `docs/design/source/dag_ml_externalization_from_code.md` | Extraction notes from current nirs4all runtime | Migration tasks are traceable to existing code |
| Core crate | `crates/dag-ml-core` | Graph, phase, OOF, RNG primitives | `cargo test -p dag-ml-core` passes |
| C ABI crate | `crates/dag-ml-capi` | FFI-safe structs/functions and header | Header mirrors Rust ABI structs |
| CLI crate | `crates/dag-ml-cli` | Local validation tool | Example graph validates |

## Validation Checklist

| Check | Command |
|---|---|
| Rust formatting | `cargo fmt --all --check` |
| Rust tests | `cargo test --workspace` |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` |
| Contract schema syntax | `python3 -m json.tool docs/contracts/coordinator_data_plan_envelope.schema.json >/dev/null` |
| Shared contract drift | `DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py` |
| Example graph | `cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json` |
