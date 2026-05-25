# Agent Handoff

Start here when implementing `dag-ml`.

## Mission

Build the Rust control core for DAG-based ML execution. The core owns graph
validation, phase planning, fold identity, OOF joins, lineage/cache metadata,
scheduler decisions and deterministic control RNG. Heavy data buffers and
operator objects stay host-owned behind handles.

## Hard Boundaries

- Do not add NIRS-specific logic.
- Do not materialize `X` or feature buffers in the core.
- Identity, predictions and `y_true` may cross the ABI as Arrow-compatible
  tables; host data and fitted objects cross as opaque handles.
- A train prediction must not become a training feature unless an explicit
  unsafe policy is added and tested.
- Slice and join by stable sample ids, not by positional indices.

## Working Gate

Run before handing work back:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
```

## First Files To Read

1. `docs/TOC.md`
2. `docs/RATIONALE.md`
3. `docs/ARCHITECTURE.md`
4. `docs/ABI.md`
5. `docs/design/source/dag_ml_synthese.md`
6. `docs/design/source/dag_ml_specification_v1.md`
