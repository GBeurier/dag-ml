# Agent Handoff

Start here when implementing `dag-ml`.

## Mission

Build the Rust control core for DAG-based ML execution. The core owns graph
validation, phase planning, fold identity, OOF joins, lineage/cache metadata,
scheduler decisions and deterministic control RNG. Heavy data buffers and
operator objects stay host-owned behind handles.

## Migration Mandate (2026-06-23, maintainer directive)

`dag-ml` is the **native (Rust / C-ABI) core that replaces the nirs4all core** and must
provide **all of nirs4all's generic functionality** natively — not just compile/plan/OOF, but
also **CV prediction aggregation** (per-fold + `avg` / `w_avg` ensemble + `final` refit rows),
**scoring/metrics**, **selection**, and **prediction/score persistence**. The end state:
**nirs4all becomes fully cross-language; only the operators/controllers stay per-language**
(host bindings). Parity with the legacy nirs4all engine is **exact and native** — do NOT
re-implement aggregation/scoring in the Python host; implement it in the core so every binding
(Python/R/MATLAB/WASM) gets it for free.

This does NOT relax the "no NIRS-specific logic" boundary below: aggregation, scoring and
persistence are *generic ML coordination* concerns. NIRS specifics (spectra transforms, model
families) remain host operators. Persistence of predictions/scores is being scoped as a
**native, light-dependency layer between `nirs4all-io` and `dag-ml-data`** (report pending).

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
