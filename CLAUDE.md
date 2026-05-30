# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## DAG-ML Development Context

You are implementing DAG-ML, a Rust low-level coordinator for reproducible,
traceable and OOF/leakage-safe ML/DL/bioinformatics pipelines.

## Product Direction

- Keep operators external. The Rust core owns graph compilation, scheduling,
  replay, lineage, OOF safety, leakage validation, fingerprints and handle
  lifecycle.
- Bindings/controllers own model fitting, transforms, augmentations, data
  backends and native library integrations.
- Keep `dag-ml` and `dag-ml-data` responsibilities separate. Cross-repo
  compatibility is enforced by shared contracts and fixtures.
- Preserve DAG-ML-specific invariants internally. Research standards such as
  W3C PROV, Workflow Run RO-Crate, OpenLineage and MLMD are export targets,
  not the internal execution model.

## Current Priorities

- Persistent and portable artifact contracts without serializing ML objects in
  the core.
- Strong replay and bundle validation across prediction caches, artifacts and
  data envelopes.
- Production-shaped host adapters and C ABI contracts.
- Research provenance export roadmap: W3C PROV plus Workflow Run RO-Crate,
  derived from validated DAG-ML lineage and bundles.

## Engineering Rules

- Prefer small, validated slices that move the final product forward.
- Do not weaken OOF, fold, group, repetition, augmentation-origin or refit
  leakage checks for convenience.
- Do not touch `nirs4all` core code.
- Use existing crate patterns and tests.
- Keep JSON compatibility unless a schema version/migration policy is updated.
- Run targeted tests first, then `cargo fmt --check`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and
  `python3 scripts/validate_contracts.py`.

## Commands

The full green gate (matches CI in `.github/workflows/ci.yml`):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
python3 scripts/validate_contracts.py        # add DAG_ML_DATA_REPO=../dag-ml-data for cross-repo schema/fixture parity
```

Targeted iteration:

```bash
# Single crate
cargo test -p dag-ml-core
cargo test -p dag-ml-capi
cargo test -p dag-ml-cli

# Single test by name (tests live in inline `#[cfg(test)] mod tests` blocks
# inside each module, plus the out-of-module `crates/dag-ml-core/src/runtime/tests.rs`
# and `crates/dag-ml-capi/src/tests.rs`).
cargo test -p dag-ml-core <substring_of_test_name>
cargo test -p dag-ml-core runtime::tests::<name> -- --exact --nocapture

# Validation CLI (see docs/STATUS.md and docs/TEST_PLAN.md for the full
# smoke list — `validate-bundle`, `run-mock-campaign`, `run-process-replay`,
# `compile-pipeline-dsl`, `export-research-provenance`, etc.).
cargo run -p dag-ml-cli -- <subcommand> --help
```

`scripts/validate_contracts.py` uses only the Python stdlib; without
`DAG_ML_DATA_REPO` it still checks local schemas and fixtures. CI checks out
`GBeurier/dag-ml-data` and points `DAG_ML_DATA_REPO` at it so contract drift
(schemas, conformance pack, header ABI) fails the build.

## Architecture Big Picture

`docs/TOC.md` is the canonical navigation map; `docs/COORDINATOR_SPEC.md` is
the short normative product contract and the alignment source for any
ambiguity in the older design docs under `docs/design/source/`. Read those two
first before changing contracts.

### Crate dependency direction

```
dag-ml-core   pure Rust contracts, validation, runtime; no host runtime deps
   ^
   |── dag-ml         stable Rust facade (re-exports core for downstream crates)
   |── dag-ml-capi    cdylib/staticlib/rlib C ABI; header at include/dag_ml.h
   |── dag-ml-cli     local validation + smoke-execution CLI (sole binary)
```

`dag-ml-core/src/lib.rs` re-exports every submodule (`graph`, `dsl`,
`plan`, `campaign`, `controller`, `data`, `runtime`, `bundle`, `oof`,
`selection`, `aggregation`, `metrics`, `policy`, `provenance`, ...). The CLI
and C ABI consume the core through these re-exports; do not add a parallel
public surface.

### Runtime flow

```
COMPILE -> PLAN -> FIT_CV -> SELECT -> REFIT -> PREDICT -> EXPLAIN
```

The control core schedules these phases over `(variant, fold)` scopes, joins
OOF predictions by stable `sample_id`, and invokes host controllers through
typed `NodeTask` / `NodeResult` JSON or the C ABI vtables. Splitters run as
campaign-plan controller calls and produce a `FoldSet`; they are never
ordinary data-transform nodes. A training-phase edge marked `requires_oof`
must be backed by validation predictions in the core `PredictionStore` — raw
upstream handles are not forwarded across that edge.

### Ownership boundary (do not violate)

| Crosses the ABI as data        | Crosses the ABI as opaque handle |
| ------------------------------ | -------------------------------- |
| sample/group/target/origin ids | host data buffers                |
| prediction tables, `y_true`    | fitted model objects             |
| canonical JSON descriptors     | artifact/prediction-cache refs   |
| fingerprints                   | data-view handles                |

The core must never inspect feature matrices, images, spectra, tensors,
sequences, graphs or fitted operator internals. Every join, split, merge and
prediction use is keyed by stable identities, not by row position. Unsafe
leakage paths (e.g. train predictions as training features) are refused by
default and require an explicit, traceable opt-in marker.

### Contracts and fixtures

JSON Schemas live in `docs/contracts/*.schema.json` (`graph_spec`,
`pipeline_dsl`, `campaign_spec`, `execution_plan`, `controller_manifest`,
`coordinator_data_plan_envelope`, `node_task`, `node_result`,
`selection_policy`, `selection_decision`, `prediction_cache_*_metadata`,
`process_adapter_*`, `research_provenance_package_profile`,
`openlineage_dagml_facets`, ...). Canonical example DSL/graph/campaign JSON
sits in `examples/`, with reusable fixtures under `examples/fixtures/` and CLI
outputs landing in `examples/generated/`. Host adapter smokes live in
`examples/adapters/` (`python_process_controller.py`,
`sklearn_process_controller.py`, `flaky_process_controller.py`).

When you change a contract, update the schema, the local fixture, the Rust
type, the C ABI version constant if exposed, and the conformance pack so
`validate_contracts.py` keeps `dag-ml-data` in lockstep.

### C ABI conventions

Defined in `crates/dag-ml-capi/src/lib.rs` and mirrored in
`crates/dag-ml-capi/include/dag_ml.h`. Rust-allocated outputs are released
through dedicated helpers (`dagml_string_free`, `dagml_owned_bytes_free`,
`dagml_f64_tensor_free`, `dagml_f64_columnar_tensor_free`); host-owned
handles are released through their vtable's `release` callback. Controller,
artifact-store and prediction-cache vtables have explicit v1/v2/v3 ABI
versions distinguishing borrowed vs. owned `user_data` lifecycle — see
`docs/ABI.md` before extending them.
