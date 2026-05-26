# dag-ml

Rust-first execution core for leakage-safe, in-process ML pipelines.

`dag-ml` owns the graph, phases, folds, OOF joins, controller ABI, lineage,
cache and deterministic control RNG. It does not own source storage or feature
buffers; those contracts live in the companion `dag-ml-data` repository.

> Status: active core scaffold. The project has executable Rust crates, C ABI
> graph validation, CLI validation, coordinator planning/runtime contracts,
> data-plan fingerprints, OOF leakage checks, deterministic selection and first
> refit/replay bundle contracts. Host bindings and real controller adapters are
> still pending.

## Repository Layout

```text
crates/
  dag-ml-core/      # graph, phase, OOF, selection, bundle and control contracts
  dag-ml/           # Rust facade re-exporting stable core APIs
  dag-ml-capi/      # C ABI surface and header for host/controller integration
  dag-ml-cli/       # small validation CLI for specs and fixtures
docs/
  TOC.md            # validation-oriented table of contents
  ARCHITECTURE.md   # module boundaries and runtime flow
  ABI.md            # C ABI ownership model and vtable roadmap
  RATIONALE.md      # why Rust/C ABI, why the data split, non-goals
  ROADMAP.md        # phase plan and delivery gates
  STATUS.md         # current state and next tasks
  TEST_PLAN.md      # invariant and conformance test strategy
  design/source/    # moved source design markdowns from nirs4all
examples/
  minimal_graph.json
```

## Quick Start

```bash
cargo fmt --all --check
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
```

## First Implementation Target

The current useful milestone is a sequential Rust core that can:

1. parse a canonical `GraphSpec`;
2. validate edge contracts and acyclicity;
3. consume identity-only fold assignments;
4. join validation predictions by `sample_id`;
5. reject train predictions as meta-model training features by default;
6. select branch/merge variants from persisted OOF metrics;
7. build a refit/replay bundle that locks plan, controller, data and artifact
   fingerprints.

That milestone is intentionally smaller than full pipeline execution. The next
gate is to expose selection/bundle/replay through CLI/C ABI and replace the
Python-side orchestration in the sklearn demonstrator with host controller
adapters driven by the Rust scheduler.
