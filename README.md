# dag-ml

Rust-first execution core for leakage-safe, in-process ML pipelines.

`dag-ml` owns the graph, phases, folds, OOF joins, controller ABI, lineage,
cache and deterministic control RNG. It does not own source storage or feature
buffers; those contracts live in the companion `dag-ml-data` repository.

> Status: foundation scaffold. The project is ready for implementation work:
> executable Rust crates, C ABI header, CLI validation, design documents,
> rationale, roadmap, CI and first invariant tests are present.

## Repository Layout

```text
crates/
  dag-ml-core/      # graph, phase, OOF and deterministic control contracts
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

The first useful milestone is a sequential Rust core that can:

1. parse a canonical `GraphSpec`;
2. validate edge contracts and acyclicity;
3. consume identity-only fold assignments;
4. join validation predictions by `sample_id`;
5. reject train predictions as meta-model training features by default;
6. expose the same checks through the C ABI.

That milestone is intentionally smaller than full pipeline execution, but it
tests the hard invariant: stacking must be OOF and aligned by identity, never by
position.
