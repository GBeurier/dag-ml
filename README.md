# dag-ml

Rust-first execution core for leakage-safe, in-process ML pipelines.

`dag-ml` owns the graph, phases, folds, OOF joins, controller ABI, lineage,
cache and deterministic control RNG. It does not own source storage or feature
buffers; those contracts live in the companion `dag-ml-data` repository.

> Status: active core scaffold. The project has executable Rust crates, C ABI
> graph/selection/bundle validation, CLI validation, coordinator
> planning/runtime contracts, data-plan fingerprints, OOF leakage checks,
> deterministic selection, ADR-11 structured error descriptors and first
> versioned refit/replay bundle contracts with CLI build/validate commands.
> Host controller adapters are still pending.

## Repository Layout

```text
crates/
  dag-ml-core/      # graph, phase, OOF, selection, bundle and control contracts
  dag-ml/           # Rust facade re-exporting stable core APIs
  dag-ml-capi/      # C ABI surface and header for host/controller integration
  dag-ml-py/        # PyO3/maturin JSON-contract bindings for Python hosts
  dag-ml-wasm/      # wasm-bindgen JSON-contract bindings for browser hosts
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
cargo +1.83.0 check --workspace --all-targets
cargo test --workspace
cargo test -p dag-ml-wasm
# dag-ml-py is excluded from the workspace (abi3-py311); test it standalone:
PYO3_PYTHON=python3.11 cargo test --manifest-path crates/dag-ml-py/Cargo.toml
python3 scripts/validate_release_metadata.py
python3 scripts/check_error_taxonomy.py
python3 scripts/check_deprecations.py
python3 scripts/check_public_docs.py
python3 scripts/release/check_publish_plan.py --dry-run
python3 scripts/validate_abi_snapshot.py
cargo audit --deny warnings
python3 -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
cargo run -p dag-ml-cli -- validate-bundle --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
(cd crates/dag-ml-py && maturin build --release --features extension-module --out ../../target/wheels)
python3 scripts/smoke_python_wheel_metadata.py target/wheels/dag_ml-*.whl
python3 scripts/smoke_python_bindings.py      # after installing the built wheel
python3 scripts/smoke_python_integration.py ../dag-ml-data  # after installing dag-ml + dag-ml-data wheels
node_out_dir="$PWD/target/wasm/dag-ml-wasm"
wasm-pack build crates/dag-ml-wasm --target nodejs --out-dir "$node_out_dir" --release
node scripts/smoke_wasm_bindings.cjs "$node_out_dir"
web_out_dir="$PWD/crates/dag-ml-wasm/pkg-web"
rm -rf "$web_out_dir"
wasm-pack build crates/dag-ml-wasm --target web --out-dir "$web_out_dir" --release
node scripts/smoke_wasm_web_bindings.mjs "$web_out_dir"
(cd crates/dag-ml-wasm && wasm-pack pack --pkg-dir pkg-web .)
node scripts/smoke_wasm_tarball_metadata.mjs "$web_out_dir"
data_web_out_dir="$PWD/target/wasm-web/dag-ml-data-wasm"
wasm-pack build ../dag-ml-data/crates/dag-ml-data-wasm --target web --out-dir "$data_web_out_dir" --release
node scripts/smoke_wasm_tarball_metadata.mjs "$data_web_out_dir"
node scripts/smoke_wasm_integration.mjs "$web_out_dir" "$data_web_out_dir" ../dag-ml-data
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

## License

`dag-ml` is dual-licensed open-source — **`CeCILL-2.1 OR AGPL-3.0-or-later`** (your choice). See
[`LICENSING.md`](LICENSING.md), the full texts under [`LICENSES/`](LICENSES/), third-party
attributions in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and the licensing decision in
[`docs/adr/ADR-18-licensing.md`](docs/adr/ADR-18-licensing.md).
