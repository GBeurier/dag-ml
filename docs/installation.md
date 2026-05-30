# Installation And Local Gates

## Prerequisites

- Rust stable plus the MSRV toolchain `1.83.0`.
- Python 3.11 or newer for validation scripts, Sphinx and Python binding
  smokes.
- Node.js 20 or 22 plus `wasm-pack` for WASM package smokes.
- A sibling `../dag-ml-data` checkout for cross-repo contract and browser
  integration smokes.

## Core Validation

```bash
cargo fmt --all --check
cargo +1.83.0 check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py
python3 scripts/validate_release_metadata.py
python3 scripts/validate_abi_snapshot.py
```

## Documentation Site

```bash
python3 -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Python And WASM Packages

```bash
(cd crates/dag-ml-py && maturin build --release --features extension-module --out ../../target/wheels)
python3 scripts/smoke_python_wheel_metadata.py target/wheels/dag_ml-*.whl

wasm-pack build crates/dag-ml-wasm --target web --out-dir crates/dag-ml-wasm/pkg-web --release
(cd crates/dag-ml-wasm && wasm-pack pack --pkg-dir pkg-web .)
node scripts/smoke_wasm_tarball_metadata.mjs crates/dag-ml-wasm/pkg-web
```
