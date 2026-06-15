# Audit release finale: dag-ml + dag-ml-data

Date: 2026-06-15
Train cible: `0.2.0`

## Verdict

`dag-ml` et `dag-ml-data` sont fermes pour une release `0.2.0` locale:
contrats Rust/JSON, C ABI, CLI, Python/WASM JSON bindings, fixtures croisees,
docs de support, gates release et probes perf sanity sont alignes.

La release reste volontairement scopee. Elle ne promet pas les frontends
Python/YAML objet, les adapters SpectroChemPy/Orange-Spectroscopy, les chemins
stateful `msc`/`simca`/`mcrals`, ni les providers host-filtered au-dela du
backend in-memory de conformance. Ces elements sont du backlog post-0.2.0, pas
des bloqueurs du tag `0.2.0`.

## Fermetures realisees

| Ancien risque | Etat 0.2.0 |
|---|---|
| Versioning encore alpha/RC | Ferme: Cargo, Python, provider Python et R sidecar sont bumpes en `0.2.0`; pas de RC. |
| Promesse publique ambigue | Ferme: README, STATUS, ROADMAP et `SUPPORTED.md` de chaque repo decrivent la surface supportee et le backlog. |
| ASan manquant cote `dag-ml-capi` | Ferme: lane AddressSanitizer ajoutee pour `dag-ml-capi`; `dag-ml-data-capi` avait deja sa lane. |
| Performance non mesuree | Ferme pour release sanity: probes ignorees en release mode documentees dans les deux repos. Benchmarks formels restent post-0.2.0. |
| Drift aggregation `dag-ml` / `dag-ml-data` | Ferme par documentation: mapping explicite, `robust_mean` / `exclude_outliers` restent data-side seulement. |
| Docs release absentes | Ferme: `SUPPORTED.md`, `PERFORMANCE.md`, `AGGREGATION_INTEROP.md` et cet audit sont integres aux index/toctrees. |
| Metadata release non contraignante | Ferme: `validate_release_metadata.py` impose les lanes ASan et le bump coherent. |

## Surface supportee 0.2.0

### Surface dag-ml

- Graph, campaign et execution-plan contracts.
- Fold identity, OOF joins, leakage refusal, group/origin/repetition guards.
- Deterministic selection, replay bundle validation and provenance exports.
- C ABI JSON-contract helpers with ABI snapshot validation.
- CLI validation flows.
- Runtime process-adapter JSONL protocol and reference adapters in conformance
  scope.
- Python and WASM JSON-contract bindings and metadata smokes.
- Pipeline DSL JSON compiler for canonical and nirs4all-compatible descriptors.

### Surface dag-ml-data

- Dataset schema, ids, axes, representation contracts and fingerprints.
- Model-input planning and adapter path solving.
- Coordinator data-plan envelope v1.
- Sample relations and FoldSet boundary validation.
- Numeric feature buffers, `.n4d` persistence and N-D tensor transport.
- Fitted adapter refs/manifests/store.
- In-memory provider vtable as conformance backend.
- Python `ctypes` provider package as conformance template.
- Arrow IPC feature-buffer reader as conformance path.

## Public signatures

No public ABI/schema/Rust/Python/WASM signature was changed by the hardening
work. The only public metadata change is the package version bump to `0.2.0`.

Downstream impact:

- no ABI/schema migration is required;
- downstream package chains that pin package versions should rebuild and test
  against `dag-ml==0.2.0` and `dag-ml-data==0.2.0`;
- `nirs4all-lite`, `nirs4all-web`, browser smokes and Python wheel smokes are
  the integration chains to validate before publishing/tagging.

## Gates a executer pour tag

### Gates dag-ml

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p dag-ml-core oof_join_large_campaign_under_1500ms --release -- --ignored --nocapture
cargo test -p dag-ml-core build_execution_plan_large_linear_graph_under_1500ms --release -- --ignored --nocapture
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
python3.11 scripts/validate_release_metadata.py
python3.11 scripts/check_error_taxonomy.py
python3.11 scripts/check_deprecations.py
python3.11 scripts/check_public_docs.py
python3.11 scripts/validate_abi_snapshot.py
DAG_ML_DATA_REPO=../dag-ml-data python3.11 scripts/validate_contracts.py
python3.11 -m sphinx -W --keep-going -b html docs docs/_build/html
```

### Gates dag-ml-data

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p dag-ml-data-core fingerprint_large_buffer_under_500ms --release -- --ignored --nocapture
cargo test -p dag-ml-data-core tensor_fingerprint_large_payload_under_500ms --release -- --ignored --nocapture
cargo run -p dag-ml-data-cli -- fingerprint-schema examples/minimal_schema.json
python3.11 scripts/validate_release_metadata.py
python3.11 scripts/check_error_taxonomy.py
python3.11 scripts/check_deprecations.py
python3.11 scripts/check_public_docs.py
python3.11 scripts/validate_abi_snapshot.py
DAG_ML_REPO=../dag-ml python3.11 scripts/validate_contracts.py
python3.11 -m sphinx -W --keep-going -b html docs docs/_build/html
```

### Packaging et aval

```bash
python3.11 scripts/release/check_publish_plan.py --dry-run
(cd crates/dag-ml-py && maturin build --release --features extension-module --out ../../target/wheels)
python3.11 scripts/smoke_python_wheel_metadata.py target/wheels/*.whl
wasm-pack build crates/dag-ml-wasm --target web --out-dir crates/dag-ml-wasm/pkg-web --release
(cd crates/dag-ml-wasm && wasm-pack pack --pkg-dir pkg-web .)
node scripts/smoke_wasm_tarball_metadata.mjs crates/dag-ml-wasm/pkg-web
```

Run the equivalent wheel/WASM packaging commands in `dag-ml-data` first, then
run the cross-repo Python/WASM integration smokes and local `nirs4all-lite` /
`nirs4all-web` rebuilds if those repos pin the old package versions.

## Post-0.2.0 backlog

- Raise public Rust documentation coverage toward the ADR target.
- Extend ASan/lifecycle tests beyond current C ABI unit coverage.
- Promote sanity perf probes into repeatable benchmark jobs.
- Decide whether `robust_mean` / `exclude_outliers` need a shared coordinator
  schema.
- Wire expected signal type through paired `dag-ml` replay contracts.
- Implement production providers for host-filtered branch views.
- Add SpectroChemPy, Orange-Spectroscopy, `msc`, `simca` and `mcrals` adapters
  when they become release scope.
