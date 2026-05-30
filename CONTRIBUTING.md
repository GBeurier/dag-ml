# Contributing to dag-ml

`dag-ml` is the Rust execution coordinator for leakage-safe, reproducible ML pipelines. It owns graph compilation, phases, folds, OOF joins, the controller ABI, lineage, caching, and the deterministic control RNG. It does **not** own source storage or feature buffers — those contracts live in the sibling [`dag-ml-data`](https://github.com/GBeurier/dag-ml-data) repository.

Read [`docs/TOC.md`](docs/TOC.md) for the navigation map and [`docs/COORDINATOR_SPEC.md`](docs/COORDINATOR_SPEC.md) for the normative product contract before changing any contract.

## Development environment

```bash
# Rust toolchain (edition 2021, MSRV 1.83)
rustup toolchain install stable 1.83.0
rustup component add rustfmt clippy

# Optional: Python/R adapter dependencies for the host-controller smokes
python3 -m pip install numpy scikit-learn
```

## The green gate

Every change must pass the full gate (it mirrors `.github/workflows/ci.yml`):

```bash
cargo fmt --all --check
cargo +1.83.0 check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py
python3 scripts/release/check_publish_plan.py --dry-run
```

Targeted iteration:

```bash
cargo test -p dag-ml-core <substring_of_test_name>
cargo test -p dag-ml-core runtime::tests::<name> -- --exact --nocapture
cargo run -p dag-ml-cli -- <subcommand> --help    # CLI smoke surface
```

## Repository layout

```
crates/
  dag-ml-core/   pure-Rust contracts, validation, runtime (all real logic)
  dag-ml/        stable Rust facade re-exporting core
  dag-ml-capi/   C ABI surface + header (include/dag_ml.h)
  dag-ml-cli/    validation + smoke-execution CLI
docs/
  TOC.md             navigation map
  COORDINATOR_SPEC.md normative product contract (source of truth on ambiguity)
  ARCHITECTURE.md / ABI.md / STATUS.md / TEST_PLAN.md / ROADMAP.md
  adr/               Architecture Decision Records (see adr/README.md)
  contracts/         JSON Schemas (shared with dag-ml-data — JSON-identical)
examples/
  minimal_graph.json, pipeline_dsl_*.json, campaign_*.json
  adapters/          host controller scripts (Python + R)
  generated/         CLI-produced bundle/cache/selection outputs
  fixtures/          reusable test fixtures
```

## Adding a new controller / operator

1. Define the operator's `ControllerManifest` (capabilities, operator selectors, supported phases, data requirements). See `examples/controller_manifests.json` for the shape and `examples/adapters/sklearn_process_controller.py` for a production adapter.
2. Wire the operator selector so the planner routes the operator to your controller.
3. Add a fixture-driven test under the relevant crate's `tests/` and, if it exercises OOF, an entry in `docs/OOF_FIXTURES.md`.
4. Never weaken OOF, fold, group, repetition, augmentation-origin, or refit leakage checks for convenience (see [ADR-05](docs/adr/ADR-05-repetition-cv-invariant.md)).

## Changing a contract (schema / C header / conformance pack)

These are **cross-repo contracts**. A contract change is a coordinated, dual-PR operation:

1. Update the JSON Schema in `docs/contracts/`, the Rust type, the C ABI version macro in `crates/dag-ml-capi/include/dag_ml.h` (only if the wire shape changes), and the conformance pack.
2. Mirror the change in the sibling `dag-ml-data` repo so the shared artifacts stay **JSON-identical**.
3. Run `DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py` — it fails on drift.
4. Update `docs/STATUS.md` (Implemented / Not implemented / Next) and the relevant ADR.
5. Schema/wire-shape evolution follows [ADR-02](docs/adr/ADR-02-schema-evolution-sla.md): land additively first, promote with a version bump and a dual-read window.
6. Open the paired PR in `dag-ml-data`; both must merge together.

## Adding a new error variant

Errors follow the [ADR-11](docs/adr/ADR-11-error-taxonomy.md) unified taxonomy.
A new `DagMlError` variant in `crates/dag-ml-core/src/error.rs` must update **all**
of these in the same change (the compiler enforces match exhaustiveness; the
`scripts/check_error_taxonomy.py` gate enforces the rest):

1. `taxonomy_parts()` — `(category, code, severity)`; the category must be one of
   the ten ADR-11 categories.
2. `remediation_hint()` — one actionable sentence (part of the public surface).
3. `context()` — structured debug fields (identifiers/counts, never raw data).
4. `numeric_taxonomy()` — a stable `(category_id, code_id)`; category_id matches
   the category string (validation=0 … internal=9) and code_id is unique within
   the category. **Never renumber a shipped pair.**
5. If the category is new to the Python surface, add a `create_exception!`
   subclass in `crates/dag-ml-py/src/lib.rs`, map it in
   `dag_ml_error_type_for_category`, export it in `python/dag_ml/__init__.py` /
   `.pyi`, and mirror in `dag-ml-data` if applicable.
6. Add binding tests: Rust `error_code` value, the C ABI `dagml_last_error_json`/
   `dagml_last_error_code` round-trip, and the Python `isinstance` subclass.

CHANGELOG: new variants land under "added"; renaming/removing one is a breaking
change under the [ADR-14](docs/adr/ADR-14-deprecation-policy.md) deprecation cycle.

## Shipping a host-controller binding

Host controllers run over the JSONL process-adapter protocol or the C ABI vtable. See `examples/adapters/` for Python (`*.py`) and R (`*.R`) references and `docs/ABI.md` for the vtable lifecycle. New adapters must respect the security boundary in [ADR-13](docs/adr/ADR-13-process-adapter-security.md) (executable allowlist, env sanitization, timeout/kill) and the artifact rules in [ADR-16](docs/adr/ADR-16-artifact-security.md) (declared serialization backend, no silent pickle).

## Pull-request rules

- Run the green gate locally before pushing.
- If you changed a contract, confirm the paired `dag-ml-data` PR and that `validate_contracts.py` is green.
- If you changed `Cargo.toml`, run `python3 scripts/release/check_publish_plan.py --dry-run`.
- Update `CHANGELOG.md` under `[Unreleased]`.
- If you changed a decision, add or supersede an ADR in `docs/adr/`.
- Every new `#[deprecated]` carries a removal version and a removal-test ([ADR-14](docs/adr/ADR-14-deprecation-policy.md)).
- Every production-path `TODO`/`FIXME` uses `TODO(owner): reason (#issue)` and
  passes `python3 scripts/check_deprecations.py`.
- Every new public item carries an invariant-grade doc comment (lead with the invariant, then failure modes, then example).

## Architecture Decision Records

Significant decisions are recorded in [`docs/adr/`](docs/adr/README.md). Eighteen Phase-0 ADRs fix the contract for the nirs4all integration. To change a decision, write a new ADR that explicitly supersedes the old one — never edit an accepted ADR silently.

## Releases

Releases follow the cross-repo release train in [ADR-10](docs/adr/ADR-10-release-train.md): `dag-ml-data` publishes first, `dag-ml`'s pinned version bumps, then `dag-ml` publishes. Version/tag policy is documented there.
