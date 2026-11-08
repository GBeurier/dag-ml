# DAG-ML contributor instructions

## Ownership and contracts

- DAG owns generic Rust/C-ABI compile -> plan -> fit-CV -> select -> refit -> predict -> explain, variants, FoldSet scheduling, deterministic control RNG, OOF/leakage, search, selection, aggregation, prediction/score persistence, lineage/cache/replay.
- [Methods](../nirs4all-methods/AGENTS.md) owns numerical kernels and splitter algorithms; [DMD](../dag-ml-data/AGENTS.md) owns data/identity constraints, not ML phases. Sibling links are optional navigation.
- Keep generic aggregation/scoring/selection/persistence native, not Python/JS shims. Aggregation includes per-fold predictions, `avg`/`w_avg` ensembles and `final` refit rows; extend this owner, not a new persistence project.
- Operators stay external: hosts/controllers own fitting, transforms, augmentations, data backends and native integrations. Hosts own language-specific fitted binaries and call owner kernels; Methods state is portable. Host parity requires the same skeleton in every language: "torch from R" must behave like "torch from Python", with the same DAG/persistence and only the controller differing. This is a parity obligation, not a universal qualification claim.
- Metric definitions/scoring context stay DAG; Methods supplies primitives. Preserve the accepted generic identity-aligned conformal score/quantile/application exception, not blanket numerical ownership. Shared fixtures qualify named profiles.
- No NIRS-specific core logic or inspection/materialization of feature matrices, images, spectra, tensors, sequences, graphs or fitted internals. Buffers/models/artifact refs/views remain opaque handles; canonical descriptors/fingerprints remain data.
- Identity, predictions and `y_true` may cross as Arrow-compatible tables. Join/split/merge by stable sample/group/target/origin identity, never row position. Preserve fold/group/repetition/augmentation-origin/refit leakage checks.
- Schedule splitters through campaign-plan controller calls producing a FoldSet, never ordinary transform nodes. A `requires_oof` edge needs validation predictions in PredictionStore; unsafe train-prediction features require explicit traceable opt-in and tests, never raw-handle forwarding.
- The old SDK-touch prohibition does not block an already authorized owner fix and affected propagation. Stay within current user scope; no unauthorized local fallback. Clean V1 removes obsolete JSON/API alternatives only after callers migrate and parity is established; supported low schema/ABI numbers are not legacy. Update version/migration/refusal policy for wire changes.

## Gate and targeted commands

- Use existing crate patterns, targeted tests first and small validated slices. Run the local gate before handing work back:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
python3 scripts/check_so_freshness.py
python3 scripts/validate_contracts.py
```

- Freshness fails if tracked `_dag_ml.abi3.so` predates Rust sources. The stdlib contract validator checks local schemas/fixtures without a sibling; when the sibling exists, also run `DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py`. CI uses that sibling for schema/conformance/header drift.
- Tests include inline modules, `crates/dag-ml-core/src/runtime/tests.rs` and `crates/dag-ml-capi/tests/`. Targeted invocations are:

```bash
cargo test -p dag-ml-core
cargo test -p dag-ml-capi
cargo test -p dag-ml-cli
cargo test -p dag-ml-core <substring_of_test_name>
cargo test -p dag-ml-core runtime::tests::<name> -- --exact --nocapture
cargo run -p dag-ml-cli -- <subcommand> --help
```

- CLI smoke recipes in `examples/README.md`/DEVELOPMENT include validate-bundle, run-mock-campaign, run-process-replay, compile-pipeline-dsl and export-research-provenance.

## Architecture and ABI navigation

- Read [DEVELOPMENT](docs/DEVELOPMENT.md) and [COORDINATOR_SPEC](docs/COORDINATOR_SPEC.md) first; the latter resolves ambiguity in older design docs. [ABI](docs/ABI.md) and the [conformal ADR](docs/adr/ADR-20-conformal-calibration-ownership.md) define their boundaries. W3C PROV/Workflow Run RO-Crate/OpenLineage/MLMD are exports, not the internal execution model.
- `dag-ml-core` is pure Rust with no host runtime dependencies; its flat lib.rs re-exports are consumed through the facade by C ABI and the sole CLI. Do not create a parallel public surface. Typed NodeTask/NodeResult JSON or C ABI vtables connect host controllers to variant/fold scopes.
- Schemas under `docs/contracts/` include graph_spec, pipeline_dsl, campaign_spec, execution_plan, controller_manifest, coordinator_data_plan_envelope, node_task/result, selection_policy/decision, prediction_cache metadata, process_adapter profiles and research/OpenLineage profiles. Canonical DSL/graph/campaign examples use `examples/`, fixtures `examples/fixtures/`, outputs `examples/generated/`; adapter smokes include python_process_controller, sklearn_process_controller and flaky_process_controller under `examples/adapters/`.
- Contract changes update schema, local fixture, Rust type, exposed C ABI version and shared conformance pack together with DMD. Preserve portable artifact/replay validation without serializing arbitrary host ML objects in core.
- ABI implementation/header are `crates/dag-ml-capi/src/lib.rs` and `crates/dag-ml-capi/include/dag_ml.h`. Rust outputs use `dagml_string_free`, `dagml_owned_bytes_free`, `dagml_f64_tensor_free` or `dagml_f64_columnar_tensor_free`; host handles use their vtable release callback. Controller/artifact-store/prediction-cache v1/v2/v3 distinguish borrowed/owned user_data lifecycle; consult ABI.md before extending them.

- Context: `docs/RATIONALE.md`, `docs/ARCHITECTURE.md`; `docs/design/README.md` is historical, not current contracts.

## Working rules

- Read applicable local and deeper instructions before editing. Preserve dirty/concurrent work; make authorized owner fixes and propagate affected contracts/bindings/consumers. Do not reset another author's work.
- Use CodeGraph FIRST for implementation navigation only if this checkout's root already has `.codegraph/`; use its exact projectPath or run the CLI here. Never auto-index or silently substitute an ancestor index.
- Unindexed code and documentation use ordinary source navigation. Explicit source, tests and build tools are final authority. Owner links are navigation; this policy works without a parent workspace.
- All application/project content is English: UI text/subtitles/tooltips/errors, docs/comments/docstrings, test/fixture prose, reports/issues/PRs, commits and generated material. Preserve identifiers, quotations and historical/provenance meaning; converse with the maintainer in French.
- Do not commit credentials, private data, generated workspaces, caches or build outputs. Follow local licensing/fixture restrictions and report actual commands/exits/skips; a command reference grants no execution/publication permission.

