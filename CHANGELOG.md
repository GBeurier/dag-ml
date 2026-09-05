# Changelog

All notable changes to `dag-ml` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Contract/wire-shape changes follow [ADR-02](docs/adr/ADR-02-schema-evolution-sla.md);
deprecations follow [ADR-14](docs/adr/ADR-14-deprecation-policy.md).

## [Unreleased]

## [0.3.24] - 2026-09-05

### Fixed

- Keep Methods HPO campaign provenance stable across downstream JSON map
  feature unification, retaining published default-build identities and all
  strict resume checks.
- Serialize concurrent native Results V2 writers, refuse replacement atomically,
  publish the manifest last, and roll back this writer's files on I/O errors.
- Make all registry publications depend on the complete same-SHA CI gate;
  refresh source-attestation packs without regenerating scientific fixtures.
- Align the advertised workspace MSRV to Rust 1.85, already required by the
  locked cross-repository test dependencies, and qualify the locked graph.

### Added

- Candidate `dafb8b6` adds an archive-bound, identity-validated
  multi-target conformal presentation projection. It presents already
  calibrated point/interval blocks without recalculating calibration,
  quantiles, or joins in the host.
- Native Methods replay now accepts N4MM format 1 raw PLS and format 2
  `SNV(ddof=0) -> Savitzky-Golay(mode=interp) -> PLS` artifacts. Format 2
  requires the ABI 2.5 typed pipeline descriptor and is refused rather than
  delegated to a Python fallback when its signed semantics do not match.
- Release patch 0.3.23: the native results crate and Python facade can read and
  write validated result files while preserving physical sample identities.
  Native Methods HPO now reports each scheduler-owned validation fold as step
  `0..N-1`, stops a candidate immediately when the native pruner decides, and
  checkpoints/resumes only through the official opaque `N4MOPT` state. The
  progressive route refuses absent, empty, resampled, non-partitioning or
  nested fold topologies before creating a tuner session. The Archive V2
  integration test keeps its explicit registry-only
  `nirs4all_archive_core = "=0.3.22"`
  baseline; it is not a claim of qualification against the release-train Core.
  Native N4MM and N4MOPT references now carry capability-derived Methods ABI
  minima (PLS 2.0+, optimizer checkpoints 2.2+, imported-linear/Ridge 2.3+),
  retain a narrow historical PLS dual-read, and fail closed before native
  materialization when the payload minimum and runtime capability disagree.

- Release patch 0.3.21: Python hosts can execute the strict callback-free
  Methods CV → REFIT → terminal PREDICT facade for raw numeric arrays with
  explicit IDs, one numeric target, no-shuffle KFold PLS and a separate X-only
  cohort. It preflights identities and feature widths before native execution,
  permits only ephemeral internal CV OOF scoring, refuses external OOF use and
  unsupported transforms/generators/HPO/calibration/groups/metadata, and
  returns frozen native V2 result and receipt objects bound to the terminal
  run. This patch introduces no C ABI or JSON-schema delta.

- Release patch 0.3.20: Python hosts can execute a closed CV → REFIT → terminal
  PREDICT replay for a V2 identity-attested external cohort. The core seals the
  bundle receipt, reuses only REFIT artifacts, and refuses selector errors,
  V1/targetless cohorts, OOF-dependent graphs, and observation/aggregation
  output paths before a host callback executes. This is a dag-ml-only patch
  under ADR-10: it introduces no paired dag-ml-data contract or C ABI delta and
  consumes the already synchronized V2 envelope.

- Release patch 0.3.19: native Methods Ridge stacking now retains its nested
  OOF/refit evidence and replays the selected portable model without a stale
  runtime handle.

- Release patch 0.3.18: the serial runtime now executes leakage-safe nested
  stacking OOF campaigns natively, retaining inner and outer fold evidence as
  distinct scheduler-owned resources.

- Release patch 0.3.15: promote callback-free native Methods Package V3 refit replay into the public DAG-ML runtime boundary for Core and Studio hosts.

- Release patch 0.3.14: native global OOF scoring now reconstructs each
  semantic unit across every validation fold before invoking the selected
  reducer and metric. The coordinator therefore makes a single attested
  selection decision over global sample/group/target units rather than
  averaging per-fold scores.

- Release patch 0.3.13: Python hosts can validate and replay target-bound
  native full-refit Package V3 contracts without a host callback. The native
  coordinator retains the exact refit recipe, portable Methods artifact
  closure and conformal presentation evidence; ZIP persistence remains owned
  by `nirs4all-core`.

### Fixed

- Native HPO resume merges the checkpoint's completed-trial prefix into the
  resumed evidence before validation and continuation. This preserves the
  official opaque `N4MOPT` state while preventing a valid completed prefix from
  being rejected or terminalized twice.
- Release patch 0.3.22: strict terminal Archive V2 persistence now emits a
  verified empty prediction-cache payload set when internal CV OOF evidence was
  intentionally discarded and the bundle and graph prove no external OOF
  requirement. It never invents nonempty OOF evidence or persists a terminal
  receipt; this is a dag-ml-only fix with no C ABI or JSON-schema delta.

- Release patch 0.3.17: conformal replay now verifies canonical independently
  rounded decimal endpoints while retaining the frozen report-only radius
  closure for historical evidence that does not carry point predictions.

- Release patch 0.3.16: TCV1 replay fingerprints now preserve their exact
  binary64 values across JSON serialization, including dynamic native Methods
  score values, so saved native packages reload and predict without a
  fingerprint drift.

- Release patch 0.3.9: target-free PREDICT replay retains the validated
  conformal intervals calibrated from the source training evidence.

- Release patch 0.3.8: conformal calibration fingerprints now stabilize their
  strict TCV1 float representation before signing, so native residuals remain
  self-validating after JSON serialization and replay.

- Release patch 0.3.7: the Python binding now exposes native conformal
  calibration attachment. The Rust coordinator derives and validates the full
  replay, cohort and provenance context; hosts supply only identity-keyed truth
  values and relationships.

- Release patch 0.3.6: the published Python Methods runtime now registers the
  native HPO controller alongside the PLS controller when an attested Methods
  HPO campaign requests it, preserving controller-owned scheduling, checkpoint
  and resume semantics.

- Release patch 0.3.5: public Python Methods PLS training and stateless native
  package replay now expose the portable runtime path without a host callback.

- Release patch 0.3.4: native training-loss role references are now attested
  in `TrainingRequest` and lowered into `ExecutionPlan` before FIT_CV/REFIT
  task construction.

- Release patch 0.3.3: Methods HPO now verifies the native runtime overlay
  before consulting host controller or provider state, preserving fail-closed
  behavior when native optimizer support is unavailable. This is a dag-ml-only
  hotfix; the pinned `dag-ml-data` contract is unchanged.

### Added

- ADR-20 and W0 JSON contracts for conformal calibration ownership,
  `ParameterPatch`, `OutputBinding`, `TrainingInfluenceManifest`, complete
  `TrainingOutcome`/`ReplayOutcome` payloads, and the existing execution-bundle
  and portable prediction-cache wire shapes.
- Canonical refit, no-refit, predict, class-probability, explain and conformal
  fixtures with TCV1 fingerprints, exact finite-sample rank/float-token
  goldens, leakage and stale-predictor refusals, and cross-contract semantic
  validation. The refit and no-refit fixtures retain a positive two-branch
  portable OOF cache, validated with an offline Draft 2020-12 registry and
  native-equivalent plan/requirement/cache/payload cross-links. Estimator
  validation now closes transitively over the predictor graph: selected variant
  leaf patches, per-node/per-fold FIT_CV lineage, per-node REFIT lineage,
  fit/selection influence, capability-driven artifacts and conformal predictor
  data/artifact bindings must agree exactly. This initial W0 fixture slice was
  contract-only; the same Unreleased line now also includes the native Rust
  conformal surface below.
- Native Rust split-conformal regression contracts and exact kernels for
  finite-sample ranks, marginal/joint calibration, tagged unbounded intervals,
  interval application and metrics. Public C, Python and WASM conformal
  bindings remain future work.
- Portable W0.6 contracts and positive/negative fixtures for four-role cohort
  manifests, identity-aligned conformal prediction blocks, conformal-specific
  metrics, domain assessments, explicit application decisions and three-mode
  robustness reports. They separate exchangeability assumptions, predictor
  state, calibrator state and coverage guarantees; bind recalibration to full
  calibration artifacts and disjoint identity closures; and require exact
  severity-zero/all-slice baselines, metric coordinates and provenance.
  Structural replacements now resolve against the baseline predictor closure
  and carry distinct plan, graph and selected-variant identities; recalibration
  diagnostics bind the exact scenario, binary64 severity and calibration input.
  The production point-only fixture uses label-free prediction-mean shift and
  explicitly carries no conformal evidence.
- Fixture-only numeric evidence now closes point predictions, interval
  midpoints, truth and conformal metrics one-to-one for marginal, joint and
  small-sample unbounded cases. Independent validators reconstruct empirical
  coverage, gap, mean/median width and Winkler score, and re-fingerprinted truth
  and translated-midpoint attacks fail at the intended numeric invariant. The
  resolved report matrix now contains 18 exact result/block/metric coordinates,
  while an additional valid structural case demonstrates explicit calibrator
  invalidation.
- Versioned conformal/robustness conformance pack with path-traversal refusal,
  sorted unique artifact paths, file-byte SHA-256 values and an omit-self TCV1
  checksum. The pack includes transitive calibration/output/influence
  contracts, exact rank and regression-metric reconstruction goldens, the
  independent Python semantic oracle, a frozen Philox4x32-10 counter oracle,
  and a locked isolated Rust oracle that proves TCV1 and restricted RFC
  8785/JCS remain distinct. CI runs the contract validator, conformal, RNG and
  cross-language Python suites, and the Rust oracle. The portable wire layer
  remains binding-neutral; native Rust conformal kernels now consume its
  semantics, while public C/Python/WASM conformal bindings are still pending.

## [0.2.0] - 2026-06-15

### Added
- `docs/adr/` — eighteen Phase-0 Architecture Decision Records fixing the
  contract for the nirs4all backend integration (compatibility ledger, schema
  evolution SLA, separation-branch semantics, tag/exclude masks, repetition-CV
  invariant, signal-type ownership, aggregation reducers, session persistence,
  docs stack, release train, error taxonomy, observability, process-adapter
  security, deprecation policy, GIL/async, artifact security, cutover/rollback,
  licensing). See `docs/adr/README.md`.
- `docs/adr/ADR-19-multisource-unit-vocabulary.md` (heterogeneous multi-source
  repetitions roadmap, phase D0) freezes the unit vocabulary
  (`physical_sample`, `source_sample`, `observation`, `combo`,
  `EntityUnitLevel`, `PredictionUnitId`, `ReductionPlan`, `RepresentationPlan`,
  `FitInfluencePolicy`), records that combos are relation-backed derived
  observations rather than a public `PredictionLevel`, gates the first-class
  public combo/source level behind a separate explicit public-contract
  decision, and adds the per-feature ADR-02 migration checklist. Vocabulary and
  ledger surfaced in `docs/COORDINATOR_SPEC.md`, `docs/ARCHITECTURE.md` and
  `docs/contracts/README.md`.
- Heterogeneous multi-source repetitions roadmap D1 extends coordinator
  relation records with `EntityUnitLevel`, optional unit/source/rep/combo
  provenance, component observation ids, sample influence weights, quality
  flags, deterministic relation fingerprinting, schema/conformance updates and
  an A=2/B=3/C=2 multisource fixture while keeping public prediction levels
  unchanged.
- Heterogeneous multi-source repetitions roadmap D2 adds optional unit-typed
  graph/DSL port metadata, relation edge contracts, broadcast/missingness edge
  policy fields and graph validation that rejects incompatible unit/alignment
  joins unless an explicit relation-backed adapter contract is declared.
- Heterogeneous multi-source repetitions roadmap D3 adds the optional
  `ReductionPlan` contract for reducer role/axis/unit/method metadata, exposes
  `robust_mean` and `exclude_outliers` reducer vocabulary, validates
  controller task/result plan echoes and supports relation-backed combo
  observation reductions to physical samples without adding public combo
  prediction levels.
- Heterogeneous multi-source repetitions roadmap D4 switches the default
  leakage split vocabulary to explicit `physical_sample`, adds optional OOF
  evaluation/refit/stacking selection contracts, and extends prediction-cache
  metadata with relation/reduction/evaluation lineage markers.
- Heterogeneous multi-source repetitions roadmap D5 adds optional
  `FitInfluencePolicy` contracts to model inputs and node tasks, explicit
  controller capabilities for sample weights/resampling/backend loss weights
  and missing masks, and node-result diagnostics so unsupported strict
  weighting fails while `auto` fallbacks are visible.
- Heterogeneous multi-source repetitions roadmap D6 adds representation and
  combination plan contracts for cartesian, sampled-cartesian, fixed-stack and
  padded/masked-stack data views, plus replay manifests and data-output
  provenance fields so hosts can materialize heterogeneous fusion views while
  the core validates only identities, fingerprints and policy metadata.
- Heterogeneous multi-source repetitions roadmap D7 adds representation
  compatibility reports for train/predict replay, explicit fallback severity
  and affected-count metadata, and bundle/data-output validation that rejects
  silent fixed-width, cartesian-count or late-fusion missingness mismatches.
- Heterogeneous multi-source repetitions roadmap D8 adds the final public
  surface audit and conformance-pack scenarios for A=2/B=3/C=2 multisource
  relations, sample-level late fusion, cartesian combo reduction, missing-source
  fallback, stacking OOF, invalid unit joins and row-vs-sample selection
  mismatch. The audit records that D1-D7 were additive JSON/Rust contract
  changes and did not add new C ABI, Python or WASM entry points.
- Heterogeneous multi-source repetitions roadmap D9 adds golden runtime
  fixtures for per-source aggregate, late fusion, full and sampled cartesian,
  fixed and padded/masked stack and combo-meta-post flows; runtime mock-run
  coverage through FitCv/OOF/Refit/Predict replay; and negative tests for
  relation replay drift, unit-join schema rejection, row-vs-sample selection,
  missing prediction-cache unit ids and missing fit-influence capability.
- Governance: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `.github/` issue/PR templates, `CODEOWNERS`, `dependabot.yml`,
  `examples/README.md` audience matrix.
- Canonical `FoldSet` fingerprints exposed in Rust, Python and WASM bindings so
  OOF partitions can participate in replay and lineage checks.
- Shared `FoldSet` fixture and contract validation keep the canonical
  fingerprint byte-identical with `dag-ml-data` for the common JSON shape.
- Shared parity-oracle handoff manifest pins the first nirs4all-lite parity
  cases, fixtures, Python/WASM gates and invariants for the future consumer
  compatibility ledger.
- Python/WASM `contract_manifest_json()` exposes the versioned integration
  surface, supported contract ids, exported helper names and shared fixture
  digests for host/browser compatibility checks.
- ADR-11 structured error descriptors now expose stable `category`, `code`,
  `severity`, remediation hints and context in Rust, Python exception
  attributes, WASM error payloads and C ABI `DagMlError` refusals.
- Web-target WASM integration smoke composes `dag-ml` with sibling
  `dag-ml-data`, validates manifests/fold fingerprints, builds a coordinator
  data-plan envelope, compiles the nirs4all-compatible DSL fixture and builds a
  scheduler-ready execution plan.
- Python package facade now exposes validated contract wrappers
  (`PipelineDslSpec`, `GraphSpec`, `CampaignSpec`, `ControllerManifests`,
  `ExecutionPlan`, `FoldSet`) and typed compile/plan helpers on top of the
  stable JSON functions.
- Installed-wheel Python integration smoke composes `dag_ml` with
  `dag_ml_data` through the typed facades, builds a nirs4all-lite data-plan
  envelope, compiles the nirs4all-compatible DSL fixture and validates the
  resulting execution plan.
- Python wheel metadata smoke now validates built wheel name/version,
  `Requires-Python`, license files, `abi3` tag, native extension, stubs and
  `py.typed` before install smokes run.
- CI now gates Rust documentation with `RUSTDOCFLAGS="-D warnings" cargo doc`
  and runs a workspace package dry-run so publishability regressions fail
  before release.
- Sphinx/MyST documentation site scaffold (`docs/conf.py`, `docs/index.md`,
  `docs/installation.md`, `docs/requirements.txt`) now builds in CI with
  warnings denied, closing the ADR-09 local docs gate before hosted publishing.
- ADR-14 managed-debt lint (`scripts/check_deprecations.py`) now rejects
  unexplained production-path `TODO`/`FIXME` markers and unmanaged
  `#[deprecated]` attributes in CI.
- Public Rust doc coverage now has a ratcheted CI gate
  (`scripts/check_public_docs.py`), making the current docstring debt visible
  without claiming the final 95% target is complete.
- ADR-10 publish-plan check (`scripts/release/check_publish_plan.py`) validates
  workspace internal dependency pinning and runs `cargo publish --dry-run` for
  currently publishable root crates before release.
- CI now gates the declared Rust MSRV with `RUST_MSRV: "1.83.0"` and
  `cargo check --workspace --all-targets`.
- CI now gates Rust dependencies with pinned `cargo-audit` and
  `cargo audit --deny warnings`.
- Web-target WASM packages are packed with `wasm-pack pack` in CI after smoke
  loading, so npm tarball regressions are caught before release.
- WASM npm tarball dry-run metadata smoke validates package name/version,
  integrity, bundled-dependency absence and required published files for both
  local and cross-repo browser packages.
- WASM smokes now validate generated npm metadata (`package.json` name,
  version, JS entry, typings, packaged files and required TypeScript exports)
  against the Rust contract manifest.
- Release metadata validation now checks Cargo workspace inheritance, internal
  path-dependency versions, Python PEP 440 wheel version, `abi3-py311`, MSRV
  pins, MSRV-sensitive dependency pins, CI tool pins, required governance files
  and the Sphinx docs-site / managed-debt / publish-plan gates before release.
- Public C ABI header snapshot validation now locks `dag_ml.h` through a
  checked-in SHA-256 manifest so ABI changes are explicit in review.
- The shared conformance pack now requires the producer-side
  `dagmldata_coordinator_multi_target_arrow_json` symbol from `dag-ml-data`,
  so multi-output target export is an explicit integration capability.

### Fixed
- Workspace path dependencies now carry explicit SemVer requirements, so
  `cargo package --workspace --allow-dirty --no-verify` succeeds for all Rust
  crates instead of failing at publish packaging time.
- `crates/dag-ml-cli/tests/cli_contracts.rs` — formatting brought back under
  `cargo fmt --all --check` (the green gate now passes clean).
- Controller YAML parsing now uses `yaml_serde` instead of the RustSec-flagged
  `serde_yml/libyml` stack.

## [0.1.0-alpha.0] - 2026-05-29

Initial active core scaffold. Executable Rust crates with:

### Added
- **Graph** model + validation (acyclicity, port-kind contracts, parallel-level
  computation).
- **Plan** — `GraphPlan`, `CampaignSpec`, `ExecutionPlan`, `NodePlan`,
  `NodeTask`, `NodeResult`; variant enumeration, split invocation, phase
  schedules.
- **Runtime** — sequential and parallel schedulers (byte-for-byte identical
  outputs) over `(variant, fold)` scopes; `PredictionStore` joining OOF
  predictions by stable `sample_id`.
- **OOF / leakage safety** — `requires_oof` edge enforcement; train predictions
  refused as meta-model training features by default; identity-only fold
  assignments; augmentation-origin and group constraints.
- **Selection** — deterministic variant ranking from persisted OOF metrics.
- **Bundle / replay** — `ExecutionBundle` locking plan/data/artifact/controller
  fingerprints; file-backed artifact manifest and prediction-cache payloads
  with SHA-256 tamper detection.
- **Provenance** — W3C PROV, Workflow Run RO-Crate, and OpenLineage export
  derived from validated lineage.
- **Pipeline DSL compiler** — linear/branching/generation/augmentation; nirs4all
  JSON import.
- **C ABI** (`crates/dag-ml-capi`, `include/dag_ml.h`) — versioned controller,
  artifact-store, and prediction-cache vtables (v1/v2/v3 ownership lifecycle).
- **CLI** (`dag-ml-cli`) — graph/bundle/replay validation and smoke execution.
- 23 JSON Schemas in `docs/contracts/`, shared JSON-identical with
  `dag-ml-data` and validated by `scripts/validate_contracts.py`.

### Not yet implemented
- EXPLAIN phase executor (lineage/provenance export exists; per-node explain
  dispatch is scaffolded).
- Production host controllers beyond the sklearn / prospectr / mdatools
  references.
- Direct Python/YAML DSL frontends (JSON-only parser today).

[Unreleased]: https://github.com/GBeurier/dag-ml/compare/v0.3.24...HEAD
[0.3.24]: https://github.com/GBeurier/dag-ml/compare/v0.3.23...v0.3.24
[0.2.0]: https://github.com/GBeurier/dag-ml/releases/tag/v0.2.0
[0.1.0-alpha.0]: https://github.com/GBeurier/dag-ml/releases/tag/v0.1.0-alpha.0
