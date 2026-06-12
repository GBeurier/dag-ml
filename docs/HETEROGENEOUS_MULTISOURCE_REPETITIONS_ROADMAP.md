# Roadmap DAG-ML - Heterogeneous Multi-Source Repetitions

Date: 2026-06-12
Status: implementation roadmap, review-required before coding
Reference design: `/home/delete/nirs4all/nirs4all/docs/_internal/specifications/heterogeneous_multisource_repetitions.md`
Scope: DAG-ML only. No NIRS-specific behavior and no challenge-specific protocol.

## Objective

Extend DAG-ML so the control core can coordinate pipelines where one physical sample has multiple observations per source, with asymmetric cardinalities across sources. The core must remain a coordinator:

- no feature-buffer materialization in Rust core;
- no fitted model internals in Rust core;
- identities, relations, prediction tables, fold/OOF contracts, reducers, selection, lineage and replay are core responsibilities;
- host adapters own feature matrices, transforms, imputation, padding, model fitting and fitted objects.

The primary output is sample-level prediction. Observation/source/combo-like units are intermediate domains for reducers, metrics, meta-features and audit.

## Decisions

- Repetitions are exchangeable by default.
- Targets are sample-level for this feature.
- Missing source at prediction time is policy-driven: default is warning plus declared fallback (`impute_declared`, `mask`, `partial_model` or representation-specific padding). `strict` is available but not the only behavior.
- Provenance/origin must always be preserved.
- Mainline implementation: `combo` is represented as a derived observation/unit in the relation table, with `component_observation_ids` and `origin_sample_id`. This covers cartesian training, combo-to-sample reducers, replay and audit without changing public prediction-cache semantics.
- Deferred public decision: promoting `combo` and `source_sample` to first-class `PredictionLevel` / `PredictionUnitId` values is a breaking/public-contract decision and is only required for graph designs where a meta-model consumes combo rows directly as public prediction units.
- DAG-ML and nirs4all have separate implementations. DAG-ML may redesign clean contracts rather than mirror nirs4all legacy.
- Public schema/ABI changes must follow ADR-02: additive fields first, schema versioning, dual-read when needed, CHANGELOG, fixtures and conformance.

## Current Anchors

Existing code already provides partial foundations:

- `crates/dag-ml-core/src/relation.rs`: `SampleRelation`, `SampleRelationSet`, `origin_sample_id`, augmentation-origin leakage checks.
- `crates/dag-ml-core/src/policy.rs`: `PredictionLevel`, `AggregationMethod`, `AggregationPolicy`, `AggregationWeights`, leakage/data-shape policies.
- `crates/dag-ml-core/src/aggregation.rs`: prediction aggregation and validation.
- `crates/dag-ml-core/src/oof.rs`: `PredictionBlock`, partitions, OOF contracts.
- `crates/dag-ml-core/src/graph.rs`: `NodeKind`, `PortKind`, `PortSpec`, `EdgeContract`.
- `crates/dag-ml-core/src/dsl.rs`: pipeline DSL, merge semantics, prediction vs feature joins.
- `crates/dag-ml-core/src/runtime.rs`: phase orchestration, refit/predict replay, OOF edge validation.
- `crates/dag-ml-core/src/selection.rs`: selection policies and decisions.
- `docs/contracts/*.schema.json`: public JSON contracts.
- `scripts/validate_contracts.py`: local and optional `dag-ml-data` conformance.

## Public Contract Matrix

Every row must be checked before coding and updated in the same PR that changes the Rust type.
D8 is an audit/conformance phase, not a catch-up phase: every D1-D7 phase that changes a public type must update its schema, fixtures, bindings/ABI decision and conformance tests in that same phase.

| Surface | Likely changes | Compatibility rule |
| --- | --- | --- |
| New internal/entity enum, e.g. `EntityUnitLevel` | add `physical_sample`, `source_sample`, `observation`, `combo` for relation/view/reducer domains | additive Rust type and schemas where surfaced |
| `dag-ml-core::policy::PredictionLevel` | keep current public prediction-cache levels in the mainline; only add `combo` / `source_sample` if the deferred public decision is accepted | decision-required; likely schema/cache/ABI migration |
| `dag-ml-core::relation::SampleRelation` / `SampleRelationSet` | add unit level, rep id, derived unit id, component observations, quality, sample influence | additive fields with defaults; schema version |
| `graph_spec.schema.json` | port/edge unit level, alignment key, target level, relation contract | optional fields first; validation stricter only when present |
| `pipeline_dsl.schema.json` | representation nodes, reducers, missingness, fit influence | additive DSL blocks |
| `node_task.schema.json` / `node_result.schema.json` | relation deltas, representation fingerprints, fit influence policy, backend capabilities | versioned fixtures |
| `selection_policy.schema.json` / `selection_decision.schema.json` | `EvaluationScope`, `RefitSlotPlan`, reduction id, unit level | additive fields, dual-read default to current behavior |
| prediction cache metadata schemas | unit ids, relation fingerprint, reduction id, optional evaluation scope | additive in the mainline; public `combo` prediction level is deferred |
| `coordinator_data_plan_envelope.schema.json` / `data_plan.schema.json` | relation table extensions and data-provider view contracts | schema version + conformance |
| `feature_fusion_selector.schema.json` | combination/fusion plans beyond sample-id joins | additive selector variants |
| C ABI header/snapshot | only if relation/prediction unit tables cross ABI | bump ABI snapshot and tests |
| Python bindings / `dag-ml-py` | expose new schema fields and validation helpers | update package fixtures even if crate excluded from workspace |
| Wasm/CLI/examples | validate and inspect new graph contracts | smoke fixtures and docs |
| `dag-ml-data` | feature fusion/combination selectors and conformance | update cross-repo pack |

## Phase D0 - ADR and Compatibility Ledger

Goal: freeze vocabulary and migration rules before schema work.

Work:

- Write an ADR for unit-level vocabulary:
  - `physical_sample`;
  - `source_sample`;
  - `observation`;
  - `combo`;
  - `EntityUnitLevel`;
  - `PredictionUnitId`;
  - `ReductionPlan`;
  - `RepresentationPlan`;
  - `FitInfluencePolicy`.
- Record the explicit mainline decision: `combo` is a derived observation/unit, not a public `PredictionLevel`.
- Record the deferred decision gate: first-class public `combo` / `source_sample` prediction levels require explicit approval, cache/schema migration notes, and ABI/binding impact review.
- Add an ADR-02 migration checklist for this feature:
  - schema fields optional first;
  - default/dual-read behavior;
  - fixture update;
  - CHANGELOG entry;
  - C ABI decision.

Files:

- `docs/adr/`
- `docs/COORDINATOR_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/contracts/README.md`
- `docs/CHANGELOG` or existing changelog path.

Definition of Done:

- contract names are frozen;
- downstream phases use derived-observation combos by default and isolate the first-class-combo option as a later decision;
- schema/ABI review checklist is explicit.

## Phase D1 - Relation Schema and Unit Identity

Goal: make the current relation model able to carry asymmetric repetitions and derived combo rows.

Work:

- Extend `SampleRelation` or introduce a versioned companion record with:
  - `unit_level`;
  - `unit_id`;
  - `observation_id`;
  - `sample_id`;
  - `source_id`;
  - `rep_id`;
  - `target_id`;
  - `group_id`;
  - `origin_sample_id`;
  - `derived_unit_id`;
  - `component_observation_ids`;
  - `sample_influence_weight`;
  - `quality_flag`;
  - `is_augmented`.
- Update closed JSON schemas with defaults and versions.
- Add relation fingerprinting.
- Validate:
  - unique unit ids;
  - component observations belong to the same physical sample;
  - a derived combo is stored as a relation-backed observation/unit until the explicit first-class-combo decision is accepted;
  - target sample-level consistency;
  - origin/augmentation never crosses train/validation folds;
  - no empty source ids or invalid rep ids.

Files:

- `crates/dag-ml-core/src/relation.rs`
- `crates/dag-ml-core/src/fold.rs`
- `docs/contracts/coordinator_data_plan_envelope.schema.json`
- `docs/contracts/data_plan.schema.json`
- `examples/fixtures/`
- `scripts/validate_contracts.py`

Tests:

- repeated observations validate under sample split;
- combo components cannot cross sample boundary;
- derived combo rows reuse observation-to-sample aggregation paths;
- augmentation origin leakage fails;
- schema round-trip for old and new relation records;
- conformance with optional missing fields.

Definition of Done:

- `A=2/B=3/C=2` can be represented as relations only;
- replay/audit has enough provenance for combos;
- public prediction caches do not need a new `PredictionLevel` in the mainline path;
- old relation fixtures still validate or are dual-read migrated.

## Phase D2 - Unit-Typed Ports and Illegal Join Rejection

Goal: reject row-position joins across incompatible unit domains.

Work:

- Extend `PortSpec` / `EdgeContract` with optional:
  - `unit_level`;
  - `alignment_key`;
  - `target_level`;
  - `relation_contract`;
  - `allows_broadcast`;
  - `missingness_policy`.
- Validate:
  - relation-aware graphs require `SampleRelationSet` / relation table, `relation_fingerprint`, unit metadata and `alignment_key`;
  - relation-aware graphs include asymmetric multisource repetitions, source-aware late fusion, `cartesian`, `stack_*`, non-sample-level feature/prediction joins, or any representation producing multiple rows per physical sample;
  - feature joins require same unit/alignment or explicit representation node;
  - prediction joins require compatible prediction/unit metadata;
  - combo -> sample requires a reducer;
  - sample -> combo broadcast is explicit;
  - graph validation reports actionable errors.

Files:

- `crates/dag-ml-core/src/graph.rs`
- `crates/dag-ml-core/src/plan.rs`
- `crates/dag-ml-core/src/dsl.rs`
- `docs/contracts/graph_spec.schema.json`
- `docs/contracts/pipeline_dsl.schema.json`

Tests:

- asymmetric multisource graph without relations is rejected;
- graph with missing `relation_fingerprint` or `alignment_key` is rejected;
- invalid observation/sample feature concat rejected;
- sample-level late fusion accepted;
- combo-level prediction join rejected without declared adapter;
- derived-observation combo -> sample reducer accepted;
- sample-to-combo broadcast accepted only with explicit adapter.

Definition of Done:

- unsafe positional joins cannot silently pass in a relation-aware profile, and relation-aware plans cannot omit relation metadata.

## Phase D3 - Reducers / `ReductionPlan` Contract

Goal: align existing `AggregationPolicy` with design-level reducers.

Work:

- Extend aggregation/reduction contracts with:
  - `role=score|persist|fold_ensemble|meta_feature|final_output`;
  - `axis=unit|fold|model|metric`;
  - `input_unit_level`;
  - `output_unit_level`;
  - `method`;
  - `weight_source`;
  - `task_compatibility`;
  - custom controller spec.
- Reconcile core and `dag-ml-data` vocabulary:
  - `mean`;
  - `weighted_mean`;
  - `median`;
  - `vote`;
  - `robust_mean`;
  - `exclude_outliers`;
  - `custom`.
- Validate reducer outputs:
  - producer;
  - partition;
  - fold id;
  - target names;
  - unit id order;
  - output unit level.

Files:

- `crates/dag-ml-core/src/policy.rs`
- `crates/dag-ml-core/src/aggregation.rs`
- `crates/dag-ml-core/src/runtime.rs`
- `docs/contracts/aggregation_controller_task.schema.json`
- `docs/contracts/aggregation_controller_result.schema.json`
- `docs/adr/ADR-07-aggregation-reducers.md`
- `../dag-ml-data` aggregation contracts/fixtures.

Tests:

- observation -> sample;
- combo -> sample;
- fold ensemble;
- model ensemble;
- classification vote;
- weighted reducer rejects bad weights;
- custom reducer task/result validation.

Definition of Done:

- scoring, final output, fold ensembles and meta-feature preparation use the same reducer contract.

## Phase D4 - OOF, Fold and Selection Safety Before Executable Representations

Goal: ensure cartesian/stacking representations cannot run unsafely.

Work:

- Extend fold policies with `split_unit=physical_sample`.
- Ensure all observations/combos of a physical sample share fold.
- Extend `PredictionBlock` / prediction cache metadata with:
  - `prediction_level`;
  - `unit_ids`;
  - relation fingerprint;
  - `evaluation_scope`;
  - reduction id if already reduced.
- Add `EvaluationResult` typed by:
  - metric;
  - partition;
  - evaluation scope;
  - reduction id;
  - unit level.
- Add `RefitSlotPlan`:
  - `refit_one`;
  - `refit_ensemble`;
  - `selection_level`;
  - `member_count`;
  - `selection_metric`;
  - `reduction_plan`.
- Add `StackingFitContract`:
  - `meta_training_features=oof`;
  - `inference_features=refit_base_predictions`;
  - `selection_protocol=nested|holdout|reuse_oof`;
  - `base_prediction_calibration=none|rank|calibrated_oof_to_refit`.
- Refuse in-sample base predictions as meta-training features by default.
- Support two meta row domains in the mainline:
  - `meta_row_domain=sample`: default production path; branch predictions are reduced/broadcast to sample-level meta-features.
  - `meta_row_domain=combo`: relation-backed derived-observation path; branch adapters map sample/source/combo predictions onto combo rows, training still uses OOF only, `FitInfluencePolicy` is explicit, and final output must declare a `combo -> sample` reducer.
- Deferred only: exposing combo rows as a public `PredictionLevel::Combo` / standalone prediction-cache domain.

Files:

- `crates/dag-ml-core/src/fold.rs`
- `crates/dag-ml-core/src/oof.rs`
- `crates/dag-ml-core/src/runtime.rs`
- `crates/dag-ml-core/src/selection.rs`
- `crates/dag-ml-core/src/bundle.rs`
- `docs/contracts/selection_policy.schema.json`
- `docs/contracts/selection_decision.schema.json`
- prediction cache metadata schemas.

Tests:

- combo OOF from a model trained on the same physical sample rejected;
- row-level metric cannot select sample-level refit unless reduced;
- `refit_one` and `refit_ensemble` produce distinct decisions;
- stacking meta receives only OOF in training;
- `meta_row_domain=sample` late fusion;
- `meta_row_domain=combo` relation-backed with adapters and mandatory final `combo -> sample` reducer;
- final-validation profile rejects `reuse_oof` selection protocols;
- fold-alignment mismatch rejected.

Definition of Done:

- representations that multiply rows cannot be marked executable until sample-level OOF safety and reducer-backed selection are in place.

## Phase D5 - Fit Influence Policy to Controllers

Goal: make sample influence a fit contract, not an aggregation weight.

Work:

- Add `FitInfluencePolicy` / `SampleInfluencePolicy`:
  - `auto`;
  - `uniform_rows`;
  - `equal_sample_influence`;
  - `resample_equalized`;
  - `backend_loss_weight`;
  - `scorer_only`;
  - `strict_weight_support`.
- Add controller capability declarations:
  - supports sample weights;
  - supports row resampling;
  - supports backend loss weights;
  - supports missing masks.
- Add fields to:
  - `ControllerManifest`;
  - `NodeTask`;
  - `ModelInputSpec`;
  - node result diagnostics.
- Runtime behavior:
  - if policy is unsupported and `strict_weight_support`, fail;
  - if `auto`, choose declared fallback and persist warning;
  - never reinterpret `AggregationWeights` as loss weights.

Files:

- `crates/dag-ml-core/src/controller.rs`
- `crates/dag-ml-core/src/policy.rs`
- `crates/dag-ml-core/src/runtime.rs`
- `docs/contracts/controller_manifest.schema.json`
- `docs/contracts/model_input_spec.schema.json`
- `docs/contracts/node_task.schema.json`
- `docs/contracts/node_result.schema.json`
- C ABI/Python bindings if manifests/tasks cross those surfaces.

Tests:

- controller without weight support fails under strict;
- `auto` emits persisted fallback warning;
- equal sample influence produces expected per-row weights in task metadata;
- aggregation weights and fit weights are kept separate.

Definition of Done:

- a backend cannot silently ignore requested sample influence.

## Phase D6 - Representation Nodes and dag-ml-data Fusion Contracts

Goal: represent feature-domain choices without core data materialization.

Prerequisite: D1-D5 complete enough to validate unit, OOF and fit influence contracts. Representation nodes may be parsed earlier, but not executable.

Work:

- Add representation specs:
  - `AggregateRepresentation`;
  - `CartesianProductRepresentation`;
  - `MonteCarloCartesianRepresentation`;
  - `StackFixedRepresentation`;
  - `StackPaddedMaskedRepresentation`.
- Defer `RaggedBagRepresentation` until the relation/reducer/replay path is stable. It is a larger model-input refactor, not required for this feature to be usable.
- Add `CombinationPlan` / fusion selector variants:
  - `cartesian`;
  - `zip`;
  - `match_by`;
  - `sample_k`;
  - `reference_broadcast`;
  - budget/cap;
  - seed;
  - component ids;
  - missing source policy.
- Coordinate with `dag-ml-data`:
  - data-provider selectors;
  - `feature_fusion_selector.schema.json`;
  - Arrow/data-plan fixtures;
  - host materialization responsibilities.
- Representation nodes emit:
  - host data handle;
  - relation delta;
  - lineage/provenance;
  - feature schema fingerprint;
  - cardinality contract.
- Cartesian and Monte Carlo representations emit derived combo relation rows. They must not require a public `PredictionLevel::Combo` in the mainline implementation.
- Representation nodes also emit a replay manifest:
  - `CombinationPlan`;
  - physical sample -> source observation mapping;
  - `combo_selection`;
  - seed;
  - caps/budgets;
  - QC/outlier policy references;
  - missing source/repetition policy;
  - prediction representation;
  - final output unit level.

Files:

- `crates/dag-ml-core/src/graph.rs`
- `crates/dag-ml-core/src/data.rs`
- `crates/dag-ml-core/src/plan.rs`
- `crates/dag-ml-core/src/controller.rs`
- `docs/contracts/feature_fusion_selector.schema.json`
- `docs/contracts/coordinator_branch_view.schema.json`
- `docs/contracts/data_output_provenance.schema.json`
- `../dag-ml-data` selectors/contracts/fixtures.

Tests:

- cartesian representation emits relation delta with component ids;
- Monte Carlo representation deterministic by seed;
- stack fixed rejects cardinality mismatch;
- padded/masked stack requires mask-aware controller capability;
- cartesian representation replay regenerates the same derived combo ids from `CombinationPlan` plus seed;
- replay manifest round-trips through schema fixtures in the same phase;
- dag-ml-data conformance for each fusion selector.

Definition of Done:

- core plans representation work as contracts and host handles only;
- dag-ml-data can validate/materialize corresponding data views.

## Phase D7 - Missingness and Serve-Time Replay

Goal: make train/predict cardinality differences explicit and replayable.

Work:

- Missingness policies:
  - `strict`;
  - `drop_incomplete`;
  - `impute_declared`;
  - `mask`;
  - `partial_model`;
  - `pad`.
- Default for this feature: warning + declared fallback when policy is not strict.
- Persist:
  - representation replay manifest from D6;
  - mapping of physical samples to observations per source;
  - `CombinationPlan`, `combo_selection`, seed and caps;
  - QC/outlier policy state references;
  - train-time and predict-time representation compatibility result;
  - warning severity;
  - fallback used;
  - affected source/repetition/sample counts;
  - representation compatibility outcome.
- Validate:
  - fixed-width representations require compatible cardinality or pad/mask;
  - cartesian can vary combo count only if final reducer stabilizes output;
  - late fusion can drop/impute branches only by policy.

Files:

- `crates/dag-ml-core/src/policy.rs`
- `crates/dag-ml-core/src/runtime.rs`
- `crates/dag-ml-core/src/bundle.rs`
- replay/prediction cache schemas.

Tests:

- missing repetition predict;
- missing source predict;
- `fit -> OOF -> refit -> predict` with same repetitions;
- `fit -> OOF -> refit -> predict` with different repetitions;
- replay refuses if a required representation manifest is missing or fingerprint-mismatched;
- strict fails;
- impute/mask/partial-model replay succeeds with warning artifact;
- relation fingerprint mismatch rejected.

Definition of Done:

- serve-time mismatch is never silent and never untraceable.

## Phase D8 - Final Schemas, ABI, Bindings and Conformance Audit

Goal: prove all public surfaces stayed coherent. This phase must not be the first place where D1-D7 schema/binding/ABI work happens.

Work:

- Audit that every D1-D7 change already updated schemas and fixtures for:
  - graph spec;
  - pipeline DSL;
  - campaign/execution plan;
  - node task/result;
  - controller manifest;
  - model input;
  - data plan/envelope;
  - feature fusion selector;
  - prediction cache metadata;
  - selection policy/decision;
- Confirm `abi_snapshot.v1.json`, C header and tests were updated in the phase that crossed ABI, or explicitly record "ABI not touched".
- Confirm Python bindings and Wasm/CLI validation were updated in the phase that exposed schema fields.
- Update `docs/STATUS.md`, `docs/TEST_PLAN.md`, `docs/OOF_FIXTURES.md`, `docs/ROADMAP.md`.
- Add conformance pack scenarios:
  - `A=2/B=3/C=2`;
  - sample-level late fusion;
  - cartesian combo -> sample reducer;
  - missing source with fallback;
  - stacking OOF contract;
  - invalid unit join;
  - row-vs-sample selection mismatch.

Validation:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
python3 scripts/validate_contracts.py
DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py
```

Definition of Done:

- Rust, JSON schema, CLI, C ABI if touched, Python if touched, Wasm if touched and dag-ml-data conformance all agree.
- no D1-D7 public contract delta remains unpaired with a schema/fixture/conformance update.

## Phase D9 - Golden DAGs and Negative E2E

Goal: prove the feature is usable and safe.

Golden DAGs:

1. `per_source_aggregate -> source models -> sample-level reducer`.
2. `late_fusion_by_source -> prediction join -> meta model`.
3. `cartesian_full -> model -> combo_to_sample reducer`.
4. `cartesian_mc -> deterministic replay`.
5. `stack_fixed -> strict cardinality`.
6. `stack_padded_masked -> missing repetition`.
7. `combo-meta-post` relation-backed with adapters, explicit fit influence and final sample reducer.

Negative tests:

- leakage fold violation;
- incompatible unit join;
- row-level metric selected as sample-level refit;
- missing source without fallback;
- train/predict relation fingerprint mismatch;
- controller lacks required fit-influence capability;
- prediction cache missing unit ids;
- C/Python/CLI schema rejection for invalid plans.

Files:

- `examples/`
- `examples/fixtures/`
- `crates/dag-ml-core/src/runtime/tests.rs`
- `crates/dag-ml-capi/src/tests.rs` if ABI touched
- Python binding tests if touched
- `docs/TEST_PLAN.md`

Definition of Done:

- every golden DAG validates and mock-runs through fit/OOF/refit/predict;
- golden replay covers same repetitions and different repetitions at predict time;
- every negative fixture fails for the intended reason.

## Phase D10 - Deferred Public/Breaking Extensions

These items are deliberately outside the no-regression mainline and require explicit approval before implementation.

- Promote `combo` and/or `source_sample` to public `PredictionLevel` / `PredictionUnitId` values:
  - required only for graph designs where downstream public prediction caches, selectors or meta-models consume combo rows directly;
  - touches prediction cache schemas, `PredictionBlock`, JSON fixtures, CLI/Wasm validation, Python bindings and possibly C ABI snapshots;
  - must follow ADR-02 and ADR-14 with versioning, dual-read or migration edge, CHANGELOG and negative fixtures.
- Add `RaggedBagRepresentation` / native multi-instance model inputs:
  - requires host/controller capability negotiation for ragged tensors or masks;
  - requires new replay, model-input and data-provider contracts;
  - not needed for `per_source_aggregate`, late fusion, cartesian full or Monte Carlo cartesian.
- Add first-class public combo prediction/cache domain:
  - not required for relation-backed `meta_row_domain=combo`;
  - final public output remains sample-level unless an explicit alternate serving contract is approved.

## Rollout Order

1. D0: ADR, derived-combo decision, compatibility ledger.
2. D1: relation schema and unit identity.
3. D2: typed ports and illegal join rejection.
4. D3: reducers as graph contracts.
5. D4: OOF/fold/selection safety.
6. D5: fit influence to controller contracts.
7. D6: representation nodes and dag-ml-data fusion contracts.
8. D7: missingness and serve-time replay.
9. D8: final schemas/ABI/bindings/conformance audit; no D1-D7 contract catch-up.
10. D9: golden and negative E2E.
11. D10: deferred breaking/public extensions, only after approval.

## Risks

- Accidentally adding spectroscopy-specific behavior. Mitigation: source ids stay opaque.
- Materializing feature buffers in core. Mitigation: host handles + relation deltas only.
- Schema drift with `dag-ml-data`. Mitigation: conformance pack and `DAG_ML_DATA_REPO`.
- ABI drift. Mitigation: explicit ABI decision row and snapshot updates.
- Representation nodes before OOF safety. Mitigation: D6 not executable until D1-D5 pass.
- Fit influence ignored by controllers. Mitigation: capability checks in manifest/task/runtime.
- Premature first-class `combo` public level. Mitigation: use relation-backed derived observations in D0-D9 and keep D10 as an explicit decision gate.

## Non-Regression Gate

Mandatory:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
python3 scripts/validate_contracts.py
```

When cross-repo contracts are touched:

```bash
DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py
```

When ABI/bindings are touched:

```bash
cargo test -p dag-ml-capi
cargo test -p dag-ml-py
```

## Mainline Closure Criteria

The feature is considered implemented without public-breaking changes when:

- `A=2/B=3/C=2` compiles into relation-backed derived combos and sample-level outputs;
- split, score, selection, refit, export and replay default to sample-level semantics;
- cartesian rows are grouped by physical sample and reduced by identity, never by position;
- missing sources in prediction emit declared warnings/fallback artifacts;
- `DAG_ML_DATA_REPO=../dag-ml-data python3 scripts/validate_contracts.py` passes when shared contracts change;
- all D10 items remain disabled unless explicitly approved.
