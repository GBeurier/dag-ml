# Shared Contracts

This directory contains wire-contract artifacts shared with `dag-ml-data`, plus
DAG-ML-specific publication schemas. `dag-ml` remains the consumer and semantic
validator: it checks fingerprints, campaign fold membership, OOF boundaries and
leakage policies before any controller receives a handle.

## Heterogeneous Multi-Source Vocabulary and Evolution

The heterogeneous multi-source repetitions roadmap
(`docs/HETEROGENEOUS_MULTISOURCE_REPETITIONS_ROADMAP.md`) extends several of the
schemas below with optional unit-level fields.
`docs/adr/ADR-19-multisource-unit-vocabulary.md` freezes the vocabulary
(`physical_sample`, `source_sample`, `observation`, `combo`, `EntityUnitLevel`,
`PredictionUnitId`, `ReductionPlan`, `RepresentationPlan`, `FitInfluencePolicy`)
and records the mainline decision that combos are relation-backed derived
observations rather than a public `PredictionLevel`. Each phase that touches a
contract here follows the ADR-19 / ADR-02 checklist: optional fields first,
defaults/dual-read, fixture and conformance-pack update, CHANGELOG entry, and an
explicit C ABI decision. A first-class public `combo` / `source_sample`
prediction level remains deferred and gated.

## Coordinator Data Plan Envelope v1

Schema: `coordinator_data_plan_envelope.schema.json`

Canonical fixture: `examples/fixtures/data/coordinator_data_plan_envelope_nir.json`

Conformance pack: `conformance_pack.v1.json`

Parity oracle handoff: `parity_oracle.v1.json`

Public C ABI snapshot: `abi_snapshot.v1.json`

Runtime type consumed here: `ExternalDataPlanEnvelope`

Producer type in `dag-ml-data`: `CoordinatorDataPlanEnvelope`

The envelope binds a data plan to stable schema, plan and relation
fingerprints. It may carry coordinator relation records for sample, target,
group, origin, source and augmentation identity. The JSON Schema documents the
portable shape of that envelope; Rust validation enforces the stronger semantic
rules that depend on the active campaign.

Short-term policy: both repositories keep a JSON-identical conformance fixture
for this envelope plus a copy of the v1 schema, and test that the published
artifact declares the Rust-supported version. `scripts/validate_contracts.py`
compares the fixture and schema copies when `DAG_ML_DATA_REPO` points to a
sibling checkout, validates the shared conformance-pack digests, and CI checks
out that peer explicitly. When development moves into a monorepo, this file
should become a single generated or shared contract artifact used by both
crates.

The D8 multisource audit extends the shared conformance pack with seven named
scenarios: `multisource_a2_b3_c2.v1`, `sample_level_late_fusion.v1`,
`cartesian_combo_to_sample_reducer.v1`, `missing_source_with_fallback.v1`,
`stacking_oof_contract.v1`, `invalid_unit_join.v1` and
`row_vs_sample_selection_mismatch.v1`. These scenarios bind the D1-D7 public
surface changes to schema digests, canonical fixtures and concrete Rust/contract
test references. They intentionally remain metadata: the core still validates
relations, fingerprints, OOF safety and representation replay without owning
feature buffers or host model objects.

## Parity Oracle v1

Manifest: `parity_oracle.v1.json`

This is the producer-side handoff for the future `nirs4all` compatibility
ledger. It does not wire `nirs4all`; instead it names the parity cases,
fixtures, Python/WASM gates and invariants that the consumer ledger must bind
to public API rows before bridge work starts. `scripts/validate_contracts.py`
checks the manifest shape, verifies referenced `dag-ml`/`dag-ml-data` fixtures
when the sibling checkout is present, pins its digest in
`conformance_pack.v1.json`, and requires the manifest to stay byte-identical
across both repositories.

## Public C ABI Snapshot v1

Snapshot: `abi_snapshot.v1.json`

Header: `crates/dag-ml-capi/include/dag_ml.h`

`scripts/validate_abi_snapshot.py` checks the header SHA-256 against the
snapshot and runs in CI. Any C ABI header change must update this manifest in
the same review so downstream hosts can see the ABI movement explicitly.
The shared `dag-ml-data` conformance pack also requires the producer-side
multi-target Arrow helper
`dagmldata_coordinator_multi_target_arrow_json`, which is consumed as a data
provider capability rather than as a `dag-ml` header symbol.

## Coordinator Branch View v1

Schema: `coordinator_branch_view.schema.json`

Mirrors `dag-ml-data`'s `coordinator_branch_view.v1` byte-for-byte except for
the `$id` (each repo declares its own domain). The normalized SHA-256 (with
`$id` stripped) is pinned identically in both repos' `conformance_pack.v1.json`
so the wire contract cannot drift. `BranchViewPlan` records in
`dag-ml`/`crates/dag-ml-core/src/data.rs` accept the same shape and the
in-memory `dag-ml-data` provider executes `by_source` natively.

## Fitted Adapter Ref v1

Schema: `fitted_adapter_ref.schema.json`

Mirrors `dag-ml-data`'s `fitted_adapter_ref.v1` byte-for-byte except for the
`$id`. Same normalized-SHA-256 enforcement in both `conformance_pack.v1.json`
files. The producer type is `FittedAdapterRef` in `dag-ml-data`; `dag-ml`
relies on this schema to validate fitted-adapter records that flow through
the data layer at refit time.

## Feature Fusion Selector v1

Schema: `feature_fusion_selector.schema.json`

Canonical fixture: `examples/fixtures/data/feature_fusion_selector_nir_chem.json`

Runtime shape passed through data-provider `feature_arrow` when the provider
supports `dag-ml-data` multi-source fusion:
`{ schema_version, feature_set_id, sources, alignment, combination_plan?,
representation_plan?, policy? }`, where each source maps a `source_id` to a
provider-owned `feature_set_id` and optional column subset. The optional D6
plans describe host-owned representation work such as cartesian, sampled
cartesian, fixed stack and padded/masked stack materialization; the core
validates identity, unit, replay and provenance contracts without materializing
feature buffers. This keeps `DagMlDataVTable` ABI-compatible while making
feature fusion explicit.

## GraphSpec v1

Schema: `graph_spec.schema.json`

Canonical fixture: `examples/branch_merge_oof_graph.json`

Runtime type: `GraphSpec`

C ABI: `DAG_ML_GRAPH_SPEC_SCHEMA_VERSION`,
`dagml_graph_spec_contract_json`, `dagml_graph_validate_json`

This is the portable graph object produced by the DSL compiler and consumed by
the execution-plan builder. The schema documents node kinds, ports, edge
contracts, OOF prediction edges and lineage propagation flags so host bindings
can reject malformed graph JSON before controller resolution or scheduling.
Rust validation remains the semantic authority for uniqueness, endpoint checks,
port-kind alignment and cycle refusal.

Prediction-stacking meta-nodes reserve the optional node metadata key
`stacking_oof_refit_contract`. Its current shape is
`{"policy": "require_full_coverage" | "cv_only" | "skip_refit_on_incomplete_oof"}`.
The default is `require_full_coverage`: REFIT consumes validation OOF only when
the producer covers the complete refit sample universe. `cv_only` always skips
the stacking node during REFIT, while `skip_refit_on_incomplete_oof` skips only
when otherwise well-formed validation OOF is incomplete. Invalid OOF still fails
with a stable cause such as `partial_oof_without_policy` or
`non_validation_partition`; Rust validates the metadata object and the OOF
coverage semantics.

## Pipeline DSL v1

Schema: `pipeline_dsl.schema.json`

Canonical compatibility fixture: `examples/pipeline_dsl_nirs4all_compat.json`

Runtime parser: `parse_pipeline_dsl_json`

C ABI: `DAG_ML_PIPELINE_DSL_SCHEMA_VERSION`,
`dagml_pipeline_dsl_contract_json`, `dagml_pipeline_dsl_validate_json`,
`dagml_pipeline_dsl_compile_json`,
`dagml_pipeline_dsl_compile_artifact_json`,
`dagml_pipeline_dsl_execution_plan_build_json`

This is the public input contract for both canonical `PipelineDslSpec` JSON and
serialized nirs4all-style list/dict JSON. The schema documents the accepted
portable surface: canonical step kinds plus compatibility keys such as
`pipeline`, `preprocessing`, `model`, `branch`, `merge`, `split`, `sources`,
`_or_`, `_cartesian_`, `_chain_`, `_grid_`, `_range_`, `_log_range_`, `_zip_`
and `_sample_`. The compatibility profile also accepts minimal nirs4all-style
operator aliases: short strings such as `SNV`/`PLSRegression`, plain
`{"class": ...}` / `{"function": ...}` / `{"ref": ...}` objects, and
`{"name": ..., "step": ...}` wrappers. Rust classifies those aliases only far
enough to build the safe plan: splitters become campaign `SplitInvocation`
entries, obvious estimators become model nodes, obvious tuners such as
`OptunaTuner` become tuner nodes, chart aliases become chart nodes, and
everything else remains an external transform for host controller resolution.
When the compiler is given controller manifests (`compile-pipeline-dsl
--controllers` or `build-pipeline-dsl-plan`), selector-only aliases can refine
this default before graph ports are frozen: a custom bare alias such as
`ElasticSpectra` may become a model if exactly one operator kind claims it
through `operator_selectors`. Cross-kind matches are rejected and must use
explicit DSL syntax.
Rust validation remains the semantic authority: it lowers
compatibility JSON into canonical DSL, compiles the graph/campaign/generation
artifact, rejects unsafe augmentation/shape contracts and enforces OOF graph
edges.

External tuner/finetune controllers are canonical operator steps.
`kind: "tuner"` and its alias `kind: "finetune"` compile to
`NodeKind::Tuner`, preserve public tuning metadata and produce fold-aligned OOF
prediction outputs like model nodes; the actual search implementation remains in
the host controller.

Runtime data generators are canonical operator steps, separate from
compile-time search-space generators. `kind: "data_generation"` and its alias
`kind: "generation"` compile to `NodeKind::Generator` and require a public
`shape` contract so synthetic samples/features can be scoped to fold-train
data, audited through origin/group/target inheritance and executed by an
external controller.

## CampaignSpec v1

Schema: `campaign_spec.schema.json`

Canonical fixture: `examples/campaign_oof_generation.json`

Runtime type: `CampaignSpec`

C ABI: `DAG_ML_CAMPAIGN_SPEC_SCHEMA_VERSION`,
`dagml_campaign_spec_contract_json`, `dagml_campaign_validate_json`

This is the portable experimental-plan contract layered beside the graph. It
keeps split invocation, concrete fold sets, leakage-unit policy, repeated-sample
aggregation policy, generation/search dimensions, data/model shape plans and
data bindings outside operator nodes. Selector-driven separation branches are
recorded here as `branch_view_plans`, so source/metadata/tag/filter branch views
can be materialized by data-provider bindings without turning splits or filters
into graph operators. Rust validation remains the semantic authority for fold
membership, leakage guards, generation consistency, shape-plan/key alignment,
branch-view selector sanity and data-binding fingerprint requirements.

## ExecutionPlan v1

Schema: `execution_plan.schema.json`

Canonical fixture:
`examples/fixtures/runtime/execution_plan_branch_merge_executable.json`

Runtime type: `ExecutionPlan`

C ABI: `DAG_ML_EXECUTION_PLAN_SCHEMA_VERSION`,
`dagml_execution_plan_contract_json`,
`dagml_execution_plan_validate_json`

This is the compiled, scheduler-ready DAG contract. It binds the validated
graph, campaign, resolved controller manifests, per-node execution policies,
generation variants, fold set and canonical fingerprints used later by bundles,
replay and provenance exports. The schema documents the portable envelope and
critical coordination fields; Rust validation remains the authority for DAG
topology, controller-policy consistency, OOF capability checks, fold semantics,
shape/data binding checks and fingerprint consistency.

## Estimator outcome and conformal foundation v1

Schemas: `parameter_patch.schema.json`, `output_binding.schema.json`,
`training_influence_manifest.schema.json`, `training_outcome.schema.json`,
`replay_outcome.schema.json`, `execution_bundle.schema.json`,
`prediction_cache_payload_set.schema.json` and
`conformal_calibration.schema.json`.

Canonical fixtures live under `examples/fixtures/estimator/` and
`examples/fixtures/conformal/`. These W0 contracts publish the portable
boundary needed by native estimator-style fitting and conformal calibration;
the conformal calibration, prediction, metric and robustness contract types are
now native in `dag-ml-core`. Their dedicated C ABI, PyO3 and WASM exposure is a
later integration step; hosts must not infer those bindings from the schemas.
Install `scripts/requirements-contracts.txt` before running
`scripts/validate_contracts.py`; the validator builds an offline Draft 2020-12
registry from every local schema, resolves all estimator/conformal references,
then applies the cross-plan and cache invariants that JSON Schema cannot express.

`ParameterPatch` records one selected change after tuning. Its canonical key
is `(node_id, namespace, path)`, arrays are sorted by that key, and duplicate
targets are invalid. `path` is an array of object-key segments, not a JSON
Pointer; `-` append semantics are unsupported. The namespaces mean:

- `operator`: estimator or transform constructor parameters;
- `fit`: fit-call-only parameters;
- `control`: scheduler/controller behavior that does not change graph shape;
- `structural`: explicit graph- or shape-affecting choices.

For example, the canonical representation of a selected PLS component count
is:

```json
{
  "schema_version": 1,
  "node_id": "model:pls",
  "namespace": "operator",
  "path": ["n_components"],
  "value": 12
}
```

The patch is not an instruction to mutate an opaque fitted model. The same
value must already be materialized in the matching effective
`ExecutionPlan.node_plans[node_id]` namespace before a `TrainingOutcome` can
be accepted.

`OutputBinding` removes output-shape guessing. It binds a graph node and port
to prediction level/unit, prediction kind and source, optional refit strategy,
aggregation fingerprint, target order, target units, per-target class
vocabularies and target space. Regression uses one explicit empty class array
per target. Class probabilities use `target_major_class_minor`; all other V1
kinds use `target_order`. `binding_fingerprint` is TCV1 over the binding with
only that field omitted. `bound_output` pairs the binding with the actual
prediction blocks and validates producer, unit, row width and target/class
column order.

`TrainingInfluenceManifest` is the exact development-data identity closure for
fit, selection, early stopping, weighting/resampling and trained aggregation.
Every entry explicitly carries `node_id` (`null` when the influence is global),
`origin_sample_ids` and `group_ids` (empty arrays when absent). It never carries
`training_outcome_fingerprint`: `TrainingOutcome` owns the manifest and a
conformal `PredictorBinding` references the two sibling fingerprints, which
avoids a recursive hash.

`TrainingOutcome` is the complete portable result of compile through `FIT_CV`
and `SELECT`, plus optional `REFIT`. It always contains the full effective plan,
selected variant, canonical patches, non-null OOF `ScoreSet`, bound outputs,
lineage, influence manifest and replay bundle. The selected patches are exactly
the leaf `param_overrides` of the selected variant, not merely values that happen
to occur in the effective plan. Starting at every bound output, validation walks
the full transitive `node_plans[*].input_nodes` closure. Every closure node that
supports `FIT_CV` has one lineage record per fold and exactly one corresponding
fit/transform/trained-meta influence entry; a completed refit has one `REFIT`
lineage record per closure node that supports `REFIT`.

Persisted refit artifacts follow controller capability, not a broad node-kind
guess: every closure node whose `controller_capabilities` contains
`emits_artifacts` must be represented in the execution bundle and its `REFIT`
lineage, while a node without that capability must not invent an artifact. Each
artifact record exactly cross-links its node's external data requirements and
incoming OOF prediction requirements. A completed refit uses `final_refit`
outputs; a skipped refit carries no refit artifact and exposes `REFIT` as its
replayable phase. Runtime handles are forbidden. `outcome_fingerprint` is TCV1
over the entire outcome with only `outcome_fingerprint` omitted.

`ReplayOutcome` extends the existing replay summary counters with actual bound
predictions, explanations and lineage. Counts must equal the payload, every
emitted producer must have lineage for the replay phase, and its outcome
fingerprint uses the same omit-self TCV1 rule. `execution_bundle.schema.json`
and `prediction_cache_payload_set.schema.json` publish the existing Rust wire
shapes used inside these outcomes; Rust bundle/replay validation remains the
runtime authority. The positive refit fixture embeds both branch-to-meta OOF
requirements, their cache records and the matching portable payload set; the
no-refit fixture preserves the same complete OOF cache because replaying `REFIT`
must not discard the fold predictions that justified selection. Requirement,
cache-record and payload key sets must match exactly in both branches.

`CalibrationArtifact` V1 implements split absolute-residual calibration at
physical-sample level. Its predictor embeds the exact selected patches,
full transitive external-data bindings, capability-required refit artifacts and
`OutputBinding`, and binds the owning `TrainingOutcome` plus its influence
manifest. Predictor artifact/data lists are derived from the execution bundle;
omitting a base estimator or upstream transform binding is stale even when the
final meta-estimator remains present. The sorted, unique
`predictor_node_ids` member is the authoritative closure: it includes required
transforms even when they emit no persisted artifact, and structural robustness
scenarios may target only nodes in that list. Calibration sample/origin closure must be
disjoint from training sample/origin closure. Coverages are strictly increasing
binary64 values; rank is `ceil((n + 1) * Decimal(shortest-roundtrip coverage))`. A finite
persisted quantile is a binary64 JSON float token (`2.0`, not `2`); an
out-of-range rank under the permissive policy uses the tagged
`{"status":"unbounded"}` value, never JSON infinity.

Fingerprint algorithms are intentionally disjoint:

- legacy DAG-ML graph/campaign/controller/parameter fingerprints use the
  existing deterministic serde JSON profile;
- new estimator/conformal fingerprints use DAG-ML Typed Canonical Value v1,
  whose object keys are normalized and ordered by UTF-8 bytes;
- nirs4all-methods fingerprints use RFC 8785/JCS and UTF-16 key ordering.

A field names exactly one profile. Values are never co-hashed or compared
across profiles. `scripts/validate_contracts.py` pins schema ownership,
cross-fixture equality, TCV1 preimages, integer-versus-float distinction,
shortest-roundtrip rank behavior, refit/no-refit semantics and stale predictor
refusal.

## Conformal prediction and robustness foundation v1

Schemas: `cohort_manifest.schema.json`,
`conformal_prediction_block.schema.json`,
`conformal_metric_set.schema.json`,
`domain_assessment_block.schema.json`, `decision_block.schema.json`,
`robustness_scenario_spec.schema.json` and
`robustness_report.schema.json`.

The versioned positive and negative fixtures are under
`examples/fixtures/conformal/` and `examples/fixtures/robustness/`. The
byte-level artifact manifest is
`conformal_robustness_conformance_pack.v1.json`. Together with
`conformal_calibration.schema.json`, it pins the eight W0
conformal/robustness contracts,
their transitive estimator/calibration contracts, fixtures, rank and metric
goldens, the independent Python semantic oracle, the frozen Philox counter
oracle and the independent Rust canonical-profile oracle. Paths are
repository-relative, sorted and unique; traversal and symlinks in any path
component are refused, each file uses a byte SHA-256, and the pack itself uses
an omit-self TCV1 checksum.

`CohortManifest` names one of four roles: `development`, `calibration`,
`external_test` or `production`. Its exchangeability unit is always
`physical_sample`. `unit_relations` gives the exact per-sample origin, group
and source membership; its ordered sample ids and the union of its relation
members must equal the corresponding flat identity closures. A calibration
cohort must be disjoint from every physical sample and origin that influenced
fit, selection, early stopping, resampling or trained aggregation. Distinct
digests alone are not evidence of disjointness.

`ConformalPredictionBlock` binds intervals to an existing point-output binding,
point-prediction fingerprint, predictor and calibration artifact. Unit rows,
target columns and coverage levels are explicit and intervals must be nested.
`assumption_status` records whether exchangeability is declared, shifted or
not assessed. `guarantee_status` is consequently separate from finite bounds:
under distribution shift a finite interval may remain useful as a diagnostic,
but it does not silently retain a marginal or joint coverage guarantee.

`ConformalMetricSet` is not a model `ScoreSet`. Its canonical coordinate is
scenario, severity, slice, target, coverage, fold, repeat and seed. Records
carry the actual sample count, empirical coverage, coverage gap, interval
widths and interval score. Marginal targets must belong to the report cohort;
`joint_max` records use a null target. Sliced or shifted measurements are
`diagnostic_only` or `unavailable` unless a separately validated recalibration
restores the stated assumption. The regression metric golden reconstructs
coverage, mean/median width and the two-sided Winkler interval score directly
from truth and bounds for marginal, joint and unbounded cases.

The fixture-only `evidence_cases` and report `evidence_sets` close the numeric
chain that the portable schemas intentionally fingerprint rather than embed.
Each record contains finite binary64 point predictions and truth, exact TCV1
fingerprints and a one-to-one block/metric-set link. The independent validators
require the point matrix and truth matrix to match the block row/target shape,
reconstruct every finite interval midpoint, and recompute empirical coverage,
coverage gap, mean/median width and Winkler score from truth and bounds.
Unbounded cells remain paired null endpoints with null width/score. These
records are conformance evidence only; they are not a production report field
or a new runtime API.

`DomainAssessmentBlock` answers whether a predictor-support diagnostic regards
each unit as in support, out of support or unknown. It makes no coverage claim.
`DecisionBlock` applies an explicit named policy to conformal or domain
evidence and owns actions such as `accept`, `reject` and `refer`; it makes no
domain-score or statistical-guarantee claim of its own. Numeric comparison
operators require finite binary64 JSON float thresholds (`2.0`, never `2`).
Numeric domain scores/thresholds and numeric members of decision membership
arrays follow the same lexical rule; string and boolean members remain typed.

`RobustnessScenarioSpec` makes the identity severity `0.0` mandatory and pins
Philox 4x32-10 plus the versioned counter/key derivation over scenario,
binary64 severity, unit, target kind/id and draw index. Its three modes have
different state transitions:

- `clean_frozen` reuses predictor and calibrator and treats non-zero shifted
  coverage as diagnostic or unavailable;
- `matched_recalibration` reuses the predictor but requires a new, fully
  checksum-addressed calibration artifact and a calibration cohort disjoint
  from both external evaluation units and predictor-training influence. Its
  diagnostics identify the exact scenario, binary64 severity and calibration
  input fingerprint;
- `structural_refit` requires a replacement node that resolves inside the
  baseline predictor closure, a new plan/graph/selected-variant predictor
  identity and either a new compatible calibrator or an explicitly invalidated
  calibration. A declaration targeting an unrelated node is refused.

`RobustnessReport` keeps predictor state, calibration state and coverage
guarantee in separate fields. Every result records before/after input,
relation, predictor and point-prediction identities. Severity-zero rows must be
identity-preserving. Every non-zero slice is paired with an exact
severity-zero baseline over the same split, environment, fold, repeat, seed,
unit level and unit ids; point-metric baseline values must come from that row.
Each declared severity includes an `all` slice. Conformal records additionally
match fold/repeat/seed and sample count exactly. Non-null calibration checksums
resolve to complete `CalibrationArtifact` documents in the report and are
covered by provenance. A production report may instead be explicitly
point-only: its calibration checksum is null, conformal artifacts/blocks/metric
sets are empty, coverage is `unavailable`, and its example monitors the
label-free `prediction_mean_shift` metric rather than implying that delayed
truth was available for MAE, R² or RMSE.

The fully resolved fixture has three calibration artifacts and exactly 18
result/block/metric-set coordinates: clean mode covers `all`, both groups and
both sources at both severities; matched recalibration covers `all` and both
groups; structural refit covers `all`. A separate valid structural result
demonstrates explicit calibration invalidation without inventing a prediction
block or metric set. Every requested metric is either produced or named by a
`metric_unavailable.<metric>` score error, and every declared group, source,
environment or target slice value is present at every declared severity.

These W0 contracts are owned by DAG-ML as portable coordination and audit
shapes. `dag-ml-core` now exposes native Rust conformal contract types and exact
split-conformal regression helpers over these semantics. Public C ABI, Python
and WASM conformal bindings remain separate follow-up work.
`parity/conformal/oracle.py`, `parity/robustness_rng/oracle.py` and
`parity/canonical/rust-oracle/` are test-only authorities and production
validation never imports them. TCV1 and the restricted RFC 8785/JCS profile
are intentionally different: TCV1 uses NFC, UTF-8 key ordering and typed
integer/binary64 tokens; restricted JCS uses no normalization, UTF-16 key
ordering and tagged binary64 strings. A fingerprint field names exactly one
profile.

Run the complete local gates from the repository root:

```bash
python3 -m pip install -r scripts/requirements-contracts.txt pytest
python3 scripts/validate_contracts.py --require-sibling --sibling-root ../dag-ml-data
python3 -m pytest parity/conformal/tests/test_conformal_robustness_contracts.py -q
python3 -m pytest parity/robustness_rng/tests/test_robustness_rng_contract.py -q
CARGO_TARGET_DIR=/tmp/dagml-canonical-rust-target \
  cargo test --locked --manifest-path parity/canonical/rust-oracle/Cargo.toml
python3 -m pytest parity/canonical/tests/test_rust_oracle_parity.py -q
```

## ModelInputSpec v1

Schema: `model_input_spec.schema.json`

Canonical fixture:
`examples/fixtures/data/model_input_spec_tabular_regressor.json`

Runtime type: `ModelInputSpec`

C ABI: `DAG_ML_MODEL_INPUT_SPEC_SCHEMA_VERSION`,
`dagml_model_input_spec_contract_json`,
`dagml_model_input_spec_validate_json`

This neutral contract is the data/model compatibility request declared by a
controller or binding. It lists required input ports, accepted
representations/types, tensor rank expectations, multi-source support and the
default fusion policy to ask from a data planner.

For a multi-source `concatenate_features` request, the concrete
`DataBinding.metadata.source_index` map is the required source-concat layout
hint. It maps each source id to its feature-axis block index and is propagated
into `DataProviderViewSpec.extra.source_index`; without it the planner refuses
the binding with a structured `dagml.data_requirement.*` code instead of
silently treating early fusion as ordinary flat features. `by_source` branches
remain single-source per branch unless a future contract explicitly supports
grouped source branches.

## DataPlan v1

Schema: `data_plan.schema.json`

Canonical fixture: `examples/fixtures/data/data_plan_tabular_fusion.json`

Runtime type: `DataPlan`

C ABI: `DAG_ML_DATA_PLAN_SCHEMA_VERSION`, `dagml_data_plan_contract_json`,
`dagml_data_plan_validate_json`

This neutral contract is the data-planner answer to a `ModelInputSpec`: a
deterministic sequence of materialize/adapt/align/join/collate steps plus the
named outputs that feed model ports. DAG-ML validates ordering, output
references and refusal metadata before such a plan can become part of an
execution plan or bundle.

## ControllerManifest v1

Schema: `controller_manifest.schema.json`

Canonical fixture:
`examples/fixtures/runtime/controller_manifest_data_aware_model.json`

Runtime type: `ControllerManifest`

C ABI: `DAG_ML_CONTROLLER_MANIFEST_SCHEMA_VERSION`,
`dagml_controller_manifest_contract_json`,
`dagml_controller_manifest_validate_json`,
`dagml_controller_manifest_list_validate_json`

This is the binding-facing contract each external controller registry must
publish. It declares the controller id/version, operator kind, phase support,
ports, deterministic/replay capabilities, fit scope, RNG policy, artifact
policy and optional `ModelInputSpec` data requirements. The schema is the
portable shape; Rust validation remains the authority for registry uniqueness,
phase/fit-scope consistency, capability/port consistency and typed
`data_requirements` semantics.

`operator_selectors` are the minimal-alias bridge used by bindings. A host can
publish a `TransformerMixin` controller that matches aliases such as `SNV`,
plain strings such as `StandardScaler`, a tuner controller that matches
`OptunaTuner`, or class/function/ref/type descriptors; Rust keeps the operator
payload opaque, uses selectors to classify bare aliases when manifests are
available at compile time, and routes the node to the matching controller before
execution.

## NodeTask / NodeResult v1

Schemas: `node_task.schema.json`, `node_result.schema.json`

Canonical fixtures:
`examples/fixtures/runtime/node_task_transform_scale.json`,
`examples/fixtures/runtime/node_result_transform_scale.json`

Runtime types: `NodeTask`, `NodeResult`

C ABI: `DAG_ML_NODE_TASK_SCHEMA_VERSION`,
`DAG_ML_NODE_RESULT_SCHEMA_VERSION`, `dagml_node_task_contract_json`,
`dagml_node_result_contract_json`,
`dagml_node_result_validate_for_task_json`

These are the direct wire contracts between the Rust coordinator and external
operator controllers. `NodeTask` carries the resolved node plan, phase,
variant/fold context, handles, data views, OOF prediction inputs, refit artifact
inputs and deterministic seed. `NodeResult` returns output handles,
sample predictions, optional observation-level predictions, optional aggregated
sample/target/group predictions, shape deltas, artifacts and lineage. Rust validates every result
against the exact task before committing it, including node/run/phase/fold,
variant, controller, seed, params fingerprint, shape fingerprints, output
ownership and artifact handle consistency.

## SelectionPolicy / SelectionDecision v1

Schemas: `selection_policy.schema.json`, `selection_decision.schema.json`

Canonical fixtures: `examples/fixtures/bundle/selection_policy_rmse.json`,
`examples/fixtures/bundle/selection_decision_branch_b0.json`

Runtime types: `SelectionPolicy`, `SelectionDecision`

C ABI: `DAG_ML_SELECTION_POLICY_SCHEMA_VERSION`,
`DAG_ML_SELECTION_DECISION_SCHEMA_VERSION`,
`dagml_selection_policy_contract_json`,
`dagml_selection_decision_contract_json`,
`dagml_selection_policy_validate_json`,
`dagml_selection_decision_validate_json`

These contracts preserve the selection boundary used before refit/replay:
metric name/objective, optional required prediction level
(`observation`/`sample`/`target`/`group`), selected candidate, selected score
and the deterministic ranked candidate list. Rust validation remains the
semantic authority for rank continuity, selected-candidate consistency,
duplicate candidates and finite selected scores.

## Native training and portable predictor contracts v1

Schemas: `training_request.schema.json`, `parameter_projection.schema.json`,
`cache_namespace.schema.json`, `training_outcome.schema.json` and
`portable_predictor_package.schema.json`.

Normative syntax and examples: [Native training contracts](../TRAINING_CONTRACTS.md).

Conformance pack: `training_contract_conformance_pack.v1.json`.

These W1 contracts turn the W0 estimator vocabulary into a strict native
training boundary. `TrainingRequest` projects through the normal graph,
campaign, controller and execution-plan types; it does not add a second runtime
ABI. The separate `ParameterProjection.nodes[*]` object maps namespaces
bijectively to `params`, `fit_params`, `control_params` and
`structural_params`; `ExecutionPlan::NodePlan` still exposes only `params`.
W1-0 validates and projects the other namespaces but does not silently execute
them. A later runtime step must materialize each supported namespace explicitly.
Data identities include schema, plan, relation, feature-content and
target-content fingerprints. Active controller capabilities derive exact
fold/refit influence slots, and cache namespaces bind every
prediction-affecting coordinate. `selection_output_id` names the sole output
whose averaged FIT_CV reports drive ranking; `TrainingOutcome` persists that
choice and reconstructs the decision from scores.

The D3 execution bindings expose that one native operation through
`dagml_training_execute`/`DagMlTrainingResult` in the C ABI and through
`dag_ml.execute_training`/`TrainingResult` in Python. Their complete field,
callback, GIL, ownership, detach and memory rules are documented in
[Native training contracts](../TRAINING_CONTRACTS.md#c-training-binding). Both
bindings require an exact map from the rendered `node_id.input_name`
requirement key to an attested `ExternalDataPlanEnvelope`; missing, extra,
duplicate or colliding keys fail before controller/data execution. The envelope
must derive the exact signed `TrainingDataIdentity`, including non-null
relation, feature-content and target-content fingerprints.

The C data vtable remains borrowed. Readable controller bindings opt into the
header's borrowed-v2 or owning-v3 lifecycle, and an opaque result keeps
controller/artifact resources alive until its exact-once free. Python releases
the outer GIL while the native scheduler runs, re-acquires it per callback and
returns an owning result whose `detach()` is idempotent. Detach preserves the
portable outcome/bundle/scores/outputs/cache payloads while discarding callback
objects and process-local data/view/artifact handles.

`PortablePredictorPackage` cross-links the request/outcome, effective plan,
execution bundle, output bindings, predictor closure, training influence, data
identities and refit artifacts. Native artifacts require portable metadata;
host-only fitted objects remain process-local sidecars and runtime handles are
recursively forbidden from the package. Bundle content and the complete ordered
data-identity array have dedicated TCV1 commitments in `TrainingOutcomeRef`, so
ids alone cannot authorize replacement payloads.

D3 does not expose a generic fitted-model resolver or `TrainingResult.predict`.
Artifact properties contain portable records, never host object handles.
`replayable_phases` therefore remains a validated capability statement; D4
replay must consume the portable bundle/package, exact envelopes, retained OOF
payloads and reloadable artifact stores. A host sidecar that must survive
detach or process exit needs a host persistence/reload implementation.

## DAG-ML OpenLineage Facets v1

Schema: `openlineage_dagml_facets.schema.json`

This is a DAG-ML-specific publication contract, not a shared `dag-ml-data`
wire contract. `export-open-lineage` derives an OpenLineage `RunEvent` from an
already validated research provenance package and uses these custom `dagml_*`
facets to preserve DAG-ML fingerprints, OOF coverage counters, unsafe flags and
bundle/plan identifiers that OpenLineage does not model natively.

## Prediction Cache Tensor Metadata v1

Schema: `prediction_cache_tensor_metadata.schema.json`

This C ABI metadata contract accompanies
`dagml_prediction_cache_payload_f64_tensor_json`. The tensor carries contiguous
row-major F64 prediction values; the metadata carries the stable requirement
key, cache id, prediction level, block offsets, fold ids, sample ids, unit ids
and target names required to interpret rows without hiding traceability inside
the value buffer.

## Prediction Cache Columnar Tensor Metadata v1

Schema: `prediction_cache_columnar_tensor_metadata.schema.json`

This C ABI metadata contract accompanies
`dagml_prediction_cache_payload_f64_columnar_tensor_json`. It keeps the same
traceability fields as the row-major export and adds `layout:
column_major_f64` plus `column_offsets` so host bindings can read each target
column contiguously without guessing buffer order.

## Data Output Provenance v1

Schema: `data_output_provenance.schema.json`

Canonical fixture:
`examples/fixtures/runtime/data_output_provenance_augmented_view.json`

Runtime type: `DataOutputProvenance`

C ABI: `DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION`,
`DAG_ML_DATA_OUTPUT_PROVENANCE_EXTRA_KEY`,
`dagml_data_output_provenance_contract_json`,
`dagml_data_output_provenance_validate_json`

This DAG-ML runtime contract is embedded under the reserved
`DataProviderViewSpec.extra["dag_ml_output"]` key when a data-producing DAG node
emits a downstream data view. It records the producer node/port/phase,
variant/fold scope, shape-plan and aggregation fingerprints, current feature
schema fingerprint and emitted shape deltas. D6/D7 add optional representation
plans, replay manifests, relation-delta fingerprints and train/predict
compatibility reports so serve-time missing source or repetition differences
are explicit, policy-bound and replayable. Controllers and host bindings can
discover and validate this metadata without reverse-engineering free-form JSON
or hardcoding Rust-only constants.

## Process Adapter Description v1

Schema: `process_adapter_description.schema.json`

Canonical fixture:
`examples/fixtures/runtime/process_adapter_description_python.json`

Runtime shape returned by process adapters from `--describe`:
`{ schema_version, protocol, adapter_id, supported_modes, capabilities }`.

C ABI: `DAG_ML_PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION`,
`dagml_process_adapter_description_contract_json`

This CLI/runtime contract lets the coordinator reject unsupported process
adapters before any `NodeTask` is sent. Version 1 requires protocol
`dag-ml-process-adapter`, mode declarations for `one_shot`/`jsonl` support and
explicit JSON task/result capabilities. Persistent worker and parallel
scheduler features remain opt-in capabilities layered on the same description
object.

## Process Adapter Frame v1

Schema: `process_adapter_frame.schema.json`

Canonical fixtures:
`examples/fixtures/runtime/process_adapter_frame_init.json`,
`process_adapter_frame_task_transform_scale.json`,
`process_adapter_frame_result_transform_scale.json`,
`process_adapter_frame_ack_initialized.json`,
`process_adapter_frame_error_retryable_timeout.json`,
`process_adapter_frame_close.json`

Runtime shape used by persistent JSONL process adapters:
`init | task | close` coordinator request frames and
`ack | result | error` adapter response frames.

C ABI: `DAG_ML_PROCESS_ADAPTER_FRAME_SCHEMA_VERSION`,
`dagml_process_adapter_frame_contract_json`

This contract is enabled only when the adapter description declares
`control_frames_v1`. It gives host adapters a stable lifecycle and error
surface: `init` carries controller and worker identity, `task` wraps a
published `NodeTask`, `result` wraps a published `NodeResult`, `error` carries
typed retryability, and `close` gives the coordinator a bounded shutdown path.

## Aggregation Controller Task/Result v1

Schemas: `aggregation_controller_task.schema.json` and
`aggregation_controller_result.schema.json`

C ABI: `DAG_ML_AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION`,
`DAG_ML_AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION`,
`dagml_aggregation_controller_task_contract_json`,
`dagml_aggregation_controller_result_contract_json`,
`dagml_aggregation_controller_task_validate_json`,
`dagml_aggregation_controller_result_validate_for_task_json`

These contracts define the leakage-sensitive payloads used when aggregation is
delegated to an external controller through `AggregationMethod::CustomController`.
The task carries the custom aggregation policy, controller id, repeated
observation or sample-to-unit inputs, relation metadata and requested output
order. The result is validated against the exact task so custom reducers cannot
change sample/unit coverage, fold scope, target names or prediction level.

## Research Provenance Package Profile v1

Profile: `research_provenance_package_profile.v1.json`

This publication profile declares the required files, optional files, checksum
rules, PROV JSON-LD sections, RO-Crate file properties, OpenLineage facets and
CLI tests for a DAG-ML research package. It is validated by
`scripts/validate_contracts.py` so the human-facing publication contract stays
aligned with the Rust/CLI validator.

## Data Provider C ABI v2

The shared provider surface is `DagMlDataVTable` guarded by
`DAG_ML_DATA_VTABLE_DEFINED` and versioned by
`DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION == 2`. `scripts/validate_contracts.py`
and the C ABI tests verify that `dag_ml.h` and `dag_ml_data.h` can be included
together in either order when the sibling checkout is available.
