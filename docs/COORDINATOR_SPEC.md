# DAG-ML Coordinator Short Spec

Status: working normative spec.

This document is the short alignment contract for the product. It resolves the
ambiguity left by the long design documents: DAG-ML is a Rust pipeline
coordinator, not an ML algorithm library and not a data framework.

## Product Goal

DAG-ML must virtually execute any data pipeline that can be expressed as:

1. a compiled DAG of data/model/control dependencies;
2. an experimental campaign plan: variants, splits, repetitions, refit, predict,
   explain;
3. external operators reachable through language bindings/controllers;
4. typed data contracts and identity tables supplied by `dag-ml-data` or an
   equivalent data provider.

The product promise is:

- reproducible execution;
- traceable execution;
- leakage-safe OOF campaigns;
- real compiled DAG semantics;
- high-performance Rust orchestration and scheduling;
- external operators, data buffers and fitted objects kept outside the core.

## Non-Negotiable Principles

1. The Rust core owns logic, not domain algorithms.
2. Operators are external payloads: sklearn estimators, torch modules, C++
   methods, bioinformatics tools, filters, splitters, augmenters.
3. Controllers live in bindings or native plugin layers. They adapt external
   operators to the Rust core contract.
4. The Rust core owns compilation, planning, validation, scheduling, RNG,
   fingerprints, lineage, cache keys, OOF joins, and campaign state.
5. The Rust core never inspects feature buffers, images, spectra, tensors,
   sequences, graphs, or fitted model internals.
6. The Rust core may inspect only identity, fold membership, prediction tables,
   target/scoring tables, descriptors, fingerprints and controller manifests.
7. Every join, split, merge and prediction use is keyed by stable identities, not
   by row position.
8. Unsafe leakage paths are refused by default and must be explicit, traceable and
   searchable when allowed.

## Ownership Boundary

| Area | Rust core owns | Binding/controller owns | Data layer owns |
|---|---|---|---|
| DSL | compiled IR target, not syntax sugar | frontend syntax helpers | none |
| DAG | graph topology, ports, branches, merges | external node behavior | source contracts only |
| Campaign | variants, split plan, phases, refit/replay | splitter/model execution | sample relation facts |
| Controllers | manifest validation, invocation protocol | actual adapter code | none |
| Data | views by identity, DataPlan refs | host handles | schema, sources, adapters, collation |
| Execution | scheduler, tasks, phase gates, reducers | fit/predict/transform/split calls | materialization |
| Reproducibility | RNG contexts, fingerprints, canonical plans | local RNG consumption from seed | schema/data fingerprints |
| Traceability | lineage records, cache keys, artifact refs | artifact serialization hooks | fitted data-adapter refs |
| OOF safety | fold validation, prediction store, OOF join | emits predictions with identity | group/origin/repetition facts |

## Vocabulary

### Operator

External business payload. Examples: sklearn estimator, torch module, C++ PLS,
BLAST wrapper, aligner, filter predicate, augmenter, splitter.

An operator is never owned by the Rust core.

### Controller

Binding-side or native adapter that knows how to call an operator. A Python
controller may call sklearn. A native controller may call C++. An R controller may
call mlr3. A bioinformatics controller may call a local library or process.

Controllers are registered through a manifest and invoked through ABI/binding
contracts.

### Rust Core

The coordinator. It compiles, plans, validates, schedules and records execution.
It treats controllers as callable endpoints with declared capabilities.

### Data Provider

The owner of source storage and heavy buffers. For this workspace, `dag-ml-data`
is the reference contract. Other data providers can exist if they satisfy the
same identity/data-plan boundary.

## Controller Contract

Each controller must expose a manifest equivalent to:

| Field | Purpose |
|---|---|
| `controller_id` | stable unique id |
| `controller_version` | semantic version used in fingerprints |
| `operator_kind` | model, transform, splitter, augmenter, filter, metric, generator, tuner |
| `supported_phases` | compile, plan, fit_cv, select, refit, predict, explain |
| `input_ports` / `output_ports` | typed port contracts |
| `data_requirements` | `ModelInputSpec`, aux inputs, required sources or target |
| `capabilities` | deterministic, thread_safe, process_safe, needs_python_gil, emits_predictions, consumes_oof_predictions, emits_artifacts, stateful, emits_relation |
| `fit_scope` | stateless, fold_train, full_train, inference_only |
| `rng_policy` | uses core seed, ignores seed, externally deterministic, nondeterministic |
| `artifact_policy` | serializable, host_only, content_addressed, replay_required |

Minimum callable operations:

| Operation | Used for |
|---|---|
| `describe` | manifest and static contracts |
| `plan` | controller-specific plan details, no heavy compute |
| `invoke` | phase-specific execution over handles and identity views |
| `release` | lifecycle cleanup for host handles |

Specialized operations such as `fit`, `predict`, `transform`, `split`,
`augment`, `score` may exist behind `invoke`, but the Rust core should schedule
typed tasks rather than know library-specific APIs.

Controller inputs:

- opaque data handles;
- selected identity view;
- phase;
- fold id, branch path, variant id, trial id;
- deterministic `SeedContext`;
- data-plan/fingerprint refs;
- controller params.

Data-plan references are `DataBinding` contracts owned by the coordinator:

- node id and input name;
- `dag-ml-data` request id;
- schema fingerprint;
- data-plan fingerprint;
- optional relation fingerprint;
- output representation;
- feature set id used for `feature_arrow` requests, defaulting to input name
  when omitted;
- feature fusion selectors may be passed through the same `feature_arrow`
  bytes-view when a data provider supports `dag-ml-data` multi-source fusion;
- source ids;
- view policy for fold train, fold validation, refit and predict.

The actual data plan and relation table remain external. `dag-ml-data` can emit
a coordinator envelope containing these fingerprints plus coordinator relation
records; `dag-ml` validates that an execution campaign binds to the exact
envelope before a controller receives any handle.

The shared serialization contract is versioned as
`CoordinatorDataPlanEnvelope` v1. `dag-ml` consumes the subset represented by
`ExternalDataPlanEnvelope`, rejects unsupported future versions at runtime, and
publishes the current JSON Schema at
`docs/contracts/coordinator_data_plan_envelope.schema.json`. The schema is the
wire-contract artifact; Rust validation remains responsible for semantic
checks such as fingerprint equality and relation membership in the active
campaign fold set.

At execution time, the scheduler does not give the controller the raw
materialized data handle directly. It asks the data provider for a scoped view
derived from the active phase and `DataViewPolicy`: `FIT_CV` tasks receive a
fold-train view for fitting and a separate fold-validation view for OOF
prediction, refit/full-train views carry the full training sample ids, and
replay predict views are marked as predict partitions. The handles visible in
`TaskRequest` are scoped data-view handles, and the same request carries a
`data_views` map keyed like `input_handles` so bindings can inspect the selected
partition without guessing from the handle. The parent handle remains
traceability state owned by the provider. Unsafe data views are rejected unless
declared explicitly: `FIT_CV` fitting cannot use full-train or validation
partitions, validation/predict views cannot include augmented rows, and excluded
rows cannot be included unless the corresponding `DataViewPolicy.unsafe_flags`
entry is present.

Controller outputs:

- opaque data/model/artifact handles;
- prediction blocks;
- sample relation deltas;
- metrics;
- structured errors;
- artifact refs or serialization bytes.

## Public Method Shape

The data/model piloting contract must be visible in public Rust and binding
APIs. It must not be hidden inside controller-specific side effects.

Minimum public coordinator flow:

| Method | Inputs | Output |
|---|---|---|
| `compile` | frontend pipeline IR, frontend registry | `GraphSpec`, `CampaignSpec` |
| `plan` | graph, campaign, controller manifests, dataset schema, data planner | immutable `ExecutionPlan` |
| `fit_cv` | execution plan, data provider handle, controller registry, stores | `CVResult` with prediction and lineage refs |
| `select` | CV result, ranking policy | `SelectedGraph` / selected variant refs |
| `refit` | selected graph, execution plan, data provider handle, stores | `RefitResult` / bundle inputs |
| `export_bundle` | selected graph, refit result, artifacts, fingerprints | `ExecutionBundle` |
| `predict` | bundle, new data provider handle | prediction blocks |
| `explain` | bundle, new data provider handle, target node/method | explanation payload refs |

Minimum controller-facing request/response shapes:

| Type | Required fields |
|---|---|
| `ControllerPlanRequest` | node id, operator params, phase set, data requirements, input/output ports, data schema fingerprint |
| `SplitRequest` | identity table, sample relation table, split policy, seed context |
| `TaskRequest` | phase, node id, fold id, branch path, variant id plus generated choices/fingerprint, data view, data-plan refs, input handles, prediction input metadata, artifact input metadata, seed context |
| `TaskResponse` | output handles, prediction blocks, sample relation deltas, metrics, artifacts, lineage payload |

Every shape-changing operation must declare the affected domain:

| Domain | Examples | Contract |
|---|---|---|
| row domain | sample filtering, sample augmentation, repetitions, group splits | changes identity/sample relation or view membership |
| feature domain | preprocessing, feature augmentation, feature selection, source fusion | changes representation/schema/feature names |
| target domain | y-transform, target aggregation, multi-target mapping | changes target space and inverse-transform requirements |
| prediction domain | model predict, OOF join, aggregation of repetitions | changes prediction block shape and aggregation level |

## DAG, Campaign Plan And Splits

This is the key correction.

There are two related but distinct plans:

### GraphPlan

The compiled DAG of dependency semantics:

- transforms;
- y-transforms;
- models;
- feature joins;
- prediction joins;
- source joins;
- fork/map/branch;
- aggregators;
- filters;
- augmentation nodes;
- generator nodes;
- tuners;
- explain nodes.

Graph nodes describe dependency and phase behavior. External behavior is executed
by controllers.

### CampaignPlan

The experimental execution plan around the graph:

- root seed;
- variants/search space;
- split strategy;
- repeated campaigns;
- nested CV policy;
- selection/ranking policy;
- refit policy;
- predict/explain replay policy;
- scheduler/resource policy.

### Generation Ownership

Generation has two different meanings and the ownership must stay explicit:

- compile-time generation of variants/search spaces belongs to `dag-ml`, because
  it changes campaign fingerprints, variant ids, seeds, lineage and selection;
- runtime generation of data, features, models or synthetic samples is an
  external operator capability. The payload lives in controllers or
  `dag-ml-data`, while `dag-ml` validates seeds, origin/sample relations,
  shape deltas, fold boundaries and lineage.

The current core represents compile-time generation through `GenerationSpec`
and `VariantPlan`. Choices may carry typed node-parameter overrides; the
scheduler lowers those into the effective `NodePlan.params` and
`params_fingerprint` sent to controllers for that variant. When a graph declares
`search_space_fingerprint`, plan compilation verifies it against the canonical
campaign `GenerationSpec` fingerprint before variants are enumerated. Runtime
generators are graph nodes/controllers with explicit capabilities such as
`generates_data`, `generates_model` or `expands_variants`; they are not allowed
to mutate identity or training boundaries without emitting relation and shape
deltas.

### Splitters

Splitters are not ordinary data-transform operators.

A splitter is a controller capability invoked by the coordinator during campaign
planning or early execution. It produces a `FoldSet`. The Rust core validates the
`FoldSet` against identity, group, target, repetition and origin constraints.

Native identity splitters such as KFold and GroupKFold may be shipped by the Rust
core for deterministic cross-language behavior. Feature-dependent splitters such
as KS/SPXY/sklearn splitters are external controller calls.

Implementation consequence:

- `NodeKind::Split` must not mean "data transform node".
- If a split appears in frontend syntax, the compiler must lower it into
  `CampaignPlan.split_invocation`, not a normal feature-flow node.
- A legacy or compatibility graph node may exist only as a control node that
  emits no feature data and is excluded from model/data transforms.

## Identity And Leakage Units

The coordinator must distinguish physical rows from logical samples. This is
critical for products with repeated measurements: several `X` observations can
share one `Y`.

| Identity | Meaning | Owned by |
|---|---|---|
| `ObservationId` | physical row/acquisition in one source | data provider |
| `SampleId` | logical sample requested by the user | data provider, validated by core |
| `TargetId` | target/label unit; may cover one or more samples | data provider, validated by core |
| `GroupId` | leakage unit such as plant, product, patient, plot, batch | data provider, validated by core |
| `OriginId` | original sample/observation from which an augmented row was derived | data provider/controller, validated by core |
| `RepetitionId` | repeated acquisition id inside a sample | data provider |

Split policies must declare the leakage unit:

| `split_unit` | Rule |
|---|---|
| `observation` | only the same observation is atomic; unsafe for repeated X / one Y unless explicitly allowed |
| `sample` | all observations of one sample stay on the same fold side |
| `target` | all samples sharing one target stay on the same fold side |
| `group` | all samples/observations sharing one group stay on the same fold side |

Defaults:

- repeated measurements default to `split_unit="sample"`;
- group ids, when present and requested, dominate sample ids for leakage;
- augmentation origin constraints are always checked unless explicitly disabled by
  a traceable unsafe policy;
- fold membership is stored at sample/target/group level and broadcast to
  observations during materialization.

The core refuses a split when:

- a requested split unit is absent from the sample relation table;
- one leakage unit appears in both train and validation of a fold;
- an augmented row derived from a validation origin appears in train;
- repeated observations of the same sample are split across train/validation
  under `split_unit="sample"`;
- a feature-dependent splitter returns folds not expressible in stable ids.

## Repetitions, Aggregation And Refit

Repeated observations are first-class. The core must support models trained on
individual observations while also evaluating and selecting on aggregated
sample-level predictions.

Example: several spectra (`ObservationId`) for the same product (`SampleId`) with
one chemical value (`TargetId`).

Required prediction levels:

| Level | Meaning |
|---|---|
| `observation` | one prediction per physical acquisition |
| `sample` | aggregation of observations for the same sample |
| `target` | aggregation of samples for the same target unit |
| `group` | aggregation at group level, only if explicitly requested |

Required aggregation policies:

| Policy field | Purpose |
|---|---|
| `aggregation_level` | observation, sample, target, group |
| `method` | mean, weighted_mean, median, vote, custom_controller |
| `weights` | none, quality, repetition_count, controller_emitted |
| `emit_parallel_metrics` | whether raw and aggregated metrics are both computed |
| `selection_metric_level` | which level drives variant/model selection |
| `store_raw_predictions` | keep observation-level predictions for audit |
| `store_aggregated_predictions` | keep aggregated predictions for ranking/replay |

The Rust core implements deterministic observation-to-sample and
sample-to-target/group aggregation for `mean`, `median`, `vote` and
`weighted_mean`. Weighted means require an explicit weight policy;
`controller_emitted` and `quality` weights are read from observation prediction
blocks, while `repetition_count` weights sample-to-target/group aggregation by
the number of observations attached to each sample. `custom_controller` remains
an explicit controller responsibility.

FIT_CV requirements:

1. Models may fit on observation-level rows if the data plan says so.
2. Validation predictions are first captured at the controller-emitted level.
3. The Rust core aggregates predictions by identity according to policy.
4. Metrics are computed in parallel when requested. The core provides
   identity-aligned regression scoring for validated sample/target/group
   prediction blocks (`mse`, `rmse`, `mae`, `r2`) and refuses positional-only
   or mismatched unit sets. Metric reports preserve the scored prediction
   origin: producer node, partition, optional fold and optional prediction id.
5. Selection must declare which metric level is authoritative. Selection
   policies can require a `metric_level`, and candidates whose metric metadata
   is missing or comes from another level are rejected before ranking.
6. OOF joins use the declared aggregation level and must not mix raw and
   aggregated predictions implicitly.
7. Replay-facing prediction contracts carry `prediction_level` explicitly.
   Sample-level replay caches carry `sample_ids`; target/group replay caches
   carry `unit_ids` typed as `PredictionUnitId`, validated in bundle records,
   payloads and file/in-memory stores. Aggregated caches are not preloaded into
   the sample OOF store.

REFIT requirements:

1. The final fit boundary is the selected training universe, excluding held-out
   test samples.
2. Refit may train on all repeated observations of selected samples if the
   `DataModelShapePlan` declares observation-level fitting.
3. Refit prediction outputs must preserve the same aggregation policy used for
   selection unless the user explicitly chooses a different predict policy.
4. Final prediction blocks may store both raw repetition predictions and
   aggregated sample predictions.
5. A bundle must record whether the selected model was chosen by observation,
   sample, target or group metrics.
6. Prediction requirements, prediction-cache records, prediction-cache payloads
   and `NodeTask.prediction_inputs` must preserve the prediction level, so a
   target/group metric decision cannot be replayed through an implicit
   sample-level cache.

## Data/Model Shape Plan

The coordinator must make shape control explicit because augmentations,
selection, fusion and aggregation change what a controller sees.

Each model or transform node receives a `DataModelShapePlan`:

| Field | Meaning |
|---|---|
| `input_granularity` | observation, sample, target, group |
| `target_granularity` | observation, sample, target, group |
| `fit_rows` | train observations/samples allowed for fit |
| `predict_rows` | validation/test/final rows expected for predict |
| `feature_namespace` | source/branch/augmentation prefixes |
| `feature_schema_fingerprint` | stable identity of column/feature layout |
| `target_space` | raw, transformed, scaled, encoded |
| `aggregation_policy` | how controller predictions are reduced |
| `augmentation_policy` | sample/feature augmentation rules |
| `selection_policy` | feature selection and supervised selection fit scope |

Shape-changing controllers must return a shape delta:

| Delta | Examples | Required validation |
|---|---|---|
| `row_delta` | sample filter, sample augmentation, separation branch | identity and fold boundaries remain valid |
| `feature_delta` | feature augmentation, selection, source fusion | feature schema fingerprint changes deterministically |
| `target_delta` | y scaling, target encoding | inverse-transform and target space are recorded |
| `prediction_delta` | probability output, repetition aggregation | prediction columns and aggregation level are recorded |

The core validates deltas before downstream tasks can consume them.
Feature deltas are checked against any declared
`feature_schema_fingerprint`; lineage may also echo the data/model shape and
aggregation-policy fingerprints, and when present those fingerprints must match
the compiled `NodePlan`.

## Augmentation, Selection, Filtering And Fusion

These operations are leakage-sensitive because they can change rows, features or
target spaces.

### Sample Augmentation

Sample augmentation creates new observations or samples. Default policy:

- FIT_CV: train partition only;
- validation/test rows are never augmented for training metrics;
- each augmented row carries `OriginId`;
- an augmented row inherits target, group and fold boundary from its origin;
- OOF and scoring are reported on original identities unless policy opts into
  augmented reporting.
The default Rust contract rejects sample augmentation across all partitions, or
sample augmentation without origin/target/group inheritance, unless the
corresponding `AugmentationPolicy.unsafe_flags` entry is present.

Forbidden by default:

- validation-origin augmentation appearing in train;
- augmented rows counted as independent samples for group/sample metrics;
- augmented validation/test rows entering view requests without an explicit
  unsafe flag;
- sample augmentation after a prediction join unless explicitly declared safe.

### Feature Augmentation

Feature augmentation changes columns/features but not identity.

Rules:

- stateless feature augmentation may run on train/validation/test if declared
  deterministic and fitted nowhere;
- stateful feature augmentation must fit on fold train and apply to validation;
- supervised feature augmentation is treated like supervised feature selection and
  must use fold-train targets only;
- feature namespaces and schema fingerprints must change deterministically;
- source/branch/augmentation provenance is preserved for merge and explain.

### Feature Selection

Feature selection is a transform that may be supervised or unsupervised.

Rules:

- supervised feature selection fits only inside the current train boundary;
- selected feature masks are artifacts with fold/refit lineage;
- supervised feature selection must store masks so CV/refit replay can audit
  which features were fitted inside each train boundary;
- validation/test/final data only receive `apply`, never `fit`;
- downstream feature joins must verify compatible selected schemas or use an
  explicit missing-feature policy;
- selection masks used at REFIT are recorded separately from CV-fold masks.

### Sample Filtering / Exclusion

Filtering changes row membership.

Rules:

- filters must declare whether they affect train only, predict only, all
  partitions, or branch-local views;
- train-only exclusion cannot silently remove validation samples from scoring;
- separation branches must produce disjoint or explicitly overlapping identity
  sets according to branch policy;
- all filters emit a `row_delta` with before/after identity fingerprints.

### Source Fusion And Merge

Feature/source fusion changes feature shape and possibly missingness.

Rules:

- sample alignment remains data-layer work, but the core records and validates
  the chosen alignment plan fingerprint;
- feature joins are namespace-stable and branch-aware;
- prediction joins are OOF-checked by the core;
- mixed joins must declare which inputs are raw features and which are
  predictions, with separate leakage checks.

## Phase Model

The coordinator executes:

```text
COMPILE -> PLAN -> FIT_CV -> SELECT -> REFIT -> PREDICT -> EXPLAIN
```

| Phase | Rust core responsibility | Controller responsibility |
|---|---|---|
| `COMPILE` | parse frontend IR into graph/campaign specs, freeze topology | provide descriptors if needed |
| `PLAN` | resolve controllers, ports, data plans, variants, split invocations | describe/plan only |
| `FIT_CV` | schedule fold/variant/branch tasks, enforce fold views, record predictions | fit/transform/predict/split/augment |
| `SELECT` | rank variants, reject unsafe variants by policy | optional metric helpers |
| `REFIT` | replay selected graph on full train boundary | fit final artifacts |
| `PREDICT` | replay bundle on new data, validate schema/fingerprints | predict/transform only |
| `EXPLAIN` | replay to explanation target | controller-specific explanation |

## ExecutionPlan

An `ExecutionPlan` is the Rust-owned, immutable plan after compile and plan:

- graph fingerprint;
- campaign fingerprint;
- data-plan fingerprints;
- controller manifests and versions;
- variant list or lazy enumerator;
- split invocation specs and resulting fold fingerprints;
- leakage unit policy;
- data/model shape plans per node;
- aggregation policies for prediction/evaluation/refit;
- topological order and deterministic parallel node levels;
- phase execution schedules expanded by variant and fold;
- phase gates per node;
- expected input/output contracts;
- cache key templates;
- lineage templates;
- scheduler policy.

No controller may mutate the `ExecutionPlan`.

## Runtime Execution

The Rust core runs task batches in deterministic dependency order.

Parallelism dimensions:

- variants;
- folds;
- branches;
- independent DAG nodes;
- controller-declared internal parallelism.

Scheduler rules:

1. Ready tasks may run in parallel only if dependencies and resource policy allow.
2. Reduction order is canonical: sort by variant id, fold id, branch path, node id.
3. Nested parallelism is controlled by controller capabilities and resource policy.
4. Python/GIL-bound controllers may be process-scheduled; native thread-safe
   controllers may be thread-scheduled.
5. A task result is accepted only after runtime validation.

## Runtime Validation

Before dispatch, the Rust core checks:

- required ports exist;
- phase is allowed by controller manifest;
- data plan fingerprint matches;
- requested data views match the active phase, fold id and partition;
- view sample ids are in the active fold/partition;
- validation prediction sample ids are contained in the fold-validation view;
- leakage unit membership is compatible with the active fold;
- task data/model shape plan matches the phase and controller manifest;
- seed is derived from the canonical path;
- unsafe policy is explicit if required.

After dispatch, the Rust core checks:

- output ports match declared contracts;
- prediction blocks carry producer, fold, partition, sample ids and target names;
- no prediction row is positional-only;
- train/validation/test/final partitions are legal for the phase;
- augmentation origins do not cross fold boundaries;
- group/repetition/target leakage units remain on one side of a fold;
- shape deltas are declared before downstream consumption;
- aggregation level and prediction columns match the node policy;
- artifact refs, portable backend/URI/content metadata and handle lifetimes are
  registered;
- lineage was recorded.

## OOF And Leakage Rules

OOF safety is a Rust core invariant.

Rules:

1. Training a downstream model on upstream predictions requires validation OOF
   predictions by default.
2. `partition="train"` predictions cannot feed downstream training unless
   `allow_train_predictions_as_features=true` and a second lineage flag records
   `leakage_acknowledged=true`.
3. Every producer must provide exactly one validation prediction per requested
   sample unless an explicit aggregation policy says otherwise.
4. Producers merged by a prediction join must share compatible fold structure.
5. Augmented observations inherit origin leakage boundaries.
6. Group, target and repetition leakage units are validated from sample relation
   facts supplied by the data layer.
7. OOF joins are by sample identity, never by row position.
8. Repetition aggregation is explicit; raw observation predictions and aggregated
   sample/target predictions are never silently mixed.
9. Refit uses the selected aggregation and shape policies unless the predict/refit
   policy explicitly overrides them and records the override.

## Traceability

Every accepted task emits or updates a `LineageRecord` containing:

- run id;
- graph/campaign/variant fingerprints;
- node id;
- phase;
- fold id;
- branch path;
- controller id/version;
- params fingerprint;
- data-plan fingerprint;
- data/model shape fingerprint;
- aggregation level and policy fingerprint;
- input lineage refs;
- output handle/artifact refs;
- seed;
- unsafe flags;
- metrics;
- timing and resource hints.

The lineage graph must be enough to answer:

- why was this sample in this fold?
- which controller produced this artifact?
- which seed and params were used?
- which data schema and data plan were used?
- did any unsafe leakage path occur?
- can this predict/explain run replay the training bundle?

Bundle and prediction-cache payload schemas are versioned artifacts. The core
publishes a `SchemaMigrationPolicy` for each artifact with current/min readable
and writable versions plus explicit automatic migration edges. No implicit
migration is allowed: old versions are accepted only if the policy declares a
migration edge, future versions are refused, and version `0` is always invalid.

## Performance Requirements

The Rust core must be designed as the high-performance coordination layer:

- compact immutable specs;
- canonical fingerprints without reparsing large payloads;
- parallel scheduler over variants/folds/branches/nodes;
- deterministic reducers;
- no feature-buffer copies through the core;
- opaque handle arenas with explicit release;
- prediction store optimized for identity joins;
- controller capability-aware scheduling;
- zero-copy Arrow/DLPack boundaries where identity, predictions or tensors cross.

The core must not chase performance by moving domain algorithms inside Rust. The
performance target is orchestration, validation, scheduling, lineage, and OOF
joins at scale.

## Confrontation With Current nirs4all Pipeline

Current nirs4all is the prototype. It proves the product need, but not the final
architecture.

| nirs4all today | DAG-ML target |
|---|---|
| Python-only runtime | Rust core with Python/R/JS/native bindings |
| Conceptual DAG reconstructed from sequential steps | Real compiled immutable DAG |
| Controllers do work and also carry some orchestration side effects | Controllers only execute declared external behavior |
| Split step mutates pipeline/dataset state | Splitter controller invocation produces `FoldSet` in `CampaignPlan` |
| Branching stored in mutable contexts/snapshots | `Fork/Map` graph semantics with explicit branch path |
| Merge reconstructs features/predictions in Python controllers | Rust-owned feature/prediction join contracts and OOF validation |
| Repetition averages such as `avg` / `w_avg` are controller/runtime conventions | Aggregation policy is explicit, fingerprinted and evaluated in parallel with raw predictions |
| Several spectra can share one product target through dataset conventions | `ObservationId`, `SampleId`, `TargetId`, `GroupId`, `OriginId` are first-class leakage identities |
| Feature/sample augmentation and selection are controller-specific effects | Shape deltas, fit scopes and augmentation/selection policies are validated by the core |
| NIRS-shaped data assumptions leak through dataset/model logic | Generic data contracts through `dag-ml-data` |
| Dynamic routing through Python registry | Binding registries export controller manifests to Rust |
| Trace, artifacts and prediction store are pipeline-specific | Rust-owned lineage/cache/artifact/prediction contracts |
| Reproducibility depends on Python conventions and local discipline | Core-derived seeds, fingerprints and deterministic reducers |
| Leakage safety exists but is embedded in controllers | Leakage safety is a core invariant checked around every task |

The migration path is not "port nirs4all as-is". It is:

1. extract the implicit DAG and campaign semantics;
2. encode them as `GraphPlan` and `CampaignPlan`;
3. move orchestration, validation and scheduling to Rust;
4. keep operators external through controllers;
5. keep data generic through data contracts;
6. replay nirs4all use cases as conformance tests, not as product constraints.

## What Must Be Implemented Next

The next implementation layer must make the coordinator visible:

1. `ControllerManifest`
2. `ControllerRegistry`
3. `CampaignSpec`
4. `GraphPlan`
5. `ExecutionPlan`
6. `NodePlan`
7. `NodeTask`
8. `NodeResult`
9. `RunContext`
10. sequential scheduler
11. in-memory prediction store
12. in-memory lineage recorder
13. mock controller conformance tests
14. split invocation model that produces and validates `FoldSet`
15. `LeakageUnitPolicy`
16. `AggregationPolicy`
17. `DataModelShapePlan`
18. `ShapeDelta`
19. augmentation, feature-selection and filtering policies

The existing OOF/fold/data-plan code remains useful, but it is not the product
shape by itself. The product shape starts when a compiled `ExecutionPlan` drives
controllers through tasks and validates their outputs.

## Acceptance Checks For This Spec

An implementation is aligned only if all answers are "yes":

1. Can the core compile a frozen graph and campaign plan without executing
   external code?
2. Can splitters be external controllers while the core owns the validated
   `FoldSet`?
3. Can model/process/augmentation/filter operators remain external payloads?
4. Can the scheduler parallelize variants, folds and branches without changing
   results?
5. Can the core reject OOF leakage without inspecting feature buffers?
6. Can a run be replayed from fingerprints, controller versions, data-plan refs,
   artifacts and lineage?
7. Can the same core coordinate Python, native C++, R, JS or bioinformatics
   controllers with the same safety rules?
8. Can repeated observations with one target be trained, scored, aggregated,
   selected and refit without leakage or hidden metric-level changes?
9. Can feature/sample augmentation, feature selection, filtering and fusion
   declare shape deltas that the core validates before downstream use?

If any answer is "no", the implementation has drifted from the product goal.
