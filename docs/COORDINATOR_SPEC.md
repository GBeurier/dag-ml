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
| `capabilities` | deterministic, thread_safe, process_safe, needs_python_gil, emits_predictions, emits_relation |
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

Controller outputs:

- opaque data/model/artifact handles;
- prediction blocks;
- sample relation deltas;
- metrics;
- structured errors;
- artifact refs or serialization bytes.

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
- topological task groups;
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
- view sample ids are in the active fold/partition;
- seed is derived from the canonical path;
- unsafe policy is explicit if required.

After dispatch, the Rust core checks:

- output ports match declared contracts;
- prediction blocks carry producer, fold, partition, sample ids and target names;
- no prediction row is positional-only;
- train/validation/test/final partitions are legal for the phase;
- augmentation origins do not cross fold boundaries;
- group/repetition/target leakage units remain on one side of a fold;
- artifact refs and handle lifetimes are registered;
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

If any answer is "no", the implementation has drifted from the product goal.
