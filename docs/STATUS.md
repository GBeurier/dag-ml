# Status

Current state: OOF/data-contract foundation plus first coordinator core.

Implemented:

- Rust workspace with core, facade, C ABI and CLI crates;
- graph model and validation;
- fold identity models and deterministic identity splitters;
- OOF campaign fixtures, joins and leakage refusal;
- campaign and OOF fixture fingerprints;
- deterministic control seed derivation;
- controller manifests, controller registry and controller resolution;
- controller manifests now carry scheduler/operator capabilities into
  `NodePlan`/`NodeTask`, including `thread_safe`, `process_safe`,
  `emits_predictions`, `consumes_oof_predictions`, `emits_artifacts` and
  statefulness flags; OOF edges and parallel scheduler modes validate these
  declarations before controller invocation;
- `GraphPlan`, `CampaignSpec`, `ExecutionPlan`, `NodePlan`, `NodeTask`,
  `NodeResult` and `RunContext`;
- `GraphPlan.parallel_levels` plus `PhaseExecutionSchedule`, so a compiled
  plan exposes deterministic node batches per phase/variant/fold before any
  controller is invoked;
- split invocation as a campaign-plan controller call;
- deterministic generation/search-space scaffold with variant fingerprints and
  variant seeds;
- optional graph `search_space_fingerprint` validation against the canonical
  campaign `GenerationSpec` fingerprint during plan compilation, so graph and
  campaign search spaces cannot drift silently;
- generation choices can carry typed node parameter overrides; the scheduler
  lowers them into the controller-facing `NodePlan.params` and
  `params_fingerprint` for each variant while refusing conflicting overrides
  and unknown target nodes;
- process controller smoke adapters now assert that generated param overrides
  targeting their node are present in the effective `NodeTask.node_plan.params`;
- controller-facing `NodeTask.variant` context with generated choices,
  fingerprint and variant seed, so external bindings can apply model,
  augmentation or processing variants without guessing from `variant_id`;
- leakage-unit policies for sample/target/group/repetition/origin boundaries;
- sample relation validation for repeated observations, shared targets, groups
  and augmentation origins;
- aggregation policy plus deterministic mean, weighted-mean, median and vote
  reducers from observation predictions to samples and from sample predictions
  to target/group units;
- data/model shape plans and runtime shape deltas;
- shape-policy hardening for sample augmentation, supervised feature-selection
  mask audit, optional lineage shape fingerprints and feature-delta schema
  continuity checks;
- data bindings from node inputs to external `dag-ml-data` plan envelopes,
  including explicit feature-set ids and ABI-compatible fusion selectors for
  provider `feature_arrow` calls;
- external data-plan envelopes carry an explicit schema version and are refused
  when a future/unsupported version is received;
- published JSON Schema artifact for external coordinator data-plan envelopes,
  with a unit smoke that keeps its declared version aligned to the Rust
  contract;
- stdlib shared-contract validation script plus CI checkout of `dag-ml-data`
  so schema copies and coordinator fixtures cannot drift silently;
- shared conformance-pack manifest with canonical schema/fixture digests, C ABI
  requirements and required cross-repo checks, kept JSON-identical with
  `dag-ml-data`;
- external data-plan envelope validation by schema, plan and relation
  fingerprints;
- campaign data envelopes with coordinator relations are checked against the
  campaign/split fold set and leakage policies before training handles are
  registered by the CLI;
- runtime data-provider trait with materialization plus fold/refit/predict view
  requests that turn data bindings into scoped opaque task handles;
- `FIT_CV` data routing now gives controllers separate fold-train and
  fold-validation views, so validation OOF predictions can be checked against
  the validation identity set;
- runtime `requires_oof` edge enforcement for training phases: downstream
  nodes receive a validated opaque prediction handle only when the upstream
  producer has emitted validation predictions for the current fold, while
  `REFIT` requires full CV OOF coverage; fold-aligned edges are checked against
  the `FoldSet`;
- controller-facing `NodeTask.prediction_inputs` metadata for validated OOF
  inputs, exposing producer/ports, fold scope, sample ids, prediction width and
  target names beside the opaque prediction handle;
- controller-facing `NodeTask.data_views` map carrying the scoped view spec
  beside each data-view handle;
- in-memory runtime data provider with handle records for schema/plan/relation
  traceability and child data-view records for sample partition, source and
  feature-set traceability;
- runtime artifact-store trait plus in-memory refit artifact handle records,
  including capture of controller-emitted refit artifact handles during `REFIT`;
- `ArtifactRef` now exposes typed optional backend, URI, content fingerprint and
  plugin/version metadata, with bundle/runtime/lineage validation while keeping
  legacy refit artifact JSON readable; portable URIs must be strictly relative
  artifact paths, rejecting absolute paths, Windows drive prefixes, URI schemes
  (`http://`, `s3://`, `file://`, any colon in the first path segment) and `..`
  traversal components;
- versioned file-backed artifact manifest store (`artifact_manifest.json`,
  schema v1): serializes a bundle's portable refit `ArtifactRef`s with their
  node/controller/params-fingerprint identity, revalidates the manifest against
  the bundle on reopen, and refuses tampered, duplicated or non-portable
  entries; it records artifact metadata only and never reads, writes or
  materializes artifact payloads;
- bundle replay executor that validates plan/bundle/request/data envelopes,
  materializes data and refit artifact handles, and invokes eligible runtime
  controllers for replay phases without CV folds;
- stricter `NodeResult` conformance validation for externally returned run,
  controller, version, variant, fold, branch, seed, params fingerprint, output
  owner, artifact controller fields and artifact handle ownership;
- in-memory prediction store and lineage recorder;
- sequential scheduler for deterministic DAG order plus campaign execution over
  variant x CV-fold scopes;
- bounded parallel scheduler for compiled DAG levels, including `FIT_CV`,
  `REFIT` artifact capture and bundle replay paths; controller invocation is
  concurrent within each level while data/view preparation and result commits
  stay deterministic;
- deterministic graph parallel-level planner for future node-batch execution
  without changing topological semantics;
- C ABI exports both compiled execution plans and phase execution schedules as
  owned JSON for non-Rust bindings;
- deterministic metric selection contracts, including grouped candidate
  selection, stable tie-breaking and optional metric-level guards that reject
  sample/target/group score mismatches before ranking;
- execution-bundle validation requires persisted selection decisions to carry
  the campaign `selection_metric_level`, preventing sample/target/group metric
  drift between selection and refit/replay packaging;
- identity-aligned regression metric reports over validated sample, target and
  group prediction blocks (`mse`, `rmse`, `mae`, `r2`), with finite score
  validation, prediction origin traceability and direct conversion to
  `CandidateScore` for selection;
- CLI and C ABI JSON entry points for identity-aligned regression scoring over
  sample or aggregated prediction blocks, plus report-to-`CandidateScore`
  conversion for non-Rust bindings;
- refit execution bundle contracts that bind selected variants, selected
  candidates, refit artifacts, plan fingerprints and replay data requirements;
- replay-facing prediction contracts now carry explicit `prediction_level`
  metadata across `NodeTask.prediction_inputs`, bundle prediction
  requirements, cache records, payloads and file/columnar cache manifests;
  target/group cache records and payloads carry typed `PredictionUnitId`
  `unit_ids`, and file/in-memory/columnar cache stores can validate, load and
  materialize aggregated replay handles without preloading them into the
  sample-level OOF store;
- execution-bundle validation now checks selected candidates against the
  rebuilt plan and requires refit artifacts for selected refittable nodes;
- explicit execution-bundle schema version with unsupported-version refusal;
- explicit schema migration policies for execution bundles and prediction-cache
  payload sets: current/min readable/writable versions are public, future/zero
  versions are refused, and old versions require declared migration edges;
- replay request validation for predict, explain and refit phases;
- mock controller conformance tests;
- CLI execution-plan validation from graph/campaign/controller JSON fixtures;
- CLI data-binding validation against a coordinator data-plan envelope;
- CLI mock campaign execution through controller manifests, data bindings,
  in-memory data provider, fold-aware data views and mock runtime controllers;
- CLI selection, bundle build and bundle replay validation commands with
  fixture-backed integration tests;
- CLI mock refit bundle command that runs `REFIT`, captures emitted model
  artifact handles and builds an `ExecutionBundle` from the captured records;
- CLI process refit bundle command proving the same artifact capture path over
  external `NodeTask`/`NodeResult` JSON adapters;
- CLI mock replay execution through execution bundles, data envelopes,
  in-memory data provider, predict-scoped data views, in-memory artifact store
  and mock runtime controllers;
- CLI process campaign and replay execution that sends `NodeTask` JSON to an
  external adapter process over stdin, reads `NodeResult` JSON from stdout and
  validates the result through the scheduler;
- scheduler-level branch/merge OOF smoke fixture with two branch models feeding
  a heterogeneous meta-model through `requires_oof` prediction edges plus an
  original-data binding; the process adapter now validates
  `NodeTask.prediction_inputs` against fold/sample scope;
- CLI process CV+refit bundle command that first runs `FIT_CV`, keeps the
  same `RunContext`/prediction store, then runs `REFIT`; this validates that
  branch/merge meta-model refit consumes complete CV OOF coverage before
  capturing base and meta refit artifacts, writes typed bundle
  `prediction_requirements` and deterministic `prediction_caches`, links the
  meta refit artifact to the OOF requirements it consumed, accepts validated
  selection decisions for branch and merge choices, and keeps an OOF summary
  (producer, folds, samples, prediction width and targets) in bundle metadata;
- materialized OOF prediction-cache payload sets for CV+refit bundles: payload
  JSON stores the actual validation `PredictionBlock` or
  `AggregatedPredictionBlock` values, validates by cache id, requirement key,
  format, row/block counts and content fingerprints against the bundle
  manifest, and refuses tampered payload values;
- runtime prediction-cache store contract plus in-memory payload-backed store:
  replay loads exact validation OOF blocks through the store, validates them
  against bundle cache records, and asks the store to materialize controller
  prediction handles for OOF-dependent refit inputs;
- columnar f64 prediction-cache store behind the same
  `RuntimePredictionCacheStore` contract: validated sample and target/group
  payloads are converted once into typed column buffers, exposed through
  deterministic manifests and used by the CLI payload-backed replay path before
  controller handles are materialized;
- file-backed prediction-cache store: validated payload sets can be exported to
  a cache directory with a manifest plus one payload file per OOF requirement,
  reopened for replay, fully revalidated against the bundle and rejected if a
  payload file is tampered;
- CLI export/validation/replay path for file-backed prediction caches, with
  `REFIT` replay accepting either a validated payload set or a validated cache
  directory, but not both;
- branch/merge process replay from that captured bundle, including three
  refit artifact handles and three data requirements that may resolve to the
  same external data-plan envelope without duplicate-registration failure;
- bundle replay validation refuses `REFIT` replay when the bundle depends on
  OOF prediction requirements but only carries cache manifests; when a validated
  prediction-cache payload set is supplied, replay preloads the validation OOF
  blocks into the `PredictionStore` before running `REFIT`;
- CLI contract proving direct branch/merge `REFIT` without a preceding
  in-context `FIT_CV` is refused because OOF validation predictions are absent;
- process-controller replay fixtures now verify that a model receives its own
  refit artifact handle, not just any artifact handle in the task inputs;
- stateful sklearn process-controller smoke that fits a real sklearn pipeline
  during `REFIT`, stores it behind an opaque model handle, then replays
  `PREDICT` through the captured handle in the same persistent process;
- stateful sklearn branch/merge smoke that runs scheduler-managed `FIT_CV`,
  builds OOF prediction requirements/cache manifests, captures refit artifacts
  and replays predictions in the same persistent process;
- persistent JSONL process-controller mode for campaign/replay smoke tests,
  avoiding one process spawn per task and preparing stateful host adapters;
- persistent process-controller pool mode with `--process-workers`: the CLI
  prewarms multiple JSONL workers per controller, routes `FIT_CV` tasks by
  node/variant/fold, routes `REFIT`/`PREDICT` by node/variant for artifact
  stickiness, and exposes observed worker counts through adapter lineage
  metrics;
- CLI execution commands expose `--scheduler sequential|parallel` plus
  `--scheduler-workers`, so branch/merge campaigns can exercise the core
  parallel scheduler against persistent process adapters;
- persistent process workers now have a coordinator-side watchdog
  (`--process-timeout-ms`) and opt-in task retry (`--process-retries`) that
  kills, replaces and replays a task on the targeted worker after timeout, EOF
  or transport failure; the flaky adapter fixture proves both timeout refusal
  and timeout/restart/retry recovery;
- process adapters now expose a required `--describe` JSON handshake declaring
  the adapter protocol version, supported modes and JSON task/result
  capabilities; the CLI validates this contract before one-shot or JSONL
  execution;
- process adapters must explicitly declare persistent-worker, worker-env and
  parallel-invocation capabilities before the CLI enables those execution
  modes;
- persistent process adapters now use `control_frames_v1`: workers receive
  explicit `init`, framed `task`, and `close` JSONL messages and return typed
  `ack`, `result`, or `error` frames, giving the coordinator a stable lifecycle
  and error surface before native bindings exist;
- `dag-ml-core` now has a bounded `ParallelScheduler` for parallel DAG levels:
  controllers are `Send + Sync`, independent nodes in the same compiled level
  are invoked concurrently, and results are committed back into prediction,
  lineage and handle stores in deterministic level order;
- Python process-controller adapter fixture for campaign/replay smoke tests,
  including data-handle, fold and refit-artifact-handle checks;
- C ABI validation and JSON output helpers for graph validation,
  graph parallel levels, execution-plan build, regression metric reports,
  selection decisions, grouped selection, execution bundles, replay envelopes,
  replay requests and prediction-cache payload sets;
- C ABI data-provider vtable shape aligned with `dag-ml-data`
  materialize/view/identity/target/feature exports plus a tested Rust runtime
  adapter over the vtable;
- `dag_ml.h` and `dag_ml_data.h` share a guarded data-provider vtable ABI
  version macro and are compiled together in both include orders when the
  sibling checkout is present;
- cross-repo C conformance replay where `dag-ml` consumes a real
  `dag-ml-data` in-memory f64 provider vtable, reads feature and target Arrow
  exports from an external controller, and verifies data/view handle release
  order through the runtime adapter;
- C ABI controller vtable generic `invoke` path for external host
  controllers, routing `NodeTask` JSON to `NodeResult` JSON with explicit
  host-returned byte release and controller-owned result handle release, plus
  a tested Rust runtime adapter over the vtable;
- C ABI artifact-store vtable for replay REFIT artifacts, returning typed
  `DagMlHandleRef` values, preserving host handle ownership and releasing
  materialized handles at adapter drop, plus a tested Rust runtime adapter over
  the vtable;
- C ABI prediction-cache vtable shape for host OOF cache stores, including
  JSON block loading for sample and target/group aggregated replay blocks,
  prediction-handle materialization, materialized handle release and explicit
  host-returned byte release, plus a tested Rust runtime adapter over the
  vtable;
- C ABI non-mock replay execution helper that composes host controller,
  data-provider, artifact-store and optional prediction-cache vtables while
  Rust owns bundle validation, replay envelope validation, DAG scheduling,
  data-view construction and `NodeResult` conformance, with branch/merge
  `REFIT` replay covered through a host OOF prediction-cache vtable;
- C ABI C-language conformance test that compiles and runs a small C program
  against `dag_ml.h` and `libdag_ml_capi`, builds an execution plan through
  the ABI, then drives non-mock replay through controller, data-provider and
  artifact-store vtables;
- C ABI mock replay execution helper that exercises execution-plan, bundle,
  replay request, data envelope and refit artifact handle materialization and
  returns a JSON summary including data view counts;
- standalone sklearn complex OOF demonstrator with repeated observations,
  group-aware splits, train-only augmentation, branch model variants,
  heterogeneous prediction+raw-data merge variants, OOF-based selection and
  final refit report;
- CLI validator for the sklearn complex demonstrator that revalidates the OOF
  campaign in Rust, recomputes branch/merge selections from report metrics and
  checks final-refit feature/sample contracts;
- C ABI graph validation entry point;
- `dag-ml-data` fixture integration through schema, plan and relation
  fingerprints;
- explicit `dag-ml-data` coordinator-envelope contract fixture using the
  `S001`/`S002` sample ids and `y` target emitted by `dag-ml-data`, kept
  JSON-identical across repos by `scripts/validate_contracts.py`, plus a
  sibling-repo CLI conformance test when `../dag-ml-data` is available;
- separate `sample:1`/`sample:2` internal envelope fixture for legacy
  scheduler/replay smoke campaigns, so internal test data cannot masquerade as
  the shared coordinator contract;
- coordinator graph/campaign/controller fixtures;
- CI workflow.

Not implemented yet:

- full DSL compiler;
- advanced search-space compiler/lowering beyond typed node-parameter
  overrides;
- custom aggregation controllers and production persistent/Arrow replay
  backends for non-sample aggregated prediction blocks;
- persistent artifact payload stores and payload materialization (reading,
  writing or deserializing the artifact binaries) beyond the implemented
  portable artifact reference contract and file-backed artifact manifest;
- Arrow prediction storage and ABI-owned prediction tensors;
- production host controller adapters with native libraries or
  language-specific bindings;
- production `dag-ml-data` provider backends beyond the current in-memory C
  conformance provider.

Next recommended task:

Continue productionizing host adapters: native binding contracts, controller
lifecycle ownership, failure recovery/timeouts, and larger replay stress
fixtures over the shared conformance pack.
