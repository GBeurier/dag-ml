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
- research provenance package profile describing the publishable package shape,
  RO-Crate checksum rules, PROV sections, OpenLineage facets and required CLI
  conformance tests, validated by the stdlib contract script;
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
- controller-facing `NodeTask.artifact_inputs` map carrying refit artifact
  metadata beside each replay artifact handle, including artifact backend/URI,
  content fingerprint, params fingerprint and data/prediction dependency keys;
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
- initial strict JSON pipeline DSL compiler: `PipelineDslSpec` lowers linear
  transform/augmentation/model steps, target processing, tag/exclude nodes,
  explicit sample/feature augmentation, concat feature fusion, model branches,
  multiple models per branch, standalone merge/join nodes and heterogeneous
  prediction-plus-original-data merge models into canonical `GraphSpec` while
  keeping split/fold planning in campaign contracts rather than graph
  operators; augmentation steps must declare a shape plan so sample/feature
  augmentation scope is validated instead of implicit, and tuning/train params
  are visible in public DSL fields instead of hidden controller state;
- canonical `data_generation` steps, plus alias `generation`, now compile to
  external `NodeKind::Generator` nodes. They must declare a public shape plan,
  so synthetic data/sample generation remains controller-owned while Rust
  validates fold scope, augmentation-origin/group/target inheritance, data
  edges and lineage before downstream training;
- canonical `tuner` steps, plus alias `finetune`, now compile to external
  `NodeKind::Tuner` prediction nodes. They preserve `tuning`/`finetune_params`
  metadata, produce fold-aligned OOF like model nodes, route through
  `operator_selectors`, and are covered by an executable CV+refit DSL smoke with
  a prediction-plus-original-data merge into a final model;
- DSL merge selectors now validate their branch/model/input scopes against the
  pending OOF prediction inputs at compile time, reject unsupported `select`
  modes, reject `top_k` above the matched scope and require a metric for
  `best`/`top_k`, while still leaving actual score computation to external
  selection/merge controllers;
- DSL structural containers now cover nirs4all-style `sequential`,
  `sample_filter`/`filter`, structural `_or_` and structural `_cartesian_`:
  generator choices are expanded into explicit OOF-producing branch choices,
  child node ids are namespaced per generated choice to avoid graph collisions,
  and generator mode/choice labels are preserved as node metadata for
  controller-side selection/refit policy;
- DSL branch compilation now carries transformed branch data outputs alongside
  branch OOF predictions. `merge` modes `features`, `sources`, `all` and
  `mixed` can build data/source/mixed joins from branch outputs, while
  prediction-consuming merges still enforce OOF/fold-aligned prediction edges;
  serialized nirs4all merge dictionaries preserve their raw merge policy in
  metadata for host-side selection and scoring controllers;
- DSL separation branches by source, metadata, tag or filter now emit explicit
  campaign `branch_view_plans` with validated selectors, overlap policy and
  branch-local metadata. This keeps graph branching separate from data-provider
  materialization while giving bindings a stable contract for provider-native
  branch views;
- the DSL parser now accepts serialized nirs4all-style list/dict JSON through a
  compatibility importer (`pipeline`, `preprocessing`, `model`, `branch`,
  `merge`, `_or_`, `_cartesian_`, `_chain_`, `_grid_`, `_range_`,
  `_log_range_`, `_zip_`, `_sample_`, `split`, `sources`) and lowers it to the
  canonical `PipelineDslSpec`; data-only preprocessing generators are fused
  with downstream model/generator stages before compilation so every expanded
  choice remains an OOF-producing branch;
- that compatibility importer prefers minimal aliases and also accepts plain operator
  references (`SNV`, `PLSRegression`, `OptunaTuner`, `chart_2d`,
  `{"class": ...}`, `{"function": ...}`, `{"name": ..., "step": ...}`); Rust
  only infers the safe planning category, keeps operators external for host
  controller resolution, can use controller `operator_selectors` during
  DSL/plan compilation to reclassify otherwise-unknown aliases such as
  `ElasticSpectra` before graph ports are frozen, and folds successive nirs4all
  splitter declarations into one campaign `SplitInvocation` chain instead of
  graph split nodes;
- controller manifests can now declare `operator_selectors` over aliases,
  classes, class prefixes, functions, refs and types. Registry resolution
  prefers matching selectors over generic same-kind controllers, which lets
  minimal payloads such as `SNV` route to a binding-specific transformer
  controller without making the DSL verbose;
- the public Pipeline DSL input contract is now published as
  `docs/contracts/pipeline_dsl.schema.json`, checked by
  `scripts/validate_contracts.py`, and exposed through C ABI
  `DAG_ML_PIPELINE_DSL_SCHEMA_VERSION`, `dagml_pipeline_dsl_contract_json` and
  `dagml_pipeline_dsl_validate_json`, so non-Rust bindings can discover and
  preflight both canonical and nirs4all-compatible DSL profiles before
  compilation;
- the same DSL profile extracts node-level parameter variants into a canonical
  `GenerationSpec`, accepts compact nirs4all-style parameter generators
  (`or`, `range`, `log_range`, `grid`, `pick`, `arrange` with deterministic
  `count` caps), also accepts coordinated generation dimensions that can
  override several branch/merge/model nodes together, validates per-node
  `DataModelShapePlan` declarations for augmentation/selection/aggregation
  safety, writes the generation fingerprint into
  `GraphSpec.search_space_fingerprint`, builds a `CampaignSpec` template that
  carries generation, shape plans, data bindings and split invocation outside
  the graph, and exposes graph-only or graph+generation+shape+campaign artifact
  outputs through CLI and C ABI;
- CLI and C ABI entry points compile that DSL surface to validated graph JSON
  or graph+generation+shape+campaign artifact JSON, with branch/merge OOF,
  generation, data-binding, shape-plan and campaign-template smoke coverage;
- `docs/design/DSL_NIRS4ALL_PARITY.md` records the working parity matrix
  against nirs4all pipeline features, separating canonical strict DSL support
  from future shorthand import and host-controller execution work;
- CLI and C ABI can build a validated `ExecutionPlan` directly from a
  `PipelineDslSpec` plus controller manifests, using the compiled campaign
  template rather than requiring separate graph and campaign JSON files;
- published `ExecutionPlan` contract with JSON Schema, branch/merge executable
  fixture, C ABI contract discovery and standalone validation, making the
  compiled scheduler-ready plan a first-class portable contract rather than an
  implicit Rust-only artifact;
- CLI process commands can run compiled DSL branch/merge/tuner campaigns through
  CV+refit bundle capture and stateful sklearn CV+refit+replay, including
  coordinated generation variants, heterogeneous prediction+original-data merge
  inputs and OOF prediction-cache contracts;
- runtime data edges propagate scoped `DataProviderViewSpec` contracts from
  data-producing operators to downstream consumers, so the executable DSL smoke
  now covers train-only augmentation feeding a branch model before OOF stacking
  and refit/replay;
- propagated data views now carry reserved `dag_ml_output` provenance metadata
  for the producing node/port/phase, selected variant/fold, shape-plan and
  aggregation fingerprints, current feature schema fingerprint and emitted
  shape deltas, making downstream controller inputs auditable beyond opaque
  smoke handles; the metadata is exposed as the typed
  `DataOutputProvenance` contract and validated when parsing
  `DataProviderViewSpec`; the corresponding JSON Schema and canonical fixture
  are published under `docs/contracts` / `examples/fixtures/runtime` and
  checked by `scripts/validate_contracts.py`;
- validation prediction sample checks now apply to propagated data views as
  well as direct data bindings, so a branch model fed by an augmentation edge
  cannot emit OOF rows outside its fold-validation view;
- C ABI exports compiled execution plans, validates canonical `ExecutionPlan`
  JSON directly, and returns phase execution schedules as owned JSON for
  non-Rust bindings;
- C ABI exposes the data-output provenance contract for host bindings through
  stable version/key macros plus JSON contract introspection and standalone
  validation helpers, so non-Rust controllers can trust propagated data views
  without hardcoding Rust-only constants;
- C ABI exposes `dagml_node_result_validate_for_task_json`, allowing
  non-Rust host bindings to preflight a `NodeResult` against the exact
  `NodeTask` before returning it to the scheduler; the scheduler still runs
  the same validation as the authoritative boundary;
- published `NodeTask` and `NodeResult` contracts with JSON Schemas,
  canonical task/result fixtures and C ABI contract discovery, covering the
  actual controller wire protocol used by process/C/Python bindings;
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
- refit artifact validation compares artifact fingerprints against the selected
  generation variant's effective node parameters, not only the base plan
  parameters;
- replay-facing prediction contracts now carry explicit `prediction_level`
  metadata across `NodeTask.prediction_inputs`, bundle prediction
  requirements, cache records, payloads and file/columnar cache manifests;
  target/group cache records and payloads carry typed `PredictionUnitId`
  `unit_ids`, and file/in-memory/columnar cache stores can validate, load and
  materialize aggregated replay handles without preloading them into the
  sample-level OOF store;
- custom aggregation-controller policy support and task/result contracts:
  `AggregationMethod::CustomController` now requires an explicit controller
  spec, controller manifests can declare `aggregates_predictions`, and C ABI
  helpers publish/validate aggregation tasks and validate controller results
  against the exact requested sample/unit order, fold scope, prediction level
  and target names;
- runtime custom aggregation dispatch: Rust can route observation-to-sample and
  sample-to-target/group aggregation tasks to a registered runtime controller,
  requires the controller manifest capability `aggregates_predictions`, and the
  C ABI controller vtable can carry the same aggregation task/result JSON path;
- `NodeResult` now has explicit observation-level and aggregated prediction
  channels. During execution the scheduler can aggregate controller-emitted
  observation predictions through the declared `DataModelShapePlan`, coordinator
  relations from data providers/envelopes, and either built-in reducers or a
  custom aggregation controller before storing live OOF sample or target/group
  blocks;
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
  (producer, prediction level, folds, samples or target/group units, prediction
  width and targets) in bundle metadata;
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
- CLI artifact manifest export/validation path for the file-backed artifact
  manifest: `export-artifact-manifest` writes `artifact_manifest.json` from a
  bundle's portable refit `ArtifactRef`s, `validate-artifact-manifest` reopens
  and revalidates it against the bundle, and `validate-bundle
  --artifact-manifest` reports the manifest entry count while refusing
  mismatched or non-portable entries; this path is manifest-only and never
  reads, writes or deserializes artifact payloads;
- file-backed artifact payload store: `FileArtifactPayloadStore` copies
  payload files referenced by portable `ArtifactRef.uri`, verifies their
  SHA-256 content fingerprint and declared size, reopens them against the
  bundle manifest, and materializes deterministic opaque artifact handles
  without deserializing model binaries. The CLI exposes
  `export-artifact-payload-store`, `validate-artifact-payload-store` and
  `run-mock-replay --artifact-payload-store`;
- first research provenance export layer: core validates the execution plan,
  bundle, optional lineage records, data envelopes, prediction-cache manifest
  and artifact manifest before emitting a checksum-rich RO-Crate package with
  `execution_plan.json`, `execution_bundle.json`, `lineage_records.json`,
  `lineage.prov.jsonld`, `ro-crate-metadata.json`, optional data envelopes and
  optional manifest files; this is a standards-facing export target for W3C
  PROV/Workflow Run RO-Crate and keeps DAG-ML's stricter OOF, replay and
  artifact contracts as the canonical internal model; the CLI can reopen the
  package with `validate-research-provenance`, verify RO-Crate checksums and
  re-run the DAG-ML contract validation; `export-open-lineage` derives an
  OpenLineage `RunEvent` from that already-validated package using custom
  `dagml_*` facets for reproducibility and OOF-safety evidence;
- coordinator-owned lineage propagation: schedulers infer `input_lineage` from
  compiled DAG edges marked `propagates_lineage`, reject adapter-declared
  mismatches and expose `--lineage-output` on refit bundle capture commands so
  provenance exports can be generated from real run records;
- branch/merge research provenance conformance path: the CLI can capture a
  branch/merge CV+refit bundle, export its lineage, OOF prediction-cache store
  and portable artifact manifest, then export a validated PROV/RO-Crate view
  that preserves OOF dependencies, data envelopes, controller agents and model
  artifacts;
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
  refit artifact handle and matching `NodeTask.artifact_inputs` metadata, not
  just any artifact handle in the task inputs;
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
  transport failure or adapter-emitted retryable error frames; the flaky adapter
  fixture proves timeout refusal, timeout/restart/retry recovery and
  retryable-error recovery;
- one-shot process adapters are also guarded by `--process-timeout-ms`, so a
  non-persistent host adapter cannot block the coordinator indefinitely;
- process adapters now expose a required `--describe` JSON handshake declaring
  the adapter protocol version, supported modes and JSON task/result
  capabilities; the CLI validates this contract before one-shot or JSONL
  execution, and the description now has a published JSON Schema plus canonical
  fixture validated by `scripts/validate_contracts.py`, with C ABI contract
  discovery for native bindings;
- process adapters must explicitly declare persistent-worker, worker-env and
  parallel-invocation capabilities before the CLI enables those execution
  modes;
- persistent process adapters now use `control_frames_v1`: workers receive
  explicit `init`, framed `task`, and `close` JSONL messages and return typed
  `ack`, `result`, or `error` frames, giving the coordinator a stable lifecycle
  and error surface before native bindings exist; that frame protocol now has a
  published JSON Schema plus canonical fixtures validated alongside the
  `NodeTask`/`NodeResult` contracts, with C ABI contract discovery for native
  bindings;
- `dag-ml-core` now has a bounded `ParallelScheduler` for parallel DAG levels:
  controllers are `Send + Sync`, independent nodes in the same compiled level
  are invoked concurrently, and results are committed back into prediction,
  lineage and handle stores in deterministic level order. Core tests now stress
  that contract across multi-level campaign scopes with three variants and
  three CV folds, comparing sequential and parallel outputs, predictions,
  lineage and seeds byte-for-byte;
- Python process-controller adapter fixture for campaign/replay smoke tests,
  including data-handle, fold and refit-artifact-handle checks;
- C ABI validation and JSON output helpers for graph validation,
  graph parallel levels, execution-plan build/validation, regression metric reports,
  selection decisions, grouped selection, execution bundles, replay envelopes,
  replay requests and prediction-cache payload sets;
- C ABI research provenance export helper that returns the same validated
  `ResearchProvenanceExport` JSON as the Rust core/CLI path, including
  optional lineage, replay envelopes, prediction-cache manifest and artifact
  manifest inputs for non-Rust bindings;
- C ABI OpenLineage `RunEvent` export helper over the same validated plan,
  bundle, lineage, envelope and manifest inputs used by the Rust provenance
  export path;
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
  a tested Rust runtime adapter over the vtable. Controller vtable lifecycle is
  now explicit: ABI v2 remains borrowed, while ABI v3 opts into Rust-owned
  teardown and calls `destroy(user_data)` after controller result handles are
  released;
- C ABI artifact-store vtable for replay REFIT artifacts, returning typed
  `DagMlHandleRef` values, preserving host handle ownership and releasing
  materialized handles at adapter drop, plus a tested Rust runtime adapter over
  the vtable. Artifact-store ABI v1 remains borrowed, while ABI v2 opts into
  Rust-owned teardown after materialized handles are released;
- C ABI prediction-cache vtable shape for host OOF cache stores, including
  JSON block loading for sample and target/group aggregated replay blocks,
  prediction-handle materialization, materialized handle release and explicit
  host-returned byte release, plus a tested Rust runtime adapter over the
  vtable. Prediction-cache ABI v1 remains borrowed, while ABI v2 opts into
  Rust-owned teardown after materialized handles are released;
- C ABI row-major F64 tensor export for validated sample-level and
  target/group aggregated prediction blocks, with explicit Rust allocation
  ownership and `dagml_f64_tensor_free` release;
- C ABI row-major and column-major F64 tensor exports for bundle-validated
  prediction-cache payload requirements, returning values as contiguous tensors
  and versioned traceability metadata as owned JSON with published JSON Schemas;
- `DataProviderViewSpec` now carries an optional `branch_view:
  Option<BranchViewPlan>` field plus validation, so the runtime can forward
  compiled `branch_view_plans` straight through `make_view` to the data
  provider without going through `extra` JSON. The in-memory `dag-ml-data`
  provider's `DataView` already accepts a matching `CoordinatorBranchView`
  via the same JSON wire shape;
- `ExecutionPlan::branch_view_for(branch_id)` and
  `branch_view_for_path(branch_path)` helpers walk the campaign's
  `branch_view_plans` and return the matching plan (innermost-wins for
  paths);
- the sequential scheduler now extracts `dsl_branch_view_plan` from each
  graph node's metadata via `branch_view_from_node_metadata` and forwards
  it through `data_view_for_scope` / `validation_data_view_for_scope` into
  `DataProviderViewSpec.branch_view`. Nodes produced inside a separation
  branch by the DSL compiler carry the matching `BranchViewPlan` in their
  metadata, so the data provider's `make_view` JSON now receives the
  branch view at runtime (previously the field was hardcoded `None`,
  making compiled `branch_view_plans` inert). Malformed metadata is
  rejected with a clear validation error before any provider call;
- C ABI row-major F32 tensor exports for the same sample-level and
  target/group aggregated prediction blocks, plus row-major and column-major
  F32 tensor exports for bundle-validated prediction-cache payloads. The
  prediction kernels still operate in f64 to preserve canonical numeric
  semantics; each value is cast to f32 at the ABI boundary and the call
  returns `VALIDATION_ERROR` if any value does not round-trip into a finite
  f32 (overflow to infinity or upstream non-finite). Returned tensors carry
  `DagMlF32Tensor`/`DagMlF32ColumnarTensor` shapes and must be released with
  `dagml_f32_tensor_free` / `dagml_f32_columnar_tensor_free`;
- C ABI aggregation-controller task/result contract discovery and validation,
  including result-vs-task checks for external custom reducers before their
  sample/unit predictions can enter a leakage-sensitive pipeline;
- C ABI non-mock replay execution helper that composes host controller,
  data-provider, artifact-store and optional prediction-cache vtables while
  Rust owns bundle validation, replay envelope validation, DAG scheduling,
  data-view construction and `NodeResult` conformance, with branch/merge
  `REFIT` replay covered through a host OOF prediction-cache vtable;
- C ABI C-language conformance test that compiles and runs a small C program
  against `dag_ml.h` and `libdag_ml_capi`, builds an execution plan through
  the ABI, validates it through the same public ABI, then drives non-mock
  replay through controller, data-provider and artifact-store vtables;
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
- published `GraphSpec` JSON Schema with the branch/merge graph as canonical
  fixture, contract validation in `scripts/validate_contracts.py`, and C ABI
  contract discovery through `dagml_graph_spec_contract_json`;
- published `CampaignSpec` JSON Schema with the OOF generation campaign as
  canonical fixture, contract validation in `scripts/validate_contracts.py`,
  and C ABI contract discovery/validation through
  `dagml_campaign_spec_contract_json` and `dagml_campaign_validate_json`;
- published neutral `ModelInputSpec` and `DataPlan` contracts with JSON
  Schemas, canonical tabular-fusion fixtures, Rust validators and C ABI
  contract/validation helpers for controller and data-planner bindings;
- controller manifests now parse and validate `data_requirements` as a
  versioned `ModelInputSpec` when present, so controller-side data/model
  compatibility is no longer an unchecked opaque JSON blob;
- published `ControllerManifest` JSON Schema and canonical data-aware fixture,
  with Rust/C ABI contract discovery and contract-script validation, so
  bindings have a versioned public shape for controller capabilities, ports,
  fit/RNG/artifact policies and typed data requirements;
- C ABI exposes single-manifest and manifest-list validation helpers, including
  `ModelInputSpec` data-requirement checks and duplicate controller-id refusal;
- published `SelectionPolicy` and `SelectionDecision` JSON Schemas with
  canonical fixtures, contract validation in `scripts/validate_contracts.py`,
  and C ABI contract discovery/validation helpers, so refit selection records
  expose metric level, objective and deterministic ranking as binding-level
  contracts;
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

- direct Python object/YAML nirs4all DSL frontends; the Rust importer expects
  serialized JSON descriptors and keeps host object resolution in bindings;
- advanced search-space compiler/lowering beyond typed node-parameter variants
  and coordinated override dimensions from `PipelineDslSpec`;
- artifact binary deserialization/loading into host-native model objects beyond
  the implemented portable artifact reference contract, file-backed artifact
  manifest and file-backed artifact payload store;
- Arrow prediction storage and typed tensor/cache ABI surfaces beyond the
  current owned row-major/column-major F64 prediction-block, bundle-cache
  and new `dag-ml-arrow` Arrow IPC codec (the codec covers both sample and
  non-sample aggregated prediction blocks);
- production host controller adapters with native libraries or
  language-specific bindings beyond the sklearn production,
  prospectr, and mdatools (pls/pca/plsda) controllers — remaining
  backlog (Python SpectroChemPy, Python Orange-Spectroscopy,
  stateful `msc` reference-spectrum persistence, mdatools `simca`
  and `mcrals`) is recorded in `docs/HOST_ADAPTER_BACKLOG.md`. The
  sklearn production slice
  (item #1) is shipped through F.1–F.3:
  `examples/adapters/sklearn_production_controller.py` (operator
  selectors over sklearn.preprocessing/linear_model/ensemble/decomposition
  + joblib disk persistence + structured error frames + signal-based
  fit timeout), and `examples/controllers/sklearn_production.controller.json`
  (validated ControllerManifest with manifest-to-registry parity test).
  The R prospectr slice (item #2) is shipped through G.1–G.2:
  `examples/adapters/prospectr_process_controller.R` (process-adapter
  JSONL framing in R from scratch, jsonlite-backed describe fast
  path, structured `AdapterTaskError` condition for worker survival,
  fold/REFIT/PREDICT partition leakage checks, dispatch for six
  stateless prospectr operators — SNV, savitzkyGolay, gapDer,
  binning, continuumRemoval) and
  `examples/controllers/prospectr.controller.json` (transform-kind
  manifest with the same alias-set parity test pattern). `msc` is
  intentionally excluded from the prospectr controller until
  reference-spectrum persistence is wired. The R mdatools slice
  (item #3) is shipped through H.1–H.2:
  `examples/adapters/mdatools_process_controller.R` (model-kind
  controller reusing the G.1 JSONL scaffold, adding `saveRDS`/
  `readRDS`-backed artifact persistence with the same basename
  confinement primitive as F.1's joblib path, dispatching `pls`
  via the regression input shape and `pca` via the unsupervised
  shape with the first principal-component score as the per-sample
  prediction value; PREDICT round-trips both operators through the
  RDS bundle and asserts byte-equal predictions),
  `examples/controllers/mdatools.controller.json` (manifest with
  `rng_policy=externally_deterministic` since mdatools is internally
  deterministic but does not accept an external seed). Slice K.1
  adds the classification dispatch shape for `plsda` (synthetic
  binary labels via median thresholding, `plsdares.c.pred`
  extraction). `simca` (alternative classifier), `mcrals` (matrix
  factorisation) and the prospectr stateful `msc` path are still
  deferred;
- production `dag-ml-data` provider backends beyond the current in-memory C
  conformance provider for `branch_view_plans` modes the arena cannot
  natively execute (`by_metadata`, `by_tag`, `by_filter` — the by_source
  contract is now pinned by C ABI conformance tests).

Next recommended task:

Items #1 (sklearn production), #2 (R prospectr), #3 (R mdatools
pls/pca/plsda), and #6 (YAML controller registry) are all shipped,
and the Apache Arrow IPC codec for prediction caches now exists in
the new `dag-ml-arrow` crate. The remaining backlog items —
SpectroChemPy and Orange-Spectroscopy Python adapters (items #4–#5),
the mdatools `simca`/`mcrals` operators, the prospectr stateful
`msc` path, host-filtered branch_view modes for non-`by_source`
selectors — are tracked in `docs/HOST_ADAPTER_BACKLOG.md` and can
be picked up in priority order based on operator demand.
