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
- `GraphPlan`, `CampaignSpec`, `ExecutionPlan`, `NodePlan`, `NodeTask`,
  `NodeResult` and `RunContext`;
- split invocation as a campaign-plan controller call;
- deterministic generation/search-space scaffold with variant fingerprints and
  variant seeds;
- leakage-unit policies for sample/target/group/repetition/origin boundaries;
- sample relation validation for repeated observations, shared targets, groups
  and augmentation origins;
- aggregation policy plus mean aggregation from observation predictions to
  sample predictions;
- data/model shape plans and runtime shape deltas;
- data bindings from node inputs to external `dag-ml-data` plan envelopes,
  including explicit feature-set ids for provider `feature_arrow` calls;
- external data-plan envelope validation by schema, plan and relation
  fingerprints;
- runtime data-provider trait with materialization plus fold/refit/predict view
  requests that turn data bindings into scoped opaque task handles;
- `FIT_CV` data routing now gives controllers separate fold-train and
  fold-validation views, so validation OOF predictions can be checked against
  the validation identity set;
- controller-facing `NodeTask.data_views` map carrying the scoped view spec
  beside each data-view handle;
- in-memory runtime data provider with handle records for schema/plan/relation
  traceability and child data-view records for sample partition, source and
  feature-set traceability;
- runtime artifact-store trait plus in-memory refit artifact handle records,
  including capture of controller-emitted refit artifact handles during `REFIT`;
- bundle replay executor that validates plan/bundle/request/data envelopes,
  materializes data and refit artifact handles, and invokes eligible runtime
  controllers for replay phases without CV folds;
- stricter `NodeResult` conformance validation for externally returned run,
  controller, version, variant, fold, branch, seed, params fingerprint, output
  owner, artifact controller fields and artifact handle ownership;
- in-memory prediction store and lineage recorder;
- sequential scheduler for DAG order plus campaign execution over
  variant x CV-fold scopes;
- deterministic metric selection contracts, including grouped candidate
  selection and stable tie-breaking;
- refit execution bundle contracts that bind selected variants, selected
  candidates, refit artifacts, plan fingerprints and replay data requirements;
- explicit execution-bundle schema version with unsupported-version refusal;
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
- stateful sklearn process-controller smoke that fits a real sklearn pipeline
  during `REFIT`, stores it behind an opaque model handle, then replays
  `PREDICT` through the captured handle in the same persistent process;
- persistent JSONL process-controller mode for campaign/replay smoke tests,
  avoiding one process spawn per task and preparing stateful host adapters;
- Python process-controller adapter fixture for campaign/replay smoke tests,
  including data-handle, fold and refit-artifact-handle checks;
- C ABI validation and JSON output helpers for graph, selection decisions,
  grouped selection, execution bundles, replay envelopes and replay requests;
- C ABI data-provider vtable shape aligned with `dag-ml-data`
  materialize/view/identity/target/feature exports plus a tested Rust runtime
  adapter over the vtable;
- C ABI mock replay execution helper that exercises execution-plan, bundle,
  replay request, data envelope and refit artifact handle materialization and
  returns a JSON summary including data view counts;
- standalone sklearn complex OOF demonstrator with repeated observations,
  group-aware splits, train-only augmentation, branch model variants,
  heterogeneous prediction+raw-data merge variants, OOF-based selection and
  final refit report;
- C ABI graph validation entry point;
- `dag-ml-data` fixture integration through schema, plan and relation
  fingerprints;
- coordinator graph/campaign/controller fixtures;
- CI workflow.

Not implemented yet:

- full DSL compiler;
- full search-space compiler/lowering into graph/campaign overrides;
- non-mean aggregation methods and custom aggregation controllers;
- persistent artifact/cache stores;
- Arrow prediction storage;
- production host controller adapters with stable process pools, native
  libraries or language-specific bindings;
- bundle schema migration policy;
- concrete `dag-ml-data` provider implementation with real buffers and handle
  lifecycle arena.

Next recommended task:

Extend the sklearn adapter smoke from the minimal graph to the complex
branch/merge demonstrator so OOF selection, refit artifacts and replay are
validated together.
