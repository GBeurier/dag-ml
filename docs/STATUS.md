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
- data bindings from node inputs to external `dag-ml-data` plan envelopes;
- external data-plan envelope validation by schema, plan and relation
  fingerprints;
- runtime data-provider trait and materialization requests that turn data
  bindings into opaque task handles;
- in-memory runtime data provider with handle records for schema/plan/relation
  traceability;
- in-memory prediction store and lineage recorder;
- sequential scheduler for DAG order plus campaign execution over
  variant x CV-fold scopes;
- deterministic metric selection contracts, including grouped candidate
  selection and stable tie-breaking;
- refit execution bundle contracts that bind selected variants, selected
  candidates, refit artifacts, plan fingerprints and replay data requirements;
- replay request validation for predict, explain and refit phases;
- mock controller conformance tests;
- CLI execution-plan validation from graph/campaign/controller JSON fixtures;
- CLI data-binding validation against a coordinator data-plan envelope;
- CLI mock campaign execution through controller manifests, data bindings,
  in-memory data provider and mock runtime controllers;
- CLI selection, bundle build and bundle replay validation commands with
  fixture-backed integration tests;
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
- artifact/cache stores;
- Arrow prediction storage;
- host controller adapters;
- stable bundle schema versioning and host-binding replay entry points;
- concrete `dag-ml-data` provider implementation with real buffers and handle
  lifecycle arena.

Next recommended task:

Expose the new bundle/selection/replay contracts through the C ABI, then turn
the sklearn demonstrator into the first controller adapter smoke: replace the
Python-side orchestration loop with `dag-ml` scheduler tasks while keeping the
same OOF campaign, heterogeneous merge and refit score policy.
