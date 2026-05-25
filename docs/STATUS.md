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
- in-memory prediction store and lineage recorder;
- sequential scheduler for DAG order plus campaign execution over
  variant x CV-fold scopes;
- mock controller conformance tests;
- CLI execution-plan validation from graph/campaign/controller JSON fixtures;
- C ABI graph validation entry point;
- `dag-ml-data` fixture integration through data-plan fingerprints;
- coordinator graph/campaign/controller fixtures;
- CI workflow.

Not implemented yet:

- full DSL compiler;
- full search-space compiler/lowering into graph/campaign overrides;
- non-mean aggregation methods and custom aggregation controllers;
- artifact/cache stores;
- Arrow prediction storage;
- host controller adapters;
- integration with `dag-ml-data` runtime plans.

Next recommended task:

Implement the first host-controller path:
Python/sklearn mock adapter or native C++ shim, backed by `dag-ml-data` relation
and data-plan requests, then run an end-to-end OOF stacking fixture through the
new `ExecutionPlan` and scheduler.
