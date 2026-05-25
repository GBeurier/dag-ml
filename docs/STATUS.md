# Status

Current state: OOF/data-contract foundation plus coordinator alignment spec.

Implemented:

- Rust workspace with core, facade, C ABI and CLI crates;
- graph model and validation;
- fold identity models and deterministic identity splitters;
- OOF campaign fixtures, joins and leakage refusal;
- campaign and OOF fixture fingerprints;
- deterministic control seed derivation;
- C ABI graph validation entry point;
- `dag-ml-data` fixture integration through data-plan fingerprints;
- example graph fixture;
- CI workflow.

Not implemented yet:

- full DSL compiler;
- search-space enumerator;
- controller manifest and registry;
- `GraphPlan`, `CampaignPlan`, `ExecutionPlan`, `NodePlan`, `NodeTask`,
  `NodeResult` and `RunContext`;
- executor and scheduler;
- split invocation as campaign-plan controller call;
- artifact/cache/lineage stores;
- Arrow prediction storage;
- host controller adapters;
- integration with `dag-ml-data` runtime plans.

Next recommended task:

Implement the coordinator-visible layer from `docs/COORDINATOR_SPEC.md`:
`ControllerManifest`, `ControllerRegistry`, `CampaignSpec`, `ExecutionPlan`,
mock controllers, sequential scheduler, prediction store and lineage recorder.
