# Roadmap

## Phase 0: Contracts Frozen

Definition of done:

- Rust core types for graph, phase, fold, prediction and OOF checks;
- C ABI status, handle, bytes and vtable conventions;
- source design docs moved into `docs/design/source`;
- first CLI and tests pass.

## Phase 1: Sequential Core

Definition of done:

- coordinator contract from `docs/COORDINATOR_SPEC.md` represented in Rust;
- controller manifest and registry validation;
- `GraphPlan`, `CampaignPlan`, `ExecutionPlan`, `NodePlan`, `NodeTask`,
  `NodeResult` and `RunContext`;
- search-space enumeration scaffold;
- split invocation model: splitters produce `FoldSet` through the campaign plan,
  not through ordinary data-transform nodes;
- leakage-unit, aggregation and data/model shape policies;
- shape deltas for augmentation, feature selection, filtering and fusion;
- sequential fold/variant executor;
- mock controllers proving external operator orchestration;
- `PredictionStore` and OOF join with leakage rejection;
- deterministic `SeedContext`.

## Phase 2: Host Controllers

Definition of done:

- Python controller adapter for sklearn smoke tests;
- native C++ controller shim for `nirs4all-methods`;
- `describe` blob validation;
- artifact handle release tests.

## Phase 3: Integration With `dag-ml-data`

Definition of done:

- `DataPlan` request/response over canonical JSON;
- data-handle liveness arena;
- schema fingerprint checks at predict/replay;
- UC6 stacking fixture passes end to end.

## Phase 4: Parallelism

Definition of done:

- thread scheduler for native/GIL-free controllers;
- process scheduler design for R and GIL-bound workloads;
- deterministic reducer order;
- stress tests across folds and variants.

## Phase 5: Bundles And Replay

Definition of done:

- train bundle includes graph, selected plan, artifacts, fingerprints and lineage;
- `PREDICT` replay works on new data;
- `EXPLAIN` hooks can pass opaque explanation payloads.
