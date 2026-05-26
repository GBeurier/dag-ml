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
- runtime enforcement of `requires_oof` prediction edges before downstream
  training controllers can consume upstream prediction inputs;
- deterministic `SeedContext`;
- identity-aligned regression metric reports for validated sample, target and
  group prediction blocks, usable directly as selection candidate scores.

Status: implemented as the first core slice. Remaining Phase 1 hardening is
mostly richer fixtures and replacing smoke adapters with production host
controllers.

## Phase 2: Host Controllers

Definition of done:

- Python controller adapter for sklearn smoke tests;
- native C++ controller shim for `nirs4all-methods`;
- `describe` blob validation;
- artifact handle release tests.

Status: first process adapter smoke implemented for campaign, refit and replay.
The CLI can invoke an external Python script per `NodeTask` either one-shot,
as a single persistent JSONL process, or as a prewarmed persistent worker pool
per controller. Pool routing spreads `FIT_CV` by node/variant/fold while keeping
`REFIT` and `PREDICT` sticky by node/variant so stateful artifact handles replay
on the worker that produced them. Process adapters must expose a `--describe`
JSON handshake so the CLI can reject unsupported protocol versions or modes
before a campaign starts. Persistent workers use `control_frames_v1` for
explicit `init`, framed `task`, typed `error` and `close` messages. They are
guarded by coordinator-side timeouts and opt-in retry/restart, with a flaky
adapter fixture covering timeout refusal and recovery. A stateful sklearn smoke
now fits a real sklearn pipeline during `REFIT`, stores it behind an opaque model
handle, and replays `PREDICT` through that handle in the same persistent pool.
The handshake now also gates persistent-worker, worker-env and
parallel-invocation capabilities. This is intentionally not yet a production
Python binding or native worker runtime.

## Phase 2b: Parallel Execution

Status: `dag-ml-core` exposes a bounded `ParallelScheduler` that runs
independent nodes from the same compiled DAG level concurrently and commits
their outputs, predictions and lineage in deterministic order. Runtime
controllers are now `Send + Sync`; the persistent process adapter pool locks per
worker, so parallel scheduler threads can use distinct workers instead of
serializing on the whole pool. The CLI exposes `--scheduler sequential|parallel`
and `--scheduler-workers` for campaign, refit and replay execution commands.
Broader stress tests remain to do.

## Phase 3: Integration With `dag-ml-data`

Definition of done:

- `DataPlan` request/response over canonical JSON;
- data-handle liveness arena;
- schema fingerprint checks at predict/replay;
- UC6 stacking fixture passes end to end.

Status: JSON/fingerprint contract started. `dag-ml-data` now emits a
versioned coordinator envelope, `dag-ml` validates node data bindings against
it, and both repositories publish and compare the v1 JSON Schema artifact for
the shared wire contract in CI. `dag-ml` also validates the sibling
`dag-ml-data` coordinator fixture when that checkout is available. The scheduler
requests an opaque parent data handle plus a fold/refit/predict provider view through
`RuntimeDataProvider`. The C ABI data-provider vtable is aligned on
materialization, view creation, identity, sample-level target and
observation-level feature exports; `feature_arrow` can stay ABI-compatible
while carrying `dag-ml-data` JSON fusion selectors for multi-source feature
joins. The two C headers now share a guarded data-provider vtable ABI version
and compile together in both include orders. A core in-memory provider records
both materialized handles and scoped data-view handles for smoke tests, and
the C conformance suite now links both libraries so `dag-ml` replay can consume
a real `dag-ml-data` f64 provider, read feature/target Arrow exports and verify
data/view handle release. Next missing piece is to turn that sibling-checkout
smoke into a shared conformance pack and broaden provider backends beyond the
in-memory fixture.

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

Status: core and CLI contract started. `ExecutionBundle` now records
plan/controller fingerprints, selected variants, deterministic selection
decisions, refit artifacts and the external data requirements needed for
replay. The CLI can select candidates, build a bundle, and validate that a
bundle plus replay request matches a rebuilt plan and external data envelopes.
Bundles carry an explicit schema version, publish migration policies, reject
unsupported future/zero versions, and refuse old versions unless an explicit
migration edge is declared. The C ABI exposes the same selection and
replay-validation contracts over JSON. The
runtime can now capture refit artifact handles emitted by controllers,
materialize replay data, create predict-scoped data views and materialize refit
artifact handles, then invoke eligible controllers for replay phases without CV
folds. The CLI can also build mock and external-process refit bundles directly
from captured refit artifact records, and can run a stateful process refit plus
immediate replay against captured handles. It can run `FIT_CV` followed by
`REFIT` in a single process context for a branch/merge OOF DAG, proving the
meta-model refit consumes complete CV OOF coverage before bundle capture, then
replay that captured branch/merge bundle through the process adapter with
separate branch/meta data requirements backed by one data-plan envelope. OOF
prediction caches can now be exported from the monolithic JSON payload into a
validated file-backed store directory and reused by replay. Payload-backed CLI
replay now converts validated sample and target/group OOF prediction payloads
into typed columnar f64 buffers behind `RuntimePredictionCacheStore`, keeping
the external bundle contract stable while removing JSON-shaped rows from the
runtime cache path.
It has both mock and external-process execution smokes for campaign,
refit-bundle and replay paths, and the C ABI exposes a mock replay execution
helper returning a JSON summary. The remaining work is production host
adapters, persistent artifact/data stores and Arrow-backed prediction cache
exports.

## Phase 6: Research Provenance Exports

Goal: make a DAG-ML campaign publishable as a reproducible research object
without weakening the internal DAG-ML contracts.

Design rule:

- the internal DAG-ML model remains canonical for OOF, folds, groups,
  repetitions, augmentation origins, leakage units, prediction levels, refit
  conditions, fingerprints and scheduler decisions;
- research standards are export targets, not the execution model used by the
  coordinator;
- exported provenance must be generated from validated plans, bundles,
  lineage, data envelopes, cache manifests and artifact refs, never from
  adapter-side free text.

Definition of done:

- W3C PROV mapping for DAG-ML lineage:
  - `LineageRecord` and phase-scoped node execution become `prov:Activity`;
  - data views, prediction blocks, prediction caches, bundles and model
    artifacts become `prov:Entity`;
  - controllers, adapters, plugins and host runtimes become `prov:Agent`;
  - DAG edges, OOF joins, cache hits, artifact capture and replay materialization
    map to `prov:used`, `prov:wasGeneratedBy`, `prov:wasDerivedFrom` and
    `prov:wasAssociatedWith`;
- Workflow Run RO-Crate export for research bundles:
  - `ro-crate-metadata.json` with a `ComputationalWorkflow` main entity;
  - embedded or referenced `graph.json`, `campaign.json`, `execution_plan.json`,
    `bundle.json`, data-plan envelopes, prediction-cache manifests and artifact
    manifests;
  - `lineage.prov.jsonld` plus checksums/fingerprints for every exported
    contract artifact;
  - software/controller/plugin versions, seeds, unsafe flags and replay
    requirements;
- a DAG-ML extension namespace for ML-specific invariants that W3C PROV and
  RO-Crate do not express natively: fold id, variant id, branch path,
  prediction level, leakage unit, aggregation policy, augmentation origin,
  refit dependency and OOF coverage;
- conformance fixtures that prove a branch/merge OOF sklearn campaign can be
  exported and revalidated as a Workflow Run RO-Crate without losing DAG-ML
  leakage-safety evidence;
- optional later adapters for OpenLineage and MLMD, derived from the same
  internal provenance graph:
  - OpenLineage for data-platform/job lineage;
  - MLMD for ML experiment stores.

Status: branch/merge conformance slice implemented. The core now builds a
validated checksum-rich RO-Crate package containing `execution_plan.json`,
`execution_bundle.json`, `lineage_records.json`, `lineage.prov.jsonld`,
`ro-crate-metadata.json`, optional data envelopes and optional cache/artifact
manifests. The schedulers attach coordinator-owned input lineage from compiled
DAG edges, and the CLI exposes `export-research-provenance` over validated
plans, bundles, real lineage records, data envelopes, prediction-cache stores
and artifact manifests; `validate-research-provenance` reopens those packages,
verifies RO-Crate checksums and reruns the DAG-ML contract validation.
`export-open-lineage` derives an OpenLineage `RunEvent` from the already
validated package and carries DAG-ML-specific reproducibility and OOF-safety
evidence in custom `dagml_*` facets. The C ABI exposes the same validated
research provenance export as owned JSON for non-Rust bindings. A branch/merge
CV+refit contract test exports lineage, OOF cache store, portable artifact
manifest and research provenance from the same captured bundle, then
revalidates the published package. Remaining work is optional MLMD export plus
any external conformance profile we decide to publish. This remains a
publication/export layer and must not replace the Rust coordinator's stricter
internal validation model.
