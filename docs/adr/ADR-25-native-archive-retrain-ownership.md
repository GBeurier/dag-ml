# ADR-25: Native archive retrain ownership

**Status**: accepted (2026-08-26)
**Supersedes**: none. It deliberately preserves ADR-21 replay ownership and
ADR-23 Archive V2 semantics.
**Blocks**: API-004 archive-backed full retrain, any Archive V3 writer/reader,
and any public claim that a loaded Archive V2 session can retrain.

## Context

`PortablePredictorPackage` V2 and Archive V2 are portable prediction
contracts. Their public replay operation is intentionally restricted to
`PREDICT` and `EXPLAIN`; it must not infer training inputs, targets, folds,
or a new training request from a prediction cohort. Although the scheduler can
execute a `REFIT` phase internally, exposing that through the prediction replay
wire would let an old package silently acquire a new model without a new
outcome, artifact inventory, data-identity attestation, or lineage.

Full retrain needs a new artifact and an auditable parent-child relationship.
Transfer and finetuning additionally require explicit controller capabilities;
they cannot be represented as aliases of full retrain.

## Decision

1. Archive V2 and Package V2 remain immutable, dual-readable, and
   `PREDICT`/`EXPLAIN`-only. A loaded V2 archive refuses every retrain request
   before materializing data or artifacts. It is not upgraded in place.
2. Archive-backed native full retrain is a distinct, future **Archive V3 /
   Package V3** operation. It consumes a closed `PortableRefitRecipe` owned by
   DAG-ML, never Python pipeline configuration or a host runner.
3. The recipe binds the parent package fingerprint, parent outcome reference,
   effective plan and selected variant fingerprints, exact portable controller
   identities/capabilities, selected parameter projection, required target
   schema, and a recipe fingerprint. It permits only controllers declaring
   native portable full-refit capability. Missing, host-sidecar, ambiguous or
   nonportable inputs are refused before data access.
   DAG-ML derives the child execution plan from that selected parent plan and
   the target request's data bindings/fold universe.  The source cohort's raw
   data-envelope schema/relation and physical fold identities are never copied
   into a new cohort, but a target request cannot supply topology, parameters,
   variants or controller policy.
4. The retrain request supplies a new, target-bound cohort with explicit stable
   sample identities, feature and target content fingerprints, relation set,
   fold policy and influence manifest. DAG-ML constructs and signs a fresh
   training operation; it does not reuse the source request fingerprint,
   validation reports, selection decision, conformal calibration, or fitted
   handles.
5. Successful retrain produces a new `TrainingOutcome`, `ExecutionBundle`,
   raw native artifact inventory and Package V3. Its mandatory provenance
   records the parent package/outcome/recipe fingerprints and the new cohort
   identities. The original archive remains byte-identical. Conformal state is
   absent by default and can only be attached through its own calibrated V3
   contract; it is never copied to a changed cohort.
6. `transfer` and `finetune` are separate capability names and separate recipe
   modes. Until their native controller contracts exist, public APIs refuse
   them before materialization. No retrain failure may invoke a legacy runner
   unless the caller selected the ADR-24 rollback profile explicitly.

## Consequences

- API-004 can expose `retrain(source=archive, mode="full", engine="native")`
  only after the V3 recipe, writer/reader, controller and end-to-end tests are
  published together.
- V2 product documentation remains honest: archive prediction is supported;
  archive retrain is a typed capability refusal.
- The proof gate is process-separated: source Archive V3 → close all source
  handles → target-bound full refit → new Archive V3 → fresh process PREDICT,
  with exact parent lineage and a new native artifact. Tamper, missing target,
  mixed schema family, host-sidecar and transfer/finetune negatives must fail
  before provider execution.

## Blocks

Package/Archive V3 schemas and migrations, a scheduler-owned retrain operation,
the Methods full-refit controller route, Core Archive V3 storage, the public
Python/Rust APIs, Studio product flows and the API-004 release gate.
