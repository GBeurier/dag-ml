# ADR-21: Public training replay ownership and port-explicit wire evolution

**Status**: accepted (2026-07-12)
**Blocks**: public training replay runtime, native bindings, conformal prediction replay,
and nirs4all replay/fine-tuning integration.

## Context

DAG-ML already exposes a low-level phase execution protocol named
`ReplayPhaseRequest` / `ReplayExecutionSummary`, and estimator compatibility fixtures
already use the name `ReplayOutcome`. A public, persisted replay operation must bind a
complete `TrainingOutcome`, a current data cohort and selected output ports without
weakening either legacy contract. It must also remove the ambiguity created when one
node exposes several prediction ports.

The first contract freeze is intentionally ahead of Rust and binding implementation.
Publishing runtime types before the authority, migration and cross-reader rules are
executable would allow incompatible Rust, C, Python and WASM interpretations.

## Decision

1. DAG-ML owns the public training-replay request/outcome contract. The future public
   Rust names remain `ReplayRequest` and `ReplayOutcome`; filenames, schema IDs and
   documentation qualify them as **training replay**. Existing low-level types and the
   legacy estimator `ReplayOutcome` wire remain unchanged.
2. `ReplayRequest` V1 authorizes only `PREDICT` or `EXPLAIN`. It binds the complete
   source outcome fingerprint, the exact sorted data-envelope keys and the exact sorted
   output-binding IDs. Its TCV1 fingerprint omits only `request_fingerprint`. `REFIT`
   is deferred to D8 because it changes predictor identity rather than replaying it.
3. `ReplayOutcome` V1 carries a complete `TrainingOutcomeRef`, the request identity,
   current data identities, plan/bundle cross-links, exact counters, sorted outputs,
   explanations, lineage and warnings. Its TCV1 fingerprint omits only
   `outcome_fingerprint`. Runtime handles and raw feature/spectral buffers are forbidden.
4. Current envelopes must preserve the source schema, data plan, representation,
   source set and `feature_set_id`. Relation, data-content and target-content identities
   describe the current cohort and therefore may change. D4 guarantees unique emitted
   identities and no identity outside the union of current coordinator relations.
   `excluded` is training-only and does not suppress replay prediction. Exact transitive
   row coverage for multi-source and missingness cases is a D4.1 runtime gate.
5. Prediction provenance is the pair `(producer_node, producer_port)`. Public replay
   outputs always use the port-explicit V2 wire. A V1 block without `producer_port` is
   accepted only by a legacy ingress wrapper and only when its node has exactly one
   prediction port. A V1 wrapper containing the field, an unknown port, a non-prediction
   port or an ambiguous omission is rejected.
6. The additive V2 family comprises `NodeResult`, `BoundTrainingOutput`, aggregation
   task/result, process-adapter frame, prediction-cache payload set, `ScoreSet`,
   `ExecutionBundle` and `TrainingOutcome`. Outside `schema_version`, required
   `producer_port`, V2 references and the cache marker
   `dag-ml-json-prediction-blocks-v2`, every V1 constraint is preserved. V1 writers
   remain byte-shaped as before and V1 readers/version validators reject V2 instances.
   The historical V1 `ScoreSet` JSON Schema uses `schema_version >= 1`, so an empty V2
   score set is a known schema-only exception; the V1 reader's exact-version check is
   authoritative. Non-empty V2 score fixtures are also rejected by the V1 schema because
   their reports carry `producer_port`. There is no package V2 in D4;
   portable-package evolution is deferred to D8.
7. A `ScoreSet` V2 report is unique by
   `(producer_node, producer_port, variant_id, partition, fold_id, level)`. Reports that
   differ only by port are distinct; duplicate full keys are invalid. Selection
   cross-links use the report matching the selected variant, validation partition and
   aggregate fold coordinate.
8. Cache payload fingerprints remain deterministic serde/stable-JSON fingerprints,
   now including `producer_port` in the canonical block preimage after
   `producer_node`. The Arrow layout does not change; only JSON payload identity changes.
   Migrating an already signed V1 outcome creates a new V2 outcome ID and fingerprint:
   the port changes the signed provenance and cache identity, so the migrated outcome is
   a new logical attestation rather than an in-place representation rewrite.
9. Classification probability rows contain one contiguous class segment per target,
   in binding vocabulary order. Values are finite in `[0, 1]`; sequential binary64
   summation in class order must be within absolute tolerance `1e-12` of one. Values are
   never normalized. Class-label outputs contain finite binary64 values integral in
   value and interpreted as zero-based vocabulary indices.
10. `EXPLAIN` freezes node/port provenance, a non-blank method and an optional target
    validated against the bound output. Its payload is strict portable JSON, may not
    contain handles and rejects the reserved raw-input keys `raw_features`,
    `feature_matrix`, `raw_spectra` and `raw_wavelengths`. This blacklist is not a
    semantic proof that an arbitrary controller-specific JSON value contains no input
    data; producers remain responsible for that boundary. D4 makes no unit, cohort or
    row attribution claim for the opaque payload. A typed attribution schema and those
    guarantees are deferred to W2-EXPLAIN E3-E6.
11. D4.0/D5a-C publishes schemas, deterministic fixtures, two independent semantic validators,
    migration/cross-reader tests and a separate exact conformance pack. It makes no
    Rust runtime support claim. D5a-R must first propagate port provenance, D5 must close
    binding/scoring by `(node, port)`, and D4.1 must then implement attached replay with
    golden serialization tests and review.

## Consequences

- Conformal calibration can later consume ordinary, identity-bound `PREDICT` replay
  without inventing a calibration scheduler phase.
- Multi-port models, trained meta-aggregators and explanations have unambiguous
  provenance across process, aggregation, cache, scoring and outcome boundaries.
- Existing V1 artifacts remain valid, but an explicit migration produces a new signed
  V2 identity; silently inserting a port into a signed artifact is forbidden.
- Bindings and nirs4all may expose ergonomic sugar only after normalizing it to the
  canonical request and validating the returned outcome.

## Blocks

The canonical critical path is
`(D4.0 + D5a-C) -> D5a-R -> D5 -> D4.1 -> D4.2 -> D8`. D5a-R owns runtime
port propagation, D5 owns binding/scoring, D4.1 owns attached `ReplayOutcome` plus exact
cohort coverage, and D4.2 alone opens owning C/PyO3/Python replay bindings. D8 owns
stateless `REFIT`, portable-package evolution and any package-level replay envelope.
