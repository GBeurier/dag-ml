# ADR-26: PREDICT cohorts are distinct from CV relations

**Status**: proposed (2026-08-26)  
**Supersedes**: none. Preserves ADR-05's fold/OOF safety invariant and
ADR-07's native aggregation ownership.  
**Blocks**: native classification target-vote aggregation on external/test
cohorts, public holdout scoring, and any binding that emits relations outside
the CV fold universe.

## Context

The existing `SampleRelationSet` is the authority for CV leakage checks, OOF
coverage, selection and training influence. It intentionally requires every
sample and origin to be a member of the `FoldSet` universe. A binding that
puts final-test or inference rows into this relation set is therefore refused.

Historical classification can run row-level CV and then use a native `vote`
reducer over a target/repetition group on a final PREDICT cohort. Making the
CV relation set broad enough to contain that cohort would weaken ADR-05 and
could let an off-fold observation influence OOF scoring or model selection.

## Decision

1. A future external-data-envelope **V2** adds an optional,
   `PREDICT`-only `predict_cohort` object. V1 remains readable and has no
   predict cohort. A V1 document carrying this member, a V2 document with an
   incomplete member, or an unknown role is refused before materialization.
2. `predict_cohort` is a closed, separately fingerprinted relation contract:
   `role` (`external_test` or `inference`), stable physical sample IDs,
   origin IDs, target names, its `SampleRelationSet`, data-content
   fingerprint, optional target-content fingerprint, and canonical relation
   fingerprint. Its records must be complete and unique for its declared
   cohort; its group/target aggregation mapping is never inferred from
   positions or from a CV relation.
3. CV `coordinator_relations` remains exact to the `FoldSet` and is the only
   relation source usable by `FIT_CV`, `REFIT`, OOF validation, scoring,
   selection, conformal calibration and training influence. Implementations
   must retain both universe checks in ADR-05; `predict_cohort` is not a
   bypass or an additive union.
4. A scheduler may resolve `predict_cohort` only for a `PREDICT` task. Native
   aggregation may use it to turn observation predictions into sample/group
   predictions. It may not create validation blocks, feed an OOF edge, update
   a `ScoreSet` used by SELECT, or be read by a fit/refit provider request.
5. `external_test` must be disjoint from the CV fold universe at both physical
   sample and origin closure. It may have labelled targets for reporting, but
   reports are explicitly held-out reports and cannot rank/select a variant.
   `inference` has no score-bearing target contract; a supplied target is
   refused. A future overlapping calibration role requires its own ADR rather
   than an exception here.
6. Provider attestation is phase-specific: a PREDICT provider returns the
   exact cohort fingerprint and relation bytes requested by the scheduler.
   Mismatch, absent cohort, extra/duplicate identity, a relation outside its
   declared cohort, or use during any other phase fails before scientific data
   access. Replay/package persistence binds the output block's sample IDs,
   aggregation policy and cohort fingerprint so a new prediction cohort cannot
   be substituted after load.

## Consequences

- The current classification repetition/vote path remains a typed refusal
  until envelope V2, scheduler/provider attestations, JSON schemas, fixtures
  and process-separated replay tests land together.
- Bindings must emit ordinary row-level `FoldSet`s for legacy target-vote CV;
  they must not reuse physical target groups to alter the split. Their test
  cohort is emitted only through `predict_cohort`.
- Required negatives include V1/V2 mixing, off-fold CV relation, overlapping
  external-test origin, missing or reordered cohort IDs, provider fingerprint
  drift, PREDICT relation used by FIT_CV/REFIT, and a re-signed package with a
  substituted cohort/output block.

## Blocks

`ExternalDataPlanEnvelope` V2 and schemas, data-provider phase attestation,
PREDICT aggregation routing, package/replay persistence, the Python/DAG-ML
binding producer, and native-versus-legacy vote parity fixtures.
