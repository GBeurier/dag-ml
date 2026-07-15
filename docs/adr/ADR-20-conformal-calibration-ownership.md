# ADR-20: Native conformal calibration ownership and identity boundary

**Status**: accepted (2026-07-11)
**Blocks**: native conformal contracts/runtime, estimator outcomes, conformal bindings and bundle integration.

## Context

Conformal calibration needs predictions, targets, stable identities and proof that
calibration units never influenced fitting or selection. DAG-ML already owns those
generic coordination facts, while fitted operators and feature buffers stay host-owned
and nirs4all-methods owns native model/optimizer algorithms. Adding a scheduler phase,
prediction partition or public ABI before the statistical contracts are frozen would
spread one operation across every controller and binding without improving safety.

The existing `.n4a` name is the complete nirs4all bundle contract. N4MM is the portable
nirs4all-methods model payload. They are different artifacts and must not compete for an
extension or ownership boundary.

## Decision

1. DAG-ML owns identity-aligned, generic conformal score/quantile/application logic and
   the cohort, influence, predictor-binding, calibration-artifact and robustness
   contracts. It does not inspect feature buffers or fitted model internals.
2. Calibration V1 is an explicit core operation after replaying the frozen predictor with
   ordinary `PREDICT` semantics. `DataCohortRole::calibration` is independent of
   `Phase`, `PredictionPartition` and `NodeKind`; none of those enums gains a calibration
   member in V1.
3. Split absolute-residual V1 calibrates at the physical `SampleId` unit only. Relation
   rows remain observation-level where repetitions exist. Leakage checks compare the
   calibration sample/origin closure against the exact sample/origin closure recorded by
   `TrainingInfluenceManifest` for fitting, HPO/selection, early stopping,
   weighting/resampling and trained meta-aggregation.
4. A digest alone is not an intersection proof. V1 retains sorted exact identity sets in
   the runtime and persisted manifest. A privacy-preserving proof representation requires
   a later ADR and must preserve deterministic disjointness checks.
5. A cohort relation fingerprint describes that cohort. Predictor data-binding and
   `TrainingInfluenceManifest.relation_fingerprint` describe the development/training
   relation and must match each other; they must not be equated with the calibration
   relation. Cohort, influence and opaque data-content fingerprints remain separate.
6. `PredictorBinding` is derived from validated plan/bundle state and becomes stale when
   any graph, campaign, controller, data-plan/relation, selected variant/patch, artifact
   content, output port, target processing/order/unit, aggregation or prediction-source
   component changes. It binds the exact `TrainingOutcome` and influence manifest as two
   sibling fingerprints. The outcome owns the influence manifest; the manifest never points
   back to the outcome, avoiding a recursive fingerprint. Every
   bound output's complete transitive `input_nodes` closure participates: each node that
   supports `FIT_CV` has per-fold lineage and fit/transform/trained-meta influence, and
   each node that supports `REFIT` has refit lineage when refit completes. Artifact
   persistence is capability-driven: closure nodes advertising `emits_artifacts` must
   have bundle, lineage and predictor bindings; nodes without that capability must not
   invent an artifact. Artifact data/OOF requirement links are exact, selected patches
   are exactly the selected variant's leaf overrides, and patches require selection
   influence. Calibration requires content fingerprints for every predictor artifact,
   including legacy artifacts for which replay permits weaker metadata. V1 output is explicitly `unit_level=physical_sample`,
   `prediction_level=sample`, `prediction_kind=regression_point`.
7. The canonical wire representation always carries a strictly increasing, unique
   `coverages` array of finite values in `(0, 1)`. Host APIs may accept scalar sugar and
   normalize it before validation. A finite-sample rank beyond the calibration size is
   either an error or the tagged JSON value `{"status":"unbounded"}`; JSON infinity is
   forbidden. Rank uses exact rational arithmetic over the shortest round-trip decimal
   token of the validated binary64 coverage value:
   `ceil((n+1) * Decimal(shortest_coverage))`; binary multiplication and ambient decimal
   precision are not normative. Integer residual inputs are restricted to the portable
   exact range `0..2^53-1`. A persisted finite quantile `value` is always a binary64 JSON
   float token (for example `2.0`, never integer token `2`) so one mathematical quantile
   cannot acquire two checksums through two JSON number types.
8. New W0 fingerprints use **DAG-ML Typed Canonical Value v1 (TCV1)**, without changing
   existing graph/campaign fingerprints. The SHA-256 preimage is ASCII
   `DAGML-TCV1` followed by NUL and one recursively encoded value:
   - null/false/true are the one-byte tags `N`/`F`/`T`;
   - integers use tag `I`, an unsigned 64-bit big-endian byte length, then canonical
     decimal ASCII without plus sign or leading zero, within `i64_min..u64_max`;
   - finite floats use tag `D` followed by IEEE-754 binary64 big-endian bytes, with
     negative zero normalized to positive zero;
   - strings use NFC normalization, tag `S`, unsigned 64-bit big-endian UTF-8 byte
     length, then bytes; surrogate code points are refused;
   - arrays use tag `A`, unsigned 64-bit big-endian member count, then ordered members;
   - objects use tag `O`, unsigned 64-bit big-endian member count, then each normalized
     key encoded as a string followed by its value, keys sorted by normalized UTF-8 bytes;
     normalization collisions are refused.

   Bindings never hash host JSON bytes: DAG-ML validates and emits TCV1 fingerprints.
   The cohort/influence manifest fingerprints omit their own fingerprint member; the
   artifact checksum omits `checksum`. All nullable/defaulted members that participate in
   these hashes are explicit on the canonical wire (`null`, `{}` or `[]`), never omitted.
   Golden preimage vectors freeze the profile.
   Legacy DAG-ML fingerprints (SHA-256 of deterministic serde JSON), DAG-ML TCV1
   fingerprints (normalized UTF-8 object-key order) and nirs4all-methods RFC 8785/JCS
   fingerprints (UTF-16 object-key order) are disjoint profiles with disjoint ownership.
   No object may be co-hashed or compared across profiles; a contract names exactly one
   profile for each fingerprint field.
9. `.n4a` remains the complete pipeline bundle. N4MM naming remains owned by
   nirs4all-methods. A `.n4am` envelope is deferred and conformal V1 cannot depend on it.
   Calibration artifacts will later enter `.n4a` additively under ADR-02.
10. Robustness V1 has exactly three lifecycle modes. `clean_frozen` reuses the
   predictor and any existing calibrator; `matched_recalibration` reuses the predictor
   and creates a calibrator bound to the perturbed calibration cohort;
   `structural_refit` replaces a node inside the authoritative predictor closure,
   refits the predictor and either recalibrates or explicitly invalidates calibration.
   Node replacement is valid if and only if the mode is `structural_refit`. Every
   scenario declares binary64 severity `0.0`; an identity-only scenario declares
   exactly `[0.0]`. At severity zero all before/after identities are equal, the
   predictor is reused, and calibration is reused when present or absent otherwise.
11. W0 publishes JSON contracts, fixtures and a test-only oracle only. No C ABI, PyO3,
   WASM or Rust public surface is exposed until the contracts and migrations pass review.

## Consequences

- Calibration data can be predicted through existing replay machinery without being
  reclassified as training or validation data.
- Recalibration is mandatory after any predictor-binding change; a stale artifact is a
  typed refusal, never a warning-only fallback.
- Group-conditional, CV+/jackknife+, classification and privacy-preserving manifests are
  additive follow-ups with their own statistical and migration gates.
- The first conformance fixture is local to DAG-ML, not copied into the shared
  `dag-ml-data` pack; the data repository continues to own relation production only.

## Blocks

This ADR must land before native conformal Rust types, estimator outcomes, bundle fields,
binding exposure or public calibration entry points.
