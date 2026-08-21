# ADR-23: Archive V2 dual-read and native-portable replay

**Status**: accepted (2026-08-21)
**Supersedes**: none. Archive V1 remains frozen and readable.
**Blocks**: Archive V2 writer/reader implementation, native-only portable replay,
and any product archive default change.

## Context

Archive V1 binds `PortablePredictorPackage` V1 while its port-explicit runtime
companions are already V2. Portable package V2 now owns conformal calibration
and calibration-replay evidence, and embeds `ExecutionBundle` V2. Adding those
members to a V1 package, even as null, would change a closed signed wire and
would make package/bundle provenance ambiguous. Host-sidecar packages also
cannot satisfy the strict native-only replay promise.

The product needs a migration decision before implementing a second writer or
reader. Archive V1 contract artifacts and their validator are frozen and must
not be rewritten to manufacture compatibility. W1 conformance-pack hashes are
governed provenance rather than Archive V1 wire bytes: when independently
approved inputs such as the ADR registry, CI, or `validate_contracts.py` change,
the owning workflow must regenerate the canonical packs instead of manually
patching individual hashes.

## Decision

1. The new manifest is exact `schema_version: 2`, profile
   `nirs4all.archive_workspace.v2`, and writer
   `nirs4all-core.archive_workspace_writer.v2`. The product aggregate remains
   the only archive writer; DAG-ML owns the referenced replay contracts.
2. A V2-capable reader dispatches from `manifest.json` before extraction and
   dual-reads exact Archive V1 and exact Archive V2. V1 is read immutably; a
   migration is a one-way, copy-on-write V1-to-V2 operation that retains and
   hashes its V1 source. A V1 reader refuses V2. Unknown versions are refused.
   V1 readability remains required through the native-default release and the
   next two ADR-17 retention releases; removing it requires a superseding ADR.
3. Version families may not mix. Archive V1 references exact
   `PortablePredictorPackage` V1. Archive V2 references exact
   `PortablePredictorPackage` V2 at
   `https://github.com/GBeurier/dag-ml/schemas/portable_predictor_package.v2.schema.json`
   and records `producer_port_required: true`. Package V1 embeds exact
   `ExecutionBundle` V1; Package V2 embeds exact `ExecutionBundle` V2.
4. Archive V2 keeps exact Graph V1 and requires the port-explicit V2
   `ExecutionBundle`, `TrainingOutcome`, prediction-cache payload set and
   `ScoreSet` companion references. The package, graph and four runtime
   companions occupy distinct, closed-inventory members. No V1 artifact is
   silently down-converted or patched in place.
5. Conformal state belongs to `PortablePredictorPackage` V2. The retained
   archive-level `payloads.conformal` slot is required to be null; it is not an
   alternative representation of the package-owned fields.
6. The intentionally narrow Archive V2 **P0 profile is Methods-only**, not a
   claim about every future native-portable artifact kind. Its package declares
   `portable_required`; every binding is `native_portable`; and every refit
   artifact is a raw, plugin-free `n4m_model` with an exact N4MM payload.
   Archive `host_artifacts` is empty, and host sidecars/external references are
   refused before write. N4MM references carry package `artifact_id`; they
   exactly cover and byte-equal `ExecutionBundle.raw_artifact_payloads`. Safe
   `.n4mm` paths, sizes and raw SHA-256 values match the bundle references.
   N4MM uses the explicit `n4mm_raw_sha256` binary semantic profile: its
   semantic fingerprint is exactly the raw N4MM SHA-256, because opaque N4MM
   bytes do not have a JCS representation. The archive never reconstructs a
   Methods model from that identifier.
   Supporting ONNX, SafeTensors or another native kind requires a later policy
   ADR/profile rather than silently widening this P0 gate.
7. The schema, positive fixture, refusal fixture and independent validator are
   the contract gate. They do not claim a Core runtime, archive reader, archive
   writer, C ABI or binding implementation. Archive V1 contract artifacts and
   `validate_archive_v1_contract.py` remain byte-for-byte unchanged. That byte
   freeze does not prohibit canonical W1 pack regeneration required by separately
   approved changes to governed inputs; such regeneration belongs to the W1
   owner/release workflow and must never be replaced by ad hoc hash edits.

## Consequences

- Portable conformal replay has one ownership site and port-explicit provenance.
- A package that needs Python, joblib, pickle, RDS or another host sidecar stays
  on Archive V1 or is refused by the strict V2 preflight; there is no fallback.
- A package with a non-Methods refit artifact is also refused by this P0 profile,
  even if that artifact could later receive a native-portable policy.
- V2 implementations must validate central-directory budgets, raw member hashes,
  closed inventory and the package portability policy before loading model code.
- Migration changes the archive identity and creates a new attestation while
  retaining the immutable source archive.

## Blocks

Archive V2 runtime implementation and the product-level SAVE cutover remain
blocked until `nirs4all-core` consumes this contract with dual-reader,
copy-on-write and pre-write refusal tests.
