---
orphan: true
---

# V1 stability conformance refresh — September 2026

The published `v0.3.23` tag contains forty stale source-byte attestations in
four conformance packs (conformal 2, criteria 3, training 17, replay 18).
An independent SHA-256 scan compared every stale artifact against
`git show v0.3.23:<path>`: the audited bytes were identical to that tag.
These were not scientific-oracle failures caused by documentation cleanup.
Later native predictor descriptors, embedded pipelines, cumulative HPO resume,
conformal presentation, Arrow persistence and registry pins changed sources
without republishing their enclosing source-attestation inventories.

The stability patch renews only pack artifact hashes, their self-checksums,
and the explicitly pinned training-pack byte/checksum authorities consumed by
the replay generator, validator and test. The inventories still attest exact
source bytes; byte mutation and omitted-dependency negative tests remain
mandatory. Legacy replay authorities, schemas, fixture data, expected numeric
values and archive fingerprints are **not** regenerated. Historical inventories
remain recoverable byte-for-byte from the published tag.

The unchanged generator also closes a missing transitive schema dependency:
`native_predictor_descriptor.v1.schema.json`, already present in the published
tag and referenced by its execution-bundle schema. Consequently the training
inventory contains 104 artifacts instead of 103. Both consumers pin the new
exact count; no dependency-presence check is removed.

The current source changes additionally attested are HPO canonical object
ordering, the corrected native test-controller parameters, and release CI
qualification. HPO default-build identities remain unchanged; the fix makes
the aggregate's `preserve_order` feature obey the same published identity.

Revalidation includes independent artifact SHA-256 comparison; conformal,
criteria, training and replay positive/negative oracles; archive fixtures;
Rust tests in default and `preserve_order` configurations; and the complete
contract validator with the corresponding Data checkout. A refreshed pack
is a source inventory, not a claim that unexecuted platform tests passed.
