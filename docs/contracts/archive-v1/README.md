# SAVE-001 archive/workspace V1 contract

Status: contract freeze only. SAVE-001 does not ship a writer, ZIP reader, C ABI,
binding, or a migration implementation.

The future product writer is singular and aggregate-owned:
`nirs4all-core.archive_workspace_writer.v1`. DAG-ML remains the owner of its
graph/training/replay contracts, not an archive writer. The V1 wire deliberately
has no `implementation_status` field.

## Physical and replay profile

An archive V1 is a ZIP file whose dispatch member is exactly `manifest.json`.
A future reader must inspect that member and the central-directory metadata first,
enforce the fixed entry, per-member, total-uncompressed, and compression-ratio
budgets, reject non-regular files and unsafe/colliding POSIX names, and only then
open payload bytes. The manifest inventories every payload member with its raw
SHA-256 and uncompressed size. Semantic fingerprints are separately named with
their profile; they never substitute for raw-byte integrity. `manifest.json` is
the intentionally non-inventoried dispatch member (a byte hash inside itself is
a fixed-point impossibility); a future non-null signature authenticates its
semantic fingerprint. Unsigned V1 therefore makes no authenticity claim.

V1 is complete for replay because it requires the exact DAG-ML
`PortablePredictorPackage` V1 and `GraphSpec` V1, plus the runtime-produced
`ExecutionBundle`, `TrainingOutcome`, cache-payload-set, and score-set **V2**
members. Those four V2 references state `producer_port_required: true`; V1/V2
mixing or a port-implicit runtime reference is refused. This archive contract
does not down-convert a V2 artifact: ADR-21 treats a port-explicit migration as
a new attestation. Any later artifact is explicitly marked
`deferred_future_contract` and is forbidden from affecting replay.
`external_reference` is not a valid V1 host state, so a `.n4a` cannot depend on
off-archive model bytes.

Those six required replay members are distinct inventory paths. Their semantic
fingerprints are non-null and profile-specific: graph uses the retained
historical DAG-ML serde JSON profile (never TCV1); package, bundle and outcome
use their TCV1 identities; cache payload and score-set identities retain their
historical serde profile. N4MM and N4MOPT members may not alias, and host
artifact IDs are unique.

N4MM and N4MOPT are separate, format-version-1, ABI-major-2 Methods references;
they are not interchangeable. An optional N4D entry is only an
`n4d_aggregate_reference` owned by `nirs4all-core`; it intentionally makes no
claim to define an N4D format or parser. Every host artifact carries the exact
controller, plugin, runtime, ABI, and capability identifiers required to load it.
Code-bearing `pickle`, `joblib`, and `rds` remain explicit host opt-in.

## Versioning, legacy, and workspaces

V1 accepts exactly schema version `1`; it does not pre-accept a future version.
An incompatible V2 needs a new schema and the ADR-02 dual-read/migration decision;
this V1 reader is fail-closed. The historical form is separately dispatched as a
ZIP with `manifest.json`, through the real retained reader
`nirs4all.pipeline.bundle.loader.BundleLoader`, whose current maximum historical
`bundle_format_version` is `1.0`. Historical forms are never silently reclassified
as V1.

Migration is one-way copy-on-write: its nullable provenance records source raw
SHA-256, legacy format/version, tool/version, and requires both `copy_on_write`
and `source_retained` when it imports a legacy source. There is no V1-to-legacy
writer. The workspace migration fixture binds that SHA-256 to a deterministic,
synthetic historical ZIP whose manifest is accepted by the retained
`BundleLoader`; it carries no user data or serialized artifact.

A workspace snapshot retains ADR-08's SQLite/Parquet/artifacts model but carries
a checkpoint ID, transaction ID, declared run IDs and a complete payload inventory.
It supports the one root workspace SQLite member plus multiple per-run Parquet,
artifact or ordinary members. The inventory is closed against every physical
`workspace/` payload and uses a frozen allowlist: the SQLite snapshot is a root
`workspace/*.sqlite` path; non-SQLite payloads are run-scoped under
`workspace/runs/<run>/...`. Its live exclusions are exactly `.session.lock` and
`live-session/**`; locks, open transactions, and session state are never serialized.
SQLite `-wal`, `-shm`, rollback/statement/super-journal, and temporary database
names are independently refused even when an inventory labels one as an
ordinary payload.

`security.signature` is either null or a structured reservation. A non-null
reservation records a canonical manifest-preimage SHA-256 and its exact profile
and preimage rule while keeping algorithm, key, signature, and trust root null.
It is deliberately not evidence that signing is implemented or complete. Core
members remain closed; ADR-02-compatible optional producer data belongs only in
the namespace-keyed `extensions` object.

Run `python3 scripts/validate_archive_v1_contract.py` for the self-contained
schema/semantic/mutation gate. It is imported additively by
`scripts/validate_contracts.py`; no shared conformance-pack or W1 hash changes
are made by SAVE-001. CI additionally executes the archive test module against
the pinned retained `nirs4all` checkout, so the synthetic legacy ZIP remains
readable through the real `BundleLoader` rather than only through dispatch
shape checks. Those tests also materialize every frozen refusal as a bounded
ZIP and create the SQLite database, WAL, SHM, and rollback-journal bytes with
the stdlib SQLite binding before asserting that the physical validator refuses
them.

The shared lockstep is intentionally not bypassed for this archive-only gate:
when a local `dag-ml-data` checkout is on an unpaired revision, its separate W1
mismatch remains evidence that the cross-repository validation is not green.
Pull requests changing these contract files still require the paired
`dag-ml-data` branch in CI.
