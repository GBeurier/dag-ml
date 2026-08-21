# SAVE-001 archive/workspace V1 threat model

This is an executable contract target for SAVE-002, not a claim that a reader
exists today.

| Threat | Required V1 reader/writer response |
| --- | --- |
| ZIP slip, Windows paths, controls, drive paths, DOS reserved names (`CON`, `NUL`, `COM1`, ...), NTFS ADS colons, dot segments or normalized-name collisions | Reject in bounded central-directory dispatch before extraction. Only canonical POSIX regular-file payloads are accepted. |
| ZIP bomb or metadata amplification | Refuse when fixed entry, member-size, total-uncompressed, or compression-ratio budget is exceeded. Tests mutate metadata only; they never create a large bomb. |
| Missing, extra, swapped, or edited payload | Require a closed member inventory and exact raw SHA-256/size match before interpretation. Semantic fingerprints have a separately named profile and cannot replace raw integrity. |
| Edited unsigned manifest | V1 does not claim authenticity for an unsigned dispatch manifest. A nullable, separately typed signature is reserved for a later trust-root contract; it must not be confused with member integrity. |
| Schema confusion or future producer | Dispatch accepts only V1. V2 requires ADR-02-compatible migration/dual-read work; unknown versions and unknown host states are refused. |
| Code execution through host serialization | Default deny. `pickle`, `joblib`, and `rds` require declared backend, matching typed host capability, verified payload hash, and explicit host opt-in. |
| Incomplete replay, port-implicit runtime evidence, V1/V2 runtime mixing, or missing model bytes | Require the exact V1 PortablePredictorPackage/GraphSpec and exact V2 ExecutionBundle, TrainingOutcome, cache-payload-set and ScoreSet references with `producer_port_required`. `external_reference` is forbidden in `.n4a`; deferred artifacts cannot affect replay. |
| Ownership/parser substitution | Keep N4MM and N4MOPT separate and Methods-owned; name N4D only as an aggregate reference, not a new format claim. |
| Legacy overwrite / downgrade | Keep the legacy source and raw source hash; import only copy-on-write. Dispatch legacy ZIPs to the retained `BundleLoader`, never to the V1 parser. |
| Workspace lock, transaction, session capture, SQLite WAL/SHM/rollback/statement/super journals (including current `-mj%06X9%02X` names) or temp files, or an undeclared multi-run payload | Snapshot only checkpointed, run-ID-bound, closed-inventory state; allow only the root SQLite snapshot and run-scoped payload paths, independently reject SQLite live names even when declared `ordinary`, and preserve ADR-08 single-writer behavior. |

Out of scope pending an owned follow-up: signature algorithms and trust roots,
encryption, malware analysis, sandboxing, network retrieval, and actual host
serializer format detection. SAVE-002 must preserve these typed refusals.
