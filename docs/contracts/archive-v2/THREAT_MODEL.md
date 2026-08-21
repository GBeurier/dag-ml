# Archive V2 P0 threat model

| Threat | Fail-closed contract response |
|---|---|
| ZIP traversal, aliases, devices or links | Inspect the central directory before payload reads; accept only unique, canonical POSIX regular-file names. |
| ZIP bomb or oversized archive | Enforce fixed entry, member, total-uncompressed and compression-ratio budgets before extraction. |
| Unknown or future producer | Dispatch exact V1 or exact V2 only; V1 is immutable and future versions are refused. |
| Archive/package version confusion | Archive V1 pairs only with Package V1; Archive V2 pairs only with Package V2 and port-explicit V2 companions. |
| Null-key smuggling into Package V1 | The closed Package V1 schema rejects both V2 conformal keys even when null and requires embedded Bundle V1. |
| Member substitution or truncation | Bind every member's exact raw SHA-256 and uncompressed size in a closed inventory. |
| Semantic-hash substitution for a model | Require the physical N4MM member; semantic identity never replaces raw payload integrity. |
| Package/archive N4MM divergence | Require artifact-ID closure and byte equality between archive N4MM members and package `raw_artifact_payloads`. |
| Host code execution | Refuse host-sidecar/external bindings before write; all bindings are `native_portable`, raw Methods refs are plugin-free, and archive `host_artifacts` is empty. |
| Ambiguous conformal state | Package V2 is the sole owner; the old archive-level conformal slot is explicitly null. |
| In-place attestation mutation | V1-to-V2 migration is copy-on-write, hashes and retains its V1 source, and creates a new identity. |
| Unsigned authenticity claim | P0 signature is null; raw hashes provide integrity only, not publisher authenticity. |

The schema and validator freeze these responses without claiming that a product
reader, writer, migration tool or trust policy is implemented.
