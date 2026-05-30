# ADR-13: Process-adapter security boundary

**Status**: accepted (2026-05-29)
**Blocks**: workstream A (runtime), workstream G (cross-cutting infra)

## Context

Host controllers run arbitrary host code — Python and R subprocesses driven over the JSONL process-adapter protocol. A `ControllerManifest` that names an executable is, in effect, code the runtime will execute. Without a defined trust boundary, loading an untrusted manifest is a remote-code-execution vector, and a runaway adapter can hang or exhaust the host.

## Decision

The runtime treats adapter execution as a privileged operation gated by an explicit, operator-controlled boundary:

1. **Executable allowlist** — `DAGML_ADAPTER_ALLOWLIST` (colon-separated absolute paths). The runtime resolves an adapter's executable and **refuses to spawn** anything not on the allowlist. Empty allowlist ⇒ no adapters spawn (default-deny).
2. **Environment sanitization** — child processes receive a scrubbed environment. Only `PATH`, `PYTHONPATH`, and `R_LIBS` entries that are themselves allowlisted pass through; everything else is dropped. The parent's full env is never inherited.
3. **Per-task timeout + kill** — every `NodeTask` carries a wall-clock timeout. On expiry the runtime `SIGTERM`s then `SIGKILL`s the worker and emits a `controller_timeout` lineage event (ADR-12 span field `controller_id`).
4. **Tempdir isolation** — each adapter task gets a per-task tempdir (`DAGML_PROCESS_ARTIFACT_DIR`). Artifact URIs are validated against absolute/traversal paths (already enforced in `bundle.rs`); a traversal attempt is a `security` error (ADR-11).
5. **Explicit warning** — the CLI and docs state plainly that adapters execute untrusted host code, and that trusting a controller manifest equals trusting its code.
6. **Optional sandbox hook** — a pluggable wrapper (`cgroups v2` / `firejail` / `bubblewrap`) gated behind `DAGML_ADAPTER_SANDBOX`. Off by default; documented for production operators who need resource and syscall confinement.

## Consequences

- The process-adapter spawn path in the runtime gains an allowlist check, an env scrubber, and a timeout/kill supervisor.
- `SECURITY.md` documents this surface alongside ADR-16 (artifact deserialization is the other half of the host-trust boundary).
- The error taxonomy (ADR-11) gains `security/adapter_not_allowlisted`, `security/adapter_timeout`, and `security/artifact_path_traversal` codes.

## Risk

- A locked-down default (empty allowlist) means a fresh install runs no adapters until configured. This is intentional: secure-by-default beats convenient-by-default for a code-execution surface. The CLI error names the env var and points at `SECURITY.md`.
