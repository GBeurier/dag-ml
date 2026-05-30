# ADR-08: Session / workspace persistence

**Status**: accepted (2026-05-28)
**Blocks**: workstream E (bridge).

## Context

`nirs4all.session()` is public API (frozen since 0.9.0 per `nirs4all/CLAUDE.md`). It returns a context manager that owns a workspace, lets the user run multiple pipelines against it, and shares cached artifacts / folds / dataset materializations across runs. dag-ml itself is single-run; the bridge must reproduce session semantics or break the API.

Roadmap v1 considered deprecation. Codex flagged: it's used by `nirs4all-studio` (the webapp) — breaking would cascade.

## Decision

Keep the session API. The bridge implements caching in the adapter layer with the following frozen semantics:

1. **Cache key** — `sha256(envelope_fingerprint + pipeline_fingerprint + controller_manifest_fingerprint)`. Cross-run hits are by full key only; partial matches are not honored.

2. **Concurrency model** — single writer per workspace, multiple readers. A workspace lock file (`workspace/.session.lock`, OS-advisory) prevents concurrent writers; readers proceed in parallel. Attempting to open a locked workspace as writer raises `WorkspaceBusy`.

3. **Persistence format** — extends the existing SQLite + Parquet workspace (`workspace/store.sqlite`, `workspace/arrays/`, `workspace/artifacts/`). A new table `session_cache` holds `(cache_key, run_id, created_at)` rows; entries point to existing artifact + array files. Per-run isolation is preserved.

4. **Bundle relation** — exported `.n4a` bundles do **not** depend on session cache state; they self-contain everything needed for replay. Removing the workspace must never invalidate previously exported bundles.

5. **Shutdown semantics** — `with` context-manager exit flushes the cache table, releases the lock, and runs `VACUUM`. SIGKILL / hard crash leaves a stale lock; `nirs4all workspace doctor` (new CLI command) detects and clears it after checking the holding PID is gone.

6. **Cache invalidation** — modifying the dataset (e.g. `set_repetition`, `convert_to_absorbance`) recomputes the envelope fingerprint and therefore the cache key — no stale cross-call hits.

## Consequences

- `nirs4all/api/session.py` keeps its current signature; internals switch from the legacy runner to the bridge.
- The webapp's reliance on the session API survives without changes.
- The dag-ml runtime stays single-run by design; the cache is a binding-layer feature, not a core feature.
- `nirs4all workspace doctor` is a new CLI subcommand; documented in CONTRIBUTING.md.

## Risk

- Cache hits across different dag-ml versions are refused (cache key includes the controller-manifest fingerprint which carries the dag-ml version). Users see slower re-runs after upgrades; that's the intended trade-off.
