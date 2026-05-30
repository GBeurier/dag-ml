# ADR-17: Feature flag / cutover / rollback

**Status**: accepted (2026-05-29)
**Blocks**: workstream E (bridge), workstream F (parity)

## Context

Switching nirs4all's backend from the legacy `PipelineRunner` to dag-ml is a high-blast-radius change. It must be reversible without reinstalls or data migration, and users must be able to validate parity on their own data before trusting the switch.

## Decision

1. **Backend selector** — `nirs4all.run(..., backend="legacy" | "dag-ml" | "dual")`. Default is `"legacy"` until the G6 cutover gate, then the default flips to `"dag-ml"` in a minor release. The selector is also a workspace-level and env-level setting (`NIRS4ALL_BACKEND`).
2. **Dual-run mode** — `backend="dual"` runs **both** backends on the same input and diffs results within the ADR-01 tolerance ledger (per model class × metric). Mismatches beyond tolerance are **reported**, never silently reconciled. Dual mode ships for one release around cutover so users can validate on their own corpora.
3. **Legacy retention** — the legacy backend stays available behind `backend="legacy"` for **two releases** past the default flip, then is removed under the ADR-14 managed-debt policy (removal version recorded in the CHANGELOG).
4. **Rollback** — a user hitting a regression sets `backend="legacy"`: no reinstall, no data migration, no bundle rebuild. Bundles produced by either backend stay loadable for predict for the retention window (ties to ADR-02's bundle-readability SLA).
5. **Release notes** — the default-flip release links the compatibility ledger (ADR-01) and the migration guide.

## Consequences

- `nirs4all/api/run.py` gains the `backend` parameter and dispatches to the legacy runner or the bridge.
- Workstream F implements the dual-run diff against the tolerance ledger; the parity oracle (`tests/integration/parity/`) is the offline analogue.
- The migration guide (G6 deliverable) documents the flag, dual-run validation, and the rollback procedure.

## Risk

- Dual-run doubles compute for the validation release. It is opt-in and intended for a one-release validation window, not steady state. The docs say so explicitly.
