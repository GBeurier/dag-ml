# ADR-14: Public-API deprecation policy (managed debt)

**Status**: accepted (2026-05-29)
**Blocks**: workstream C (maturity), workstream E (bridge)

## Context

The ecosystem house rule is "no dead code, no deprecated code, no backward-compatibility shims." That rule is correct in steady state but **wrong during a multi-release migration**: replacing the nirs4all backend with dag-ml legitimately requires temporary compatibility code (a `backend="legacy"` path, dual-read bundle loaders, deprecated-but-still-working symbols). Codex flagged the blanket rule as counterproductive for the migration window.

## Decision

For the migration window (until the G6 cutover completes), the absolute no-deprecated-code rule is replaced by a **managed-debt policy**. After G6, the absolute rule resumes.

Rules during the window:

1. Every Rust `#[deprecated(since = "X.Y.Z", note = "...")]` **must** name a target removal version and link a tracking issue in the note.
2. CI fails if any `#[deprecated]` lacks a removal version, **or** if a release whose version ≥ the declared removal version still contains the symbol.
3. Every deprecated public symbol **must** carry a removal-test (asserts the deprecation fires and the replacement works) until it is removed.
4. Production-path `TODO`s must be justified: `TODO(owner): reason (#issue)`. Unexplained `TODO`/`FIXME` on a production path fails the CI lint.
5. nirs4all's public API (frozen since 0.9.0) follows the same rule **plus** a minimum two-release working window before any removal, because external consumers (nirs4all-studio) depend on it.

## Consequences

- A CI lint (`scripts/check_deprecations.py`, workstream C) parses `#[deprecated]` attributes and the `TODO` corpus and enforces rules 1–4.
- The CHANGELOG records every new deprecation under "Deprecated" with its removal version, and every removal under "Removed."
- `CONTRIBUTING.md` documents the policy so a contributor knows a deprecation is a commitment, not a free escape hatch.

## Risk

- A managed-debt window can drift into permanent debt if removal versions slip. The CI gate (rule 2) is the backstop: a shipped removal version with the symbol still present is a hard failure, forcing the removal or an explicit ADR-superseder that re-schedules it.
