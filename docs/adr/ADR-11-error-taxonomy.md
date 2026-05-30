# ADR-11: Unified error taxonomy

**Status**: accepted (2026-05-28)
**Blocks**: workstream A (core errors), workstream D (Python bindings), workstream E (bridge), workstream G (cross-cutting infra).

## Context

Rust returns `Result<T, DagMlError>`; the C ABI uses integer return codes; Python bindings raise exceptions; nirs4all has its own exception hierarchy. Without a unified taxonomy, a single failure becomes a different opaque error in every binding layer. Users can't write robust catch-blocks; debugging crosses three layers.

## Decision

One **stable error code surface** that maps deterministically across all layers:

### Error code structure

`error_code = (category, code, severity)`:

- `category`: one of `validation`, `runtime`, `data`, `controller`, `bundle`, `lineage`, `replay`, `security`, `compatibility`, `internal`.
- `code`: short stable identifier (e.g. `oof_leakage`, `envelope_version_unsupported`, `schema_fingerprint_mismatch`, `controller_dispatch_failed`, `repetition_leakage`).
- `severity`: `fatal | error | warning`. `warning` doesn't fail by default but flips to error under `--strict`.

### Per-layer mapping

- **Rust** — `DagMlError` enum with one variant per `(category, code)`. Carries structured context: stable code, category, severity, optional `cause: Box<dyn Error>`, optional `remediation_hint: &'static str`, and a `context: BTreeMap<&'static str, serde_json::Value>` for query-time debug fields.
- **C ABI** — returns `int32_t`: `0` on success, otherwise `(category << 16) | code`. A thread-local last-error buffer holds the JSON-serialized `DagMlError` so callers can fetch the structured payload via `dagml_last_error_json`.
- **Python** — `DagMlError` base class with one subclass per category (`DagMlValidationError`, `DagMlRuntimeError`, etc.). Each subclass exposes `.code`, `.category`, `.severity`, `.remediation_hint`, `.context`. Catch hierarchy mirrors the Rust enum.
- **nirs4all** — bridge maps `DagMlError.category` to nirs4all's existing exception classes (`PipelineError`, `DataError`, `ValidationError` from `nirs4all/core/exceptions.py`). The mapping is exhaustive and tested.

### Remediation hints

Every error variant carries a `remediation_hint` — a single sentence telling the user the next thing to do. Hints are part of the public ABI; CI fails if a new variant ships without one. Examples:

- `oof_leakage` → "Refit the offending edge with `requires_oof = true` or remove the leakage path; see ADR-05."
- `envelope_version_unsupported` → "Re-export your bundle with `nirs4all workspace migrate`; see ADR-02."
- `repetition_leakage` → "Pass `respect_repetition=True` to your splitter; see ADR-05."

## Consequences

- Workstream A task 6 introduces the `DagMlError` enum + builders.
- Workstream G task 1 lands the C ABI side and the Python exception hierarchy.
- CONTRIBUTING.md (workstream C task 1) documents how to add a new error variant, including the remediation-hint requirement.
- The bridge logs every refusal with `code`, `category`, `context` — feeds the observability hooks (ADR-12).

## Risk

- An exhaustive enum is a stable surface; adding variants is non-breaking but renaming/removing one is. CHANGELOG entries flag any new variants under "added" and removals under "breaking change" with the deprecation cycle from ADR-14.
