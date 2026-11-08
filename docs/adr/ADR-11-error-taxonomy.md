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
- `target_missing_exclusion` → "Keep target-missing samples out of the CV universe (FoldSet.target_missing_sample_ids) and keep enough target-observed samples for every fold."

### Captured host HPO refusals

Both codes are in the existing `runtime` category. The packed values follow the `(category << 16) | code` rule above.

| Code | Packed | Meaning |
| --- | --- | --- |
| `host_hpo_capture_abort` | `(1, 3)` | A captured search propagated an error. The report keeps primary, secondary, cleanup and evidence errors separate and carries no score, completion or commitment. |
| `host_hpo_predecessor_refused` | `(1, 4)` | A restored predecessor failed qualified resume before any ask, recovery or seal. The detail starts with a stable refusal code such as `pair_mismatch`. |

`host_hpo_capture_abort` reports one disposition:

- `aborted`: nothing new was sealed for the failure, or cleanup moved the owner beyond the last publication. The owner may hold rows that no publication covers, and its pair no longer holds.
- `published_failed_prefix`: the failing trial reached an ordinary sealed Failed publication, no cleanup ran, and every observation lies inside it, so a paired resume may be possible.
- `evidence_incomplete`: the search itself returned, but its evidence stream could not be completed.

The report also lists `uncaptured_pending`: restored trials the owner still holds RUNNING. They lie outside the failed window, so they are not touched, and they keep the stream from closing.

A returned outcome is not an error. When the owner still holds RUNNING trials that no stream closed, it returns no completion, names them in `uncaptured_pending`, and reports the stream's refusal code and flush result in `refusal`.

`host_hpo_predecessor_refused` has two known limits. A prior attempt of unknown ownership refuses even when the pair verifies, so a host that reports a raised ask is refused while one that omits it passes on the digest alone. A publication does not name its publisher, so reuse of a capture id whose publication holds only unterminated replays cannot be detected.

## Consequences

- Workstream A task 6 introduces the `DagMlError` enum + builders.
- Workstream G task 1 lands the C ABI side and the Python exception hierarchy.
- CONTRIBUTING.md (workstream C task 1) documents how to add a new error variant, including the remediation-hint requirement.
- The bridge logs every refusal with `code`, `category`, `context` — feeds the observability hooks (ADR-12).

## Risk

- An exhaustive enum is a stable surface; adding variants is non-breaking but renaming/removing one is. CHANGELOG entries flag any new variants under "added" and removals under "breaking change" with the deprecation cycle from ADR-14.
