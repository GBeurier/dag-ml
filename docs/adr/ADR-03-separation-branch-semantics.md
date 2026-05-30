# ADR-03: Separation-branch semantics

**Status**: accepted (2026-05-28)
**Blocks**: workstream E (bridge). DSL translator depends on this answer.

## Context

nirs4all's `{"branch": {"by_metadata": "col", "steps": {...}}}` (and the `by_tag` / `by_filter` / `by_source` variants) partition samples dynamically at fit time. dag-ml's DSL is statically compiled — variant enumeration happens once, in `dsl::compile_pipeline_dsl_to_graph`. The mismatch was Codex's "single biggest weakness" of v1.

The v1 roadmap proposed an arbitrary `cardinality ≤ 10` cap. Codex rejected that as a "silent product limit" — caps are policy, not semantics.

## Decision

Separation-branch behavior is decomposed into three orthogonal pieces:

1. **Partition semantics** — branches are keyed by **deterministic partition IDs** derived from the partition function. For `by_metadata`, IDs are stable sorted unique values of the column. For `by_tag` and `by_filter`, IDs come from the controller's declared partition vocabulary (bool / categorical). For `by_source`, IDs are source names from the data plan.

2. **Cardinality policy** — explicit, configured, not hard-coded:
   - `cardinality_policy.max_partitions` (default: 16) — soft warning above, hard refusal at the explicit cap;
   - `cardinality_policy.on_exceeded`: `"refuse" | "truncate" | "error_with_remediation"` (default: `"refuse"`).
   The bridge refuses with a typed error pointing the user at the pre-partition escape hatch (`dataset.partition(by=...)`).

3. **Escape hatch** — `dataset.partition(by=...)` materializes a fixed partition before pipeline compilation, lifting separation outside the DAG. The bridge documents this as the canonical workaround for high-cardinality cases.

## Consequences

- The DSL translator (workstream E task 1) accepts separation branches, computes the partition ID set, checks `cardinality_policy`, then emits N parallel subgraphs in the compiled `GraphSpec`.
- `dag-ml-data`'s `DataPlan` learns a per-branch `partition_id` field so each subgraph's data view is identity-scoped (no implicit row-position joins).
- `examples/developer/05_advanced_features/D01_metadata_branching.py` is the documented escape-hatch example.
- Any pipeline above the cardinality cap is **explicitly refused** with the remediation hint, never silently truncated.

## Open follow-ups

- Confirm `branch.by_filter` works end-to-end on the legacy backend; the parity oracle surfaced a missing `nirs4all.pipeline.steps.deserializer` import (logged as `skip_kind="legacy_bug"` on `branch_separation_by_filter`).
- The bridge's structured-merge case (`{"merge": {"predictions": [...], "output_as": "features"}}`) is documented separately under ADR-03b (not authored here; defer until merge implementation lands).
