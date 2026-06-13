# ADR-04: Tag / exclude mask materialization

**Status**: accepted (2026-05-28)
**Blocks**: workstream E (bridge data-provider), workstream G (lineage taxonomy).

## Context

nirs4all's `{"tag": filter}` adds a column without removing samples. `{"exclude": filter}` removes samples from training. `{"exclude": [filter_a, filter_b], "mode": "any" | "all"}` produces a union or intersection mask. The bridge must reproduce **and audit** the resulting masks so leakage refusal still applies.

## Decision

The bridge **pre-computes masks** in the adapter layer (i.e. inside `nirs4all/bridge/dag_ml_provider.py`), then ships the materialized mask + an audit record to `dag-ml-data`. The materialization itself is offline; the audit is part of the lineage envelope.

### Mask materialization contract

1. The adapter walks every `tag` / `exclude` step at compile time, calls the corresponding filter operator's `fit_predict` on the training partition, and records the resulting boolean mask in the `SampleRelationTable` as a sample-level metadata column (`__tag_<tag_name>__` / `__exclude_<mode>_<hash>__`).
2. Multi-filter `exclude` masks are materialized via Python set ops (`mask_any = mask_a | mask_b`, `mask_all = mask_a & mask_b`). The combined mask carries the canonical key `__exclude_{mode}_{sha256(sorted(filter_ids))[:12]}__`.
3. `tag` masks are **additive** — they create new metadata columns but never alter the train/test partition. `exclude` masks set `partition = "excluded_from_training"` for the matching rows; the rows remain in the dataset for prediction.

### Lineage audit fields (ADR-11 / ADR-12)

Every materialized mask is recorded in the lineage envelope with:

- **expression** — the original Python expression (operator class name + constructor args).
- **mode** — `"single"` / `"any"` / `"all"` for `exclude`.
- **source_columns** — which columns the filter consumed (X, y, metadata key).
- **sample_id_set** — the list of `SampleId` values the mask flagged (full list, not just count, so audits are reproducible).
- **mask_fingerprint** — SHA-256 of the canonical mask blob; lets the bridge's bundle replay assert masks are identical.

## Consequences

- The bridge's data provider materializes masks before the DAG runs; the runtime never re-computes them.
- `dag-ml-data`'s `SampleRelationTable` learns an `MaskAuditRecord` reference per materialized mask.
- The lineage taxonomy (ADR-11) carries `mask_*` fields that survive RO-Crate / PROV / OpenLineage export.
- "Tagged but not excluded" remains expressible — downstream `branch.by_tag` consumes the tag column without partitioning fixed rows.

## Open follow-ups

- The parity oracle's `exclude_multi_any_y_and_x` case skips today because `sample_data/regression` is too small for a 2-filter UNION exclusion to leave a viable train set. Pick `regression_2` once available — fixture-size, not contract.
