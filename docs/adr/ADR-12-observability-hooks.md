# ADR-12: Observability hooks

**Status**: accepted (2026-05-28)
**Blocks**: workstream A (runtime instrumentation), workstream G (cross-cutting infra).

## Context

Production users need to observe what the runtime did — which controller, which fold, which OOF refusal, which cache hit. Without structured spans, post-mortem debugging means tail-grepping log files. The bridge needs to surface these to nirs4all-studio's existing telemetry.

## Decision

**`tracing` spans + JSON logfmt output + OpenTelemetry-compatible IDs + a small metrics surface.**

### Span structure

The Rust runtime emits one `tracing` span per phase invocation with the following stable fields:

| Field | Type | Always present? |
|---|---|---|
| `run_id` | UUID v7 | yes |
| `plan_id` | str | yes |
| `variant_id` | str | yes (after PLAN) |
| `fold_id` | str | inside FIT_CV / SELECT / REFIT |
| `controller_id` | str | inside controller dispatch |
| `phase` | enum | yes |
| `node_id` | str | inside node execution |
| `partition_id` | str | inside separation-branch subgraphs (ADR-03) |
| `cache_hit` | bool | when checked |
| `oof_refused` | bool | on OOF leakage refusal |

The span field set is **frozen**: adding fields is non-breaking, renaming or removing is governed by ADR-14.

### Sink format

- Default: JSON-logfmt to stderr (one event per line, parsable by `jq`).
- Optional: OpenTelemetry OTLP exporter (gated behind `--features otel`, off by default to avoid dependency bloat).
- Span IDs are 128-bit, OTEL-compatible.

### Metrics surface

The runtime exports a small set of counters / histograms (Prometheus-compatible names):

- `dagml_phase_duration_seconds{phase}` (histogram)
- `dagml_oof_refusals_total{category}` (counter)
- `dagml_controller_duration_seconds{controller_id}` (histogram)
- `dagml_cache_hits_total / dagml_cache_misses_total`
- `dagml_lineage_records_emitted_total`

### Privacy

Span fields contain **only identifiers** — never raw feature matrices, never targets, never metadata column contents. The lineage envelope (ADR-04) already enforces the same boundary at the data-layer ABI; ADR-12 extends it to the telemetry layer. Adding a non-identifier field requires an ADR-superseder and a CI lint that flags new field names matching `.*(data|features|targets|samples).*` for review.

## Consequences

- Workstream A task 6 lands the `tracing` instrumentation; the C ABI exposes a `dagml_set_tracing_subscriber` hook for hosts.
- Workstream G task 2 wires the field allowlist + the OTEL feature.
- `nirs4all-studio`'s backend reads the JSON logfmt stream and feeds its existing telemetry widgets.

## Risk

- `tracing` adds binary size; the default subscriber is lightweight (`tracing_subscriber::fmt`). The OTEL exporter is optional.
