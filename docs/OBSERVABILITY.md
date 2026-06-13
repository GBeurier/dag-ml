# Observability (ADR-12)

`dag-ml` emits structured [`tracing`](https://docs.rs/tracing) spans and events
from the control core. The core only emits through the `tracing` facade; it never
installs a subscriber, so it stays a no-op (and WASM-safe) until a binary or host
chooses a sink.

## What is emitted

All telemetry is centralized in `crates/dag-ml-core/src/observability.rs`:

| Site | Kind | Fields |
|---|---|---|
| Each phase scope (`FIT_CV`, `SELECT`, `REFIT`, `PREDICT`, …) | `info` span `dag_ml.phase` | `phase`, `variant_id`, `fold_id` |
| Each node invocation (nested under its phase span) | `info` span `dag_ml.node` | `phase`, `node_id`, `controller_id` |
| Out-of-fold leakage refusal | `warn` event | `oof_refused=true`, `category="validation"`, `code="oof_leakage"`, `node_id`, `violator_count` |

The parallel scheduler clones the phase span into each worker thread so node
spans nest correctly across `thread::scope`.

The `category`/`code` fields mirror the ADR-11 error taxonomy, so a log consumer
can alert on refusals without parsing human messages.

## Privacy invariant

Telemetry carries **identifiers and counts only** — never feature matrices,
targets, sample values or metadata contents. This extends the data-ABI boundary
(the core never inspects raw data) to the telemetry layer.

Two guards enforce it:

- The frozen `OBSERVABILITY_FIELD_ALLOWLIST` constant lists every permitted field.
- The CI lint `scripts/lint_tracing_fields.py` fails the build if any `tracing`
  usage appears outside `observability.rs`, or if any field name in that module
  matches `data|features|targets|samples|metadata`.

Adding a field requires updating the allowlist and ADR-12, and passes through the
same review.

## Consuming the stream

### CLI

The CLI installs a JSON-logfmt subscriber to stderr, gated by `RUST_LOG`:

```bash
RUST_LOG=dag_ml=info cargo run -p dag-ml-cli -- run-mock-campaign …  2> events.jsonl
# only OOF refusals:
RUST_LOG=dag_ml=warn cargo run -p dag-ml-cli -- …  2> >(jq 'select(.fields.oof_refused == true)')
```

With `RUST_LOG` unset the CLI stays quiet (no subscriber is installed).

### Hosts (Rust)

A host that links the crates installs its own subscriber (e.g.
`tracing_subscriber::fmt().json().init()`, or an OpenTelemetry layer) before
driving a run. `nirs4all-studio`'s backend reads the JSON-logfmt stream into its
existing telemetry widgets.

### Hosts (C / non-Rust)

Call `dagml_init_tracing(json_output)` once near startup (the minimal C hook): it
installs a process-global subscriber to stderr, `RUST_LOG`-filtered (default
`info`), emitting JSON-logfmt when `json_output != 0`. It returns
`VALIDATION_ERROR` (no-op) if a subscriber is already installed.

## Not yet implemented

The following ADR-12 items are deferred to a follow-up tranche and are not part
of the current surface:

- the `cache_hit` span field (prediction-cache instrumentation);
- the Prometheus metrics surface (`dagml_phase_duration_seconds`, …);
- the optional OpenTelemetry OTLP exporter (`--features otel`).
