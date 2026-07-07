# Aggregation Interop

`dag-ml` and `dag-ml-data` intentionally own different parts of aggregation:

- `dag-ml-data` describes data-side reducer vocabulary and validates
  reducer-shaped payloads.
- `dag-ml` owns ML-phase safety, OOF/refit legality, prediction levels,
  custom-controller routing and leakage-sensitive aggregation execution.

This document is the 0.2.x RC mapping. It is documentation-only and does not
change any public schema or ABI signature.

## Mapping

| dag-ml coordinator policy | dag-ml-data reducer | Release behavior |
|---|---|---|
| `mean` | `mean` | Directly compatible. |
| `weighted_mean` | `weighted_mean` | Compatible when a weight column or fit-influence contract is present. |
| `median` | `median` | Directly compatible. |
| `vote` | `vote` | Classification-only; both layers reject regression vote tasks. |
| `none` | none | Coordinator-only mode; no data-side reducer is expected. |
| `custom_controller` | `custom` | Coordinator routes an aggregation task to a controller and validates result-vs-task identity. |
| not exposed | `robust_mean` | Not a final-release coordinator mode; use `custom_controller` until a shared reducer schema is added. |
| not exposed | `exclude_outliers` | Not a final-release coordinator mode; use `custom_controller` until a shared reducer schema is added. |

## Release Rule

No implicit mapping may be added during the 0.2.x RC release window. If `robust_mean` or
`exclude_outliers` becomes a first-class `dag-ml` coordinator method, the change
must include:

1. a shared schema or explicit migration note;
2. JSON Schema fixture updates in both repositories;
3. C ABI/Python/WASM contract manifest updates if any public symbol or exported
   helper changes;
4. downstream rebuild of chains consuming public signatures.

## Signal-Type Replay

`dag-ml-data` exposes signal-type validation helpers. `dag-ml` does not yet
carry an expected signal type through bundle replay. Until that paired contract
exists, 0.2.x documents signal-type replay enforcement as a
provider/backlog item rather than silently accepting it as a supported
coordinator invariant.
