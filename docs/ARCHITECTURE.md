# Architecture

`dag-ml` is a control engine. It coordinates ML execution without owning the
heavy data buffers or fitted model objects.

For the short normative product contract, read `docs/COORDINATOR_SPEC.md` first.
That document resolves the controller/splitter boundary and is the alignment
source for implementation work.

## Layers

| Layer | Owned here | Not owned here |
|---|---|---|
| Graph contract | `GraphSpec`, nodes, ports, edges, variants | DSL syntax sugar |
| Planning | phase order, edge validation, data-plan requests | source storage and representation search |
| Execution control | fold identity, OOF joins, scheduler decisions | model fitting internals |
| Stores | lineage/cache/artifact references | artifact byte formats owned by hosts |
| ABI | controller/data vtables, handles, release contracts | host object allocation |

## Crates

| Crate | Responsibility |
|---|---|
| `dag-ml-core` | Pure Rust contracts and invariant checks. No host runtime dependency. |
| `dag-ml` | Stable Rust facade for downstream crates and bindings. |
| `dag-ml-capi` | C ABI entry points, vtable definitions and header. |
| `dag-ml-cli` | Local validation utilities for specs and fixtures. |

## Runtime Flow

```text
COMPILE -> PLAN -> FIT_CV -> SELECT -> REFIT -> PREDICT -> EXPLAIN
```

The first implementation should make `PLAN` and the leakage-sensitive subset of
`FIT_CV` concrete:

1. validate a `GraphSpec`;
2. ask `dag-ml-data` for compatible `DataPlan` blobs where data is consumed;
3. execute splitters over identity tables;
4. call controller vtables with opaque data/model handles;
5. store validation predictions as identity-aligned prediction blocks;
6. join OOF predictions by sample id for downstream meta-models.

## Boundary With `dag-ml-data`

`dag-ml-data` describes data possibilities and produces data plans. `dag-ml`
decides when those plans execute and whether using their outputs would violate
ML invariants.

The control core may inspect:

- sample ids, group ids, target ids and origin ids;
- prediction tables;
- `y_true` tables for scoring;
- canonical JSON descriptors and fingerprints.

The control core must not inspect:

- feature matrices;
- images, spectra, time series or graph buffers;
- fitted operator internals.
