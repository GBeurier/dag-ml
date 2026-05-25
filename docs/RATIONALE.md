# Rationale

## Why Rust

The hard part is not model compute. Model compute stays in sklearn, PyTorch,
C++, R or native controllers. The hard part is control correctness:

- liveness of opaque handles across a DAG;
- concurrent scheduler bookkeeping;
- fold and sample identity invariants;
- deterministic control RNG;
- cache and lineage consistency.

Those are Rust-shaped problems.

## Why A C ABI

A C ABI gives every host the same boundary: Python, R, C++, JS/WASM and native
applications can provide controllers and data providers without coupling the core
to one runtime.

The ABI also prevents an accidental dependency on Python object semantics in the
core. Python can be a first binding, but not the architecture.

## Why Split `dag-ml-data`

The data layer has different responsibilities and release cadence. It owns source
schemas, axes, representations, adapters, alignment, collation and fingerprints.
`dag-ml` only needs stable descriptors, identity tables and opaque data handles.

Keeping the split explicit reduces leakage risk: the execution core cannot
silently inspect or transform `X`.

## Non-Goals

- no NIRS-specific primitives;
- no file readers;
- no feature engineering baked into the core;
- no model implementation;
- no distributed orchestrator in the first phase.
