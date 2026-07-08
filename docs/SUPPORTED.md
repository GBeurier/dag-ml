# Supported Surface

This page is the 0.2.x RC support contract for `dag-ml` (current package
version: 0.2.7). It separates
production-facing surfaces from conformance fixtures and backlog work. It does
not change any public ABI, JSON schema, Rust, Python or WASM signature.

## Support Levels

| Level | Meaning |
|---|---|
| Supported | Included in the release scope and covered by CI gates. |
| Conformance | Stable enough for binding authors and integration tests, but not a complete production adapter. |
| Experimental | Public shape may exist, but release notes must call out limitations. |
| Backlog | Not part of the release promise. |

## dag-ml Surface

| Area | Level | Notes |
|---|---|---|
| Graph, campaign and execution-plan contracts | Supported | Rust validation, JSON Schemas, C ABI discovery and CLI validation are gated. |
| Fold identity, OOF joins and leakage refusal | Supported | Sample-id joins, validation-only OOF, group/origin/repetition guards and D9 multisource negative cases are tested. |
| Deterministic selection and replay bundle validation | Supported | Plan/controller/data/artifact fingerprints and selection metric levels are validated. |
| Research provenance export | Supported | RO-Crate, PROV and OpenLineage exports are generated from validated internal contracts. |
| C ABI JSON contract helpers | Supported | Header snapshot, C conformance and non-mock replay paths are gated. |
| Runtime process adapter protocol | Supported | JSONL frames, describe handshake, timeouts, retries and worker pools are covered by CLI tests. |
| Python and WASM JSON-contract bindings | Supported | Wheel/package metadata and smoke tests are CI-gated; object-native Python DSL frontend is not included. |
| Pipeline DSL JSON compiler | Supported | Canonical JSON plus nirs4all-compatible serialized JSON descriptors are covered. |
| Direct Python/YAML object DSL frontend | Backlog | Host object resolution remains binding-owned. |
| sklearn production process adapter | Conformance | The reference adapter is tested and useful for integration, but release notes must list supported estimator families and persistence limits. |
| prospectr and mdatools process adapters | Conformance | Tested reference adapters for selected R operators; stateful `msc`, `simca` and `mcrals` remain backlog. |
| SpectroChemPy and Orange-Spectroscopy adapters | Backlog | Tracked in `docs/HOST_ADAPTER_BACKLOG.md`. |
| EXPLAIN phase execution through host adapters | Experimental | Contracts and mock replay exist; production adapter dispatch is not a final-release promise. |
| Controller-side task batching | Backlog | Parallel scalar scheduling is supported; batch requests/static subgraphs remain future hardening. |

## dag-ml-data Dependency

`dag-ml` 0.2.7 consumes the sibling `dag-ml-data` contracts through
JSON-identical schemas and fixtures. The supported cross-repo contract for this
release is:

- `CoordinatorDataPlanEnvelope` v1;
- `FeatureFusionSelector` v1;
- `CoordinatorBranchView` v1;
- `FittedAdapterRef` v1 as a data-side replay/persistence contract;
- shared `FoldSet`, conformance pack and parity-oracle manifests.

The following `dag-ml-data` capabilities are required for release validation but
remain provider-specific at runtime:

- host-side execution of `branch_view` modes `by_metadata`, `by_tag` and
  `by_filter`;
- materialize/predict signal-type enforcement once `dag-ml` carries expected
  signal type through replay;
- production provider arenas beyond the in-memory conformance provider.

## Public-Signature Policy

For the 0.2.x RC release window:

- no C ABI symbol, struct layout, JSON schema id/version, Rust public function,
  Python facade function or WASM export changes without an explicit contract
  entry and ABI/schema snapshot update;
- if such a change is accepted, downstream chains such as `nirs4all-core`,
  `nirs4all-web` and browser/Python smoke packages must be rebuilt before tag;
- documentation, CI jobs, tests and private benchmark helpers are allowed when
  they do not alter exported signatures.

## Post-0.2.x Backlog

1. Keep the `dag-ml-capi` AddressSanitizer lane green and extend it beyond
   library unit tests when C ABI lifecycle coverage expands.
2. Extend the initial performance probes to replay cache export and process
   worker pools.
3. Raise public Rust documentation coverage toward the ADR target in follow-up
   hardening.
