# Supported Surface

This page is the 0.3.x RC support contract for `dag-ml` (current package
version: 0.3.23). It separates
production-facing surfaces from conformance fixtures and backlog work. It does
not change existing C ABI, JSON schema, or WASM signatures. The explicitly
scoped V2 terminal PREDICT Rust/Python surface is documented below.

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
| Runtime process adapter protocol | Supported | JSONL frames, describe handshake, timeouts, retries and worker pools are covered by CLI tests. It is not an HPO protocol: process controllers cannot attest native optimizer state, trial history, checkpoints or terminalization and therefore refuse tuner-session creation. |
| Native Methods HPO session (R1) | Experimental | Only the native Methods controller may create a `RuntimeHpoExecutionContext`-bound HPO session. Plugin/process controllers, including Optuna adapters, are not release-promised for HPO. |
| Native Methods HPO resume | Experimental | DAG-ML validates and merges the completed-trial prefix before continuation; Methods alone restores and advances the opaque `N4MOPT` bytes. Resumed evidence cannot duplicate a terminal transition. |
| Archive V2 Methods replay | Experimental | N4MM format 1 raw PLS and format 2 `SNV(ddof=0) -> Savitzky-Golay(mode=interp) -> PLS` are accepted through the native Methods controller. Format 2 requires its ABI 2.5 typed descriptor; mismatches fail closed without a Python fallback. |
| Archive-bound conformal presentation V2 | Experimental | The identity-bound projection supports multiple named targets and self-validates point/interval evidence. It presents already calibrated blocks; DAG-ML does not recalculate calibration or join rows positionally. |
| Callback-free Methods Package V2 PREDICT replay | Experimental | `dag_ml_core::execute_loaded_methods_predictor_replay` accepts only signed replay contracts, explicitly attested numeric input views and an exact `MethodsRuntime`; it registers and releases the native controller per invocation. This is the Rust bridge for Core/Studio, not a generic host-callback route. |
| Closed V2 terminal PREDICT replay | Experimental | `dag_ml::execute_terminal_prediction` and Python `run_cv_refit_predict_in_process` execute CV → REFIT → one direct sample-level PREDICT against an identity-attested V2 cohort, returning a sealed receipt. They reject V1/targetless cohorts, OOF-dependent graphs, and observation/aggregation output paths before controller callbacks run. |
| Strict callback-free Methods terminal PREDICT | Experimental | Python `execute_methods_cv_refit_terminal_predict` accepts only raw numeric arrays, explicit IDs, one numeric target, no-shuffle KFold PLS, and a separate X-only cohort. It performs native CV/REFIT/PREDICT without a host callback, rejects transforms, generators, HPO, calibration, groups, metadata and externally consumed/retained OOF, and returns frozen native result/receipt objects. Internal CV OOF remains required and ephemeral for scoring only. |
| Python and WASM JSON-contract bindings | Supported | Wheel/package metadata and smoke tests are CI-gated; object-native Python DSL frontend is not included. |
| Pipeline DSL JSON compiler | Supported | Canonical JSON plus nirs4all-compatible serialized JSON descriptors are covered. |
| Direct Python/YAML object DSL frontend | Backlog | Host object resolution remains binding-owned. |
| sklearn production process adapter | Conformance | The reference adapter is tested and useful for integration, but release notes must list supported estimator families and persistence limits. |
| prospectr and mdatools process adapters | Conformance | Tested reference adapters for selected R operators; stateful `msc`, `simca` and `mcrals` remain backlog. |
| SpectroChemPy and Orange-Spectroscopy adapters | Backlog | Tracked in `docs/HOST_ADAPTER_BACKLOG.md`. |
| EXPLAIN phase execution through host adapters | Experimental | Contracts and mock replay exist; production adapter dispatch is not a final-release promise. |
| Controller-side task batching | Backlog | Parallel scalar scheduling is supported; batch requests/static subgraphs remain future hardening. |

## dag-ml-data Dependency

`dag-ml` 0.3.23 consumes the sibling `dag-ml-data` contracts through
JSON-identical schemas and fixtures. The supported cross-repo contract for this
release is:

- `CoordinatorDataPlanEnvelope` v1;
- `CoordinatorDataPlanEnvelope` v2 only for closed terminal PREDICT replay
  with an identity-attested `predict_cohort`;
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

For the 0.3.x RC release window:

- no C ABI symbol, struct layout, JSON schema id/version, Rust public function,
  Python facade function or WASM export changes without an explicit contract
  entry and ABI/schema snapshot update;
- if such a change is accepted, downstream chains such as `nirs4all-core`,
  `nirs4all-web` and browser/Python smoke packages must be rebuilt before tag;
- documentation, CI jobs, tests and private benchmark helpers are allowed when
  they do not alter exported signatures.

## Post-0.3.x Backlog

1. Keep the `dag-ml-capi` AddressSanitizer lane green and extend it beyond
   library unit tests when C ABI lifecycle coverage expands.
2. Extend the initial performance probes to replay cache export and process
   worker pools.
3. Raise public Rust documentation coverage toward the ADR target in follow-up
   hardening.
