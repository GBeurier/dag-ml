# Test Plan

## Unit Tests

| Area | First tests |
|---|---|
| Graph | missing endpoints, port-kind mismatch, cycles, valid graph |
| OOF | rejects train predictions, aligns by sample id, duplicate detection |
| OOF campaign fixtures | UC6 joins, UC11 refuses, fold prediction samples match fold partitions, campaign fingerprint is stable |
| RNG | same path gives same seed, different labels split streams |
| ABI | null pointer handling, invalid JSON, valid graph |

## Conformance Tests

Add after the executor exists:

- UC6 stacking with intentionally shuffled prediction order;
- UC11 train-prediction leakage refusal;
- group-aware split where no group crosses train/validation;
- replay rejects schema fingerprint mismatch.

## ABI Tests

Add a C smoke test that:

1. links `dag-ml-capi`;
2. calls `dagml_version`;
3. validates `examples/minimal_graph.json`;
4. validates that Rust-allocated error strings are released by
   `dagml_string_free`.
