# Test Plan

## Unit Tests

| Area | First tests |
|---|---|
| Graph | missing endpoints, port-kind mismatch, cycles, valid graph |
| OOF | rejects train predictions, aligns by sample id, duplicate detection |
| OOF campaign fixtures | UC6 joins, UC11 refuses, fold prediction samples match fold partitions, campaign fingerprint is stable |
| RNG | same path gives same seed, different labels split streams |
| Data binding | validates envelope fingerprints, refuses mismatches, materializes in-memory handles |
| Runtime | sequential DAG order, campaign variant x fold execution, data-provider-required paths |
| sklearn demonstrator | group OOF, repeated observations, train-only augmentation, branch variant selection, heterogeneous merge selection, refit report |
| ABI | null pointer handling, invalid JSON, valid graph |

## Conformance Tests

Add after the executor exists:

- UC6 stacking with intentionally shuffled prediction order;
- UC11 train-prediction leakage refusal;
- group-aware split where no group crosses train/validation;
- replay rejects schema fingerprint mismatch.
- mock campaign run materializes data handles before invoking controllers.

Current CLI smoke commands:

```bash
cargo run -p dag-ml-cli -- validate-execution-plan --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json
cargo run -p dag-ml-cli -- validate-data-binding --campaign examples/campaign_oof_generation.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --node model:base --input x
cargo run -p dag-ml-cli -- run-mock-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json
python examples/sklearn_complex_oof_demo.py
cargo run -p dag-ml-cli -- validate-oof-campaign examples/generated/sklearn_complex_oof_campaign.json
```

## ABI Tests

Add a C smoke test that:

1. links `dag-ml-capi`;
2. calls `dagml_version`;
3. validates `examples/minimal_graph.json`;
4. validates that Rust-allocated error strings are released by
   `dagml_string_free`.
