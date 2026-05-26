# Test Plan

## Unit Tests

| Area | First tests |
|---|---|
| Graph | missing endpoints, port-kind mismatch, cycles, valid graph |
| OOF | rejects train predictions, aligns by sample id, duplicate detection |
| OOF campaign fixtures | UC6 joins, UC11 refuses, fold prediction samples match fold partitions, campaign fingerprint is stable |
| RNG | same path gives same seed, different labels split streams |
| Data binding | validates envelope fingerprints, feature-set ids, refuses mismatches, materializes in-memory handles and creates scoped data views |
| Selection | deterministic metric ranking, stable tie-breaking, grouped branch selection, sklearn demo merge selection |
| Bundle/replay | bundle matches execution plan fingerprints, refit artifacts match node plans, replay envelopes match data requirements, unsupported bundle schema version refused |
| Runtime | sequential DAG order, campaign variant x fold execution, data-provider-required paths, fold train/validation data view routing, `NodeTask.data_views`, validation prediction sample checks, refit artifact-handle capture, replay materializes predict views and refit artifact handles, external controller result conformance |
| CLI contracts | selection command, bundle build command, mock refit bundle capture, bundle validation with replay request and data envelope, mock replay execution, process adapter campaign/replay execution, persistent JSONL process campaign |
| sklearn demonstrator | group OOF, repeated observations, train-only augmentation, branch variant selection, heterogeneous merge selection, refit report |
| ABI | null pointer handling, invalid JSON, valid graph, grouped selection output, bundle/replay validation, data-provider vtable routing, mock replay execution summary |

## Conformance Tests

Add after the executor exists:

- UC6 stacking with intentionally shuffled prediction order;
- UC11 train-prediction leakage refusal;
- group-aware split where no group crosses train/validation;
- replay rejects schema fingerprint mismatch;
- replay rejects missing refit artifacts and unsupported phases;
- selected branch/merge candidates are reproducible from persisted metrics;
- mock campaign run materializes data handles and fold-aware data views before
  invoking controllers.
- process adapter replay refuses malformed `NodeResult` lineage or output
  ownership.

Current CLI smoke commands:

```bash
cargo run -p dag-ml-cli -- validate-execution-plan --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json
cargo run -p dag-ml-cli -- validate-data-binding --campaign examples/campaign_oof_generation.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --node model:base --input x
cargo run -p dag-ml-cli -- run-mock-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.process
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:cli.process
cargo run -p dag-ml-cli -- select-candidates --policy examples/fixtures/bundle/selection_policy_rmse.json --candidates examples/fixtures/bundle/candidate_scores_demo.json --groups examples/fixtures/bundle/candidate_groups_demo.json --output examples/generated/selection_decisions_demo.json
cargo run -p dag-ml-cli -- build-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --bundle-spec examples/fixtures/bundle/bundle_build_spec_minimal.json --output examples/generated/execution_bundle_minimal.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-refit-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --bundle-id bundle:cli.refit.capture --output examples/generated/execution_bundle_refit_capture.json --plan-id plan:cli.refit.capture
cargo run -p dag-ml-cli -- validate-bundle --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.bundle
python examples/sklearn_complex_oof_demo.py
cargo run -p dag-ml-cli -- validate-oof-campaign examples/generated/sklearn_complex_oof_campaign.json
```

## ABI Tests

Add a C smoke test that:

1. links `dag-ml-capi`;
2. calls `dagml_version`;
3. validates `examples/minimal_graph.json`;
4. validates that Rust-allocated error strings are released by
   `dagml_string_free`;
5. validates that Rust-allocated JSON byte outputs are released by
   `dagml_owned_bytes_free`.
6. executes the mock replay ABI helper and validates the returned summary.
