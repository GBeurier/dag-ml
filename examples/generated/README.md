# Generated Demonstrators

This directory contains deterministic outputs from example generators.

## sklearn complex OOF

Regenerate with:

```bash
python examples/sklearn_complex_oof_demo.py
cargo run -p dag-ml-cli -- validate-oof-campaign examples/generated/sklearn_complex_oof_campaign.json
```

The fixture is independent from `nirs4all`. It demonstrates repeated
observations, group-safe OOF, train-only augmentation, branch model variants,
heterogeneous merge variants using predictions plus original data, OOF-based
selection and final refit reporting.

## Bundle and replay CLI

Regenerate with:

```bash
cargo run -p dag-ml-cli -- select-candidates --policy examples/fixtures/bundle/selection_policy_rmse.json --candidates examples/fixtures/bundle/candidate_scores_demo.json --groups examples/fixtures/bundle/candidate_groups_demo.json --output examples/generated/selection_decisions_demo.json
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.process
cargo run -p dag-ml-cli -- build-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --bundle-spec examples/fixtures/bundle/bundle_build_spec_minimal.json --output examples/generated/execution_bundle_minimal.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- validate-bundle --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.bundle
```

These fixtures demonstrate deterministic candidate selection, bundle export
from a rebuilt execution plan, replay validation against external data
fingerprints, mock replay execution with opaque data/artifact handles, and
process-based controller campaign/replay execution through a JSON
`NodeTask`/`NodeResult` adapter boundary.
