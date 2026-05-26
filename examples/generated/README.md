# Generated Demonstrators

This directory contains deterministic outputs from example generators.

## sklearn complex OOF

Regenerate with:

```bash
python examples/sklearn_complex_oof_demo.py
cargo run -p dag-ml-cli -- validate-oof-campaign examples/generated/sklearn_complex_oof_campaign.json
cargo run -p dag-ml-cli -- validate-sklearn-complex-demo --campaign examples/generated/sklearn_complex_oof_campaign.json --report examples/generated/sklearn_complex_report.json
```

The fixture is independent from `nirs4all`. It demonstrates repeated
observations, group-safe OOF, train-only augmentation, branch model variants,
heterogeneous merge variants using predictions plus original data, OOF-based
selection and final refit reporting. The CLI validator recomputes branch and
merge selections from the report metrics and checks final-refit feature/sample
contracts from Rust.

## Bundle and replay CLI

Regenerate with:

```bash
cargo run -p dag-ml-cli -- select-candidates --policy examples/fixtures/bundle/selection_policy_rmse.json --candidates examples/fixtures/bundle/candidate_scores_demo.json --groups examples/fixtures/bundle/candidate_groups_demo.json --output examples/generated/selection_decisions_demo.json
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.process
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:cli.process
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:cli.branch.merge
cargo run -p dag-ml-cli -- run-process-cv-refit-bundle --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --bundle-id bundle:generated.branch.merge.cv.refit --selections examples/fixtures/bundle/selection_decisions_branch_merge.json --output examples/generated/execution_bundle_branch_merge_cv_refit.json --prediction-cache-output examples/generated/prediction_cache_branch_merge_cv_refit.json --plan-id plan:generated.branch.merge.cv.refit
cargo run -p dag-ml-cli -- validate-prediction-cache --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --payload examples/generated/prediction_cache_branch_merge_cv_refit.json
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_branch_merge_predict.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:generated.branch.merge.cv.refit
cargo run -p dag-ml-cli -- run-process-cv-refit-replay --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/sklearn_process_controller.py --bundle-id bundle:cli.branch.merge.sklearn.cv.refit.replay --selections examples/fixtures/bundle/selection_decisions_branch_merge.json --plan-id plan:cli.branch.merge.sklearn.cv.refit.replay
cargo run -p dag-ml-cli -- build-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --bundle-spec examples/fixtures/bundle/bundle_build_spec_minimal.json --output examples/generated/execution_bundle_minimal.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-refit-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --bundle-id bundle:cli.refit.capture --output examples/generated/execution_bundle_refit_capture.json --plan-id plan:cli.refit.capture
cargo run -p dag-ml-cli -- run-process-refit-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --bundle-id bundle:cli.process.refit.capture --output examples/generated/execution_bundle_process_refit_capture.json --plan-id plan:cli.process.refit.capture
cargo run -p dag-ml-cli -- run-process-refit-replay --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/sklearn_process_controller.py --bundle-id bundle:cli.sklearn.refit.replay --plan-id plan:cli.sklearn.refit.replay
cargo run -p dag-ml-cli -- validate-bundle --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.bundle
```

These fixtures demonstrate deterministic candidate selection, bundle export
from a rebuilt execution plan, replay validation against external data
fingerprints, mock replay execution with opaque data/artifact handles, refit
bundle creation from captured model artifacts, and process-based controller
campaign/refit/replay execution through a JSON `NodeTask`/`NodeResult` adapter
boundary. The branch/merge process smoke proves scheduler-level `requires_oof`
prediction inputs into a heterogeneous meta-model that also receives original
data. The branch/merge CV+refit bundle smoke keeps the CV prediction store
alive through refit, proving that the meta-model refit sees complete validation
OOF coverage before base and meta artifacts are captured, and the generated
bundle includes typed prediction requirements, deterministic prediction cache
fingerprints, a compact OOF summary by producer/fold/sample, a materialized
prediction-cache payload set containing the actual validation prediction
blocks, and persisted branch/merge selection decisions. The paired replay smoke
reuses that bundle for the two branch models and the meta-model while all three
data requirements point at the same validated data-plan envelope. The sklearn
refit/replay smoke fits a real sklearn pipeline and reuses the captured model
handle in a persistent JSONL process. The branch/merge sklearn CV+refit+replay
smoke keeps three captured model handles alive through replay inside the same
persistent adapter process.
