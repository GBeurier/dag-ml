# Test Plan

## Unit Tests

| Area | First tests |
|---|---|
| Graph | missing endpoints, port-kind mismatch, cycles, valid graph, deterministic parallel levels |
| OOF | rejects train predictions, aligns by sample id, duplicate detection |
| OOF campaign fixtures | UC6 joins, UC11 refuses, fold prediction samples match fold partitions, campaign fingerprint is stable |
| RNG | same path gives same seed, different labels split streams |
| Data binding | validates envelope schema version, published envelope JSON Schema version, envelope fingerprints, feature-set ids, refuses mismatches, checks coordinator relations against campaign folds/leakage policy, validates the sibling `dag-ml-data` coordinator fixture when available, idempotently registers the same envelope for multiple replay requirements, materializes in-memory handles and creates scoped data views |
| Selection | deterministic metric ranking, stable tie-breaking, grouped branch selection, sklearn demo merge selection |
| Bundle/replay | bundle matches execution plan fingerprints, selected candidates match the plan and selected refittable nodes have artifacts, refit artifacts match node plans, typed prediction requirements and cache manifests match OOF edges, materialized prediction-cache payloads match manifests and refuse tampering, file-backed prediction-cache manifests match bundle cache records, REFIT replay refuses manifest-only OOF caches and accepts validated payload-backed or file-store-backed OOF caches, replay envelopes match data requirements, unsupported bundle schema version refused |
| Runtime | sequential DAG order, precomputed phase node-level scheduling, campaign variant x fold execution, `NodeTask.variant` generation context, variant node-parameter override lowering, external adapter validation of effective generated params, data-provider-required paths, fold train/validation data view routing, `NodeTask.data_views`, `NodeTask.prediction_inputs`, validation prediction sample checks, `requires_oof` edge enforcement with missing/train/misaligned refusal and refit OOF coverage checks, in-memory and file prediction-cache store loading/materialization, file cache tamper refusal, refit artifact-handle capture, replay materializes predict views, prediction-cache handles and refit artifact handles, external controller result conformance |
| CLI contracts | selection command, execution schedule export, bundle build command, mock/process refit bundle capture, generated parameter override process campaign, branch/merge direct refit refusal without CV OOF, branch/merge CV+refit process bundle capture, prediction-cache payload export/validation, file prediction-cache store export/validation and store-backed replay, branch/merge REFIT replay refusal with manifest-only OOF caches, branch/merge REFIT replay acceptance with validated OOF payloads, branch/merge stateful sklearn CV+refit+replay, bundle validation with replay request and data envelope, mock replay execution, process adapter campaign/replay execution, persistent JSONL process campaign, branch/merge OOF process campaign, stateful sklearn refit/replay, sklearn complex demo report validation |
| sklearn demonstrator | group OOF, repeated observations, train-only augmentation, branch variant selection, heterogeneous merge selection, refit report |
| ABI | null pointer handling, invalid JSON, valid graph, graph parallel levels, execution-plan build, execution schedule export, grouped selection output, bundle/replay validation, prediction-cache payload validation, controller vtable routing, data-provider vtable routing, artifact-store vtable routing, prediction-cache vtable routing, mock replay execution summary, non-mock vtable replay execution summary, non-mock vtable REFIT replay with OOF prediction-cache store, compiled C conformance plan-build and replay against `dag_ml.h` |

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
cargo run -p dag-ml-cli -- print-execution-schedule --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --phase FIT_CV --plan-id plan:cli.schedule
cargo run -p dag-ml-cli -- validate-data-binding --campaign examples/campaign_oof_generation.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --node model:base --input x
cargo run -p dag-ml-cli -- validate-data-binding --campaign examples/campaign_data_contract_nir_s001.json --envelope ../dag-ml-data/examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json --node model:base --input x
cargo run -p dag-ml-cli -- run-mock-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.process
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:cli.process
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation_param_overrides.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:cli.process.param-overrides --run-id run:cli.process.param-overrides
cargo run -p dag-ml-cli -- run-process-campaign --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:cli.branch.merge
cargo run -p dag-ml-cli -- run-process-cv-refit-bundle --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --bundle-id bundle:generated.branch.merge.cv.refit --selections examples/fixtures/bundle/selection_decisions_branch_merge.json --output examples/generated/execution_bundle_branch_merge_cv_refit.json --prediction-cache-output examples/generated/prediction_cache_branch_merge_cv_refit.json --plan-id plan:generated.branch.merge.cv.refit
cargo run -p dag-ml-cli -- validate-prediction-cache --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --payload examples/generated/prediction_cache_branch_merge_cv_refit.json
cargo run -p dag-ml-cli -- export-prediction-cache-store --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --payload examples/generated/prediction_cache_branch_merge_cv_refit.json --output-dir examples/generated/prediction_cache_store_branch_merge_cv_refit
cargo run -p dag-ml-cli -- validate-prediction-cache-store --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --store-dir examples/generated/prediction_cache_store_branch_merge_cv_refit
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_branch_merge_predict.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:generated.branch.merge.cv.refit
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_branch_merge_refit.json --prediction-cache-payload examples/generated/prediction_cache_branch_merge_cv_refit.json --adapter examples/adapters/python_process_controller.py --persistent --plan-id plan:generated.branch.merge.cv.refit
cargo run -p dag-ml-cli -- run-mock-replay --bundle examples/generated/execution_bundle_branch_merge_cv_refit.json --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --envelope merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_branch_merge_refit.json --prediction-cache-store examples/generated/prediction_cache_store_branch_merge_cv_refit --plan-id plan:generated.branch.merge.cv.refit
cargo run -p dag-ml-cli -- run-process-cv-refit-replay --graph examples/branch_merge_oof_graph.json --campaign examples/campaign_branch_merge_oof.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/sklearn_process_controller.py --bundle-id bundle:cli.branch.merge.sklearn.cv.refit.replay --selections examples/fixtures/bundle/selection_decisions_branch_merge.json --plan-id plan:cli.branch.merge.sklearn.cv.refit.replay
cargo run -p dag-ml-cli -- select-candidates --policy examples/fixtures/bundle/selection_policy_rmse.json --candidates examples/fixtures/bundle/candidate_scores_demo.json --groups examples/fixtures/bundle/candidate_groups_demo.json --output examples/generated/selection_decisions_demo.json
cargo run -p dag-ml-cli -- build-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --bundle-spec examples/fixtures/bundle/bundle_build_spec_minimal.json --output examples/generated/execution_bundle_minimal.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-refit-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --bundle-id bundle:cli.refit.capture --output examples/generated/execution_bundle_refit_capture.json --plan-id plan:cli.refit.capture
cargo run -p dag-ml-cli -- run-process-refit-bundle --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/python_process_controller.py --persistent --bundle-id bundle:cli.process.refit.capture --output examples/generated/execution_bundle_process_refit_capture.json --plan-id plan:cli.process.refit.capture
cargo run -p dag-ml-cli -- run-process-refit-replay --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope examples/fixtures/data/coordinator_data_plan_envelope_nir.json --adapter examples/adapters/sklearn_process_controller.py --bundle-id bundle:cli.sklearn.refit.replay --plan-id plan:cli.sklearn.refit.replay
cargo run -p dag-ml-cli -- validate-sklearn-complex-demo --campaign examples/generated/sklearn_complex_oof_campaign.json --report examples/generated/sklearn_complex_report.json
cargo run -p dag-ml-cli -- validate-bundle --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-mock-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --plan-id plan:cli.bundle
cargo run -p dag-ml-cli -- run-process-replay --bundle examples/generated/execution_bundle_minimal.json --graph examples/minimal_graph.json --campaign examples/campaign_oof_generation.json --controllers examples/controller_manifests.json --envelope model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json --replay-request examples/fixtures/bundle/replay_request_predict.json --adapter examples/adapters/python_process_controller.py --plan-id plan:cli.bundle
python examples/sklearn_complex_oof_demo.py
cargo run -p dag-ml-cli -- validate-oof-campaign examples/generated/sklearn_complex_oof_campaign.json
python3 -m json.tool docs/contracts/coordinator_data_plan_envelope.schema.json >/dev/null
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
6. executes the mock replay ABI helper and validates the returned summary;
7. compiles a C conformance executable against `dag_ml.h`, links it with
   `libdag_ml_capi`, builds the execution plan through the ABI, and drives
   non-mock vtable replay.
