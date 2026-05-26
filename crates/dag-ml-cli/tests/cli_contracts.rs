use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("dag-ml-cli crate lives under crates/dag-ml-cli")
        .to_path_buf()
}

fn cli() -> &'static str {
    env!("CARGO_BIN_EXE_dag-ml-cli")
}

#[test]
fn cli_scores_regression_prediction_blocks() {
    let root = repo_root();
    let suffix = unique_suffix();
    let predictions_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_regression_predictions_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let targets_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_regression_targets_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let output_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_regression_score_{}_{}.json",
        std::process::id(),
        suffix
    ));
    std::fs::write(
        &predictions_path,
        r#"{
  "prediction_id": "pred:sample",
  "producer_node": "model:pls",
  "partition": "validation",
  "sample_ids": ["sample:1", "sample:2"],
  "values": [[2.0], [4.0]],
  "target_names": ["y"]
}
"#,
    )
    .expect("prediction fixture was written");
    std::fs::write(
        &targets_path,
        r#"{
  "level": "sample",
  "unit_ids": [
    {"level": "sample", "id": "sample:2"},
    {"level": "sample", "id": "sample:1"}
  ],
  "values": [[5.0], [1.0]],
  "target_names": ["y"]
}
"#,
    )
    .expect("target fixture was written");

    let score = Command::new(cli())
        .current_dir(&root)
        .args([
            "score-regression",
            "--prediction-block",
            "sample",
            "--predictions",
            predictions_path.to_str().expect("temp path is valid utf-8"),
            "--targets",
            targets_path.to_str().expect("temp path is valid utf-8"),
            "--metric",
            "rmse",
            "--metric",
            "r2",
            "--candidate-id",
            "model:pls",
            "--output",
            output_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli score-regression");
    assert!(
        score.status.success(),
        "score-regression failed: {}",
        String::from_utf8_lossy(&score.stderr)
    );

    let scored: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&output_path).expect("regression score output was written"),
    )
    .expect("regression score output is JSON");
    assert_eq!(scored["report"]["metrics"]["rmse"], 1.0);
    assert_eq!(scored["report"]["metrics"]["r2"], 0.75);
    assert_eq!(scored["candidate_score"]["candidate_id"], "model:pls");
    assert_eq!(
        scored["candidate_score"]["metadata"]["producer_node"],
        "model:pls"
    );
}

#[test]
fn cli_selects_builds_and_validates_replay_bundle() {
    let root = repo_root();
    let temp_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_refit_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_refit_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_process_refit_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_process_refit_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_cv_refit_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_cv_refit_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_lineage = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_lineage_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_prediction_cache = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_prediction_cache_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_prediction_cache_store = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_prediction_cache_store_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_artifact_manifest_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_artifact_manifest_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_provenance_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_research_provenance_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_prediction_cache_tampered = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_prediction_cache_tampered_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_refit_request = std::env::temp_dir().join(format!(
        "dag_ml_cli_refit_request_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_process_refit_request = std::env::temp_dir().join(format!(
        "dag_ml_cli_process_refit_request_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_branch_merge_replay_request = std::env::temp_dir().join(format!(
        "dag_ml_cli_branch_merge_replay_request_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_selection = std::env::temp_dir().join(format!(
        "dag_ml_cli_selection_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_schedule = std::env::temp_dir().join(format!(
        "dag_ml_cli_schedule_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_sklearn_demo_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_sklearn_complex_{}_{}",
        std::process::id(),
        unique_suffix()
    ));

    let select = Command::new(cli())
        .current_dir(&root)
        .args([
            "select-candidates",
            "--policy",
            "examples/fixtures/bundle/selection_policy_rmse.json",
            "--candidates",
            "examples/fixtures/bundle/candidate_scores_demo.json",
            "--groups",
            "examples/fixtures/bundle/candidate_groups_demo.json",
            "--output",
            temp_selection.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli select-candidates");
    assert!(
        select.status.success(),
        "select-candidates failed: {}",
        String::from_utf8_lossy(&select.stderr)
    );

    let schedule = Command::new(cli())
        .current_dir(&root)
        .args([
            "print-execution-schedule",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--phase",
            "FIT_CV",
            "--output",
            temp_schedule.to_str().expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.schedule",
        ])
        .output()
        .expect("failed to run dag-ml-cli print-execution-schedule");
    assert!(
        schedule.status.success(),
        "print-execution-schedule failed: {}",
        String::from_utf8_lossy(&schedule.stderr)
    );
    let schedule_json = std::fs::read_to_string(&temp_schedule).expect("schedule was written");
    assert!(
        schedule_json.contains("\"phase\": \"FIT_CV\"")
            && schedule_json.contains("\"node_levels\"")
            && schedule_json.contains("transform:snv")
            && schedule_json.contains("model:base")
            && schedule_json.contains("fold:0"),
        "unexpected print-execution-schedule JSON: {}",
        schedule_json
    );

    let build = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-bundle",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--bundle-spec",
            "examples/fixtures/bundle/bundle_build_spec_minimal.json",
            "--output",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli build-bundle");
    assert!(
        build.status.success(),
        "build-bundle failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let refit_bundle = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-mock-refit-bundle",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--bundle-id",
            "bundle:cli.refit.capture",
            "--output",
            temp_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.refit.capture",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-mock-refit-bundle");
    assert!(
        refit_bundle.status.success(),
        "run-mock-refit-bundle failed: {}",
        String::from_utf8_lossy(&refit_bundle.stderr)
    );
    let refit_bundle_json =
        std::fs::read_to_string(&temp_refit_bundle).expect("refit bundle was written");
    assert!(
        refit_bundle_json.contains("artifact:model:base:refit")
            && refit_bundle_json.contains("refit_result_count"),
        "unexpected run-mock-refit-bundle JSON: {}",
        refit_bundle_json
    );
    std::fs::write(
        &temp_refit_request,
        r#"{
  "bundle_id": "bundle:cli.refit.capture",
  "phase": "PREDICT",
  "data_envelope_keys": ["model:base.x"]
}
"#,
    )
    .expect("refit replay request was written");

    let process_refit_bundle = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-refit-bundle",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--bundle-id",
            "bundle:cli.process.refit.capture",
            "--output",
            temp_process_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.process.refit.capture",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-process-refit-bundle");
    assert!(
        process_refit_bundle.status.success(),
        "run-process-refit-bundle failed: {}",
        String::from_utf8_lossy(&process_refit_bundle.stderr)
    );
    let process_refit_bundle_json = std::fs::read_to_string(&temp_process_refit_bundle)
        .expect("process refit bundle was written");
    assert!(
        process_refit_bundle_json.contains("artifact:model:base:refit")
            && process_refit_bundle_json.contains("refit_result_count"),
        "unexpected run-process-refit-bundle JSON: {}",
        process_refit_bundle_json
    );
    std::fs::write(
        &temp_process_refit_request,
        r#"{
  "bundle_id": "bundle:cli.process.refit.capture",
  "phase": "PREDICT",
  "data_envelope_keys": ["model:base.x"]
}
"#,
    )
    .expect("process refit replay request was written");

    let process_campaign = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--plan-id",
            "plan:cli.process",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-process-campaign");
    assert!(
        process_campaign.status.success(),
        "run-process-campaign failed: {}",
        String::from_utf8_lossy(&process_campaign.stderr)
    );
    let process_campaign_stdout = String::from_utf8_lossy(&process_campaign.stdout);
    assert!(
        process_campaign_stdout.contains("process campaign run: 8 result(s)")
            && process_campaign_stdout.contains("4 prediction block(s)")
            && process_campaign_stdout.contains("4 data handle(s)"),
        "unexpected run-process-campaign output: {}",
        process_campaign_stdout
    );

    let persistent_process_campaign = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--plan-id",
            "plan:cli.process",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-process-campaign --persistent");
    assert!(
        persistent_process_campaign.status.success(),
        "run-process-campaign --persistent failed: {}",
        String::from_utf8_lossy(&persistent_process_campaign.stderr)
    );
    let persistent_process_campaign_stdout =
        String::from_utf8_lossy(&persistent_process_campaign.stdout);
    assert!(
        persistent_process_campaign_stdout.contains("process campaign run: 8 result(s)")
            && persistent_process_campaign_stdout.contains("4 prediction block(s)")
            && persistent_process_campaign_stdout.contains("4 data handle(s)"),
        "unexpected run-process-campaign --persistent output: {}",
        persistent_process_campaign_stdout
    );

    let process_campaign_with_param_overrides = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation_param_overrides.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--plan-id",
            "plan:cli.process.param-overrides",
            "--run-id",
            "run:cli.process.param-overrides",
        ])
        .output()
        .expect("failed to run generated-param override process campaign");
    assert!(
        process_campaign_with_param_overrides.status.success(),
        "generated-param override process campaign failed: {}",
        String::from_utf8_lossy(&process_campaign_with_param_overrides.stderr)
    );
    let process_campaign_with_param_overrides_stdout =
        String::from_utf8_lossy(&process_campaign_with_param_overrides.stdout);
    assert!(
        process_campaign_with_param_overrides_stdout.contains("process campaign run: 8 result(s)")
            && process_campaign_with_param_overrides_stdout.contains("4 prediction block(s)")
            && process_campaign_with_param_overrides_stdout.contains("4 data handle(s)"),
        "unexpected generated-param override process campaign output: {}",
        process_campaign_with_param_overrides_stdout
    );

    let branch_merge_campaign = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--process-workers",
            "2",
            "--plan-id",
            "plan:cli.branch.merge",
            "--run-id",
            "run:cli.branch.merge",
        ])
        .output()
        .expect("failed to run branch/merge process campaign");
    assert!(
        branch_merge_campaign.status.success(),
        "branch/merge process campaign failed: {}",
        String::from_utf8_lossy(&branch_merge_campaign.stderr)
    );
    let branch_merge_stdout = String::from_utf8_lossy(&branch_merge_campaign.stdout);
    assert!(
        branch_merge_stdout.contains("process campaign run: 6 result(s)")
            && branch_merge_stdout.contains("6 prediction block(s)")
            && branch_merge_stdout.contains("6 data handle(s)")
            && branch_merge_stdout.contains("12 data view(s)")
            && branch_merge_stdout.contains("configured process worker(s)=2")
            && branch_merge_stdout.contains("observed process worker(s)=2"),
        "unexpected branch/merge process campaign output: {}",
        branch_merge_stdout
    );

    let branch_merge_refit_without_cv = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-refit-bundle",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--process-workers",
            "2",
            "--bundle-id",
            "bundle:cli.branch.merge.refit.without.cv",
            "--plan-id",
            "plan:cli.branch.merge.refit.without.cv",
            "--run-id",
            "run:cli.branch.merge.refit.without.cv",
        ])
        .output()
        .expect("failed to run branch/merge refit without CV");
    assert!(
        !branch_merge_refit_without_cv.status.success(),
        "branch/merge direct refit unexpectedly succeeded: {}",
        String::from_utf8_lossy(&branch_merge_refit_without_cv.stdout)
    );
    assert!(
        String::from_utf8_lossy(&branch_merge_refit_without_cv.stderr)
            .contains("requires OOF validation predictions"),
        "unexpected branch/merge direct refit error: {}",
        String::from_utf8_lossy(&branch_merge_refit_without_cv.stderr)
    );

    let branch_merge_cv_refit_bundle = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-cv-refit-bundle",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--process-workers",
            "2",
            "--bundle-id",
            "bundle:cli.branch.merge.cv.refit",
            "--selections",
            "examples/fixtures/bundle/selection_decisions_branch_merge.json",
            "--output",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--lineage-output",
            temp_branch_merge_lineage
                .to_str()
                .expect("temp path is valid utf-8"),
            "--prediction-cache-output",
            temp_branch_merge_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
            "--run-id",
            "run:cli.branch.merge.cv.refit",
        ])
        .output()
        .expect("failed to run branch/merge CV+refit process bundle");
    assert!(
        branch_merge_cv_refit_bundle.status.success(),
        "branch/merge CV+refit process bundle failed: {}",
        String::from_utf8_lossy(&branch_merge_cv_refit_bundle.stderr)
    );
    let branch_merge_cv_refit_stdout =
        String::from_utf8_lossy(&branch_merge_cv_refit_bundle.stdout);
    assert!(
        branch_merge_cv_refit_stdout.contains("process cv refit bundle run: 6 fit_cv result(s)")
            && branch_merge_cv_refit_stdout.contains("6 OOF prediction block(s)")
            && branch_merge_cv_refit_stdout.contains("3 refit result(s)")
            && branch_merge_cv_refit_stdout.contains("3 captured artifact handle(s)")
            && branch_merge_cv_refit_stdout.contains("2 prediction cache(s)")
            && branch_merge_cv_refit_stdout.contains("configured process worker(s)=2")
            && branch_merge_cv_refit_stdout.contains("observed process worker(s)=2"),
        "unexpected branch/merge CV+refit process bundle output: {}",
        branch_merge_cv_refit_stdout
    );
    let branch_merge_cv_refit_bundle_json =
        std::fs::read_to_string(&temp_branch_merge_cv_refit_bundle)
            .expect("branch/merge CV+refit bundle was written");
    assert!(
        branch_merge_cv_refit_bundle_json.contains("artifact:branch:b0.model:ridge:refit")
            && branch_merge_cv_refit_bundle_json.contains("artifact:branch:b1.model:rf:refit")
            && branch_merge_cv_refit_bundle_json
                .contains("artifact:merge:stack.pred_plus_original.meta:ridge:refit")
            && branch_merge_cv_refit_bundle_json.contains("prediction_requirements")
            && branch_merge_cv_refit_bundle_json.contains("prediction_caches")
            && branch_merge_cv_refit_bundle_json.contains("prediction_requirement_keys")
            && branch_merge_cv_refit_bundle_json.contains("dag-ml-json-prediction-blocks-v1")
            && branch_merge_cv_refit_bundle_json.contains(
                "prediction-cache:branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"
            )
            && branch_merge_cv_refit_bundle_json.contains(
                "branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"
            )
            && branch_merge_cv_refit_bundle_json.contains(
                "branch:b1.model:rf.oof->merge:stack.pred_plus_original.meta:ridge.b1_oof"
            )
            && branch_merge_cv_refit_bundle_json.contains("select:branch:b0.rmse_sample")
            && branch_merge_cv_refit_bundle_json.contains("select:merge.stack.rmse_sample")
            && branch_merge_cv_refit_bundle_json.contains("fit_cv_result_count")
            && branch_merge_cv_refit_bundle_json.contains("fit_cv_oof_prediction_block_count")
            && branch_merge_cv_refit_bundle_json.contains("oof_prediction_summary")
            && branch_merge_cv_refit_bundle_json.contains("\"block_count\": 2")
            && branch_merge_cv_refit_bundle_json.contains("refit_prediction_block_count"),
        "unexpected branch/merge CV+refit bundle JSON: {}",
        branch_merge_cv_refit_bundle_json
    );
    let branch_merge_lineage_json = std::fs::read_to_string(&temp_branch_merge_lineage)
        .expect("branch/merge lineage records were written");
    let branch_merge_lineage: serde_json::Value =
        serde_json::from_str(&branch_merge_lineage_json).expect("lineage JSON parses");
    let branch_merge_lineage_records = branch_merge_lineage
        .as_array()
        .expect("lineage export is an array");
    assert_eq!(branch_merge_lineage_records.len(), 9);
    assert!(
        branch_merge_lineage_records.iter().any(|record| {
            record["node_id"] == "merge:stack.pred_plus_original.meta:ridge"
                && record["phase"] == "FIT_CV"
                && record["input_lineage"]
                    .as_array()
                    .is_some_and(|lineage| lineage.len() == 2)
        }) && branch_merge_lineage_records.iter().any(|record| {
            record["node_id"] == "merge:stack.pred_plus_original.meta:ridge"
                && record["phase"] == "REFIT"
                && record["input_lineage"]
                    .as_array()
                    .is_some_and(|lineage| lineage.len() == 2)
        }),
        "unexpected branch/merge lineage JSON: {}",
        branch_merge_lineage_json
    );
    let branch_merge_prediction_cache_json =
        std::fs::read_to_string(&temp_branch_merge_prediction_cache)
            .expect("branch/merge prediction cache payload was written");
    assert!(
        branch_merge_prediction_cache_json.contains("\"bundle_id\": \"bundle:cli.branch.merge.cv.refit\"")
            && branch_merge_prediction_cache_json.contains("\"schema_version\": 1")
            && branch_merge_prediction_cache_json.contains("\"caches\"")
            && branch_merge_prediction_cache_json.contains("\"blocks\"")
            && branch_merge_prediction_cache_json.contains("\"values\"")
            && branch_merge_prediction_cache_json.contains(
                "prediction-cache:branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"
            )
            && branch_merge_prediction_cache_json.contains(
                "prediction-cache:branch:b1.model:rf.oof->merge:stack.pred_plus_original.meta:ridge.b1_oof"
            ),
        "unexpected branch/merge prediction cache payload JSON: {}",
        branch_merge_prediction_cache_json
    );
    let validate_prediction_cache = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-prediction-cache",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--payload",
            temp_branch_merge_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to validate branch/merge prediction cache payload");
    assert!(
        validate_prediction_cache.status.success(),
        "validate-prediction-cache failed: {}",
        String::from_utf8_lossy(&validate_prediction_cache.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_prediction_cache.stdout).contains(
            "valid prediction cache payload set: bundle=bundle:cli.branch.merge.cv.refit, cache(s)=2"
        ),
        "unexpected validate-prediction-cache output: {}",
        String::from_utf8_lossy(&validate_prediction_cache.stdout)
    );
    let export_prediction_cache_store = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-prediction-cache-store",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--payload",
            temp_branch_merge_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--output-dir",
            temp_branch_merge_prediction_cache_store
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to export branch/merge prediction cache store");
    assert!(
        export_prediction_cache_store.status.success(),
        "export-prediction-cache-store failed: {}",
        String::from_utf8_lossy(&export_prediction_cache_store.stderr)
    );
    assert!(
        String::from_utf8_lossy(&export_prediction_cache_store.stdout).contains(
            "wrote prediction cache store: bundle=bundle:cli.branch.merge.cv.refit, cache(s)=2"
        ),
        "unexpected export-prediction-cache-store output: {}",
        String::from_utf8_lossy(&export_prediction_cache_store.stdout)
    );
    let validate_prediction_cache_store = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-prediction-cache-store",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--store-dir",
            temp_branch_merge_prediction_cache_store
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to validate branch/merge prediction cache store");
    assert!(
        validate_prediction_cache_store.status.success(),
        "validate-prediction-cache-store failed: {}",
        String::from_utf8_lossy(&validate_prediction_cache_store.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_prediction_cache_store.stdout).contains(
            "valid prediction cache store: bundle=bundle:cli.branch.merge.cv.refit, cache(s)=2"
        ),
        "unexpected validate-prediction-cache-store output: {}",
        String::from_utf8_lossy(&validate_prediction_cache_store.stdout)
    );
    let export_branch_merge_artifact_manifest = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-artifact-manifest",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--output-dir",
            temp_branch_merge_artifact_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to export branch/merge artifact manifest");
    assert!(
        export_branch_merge_artifact_manifest.status.success(),
        "export branch/merge artifact manifest failed: {}",
        String::from_utf8_lossy(&export_branch_merge_artifact_manifest.stderr)
    );
    assert!(
        String::from_utf8_lossy(&export_branch_merge_artifact_manifest.stdout).contains(
            "wrote artifact manifest: bundle=bundle:cli.branch.merge.cv.refit, artifact(s)=3"
        ),
        "unexpected branch/merge artifact manifest output: {}",
        String::from_utf8_lossy(&export_branch_merge_artifact_manifest.stdout)
    );
    let export_branch_merge_research_provenance = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-research-provenance",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--lineage",
            temp_branch_merge_lineage
                .to_str()
                .expect("temp path is valid utf-8"),
            "--prediction-cache-store",
            temp_branch_merge_prediction_cache_store
                .to_str()
                .expect("temp path is valid utf-8"),
            "--artifact-manifest",
            temp_branch_merge_artifact_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--output-dir",
            temp_branch_merge_provenance_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
        ])
        .output()
        .expect("failed to export branch/merge research provenance");
    assert!(
        export_branch_merge_research_provenance.status.success(),
        "export branch/merge research provenance failed: {}",
        String::from_utf8_lossy(&export_branch_merge_research_provenance.stderr)
    );
    assert!(
        String::from_utf8_lossy(&export_branch_merge_research_provenance.stdout).contains(
            "wrote research provenance export: bundle=bundle:cli.branch.merge.cv.refit, lineage record(s)=9, data envelope(s)=3, prediction cache manifest=true, artifact manifest=true"
        ),
        "unexpected branch/merge research provenance output: {}",
        String::from_utf8_lossy(&export_branch_merge_research_provenance.stdout)
    );
    let validate_branch_merge_research_provenance = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-research-provenance",
            "--input-dir",
            temp_branch_merge_provenance_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to validate branch/merge research provenance");
    assert!(
        validate_branch_merge_research_provenance.status.success(),
        "validate branch/merge research provenance failed: {}",
        String::from_utf8_lossy(&validate_branch_merge_research_provenance.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_branch_merge_research_provenance.stdout).contains(
            "valid research provenance package: bundle=bundle:cli.branch.merge.cv.refit, plan=plan:cli.branch.merge.cv.refit"
        ),
        "unexpected branch/merge research provenance validation output: {}",
        String::from_utf8_lossy(&validate_branch_merge_research_provenance.stdout)
    );
    let branch_merge_prov_json =
        std::fs::read_to_string(temp_branch_merge_provenance_dir.join("lineage.prov.jsonld"))
            .expect("branch/merge PROV JSON-LD was written");
    assert!(
        branch_merge_prov_json.contains("dagml:LineageRecord")
            && branch_merge_prov_json.contains("dagml:PredictionCacheManifest")
            && branch_merge_prov_json.contains("dagml:ArtifactManifest")
            && branch_merge_prov_json.contains("dagml:oof_dependency")
            && branch_merge_prov_json.contains("dagml:lineage_dependency")
            && branch_merge_prov_json.contains("merge:stack.pred_plus_original.meta:ridge"),
        "unexpected branch/merge PROV JSON-LD: {}",
        branch_merge_prov_json
    );
    let branch_merge_ro_crate_json =
        std::fs::read_to_string(temp_branch_merge_provenance_dir.join("ro-crate-metadata.json"))
            .expect("branch/merge RO-Crate metadata was written");
    for path in [
        "execution_plan.json",
        "execution_bundle.json",
        "lineage_records.json",
        "lineage.prov.jsonld",
        "ro-crate-metadata.json",
        "prediction_cache_manifest.json",
        "artifact_manifest.json",
        "data_envelopes/branch:b0.model:ridge.x.json",
        "data_envelopes/branch:b1.model:rf.x.json",
        "data_envelopes/merge:stack.pred_plus_original.meta:ridge.x_original.json",
    ] {
        assert!(
            temp_branch_merge_provenance_dir.join(path).exists(),
            "research provenance package is missing {path}"
        );
    }
    assert!(
        branch_merge_ro_crate_json.contains("ComputationalWorkflow")
            && branch_merge_ro_crate_json.contains("prediction_cache_manifest.json")
            && branch_merge_ro_crate_json.contains("artifact_manifest.json")
            && branch_merge_ro_crate_json.contains("data_envelopes/branch:b0.model:ridge.x.json")
            && branch_merge_ro_crate_json.contains("lineage.prov.jsonld")
            && branch_merge_ro_crate_json.contains("\"sha256\"")
            && branch_merge_ro_crate_json.contains("\"contentSize\""),
        "unexpected branch/merge RO-Crate metadata: {}",
        branch_merge_ro_crate_json
    );
    let mut tampered_prediction_cache: serde_json::Value =
        serde_json::from_str(&branch_merge_prediction_cache_json)
            .expect("prediction cache payload JSON parses");
    tampered_prediction_cache["caches"][0]["blocks"][0]["values"][0][0] =
        serde_json::json!(123456.0);
    std::fs::write(
        &temp_branch_merge_prediction_cache_tampered,
        serde_json::to_string_pretty(&tampered_prediction_cache)
            .expect("tampered prediction cache payload serializes"),
    )
    .expect("tampered prediction cache payload was written");
    let validate_tampered_prediction_cache = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-prediction-cache",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--payload",
            temp_branch_merge_prediction_cache_tampered
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to validate tampered branch/merge prediction cache payload");
    assert!(
        !validate_tampered_prediction_cache.status.success(),
        "tampered prediction cache payload unexpectedly validated: {}",
        String::from_utf8_lossy(&validate_tampered_prediction_cache.stdout)
    );
    assert!(
        String::from_utf8_lossy(&validate_tampered_prediction_cache.stderr)
            .contains("content fingerprint does not match blocks"),
        "unexpected tampered prediction cache validation error: {}",
        String::from_utf8_lossy(&validate_tampered_prediction_cache.stderr)
    );
    std::fs::write(
        &temp_branch_merge_replay_request,
        r#"{
  "bundle_id": "bundle:cli.branch.merge.cv.refit",
  "phase": "PREDICT",
  "data_envelope_keys": [
    "branch:b0.model:ridge.x",
    "branch:b1.model:rf.x",
    "merge:stack.pred_plus_original.meta:ridge.x_original"
  ]
}
"#,
    )
    .expect("branch/merge replay request was written");

    let branch_merge_replay = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-replay",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_branch_merge_replay_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
            "--run-id",
            "run:cli.branch.merge.replay",
        ])
        .output()
        .expect("failed to run branch/merge process replay");
    assert!(
        branch_merge_replay.status.success(),
        "branch/merge process replay failed: {}",
        String::from_utf8_lossy(&branch_merge_replay.stderr)
    );
    let branch_merge_replay_stdout = String::from_utf8_lossy(&branch_merge_replay.stdout);
    assert!(
        branch_merge_replay_stdout.contains("process replay run: 3 result(s)")
            && branch_merge_replay_stdout.contains("3 prediction block(s)")
            && branch_merge_replay_stdout.contains("3 data handle(s)")
            && branch_merge_replay_stdout.contains("3 data view(s)")
            && branch_merge_replay_stdout.contains("3 artifact handle(s)"),
        "unexpected branch/merge process replay output: {}",
        branch_merge_replay_stdout
    );

    std::fs::write(
        &temp_branch_merge_replay_request,
        r#"{
  "bundle_id": "bundle:cli.branch.merge.cv.refit",
  "phase": "REFIT",
  "data_envelope_keys": [
    "branch:b0.model:ridge.x",
    "branch:b1.model:rf.x",
    "merge:stack.pred_plus_original.meta:ridge.x_original"
  ]
}
"#,
    )
    .expect("branch/merge refit replay request was written");
    let validate_branch_merge_refit_replay = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-bundle",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_branch_merge_replay_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
        ])
        .output()
        .expect("failed to validate branch/merge refit replay request");
    assert!(
        !validate_branch_merge_refit_replay.status.success(),
        "branch/merge REFIT replay unexpectedly validated: {}",
        String::from_utf8_lossy(&validate_branch_merge_refit_replay.stdout)
    );
    assert!(
        String::from_utf8_lossy(&validate_branch_merge_refit_replay.stderr)
            .contains("cannot replay REFIT"),
        "unexpected branch/merge REFIT replay validation error: {}",
        String::from_utf8_lossy(&validate_branch_merge_refit_replay.stderr)
    );

    let validate_branch_merge_refit_replay_with_payload = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-bundle",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_branch_merge_replay_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--prediction-cache-payload",
            temp_branch_merge_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
        ])
        .output()
        .expect("failed to validate branch/merge refit replay request with payload");
    assert!(
        validate_branch_merge_refit_replay_with_payload
            .status
            .success(),
        "branch/merge REFIT replay with payload failed validation: {}",
        String::from_utf8_lossy(&validate_branch_merge_refit_replay_with_payload.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_branch_merge_refit_replay_with_payload.stdout)
            .contains("prediction cache payload(s)=2"),
        "unexpected branch/merge REFIT replay validation with payload output: {}",
        String::from_utf8_lossy(&validate_branch_merge_refit_replay_with_payload.stdout)
    );

    let branch_merge_refit_mock_replay_with_store = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-mock-replay",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_branch_merge_replay_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--prediction-cache-store",
            temp_branch_merge_prediction_cache_store
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
            "--run-id",
            "run:cli.branch.merge.refit.mock.replay.with.store",
        ])
        .output()
        .expect("failed to run branch/merge refit mock replay with prediction cache store");
    assert!(
        branch_merge_refit_mock_replay_with_store.status.success(),
        "branch/merge REFIT mock replay with store failed: {}",
        String::from_utf8_lossy(&branch_merge_refit_mock_replay_with_store.stderr)
    );
    let branch_merge_refit_mock_replay_with_store_stdout =
        String::from_utf8_lossy(&branch_merge_refit_mock_replay_with_store.stdout);
    assert!(
        branch_merge_refit_mock_replay_with_store_stdout.contains("mock replay run: 3 result(s)")
            && branch_merge_refit_mock_replay_with_store_stdout
                .contains("2 prediction cache handle(s)"),
        "unexpected branch/merge REFIT mock replay with store output: {}",
        branch_merge_refit_mock_replay_with_store_stdout
    );

    let branch_merge_refit_replay_with_payload = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-replay",
            "--bundle",
            temp_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_branch_merge_replay_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--prediction-cache-payload",
            temp_branch_merge_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--plan-id",
            "plan:cli.branch.merge.cv.refit",
            "--run-id",
            "run:cli.branch.merge.refit.replay.with.payload",
        ])
        .output()
        .expect("failed to run branch/merge refit replay with prediction cache payload");
    assert!(
        branch_merge_refit_replay_with_payload.status.success(),
        "branch/merge REFIT replay with payload failed: {}",
        String::from_utf8_lossy(&branch_merge_refit_replay_with_payload.stderr)
    );
    let branch_merge_refit_replay_with_payload_stdout =
        String::from_utf8_lossy(&branch_merge_refit_replay_with_payload.stdout);
    assert!(
        branch_merge_refit_replay_with_payload_stdout.contains("process replay run: 3 result(s)")
            && branch_merge_refit_replay_with_payload_stdout.contains("7 prediction block(s)")
            && branch_merge_refit_replay_with_payload_stdout.contains("3 data handle(s)")
            && branch_merge_refit_replay_with_payload_stdout.contains("3 data view(s)")
            && branch_merge_refit_replay_with_payload_stdout.contains("3 artifact handle(s)")
            && branch_merge_refit_replay_with_payload_stdout
                .contains("2 prediction cache handle(s)"),
        "unexpected branch/merge REFIT replay with payload output: {}",
        branch_merge_refit_replay_with_payload_stdout
    );

    let branch_merge_sklearn_cv_refit_replay = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-cv-refit-replay",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/sklearn_process_controller.py",
            "--process-workers",
            "2",
            "--bundle-id",
            "bundle:cli.branch.merge.sklearn.cv.refit.replay",
            "--selections",
            "examples/fixtures/bundle/selection_decisions_branch_merge.json",
            "--plan-id",
            "plan:cli.branch.merge.sklearn.cv.refit.replay",
            "--run-id",
            "run:cli.branch.merge.sklearn.cv.refit.replay",
        ])
        .output()
        .expect("failed to run branch/merge sklearn CV+refit+replay");
    assert!(
        branch_merge_sklearn_cv_refit_replay.status.success(),
        "branch/merge sklearn CV+refit+replay failed: {}",
        String::from_utf8_lossy(&branch_merge_sklearn_cv_refit_replay.stderr)
    );
    let branch_merge_sklearn_stdout =
        String::from_utf8_lossy(&branch_merge_sklearn_cv_refit_replay.stdout);
    assert!(
        branch_merge_sklearn_stdout.contains("process cv refit replay run: 6 fit_cv result(s)")
            && branch_merge_sklearn_stdout.contains("6 OOF prediction block(s)")
            && branch_merge_sklearn_stdout.contains("3 refit result(s)")
            && branch_merge_sklearn_stdout.contains("3 replay result(s)")
            && branch_merge_sklearn_stdout.contains("3 replay prediction block(s)")
            && branch_merge_sklearn_stdout.contains("3 captured artifact handle(s)")
            && branch_merge_sklearn_stdout.contains("2 prediction cache(s)")
            && branch_merge_sklearn_stdout.contains("configured process worker(s)=2")
            && branch_merge_sklearn_stdout.contains("observed process worker(s)=2"),
        "unexpected branch/merge sklearn CV+refit+replay output: {}",
        branch_merge_sklearn_stdout
    );

    let validate = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-bundle",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            "examples/fixtures/bundle/replay_request_predict.json",
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-bundle");
    assert!(
        validate.status.success(),
        "validate-bundle failed: {}",
        String::from_utf8_lossy(&validate.stderr)
    );
    let validate_stdout = String::from_utf8_lossy(&validate.stdout);
    assert!(
        validate_stdout.contains("valid bundle: bundle:cli.demo")
            && validate_stdout.contains("prediction requirement(s)=0, prediction cache(s)=0"),
        "unexpected validate-bundle output: {}",
        validate_stdout
    );

    let validate_refit = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-bundle",
            "--bundle",
            temp_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_refit_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.refit.capture",
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-bundle for refit bundle");
    assert!(
        validate_refit.status.success(),
        "validate refit bundle failed: {}",
        String::from_utf8_lossy(&validate_refit.stderr)
    );
    let validate_refit_stdout = String::from_utf8_lossy(&validate_refit.stdout);
    assert!(
        validate_refit_stdout.contains("valid bundle: bundle:cli.refit.capture")
            && validate_refit_stdout.contains("prediction requirement(s)=0, prediction cache(s)=0"),
        "unexpected validate refit bundle output: {}",
        validate_refit_stdout
    );

    let validate_process_refit = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-bundle",
            "--bundle",
            temp_process_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            temp_process_refit_request
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.process.refit.capture",
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-bundle for process refit bundle");
    assert!(
        validate_process_refit.status.success(),
        "validate process refit bundle failed: {}",
        String::from_utf8_lossy(&validate_process_refit.stderr)
    );
    let validate_process_refit_stdout = String::from_utf8_lossy(&validate_process_refit.stdout);
    assert!(
        validate_process_refit_stdout.contains("valid bundle: bundle:cli.process.refit.capture")
            && validate_process_refit_stdout
                .contains("prediction requirement(s)=0, prediction cache(s)=0"),
        "unexpected validate process refit bundle output: {}",
        validate_process_refit_stdout
    );

    let replay = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-mock-replay",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            "examples/fixtures/bundle/replay_request_predict.json",
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-mock-replay");
    assert!(
        replay.status.success(),
        "run-mock-replay failed: {}",
        String::from_utf8_lossy(&replay.stderr)
    );
    assert!(
        String::from_utf8_lossy(&replay.stdout).contains("1 artifact handle(s)"),
        "unexpected run-mock-replay output: {}",
        String::from_utf8_lossy(&replay.stdout)
    );

    let process_replay = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-replay",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            "examples/fixtures/bundle/replay_request_predict.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-process-replay");
    assert!(
        process_replay.status.success(),
        "run-process-replay failed: {}",
        String::from_utf8_lossy(&process_replay.stderr)
    );
    let process_stdout = String::from_utf8_lossy(&process_replay.stdout);
    assert!(
        process_stdout.contains("process replay run: 2 result(s)")
            && process_stdout.contains("1 artifact handle(s)"),
        "unexpected run-process-replay output: {}",
        process_stdout
    );

    if python_has_sklearn(&root) {
        let sklearn_refit_replay = Command::new(cli())
            .current_dir(&root)
            .args([
                "run-process-refit-replay",
                "--graph",
                "examples/minimal_graph.json",
                "--campaign",
                "examples/campaign_oof_generation.json",
                "--controllers",
                "examples/controller_manifests.json",
                "--envelope",
                "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
                "--adapter",
                "examples/adapters/sklearn_process_controller.py",
                "--bundle-id",
                "bundle:cli.sklearn.refit.replay",
                "--plan-id",
                "plan:cli.sklearn.refit.replay",
            ])
            .output()
            .expect("failed to run dag-ml-cli run-process-refit-replay");
        assert!(
            sklearn_refit_replay.status.success(),
            "run-process-refit-replay failed: {}",
            String::from_utf8_lossy(&sklearn_refit_replay.stderr)
        );
        let sklearn_stdout = String::from_utf8_lossy(&sklearn_refit_replay.stdout);
        assert!(
            sklearn_stdout.contains("process refit replay run: 2 refit result(s)")
                && sklearn_stdout.contains("1 replay prediction block(s)")
                && sklearn_stdout.contains("1 captured artifact handle(s)"),
            "unexpected run-process-refit-replay output: {}",
            sklearn_stdout
        );

        let sklearn_demo = Command::new("python3")
            .current_dir(&root)
            .args([
                "examples/sklearn_complex_oof_demo.py",
                "--out-dir",
                temp_sklearn_demo_dir
                    .to_str()
                    .expect("temp path is valid utf-8"),
                "--seed",
                "20260526",
            ])
            .output()
            .expect("failed to run sklearn complex demo");
        assert!(
            sklearn_demo.status.success(),
            "sklearn complex demo failed: {}",
            String::from_utf8_lossy(&sklearn_demo.stderr)
        );
        let sklearn_demo_campaign = temp_sklearn_demo_dir.join("sklearn_complex_oof_campaign.json");
        let sklearn_demo_report = temp_sklearn_demo_dir.join("sklearn_complex_report.json");
        let validate_sklearn_demo = Command::new(cli())
            .current_dir(&root)
            .args([
                "validate-sklearn-complex-demo",
                "--campaign",
                sklearn_demo_campaign
                    .to_str()
                    .expect("temp path is valid utf-8"),
                "--report",
                sklearn_demo_report
                    .to_str()
                    .expect("temp path is valid utf-8"),
            ])
            .output()
            .expect("failed to run validate-sklearn-complex-demo");
        assert!(
            validate_sklearn_demo.status.success(),
            "validate-sklearn-complex-demo failed: {}",
            String::from_utf8_lossy(&validate_sklearn_demo.stderr)
        );
        let validate_sklearn_stdout = String::from_utf8_lossy(&validate_sklearn_demo.stdout);
        assert!(
            validate_sklearn_stdout.contains("valid sklearn complex demo: 60 sample(s)")
                && validate_sklearn_stdout.contains("9 OOF column(s)")
                && validate_sklearn_stdout.contains("3 branch selection(s)")
                && validate_sklearn_stdout.contains("merge:m1.pred_meta_original.meta:ridge"),
            "unexpected validate-sklearn-complex-demo output: {}",
            validate_sklearn_stdout
        );
    }

    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_file(temp_refit_bundle);
    let _ = std::fs::remove_file(temp_process_refit_bundle);
    let _ = std::fs::remove_file(temp_branch_merge_cv_refit_bundle);
    let _ = std::fs::remove_file(temp_branch_merge_lineage);
    let _ = std::fs::remove_file(temp_branch_merge_prediction_cache);
    let _ = std::fs::remove_file(temp_branch_merge_prediction_cache_tampered);
    let _ = std::fs::remove_dir_all(temp_branch_merge_prediction_cache_store);
    let _ = std::fs::remove_dir_all(temp_branch_merge_artifact_manifest_dir);
    let _ = std::fs::remove_dir_all(temp_branch_merge_provenance_dir);
    let _ = std::fs::remove_file(temp_refit_request);
    let _ = std::fs::remove_file(temp_process_refit_request);
    let _ = std::fs::remove_file(temp_branch_merge_replay_request);
    let _ = std::fs::remove_file(temp_selection);
    let _ = std::fs::remove_file(temp_schedule);
    let _ = std::fs::remove_dir_all(temp_sklearn_demo_dir);
}

#[test]
fn cli_exports_and_validates_artifact_manifest() {
    let root = repo_root();
    let temp_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_artifact_manifest_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_manifest_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_artifact_manifest_dir_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let temp_legacy_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_artifact_manifest_legacy_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_legacy_manifest_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_artifact_manifest_legacy_dir_{}_{}",
        std::process::id(),
        unique_suffix()
    ));

    // Build a portable bundle from the manifest-ready fixture.
    let build = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-bundle",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--bundle-spec",
            "examples/fixtures/bundle/bundle_build_spec_minimal.json",
            "--output",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli build-bundle");
    assert!(
        build.status.success(),
        "build-bundle failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let export = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-artifact-manifest",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--output-dir",
            temp_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli export-artifact-manifest");
    assert!(
        export.status.success(),
        "export-artifact-manifest failed: {}",
        String::from_utf8_lossy(&export.stderr)
    );
    assert!(
        String::from_utf8_lossy(&export.stdout)
            .contains("wrote artifact manifest: bundle=bundle:cli.demo, artifact(s)=1"),
        "unexpected export-artifact-manifest output: {}",
        String::from_utf8_lossy(&export.stdout)
    );
    let manifest_json = std::fs::read_to_string(temp_manifest_dir.join("artifact_manifest.json"))
        .expect("artifact manifest was written");
    assert!(
        manifest_json.contains("\"bundle_id\": \"bundle:cli.demo\"")
            && manifest_json.contains("\"schema_version\": 1")
            && manifest_json.contains("artifact:model:base:refit")
            && manifest_json.contains("\"backend\": \"joblib\"")
            && manifest_json.contains("\"content_fingerprint\""),
        "unexpected artifact manifest JSON: {}",
        manifest_json
    );

    let validate_manifest = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-artifact-manifest",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--manifest-dir",
            temp_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-artifact-manifest");
    assert!(
        validate_manifest.status.success(),
        "validate-artifact-manifest failed: {}",
        String::from_utf8_lossy(&validate_manifest.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_manifest.stdout)
            .contains("valid artifact manifest: bundle=bundle:cli.demo, artifact(s)=1"),
        "unexpected validate-artifact-manifest output: {}",
        String::from_utf8_lossy(&validate_manifest.stdout)
    );

    let validate_bundle = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-bundle",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            "examples/fixtures/bundle/replay_request_predict.json",
            "--artifact-manifest",
            temp_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-bundle with artifact manifest");
    assert!(
        validate_bundle.status.success(),
        "validate-bundle with artifact manifest failed: {}",
        String::from_utf8_lossy(&validate_bundle.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_bundle.stdout).contains("valid bundle: bundle:cli.demo")
            && String::from_utf8_lossy(&validate_bundle.stdout)
                .contains("artifact manifest entries=1"),
        "unexpected validate-bundle with artifact manifest output: {}",
        String::from_utf8_lossy(&validate_bundle.stdout)
    );

    // A legacy/non-portable refit bundle has no portable artifact references, so
    // artifact manifest export must refuse it with a clear error and write no
    // manifest file.
    let legacy_bundle = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-mock-refit-bundle",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--bundle-id",
            "bundle:cli.artifact.manifest.legacy",
            "--output",
            temp_legacy_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.artifact.manifest.legacy",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-mock-refit-bundle");
    assert!(
        legacy_bundle.status.success(),
        "run-mock-refit-bundle failed: {}",
        String::from_utf8_lossy(&legacy_bundle.stderr)
    );

    let export_legacy = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-artifact-manifest",
            "--bundle",
            temp_legacy_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--output-dir",
            temp_legacy_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli export-artifact-manifest for legacy bundle");
    assert!(
        !export_legacy.status.success(),
        "export-artifact-manifest unexpectedly accepted a non-portable bundle: {}",
        String::from_utf8_lossy(&export_legacy.stdout)
    );
    assert!(
        String::from_utf8_lossy(&export_legacy.stderr).contains("not portable"),
        "unexpected legacy artifact manifest export error: {}",
        String::from_utf8_lossy(&export_legacy.stderr)
    );
    assert!(
        !temp_legacy_manifest_dir
            .join("artifact_manifest.json")
            .exists(),
        "legacy artifact manifest file was unexpectedly written at {}",
        temp_legacy_manifest_dir.display()
    );

    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_dir_all(temp_manifest_dir);
    let _ = std::fs::remove_file(temp_legacy_bundle);
    let _ = std::fs::remove_dir_all(temp_legacy_manifest_dir);
}

#[test]
fn cli_exports_research_provenance_bundle() {
    let root = repo_root();
    let suffix = unique_suffix();
    let temp_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_research_provenance_bundle_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_manifest_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_research_provenance_artifacts_{}_{}",
        std::process::id(),
        suffix
    ));
    let temp_provenance_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_research_provenance_dir_{}_{}",
        std::process::id(),
        suffix
    ));
    let temp_openlineage = std::env::temp_dir().join(format!(
        "dag_ml_cli_openlineage_{}_{}.json",
        std::process::id(),
        suffix
    ));

    let build = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-bundle",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--bundle-spec",
            "examples/fixtures/bundle/bundle_build_spec_minimal.json",
            "--output",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli build-bundle");
    assert!(
        build.status.success(),
        "build-bundle failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let export_artifacts = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-artifact-manifest",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--output-dir",
            temp_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli export-artifact-manifest");
    assert!(
        export_artifacts.status.success(),
        "export-artifact-manifest failed: {}",
        String::from_utf8_lossy(&export_artifacts.stderr)
    );

    let export_provenance = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-research-provenance",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--artifact-manifest",
            temp_manifest_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--output-dir",
            temp_provenance_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.bundle",
        ])
        .output()
        .expect("failed to run dag-ml-cli export-research-provenance");
    assert!(
        export_provenance.status.success(),
        "export-research-provenance failed: {}",
        String::from_utf8_lossy(&export_provenance.stderr)
    );
    assert!(
        String::from_utf8_lossy(&export_provenance.stdout)
            .contains("wrote research provenance export: bundle=bundle:cli.demo"),
        "unexpected export-research-provenance output: {}",
        String::from_utf8_lossy(&export_provenance.stdout)
    );
    let validate_provenance = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-research-provenance",
            "--input-dir",
            temp_provenance_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-research-provenance");
    assert!(
        validate_provenance.status.success(),
        "validate-research-provenance failed: {}",
        String::from_utf8_lossy(&validate_provenance.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_provenance.stdout).contains(
            "valid research provenance package: bundle=bundle:cli.demo, plan=plan:cli.bundle"
        ),
        "unexpected validate-research-provenance output: {}",
        String::from_utf8_lossy(&validate_provenance.stdout)
    );
    let export_openlineage = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-open-lineage",
            "--input-dir",
            temp_provenance_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--event-time",
            "2026-05-27T00:00:00Z",
            "--namespace",
            "dag-ml-cli-test",
            "--output",
            temp_openlineage.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli export-open-lineage");
    assert!(
        export_openlineage.status.success(),
        "export-open-lineage failed: {}",
        String::from_utf8_lossy(&export_openlineage.stderr)
    );
    let openlineage_json =
        std::fs::read_to_string(&temp_openlineage).expect("OpenLineage export was written");
    assert!(
        openlineage_json.contains("\"eventType\": \"COMPLETE\"")
            && openlineage_json.contains("\"schemaURL\"")
            && openlineage_json.contains("\"dagml_reproducibility\"")
            && openlineage_json.contains("\"dag-ml-cli-test\""),
        "unexpected OpenLineage export: {}",
        openlineage_json
    );

    let prov_json = std::fs::read_to_string(temp_provenance_dir.join("lineage.prov.jsonld"))
        .expect("PROV JSON-LD export was written");
    assert!(
        prov_json.contains("\"prov\"")
            && prov_json.contains("\"wasDerivedFrom\"")
            && prov_json.contains("dagml:ExecutionBundle")
            && prov_json.contains("dagml:ArtifactManifest"),
        "unexpected PROV JSON-LD export: {}",
        prov_json
    );
    let ro_crate_json = std::fs::read_to_string(temp_provenance_dir.join("ro-crate-metadata.json"))
        .expect("RO-Crate metadata export was written");
    for path in [
        "execution_plan.json",
        "execution_bundle.json",
        "lineage_records.json",
        "lineage.prov.jsonld",
        "ro-crate-metadata.json",
        "artifact_manifest.json",
    ] {
        assert!(
            temp_provenance_dir.join(path).exists(),
            "research provenance package is missing {path}"
        );
    }
    assert!(
        ro_crate_json.contains("ComputationalWorkflow")
            && ro_crate_json.contains("execution_bundle.json")
            && ro_crate_json.contains("artifact_manifest.json")
            && ro_crate_json.contains("lineage.prov.jsonld")
            && ro_crate_json.contains("\"sha256\""),
        "unexpected RO-Crate metadata export: {}",
        ro_crate_json
    );

    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_dir_all(temp_manifest_dir);
    let _ = std::fs::remove_dir_all(temp_provenance_dir);
    let _ = std::fs::remove_file(temp_openlineage);
}

#[test]
fn process_adapters_describe_supported_protocol_modes() {
    let root = repo_root();
    for adapter in [
        "examples/adapters/python_process_controller.py",
        "examples/adapters/sklearn_process_controller.py",
        "examples/adapters/flaky_process_controller.py",
    ] {
        let describe = Command::new("python3")
            .current_dir(&root)
            .args([adapter, "--describe"])
            .output()
            .expect("failed to run process adapter describe handshake");
        assert!(
            describe.status.success(),
            "adapter `{adapter}` describe failed: {}",
            String::from_utf8_lossy(&describe.stderr)
        );
        let stdout = String::from_utf8_lossy(&describe.stdout);
        assert!(
            stdout.contains("\"protocol\": \"dag-ml-process-adapter\"")
                && stdout.contains("\"schema_version\": 1")
                && stdout.contains("\"one_shot\"")
                && stdout.contains("\"jsonl\"")
                && stdout.contains("\"control_frames_v1\"")
                && stdout.contains("\"node_task_json_v1\"")
                && stdout.contains("\"node_result_json_v1\"")
                && stdout.contains("\"parallel_invocation_v1\"")
                && stdout.contains("\"persistent_workers\"")
                && stdout.contains("\"worker_env\""),
            "unexpected adapter `{adapter}` describe output: {}",
            stdout
        );
    }
}

#[test]
fn cli_restarts_persistent_process_worker_after_timeout_when_retry_is_enabled() {
    let root = repo_root();
    let timeout_marker_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_flaky_timeout_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let retry_marker_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_flaky_retry_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let retry_lifecycle_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_lifecycle_retry_{}_{}",
        std::process::id(),
        unique_suffix()
    ));

    let timeout_run_id = format!("run:cli.process.timeout.{}", unique_suffix());
    let timeout = Command::new(cli())
        .current_dir(&root)
        .env("DAG_ML_FLAKY_MARKER_DIR", &timeout_marker_dir)
        .env("DAG_ML_FLAKY_SLEEP_SECONDS", "2.0")
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/flaky_process_controller.py",
            "--persistent",
            "--process-timeout-ms",
            "750",
            "--plan-id",
            "plan:cli.process.timeout",
            "--run-id",
            timeout_run_id.as_str(),
        ])
        .output()
        .expect("failed to run flaky process campaign without retry");
    assert!(
        !timeout.status.success(),
        "flaky process campaign unexpectedly succeeded without retry: {}",
        String::from_utf8_lossy(&timeout.stdout)
    );
    let timeout_stderr = String::from_utf8_lossy(&timeout.stderr);
    assert!(
        timeout_stderr.contains("timed out after 750 ms")
            && timeout_stderr.contains("after 1 attempt(s)"),
        "unexpected flaky timeout error: {}",
        timeout_stderr
    );

    let retry_run_id = format!("run:cli.process.retry.{}", unique_suffix());
    let retry = Command::new(cli())
        .current_dir(&root)
        .env("DAG_ML_FLAKY_MARKER_DIR", &retry_marker_dir)
        .env("DAG_ML_PROCESS_LIFECYCLE_MARKER_DIR", &retry_lifecycle_dir)
        .env("DAG_ML_FLAKY_SLEEP_SECONDS", "2.0")
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/flaky_process_controller.py",
            "--persistent",
            "--process-timeout-ms",
            "750",
            "--process-retries",
            "1",
            "--plan-id",
            "plan:cli.process.retry",
            "--run-id",
            retry_run_id.as_str(),
        ])
        .output()
        .expect("failed to run flaky process campaign with retry");
    assert!(
        retry.status.success(),
        "flaky process campaign with retry failed: {}",
        String::from_utf8_lossy(&retry.stderr)
    );
    let retry_stdout = String::from_utf8_lossy(&retry.stdout);
    assert!(
        retry_stdout.contains("process campaign run: 8 result(s)")
            && retry_stdout.contains("4 prediction block(s)")
            && retry_stdout.contains("4 data handle(s)"),
        "unexpected flaky retry output: {}",
        retry_stdout
    );
    assert!(
        lifecycle_marker_count(&retry_lifecycle_dir, "init") >= 2
            && lifecycle_marker_count(&retry_lifecycle_dir, "close") >= 2,
        "expected persistent worker init/close lifecycle markers in {}",
        retry_lifecycle_dir.display()
    );

    let invalid_timeout = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--process-timeout-ms",
            "0",
            "--plan-id",
            "plan:cli.process.invalid-timeout",
            "--run-id",
            "run:cli.process.invalid-timeout",
        ])
        .output()
        .expect("failed to run process campaign with invalid timeout");
    assert!(
        !invalid_timeout.status.success(),
        "invalid process timeout unexpectedly succeeded: {}",
        String::from_utf8_lossy(&invalid_timeout.stdout)
    );
    assert!(
        String::from_utf8_lossy(&invalid_timeout.stderr)
            .contains("--process-timeout-ms must be at least 1"),
        "unexpected invalid timeout error: {}",
        String::from_utf8_lossy(&invalid_timeout.stderr)
    );

    let _ = std::fs::remove_dir_all(timeout_marker_dir);
    let _ = std::fs::remove_dir_all(retry_marker_dir);
    let _ = std::fs::remove_dir_all(retry_lifecycle_dir);
}

#[test]
fn cli_parallel_scheduler_runs_branch_merge_process_campaign() {
    let root = repo_root();
    let run_id = format!("run:cli.parallel.branch.merge.{}", unique_suffix());
    let output = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--process-workers",
            "2",
            "--scheduler",
            "parallel",
            "--scheduler-workers",
            "2",
            "--plan-id",
            "plan:cli.parallel.branch.merge",
            "--run-id",
            run_id.as_str(),
        ])
        .output()
        .expect("failed to run parallel branch/merge process campaign");

    assert!(
        output.status.success(),
        "parallel branch/merge process campaign failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("process campaign run: 6 result(s)")
            && stdout.contains("6 prediction block(s)")
            && stdout.contains("scheduler=parallel")
            && stdout.contains("scheduler worker(s)=2")
            && stdout.contains("configured process worker(s)=2")
            && stdout.contains("observed process worker(s)=2"),
        "unexpected parallel branch/merge process campaign output: {}",
        stdout
    );
}

#[test]
fn cli_validates_sibling_dag_ml_data_coordinator_fixture_when_available() {
    let root = repo_root();
    let dag_ml_data_root = if let Some(path) = std::env::var_os("DAG_ML_DATA_REPO") {
        PathBuf::from(path)
    } else {
        let Some(workspace_parent) = root.parent() else {
            return;
        };
        workspace_parent.join("dag-ml-data")
    };
    let dag_ml_data_envelope = dag_ml_data_root
        .join("examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json");
    if !dag_ml_data_envelope.exists() {
        return;
    }

    let validate = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-data-binding",
            "--campaign",
            "examples/campaign_data_contract_nir_s001.json",
            "--envelope",
            dag_ml_data_envelope
                .to_str()
                .expect("dag-ml-data fixture path is valid utf-8"),
            "--node",
            "model:base",
            "--input",
            "x",
        ])
        .output()
        .expect("failed to run dag-ml-data coordinator fixture validation");
    assert!(
        validate.status.success(),
        "dag-ml-data coordinator fixture failed dag-ml validation\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&validate.stdout),
        String::from_utf8_lossy(&validate.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate.stdout)
            .contains("valid data binding: model:base.x -> 7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d"),
        "unexpected dag-ml-data coordinator fixture validation output: {}",
        String::from_utf8_lossy(&validate.stdout)
    );
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after UNIX_EPOCH")
        .as_nanos()
}

fn lifecycle_marker_count(dir: &Path, prefix: &str) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with(prefix))
                })
                .count()
        })
        .unwrap_or(0)
}

fn python_has_sklearn(root: &Path) -> bool {
    Command::new("python3")
        .current_dir(root)
        .args(["-c", "import sklearn"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
