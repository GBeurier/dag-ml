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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
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
            && branch_merge_stdout.contains("12 data view(s)"),
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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
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
            "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--bundle-id",
            "bundle:cli.branch.merge.cv.refit",
            "--selections",
            "examples/fixtures/bundle/selection_decisions_branch_merge.json",
            "--output",
            temp_branch_merge_cv_refit_bundle
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
            && branch_merge_cv_refit_stdout.contains("3 captured artifact handle(s)"),
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
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
    assert!(
        String::from_utf8_lossy(&validate.stdout).contains("valid bundle: bundle:cli.demo"),
        "unexpected validate-bundle output: {}",
        String::from_utf8_lossy(&validate.stdout)
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
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
    assert!(
        String::from_utf8_lossy(&validate_refit.stdout)
            .contains("valid bundle: bundle:cli.refit.capture"),
        "unexpected validate refit bundle output: {}",
        String::from_utf8_lossy(&validate_refit.stdout)
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
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
    assert!(
        String::from_utf8_lossy(&validate_process_refit.stdout)
            .contains("valid bundle: bundle:cli.process.refit.capture"),
        "unexpected validate process refit bundle output: {}",
        String::from_utf8_lossy(&validate_process_refit.stdout)
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
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
            "model:base.x=examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
                "examples/fixtures/data/coordinator_data_plan_envelope_nir.json",
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
    let _ = std::fs::remove_file(temp_refit_request);
    let _ = std::fs::remove_file(temp_process_refit_request);
    let _ = std::fs::remove_file(temp_branch_merge_replay_request);
    let _ = std::fs::remove_file(temp_selection);
    let _ = std::fs::remove_dir_all(temp_sklearn_demo_dir);
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after UNIX_EPOCH")
        .as_nanos()
}

fn python_has_sklearn(root: &Path) -> bool {
    Command::new("python3")
        .current_dir(root)
        .args(["-c", "import sklearn"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
