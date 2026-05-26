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
    let temp_selection = std::env::temp_dir().join(format!(
        "dag_ml_cli_selection_{}_{}.json",
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
    }

    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_file(temp_refit_bundle);
    let _ = std::fs::remove_file(temp_process_refit_bundle);
    let _ = std::fs::remove_file(temp_refit_request);
    let _ = std::fs::remove_file(temp_process_refit_request);
    let _ = std::fs::remove_file(temp_selection);
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
