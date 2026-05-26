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

    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_file(temp_selection);
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after UNIX_EPOCH")
        .as_nanos()
}
