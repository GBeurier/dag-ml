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

fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "dag_ml_cli_{label}_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos()
    ))
}

#[test]
fn cli_validates_and_projects_w10_training_contracts() {
    let root = repo_root();
    let training = "examples/fixtures/training/training_request_active_influence.v1.json";
    let validate = Command::new(cli())
        .current_dir(&root)
        .args(["validate-training-request", training])
        .output()
        .expect("validate-training-request runs");
    assert!(
        validate.status.success(),
        "training validation failed: {}",
        String::from_utf8_lossy(&validate.stderr)
    );
    assert!(String::from_utf8_lossy(&validate.stdout).contains("training:fixture.active_influence"));

    let projection_path = temp_path("training_projection");
    let project = Command::new(cli())
        .current_dir(&root)
        .args([
            "project-training-request",
            training,
            "--output",
            projection_path.to_str().expect("temp path is UTF-8"),
        ])
        .output()
        .expect("project-training-request runs");
    assert!(
        project.status.success(),
        "training projection failed: {}",
        String::from_utf8_lossy(&project.stderr)
    );
    let projection: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&projection_path).expect("projection was written"))
            .expect("projection is JSON");
    assert_eq!(
        projection["request_id"],
        "training:fixture.active_influence"
    );
    assert_eq!(projection["outputs"][0]["port_name"], "oof");
    assert_eq!(projection["parameters"]["structural_patch_count"], 0);
    let _ = std::fs::remove_file(projection_path);

    for (command, fixture) in [
        (
            "validate-portable-predictor-package",
            "examples/fixtures/training/portable_predictor_package.v1.json",
        ),
        (
            "validate-cache-namespace",
            "examples/fixtures/training/cache_namespace_fit_cv.v1.json",
        ),
    ] {
        let output = Command::new(cli())
            .current_dir(&root)
            .args([command, fixture])
            .output()
            .expect("W1 CLI validation runs");
        assert!(
            output.status.success(),
            "{command} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn cli_rejects_refingerprinted_missing_influence_slot() {
    let root = repo_root();
    let negatives: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("examples/fixtures/training/negative_cases.v1.json"))
            .expect("negative fixture is readable"),
    )
    .expect("negative fixture is JSON");
    let case = negatives["cases"]
        .as_array()
        .expect("cases is an array")
        .iter()
        .find(|case| case["id"] == "training_influence_missing_slot")
        .expect("missing-slot case exists");
    let path = temp_path("missing_influence_slot");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&case["document"]).expect("case serializes"),
    )
    .expect("negative request is written");
    let output = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-training-request",
            path.to_str().expect("temp path is UTF-8"),
        ])
        .output()
        .expect("negative validation runs");
    let _ = std::fs::remove_file(path);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("exactly cover active capability scopes")
    );
}
