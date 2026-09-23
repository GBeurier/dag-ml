use std::path::{Path, PathBuf};
use std::process::Command;

use dag_ml_core::{ExternalDataPlanEnvelope, InitialFullRefitPackage};
use serde_json::json;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

#[test]
fn cli_captures_closed_initial_full_refit_without_cv() {
    let root = root();
    let temp = std::env::temp_dir().join(format!("dag_ml_initial_refit_{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let mut dsl = json!({
        "id": "dsl:initial.refit.test",
        "input": {"name": "x", "representation": "tabular_numeric"},
        "campaign_id": "campaign:initial.refit.test",
        "root_seed": 7,
        "leakage_policy": {"split_unit": "sample", "forbid_origin_cross_fold": true,
            "allow_observation_split_with_shared_target": false, "require_group_ids": false, "unsafe_flags": []},
        "data_bindings": [{"node_id": "model:initial", "input_name": "x",
            "request_id": "nir-to-tabular",
            "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
            "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
            "relation_fingerprint": "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10",
            "output_representation": "tabular_numeric", "feature_set_id": "x",
            "source_ids": ["nir"], "require_relations": true}],
        "steps": [{"kind": "model", "id": "model:initial", "operator": {"type": "MockModel"}, "params": {}}]
    });
    let mut envelope: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            root.join("examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"),
        )
        .unwrap(),
    )
    .unwrap();
    envelope["data_content_fingerprint"] = json!("e".repeat(64));
    envelope["target_content_fingerprint"] = json!("f".repeat(64));
    let typed_envelope: ExternalDataPlanEnvelope =
        serde_json::from_value(envelope.clone()).unwrap();
    let relation_fingerprint = typed_envelope
        .coordinator_relations
        .as_ref()
        .unwrap()
        .fingerprint()
        .unwrap();
    envelope["relation_fingerprint"] = json!(relation_fingerprint);
    dsl["data_bindings"][0]["relation_fingerprint"] = json!(relation_fingerprint);
    let dsl_path = temp.join("dsl.json");
    let envelope_path = temp.join("envelope.json");
    let ids_path = temp.join("ids.json");
    let package_path = temp.join("package.json");
    let outcome_path = temp.join("outcome.json");
    std::fs::write(&dsl_path, dsl.to_string()).unwrap();
    std::fs::write(&envelope_path, envelope.to_string()).unwrap();
    std::fs::write(&ids_path, json!(["sample:2", "sample:1"]).to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
        .current_dir(&root)
        .arg("run-process-dsl-refit-phase")
        .args([
            "--dsl",
            dsl_path.to_str().unwrap(),
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            envelope_path.to_str().unwrap(),
            "--training-sample-ids",
            ids_path.to_str().unwrap(),
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--package-output",
            package_path.to_str().unwrap(),
            "--output",
            outcome_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let package_json = std::fs::read_to_string(&package_path).unwrap();
    let package = InitialFullRefitPackage::from_json(&package_json).unwrap();
    assert_eq!(
        package
            .training_sample_ids
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        vec!["sample:2", "sample:1"]
    );
    assert_eq!(package.artifacts.len(), 1);
    assert_eq!(package.execution_root_seed, Some(12345));
    let outcome: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&outcome_path).unwrap()).unwrap();
    assert!(outcome.get("scores").is_none());
    assert_eq!(
        outcome["initial_full_refit_package"]["package_fingerprint"],
        package.package_fingerprint
    );
    let valid = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
        .arg("validate-initial-full-refit-package")
        .arg(&package_path)
        .output()
        .unwrap();
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let mut tampered: serde_json::Value = serde_json::from_str(&package_json).unwrap();
    tampered["training_sample_ids"][0] = json!("sample:wrong");
    let tampered_path = temp.join("tampered.json");
    std::fs::write(&tampered_path, tampered.to_string()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
        .arg("validate-initial-full-refit-package")
        .arg(&tampered_path)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    std::fs::remove_dir_all(temp).unwrap();
}
