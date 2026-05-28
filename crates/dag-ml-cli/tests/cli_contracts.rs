use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;
use sha2::{Digest, Sha256};

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

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[test]
fn cli_compiles_pipeline_dsl_to_graph() {
    let root = repo_root();
    let output_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_compiled_graph_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));

    let compile = Command::new(cli())
        .current_dir(&root)
        .args([
            "compile-pipeline-dsl",
            "--dsl",
            "examples/pipeline_dsl_branch_merge.json",
            "--output",
            output_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli compile-pipeline-dsl");
    assert!(
        compile.status.success(),
        "compile-pipeline-dsl failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let graph: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&output_path).expect("compiled graph output was written"),
    )
    .expect("compiled graph output is JSON");
    assert_eq!(graph["id"], "dsl-branch-merge-oof-smoke");
    assert_eq!(graph["nodes"].as_array().expect("nodes array").len(), 4);
    assert!(graph["edges"]
        .as_array()
        .expect("edges array")
        .iter()
        .any(|edge| edge["contract"]["requires_oof"] == true
            && edge["target"]["port_name"] == "b0_oof"));

    let validate = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-graph",
            output_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-graph");
    assert!(
        validate.status.success(),
        "validate-graph failed for compiled DSL graph: {}",
        String::from_utf8_lossy(&validate.stderr)
    );

    let artifact_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_compiled_dsl_artifact_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let compile_artifact = Command::new(cli())
        .current_dir(&root)
        .args([
            "compile-pipeline-dsl",
            "--dsl",
            "examples/pipeline_dsl_generation.json",
            "--artifact",
            "--output",
            artifact_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli compile-pipeline-dsl --artifact");
    assert!(
        compile_artifact.status.success(),
        "compile-pipeline-dsl --artifact failed: {}",
        String::from_utf8_lossy(&compile_artifact.stderr)
    );
    let artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&artifact_path).expect("compiled DSL artifact output was written"),
    )
    .expect("compiled DSL artifact output is JSON");
    assert_eq!(artifact["generation"]["strategy"], "cartesian");
    assert_eq!(artifact["generation"]["max_variants"], 4);
    assert_eq!(
        artifact["graph"]["search_space_fingerprint"],
        artifact["generation_fingerprint"]
    );

    let shape_artifact_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_compiled_dsl_shape_artifact_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let compile_shape_artifact = Command::new(cli())
        .current_dir(&root)
        .args([
            "compile-pipeline-dsl",
            "--dsl",
            "examples/pipeline_dsl_shape_plan.json",
            "--artifact",
            "--output",
            shape_artifact_path
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli compile-pipeline-dsl --artifact for shape plan");
    assert!(
        compile_shape_artifact.status.success(),
        "compile-pipeline-dsl --artifact shape plan failed: {}",
        String::from_utf8_lossy(&compile_shape_artifact.stderr)
    );
    let shape_artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&shape_artifact_path).expect("compiled DSL shape artifact was written"),
    )
    .expect("compiled DSL shape artifact is JSON");
    assert_eq!(
        shape_artifact["shape_plans"]["augment:synthetic"]["augmentation_policy"]["sample_scope"],
        "train_only"
    );
    assert_eq!(
        shape_artifact["shape_plans"]["transform:select"]["selection_policy"]["scope"],
        "supervised_fold_train"
    );

    let coordinated_artifact_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_compiled_dsl_coordinated_artifact_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let compile_coordinated_artifact = Command::new(cli())
        .current_dir(&root)
        .args([
            "compile-pipeline-dsl",
            "--dsl",
            "examples/pipeline_dsl_coordinated_generation.json",
            "--artifact",
            "--output",
            coordinated_artifact_path
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect(
            "failed to run dag-ml-cli compile-pipeline-dsl --artifact for coordinated generation",
        );
    assert!(
        compile_coordinated_artifact.status.success(),
        "compile-pipeline-dsl --artifact coordinated generation failed: {}",
        String::from_utf8_lossy(&compile_coordinated_artifact.stderr)
    );
    let coordinated_artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&coordinated_artifact_path)
            .expect("compiled DSL coordinated artifact was written"),
    )
    .expect("compiled DSL coordinated artifact is JSON");
    assert_eq!(
        coordinated_artifact["generation"]["dimensions"][0]["name"],
        "stack_profile"
    );
    assert_eq!(
        coordinated_artifact["generation"]["dimensions"][0]["choices"][1]["param_overrides"][2]
            ["node_id"],
        "merge:stack.pred_plus_original.meta:ridge"
    );
    assert_eq!(
        coordinated_artifact["campaign_template"]["split_invocation"]["id"],
        "split:group-kfold"
    );
    assert_eq!(
        coordinated_artifact["campaign_template"]["generation"],
        coordinated_artifact["generation"]
    );
    assert_eq!(
        coordinated_artifact["data_bindings"]["merge:stack.pred_plus_original.meta:ridge"][0]
            ["input_name"],
        "x_original"
    );
    assert_eq!(
        coordinated_artifact["campaign_template"]["data_bindings"],
        coordinated_artifact["data_bindings"]
    );

    let compat_artifact_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_compiled_dsl_compat_artifact_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let compile_compat_artifact = Command::new(cli())
        .current_dir(&root)
        .args([
            "compile-pipeline-dsl",
            "--dsl",
            "examples/pipeline_dsl_nirs4all_compat.json",
            "--artifact",
            "--output",
            compat_artifact_path
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli compile-pipeline-dsl --artifact for compat DSL");
    assert!(
        compile_compat_artifact.status.success(),
        "compile-pipeline-dsl --artifact compat DSL failed: {}",
        String::from_utf8_lossy(&compile_compat_artifact.stderr)
    );
    let compat_artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&compat_artifact_path).expect("compiled compat DSL artifact was written"),
    )
    .expect("compiled compat DSL artifact is JSON");
    assert_eq!(
        compat_artifact["campaign_template"]["split_invocation"]["params"]["kind"],
        "compat_split_chain"
    );
    assert_eq!(
        compat_artifact["campaign_template"]["split_invocation"]["params"]["compat_split_chain"][0]
            ["params"]["type"],
        "GroupKFold"
    );
    assert_eq!(
        compat_artifact["campaign_template"]["split_invocation"]["params"]["compat_split_chain"][1]
            ["params"]["class"],
        "sklearn.model_selection.KFold"
    );
    assert!(compat_artifact["graph"]["edges"]
        .as_array()
        .expect("edges array")
        .iter()
        .any(|edge| edge["target"]["node_id"] == "model:compat.meta"
            && edge["contract"]["requires_oof"] == true));

    let dsl_plan_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_dsl_execution_plan_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let build_dsl_plan = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-pipeline-dsl-plan",
            "--dsl",
            "examples/pipeline_dsl_coordinated_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--plan-id",
            "plan:cli.dsl.coordinated",
            "--output",
            dsl_plan_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli build-pipeline-dsl-plan");
    assert!(
        build_dsl_plan.status.success(),
        "build-pipeline-dsl-plan failed: {}",
        String::from_utf8_lossy(&build_dsl_plan.stderr)
    );
    let dsl_plan: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&dsl_plan_path).expect("compiled DSL execution plan was written"),
    )
    .expect("compiled DSL execution plan is JSON");
    assert_eq!(dsl_plan["id"], "plan:cli.dsl.coordinated");
    assert_eq!(dsl_plan["variants"].as_array().unwrap().len(), 2);
    assert_eq!(
        dsl_plan["campaign"]["data_bindings"]["merge:stack.pred_plus_original.meta:ridge"][0]
            ["input_name"],
        "x_original"
    );
}

#[test]
fn cli_builds_dsl_plan_with_registry_inferred_minimal_alias_kind() {
    let root = repo_root();
    let output_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_registry_alias_plan_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let artifact_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_registry_alias_artifact_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));

    let build_plan = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-pipeline-dsl-plan",
            "--dsl",
            "examples/pipeline_dsl_registry_inferred_alias.json",
            "--controllers",
            "examples/controller_manifests_alias_registry.json",
            "--plan-id",
            "plan:cli.dsl.registry.alias",
            "--output",
            output_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli build-pipeline-dsl-plan for registry alias");
    assert!(
        build_plan.status.success(),
        "build-pipeline-dsl-plan registry alias failed: {}",
        String::from_utf8_lossy(&build_plan.stderr)
    );

    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output_path).expect("plan was written"))
            .expect("plan is JSON");
    let nodes = plan["graph_plan"]["graph"]["nodes"]
        .as_array()
        .expect("graph nodes array");
    let model = nodes
        .iter()
        .find(|node| node["operator"].as_str() == Some("ElasticSpectra"))
        .expect("registry-inferred model node");

    assert_eq!(model["kind"], "model");
    assert_eq!(model["metadata"]["dsl_registry_inferred_kind"], "model");
    assert_eq!(
        model["metadata"]["dsl_compat_original_keyword"],
        "preprocessing"
    );

    let compile_artifact = Command::new(cli())
        .current_dir(&root)
        .args([
            "compile-pipeline-dsl",
            "--dsl",
            "examples/pipeline_dsl_registry_inferred_alias.json",
            "--controllers",
            "examples/controller_manifests_alias_registry.json",
            "--artifact",
            "--output",
            artifact_path.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli compile-pipeline-dsl for registry alias");
    assert!(
        compile_artifact.status.success(),
        "compile-pipeline-dsl registry alias failed: {}",
        String::from_utf8_lossy(&compile_artifact.stderr)
    );
    let artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&artifact_path).expect("artifact was written"))
            .expect("artifact is JSON");
    assert!(artifact["graph"]["nodes"]
        .as_array()
        .expect("artifact graph nodes")
        .iter()
        .any(|node| {
            node["operator"].as_str() == Some("ElasticSpectra") && node["kind"] == "model"
        }));
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
    let temp_dsl_branch_merge_cv_refit_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_dsl_branch_merge_cv_refit_bundle_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_dsl_branch_merge_lineage = std::env::temp_dir().join(format!(
        "dag_ml_cli_dsl_branch_merge_lineage_{}_{}.json",
        std::process::id(),
        unique_suffix()
    ));
    let temp_dsl_branch_merge_prediction_cache = std::env::temp_dir().join(format!(
        "dag_ml_cli_dsl_branch_merge_prediction_cache_{}_{}.json",
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

    let dsl_branch_merge_cv_refit_bundle = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-dsl-cv-refit-bundle",
            "--dsl",
            "examples/pipeline_dsl_branch_merge_executable.json",
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
            "bundle:cli.dsl.branch.merge.cv.refit",
            "--selections",
            "examples/fixtures/bundle/selection_decisions_branch_merge.json",
            "--output",
            temp_dsl_branch_merge_cv_refit_bundle
                .to_str()
                .expect("temp path is valid utf-8"),
            "--lineage-output",
            temp_dsl_branch_merge_lineage
                .to_str()
                .expect("temp path is valid utf-8"),
            "--prediction-cache-output",
            temp_dsl_branch_merge_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.dsl.branch.merge.cv.refit",
            "--run-id",
            "run:cli.dsl.branch.merge.cv.refit",
        ])
        .output()
        .expect("failed to run DSL branch/merge CV+refit process bundle");
    assert!(
        dsl_branch_merge_cv_refit_bundle.status.success(),
        "DSL branch/merge CV+refit process bundle failed: {}",
        String::from_utf8_lossy(&dsl_branch_merge_cv_refit_bundle.stderr)
    );
    let dsl_branch_merge_stdout = String::from_utf8_lossy(&dsl_branch_merge_cv_refit_bundle.stdout);
    assert!(
        dsl_branch_merge_stdout.contains("process DSL cv refit bundle run: 8 fit_cv result(s)")
            && dsl_branch_merge_stdout.contains("6 OOF prediction block(s)")
            && dsl_branch_merge_stdout.contains("4 refit result(s)")
            && dsl_branch_merge_stdout.contains("3 captured artifact handle(s)")
            && dsl_branch_merge_stdout.contains("2 prediction cache(s)")
            && dsl_branch_merge_stdout.contains("configured process worker(s)=2")
            && dsl_branch_merge_stdout.contains("observed process worker(s)=2"),
        "unexpected DSL branch/merge CV+refit process bundle output: {}",
        dsl_branch_merge_stdout
    );
    let dsl_branch_merge_bundle_json =
        std::fs::read_to_string(&temp_dsl_branch_merge_cv_refit_bundle)
            .expect("DSL branch/merge CV+refit bundle was written");
    assert!(
        dsl_branch_merge_bundle_json.contains("\"selected_variant_id\": \"variant:")
            && dsl_branch_merge_bundle_json.contains("\"node_id\": \"branch:b1.augment:noise\"")
            && dsl_branch_merge_bundle_json.contains("artifact:branch:b0.model:ridge:refit")
            && dsl_branch_merge_bundle_json.contains("artifact:branch:b1.model:rf:refit")
            && dsl_branch_merge_bundle_json
                .contains("artifact:merge:stack.pred_plus_original.meta:ridge:refit")
            && dsl_branch_merge_bundle_json.contains(
                "branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"
            )
            && dsl_branch_merge_bundle_json.contains(
                "branch:b1.model:rf.oof->merge:stack.pred_plus_original.meta:ridge.b1_oof"
            ),
        "unexpected DSL branch/merge CV+refit bundle JSON: {}",
        dsl_branch_merge_bundle_json
    );
    let dsl_branch_merge_lineage_json = std::fs::read_to_string(&temp_dsl_branch_merge_lineage)
        .expect("DSL branch/merge lineage records were written");
    assert!(
        dsl_branch_merge_lineage_json.contains("variant:a964828b1417c6e7")
            && dsl_branch_merge_lineage_json.contains("branch:b1.augment:noise")
            && dsl_branch_merge_lineage_json.contains("merge:stack.pred_plus_original.meta:ridge")
            && dsl_branch_merge_lineage_json.contains("input_lineage"),
        "unexpected DSL branch/merge lineage JSON: {}",
        dsl_branch_merge_lineage_json
    );
    let dsl_branch_merge_prediction_cache_json =
        std::fs::read_to_string(&temp_dsl_branch_merge_prediction_cache)
            .expect("DSL branch/merge prediction cache payload was written");
    assert!(
        dsl_branch_merge_prediction_cache_json
            .contains("\"bundle_id\": \"bundle:cli.dsl.branch.merge.cv.refit\"")
            && dsl_branch_merge_prediction_cache_json.contains("\"schema_version\": 1")
            && dsl_branch_merge_prediction_cache_json.contains("prediction-cache:branch:b0")
            && dsl_branch_merge_prediction_cache_json.contains("prediction-cache:branch:b1"),
        "unexpected DSL branch/merge prediction cache payload JSON: {}",
        dsl_branch_merge_prediction_cache_json
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

    let dsl_branch_merge_sklearn_cv_refit_replay = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-dsl-cv-refit-replay",
            "--dsl",
            "examples/pipeline_dsl_branch_merge_executable.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/sklearn_process_controller.py",
            "--process-workers",
            "2",
            "--bundle-id",
            "bundle:cli.dsl.branch.merge.sklearn.cv.refit.replay",
            "--selections",
            "examples/fixtures/bundle/selection_decisions_branch_merge.json",
            "--plan-id",
            "plan:cli.dsl.branch.merge.sklearn.cv.refit.replay",
            "--run-id",
            "run:cli.dsl.branch.merge.sklearn.cv.refit.replay",
        ])
        .output()
        .expect("failed to run DSL branch/merge sklearn CV+refit+replay");
    assert!(
        dsl_branch_merge_sklearn_cv_refit_replay.status.success(),
        "DSL branch/merge sklearn CV+refit+replay failed: {}",
        String::from_utf8_lossy(&dsl_branch_merge_sklearn_cv_refit_replay.stderr)
    );
    let dsl_branch_merge_sklearn_stdout =
        String::from_utf8_lossy(&dsl_branch_merge_sklearn_cv_refit_replay.stdout);
    assert!(
        dsl_branch_merge_sklearn_stdout
            .contains("process DSL cv refit replay run: 8 fit_cv result(s)")
            && dsl_branch_merge_sklearn_stdout.contains("6 OOF prediction block(s)")
            && dsl_branch_merge_sklearn_stdout.contains("4 refit result(s)")
            && dsl_branch_merge_sklearn_stdout.contains("4 replay result(s)")
            && dsl_branch_merge_sklearn_stdout.contains("3 replay prediction block(s)")
            && dsl_branch_merge_sklearn_stdout.contains("3 captured artifact handle(s)")
            && dsl_branch_merge_sklearn_stdout.contains("2 prediction cache(s)")
            && dsl_branch_merge_sklearn_stdout.contains("configured process worker(s)=2")
            && dsl_branch_merge_sklearn_stdout.contains("observed process worker(s)=2"),
        "unexpected DSL branch/merge sklearn CV+refit+replay output: {}",
        dsl_branch_merge_sklearn_stdout
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
    let _ = std::fs::remove_file(temp_dsl_branch_merge_cv_refit_bundle);
    let _ = std::fs::remove_file(temp_dsl_branch_merge_lineage);
    let _ = std::fs::remove_file(temp_dsl_branch_merge_prediction_cache);
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
    let temp_payload_source_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_artifact_payload_source_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let temp_payload_store_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_artifact_payload_store_{}_{}",
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
    let payload_bytes = b"dag-ml cli portable artifact payload\n";
    let payload_fingerprint = sha256_hex(payload_bytes);
    let payload_uri = format!("artifacts/{payload_fingerprint}.joblib");
    let mut bundle_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&temp_bundle).expect("bundle was written"))
            .expect("bundle JSON parses");
    bundle_json["refit_artifacts"][0]["artifact"]["uri"] = json!(payload_uri);
    bundle_json["refit_artifacts"][0]["artifact"]["content_fingerprint"] =
        json!(payload_fingerprint);
    bundle_json["refit_artifacts"][0]["artifact"]["size_bytes"] = json!(payload_bytes.len());
    std::fs::write(
        &temp_bundle,
        serde_json::to_vec_pretty(&bundle_json).expect("bundle JSON serializes"),
    )
    .expect("portable payload bundle was written");
    let payload_path = temp_payload_source_dir.join(
        bundle_json["refit_artifacts"][0]["artifact"]["uri"]
            .as_str()
            .expect("payload uri is string"),
    );
    std::fs::create_dir_all(payload_path.parent().expect("payload has parent"))
        .expect("payload source dir was created");
    std::fs::write(&payload_path, payload_bytes).expect("payload source was written");

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

    let export_payload_store = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-artifact-payload-store",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--source-dir",
            temp_payload_source_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--output-dir",
            temp_payload_store_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli export-artifact-payload-store");
    assert!(
        export_payload_store.status.success(),
        "export-artifact-payload-store failed: {}",
        String::from_utf8_lossy(&export_payload_store.stderr)
    );
    assert!(
        String::from_utf8_lossy(&export_payload_store.stdout)
            .contains("wrote artifact payload store: bundle=bundle:cli.demo, artifact(s)=1"),
        "unexpected export-artifact-payload-store output: {}",
        String::from_utf8_lossy(&export_payload_store.stdout)
    );
    assert!(
        temp_payload_store_dir
            .join(
                bundle_json["refit_artifacts"][0]["artifact"]["uri"]
                    .as_str()
                    .expect("payload uri is string")
            )
            .exists(),
        "artifact payload was not copied into {}",
        temp_payload_store_dir.display()
    );

    let validate_payload_store = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-artifact-payload-store",
            "--bundle",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--store-dir",
            temp_payload_store_dir
                .to_str()
                .expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to run dag-ml-cli validate-artifact-payload-store");
    assert!(
        validate_payload_store.status.success(),
        "validate-artifact-payload-store failed: {}",
        String::from_utf8_lossy(&validate_payload_store.stderr)
    );
    assert!(
        String::from_utf8_lossy(&validate_payload_store.stdout)
            .contains("valid artifact payload store: bundle=bundle:cli.demo, artifact(s)=1"),
        "unexpected validate-artifact-payload-store output: {}",
        String::from_utf8_lossy(&validate_payload_store.stdout)
    );

    let replay_with_payload_store = Command::new(cli())
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
            "--artifact-payload-store",
            temp_payload_store_dir
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            "plan:cli.bundle",
            "--run-id",
            "run:cli.artifact.payload.replay",
        ])
        .output()
        .expect("failed to run dag-ml-cli run-mock-replay with artifact payload store");
    assert!(
        replay_with_payload_store.status.success(),
        "run-mock-replay with artifact payload store failed: {}",
        String::from_utf8_lossy(&replay_with_payload_store.stderr)
    );
    assert!(
        String::from_utf8_lossy(&replay_with_payload_store.stdout).contains("1 artifact handle(s)"),
        "unexpected run-mock-replay with artifact payload store output: {}",
        String::from_utf8_lossy(&replay_with_payload_store.stdout)
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
    let _ = std::fs::remove_dir_all(temp_payload_source_dir);
    let _ = std::fs::remove_dir_all(temp_payload_store_dir);
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
        "examples/adapters/sklearn_production_controller.py",
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
        let description: serde_json::Value =
            serde_json::from_slice(&describe.stdout).expect("adapter description JSON");
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
        if adapter == "examples/adapters/python_process_controller.py" {
            let fixture: serde_json::Value = serde_json::from_str(include_str!(
                "../../../examples/fixtures/runtime/process_adapter_description_python.json"
            ))
            .unwrap();
            assert_eq!(description, fixture);
        }
        if adapter == "examples/adapters/sklearn_production_controller.py" {
            assert!(
                stdout.contains("\"dag-ml-sklearn-production-controller\"")
                    && stdout.contains("\"sklearn_production\""),
                "production adapter description missing production capability: {}",
                stdout
            );
        }
    }
}

#[test]
fn prospectr_process_controller_describes_supported_protocol_modes() {
    let root = repo_root();
    if !r_has_prospectr(&root) {
        return;
    }
    let describe = Command::new("Rscript")
        .current_dir(&root)
        .args([
            "examples/adapters/prospectr_process_controller.R",
            "--describe",
        ])
        .output()
        .expect("run prospectr adapter describe handshake");
    assert!(
        describe.status.success(),
        "prospectr adapter describe failed: {}",
        String::from_utf8_lossy(&describe.stderr)
    );
    let stdout = String::from_utf8_lossy(&describe.stdout);
    let description: serde_json::Value =
        serde_json::from_slice(&describe.stdout).expect("prospectr adapter description JSON");
    assert_eq!(
        description["adapter_id"].as_str(),
        Some("dag-ml-prospectr-process-controller")
    );
    assert_eq!(
        description["protocol"].as_str(),
        Some("dag-ml-process-adapter")
    );
    assert_eq!(description["schema_version"].as_u64(), Some(1));
    for required in [
        "control_frames_v1",
        "node_task_json_v1",
        "node_result_json_v1",
        "parallel_invocation_v1",
        "persistent_workers",
        "worker_env",
        "prospectr_smoke",
    ] {
        assert!(
            stdout.contains(&format!("\"{required}\"")),
            "prospectr adapter description missing capability `{required}`: {}",
            stdout
        );
    }
}

#[test]
fn prospectr_process_controller_runs_snv_transform_one_shot() {
    let root = repo_root();
    if !r_has_prospectr(&root) {
        return;
    }

    let node_plan = json!({
        "node_id": "snv:0",
        "kind": "transform",
        "controller_id": "controller:prospectr",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "stateless",
        "rng_policy": "ignores_seed",
        "artifact_policy": "replay_required",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "SNV"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let task = json!({
        "run_id": "run:cli.prospectr",
        "node_plan": node_plan,
        "phase": "FIT_CV",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {},
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {},
        "seed": 7
    });
    let output = run_r_adapter_one_shot(&root, &task);
    let value: serde_json::Value =
        serde_json::from_slice(&output).expect("R adapter one_shot output is JSON");
    assert_eq!(value["node_id"].as_str(), Some("snv:0"));
    assert!(value["outputs"]["x_out"]["handle"].is_u64());
    assert_eq!(value["outputs"]["x_out"]["kind"].as_str(), Some("data"));
    assert_eq!(value["predictions"].as_array().map(|v| v.len()), Some(0));
    assert_eq!(
        value["lineage"]["metrics"]["transform_columns"].as_u64(),
        Some(4),
        "SNV transform should preserve the 4-column synthetic feature matrix"
    );
    assert_eq!(
        value["lineage"]["metrics"]["transform_rows"].as_u64(),
        Some(4),
        "SNV transform should preserve the 4-row default sample count"
    );
    assert_eq!(
        value["lineage"]["metrics"]["prospectr_adapter"].as_f64(),
        Some(1.0)
    );
}

#[test]
fn prospectr_process_controller_one_shot_unknown_operator_exits_nonzero() {
    let root = repo_root();
    if !r_has_prospectr(&root) {
        return;
    }
    let node_plan = json!({
        "node_id": "transform:bad",
        "kind": "transform",
        "controller_id": "controller:prospectr",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "stateless",
        "rng_policy": "ignores_seed",
        "artifact_policy": "replay_required",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "DefinitelyNotAProspectrFunction"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let task = json!({
        "run_id": "run:cli.prospectr-bad",
        "node_plan": node_plan,
        "phase": "FIT_CV",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {},
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {},
        "seed": 7
    });
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("Rscript")
        .current_dir(&root)
        .args(["examples/adapters/prospectr_process_controller.R"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn prospectr adapter");
    {
        let stdin = child.stdin.as_mut().expect("R adapter stdin is piped");
        stdin
            .write_all(
                serde_json::to_vec(&task)
                    .expect("task JSON serializes")
                    .as_slice(),
            )
            .expect("write task JSON to R adapter stdin");
        stdin.write_all(b"\n").expect("write trailing newline");
    }
    let output = child.wait_with_output().expect("wait for R adapter");
    assert!(
        !output.status.success(),
        "prospectr one_shot accepted an unknown operator instead of failing"
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr_text.contains("unknown_operator"),
        "expected `unknown_operator` code in stderr: {}",
        stderr_text
    );
}

#[test]
fn prospectr_process_controller_jsonl_emits_error_frame_and_survives() {
    let root = repo_root();
    if !r_has_prospectr(&root) {
        return;
    }

    let bad_node_plan = json!({
        "node_id": "snv:bad",
        "kind": "transform",
        "controller_id": "controller:prospectr",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "stateless",
        "rng_policy": "ignores_seed",
        "artifact_policy": "replay_required",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "DefinitelyNotAProspectrFunction"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let good_node_plan = json!({
        "node_id": "snv:ok",
        "kind": "transform",
        "controller_id": "controller:prospectr",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "stateless",
        "rng_policy": "ignores_seed",
        "artifact_policy": "replay_required",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "SNV"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let variant = json!({
        "variant_id": "variant:base",
        "choices": {},
        "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "seed": 7
    });
    let bad_task = json!({
        "type": "task",
        "schema_version": 1,
        "task": {
            "run_id": "run:cli.prospectr-jsonl",
            "node_plan": bad_node_plan,
            "phase": "FIT_CV",
            "variant_id": "variant:base",
            "variant": variant,
            "fold_id": null,
            "branch_path": [],
            "input_handles": {},
            "data_views": {},
            "prediction_inputs": {},
            "artifact_inputs": {},
            "seed": 7
        }
    });
    let good_task = json!({
        "type": "task",
        "schema_version": 1,
        "task": {
            "run_id": "run:cli.prospectr-jsonl",
            "node_plan": good_node_plan,
            "phase": "FIT_CV",
            "variant_id": "variant:base",
            "variant": variant,
            "fold_id": null,
            "branch_path": [],
            "input_handles": {},
            "data_views": {},
            "prediction_inputs": {},
            "artifact_inputs": {},
            "seed": 7
        }
    });
    let close = json!({"type": "close", "schema_version": 1});

    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("Rscript")
        .current_dir(&root)
        .args([
            "examples/adapters/prospectr_process_controller.R",
            "--jsonl",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn prospectr adapter");
    {
        let stdin = child.stdin.as_mut().expect("R adapter stdin is piped");
        for frame in [&bad_task, &good_task, &close] {
            stdin
                .write_all(
                    serde_json::to_vec(frame)
                        .expect("frame JSON serializes")
                        .as_slice(),
                )
                .expect("write frame to R adapter stdin");
            stdin.write_all(b"\n").expect("write trailing newline");
        }
    }
    let output = child.wait_with_output().expect("wait for R adapter");
    assert!(
        output.status.success(),
        "JSONL R adapter exited with failure after a bad-task error: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout_text = String::from_utf8(output.stdout).expect("R adapter stdout is utf-8");
    let mut frames: Vec<serde_json::Value> = Vec::new();
    for line in stdout_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        frames.push(serde_json::from_str(line).expect("each R adapter frame is JSON"));
    }
    let error_frames: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"].as_str() == Some("error"))
        .collect();
    let result_frames: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"].as_str() == Some("result"))
        .collect();
    let ack_frames: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"].as_str() == Some("ack"))
        .collect();
    assert_eq!(
        frames.len(),
        3,
        "expected exactly 3 frames (error + result + close ack), got: {:#?}",
        frames
    );
    assert_eq!(error_frames.len(), 1, "expected one error frame");
    assert_eq!(
        error_frames[0]["error"]["code"].as_str(),
        Some("unknown_operator")
    );
    assert_eq!(result_frames.len(), 1, "expected one result frame");
    assert_eq!(
        result_frames[0]["result"]["node_id"].as_str(),
        Some("snv:ok")
    );
    assert_eq!(ack_frames.len(), 1, "expected one close ack");
    assert_eq!(ack_frames[0]["status"].as_str(), Some("closed"));
}

#[test]
fn prospectr_process_controller_runs_savitzky_golay_one_shot() {
    let root = repo_root();
    if !r_has_prospectr(&root) {
        return;
    }
    let node_plan = json!({
        "node_id": "sg:0",
        "kind": "transform",
        "controller_id": "controller:prospectr",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "stateless",
        "rng_policy": "ignores_seed",
        "artifact_policy": "replay_required",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "savitzkyGolay", "params": {"m": 0, "p": 1, "w": 3}},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let task = json!({
        "run_id": "run:cli.prospectr-sg",
        "node_plan": node_plan,
        "phase": "FIT_CV",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {},
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {},
        "seed": 7
    });
    let output = run_r_adapter_one_shot(&root, &task);
    let value: serde_json::Value =
        serde_json::from_slice(&output).expect("R adapter savitzkyGolay output is JSON");
    // savitzkyGolay with w=3 drops the column count by (w-1)=2; the
    // 4-column synthetic feature matrix becomes 4 rows x 2 cols.
    assert_eq!(
        value["lineage"]["metrics"]["transform_rows"].as_u64(),
        Some(4),
        "savitzkyGolay should preserve the 4-row sample count"
    );
    assert_eq!(
        value["lineage"]["metrics"]["transform_columns"].as_u64(),
        Some(2),
        "savitzkyGolay with w=3 should drop the column count from 4 to 2"
    );
}

#[test]
fn prospectr_process_controller_manifest_validates_and_matches_registry() {
    let root = repo_root();
    let manifest_path = root.join("examples/controllers/prospectr.controller.json");
    let manifest_text =
        std::fs::read_to_string(&manifest_path).expect("prospectr controller manifest is readable");
    let manifest: dag_ml_core::controller::ControllerManifest =
        serde_json::from_str(&manifest_text).expect("manifest deserializes as ControllerManifest");
    manifest
        .validate()
        .expect("prospectr controller manifest passes ControllerManifest::validate");
    assert_eq!(manifest.controller_id.as_str(), "controller:prospectr");
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::FitCv));
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::Refit));
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::Predict));

    let manifest_value: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("manifest parses as JSON value");
    let manifest_aliases: std::collections::BTreeSet<String> = manifest_value["operator_selectors"]
        .as_array()
        .expect("operator_selectors is an array")
        .iter()
        .find_map(|selector| selector.get("aliases"))
        .and_then(|aliases| aliases.as_array())
        .expect("operator_selectors contains an aliases selector")
        .iter()
        .map(|alias| alias.as_str().expect("alias is string").to_string())
        .collect();

    // Static manifest validation passes without R. The registry-parity
    // probe below sources the R controller and therefore requires
    // R + jsonlite + prospectr.
    if !r_has_prospectr(&root) {
        return;
    }
    let probe_script = "e <- new.env()\n\
# Source under a fresh env. The controller's source guard\n\
# `if (sys.nframe() == 0L) main()` keeps main() from firing because\n\
# `sys.nframe()` is > 0 inside `source()`. The startup messages from\n\
# the controller's own `library(jsonlite)` / `library(prospectr)`\n\
# calls are suppressed so they cannot mix with the JSON output line\n\
# we parse.\n\
suppressPackageStartupMessages(\n\
  source('examples/adapters/prospectr_process_controller.R', local = e)\n\
)\n\
cat(jsonlite::toJSON(sort(names(e$OPERATOR_SELECTORS))), '\\n')\n";
    let suffix = unique_suffix();
    let probe_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_prospectr_manifest_probe_{}_{}.R",
        std::process::id(),
        suffix
    ));
    std::fs::write(&probe_path, probe_script).expect("write parity probe script");
    let probe = Command::new("Rscript")
        .current_dir(&root)
        .arg(&probe_path)
        .output()
        .expect("run R parity probe");
    let _ = std::fs::remove_file(&probe_path);
    assert!(
        probe.status.success(),
        "R parity probe failed: stdout=`{}` stderr=`{}`",
        String::from_utf8_lossy(&probe.stdout),
        String::from_utf8_lossy(&probe.stderr)
    );
    let stdout_text = String::from_utf8_lossy(&probe.stdout);
    let first_line = stdout_text
        .lines()
        .find(|line| line.trim().starts_with('['))
        .expect("R probe stdout contains a JSON array line");
    let registry_aliases: std::collections::BTreeSet<String> =
        serde_json::from_str(first_line).expect("registry probe line is a JSON list");
    assert_eq!(
        manifest_aliases, registry_aliases,
        "manifest aliases drift from R's OPERATOR_SELECTORS: manifest_only={:?}, registry_only={:?}",
        manifest_aliases.difference(&registry_aliases).collect::<Vec<_>>(),
        registry_aliases.difference(&manifest_aliases).collect::<Vec<_>>()
    );
}

fn run_r_adapter_one_shot(root: &Path, task: &serde_json::Value) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("Rscript")
        .current_dir(root)
        .args(["examples/adapters/prospectr_process_controller.R"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn prospectr adapter");
    {
        let stdin = child.stdin.as_mut().expect("R adapter stdin is piped");
        stdin
            .write_all(
                serde_json::to_vec(task)
                    .expect("task JSON serializes")
                    .as_slice(),
            )
            .expect("write task JSON to R adapter stdin");
        stdin.write_all(b"\n").expect("write trailing newline");
    }
    let output = child.wait_with_output().expect("wait for R adapter");
    assert!(
        output.status.success(),
        "prospectr R adapter exited with failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn mdatools_process_controller_describes_supported_protocol_modes() {
    let root = repo_root();
    if !r_has_mdatools(&root) {
        return;
    }
    let describe = Command::new("Rscript")
        .current_dir(&root)
        .args([
            "examples/adapters/mdatools_process_controller.R",
            "--describe",
        ])
        .output()
        .expect("run mdatools adapter describe handshake");
    assert!(
        describe.status.success(),
        "mdatools adapter describe failed: {}",
        String::from_utf8_lossy(&describe.stderr)
    );
    let description: serde_json::Value =
        serde_json::from_slice(&describe.stdout).expect("mdatools adapter description JSON");
    assert_eq!(
        description["adapter_id"].as_str(),
        Some("dag-ml-mdatools-process-controller")
    );
    let stdout = String::from_utf8_lossy(&describe.stdout);
    for required in [
        "control_frames_v1",
        "node_task_json_v1",
        "node_result_json_v1",
        "parallel_invocation_v1",
        "persistent_workers",
        "worker_env",
        "stateful_refit_artifacts",
        "mdatools_smoke",
    ] {
        assert!(
            stdout.contains(&format!("\"{required}\"")),
            "mdatools adapter description missing capability `{required}`: {}",
            stdout
        );
    }
}

#[test]
fn mdatools_process_controller_refits_then_predicts_via_rds() {
    let root = repo_root();
    if !r_has_mdatools(&root) {
        return;
    }

    let suffix = unique_suffix();
    let artifact_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_mdatools_artifacts_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

    let node_plan = json!({
        "node_id": "pls:0",
        "kind": "model",
        "controller_id": "controller:mdatools",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        // ncomp=1 is the smallest valid PLS model on the 4x4 synthetic
        // feature matrix.
        "params": {"operator": "pls", "params": {"ncomp": 1}},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let variant = json!({
        "variant_id": "variant:base",
        "choices": {},
        "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "seed": 7
    });

    let refit_task = json!({
        "run_id": "run:cli.mdatools",
        "node_plan": node_plan,
        "phase": "REFIT",
        "variant_id": "variant:base",
        "variant": variant,
        "fold_id": null,
        "branch_path": [],
        "input_handles": {},
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {},
        "seed": 7
    });
    let refit_output = run_mdatools_adapter_one_shot(&root, &artifact_dir, &refit_task);
    let refit_value: serde_json::Value =
        serde_json::from_slice(&refit_output).expect("REFIT result is JSON");
    let artifacts = refit_value["artifacts"]
        .as_array()
        .expect("REFIT result has artifacts");
    assert_eq!(
        artifacts.len(),
        1,
        "REFIT produced unexpected artifact count: {refit_value:#}"
    );
    let artifact = &artifacts[0];
    assert_eq!(artifact["backend"].as_str(), Some("rds"));
    assert_eq!(artifact["kind"].as_str(), Some("mdatools_model"));
    let artifact_id = artifact["id"].as_str().expect("artifact id").to_string();
    let artifact_uri = artifact["uri"].as_str().expect("artifact uri").to_string();
    assert!(
        artifact["size_bytes"].as_u64().is_some_and(|n| n > 0),
        "REFIT artifact size_bytes must be positive"
    );
    let artifact_path = if std::path::Path::new(&artifact_uri).is_absolute() {
        std::path::PathBuf::from(&artifact_uri)
    } else {
        artifact_dir.join(
            std::path::Path::new(&artifact_uri)
                .file_name()
                .expect("artifact uri has file name"),
        )
    };
    assert!(
        artifact_path.exists(),
        "RDS artifact was not written to disk: {}",
        artifact_path.display()
    );
    let artifact_handle = refit_value["artifact_handles"][&artifact_id]["handle"]
        .as_u64()
        .expect("artifact_handles carry numeric handle");

    // Drive PREDICT against the same sample IDs REFIT trained on so
    // we can prove RDS round-trip determinism. data_views with
    // partition="predict" carries the IDs; node_plan.data_bindings
    // names "x" so the controller's `data_view(task)` resolves the
    // view by key.
    let refit_sample_ids = refit_value["predictions"][0]["sample_ids"]
        .as_array()
        .expect("REFIT predictions carry sample_ids")
        .iter()
        .map(|v| v.as_str().expect("sample id is str").to_string())
        .collect::<Vec<_>>();
    let mut predict_node_plan = node_plan.clone();
    predict_node_plan["data_bindings"] = json!([{"input_name": "x"}]);
    let predict_task = json!({
        "run_id": "run:cli.mdatools",
        "node_plan": predict_node_plan,
        "phase": "PREDICT",
        "variant_id": "variant:base",
        "variant": variant,
        "fold_id": null,
        "branch_path": [],
        "input_handles": {
            artifact_id.clone(): {
                "handle": artifact_handle,
                "kind": "model",
                "owner_controller": "controller:mdatools"
            },
            "data:x": {
                "handle": 1,
                "kind": "data",
                "owner_controller": "controller:mdatools"
            }
        },
        "data_views": {
            "data:x": {
                "partition": "predict",
                "sample_ids": refit_sample_ids
            }
        },
        "prediction_inputs": {},
        "artifact_inputs": {
            artifact_id.clone(): {
                "node_id": "pls:0",
                "controller_id": "controller:mdatools",
                "uri": artifact_uri
            }
        },
        "seed": 7
    });

    let predict_output = run_mdatools_adapter_one_shot(&root, &artifact_dir, &predict_task);
    let predict_value: serde_json::Value =
        serde_json::from_slice(&predict_output).expect("PREDICT result is JSON");
    let predictions = predict_value["predictions"]
        .as_array()
        .expect("PREDICT result has predictions");
    assert_eq!(predictions.len(), 1);
    let predict_values = predictions[0]["values"]
        .as_array()
        .expect("PREDICT values is an array");
    assert!(
        !predict_values.is_empty(),
        "PREDICT returned empty value rows"
    );
    for row in predict_values {
        let row = row.as_array().expect("prediction row is an array");
        assert_eq!(row.len(), 1, "pls produces 1 target per sample");
        assert!(
            row[0].as_f64().is_some_and(f64::is_finite),
            "PREDICT row must be finite: {row:?}"
        );
    }

    // Round-trip determinism: both phases now predict on the SAME
    // sample IDs through the SAME synthetic feature pipeline. The
    // RDS artifact must round-trip the fitted PLS model exactly.
    let refit_values = refit_value["predictions"]
        .as_array()
        .and_then(|blocks| blocks.first())
        .and_then(|block| block["values"].as_array())
        .expect("REFIT predictions exist");
    assert_eq!(
        refit_values.len(),
        predict_values.len(),
        "REFIT and PREDICT row counts must match after sample alignment"
    );
    for (refit_row, predict_row) in refit_values.iter().zip(predict_values.iter()) {
        let refit_v = refit_row[0]
            .as_f64()
            .expect("REFIT row value is finite f64");
        let predict_v = predict_row[0]
            .as_f64()
            .expect("PREDICT row value is finite f64");
        assert!(
            (refit_v - predict_v).abs() < 1e-9,
            "REFIT vs PREDICT value drift: refit={refit_v}, predict={predict_v}"
        );
    }

    let _ = std::fs::remove_dir_all(artifact_dir);
}

#[test]
fn mdatools_process_controller_rejects_artifact_uri_outside_artifact_dir() {
    let root = repo_root();
    if !r_has_mdatools(&root) {
        return;
    }
    let suffix = unique_suffix();
    let artifact_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_mdatools_traversal_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

    let node_plan = json!({
        "node_id": "pls:0",
        "kind": "model",
        "controller_id": "controller:mdatools",
        "controller_version": "1.0.0",
        "supported_phases": ["PREDICT"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "pls", "params": {"ncomp": 1}},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let artifact_id = "artifact:pls:0:mdatools:refit";
    let predict_task = json!({
        "run_id": "run:cli.mdatools-traversal",
        "node_plan": node_plan,
        "phase": "PREDICT",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {
            artifact_id: {
                "handle": 1,
                "kind": "model",
                "owner_controller": "controller:mdatools"
            }
        },
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {
            artifact_id: {
                "node_id": "pls:0",
                "controller_id": "controller:mdatools",
                "uri": "/etc/passwd"
            }
        },
        "seed": 7
    });

    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("Rscript")
        .current_dir(&root)
        .args(["examples/adapters/mdatools_process_controller.R"])
        .env(
            "DAG_ML_PROCESS_ARTIFACT_DIR",
            artifact_dir
                .to_str()
                .expect("artifact dir path is valid utf-8"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mdatools adapter");
    {
        let stdin = child.stdin.as_mut().expect("adapter stdin is piped");
        stdin
            .write_all(
                serde_json::to_vec(&predict_task)
                    .expect("task JSON serializes")
                    .as_slice(),
            )
            .expect("write task JSON to adapter stdin");
        stdin.write_all(b"\n").expect("write trailing newline");
    }
    let output = child.wait_with_output().expect("wait for mdatools adapter");
    assert!(
        !output.status.success(),
        "mdatools adapter accepted a traversal URI instead of rejecting it"
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    // The basename of `/etc/passwd` resolves to `passwd`, which is
    // joined under the artifact dir; that path does not exist, so
    // the error message references resolution under the artifact dir
    // (proving the basename strip worked and the controller did not
    // touch /etc/passwd).
    let artifact_dir_str = artifact_dir
        .to_str()
        .expect("artifact dir path is valid utf-8");
    assert!(
        stderr_text.contains("resolved under artifact dir")
            && stderr_text.contains(artifact_dir_str)
            && !stderr_text.contains("/etc/passwd readRDS"),
        "mdatools adapter did not confine the traversal URI under artifact dir: {}",
        stderr_text
    );

    let _ = std::fs::remove_dir_all(artifact_dir);
}

#[test]
fn mdatools_process_controller_runs_pca_one_shot() {
    let root = repo_root();
    if !r_has_mdatools(&root) {
        return;
    }

    let suffix = unique_suffix();
    let artifact_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_mdatools_pca_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

    let node_plan = json!({
        "node_id": "pca:0",
        "kind": "model",
        "controller_id": "controller:mdatools",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        // pca uses the unsupervised dispatch shape — no `y` argument.
        // ncomp=1 returns the first principal component score as the
        // single per-sample prediction.
        "params": {"operator": "pca", "params": {"ncomp": 1}},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let task = json!({
        "run_id": "run:cli.mdatools-pca",
        "node_plan": node_plan,
        "phase": "REFIT",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {},
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {},
        "seed": 7
    });
    let output = run_mdatools_adapter_one_shot(&root, &artifact_dir, &task);
    let value: serde_json::Value =
        serde_json::from_slice(&output).expect("pca REFIT output is JSON");
    let predictions = value["predictions"]
        .as_array()
        .expect("pca REFIT result has predictions");
    assert_eq!(predictions.len(), 1);
    let prediction_values = predictions[0]["values"]
        .as_array()
        .expect("pca predictions[0].values is an array");
    assert_eq!(prediction_values.len(), 4, "REFIT default produces 4 rows");
    for row in prediction_values {
        let row = row.as_array().expect("row is an array");
        assert_eq!(row.len(), 1, "pca produces one score per sample");
        assert!(
            row[0].as_f64().is_some_and(f64::is_finite),
            "pca prediction must be finite: {row:?}"
        );
    }
    // pca emits a refit artifact like pls does.
    let artifacts = value["artifacts"]
        .as_array()
        .expect("REFIT artifacts array");
    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0]["backend"].as_str(), Some("rds"));

    // Round-trip PCA model through RDS to exercise the `pcares`
    // branch in `extract_prediction_vector` end-to-end. The PREDICT
    // task targets the same sample IDs as REFIT so the first-PC
    // scores must match exactly.
    let artifact = &artifacts[0];
    let artifact_id = artifact["id"].as_str().expect("artifact id").to_string();
    let artifact_uri = artifact["uri"].as_str().expect("artifact uri").to_string();
    let artifact_handle = value["artifact_handles"][&artifact_id]["handle"]
        .as_u64()
        .expect("artifact handle is u64");
    let refit_sample_ids = predictions[0]["sample_ids"]
        .as_array()
        .expect("REFIT prediction sample_ids")
        .iter()
        .map(|v| v.as_str().expect("sample id is str").to_string())
        .collect::<Vec<_>>();
    let mut predict_node_plan = node_plan.clone();
    predict_node_plan["data_bindings"] = json!([{"input_name": "x"}]);
    let predict_task = json!({
        "run_id": "run:cli.mdatools-pca",
        "node_plan": predict_node_plan,
        "phase": "PREDICT",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {
            artifact_id.clone(): {
                "handle": artifact_handle,
                "kind": "model",
                "owner_controller": "controller:mdatools"
            },
            "data:x": {
                "handle": 1,
                "kind": "data",
                "owner_controller": "controller:mdatools"
            }
        },
        "data_views": {
            "data:x": {
                "partition": "predict",
                "sample_ids": refit_sample_ids
            }
        },
        "prediction_inputs": {},
        "artifact_inputs": {
            artifact_id.clone(): {
                "node_id": "pca:0",
                "controller_id": "controller:mdatools",
                "uri": artifact_uri
            }
        },
        "seed": 7
    });
    let predict_output = run_mdatools_adapter_one_shot(&root, &artifact_dir, &predict_task);
    let predict_value: serde_json::Value =
        serde_json::from_slice(&predict_output).expect("pca PREDICT output is JSON");
    let predict_values = predict_value["predictions"][0]["values"]
        .as_array()
        .expect("pca PREDICT values is an array");
    assert_eq!(predict_values.len(), prediction_values.len());
    for (refit_row, predict_row) in prediction_values.iter().zip(predict_values.iter()) {
        let refit_v = refit_row[0].as_f64().expect("REFIT score is f64");
        let predict_v = predict_row[0].as_f64().expect("PREDICT score is f64");
        assert!(
            (refit_v - predict_v).abs() < 1e-9,
            "pca REFIT vs PREDICT score drift: refit={refit_v}, predict={predict_v}"
        );
    }

    let _ = std::fs::remove_dir_all(artifact_dir);
}

#[test]
fn mdatools_process_controller_manifest_validates_and_matches_registry() {
    let root = repo_root();
    let manifest_path = root.join("examples/controllers/mdatools.controller.json");
    let manifest_text =
        std::fs::read_to_string(&manifest_path).expect("mdatools controller manifest is readable");
    let manifest: dag_ml_core::controller::ControllerManifest =
        serde_json::from_str(&manifest_text).expect("manifest deserializes as ControllerManifest");
    manifest
        .validate()
        .expect("mdatools controller manifest passes ControllerManifest::validate");
    assert_eq!(manifest.controller_id.as_str(), "controller:mdatools");
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::FitCv));
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::Refit));
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::Predict));

    let manifest_value: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("manifest parses as JSON value");
    let manifest_aliases: std::collections::BTreeSet<String> = manifest_value["operator_selectors"]
        .as_array()
        .expect("operator_selectors is an array")
        .iter()
        .find_map(|selector| selector.get("aliases"))
        .and_then(|aliases| aliases.as_array())
        .expect("operator_selectors contains an aliases selector")
        .iter()
        .map(|alias| alias.as_str().expect("alias is string").to_string())
        .collect();

    if !r_has_mdatools(&root) {
        return;
    }
    let probe_script = "e <- new.env()\n\
suppressPackageStartupMessages(\n\
  source('examples/adapters/mdatools_process_controller.R', local = e)\n\
)\n\
cat(jsonlite::toJSON(sort(names(e$OPERATOR_SELECTORS))), '\\n')\n";
    let suffix = unique_suffix();
    let probe_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_mdatools_manifest_probe_{}_{}.R",
        std::process::id(),
        suffix
    ));
    std::fs::write(&probe_path, probe_script).expect("write parity probe script");
    let probe = Command::new("Rscript")
        .current_dir(&root)
        .arg(&probe_path)
        .output()
        .expect("run R parity probe");
    let _ = std::fs::remove_file(&probe_path);
    assert!(
        probe.status.success(),
        "R parity probe failed: stdout=`{}` stderr=`{}`",
        String::from_utf8_lossy(&probe.stdout),
        String::from_utf8_lossy(&probe.stderr)
    );
    let stdout_text = String::from_utf8_lossy(&probe.stdout);
    let first_line = stdout_text
        .lines()
        .find(|line| line.trim().starts_with('['))
        .expect("R probe stdout contains a JSON array line");
    let registry_aliases: std::collections::BTreeSet<String> =
        serde_json::from_str(first_line).expect("registry probe line is a JSON list");
    assert_eq!(
        manifest_aliases, registry_aliases,
        "manifest aliases drift from R's OPERATOR_SELECTORS: manifest_only={:?}, registry_only={:?}",
        manifest_aliases.difference(&registry_aliases).collect::<Vec<_>>(),
        registry_aliases.difference(&manifest_aliases).collect::<Vec<_>>()
    );
}

fn run_mdatools_adapter_one_shot(
    root: &Path,
    artifact_dir: &Path,
    task: &serde_json::Value,
) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("Rscript")
        .current_dir(root)
        .args(["examples/adapters/mdatools_process_controller.R"])
        .env(
            "DAG_ML_PROCESS_ARTIFACT_DIR",
            artifact_dir
                .to_str()
                .expect("artifact dir path is valid utf-8"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mdatools adapter");
    {
        let stdin = child.stdin.as_mut().expect("R adapter stdin is piped");
        stdin
            .write_all(
                serde_json::to_vec(task)
                    .expect("task JSON serializes")
                    .as_slice(),
            )
            .expect("write task JSON to R adapter stdin");
        stdin.write_all(b"\n").expect("write trailing newline");
    }
    let output = child.wait_with_output().expect("wait for mdatools adapter");
    assert!(
        output.status.success(),
        "mdatools R adapter exited with failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn sklearn_production_controller_refits_then_predicts_via_joblib() {
    let root = repo_root();
    if !python_has_sklearn(&root) {
        return;
    }
    let joblib_available = Command::new("python3")
        .current_dir(&root)
        .args(["-c", "import joblib"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !joblib_available {
        return;
    }

    let suffix = unique_suffix();
    let artifact_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_sklearn_production_artifacts_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

    let node_plan = json!({
        "node_id": "model:ridge",
        "kind": "model",
        "controller_id": "controller:sklearn.production",
        "controller_version": "1.0.0",
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "controller_capabilities": ["deterministic", "thread_safe"],
        "fit_scope": "full_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {
            "operator": "Ridge",
            "params": {"alpha": 0.5}
        },
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });

    let refit_task = json!({
        "run_id": "run:cli.sklearn-production",
        "node_plan": node_plan,
        "phase": "REFIT",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {},
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {},
        "seed": 7
    });

    let refit_output = run_production_adapter_one_shot(&root, &artifact_dir, &refit_task);
    let refit_value: serde_json::Value =
        serde_json::from_slice(&refit_output).expect("REFIT result is JSON");
    let artifacts = refit_value["artifacts"]
        .as_array()
        .expect("REFIT result has artifacts");
    assert_eq!(
        artifacts.len(),
        1,
        "REFIT produced unexpected artifact count: {refit_value:#}"
    );
    let artifact = &artifacts[0];
    assert_eq!(artifact["backend"].as_str(), Some("joblib"));
    assert_eq!(artifact["kind"].as_str(), Some("sklearn_pipeline"));
    let artifact_id = artifact["id"].as_str().expect("artifact id").to_string();
    let artifact_uri = artifact["uri"].as_str().expect("artifact uri").to_string();
    let artifact_size = artifact["size_bytes"]
        .as_u64()
        .expect("artifact size_bytes");
    assert!(artifact_size > 0, "joblib artifact reported zero bytes");
    let artifact_path = if std::path::Path::new(&artifact_uri).is_absolute() {
        std::path::PathBuf::from(&artifact_uri)
    } else {
        artifact_dir.join(
            std::path::Path::new(&artifact_uri)
                .file_name()
                .expect("artifact uri has file name"),
        )
    };
    assert!(
        artifact_path.exists(),
        "joblib artifact was not written to disk: {}",
        artifact_path.display()
    );
    let artifact_handle = refit_value["artifact_handles"][&artifact_id]["handle"]
        .as_u64()
        .expect("artifact_handles carry numeric handle");

    let predict_task = json!({
        "run_id": "run:cli.sklearn-production",
        "node_plan": node_plan,
        "phase": "PREDICT",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {
            artifact_id.clone(): {
                "handle": artifact_handle,
                "kind": "model",
                "owner_controller": "controller:sklearn.production"
            }
        },
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {
            artifact_id.clone(): {
                "node_id": "model:ridge",
                "controller_id": "controller:sklearn.production",
                "uri": artifact_uri
            }
        },
        "seed": 7
    });

    let predict_output = run_production_adapter_one_shot(&root, &artifact_dir, &predict_task);
    let predict_value: serde_json::Value =
        serde_json::from_slice(&predict_output).expect("PREDICT result is JSON");
    let predictions = predict_value["predictions"]
        .as_array()
        .expect("PREDICT result has predictions");
    assert_eq!(
        predictions.len(),
        1,
        "PREDICT produced unexpected block count"
    );
    let values = predictions[0]["values"]
        .as_array()
        .expect("PREDICT prediction values are an array");
    assert!(!values.is_empty(), "PREDICT returned empty value rows");
    for row in values {
        let row = row.as_array().expect("each prediction row is an array");
        assert_eq!(row.len(), 1, "Ridge prediction width is 1");
        assert!(
            row[0].as_f64().is_some_and(f64::is_finite),
            "PREDICT row carries non-finite value: {row:?}"
        );
    }

    let _ = std::fs::remove_dir_all(artifact_dir);
}

#[test]
fn sklearn_production_controller_jsonl_emits_error_frame_and_survives() {
    let root = repo_root();
    if !python_has_sklearn(&root) {
        return;
    }
    let joblib_available = Command::new("python3")
        .current_dir(&root)
        .args(["-c", "import joblib"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !joblib_available {
        return;
    }

    let suffix = unique_suffix();
    let artifact_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_sklearn_production_jsonl_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

    // A bad task (unknown operator) followed by a good task that
    // should still produce a result frame, proving the worker survived
    // the first failure.
    let node_plan_bad = json!({
        "node_id": "model:bad",
        "kind": "model",
        "controller_id": "controller:sklearn.production",
        "controller_version": "1.0.0",
        "supported_phases": ["REFIT"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "full_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "DefinitelyNotAnSklearnClass"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let node_plan_good = json!({
        "node_id": "model:ridge",
        "kind": "model",
        "controller_id": "controller:sklearn.production",
        "controller_version": "1.0.0",
        "supported_phases": ["REFIT"],
        "controller_capabilities": ["deterministic"],
        "fit_scope": "full_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "Ridge"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let variant = json!({
        "variant_id": "variant:base",
        "choices": {},
        "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "seed": 7
    });
    let bad_task = json!({
        "type": "task",
        "schema_version": 1,
        "task": {
            "run_id": "run:cli.sklearn-production-jsonl",
            "node_plan": node_plan_bad,
            "phase": "REFIT",
            "variant_id": "variant:base",
            "variant": variant,
            "fold_id": null,
            "branch_path": [],
            "input_handles": {},
            "data_views": {},
            "prediction_inputs": {},
            "artifact_inputs": {},
            "seed": 7
        }
    });
    let good_task = json!({
        "type": "task",
        "schema_version": 1,
        "task": {
            "run_id": "run:cli.sklearn-production-jsonl",
            "node_plan": node_plan_good,
            "phase": "REFIT",
            "variant_id": "variant:base",
            "variant": variant,
            "fold_id": null,
            "branch_path": [],
            "input_handles": {},
            "data_views": {},
            "prediction_inputs": {},
            "artifact_inputs": {},
            "seed": 7
        }
    });
    let close = json!({"type": "close", "schema_version": 1});

    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("python3")
        .current_dir(&root)
        .args([
            "examples/adapters/sklearn_production_controller.py",
            "--jsonl",
        ])
        .env(
            "DAG_ML_PROCESS_ARTIFACT_DIR",
            artifact_dir
                .to_str()
                .expect("artifact dir path is valid utf-8"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sklearn production adapter");
    {
        let stdin = child.stdin.as_mut().expect("adapter stdin is piped");
        for frame in [&bad_task, &good_task, &close] {
            stdin
                .write_all(
                    serde_json::to_vec(frame)
                        .expect("frame JSON serializes")
                        .as_slice(),
                )
                .expect("write frame to adapter stdin");
            stdin.write_all(b"\n").expect("write trailing newline");
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for sklearn production adapter");
    assert!(
        output.status.success(),
        "JSONL adapter exited with failure after a bad-task error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_text = String::from_utf8(output.stdout).expect("adapter stdout is utf-8");
    let mut frames: Vec<serde_json::Value> = Vec::new();
    for line in stdout_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        frames.push(serde_json::from_str(line).expect("each adapter frame is JSON"));
    }
    let error_frames: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"].as_str() == Some("error"))
        .collect();
    let result_frames: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"].as_str() == Some("result"))
        .collect();
    let ack_frames: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"].as_str() == Some("ack"))
        .collect();
    assert_eq!(
        frames.len(),
        3,
        "expected exactly 3 frames (error + result + close ack), got: {:#?}",
        frames
    );
    assert_eq!(
        error_frames.len(),
        1,
        "expected exactly one error frame, got: {:#?}",
        frames
    );
    assert_eq!(
        error_frames[0]["error"]["code"].as_str(),
        Some("unknown_operator")
    );
    assert!(error_frames[0]["error"]["message"]
        .as_str()
        .is_some_and(|m| m.contains("DefinitelyNotAnSklearnClass")));
    assert_eq!(
        result_frames.len(),
        1,
        "worker should still produce one result after the error, frames: {:#?}",
        frames
    );
    assert_eq!(
        result_frames[0]["result"]["node_id"].as_str(),
        Some("model:ridge")
    );
    assert_eq!(
        ack_frames.len(),
        1,
        "expected one close ack frame, got: {:#?}",
        frames
    );
    assert_eq!(ack_frames[0]["status"].as_str(), Some("closed"));

    let _ = std::fs::remove_dir_all(artifact_dir);
}

#[test]
fn sklearn_production_controller_manifest_validates_and_matches_registry() {
    let root = repo_root();
    let manifest_path = root.join("examples/controllers/sklearn_production.controller.json");
    let manifest_text =
        std::fs::read_to_string(&manifest_path).expect("sklearn production manifest is readable");
    let manifest: dag_ml_core::controller::ControllerManifest =
        serde_json::from_str(&manifest_text).expect("manifest deserializes as ControllerManifest");
    manifest
        .validate()
        .expect("sklearn production manifest passes ControllerManifest::validate");

    // Static manifest validation passes without sklearn; the
    // registry-parity probe below imports the controller module and
    // therefore requires sklearn (and transitively joblib/numpy) to
    // be available. Skip the parity portion on hosts that lack them,
    // matching the guard used by the timeout test.
    if !python_has_sklearn(&root) {
        return;
    }

    assert_eq!(
        manifest.controller_id.as_str(),
        "controller:sklearn.production"
    );
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::FitCv));
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::Refit));
    assert!(manifest
        .supported_phases
        .contains(&dag_ml_core::phase::Phase::Predict));

    let manifest_value: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("manifest parses as JSON value");
    let manifest_aliases: std::collections::BTreeSet<String> = manifest_value["operator_selectors"]
        .as_array()
        .expect("operator_selectors is an array")
        .iter()
        .find_map(|selector| selector.get("aliases"))
        .and_then(|aliases| aliases.as_array())
        .expect("operator_selectors contains an aliases selector")
        .iter()
        .map(|alias| alias.as_str().expect("alias is string").to_string())
        .collect();

    // Parity check: the manifest aliases must match the controller's
    // OPERATOR_SELECTORS keys exactly. Drift between the two would
    // leave the manifest advertising operators the controller refuses
    // to dispatch, or vice versa.
    let probe_script = "import importlib.util\n\
spec = importlib.util.spec_from_file_location(\n\
    'sklearn_production_controller',\n\
    'examples/adapters/sklearn_production_controller.py',\n\
)\n\
mod = importlib.util.module_from_spec(spec)\n\
spec.loader.exec_module(mod)\n\
import json\n\
print(json.dumps(sorted(mod.OPERATOR_SELECTORS.keys())))\n\
";
    let suffix = unique_suffix();
    let probe_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_sklearn_production_manifest_probe_{}_{}.py",
        std::process::id(),
        suffix
    ));
    std::fs::write(&probe_path, probe_script).expect("write parity probe script");
    let probe = Command::new("python3")
        .current_dir(&root)
        .arg(&probe_path)
        .output()
        .expect("run parity probe");
    let _ = std::fs::remove_file(&probe_path);
    assert!(
        probe.status.success(),
        "parity probe failed: stderr=`{}`",
        String::from_utf8_lossy(&probe.stderr)
    );
    let registry_aliases: std::collections::BTreeSet<String> =
        serde_json::from_slice(&probe.stdout).expect("registry probe stdout is a JSON list");
    assert_eq!(
        manifest_aliases, registry_aliases,
        "manifest aliases drift from controller's OPERATOR_SELECTORS: manifest_only={:?}, registry_only={:?}",
        manifest_aliases.difference(&registry_aliases).collect::<Vec<_>>(),
        registry_aliases.difference(&manifest_aliases).collect::<Vec<_>>()
    );
}

#[test]
fn sklearn_production_controller_fit_timeout_raises_retryable_error() {
    let root = repo_root();
    if !python_has_sklearn(&root) {
        return;
    }
    let suffix = unique_suffix();
    let probe_path = std::env::temp_dir().join(format!(
        "dag_ml_cli_sklearn_production_timeout_probe_{}_{}.py",
        std::process::id(),
        suffix
    ));
    let probe_script = r#"import importlib.util, os, sys, time
os.environ['DAG_ML_PROCESS_FIT_TIMEOUT_SECONDS'] = '1'
sys.argv = ['sklearn_production_controller.py']
spec = importlib.util.spec_from_file_location(
    'sklearn_production_controller',
    'examples/adapters/sklearn_production_controller.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    mod.with_fit_timeout(lambda: time.sleep(3))
    print('NO_TIMEOUT')
except mod.AdapterTaskError as exc:
    print(f'OK:{exc.code}:{int(exc.retryable)}')
"#;
    std::fs::write(&probe_path, probe_script).expect("write timeout probe script");
    let probe = Command::new("python3")
        .current_dir(&root)
        .arg(&probe_path)
        .output()
        .expect("spawn python timeout probe");
    let _ = std::fs::remove_file(&probe_path);
    assert!(
        probe.status.success(),
        "timeout probe failed: stdout=`{}` stderr=`{}`",
        String::from_utf8_lossy(&probe.stdout),
        String::from_utf8_lossy(&probe.stderr)
    );
    let stdout_text = String::from_utf8_lossy(&probe.stdout);
    assert!(
        stdout_text.contains("OK:fit_timeout:1"),
        "expected `OK:fit_timeout:1` in timeout probe stdout, got: {}",
        stdout_text
    );
}

fn run_production_adapter_one_shot(
    root: &Path,
    artifact_dir: &Path,
    task: &serde_json::Value,
) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("python3")
        .current_dir(root)
        .args(["examples/adapters/sklearn_production_controller.py"])
        .env(
            "DAG_ML_PROCESS_ARTIFACT_DIR",
            artifact_dir
                .to_str()
                .expect("artifact dir path is valid utf-8"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sklearn production adapter");
    {
        let stdin = child.stdin.as_mut().expect("adapter stdin is piped");
        stdin
            .write_all(
                serde_json::to_vec(task)
                    .expect("task JSON serializes")
                    .as_slice(),
            )
            .expect("write task JSON to adapter stdin");
        stdin.write_all(b"\n").expect("write trailing newline");
    }
    let output = child
        .wait_with_output()
        .expect("wait for sklearn production adapter");
    assert!(
        output.status.success(),
        "sklearn production adapter exited with failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn sklearn_production_controller_rejects_artifact_uri_outside_artifact_dir() {
    let root = repo_root();
    if !python_has_sklearn(&root) {
        return;
    }
    let joblib_available = Command::new("python3")
        .current_dir(&root)
        .args(["-c", "import joblib"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !joblib_available {
        return;
    }

    let suffix = unique_suffix();
    let artifact_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_sklearn_production_traversal_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

    let node_plan = json!({
        "node_id": "model:ridge",
        "kind": "model",
        "controller_id": "controller:sklearn.production",
        "controller_version": "1.0.0",
        "supported_phases": ["PREDICT"],
        "controller_capabilities": ["deterministic", "thread_safe"],
        "fit_scope": "full_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
        "input_nodes": [],
        "output_nodes": [],
        "shape_plan": null,
        "data_bindings": [],
        "params": {"operator": "Ridge"},
        "params_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    });
    let artifact_id = "artifact:model:ridge:sklearn:refit";
    let predict_task = json!({
        "run_id": "run:cli.sklearn-production-traversal",
        "node_plan": node_plan,
        "phase": "PREDICT",
        "variant_id": "variant:base",
        "variant": {
            "variant_id": "variant:base",
            "choices": {},
            "fingerprint": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "seed": 7
        },
        "fold_id": null,
        "branch_path": [],
        "input_handles": {
            artifact_id: {
                "handle": 1,
                "kind": "model",
                "owner_controller": "controller:sklearn.production"
            }
        },
        "data_views": {},
        "prediction_inputs": {},
        "artifact_inputs": {
            artifact_id: {
                "node_id": "model:ridge",
                "controller_id": "controller:sklearn.production",
                "uri": "/etc/passwd"
            }
        },
        "seed": 7
    });

    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("python3")
        .current_dir(&root)
        .args(["examples/adapters/sklearn_production_controller.py"])
        .env(
            "DAG_ML_PROCESS_ARTIFACT_DIR",
            artifact_dir
                .to_str()
                .expect("artifact dir path is valid utf-8"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sklearn production adapter");
    {
        let stdin = child.stdin.as_mut().expect("adapter stdin is piped");
        stdin
            .write_all(
                serde_json::to_vec(&predict_task)
                    .expect("task JSON serializes")
                    .as_slice(),
            )
            .expect("write task JSON to adapter stdin");
        stdin.write_all(b"\n").expect("write trailing newline");
    }
    let output = child
        .wait_with_output()
        .expect("wait for sklearn production adapter");
    assert!(
        !output.status.success(),
        "production adapter accepted a traversal URI instead of rejecting it"
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    let artifact_dir_str = artifact_dir
        .to_str()
        .expect("artifact dir path is valid utf-8");
    assert!(
        stderr_text.contains("resolved under artifact dir")
            && stderr_text.contains(artifact_dir_str)
            && !stderr_text.contains("loaded /etc/passwd")
            && !stderr_text.contains("UnpicklingError"),
        "production adapter did not confine the traversal URI under artifact dir: {}",
        stderr_text
    );

    let _ = std::fs::remove_dir_all(artifact_dir);
}

#[test]
fn cli_executes_mixed_branch_merge_with_minimal_aliases() {
    let root = repo_root();
    if !python_has_sklearn(&root) {
        return;
    }

    let suffix = unique_suffix();
    let temp_plan = std::env::temp_dir().join(format!(
        "dag_ml_cli_mixed_alias_plan_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_mixed_alias_bundle_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_lineage = std::env::temp_dir().join(format!(
        "dag_ml_cli_mixed_alias_lineage_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_prediction_cache = std::env::temp_dir().join(format!(
        "dag_ml_cli_mixed_alias_prediction_cache_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let bundle_id = format!("bundle:cli.mixed.alias.{suffix}");
    let plan_id = format!("plan:cli.mixed.alias.{suffix}");
    let run_id = format!("run:cli.mixed.alias.{suffix}");

    let build_plan = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-pipeline-dsl-plan",
            "--dsl",
            "examples/pipeline_dsl_mixed_branch_merge_executable.json",
            "--controllers",
            "examples/controller_manifests_alias_registry.json",
            "--plan-id",
            plan_id.as_str(),
            "--output",
            temp_plan.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to build mixed branch merge DSL plan");
    assert!(
        build_plan.status.success(),
        "mixed branch merge DSL plan failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build_plan.stdout),
        String::from_utf8_lossy(&build_plan.stderr)
    );

    let plan_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&temp_plan).expect("plan was written"))
            .expect("plan is JSON");
    let graph_nodes = plan_json["graph_plan"]["graph"]["nodes"]
        .as_array()
        .expect("graph nodes array");
    assert!(
        graph_nodes.iter().any(|node| {
            node["id"] == "branch:scaled.transform"
                && node["operator"].as_str() == Some("StandardScaler")
        }) && graph_nodes.iter().any(|node| {
            node["id"] == "branch:raw_copy.transform"
                && node["operator"].as_str() == Some("FunctionTransformer")
        }) && graph_nodes.iter().any(|node| {
            node["id"] == "merge:mixed"
                && node["kind"] == "mixed_join"
                && node["metadata"]["include_original_data"] == true
                && node["metadata"]["branch_data_inputs"]
                    .as_array()
                    .is_some_and(|inputs| inputs.len() == 2)
        }),
        "unexpected mixed branch merge plan JSON: {}",
        plan_json
    );
    assert_eq!(
        plan_json["node_plans"]["branch:scaled.transform"]["controller_id"],
        "controller:transformer-mixin.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["branch:raw_copy.transform"]["controller_id"],
        "controller:transformer-mixin.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["branch:scaled.model:ridge"]["controller_id"],
        "controller:sklearn-estimator.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["merge:mixed"]["controller_id"],
        "controller:mixed-join.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["model:final.ridge"]["controller_id"],
        "controller:sklearn-estimator.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["merge:mixed"]["input_nodes"]
            .as_array()
            .expect("mixed merge input nodes")
            .len(),
        3
    );

    let run = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-dsl-cv-refit-bundle",
            "--dsl",
            "examples/pipeline_dsl_mixed_branch_merge_executable.json",
            "--controllers",
            "examples/controller_manifests_alias_registry.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/sklearn_process_controller.py",
            "--persistent",
            "--process-workers",
            "2",
            "--bundle-id",
            bundle_id.as_str(),
            "--output",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--lineage-output",
            temp_lineage.to_str().expect("temp path is valid utf-8"),
            "--prediction-cache-output",
            temp_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            plan_id.as_str(),
            "--run-id",
            run_id.as_str(),
        ])
        .output()
        .expect("failed to run mixed branch merge DSL bundle");
    assert!(
        run.status.success(),
        "mixed branch merge DSL bundle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("process DSL cv refit bundle run: 10 fit_cv result(s)")
            && stdout.contains("4 OOF prediction block(s)")
            && stdout.contains("5 refit result(s)")
            && stdout.contains("2 captured artifact handle(s)")
            && stdout.contains("1 prediction cache(s)")
            && stdout.contains("configured process worker(s)=2")
            && stdout.contains("observed process worker(s)=2"),
        "unexpected mixed branch merge DSL bundle output: {}",
        stdout
    );

    let bundle_json =
        std::fs::read_to_string(&temp_bundle).expect("mixed branch merge bundle was written");
    assert!(
        bundle_json.contains("branch:scaled.transform")
            && bundle_json.contains("branch:raw_copy.transform")
            && bundle_json.contains("merge:mixed")
            && bundle_json.contains("model:final.ridge")
            && bundle_json.contains("artifact:model:final.ridge:sklearn:refit"),
        "unexpected mixed branch merge bundle JSON: {}",
        bundle_json
    );
    let lineage_json =
        std::fs::read_to_string(&temp_lineage).expect("mixed branch merge lineage was written");
    assert!(
        lineage_json.contains("merge:mixed")
            && lineage_json.contains("model:final.ridge")
            && lineage_json.contains("input_lineage"),
        "unexpected mixed branch merge lineage JSON: {}",
        lineage_json
    );
    let prediction_cache_json = std::fs::read_to_string(&temp_prediction_cache)
        .expect("mixed branch merge prediction cache was written");
    assert!(
        prediction_cache_json.contains(&format!("\"bundle_id\": \"{bundle_id}\""))
            && prediction_cache_json
                .contains("prediction-cache:branch:scaled.model:ridge.oof->merge:mixed.scaled_oof"),
        "unexpected mixed branch merge prediction cache JSON: {}",
        prediction_cache_json
    );

    let _ = std::fs::remove_file(temp_plan);
    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_file(temp_lineage);
    let _ = std::fs::remove_file(temp_prediction_cache);
}

#[test]
fn cli_executes_tuner_operator_with_minimal_aliases() {
    let root = repo_root();
    let suffix = unique_suffix();
    let temp_plan = std::env::temp_dir().join(format!(
        "dag_ml_cli_tuner_plan_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_tuner_bundle_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_lineage = std::env::temp_dir().join(format!(
        "dag_ml_cli_tuner_lineage_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_prediction_cache = std::env::temp_dir().join(format!(
        "dag_ml_cli_tuner_prediction_cache_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let bundle_id = format!("bundle:cli.tuner.alias.{suffix}");
    let plan_id = format!("plan:cli.tuner.alias.{suffix}");
    let run_id = format!("run:cli.tuner.alias.{suffix}");

    let build_plan = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-pipeline-dsl-plan",
            "--dsl",
            "examples/pipeline_dsl_tuner_executable.json",
            "--controllers",
            "examples/controller_manifests_alias_registry.json",
            "--plan-id",
            plan_id.as_str(),
            "--output",
            temp_plan.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to build tuner DSL plan");
    assert!(
        build_plan.status.success(),
        "tuner DSL plan failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build_plan.stdout),
        String::from_utf8_lossy(&build_plan.stderr)
    );

    let plan_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&temp_plan).expect("plan was written"))
            .expect("plan is JSON");
    assert_eq!(
        plan_json["node_plans"]["transform:snv"]["controller_id"],
        "controller:transformer-mixin.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["tuner:optuna"]["controller_id"],
        "controller:tuner.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["merge:tuned_features"]["controller_id"],
        "controller:mixed-join.mock"
    );
    assert_eq!(
        plan_json["node_plans"]["model:final.ridge"]["controller_id"],
        "controller:sklearn-estimator.mock"
    );
    assert_eq!(
        plan_json["graph_plan"]["graph"]["nodes"]
            .as_array()
            .expect("graph nodes array")
            .iter()
            .find(|node| node["id"] == "tuner:optuna")
            .expect("tuner node")["kind"],
        "tuner"
    );

    let run = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-dsl-cv-refit-bundle",
            "--dsl",
            "examples/pipeline_dsl_tuner_executable.json",
            "--controllers",
            "examples/controller_manifests_alias_registry.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--bundle-id",
            bundle_id.as_str(),
            "--output",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--lineage-output",
            temp_lineage.to_str().expect("temp path is valid utf-8"),
            "--prediction-cache-output",
            temp_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            plan_id.as_str(),
            "--run-id",
            run_id.as_str(),
        ])
        .output()
        .expect("failed to run tuner DSL bundle");
    assert!(
        run.status.success(),
        "tuner DSL bundle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("process DSL cv refit bundle run: 8 fit_cv result(s)")
            && stdout.contains("4 OOF prediction block(s)")
            && stdout.contains("4 refit result(s)")
            && stdout.contains("2 captured artifact handle(s)")
            && stdout.contains("1 prediction cache(s)"),
        "unexpected tuner DSL bundle output: {}",
        stdout
    );

    let prediction_cache_json = std::fs::read_to_string(&temp_prediction_cache)
        .expect("tuner prediction cache was written");
    assert!(
        prediction_cache_json.contains(&format!("\"bundle_id\": \"{bundle_id}\""))
            && prediction_cache_json
                .contains("prediction-cache:tuner:optuna.oof->merge:tuned_features.tuned_oof"),
        "unexpected tuner prediction cache JSON: {}",
        prediction_cache_json
    );

    let _ = std::fs::remove_file(temp_plan);
    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_file(temp_lineage);
    let _ = std::fs::remove_file(temp_prediction_cache);
}

#[test]
fn cli_executes_runtime_data_generation_operator() {
    let root = repo_root();
    let suffix = unique_suffix();
    let temp_plan = std::env::temp_dir().join(format!(
        "dag_ml_cli_runtime_generation_plan_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_bundle = std::env::temp_dir().join(format!(
        "dag_ml_cli_runtime_generation_bundle_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_lineage = std::env::temp_dir().join(format!(
        "dag_ml_cli_runtime_generation_lineage_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let temp_prediction_cache = std::env::temp_dir().join(format!(
        "dag_ml_cli_runtime_generation_prediction_cache_{}_{}.json",
        std::process::id(),
        suffix
    ));
    let bundle_id = format!("bundle:cli.runtime-generation.{suffix}");
    let plan_id = format!("plan:cli.runtime-generation.{suffix}");
    let run_id = format!("run:cli.runtime-generation.{suffix}");

    let build_plan = Command::new(cli())
        .current_dir(&root)
        .args([
            "build-pipeline-dsl-plan",
            "--dsl",
            "examples/pipeline_dsl_runtime_generation_executable.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--plan-id",
            plan_id.as_str(),
            "--output",
            temp_plan.to_str().expect("temp path is valid utf-8"),
        ])
        .output()
        .expect("failed to build runtime generation DSL plan");
    assert!(
        build_plan.status.success(),
        "runtime generation DSL plan failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build_plan.stdout),
        String::from_utf8_lossy(&build_plan.stderr)
    );
    let plan_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&temp_plan).expect("plan was written"))
            .expect("plan is JSON");
    assert_eq!(
        plan_json["node_plans"]["generator:synthetic.train"]["kind"],
        "generator"
    );
    assert_eq!(
        plan_json["node_plans"]["generator:synthetic.train"]["controller_id"],
        "controller:data-generator.mock"
    );
    assert_eq!(
        plan_json["graph_plan"]["graph"]["nodes"]
            .as_array()
            .expect("nodes")
            .iter()
            .find(|node| node["id"] == "generator:synthetic.train")
            .expect("generator node")["metadata"]["dsl_generation_kind"],
        "data"
    );
    assert!(plan_json["campaign"]["shape_plans"]
        .as_object()
        .expect("shape plans")
        .contains_key("generator:synthetic.train"));

    let run = Command::new(cli())
        .current_dir(&root)
        .args([
            "run-process-dsl-cv-refit-bundle",
            "--dsl",
            "examples/pipeline_dsl_runtime_generation_executable.json",
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
            bundle_id.as_str(),
            "--output",
            temp_bundle.to_str().expect("temp path is valid utf-8"),
            "--lineage-output",
            temp_lineage.to_str().expect("temp path is valid utf-8"),
            "--prediction-cache-output",
            temp_prediction_cache
                .to_str()
                .expect("temp path is valid utf-8"),
            "--plan-id",
            plan_id.as_str(),
            "--run-id",
            run_id.as_str(),
        ])
        .output()
        .expect("failed to run runtime generation DSL bundle");
    assert!(
        run.status.success(),
        "runtime generation DSL bundle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("process DSL cv refit bundle run: 4 fit_cv result(s)")
            && stdout.contains("2 OOF prediction block(s)")
            && stdout.contains("2 refit result(s)")
            && stdout.contains("1 captured artifact handle(s)")
            && stdout.contains("0 prediction cache(s)")
            && stdout.contains("configured process worker(s)=2")
            && stdout.contains("observed process worker(s)=2"),
        "unexpected runtime generation DSL bundle output: {}",
        stdout
    );
    let bundle_json =
        std::fs::read_to_string(&temp_bundle).expect("runtime generation bundle was written");
    assert!(
        bundle_json.contains("generator:synthetic.train")
            && bundle_json.contains("model:ridge.after-generation"),
        "unexpected runtime generation bundle JSON: {}",
        bundle_json
    );
    let lineage_json =
        std::fs::read_to_string(&temp_lineage).expect("runtime generation lineage was written");
    assert!(
        lineage_json.contains("generator:synthetic.train")
            && lineage_json.contains("model:ridge.after-generation"),
        "unexpected runtime generation lineage JSON: {}",
        lineage_json
    );

    let _ = std::fs::remove_file(temp_plan);
    let _ = std::fs::remove_file(temp_bundle);
    let _ = std::fs::remove_file(temp_lineage);
    let _ = std::fs::remove_file(temp_prediction_cache);
}

#[test]
fn cli_enforces_process_timeouts_and_restarts_persistent_workers() {
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
    let one_shot_timeout_marker_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_flaky_one_shot_timeout_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let error_no_retry_marker_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_flaky_error_no_retry_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    let error_retry_marker_dir = std::env::temp_dir().join(format!(
        "dag_ml_cli_flaky_error_retry_{}_{}",
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

    let error_no_retry_run_id = format!("run:cli.process.retryable-error.{}", unique_suffix());
    let error_no_retry = Command::new(cli())
        .current_dir(&root)
        .env("DAG_ML_FLAKY_MARKER_DIR", &error_no_retry_marker_dir)
        .env("DAG_ML_FLAKY_ERROR_ONCE", "1")
        .env("DAG_ML_FLAKY_SLEEP_SECONDS", "0.0")
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
            "2000",
            "--plan-id",
            "plan:cli.process.retryable-error.no-retry",
            "--run-id",
            error_no_retry_run_id.as_str(),
        ])
        .output()
        .expect("failed to run flaky process campaign with retryable error");
    assert!(
        !error_no_retry.status.success(),
        "flaky retryable-error campaign unexpectedly succeeded without retry: {}",
        String::from_utf8_lossy(&error_no_retry.stdout)
    );
    let error_no_retry_stderr = String::from_utf8_lossy(&error_no_retry.stderr);
    assert!(
        error_no_retry_stderr.contains("adapter task returned error `retryable_test_error`")
            && error_no_retry_stderr.contains("after 1 attempt(s)"),
        "unexpected retryable adapter error without retry: {}",
        error_no_retry_stderr
    );

    let error_retry_run_id = format!("run:cli.process.retryable-error-retry.{}", unique_suffix());
    let error_retry = Command::new(cli())
        .current_dir(&root)
        .env("DAG_ML_FLAKY_MARKER_DIR", &error_retry_marker_dir)
        .env("DAG_ML_FLAKY_ERROR_ONCE", "1")
        .env("DAG_ML_FLAKY_SLEEP_SECONDS", "0.0")
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
            "2000",
            "--process-retries",
            "1",
            "--plan-id",
            "plan:cli.process.retryable-error.retry",
            "--run-id",
            error_retry_run_id.as_str(),
        ])
        .output()
        .expect("failed to run flaky process campaign with retryable error and retry");
    assert!(
        error_retry.status.success(),
        "flaky retryable-error campaign with retry failed: {}",
        String::from_utf8_lossy(&error_retry.stderr)
    );
    assert!(
        String::from_utf8_lossy(&error_retry.stdout).contains("process campaign run: 8 result(s)"),
        "unexpected retryable-error retry output: {}",
        String::from_utf8_lossy(&error_retry.stdout)
    );

    let one_shot_timeout_run_id = format!("run:cli.process.one-shot-timeout.{}", unique_suffix());
    let one_shot_timeout = Command::new(cli())
        .current_dir(&root)
        .env("DAG_ML_FLAKY_MARKER_DIR", &one_shot_timeout_marker_dir)
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
            "--process-timeout-ms",
            "750",
            "--plan-id",
            "plan:cli.process.one-shot-timeout",
            "--run-id",
            one_shot_timeout_run_id.as_str(),
        ])
        .output()
        .expect("failed to run flaky one-shot process campaign with timeout");
    assert!(
        !one_shot_timeout.status.success(),
        "flaky one-shot process campaign unexpectedly succeeded after timeout: {}",
        String::from_utf8_lossy(&one_shot_timeout.stdout)
    );
    let one_shot_timeout_stderr = String::from_utf8_lossy(&one_shot_timeout.stderr);
    assert!(
        one_shot_timeout_stderr.contains("timed out after 750 ms"),
        "unexpected flaky one-shot timeout error: {}",
        one_shot_timeout_stderr
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
    let _ = std::fs::remove_dir_all(one_shot_timeout_marker_dir);
    let _ = std::fs::remove_dir_all(error_no_retry_marker_dir);
    let _ = std::fs::remove_dir_all(error_retry_marker_dir);
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

fn r_has_prospectr(root: &Path) -> bool {
    Command::new("Rscript")
        .current_dir(root)
        .args([
            "-e",
            "suppressPackageStartupMessages({library(jsonlite); library(prospectr)})",
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn r_has_mdatools(root: &Path) -> bool {
    Command::new("Rscript")
        .current_dir(root)
        .args([
            "-e",
            "suppressPackageStartupMessages({library(jsonlite); library(mdatools); library(digest)})",
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
