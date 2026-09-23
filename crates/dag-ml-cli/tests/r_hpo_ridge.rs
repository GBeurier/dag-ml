use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use dag_ml_core::{
    build_execution_plan, CampaignSpec, ControllerManifest, ControllerRegistry,
    ExternalDataPlanEnvelope, GraphSpec, ObservationId, SampleId, SampleRelation,
    SampleRelationSet,
};
use serde_json::{json, Value};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn test_directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "dagml-r-hpo-ridge-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    path
}

fn write_json(path: &Path, value: &impl serde::Serialize) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn plan() -> dag_ml_core::ExecutionPlan {
    let graph: GraphSpec = serde_json::from_value(json!({
        "id": "graph:r-hpo-ridge", "interface": {"inputs": [], "outputs": []},
        "nodes": [{
            "id": "model:m", "kind": "model", "operator": {"type": "RidgeR"},
            "params": {}, "ports": {"inputs": [], "outputs": [{
                "name": "oof", "kind": "prediction", "representation": null,
                "cardinality": "one", "description": ""
            }]}, "metadata": {}, "seed_label": null
        }],
        "edges": [], "search_space_fingerprint": null, "metadata": {}
    }))
    .unwrap();
    let campaign: CampaignSpec = serde_json::from_value(json!({
        "id": "campaign:r-hpo-ridge", "root_seed": 7,
        "split_invocation": {
            "id": "split:outer", "controller_id": null,
            "leakage_policy": {
                "split_unit": "sample", "forbid_origin_cross_fold": true,
                "allow_observation_split_with_shared_target": false,
                "require_group_ids": false, "unsafe_flags": []
            },
            "params": {},
            "fold_set": {
                "id": "folds:r-hpo-ridge", "sample_ids": ["s1", "s2"],
                "sample_groups": {},
                "folds": [
                    {"fold_id": "fold:0", "train_sample_ids": ["s2"],
                     "validation_sample_ids": ["s1"], "metadata": {}},
                    {"fold_id": "fold:1", "train_sample_ids": ["s1"],
                     "validation_sample_ids": ["s2"], "metadata": {}}
                ]
            }
        }
    }))
    .unwrap();
    let manifest: ControllerManifest = serde_json::from_value(json!({
        "controller_id": "controller:r-ridge", "controller_version": "1.0.0",
        "operator_kind": "model", "priority": 0,
        "supported_phases": ["FIT_CV"], "input_ports": [],
        "output_ports": graph.nodes[0].ports.outputs,
        "data_requirements": null,
        "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions"],
        "fit_scope": "fold_train", "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable"
    }))
    .unwrap();
    let mut registry = ControllerRegistry::new();
    registry.register(manifest).unwrap();
    build_execution_plan("plan:r-hpo-ridge", graph, campaign, &registry).unwrap()
}

fn assert_score(actual: &Value, expected: f64) {
    let actual = actual.as_f64().expect("native HPO score is numeric");
    assert!(
        (actual - expected).abs() < 1e-9,
        "native HPO score {actual} differs from R Ridge oracle {expected}"
    );
}

#[test]
fn r_ridge_hpo_uses_fold_train_ids_and_native_parallel_pruning_resume() {
    if Command::new("Rscript").arg("--version").output().is_err() {
        assert_ne!(std::env::var("DAGML_REQUIRE_HPO_R").as_deref(), Ok("1"));
        return;
    }
    let repo = root();
    let work = test_directory();
    let plan_path = work.join("plan.json");
    let data_path = work.join("data.csv");
    let envelope_path = work.join("envelope.json");
    let request_path = work.join("request.json");
    let checkpoint_path = work.join("checkpoint.json");
    let outcome_path = work.join("outcome.json");
    let evidence_dir = work.join("operator-evidence");
    std::fs::create_dir(&evidence_dir).unwrap();
    write_json(&plan_path, &plan());
    std::fs::write(&data_path, "id,x,y\ns1,1,1\ns2,2,2\n").unwrap();
    let mut envelope: ExternalDataPlanEnvelope = serde_json::from_slice(&std::fs::read(
        repo.join("crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"),
    ).unwrap()).unwrap();
    envelope.coordinator_relations = Some(SampleRelationSet {
        records: ["s1", "s2"]
            .into_iter()
            .map(|id| {
                SampleRelation::new(
                    ObservationId::new(format!("obs:{id}")).unwrap(),
                    SampleId::new(id).unwrap(),
                )
            })
            .collect(),
    });
    envelope.relation_fingerprint = None;
    write_json(&envelope_path, &envelope);
    let adapter = repo.join("examples/adapters/hpo_ridge_operator.R");
    let optimizer = repo.join("examples/adapters/hpo_ridge_optimizer_jsonl.R");
    let wrapper = repo.join("scripts/run_r_hpo_ridge_oracle.R");
    for budget in [2, 3] {
        write_json(
            &request_path,
            &json!({
                "target_node": "model:m", "trial_budget": budget,
                "metric": "rmse", "direction": "minimize",
                "optimizer_descriptor": {"owner": "r-ridge-oracle"},
                "progressive_pruning": true
            }),
        );
        let output = Command::new("Rscript")
            .arg(&wrapper)
            .args([
                &repo,
                &plan_path,
                &envelope_path,
                &request_path,
                &adapter,
                &optimizer,
                Path::new(env!("CARGO_BIN_EXE_dag-ml-cli")),
                &checkpoint_path,
                &outcome_path,
            ])
            .env("DAGML_R_HPO_PLAN", &plan_path)
            .env("DAGML_R_HPO_DATA", &data_path)
            .env("DAGML_R_HPO_EVIDENCE_DIR", &evidence_dir)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "R wrapper/native CLI failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&std::fs::read(&outcome_path).unwrap()).unwrap();
        assert_eq!(result["status"], "completed");
        // SearchResult lists completed candidates; the sealed checkpoint also
        // retains the pruned candidate and its first-fold evidence.
        assert_eq!(result["trials"].as_array().unwrap().len(), 2);
        assert_eq!(result["selected_trial_index"], 0);
        let trials = result["trials"].as_array().unwrap();
        assert_score(&trials[0]["score"], 0.0);
        assert_score(
            &trials[1]["score"],
            (0.2_f64.powi(2) + 1.0).sqrt() / 2.0_f64.sqrt(),
        );
        let fold_reports = trials[1]["scores"]["reports"].as_array().unwrap();
        let folds = fold_reports
            .iter()
            .filter_map(|report| report["fold_id"].as_str())
            .collect::<BTreeSet<_>>();
        assert!(folds.contains("fold:0") && folds.contains("fold:1"));
        for (fold, expected) in [("fold:0", 0.2), ("fold:1", 1.0)] {
            let report = fold_reports
                .iter()
                .find(|report| report["fold_id"] == fold)
                .unwrap();
            assert_score(&report["metrics"]["rmse"], expected);
        }
        let checkpoint: Value =
            serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
        assert_eq!(checkpoint["trials"].as_array().unwrap().len(), budget);
        if budget == 3 {
            assert_eq!(checkpoint["trials"][2]["state"], "pruned");
            assert_eq!(
                checkpoint["trials"][2]["evidence"]["scores"]["reports"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            assert_score(
                &checkpoint["trials"][2]["evidence"]["intermediate_scores"][0],
                1.0 / 3.0,
            );
            for (trial, fold, train, validation, prediction) in [
                (0, 0, "s2", "s1", 1.0),
                (0, 1, "s1", "s2", 2.0),
                (1, 0, "s2", "s1", 0.8),
                (1, 1, "s1", "s2", 1.0),
                (2, 0, "s2", "s1", 2.0 / 3.0),
            ] {
                let evidence_path =
                    evidence_dir.join(format!("host_hpo_trial_{trial:010}_fold_{fold}.json"));
                let evidence: Value =
                    serde_json::from_slice(&std::fs::read(evidence_path).unwrap()).unwrap();
                assert_eq!(evidence["train_ids"], json!([train]));
                assert_eq!(evidence["validation_ids"], json!([validation]));
                assert_score(&evidence["predictions"][0], prediction);
            }
        }
    }
    std::fs::remove_dir_all(work).unwrap();
}
