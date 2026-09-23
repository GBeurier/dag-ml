use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use dag_ml_core::{
    build_execution_plan, CampaignSpec, ControllerManifest, ControllerRegistry,
    ExternalDataPlanEnvelope, GraphSpec, InitialFullRefitPackage, ObservationId, PredictCohort,
    PredictCohortRole, SampleId, SampleRelation, SampleRelationSet,
    EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2,
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
        "dagml-octave-hpo-ridge-{}-{}",
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
        "id": "graph:octave-hpo-ridge", "interface": {"inputs": [], "outputs": []},
        "nodes": [{
            "id": "model:m", "kind": "model", "operator": {"type": "RidgeOctave"},
            "params": {}, "ports": {"inputs": [], "outputs": [{
                "name": "oof", "kind": "prediction", "representation": null,
                "cardinality": "one", "description": ""
            }]}, "metadata": {}, "seed_label": null
        }],
        "edges": [], "search_space_fingerprint": null, "metadata": {}
    }))
    .unwrap();
    let campaign: CampaignSpec = serde_json::from_value(json!({
        "id": "campaign:octave-hpo-ridge", "root_seed": 7,
        "split_invocation": {
            "id": "split:outer", "controller_id": null,
            "leakage_policy": {
                "split_unit": "sample", "forbid_origin_cross_fold": true,
                "allow_observation_split_with_shared_target": false,
                "require_group_ids": false, "unsafe_flags": []
            },
            "params": {},
            "fold_set": {
                "id": "folds:octave-hpo-ridge", "sample_ids": ["s1", "s2"],
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
        "controller_id": "controller:octave-ridge", "controller_version": "1.0.0",
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
    build_execution_plan("plan:octave-hpo-ridge", graph, campaign, &registry).unwrap()
}

fn assert_score(actual: &Value, expected: f64) {
    let actual = actual.as_f64().expect("native HPO score is numeric");
    assert!(
        (actual - expected).abs() < 1e-9,
        "native HPO score {actual} differs from Octave Ridge oracle {expected}"
    );
}

#[test]
fn octave_ridge_hpo_uses_fold_train_ids_and_native_parallel_pruning_resume() {
    if Command::new("octave").arg("--version").output().is_err() {
        assert_ne!(
            std::env::var("DAGML_REQUIRE_HPO_OCTAVE").as_deref(),
            Ok("1")
        );
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
    std::fs::write(
        &data_path,
        "id,x,y\ns1,1,1\ns2,2,2\nsample:1,1,1\nsample:2,2,2\nheldout:1,3,3\nheldout:2,4,4\n",
    )
    .unwrap();
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
    let adapter = repo.join("examples/adapters/octave_ridge_oracle.sh");
    let optimizer = repo.join("examples/adapters/octave_ridge_optimizer.sh");
    for budget in [2, 3] {
        write_json(
            &request_path,
            &json!({
                "target_node": "model:m", "trial_budget": budget,
                "metric": "rmse", "direction": "minimize",
                "optimizer_descriptor": {"owner": "octave-ridge-oracle"},
                "progressive_pruning": true
            }),
        );
        let output = Command::new("octave")
            .args(["--quiet", "--no-gui", "--no-init-file", "--eval"])
            .arg(format!(
                "addpath('{}'); run_octave_hpo_ridge_oracle",
                repo.join("scripts").display()
            ))
            .env("DAGML_OCTAVE_REPO", &repo)
            .env("DAGML_OCTAVE_PLAN", &plan_path)
            .env("DAGML_OCTAVE_ENVELOPE", &envelope_path)
            .env("DAGML_OCTAVE_REQUEST", &request_path)
            .env("DAGML_OCTAVE_OPERATOR", &adapter)
            .env("DAGML_OCTAVE_OPTIMIZER", &optimizer)
            .env("DAGML_OCTAVE_CLI", env!("CARGO_BIN_EXE_dag-ml-cli"))
            .env("DAGML_OCTAVE_CHECKPOINT", &checkpoint_path)
            .env("DAGML_OCTAVE_OUTCOME", &outcome_path)
            .env("DAGML_OCTAVE_HPO_PLAN", &plan_path)
            .env("DAGML_OCTAVE_HPO_DATA", &data_path)
            .env("DAGML_OCTAVE_HPO_EVIDENCE_DIR", &evidence_dir)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Octave wrapper/native CLI failed: {} {}",
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
    let selected: Value = serde_json::from_slice(&std::fs::read(&outcome_path).unwrap()).unwrap();
    assert_eq!(selected["trials"][0]["params"]["n_components"], 1);
    assert_selected_octave_ridge_refit_replay(&repo, &work, &data_path, &selected);
    std::fs::remove_dir_all(work).unwrap();
}

fn assert_selected_octave_ridge_refit_replay(repo: &Path, work: &Path, data: &Path, hpo: &Value) {
    let mut dsl = json!({
        "id": "dsl:octave-hpo-ridge-refit",
        "input": {"name": "x", "representation": "tabular_numeric"},
        "campaign_id": "campaign:octave-hpo-ridge-refit", "root_seed": 7,
        "leakage_policy": {"split_unit": "sample", "forbid_origin_cross_fold": true,
            "allow_observation_split_with_shared_target": false, "require_group_ids": false,
            "unsafe_flags": []},
        "data_bindings": [{"node_id": "model:initial", "input_name": "x",
            "request_id": "nir-to-tabular",
            "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
            "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
            "relation_fingerprint": "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10",
            "output_representation": "tabular_numeric", "feature_set_id": "x",
            "source_ids": ["nir"], "require_relations": true}],
        "steps": [{"kind": "model", "id": "model:initial", "operator": {"type": "RidgeOctave"},
            "params": {"n_components": hpo["trials"][0]["params"]["n_components"]}}]
    });
    let mut envelope: Value = serde_json::from_slice(
        &std::fs::read(
            repo.join("examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"),
        )
        .unwrap(),
    )
    .unwrap();
    envelope["data_content_fingerprint"] = json!("e".repeat(64));
    envelope["target_content_fingerprint"] = json!("f".repeat(64));
    let typed: ExternalDataPlanEnvelope = serde_json::from_value(envelope.clone()).unwrap();
    let relations = typed.coordinator_relations.as_ref().unwrap();
    envelope["relation_fingerprint"] = json!(relations.fingerprint().unwrap());
    dsl["data_bindings"][0]["relation_fingerprint"] = envelope["relation_fingerprint"].clone();
    let dsl_path = work.join("refit-dsl.json");
    let envelope_path = work.join("refit-envelope.json");
    let ids_path = work.join("refit-ids.json");
    let package_path = work.join("refit-package.json");
    let outcome_path = work.join("refit-outcome.json");
    let sidecar = work.join("artifacts/octave-ridge.mat");
    let adapter = repo.join("examples/adapters/octave_ridge_oracle.sh");
    write_json(&dsl_path, &dsl);
    write_json(&envelope_path, &envelope);
    write_json(&ids_path, &json!(["sample:2", "sample:1"]));
    let output = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
        .current_dir(repo)
        .env("DAGML_OCTAVE_HPO_DATA", data)
        .env("DAGML_OCTAVE_RIDGE_SIDECAR", &sidecar)
        .args(["run-process-dsl-refit-phase", "--dsl"])
        .arg(&dsl_path)
        .arg("--controllers")
        .arg(repo.join("examples/controller_manifests.json"))
        .arg("--envelope")
        .arg(&envelope_path)
        .arg("--training-sample-ids")
        .arg(&ids_path)
        .arg("--adapter")
        .arg(&adapter)
        .arg("--package-output")
        .arg(&package_path)
        .arg("--output")
        .arg(&outcome_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Octave Ridge REFIT failed: {} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let package =
        InitialFullRefitPackage::from_json(&std::fs::read_to_string(&package_path).unwrap())
            .unwrap();
    assert_eq!(package.artifacts.len(), 1);
    assert_eq!(
        package.artifacts[0].record.artifact.backend,
        Some(dag_ml_core::ArtifactBackend::Mat)
    );
    assert!(sidecar.is_file());
    let outcome: Value = serde_json::from_slice(&std::fs::read(&outcome_path).unwrap()).unwrap();
    let mut replay_envelope: ExternalDataPlanEnvelope = serde_json::from_value(envelope).unwrap();
    let heldout: SampleRelationSet = serde_json::from_value(json!({"records": [
        {"observation_id": "obs.h1", "sample_id": "heldout:1", "target_id": "target:h1", "group_id": "group:h", "origin_sample_id": null, "source_id": "nir", "is_augmented": false},
        {"observation_id": "obs.h2", "sample_id": "heldout:2", "target_id": "target:h2", "group_id": "group:h", "origin_sample_id": null, "source_id": "nir", "is_augmented": false}
    ]})).unwrap();
    replay_envelope.schema_version = EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2;
    replay_envelope.predict_cohort = Some(
        PredictCohort::from_relations(
            PredictCohortRole::ExternalTest,
            heldout,
            vec!["y".into()],
            "a".repeat(64),
            Some("b".repeat(64)),
        )
        .unwrap(),
    );
    let replay_envelope_path = work.join("replay-envelope.json");
    let handles_path = work.join("artifact-handles.json");
    let output_ids_path = work.join("output-ids.json");
    let replay_path = work.join("replay-outcome.json");
    write_json(&replay_envelope_path, &replay_envelope);
    write_json(
        &handles_path,
        &outcome["node_results"][0]["artifact_handles"],
    );
    write_json(&output_ids_path, &json!([package.outputs[0].output_id]));
    let replay = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
        .current_dir(repo)
        .env("DAGML_OCTAVE_HPO_DATA", data)
        .env("DAGML_OCTAVE_RIDGE_SIDECAR", &sidecar)
        .args(["run-process-initial-full-refit-predict", "--package"])
        .arg(&package_path)
        .arg("--envelope")
        .arg(&replay_envelope_path)
        .arg("--adapter")
        .arg(&adapter)
        .arg("--artifact-handles")
        .arg(&handles_path)
        .arg("--output-ids")
        .arg(&output_ids_path)
        .arg("--output")
        .arg(&replay_path)
        .output()
        .unwrap();
    assert!(
        replay.status.success(),
        "Octave Ridge fresh-process replay failed: {} {}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
    let replay: Value = serde_json::from_slice(&std::fs::read(&replay_path).unwrap()).unwrap();
    assert_eq!(
        replay["replay_outcome"]["outputs"][0]["prediction"]["sample_ids"],
        json!(["heldout:1", "heldout:2"])
    );
    assert_eq!(
        replay["replay_outcome"]["outputs"][0]["prediction"]["values"],
        json!([[3.0], [4.0]])
    );
    std::fs::write(&sidecar, b"tampered MAT").unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
        .current_dir(repo)
        .env("DAGML_OCTAVE_HPO_DATA", data)
        .env("DAGML_OCTAVE_RIDGE_SIDECAR", &sidecar)
        .args(["run-process-initial-full-refit-predict", "--package"])
        .arg(&package_path)
        .arg("--envelope")
        .arg(&replay_envelope_path)
        .arg("--adapter")
        .arg(&adapter)
        .arg("--artifact-handles")
        .arg(&handles_path)
        .arg("--output-ids")
        .arg(&output_ids_path)
        .output()
        .unwrap();
    assert!(
        !rejected.status.success(),
        "Octave Ridge replay accepted a corrupted MAT sidecar"
    );
}
