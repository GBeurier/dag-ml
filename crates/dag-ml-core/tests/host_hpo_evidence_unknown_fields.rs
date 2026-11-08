//! Common HPO evidence shapes share the same closed owner DTOs.
use dag_ml_core::{
    HostHpoPrunedTrialEvidence, HostHpoSearchResult, HostHpoTrialEvidence,
    MethodsHpoCompletedReport, RegressionMetricReport, ScoreSet,
};
use serde_json::{json, Value};

fn scores() -> Value {
    serde_json::from_str(include_str!("../../../examples/fixtures/score_set.json")).unwrap()
}

fn trial() -> Value {
    json!({"trial_index": 0, "params": {"nested": {"kept": true}}, "score": 1.0,
        "variant_id": "variant:one", "scores": scores(),
        "generated_view_manifest": {"arbitrary": [1, 2]}, "objective_fold_scores": {"fold": -0.0}})
}

#[test]
fn complete_pruned_scores_reports_and_open_maps() {
    let value = trial();
    let decoded: HostHpoTrialEvidence = serde_json::from_value(value.clone()).unwrap();
    let encoded = serde_json::to_value(decoded).unwrap();
    assert_eq!(encoded["params"], value["params"]);
    assert_eq!(
        encoded["generated_view_manifest"],
        value["generated_view_manifest"]
    );
    for path in [0, 1, 2] {
        let mut changed = value.clone();
        let target = match path {
            0 => &mut changed,
            1 => &mut changed["scores"],
            _ => &mut changed["scores"]["reports"][0],
        };
        target["unknown_probe"] = json!(true);
        assert!(serde_json::from_value::<HostHpoTrialEvidence>(changed)
            .unwrap_err()
            .to_string()
            .contains("unknown field"));
    }
    let mut pruned = value.clone();
    pruned.as_object_mut().unwrap().remove("score");
    pruned
        .as_object_mut()
        .unwrap()
        .remove("objective_fold_scores");
    pruned["intermediate_scores"] = json!([1.0]);
    assert!(serde_json::from_value::<HostHpoPrunedTrialEvidence>(pruned.clone()).is_ok());
    for path in [0, 1, 2] {
        let mut changed = pruned.clone();
        let target = match path {
            0 => &mut changed,
            1 => &mut changed["scores"],
            _ => &mut changed["scores"]["reports"][0],
        };
        target["unknown_probe"] = json!(true);
        assert!(
            serde_json::from_value::<HostHpoPrunedTrialEvidence>(changed)
                .unwrap_err()
                .to_string()
                .contains("unknown field")
        );
    }
    let mut report = scores()["reports"][0].clone();
    assert!(serde_json::from_value::<RegressionMetricReport>(report.clone()).is_ok());
    report["unknown_probe"] = json!(1);
    let wrapper = json!({"trial_id": 1, "variant_id": "variant:one", "terminal_state": "completed", "score": 1.0, "report": report});
    assert!(serde_json::from_value::<MethodsHpoCompletedReport>(wrapper)
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
}

#[test]
fn search_result_remains_open_but_nested_evidence_is_closed() {
    let value = json!({"profile":"host_optimizer_search_v1", "portable":false, "request_fingerprint":"r", "graph_fingerprint":"g", "controller_fingerprint":"c", "campaign_fingerprint":"c", "fold_set_fingerprint":"f", "trials":[trial()], "selected_trial_index":0, "selected_params":{}, "status":"completed", "checkpoint":{}});
    assert!(serde_json::from_value::<HostHpoSearchResult>(value.clone()).is_ok());
    let mut changed = value.clone();
    changed["trials"][0]["unknown_probe"] = json!(1);
    assert!(serde_json::from_value::<HostHpoSearchResult>(changed)
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
    let mut pruned = trial();
    pruned.as_object_mut().unwrap().remove("score");
    pruned
        .as_object_mut()
        .unwrap()
        .remove("objective_fold_scores");
    pruned["intermediate_scores"] = json!([1.0]);
    let mut with_pruned = value;
    with_pruned["pruned_trials"] = json!([pruned]);
    assert!(serde_json::from_value::<HostHpoSearchResult>(with_pruned.clone()).is_ok());
    with_pruned["pruned_trials"][0]["unknown_probe"] = json!(1);
    assert!(serde_json::from_value::<HostHpoSearchResult>(with_pruned)
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
}

#[test]
fn supported_score_versions_keep_optional_controls() {
    let mut value = scores();
    value["reports"][0]["level"] = json!("group");
    value["reports"][0]["grouping_key"] =
        json!({"kind":"relation_metadata","key":"independent_unit_id"});
    let v1: ScoreSet = serde_json::from_value(value.clone()).unwrap();
    assert!(v1
        .validate()
        .unwrap_err()
        .to_string()
        .contains("V1 reports must not carry grouping_key"));
    value["reports"][0]["grouping_key"] = Value::Null;
    let null_key: ScoreSet = serde_json::from_value(value.clone()).unwrap();
    assert!(null_key.validate().is_ok());
    value.as_object_mut().unwrap().remove("schema_version");
    assert_eq!(
        serde_json::from_value::<ScoreSet>(value.clone())
            .unwrap()
            .schema_version,
        1
    );
    value["schema_version"] = json!(2);
    for report in value["reports"].as_array_mut().unwrap() {
        report["producer_port"] = json!("oof");
    }
    value["reports"][0]["grouping_key"] =
        json!({"kind":"relation_metadata","key":"independent_unit_id"});
    assert!(serde_json::from_value::<ScoreSet>(value)
        .unwrap()
        .validate()
        .is_ok());
}
