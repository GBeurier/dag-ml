//! Relation-attested FIT_CV phase execution and observation-to-sample scoring
//! for browser hosts. These are thin facades over existing core composition:
//! the same provider/scheduler/ScoreSet sequence the Python in-process host and
//! the CLI run, plus the public aggregation and regression scorer. No numerics
//! are implemented here.

use super::*;

use dag_ml_core::{
    aggregate_observation_predictions, plan_oof_partition_mode, score_regression_prediction_block,
    score_regression_prediction_block_with_profile, AggregationPolicy, ExplicitPhaseDataProvider,
    ExternalDataPlanEnvelope, ObservationPredictionBlock, PredictionBlock, RegressionMetricKind,
    RegressionReportProfile, RegressionTargetBlock, SampleRelationSet,
};

fn contract<T: DeserializeOwned + serde::Serialize>(json: &str, label: &str) -> CoreResult<T> {
    deserialize_external_contract(json, label, CoreDagMlError::CampaignValidation)
}

/// Execute FIT_CV of a built plan against an attested data envelope.
///
/// The envelope's coordinator relations make native observation-to-sample
/// aggregation, cross-fold OOF scoring and the returned `ScoreSet` available to
/// hosts whose controllers emit observation-level predictions. `variant_id`
/// must name one plan variant; the run context is scoped to it.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn execute_execution_plan_phase_with_envelope_json(
    execution_plan_json: &js_sys::JsString,
    trusted_controller_manifests_json: &js_sys::JsString,
    envelope_json: &js_sys::JsString,
    variant_id: &js_sys::JsString,
    run_id: &js_sys::JsString,
    root_seed: &js_sys::JsString,
    phase: &js_sys::JsString,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let execution_plan_json =
        crate::js_text::text_argument(execution_plan_json, "execution_plan_json")?;
    let trusted_controller_manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let envelope_json = crate::js_text::text_argument(envelope_json, "envelope_json")?;
    let variant_id = crate::js_text::text_argument(variant_id, "variant_id")?;
    let run_id = crate::js_text::text_argument(run_id, "run_id")?;
    let root_seed = crate::js_text::text_argument(root_seed, "root_seed")?;
    let phase = crate::js_text::text_argument(phase, "phase")?;
    if parse_phase(&phase)? != Phase::FitCv {
        return Err(js_core_error(CoreDagMlError::RuntimeValidation(
            "envelope phase execution supports FIT_CV only; REFIT uses execute_initial_full_refit_json".into(),
        )));
    }
    let seed = root_seed
        .parse::<u64>()
        .map_err(|_| JsValue::from_str("root_seed must be an exact decimal u64"))?;
    let plan = ExecutionPlan::from_json(&execution_plan_json).map_err(js_core_error)?;
    let trusted = controller_registry_from_json(&trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&plan, &trusted).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        contract(&envelope_json, "FIT_CV data envelope").map_err(js_core_error)?;
    plan.campaign
        .validate_data_envelope_relations(&envelope)
        .map_err(js_core_error)?;
    let variant = plan
        .variants
        .iter()
        .find(|variant| variant.variant_id.to_string() == variant_id)
        .map(|variant| variant.variant_id.clone())
        .ok_or_else(|| {
            js_core_error(CoreDagMlError::RuntimeValidation(format!(
                "variant `{variant_id}` is not a variant of plan `{}`",
                plan.id
            )))
        })?;
    let provider = ExplicitPhaseDataProvider::new(
        ControllerId::new("controller:data.provider").map_err(js_core_error)?,
        envelope,
        None,
    )
    .map_err(js_core_error)?;
    let controllers =
        initial_refit::controllers_for_plan(&plan, js_invoke).map_err(js_core_error)?;
    let mut ctx = RunContext::new(RunId::new(run_id).map_err(js_core_error)?, Some(seed));
    ctx.variant_id = Some(variant);
    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::FitCv,
        )
        .map_err(js_core_error)?;
    ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(&plan))
        .map_err(js_core_error)?;
    let scores = ctx.build_score_set(plan.id.clone(), None);
    serde_json::to_string(&serde_json::json!({"node_results": results, "scores": scores}))
        .map_err(js_serde_error)
}

/// Aggregate observation-level predictions to physical samples with an
/// existing `AggregationPolicy` and the attested sample relations.
#[wasm_bindgen]
pub fn aggregate_observation_predictions_json(
    block_json: &js_sys::JsString,
    relations_json: &js_sys::JsString,
    policy_json: &js_sys::JsString,
    requested_sample_order_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    let block_json = crate::js_text::text_argument(block_json, "block_json")?;
    let relations_json = crate::js_text::text_argument(relations_json, "relations_json")?;
    let policy_json = crate::js_text::text_argument(policy_json, "policy_json")?;
    let requested_sample_order_json =
        crate::js_text::text_argument(requested_sample_order_json, "requested_sample_order_json")?;
    aggregate_observation_predictions_json_impl(
        &block_json,
        &relations_json,
        &policy_json,
        &requested_sample_order_json,
    )
    .map_err(js_core_error)
}

fn aggregate_observation_predictions_json_impl(
    block_json: &str,
    relations_json: &str,
    policy_json: &str,
    requested_sample_order_json: &str,
) -> CoreResult<String> {
    let block: ObservationPredictionBlock = contract(block_json, "observation prediction block")?;
    let relations: SampleRelationSet = contract(relations_json, "sample relations")?;
    let policy: AggregationPolicy = contract(policy_json, "aggregation policy")?;
    let order: Vec<SampleId> = contract(requested_sample_order_json, "requested sample order")?;
    let aggregated = aggregate_observation_predictions(&block, &relations, &policy, &order)?;
    serde_json::to_string(&aggregated).map_err(CoreDagMlError::Serialization)
}

/// Score a sample-level prediction block against sample-level targets with the
/// named DAG metrics (mirrors `dagml_score_regression_prediction_block_json`).
#[wasm_bindgen]
pub fn score_regression_prediction_block_json(
    predictions_json: &js_sys::JsString,
    targets_json: &js_sys::JsString,
    metrics_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    let predictions_json = crate::js_text::text_argument(predictions_json, "predictions_json")?;
    let targets_json = crate::js_text::text_argument(targets_json, "targets_json")?;
    let metrics_json = crate::js_text::text_argument(metrics_json, "metrics_json")?;
    score_regression_prediction_block_json_impl(&predictions_json, &targets_json, &metrics_json)
        .map_err(js_core_error)
}

fn score_regression_prediction_block_json_impl(
    predictions_json: &str,
    targets_json: &str,
    metrics_json: &str,
) -> CoreResult<String> {
    let predictions: PredictionBlock = contract(predictions_json, "sample prediction block")?;
    let targets: RegressionTargetBlock = contract(targets_json, "regression target block")?;
    let metrics: Vec<RegressionMetricKind> = contract(metrics_json, "regression metric list")?;
    let report = score_regression_prediction_block(&predictions, &targets, &metrics)?;
    serde_json::to_string(&report).map_err(CoreDagMlError::Serialization)
}

/// Score known-reference predictions with an explicit versioned owner profile.
/// `profile_json` is a JSON string, e.g. `"regression_report.population_rmse_ratios.v1"`.
#[wasm_bindgen]
pub fn score_regression_prediction_block_with_profile_json(
    predictions_json: &js_sys::JsString,
    targets_json: &js_sys::JsString,
    profile_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    let predictions_json = crate::js_text::text_argument(predictions_json, "predictions_json")?;
    let targets_json = crate::js_text::text_argument(targets_json, "targets_json")?;
    let profile_json = crate::js_text::text_argument(profile_json, "profile_json")?;
    score_regression_prediction_block_with_profile_json_impl(
        &predictions_json,
        &targets_json,
        &profile_json,
    )
    .map_err(js_core_error)
}

fn score_regression_prediction_block_with_profile_json_impl(
    predictions_json: &str,
    targets_json: &str,
    profile_json: &str,
) -> CoreResult<String> {
    let predictions: PredictionBlock = contract(predictions_json, "sample prediction block")?;
    let targets: RegressionTargetBlock = contract(targets_json, "regression target block")?;
    let profile: RegressionReportProfile = contract(profile_json, "regression report profile")?;
    let report = score_regression_prediction_block_with_profile(&predictions, &targets, profile)?;
    serde_json::to_string(&report).map_err(CoreDagMlError::Serialization)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relations() -> String {
        serde_json::json!({"records": [
            {"observation_id": "s0", "sample_id": "p0"},
            {"observation_id": "s1", "sample_id": "p0"},
            {"observation_id": "s2", "sample_id": "p1"}
        ]})
        .to_string()
    }

    fn observations() -> String {
        serde_json::json!({
            "prediction_id": "pred:obs", "producer_node": "model:web.pipeline",
            "partition": "test", "fold_id": null,
            "observation_ids": ["s0", "s1", "s2"], "values": [[1.0], [3.0], [10.0]],
            "target_names": ["y"]
        })
        .to_string()
    }

    #[test]
    fn aggregates_repeated_observations_to_physical_samples() {
        let policy =
            serde_json::json!({"aggregation_level": "sample", "method": "mean"}).to_string();
        let block: PredictionBlock = serde_json::from_str(
            &aggregate_observation_predictions_json_impl(
                &observations(),
                &relations(),
                &policy,
                r#"["p0","p1"]"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            block.sample_ids,
            vec![SampleId::new("p0").unwrap(), SampleId::new("p1").unwrap()]
        );
        assert_eq!(block.values, vec![vec![2.0], vec![10.0]]);
        let none = serde_json::json!({"aggregation_level": "sample", "method": "none"}).to_string();
        assert!(aggregate_observation_predictions_json_impl(
            &observations(),
            &relations(),
            &none,
            r#"["p0","p1"]"#
        )
        .is_err());
    }

    #[test]
    fn scores_aggregated_samples_with_named_metrics() {
        let predictions = serde_json::json!({
            "producer_node": "model:web.pipeline", "partition": "test", "fold_id": null,
            "sample_ids": ["p0", "p1"], "values": [[2.0], [10.0]], "target_names": ["y"]
        })
        .to_string();
        let targets = serde_json::json!({
            "level": "sample",
            "unit_ids": [{"level": "sample", "id": "p0"}, {"level": "sample", "id": "p1"}],
            "values": [[1.0], [10.0]], "target_names": ["y"]
        })
        .to_string();
        let report: serde_json::Value = serde_json::from_str(
            &score_regression_prediction_block_json_impl(
                &predictions,
                &targets,
                r#"["rmse","mae","r2"]"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(report["row_count"], 2);
        assert_eq!(report["level"], "sample");
        assert!((report["metrics"]["mae"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert!((report["metrics"]["rmse"].as_f64().unwrap() - 0.5_f64.sqrt()).abs() < 1e-12);
        let exports = contract_manifest()["wasm_exports"].clone();
        for name in [
            "execute_execution_plan_phase_with_envelope_json",
            "aggregate_observation_predictions_json",
            "score_regression_prediction_block_json",
        ] {
            assert!(exports
                .as_array()
                .unwrap()
                .contains(&serde_json::json!(name)));
        }
    }

    #[test]
    fn profile_scoring_export_preserves_identity_zero_rmse_and_reasons() {
        let mut predictions = serde_json::json!({
            "prediction_id":"pred:replay", "producer_node":"model:web.pipeline",
            "partition":"final", "fold_id":null, "sample_ids":["p1","p0"],
            "values":[[3.0],[1.0]], "target_names":["y"]
        });
        let mut targets = serde_json::json!({
            "level":"sample", "unit_ids":[{"level":"sample","id":"p0"},{"level":"sample","id":"p1"}],
            "values":[[0.0],[2.0]], "target_names":["y"]
        });
        let profile_json = r#""regression_report.population_rmse_ratios.v1""#;
        let report: serde_json::Value = serde_json::from_str(
            &score_regression_prediction_block_with_profile_json_impl(
                &predictions.to_string(),
                &targets.to_string(),
                profile_json,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(report["prediction_id"], "pred:replay");
        assert_eq!(report["producer_node"], "model:web.pipeline");
        assert_eq!(report["partition"], "final");
        assert_eq!(report["row_count"], 2);
        assert_eq!(
            report["metric_profile"],
            "regression_report.population_rmse_ratios.v1"
        );
        // Reversed target identities must join by ID, never by row position.
        assert_eq!(
            report["metrics"],
            serde_json::json!({"rmse":1.0,"bias":1.0,"r2":0.0,"rpd":1.0,"rpiq":1.0})
        );
        assert!(report.get("undefined_metrics").is_none());
        predictions["values"] = serde_json::json!([[2.0], [2.0]]);
        targets["values"] = serde_json::json!([[2.0], [2.0]]);
        let report: serde_json::Value = serde_json::from_str(
            &score_regression_prediction_block_with_profile_json_impl(
                &predictions.to_string(),
                &targets.to_string(),
                profile_json,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            report["metrics"],
            serde_json::json!({"rmse":0.0,"bias":0.0})
        );
        assert_eq!(
            report["undefined_metrics"],
            serde_json::json!({
                "r2":"constant_target", "rpd":"zero_rmse", "rpiq":"zero_rmse"
            })
        );
        for invalid_profile in [
            r#""regression_report.population_rmse_ratios.v2""#,
            r#"{"profile":"regression_report.population_rmse_ratios.v1"}"#,
        ] {
            assert!(score_regression_prediction_block_with_profile_json_impl(
                &predictions.to_string(),
                &targets.to_string(),
                invalid_profile,
            )
            .is_err());
        }
        assert!(score_regression_prediction_block_with_profile_json_impl(
            &predictions.to_string(),
            "[]",
            profile_json,
        )
        .is_err());
        targets["unit_ids"][0]["id"] = serde_json::json!("missing-reference");
        assert!(score_regression_prediction_block_with_profile_json_impl(
            &predictions.to_string(),
            &targets.to_string(),
            profile_json,
        )
        .is_err());
        assert!(contract_manifest()["wasm_exports"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!(
                "score_regression_prediction_block_with_profile_json"
            )));
    }
}

/// Execute a whole ordered Cartesian bank with native FIT_CV + holdout SELECT.
/// No host candidate loop, REFIT, or synthetic cross-fold average participates.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn execute_single_holdout_bank_json(
    execution_plan_json: &js_sys::JsString,
    trusted_controller_manifests_json: &js_sys::JsString,
    envelope_json: &js_sys::JsString,
    request_json: &js_sys::JsString,
    run_id: &js_sys::JsString,
    root_seed: &js_sys::JsString,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let plan_json = crate::js_text::text_argument(execution_plan_json, "execution_plan_json")?;
    let manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let envelope_json = crate::js_text::text_argument(envelope_json, "envelope_json")?;
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    let run_id = crate::js_text::text_argument(run_id, "run_id")?;
    let root_seed = crate::js_text::text_argument(root_seed, "root_seed")?;
    let seed = root_seed
        .parse::<u64>()
        .map_err(|_| JsValue::from_str("root_seed must be an exact decimal u64"))?;
    let plan = ExecutionPlan::from_json(&plan_json).map_err(js_core_error)?;
    let trusted = controller_registry_from_json(&manifests_json)?;
    validate_runtime_controller_manifests(&plan, &trusted).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        contract(&envelope_json, "single holdout data envelope").map_err(js_core_error)?;
    plan.campaign
        .validate_data_envelope_relations(&envelope)
        .map_err(js_core_error)?;
    let request: dag_ml_core::SingleHoldoutBankRequest =
        contract(&request_json, "single holdout bank request").map_err(js_core_error)?;
    let provider = ExplicitPhaseDataProvider::new(
        ControllerId::new("controller:data.provider").map_err(js_core_error)?,
        envelope,
        None,
    )
    .map_err(js_core_error)?;
    let controllers =
        initial_refit::controllers_for_plan(&plan, js_invoke).map_err(js_core_error)?;
    let result = dag_ml_core::execute_single_holdout_bank(
        &plan,
        &request,
        &RunId::new(run_id).map_err(js_core_error)?,
        Some(seed),
        &controllers,
        &provider,
    )
    .map_err(js_core_error)?;
    serde_json::to_string(&result).map_err(js_serde_error)
}
