//! Closed, bundle-backed terminal PREDICT execution.
//!
//! This module is intentionally narrow.  It consumes the selected variant and
//! REFIT artifacts already captured in an [`ExecutionBundle`], runs exactly one
//! scheduler-owned PREDICT replay, and attests one explicitly selected terminal
//! prediction port against a V2 [`PredictCohort`].  It never asks a host binding
//! to recover a model by key, replay a Python-side model store, or substitute a
//! cohort after REFIT.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::aggregation::{AggregatedPredictionBlock, ObservationPredictionBlock};
use crate::bundle::{ExecutionBundle, RefitArtifactRecord, ReplayPhaseRequest};
use crate::campaign::stable_json_fingerprint;
use crate::controller::ControllerCapability;
use crate::data::{
    ExternalDataPlanEnvelope, PredictCohort, EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2,
};
use crate::error::{DagMlError, Result};
use crate::graph::PortKind;
use crate::ids::{BundleId, NodeId, VariantId};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::phase::Phase;
use crate::plan::ExecutionPlan;
use crate::policy::PredictionLevel;
use crate::runtime::{
    BundleReplayExecution, RunContext, RuntimeArtifactStore, RuntimeControllerRegistry,
    RuntimeDataProvider, SequentialScheduler,
};

/// First public receipt shape for the V2 terminal-prediction boundary.
pub const TERMINAL_PREDICTION_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Explicit terminal output selected by the caller.
///
/// A node alone is not enough: graphs may expose more than one prediction
/// output, so the selected producer port is an integrity boundary as well.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalPredictionSelector {
    pub node_id: NodeId,
    pub port: String,
}

impl TerminalPredictionSelector {
    pub fn new(node_id: NodeId, port: impl Into<String>) -> Result<Self> {
        let selector = Self {
            node_id,
            port: port.into(),
        };
        selector.validate()?;
        Ok(selector)
    }

    pub fn validate(&self) -> Result<()> {
        if self.port.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "terminal prediction selector has an empty port".to_string(),
            ));
        }
        Ok(())
    }
}

/// Durable attestation for one terminal PREDICT result.
///
/// The receipt contains only logical artifact references from the bundle; it
/// deliberately never serializes invocation-local `HandleRef` values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalPredictionReceipt {
    pub schema_version: u32,
    pub bundle_id: BundleId,
    pub plan_id: String,
    pub graph_fingerprint: String,
    pub campaign_fingerprint: String,
    pub controller_fingerprint: String,
    pub selected_variant_id: VariantId,
    pub terminal_node_id: NodeId,
    pub terminal_port: String,
    pub cohort_fingerprint: String,
    pub refit_artifacts: Vec<RefitArtifactRecord>,
    pub output_fingerprint: String,
}

/// One fully attested terminal PREDICT outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalPredictionExecution {
    pub prediction: PredictionBlock,
    pub receipt: TerminalPredictionReceipt,
}

/// Runtime resources for one closed terminal PREDICT replay.
///
/// This groups the borrowed execution boundary so the public API remains
/// explicit without leaking a positional list of host/runtime dependencies.
pub struct TerminalPredictionReplay<'a> {
    pub plan: &'a ExecutionPlan,
    pub bundle: &'a ExecutionBundle,
    pub envelope: &'a ExternalDataPlanEnvelope,
    pub selector: &'a TerminalPredictionSelector,
    pub controllers: &'a RuntimeControllerRegistry,
    pub data_provider: &'a dyn RuntimeDataProvider,
    pub artifact_store: &'a dyn RuntimeArtifactStore,
}

/// Require the V2, separately attested PREDICT cohort used by this API.
///
/// This is intentionally stricter than generic envelope validation: valid V1
/// envelopes remain supported by frozen CV/REFIT interfaces, but cannot enter
/// this terminal PREDICT route.
pub fn require_terminal_predict_cohort(
    envelope: &ExternalDataPlanEnvelope,
) -> Result<&PredictCohort> {
    if envelope.schema_version != EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2 {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT requires external data-plan envelope V2, got V{}",
            envelope.schema_version
        )));
    }
    envelope.validate()?;
    envelope.predict_cohort.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(
            "terminal PREDICT requires a V2 envelope with predict_cohort".to_string(),
        )
    })
}

/// Validate the closed V2 request before any PREDICT controller is invoked.
///
/// The selected output must be a graph-terminal prediction port.  This first
/// slice intentionally supports direct sample-level predictions only; relation
/// aggregation is refused rather than accidentally using the CV coordinator
/// relation universe for an external cohort.
pub fn validate_terminal_prediction_request(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    envelope: &ExternalDataPlanEnvelope,
    selector: &TerminalPredictionSelector,
) -> Result<()> {
    let _cohort = require_terminal_predict_cohort(envelope)?;
    selector.validate()?;
    plan.validate()?;
    bundle.validate_against_plan(plan)?;

    let _selected_variant = bundle.selected_variant_id.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "terminal PREDICT requires bundle `{}` to select one variant",
            bundle.bundle_id
        ))
    })?;

    let node_plan = plan.node_plans.get(&selector.node_id).ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector references unknown node `{}`",
            selector.node_id
        ))
    })?;
    if !node_plan.supported_phases.contains(&Phase::Predict) {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT node `{}` does not support PREDICT",
            selector.node_id
        )));
    }
    if !node_plan
        .controller_capabilities
        .contains(&ControllerCapability::EmitsPredictions)
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT node `{}` lacks the emits_predictions capability",
            selector.node_id
        )));
    }

    let graph_node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == selector.node_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "terminal PREDICT selector node `{}` is absent from the graph",
                selector.node_id
            ))
        })?;
    let output = graph_node
        .ports
        .outputs
        .iter()
        .find(|port| port.name == selector.port)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "terminal PREDICT selector `{}` has no output port `{}`",
                selector.node_id, selector.port
            ))
        })?;
    if output.kind != PortKind::Prediction {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` is not a prediction port",
            selector.node_id, selector.port
        )));
    }
    if plan.graph_plan.graph.edges.iter().any(|edge| {
        edge.source.node_id == selector.node_id && edge.source.port_name == selector.port
    }) {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` is consumed by another graph node",
            selector.node_id, selector.port
        )));
    }

    if node_plan.shape_plan.as_ref().is_some_and(|shape_plan| {
        shape_plan.aggregation_policy.aggregation_level != PredictionLevel::Sample
    }) {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` uses unsupported non-sample aggregation",
            selector.node_id, selector.port
        )));
    }

    if bundle.data_requirements.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT bundle `{}` has no relation-attested data requirement",
            bundle.bundle_id
        )));
    }
    let envelopes = terminal_prediction_envelopes(bundle, envelope);
    bundle.validate_replay_envelopes(&envelopes)?;

    for stateful_node in plan.node_plans.values().filter(|node| {
        node.supported_phases.contains(&Phase::Predict)
            && node
                .controller_capabilities
                .contains(&ControllerCapability::Stateful)
    }) {
        if !bundle
            .refit_artifacts
            .iter()
            .any(|artifact| artifact.node_id == stateful_node.node_id)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "terminal PREDICT stateful node `{}` has no REFIT artifact in bundle `{}`",
                stateful_node.node_id, bundle.bundle_id
            )));
        }
    }
    Ok(())
}

/// Execute one scheduler-owned PREDICT replay from a captured bundle.
///
/// `artifact_store` is the runtime's REFIT artifact store.  It is the only
/// model source used here: no Python model registry, model refetch, or host
/// replay fallback is consulted by this API.
pub fn execute_terminal_prediction(
    replay: TerminalPredictionReplay<'_>,
    ctx: &mut RunContext,
) -> Result<TerminalPredictionExecution> {
    let TerminalPredictionReplay {
        plan,
        bundle,
        envelope,
        selector,
        controllers,
        data_provider,
        artifact_store,
    } = replay;
    validate_terminal_prediction_request(plan, bundle, envelope, selector)?;

    let data_envelopes = terminal_prediction_envelopes(bundle, envelope);
    let replay_request = ReplayPhaseRequest {
        bundle_id: bundle.bundle_id.clone(),
        phase: Phase::Predict,
        data_envelope_keys: data_envelopes.keys().cloned().collect(),
    };
    let results = SequentialScheduler.execute_bundle_replay(
        BundleReplayExecution {
            plan,
            bundle,
            replay_request: &replay_request,
            prediction_cache_store: None,
            controllers,
            data_provider,
            artifact_store,
            data_envelopes: &data_envelopes,
        },
        ctx,
    )?;

    let predictions = results
        .iter()
        .flat_map(|result| result.predictions.iter().cloned())
        .collect::<Vec<_>>();
    let observation_predictions = results
        .iter()
        .flat_map(|result| result.observation_predictions.iter().cloned())
        .collect::<Vec<_>>();
    let aggregated_predictions = results
        .iter()
        .flat_map(|result| result.aggregated_predictions.iter().cloned())
        .collect::<Vec<_>>();
    attest_terminal_prediction_output(
        plan,
        bundle,
        envelope,
        selector,
        &predictions,
        &observation_predictions,
        &aggregated_predictions,
    )
}

/// Attest scheduler-produced blocks as one exact, sample-level terminal result.
///
/// This remains public so non-Python runtime embeddings can use the same
/// receipt gate after invoking [`SequentialScheduler::execute_bundle_replay`].
/// Callers must pass the complete PREDICT blocks emitted for the replay, not a
/// host-filtered subset.
pub fn attest_terminal_prediction_output(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    envelope: &ExternalDataPlanEnvelope,
    selector: &TerminalPredictionSelector,
    predictions: &[PredictionBlock],
    observation_predictions: &[ObservationPredictionBlock],
    aggregated_predictions: &[AggregatedPredictionBlock],
) -> Result<TerminalPredictionExecution> {
    validate_terminal_prediction_request(plan, bundle, envelope, selector)?;
    let cohort = require_terminal_predict_cohort(envelope)?;

    if observation_predictions.iter().any(|block| {
        block.producer_node == selector.node_id
            && block.producer_port.as_deref() == Some(selector.port.as_str())
    }) || aggregated_predictions.iter().any(|block| {
        block.producer_node == selector.node_id
            && block.producer_port.as_deref() == Some(selector.port.as_str())
    }) {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` emitted unsupported aggregated predictions",
            selector.node_id, selector.port
        )));
    }

    let matching = predictions
        .iter()
        .filter(|block| {
            block.producer_node == selector.node_id
                && block.producer_port.as_deref() == Some(selector.port.as_str())
        })
        .collect::<Vec<_>>();
    let [prediction] = matching.as_slice() else {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` must emit exactly one prediction block, got {}",
            selector.node_id,
            selector.port,
            matching.len()
        )));
    };
    prediction.validate_content()?;
    if prediction.partition != PredictionPartition::Final || prediction.fold_id.is_some() {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` must emit a top-level Final prediction block",
            selector.node_id, selector.port
        )));
    }
    if prediction.sample_ids != cohort.physical_sample_ids {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` sample identities do not exactly match the V2 predict cohort",
            selector.node_id, selector.port
        )));
    }
    if prediction.target_names != cohort.target_names {
        return Err(DagMlError::RuntimeValidation(format!(
            "terminal PREDICT selector `{}.{}` target names do not exactly match the V2 predict cohort",
            selector.node_id, selector.port
        )));
    }

    let selected_variant_id = bundle.selected_variant_id.clone().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "terminal PREDICT bundle `{}` lost its selected variant during attestation",
            bundle.bundle_id
        ))
    })?;
    Ok(TerminalPredictionExecution {
        prediction: (*prediction).clone(),
        receipt: TerminalPredictionReceipt {
            schema_version: TERMINAL_PREDICTION_RECEIPT_SCHEMA_VERSION,
            bundle_id: bundle.bundle_id.clone(),
            plan_id: bundle.plan_id.clone(),
            graph_fingerprint: bundle.graph_fingerprint.clone(),
            campaign_fingerprint: bundle.campaign_fingerprint.clone(),
            controller_fingerprint: bundle.controller_fingerprint.clone(),
            selected_variant_id,
            terminal_node_id: selector.node_id.clone(),
            terminal_port: selector.port.clone(),
            cohort_fingerprint: cohort.cohort_fingerprint.clone(),
            refit_artifacts: bundle.refit_artifacts.clone(),
            output_fingerprint: stable_json_fingerprint(prediction)?,
        },
    })
}

fn terminal_prediction_envelopes(
    bundle: &ExecutionBundle,
    envelope: &ExternalDataPlanEnvelope,
) -> BTreeMap<String, ExternalDataPlanEnvelope> {
    bundle
        .data_requirements
        .iter()
        .map(|requirement| (requirement.key(), envelope.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::bundle::build_execution_bundle;
    use crate::controller::{ControllerManifest, ControllerRegistry};
    use crate::data::PredictCohortRole;
    use crate::graph::GraphSpec;
    use crate::ids::{ArtifactId, ControllerId};
    use crate::plan::{build_execution_plan, CampaignSpec};
    use crate::relation::SampleRelationSet;
    use crate::runtime::ArtifactRef;

    use super::*;

    const SCHEMA_FINGERPRINT: &str =
        "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b";
    const PLAN_FINGERPRINT: &str =
        "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d";

    fn terminal_fixture() -> (ExecutionPlan, ExecutionBundle, ExternalDataPlanEnvelope) {
        let cv_envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .expect("fixture envelope parses");
        assert_eq!(cv_envelope.schema_fingerprint, SCHEMA_FINGERPRINT);
        assert_eq!(cv_envelope.plan_fingerprint, PLAN_FINGERPRINT);

        let graph: GraphSpec = serde_json::from_str(
            r#"{
              "id": "graph:terminal.predict",
              "interface": {"inputs": [], "outputs": []},
              "nodes": [{
                "id": "model:terminal",
                "kind": "model",
                "operator": null,
                "params": {},
                "ports": {
                  "inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
                  "outputs": [{"name": "prediction", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}]
                },
                "metadata": {},
                "seed_label": null
              }],
              "edges": [],
              "search_space_fingerprint": null,
              "metadata": {}
            }"#,
        )
        .expect("terminal graph parses");
        let campaign: CampaignSpec = serde_json::from_str(&format!(
            r#"{{
              "id": "campaign:terminal.predict",
              "root_seed": 7,
              "leakage_policy": {{"split_unit": "sample", "forbid_origin_cross_fold": true,
                "allow_observation_split_with_shared_target": false, "require_group_ids": false, "unsafe_flags": []}},
              "aggregation_policy": {{"aggregation_level": "sample", "method": "mean", "weights": "none",
                "emit_parallel_metrics": true, "selection_metric_level": "sample",
                "store_raw_predictions": true, "store_aggregated_predictions": true}},
              "split_invocation": {{
                "id": "split:terminal.predict", "controller_id": null,
                "leakage_policy": {{"split_unit": "sample", "forbid_origin_cross_fold": true,
                  "allow_observation_split_with_shared_target": false, "require_group_ids": false, "unsafe_flags": []}},
                "params": {{}},
                "fold_set": {{
                  "id": "folds:terminal.predict", "sample_ids": ["sample:1", "sample:2"],
                  "folds": [
                    {{"fold_id": "fold:0", "train_sample_ids": ["sample:2"], "validation_sample_ids": ["sample:1"], "metadata": {{}}}},
                    {{"fold_id": "fold:1", "train_sample_ids": ["sample:1"], "validation_sample_ids": ["sample:2"], "metadata": {{}}}}
                  ], "sample_groups": {{}}
                }}
              }},
              "generation": {{"strategy": "none", "dimensions": [], "max_variants": 1}},
              "shape_plans": {{}},
              "data_bindings": {{"model:terminal": [{{
                "node_id": "model:terminal", "input_name": "x", "request_id": "nir-to-tabular",
                "schema_fingerprint": "{SCHEMA_FINGERPRINT}", "plan_fingerprint": "{PLAN_FINGERPRINT}",
                "relation_fingerprint": "{}", "output_representation": "tabular_numeric",
                "feature_set_id": "x", "source_ids": ["nir"], "require_relations": true
              }}]}},
              "metadata": {{}}
            }}"#,
            cv_envelope
                .relation_fingerprint
                .as_deref()
                .expect("fixture relation fingerprint"),
        ))
        .expect("terminal campaign parses");
        let manifest: ControllerManifest = serde_json::from_str(
            r#"{
              "controller_id": "controller:model",
              "controller_version": "0.1.0",
              "operator_kind": "model",
              "priority": 0,
              "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
              "input_ports": [],
              "output_ports": [],
              "data_requirements": null,
              "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions", "emits_artifacts", "stateful"],
              "fit_scope": "fold_train",
              "rng_policy": "uses_core_seed",
              "artifact_policy": "serializable"
            }"#,
        )
        .expect("terminal controller manifest parses");
        let mut controllers = ControllerRegistry::new();
        controllers.register(manifest).expect("manifest registers");
        let plan = build_execution_plan("plan:terminal.predict", graph, campaign, &controllers)
            .expect("terminal plan builds");

        let heldout_relations: SampleRelationSet = serde_json::from_str(
            r#"{
              "records": [
                {"observation_id": "obs.H001", "sample_id": "sample:heldout:1", "target_id": "target:heldout:1", "group_id": "group:heldout", "origin_sample_id": null, "source_id": "nir", "is_augmented": false},
                {"observation_id": "obs.H002", "sample_id": "sample:heldout:2", "target_id": "target:heldout:2", "group_id": "group:heldout", "origin_sample_id": null, "source_id": "nir", "is_augmented": false}
              ]
            }"#,
        )
        .expect("heldout relations parse");
        let cohort = PredictCohort::from_relations(
            PredictCohortRole::ExternalTest,
            heldout_relations,
            vec!["protein".to_string()],
            "a".repeat(64),
            Some("b".repeat(64)),
        )
        .expect("predict cohort builds");
        let mut envelope = cv_envelope;
        envelope.schema_version = EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2;
        envelope.predict_cohort = Some(cohort);
        envelope.validate().expect("V2 envelope validates");
        plan.campaign
            .validate_data_envelope_relations(&envelope)
            .expect("campaign accepts closed external cohort");

        let selected_variant_id = plan
            .variants
            .first()
            .expect("single base variant")
            .variant_id
            .clone();
        let node_plan = plan
            .node_plans
            .get(&NodeId::new("model:terminal").unwrap())
            .expect("terminal node plan");
        let artifact = RefitArtifactRecord {
            node_id: node_plan.node_id.clone(),
            controller_id: node_plan.controller_id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:model:terminal:refit").unwrap(),
                kind: "mock_model".to_string(),
                controller_id: ControllerId::new("controller:model").unwrap(),
                backend: None,
                uri: None,
                content_fingerprint: None,
                size_bytes: Some(1),
                plugin: None,
                plugin_version: None,
            },
            params_fingerprint: node_plan.params_fingerprint.clone(),
            training_loss_fingerprint: None,
            data_requirement_keys: vec!["model:terminal.x".to_string()],
            prediction_requirement_keys: Vec::new(),
        };
        let bundle = build_execution_bundle(
            BundleId::new("bundle:terminal.predict").unwrap(),
            &plan,
            Some(selected_variant_id),
            BTreeMap::new(),
            vec![artifact],
        )
        .expect("terminal bundle builds");
        (plan, bundle, envelope)
    }

    fn terminal_prediction(envelope: &ExternalDataPlanEnvelope) -> PredictionBlock {
        let cohort = envelope.predict_cohort.as_ref().expect("V2 cohort");
        PredictionBlock {
            prediction_id: Some("prediction:terminal".to_string()),
            producer_node: NodeId::new("model:terminal").unwrap(),
            producer_port: Some("prediction".to_string()),
            partition: PredictionPartition::Final,
            fold_id: None,
            sample_ids: cohort.physical_sample_ids.clone(),
            values: vec![vec![0.1], vec![0.2]],
            target_names: cohort.target_names.clone(),
        }
    }

    fn selector() -> TerminalPredictionSelector {
        TerminalPredictionSelector::new(NodeId::new("model:terminal").unwrap(), "prediction")
            .unwrap()
    }

    #[test]
    fn terminal_receipt_binds_exact_v2_cohort_and_logical_refit_artifact() {
        let (plan, bundle, envelope) = terminal_fixture();
        let prediction = terminal_prediction(&envelope);
        let execution = attest_terminal_prediction_output(
            &plan,
            &bundle,
            &envelope,
            &selector(),
            std::slice::from_ref(&prediction),
            &[],
            &[],
        )
        .expect("exact terminal prediction is accepted");
        assert_eq!(execution.prediction, prediction);
        assert_eq!(
            execution.receipt.cohort_fingerprint,
            envelope
                .predict_cohort
                .as_ref()
                .expect("V2 cohort")
                .cohort_fingerprint
        );
        assert_eq!(execution.receipt.refit_artifacts.len(), 1);
        assert_ne!(execution.receipt.output_fingerprint, "");
        let receipt_json = serde_json::to_value(&execution.receipt).unwrap();
        assert!(receipt_json.get("handle").is_none());
    }

    #[test]
    fn terminal_request_refuses_v1_or_v2_without_cohort() {
        let (plan, bundle, envelope) = terminal_fixture();
        let mut v1 = envelope.clone();
        v1.schema_version = crate::data::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V1;
        v1.predict_cohort = None;
        let error = validate_terminal_prediction_request(&plan, &bundle, &v1, &selector())
            .expect_err("V1 must not enter terminal PREDICT");
        assert!(error
            .to_string()
            .contains("requires external data-plan envelope V2"));

        let mut missing = envelope;
        missing.predict_cohort = None;
        let error = require_terminal_predict_cohort(&missing)
            .expect_err("V2 without a cohort must fail closed");
        assert!(error.to_string().contains("V2 requires predict_cohort"));
    }

    #[test]
    fn terminal_receipt_refuses_identity_mismatch_and_aggregation() {
        let (plan, bundle, envelope) = terminal_fixture();
        let mut altered = terminal_prediction(&envelope);
        altered.sample_ids.reverse();
        let error = attest_terminal_prediction_output(
            &plan,
            &bundle,
            &envelope,
            &selector(),
            &[altered],
            &[],
            &[],
        )
        .expect_err("reordered cohort identities must fail");
        assert!(error
            .to_string()
            .contains("sample identities do not exactly match"));

        let observation = ObservationPredictionBlock {
            prediction_id: Some("prediction:terminal:observation".to_string()),
            producer_node: NodeId::new("model:terminal").unwrap(),
            producer_port: Some("prediction".to_string()),
            partition: PredictionPartition::Final,
            fold_id: None,
            observation_ids: Vec::new(),
            values: Vec::new(),
            weights: Vec::new(),
            target_names: vec!["protein".to_string()],
        };
        let error = attest_terminal_prediction_output(
            &plan,
            &bundle,
            &envelope,
            &selector(),
            &[terminal_prediction(&envelope)],
            &[observation],
            &[],
        )
        .expect_err("aggregation output must not enter this first terminal slice");
        assert!(error
            .to_string()
            .contains("unsupported aggregated predictions"));
    }
}
