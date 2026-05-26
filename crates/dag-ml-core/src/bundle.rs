use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::data::ExternalDataPlanEnvelope;
use crate::error::{DagMlError, Result};
use crate::ids::{BundleId, ControllerId, FoldId, NodeId, SampleId, VariantId};
use crate::oof::PredictionPartition;
use crate::phase::Phase;
use crate::plan::ExecutionPlan;
use crate::runtime::ArtifactRef;
use crate::selection::SelectionDecision;

pub const EXECUTION_BUNDLE_SCHEMA_VERSION: u32 = 1;

fn default_execution_bundle_schema_version() -> u32 {
    EXECUTION_BUNDLE_SCHEMA_VERSION
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BundleDataRequirement {
    pub node_id: NodeId,
    pub input_name: String,
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    pub output_representation: String,
    #[serde(default)]
    pub feature_set_id: Option<String>,
}

impl BundleDataRequirement {
    pub fn key(&self) -> String {
        format!("{}.{}", self.node_id, self.input_name)
    }

    pub fn validate(&self) -> Result<()> {
        if self.input_name.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "bundle data requirement for `{}` has empty input_name",
                self.node_id
            )));
        }
        validate_fingerprint("schema", &self.schema_fingerprint)?;
        validate_fingerprint("plan", &self.plan_fingerprint)?;
        if let Some(relation_fingerprint) = &self.relation_fingerprint {
            validate_fingerprint("relation", relation_fingerprint)?;
        }
        if self.output_representation.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "bundle data requirement `{}` has empty output representation",
                self.key()
            )));
        }
        if let Some(feature_set_id) = &self.feature_set_id {
            if feature_set_id.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "bundle data requirement `{}` has empty feature_set_id",
                    self.key()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundlePredictionRequirement {
    pub producer_node: NodeId,
    pub source_port: String,
    pub consumer_node: NodeId,
    pub target_port: String,
    pub partition: PredictionPartition,
    #[serde(default)]
    pub fold_ids: Vec<FoldId>,
    pub sample_ids: Vec<SampleId>,
    pub prediction_width: usize,
    pub target_names: Vec<String>,
}

impl BundlePredictionRequirement {
    pub fn key(&self) -> String {
        bundle_prediction_requirement_key(
            &self.producer_node,
            &self.source_port,
            &self.consumer_node,
            &self.target_port,
        )
    }

    pub fn validate(&self) -> Result<()> {
        validate_non_empty("source_port", &self.source_port)?;
        validate_non_empty("target_port", &self.target_port)?;
        if self.partition != PredictionPartition::Validation {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle prediction requirement `{}` must use validation OOF predictions",
                self.key()
            )));
        }
        validate_unique_ids("fold id", &self.fold_ids)?;
        validate_unique_ids("sample id", &self.sample_ids)?;
        if self.sample_ids.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle prediction requirement `{}` has no sample ids",
                self.key()
            )));
        }
        if self.prediction_width == 0 {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle prediction requirement `{}` has zero prediction width",
                self.key()
            )));
        }
        if self.target_names.len() != self.prediction_width {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle prediction requirement `{}` target name count does not match prediction width",
                self.key()
            )));
        }
        for target_name in &self.target_names {
            validate_non_empty("target_name", target_name)?;
        }
        Ok(())
    }
}

pub fn bundle_prediction_requirement_key(
    producer_node: &NodeId,
    source_port: &str,
    consumer_node: &NodeId,
    target_port: &str,
) -> String {
    format!("{producer_node}.{source_port}->{consumer_node}.{target_port}")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RefitArtifactRecord {
    pub node_id: NodeId,
    pub controller_id: ControllerId,
    pub artifact: ArtifactRef,
    pub params_fingerprint: String,
    #[serde(default)]
    pub data_requirement_keys: Vec<String>,
    #[serde(default)]
    pub prediction_requirement_keys: Vec<String>,
}

impl RefitArtifactRecord {
    pub fn validate(&self) -> Result<()> {
        if self.artifact.id.as_str().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "refit artifact for `{}` has empty artifact id",
                self.node_id
            )));
        }
        if self.artifact.kind.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "refit artifact `{}` has empty artifact kind",
                self.artifact.id
            )));
        }
        if self.artifact.controller_id != self.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "refit artifact `{}` controller `{}` does not match record controller `{}`",
                self.artifact.id, self.artifact.controller_id, self.controller_id
            )));
        }
        validate_fingerprint("params", &self.params_fingerprint)?;
        let mut seen_keys = BTreeSet::new();
        for key in &self.data_requirement_keys {
            if key.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "refit artifact `{}` has empty data requirement key",
                    self.artifact.id
                )));
            }
            if !seen_keys.insert(key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "refit artifact `{}` has duplicate data requirement key `{key}`",
                    self.artifact.id
                )));
            }
        }
        let mut seen_prediction_keys = BTreeSet::new();
        for key in &self.prediction_requirement_keys {
            if key.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "refit artifact `{}` has empty prediction requirement key",
                    self.artifact.id
                )));
            }
            if !seen_prediction_keys.insert(key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "refit artifact `{}` has duplicate prediction requirement key `{key}`",
                    self.artifact.id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionBundle {
    pub bundle_id: BundleId,
    #[serde(default = "default_execution_bundle_schema_version")]
    pub schema_version: u32,
    pub plan_id: String,
    pub graph_fingerprint: String,
    pub campaign_fingerprint: String,
    pub controller_fingerprint: String,
    #[serde(default)]
    pub selected_variant_id: Option<VariantId>,
    #[serde(default)]
    pub selections: BTreeMap<String, SelectionDecision>,
    #[serde(default)]
    pub refit_artifacts: Vec<RefitArtifactRecord>,
    #[serde(default)]
    pub prediction_requirements: Vec<BundlePredictionRequirement>,
    #[serde(default)]
    pub data_requirements: Vec<BundleDataRequirement>,
    #[serde(default)]
    pub unsafe_flags: BTreeSet<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl ExecutionBundle {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != EXECUTION_BUNDLE_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` uses unsupported schema_version {}, expected {}",
                self.bundle_id, self.schema_version, EXECUTION_BUNDLE_SCHEMA_VERSION
            )));
        }
        if self.plan_id.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` has empty plan_id",
                self.bundle_id
            )));
        }
        validate_fingerprint("graph", &self.graph_fingerprint)?;
        validate_fingerprint("campaign", &self.campaign_fingerprint)?;
        validate_fingerprint("controller", &self.controller_fingerprint)?;
        for (key, decision) in &self.selections {
            if key.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` contains empty selection key",
                    self.bundle_id
                )));
            }
            decision.validate()?;
        }
        let mut data_keys = BTreeMap::new();
        for requirement in &self.data_requirements {
            requirement.validate()?;
            let key = requirement.key();
            if data_keys.insert(key.clone(), requirement).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` has duplicate data requirement `{}`",
                    self.bundle_id, key
                )));
            }
        }
        let mut prediction_keys = BTreeMap::new();
        for requirement in &self.prediction_requirements {
            requirement.validate()?;
            let key = requirement.key();
            if prediction_keys.insert(key.clone(), requirement).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` has duplicate prediction requirement `{}`",
                    self.bundle_id, key
                )));
            }
        }
        for artifact in &self.refit_artifacts {
            artifact.validate()?;
            for key in &artifact.data_requirement_keys {
                match data_keys.get(key) {
                    Some(requirement) if requirement.node_id == artifact.node_id => {}
                    Some(requirement) => {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "refit artifact `{}` for `{}` references data requirement `{key}` owned by `{}`",
                            artifact.artifact.id, artifact.node_id, requirement.node_id
                        )));
                    }
                    None => {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "refit artifact `{}` references unknown data requirement `{key}`",
                            artifact.artifact.id
                        )));
                    }
                }
            }
            for key in &artifact.prediction_requirement_keys {
                match prediction_keys.get(key) {
                    Some(requirement) if requirement.consumer_node == artifact.node_id => {}
                    Some(requirement) => {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "refit artifact `{}` for `{}` references prediction requirement `{key}` consumed by `{}`",
                            artifact.artifact.id, artifact.node_id, requirement.consumer_node
                        )));
                    }
                    None => {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "refit artifact `{}` references unknown prediction requirement `{key}`",
                            artifact.artifact.id
                        )));
                    }
                }
            }
        }
        for unsafe_flag in &self.unsafe_flags {
            if unsafe_flag.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` contains an empty unsafe flag",
                    self.bundle_id
                )));
            }
        }
        Ok(())
    }

    pub fn validate_against_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        self.validate()?;
        plan.validate()?;
        if self.plan_id != plan.id {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` plan_id `{}` does not match plan `{}`",
                self.bundle_id, self.plan_id, plan.id
            )));
        }
        if self.graph_fingerprint != plan.graph_fingerprint
            || self.campaign_fingerprint != plan.campaign_fingerprint
            || self.controller_fingerprint != plan.controller_fingerprint
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` fingerprints do not match execution plan",
                self.bundle_id
            )));
        }
        if let Some(selected_variant_id) = &self.selected_variant_id {
            if !plan
                .variants
                .iter()
                .any(|variant| &variant.variant_id == selected_variant_id)
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` selected unknown variant `{selected_variant_id}`",
                    self.bundle_id
                )));
            }
        }
        let expected_requirements = collect_data_requirements(plan)?;
        if self.data_requirements != expected_requirements {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` data requirements do not match execution plan",
                self.bundle_id
            )));
        }
        for artifact in &self.refit_artifacts {
            let node_plan = plan.node_plans.get(&artifact.node_id).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "bundle `{}` artifact references unknown node `{}`",
                    self.bundle_id, artifact.node_id
                ))
            })?;
            if artifact.controller_id != node_plan.controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` artifact controller for `{}` does not match plan",
                    self.bundle_id, artifact.node_id
                )));
            }
            if artifact.params_fingerprint != node_plan.params_fingerprint {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` artifact params for `{}` do not match plan",
                    self.bundle_id, artifact.node_id
                )));
            }
        }
        for requirement in &self.prediction_requirements {
            let edge_exists = plan.graph_plan.graph.edges.iter().any(|edge| {
                edge.source.node_id == requirement.producer_node
                    && edge.source.port_name == requirement.source_port
                    && edge.target.node_id == requirement.consumer_node
                    && edge.target.port_name == requirement.target_port
                    && edge.contract.requires_oof
            });
            if !edge_exists {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` prediction requirement `{}` does not match an OOF edge in the plan",
                    self.bundle_id,
                    requirement.key()
                )));
            }
        }
        Ok(())
    }

    pub fn validate_replay_envelopes(
        &self,
        envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    ) -> Result<()> {
        self.validate()?;
        for requirement in &self.data_requirements {
            let key = requirement.key();
            let envelope = envelopes.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "replay is missing external data envelope for `{key}`"
                ))
            })?;
            envelope.validate()?;
            if requirement.schema_fingerprint != envelope.schema_fingerprint
                || requirement.plan_fingerprint != envelope.plan_fingerprint
                || requirement.relation_fingerprint != envelope.relation_fingerprint
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "replay envelope for `{key}` does not match bundle data requirement"
                )));
            }
        }
        Ok(())
    }
}

pub fn build_execution_bundle(
    bundle_id: BundleId,
    plan: &ExecutionPlan,
    selected_variant_id: Option<VariantId>,
    selections: BTreeMap<String, SelectionDecision>,
    refit_artifacts: Vec<RefitArtifactRecord>,
) -> Result<ExecutionBundle> {
    build_execution_bundle_with_prediction_requirements(
        bundle_id,
        plan,
        selected_variant_id,
        selections,
        refit_artifacts,
        Vec::new(),
    )
}

pub fn build_execution_bundle_with_prediction_requirements(
    bundle_id: BundleId,
    plan: &ExecutionPlan,
    selected_variant_id: Option<VariantId>,
    selections: BTreeMap<String, SelectionDecision>,
    refit_artifacts: Vec<RefitArtifactRecord>,
    prediction_requirements: Vec<BundlePredictionRequirement>,
) -> Result<ExecutionBundle> {
    plan.validate()?;
    let bundle = ExecutionBundle {
        bundle_id,
        schema_version: EXECUTION_BUNDLE_SCHEMA_VERSION,
        plan_id: plan.id.clone(),
        graph_fingerprint: plan.graph_fingerprint.clone(),
        campaign_fingerprint: plan.campaign_fingerprint.clone(),
        controller_fingerprint: plan.controller_fingerprint.clone(),
        selected_variant_id,
        selections,
        refit_artifacts,
        prediction_requirements,
        data_requirements: collect_data_requirements(plan)?,
        unsafe_flags: BTreeSet::new(),
        metadata: BTreeMap::new(),
    };
    bundle.validate_against_plan(plan)?;
    Ok(bundle)
}

fn collect_data_requirements(plan: &ExecutionPlan) -> Result<Vec<BundleDataRequirement>> {
    let mut requirements = Vec::new();
    for node_plan in plan.node_plans.values() {
        for binding in &node_plan.data_bindings {
            requirements.push(BundleDataRequirement {
                node_id: node_plan.node_id.clone(),
                input_name: binding.input_name.clone(),
                schema_fingerprint: binding.schema_fingerprint.clone(),
                plan_fingerprint: binding.plan_fingerprint.clone(),
                relation_fingerprint: binding.relation_fingerprint.clone(),
                output_representation: binding.output_representation.clone(),
                feature_set_id: binding.feature_set_id.clone(),
            });
        }
    }
    requirements.sort_by_key(BundleDataRequirement::key);
    for requirement in &requirements {
        requirement.validate()?;
    }
    Ok(requirements)
}

fn validate_fingerprint(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::RuntimeValidation(format!(
            "{label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

fn validate_non_empty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(DagMlError::RuntimeValidation(format!("{label} is empty")));
    }
    Ok(())
}

fn validate_unique_ids<T>(label: &str, values: &[T]) -> Result<()>
where
    T: Ord + ToString,
{
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate {label} `{}`",
                value.to_string()
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayPhaseRequest {
    pub bundle_id: BundleId,
    pub phase: Phase,
    #[serde(default)]
    pub data_envelope_keys: Vec<String>,
}

impl ReplayPhaseRequest {
    pub fn validate_for_bundle(&self, bundle: &ExecutionBundle) -> Result<()> {
        bundle.validate()?;
        if self.bundle_id != bundle.bundle_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "replay request bundle `{}` does not match bundle `{}`",
                self.bundle_id, bundle.bundle_id
            )));
        }
        if !matches!(self.phase, Phase::Predict | Phase::Explain | Phase::Refit) {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle replay phase {:?} is not supported",
                self.phase
            )));
        }
        let expected = bundle
            .data_requirements
            .iter()
            .map(BundleDataRequirement::key)
            .collect::<BTreeSet<_>>();
        let mut requested = BTreeSet::new();
        for key in &self.data_envelope_keys {
            if key.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(
                    "replay request contains an empty data envelope key".to_string(),
                ));
            }
            if !requested.insert(key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "replay request contains duplicate data envelope key `{key}`"
                )));
            }
            if !expected.contains(key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "replay request references unknown data envelope key `{key}`"
                )));
            }
        }
        for requirement in &bundle.data_requirements {
            let key = requirement.key();
            if !requested.contains(key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "replay request is missing data envelope key `{key}`"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{ControllerManifest, ControllerRegistry};
    use crate::graph::GraphSpec;
    use crate::ids::{ArtifactId, FoldId, SampleId};
    use crate::plan::{build_execution_plan, CampaignSpec};
    use crate::selection::{
        select_candidate, CandidateScore, MetricObjective, SelectionMetric, SelectionPolicy,
    };

    fn plan() -> ExecutionPlan {
        let graph: GraphSpec =
            serde_json::from_str(include_str!("../../../examples/minimal_graph.json")).unwrap();
        let campaign: CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_oof_generation.json"
        ))
        .unwrap();
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(include_str!("../../../examples/controller_manifests.json"))
                .unwrap();
        let mut registry = ControllerRegistry::new();
        for manifest in manifests {
            registry.register(manifest).unwrap();
        }
        build_execution_plan("plan:bundle", graph, campaign, &registry).unwrap()
    }

    fn branch_merge_plan() -> ExecutionPlan {
        let graph: GraphSpec = serde_json::from_str(include_str!(
            "../../../examples/branch_merge_oof_graph.json"
        ))
        .unwrap();
        let campaign: CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_branch_merge_oof.json"
        ))
        .unwrap();
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(include_str!("../../../examples/controller_manifests.json"))
                .unwrap();
        let mut registry = ControllerRegistry::new();
        for manifest in manifests {
            registry.register(manifest).unwrap();
        }
        build_execution_plan("plan:branch.merge.bundle", graph, campaign, &registry).unwrap()
    }

    fn decision() -> SelectionDecision {
        select_candidate(
            &SelectionPolicy {
                id: "select:merge".to_string(),
                metric: SelectionMetric {
                    name: "rmse".to_string(),
                    objective: MetricObjective::Minimize,
                },
                require_finite: true,
            },
            &[
                CandidateScore {
                    candidate_id: "merge:a".to_string(),
                    metrics: BTreeMap::from([("rmse".to_string(), 2.0)]),
                    metadata: BTreeMap::new(),
                },
                CandidateScore {
                    candidate_id: "merge:b".to_string(),
                    metrics: BTreeMap::from([("rmse".to_string(), 1.0)]),
                    metadata: BTreeMap::new(),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn builds_bundle_from_execution_plan() {
        let plan = plan();
        let model_plan = plan
            .node_plans
            .get(&NodeId::new("model:base").unwrap())
            .unwrap();
        let artifact = RefitArtifactRecord {
            node_id: model_plan.node_id.clone(),
            controller_id: model_plan.controller_id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:model:base:refit").unwrap(),
                kind: "sklearn_pickle".to_string(),
                controller_id: model_plan.controller_id.clone(),
                size_bytes: Some(128),
            },
            params_fingerprint: model_plan.params_fingerprint.clone(),
            data_requirement_keys: vec!["model:base.x".to_string()],
            prediction_requirement_keys: Vec::new(),
        };

        let bundle = build_execution_bundle(
            BundleId::new("bundle:demo").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::from([("merge".to_string(), decision())]),
            vec![artifact],
        )
        .unwrap();

        bundle.validate_against_plan(&plan).unwrap();
        assert_eq!(bundle.data_requirements.len(), 1);
    }

    #[test]
    fn prediction_requirements_are_typed_and_validate_against_oof_edges() {
        let plan = branch_merge_plan();
        let meta_plan = plan
            .node_plans
            .get(&NodeId::new("merge:stack.pred_plus_original.meta:ridge").unwrap())
            .unwrap();
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("branch:b0.model:ridge").unwrap(),
            source_port: "oof".to_string(),
            consumer_node: meta_plan.node_id.clone(),
            target_port: "b0_oof".to_string(),
            partition: PredictionPartition::Validation,
            fold_ids: vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
            ],
            sample_ids: vec![
                SampleId::new("sample:1").unwrap(),
                SampleId::new("sample:2").unwrap(),
                SampleId::new("sample:3").unwrap(),
                SampleId::new("sample:4").unwrap(),
            ],
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let prediction_key = requirement.key();
        let artifact = RefitArtifactRecord {
            node_id: meta_plan.node_id.clone(),
            controller_id: meta_plan.controller_id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:merge:stack.pred_plus_original.meta:ridge:refit")
                    .unwrap(),
                kind: "mock_model".to_string(),
                controller_id: meta_plan.controller_id.clone(),
                size_bytes: Some(128),
            },
            params_fingerprint: meta_plan.params_fingerprint.clone(),
            data_requirement_keys: vec![
                "merge:stack.pred_plus_original.meta:ridge.x_original".to_string()
            ],
            prediction_requirement_keys: vec![prediction_key],
        };

        assert!(build_execution_bundle(
            BundleId::new("bundle:missing.prediction.requirement").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            vec![artifact.clone()],
        )
        .is_err());

        let bundle = build_execution_bundle_with_prediction_requirements(
            BundleId::new("bundle:typed.prediction.requirement").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            vec![artifact],
            vec![requirement],
        )
        .unwrap();
        bundle.validate_against_plan(&plan).unwrap();
        assert_eq!(bundle.prediction_requirements.len(), 1);
        assert_eq!(
            bundle.refit_artifacts[0].prediction_requirement_keys,
            vec!["branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"]
        );

        let mut wrong_data_owner = bundle.clone();
        wrong_data_owner.refit_artifacts[0].data_requirement_keys =
            vec!["branch:b0.model:ridge.x".to_string()];
        assert!(wrong_data_owner.validate().is_err());

        let mut wrong_prediction_consumer = bundle;
        wrong_prediction_consumer.refit_artifacts[0].node_id =
            NodeId::new("branch:b0.model:ridge").unwrap();
        wrong_prediction_consumer.refit_artifacts[0]
            .data_requirement_keys
            .clear();
        assert!(wrong_prediction_consumer.validate().is_err());
    }

    #[test]
    fn replay_envelopes_must_match_bundle_requirements() {
        let plan = plan();
        let bundle = build_execution_bundle(
            BundleId::new("bundle:demo").unwrap(),
            &plan,
            None,
            BTreeMap::from([("merge".to_string(), decision())]),
            Vec::new(),
        )
        .unwrap();
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
        ))
        .unwrap();

        bundle
            .validate_replay_envelopes(&BTreeMap::from([(
                "model:base.x".to_string(),
                envelope.clone(),
            )]))
            .unwrap();

        let mut mismatched = envelope;
        mismatched.schema_fingerprint = "0".repeat(64);
        assert!(bundle
            .validate_replay_envelopes(&BTreeMap::from([("model:base.x".to_string(), mismatched,)]))
            .is_err());
    }

    #[test]
    fn rejects_unsupported_bundle_schema_version() {
        let mut bundle = build_execution_bundle(
            BundleId::new("bundle:demo").unwrap(),
            &plan(),
            None,
            BTreeMap::from([("merge".to_string(), decision())]),
            Vec::new(),
        )
        .unwrap();
        bundle.schema_version = EXECUTION_BUNDLE_SCHEMA_VERSION + 1;

        assert!(bundle.validate().is_err());
    }

    #[test]
    fn replay_request_requires_predict_explain_or_refit_phase() {
        let bundle = build_execution_bundle(
            BundleId::new("bundle:demo").unwrap(),
            &plan(),
            None,
            BTreeMap::from([("merge".to_string(), decision())]),
            Vec::new(),
        )
        .unwrap();

        ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Predict,
            data_envelope_keys: vec!["model:base.x".to_string()],
        }
        .validate_for_bundle(&bundle)
        .unwrap();
        assert!(ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::FitCv,
            data_envelope_keys: vec!["model:base.x".to_string()],
        }
        .validate_for_bundle(&bundle)
        .is_err());
        assert!(ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Predict,
            data_envelope_keys: vec!["model:base.x".to_string(), "model:base.x".to_string()],
        }
        .validate_for_bundle(&bundle)
        .is_err());
        assert!(ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Predict,
            data_envelope_keys: vec!["model:base.y".to_string()],
        }
        .validate_for_bundle(&bundle)
        .is_err());
    }
}
