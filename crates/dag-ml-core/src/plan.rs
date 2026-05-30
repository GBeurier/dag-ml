use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::campaign::stable_json_fingerprint;
use crate::controller::{
    ArtifactPolicy, ControllerCapability, ControllerFitScope, ControllerManifest,
    ControllerRegistry, RngPolicy,
};
use crate::data::{BranchViewPlan, DataBinding, ExternalDataPlanEnvelope};
use crate::error::{DagMlError, Result};
use crate::fold::{FoldSet, NestedCvSpec};
use crate::generation::{
    enumerate_variants, generation_spec_fingerprint, GenerationSpec, VariantPlan,
};
use crate::graph::{GraphSpec, NodeKind};
use crate::ids::{ControllerId, FoldId, NodeId, VariantId};
use crate::phase::Phase;
use crate::policy::{AggregationPolicy, DataModelShapePlan, LeakageUnitPolicy};

pub const CAMPAIGN_SPEC_SCHEMA_VERSION: u32 = 1;
pub const CAMPAIGN_SPEC_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/campaign_spec.v1.schema.json";
pub const EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;
pub const EXECUTION_PLAN_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/execution_plan.v1.schema.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitInvocation {
    pub id: String,
    #[serde(default)]
    pub controller_id: Option<ControllerId>,
    #[serde(default)]
    pub leakage_policy: LeakageUnitPolicy,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub fold_set: Option<FoldSet>,
}

impl SplitInvocation {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "split invocation id is empty".to_string(),
            ));
        }
        self.leakage_policy.validate()?;
        if let Some(fold_set) = &self.fold_set {
            fold_set.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignSpec {
    pub id: String,
    pub root_seed: Option<u64>,
    #[serde(default)]
    pub leakage_policy: LeakageUnitPolicy,
    #[serde(default)]
    pub aggregation_policy: AggregationPolicy,
    #[serde(default)]
    pub split_invocation: Option<SplitInvocation>,
    #[serde(default)]
    pub generation: GenerationSpec,
    #[serde(default)]
    pub shape_plans: BTreeMap<NodeId, DataModelShapePlan>,
    #[serde(default)]
    pub data_bindings: BTreeMap<NodeId, Vec<DataBinding>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub branch_view_plans: Vec<BranchViewPlan>,
    /// Campaign-wide default nested (inner) CV policy. A node-level
    /// `NodePlan.inner_cv` overrides it; see [`crate::fold::resolve_inner_cv`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_cv: Option<NestedCvSpec>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl CampaignSpec {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "campaign id is empty".to_string(),
            ));
        }
        self.leakage_policy.validate()?;
        self.aggregation_policy.validate()?;
        if let Some(inner_cv) = &self.inner_cv {
            inner_cv.validate()?;
        }
        if let Some(split) = &self.split_invocation {
            split.validate()?;
        }
        self.generation.validate()?;
        for (node_id, shape_plan) in &self.shape_plans {
            if node_id != &shape_plan.node_id {
                return Err(DagMlError::CampaignValidation(format!(
                    "shape plan key `{node_id}` does not match node_id `{}`",
                    shape_plan.node_id
                )));
            }
            shape_plan.validate()?;
        }
        for (node_id, bindings) in &self.data_bindings {
            for binding in bindings {
                if node_id != &binding.node_id {
                    return Err(DagMlError::CampaignValidation(format!(
                        "data binding key `{node_id}` does not match node_id `{}`",
                        binding.node_id
                    )));
                }
                binding.validate()?;
            }
        }
        let mut branch_views = BTreeSet::new();
        for plan in &self.branch_view_plans {
            plan.validate()?;
            if !branch_views.insert(plan.view_id.as_str()) {
                return Err(DagMlError::CampaignValidation(format!(
                    "campaign `{}` contains duplicate branch view `{}`",
                    self.id, plan.view_id
                )));
            }
        }
        Ok(())
    }

    pub fn validate_data_envelope_relations(
        &self,
        envelope: &ExternalDataPlanEnvelope,
    ) -> Result<()> {
        envelope.validate()?;
        let Some(relations) = &envelope.coordinator_relations else {
            return Ok(());
        };
        let Some(split) = &self.split_invocation else {
            return Ok(());
        };
        let Some(fold_set) = &split.fold_set else {
            return Ok(());
        };
        relations.validate_against_fold_set(fold_set, &self.leakage_policy)?;
        relations.validate_against_fold_set(fold_set, &split.leakage_policy)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphPlan {
    pub graph: GraphSpec,
    pub topological_order: Vec<NodeId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parallel_levels: Vec<Vec<NodeId>>,
}

impl GraphPlan {
    pub fn from_graph(graph: GraphSpec) -> Result<Self> {
        let topological_order = graph.topological_order()?;
        let parallel_levels = graph.parallel_levels()?;
        Ok(Self {
            graph,
            topological_order,
            parallel_levels,
        })
    }

    pub fn parallel_levels(&self) -> Result<Vec<Vec<NodeId>>> {
        if self.parallel_levels.is_empty() {
            return self.graph.parallel_levels();
        }
        Ok(self.parallel_levels.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodePlan {
    pub node_id: NodeId,
    pub kind: NodeKind,
    pub controller_id: ControllerId,
    pub controller_version: String,
    pub supported_phases: BTreeSet<Phase>,
    #[serde(default)]
    pub controller_capabilities: BTreeSet<ControllerCapability>,
    pub fit_scope: ControllerFitScope,
    pub rng_policy: RngPolicy,
    pub artifact_policy: ArtifactPolicy,
    pub input_nodes: Vec<NodeId>,
    pub output_nodes: Vec<NodeId>,
    pub shape_plan: Option<DataModelShapePlan>,
    #[serde(default)]
    pub data_bindings: Vec<DataBinding>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, serde_json::Value>,
    /// Node-local nested (inner) CV policy (e.g. for a finetune/tuner or branch
    /// node); overrides the campaign-wide default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_cv: Option<NestedCvSpec>,
    pub params_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub id: String,
    pub graph_plan: GraphPlan,
    pub campaign: CampaignSpec,
    pub node_plans: BTreeMap<NodeId, NodePlan>,
    pub controller_manifests: BTreeMap<ControllerId, ControllerManifest>,
    pub variants: Vec<VariantPlan>,
    pub fold_set: Option<FoldSet>,
    pub graph_fingerprint: String,
    pub campaign_fingerprint: String,
    pub controller_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionScopePlan {
    pub scope_id: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub variant_seed: Option<u64>,
    pub fold_id: Option<FoldId>,
    pub node_levels: Vec<Vec<NodeId>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhaseExecutionSchedule {
    pub plan_id: String,
    pub phase: Phase,
    pub scopes: Vec<ExecutionScopePlan>,
}

impl ExecutionPlan {
    pub fn validate(&self) -> Result<()> {
        self.graph_plan.graph.validate()?;
        self.campaign.validate()?;
        if !self.graph_plan.parallel_levels.is_empty()
            && self.graph_plan.parallel_levels != self.graph_plan.graph.parallel_levels()?
        {
            return Err(DagMlError::Planning(
                "graph plan parallel levels do not match graph".to_string(),
            ));
        }
        if self.node_plans.len() != self.graph_plan.graph.nodes.len() {
            return Err(DagMlError::Planning(
                "execution plan node count does not match graph".to_string(),
            ));
        }
        for node_id in &self.graph_plan.topological_order {
            let plan = self.node_plans.get(node_id).ok_or_else(|| {
                DagMlError::Planning(format!("missing node plan for `{node_id}`"))
            })?;
            let manifest = self
                .controller_manifests
                .get(&plan.controller_id)
                .ok_or_else(|| {
                    DagMlError::Planning(format!(
                        "missing controller manifest `{}` for node `{node_id}`",
                        plan.controller_id
                    ))
                })?;
            if manifest.operator_kind != plan.kind {
                return Err(DagMlError::Planning(format!(
                    "node `{node_id}` planned with incompatible controller `{}`",
                    manifest.controller_id
                )));
            }
            if plan.controller_capabilities != manifest.capabilities {
                return Err(DagMlError::Planning(format!(
                    "node `{node_id}` controller capabilities do not match manifest `{}`",
                    manifest.controller_id
                )));
            }
            if plan.fit_scope != manifest.fit_scope
                || plan.rng_policy != manifest.rng_policy
                || plan.artifact_policy != manifest.artifact_policy
            {
                return Err(DagMlError::Planning(format!(
                    "node `{node_id}` controller policy fields do not match manifest `{}`",
                    manifest.controller_id
                )));
            }
            for binding in &plan.data_bindings {
                if binding.node_id != *node_id {
                    return Err(DagMlError::Planning(format!(
                        "node plan `{node_id}` contains data binding for `{}`",
                        binding.node_id
                    )));
                }
                binding.validate()?;
            }
            let actual_params_fingerprint = stable_json_fingerprint(&plan.params)?;
            if actual_params_fingerprint != plan.params_fingerprint {
                return Err(DagMlError::Planning(format!(
                    "node plan `{node_id}` params fingerprint does not match params"
                )));
            }
        }
        // Validate every node-local inner_cv over ALL node plans (not just the
        // cached topological order): a hand-loaded ExecutionPlan JSON with a
        // stale/tampered order could omit a FIT_CV node from that order while
        // still scheduling it via parallel levels, so a malformed inner_cv must
        // be refused here rather than deferred to FIT_CV fold building.
        for (node_id, plan) in &self.node_plans {
            if let Some(inner_cv) = &plan.inner_cv {
                inner_cv.validate().map_err(|error| {
                    DagMlError::Planning(format!(
                        "node plan `{node_id}` has invalid inner_cv: {error}"
                    ))
                })?;
            }
        }
        self.validate_oof_controller_capabilities()?;
        if let Some(fold_set) = &self.fold_set {
            fold_set.validate()?;
        }
        if self.variants.is_empty() {
            return Err(DagMlError::Planning(
                "execution plan has no variants".to_string(),
            ));
        }
        for variant in &self.variants {
            variant.validate()?;
        }
        Ok(())
    }

    pub fn validate_parallel_controller_capabilities(
        &self,
        max_workers: usize,
        phase: Phase,
    ) -> Result<()> {
        if max_workers <= 1 {
            return Ok(());
        }
        let node_ids = self
            .node_parallel_levels_for_phase(phase)?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for node_id in node_ids {
            let node_plan = self.node_plans.get(&node_id).ok_or_else(|| {
                DagMlError::Planning(format!("missing node plan for `{node_id}`"))
            })?;
            let manifest = self
                .controller_manifests
                .get(&node_plan.controller_id)
                .ok_or_else(|| {
                    DagMlError::Planning(format!(
                        "missing controller manifest `{}` for node `{}`",
                        node_plan.controller_id, node_plan.node_id
                    ))
                })?;
            if !manifest.supports_parallel_invocation() {
                return Err(DagMlError::Planning(format!(
                    "parallel scheduler with {max_workers} workers requires controller `{}` for node `{}` to declare thread_safe or process_safe",
                    manifest.controller_id, node_plan.node_id
                )));
            }
        }
        Ok(())
    }

    fn validate_oof_controller_capabilities(&self) -> Result<()> {
        for edge in &self.graph_plan.graph.edges {
            if !edge.contract.requires_oof {
                continue;
            }
            let source_plan = self.node_plans.get(&edge.source.node_id).ok_or_else(|| {
                DagMlError::Planning(format!(
                    "OOF edge source node `{}` has no node plan",
                    edge.source.node_id
                ))
            })?;
            if !source_plan
                .controller_capabilities
                .contains(&ControllerCapability::EmitsPredictions)
            {
                return Err(DagMlError::Planning(format!(
                    "OOF edge `{}.{}` -> `{}.{}` requires source controller `{}` to declare emits_predictions",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name,
                    source_plan.controller_id
                )));
            }
            let target_plan = self.node_plans.get(&edge.target.node_id).ok_or_else(|| {
                DagMlError::Planning(format!(
                    "OOF edge target node `{}` has no node plan",
                    edge.target.node_id
                ))
            })?;
            if !target_plan
                .controller_capabilities
                .contains(&ControllerCapability::ConsumesOofPredictions)
            {
                return Err(DagMlError::Planning(format!(
                    "OOF edge `{}.{}` -> `{}.{}` requires target controller `{}` to declare consumes_oof_predictions",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name,
                    target_plan.controller_id
                )));
            }
        }
        Ok(())
    }

    pub fn node_parallel_levels_for_phase(&self, phase: Phase) -> Result<Vec<Vec<NodeId>>> {
        let levels = self
            .graph_plan
            .parallel_levels()?
            .into_iter()
            .map(|level| {
                level
                    .into_iter()
                    .filter(|node_id| {
                        self.node_plans
                            .get(node_id)
                            .is_some_and(|node_plan| node_plan.supported_phases.contains(&phase))
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|level| !level.is_empty())
            .collect::<Vec<_>>();
        Ok(levels)
    }

    pub fn campaign_phase_schedule(&self, phase: Phase) -> Result<PhaseExecutionSchedule> {
        self.validate()?;
        let node_levels = self.node_parallel_levels_for_phase(phase)?;
        let fold_ids = if phase == Phase::FitCv {
            self.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        let mut scopes = Vec::new();
        for variant in &self.variants {
            for fold_id in &fold_ids {
                scopes.push(ExecutionScopePlan {
                    scope_id: execution_scope_id(
                        phase,
                        Some(&variant.variant_id),
                        fold_id.as_ref(),
                    ),
                    phase,
                    variant_id: Some(variant.variant_id.clone()),
                    variant_seed: variant.seed,
                    fold_id: fold_id.clone(),
                    node_levels: node_levels.clone(),
                });
            }
        }
        Ok(PhaseExecutionSchedule {
            plan_id: self.id.clone(),
            phase,
            scopes,
        })
    }

    /// Returns the `BranchViewPlan` whose `branch_id` matches `branch_id`,
    /// if any. The match is exact; callers that need fuzzy or prefix matching
    /// must iterate `self.campaign.branch_view_plans` themselves.
    pub fn branch_view_for(&self, branch_id: &str) -> Option<&BranchViewPlan> {
        branch_view_for_in(&self.campaign.branch_view_plans, branch_id)
    }

    /// Returns the `BranchViewPlan` for the deepest branch in `branch_path`
    /// that has a matching plan, if any. The path is walked tip-first so the
    /// closest enclosing branch wins; an empty path returns `None`. The
    /// returned reference borrows the plan from the campaign; the caller can
    /// `.clone()` it into a `DataProviderViewSpec.branch_view` field when
    /// constructing a provider view for an in-branch node.
    pub fn branch_view_for_path(&self, branch_path: &[String]) -> Option<&BranchViewPlan> {
        branch_view_for_path_in(&self.campaign.branch_view_plans, branch_path)
    }
}

fn branch_view_for_in<'a>(
    plans: &'a [BranchViewPlan],
    branch_id: &str,
) -> Option<&'a BranchViewPlan> {
    plans.iter().find(|plan| plan.branch_id == branch_id)
}

fn branch_view_for_path_in<'a>(
    plans: &'a [BranchViewPlan],
    branch_path: &[String],
) -> Option<&'a BranchViewPlan> {
    for branch_id in branch_path.iter().rev() {
        if let Some(plan) = branch_view_for_in(plans, branch_id) {
            return Some(plan);
        }
    }
    None
}

fn execution_scope_id(
    phase: Phase,
    variant_id: Option<&VariantId>,
    fold_id: Option<&FoldId>,
) -> String {
    format!(
        "scope:{}:{}:{}",
        phase_scope_label(phase),
        variant_id
            .map(ToString::to_string)
            .unwrap_or_else(|| "base".to_string()),
        fold_id
            .map(ToString::to_string)
            .unwrap_or_else(|| "nofold".to_string())
    )
}

fn phase_scope_label(phase: Phase) -> &'static str {
    match phase {
        Phase::Compile => "COMPILE",
        Phase::Plan => "PLAN",
        Phase::FitCv => "FIT_CV",
        Phase::Select => "SELECT",
        Phase::Refit => "REFIT",
        Phase::Predict => "PREDICT",
        Phase::Explain => "EXPLAIN",
    }
}

pub fn build_execution_plan(
    id: impl Into<String>,
    graph: GraphSpec,
    campaign: CampaignSpec,
    registry: &ControllerRegistry,
) -> Result<ExecutionPlan> {
    let id = id.into();
    if id.trim().is_empty() {
        return Err(DagMlError::Planning(
            "execution plan id is empty".to_string(),
        ));
    }
    campaign.validate()?;
    let graph_plan = GraphPlan::from_graph(graph)?;
    validate_campaign_node_targets(&graph_plan.graph, &campaign)?;

    let mut node_plans = BTreeMap::new();
    let mut controller_manifests = BTreeMap::new();
    for node_id in &graph_plan.topological_order {
        let node = graph_plan
            .graph
            .nodes
            .iter()
            .find(|node| &node.id == node_id)
            .expect("topological node exists");
        let manifest = registry.resolve_for_node(node)?;
        let params = node.params.clone();
        let params_fingerprint = stable_json_fingerprint(&params)?;
        // Lower a node-local nested-CV policy carried by the DSL compiler in the
        // graph node metadata into the typed NodePlan field. Malformed metadata
        // fails the plan rather than silently dropping nested CV.
        let inner_cv = match node.metadata.get("dsl_inner_cv") {
            Some(value) => {
                let spec =
                    serde_json::from_value::<NestedCvSpec>(value.clone()).map_err(|error| {
                        DagMlError::Planning(format!(
                            "node `{}` has invalid dsl_inner_cv metadata: {error}",
                            node.id
                        ))
                    })?;
                // Reject semantically malformed specs (e.g. n_splits < 2) here, at
                // the plan boundary, rather than deferring to FIT_CV fold building.
                spec.validate().map_err(|error| {
                    DagMlError::Planning(format!(
                        "node `{}` has invalid dsl_inner_cv metadata: {error}",
                        node.id
                    ))
                })?;
                Some(spec)
            }
            None => None,
        };
        let shape_plan = campaign.shape_plans.get(&node.id).cloned();
        let data_bindings = campaign
            .data_bindings
            .get(&node.id)
            .cloned()
            .unwrap_or_default();
        node_plans.insert(
            node.id.clone(),
            NodePlan {
                inner_cv,
                node_id: node.id.clone(),
                kind: node.kind.clone(),
                controller_id: manifest.controller_id.clone(),
                controller_version: manifest.controller_version.clone(),
                supported_phases: manifest.supported_phases.clone(),
                controller_capabilities: manifest.capabilities.clone(),
                fit_scope: manifest.fit_scope,
                rng_policy: manifest.rng_policy,
                artifact_policy: manifest.artifact_policy,
                input_nodes: graph_plan.graph.upstream_nodes(&node.id),
                output_nodes: graph_plan.graph.downstream_nodes(&node.id),
                shape_plan,
                data_bindings,
                params,
                params_fingerprint,
            },
        );
        controller_manifests.insert(manifest.controller_id.clone(), manifest);
    }

    let fold_set = campaign
        .split_invocation
        .as_ref()
        .and_then(|split| split.fold_set.clone());
    validate_search_space_fingerprint(&graph_plan.graph, &campaign)?;
    let variants = enumerate_variants(&campaign.generation, campaign.root_seed)?;
    validate_generation_override_targets(&graph_plan.graph, &variants)?;
    let graph_fingerprint = stable_json_fingerprint(&graph_plan.graph)?;
    let campaign_fingerprint = stable_json_fingerprint(&campaign)?;
    let controller_fingerprint = stable_json_fingerprint(&controller_manifests)?;
    let plan = ExecutionPlan {
        id,
        graph_plan,
        campaign,
        node_plans,
        controller_manifests,
        variants,
        fold_set,
        graph_fingerprint,
        campaign_fingerprint,
        controller_fingerprint,
    };
    plan.validate()?;
    Ok(plan)
}

fn validate_search_space_fingerprint(graph: &GraphSpec, campaign: &CampaignSpec) -> Result<()> {
    let Some(expected_fingerprint) = &graph.search_space_fingerprint else {
        return Ok(());
    };
    if expected_fingerprint.trim().is_empty() {
        return Err(DagMlError::Planning(format!(
            "graph `{}` has empty search_space_fingerprint",
            graph.id
        )));
    }
    let actual_fingerprint = generation_spec_fingerprint(&campaign.generation)?;
    if expected_fingerprint != &actual_fingerprint {
        return Err(DagMlError::Planning(format!(
            "graph `{}` search_space_fingerprint does not match campaign generation spec",
            graph.id
        )));
    }
    Ok(())
}

fn validate_generation_override_targets(graph: &GraphSpec, variants: &[VariantPlan]) -> Result<()> {
    let node_ids = graph
        .nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    for variant in variants {
        for node_id in variant.param_override_targets()? {
            if !node_ids.contains(&node_id) {
                return Err(DagMlError::Planning(format!(
                    "variant `{}` overrides params for unknown node `{node_id}`",
                    variant.variant_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_campaign_node_targets(graph: &GraphSpec, campaign: &CampaignSpec) -> Result<()> {
    let node_ids = graph
        .nodes
        .iter()
        .map(|node| &node.id)
        .collect::<BTreeSet<_>>();
    for node_id in campaign.shape_plans.keys() {
        if !node_ids.contains(node_id) {
            return Err(DagMlError::Planning(format!(
                "shape plan references unknown node `{node_id}`"
            )));
        }
    }
    for node_id in campaign.data_bindings.keys() {
        if !node_ids.contains(node_id) {
            return Err(DagMlError::Planning(format!(
                "data binding references unknown node `{node_id}`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::controller::{
        ArtifactPolicy, ControllerCapability, ControllerFitScope, ControllerManifest, RngPolicy,
    };

    #[test]
    fn inner_cv_is_declarable_at_campaign_and_node_level() {
        // Campaign-level (global) declaration round-trips through JSON.
        let campaign_json = r#"{"id":"c","root_seed":null,"inner_cv":{"kind":"kfold","n_splits":3,"shuffle":false,"seed":5}}"#;
        let campaign: CampaignSpec = serde_json::from_str(campaign_json).unwrap();
        campaign.validate().unwrap();
        assert!(campaign.inner_cv.is_some());

        // A node-local declaration overrides the campaign default.
        let node_inner = crate::fold::NestedCvSpec::KFold(crate::fold::KFoldSpec {
            n_splits: 4,
            shuffle: false,
            seed: Some(6),
        });
        let resolved = crate::fold::resolve_inner_cv(Some(&node_inner), campaign.inner_cv.as_ref());
        assert_eq!(resolved, Some(&node_inner));

        // Absent on both campaign and node serializes away (skip_serializing_if).
        let bare = r#"{"id":"c","root_seed":null}"#;
        let bare_campaign: CampaignSpec = serde_json::from_str(bare).unwrap();
        assert!(bare_campaign.inner_cv.is_none());
        let reserialized = serde_json::to_string(&bare_campaign).unwrap();
        assert!(!reserialized.contains("inner_cv"));

        // A semantically-malformed campaign-global inner_cv (n_splits < 2) is
        // rejected by CampaignSpec::validate (the plan boundary), not deferred.
        let bad: CampaignSpec = serde_json::from_str(
            r#"{"id":"c","root_seed":null,"inner_cv":{"kind":"kfold","n_splits":1,"shuffle":false,"seed":null}}"#,
        )
        .unwrap();
        let error = bad.validate().unwrap_err();
        assert!(error.to_string().contains("at least two splits"));
    }

    #[test]
    fn execution_plan_validate_rejects_invalid_node_local_inner_cv() {
        // A canonical ExecutionPlan loaded from JSON (bypassing DSL lowering) can
        // carry a malformed node-local inner_cv; ExecutionPlan::validate must
        // refuse it rather than deferring to FIT_CV fold building.
        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:plan-validate".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };
        let mut plan =
            build_execution_plan("plan:validate", graph(), campaign, &registry()).unwrap();
        plan.validate().unwrap();
        plan.node_plans
            .get_mut(&NodeId::new("model:pls").unwrap())
            .unwrap()
            .inner_cv = Some(crate::fold::NestedCvSpec::KFold(crate::fold::KFoldSpec {
            n_splits: 1,
            shuffle: false,
            seed: None,
        }));
        let error = plan.validate().unwrap_err();
        assert!(matches!(error, DagMlError::Planning(_)));
        assert!(error.to_string().contains("invalid inner_cv"));
        assert!(error.to_string().contains("at least two splits"));
    }

    #[test]
    fn build_execution_plan_lowers_dsl_inner_cv_metadata_into_node_plan() {
        let mut graph = graph();
        graph
            .nodes
            .iter_mut()
            .find(|node| node.id.as_str() == "model:pls")
            .unwrap()
            .metadata
            .insert(
                "dsl_inner_cv".to_string(),
                serde_json::json!({"kind": "kfold", "n_splits": 3, "shuffle": false, "seed": 9}),
            );

        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:inner-cv".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let plan = build_execution_plan("plan:inner-cv", graph, campaign, &registry()).unwrap();
        match &plan.node_plans[&NodeId::new("model:pls").unwrap()].inner_cv {
            Some(crate::fold::NestedCvSpec::KFold(k)) => {
                assert_eq!(k.n_splits, 3);
                assert_eq!(k.seed, Some(9));
            }
            other => panic!("expected lowered KFold inner_cv, got {other:?}"),
        }
        assert!(plan.node_plans[&NodeId::new("transform:snv").unwrap()]
            .inner_cv
            .is_none());
    }

    #[test]
    fn build_execution_plan_rejects_malformed_dsl_inner_cv_metadata() {
        let mut graph = graph();
        graph
            .nodes
            .iter_mut()
            .find(|node| node.id.as_str() == "model:pls")
            .unwrap()
            .metadata
            .insert(
                "dsl_inner_cv".to_string(),
                serde_json::json!({"kind": "not_a_real_kind"}),
            );

        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:inner-cv.bad".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let error =
            build_execution_plan("plan:inner-cv.bad", graph, campaign, &registry()).unwrap_err();
        assert!(matches!(error, DagMlError::Planning(_)));
        assert!(error.to_string().contains("invalid dsl_inner_cv metadata"));
    }

    #[test]
    fn build_execution_plan_rejects_semantically_invalid_dsl_inner_cv() {
        // Right discriminator, invalid value: a single split is rejected at the
        // plan boundary rather than deferred to FIT_CV fold building.
        let mut graph = graph();
        graph
            .nodes
            .iter_mut()
            .find(|node| node.id.as_str() == "model:pls")
            .unwrap()
            .metadata
            .insert(
                "dsl_inner_cv".to_string(),
                serde_json::json!({"kind": "kfold", "n_splits": 1, "shuffle": false, "seed": null}),
            );

        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:inner-cv.nsplits".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let error = build_execution_plan("plan:inner-cv.nsplits", graph, campaign, &registry())
            .unwrap_err();
        assert!(matches!(error, DagMlError::Planning(_)));
        assert!(error.to_string().contains("at least two splits"));
    }
    use crate::data::DataBinding;
    use crate::generation::{
        GenerationChoice, GenerationDimension, GenerationParamOverride, GenerationStrategy,
    };
    use crate::graph::{
        EdgeContract, EdgeSpec, GraphInterface, NodeSpec, PortCardinality, PortKind, PortRef,
        PortSchema, PortSpec,
    };
    use crate::ids::{ControllerId, FoldId, ObservationId, SampleId, TargetId};
    use crate::phase::Phase;
    use crate::policy::{DataModelShapePlan, Granularity};
    use crate::relation::{SampleRelation, SampleRelationSet};

    fn port(name: &str, kind: PortKind) -> PortSpec {
        PortSpec {
            name: name.to_string(),
            kind,
            representation: None,
            cardinality: PortCardinality::One,
            description: String::new(),
        }
    }

    fn node(id: &str, kind: NodeKind, inputs: Vec<PortSpec>, outputs: Vec<PortSpec>) -> NodeSpec {
        NodeSpec {
            id: NodeId::new(id).unwrap(),
            kind,
            operator: None,
            params: BTreeMap::new(),
            ports: PortSchema { inputs, outputs },
            metadata: BTreeMap::new(),
            seed_label: None,
        }
    }

    fn graph() -> GraphSpec {
        GraphSpec {
            id: "g".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node(
                    "transform:snv",
                    NodeKind::Transform,
                    vec![],
                    vec![port("x", PortKind::Data)],
                ),
                node(
                    "model:pls",
                    NodeKind::Model,
                    vec![port("x", PortKind::Data)],
                    vec![port("pred", PortKind::Prediction)],
                ),
            ],
            edges: vec![EdgeSpec {
                source: PortRef {
                    node_id: NodeId::new("transform:snv").unwrap(),
                    port_name: "x".to_string(),
                },
                target: PortRef {
                    node_id: NodeId::new("model:pls").unwrap(),
                    port_name: "x".to_string(),
                },
                contract: EdgeContract {
                    kind: PortKind::Data,
                    representation: None,
                    requires_oof: false,
                    requires_fold_alignment: false,
                    propagates_lineage: true,
                },
            }],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        }
    }

    fn manifest(id: &str, kind: NodeKind) -> ControllerManifest {
        let mut capabilities = BTreeSet::from([
            ControllerCapability::Deterministic,
            ControllerCapability::ThreadSafe,
            ControllerCapability::ProcessSafe,
        ]);
        if kind == NodeKind::Model {
            capabilities.insert(ControllerCapability::EmitsPredictions);
            capabilities.insert(ControllerCapability::ConsumesOofPredictions);
        }
        ControllerManifest {
            controller_id: ControllerId::new(id).unwrap(),
            controller_version: "0.1.0".to_string(),
            operator_kind: kind,
            priority: 0,
            supported_phases: BTreeSet::from([Phase::FitCv, Phase::Refit, Phase::Predict]),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            data_requirements: None,
            capabilities,
            operator_selectors: Vec::new(),
            fit_scope: ControllerFitScope::FoldTrain,
            rng_policy: RngPolicy::UsesCoreSeed,
            artifact_policy: ArtifactPolicy::Serializable,
        }
    }

    fn registry() -> ControllerRegistry {
        let mut registry = ControllerRegistry::new();
        registry
            .register(manifest("controller:transform", NodeKind::Transform))
            .unwrap();
        registry
            .register(manifest("controller:model", NodeKind::Model))
            .unwrap();
        registry
    }

    fn oof_graph() -> GraphSpec {
        GraphSpec {
            id: "g:oof.capabilities".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node(
                    "model:base",
                    NodeKind::Model,
                    vec![],
                    vec![port("pred", PortKind::Prediction)],
                ),
                node(
                    "model:meta",
                    NodeKind::Model,
                    vec![port("pred", PortKind::Prediction)],
                    vec![port("pred", PortKind::Prediction)],
                ),
            ],
            edges: vec![EdgeSpec {
                source: PortRef {
                    node_id: NodeId::new("model:base").unwrap(),
                    port_name: "pred".to_string(),
                },
                target: PortRef {
                    node_id: NodeId::new("model:meta").unwrap(),
                    port_name: "pred".to_string(),
                },
                contract: EdgeContract {
                    kind: PortKind::Prediction,
                    representation: None,
                    requires_oof: true,
                    requires_fold_alignment: true,
                    propagates_lineage: true,
                },
            }],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        }
    }

    fn data_binding(node_id: &NodeId) -> DataBinding {
        DataBinding {
            node_id: node_id.clone(),
            input_name: "x".to_string(),
            request_id: "nir-to-tabular".to_string(),
            schema_fingerprint: "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b"
                .to_string(),
            plan_fingerprint: "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d"
                .to_string(),
            relation_fingerprint: Some(
                "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10".to_string(),
            ),
            output_representation: "tabular_numeric".to_string(),
            feature_set_id: Some("x".to_string()),
            source_ids: vec!["nir".to_string()],
            require_relations: true,
            view_policy: Default::default(),
            metadata: BTreeMap::new(),
        }
    }

    fn levels_as_strings(levels: &[Vec<NodeId>]) -> Vec<Vec<String>> {
        levels
            .iter()
            .map(|level| level.iter().map(ToString::to_string).collect())
            .collect()
    }

    #[test]
    fn published_campaign_spec_schema_declares_current_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/campaign_spec.schema.json"
        ))
        .unwrap();

        assert_eq!(schema["$id"], CAMPAIGN_SPEC_SCHEMA_ID);
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("id")));
        assert!(schema["$defs"]["split_invocation"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("fold_set"));
        assert!(schema["$defs"]["aggregation_policy"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("selection_metric_level"));
        assert!(schema["$defs"]["aggregation_policy"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("custom_controller"));
        assert!(schema["$defs"]["data_binding"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("view_policy"));
        assert!(schema["properties"]
            .as_object()
            .unwrap()
            .contains_key("branch_view_plans"));
        assert!(schema["$defs"]["branch_view_plan"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("selector"));
    }

    #[test]
    fn published_execution_plan_schema_declares_current_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/execution_plan.schema.json"
        ))
        .unwrap();

        assert_eq!(schema["$id"], EXECUTION_PLAN_SCHEMA_ID);
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("node_plans")));
        assert!(schema["properties"]
            .as_object()
            .unwrap()
            .contains_key("controller_fingerprint"));
        assert!(schema["$defs"]["node_plan"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("shape_plan"));
        assert!(schema["$defs"]["variant_plan"]["properties"]
            .as_object()
            .unwrap()
            .contains_key("choices"));
    }

    #[test]
    fn published_execution_plan_fixture_validates_current_contract() {
        let plan: ExecutionPlan = serde_json::from_str(include_str!(
            "../../../examples/fixtures/runtime/execution_plan_branch_merge_executable.json"
        ))
        .unwrap();

        plan.validate().unwrap();
        assert_eq!(plan.id, "plan:fixture.execution.branch_merge");
        assert_eq!(plan.variants.len(), 2);
        assert_eq!(plan.node_plans.len(), plan.graph_plan.graph.nodes.len());
    }

    #[test]
    fn builds_execution_plan_with_shape_and_fold_contracts() {
        let model_id = NodeId::new("model:pls").unwrap();
        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:oof".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer".to_string(),
                controller_id: None,
                leakage_policy: LeakageUnitPolicy::default(),
                params: BTreeMap::new(),
                fold_set: Some(FoldSet {
                    id: "outer".to_string(),
                    sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
                    folds: vec![
                        crate::fold::FoldAssignment {
                            fold_id: FoldId::new("fold0").unwrap(),
                            train_sample_ids: vec![SampleId::new("s2").unwrap()],
                            validation_sample_ids: vec![SampleId::new("s1").unwrap()],
                            metadata: BTreeMap::new(),
                        },
                        crate::fold::FoldAssignment {
                            fold_id: FoldId::new("fold1").unwrap(),
                            train_sample_ids: vec![SampleId::new("s1").unwrap()],
                            validation_sample_ids: vec![SampleId::new("s2").unwrap()],
                            metadata: BTreeMap::new(),
                        },
                    ],
                    sample_groups: BTreeMap::new(),
                }),
            }),
            generation: Default::default(),
            shape_plans: BTreeMap::from([(
                model_id.clone(),
                DataModelShapePlan {
                    node_id: model_id.clone(),
                    input_granularity: Granularity::Observation,
                    ..DataModelShapePlan {
                        node_id: model_id.clone(),
                        input_granularity: Granularity::Sample,
                        target_granularity: Granularity::Sample,
                        fit_rows: crate::policy::FitBoundary::FoldTrain,
                        predict_rows: crate::policy::FitBoundary::FoldValidation,
                        feature_namespace: None,
                        feature_schema_fingerprint: None,
                        target_space: "raw".to_string(),
                        aggregation_policy: AggregationPolicy::default(),
                        augmentation_policy: crate::policy::AugmentationPolicy::default(),
                        selection_policy: crate::policy::FeatureSelectionPolicy::default(),
                    }
                },
            )]),
            data_bindings: BTreeMap::from([(model_id.clone(), vec![data_binding(&model_id)])]),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let plan = build_execution_plan("plan:oof", graph(), campaign, &registry()).unwrap();

        assert_eq!(
            plan.graph_plan
                .topological_order
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            vec!["transform:snv", "model:pls"]
        );
        assert_eq!(
            levels_as_strings(&plan.graph_plan.parallel_levels),
            vec![vec!["transform:snv"], vec!["model:pls"]]
        );
        assert!(plan.node_plans[&model_id]
            .controller_capabilities
            .contains(&ControllerCapability::EmitsPredictions));
        assert!(plan.fold_set.is_some());
        let schedule = plan.campaign_phase_schedule(Phase::FitCv).unwrap();
        assert_eq!(schedule.scopes.len(), 2);
        assert!(schedule.scopes[0].scope_id.starts_with("scope:FIT_CV:"));
        assert!(schedule
            .scopes
            .iter()
            .all(|scope| levels_as_strings(&scope.node_levels)
                == vec![vec!["transform:snv"], vec!["model:pls"]]));
        assert_eq!(
            schedule
                .scopes
                .iter()
                .filter_map(|scope| scope.fold_id.as_ref().map(ToString::to_string))
                .collect::<Vec<_>>(),
            vec!["fold0", "fold1"]
        );
        assert_eq!(
            plan.node_plans
                .get(&model_id)
                .unwrap()
                .controller_id
                .as_str(),
            "controller:model"
        );
        assert_eq!(
            plan.node_plans.get(&model_id).unwrap().data_bindings.len(),
            1
        );

        let mut bad_plan = plan.clone();
        bad_plan.graph_plan.parallel_levels =
            vec![vec![model_id], vec![NodeId::new("transform:snv").unwrap()]];
        assert!(bad_plan
            .validate()
            .unwrap_err()
            .to_string()
            .contains("parallel levels"));

        let bad_envelope = ExternalDataPlanEnvelope {
            schema_version: crate::data::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
            schema_fingerprint: "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b"
                .to_string(),
            plan_fingerprint: "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d"
                .to_string(),
            relation_fingerprint: None,
            coordinator_relations: Some(SampleRelationSet {
                records: vec![SampleRelation {
                    observation_id: ObservationId::new("obs:outside").unwrap(),
                    sample_id: SampleId::new("sample:outside").unwrap(),
                    target_id: Some(TargetId::new("target:outside").unwrap()),
                    group_id: None,
                    origin_sample_id: None,
                    source_id: Some("nir".to_string()),
                    is_augmented: false,
                }],
            }),
        };
        assert!(plan
            .campaign
            .validate_data_envelope_relations(&bad_envelope)
            .unwrap_err()
            .to_string()
            .contains("outside fold set"));
    }

    #[test]
    fn planning_refuses_shape_plan_for_unknown_node() {
        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:oof".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::from([(
                NodeId::new("model:missing").unwrap(),
                DataModelShapePlan {
                    node_id: NodeId::new("model:missing").unwrap(),
                    input_granularity: Granularity::Sample,
                    target_granularity: Granularity::Sample,
                    fit_rows: crate::policy::FitBoundary::FoldTrain,
                    predict_rows: crate::policy::FitBoundary::FoldValidation,
                    feature_namespace: None,
                    feature_schema_fingerprint: None,
                    target_space: "raw".to_string(),
                    aggregation_policy: AggregationPolicy::default(),
                    augmentation_policy: crate::policy::AugmentationPolicy::default(),
                    selection_policy: crate::policy::FeatureSelectionPolicy::default(),
                },
            )]),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };

        assert!(build_execution_plan("plan:oof", graph(), campaign, &registry()).is_err());
    }

    #[test]
    fn planning_refuses_oof_edge_without_controller_capabilities() {
        let mut registry = ControllerRegistry::new();
        let mut model_manifest = manifest("controller:model", NodeKind::Model);
        model_manifest
            .capabilities
            .remove(&ControllerCapability::ConsumesOofPredictions);
        registry.register(model_manifest).unwrap();

        let err = build_execution_plan(
            "plan:oof.capability",
            oof_graph(),
            CampaignSpec {
                inner_cv: None,
                id: "campaign:oof.capability".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                branch_view_plans: Vec::new(),
                metadata: BTreeMap::new(),
            },
            &registry,
        )
        .unwrap_err();

        assert!(err.to_string().contains("consumes_oof_predictions"));
    }

    #[test]
    fn parallel_controller_capability_validation_requires_safe_manifest() {
        let mut registry = ControllerRegistry::new();
        let mut transform_manifest = manifest("controller:transform", NodeKind::Transform);
        transform_manifest
            .capabilities
            .remove(&ControllerCapability::ThreadSafe);
        transform_manifest
            .capabilities
            .remove(&ControllerCapability::ProcessSafe);
        registry.register(transform_manifest).unwrap();
        registry
            .register(manifest("controller:model", NodeKind::Model))
            .unwrap();
        let plan = build_execution_plan(
            "plan:parallel.capability",
            graph(),
            CampaignSpec {
                inner_cv: None,
                id: "campaign:parallel.capability".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                branch_view_plans: Vec::new(),
                metadata: BTreeMap::new(),
            },
            &registry,
        )
        .unwrap();

        assert!(plan
            .validate_parallel_controller_capabilities(1, Phase::FitCv)
            .is_ok());
        let err = plan
            .validate_parallel_controller_capabilities(2, Phase::FitCv)
            .unwrap_err();
        assert!(err.to_string().contains("thread_safe or process_safe"));
    }

    #[test]
    fn planning_refuses_generation_override_for_unknown_node() {
        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:oof".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: GenerationSpec {
                strategy: GenerationStrategy::Cartesian,
                dimensions: vec![GenerationDimension {
                    name: "model_family".to_string(),
                    choices: vec![GenerationChoice {
                        label: "pls".to_string(),
                        value: serde_json::json!("pls"),
                        param_overrides: vec![GenerationParamOverride {
                            node_id: NodeId::new("model:missing").unwrap(),
                            params: BTreeMap::from([(
                                "n_components".to_string(),
                                serde_json::json!(8),
                            )]),
                        }],
                    }],
                }],
                max_variants: Some(1),
            },
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let error = build_execution_plan("plan:oof", graph(), campaign, &registry())
            .unwrap_err()
            .to_string();

        assert!(error.contains("overrides params for unknown node"));
    }

    #[test]
    fn planning_validates_declared_search_space_fingerprint() {
        let campaign = CampaignSpec {
            inner_cv: None,
            id: "campaign:search.fingerprint".to_string(),
            root_seed: Some(7),
            leakage_policy: LeakageUnitPolicy::default(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: None,
            generation: GenerationSpec {
                strategy: GenerationStrategy::Cartesian,
                dimensions: vec![GenerationDimension {
                    name: "model_family".to_string(),
                    choices: vec![GenerationChoice {
                        label: "pls".to_string(),
                        value: serde_json::json!("pls"),
                        param_overrides: vec![GenerationParamOverride {
                            node_id: NodeId::new("model:pls").unwrap(),
                            params: BTreeMap::from([(
                                "n_components".to_string(),
                                serde_json::json!(8),
                            )]),
                        }],
                    }],
                }],
                max_variants: Some(1),
            },
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        };
        let mut graph = graph();
        graph.search_space_fingerprint =
            Some(generation_spec_fingerprint(&campaign.generation).unwrap());

        let plan = build_execution_plan(
            "plan:search.fingerprint",
            graph.clone(),
            campaign.clone(),
            &registry(),
        )
        .unwrap();
        assert_eq!(plan.variants.len(), 1);

        graph.search_space_fingerprint = Some("sha256:not-the-generation-spec".to_string());
        let error = build_execution_plan("plan:search.fingerprint", graph, campaign, &registry())
            .unwrap_err()
            .to_string();
        assert!(error.contains("search_space_fingerprint"));
    }

    #[test]
    fn branch_view_lookup_helpers_match_by_branch_id_and_innermost_path() {
        use crate::data::{BranchViewMode, DataViewSelector};

        let outer = BranchViewPlan {
            view_id: "branch_view:outer".to_string(),
            branch_id: "branch:outer".to_string(),
            mode: BranchViewMode::BySource,
            selector: DataViewSelector {
                source_ids: vec!["nir".to_string()],
                ..Default::default()
            },
            allow_overlap: false,
            metadata: BTreeMap::new(),
        };
        let inner = BranchViewPlan {
            view_id: "branch_view:inner".to_string(),
            branch_id: "branch:inner".to_string(),
            mode: BranchViewMode::Separation,
            selector: DataViewSelector {
                source_ids: vec!["chem".to_string()],
                ..Default::default()
            },
            allow_overlap: false,
            metadata: BTreeMap::new(),
        };
        let plans = vec![outer.clone(), inner.clone()];

        assert_eq!(
            super::branch_view_for_in(&plans, "branch:outer"),
            Some(&outer)
        );
        assert_eq!(
            super::branch_view_for_in(&plans, "branch:inner"),
            Some(&inner)
        );
        assert_eq!(super::branch_view_for_in(&plans, "branch:missing"), None);

        let path = vec!["branch:outer".to_string(), "branch:inner".to_string()];
        // tip-first: innermost matching branch wins
        assert_eq!(super::branch_view_for_path_in(&plans, &path), Some(&inner));

        let path_outer_only = vec!["branch:outer".to_string()];
        assert_eq!(
            super::branch_view_for_path_in(&plans, &path_outer_only),
            Some(&outer)
        );

        let empty_path: Vec<String> = Vec::new();
        assert_eq!(super::branch_view_for_path_in(&plans, &empty_path), None);

        let path_no_match = vec!["branch:other".to_string()];
        assert_eq!(super::branch_view_for_path_in(&plans, &path_no_match), None);
    }
}
