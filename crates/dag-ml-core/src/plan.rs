use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::campaign::stable_json_fingerprint;
use crate::controller::{ControllerManifest, ControllerRegistry};
use crate::data::DataBinding;
use crate::error::{DagMlError, Result};
use crate::fold::FoldSet;
use crate::generation::{enumerate_variants, GenerationSpec, VariantPlan};
use crate::graph::{GraphSpec, NodeKind};
use crate::ids::{ControllerId, NodeId};
use crate::phase::Phase;
use crate::policy::{AggregationPolicy, DataModelShapePlan, LeakageUnitPolicy};

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
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphPlan {
    pub graph: GraphSpec,
    pub topological_order: Vec<NodeId>,
}

impl GraphPlan {
    pub fn from_graph(graph: GraphSpec) -> Result<Self> {
        let topological_order = graph.topological_order()?;
        Ok(Self {
            graph,
            topological_order,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodePlan {
    pub node_id: NodeId,
    pub kind: NodeKind,
    pub controller_id: ControllerId,
    pub controller_version: String,
    pub supported_phases: BTreeSet<Phase>,
    pub input_nodes: Vec<NodeId>,
    pub output_nodes: Vec<NodeId>,
    pub shape_plan: Option<DataModelShapePlan>,
    #[serde(default)]
    pub data_bindings: Vec<DataBinding>,
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

impl ExecutionPlan {
    pub fn validate(&self) -> Result<()> {
        self.graph_plan.graph.validate()?;
        self.campaign.validate()?;
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
            for binding in &plan.data_bindings {
                if binding.node_id != *node_id {
                    return Err(DagMlError::Planning(format!(
                        "node plan `{node_id}` contains data binding for `{}`",
                        binding.node_id
                    )));
                }
                binding.validate()?;
            }
        }
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
        let params_fingerprint = stable_json_fingerprint(&node.params)?;
        let shape_plan = campaign.shape_plans.get(&node.id).cloned();
        let data_bindings = campaign
            .data_bindings
            .get(&node.id)
            .cloned()
            .unwrap_or_default();
        node_plans.insert(
            node.id.clone(),
            NodePlan {
                node_id: node.id.clone(),
                kind: node.kind.clone(),
                controller_id: manifest.controller_id.clone(),
                controller_version: manifest.controller_version.clone(),
                supported_phases: manifest.supported_phases.clone(),
                input_nodes: graph_plan.graph.upstream_nodes(&node.id),
                output_nodes: graph_plan.graph.downstream_nodes(&node.id),
                shape_plan,
                data_bindings,
                params_fingerprint,
            },
        );
        controller_manifests.insert(manifest.controller_id.clone(), manifest);
    }

    let fold_set = campaign
        .split_invocation
        .as_ref()
        .and_then(|split| split.fold_set.clone());
    let variants = enumerate_variants(&campaign.generation, campaign.root_seed)?;
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
    use crate::data::DataBinding;
    use crate::graph::{
        EdgeContract, EdgeSpec, GraphInterface, NodeSpec, PortCardinality, PortKind, PortRef,
        PortSchema, PortSpec,
    };
    use crate::ids::{ControllerId, FoldId, SampleId};
    use crate::phase::Phase;
    use crate::policy::{DataModelShapePlan, Granularity};

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
        ControllerManifest {
            controller_id: ControllerId::new(id).unwrap(),
            controller_version: "0.1.0".to_string(),
            operator_kind: kind,
            priority: 0,
            supported_phases: BTreeSet::from([Phase::FitCv, Phase::Refit, Phase::Predict]),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            data_requirements: None,
            capabilities: BTreeSet::from([ControllerCapability::Deterministic]),
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
            source_ids: vec!["nir".to_string()],
            require_relations: true,
            view_policy: Default::default(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn builds_execution_plan_with_shape_and_fold_contracts() {
        let model_id = NodeId::new("model:pls").unwrap();
        let campaign = CampaignSpec {
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
        assert!(plan.fold_set.is_some());
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
    }

    #[test]
    fn planning_refuses_shape_plan_for_unknown_node() {
        let campaign = CampaignSpec {
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
            metadata: BTreeMap::new(),
        };

        assert!(build_execution_plan("plan:oof", graph(), campaign, &registry()).is_err());
    }
}
