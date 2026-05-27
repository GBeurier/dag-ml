use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::data::DataBinding;
use crate::error::{DagMlError, Result};
use crate::generation::{
    generation_spec_fingerprint, GenerationChoice, GenerationDimension, GenerationParamOverride,
    GenerationSpec, GenerationStrategy,
};
use crate::graph::{
    EdgeContract, EdgeSpec, GraphInterface, GraphSpec, NodeKind, NodeSpec, PortCardinality,
    PortKind, PortRef, PortSchema, PortSpec,
};
use crate::ids::NodeId;
use crate::plan::{CampaignSpec, SplitInvocation};
use crate::policy::{
    AggregationPolicy, AugmentationPolicy, DataModelShapePlan, FeatureSelectionPolicy, FitBoundary,
    Granularity, LeakageUnitPolicy,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslSpec {
    pub id: String,
    #[serde(default)]
    pub input: PipelineDslDataPort,
    #[serde(default)]
    pub output: PipelineDslPredictionPort,
    #[serde(default)]
    pub generation_strategy: Option<GenerationStrategy>,
    #[serde(default)]
    pub max_variants: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generation_dimensions: Vec<PipelineDslGenerationDimension>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign_id: Option<String>,
    #[serde(default)]
    pub root_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leakage_policy: Option<LeakageUnitPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregation_policy: Option<AggregationPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_invocation: Option<SplitInvocation>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub campaign_metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_bindings: Vec<DataBinding>,
    #[serde(default)]
    pub steps: Vec<PipelineDslStep>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslDataPort {
    #[serde(default = "default_input_name")]
    pub name: String,
    #[serde(default = "default_data_representation")]
    pub representation: String,
    #[serde(default)]
    pub description: String,
}

impl Default for PipelineDslDataPort {
    fn default() -> Self {
        Self {
            name: default_input_name(),
            representation: default_data_representation(),
            description: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslPredictionPort {
    #[serde(default = "default_output_name")]
    pub name: String,
    #[serde(default)]
    pub description: String,
}

impl Default for PipelineDslPredictionPort {
    fn default() -> Self {
        Self {
            name: default_output_name(),
            description: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PipelineDslStep {
    Transform(PipelineDslOperatorStep),
    Augmentation(PipelineDslOperatorStep),
    Model(PipelineDslOperatorStep),
    Branch(PipelineDslBranchStep),
    MergeModel(PipelineDslMergeModelStep),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslOperatorStep {
    pub id: NodeId,
    pub operator: serde_json::Value,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub seed_label: Option<String>,
    #[serde(default)]
    pub representation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<PipelineDslVariantChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<PipelineDslShapePlan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslVariantChoice {
    pub label: String,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslGenerationDimension {
    pub name: String,
    #[serde(default)]
    pub choices: Vec<PipelineDslGenerationChoice>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslGenerationChoice {
    pub label: String,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub param_overrides: Vec<PipelineDslGenerationParamOverride>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslGenerationParamOverride {
    pub node_id: NodeId,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslBranchStep {
    pub branches: Vec<PipelineDslBranch>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslBranch {
    pub id: String,
    #[serde(default)]
    pub steps: Vec<PipelineDslStep>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslMergeModelStep {
    pub id: NodeId,
    pub operator: serde_json::Value,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub seed_label: Option<String>,
    #[serde(default = "default_true")]
    pub include_original_data: bool,
    #[serde(default = "default_merge_mode")]
    pub merge_mode: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<PipelineDslVariantChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<PipelineDslShapePlan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslShapePlan {
    #[serde(default)]
    pub input_granularity: Option<Granularity>,
    #[serde(default)]
    pub target_granularity: Option<Granularity>,
    #[serde(default)]
    pub fit_rows: Option<FitBoundary>,
    #[serde(default)]
    pub predict_rows: Option<FitBoundary>,
    #[serde(default)]
    pub feature_namespace: Option<String>,
    #[serde(default)]
    pub feature_schema_fingerprint: Option<String>,
    #[serde(default)]
    pub target_space: Option<String>,
    #[serde(default)]
    pub aggregation_policy: Option<AggregationPolicy>,
    #[serde(default)]
    pub augmentation_policy: Option<AugmentationPolicy>,
    #[serde(default)]
    pub selection_policy: Option<FeatureSelectionPolicy>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledPipelineDsl {
    pub graph: GraphSpec,
    pub generation: GenerationSpec,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub shape_plans: BTreeMap<NodeId, DataModelShapePlan>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub data_bindings: BTreeMap<NodeId, Vec<DataBinding>>,
    pub campaign_template: CampaignSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_fingerprint: Option<String>,
}

pub fn compile_pipeline_dsl(spec: &PipelineDslSpec) -> Result<GraphSpec> {
    Ok(compile_pipeline_dsl_with_generation(spec)?.graph)
}

pub fn compile_pipeline_dsl_with_generation(spec: &PipelineDslSpec) -> Result<CompiledPipelineDsl> {
    validate_pipeline_dsl(spec)?;
    let input_representation = Some(spec.input.representation.clone());
    let external_data = DataSource {
        node_id: None,
        port_name: spec.input.name.clone(),
        representation: input_representation.clone(),
    };
    let mut compiler = PipelineCompiler {
        graph_id: spec.id.clone(),
        input_representation: input_representation.clone(),
        nodes: Vec::new(),
        edges: Vec::new(),
        generation_dimensions: Vec::new(),
        shape_plans: BTreeMap::new(),
    };
    let mut current_data = external_data.clone();
    let mut pending_predictions = Vec::new();

    for step in &spec.steps {
        compiler.compile_top_level_step(
            step,
            &external_data,
            &mut current_data,
            &mut pending_predictions,
        )?;
    }

    let mut generation_dimensions =
        compile_explicit_generation_dimensions(&spec.generation_dimensions, &compiler.nodes)?;
    generation_dimensions.extend(compiler.generation_dimensions);
    let generation = build_generation_spec(
        spec.generation_strategy,
        spec.max_variants,
        generation_dimensions,
    )?;
    let generation_fingerprint = if generation.strategy == GenerationStrategy::None {
        None
    } else {
        Some(generation_spec_fingerprint(&generation)?)
    };
    let graph = GraphSpec {
        id: spec.id.clone(),
        interface: GraphInterface {
            inputs: vec![data_port(
                &spec.input.name,
                input_representation.clone(),
                &spec.input.description,
            )],
            outputs: vec![prediction_port(&spec.output.name, &spec.output.description)],
        },
        nodes: compiler.nodes,
        edges: compiler.edges,
        search_space_fingerprint: generation_fingerprint.clone(),
        metadata: spec.metadata.clone(),
    };
    graph.validate()?;
    validate_shape_plan_targets(&compiler.shape_plans, &graph)?;
    let data_bindings = compile_data_bindings(&spec.data_bindings, &graph)?;
    let campaign_template =
        build_campaign_template(spec, &generation, &compiler.shape_plans, &data_bindings)?;
    Ok(CompiledPipelineDsl {
        graph,
        generation,
        shape_plans: compiler.shape_plans,
        data_bindings,
        campaign_template,
        generation_fingerprint,
    })
}

fn validate_pipeline_dsl(spec: &PipelineDslSpec) -> Result<()> {
    if spec.id.trim().is_empty() {
        return Err(DagMlError::GraphValidation(
            "pipeline DSL graph id must not be empty".to_string(),
        ));
    }
    if spec.input.name.trim().is_empty() {
        return Err(DagMlError::GraphValidation(
            "pipeline DSL input name must not be empty".to_string(),
        ));
    }
    if spec.input.representation.trim().is_empty() {
        return Err(DagMlError::GraphValidation(
            "pipeline DSL input representation must not be empty".to_string(),
        ));
    }
    if spec.output.name.trim().is_empty() {
        return Err(DagMlError::GraphValidation(
            "pipeline DSL output name must not be empty".to_string(),
        ));
    }
    if spec.steps.is_empty() {
        return Err(DagMlError::GraphValidation(
            "pipeline DSL must contain at least one step".to_string(),
        ));
    }
    Ok(())
}

struct PipelineCompiler {
    graph_id: String,
    input_representation: Option<String>,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    generation_dimensions: Vec<GenerationDimension>,
    shape_plans: BTreeMap<NodeId, DataModelShapePlan>,
}

#[derive(Clone, Debug)]
struct DataSource {
    node_id: Option<NodeId>,
    port_name: String,
    representation: Option<String>,
}

#[derive(Clone, Debug)]
struct PredictionSource {
    node_id: NodeId,
    port_name: String,
    input_name: String,
}

impl PipelineCompiler {
    fn compile_top_level_step(
        &mut self,
        step: &PipelineDslStep,
        external_data: &DataSource,
        current_data: &mut DataSource,
        pending_predictions: &mut Vec<PredictionSource>,
    ) -> Result<()> {
        match step {
            PipelineDslStep::Transform(step) => {
                *current_data =
                    self.compile_data_operator(NodeKind::Transform, step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Augmentation(step) => {
                *current_data =
                    self.compile_data_operator(NodeKind::Augmentation, step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Model(step) => {
                pending_predictions.clear();
                pending_predictions.push(self.compile_model(step, current_data, None)?);
                Ok(())
            }
            PipelineDslStep::Branch(step) => {
                *pending_predictions = self.compile_branch(step, current_data)?;
                Ok(())
            }
            PipelineDslStep::MergeModel(step) => {
                let prediction =
                    self.compile_merge_model(step, pending_predictions, external_data)?;
                pending_predictions.clear();
                pending_predictions.push(prediction);
                Ok(())
            }
        }
    }

    fn compile_branch(
        &mut self,
        step: &PipelineDslBranchStep,
        current_data: &DataSource,
    ) -> Result<Vec<PredictionSource>> {
        if step.branches.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL graph `{}` has a branch step without branches",
                self.graph_id
            )));
        }
        let mut predictions = Vec::with_capacity(step.branches.len());
        for (index, branch) in step.branches.iter().enumerate() {
            validate_branch_id(&branch.id)?;
            if branch.steps.is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL branch `{}` has no steps",
                    branch.id
                )));
            }
            let mut branch_data = current_data.clone();
            let mut branch_prediction = None;
            for branch_step in &branch.steps {
                match branch_step {
                    PipelineDslStep::Transform(step) => {
                        branch_data =
                            self.compile_data_operator(NodeKind::Transform, step, &branch_data)?;
                        branch_prediction = None;
                    }
                    PipelineDslStep::Augmentation(step) => {
                        branch_data =
                            self.compile_data_operator(NodeKind::Augmentation, step, &branch_data)?;
                        branch_prediction = None;
                    }
                    PipelineDslStep::Model(step) => {
                        branch_prediction =
                            Some(self.compile_model(step, &branch_data, Some(&branch.id))?);
                    }
                    PipelineDslStep::Branch(_) | PipelineDslStep::MergeModel(_) => {
                        return Err(DagMlError::GraphValidation(format!(
                            "pipeline DSL branch `{}` cannot contain nested branch or merge_model steps in this compiler profile",
                            branch.id
                        )));
                    }
                }
            }
            let prediction = branch_prediction.ok_or_else(|| {
                DagMlError::GraphValidation(format!(
                    "pipeline DSL branch `{}` must end with a model step",
                    branch.id
                ))
            })?;
            predictions.push(PredictionSource {
                input_name: format!("{}_oof", branch_input_prefix(&branch.id, index)),
                ..prediction
            });
        }
        Ok(predictions)
    }

    fn compile_data_operator(
        &mut self,
        kind: NodeKind,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
    ) -> Result<DataSource> {
        if kind == NodeKind::Augmentation && step.shape.is_none() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL augmentation `{}` requires a shape plan for leakage-safe scope validation",
                step.id
            )));
        }
        let representation = step
            .representation
            .clone()
            .or_else(|| input.representation.clone())
            .or_else(|| self.input_representation.clone());
        let node = NodeSpec {
            id: step.id.clone(),
            kind,
            operator: Some(step.operator.clone()),
            params: step.params.clone(),
            ports: PortSchema {
                inputs: vec![data_port("x", input.representation.clone(), "")],
                outputs: vec![data_port("x_out", representation.clone(), "")],
            },
            metadata: step.metadata.clone(),
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())?;
        self.connect_data(input, &step.id, "x")?;
        Ok(DataSource {
            node_id: Some(step.id.clone()),
            port_name: "x_out".to_string(),
            representation,
        })
    }

    fn compile_model(
        &mut self,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
        branch_id: Option<&str>,
    ) -> Result<PredictionSource> {
        let mut metadata = step.metadata.clone();
        if let Some(branch_id) = branch_id {
            metadata.insert(
                "dsl_branch".to_string(),
                serde_json::Value::String(branch_id.to_string()),
            );
        }
        let node = NodeSpec {
            id: step.id.clone(),
            kind: NodeKind::Model,
            operator: Some(step.operator.clone()),
            params: step.params.clone(),
            ports: PortSchema {
                inputs: vec![data_port("x", input.representation.clone(), "")],
                outputs: vec![prediction_port("oof", "")],
            },
            metadata,
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())?;
        self.connect_data(input, &step.id, "x")?;
        Ok(PredictionSource {
            node_id: step.id.clone(),
            port_name: "oof".to_string(),
            input_name: "oof".to_string(),
        })
    }

    fn compile_merge_model(
        &mut self,
        step: &PipelineDslMergeModelStep,
        predictions: &[PredictionSource],
        external_data: &DataSource,
    ) -> Result<PredictionSource> {
        if predictions.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL merge_model `{}` has no pending branch predictions",
                step.id
            )));
        }
        let mut input_ports = Vec::with_capacity(predictions.len() + 1);
        for prediction in predictions {
            input_ports.push(prediction_port(&prediction.input_name, ""));
        }
        if step.include_original_data {
            input_ports.push(data_port(
                "x_original",
                external_data.representation.clone(),
                "",
            ));
        }
        let mut metadata = step.metadata.clone();
        metadata.insert(
            "merge_mode".to_string(),
            serde_json::Value::String(step.merge_mode.clone()),
        );
        let node = NodeSpec {
            id: step.id.clone(),
            kind: NodeKind::Model,
            operator: Some(step.operator.clone()),
            params: step.params.clone(),
            ports: PortSchema {
                inputs: input_ports,
                outputs: vec![prediction_port("oof", "")],
            },
            metadata,
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())?;
        for prediction in predictions {
            self.edges.push(EdgeSpec {
                source: PortRef {
                    node_id: prediction.node_id.clone(),
                    port_name: prediction.port_name.clone(),
                },
                target: PortRef {
                    node_id: step.id.clone(),
                    port_name: prediction.input_name.clone(),
                },
                contract: EdgeContract {
                    kind: PortKind::Prediction,
                    representation: None,
                    requires_oof: true,
                    requires_fold_alignment: true,
                    propagates_lineage: true,
                },
            });
        }
        if step.include_original_data {
            self.connect_data_to_port(external_data, &step.id, "x_original")?;
        }
        Ok(PredictionSource {
            node_id: step.id.clone(),
            port_name: "oof".to_string(),
            input_name: "oof".to_string(),
        })
    }

    fn push_node(&mut self, node: NodeSpec) -> Result<()> {
        if self.nodes.iter().any(|existing| existing.id == node.id) {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL graph `{}` produced duplicate node `{}`",
                self.graph_id, node.id
            )));
        }
        self.nodes.push(node);
        Ok(())
    }

    fn collect_operator_generation(
        &mut self,
        node_id: &NodeId,
        choices: &[PipelineDslVariantChoice],
    ) -> Result<()> {
        if choices.is_empty() {
            return Ok(());
        }
        let dimension = GenerationDimension {
            name: format!("{node_id}.params"),
            choices: choices
                .iter()
                .map(|choice| {
                    if choice.params.is_empty() {
                        return Err(DagMlError::GraphValidation(format!(
                            "pipeline DSL variant `{}` for node `{node_id}` has no params",
                            choice.label
                        )));
                    }
                    let value = match &choice.value {
                        Some(value) => value.clone(),
                        None => serde_json::to_value(&choice.params).map_err(|error| {
                            DagMlError::GraphValidation(format!(
                                "failed to serialize pipeline DSL variant `{}` for node `{node_id}`: {error}",
                                choice.label
                            ))
                        })?,
                    };
                    Ok(GenerationChoice {
                        label: choice.label.clone(),
                        value,
                        param_overrides: vec![GenerationParamOverride {
                            node_id: node_id.clone(),
                            params: choice.params.clone(),
                        }],
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        self.generation_dimensions.push(dimension);
        Ok(())
    }

    fn collect_shape_plan(
        &mut self,
        node_id: &NodeId,
        shape: Option<&PipelineDslShapePlan>,
    ) -> Result<()> {
        let Some(shape) = shape else {
            return Ok(());
        };
        let plan = shape.to_data_model_shape_plan(node_id)?;
        if self.shape_plans.insert(node_id.clone(), plan).is_some() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL graph `{}` produced duplicate shape plan for `{node_id}`",
                self.graph_id
            )));
        }
        Ok(())
    }

    fn connect_data(
        &mut self,
        input: &DataSource,
        target_id: &NodeId,
        target_port: &str,
    ) -> Result<()> {
        self.connect_data_to_port(input, target_id, target_port)
    }

    fn connect_data_to_port(
        &mut self,
        input: &DataSource,
        target_id: &NodeId,
        target_port: &str,
    ) -> Result<()> {
        if let Some(source_id) = &input.node_id {
            self.edges.push(EdgeSpec {
                source: PortRef {
                    node_id: source_id.clone(),
                    port_name: input.port_name.clone(),
                },
                target: PortRef {
                    node_id: target_id.clone(),
                    port_name: target_port.to_string(),
                },
                contract: EdgeContract {
                    kind: PortKind::Data,
                    representation: input.representation.clone(),
                    requires_oof: false,
                    requires_fold_alignment: true,
                    propagates_lineage: true,
                },
            });
        }
        Ok(())
    }
}

impl PipelineDslShapePlan {
    fn to_data_model_shape_plan(&self, node_id: &NodeId) -> Result<DataModelShapePlan> {
        let plan = DataModelShapePlan {
            node_id: node_id.clone(),
            input_granularity: self.input_granularity.unwrap_or(Granularity::Sample),
            target_granularity: self.target_granularity.unwrap_or(Granularity::Sample),
            fit_rows: self.fit_rows.unwrap_or(FitBoundary::FoldTrain),
            predict_rows: self.predict_rows.unwrap_or(FitBoundary::FoldValidation),
            feature_namespace: self.feature_namespace.clone(),
            feature_schema_fingerprint: self.feature_schema_fingerprint.clone(),
            target_space: self
                .target_space
                .clone()
                .unwrap_or_else(|| "raw".to_string()),
            aggregation_policy: self.aggregation_policy.clone().unwrap_or_default(),
            augmentation_policy: self.augmentation_policy.clone().unwrap_or_default(),
            selection_policy: self.selection_policy.clone().unwrap_or_default(),
        };
        plan.validate()?;
        Ok(plan)
    }
}

fn validate_shape_plan_targets(
    shape_plans: &BTreeMap<NodeId, DataModelShapePlan>,
    graph: &GraphSpec,
) -> Result<()> {
    for (node_id, plan) in shape_plans {
        if node_id != &plan.node_id {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL shape plan key `{node_id}` does not match node_id `{}`",
                plan.node_id
            )));
        }
        if !graph.nodes.iter().any(|node| &node.id == node_id) {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL shape plan references unknown node `{node_id}`"
            )));
        }
    }
    Ok(())
}

fn compile_explicit_generation_dimensions(
    dimensions: &[PipelineDslGenerationDimension],
    nodes: &[NodeSpec],
) -> Result<Vec<GenerationDimension>> {
    if dimensions.is_empty() {
        return Ok(Vec::new());
    }
    let node_ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    dimensions
        .iter()
        .map(|dimension| compile_explicit_generation_dimension(dimension, &node_ids))
        .collect()
}

fn compile_explicit_generation_dimension(
    dimension: &PipelineDslGenerationDimension,
    node_ids: &BTreeSet<NodeId>,
) -> Result<GenerationDimension> {
    let choices = dimension
        .choices
        .iter()
        .map(|choice| compile_explicit_generation_choice(&dimension.name, choice, node_ids))
        .collect::<Result<Vec<_>>>()?;
    Ok(GenerationDimension {
        name: dimension.name.clone(),
        choices,
    })
}

fn compile_explicit_generation_choice(
    dimension_name: &str,
    choice: &PipelineDslGenerationChoice,
    node_ids: &BTreeSet<NodeId>,
) -> Result<GenerationChoice> {
    if choice.param_overrides.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generation choice `{}` in dimension `{dimension_name}` has no param_overrides",
            choice.label
        )));
    }
    let param_overrides = choice
        .param_overrides
        .iter()
        .map(|override_spec| {
            if !node_ids.contains(&override_spec.node_id) {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL generation choice `{}` in dimension `{dimension_name}` references unknown node `{}`",
                    choice.label, override_spec.node_id
                )));
            }
            Ok(GenerationParamOverride {
                node_id: override_spec.node_id.clone(),
                params: override_spec.params.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let value = match &choice.value {
        Some(value) => value.clone(),
        None => explicit_generation_choice_value(&param_overrides)?,
    };
    Ok(GenerationChoice {
        label: choice.label.clone(),
        value,
        param_overrides,
    })
}

fn explicit_generation_choice_value(
    param_overrides: &[GenerationParamOverride],
) -> Result<serde_json::Value> {
    let mut by_node = serde_json::Map::new();
    for override_spec in param_overrides {
        let value = serde_json::to_value(&override_spec.params).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize DSL generation override for node `{}`: {error}",
                override_spec.node_id
            ))
        })?;
        by_node.insert(override_spec.node_id.to_string(), value);
    }
    Ok(serde_json::Value::Object(by_node))
}

fn build_campaign_template(
    spec: &PipelineDslSpec,
    generation: &GenerationSpec,
    shape_plans: &BTreeMap<NodeId, DataModelShapePlan>,
    data_bindings: &BTreeMap<NodeId, Vec<DataBinding>>,
) -> Result<CampaignSpec> {
    let campaign = CampaignSpec {
        id: spec
            .campaign_id
            .clone()
            .unwrap_or_else(|| format!("campaign:{}", spec.id)),
        root_seed: spec.root_seed,
        leakage_policy: spec.leakage_policy.clone().unwrap_or_default(),
        aggregation_policy: spec.aggregation_policy.clone().unwrap_or_default(),
        split_invocation: spec.split_invocation.clone(),
        generation: generation.clone(),
        shape_plans: shape_plans.clone(),
        data_bindings: data_bindings.clone(),
        metadata: spec.campaign_metadata.clone(),
    };
    campaign.validate()?;
    Ok(campaign)
}

fn compile_data_bindings(
    bindings: &[DataBinding],
    graph: &GraphSpec,
) -> Result<BTreeMap<NodeId, Vec<DataBinding>>> {
    let mut by_node = BTreeMap::<NodeId, Vec<DataBinding>>::new();
    for binding in bindings {
        validate_dsl_data_binding(binding, graph)?;
        by_node
            .entry(binding.node_id.clone())
            .or_default()
            .push(binding.clone());
    }
    Ok(by_node)
}

fn validate_dsl_data_binding(binding: &DataBinding, graph: &GraphSpec) -> Result<()> {
    binding.validate()?;
    let node = graph
        .nodes
        .iter()
        .find(|node| node.id == binding.node_id)
        .ok_or_else(|| {
            DagMlError::GraphValidation(format!(
                "pipeline DSL data binding references unknown node `{}`",
                binding.node_id
            ))
        })?;
    let Some(input_port) = node
        .ports
        .inputs
        .iter()
        .find(|port| port.name == binding.input_name)
    else {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL data binding `{}` references unknown input port `{}` on node `{}`",
            binding.request_id, binding.input_name, binding.node_id
        )));
    };
    if input_port.kind != PortKind::Data {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL data binding `{}` targets non-data input `{}.{}`",
            binding.request_id, binding.node_id, binding.input_name
        )));
    }
    Ok(())
}

fn build_generation_spec(
    requested_strategy: Option<GenerationStrategy>,
    max_variants: Option<usize>,
    dimensions: Vec<GenerationDimension>,
) -> Result<GenerationSpec> {
    let strategy = requested_strategy.unwrap_or(if dimensions.is_empty() {
        GenerationStrategy::None
    } else {
        GenerationStrategy::Cartesian
    });
    let generation = GenerationSpec {
        strategy,
        dimensions,
        max_variants: if strategy == GenerationStrategy::None {
            Some(1)
        } else {
            max_variants
        },
    };
    generation.validate()?;
    Ok(generation)
}

fn data_port(name: &str, representation: Option<String>, description: &str) -> PortSpec {
    PortSpec {
        name: name.to_string(),
        kind: PortKind::Data,
        representation,
        cardinality: PortCardinality::One,
        description: description.to_string(),
    }
}

fn prediction_port(name: &str, description: &str) -> PortSpec {
    PortSpec {
        name: name.to_string(),
        kind: PortKind::Prediction,
        representation: None,
        cardinality: PortCardinality::One,
        description: description.to_string(),
    }
}

fn validate_branch_id(branch_id: &str) -> Result<()> {
    if branch_id.trim().is_empty() {
        return Err(DagMlError::GraphValidation(
            "pipeline DSL branch id must not be empty".to_string(),
        ));
    }
    if !branch_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
    {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL branch id `{branch_id}` contains unsupported characters"
        )));
    }
    Ok(())
}

fn branch_input_prefix(branch_id: &str, index: usize) -> String {
    let sanitized = branch_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if sanitized.is_empty() {
        format!("branch{index}")
    } else {
        sanitized
    }
}

fn default_input_name() -> String {
    "x".to_string()
}

fn default_output_name() -> String {
    "prediction".to_string()
}

fn default_data_representation() -> String {
    "tabular_numeric".to_string()
}

fn default_true() -> bool {
    true
}

fn default_merge_mode() -> String {
    "predictions_plus_original".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_linear_pipeline_dsl_to_valid_graph() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-linear-smoke",
  "steps": [
    {
      "kind": "transform",
      "id": "transform:snv",
      "operator": {"type": "StandardNormalVariate"},
      "seed_label": "snv"
    },
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "RandomForestRegressor"},
      "params": {"n_estimators": 100},
      "seed_label": "base"
    }
  ]
}"#,
        )
        .unwrap();

        let graph = compile_pipeline_dsl(&spec).unwrap();

        assert_eq!(graph.id, "dsl-linear-smoke");
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.nodes[0].kind, NodeKind::Transform);
        assert_eq!(graph.nodes[1].kind, NodeKind::Model);
        assert_eq!(graph.edges[0].source.node_id.as_str(), "transform:snv");
        assert_eq!(graph.edges[0].target.node_id.as_str(), "model:base");
        assert_eq!(graph.edges[0].contract.kind, PortKind::Data);
        graph.validate().unwrap();
    }

    #[test]
    fn compiles_branch_merge_predictions_plus_original_dsl() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-branch-merge-smoke",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b0.model:ridge",
              "operator": {"type": "Ridge"},
              "params": {"alpha": 0.3},
              "seed_label": "branch:b0"
            }
          ]
        },
        {
          "id": "b1",
          "steps": [
            {
              "kind": "augmentation",
              "id": "branch:b1.augment:noise",
              "operator": {"type": "GaussianNoise"},
              "params": {"scope": "train_only"},
              "seed_label": "branch:b1.augment",
              "shape": {
                "fit_rows": "fold_train",
                "predict_rows": "fold_validation",
                "augmentation_policy": {
                  "sample_scope": "train_only",
                  "feature_scope": "none",
                  "require_origin_id": true,
                  "inherit_group": true,
                  "inherit_target": true
                }
              }
            },
            {
              "kind": "model",
              "id": "branch:b1.model:rf",
              "operator": {"type": "RandomForestRegressor"},
              "params": {"n_estimators": 64},
              "seed_label": "branch:b1"
            }
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "merge:stack.pred_plus_original.meta:ridge",
      "operator": {"type": "RidgeMetaStacker"},
      "params": {"alpha": 0.2},
      "seed_label": "merge:stack"
    }
  ]
}"#,
        )
        .unwrap();

        let graph = compile_pipeline_dsl(&spec).unwrap();

        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 3);
        let merge = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "merge:stack.pred_plus_original.meta:ridge")
            .unwrap();
        assert_eq!(merge.ports.inputs.len(), 3);
        assert_eq!(merge.ports.inputs[0].name, "b0_oof");
        assert_eq!(merge.ports.inputs[1].name, "b1_oof");
        assert_eq!(merge.ports.inputs[2].name, "x_original");
        let prediction_edges = graph
            .edges
            .iter()
            .filter(|edge| edge.contract.kind == PortKind::Prediction)
            .collect::<Vec<_>>();
        assert_eq!(prediction_edges.len(), 2);
        assert!(prediction_edges
            .iter()
            .all(|edge| edge.contract.requires_oof));
        assert!(prediction_edges
            .iter()
            .all(|edge| edge.contract.requires_fold_alignment));
        assert!(graph.edges.iter().any(|edge| edge.source.node_id.as_str()
            == "branch:b1.augment:noise"
            && edge.target.node_id.as_str() == "branch:b1.model:rf"));
        graph.validate().unwrap();
    }

    #[test]
    fn extracts_node_param_variants_into_generation_spec() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-generation-smoke",
  "max_variants": 4,
  "steps": [
    {
      "kind": "transform",
      "id": "transform:preprocess",
      "operator": {"type": "Preprocess"},
      "variants": [
        {
          "label": "snv",
          "params": {"method": "snv"}
        },
        {
          "label": "msc",
          "params": {"method": "msc"}
        }
      ]
    },
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"},
      "variants": [
        {
          "label": "low",
          "params": {"alpha": 0.1}
        },
        {
          "label": "high",
          "params": {"alpha": 1.0}
        }
      ]
    }
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

        assert_eq!(compiled.generation.strategy, GenerationStrategy::Cartesian);
        assert_eq!(compiled.generation.max_variants, Some(4));
        assert_eq!(compiled.generation.dimensions.len(), 2);
        assert_eq!(
            compiled.generation.dimensions[0].name,
            "transform:preprocess.params"
        );
        assert_eq!(compiled.generation.dimensions[0].choices[0].label, "snv");
        assert_eq!(
            compiled.generation.dimensions[0].choices[0].param_overrides[0].node_id,
            NodeId::new("transform:preprocess").unwrap()
        );
        assert_eq!(
            compiled.generation.dimensions[1].choices[1].param_overrides[0].params["alpha"],
            1.0
        );
        assert!(compiled.generation_fingerprint.is_some());
        assert_eq!(
            compiled.graph.search_space_fingerprint,
            compiled.generation_fingerprint
        );
        compiled.graph.validate().unwrap();
    }

    #[test]
    fn compiles_coordinated_generation_dimensions() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-coordinated-generation",
  "max_variants": 2,
  "generation_dimensions": [
    {
      "name": "stack_profile",
      "choices": [
        {
          "label": "linear_stack",
          "param_overrides": [
            {"node_id": "branch:b0.model:ridge", "params": {"alpha": 0.1}},
            {"node_id": "branch:b1.model:rf", "params": {"max_depth": 4}},
            {"node_id": "merge:stack.pred_plus_original.meta:ridge", "params": {"alpha": 0.05}}
          ]
        },
        {
          "label": "robust_stack",
          "param_overrides": [
            {"node_id": "branch:b0.model:ridge", "params": {"alpha": 1.0}},
            {"node_id": "branch:b1.model:rf", "params": {"max_depth": 8}},
            {"node_id": "merge:stack.pred_plus_original.meta:ridge", "params": {"alpha": 0.5}}
          ]
        }
      ]
    }
  ],
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b0.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        },
        {
          "id": "b1",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b1.model:rf",
              "operator": {"type": "RandomForestRegressor"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "merge:stack.pred_plus_original.meta:ridge",
      "operator": {"type": "RidgeMetaStacker"}
    }
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

        assert_eq!(compiled.generation.strategy, GenerationStrategy::Cartesian);
        assert_eq!(compiled.generation.max_variants, Some(2));
        assert_eq!(compiled.generation.dimensions.len(), 1);
        assert_eq!(compiled.generation.dimensions[0].name, "stack_profile");
        assert_eq!(
            compiled.generation.dimensions[0].choices[0]
                .param_overrides
                .len(),
            3
        );
        assert_eq!(
            compiled.generation.dimensions[0].choices[1].param_overrides[2].node_id,
            NodeId::new("merge:stack.pred_plus_original.meta:ridge").unwrap()
        );
        assert_eq!(
            compiled.generation.dimensions[0].choices[1].value
                ["merge:stack.pred_plus_original.meta:ridge"]["alpha"],
            0.5
        );
        assert_eq!(
            compiled.graph.search_space_fingerprint,
            compiled.generation_fingerprint
        );
        compiled.graph.validate().unwrap();
    }

    #[test]
    fn refuses_coordinated_generation_for_unknown_node() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-generation-target",
  "generation_dimensions": [
    {
      "name": "bad_target",
      "choices": [
        {
          "label": "bad",
          "param_overrides": [
            {"node_id": "model:missing", "params": {"alpha": 0.1}}
          ]
        }
      ]
    }
  ],
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ]
}"#,
        )
        .unwrap();

        let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
        assert!(format!("{error}").contains("references unknown node `model:missing`"));
    }

    #[test]
    fn artifact_contains_campaign_template_without_split_graph_nodes() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-campaign-template",
  "campaign_id": "campaign:dsl.template",
  "root_seed": 123,
  "leakage_policy": {
    "split_unit": "group",
    "require_group_ids": true
  },
  "split_invocation": {
    "id": "split:group-kfold",
    "leakage_policy": {
      "split_unit": "group",
      "require_group_ids": true
    },
    "params": {
      "n_splits": 3
    }
  },
  "generation_dimensions": [
    {
      "name": "model_family",
      "choices": [
        {
          "label": "ridge_low",
          "param_overrides": [
            {"node_id": "model:base", "params": {"alpha": 0.1}}
          ]
        },
        {
          "label": "ridge_high",
          "param_overrides": [
            {"node_id": "model:base", "params": {"alpha": 1.0}}
          ]
        }
      ]
    }
  ],
  "data_bindings": [
    {
      "node_id": "model:base",
      "input_name": "x",
      "request_id": "data:model.base.x",
      "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
      "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
      "relation_fingerprint": "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10",
      "output_representation": "tabular_numeric",
      "feature_set_id": "x",
      "source_ids": ["nir"],
      "require_relations": true
    }
  ],
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ],
  "campaign_metadata": {
    "owner": "dsl-test"
  }
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

        assert_eq!(compiled.campaign_template.id, "campaign:dsl.template");
        assert_eq!(compiled.campaign_template.root_seed, Some(123));
        assert_eq!(
            compiled
                .campaign_template
                .split_invocation
                .as_ref()
                .unwrap()
                .id,
            "split:group-kfold"
        );
        assert_eq!(compiled.campaign_template.generation, compiled.generation);
        assert_eq!(
            compiled.data_bindings[&NodeId::new("model:base").unwrap()][0].request_id,
            "data:model.base.x"
        );
        assert_eq!(
            compiled.campaign_template.data_bindings,
            compiled.data_bindings
        );
        assert_eq!(compiled.graph.nodes.len(), 1);
        assert!(compiled
            .graph
            .nodes
            .iter()
            .all(|node| !node.id.as_str().starts_with("split:")));
    }

    #[test]
    fn refuses_data_binding_for_unknown_or_non_data_port() {
        let unknown_input_spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-data-binding",
  "data_bindings": [
    {
      "node_id": "model:base",
      "input_name": "missing",
      "request_id": "data:bad",
      "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
      "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
      "output_representation": "tabular_numeric"
    }
  ],
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ]
}"#,
        )
        .unwrap();
        let error = compile_pipeline_dsl_with_generation(&unknown_input_spec).unwrap_err();
        assert!(format!("{error}").contains("unknown input port `missing`"));

        let prediction_input_spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-prediction-port-data-binding",
  "data_bindings": [
    {
      "node_id": "merge:stack.pred_plus_original.meta:ridge",
      "input_name": "b0_oof",
      "request_id": "data:bad.prediction-port",
      "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
      "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
      "output_representation": "tabular_numeric"
    }
  ],
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b0.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "merge:stack.pred_plus_original.meta:ridge",
      "operator": {"type": "RidgeMetaStacker"}
    }
  ]
}"#,
        )
        .unwrap();
        let error = compile_pipeline_dsl_with_generation(&prediction_input_spec).unwrap_err();
        assert!(format!("{error}").contains("targets non-data input"));
    }

    #[test]
    fn extracts_shape_plans_into_compiled_artifact() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-shape-plan-smoke",
  "steps": [
    {
      "kind": "augmentation",
      "id": "augment:synthetic",
      "operator": {"type": "SampleAugmenter"},
      "shape": {
        "input_granularity": "sample",
        "target_granularity": "sample",
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "aug.synthetic",
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "none",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        }
      }
    },
    {
      "kind": "transform",
      "id": "transform:select",
      "operator": {"type": "SupervisedFeatureSelector"},
      "shape": {
        "fit_rows": "fold_train",
        "feature_namespace": "selected",
        "selection_policy": {
          "scope": "supervised_fold_train",
          "store_masks": true
        }
      }
    },
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

        assert_eq!(compiled.shape_plans.len(), 2);
        let augment_plan = compiled
            .shape_plans
            .get(&NodeId::new("augment:synthetic").unwrap())
            .unwrap();
        assert_eq!(
            augment_plan.feature_namespace.as_deref(),
            Some("aug.synthetic")
        );
        assert_eq!(
            augment_plan.augmentation_policy.sample_scope,
            crate::policy::AugmentationScope::TrainOnly
        );
        let select_plan = compiled
            .shape_plans
            .get(&NodeId::new("transform:select").unwrap())
            .unwrap();
        assert_eq!(
            select_plan.selection_policy.scope,
            crate::policy::FeatureSelectionScope::SupervisedFoldTrain
        );
        assert_eq!(compiled.generation.strategy, GenerationStrategy::None);
        compiled.graph.validate().unwrap();
    }

    #[test]
    fn refuses_unsafe_shape_plan_from_dsl() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-unsafe-shape-plan",
  "steps": [
    {
      "kind": "augmentation",
      "id": "augment:bad",
      "operator": {"type": "LeakyAugmenter"},
      "shape": {
        "augmentation_policy": {
          "sample_scope": "all_partitions"
        }
      }
    }
  ]
}"#,
        )
        .unwrap();

        let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
        assert!(format!("{error}").contains("sample augmentation over all partitions"));
    }

    #[test]
    fn refuses_augmentation_without_shape_plan() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-augmentation-without-shape",
  "steps": [
    {
      "kind": "augmentation",
      "id": "augment:missing-shape",
      "operator": {"type": "GaussianNoise"}
    }
  ]
}"#,
        )
        .unwrap();

        let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
        assert!(format!("{error}").contains("requires a shape plan"));
    }

    #[test]
    fn refuses_branch_without_model_output() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-branch",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "transform",
              "id": "transform:only",
              "operator": {"type": "SNV"}
            }
          ]
        }
      ]
    }
  ]
}"#,
        )
        .unwrap();

        let error = compile_pipeline_dsl(&spec).unwrap_err();
        assert!(format!("{error}").contains("must end with a model step"));
    }
}
