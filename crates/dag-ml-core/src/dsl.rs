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
    YTransform(PipelineDslOperatorStep),
    Tag(PipelineDslOperatorStep),
    Exclude(PipelineDslOperatorStep),
    Augmentation(PipelineDslOperatorStep),
    FeatureAugmentation(PipelineDslOperatorStep),
    SampleAugmentation(PipelineDslOperatorStep),
    ConcatTransform(PipelineDslConcatTransformStep),
    Model(PipelineDslOperatorStep),
    Branch(PipelineDslBranchStep),
    Merge(PipelineDslMergeStep),
    MergeModel(PipelineDslMergeModelStep),
    Chart(PipelineDslOperatorStep),
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub train_params: BTreeMap<String, serde_json::Value>,
    #[serde(
        default,
        alias = "finetune_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub tuning: Option<PipelineDslTuningSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<PipelineDslVariantChoice>,
    #[serde(default, alias = "generators", skip_serializing_if = "Vec::is_empty")]
    pub param_generators: Vec<PipelineDslParamGenerator>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<PipelineDslShapePlan>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslTuningSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_trials: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approach: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub model_params: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub train_params: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
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
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PipelineDslParamGenerator {
    Or {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        param: String,
        values: Vec<PipelineDslGeneratorValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<usize>,
    },
    Range {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        param: String,
        start: f64,
        stop: f64,
        step: f64,
        #[serde(default = "default_true")]
        inclusive: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<usize>,
    },
    LogRange {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        param: String,
        start: f64,
        stop: f64,
        count: usize,
        #[serde(default = "default_log_base")]
        base: f64,
    },
    Grid {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        params: BTreeMap<String, Vec<PipelineDslGeneratorValue>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<usize>,
    },
    Pick {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        param: String,
        values: Vec<PipelineDslGeneratorValue>,
        sizes: Vec<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<usize>,
    },
    Arrange {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        param: String,
        values: Vec<PipelineDslGeneratorValue>,
        sizes: Vec<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PipelineDslGeneratorValue {
    Labeled {
        label: String,
        value: serde_json::Value,
    },
    Value(serde_json::Value),
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
    #[serde(default)]
    pub mode: PipelineDslBranchMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub branches: Vec<PipelineDslBranch>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineDslBranchMode {
    #[default]
    Duplication,
    Separation,
    BySource,
    ByMetadata,
    ByTag,
    ByFilter,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslBranch {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub steps: Vec<PipelineDslStep>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslConcatTransformStep {
    pub id: NodeId,
    #[serde(default)]
    pub branches: Vec<PipelineDslConcatBranch>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub seed_label: Option<String>,
    #[serde(default)]
    pub representation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<PipelineDslVariantChoice>,
    #[serde(default, alias = "generators", skip_serializing_if = "Vec::is_empty")]
    pub param_generators: Vec<PipelineDslParamGenerator>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<PipelineDslShapePlan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslConcatBranch {
    pub id: String,
    #[serde(default)]
    pub steps: Vec<PipelineDslOperatorStep>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslMergeStep {
    pub id: NodeId,
    #[serde(default = "default_merge_mode")]
    pub merge_mode: String,
    #[serde(default)]
    pub output_as: PipelineDslMergeOutput,
    #[serde(default = "default_true")]
    pub include_original_data: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_missing: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selectors: Vec<PipelineDslMergeSelector>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub seed_label: Option<String>,
    #[serde(default)]
    pub representation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<PipelineDslVariantChoice>,
    #[serde(default, alias = "generators", skip_serializing_if = "Vec::is_empty")]
    pub param_generators: Vec<PipelineDslParamGenerator>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<PipelineDslShapePlan>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineDslMergeOutput {
    #[default]
    Features,
    Predictions,
    Sources,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslMergeSelector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<NodeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub select: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub train_params: BTreeMap<String, serde_json::Value>,
    #[serde(
        default,
        alias = "finetune_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub tuning: Option<PipelineDslTuningSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<PipelineDslVariantChoice>,
    #[serde(default, alias = "generators", skip_serializing_if = "Vec::is_empty")]
    pub param_generators: Vec<PipelineDslParamGenerator>,
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
    branch_id: Option<String>,
}

#[derive(Clone, Debug)]
enum MergeOutputSource {
    Data(DataSource),
    Prediction(PredictionSource),
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
            PipelineDslStep::YTransform(step) => {
                self.compile_y_transform(step)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Tag(step) => {
                *current_data = self.compile_data_operator(NodeKind::Tag, step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Exclude(step) => {
                *current_data =
                    self.compile_data_operator(NodeKind::Exclude, step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Augmentation(step) => {
                *current_data =
                    self.compile_data_operator(NodeKind::Augmentation, step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::FeatureAugmentation(step) => {
                *current_data =
                    self.compile_augmentation_operator("feature", step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::SampleAugmentation(step) => {
                *current_data = self.compile_augmentation_operator("sample", step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::ConcatTransform(step) => {
                *current_data = self.compile_concat_transform(step, current_data)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Model(step) => {
                pending_predictions.push(self.compile_model(step, current_data, None)?);
                Ok(())
            }
            PipelineDslStep::Branch(step) => {
                *pending_predictions = self.compile_branch(step, current_data)?;
                Ok(())
            }
            PipelineDslStep::Merge(step) => {
                match self.compile_merge(step, pending_predictions, external_data)? {
                    MergeOutputSource::Data(data) => {
                        *current_data = data;
                        pending_predictions.clear();
                    }
                    MergeOutputSource::Prediction(prediction) => {
                        pending_predictions.clear();
                        pending_predictions.push(prediction);
                    }
                }
                Ok(())
            }
            PipelineDslStep::MergeModel(step) => {
                let prediction =
                    self.compile_merge_model(step, pending_predictions, external_data)?;
                pending_predictions.clear();
                pending_predictions.push(prediction);
                Ok(())
            }
            PipelineDslStep::Chart(step) => {
                *current_data = self.compile_data_operator(NodeKind::Chart, step, current_data)?;
                pending_predictions.clear();
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
        let mut predictions = Vec::new();
        for (index, branch) in step.branches.iter().enumerate() {
            validate_branch_id(&branch.id)?;
            if branch.steps.is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL branch `{}` has no steps",
                    branch.id
                )));
            }
            let mut branch_data = current_data.clone();
            let mut branch_predictions = Vec::new();
            let branch_metadata = branch_context_metadata(step, branch)?;
            for branch_step in &branch.steps {
                match branch_step {
                    PipelineDslStep::Transform(step) => {
                        branch_data = self.compile_data_operator_with_extra(
                            NodeKind::Transform,
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::YTransform(step) => {
                        self.compile_y_transform_with_extra(step, branch_metadata.clone())?;
                    }
                    PipelineDslStep::Tag(step) => {
                        branch_data = self.compile_data_operator_with_extra(
                            NodeKind::Tag,
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::Exclude(step) => {
                        branch_data = self.compile_data_operator_with_extra(
                            NodeKind::Exclude,
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::Augmentation(step) => {
                        branch_data = self.compile_data_operator_with_extra(
                            NodeKind::Augmentation,
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::FeatureAugmentation(step) => {
                        branch_data = self.compile_augmentation_operator_with_extra(
                            "feature",
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::SampleAugmentation(step) => {
                        branch_data = self.compile_augmentation_operator_with_extra(
                            "sample",
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::ConcatTransform(step) => {
                        branch_data = self.compile_concat_transform_with_extra(
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                    PipelineDslStep::Model(step) => {
                        branch_predictions.push(self.compile_model_with_extra(
                            step,
                            &branch_data,
                            Some(&branch.id),
                            branch_metadata.clone(),
                        )?);
                    }
                    PipelineDslStep::Branch(step) => {
                        branch_predictions.extend(self.compile_branch(step, &branch_data)?);
                    }
                    PipelineDslStep::Merge(step) => {
                        match self.compile_merge_with_extra(
                            step,
                            &branch_predictions,
                            current_data,
                            branch_metadata.clone(),
                        )? {
                            MergeOutputSource::Data(data) => {
                                branch_data = data;
                                branch_predictions.clear();
                            }
                            MergeOutputSource::Prediction(prediction) => {
                                branch_predictions.clear();
                                branch_predictions.push(prediction);
                            }
                        }
                    }
                    PipelineDslStep::MergeModel(step) => {
                        let prediction = self.compile_merge_model_with_extra(
                            step,
                            &branch_predictions,
                            current_data,
                            branch_metadata.clone(),
                        )?;
                        branch_predictions.clear();
                        branch_predictions.push(prediction);
                    }
                    PipelineDslStep::Chart(step) => {
                        branch_data = self.compile_data_operator_with_extra(
                            NodeKind::Chart,
                            step,
                            &branch_data,
                            branch_metadata.clone(),
                        )?;
                    }
                }
            }
            if branch_predictions.is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL branch `{}` must produce at least one model or merge prediction",
                    branch.id
                )));
            }
            let prediction_count = branch_predictions.len();
            for (prediction_index, prediction) in branch_predictions.into_iter().enumerate() {
                let input_name = if prediction_count == 1 {
                    format!("{}_oof", branch_input_prefix(&branch.id, index))
                } else {
                    branch_prediction_input_name(
                        &branch.id,
                        index,
                        prediction_index,
                        &prediction.node_id,
                    )
                };
                predictions.push(PredictionSource {
                    input_name,
                    ..prediction
                });
            }
        }
        Ok(predictions)
    }

    fn compile_data_operator(
        &mut self,
        kind: NodeKind,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
    ) -> Result<DataSource> {
        self.compile_data_operator_with_extra(kind, step, input, BTreeMap::new())
    }

    fn compile_augmentation_operator(
        &mut self,
        augmentation_kind: &str,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
    ) -> Result<DataSource> {
        self.compile_augmentation_operator_with_extra(
            augmentation_kind,
            step,
            input,
            BTreeMap::new(),
        )
    }

    fn compile_augmentation_operator_with_extra(
        &mut self,
        augmentation_kind: &str,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
        mut extra: BTreeMap<String, serde_json::Value>,
    ) -> Result<DataSource> {
        extra.insert(
            "dsl_augmentation_kind".to_string(),
            serde_json::Value::String(augmentation_kind.to_string()),
        );
        self.compile_data_operator_with_extra(NodeKind::Augmentation, step, input, extra)
    }

    fn compile_data_operator_with_extra(
        &mut self,
        kind: NodeKind,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
        extra_metadata: BTreeMap<String, serde_json::Value>,
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
        let mut metadata = operator_runtime_metadata(step, None)?;
        metadata.extend(extra_metadata);
        let node = NodeSpec {
            id: step.id.clone(),
            kind,
            operator: Some(step.operator.clone()),
            params: step.params.clone(),
            ports: PortSchema {
                inputs: vec![data_port("x", input.representation.clone(), "")],
                outputs: vec![data_port("x_out", representation.clone(), "")],
            },
            metadata,
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants, &step.param_generators)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())?;
        self.connect_data(input, &step.id, "x")?;
        Ok(DataSource {
            node_id: Some(step.id.clone()),
            port_name: "x_out".to_string(),
            representation,
        })
    }

    fn compile_y_transform(&mut self, step: &PipelineDslOperatorStep) -> Result<()> {
        self.compile_y_transform_with_extra(step, BTreeMap::new())
    }

    fn compile_y_transform_with_extra(
        &mut self,
        step: &PipelineDslOperatorStep,
        extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<()> {
        let mut metadata = operator_runtime_metadata(step, None)?;
        metadata.extend(extra_metadata);
        let node = NodeSpec {
            id: step.id.clone(),
            kind: NodeKind::YTransform,
            operator: Some(step.operator.clone()),
            params: step.params.clone(),
            ports: PortSchema {
                inputs: vec![target_port("y", "")],
                outputs: vec![target_port("y_out", "")],
            },
            metadata,
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants, &step.param_generators)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())
    }

    fn compile_concat_transform(
        &mut self,
        step: &PipelineDslConcatTransformStep,
        input: &DataSource,
    ) -> Result<DataSource> {
        self.compile_concat_transform_with_extra(step, input, BTreeMap::new())
    }

    fn compile_concat_transform_with_extra(
        &mut self,
        step: &PipelineDslConcatTransformStep,
        input: &DataSource,
        extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<DataSource> {
        if step.branches.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL concat_transform `{}` has no branches",
                step.id
            )));
        }
        let representation = step
            .representation
            .clone()
            .or_else(|| input.representation.clone())
            .or_else(|| self.input_representation.clone());
        let mut branch_outputs = Vec::with_capacity(step.branches.len());
        for (index, branch) in step.branches.iter().enumerate() {
            validate_branch_id(&branch.id)?;
            let mut branch_data = input.clone();
            for branch_step in &branch.steps {
                branch_data =
                    self.compile_data_operator(NodeKind::Transform, branch_step, &branch_data)?;
            }
            let input_name = format!("{}_x", branch_input_prefix(&branch.id, index));
            branch_outputs.push((input_name, branch_data));
        }
        let node = NodeSpec {
            id: step.id.clone(),
            kind: NodeKind::FeatureJoin,
            operator: None,
            params: BTreeMap::new(),
            ports: PortSchema {
                inputs: branch_outputs
                    .iter()
                    .map(|(name, source)| data_port(name, source.representation.clone(), ""))
                    .collect(),
                outputs: vec![data_port("x_out", representation.clone(), "")],
            },
            metadata: {
                let mut metadata = step.metadata.clone();
                metadata.extend(extra_metadata);
                metadata
            },
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants, &step.param_generators)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())?;
        for (input_name, source) in &branch_outputs {
            self.connect_data_to_port(source, &step.id, input_name)?;
        }
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
        self.compile_model_with_extra(step, input, branch_id, BTreeMap::new())
    }

    fn compile_model_with_extra(
        &mut self,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
        branch_id: Option<&str>,
        extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<PredictionSource> {
        let mut metadata = operator_runtime_metadata(step, branch_id)?;
        metadata.extend(extra_metadata);
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
        self.collect_operator_generation(&step.id, &step.variants, &step.param_generators)?;
        self.collect_shape_plan(&step.id, step.shape.as_ref())?;
        self.connect_data(input, &step.id, "x")?;
        Ok(PredictionSource {
            node_id: step.id.clone(),
            port_name: "oof".to_string(),
            input_name: "oof".to_string(),
            branch_id: branch_id.map(str::to_string),
        })
    }

    fn compile_merge(
        &mut self,
        step: &PipelineDslMergeStep,
        predictions: &[PredictionSource],
        original_data: &DataSource,
    ) -> Result<MergeOutputSource> {
        self.compile_merge_with_extra(step, predictions, original_data, BTreeMap::new())
    }

    fn compile_merge_with_extra(
        &mut self,
        step: &PipelineDslMergeStep,
        predictions: &[PredictionSource],
        original_data: &DataSource,
        extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<MergeOutputSource> {
        if predictions.is_empty() && !step.include_original_data {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL merge `{}` has no pending predictions and no original data input",
                step.id
            )));
        }
        validate_merge_selectors(&step.id, &step.selectors, predictions)?;
        let outputs_prediction = step.output_as == PipelineDslMergeOutput::Predictions;
        let representation = step
            .representation
            .clone()
            .or_else(|| original_data.representation.clone())
            .or_else(|| self.input_representation.clone());
        let mut input_ports = Vec::with_capacity(predictions.len() + 1);
        for prediction in predictions {
            input_ports.push(prediction_port(&prediction.input_name, ""));
        }
        if step.include_original_data {
            input_ports.push(data_port(
                "x_original",
                original_data.representation.clone(),
                "",
            ));
        }
        let mut metadata = step.metadata.clone();
        metadata.insert(
            "merge_mode".to_string(),
            serde_json::Value::String(step.merge_mode.clone()),
        );
        metadata.insert(
            "output_as".to_string(),
            serde_json::to_value(step.output_as).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL merge `{}` output mode: {error}",
                    step.id
                ))
            })?,
        );
        metadata.insert(
            "include_original_data".to_string(),
            serde_json::Value::Bool(step.include_original_data),
        );
        if let Some(on_missing) = &step.on_missing {
            metadata.insert(
                "on_missing".to_string(),
                serde_json::Value::String(on_missing.clone()),
            );
        }
        if !step.selectors.is_empty() {
            metadata.insert(
                "selectors".to_string(),
                serde_json::to_value(&step.selectors).map_err(|error| {
                    DagMlError::GraphValidation(format!(
                        "failed to serialize pipeline DSL merge `{}` selectors: {error}",
                        step.id
                    ))
                })?,
            );
        }
        let branch_id = branch_id_from_metadata(&extra_metadata);
        metadata.extend(extra_metadata);
        let node = NodeSpec {
            id: step.id.clone(),
            kind: merge_node_kind(step, !predictions.is_empty()),
            operator: None,
            params: BTreeMap::new(),
            ports: PortSchema {
                inputs: input_ports,
                outputs: if outputs_prediction {
                    vec![prediction_port("prediction", "")]
                } else {
                    vec![data_port("x_out", representation.clone(), "")]
                },
            },
            metadata,
            seed_label: step.seed_label.clone(),
        };
        self.push_node(node)?;
        self.collect_operator_generation(&step.id, &step.variants, &step.param_generators)?;
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
            self.connect_data_to_port(original_data, &step.id, "x_original")?;
        }
        if outputs_prediction {
            Ok(MergeOutputSource::Prediction(PredictionSource {
                node_id: step.id.clone(),
                port_name: "prediction".to_string(),
                input_name: "oof".to_string(),
                branch_id,
            }))
        } else {
            Ok(MergeOutputSource::Data(DataSource {
                node_id: Some(step.id.clone()),
                port_name: "x_out".to_string(),
                representation,
            }))
        }
    }

    fn compile_merge_model(
        &mut self,
        step: &PipelineDslMergeModelStep,
        predictions: &[PredictionSource],
        external_data: &DataSource,
    ) -> Result<PredictionSource> {
        self.compile_merge_model_with_extra(step, predictions, external_data, BTreeMap::new())
    }

    fn compile_merge_model_with_extra(
        &mut self,
        step: &PipelineDslMergeModelStep,
        predictions: &[PredictionSource],
        external_data: &DataSource,
        extra_metadata: BTreeMap<String, serde_json::Value>,
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
        insert_training_metadata(
            &mut metadata,
            &step.train_params,
            step.tuning.as_ref(),
            &step.id,
        )?;
        metadata.insert(
            "merge_mode".to_string(),
            serde_json::Value::String(step.merge_mode.clone()),
        );
        let branch_id = branch_id_from_metadata(&extra_metadata);
        metadata.extend(extra_metadata);
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
        self.collect_operator_generation(&step.id, &step.variants, &step.param_generators)?;
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
            branch_id,
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
        generators: &[PipelineDslParamGenerator],
    ) -> Result<()> {
        if !choices.is_empty() {
            self.generation_dimensions
                .push(compile_variant_choice_dimension(node_id, choices)?);
        }
        for generator in generators {
            self.generation_dimensions
                .push(compile_param_generator_dimension(node_id, generator)?);
        }
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

fn compile_variant_choice_dimension(
    node_id: &NodeId,
    choices: &[PipelineDslVariantChoice],
) -> Result<GenerationDimension> {
    Ok(GenerationDimension {
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
    })
}

fn compile_param_generator_dimension(
    node_id: &NodeId,
    generator: &PipelineDslParamGenerator,
) -> Result<GenerationDimension> {
    match generator {
        PipelineDslParamGenerator::Or {
            name,
            param,
            values,
            count,
        } => compile_or_generator(node_id, name.as_deref(), param, values, *count),
        PipelineDslParamGenerator::Range {
            name,
            param,
            start,
            stop,
            step,
            inclusive,
            count,
        } => compile_range_generator(RangeGeneratorSpec {
            node_id,
            name: name.as_deref(),
            param,
            start: *start,
            stop: *stop,
            step: *step,
            inclusive: *inclusive,
            count: *count,
        }),
        PipelineDslParamGenerator::LogRange {
            name,
            param,
            start,
            stop,
            count,
            base,
        } => compile_log_range_generator(
            node_id,
            name.as_deref(),
            param,
            *start,
            *stop,
            *count,
            *base,
        ),
        PipelineDslParamGenerator::Grid {
            name,
            params,
            count,
        } => compile_grid_generator(node_id, name.as_deref(), params, *count),
        PipelineDslParamGenerator::Pick {
            name,
            param,
            values,
            sizes,
            count,
        } => compile_pick_arrange_generator(
            node_id,
            name.as_deref(),
            param,
            values,
            sizes,
            *count,
            PickArrangeMode::Pick,
        ),
        PipelineDslParamGenerator::Arrange {
            name,
            param,
            values,
            sizes,
            count,
        } => compile_pick_arrange_generator(
            node_id,
            name.as_deref(),
            param,
            values,
            sizes,
            *count,
            PickArrangeMode::Arrange,
        ),
    }
}

fn compile_or_generator(
    node_id: &NodeId,
    name: Option<&str>,
    param: &str,
    values: &[PipelineDslGeneratorValue],
    count: Option<usize>,
) -> Result<GenerationDimension> {
    validate_param_name(node_id, param)?;
    validate_count(node_id, name, count)?;
    if values.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` for node `{node_id}` has no values",
            generator_dimension_name(node_id, name, Some(param), "or")
        )));
    }
    let mut choices = values
        .iter()
        .enumerate()
        .map(|(index, value)| single_param_generation_choice(node_id, param, index, value))
        .collect::<Result<Vec<_>>>()?;
    apply_choice_count(&mut choices, count);
    Ok(GenerationDimension {
        name: generator_dimension_name(node_id, name, Some(param), "or"),
        choices,
    })
}

struct RangeGeneratorSpec<'a> {
    node_id: &'a NodeId,
    name: Option<&'a str>,
    param: &'a str,
    start: f64,
    stop: f64,
    step: f64,
    inclusive: bool,
    count: Option<usize>,
}

fn compile_range_generator(spec: RangeGeneratorSpec<'_>) -> Result<GenerationDimension> {
    validate_param_name(spec.node_id, spec.param)?;
    validate_count(spec.node_id, spec.name, spec.count)?;
    validate_finite(spec.node_id, spec.param, "range start", spec.start)?;
    validate_finite(spec.node_id, spec.param, "range stop", spec.stop)?;
    validate_finite(spec.node_id, spec.param, "range step", spec.step)?;
    if spec.step == 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` has zero step",
            spec.node_id, spec.param
        )));
    }
    if spec.start < spec.stop && spec.step < 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` steps away from stop",
            spec.node_id, spec.param
        )));
    }
    if spec.start > spec.stop && spec.step > 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` steps away from stop",
            spec.node_id, spec.param
        )));
    }
    let mut values = Vec::new();
    let mut current = spec.start;
    let mut guard = 0usize;
    while range_contains(current, spec.stop, spec.step, spec.inclusive) {
        values.push(json_number(current, spec.node_id, spec.param)?);
        current += spec.step;
        guard += 1;
        if guard > 10_000 {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL range generator for `{}.{}` produced more than 10000 values",
                spec.node_id, spec.param
            )));
        }
    }
    if values.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` produced no values",
            spec.node_id, spec.param
        )));
    }
    let wrapped = values
        .into_iter()
        .map(PipelineDslGeneratorValue::Value)
        .collect::<Vec<_>>();
    compile_or_generator(spec.node_id, spec.name, spec.param, &wrapped, spec.count).map(
        |mut dimension| {
            dimension.name =
                generator_dimension_name(spec.node_id, spec.name, Some(spec.param), "range");
            dimension
        },
    )
}

fn compile_log_range_generator(
    node_id: &NodeId,
    name: Option<&str>,
    param: &str,
    start: f64,
    stop: f64,
    count: usize,
    base: f64,
) -> Result<GenerationDimension> {
    validate_param_name(node_id, param)?;
    validate_finite(node_id, param, "log_range start", start)?;
    validate_finite(node_id, param, "log_range stop", stop)?;
    validate_finite(node_id, param, "log_range base", base)?;
    if start <= 0.0 || stop <= 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL log_range generator for `{node_id}.{param}` requires positive start and stop"
        )));
    }
    if count == 0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL log_range generator for `{node_id}.{param}` has count=0"
        )));
    }
    if base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL log_range generator for `{node_id}.{param}` requires base > 0 and != 1"
        )));
    }
    let start_log = start.log(base);
    let stop_log = stop.log(base);
    let values = if count == 1 {
        vec![json_number(start, node_id, param)?]
    } else {
        (0..count)
            .map(|index| {
                let ratio = index as f64 / (count - 1) as f64;
                json_number(
                    base.powf(start_log + (stop_log - start_log) * ratio),
                    node_id,
                    param,
                )
            })
            .collect::<Result<Vec<_>>>()?
    };
    let wrapped = values
        .into_iter()
        .map(PipelineDslGeneratorValue::Value)
        .collect::<Vec<_>>();
    compile_or_generator(node_id, name, param, &wrapped, None).map(|mut dimension| {
        dimension.name = generator_dimension_name(node_id, name, Some(param), "log_range");
        dimension
    })
}

fn compile_grid_generator(
    node_id: &NodeId,
    name: Option<&str>,
    params: &BTreeMap<String, Vec<PipelineDslGeneratorValue>>,
    count: Option<usize>,
) -> Result<GenerationDimension> {
    validate_count(node_id, name, count)?;
    if params.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL grid generator for node `{node_id}` has no params"
        )));
    }
    for (param, values) in params {
        validate_param_name(node_id, param)?;
        if values.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL grid generator for `{node_id}.{param}` has no values"
            )));
        }
    }
    let entries = params
        .iter()
        .map(|(param, values)| (param.as_str(), values.as_slice()))
        .collect::<Vec<_>>();
    let mut rows = Vec::<BTreeMap<String, PipelineDslGeneratorValue>>::new();
    build_grid_rows(&entries, 0, &mut BTreeMap::new(), &mut rows, count);
    let choices = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| multi_param_generation_choice(node_id, index, row))
        .collect::<Result<Vec<_>>>()?;
    Ok(GenerationDimension {
        name: generator_dimension_name(node_id, name, None, "grid"),
        choices,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PickArrangeMode {
    Pick,
    Arrange,
}

fn compile_pick_arrange_generator(
    node_id: &NodeId,
    name: Option<&str>,
    param: &str,
    values: &[PipelineDslGeneratorValue],
    sizes: &[usize],
    count: Option<usize>,
    mode: PickArrangeMode,
) -> Result<GenerationDimension> {
    validate_param_name(node_id, param)?;
    validate_count(node_id, name, count)?;
    if values.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL {:?} generator for `{node_id}.{param}` has no values",
            mode
        )));
    }
    if sizes.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL {:?} generator for `{node_id}.{param}` has no sizes",
            mode
        )));
    }
    let mut selections = Vec::<Vec<usize>>::new();
    for size in sizes {
        if *size == 0 || *size > values.len() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL {:?} generator for `{node_id}.{param}` has invalid size `{size}`",
                mode
            )));
        }
        match mode {
            PickArrangeMode::Pick => build_combinations(
                values.len(),
                *size,
                0,
                &mut Vec::new(),
                &mut selections,
                count,
            ),
            PickArrangeMode::Arrange => build_permutations(
                values.len(),
                *size,
                &mut BTreeSet::new(),
                &mut Vec::new(),
                &mut selections,
                count,
            ),
        }
        if count.is_some_and(|limit| selections.len() >= limit) {
            break;
        }
    }
    let mut choices = selections
        .into_iter()
        .enumerate()
        .map(|(index, selection)| {
            let selected_values = selection
                .iter()
                .map(|selected| values[*selected].value().clone())
                .collect::<Vec<_>>();
            let selected_labels = selection
                .iter()
                .map(|selected| values[*selected].label_fragment())
                .collect::<Vec<_>>();
            let mut params = BTreeMap::new();
            params.insert(param.to_string(), serde_json::Value::Array(selected_values));
            Ok(GenerationChoice {
                label: format!(
                    "{index:04}_{}_{}",
                    match mode {
                        PickArrangeMode::Pick => "pick",
                        PickArrangeMode::Arrange => "arrange",
                    },
                    sanitize_generation_label(&selected_labels.join("_"))
                ),
                value: serde_json::to_value(&params).map_err(|error| {
                    DagMlError::GraphValidation(format!(
                        "failed to serialize pipeline DSL {:?} generator choice for `{node_id}.{param}`: {error}",
                        mode
                    ))
                })?,
                param_overrides: vec![GenerationParamOverride {
                    node_id: node_id.clone(),
                    params,
                }],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    apply_choice_count(&mut choices, count);
    Ok(GenerationDimension {
        name: generator_dimension_name(
            node_id,
            name,
            Some(param),
            match mode {
                PickArrangeMode::Pick => "pick",
                PickArrangeMode::Arrange => "arrange",
            },
        ),
        choices,
    })
}

fn single_param_generation_choice(
    node_id: &NodeId,
    param: &str,
    index: usize,
    value: &PipelineDslGeneratorValue,
) -> Result<GenerationChoice> {
    let mut params = BTreeMap::new();
    params.insert(param.to_string(), value.value().clone());
    Ok(GenerationChoice {
        label: format!(
            "{index:04}_{}_{}",
            sanitize_generation_label(param),
            value.label_fragment()
        ),
        value: serde_json::to_value(&params).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL generator choice for `{node_id}.{param}`: {error}"
            ))
        })?,
        param_overrides: vec![GenerationParamOverride {
            node_id: node_id.clone(),
            params,
        }],
    })
}

fn multi_param_generation_choice(
    node_id: &NodeId,
    index: usize,
    row: BTreeMap<String, PipelineDslGeneratorValue>,
) -> Result<GenerationChoice> {
    let mut params = BTreeMap::new();
    let mut label_parts = Vec::new();
    for (param, value) in row {
        label_parts.push(format!(
            "{}_{}",
            sanitize_generation_label(&param),
            value.label_fragment()
        ));
        params.insert(param, value.value().clone());
    }
    Ok(GenerationChoice {
        label: format!("{index:04}_{}", label_parts.join("__")),
        value: serde_json::to_value(&params).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL grid generator choice for node `{node_id}`: {error}"
            ))
        })?,
        param_overrides: vec![GenerationParamOverride {
            node_id: node_id.clone(),
            params,
        }],
    })
}

fn build_grid_rows(
    entries: &[(&str, &[PipelineDslGeneratorValue])],
    entry_index: usize,
    current: &mut BTreeMap<String, PipelineDslGeneratorValue>,
    rows: &mut Vec<BTreeMap<String, PipelineDslGeneratorValue>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| rows.len() >= limit) {
        return;
    }
    if entry_index == entries.len() {
        rows.push(current.clone());
        return;
    }
    let (param, values) = entries[entry_index];
    for value in values {
        current.insert(param.to_string(), value.clone());
        build_grid_rows(entries, entry_index + 1, current, rows, count);
        current.remove(param);
        if count.is_some_and(|limit| rows.len() >= limit) {
            break;
        }
    }
}

fn build_combinations(
    value_count: usize,
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    selections: &mut Vec<Vec<usize>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| selections.len() >= limit) {
        return;
    }
    if current.len() == size {
        selections.push(current.clone());
        return;
    }
    let remaining = size - current.len();
    if value_count < remaining {
        return;
    }
    for index in start..=value_count - remaining {
        current.push(index);
        build_combinations(value_count, size, index + 1, current, selections, count);
        current.pop();
        if count.is_some_and(|limit| selections.len() >= limit) {
            break;
        }
    }
}

fn build_permutations(
    value_count: usize,
    size: usize,
    used: &mut BTreeSet<usize>,
    current: &mut Vec<usize>,
    selections: &mut Vec<Vec<usize>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| selections.len() >= limit) {
        return;
    }
    if current.len() == size {
        selections.push(current.clone());
        return;
    }
    for index in 0..value_count {
        if used.contains(&index) {
            continue;
        }
        used.insert(index);
        current.push(index);
        build_permutations(value_count, size, used, current, selections, count);
        current.pop();
        used.remove(&index);
        if count.is_some_and(|limit| selections.len() >= limit) {
            break;
        }
    }
}

fn apply_choice_count(choices: &mut Vec<GenerationChoice>, count: Option<usize>) {
    if let Some(limit) = count {
        choices.truncate(limit);
    }
}

fn validate_count(node_id: &NodeId, name: Option<&str>, count: Option<usize>) -> Result<()> {
    if count == Some(0) {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` for node `{node_id}` has count=0",
            generator_dimension_name(node_id, name, None, "params")
        )));
    }
    Ok(())
}

fn validate_param_name(node_id: &NodeId, param: &str) -> Result<()> {
    if param.trim().is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL param generator for node `{node_id}` has an empty param name"
        )));
    }
    Ok(())
}

fn validate_finite(node_id: &NodeId, param: &str, field: &str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL {field} for `{node_id}.{param}` must be finite"
        )));
    }
    Ok(())
}

fn range_contains(current: f64, stop: f64, step: f64, inclusive: bool) -> bool {
    let epsilon = step.abs() * 1e-12 + f64::EPSILON;
    if step > 0.0 {
        if inclusive {
            current <= stop + epsilon
        } else {
            current < stop - epsilon
        }
    } else if inclusive {
        current >= stop - epsilon
    } else {
        current > stop + epsilon
    }
}

fn json_number(value: f64, node_id: &NodeId, param: &str) -> Result<serde_json::Value> {
    let number = serde_json::Number::from_f64(value).ok_or_else(|| {
        DagMlError::GraphValidation(format!(
            "pipeline DSL numeric generator for `{node_id}.{param}` produced a non-finite value"
        ))
    })?;
    Ok(serde_json::Value::Number(number))
}

fn generator_dimension_name(
    node_id: &NodeId,
    name: Option<&str>,
    param: Option<&str>,
    suffix: &str,
) -> String {
    if let Some(name) = name {
        return name.to_string();
    }
    match param {
        Some(param) => format!("{node_id}.{param}.{suffix}"),
        None => format!("{node_id}.{suffix}"),
    }
}

impl PipelineDslGeneratorValue {
    fn value(&self) -> &serde_json::Value {
        match self {
            Self::Labeled { value, .. } | Self::Value(value) => value,
        }
    }

    fn label_fragment(&self) -> String {
        match self {
            Self::Labeled { label, .. } => sanitize_generation_label(label),
            Self::Value(value) => {
                let rendered = match value {
                    serde_json::Value::String(value) => value.clone(),
                    _ => serde_json::to_string(value).unwrap_or_else(|_| "value".to_string()),
                };
                sanitize_generation_label(&rendered)
            }
        }
    }
}

fn sanitize_generation_label(input: &str) -> String {
    let sanitized = input
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if sanitized.is_empty() {
        "value".to_string()
    } else {
        sanitized
    }
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

fn operator_runtime_metadata(
    step: &PipelineDslOperatorStep,
    branch_id: Option<&str>,
) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut metadata = step.metadata.clone();
    if let Some(branch_id) = branch_id {
        metadata.insert(
            "dsl_branch".to_string(),
            serde_json::Value::String(branch_id.to_string()),
        );
    }
    insert_training_metadata(
        &mut metadata,
        &step.train_params,
        step.tuning.as_ref(),
        &step.id,
    )?;
    Ok(metadata)
}

fn branch_context_metadata(
    branch_step: &PipelineDslBranchStep,
    branch: &PipelineDslBranch,
) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "dsl_branch".to_string(),
        serde_json::Value::String(branch.id.clone()),
    );
    metadata.insert(
        "dsl_branch_mode".to_string(),
        serde_json::to_value(branch_step.mode).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL branch mode for `{}`: {error}",
                branch.id
            ))
        })?,
    );
    if let Some(selector) = &branch_step.selector {
        metadata.insert("dsl_branch_step_selector".to_string(), selector.clone());
    }
    if !branch_step.metadata.is_empty() {
        metadata.insert(
            "dsl_branch_step_metadata".to_string(),
            serde_json::to_value(&branch_step.metadata).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL branch step metadata for `{}`: {error}",
                    branch.id
                ))
            })?,
        );
    }
    if let Some(selector) = &branch.selector {
        metadata.insert("dsl_branch_selector".to_string(), selector.clone());
    }
    if !branch.metadata.is_empty() {
        metadata.insert(
            "dsl_branch_metadata".to_string(),
            serde_json::to_value(&branch.metadata).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL branch metadata for `{}`: {error}",
                    branch.id
                ))
            })?,
        );
    }
    Ok(metadata)
}

fn branch_id_from_metadata(metadata: &BTreeMap<String, serde_json::Value>) -> Option<String> {
    metadata
        .get("dsl_branch")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn validate_merge_selectors(
    merge_id: &NodeId,
    selectors: &[PipelineDslMergeSelector],
    predictions: &[PredictionSource],
) -> Result<()> {
    if selectors.is_empty() {
        return Ok(());
    }
    if predictions.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL merge `{merge_id}` declares selectors but has no prediction inputs"
        )));
    }
    for (selector_index, selector) in selectors.iter().enumerate() {
        let mut matched = predictions.iter().collect::<Vec<_>>();
        if let Some(input_name) = &selector.input_name {
            if input_name.trim().is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL merge `{merge_id}` selector {selector_index} has an empty input_name"
                )));
            }
            matched.retain(|prediction| prediction.input_name == *input_name);
        }
        if let Some(branch) = &selector.branch {
            if branch.trim().is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL merge `{merge_id}` selector {selector_index} has an empty branch"
                )));
            }
            matched.retain(|prediction| prediction.branch_id.as_deref() == Some(branch.as_str()));
        }
        if let Some(model) = &selector.model {
            matched.retain(|prediction| prediction.node_id == *model);
        }
        if matched.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL merge `{merge_id}` selector {selector_index} does not match any pending prediction input"
            )));
        }
        validate_merge_selector_select(merge_id, selector_index, selector, matched.len())?;
    }
    Ok(())
}

fn validate_merge_selector_select(
    merge_id: &NodeId,
    selector_index: usize,
    selector: &PipelineDslMergeSelector,
    matched_count: usize,
) -> Result<()> {
    let Some(select) = &selector.select else {
        return Ok(());
    };
    if let Some(mode) = select.as_str() {
        match mode {
            "all" => return Ok(()),
            "best" => {
                require_selector_metric(merge_id, selector_index, selector, mode)?;
                return Ok(());
            }
            _ => {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL merge `{merge_id}` selector {selector_index} has unsupported select mode `{mode}`"
                )));
            }
        }
    }
    let Some(object) = select.as_object() else {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL merge `{merge_id}` selector {selector_index} select must be `all`, `best` or an object with `top_k`"
        )));
    };
    if object.len() != 1 || !object.contains_key("top_k") {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL merge `{merge_id}` selector {selector_index} object select currently supports only `top_k`"
        )));
    }
    let Some(top_k) = object.get("top_k").and_then(|value| value.as_u64()) else {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL merge `{merge_id}` selector {selector_index} top_k must be a positive integer"
        )));
    };
    if top_k == 0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL merge `{merge_id}` selector {selector_index} top_k must be positive"
        )));
    }
    if top_k as usize > matched_count {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL merge `{merge_id}` selector {selector_index} top_k={top_k} exceeds {matched_count} matched prediction inputs"
        )));
    }
    require_selector_metric(merge_id, selector_index, selector, "top_k")
}

fn require_selector_metric(
    merge_id: &NodeId,
    selector_index: usize,
    selector: &PipelineDslMergeSelector,
    select_mode: &str,
) -> Result<()> {
    if selector
        .metric
        .as_ref()
        .is_some_and(|metric| !metric.trim().is_empty())
    {
        return Ok(());
    }
    Err(DagMlError::GraphValidation(format!(
        "pipeline DSL merge `{merge_id}` selector {selector_index} select `{select_mode}` requires a non-empty metric"
    )))
}

fn insert_training_metadata(
    metadata: &mut BTreeMap<String, serde_json::Value>,
    train_params: &BTreeMap<String, serde_json::Value>,
    tuning: Option<&PipelineDslTuningSpec>,
    node_id: &NodeId,
) -> Result<()> {
    if !train_params.is_empty() {
        metadata.insert(
            "dsl_train_params".to_string(),
            serde_json::to_value(train_params).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL train params for node `{node_id}`: {error}"
                ))
            })?,
        );
    }
    if let Some(tuning) = tuning {
        metadata.insert(
            "dsl_tuning".to_string(),
            serde_json::to_value(tuning).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL tuning for node `{node_id}`: {error}"
                ))
            })?,
        );
    }
    Ok(())
}

fn merge_node_kind(step: &PipelineDslMergeStep, has_predictions: bool) -> NodeKind {
    match step.output_as {
        PipelineDslMergeOutput::Predictions => NodeKind::PredictionJoin,
        PipelineDslMergeOutput::Sources => NodeKind::SourceJoin,
        PipelineDslMergeOutput::Features => {
            if step.include_original_data && has_predictions {
                NodeKind::MixedJoin
            } else if has_predictions {
                NodeKind::PredictionJoin
            } else {
                NodeKind::FeatureJoin
            }
        }
    }
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

fn target_port(name: &str, description: &str) -> PortSpec {
    PortSpec {
        name: name.to_string(),
        kind: PortKind::Target,
        representation: None,
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

fn branch_prediction_input_name(
    branch_id: &str,
    branch_index: usize,
    prediction_index: usize,
    node_id: &NodeId,
) -> String {
    let branch = branch_input_prefix(branch_id, branch_index);
    let model = node_id
        .as_str()
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
    if model.is_empty() {
        format!("{branch}_model{prediction_index}_oof")
    } else {
        format!("{branch}_{model}_oof")
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

fn default_log_base() -> f64 {
    10.0
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
    fn compiles_nirs4all_style_multi_model_branch_and_separate_merge() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-nirs4all-branch-parity",
  "steps": [
    {
      "kind": "branch",
      "mode": "duplication",
      "selector": {"scope": "all_samples"},
      "branches": [
        {
          "id": "pls_path",
          "steps": [
            {
              "kind": "model",
              "id": "branch:pls.model:pls5",
              "operator": {"class": "sklearn.cross_decomposition.PLSRegression"},
              "params": {"n_components": 5}
            },
            {
              "kind": "model",
              "id": "branch:pls.model:pls10",
              "operator": {"class": "sklearn.cross_decomposition.PLSRegression"},
              "params": {"n_components": 10}
            }
          ]
        },
        {
          "id": "rf_path",
          "selector": {"source": "nir"},
          "steps": [
            {
              "kind": "transform",
              "id": "branch:rf.transform:snv",
              "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate"}
            },
            {
              "kind": "model",
              "id": "branch:rf.model:rf",
              "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
              "params": {"n_estimators": 64}
            },
            {
              "kind": "model",
              "id": "branch:rf.model:gbr",
              "operator": {"class": "sklearn.ensemble.GradientBoostingRegressor"},
              "params": {"n_estimators": 32}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:stack.predictions_plus_original",
      "merge_mode": "predictions_plus_original",
      "output_as": "features",
      "include_original_data": true,
      "selectors": [
        {"branch": "pls_path", "select": "best", "metric": "rmse"},
        {"branch": "rf_path", "select": {"top_k": 2}, "metric": "r2"}
      ],
      "metadata": {"on_missing": "warn"}
    },
    {
      "kind": "model",
      "id": "model:meta.ridge",
      "operator": {"class": "sklearn.linear_model.Ridge"},
      "variants": [
        {"label": "low", "params": {"alpha": 0.1}},
        {"label": "mid", "params": {"alpha": 0.5}}
      ]
    },
    {
      "kind": "model",
      "id": "model:meta.rf",
      "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
      "params": {"n_estimators": 30}
    }
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
        let graph = compiled.graph;
        let merge = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "merge:stack.predictions_plus_original")
            .unwrap();

        assert_eq!(merge.kind, NodeKind::MixedJoin);
        assert_eq!(merge.ports.inputs.len(), 5);
        assert_eq!(merge.ports.outputs[0].kind, PortKind::Data);
        assert_eq!(merge.metadata["merge_mode"], "predictions_plus_original");
        assert_eq!(merge.metadata["selectors"][0]["branch"], "pls_path");
        let rf_model = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "branch:rf.model:rf")
            .unwrap();
        assert_eq!(rf_model.metadata["dsl_branch"], "rf_path");
        assert_eq!(rf_model.metadata["dsl_branch_mode"], "duplication");
        assert_eq!(
            rf_model.metadata["dsl_branch_step_selector"]["scope"],
            "all_samples"
        );
        assert_eq!(rf_model.metadata["dsl_branch_selector"]["source"], "nir");
        assert_eq!(
            graph
                .edges
                .iter()
                .filter(|edge| edge.target.node_id == merge.id
                    && edge.contract.kind == PortKind::Prediction
                    && edge.contract.requires_oof)
                .count(),
            4
        );
        assert!(graph
            .edges
            .iter()
            .any(|edge| edge.source.node_id == merge.id
                && edge.target.node_id.as_str() == "model:meta.ridge"
                && edge.contract.kind == PortKind::Data));
        assert!(graph
            .edges
            .iter()
            .any(|edge| edge.source.node_id == merge.id
                && edge.target.node_id.as_str() == "model:meta.rf"
                && edge.contract.kind == PortKind::Data));
        assert_eq!(compiled.generation.dimensions.len(), 1);
        assert_eq!(
            compiled.generation.dimensions[0].name,
            "model:meta.ridge.params"
        );
        graph.validate().unwrap();
    }

    #[test]
    fn merge_selectors_reject_unknown_branch_and_missing_metric() {
        let unknown_branch: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-merge-selector-branch",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "known",
          "steps": [
            {
              "kind": "model",
              "id": "branch:known.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:bad.selector",
      "selectors": [
        {"branch": "missing", "select": "all"}
      ]
    }
  ]
}"#,
        )
        .unwrap();
        let error = compile_pipeline_dsl_with_generation(&unknown_branch).unwrap_err();
        assert!(format!("{error}").contains("does not match any pending prediction input"));

        let missing_metric: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-merge-selector-metric",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "known",
          "steps": [
            {
              "kind": "model",
              "id": "branch:known.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:bad.metric",
      "selectors": [
        {"branch": "known", "select": "best"}
      ]
    }
  ]
}"#,
        )
        .unwrap();
        let error = compile_pipeline_dsl_with_generation(&missing_metric).unwrap_err();
        assert!(format!("{error}").contains("requires a non-empty metric"));
    }

    #[test]
    fn merge_selectors_reject_top_k_above_scope() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-merge-selector-top-k",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "known",
          "steps": [
            {
              "kind": "model",
              "id": "branch:known.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:bad.topk",
      "selectors": [
        {"branch": "known", "select": {"top_k": 2}, "metric": "rmse"}
      ]
    }
  ]
}"#,
        )
        .unwrap();

        let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
        assert!(format!("{error}").contains("top_k=2 exceeds 1 matched prediction inputs"));
    }

    #[test]
    fn compiles_nirs4all_shape_changing_and_tuning_surface() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-nirs4all-shape-parity",
  "steps": [
    {
      "kind": "y_transform",
      "id": "target:scale",
      "operator": {"class": "sklearn.preprocessing.StandardScaler"}
    },
    {
      "kind": "tag",
      "id": "tag:y_outliers",
      "operator": {"class": "nirs4all.filters.YOutlierFilter"},
      "params": {"method": "iqr"}
    },
    {
      "kind": "exclude",
      "id": "exclude:train_outliers",
      "operator": {"class": "nirs4all.filters.YOutlierFilter"},
      "params": {"mode": "any"}
    },
    {
      "kind": "sample_augmentation",
      "id": "augment:sample.noise",
      "operator": {"class": "nirs4all.operators.transforms.GaussianAdditiveNoise"},
      "params": {"count": 3, "selection": "random"},
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
      "kind": "feature_augmentation",
      "id": "augment:feature.views",
      "operator": {"class": "nirs4all.operators.transforms.FeatureAugmentation"},
      "params": {"action": "extend"},
      "shape": {
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "augmented_views",
        "augmentation_policy": {
          "sample_scope": "none",
          "feature_scope": "train_only",
          "require_origin_id": false
        }
      }
    },
    {
      "kind": "concat_transform",
      "id": "join:concat.multi_view",
      "branches": [
        {
          "id": "pca",
          "steps": [
            {
              "id": "concat:pca",
              "operator": {"class": "sklearn.decomposition.PCA"},
              "params": {"n_components": 20}
            }
          ]
        },
        {
          "id": "derivative_pca",
          "steps": [
            {
              "id": "concat:derivative",
              "operator": {"class": "nirs4all.operators.transforms.FirstDerivative"}
            },
            {
              "id": "concat:derivative.pca",
              "operator": {"class": "sklearn.decomposition.PCA"},
              "params": {"n_components": 10}
            }
          ]
        }
      ],
      "shape": {
        "fit_rows": "fold_train",
        "feature_namespace": "concat.multi_view",
        "selection_policy": {
          "scope": "unsupervised"
        }
      }
    },
    {
      "kind": "model",
      "id": "model:tuned",
      "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
      "finetune_params": {
        "n_trials": 10,
        "approach": "single",
        "eval_mode": "mean",
        "sampler": "random",
        "metric": "rmse",
        "model_params": {
          "max_depth": [3, 5, 7]
        }
      },
      "train_params": {
        "sample_weight": "balanced"
      }
    }
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
        let graph = compiled.graph;
        let kinds = graph
            .nodes
            .iter()
            .map(|node| node.kind.clone())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&NodeKind::YTransform));
        assert!(kinds.contains(&NodeKind::Tag));
        assert!(kinds.contains(&NodeKind::Exclude));
        assert!(kinds.contains(&NodeKind::Augmentation));
        assert!(kinds.contains(&NodeKind::FeatureJoin));
        assert_eq!(compiled.shape_plans.len(), 3);

        let sample_aug = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "augment:sample.noise")
            .unwrap();
        assert_eq!(sample_aug.metadata["dsl_augmentation_kind"], "sample");
        let feature_aug = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "augment:feature.views")
            .unwrap();
        assert_eq!(feature_aug.metadata["dsl_augmentation_kind"], "feature");
        let model = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "model:tuned")
            .unwrap();
        assert_eq!(model.metadata["dsl_tuning"]["n_trials"], 10);
        assert_eq!(
            model.metadata["dsl_train_params"]["sample_weight"],
            "balanced"
        );
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
    fn expands_compact_param_generators_into_generation_dimensions() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-compact-generation",
  "steps": [
    {
      "kind": "model",
      "id": "model:tuned",
      "operator": {"type": "TunedModel"},
      "generators": [
        {
          "kind": "or",
          "name": "model_family",
          "param": "family",
          "values": [
            {"label": "ridge", "value": "ridge"},
            {"label": "rf", "value": "random_forest"}
          ]
        },
        {
          "kind": "range",
          "param": "alpha",
          "start": 0.1,
          "stop": 0.9,
          "step": 0.4
        },
        {
          "kind": "log_range",
          "param": "lambda",
          "start": 0.01,
          "stop": 1.0,
          "count": 3
        },
        {
          "kind": "grid",
          "name": "tree_grid",
          "params": {
            "max_depth": [3, 5],
            "n_estimators": [50, 100]
          },
          "count": 3
        },
        {
          "kind": "pick",
          "param": "views",
          "values": ["snv", "msc", "derivative"],
          "sizes": [1, 2],
          "count": 4
        },
        {
          "kind": "arrange",
          "param": "chain",
          "values": ["snv", "pca", "pls"],
          "sizes": [2],
          "count": 3
        }
      ]
    }
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

        assert_eq!(compiled.generation.strategy, GenerationStrategy::Cartesian);
        assert_eq!(compiled.generation.dimensions.len(), 6);
        assert_eq!(compiled.generation.dimensions[0].name, "model_family");
        assert_eq!(compiled.generation.dimensions[0].choices.len(), 2);
        assert_eq!(
            compiled.generation.dimensions[1].name,
            "model:tuned.alpha.range"
        );
        assert_eq!(compiled.generation.dimensions[1].choices.len(), 3);
        assert_eq!(
            compiled.generation.dimensions[1].choices[1].param_overrides[0].params["alpha"],
            0.5
        );
        assert_eq!(
            compiled.generation.dimensions[2].name,
            "model:tuned.lambda.log_range"
        );
        assert_eq!(compiled.generation.dimensions[2].choices.len(), 3);
        assert_eq!(compiled.generation.dimensions[3].name, "tree_grid");
        assert_eq!(compiled.generation.dimensions[3].choices.len(), 3);
        assert_eq!(
            compiled.generation.dimensions[3].choices[2].param_overrides[0].params["n_estimators"],
            50
        );
        assert_eq!(
            compiled.generation.dimensions[4].choices[3].param_overrides[0].params["views"],
            serde_json::json!(["snv", "msc"])
        );
        assert_eq!(
            compiled.generation.dimensions[5].choices[2].param_overrides[0].params["chain"],
            serde_json::json!(["pca", "snv"])
        );
        assert!(compiled.generation_fingerprint.is_some());
    }

    #[test]
    fn compact_param_generators_reject_invalid_counts() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-bad-compact-generation",
  "steps": [
    {
      "kind": "model",
      "id": "model:bad",
      "operator": {"type": "Ridge"},
      "generators": [
        {
          "kind": "or",
          "param": "alpha",
          "values": [0.1, 1.0],
          "count": 0
        }
      ]
    }
  ]
}"#,
        )
        .unwrap();

        let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
        assert!(format!("{error}").contains("count=0"));
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
        assert!(format!("{error}").contains("must produce at least one model or merge prediction"));
    }
}
