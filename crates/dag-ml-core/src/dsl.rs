use std::collections::{BTreeMap, BTreeSet};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

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
    Filter(PipelineDslOperatorStep),
    SampleFilter(PipelineDslOperatorStep),
    Augmentation(PipelineDslOperatorStep),
    FeatureAugmentation(PipelineDslOperatorStep),
    SampleAugmentation(PipelineDslOperatorStep),
    ConcatTransform(PipelineDslConcatTransformStep),
    Model(PipelineDslOperatorStep),
    Branch(PipelineDslBranchStep),
    Generator(PipelineDslGeneratorStep),
    Sequential(PipelineDslSequenceStep),
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
pub struct PipelineDslSequenceStep {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<NodeId>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub steps: Vec<PipelineDslStep>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslGeneratorStep {
    pub id: NodeId,
    #[serde(default)]
    pub mode: PipelineDslGeneratorMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<PipelineDslBranch>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<PipelineDslGeneratorStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pick: Option<PipelineDslSelectionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrange: Option<PipelineDslSelectionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub then_pick: Option<PipelineDslSelectionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub then_arrange: Option<PipelineDslSelectionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineDslGeneratorMode {
    #[default]
    Or,
    Cartesian,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineDslGeneratorStage {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub branches: Vec<PipelineDslBranch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PipelineDslSelectionSpec {
    Single(usize),
    Range([usize; 2]),
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

pub fn parse_pipeline_dsl_json(data: &[u8]) -> Result<PipelineDslSpec> {
    match serde_json::from_slice::<PipelineDslSpec>(data) {
        Ok(spec) if validate_pipeline_dsl(&spec).is_ok() => Ok(spec),
        Ok(spec) => {
            let strict_error = validate_pipeline_dsl(&spec)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_else(|| "unknown validation error".to_string());
            let value = serde_json::from_slice::<serde_json::Value>(data).map_err(|error| {
                DagMlError::GraphValidation(format!("failed to parse pipeline DSL JSON: {error}"))
            })?;
            lower_nirs4all_compat_pipeline_dsl(&value).map_err(|compat_error| {
                DagMlError::GraphValidation(format!(
                    "failed to parse pipeline DSL as valid canonical PipelineDslSpec ({strict_error}) or nirs4all-compatible JSON ({compat_error})"
                ))
            })
        }
        Err(strict_error) => {
            let value = serde_json::from_slice::<serde_json::Value>(data).map_err(|error| {
                DagMlError::GraphValidation(format!("failed to parse pipeline DSL JSON: {error}"))
            })?;
            lower_nirs4all_compat_pipeline_dsl(&value).map_err(|compat_error| {
                DagMlError::GraphValidation(format!(
                    "failed to parse pipeline DSL as canonical PipelineDslSpec ({strict_error}) or nirs4all-compatible JSON ({compat_error})"
                ))
            })
        }
    }
}

pub fn lower_nirs4all_compat_pipeline_dsl(value: &serde_json::Value) -> Result<PipelineDslSpec> {
    CompatDslLowerer::default().lower_root(value)
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

#[derive(Clone, Debug)]
struct GeneratedSequence {
    id: String,
    labels: Vec<String>,
    steps: Vec<PipelineDslStep>,
    metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
struct CompatGenerationAttachment {
    variants: Vec<PipelineDslVariantChoice>,
    param_generators: Vec<PipelineDslParamGenerator>,
}

#[derive(Default)]
struct CompatDslLowerer {
    node_counter: usize,
    generator_counter: usize,
    split_invocation: Option<SplitInvocation>,
    metadata: BTreeMap<String, serde_json::Value>,
}

impl CompatDslLowerer {
    fn lower_root(mut self, value: &serde_json::Value) -> Result<PipelineDslSpec> {
        let root = value.as_object();
        let pipeline = match value {
            serde_json::Value::Array(_) => value,
            serde_json::Value::Object(object) => object
                .get("pipeline")
                .or_else(|| object.get("steps"))
                .ok_or_else(|| {
                    DagMlError::GraphValidation(
                        "nirs4all-compatible pipeline DSL must be a JSON array or an object with `pipeline`/`steps`".to_string(),
                    )
                })?,
            _ => {
                return Err(DagMlError::GraphValidation(
                    "nirs4all-compatible pipeline DSL must be a JSON array or object".to_string(),
                ));
            }
        };
        let pipeline = pipeline.as_array().ok_or_else(|| {
            DagMlError::GraphValidation(
                "nirs4all-compatible pipeline field must be an array".to_string(),
            )
        })?;
        let steps = self.lower_steps(pipeline, "pipeline")?;
        let id = root
            .and_then(|object| object.get("id"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("dsl-nirs4all-compat")
            .to_string();
        let mut metadata: BTreeMap<String, serde_json::Value> =
            optional_root_field(root, "metadata")?.unwrap_or_default();
        metadata.extend(std::mem::take(&mut self.metadata));
        metadata.insert(
            "dsl_compat_profile".to_string(),
            serde_json::Value::String("nirs4all_json_v1".to_string()),
        );
        let root_split = optional_root_field(root, "split_invocation")?;
        let split_invocation = match (root_split, self.split_invocation) {
            (Some(_), Some(_)) => {
                return Err(DagMlError::GraphValidation(
                    "nirs4all-compatible pipeline declares split_invocation and a pipeline split step".to_string(),
                ));
            }
            (Some(split), None) | (None, Some(split)) => Some(split),
            (None, None) => None,
        };
        Ok(PipelineDslSpec {
            id,
            input: optional_root_field(root, "input")?.unwrap_or_default(),
            output: optional_root_field(root, "output")?.unwrap_or_default(),
            generation_strategy: optional_root_field(root, "generation_strategy")?,
            max_variants: optional_root_field(root, "max_variants")?,
            generation_dimensions: optional_root_field(root, "generation_dimensions")?
                .unwrap_or_default(),
            campaign_id: optional_root_field(root, "campaign_id")?,
            root_seed: optional_root_field(root, "root_seed")?,
            leakage_policy: optional_root_field(root, "leakage_policy")?,
            aggregation_policy: optional_root_field(root, "aggregation_policy")?,
            split_invocation,
            campaign_metadata: optional_root_field(root, "campaign_metadata")?.unwrap_or_default(),
            data_bindings: optional_root_field(root, "data_bindings")?.unwrap_or_default(),
            steps,
            metadata,
        })
    }

    fn lower_steps(
        &mut self,
        values: &[serde_json::Value],
        path: &str,
    ) -> Result<Vec<PipelineDslStep>> {
        let mut lowered = Vec::new();
        let mut index = 0usize;
        while index < values.len() {
            let current_path = format!("{path}[{index}]");
            if self.consume_side_effect_step(&values[index], &current_path)? {
                index += 1;
                continue;
            }
            if let Some(attachment) =
                self.parse_attached_generation(&values[index], &current_path)?
            {
                let next = values.get(index + 1).ok_or_else(|| {
                    DagMlError::GraphValidation(format!(
                        "{current_path} declares a parameter generator but has no following operator/model step"
                    ))
                })?;
                let mut attached = self.lower_value_with_attachment(
                    next,
                    &format!("{path}[{}]", index + 1),
                    attachment,
                )?;
                lowered.append(&mut attached);
                index += 2;
                continue;
            }
            if let Some(merge_model) =
                self.lower_merge_followed_by_model(values, index, &current_path)?
            {
                lowered.push(PipelineDslStep::MergeModel(merge_model));
                index += 2;
                continue;
            }

            let steps = self.lower_value_as_steps(&values[index], &current_path)?;
            if let [PipelineDslStep::Generator(generator)] = steps.as_slice() {
                if !generator_step_has_prediction(generator) {
                    if let Some((combined, consumed)) = self.combine_data_generator_with_following(
                        generator.clone(),
                        &values[index + 1..],
                        path,
                        index + 1,
                    )? {
                        lowered.push(PipelineDslStep::Generator(combined));
                        index += consumed + 1;
                        continue;
                    }
                }
            }
            lowered.extend(steps);
            index += 1;
        }
        Ok(lowered)
    }

    fn consume_side_effect_step(&mut self, value: &serde_json::Value, path: &str) -> Result<bool> {
        let Some(object) = value.as_object() else {
            return Ok(false);
        };
        if let Some(split) = object.get("split") {
            self.set_split_invocation(self.lower_split_invocation(split, object, path)?, path)?;
            return Ok(true);
        }
        if let Some(sources) = object.get("sources") {
            self.metadata
                .insert("compat_sources".to_string(), sources.clone());
            return Ok(true);
        }
        Ok(false)
    }

    fn lower_value_as_steps(
        &mut self,
        value: &serde_json::Value,
        path: &str,
    ) -> Result<Vec<PipelineDslStep>> {
        match value {
            serde_json::Value::Null => Ok(Vec::new()),
            serde_json::Value::Array(children) => {
                Ok(vec![PipelineDslStep::Sequential(PipelineDslSequenceStep {
                    id: None,
                    metadata: BTreeMap::new(),
                    steps: self.lower_steps(children, path)?,
                })])
            }
            serde_json::Value::String(_) => Ok(vec![PipelineDslStep::Transform(
                self.compat_operator_step(None, "preprocessing", value, None, None)?,
            )]),
            serde_json::Value::Object(object) => {
                if object.contains_key("kind") {
                    let step = serde_json::from_value::<PipelineDslStep>(value.clone()).map_err(
                        |error| {
                            DagMlError::GraphValidation(format!(
                                "failed to parse canonical DSL step at {path}: {error}"
                            ))
                        },
                    )?;
                    return Ok(vec![step]);
                }
                if self.consume_side_effect_step(value, path)? {
                    return Ok(Vec::new());
                }
                if let Some(operator) =
                    first_object_value(object, &["preprocessing", "processing", "transform"])
                {
                    return Ok(vec![PipelineDslStep::Transform(
                        self.compat_operator_step(
                            Some(object),
                            "preprocessing",
                            operator,
                            None,
                            None,
                        )?,
                    )]);
                }
                if let Some(operator) = first_object_value(object, &["y_processing", "y_transform"])
                {
                    return Ok(vec![PipelineDslStep::YTransform(
                        self.compat_operator_step(
                            Some(object),
                            "y_processing",
                            operator,
                            None,
                            None,
                        )?,
                    )]);
                }
                if let Some(operator) = object.get("tag") {
                    return Ok(vec![PipelineDslStep::Tag(self.compat_operator_step(
                        Some(object),
                        "tag",
                        operator,
                        None,
                        None,
                    )?)]);
                }
                if let Some(operator) = object.get("exclude") {
                    return Ok(vec![PipelineDslStep::Exclude(self.compat_operator_step(
                        Some(object),
                        "exclude",
                        operator,
                        None,
                        None,
                    )?)]);
                }
                if let Some(operator) = object.get("filter") {
                    return Ok(vec![PipelineDslStep::Filter(self.compat_operator_step(
                        Some(object),
                        "filter",
                        operator,
                        None,
                        None,
                    )?)]);
                }
                if let Some(operator) = object.get("sample_filter") {
                    return Ok(vec![PipelineDslStep::SampleFilter(
                        self.compat_operator_step(
                            Some(object),
                            "sample_filter",
                            operator,
                            None,
                            None,
                        )?,
                    )]);
                }
                if let Some(operator) = object.get("sample_augmentation") {
                    return Ok(vec![PipelineDslStep::SampleAugmentation(
                        self.compat_operator_step(
                            Some(object),
                            "sample_augmentation",
                            operator,
                            None,
                            Some(compat_augmentation_shape("sample", object)?),
                        )?,
                    )]);
                }
                if let Some(operator) = object.get("feature_augmentation") {
                    return Ok(vec![PipelineDslStep::FeatureAugmentation(
                        self.compat_operator_step(
                            Some(object),
                            "feature_augmentation",
                            operator,
                            None,
                            Some(compat_augmentation_shape("feature", object)?),
                        )?,
                    )]);
                }
                if let Some(operator) = object.get("augmentation") {
                    return Ok(vec![PipelineDslStep::Augmentation(
                        self.compat_operator_step(
                            Some(object),
                            "augmentation",
                            operator,
                            None,
                            Some(compat_augmentation_shape("both", object)?),
                        )?,
                    )]);
                }
                if let Some(operator) = object.get("model") {
                    return Ok(vec![PipelineDslStep::Model(self.compat_operator_step(
                        Some(object),
                        "model",
                        operator,
                        None,
                        None,
                    )?)]);
                }
                if let Some(operator) = object.get("chart") {
                    return Ok(vec![PipelineDslStep::Chart(self.compat_operator_step(
                        Some(object),
                        "chart",
                        operator,
                        None,
                        None,
                    )?)]);
                }
                if object.contains_key("branch") {
                    return Ok(vec![PipelineDslStep::Branch(
                        self.lower_branch_step(object, path)?,
                    )]);
                }
                if object.contains_key("concat_transform") {
                    return Ok(vec![PipelineDslStep::ConcatTransform(
                        self.lower_concat_transform_step(object, path)?,
                    )]);
                }
                if object.contains_key("merge") {
                    return Ok(vec![PipelineDslStep::Merge(
                        self.lower_merge_step(object, path)?,
                    )]);
                }
                if object.contains_key("_or_") {
                    return Ok(vec![PipelineDslStep::Generator(
                        self.lower_or_generator(object, "_or_", path)?,
                    )]);
                }
                if object.contains_key("_chain_") {
                    return Ok(vec![PipelineDslStep::Generator(
                        self.lower_or_generator(object, "_chain_", path)?,
                    )]);
                }
                if object.contains_key("_cartesian_") {
                    return Ok(vec![PipelineDslStep::Generator(
                        self.lower_cartesian_generator(object, path)?,
                    )]);
                }
                if object.contains_key("_grid_") {
                    return Ok(vec![PipelineDslStep::Generator(
                        self.lower_grid_generator(object, path)?,
                    )]);
                }
                if object.contains_key("_sample_") {
                    return Ok(vec![PipelineDslStep::Generator(
                        self.lower_sample_generator(object, path)?,
                    )]);
                }
                if object.contains_key("type") || object.contains_key("ref") {
                    return Ok(vec![PipelineDslStep::Transform(
                        self.compat_operator_step(None, "preprocessing", value, None, None)?,
                    )]);
                }
                Err(DagMlError::GraphValidation(format!(
                    "unsupported nirs4all-compatible DSL object at {path}"
                )))
            }
            _ => Err(DagMlError::GraphValidation(format!(
                "unsupported nirs4all-compatible DSL value at {path}"
            ))),
        }
    }

    fn lower_value_with_attachment(
        &mut self,
        value: &serde_json::Value,
        path: &str,
        attachment: CompatGenerationAttachment,
    ) -> Result<Vec<PipelineDslStep>> {
        match value {
            serde_json::Value::String(_) => Ok(vec![PipelineDslStep::Transform(
                self.compat_operator_step(None, "preprocessing", value, Some(attachment), None)?,
            )]),
            serde_json::Value::Object(object) => {
                if let Some(operator) = object.get("model") {
                    return Ok(vec![PipelineDslStep::Model(self.compat_operator_step(
                        Some(object),
                        "model",
                        operator,
                        Some(attachment),
                        None,
                    )?)]);
                }
                if let Some(operator) =
                    first_object_value(object, &["preprocessing", "processing", "transform"])
                {
                    return Ok(vec![PipelineDslStep::Transform(self.compat_operator_step(
                        Some(object),
                        "preprocessing",
                        operator,
                        Some(attachment),
                        None,
                    )?)]);
                }
                Err(DagMlError::GraphValidation(format!(
                    "{path} cannot receive a preceding nirs4all parameter generator; expected model or preprocessing"
                )))
            }
            _ => Err(DagMlError::GraphValidation(format!(
                "{path} cannot receive a preceding nirs4all parameter generator; expected model or preprocessing"
            ))),
        }
    }

    fn lower_merge_followed_by_model(
        &mut self,
        values: &[serde_json::Value],
        index: usize,
        _path: &str,
    ) -> Result<Option<PipelineDslMergeModelStep>> {
        let Some(merge_object) = values[index].as_object() else {
            return Ok(None);
        };
        if !merge_object.contains_key("merge") {
            return Ok(None);
        }
        let Some(next) = values.get(index + 1).and_then(serde_json::Value::as_object) else {
            return Ok(None);
        };
        let Some(operator) = next.get("model") else {
            return Ok(None);
        };
        let (merge_mode, include_original_data, _) = compat_merge_modes(merge_object)?;
        let operator_step = self.compat_operator_step(Some(next), "model", operator, None, None)?;
        Ok(Some(PipelineDslMergeModelStep {
            id: operator_step.id,
            operator: operator_step.operator,
            params: operator_step.params,
            metadata: operator_step.metadata,
            seed_label: operator_step.seed_label,
            include_original_data,
            merge_mode,
            train_params: operator_step.train_params,
            tuning: operator_step.tuning,
            variants: operator_step.variants,
            param_generators: operator_step.param_generators,
            shape: operator_step.shape,
        }))
    }

    fn combine_data_generator_with_following(
        &mut self,
        generator: PipelineDslGeneratorStep,
        remaining: &[serde_json::Value],
        path: &str,
        absolute_start: usize,
    ) -> Result<Option<(PipelineDslGeneratorStep, usize)>> {
        let fused_id = generator.id.clone();
        let mut stages = generator_to_cartesian_stages(generator)?;
        let mut prefix_steps = Vec::new();
        let mut consumed = 0usize;
        while consumed < remaining.len() {
            let current_path = format!("{path}[{}]", absolute_start + consumed);
            if self.consume_side_effect_step(&remaining[consumed], &current_path)? {
                consumed += 1;
                continue;
            }
            let steps = if let Some(attachment) =
                self.parse_attached_generation(&remaining[consumed], &current_path)?
            {
                let next = remaining.get(consumed + 1).ok_or_else(|| {
                    DagMlError::GraphValidation(format!(
                        "{current_path} declares a parameter generator but has no following operator/model step"
                    ))
                })?;
                consumed += 1;
                self.lower_value_with_attachment(
                    next,
                    &format!("{path}[{}]", absolute_start + consumed),
                    attachment,
                )?
            } else if let Some(merge_model) =
                self.lower_merge_followed_by_model(remaining, consumed, &current_path)?
            {
                consumed += 1;
                vec![PipelineDslStep::MergeModel(merge_model)]
            } else {
                self.lower_value_as_steps(&remaining[consumed], &current_path)?
            };
            consumed += 1;
            if steps.is_empty() {
                continue;
            }
            if let [PipelineDslStep::Generator(next_generator)] = steps.as_slice() {
                if !prefix_steps.is_empty() {
                    stages.push(single_stage(
                        format!("stage{}", stages.len()),
                        "prefix",
                        std::mem::take(&mut prefix_steps),
                    ));
                }
                let next_has_prediction = generator_step_has_prediction(next_generator);
                stages.extend(generator_to_cartesian_stages(next_generator.clone())?);
                if next_has_prediction {
                    return Ok(Some((
                        combined_cartesian_generator(fused_id.clone(), stages),
                        consumed,
                    )));
                }
                continue;
            }
            let has_prediction = steps.iter().any(step_has_prediction);
            prefix_steps.extend(steps);
            if has_prediction {
                stages.push(single_stage(
                    format!("stage{}", stages.len()),
                    "then",
                    std::mem::take(&mut prefix_steps),
                ));
                return Ok(Some((
                    combined_cartesian_generator(fused_id.clone(), stages),
                    consumed,
                )));
            }
        }
        Ok(None)
    }

    fn lower_branch_step(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<PipelineDslBranchStep> {
        let branch_value = object.get("branch").expect("checked by caller");
        let mode = optional_object_field(object, "mode")?.unwrap_or_default();
        let selector = object.get("selector").cloned();
        let metadata = optional_object_field(object, "metadata")?.unwrap_or_default();
        let branches = match branch_value {
            serde_json::Value::Array(values) => values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let id = compat_branch_id(value, index);
                    Ok(PipelineDslBranch {
                        id,
                        selector: None,
                        metadata: BTreeMap::new(),
                        steps: self
                            .lower_pipeline_fragment(value, &format!("{path}.branch[{index}]"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            serde_json::Value::Object(branch_object) => {
                if let Some(values) = branch_object
                    .get("branches")
                    .and_then(serde_json::Value::as_array)
                {
                    values
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            self.lower_named_branch(
                                value,
                                index,
                                &format!("{path}.branch.branches[{index}]"),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    branch_object
                        .iter()
                        .filter(|(key, _)| {
                            !matches!(key.as_str(), "mode" | "selector" | "metadata")
                        })
                        .enumerate()
                        .map(|(index, (key, value))| {
                            Ok(PipelineDslBranch {
                                id: sanitize_branch_id(key, index),
                                selector: None,
                                metadata: BTreeMap::new(),
                                steps: self.lower_pipeline_fragment(
                                    value,
                                    &format!("{path}.branch.{key}"),
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?
                }
            }
            _ => {
                return Err(DagMlError::GraphValidation(format!(
                    "{path}.branch must be an array or object"
                )));
            }
        };
        Ok(PipelineDslBranchStep {
            mode,
            selector,
            metadata,
            branches,
        })
    }

    fn lower_named_branch(
        &mut self,
        value: &serde_json::Value,
        index: usize,
        path: &str,
    ) -> Result<PipelineDslBranch> {
        if let Some(object) = value.as_object() {
            if object.contains_key("steps") || object.contains_key("pipeline") {
                let id = object
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(|id| sanitize_branch_id(id, index))
                    .unwrap_or_else(|| format!("branch{index}"));
                let selector = object.get("selector").cloned();
                let metadata = optional_object_field(object, "metadata")?.unwrap_or_default();
                let steps_value = object
                    .get("steps")
                    .or_else(|| object.get("pipeline"))
                    .ok_or_else(|| {
                        DagMlError::GraphValidation(format!(
                            "{path} branch object must contain steps or pipeline"
                        ))
                    })?;
                return Ok(PipelineDslBranch {
                    id,
                    selector,
                    metadata,
                    steps: self.lower_pipeline_fragment(steps_value, path)?,
                });
            }
        }
        Ok(PipelineDslBranch {
            id: compat_branch_id(value, index),
            selector: None,
            metadata: BTreeMap::new(),
            steps: self.lower_pipeline_fragment(value, path)?,
        })
    }

    fn lower_concat_transform_step(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<PipelineDslConcatTransformStep> {
        let value = object.get("concat_transform").expect("checked by caller");
        let branches = match value {
            serde_json::Value::Array(values) => values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    Ok(PipelineDslConcatBranch {
                        id: compat_branch_id(value, index),
                        steps: self.lower_concat_operator_steps(
                            value,
                            &format!("{path}.concat_transform[{index}]"),
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            serde_json::Value::Object(map) => map
                .iter()
                .enumerate()
                .map(|(index, (key, value))| {
                    Ok(PipelineDslConcatBranch {
                        id: sanitize_branch_id(key, index),
                        steps: self.lower_concat_operator_steps(
                            value,
                            &format!("{path}.concat_transform.{key}"),
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => {
                return Err(DagMlError::GraphValidation(format!(
                    "{path}.concat_transform must be an array or object"
                )));
            }
        };
        Ok(PipelineDslConcatTransformStep {
            id: explicit_or_generated_node_id(object, "id", || self.next_node_id("join"))?,
            branches,
            metadata: optional_object_field(object, "metadata")?.unwrap_or_default(),
            seed_label: optional_object_field(object, "seed_label")?,
            representation: optional_object_field(object, "representation")?,
            variants: Vec::new(),
            param_generators: Vec::new(),
            shape: optional_object_field(object, "shape")?,
        })
    }

    fn lower_concat_operator_steps(
        &mut self,
        value: &serde_json::Value,
        path: &str,
    ) -> Result<Vec<PipelineDslOperatorStep>> {
        let steps = self.lower_pipeline_fragment(value, path)?;
        steps
            .into_iter()
            .map(|step| match step {
                PipelineDslStep::Transform(step) => Ok(step),
                _ => Err(DagMlError::GraphValidation(format!(
                    "{path} concat_transform branches currently accept only preprocessing/transform steps"
                ))),
            })
            .collect()
    }

    fn lower_merge_step(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        _path: &str,
    ) -> Result<PipelineDslMergeStep> {
        let (merge_mode, include_original_data, output_as) = compat_merge_modes(object)?;
        Ok(PipelineDslMergeStep {
            id: explicit_or_generated_node_id(object, "id", || self.next_node_id("merge"))?,
            merge_mode,
            output_as,
            include_original_data,
            on_missing: optional_object_field(object, "on_missing")?,
            selectors: optional_object_field(object, "selectors")?.unwrap_or_default(),
            metadata: optional_object_field(object, "metadata")?.unwrap_or_default(),
            seed_label: optional_object_field(object, "seed_label")?,
            representation: optional_object_field(object, "representation")?,
            variants: Vec::new(),
            param_generators: Vec::new(),
            shape: optional_object_field(object, "shape")?,
        })
    }

    fn lower_or_generator(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        key: &str,
        path: &str,
    ) -> Result<PipelineDslGeneratorStep> {
        let values = object
            .get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| DagMlError::GraphValidation(format!("{path}.{key} must be an array")))?;
        let branches = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                Ok(PipelineDslBranch {
                    id: compat_branch_id(value, index),
                    selector: None,
                    metadata: BTreeMap::new(),
                    steps: self
                        .lower_pipeline_fragment(value, &format!("{path}.{key}[{index}]"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PipelineDslGeneratorStep {
            id: explicit_or_generated_node_id(object, "id", || self.next_generator_id())?,
            mode: PipelineDslGeneratorMode::Or,
            branches,
            stages: Vec::new(),
            pick: optional_object_field(object, "pick")?,
            arrange: optional_object_field(object, "arrange")?,
            then_pick: optional_object_field(object, "then_pick")?,
            then_arrange: optional_object_field(object, "then_arrange")?,
            count: optional_object_field(object, "count")?,
            metadata: compat_generator_metadata(object, key)?,
        })
    }

    fn lower_cartesian_generator(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<PipelineDslGeneratorStep> {
        let values = object
            .get("_cartesian_")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                DagMlError::GraphValidation(format!("{path}._cartesian_ must be an array"))
            })?;
        let stages = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                self.lower_cartesian_stage(value, index, &format!("{path}._cartesian_[{index}]"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PipelineDslGeneratorStep {
            id: explicit_or_generated_node_id(object, "id", || self.next_generator_id())?,
            mode: PipelineDslGeneratorMode::Cartesian,
            branches: Vec::new(),
            stages,
            pick: None,
            arrange: None,
            then_pick: None,
            then_arrange: None,
            count: optional_object_field(object, "count")?,
            metadata: compat_generator_metadata(object, "_cartesian_")?,
        })
    }

    fn lower_cartesian_stage(
        &mut self,
        value: &serde_json::Value,
        index: usize,
        path: &str,
    ) -> Result<PipelineDslGeneratorStage> {
        if let Some(object) = value.as_object() {
            if object.contains_key("_or_") {
                let generator = self.lower_or_generator(object, "_or_", path)?;
                return Ok(PipelineDslGeneratorStage {
                    id: format!("stage{index}"),
                    selector: None,
                    metadata: BTreeMap::new(),
                    branches: generator.branches,
                });
            }
            if object.contains_key("_chain_") {
                let generator = self.lower_or_generator(object, "_chain_", path)?;
                return Ok(PipelineDslGeneratorStage {
                    id: format!("stage{index}"),
                    selector: None,
                    metadata: BTreeMap::new(),
                    branches: generator.branches,
                });
            }
            if object.contains_key("_grid_") {
                return Ok(PipelineDslGeneratorStage {
                    id: format!("stage{index}"),
                    selector: None,
                    metadata: BTreeMap::new(),
                    branches: self.lower_grid_branches(object.get("_grid_").unwrap(), path)?,
                });
            }
            if object.contains_key("_sample_") {
                let generator = self.lower_sample_generator(object, path)?;
                return Ok(PipelineDslGeneratorStage {
                    id: format!("stage{index}"),
                    selector: None,
                    metadata: BTreeMap::new(),
                    branches: generator.branches,
                });
            }
        }
        Ok(PipelineDslGeneratorStage {
            id: format!("stage{index}"),
            selector: None,
            metadata: BTreeMap::new(),
            branches: vec![PipelineDslBranch {
                id: "option0".to_string(),
                selector: None,
                metadata: BTreeMap::new(),
                steps: self.lower_pipeline_fragment(value, path)?,
            }],
        })
    }

    fn lower_grid_generator(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<PipelineDslGeneratorStep> {
        Ok(PipelineDslGeneratorStep {
            id: explicit_or_generated_node_id(object, "id", || self.next_generator_id())?,
            mode: PipelineDslGeneratorMode::Or,
            branches: self.lower_grid_branches(object.get("_grid_").unwrap(), path)?,
            stages: Vec::new(),
            pick: None,
            arrange: None,
            then_pick: None,
            then_arrange: None,
            count: optional_object_field(object, "count")?,
            metadata: compat_generator_metadata(object, "_grid_")?,
        })
    }

    fn lower_sample_generator(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<PipelineDslGeneratorStep> {
        Ok(PipelineDslGeneratorStep {
            id: explicit_or_generated_node_id(object, "id", || self.next_generator_id())?,
            mode: PipelineDslGeneratorMode::Or,
            branches: self.lower_sample_branches(object.get("_sample_").unwrap(), path)?,
            stages: Vec::new(),
            pick: None,
            arrange: None,
            then_pick: None,
            then_arrange: None,
            count: optional_object_field(object, "count")?,
            metadata: compat_generator_metadata(object, "_sample_")?,
        })
    }

    fn lower_sample_branches(
        &mut self,
        value: &serde_json::Value,
        path: &str,
    ) -> Result<Vec<PipelineDslBranch>> {
        let sample = value.as_object().ok_or_else(|| {
            DagMlError::GraphValidation(format!("{path}._sample_ must be an object"))
        })?;
        let rows = compat_sample_rows(sample, path)?;
        let operator = sample
            .get("model")
            .or_else(|| sample.get("preprocessing"))
            .or_else(|| sample.get("transform"))
            .ok_or_else(|| {
                DagMlError::GraphValidation(format!(
                    "{path}._sample_ structural lowering requires `model`, `preprocessing` or `transform`"
                ))
            })?
            .clone();
        let keyword = if sample.contains_key("model") {
            "model"
        } else {
            "preprocessing"
        };
        let fixed_params = sample
            .iter()
            .filter(|(key, _)| {
                !matches!(
                    key.as_str(),
                    "model"
                        | "preprocessing"
                        | "transform"
                        | "distribution"
                        | "from"
                        | "to"
                        | "num"
                        | "count"
                        | "param"
                        | "tune"
                )
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        rows.into_iter()
            .enumerate()
            .map(|(index, mut row)| {
                row.extend(fixed_params.clone());
                let step = self.compat_operator_step_from_parts(
                    keyword,
                    operator.clone(),
                    row,
                    None,
                    None,
                )?;
                Ok(PipelineDslBranch {
                    id: format!("sample{index}"),
                    selector: None,
                    metadata: BTreeMap::new(),
                    steps: vec![if keyword == "model" {
                        PipelineDslStep::Model(step)
                    } else {
                        PipelineDslStep::Transform(step)
                    }],
                })
            })
            .collect()
    }

    fn lower_grid_branches(
        &mut self,
        value: &serde_json::Value,
        path: &str,
    ) -> Result<Vec<PipelineDslBranch>> {
        let rows = compat_grid_rows(value, path)?;
        rows.into_iter()
            .enumerate()
            .map(|(index, row)| {
                let metadata = BTreeMap::from([(
                    "compat_grid_row".to_string(),
                    serde_json::to_value(&row).map_err(|error| {
                        DagMlError::GraphValidation(format!(
                            "failed to serialize grid row at {path}: {error}"
                        ))
                    })?,
                )]);
                Ok(PipelineDslBranch {
                    id: format!("grid{index}"),
                    selector: None,
                    metadata,
                    steps: self.lower_grid_row(row, path)?,
                })
            })
            .collect()
    }

    fn lower_grid_row(
        &mut self,
        mut row: BTreeMap<String, serde_json::Value>,
        path: &str,
    ) -> Result<Vec<PipelineDslStep>> {
        if let Some(operator) = row.remove("model") {
            return Ok(vec![PipelineDslStep::Model(
                self.compat_operator_step_from_parts("model", operator, row, None, None)?,
            )]);
        }
        if let Some(operator) = row
            .remove("preprocessing")
            .or_else(|| row.remove("processing"))
            .or_else(|| row.remove("transform"))
        {
            return Ok(vec![PipelineDslStep::Transform(
                self.compat_operator_step_from_parts("preprocessing", operator, row, None, None)?,
            )]);
        }
        Err(DagMlError::GraphValidation(format!(
            "{path}._grid_ rows must contain `model`, `preprocessing` or `transform` for structural lowering"
        )))
    }

    fn lower_pipeline_fragment(
        &mut self,
        value: &serde_json::Value,
        path: &str,
    ) -> Result<Vec<PipelineDslStep>> {
        match value {
            serde_json::Value::Null => Ok(Vec::new()),
            serde_json::Value::Array(values) => self.lower_steps(values, path),
            _ => self.lower_value_as_steps(value, path),
        }
    }

    fn parse_attached_generation(
        &mut self,
        value: &serde_json::Value,
        path: &str,
    ) -> Result<Option<CompatGenerationAttachment>> {
        let Some(object) = value.as_object() else {
            return Ok(None);
        };
        if let Some(range) = object.get("_range_") {
            return Ok(Some(CompatGenerationAttachment {
                variants: Vec::new(),
                param_generators: vec![compat_range_generator(range, object, path)?],
            }));
        }
        if let Some(range) = object.get("_log_range_") {
            return Ok(Some(CompatGenerationAttachment {
                variants: Vec::new(),
                param_generators: vec![compat_log_range_generator(range, object, path)?],
            }));
        }
        if let Some(grid) = object.get("_grid_") {
            if grid.as_object().is_some_and(|grid| {
                !grid.contains_key("model")
                    && !grid.contains_key("preprocessing")
                    && !grid.contains_key("transform")
            }) {
                return Ok(Some(CompatGenerationAttachment {
                    variants: Vec::new(),
                    param_generators: vec![compat_grid_param_generator(grid, object, path)?],
                }));
            }
        }
        if let Some(zip) = object.get("_zip_") {
            return Ok(Some(CompatGenerationAttachment {
                variants: compat_zip_variants(zip, path)?,
                param_generators: Vec::new(),
            }));
        }
        if let Some(sample) = object.get("_sample_") {
            if sample.as_object().is_some_and(|sample| {
                sample.contains_key("model")
                    || sample.contains_key("preprocessing")
                    || sample.contains_key("transform")
            }) {
                return Ok(None);
            }
            return Ok(Some(CompatGenerationAttachment {
                variants: compat_sample_variants(sample, path)?,
                param_generators: Vec::new(),
            }));
        }
        Ok(None)
    }

    fn compat_operator_step(
        &mut self,
        object: Option<&serde_json::Map<String, serde_json::Value>>,
        keyword: &str,
        operator: &serde_json::Value,
        attachment: Option<CompatGenerationAttachment>,
        fallback_shape: Option<PipelineDslShapePlan>,
    ) -> Result<PipelineDslOperatorStep> {
        let id_prefix = compat_node_prefix(keyword);
        let mut params = object
            .and_then(|object| object_value_as_map(object.get("params")))
            .unwrap_or_default();
        if let Some(object) = object {
            for alias in compat_param_aliases(keyword) {
                if let Some(alias_params) = object_value_as_map(object.get(*alias)) {
                    params.extend(alias_params);
                }
            }
        }
        let shape = match object.and_then(|object| object.get("shape")) {
            Some(shape) => Some(deserialize_value(
                shape.clone(),
                "pipeline DSL compat shape",
            )?),
            None => fallback_shape,
        };
        let mut step = PipelineDslOperatorStep {
            id: match object {
                Some(object) => {
                    explicit_or_generated_node_id(object, "id", || self.next_node_id(id_prefix))?
                }
                None => self.next_node_id(id_prefix)?,
            },
            operator: operator.clone(),
            params,
            metadata: optional_object_field_from_option(object, "metadata")?.unwrap_or_default(),
            seed_label: optional_object_field_from_option(object, "seed_label")?,
            representation: optional_object_field_from_option(object, "representation")?,
            train_params: optional_object_field_from_option(object, "train_params")?
                .unwrap_or_default(),
            tuning: optional_object_field_from_option(object, "tuning")?.or(
                optional_object_field_from_option(object, "finetune_params")?,
            ),
            variants: optional_object_field_from_option(object, "variants")?.unwrap_or_default(),
            param_generators: optional_object_field_from_option(object, "generators")?
                .unwrap_or_default(),
            shape,
        };
        step.metadata.insert(
            "dsl_compat_keyword".to_string(),
            serde_json::Value::String(keyword.to_string()),
        );
        if let Some(policy) = object.and_then(|object| object.get("policy")) {
            step.metadata
                .insert("dsl_compat_policy".to_string(), policy.clone());
        }
        if let Some(attachment) = attachment {
            step.variants.extend(attachment.variants);
            step.param_generators.extend(attachment.param_generators);
        }
        Ok(step)
    }

    fn compat_operator_step_from_parts(
        &mut self,
        keyword: &str,
        operator: serde_json::Value,
        params: BTreeMap<String, serde_json::Value>,
        attachment: Option<CompatGenerationAttachment>,
        shape: Option<PipelineDslShapePlan>,
    ) -> Result<PipelineDslOperatorStep> {
        let mut step = PipelineDslOperatorStep {
            id: self.next_node_id(compat_node_prefix(keyword))?,
            operator,
            params,
            metadata: BTreeMap::from([(
                "dsl_compat_keyword".to_string(),
                serde_json::Value::String(keyword.to_string()),
            )]),
            seed_label: None,
            representation: None,
            train_params: BTreeMap::new(),
            tuning: None,
            variants: Vec::new(),
            param_generators: Vec::new(),
            shape,
        };
        if let Some(attachment) = attachment {
            step.variants.extend(attachment.variants);
            step.param_generators.extend(attachment.param_generators);
        }
        Ok(step)
    }

    fn lower_split_invocation(
        &self,
        split: &serde_json::Value,
        object: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<SplitInvocation> {
        let mut params = BTreeMap::new();
        let mut id = object
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("split:compat")
            .to_string();
        let mut controller_id = optional_object_field(object, "controller_id")?;
        let mut leakage_policy =
            optional_object_field(object, "leakage_policy")?.unwrap_or_default();
        let fold_set = optional_object_field(object, "fold_set")?;
        match split {
            serde_json::Value::String(kind) => {
                params.insert("kind".to_string(), serde_json::Value::String(kind.clone()));
                id = format!("split:{}", sanitize_generation_label(kind));
            }
            serde_json::Value::Object(split_object) => {
                if let Some(split_id) = split_object.get("id").and_then(serde_json::Value::as_str) {
                    id = split_id.to_string();
                }
                if controller_id.is_none() {
                    controller_id = optional_object_field(split_object, "controller_id")?;
                }
                if let Some(policy) = optional_object_field(split_object, "leakage_policy")? {
                    leakage_policy = policy;
                }
                if let Some(explicit_params) = object_value_as_map(split_object.get("params")) {
                    params.extend(explicit_params);
                } else {
                    for (key, value) in split_object {
                        if !matches!(
                            key.as_str(),
                            "id" | "controller_id" | "leakage_policy" | "fold_set"
                        ) {
                            params.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
            _ => {
                return Err(DagMlError::GraphValidation(format!(
                    "{path}.split must be a string or object"
                )));
            }
        }
        Ok(SplitInvocation {
            id,
            controller_id,
            leakage_policy,
            params,
            fold_set,
        })
    }

    fn set_split_invocation(&mut self, split: SplitInvocation, path: &str) -> Result<()> {
        if self.split_invocation.replace(split).is_some() {
            return Err(DagMlError::GraphValidation(format!(
                "{path} declares a second split; DAG-ML expects one campaign split invocation"
            )));
        }
        Ok(())
    }

    fn next_node_id(&mut self, prefix: &str) -> Result<NodeId> {
        let id = NodeId::new(format!("{prefix}:compat.{}", self.node_counter))?;
        self.node_counter += 1;
        Ok(id)
    }

    fn next_generator_id(&mut self) -> Result<NodeId> {
        let id = NodeId::new(format!("generator:compat.{}", self.generator_counter))?;
        self.generator_counter += 1;
        Ok(id)
    }
}

fn optional_root_field<T>(
    root: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    match root.and_then(|object| object.get(key)) {
        Some(value) => Ok(Some(deserialize_value(value.clone(), key)?)),
        None => Ok(None),
    }
}

fn optional_object_field<T>(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    match object.get(key) {
        Some(value) => Ok(Some(deserialize_value(value.clone(), key)?)),
        None => Ok(None),
    }
}

fn optional_object_field_from_option<T>(
    object: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    match object.and_then(|object| object.get(key)) {
        Some(value) => Ok(Some(deserialize_value(value.clone(), key)?)),
        None => Ok(None),
    }
}

fn deserialize_value<T>(value: serde_json::Value, label: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    serde_json::from_value(value)
        .map_err(|error| DagMlError::GraphValidation(format!("failed to parse {label}: {error}")))
}

fn explicit_or_generated_node_id<F>(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    generated: F,
) -> Result<NodeId>
where
    F: FnOnce() -> Result<NodeId>,
{
    match object.get(key).and_then(serde_json::Value::as_str) {
        Some(id) => NodeId::new(id),
        None => generated(),
    }
}

fn first_object_value<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<&'a serde_json::Value> {
    keys.iter().find_map(|key| object.get(*key))
}

fn object_value_as_map(
    value: Option<&serde_json::Value>,
) -> Option<BTreeMap<String, serde_json::Value>> {
    value.and_then(|value| {
        value.as_object().map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
    })
}

fn compat_node_prefix(keyword: &str) -> &'static str {
    match keyword {
        "model" => "model",
        "y_processing" | "y_transform" => "target",
        "tag" => "tag",
        "exclude" | "filter" | "sample_filter" => "filter",
        "sample_augmentation" | "feature_augmentation" | "augmentation" => "augment",
        "chart" => "chart",
        _ => "transform",
    }
}

fn compat_param_aliases(keyword: &str) -> &'static [&'static str] {
    match keyword {
        "model" => &["model_params"],
        "preprocessing" | "processing" | "transform" => &[
            "preprocessing_params",
            "processing_params",
            "transform_params",
        ],
        "sample_augmentation" | "feature_augmentation" | "augmentation" => &["augmentation_params"],
        _ => &[],
    }
}

fn compat_augmentation_shape(
    kind: &str,
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<PipelineDslShapePlan> {
    if let Some(shape) = object.get("shape") {
        return deserialize_value(shape.clone(), "augmentation shape");
    }
    let mut sample_scope = crate::policy::AugmentationScope::None;
    let mut feature_scope = crate::policy::AugmentationScope::None;
    match kind {
        "sample" => sample_scope = crate::policy::AugmentationScope::TrainOnly,
        "feature" => feature_scope = crate::policy::AugmentationScope::TrainOnly,
        _ => {
            sample_scope = crate::policy::AugmentationScope::TrainOnly;
            feature_scope = crate::policy::AugmentationScope::TrainOnly;
        }
    }
    if let Some(apply_to) = object
        .get("policy")
        .and_then(serde_json::Value::as_object)
        .and_then(|policy| policy.get("apply_to"))
        .and_then(serde_json::Value::as_str)
    {
        match apply_to {
            "train_only" => {}
            "all" | "all_partitions" => {
                if sample_scope != crate::policy::AugmentationScope::None {
                    sample_scope = crate::policy::AugmentationScope::AllPartitions;
                }
                if feature_scope != crate::policy::AugmentationScope::None {
                    feature_scope = crate::policy::AugmentationScope::AllPartitions;
                }
            }
            "none" => {
                sample_scope = crate::policy::AugmentationScope::None;
                feature_scope = crate::policy::AugmentationScope::None;
            }
            other => {
                return Err(DagMlError::GraphValidation(format!(
                    "unsupported nirs4all augmentation policy apply_to `{other}`"
                )));
            }
        }
    }
    Ok(PipelineDslShapePlan {
        input_granularity: None,
        target_granularity: None,
        fit_rows: Some(FitBoundary::FoldTrain),
        predict_rows: Some(FitBoundary::FoldValidation),
        feature_namespace: None,
        feature_schema_fingerprint: None,
        target_space: None,
        aggregation_policy: None,
        augmentation_policy: Some(AugmentationPolicy {
            sample_scope,
            feature_scope,
            require_origin_id: true,
            inherit_group: true,
            inherit_target: true,
            unsafe_flags: BTreeSet::new(),
        }),
        selection_policy: None,
    })
}

fn compat_merge_modes(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<(String, bool, PipelineDslMergeOutput)> {
    let merge = object
        .get("merge")
        .ok_or_else(|| DagMlError::GraphValidation("merge step lacks `merge`".to_string()))?;
    let mode = merge
        .as_str()
        .or_else(|| {
            merge
                .as_object()
                .and_then(|object| object.get("mode"))
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("predictions");
    let include_original_data = object
        .get("include_original_data")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(matches!(
            mode,
            "all" | "mixed" | "predictions_plus_original"
        ));
    let output_as = match mode {
        "predictions" | "prediction" => PipelineDslMergeOutput::Predictions,
        "sources" | "source" => PipelineDslMergeOutput::Sources,
        "features" | "feature" | "concat" | "all" | "mixed" | "predictions_plus_original" => {
            PipelineDslMergeOutput::Features
        }
        other => {
            return Err(DagMlError::GraphValidation(format!(
                "unsupported nirs4all merge mode `{other}`"
            )));
        }
    };
    Ok((mode.to_string(), include_original_data, output_as))
}

fn compat_generator_metadata(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut metadata: BTreeMap<String, serde_json::Value> =
        optional_object_field(object, "metadata")?.unwrap_or_default();
    metadata.insert(
        "dsl_compat_generator".to_string(),
        serde_json::Value::String(key.to_string()),
    );
    Ok(metadata)
}

fn compat_branch_id(value: &serde_json::Value, index: usize) -> String {
    value
        .as_object()
        .and_then(|object| object.get("id"))
        .and_then(serde_json::Value::as_str)
        .map(|id| sanitize_branch_id(id, index))
        .unwrap_or_else(|| format!("choice{index}"))
}

fn sanitize_branch_id(input: &str, index: usize) -> String {
    let sanitized = sanitize_generation_label(input);
    if sanitized == "value" {
        format!("branch{index}")
    } else {
        sanitized
    }
}

fn step_has_prediction(step: &PipelineDslStep) -> bool {
    match step {
        PipelineDslStep::Model(_) | PipelineDslStep::MergeModel(_) => true,
        PipelineDslStep::Merge(step) => step.output_as == PipelineDslMergeOutput::Predictions,
        PipelineDslStep::Branch(step) => step
            .branches
            .iter()
            .any(|branch| branch.steps.iter().any(step_has_prediction)),
        PipelineDslStep::Generator(step) => generator_step_has_prediction(step),
        PipelineDslStep::Sequential(step) => step.steps.iter().any(step_has_prediction),
        _ => false,
    }
}

fn generator_step_has_prediction(generator: &PipelineDslGeneratorStep) -> bool {
    generator
        .branches
        .iter()
        .any(|branch| branch.steps.iter().any(step_has_prediction))
        || generator.stages.iter().any(|stage| {
            stage
                .branches
                .iter()
                .any(|branch| branch.steps.iter().any(step_has_prediction))
        })
}

fn generator_to_cartesian_stages(
    generator: PipelineDslGeneratorStep,
) -> Result<Vec<PipelineDslGeneratorStage>> {
    match generator.mode {
        PipelineDslGeneratorMode::Cartesian => Ok(generator.stages),
        PipelineDslGeneratorMode::Or => {
            if generator.pick.is_some()
                || generator.arrange.is_some()
                || generator.then_pick.is_some()
                || generator.then_arrange.is_some()
            {
                return Err(DagMlError::GraphValidation(format!(
                    "nirs4all-compatible data-only generator `{}` cannot be fused across downstream models when pick/arrange selectors are present",
                    generator.id
                )));
            }
            Ok(vec![PipelineDslGeneratorStage {
                id: sanitize_generation_label(generator.id.as_str()),
                selector: None,
                metadata: generator.metadata,
                branches: generator.branches,
            }])
        }
    }
}

fn single_stage(
    id: String,
    branch_id: &str,
    steps: Vec<PipelineDslStep>,
) -> PipelineDslGeneratorStage {
    PipelineDslGeneratorStage {
        id,
        selector: None,
        metadata: BTreeMap::new(),
        branches: vec![PipelineDslBranch {
            id: branch_id.to_string(),
            selector: None,
            metadata: BTreeMap::new(),
            steps,
        }],
    }
}

fn combined_cartesian_generator(
    id: NodeId,
    stages: Vec<PipelineDslGeneratorStage>,
) -> PipelineDslGeneratorStep {
    PipelineDslGeneratorStep {
        id,
        mode: PipelineDslGeneratorMode::Cartesian,
        branches: Vec::new(),
        stages,
        pick: None,
        arrange: None,
        then_pick: None,
        then_arrange: None,
        count: None,
        metadata: BTreeMap::from([(
            "dsl_compat_generator".to_string(),
            serde_json::Value::String("fused_data_to_prediction".to_string()),
        )]),
    }
}

fn compat_grid_rows(
    value: &serde_json::Value,
    path: &str,
) -> Result<Vec<BTreeMap<String, serde_json::Value>>> {
    let object = value
        .as_object()
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._grid_ must be an object")))?;
    if object.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "{path}._grid_ must contain at least one parameter"
        )));
    }
    let entries = object
        .iter()
        .map(|(key, value)| {
            let values = match value {
                serde_json::Value::Array(values) => values.clone(),
                _ => vec![value.clone()],
            };
            if values.is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "{path}._grid_.{key} has no values"
                )));
            }
            Ok((key.clone(), values))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rows = Vec::new();
    build_compat_grid_rows(&entries, 0, &mut BTreeMap::new(), &mut rows);
    Ok(rows)
}

fn build_compat_grid_rows(
    entries: &[(String, Vec<serde_json::Value>)],
    index: usize,
    current: &mut BTreeMap<String, serde_json::Value>,
    rows: &mut Vec<BTreeMap<String, serde_json::Value>>,
) {
    if index == entries.len() {
        rows.push(current.clone());
        return;
    }
    let (key, values) = &entries[index];
    for value in values {
        current.insert(key.clone(), value.clone());
        build_compat_grid_rows(entries, index + 1, current, rows);
        current.remove(key);
    }
}

fn compat_range_generator(
    value: &serde_json::Value,
    object: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<PipelineDslParamGenerator> {
    let param = object
        .get("param")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("n_components")
        .to_string();
    let (start, stop, step) = if let Some(values) = value.as_array() {
        if values.len() != 3 {
            return Err(DagMlError::GraphValidation(format!(
                "{path}._range_ array must be [start, stop, step]"
            )));
        }
        (
            json_f64(&values[0], path, "_range_[0]")?,
            json_f64(&values[1], path, "_range_[1]")?,
            json_f64(&values[2], path, "_range_[2]")?,
        )
    } else if let Some(spec) = value.as_object() {
        (
            json_f64(
                spec.get("start").ok_or_else(|| {
                    DagMlError::GraphValidation(format!("{path}._range_ lacks start"))
                })?,
                path,
                "start",
            )?,
            json_f64(
                spec.get("stop").ok_or_else(|| {
                    DagMlError::GraphValidation(format!("{path}._range_ lacks stop"))
                })?,
                path,
                "stop",
            )?,
            json_f64(
                spec.get("step").ok_or_else(|| {
                    DagMlError::GraphValidation(format!("{path}._range_ lacks step"))
                })?,
                path,
                "step",
            )?,
        )
    } else {
        return Err(DagMlError::GraphValidation(format!(
            "{path}._range_ must be an array or object"
        )));
    };
    Ok(PipelineDslParamGenerator::Range {
        name: optional_object_field(object, "name")?,
        param,
        start,
        stop,
        step,
        inclusive: object
            .get("inclusive")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        count: optional_object_field(object, "count")?,
    })
}

fn compat_log_range_generator(
    value: &serde_json::Value,
    object: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<PipelineDslParamGenerator> {
    let param = object
        .get("param")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("alpha")
        .to_string();
    let spec = value.as_object().ok_or_else(|| {
        DagMlError::GraphValidation(format!("{path}._log_range_ must be an object"))
    })?;
    let start = json_f64(
        spec.get("start")
            .or_else(|| spec.get("from"))
            .ok_or_else(|| {
                DagMlError::GraphValidation(format!("{path}._log_range_ lacks start/from"))
            })?,
        path,
        "start",
    )?;
    let stop = json_f64(
        spec.get("stop").or_else(|| spec.get("to")).ok_or_else(|| {
            DagMlError::GraphValidation(format!("{path}._log_range_ lacks stop/to"))
        })?,
        path,
        "stop",
    )?;
    let count = spec
        .get("count")
        .or_else(|| spec.get("num"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._log_range_ lacks count/num")))?
        as usize;
    Ok(PipelineDslParamGenerator::LogRange {
        name: optional_object_field(object, "name")?,
        param,
        start,
        stop,
        count,
        base: spec
            .get("base")
            .map(|value| json_f64(value, path, "base"))
            .transpose()?
            .unwrap_or(10.0),
    })
}

fn compat_grid_param_generator(
    value: &serde_json::Value,
    object: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<PipelineDslParamGenerator> {
    let grid = value
        .as_object()
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._grid_ must be an object")))?;
    let params = grid
        .iter()
        .map(|(key, value)| {
            let values = match value {
                serde_json::Value::Array(values) => values.clone(),
                _ => vec![value.clone()],
            };
            Ok((
                key.clone(),
                values
                    .into_iter()
                    .map(PipelineDslGeneratorValue::Value)
                    .collect::<Vec<_>>(),
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(PipelineDslParamGenerator::Grid {
        name: optional_object_field(object, "name")?,
        params,
        count: optional_object_field(object, "count")?,
    })
}

fn compat_zip_variants(
    value: &serde_json::Value,
    path: &str,
) -> Result<Vec<PipelineDslVariantChoice>> {
    let object = value
        .as_object()
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._zip_ must be an object")))?;
    let mut length = None;
    let mut columns = Vec::new();
    for (key, value) in object {
        let values = value.as_array().ok_or_else(|| {
            DagMlError::GraphValidation(format!("{path}._zip_.{key} must be an array"))
        })?;
        if let Some(expected) = length {
            if values.len() != expected {
                return Err(DagMlError::GraphValidation(format!(
                    "{path}._zip_ arrays must have equal lengths"
                )));
            }
        } else {
            length = Some(values.len());
        }
        columns.push((key.clone(), values.clone()));
    }
    let length = length.unwrap_or(0);
    if length == 0 {
        return Err(DagMlError::GraphValidation(format!(
            "{path}._zip_ must contain non-empty arrays"
        )));
    }
    Ok((0..length)
        .map(|index| {
            let params = columns
                .iter()
                .map(|(key, values)| (key.clone(), values[index].clone()))
                .collect::<BTreeMap<_, _>>();
            PipelineDslVariantChoice {
                label: format!("zip{index}"),
                params,
                value: None,
            }
        })
        .collect())
}

fn compat_sample_rows(
    object: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<Vec<BTreeMap<String, serde_json::Value>>> {
    let param_names = if let Some(param) = object.get("param").and_then(serde_json::Value::as_str) {
        vec![param.to_string()]
    } else if let Some(tune) = object.get("tune").and_then(serde_json::Value::as_array) {
        let params = tune
            .iter()
            .map(|value| {
                value.as_str().map(str::to_string).ok_or_else(|| {
                    DagMlError::GraphValidation(format!(
                        "{path}._sample_.tune entries must be strings"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if params.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "{path}._sample_.tune cannot be empty"
            )));
        }
        params
    } else {
        return Err(DagMlError::GraphValidation(format!(
            "{path}._sample_ requires `param` or `tune` for deterministic JSON lowering"
        )));
    };
    let from = json_f64(
        object
            .get("from")
            .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._sample_ lacks from")))?,
        path,
        "from",
    )?;
    let to = json_f64(
        object
            .get("to")
            .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._sample_ lacks to")))?,
        path,
        "to",
    )?;
    let count = object
        .get("num")
        .or_else(|| object.get("count"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._sample_ lacks num/count")))?
        as usize;
    if count == 0 {
        return Err(DagMlError::GraphValidation(format!(
            "{path}._sample_ count cannot be zero"
        )));
    }
    let distribution = object
        .get("distribution")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("uniform");
    if distribution == "log_uniform" && (from <= 0.0 || to <= 0.0) {
        return Err(DagMlError::GraphValidation(format!(
            "{path}._sample_ log_uniform requires positive from/to"
        )));
    }
    (0..count)
        .map(|index| {
            let ratio = if count == 1 {
                0.0
            } else {
                index as f64 / (count - 1) as f64
            };
            let sampled = match distribution {
                "uniform" => from + (to - from) * ratio,
                "log_uniform" => {
                    let start = from.log10();
                    let stop = to.log10();
                    10f64.powf(start + (stop - start) * ratio)
                }
                other => {
                    return Err(DagMlError::GraphValidation(format!(
                        "{path}._sample_ unsupported deterministic distribution `{other}`"
                    )));
                }
            };
            let mut row = BTreeMap::new();
            let value = serde_json::Value::Number(
                serde_json::Number::from_f64(sampled).ok_or_else(|| {
                    DagMlError::GraphValidation(format!(
                        "{path}._sample_ produced non-finite value"
                    ))
                })?,
            );
            for param in &param_names {
                row.insert(param.clone(), value.clone());
            }
            Ok(row)
        })
        .collect()
}

fn compat_sample_variants(
    value: &serde_json::Value,
    path: &str,
) -> Result<Vec<PipelineDslVariantChoice>> {
    let object = value
        .as_object()
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}._sample_ must be an object")))?;
    compat_sample_rows(object, path)?
        .into_iter()
        .enumerate()
        .map(|(index, params)| {
            Ok(PipelineDslVariantChoice {
                label: format!("sample{index}"),
                params,
                value: None,
            })
        })
        .collect()
}

fn json_f64(value: &serde_json::Value, path: &str, field: &str) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| DagMlError::GraphValidation(format!("{path}.{field} must be numeric")))
}

impl PipelineCompiler {
    fn compile_top_level_step(
        &mut self,
        step: &PipelineDslStep,
        external_data: &DataSource,
        current_data: &mut DataSource,
        pending_predictions: &mut Vec<PredictionSource>,
    ) -> Result<()> {
        self.compile_sequence_step(
            step,
            external_data,
            current_data,
            pending_predictions,
            None,
            BTreeMap::new(),
        )
    }

    fn compile_sequence_step(
        &mut self,
        step: &PipelineDslStep,
        original_data: &DataSource,
        current_data: &mut DataSource,
        pending_predictions: &mut Vec<PredictionSource>,
        branch_id: Option<&str>,
        extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<()> {
        match step {
            PipelineDslStep::Transform(step) => {
                *current_data = self.compile_data_operator_with_extra(
                    NodeKind::Transform,
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::YTransform(step) => {
                self.compile_y_transform_with_extra(step, extra_metadata)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Tag(step) => {
                *current_data = self.compile_data_operator_with_extra(
                    NodeKind::Tag,
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Exclude(step) => {
                *current_data = self.compile_data_operator_with_extra(
                    NodeKind::Exclude,
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Filter(step) => {
                *current_data =
                    self.compile_filter_operator("filter", step, current_data, extra_metadata)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::SampleFilter(step) => {
                *current_data =
                    self.compile_filter_operator("sample", step, current_data, extra_metadata)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Augmentation(step) => {
                *current_data = self.compile_data_operator_with_extra(
                    NodeKind::Augmentation,
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::FeatureAugmentation(step) => {
                *current_data = self.compile_augmentation_operator_with_extra(
                    "feature",
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::SampleAugmentation(step) => {
                *current_data = self.compile_augmentation_operator_with_extra(
                    "sample",
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::ConcatTransform(step) => {
                *current_data =
                    self.compile_concat_transform_with_extra(step, current_data, extra_metadata)?;
                pending_predictions.clear();
                Ok(())
            }
            PipelineDslStep::Model(step) => {
                pending_predictions.push(self.compile_model_with_extra(
                    step,
                    current_data,
                    branch_id,
                    extra_metadata,
                )?);
                Ok(())
            }
            PipelineDslStep::Branch(step) => {
                *pending_predictions =
                    self.compile_branch_with_extra(step, current_data, extra_metadata)?;
                Ok(())
            }
            PipelineDslStep::Generator(step) => {
                *pending_predictions =
                    self.compile_generator_with_extra(step, current_data, extra_metadata)?;
                Ok(())
            }
            PipelineDslStep::Sequential(step) => {
                self.compile_sequence_container(
                    step,
                    original_data,
                    current_data,
                    pending_predictions,
                    branch_id,
                    extra_metadata,
                )?;
                Ok(())
            }
            PipelineDslStep::Merge(step) => {
                match self.compile_merge_with_extra(
                    step,
                    pending_predictions,
                    original_data,
                    extra_metadata,
                )? {
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
                let prediction = self.compile_merge_model_with_extra(
                    step,
                    pending_predictions,
                    original_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                pending_predictions.push(prediction);
                Ok(())
            }
            PipelineDslStep::Chart(step) => {
                *current_data = self.compile_data_operator_with_extra(
                    NodeKind::Chart,
                    step,
                    current_data,
                    extra_metadata,
                )?;
                pending_predictions.clear();
                Ok(())
            }
        }
    }

    fn compile_sequence_container(
        &mut self,
        step: &PipelineDslSequenceStep,
        original_data: &DataSource,
        current_data: &mut DataSource,
        pending_predictions: &mut Vec<PredictionSource>,
        branch_id: Option<&str>,
        mut extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<()> {
        if step.steps.is_empty() {
            return Err(DagMlError::GraphValidation(
                "pipeline DSL sequential step has no child steps".to_string(),
            ));
        }
        if let Some(sequence_id) = &step.id {
            extra_metadata.insert(
                "dsl_sequence".to_string(),
                serde_json::Value::String(sequence_id.to_string()),
            );
        }
        if !step.metadata.is_empty() {
            extra_metadata.insert(
                "dsl_sequence_metadata".to_string(),
                serde_json::to_value(&step.metadata).map_err(|error| {
                    DagMlError::GraphValidation(format!(
                        "failed to serialize pipeline DSL sequential metadata: {error}"
                    ))
                })?,
            );
        }
        for child in &step.steps {
            self.compile_sequence_step(
                child,
                original_data,
                current_data,
                pending_predictions,
                branch_id,
                extra_metadata.clone(),
            )?;
        }
        Ok(())
    }

    fn compile_branch_with_extra(
        &mut self,
        step: &PipelineDslBranchStep,
        current_data: &DataSource,
        extra_metadata: BTreeMap<String, serde_json::Value>,
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
            let mut branch_metadata = branch_context_metadata(step, branch)?;
            branch_metadata.extend(extra_metadata.clone());
            for branch_step in &branch.steps {
                self.compile_sequence_step(
                    branch_step,
                    current_data,
                    &mut branch_data,
                    &mut branch_predictions,
                    Some(&branch.id),
                    branch_metadata.clone(),
                )?;
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

    fn compile_generator_with_extra(
        &mut self,
        step: &PipelineDslGeneratorStep,
        current_data: &DataSource,
        extra_metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<Vec<PredictionSource>> {
        let choices = expand_generator_sequences(step)?;
        if choices.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{}` produced no choices",
                step.id
            )));
        }
        let mut predictions = Vec::new();
        for (choice_index, choice) in choices.into_iter().enumerate() {
            let choice = namespace_generated_sequence(step, choice, choice_index)?;
            validate_branch_id(&choice.id)?;
            if choice.steps.is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL generator `{}` choice `{}` has no steps",
                    step.id, choice.id
                )));
            }
            let mut choice_data = current_data.clone();
            let mut choice_predictions = Vec::new();
            let mut choice_metadata = generator_choice_metadata(step, &choice)?;
            choice_metadata.extend(extra_metadata.clone());
            for choice_step in &choice.steps {
                self.compile_sequence_step(
                    choice_step,
                    current_data,
                    &mut choice_data,
                    &mut choice_predictions,
                    Some(&choice.id),
                    choice_metadata.clone(),
                )?;
            }
            if choice_predictions.is_empty() {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL generator `{}` choice `{}` must produce at least one model or merge prediction",
                    step.id, choice.id
                )));
            }
            let prediction_count = choice_predictions.len();
            for (prediction_index, prediction) in choice_predictions.into_iter().enumerate() {
                let input_name = if prediction_count == 1 {
                    format!("{}_oof", branch_input_prefix(&choice.id, choice_index))
                } else {
                    branch_prediction_input_name(
                        &choice.id,
                        choice_index,
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

    fn compile_filter_operator(
        &mut self,
        filter_kind: &str,
        step: &PipelineDslOperatorStep,
        input: &DataSource,
        mut extra: BTreeMap<String, serde_json::Value>,
    ) -> Result<DataSource> {
        extra.insert(
            "dsl_filter_kind".to_string(),
            serde_json::Value::String(filter_kind.to_string()),
        );
        self.compile_data_operator_with_extra(NodeKind::Exclude, step, input, extra)
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

fn expand_generator_sequences(step: &PipelineDslGeneratorStep) -> Result<Vec<GeneratedSequence>> {
    if step.count == Some(0) {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` count cannot be zero",
            step.id
        )));
    }
    match step.mode {
        PipelineDslGeneratorMode::Or => expand_or_generator_sequences(step),
        PipelineDslGeneratorMode::Cartesian => expand_cartesian_generator_sequences(step),
    }
}

fn expand_or_generator_sequences(
    step: &PipelineDslGeneratorStep,
) -> Result<Vec<GeneratedSequence>> {
    if !step.stages.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` uses mode `or` but declares cartesian stages",
            step.id
        )));
    }
    if step.branches.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` has no branches",
            step.id
        )));
    }
    let options = step
        .branches
        .iter()
        .enumerate()
        .map(|(index, branch)| {
            validate_branch_id(&branch.id)?;
            Ok(GeneratedSequence {
                id: generator_choice_id(&step.id, index),
                labels: vec![branch.id.clone()],
                steps: branch.steps.clone(),
                metadata: branch.metadata.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let choices = if let Some(sizes) = selection_sizes(step.pick)? {
        generated_pick_sequences(&options, &step.id, "pick", &sizes, step.count)?
    } else if let Some(sizes) = selection_sizes(step.arrange)? {
        generated_arrange_sequences(&options, &step.id, "arrange", &sizes, step.count)?
    } else {
        truncate_generated_sequences(options, step.count)
    };

    let choices = if let Some(sizes) = selection_sizes(step.then_pick)? {
        generated_pick_sequences(&choices, &step.id, "then_pick", &sizes, step.count)?
    } else if let Some(sizes) = selection_sizes(step.then_arrange)? {
        generated_arrange_sequences(&choices, &step.id, "then_arrange", &sizes, step.count)?
    } else {
        choices
    };
    Ok(truncate_generated_sequences(choices, step.count))
}

fn expand_cartesian_generator_sequences(
    step: &PipelineDslGeneratorStep,
) -> Result<Vec<GeneratedSequence>> {
    if !step.branches.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` uses mode `cartesian` but declares direct branches",
            step.id
        )));
    }
    if step.stages.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` has no cartesian stages",
            step.id
        )));
    }
    if step.pick.is_some()
        || step.arrange.is_some()
        || step.then_pick.is_some()
        || step.then_arrange.is_some()
    {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` cannot combine cartesian mode with pick/arrange selectors",
            step.id
        )));
    }

    let mut stage_options = Vec::<Vec<GeneratedSequence>>::new();
    for (stage_index, stage) in step.stages.iter().enumerate() {
        validate_branch_id(&stage.id)?;
        if stage.branches.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{}` stage `{}` has no branches",
                step.id, stage.id
            )));
        }
        let mut options = Vec::new();
        for branch in &stage.branches {
            validate_branch_id(&branch.id)?;
            let mut metadata = branch.metadata.clone();
            if let Some(selector) = &stage.selector {
                metadata.insert("dsl_generator_stage_selector".to_string(), selector.clone());
            }
            if !stage.metadata.is_empty() {
                metadata.insert(
                    "dsl_generator_stage_metadata".to_string(),
                    serde_json::to_value(&stage.metadata).map_err(|error| {
                        DagMlError::GraphValidation(format!(
                            "failed to serialize pipeline DSL generator `{}` stage `{}` metadata: {error}",
                            step.id, stage.id
                        ))
                    })?,
                );
            }
            options.push(GeneratedSequence {
                id: format!("{stage_index}:{}", branch.id),
                labels: vec![format!("{}:{}", stage.id, branch.id)],
                steps: branch.steps.clone(),
                metadata,
            });
        }
        stage_options.push(options);
    }

    let mut rows = Vec::<Vec<usize>>::new();
    build_cartesian_indices(&stage_options, 0, &mut Vec::new(), &mut rows, step.count);
    let mut choices = Vec::with_capacity(rows.len());
    for (choice_index, row) in rows.into_iter().enumerate() {
        let selected = row
            .into_iter()
            .enumerate()
            .map(|(stage_index, option_index)| stage_options[stage_index][option_index].clone())
            .collect::<Vec<_>>();
        choices.push(merge_generated_sequence(
            generator_choice_id(&step.id, choice_index),
            selected,
        )?);
    }
    Ok(choices)
}

fn generated_pick_sequences(
    options: &[GeneratedSequence],
    generator_id: &NodeId,
    mode: &str,
    sizes: &[usize],
    count: Option<usize>,
) -> Result<Vec<GeneratedSequence>> {
    let mut selections = Vec::<Vec<usize>>::new();
    for size in sizes {
        if *size == 0 || *size > options.len() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{generator_id}` {mode} size {size} is outside 1..={}",
                options.len()
            )));
        }
        build_combinations(
            options.len(),
            *size,
            0,
            &mut Vec::new(),
            &mut selections,
            count,
        );
    }
    selections
        .into_iter()
        .enumerate()
        .map(|(index, selection)| {
            let selected = selection
                .into_iter()
                .map(|option_index| options[option_index].clone())
                .collect::<Vec<_>>();
            merge_generated_sequence(generator_choice_id(generator_id, index), selected)
        })
        .collect()
}

fn generated_arrange_sequences(
    options: &[GeneratedSequence],
    generator_id: &NodeId,
    mode: &str,
    sizes: &[usize],
    count: Option<usize>,
) -> Result<Vec<GeneratedSequence>> {
    let mut selections = Vec::<Vec<usize>>::new();
    for size in sizes {
        if *size == 0 || *size > options.len() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{generator_id}` {mode} size {size} is outside 1..={}",
                options.len()
            )));
        }
        build_permutations(
            options.len(),
            *size,
            &mut BTreeSet::new(),
            &mut Vec::new(),
            &mut selections,
            count,
        );
    }
    selections
        .into_iter()
        .enumerate()
        .map(|(index, selection)| {
            let selected = selection
                .into_iter()
                .map(|option_index| options[option_index].clone())
                .collect::<Vec<_>>();
            merge_generated_sequence(generator_choice_id(generator_id, index), selected)
        })
        .collect()
}

fn merge_generated_sequence(
    id: String,
    sequences: Vec<GeneratedSequence>,
) -> Result<GeneratedSequence> {
    if sequences.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generated sequence `{id}` has no selected options"
        )));
    }
    let mut labels = Vec::new();
    let mut steps = Vec::new();
    let mut metadata = BTreeMap::new();
    for sequence in sequences {
        labels.extend(sequence.labels);
        steps.extend(sequence.steps);
        if !sequence.metadata.is_empty() {
            metadata.insert(
                format!("option:{}", metadata.len()),
                serde_json::to_value(sequence.metadata).map_err(|error| {
                    DagMlError::GraphValidation(format!(
                        "failed to serialize generated sequence `{id}` metadata: {error}"
                    ))
                })?,
            );
        }
    }
    Ok(GeneratedSequence {
        id,
        labels,
        steps,
        metadata,
    })
}

fn truncate_generated_sequences(
    mut sequences: Vec<GeneratedSequence>,
    count: Option<usize>,
) -> Vec<GeneratedSequence> {
    if let Some(limit) = count {
        sequences.truncate(limit);
    }
    sequences
}

fn build_cartesian_indices<T>(
    stages: &[Vec<T>],
    stage_index: usize,
    current: &mut Vec<usize>,
    rows: &mut Vec<Vec<usize>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| rows.len() >= limit) {
        return;
    }
    if stage_index == stages.len() {
        rows.push(current.clone());
        return;
    }
    for option_index in 0..stages[stage_index].len() {
        current.push(option_index);
        build_cartesian_indices(stages, stage_index + 1, current, rows, count);
        current.pop();
        if count.is_some_and(|limit| rows.len() >= limit) {
            break;
        }
    }
}

fn selection_sizes(selection: Option<PipelineDslSelectionSpec>) -> Result<Option<Vec<usize>>> {
    selection
        .map(|selection| match selection {
            PipelineDslSelectionSpec::Single(size) => {
                if size == 0 {
                    return Err(DagMlError::GraphValidation(
                        "pipeline DSL generator selection size cannot be zero".to_string(),
                    ));
                }
                Ok(vec![size])
            }
            PipelineDslSelectionSpec::Range([start, stop]) => {
                if start == 0 || stop == 0 || start > stop {
                    return Err(DagMlError::GraphValidation(format!(
                        "pipeline DSL generator selection range [{start}, {stop}] is invalid"
                    )));
                }
                Ok((start..=stop).collect())
            }
        })
        .transpose()
}

fn generator_choice_id(generator_id: &NodeId, choice_index: usize) -> String {
    format!("{generator_id}:choice{choice_index}")
}

fn generator_choice_metadata(
    step: &PipelineDslGeneratorStep,
    choice: &GeneratedSequence,
) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut metadata = step.metadata.clone();
    metadata.insert(
        "dsl_generator".to_string(),
        serde_json::Value::String(step.id.to_string()),
    );
    metadata.insert(
        "dsl_generator_mode".to_string(),
        serde_json::to_value(step.mode).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL generator `{}` mode: {error}",
                step.id
            ))
        })?,
    );
    metadata.insert(
        "dsl_generator_choice".to_string(),
        serde_json::Value::String(choice.id.clone()),
    );
    metadata.insert(
        "dsl_generator_labels".to_string(),
        serde_json::to_value(&choice.labels).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL generator `{}` choice labels: {error}",
                step.id
            ))
        })?,
    );
    if !choice.metadata.is_empty() {
        metadata.insert(
            "dsl_generator_choice_metadata".to_string(),
            serde_json::to_value(&choice.metadata).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL generator `{}` choice metadata: {error}",
                    step.id
                ))
            })?,
        );
    }
    Ok(metadata)
}

fn namespace_generated_sequence(
    generator: &PipelineDslGeneratorStep,
    mut choice: GeneratedSequence,
    choice_index: usize,
) -> Result<GeneratedSequence> {
    let mut node_map = BTreeMap::<NodeId, NodeId>::new();
    let mut counter = 0usize;
    for step in &mut choice.steps {
        namespace_step_ids(generator, choice_index, step, &mut counter, &mut node_map)?;
    }
    for step in &mut choice.steps {
        rewrite_step_node_refs(step, &node_map);
    }
    Ok(choice)
}

fn namespace_step_ids(
    generator: &PipelineDslGeneratorStep,
    choice_index: usize,
    step: &mut PipelineDslStep,
    counter: &mut usize,
    node_map: &mut BTreeMap<NodeId, NodeId>,
) -> Result<()> {
    match step {
        PipelineDslStep::Transform(step)
        | PipelineDslStep::YTransform(step)
        | PipelineDslStep::Tag(step)
        | PipelineDslStep::Exclude(step)
        | PipelineDslStep::Filter(step)
        | PipelineDslStep::SampleFilter(step)
        | PipelineDslStep::Augmentation(step)
        | PipelineDslStep::FeatureAugmentation(step)
        | PipelineDslStep::SampleAugmentation(step)
        | PipelineDslStep::Model(step)
        | PipelineDslStep::Chart(step) => {
            namespace_operator_step_id(generator, choice_index, step, counter, node_map)?;
        }
        PipelineDslStep::ConcatTransform(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    namespace_operator_step_id(
                        generator,
                        choice_index,
                        branch_step,
                        counter,
                        node_map,
                    )?;
                }
            }
        }
        PipelineDslStep::Branch(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    namespace_step_ids(generator, choice_index, branch_step, counter, node_map)?;
                }
            }
        }
        PipelineDslStep::Generator(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    namespace_step_ids(generator, choice_index, branch_step, counter, node_map)?;
                }
            }
            for stage in &mut step.stages {
                for branch in &mut stage.branches {
                    for branch_step in &mut branch.steps {
                        namespace_step_ids(
                            generator,
                            choice_index,
                            branch_step,
                            counter,
                            node_map,
                        )?;
                    }
                }
            }
        }
        PipelineDslStep::Sequential(step) => {
            if let Some(id) = &mut step.id {
                namespace_node_id_field(generator, choice_index, id, counter, node_map)?;
            }
            for child in &mut step.steps {
                namespace_step_ids(generator, choice_index, child, counter, node_map)?;
            }
        }
        PipelineDslStep::Merge(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
        }
        PipelineDslStep::MergeModel(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
        }
    }
    Ok(())
}

fn namespace_operator_step_id(
    generator: &PipelineDslGeneratorStep,
    choice_index: usize,
    step: &mut PipelineDslOperatorStep,
    counter: &mut usize,
    node_map: &mut BTreeMap<NodeId, NodeId>,
) -> Result<()> {
    namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)
}

fn namespace_node_id_field(
    generator: &PipelineDslGeneratorStep,
    choice_index: usize,
    node_id: &mut NodeId,
    counter: &mut usize,
    node_map: &mut BTreeMap<NodeId, NodeId>,
) -> Result<()> {
    let original = node_id.clone();
    if node_map.contains_key(&original) {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` choice `{}` reuses node id `{original}`; generated choices require unique node ids inside each expanded sequence",
            generator.id, choice_index
        )));
    }
    let next = namespaced_generated_node_id(&generator.id, choice_index, *counter, &original)?;
    *counter += 1;
    *node_id = next.clone();
    node_map.insert(original, next);
    Ok(())
}

fn namespaced_generated_node_id(
    generator_id: &NodeId,
    choice_index: usize,
    node_index: usize,
    original: &NodeId,
) -> Result<NodeId> {
    let generator = sanitized_id_fragment(generator_id.as_str(), 32);
    let suffix = sanitized_id_fragment(original.as_str(), 28);
    NodeId::new(format!(
        "gen:{generator}:c{choice_index}:n{node_index}.{suffix}"
    ))
}

fn sanitized_id_fragment(input: &str, max_len: usize) -> String {
    let sanitized = sanitize_generation_label(input);
    let mut fragment = sanitized.chars().take(max_len).collect::<String>();
    if fragment.is_empty() {
        fragment = "x".to_string();
    }
    fragment
}

fn rewrite_step_node_refs(step: &mut PipelineDslStep, node_map: &BTreeMap<NodeId, NodeId>) {
    match step {
        PipelineDslStep::Transform(_)
        | PipelineDslStep::YTransform(_)
        | PipelineDslStep::Tag(_)
        | PipelineDslStep::Exclude(_)
        | PipelineDslStep::Filter(_)
        | PipelineDslStep::SampleFilter(_)
        | PipelineDslStep::Augmentation(_)
        | PipelineDslStep::FeatureAugmentation(_)
        | PipelineDslStep::SampleAugmentation(_)
        | PipelineDslStep::Model(_)
        | PipelineDslStep::Chart(_) => {}
        PipelineDslStep::ConcatTransform(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    rewrite_operator_step_refs(branch_step, node_map);
                }
            }
        }
        PipelineDslStep::Branch(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    rewrite_step_node_refs(branch_step, node_map);
                }
            }
        }
        PipelineDslStep::Generator(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    rewrite_step_node_refs(branch_step, node_map);
                }
            }
            for stage in &mut step.stages {
                for branch in &mut stage.branches {
                    for branch_step in &mut branch.steps {
                        rewrite_step_node_refs(branch_step, node_map);
                    }
                }
            }
        }
        PipelineDslStep::Sequential(step) => {
            for child in &mut step.steps {
                rewrite_step_node_refs(child, node_map);
            }
        }
        PipelineDslStep::Merge(step) => {
            rewrite_merge_selectors(&mut step.selectors, node_map);
        }
        PipelineDslStep::MergeModel(_) => {}
    }
}

fn rewrite_operator_step_refs(
    _step: &mut PipelineDslOperatorStep,
    _node_map: &BTreeMap<NodeId, NodeId>,
) {
}

fn rewrite_merge_selectors(
    selectors: &mut [PipelineDslMergeSelector],
    node_map: &BTreeMap<NodeId, NodeId>,
) {
    for selector in selectors {
        if let Some(model) = &selector.model {
            if let Some(rewritten) = node_map.get(model) {
                selector.model = Some(rewritten.clone());
            }
        }
    }
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
    fn compiles_sequential_filter_and_or_generator_surface() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-generator-or-parity",
  "steps": [
    {
      "kind": "sequential",
      "id": "seq:pre",
      "steps": [
        {
          "kind": "sample_filter",
          "id": "filter:y_outlier",
          "operator": {"class": "nirs4all.operators.filters.YOutlierFilter"},
          "params": {"mode": "any"}
        },
        {
          "kind": "transform",
          "id": "transform:scale",
          "operator": {"class": "sklearn.preprocessing.StandardScaler"}
        }
      ]
    },
    {
      "kind": "generator",
      "id": "generator:model_choices",
      "mode": "or",
      "pick": 1,
      "branches": [
        {
          "id": "pls",
          "steps": [
            {
              "kind": "model",
              "id": "model:pls",
              "operator": {"class": "sklearn.cross_decomposition.PLSRegression"},
              "params": {"n_components": 8}
            }
          ]
        },
        {
          "id": "rf",
          "steps": [
            {
              "kind": "model",
              "id": "model:rf",
              "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
              "params": {"n_estimators": 64}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:generated",
      "output_as": "features",
      "include_original_data": false,
      "selectors": [
        {"branch": "generator:model_choices:choice0", "select": "all"}
      ]
    }
  ]
}"#,
        )
        .unwrap();

        let graph = compile_pipeline_dsl(&spec).unwrap();
        graph.validate().unwrap();
        let filter = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "filter:y_outlier")
            .unwrap();
        assert_eq!(filter.kind, NodeKind::Exclude);
        assert_eq!(filter.metadata["dsl_filter_kind"], "sample");

        let generated_models = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Model)
            .collect::<Vec<_>>();
        assert_eq!(generated_models.len(), 2);
        assert!(generated_models
            .iter()
            .all(|node| node.id.as_str().starts_with("gen:generator_model_choices")));
        assert!(generated_models.iter().all(|node| {
            node.metadata
                .get("dsl_generator")
                .and_then(|value| value.as_str())
                == Some("generator:model_choices")
        }));

        let merge_inputs = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "merge:generated")
            .unwrap()
            .ports
            .inputs
            .iter()
            .map(|port| port.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            merge_inputs,
            vec![
                "generator_model_choices_choice0_oof",
                "generator_model_choices_choice1_oof"
            ]
        );
    }

    #[test]
    fn compiles_cartesian_generator_as_explicit_prediction_choices() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-generator-cartesian-parity",
  "steps": [
    {
      "kind": "generator",
      "id": "generator:cartesian",
      "mode": "cartesian",
      "stages": [
        {
          "id": "preproc",
          "branches": [
            {
              "id": "snv",
              "steps": [
                {
                  "kind": "transform",
                  "id": "transform:snv",
                  "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate"}
                }
              ]
            },
            {
              "id": "msc",
              "steps": [
                {
                  "kind": "transform",
                  "id": "transform:msc",
                  "operator": {"class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"}
                }
              ]
            }
          ]
        },
        {
          "id": "model",
          "branches": [
            {
              "id": "ridge",
              "steps": [
                {
                  "kind": "model",
                  "id": "model:ridge",
                  "operator": {"class": "sklearn.linear_model.Ridge"}
                }
              ]
            },
            {
              "id": "lasso",
              "steps": [
                {
                  "kind": "model",
                  "id": "model:lasso",
                  "operator": {"class": "sklearn.linear_model.Lasso"}
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:cartesian",
      "output_as": "features",
      "include_original_data": false
    }
  ]
}"#,
        )
        .unwrap();

        let graph = compile_pipeline_dsl(&spec).unwrap();
        graph.validate().unwrap();
        let models = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Model)
            .collect::<Vec<_>>();
        assert_eq!(models.len(), 4);
        assert!(models.iter().all(|node| {
            node.metadata
                .get("dsl_generator_mode")
                .and_then(|value| value.as_str())
                == Some("cartesian")
        }));
        let merge = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "merge:cartesian")
            .unwrap();
        assert_eq!(merge.ports.inputs.len(), 4);
        assert_eq!(
            graph
                .edges
                .iter()
                .filter(|edge| edge.target.node_id.as_str() == "merge:cartesian")
                .count(),
            4
        );
    }

    #[test]
    fn refuses_generator_choice_without_prediction_output() {
        let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-generator-bad-choice",
  "steps": [
    {
      "kind": "generator",
      "id": "generator:bad",
      "branches": [
        {
          "id": "transform_only",
          "steps": [
            {
              "kind": "transform",
              "id": "transform:only",
              "operator": {"class": "sklearn.preprocessing.StandardScaler"}
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

    #[test]
    fn parses_nirs4all_compat_pipeline_and_fuses_data_generators() {
        let spec = parse_pipeline_dsl_json(
            br#"{
  "id": "dsl-nirs4all-compat-fused",
  "pipeline": [
    {"sources": ["nir"]},
    {"_cartesian_": [
      {"_or_": ["SNV", "MSC", null]},
      {"_or_": [null, {"preprocessing": "SavitzkyGolay", "params": {"window": 11, "deriv": 1}}]}
    ]},
    {"split": {"type": "GroupKFold", "n_splits": 3}},
    {"_chain_": [
      {"_grid_": {"model": ["PLSRegression"], "n_components": [5, 10]}},
      {"_grid_": {"model": ["Ridge"], "alpha": [0.1, 1.0]}},
      {"_sample_": {"model": "SVR", "distribution": "log_uniform", "from": 0.001, "to": 1.0, "num": 2, "tune": ["C", "gamma"], "kernel": "rbf"}}
    ]},
    {"merge": "all"},
    {"model": "Ridge", "id": "model:meta", "params": {"alpha": 0.5}}
  ]
}"#,
        )
        .unwrap();

        assert_eq!(spec.steps.len(), 2);
        assert_eq!(
            spec.split_invocation
                .as_ref()
                .unwrap()
                .params
                .get("type")
                .unwrap(),
            "GroupKFold"
        );

        let graph = compile_pipeline_dsl(&spec).unwrap();
        graph.validate().unwrap();
        let meta = graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "model:meta")
            .unwrap();
        assert_eq!(meta.kind, NodeKind::Model);
        assert!(meta
            .ports
            .inputs
            .iter()
            .any(|port| port.name == "x_original"));
        assert!(graph.edges.iter().any(|edge| {
            edge.target.node_id.as_str() == "model:meta"
                && edge.contract.kind == PortKind::Prediction
                && edge.contract.requires_oof
        }));
        assert!(graph.nodes.iter().any(|node| {
            node.metadata
                .get("dsl_compat_keyword")
                .and_then(serde_json::Value::as_str)
                == Some("preprocessing")
        }));
        assert!(graph.nodes.iter().any(|node| {
            node.kind == NodeKind::Model
                && node.params.contains_key("C")
                && node.params.contains_key("gamma")
        }));
    }

    #[test]
    fn parses_nirs4all_range_attached_to_following_model() {
        let spec = parse_pipeline_dsl_json(
            br#"{
  "id": "dsl-nirs4all-compat-range",
  "pipeline": [
    {"_range_": [5, 15, 5]},
    {"model": "PLSRegression", "id": "model:pls"}
  ]
}"#,
        )
        .unwrap();

        let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
        assert_eq!(compiled.generation.dimensions.len(), 1);
        assert_eq!(compiled.generation.dimensions[0].choices.len(), 3);
        assert_eq!(
            compiled.generation.dimensions[0].choices[0].param_overrides[0].params["n_components"],
            5.0
        );
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
