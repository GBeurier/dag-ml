//! Backend-neutral training-loss and metric semantic contracts.
//!
//! This module owns serializable semantics and implementation identity only.
//! Host controllers retain model objects, tensors, autodiff graphs and callback
//! objects. Process-local executable implementations are resolved by binding
//! registries from opaque [`ImplementationDescriptor::registry_key`] values.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::canonical::{
    deserialize_external_contract, parse_typed_json, validate_typed_serde_value,
};
use crate::error::{DagMlError, Result};
use crate::ids::NodeId;
use crate::oof::PredictionPartition;
use crate::phase::Phase;
use crate::policy::PredictionLevel;
use crate::selection::MetricObjective;
use crate::training::PredictionKind;

pub const LOSS_SPEC_SCHEMA_VERSION: u32 = 1;
pub const METRIC_SPEC_SCHEMA_VERSION: u32 = 1;
pub const IMPLEMENTATION_DESCRIPTOR_SCHEMA_VERSION: u32 = 1;
pub const LOSS_ROLE_SCHEMA_VERSION: u32 = 1;
pub const METRIC_ROLE_SCHEMA_VERSION: u32 = 1;

pub const LOSS_SPEC_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/loss_spec.v1.schema.json";
pub const METRIC_SPEC_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/metric_spec.v1.schema.json";
pub const IMPLEMENTATION_DESCRIPTOR_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/implementation_descriptor.v1.schema.json";
pub const TRAINING_LOSS_ROLE_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/training_loss_role.v1.schema.json";
pub const METRIC_ROLE_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/metric_role.v1.schema.json";

const FINGERPRINT_LEN: usize = 64;
const FORBIDDEN_EXECUTABLE_KEYS: &[&str] = &[
    "bytecode",
    "callable",
    "code",
    "function_source",
    "import_path",
    "module_path",
    "pickle",
    "serialized_callable",
    "source_code",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticSpecKind {
    BuiltIn,
    Custom,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningTaskKind {
    Regression,
    BinaryClassification,
    MulticlassClassification,
    MultilabelClassification,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionInput {
    Target,
    Prediction,
    SampleWeight,
    MissingMask,
    Group,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossReduction {
    Mean,
    Sum,
    WeightedMean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossCapability {
    Differentiable,
    DistributedReduction,
    PerOutput,
    SupportsMissingMask,
    SupportsSampleWeights,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricReduction {
    Global,
    Mean,
    Sum,
    WeightedMean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDecomposition {
    Global,
    PerOutput,
    PerUnit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricCapability {
    Decomposable,
    DistributedReduction,
    SupportsMissingMask,
    SupportsSampleWeights,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationSemanticKind {
    Loss,
    Metric,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationCapability {
    Deterministic,
    Differentiable,
    DistributedReduction,
    NeedsGil,
    ProcessSafe,
    SupportsMissingMask,
    SupportsSampleWeights,
    ThreadSafe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortabilityClass {
    HostLocal,
    PortableRegistered,
    PortableBuiltIn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayabilityClass {
    ProcessLocal,
    RegistryRequired,
    Detached,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossSpec {
    pub schema_version: u32,
    pub loss_id: String,
    pub kind: SemanticSpecKind,
    pub task_kinds: BTreeSet<LearningTaskKind>,
    pub prediction_kinds: BTreeSet<PredictionKind>,
    pub objective: MetricObjective,
    pub reduction: LossReduction,
    pub required_inputs: BTreeSet<CriterionInput>,
    pub capabilities: BTreeSet<LossCapability>,
    pub parameters: serde_json::Value,
    pub spec_fingerprint: String,
}

impl LossSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss_id: impl Into<String>,
        kind: SemanticSpecKind,
        task_kinds: BTreeSet<LearningTaskKind>,
        prediction_kinds: BTreeSet<PredictionKind>,
        reduction: LossReduction,
        required_inputs: BTreeSet<CriterionInput>,
        capabilities: BTreeSet<LossCapability>,
        parameters: serde_json::Value,
    ) -> Result<Self> {
        let mut spec = Self {
            schema_version: LOSS_SPEC_SCHEMA_VERSION,
            loss_id: loss_id.into(),
            kind,
            task_kinds,
            prediction_kinds,
            objective: MetricObjective::Minimize,
            reduction,
            required_inputs,
            capabilities,
            parameters,
            spec_fingerprint: String::new(),
        };
        spec.spec_fingerprint = spec.compute_fingerprint()?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let spec: Self =
            deserialize_external_contract(json, "loss spec", DagMlError::CampaignValidation)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        fingerprint_without(self, "spec_fingerprint", "loss spec")
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version("loss spec", self.schema_version, LOSS_SPEC_SCHEMA_VERSION)?;
        validate_versioned_id("loss", &self.loss_id)?;
        validate_nonempty_set("loss task_kinds", &self.task_kinds)?;
        validate_nonempty_set("loss prediction_kinds", &self.prediction_kinds)?;
        if self.objective != MetricObjective::Minimize {
            return contract_error("loss objective must be minimize in schema version 1");
        }
        validate_required_target_prediction("loss", &self.required_inputs)?;
        validate_parameters("loss", &self.parameters)?;
        if self.reduction == LossReduction::WeightedMean
            && !self.required_inputs.contains(&CriterionInput::SampleWeight)
        {
            return contract_error("weighted_mean loss requires sample_weight input");
        }
        if self.required_inputs.contains(&CriterionInput::SampleWeight)
            && !self
                .capabilities
                .contains(&LossCapability::SupportsSampleWeights)
        {
            return contract_error(
                "loss requiring sample_weight must declare supports_sample_weights",
            );
        }
        if self.required_inputs.contains(&CriterionInput::MissingMask)
            && !self
                .capabilities
                .contains(&LossCapability::SupportsMissingMask)
        {
            return contract_error(
                "loss requiring missing_mask must declare supports_missing_mask",
            );
        }
        validate_fingerprint("loss spec", &self.spec_fingerprint)?;
        let expected = self.compute_fingerprint()?;
        if self.spec_fingerprint != expected {
            return contract_error(format!(
                "loss spec fingerprint mismatch: declared {}, expected {expected}",
                self.spec_fingerprint
            ));
        }
        Ok(())
    }

    pub fn validate_compatibility(
        &self,
        task_kind: LearningTaskKind,
        prediction_kind: PredictionKind,
    ) -> Result<()> {
        self.validate()?;
        if !self.task_kinds.contains(&task_kind)
            || !self.prediction_kinds.contains(&prediction_kind)
        {
            return contract_error(format!(
                "loss `{}` is not compatible with task {task_kind:?} and prediction {prediction_kind:?}",
                self.loss_id
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSpec {
    pub schema_version: u32,
    pub metric_id: String,
    pub kind: SemanticSpecKind,
    pub task_kinds: BTreeSet<LearningTaskKind>,
    pub prediction_kinds: BTreeSet<PredictionKind>,
    pub objective: MetricObjective,
    pub supported_levels: BTreeSet<PredictionLevel>,
    pub decomposition: MetricDecomposition,
    pub reduction: MetricReduction,
    pub required_inputs: BTreeSet<CriterionInput>,
    pub capabilities: BTreeSet<MetricCapability>,
    pub parameters: serde_json::Value,
    pub spec_fingerprint: String,
}

impl MetricSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metric_id: impl Into<String>,
        kind: SemanticSpecKind,
        task_kinds: BTreeSet<LearningTaskKind>,
        prediction_kinds: BTreeSet<PredictionKind>,
        objective: MetricObjective,
        supported_levels: BTreeSet<PredictionLevel>,
        decomposition: MetricDecomposition,
        reduction: MetricReduction,
        required_inputs: BTreeSet<CriterionInput>,
        capabilities: BTreeSet<MetricCapability>,
        parameters: serde_json::Value,
    ) -> Result<Self> {
        let mut spec = Self {
            schema_version: METRIC_SPEC_SCHEMA_VERSION,
            metric_id: metric_id.into(),
            kind,
            task_kinds,
            prediction_kinds,
            objective,
            supported_levels,
            decomposition,
            reduction,
            required_inputs,
            capabilities,
            parameters,
            spec_fingerprint: String::new(),
        };
        spec.spec_fingerprint = spec.compute_fingerprint()?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let spec: Self =
            deserialize_external_contract(json, "metric spec", DagMlError::CampaignValidation)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        fingerprint_without(self, "spec_fingerprint", "metric spec")
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version(
            "metric spec",
            self.schema_version,
            METRIC_SPEC_SCHEMA_VERSION,
        )?;
        validate_versioned_id("metric", &self.metric_id)?;
        validate_nonempty_set("metric task_kinds", &self.task_kinds)?;
        validate_nonempty_set("metric prediction_kinds", &self.prediction_kinds)?;
        validate_nonempty_set("metric supported_levels", &self.supported_levels)?;
        validate_required_target_prediction("metric", &self.required_inputs)?;
        validate_parameters("metric", &self.parameters)?;
        match (self.decomposition, self.reduction) {
            (MetricDecomposition::Global, MetricReduction::Global) => {}
            (MetricDecomposition::Global, _) => {
                return contract_error("global metric decomposition requires global reduction");
            }
            (_, MetricReduction::Global) => {
                return contract_error("decomposed metric cannot use global reduction");
            }
            _ => {}
        }
        if self.reduction == MetricReduction::WeightedMean
            && !self.required_inputs.contains(&CriterionInput::SampleWeight)
        {
            return contract_error("weighted_mean metric requires sample_weight input");
        }
        if self.reduction == MetricReduction::WeightedMean
            && self.decomposition != MetricDecomposition::PerUnit
        {
            return contract_error("weighted_mean metric requires per_unit decomposition");
        }
        if self.decomposition != MetricDecomposition::Global
            && !self.capabilities.contains(&MetricCapability::Decomposable)
        {
            return contract_error("decomposed metric must declare decomposable capability");
        }
        if self.required_inputs.contains(&CriterionInput::SampleWeight)
            && !self
                .capabilities
                .contains(&MetricCapability::SupportsSampleWeights)
        {
            return contract_error(
                "metric requiring sample_weight must declare supports_sample_weights",
            );
        }
        if self.required_inputs.contains(&CriterionInput::MissingMask)
            && !self
                .capabilities
                .contains(&MetricCapability::SupportsMissingMask)
        {
            return contract_error(
                "metric requiring missing_mask must declare supports_missing_mask",
            );
        }
        validate_fingerprint("metric spec", &self.spec_fingerprint)?;
        let expected = self.compute_fingerprint()?;
        if self.spec_fingerprint != expected {
            return contract_error(format!(
                "metric spec fingerprint mismatch: declared {}, expected {expected}",
                self.spec_fingerprint
            ));
        }
        Ok(())
    }

    pub fn validate_compatibility(
        &self,
        task_kind: LearningTaskKind,
        prediction_kind: PredictionKind,
        level: PredictionLevel,
    ) -> Result<()> {
        self.validate()?;
        if !self.task_kinds.contains(&task_kind)
            || !self.prediction_kinds.contains(&prediction_kind)
            || !self.supported_levels.contains(&level)
        {
            return contract_error(format!(
                "metric `{}` is not compatible with task {task_kind:?}, prediction {prediction_kind:?}, and level {level:?}",
                self.metric_id
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImplementationDescriptor {
    pub schema_version: u32,
    pub semantic_kind: ImplementationSemanticKind,
    pub semantic_id: String,
    pub semantic_fingerprint: String,
    pub provider_id: String,
    pub binding_id: String,
    pub implementation_version: String,
    pub implementation_fingerprint: String,
    pub supported_controller_families: BTreeSet<String>,
    pub runtime_requirements: BTreeSet<String>,
    pub capabilities: BTreeSet<ImplementationCapability>,
    pub portability: PortabilityClass,
    pub replayability: ReplayabilityClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry_key: Option<String>,
    pub descriptor_fingerprint: String,
}

impl ImplementationDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        semantic_kind: ImplementationSemanticKind,
        semantic_id: impl Into<String>,
        semantic_fingerprint: impl Into<String>,
        provider_id: impl Into<String>,
        binding_id: impl Into<String>,
        implementation_version: impl Into<String>,
        implementation_fingerprint: impl Into<String>,
        supported_controller_families: BTreeSet<String>,
        runtime_requirements: BTreeSet<String>,
        capabilities: BTreeSet<ImplementationCapability>,
        portability: PortabilityClass,
        replayability: ReplayabilityClass,
        registry_key: Option<String>,
    ) -> Result<Self> {
        let mut descriptor = Self {
            schema_version: IMPLEMENTATION_DESCRIPTOR_SCHEMA_VERSION,
            semantic_kind,
            semantic_id: semantic_id.into(),
            semantic_fingerprint: semantic_fingerprint.into(),
            provider_id: provider_id.into(),
            binding_id: binding_id.into(),
            implementation_version: implementation_version.into(),
            implementation_fingerprint: implementation_fingerprint.into(),
            supported_controller_families,
            runtime_requirements,
            capabilities,
            portability,
            replayability,
            registry_key,
            descriptor_fingerprint: String::new(),
        };
        descriptor.descriptor_fingerprint = descriptor.compute_fingerprint()?;
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let descriptor: Self = deserialize_external_contract(
            json,
            "implementation descriptor",
            DagMlError::CampaignValidation,
        )?;
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        fingerprint_without(self, "descriptor_fingerprint", "implementation descriptor")
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version(
            "implementation descriptor",
            self.schema_version,
            IMPLEMENTATION_DESCRIPTOR_SCHEMA_VERSION,
        )?;
        validate_versioned_id("implementation semantic", &self.semantic_id)?;
        validate_fingerprint("implementation semantic", &self.semantic_fingerprint)?;
        validate_token("provider_id", &self.provider_id)?;
        validate_token("binding_id", &self.binding_id)?;
        validate_token("implementation_version", &self.implementation_version)?;
        validate_fingerprint("implementation", &self.implementation_fingerprint)?;
        validate_string_set(
            "supported_controller_families",
            &self.supported_controller_families,
        )?;
        validate_string_set("runtime_requirements", &self.runtime_requirements)?;
        if let Some(registry_key) = &self.registry_key {
            validate_token("registry_key", registry_key)?;
        }
        match self.portability {
            PortabilityClass::HostLocal => {
                if self.registry_key.is_none() {
                    return contract_error("host_local implementation requires registry_key");
                }
                if self.replayability == ReplayabilityClass::Detached {
                    return contract_error(
                        "host_local implementation cannot be detached-replayable",
                    );
                }
            }
            PortabilityClass::PortableRegistered => {
                if self.registry_key.is_none() {
                    return contract_error(
                        "portable_registered implementation requires registry_key",
                    );
                }
                if self.replayability != ReplayabilityClass::RegistryRequired {
                    return contract_error(
                        "portable_registered implementation requires registry_required replay",
                    );
                }
            }
            PortabilityClass::PortableBuiltIn => {
                if self.registry_key.is_some() {
                    return contract_error("portable_builtin implementation forbids registry_key");
                }
                if self.replayability != ReplayabilityClass::Detached {
                    return contract_error(
                        "portable_builtin implementation must be detached-replayable",
                    );
                }
            }
        }
        validate_fingerprint("implementation descriptor", &self.descriptor_fingerprint)?;
        let expected = self.compute_fingerprint()?;
        if self.descriptor_fingerprint != expected {
            return contract_error(format!(
                "implementation descriptor fingerprint mismatch: declared {}, expected {expected}",
                self.descriptor_fingerprint
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossReference {
    pub spec: LossSpec,
    pub implementation: ImplementationDescriptor,
}

impl LossReference {
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        self.implementation.validate()?;
        validate_descriptor_semantic(
            &self.implementation,
            ImplementationSemanticKind::Loss,
            &self.spec.loss_id,
            &self.spec.spec_fingerprint,
        )?;
        for (required, provided, label) in [
            (
                self.spec
                    .capabilities
                    .contains(&LossCapability::Differentiable),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::Differentiable),
                "differentiable",
            ),
            (
                self.spec
                    .capabilities
                    .contains(&LossCapability::SupportsSampleWeights),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::SupportsSampleWeights),
                "supports_sample_weights",
            ),
            (
                self.spec
                    .capabilities
                    .contains(&LossCapability::SupportsMissingMask),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::SupportsMissingMask),
                "supports_missing_mask",
            ),
            (
                self.spec
                    .capabilities
                    .contains(&LossCapability::DistributedReduction),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::DistributedReduction),
                "distributed_reduction",
            ),
        ] {
            if required && !provided {
                return contract_error(format!(
                    "loss implementation lacks required `{label}` capability"
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricReference {
    pub spec: MetricSpec,
    pub implementation: ImplementationDescriptor,
}

impl MetricReference {
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        self.implementation.validate()?;
        validate_descriptor_semantic(
            &self.implementation,
            ImplementationSemanticKind::Metric,
            &self.spec.metric_id,
            &self.spec.spec_fingerprint,
        )?;
        for (required, provided, label) in [
            (
                self.spec
                    .capabilities
                    .contains(&MetricCapability::SupportsSampleWeights),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::SupportsSampleWeights),
                "supports_sample_weights",
            ),
            (
                self.spec
                    .capabilities
                    .contains(&MetricCapability::SupportsMissingMask),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::SupportsMissingMask),
                "supports_missing_mask",
            ),
            (
                self.spec
                    .capabilities
                    .contains(&MetricCapability::DistributedReduction),
                self.implementation
                    .capabilities
                    .contains(&ImplementationCapability::DistributedReduction),
                "distributed_reduction",
            ),
        ] {
            if required && !provided {
                return contract_error(format!(
                    "metric implementation lacks required `{label}` capability"
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingLossRoleReference {
    pub schema_version: u32,
    pub node_id: NodeId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_id: Option<String>,
    pub phases: BTreeSet<Phase>,
    pub loss: LossReference,
}

impl TrainingLossRoleReference {
    pub fn from_json(json: &str) -> Result<Self> {
        let role: Self = deserialize_external_contract(
            json,
            "training loss role",
            DagMlError::CampaignValidation,
        )?;
        role.validate()?;
        Ok(role)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version(
            "training loss role",
            self.schema_version,
            LOSS_ROLE_SCHEMA_VERSION,
        )?;
        if let Some(output_id) = &self.output_id {
            validate_token("loss output_id", output_id)?;
        }
        if self.phases.is_empty()
            || self
                .phases
                .iter()
                .any(|phase| !matches!(phase, Phase::FitCv | Phase::Refit))
        {
            return contract_error(
                "training loss phases must be a non-empty subset of FIT_CV/REFIT",
            );
        }
        self.loss.validate()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricRoleKind {
    EarlyStopping,
    Selection,
    Reporting,
    Tuning,
    Pruning,
    Threshold,
    EnsembleWeighting,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingMetricPolicy {
    Error,
    Skip,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricRoleReference {
    pub schema_version: u32,
    pub role_id: String,
    pub role: MetricRoleKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_id: Option<String>,
    pub partition: PredictionPartition,
    pub level: PredictionLevel,
    pub missing_value_policy: MissingMetricPolicy,
    pub metric: MetricReference,
}

impl MetricRoleReference {
    pub fn from_json(json: &str) -> Result<Self> {
        let role: Self =
            deserialize_external_contract(json, "metric role", DagMlError::CampaignValidation)?;
        role.validate()?;
        Ok(role)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version(
            "metric role",
            self.schema_version,
            METRIC_ROLE_SCHEMA_VERSION,
        )?;
        validate_token("metric role_id", &self.role_id)?;
        if let Some(output_id) = &self.output_id {
            validate_token("metric output_id", output_id)?;
        }
        if self.missing_value_policy == MissingMetricPolicy::Skip
            && self.role != MetricRoleKind::Reporting
        {
            return contract_error("only reporting metrics may skip missing values");
        }
        if !self.metric.spec.supported_levels.contains(&self.level) {
            return contract_error(format!(
                "metric role level {:?} is not supported by `{}`",
                self.level, self.metric.spec.metric_id
            ));
        }
        self.metric.validate()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossResolutionSource {
    ExplicitNodeOutput,
    ControllerProfile,
    CampaignDefault,
    TaskFamilyDefault,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossResolutionRequest {
    pub task_kind: LearningTaskKind,
    pub prediction_kind: PredictionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explicit_node_output: Option<LossSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controller_profile: Option<LossSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign_default: Option<LossSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_family_default: Option<LossSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedLossSpec {
    pub source: LossResolutionSource,
    pub spec: LossSpec,
}

impl LossResolutionRequest {
    pub fn resolve(&self) -> Result<ResolvedLossSpec> {
        let candidates = [
            (
                LossResolutionSource::ExplicitNodeOutput,
                self.explicit_node_output.as_ref(),
            ),
            (
                LossResolutionSource::ControllerProfile,
                self.controller_profile.as_ref(),
            ),
            (
                LossResolutionSource::CampaignDefault,
                self.campaign_default.as_ref(),
            ),
            (
                LossResolutionSource::TaskFamilyDefault,
                self.task_family_default.as_ref(),
            ),
        ];
        let Some((source, spec)) = candidates
            .into_iter()
            .find_map(|(source, spec)| spec.map(|spec| (source, spec)))
        else {
            return contract_error(format!(
                "no loss resolves for task {:?} and prediction {:?}",
                self.task_kind, self.prediction_kind
            ));
        };
        spec.validate_compatibility(self.task_kind, self.prediction_kind)?;
        Ok(ResolvedLossSpec {
            source,
            spec: spec.clone(),
        })
    }
}

pub fn builtin_loss_catalog() -> Result<BTreeMap<String, LossSpec>> {
    let target_prediction = BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]);
    let differentiable = BTreeSet::from([LossCapability::Differentiable]);
    let regression = LossSpec::new(
        "dagml.loss.mse@1",
        SemanticSpecKind::BuiltIn,
        BTreeSet::from([LearningTaskKind::Regression]),
        BTreeSet::from([PredictionKind::RegressionPoint]),
        LossReduction::Mean,
        target_prediction.clone(),
        differentiable.clone(),
        empty_parameters(),
    )?;
    let binary = LossSpec::new(
        "dagml.loss.binary_cross_entropy@1",
        SemanticSpecKind::BuiltIn,
        BTreeSet::from([
            LearningTaskKind::BinaryClassification,
            LearningTaskKind::MultilabelClassification,
        ]),
        BTreeSet::from([PredictionKind::ClassLabel, PredictionKind::ClassProbability]),
        LossReduction::Mean,
        target_prediction.clone(),
        differentiable.clone(),
        empty_parameters(),
    )?;
    let multiclass = LossSpec::new(
        "dagml.loss.sparse_categorical_cross_entropy@1",
        SemanticSpecKind::BuiltIn,
        BTreeSet::from([LearningTaskKind::MulticlassClassification]),
        BTreeSet::from([PredictionKind::ClassLabel, PredictionKind::ClassProbability]),
        LossReduction::Mean,
        target_prediction,
        differentiable,
        empty_parameters(),
    )?;
    Ok([regression, binary, multiclass]
        .into_iter()
        .map(|spec| (spec.loss_id.clone(), spec))
        .collect())
}

pub fn builtin_metric_catalog() -> Result<BTreeMap<String, MetricSpec>> {
    let all_levels = BTreeSet::from([
        PredictionLevel::Observation,
        PredictionLevel::Sample,
        PredictionLevel::Target,
        PredictionLevel::Group,
    ]);
    let target_prediction = BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]);
    let decomposable = BTreeSet::from([MetricCapability::Decomposable]);
    let mut specs = Vec::new();
    for (name, objective) in [
        ("mse", MetricObjective::Minimize),
        ("rmse", MetricObjective::Minimize),
        ("mae", MetricObjective::Minimize),
        ("r2", MetricObjective::Maximize),
    ] {
        specs.push(MetricSpec::new(
            format!("dagml.metric.{name}@1"),
            SemanticSpecKind::BuiltIn,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            objective,
            all_levels.clone(),
            MetricDecomposition::PerOutput,
            MetricReduction::Mean,
            target_prediction.clone(),
            decomposable.clone(),
            empty_parameters(),
        )?);
    }
    for (name, tasks) in [
        (
            "accuracy",
            BTreeSet::from([
                LearningTaskKind::BinaryClassification,
                LearningTaskKind::MulticlassClassification,
                LearningTaskKind::MultilabelClassification,
            ]),
        ),
        (
            "balanced_accuracy",
            BTreeSet::from([
                LearningTaskKind::BinaryClassification,
                LearningTaskKind::MulticlassClassification,
            ]),
        ),
    ] {
        specs.push(MetricSpec::new(
            format!("dagml.metric.{name}@1"),
            SemanticSpecKind::BuiltIn,
            tasks,
            BTreeSet::from([PredictionKind::ClassLabel]),
            MetricObjective::Maximize,
            all_levels.clone(),
            MetricDecomposition::PerOutput,
            MetricReduction::Mean,
            target_prediction.clone(),
            decomposable.clone(),
            empty_parameters(),
        )?);
    }
    Ok(specs
        .into_iter()
        .map(|spec| (spec.metric_id.clone(), spec))
        .collect())
}

fn empty_parameters() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

fn validate_schema_version(label: &str, actual: u32, expected: u32) -> Result<()> {
    if actual != expected {
        return contract_error(format!(
            "{label} schema_version {actual} is unsupported (expected {expected})"
        ));
    }
    Ok(())
}

fn validate_versioned_id(label: &str, value: &str) -> Result<()> {
    validate_token(&format!("{label}_id"), value)?;
    let Some((base, version)) = value.rsplit_once('@') else {
        return contract_error(format!("{label} id `{value}` is not versioned"));
    };
    if base.is_empty()
        || version.is_empty()
        || version.starts_with('0')
        || !version.bytes().all(|byte| byte.is_ascii_digit())
        || base.contains('@')
    {
        return contract_error(format!(
            "{label} id `{value}` must end in exactly one positive `@<version>` suffix"
        ));
    }
    Ok(())
}

pub(crate) fn validate_token(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        return contract_error(format!("{label} must be non-blank canonical text"));
    }
    Ok(())
}

fn validate_string_set(label: &str, values: &BTreeSet<String>) -> Result<()> {
    for value in values {
        validate_token(label, value)?;
    }
    Ok(())
}

fn validate_nonempty_set<T>(label: &str, values: &BTreeSet<T>) -> Result<()> {
    if values.is_empty() {
        return contract_error(format!("{label} must be non-empty"));
    }
    Ok(())
}

fn validate_required_target_prediction(
    label: &str,
    required_inputs: &BTreeSet<CriterionInput>,
) -> Result<()> {
    if !required_inputs.contains(&CriterionInput::Target)
        || !required_inputs.contains(&CriterionInput::Prediction)
    {
        return contract_error(format!("{label} requires target and prediction inputs"));
    }
    Ok(())
}

fn validate_parameters(label: &str, parameters: &serde_json::Value) -> Result<()> {
    if !parameters.is_object() {
        return contract_error(format!("{label} parameters must be a JSON object"));
    }
    validate_typed_serde_value(parameters).map_err(|error| {
        DagMlError::CampaignValidation(format!(
            "{label} parameters are outside strict TCV1: {error}"
        ))
    })?;
    validate_no_executable_payload(parameters, &format!("{label}.parameters"))
}

fn validate_no_executable_payload(value: &serde_json::Value, path: &str) -> Result<()> {
    match value {
        serde_json::Value::Object(entries) => {
            for (key, value) in entries {
                let normalized_key = key.to_ascii_lowercase();
                if FORBIDDEN_EXECUTABLE_KEYS.contains(&normalized_key.as_str()) {
                    return contract_error(format!(
                        "{path}.{key} is an executable-code payload field"
                    ));
                }
                validate_no_executable_payload(value, &format!("{path}.{key}"))?;
            }
        }
        serde_json::Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_no_executable_payload(value, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_descriptor_semantic(
    descriptor: &ImplementationDescriptor,
    expected_kind: ImplementationSemanticKind,
    expected_id: &str,
    expected_fingerprint: &str,
) -> Result<()> {
    if descriptor.semantic_kind != expected_kind
        || descriptor.semantic_id != expected_id
        || descriptor.semantic_fingerprint != expected_fingerprint
    {
        return contract_error(format!(
            "implementation descriptor semantic identity does not match {expected_kind:?} `{expected_id}`"
        ));
    }
    Ok(())
}

pub(crate) fn validate_fingerprint(label: &str, value: &str) -> Result<()> {
    if value.len() != FINGERPRINT_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return contract_error(format!("{label} fingerprint must be lowercase sha256 hex"));
    }
    Ok(())
}

pub(crate) fn fingerprint_without<T: Serialize>(
    value: &T,
    field: &str,
    label: &str,
) -> Result<String> {
    let json = serde_json::to_string(value)?;
    parse_typed_json(&json)
        .and_then(|value| value.fingerprint_without(field))
        .map_err(|error| {
            DagMlError::CampaignValidation(format!(
                "cannot compute {label} TCV1 fingerprint: {error}"
            ))
        })
}

fn contract_error<T>(message: impl Into<String>) -> Result<T> {
    Err(DagMlError::CampaignValidation(message.into()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn custom_loss() -> LossSpec {
        LossSpec::new(
            "example.loss.asymmetric@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            LossReduction::Mean,
            BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]),
            BTreeSet::from([LossCapability::Differentiable]),
            json!({"under_weight": 2.0, "over_weight": 1.0}),
        )
        .unwrap()
    }

    fn host_local_descriptor(
        kind: ImplementationSemanticKind,
        semantic_id: &str,
        semantic_fingerprint: &str,
    ) -> ImplementationDescriptor {
        ImplementationDescriptor::new(
            kind,
            semantic_id,
            semantic_fingerprint,
            "provider:python-local",
            "binding:python",
            "1.0.0",
            "1f4c71b0b758c5ed25b4e38b132b9ad56fb2f5ff2cf490f7eb8786c4350a62f7",
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::from([
                ImplementationCapability::Deterministic,
                ImplementationCapability::Differentiable,
                ImplementationCapability::NeedsGil,
            ]),
            PortabilityClass::HostLocal,
            ReplayabilityClass::RegistryRequired,
            Some("loss:run-123:asymmetric".to_string()),
        )
        .unwrap()
    }

    #[test]
    fn built_in_catalogs_are_versioned_and_self_fingerprinted() {
        let losses = builtin_loss_catalog().unwrap();
        assert_eq!(losses.len(), 3);
        assert!(losses.contains_key("dagml.loss.mse@1"));
        assert!(losses.values().all(|spec| spec.validate().is_ok()));

        let metrics = builtin_metric_catalog().unwrap();
        assert_eq!(metrics.len(), 6);
        assert_eq!(
            metrics["dagml.metric.rmse@1"].objective,
            MetricObjective::Minimize
        );
        assert_eq!(
            metrics["dagml.metric.balanced_accuracy@1"].objective,
            MetricObjective::Maximize
        );
        assert!(metrics.values().all(|spec| spec.validate().is_ok()));
    }

    #[test]
    fn loss_resolution_uses_declared_priority_and_never_falls_back_on_incompatibility() {
        let builtins = builtin_loss_catalog().unwrap();
        let explicit = custom_loss();
        let request = LossResolutionRequest {
            task_kind: LearningTaskKind::Regression,
            prediction_kind: PredictionKind::RegressionPoint,
            explicit_node_output: Some(explicit.clone()),
            controller_profile: Some(builtins["dagml.loss.mse@1"].clone()),
            campaign_default: None,
            task_family_default: None,
        };
        let resolved = request.resolve().unwrap();
        assert_eq!(resolved.source, LossResolutionSource::ExplicitNodeOutput);
        assert_eq!(resolved.spec, explicit);

        let mut incompatible = request;
        incompatible.task_kind = LearningTaskKind::BinaryClassification;
        let error = incompatible.resolve().unwrap_err().to_string();
        assert!(error.contains("not compatible"));
    }

    #[test]
    fn empty_unversioned_and_tampered_loss_specs_are_rejected() {
        let mut spec = custom_loss();
        spec.loss_id = "example.loss.asymmetric".to_string();
        assert!(spec
            .validate()
            .unwrap_err()
            .to_string()
            .contains("not versioned"));

        let mut spec = custom_loss();
        spec.task_kinds.clear();
        assert!(spec
            .validate()
            .unwrap_err()
            .to_string()
            .contains("non-empty"));

        let mut spec = custom_loss();
        spec.spec_fingerprint = "0".repeat(64);
        assert!(spec
            .validate()
            .unwrap_err()
            .to_string()
            .contains("fingerprint mismatch"));
    }

    #[test]
    fn versioned_ids_are_canonical_decimal_without_an_artificial_width_limit() {
        let mut leading_zero = custom_loss();
        leading_zero.loss_id = "example.loss.asymmetric@01".to_string();
        leading_zero.spec_fingerprint = leading_zero.compute_fingerprint().unwrap();
        assert!(leading_zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("positive `@<version>` suffix"));

        let mut large_version = custom_loss();
        large_version.loss_id = "example.loss.asymmetric@4294967296".to_string();
        large_version.spec_fingerprint = large_version.compute_fingerprint().unwrap();
        large_version.validate().unwrap();
    }

    #[test]
    fn executable_payloads_and_nfc_colliding_parameters_are_rejected() {
        let error = LossSpec::new(
            "example.loss.code@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            LossReduction::Mean,
            BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]),
            BTreeSet::new(),
            json!({"callable": "lambda y, p: 0"}),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("executable-code payload"));

        let mut values = serde_json::Map::new();
        values.insert("é".to_string(), json!(1));
        values.insert("e\u{301}".to_string(), json!(2));
        let error = LossSpec::new(
            "example.loss.nfc@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            LossReduction::Mean,
            BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]),
            BTreeSet::new(),
            serde_json::Value::Object(values),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("NFC-colliding"));
    }

    #[test]
    fn reduction_and_input_capabilities_are_consistent() {
        let error = LossSpec::new(
            "example.loss.weighted@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            LossReduction::WeightedMean,
            BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]),
            BTreeSet::new(),
            empty_parameters(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires sample_weight"));

        let error = MetricSpec::new(
            "example.metric.bad-decomposition@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            MetricObjective::Minimize,
            BTreeSet::from([PredictionLevel::Sample]),
            MetricDecomposition::PerUnit,
            MetricReduction::Global,
            BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]),
            BTreeSet::from([MetricCapability::Decomposable]),
            empty_parameters(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("cannot use global reduction"));

        let error = MetricSpec::new(
            "example.metric.bad-weighted-output@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            MetricObjective::Minimize,
            BTreeSet::from([PredictionLevel::Sample]),
            MetricDecomposition::PerOutput,
            MetricReduction::WeightedMean,
            BTreeSet::from([
                CriterionInput::Target,
                CriterionInput::Prediction,
                CriterionInput::SampleWeight,
            ]),
            BTreeSet::from([
                MetricCapability::Decomposable,
                MetricCapability::SupportsSampleWeights,
            ]),
            empty_parameters(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires per_unit decomposition"));
    }

    #[test]
    fn one_descriptor_contract_binds_loss_and_metric_without_merging_semantics() {
        let loss = custom_loss();
        let loss_reference = LossReference {
            implementation: host_local_descriptor(
                ImplementationSemanticKind::Loss,
                &loss.loss_id,
                &loss.spec_fingerprint,
            ),
            spec: loss,
        };
        loss_reference.validate().unwrap();

        let metric = MetricSpec::new(
            "example.metric.bias@1",
            SemanticSpecKind::Custom,
            BTreeSet::from([LearningTaskKind::Regression]),
            BTreeSet::from([PredictionKind::RegressionPoint]),
            MetricObjective::Minimize,
            BTreeSet::from([PredictionLevel::Sample]),
            MetricDecomposition::Global,
            MetricReduction::Global,
            BTreeSet::from([CriterionInput::Target, CriterionInput::Prediction]),
            BTreeSet::new(),
            empty_parameters(),
        )
        .unwrap();
        let metric_reference = MetricReference {
            implementation: host_local_descriptor(
                ImplementationSemanticKind::Metric,
                &metric.metric_id,
                &metric.spec_fingerprint,
            ),
            spec: metric,
        };
        metric_reference.validate().unwrap();
        assert_ne!(
            loss_reference.implementation.semantic_kind,
            metric_reference.implementation.semantic_kind
        );
    }

    #[test]
    fn local_descriptor_serializes_only_an_opaque_registry_key() {
        let loss = custom_loss();
        let descriptor = host_local_descriptor(
            ImplementationSemanticKind::Loss,
            &loss.loss_id,
            &loss.spec_fingerprint,
        );
        let serialized = serde_json::to_string(&descriptor).unwrap();
        assert!(serialized.contains("loss:run-123:asymmetric"));
        for forbidden in FORBIDDEN_EXECUTABLE_KEYS {
            assert!(!serialized.contains(&format!("\"{forbidden}\"")));
        }
    }

    #[test]
    fn semantic_reference_rejects_wrong_kind_or_fingerprint() {
        let loss = custom_loss();
        let mut descriptor = host_local_descriptor(
            ImplementationSemanticKind::Loss,
            &loss.loss_id,
            &loss.spec_fingerprint,
        );
        descriptor.semantic_kind = ImplementationSemanticKind::Metric;
        descriptor.descriptor_fingerprint = descriptor.compute_fingerprint().unwrap();
        let error = LossReference {
            spec: loss,
            implementation: descriptor,
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("semantic identity"));
    }

    #[test]
    fn roles_keep_training_loss_and_metric_policy_distinct() {
        let loss = custom_loss();
        let loss_role = TrainingLossRoleReference {
            schema_version: LOSS_ROLE_SCHEMA_VERSION,
            node_id: NodeId::new("model:custom").unwrap(),
            output_id: Some("prediction".to_string()),
            phases: BTreeSet::from([Phase::FitCv, Phase::Refit]),
            loss: LossReference {
                implementation: host_local_descriptor(
                    ImplementationSemanticKind::Loss,
                    &loss.loss_id,
                    &loss.spec_fingerprint,
                ),
                spec: loss,
            },
        };
        loss_role.validate().unwrap();

        let metric = builtin_metric_catalog().unwrap()["dagml.metric.rmse@1"].clone();
        let metric_role = MetricRoleReference {
            schema_version: METRIC_ROLE_SCHEMA_VERSION,
            role_id: "selection:rmse".to_string(),
            role: MetricRoleKind::Selection,
            output_id: Some("prediction".to_string()),
            partition: PredictionPartition::Validation,
            level: PredictionLevel::Sample,
            missing_value_policy: MissingMetricPolicy::Error,
            metric: MetricReference {
                implementation: ImplementationDescriptor::new(
                    ImplementationSemanticKind::Metric,
                    &metric.metric_id,
                    &metric.spec_fingerprint,
                    "provider:dag-ml-core",
                    "binding:rust",
                    "1.0.0",
                    "4991854599d650fd613dfd02b10d90a649ad7fec85f20a027d5e7b2a553f628b",
                    BTreeSet::new(),
                    BTreeSet::new(),
                    BTreeSet::from([ImplementationCapability::Deterministic]),
                    PortabilityClass::PortableBuiltIn,
                    ReplayabilityClass::Detached,
                    None,
                )
                .unwrap(),
                spec: metric,
            },
        };
        metric_role.validate().unwrap();

        let mut invalid_missing = metric_role;
        invalid_missing.missing_value_policy = MissingMetricPolicy::Skip;
        assert!(invalid_missing
            .validate()
            .unwrap_err()
            .to_string()
            .contains("only reporting"));
    }

    #[test]
    fn strict_json_rejects_duplicate_keys_before_deserialization() {
        let valid = serde_json::to_string(&custom_loss()).unwrap();
        LossSpec::from_json(&valid).unwrap();

        let duplicate = valid.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        );
        assert!(LossSpec::from_json(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate JSON object key"));
    }

    #[test]
    fn published_criteria_fixture_matches_rust_contracts_and_negative_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../examples/fixtures/criteria/criteria_contracts.v1.json"
        ))
        .unwrap();
        let valid = fixture["valid"].as_object().unwrap();

        let loss = LossSpec::from_json(&valid["loss_spec"].to_string()).unwrap();
        assert_eq!(
            loss.spec_fingerprint,
            "cf661225cc7137ab5ef9b87871ed5736a8479dd21587b7e17150c442b1e43eb0"
        );
        let metric = MetricSpec::from_json(&valid["metric_spec"].to_string()).unwrap();
        assert_eq!(
            metric.spec_fingerprint,
            "be54e6824479b10c398c229169bd324387c2fbb932e8fa879e37eba7d8821006"
        );
        for key in ["loss_implementation", "metric_implementation"] {
            ImplementationDescriptor::from_json(&valid[key].to_string()).unwrap();
        }
        TrainingLossRoleReference::from_json(&valid["training_loss_role"].to_string()).unwrap();
        MetricRoleReference::from_json(&valid["metric_role"].to_string()).unwrap();

        for case in fixture["invalid"].as_array().unwrap() {
            let document = case["document"].to_string();
            let result = match case["contract"].as_str().unwrap() {
                "loss_spec" => LossSpec::from_json(&document).map(|_| ()),
                "metric_spec" => MetricSpec::from_json(&document).map(|_| ()),
                "implementation_descriptor" => {
                    ImplementationDescriptor::from_json(&document).map(|_| ())
                }
                "training_loss_role" => TrainingLossRoleReference::from_json(&document).map(|_| ()),
                "metric_role" => MetricRoleReference::from_json(&document).map(|_| ()),
                contract => panic!("unknown criteria fixture contract `{contract}`"),
            };
            assert!(
                result.is_err(),
                "negative case `{}` was accepted",
                case["id"]
            );
        }
    }
}
