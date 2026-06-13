use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::{ControllerId, FoldId, NodeId, RunId, VariantId};
use crate::phase::Phase;
use crate::relation::SampleRelationSet;
use crate::runtime::{
    DataMaterializationRequest, DataProviderViewSpec, DataViewRequest, HandleKind, HandleRef,
    RuntimeDataProvider,
};

pub const EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION: u32 = 1;
pub const MODEL_INPUT_SPEC_SCHEMA_VERSION: u32 = 1;
pub const MODEL_INPUT_SPEC_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/model_input_spec.v1.schema.json";
pub const DATA_PLAN_SCHEMA_VERSION: u32 = 1;
pub const DATA_PLAN_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/data_plan.v1.schema.json";

fn default_external_data_plan_envelope_schema_version() -> u32 {
    EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION
}

fn default_model_input_spec_schema_version() -> u32 {
    MODEL_INPUT_SPEC_SCHEMA_VERSION
}

fn default_data_plan_schema_version() -> u32 {
    DATA_PLAN_SCHEMA_VERSION
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRequestPartition {
    FoldTrain,
    FoldValidation,
    FullTrain,
    Predict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInputFusionMode {
    SingleSource,
    ConcatenateFeatures,
    StackSamples,
    DictBySource,
    Custom,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchViewMode {
    Separation,
    BySource,
    ByMetadata,
    ByTag,
    ByFilter,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DataViewSelector {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<serde_json::Value>,
}

impl DataViewSelector {
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.source_ids.is_empty()
            && self.metadata.is_empty()
            && self.tags.is_empty()
            && self.filter.is_none()
        {
            return Err(DagMlError::CampaignValidation(format!(
                "{label} selector must constrain source_ids, metadata, tags or filter"
            )));
        }
        validate_string_list_entries(&format!("{label} selector source_ids"), &self.source_ids)?;
        validate_unique_strings(&format!("{label} selector source_ids"), &self.source_ids)?;
        validate_string_list_entries(&format!("{label} selector tags"), &self.tags)?;
        validate_unique_strings(&format!("{label} selector tags"), &self.tags)?;
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "{label} selector contains an empty metadata key"
                )));
            }
        }
        if matches!(self.filter, Some(serde_json::Value::Null)) {
            return Err(DagMlError::CampaignValidation(format!(
                "{label} selector filter must not be null"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BranchViewPlan {
    pub view_id: String,
    pub branch_id: String,
    pub mode: BranchViewMode,
    pub selector: DataViewSelector,
    #[serde(default)]
    pub allow_overlap: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl BranchViewPlan {
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("branch view plan view_id", &self.view_id)?;
        validate_non_empty("branch view plan branch_id", &self.branch_id)?;
        self.selector
            .validate(&format!("branch view `{}`", self.view_id))?;
        match self.mode {
            BranchViewMode::BySource if self.selector.source_ids.is_empty() => {
                return Err(DagMlError::CampaignValidation(format!(
                    "branch view `{}` mode=by_source requires source_ids",
                    self.view_id
                )));
            }
            BranchViewMode::ByMetadata if self.selector.metadata.is_empty() => {
                return Err(DagMlError::CampaignValidation(format!(
                    "branch view `{}` mode=by_metadata requires metadata",
                    self.view_id
                )));
            }
            BranchViewMode::ByTag if self.selector.tags.is_empty() => {
                return Err(DagMlError::CampaignValidation(format!(
                    "branch view `{}` mode=by_tag requires tags",
                    self.view_id
                )));
            }
            BranchViewMode::ByFilter if self.selector.filter.is_none() => {
                return Err(DagMlError::CampaignValidation(format!(
                    "branch view `{}` mode=by_filter requires filter",
                    self.view_id
                )));
            }
            _ => {}
        }
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "branch view `{}` metadata contains an empty key",
                    self.view_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInputFusionPolicy {
    pub mode: ModelInputFusionMode,
    #[serde(default)]
    pub alignment: Option<String>,
    #[serde(default)]
    pub adapter_id: Option<String>,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

impl ModelInputFusionPolicy {
    pub fn validate(&self) -> Result<()> {
        if self
            .alignment
            .as_ref()
            .is_some_and(|alignment| alignment.trim().is_empty())
        {
            return Err(DagMlError::CampaignValidation(
                "model input fusion policy has empty alignment".to_string(),
            ));
        }
        if self
            .adapter_id
            .as_ref()
            .is_some_and(|adapter_id| adapter_id.trim().is_empty())
        {
            return Err(DagMlError::CampaignValidation(
                "model input fusion policy has empty adapter_id".to_string(),
            ));
        }
        if self.mode == ModelInputFusionMode::Custom && self.adapter_id.is_none() {
            return Err(DagMlError::CampaignValidation(
                "custom model input fusion policy requires adapter_id".to_string(),
            ));
        }
        for key in self.params.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(
                    "model input fusion policy contains an empty param key".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInputPortSpec {
    pub name: String,
    pub accepted_representations: Vec<String>,
    pub accepted_types: Vec<String>,
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    pub multi_source: bool,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl ModelInputPortSpec {
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("model input port name", &self.name)?;
        validate_non_empty_list(
            "model input port accepted_representations",
            &self.accepted_representations,
        )?;
        validate_non_empty_list("model input port accepted_types", &self.accepted_types)?;
        validate_unique_strings(
            "model input port accepted_representations",
            &self.accepted_representations,
        )?;
        validate_unique_strings("model input port accepted_types", &self.accepted_types)?;
        if self.rank.is_some_and(|rank| rank > 16) {
            return Err(DagMlError::CampaignValidation(format!(
                "model input port `{}` rank must be <= 16",
                self.name
            )));
        }
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "model input port `{}` contains an empty metadata key",
                    self.name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInputSpec {
    #[serde(default = "default_model_input_spec_schema_version")]
    pub schema_version: u32,
    pub ports: Vec<ModelInputPortSpec>,
    #[serde(default)]
    pub default_fusion: Option<ModelInputFusionPolicy>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl ModelInputSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MODEL_INPUT_SPEC_SCHEMA_VERSION {
            return Err(DagMlError::CampaignValidation(format!(
                "model input spec uses unsupported schema_version {}, expected {}",
                self.schema_version, MODEL_INPUT_SPEC_SCHEMA_VERSION
            )));
        }
        if self.ports.is_empty() {
            return Err(DagMlError::CampaignValidation(
                "model input spec must declare at least one port".to_string(),
            ));
        }
        let mut names = BTreeSet::new();
        for port in &self.ports {
            port.validate()?;
            if !names.insert(port.name.as_str()) {
                return Err(DagMlError::CampaignValidation(format!(
                    "model input spec contains duplicate port `{}`",
                    port.name
                )));
            }
        }
        if let Some(default_fusion) = &self.default_fusion {
            default_fusion.validate()?;
        }
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(
                    "model input spec contains an empty metadata key".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataPlanStepKind {
    Materialize,
    Adapt,
    Align,
    Join,
    Collate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataPlanStep {
    pub kind: DataPlanStepKind,
    #[serde(default)]
    pub inputs: Vec<String>,
    pub output: String,
    #[serde(default)]
    pub adapter_id: Option<String>,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

impl DataPlanStep {
    pub fn validate(&self, previous_outputs: &BTreeSet<String>) -> Result<()> {
        validate_non_empty("data plan step output", &self.output)?;
        if self.kind != DataPlanStepKind::Materialize && self.inputs.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "data plan step `{}` requires at least one input",
                self.output
            )));
        }
        for (index, input) in self.inputs.iter().enumerate() {
            validate_non_empty("data plan step input", input)?;
            if self.kind != DataPlanStepKind::Materialize && !previous_outputs.contains(input) {
                return Err(DagMlError::CampaignValidation(format!(
                    "data plan step `{}` input #{index} references `{input}` before it is produced",
                    self.output
                )));
            }
        }
        if self
            .adapter_id
            .as_ref()
            .is_some_and(|adapter_id| adapter_id.trim().is_empty())
        {
            return Err(DagMlError::CampaignValidation(format!(
                "data plan step `{}` has empty adapter_id",
                self.output
            )));
        }
        for key in self.params.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "data plan step `{}` contains an empty param key",
                    self.output
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataPlan {
    #[serde(default = "default_data_plan_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub steps: Vec<DataPlanStep>,
    pub output_ports: BTreeMap<String, String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub requires_user_choice: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl DataPlan {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != DATA_PLAN_SCHEMA_VERSION {
            return Err(DagMlError::CampaignValidation(format!(
                "data plan uses unsupported schema_version {}, expected {}",
                self.schema_version, DATA_PLAN_SCHEMA_VERSION
            )));
        }
        validate_non_empty("data plan id", &self.id)?;
        if self.steps.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "data plan `{}` must contain at least one step",
                self.id
            )));
        }
        let mut outputs = BTreeSet::new();
        for step in &self.steps {
            step.validate(&outputs)?;
            if !outputs.insert(step.output.clone()) {
                return Err(DagMlError::CampaignValidation(format!(
                    "data plan `{}` contains duplicate step output `{}`",
                    self.id, step.output
                )));
            }
        }
        if self.output_ports.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "data plan `{}` must declare at least one output port",
                self.id
            )));
        }
        for (port_name, output) in &self.output_ports {
            validate_non_empty("data plan output port", port_name)?;
            validate_non_empty("data plan output reference", output)?;
            if !outputs.contains(output) {
                return Err(DagMlError::CampaignValidation(format!(
                    "data plan `{}` output port `{port_name}` references unknown output `{output}`",
                    self.id
                )));
            }
        }
        validate_string_list_entries("data plan warnings", &self.warnings)?;
        validate_string_list_entries("data plan requires_user_choice", &self.requires_user_choice)?;
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "data plan `{}` contains an empty metadata key",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DataViewPolicy {
    #[serde(default = "default_fit_partition")]
    pub fit_partition: DataRequestPartition,
    #[serde(default = "default_predict_partition")]
    pub predict_partition: DataRequestPartition,
    #[serde(default)]
    pub include_augmented_train: bool,
    #[serde(default)]
    pub include_augmented_validation: bool,
    #[serde(default)]
    pub include_excluded: bool,
    #[serde(default = "default_true")]
    pub require_sample_ids: bool,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub unsafe_flags: BTreeSet<String>,
}

impl Default for DataViewPolicy {
    fn default() -> Self {
        Self {
            fit_partition: DataRequestPartition::FoldTrain,
            predict_partition: DataRequestPartition::FoldValidation,
            include_augmented_train: true,
            include_augmented_validation: false,
            include_excluded: false,
            require_sample_ids: true,
            unsafe_flags: BTreeSet::new(),
        }
    }
}

impl DataViewPolicy {
    pub const ALLOW_FIT_CV_FULL_TRAIN_VIEW: &'static str = "allow_fit_cv_full_train_view";
    pub const ALLOW_FIT_CV_VALIDATION_VIEW: &'static str = "allow_fit_cv_validation_view";
    pub const ALLOW_AUGMENTED_VALIDATION_VIEW: &'static str = "allow_augmented_validation_view";
    pub const ALLOW_EXCLUDED_ROWS: &'static str = "allow_excluded_rows";

    pub fn validate(&self) -> Result<()> {
        for unsafe_flag in &self.unsafe_flags {
            if unsafe_flag.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(
                    "data view policy contains an empty unsafe flag".to_string(),
                ));
            }
        }
        match self.fit_partition {
            DataRequestPartition::FoldTrain => {}
            DataRequestPartition::FullTrain
                if self
                    .unsafe_flags
                    .contains(Self::ALLOW_FIT_CV_FULL_TRAIN_VIEW) => {}
            DataRequestPartition::FoldValidation
                if self
                    .unsafe_flags
                    .contains(Self::ALLOW_FIT_CV_VALIDATION_VIEW) => {}
            DataRequestPartition::FullTrain => {
                return Err(DagMlError::CampaignValidation(
                    "data view policy fit_partition=full_train would leak validation rows during FIT_CV; add explicit unsafe flag allow_fit_cv_full_train_view".to_string(),
                ));
            }
            DataRequestPartition::FoldValidation => {
                return Err(DagMlError::CampaignValidation(
                    "data view policy fit_partition=fold_validation would train on validation rows during FIT_CV; add explicit unsafe flag allow_fit_cv_validation_view".to_string(),
                ));
            }
            DataRequestPartition::Predict => {
                return Err(DagMlError::CampaignValidation(
                    "data view policy fit_partition=predict is not valid for FIT_CV".to_string(),
                ));
            }
        }
        match self.predict_partition {
            DataRequestPartition::FoldValidation | DataRequestPartition::Predict => {}
            DataRequestPartition::FoldTrain | DataRequestPartition::FullTrain => {
                return Err(DagMlError::CampaignValidation(format!(
                    "data view policy predict_partition={:?} is not valid for validation/predict views",
                    self.predict_partition
                )));
            }
        }
        if self.include_augmented_validation
            && !self
                .unsafe_flags
                .contains(Self::ALLOW_AUGMENTED_VALIDATION_VIEW)
        {
            return Err(DagMlError::CampaignValidation(
                "data view policy include_augmented_validation=true can leak augmented validation/test rows; add explicit unsafe flag allow_augmented_validation_view".to_string(),
            ));
        }
        if self.include_excluded && !self.unsafe_flags.contains(Self::ALLOW_EXCLUDED_ROWS) {
            return Err(DagMlError::CampaignValidation(
                "data view policy include_excluded=true requires explicit unsafe flag allow_excluded_rows".to_string(),
            ));
        }
        Ok(())
    }
}

fn default_fit_partition() -> DataRequestPartition {
    DataRequestPartition::FoldTrain
}

fn default_predict_partition() -> DataRequestPartition {
    DataRequestPartition::FoldValidation
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DataBinding {
    pub node_id: NodeId,
    pub input_name: String,
    pub request_id: String,
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    pub output_representation: String,
    #[serde(default)]
    pub feature_set_id: Option<String>,
    #[serde(default)]
    pub source_ids: Vec<String>,
    #[serde(default)]
    pub require_relations: bool,
    #[serde(default)]
    pub view_policy: DataViewPolicy,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl DataBinding {
    pub fn validate(&self) -> Result<()> {
        self.view_policy.validate()?;
        if self.input_name.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding for `{}` has empty input_name",
                self.node_id
            )));
        }
        if self.request_id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` has empty request_id",
                self.input_name, self.node_id
            )));
        }
        validate_fingerprint("schema", &self.schema_fingerprint)?;
        validate_fingerprint("plan", &self.plan_fingerprint)?;
        if let Some(relation_fingerprint) = &self.relation_fingerprint {
            validate_fingerprint("relation", relation_fingerprint)?;
        } else if self.require_relations {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` requires relations but has no relation_fingerprint",
                self.input_name, self.node_id
            )));
        }
        if self.output_representation.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` has empty output_representation",
                self.input_name, self.node_id
            )));
        }
        if let Some(feature_set_id) = &self.feature_set_id {
            if feature_set_id.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "data binding `{}` on `{}` has empty feature_set_id",
                    self.input_name, self.node_id
                )));
            }
        }
        for source_id in &self.source_ids {
            if source_id.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "data binding `{}` on `{}` has empty source id",
                    self.input_name, self.node_id
                )));
            }
        }
        Ok(())
    }

    pub fn feature_set_id(&self) -> &str {
        self.feature_set_id.as_deref().unwrap_or(&self.input_name)
    }

    pub fn validate_envelope(&self, envelope: &ExternalDataPlanEnvelope) -> Result<()> {
        self.validate()?;
        envelope.validate()?;
        if self.schema_fingerprint != envelope.schema_fingerprint {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` schema fingerprint mismatch",
                self.input_name, self.node_id
            )));
        }
        if self.plan_fingerprint != envelope.plan_fingerprint {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` plan fingerprint mismatch",
                self.input_name, self.node_id
            )));
        }
        if self.relation_fingerprint != envelope.relation_fingerprint {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` relation fingerprint mismatch",
                self.input_name, self.node_id
            )));
        }
        if self.require_relations && envelope.coordinator_relations.is_none() {
            return Err(DagMlError::CampaignValidation(format!(
                "data binding `{}` on `{}` requires coordinator relations",
                self.input_name, self.node_id
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalDataPlanEnvelope {
    #[serde(default = "default_external_data_plan_envelope_schema_version")]
    pub schema_version: u32,
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    #[serde(default)]
    pub coordinator_relations: Option<SampleRelationSet>,
}

impl ExternalDataPlanEnvelope {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION {
            return Err(DagMlError::CampaignValidation(format!(
                "external data-plan envelope uses unsupported schema_version {}, expected {}",
                self.schema_version, EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION
            )));
        }
        validate_fingerprint("schema", &self.schema_fingerprint)?;
        validate_fingerprint("plan", &self.plan_fingerprint)?;
        if let Some(relation_fingerprint) = &self.relation_fingerprint {
            validate_fingerprint("relation", relation_fingerprint)?;
            if self.coordinator_relations.is_none() {
                return Err(DagMlError::CampaignValidation(
                    "relation_fingerprint requires coordinator_relations".to_string(),
                ));
            }
        }
        if let Some(relations) = &self.coordinator_relations {
            relations.validate()?;
        }
        Ok(())
    }
}

pub fn validate_data_binding_envelope(
    binding: &DataBinding,
    envelope: &ExternalDataPlanEnvelope,
) -> Result<()> {
    binding.validate_envelope(envelope)
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
struct DataEnvelopeKey {
    schema_fingerprint: String,
    plan_fingerprint: String,
    relation_fingerprint: Option<String>,
}

impl DataEnvelopeKey {
    fn from_binding(binding: &DataBinding) -> Self {
        Self {
            schema_fingerprint: binding.schema_fingerprint.clone(),
            plan_fingerprint: binding.plan_fingerprint.clone(),
            relation_fingerprint: binding.relation_fingerprint.clone(),
        }
    }

    fn from_envelope(envelope: &ExternalDataPlanEnvelope) -> Self {
        Self {
            schema_fingerprint: envelope.schema_fingerprint.clone(),
            plan_fingerprint: envelope.plan_fingerprint.clone(),
            relation_fingerprint: envelope.relation_fingerprint.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DataHandleRecord {
    pub handle: HandleRef,
    pub run_id: RunId,
    pub node_id: NodeId,
    pub input_name: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    pub request_id: String,
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    pub relation_fingerprint: Option<String>,
    pub output_representation: String,
    #[serde(default)]
    pub feature_set_id: Option<String>,
    #[serde(default)]
    pub source_ids: Vec<String>,
    pub relation_record_count: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataViewHandleRecord {
    pub handle: HandleRef,
    pub parent_handle: HandleRef,
    pub run_id: RunId,
    pub node_id: NodeId,
    pub input_name: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    pub request_id: String,
    pub feature_set_id: String,
    pub view: DataProviderViewSpec,
}

#[derive(Debug)]
pub struct InMemoryDataProvider {
    owner_controller: ControllerId,
    envelopes: BTreeMap<DataEnvelopeKey, ExternalDataPlanEnvelope>,
    next_handle: RefCell<u64>,
    records: RefCell<BTreeMap<u64, DataHandleRecord>>,
    view_records: RefCell<BTreeMap<u64, DataViewHandleRecord>>,
}

impl InMemoryDataProvider {
    pub fn new(owner_controller: ControllerId) -> Self {
        Self {
            owner_controller,
            envelopes: BTreeMap::new(),
            next_handle: RefCell::new(1),
            records: RefCell::new(BTreeMap::new()),
            view_records: RefCell::new(BTreeMap::new()),
        }
    }

    pub fn with_envelope(
        owner_controller: ControllerId,
        envelope: ExternalDataPlanEnvelope,
    ) -> Result<Self> {
        let mut provider = Self::new(owner_controller);
        provider.register_envelope(envelope)?;
        Ok(provider)
    }

    pub fn register_envelope(&mut self, envelope: ExternalDataPlanEnvelope) -> Result<()> {
        envelope.validate()?;
        let key = DataEnvelopeKey::from_envelope(&envelope);
        if let Some(existing) = self.envelopes.get(&key) {
            if existing == &envelope {
                return Ok(());
            }
            return Err(DagMlError::RuntimeValidation(
                "duplicate external data-plan envelope with different payload".to_string(),
            ));
        }
        self.envelopes.insert(key, envelope);
        Ok(())
    }

    pub fn handle_record(&self, handle: u64) -> Option<DataHandleRecord> {
        self.records.borrow().get(&handle).cloned()
    }

    pub fn handle_records(&self) -> Vec<DataHandleRecord> {
        self.records.borrow().values().cloned().collect()
    }

    pub fn view_record(&self, handle: u64) -> Option<DataViewHandleRecord> {
        self.view_records.borrow().get(&handle).cloned()
    }

    pub fn view_records(&self) -> Vec<DataViewHandleRecord> {
        self.view_records.borrow().values().cloned().collect()
    }

    fn next_handle(&self) -> u64 {
        let mut next = self.next_handle.borrow_mut();
        let handle = *next;
        *next += 1;
        handle
    }
}

impl RuntimeDataProvider for InMemoryDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> Result<HandleRef> {
        if request.node_id != request.binding.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "data materialization request node `{}` does not match binding node `{}`",
                request.node_id, request.binding.node_id
            )));
        }
        if request.input_name != request.binding.input_name {
            return Err(DagMlError::RuntimeValidation(format!(
                "data materialization request input `{}` does not match binding input `{}`",
                request.input_name, request.binding.input_name
            )));
        }
        let envelope = self
            .envelopes
            .get(&DataEnvelopeKey::from_binding(&request.binding))
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "no external data-plan envelope registered for binding `{}` on `{}`",
                    request.binding.input_name, request.binding.node_id
                ))
            })?;
        request.binding.validate_envelope(envelope)?;

        let handle = HandleRef {
            handle: self.next_handle(),
            kind: HandleKind::Data,
            owner_controller: self.owner_controller.clone(),
        };
        let record = DataHandleRecord {
            handle: handle.clone(),
            run_id: request.run_id.clone(),
            node_id: request.node_id.clone(),
            input_name: request.input_name.clone(),
            phase: request.phase,
            variant_id: request.variant_id.clone(),
            fold_id: request.fold_id.clone(),
            request_id: request.binding.request_id.clone(),
            schema_fingerprint: request.binding.schema_fingerprint.clone(),
            plan_fingerprint: request.binding.plan_fingerprint.clone(),
            relation_fingerprint: request.binding.relation_fingerprint.clone(),
            output_representation: request.binding.output_representation.clone(),
            feature_set_id: request.binding.feature_set_id.clone(),
            source_ids: request.binding.source_ids.clone(),
            relation_record_count: envelope
                .coordinator_relations
                .as_ref()
                .map(|relations| relations.records.len()),
        };
        self.records.borrow_mut().insert(handle.handle, record);
        Ok(handle)
    }

    fn make_view(&self, request: &DataViewRequest) -> Result<HandleRef> {
        request.view.validate()?;
        if request.node_id != request.binding.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "data view request node `{}` does not match binding node `{}`",
                request.node_id, request.binding.node_id
            )));
        }
        if request.input_name != request.binding.input_name {
            return Err(DagMlError::RuntimeValidation(format!(
                "data view request input `{}` does not match binding input `{}`",
                request.input_name, request.binding.input_name
            )));
        }
        if request.data_handle.kind != HandleKind::Data {
            return Err(DagMlError::RuntimeValidation(format!(
                "data view request for `{}` on `{}` received non-data parent handle",
                request.input_name, request.node_id
            )));
        }
        let parent = self
            .records
            .borrow()
            .get(&request.data_handle.handle)
            .cloned()
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "unknown data handle `{}` for view request `{}` on `{}`",
                    request.data_handle.handle, request.input_name, request.node_id
                ))
            })?;
        if parent.handle != request.data_handle {
            return Err(DagMlError::RuntimeValidation(format!(
                "data view request parent handle `{}` does not match provider record",
                request.data_handle.handle
            )));
        }
        request.binding.validate()?;
        let handle = HandleRef {
            handle: self.next_handle(),
            kind: HandleKind::DataView,
            owner_controller: self.owner_controller.clone(),
        };
        let record = DataViewHandleRecord {
            handle: handle.clone(),
            parent_handle: request.data_handle.clone(),
            run_id: request.run_id.clone(),
            node_id: request.node_id.clone(),
            input_name: request.input_name.clone(),
            phase: request.phase,
            variant_id: request.variant_id.clone(),
            fold_id: request.fold_id.clone(),
            request_id: request.binding.request_id.clone(),
            feature_set_id: request.binding.feature_set_id().to_string(),
            view: request.view.clone(),
        };
        self.view_records.borrow_mut().insert(handle.handle, record);
        Ok(handle)
    }

    fn coordinator_relations(&self, binding: &DataBinding) -> Result<Option<SampleRelationSet>> {
        let envelope = self
            .envelopes
            .get(&DataEnvelopeKey::from_binding(binding))
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "no external data-plan envelope registered for binding `{}` on `{}`",
                    binding.input_name, binding.node_id
                ))
            })?;
        binding.validate_envelope(envelope)?;
        Ok(envelope.coordinator_relations.clone())
    }
}

fn validate_fingerprint(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::CampaignValidation(format!(
            "{label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

fn validate_non_empty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(DagMlError::CampaignValidation(format!(
            "{label} must be a non-empty string"
        )));
    }
    Ok(())
}

fn validate_non_empty_list(label: &str, values: &[String]) -> Result<()> {
    if values.is_empty() {
        return Err(DagMlError::CampaignValidation(format!(
            "{label} must be a non-empty list"
        )));
    }
    validate_string_list_entries(label, values)
}

fn validate_string_list_entries(label: &str, values: &[String]) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if value.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "{label}[{index}] must be a non-empty string"
            )));
        }
    }
    Ok(())
}

fn validate_unique_strings(label: &str, values: &[String]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value.as_str()) {
            return Err(DagMlError::CampaignValidation(format!(
                "{label} contains duplicate value `{value}`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::NodeId;
    use crate::runtime::DataMaterializationRequest;

    fn binding() -> DataBinding {
        DataBinding {
            node_id: NodeId::new("model:base").unwrap(),
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
            view_policy: DataViewPolicy::default(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn validates_data_binding_contract() {
        let binding = binding();
        binding.validate().unwrap();
        assert_eq!(binding.feature_set_id(), "x");
    }

    #[test]
    fn published_model_input_and_data_plan_schemas_declare_current_contract() {
        let model_input_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/model_input_spec.schema.json"
        ))
        .unwrap();
        assert_eq!(model_input_schema["$id"], MODEL_INPUT_SPEC_SCHEMA_ID);
        assert_eq!(
            model_input_schema["properties"]["schema_version"]["const"].as_u64(),
            Some(MODEL_INPUT_SPEC_SCHEMA_VERSION as u64)
        );
        assert!(model_input_schema["$defs"]["input_port"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("accepted_representations")));

        let data_plan_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/data_plan.schema.json"
        ))
        .unwrap();
        assert_eq!(data_plan_schema["$id"], DATA_PLAN_SCHEMA_ID);
        assert_eq!(
            data_plan_schema["properties"]["schema_version"]["const"].as_u64(),
            Some(DATA_PLAN_SCHEMA_VERSION as u64)
        );
        assert!(data_plan_schema["$defs"]["data_plan_step_kind"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kind| kind.as_str() == Some("collate")));
    }

    #[test]
    fn validates_model_input_and_data_plan_fixtures() {
        let model_input: ModelInputSpec = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/model_input_spec_tabular_regressor.json"
        ))
        .unwrap();
        model_input.validate().unwrap();
        assert_eq!(model_input.ports[0].rank, Some(2));
        assert!(model_input.ports[0].multi_source);

        let data_plan: DataPlan = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/data_plan_tabular_fusion.json"
        ))
        .unwrap();
        data_plan.validate().unwrap();
        assert_eq!(data_plan.output_ports.get("x").unwrap(), "x_collated");
    }

    #[test]
    fn data_plan_rejects_forward_step_references() {
        let data_plan = DataPlan {
            schema_version: DATA_PLAN_SCHEMA_VERSION,
            id: "data-plan:bad".to_string(),
            steps: vec![DataPlanStep {
                kind: DataPlanStepKind::Adapt,
                inputs: vec!["missing".to_string()],
                output: "adapted".to_string(),
                adapter_id: Some("adapter:adapt".to_string()),
                params: BTreeMap::new(),
            }],
            output_ports: BTreeMap::from([("x".to_string(), "adapted".to_string())]),
            warnings: Vec::new(),
            requires_user_choice: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let error = data_plan.validate().unwrap_err().to_string();
        assert!(error.contains("before it is produced"));
    }

    #[test]
    fn data_view_policy_rejects_unsafe_fit_and_validation_augmentation_by_default() {
        let mut full_train_binding = binding();
        full_train_binding.view_policy.fit_partition = DataRequestPartition::FullTrain;
        let full_train_error = full_train_binding.validate().unwrap_err().to_string();
        assert!(
            full_train_error.contains("fit_partition=full_train"),
            "unexpected full-train error: {full_train_error}"
        );

        let mut augmented_validation_binding = binding();
        augmented_validation_binding
            .view_policy
            .include_augmented_validation = true;
        let augmented_error = augmented_validation_binding
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            augmented_error.contains("include_augmented_validation=true"),
            "unexpected augmented-validation error: {augmented_error}"
        );

        let mut excluded_binding = binding();
        excluded_binding.view_policy.include_excluded = true;
        let excluded_error = excluded_binding.validate().unwrap_err().to_string();
        assert!(
            excluded_error.contains("include_excluded=true"),
            "unexpected excluded-row error: {excluded_error}"
        );
    }

    #[test]
    fn data_view_policy_requires_explicit_unsafe_flags_for_debug_views() {
        let mut binding = binding();
        binding.view_policy.fit_partition = DataRequestPartition::FullTrain;
        binding.view_policy.include_augmented_validation = true;
        binding.view_policy.include_excluded = true;
        binding.view_policy.unsafe_flags = BTreeSet::from([
            DataViewPolicy::ALLOW_FIT_CV_FULL_TRAIN_VIEW.to_string(),
            DataViewPolicy::ALLOW_AUGMENTED_VALIDATION_VIEW.to_string(),
            DataViewPolicy::ALLOW_EXCLUDED_ROWS.to_string(),
        ]);

        binding.validate().unwrap();
    }

    #[test]
    fn validates_external_data_envelope_subset() {
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();

        assert_eq!(
            envelope.schema_version,
            EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION
        );
        binding().validate_envelope(&envelope).unwrap();
    }

    #[test]
    fn validates_multisource_repetition_envelope_fixture() {
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_multisource_repetitions.json"
        ))
        .unwrap();

        envelope.validate().unwrap();
        let relations = envelope.coordinator_relations.as_ref().unwrap();
        assert_eq!(relations.records.len(), 8);
        assert_eq!(
            relations
                .sample_for_observation(
                    &crate::ids::ObservationId::new("obs.s1.combo.A0.B0.C0").unwrap()
                )
                .unwrap()
                .as_str(),
            "sample:1"
        );
    }

    #[test]
    fn published_external_data_envelope_schema_declares_current_version() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/coordinator_data_plan_envelope.schema.json"
        ))
        .unwrap();

        assert_eq!(
            schema["properties"]["schema_version"]["const"].as_u64(),
            Some(EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION as u64)
        );
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("schema_version")));
    }

    #[test]
    fn refuses_unsupported_external_data_envelope_schema_version() {
        let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        envelope.schema_version = EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION + 1;

        assert!(binding().validate_envelope(&envelope).is_err());
    }

    #[test]
    fn refuses_envelope_fingerprint_mismatch() {
        let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        envelope.plan_fingerprint = "0".repeat(64);

        assert!(binding().validate_envelope(&envelope).is_err());
    }

    #[test]
    fn in_memory_provider_materializes_validated_data_handles() {
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        let provider = InMemoryDataProvider::with_envelope(
            ControllerId::new("controller:data.provider").unwrap(),
            envelope,
        )
        .unwrap();

        let handle = provider
            .materialize(&DataMaterializationRequest {
                run_id: RunId::new("run:data").unwrap(),
                node_id: NodeId::new("model:base").unwrap(),
                input_name: "x".to_string(),
                phase: Phase::FitCv,
                variant_id: Some(VariantId::new("variant:base").unwrap()),
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                binding: binding(),
            })
            .unwrap();

        let record = provider.handle_record(handle.handle).unwrap();
        assert_eq!(record.input_name, "x");
        assert_eq!(record.relation_record_count, Some(4));
        assert_eq!(provider.handle_records().len(), 1);
    }

    #[test]
    fn in_memory_provider_registration_is_idempotent_for_same_envelope() {
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        let mut provider =
            InMemoryDataProvider::new(ControllerId::new("controller:data.provider").unwrap());

        provider.register_envelope(envelope.clone()).unwrap();
        provider.register_envelope(envelope).unwrap();
    }

    #[test]
    fn in_memory_provider_refuses_unknown_envelope() {
        let provider =
            InMemoryDataProvider::new(ControllerId::new("controller:data.provider").unwrap());

        assert!(provider
            .materialize(&DataMaterializationRequest {
                run_id: RunId::new("run:data").unwrap(),
                node_id: NodeId::new("model:base").unwrap(),
                input_name: "x".to_string(),
                phase: Phase::FitCv,
                variant_id: None,
                fold_id: None,
                binding: binding(),
            })
            .is_err());
    }
}
