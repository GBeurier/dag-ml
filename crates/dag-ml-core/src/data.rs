use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::NodeId;
use crate::relation::SampleRelationSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRequestPartition {
    FoldTrain,
    FoldValidation,
    FullTrain,
    Predict,
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
        }
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
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    #[serde(default)]
    pub coordinator_relations: Option<SampleRelationSet>,
}

impl ExternalDataPlanEnvelope {
    pub fn validate(&self) -> Result<()> {
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

fn validate_fingerprint(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::CampaignValidation(format!(
            "{label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::NodeId;

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
            source_ids: vec!["nir".to_string()],
            require_relations: true,
            view_policy: DataViewPolicy::default(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn validates_data_binding_contract() {
        binding().validate().unwrap();
    }

    #[test]
    fn validates_external_data_envelope_subset() {
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
        ))
        .unwrap();

        binding().validate_envelope(&envelope).unwrap();
    }

    #[test]
    fn refuses_envelope_fingerprint_mismatch() {
        let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
        ))
        .unwrap();
        envelope.plan_fingerprint = "0".repeat(64);

        assert!(binding().validate_envelope(&envelope).is_err());
    }
}
