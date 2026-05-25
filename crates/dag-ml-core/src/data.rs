use std::cell::RefCell;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::{ControllerId, FoldId, NodeId, RunId, VariantId};
use crate::phase::Phase;
use crate::relation::SampleRelationSet;
use crate::runtime::{DataMaterializationRequest, HandleKind, HandleRef, RuntimeDataProvider};

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
    pub source_ids: Vec<String>,
    pub relation_record_count: Option<usize>,
}

#[derive(Debug)]
pub struct InMemoryDataProvider {
    owner_controller: ControllerId,
    envelopes: BTreeMap<DataEnvelopeKey, ExternalDataPlanEnvelope>,
    next_handle: RefCell<u64>,
    records: RefCell<BTreeMap<u64, DataHandleRecord>>,
}

impl InMemoryDataProvider {
    pub fn new(owner_controller: ControllerId) -> Self {
        Self {
            owner_controller,
            envelopes: BTreeMap::new(),
            next_handle: RefCell::new(1),
            records: RefCell::new(BTreeMap::new()),
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
        if self.envelopes.insert(key, envelope).is_some() {
            return Err(DagMlError::RuntimeValidation(
                "duplicate external data-plan envelope".to_string(),
            ));
        }
        Ok(())
    }

    pub fn handle_record(&self, handle: u64) -> Option<DataHandleRecord> {
        self.records.borrow().get(&handle).cloned()
    }

    pub fn handle_records(&self) -> Vec<DataHandleRecord> {
        self.records.borrow().values().cloned().collect()
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
            source_ids: request.binding.source_ids.clone(),
            relation_record_count: envelope
                .coordinator_relations
                .as_ref()
                .map(|relations| relations.records.len()),
        };
        self.records.borrow_mut().insert(handle.handle, record);
        Ok(handle)
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

    #[test]
    fn in_memory_provider_materializes_validated_data_handles() {
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
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
