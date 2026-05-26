use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::campaign::stable_json_fingerprint;
use crate::data::ExternalDataPlanEnvelope;
use crate::error::{DagMlError, Result};
use crate::ids::{BundleId, ControllerId, FoldId, NodeId, SampleId, VariantId};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::phase::Phase;
use crate::plan::ExecutionPlan;
use crate::runtime::ArtifactRef;
use crate::selection::SelectionDecision;

pub const EXECUTION_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION: u32 = 1;
pub const BUNDLE_PREDICTION_CACHE_FORMAT: &str = "dag-ml-json-prediction-blocks-v1";

pub const MIN_READABLE_EXECUTION_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const MIN_WRITABLE_EXECUTION_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const MIN_READABLE_PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION: u32 = 1;
pub const MIN_WRITABLE_PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION: u32 = 1;

fn default_execution_bundle_schema_version() -> u32 {
    EXECUTION_BUNDLE_SCHEMA_VERSION
}

fn default_prediction_cache_payload_schema_version() -> u32 {
    PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchemaMigrationPolicy {
    pub artifact: String,
    pub current_version: u32,
    pub min_readable_version: u32,
    pub min_writable_version: u32,
    #[serde(default)]
    pub automatic_migrations: BTreeMap<u32, u32>,
}

impl SchemaMigrationPolicy {
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("schema migration artifact", &self.artifact)?;
        if self.current_version == 0
            || self.min_readable_version == 0
            || self.min_writable_version == 0
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "schema migration policy `{}` has zero version boundary",
                self.artifact
            )));
        }
        if self.min_readable_version > self.current_version {
            return Err(DagMlError::RuntimeValidation(format!(
                "schema migration policy `{}` min_readable_version exceeds current_version",
                self.artifact
            )));
        }
        if self.min_writable_version > self.current_version {
            return Err(DagMlError::RuntimeValidation(format!(
                "schema migration policy `{}` min_writable_version exceeds current_version",
                self.artifact
            )));
        }
        for (from, to) in &self.automatic_migrations {
            if *from == 0 || *to == 0 {
                return Err(DagMlError::RuntimeValidation(format!(
                    "schema migration policy `{}` contains a zero migration version",
                    self.artifact
                )));
            }
            if from == to {
                return Err(DagMlError::RuntimeValidation(format!(
                    "schema migration policy `{}` contains a no-op migration {from}->{to}",
                    self.artifact
                )));
            }
            if *to > self.current_version {
                return Err(DagMlError::RuntimeValidation(format!(
                    "schema migration policy `{}` migrates to unsupported future version {to}",
                    self.artifact
                )));
            }
        }
        Ok(())
    }

    pub fn validate_read_version(&self, version: u32, owner: &str) -> Result<()> {
        self.validate()?;
        if version < self.min_readable_version {
            return Err(DagMlError::RuntimeValidation(format!(
                "{owner} uses schema_version {version}, below minimum readable {} for {}",
                self.min_readable_version, self.artifact
            )));
        }
        if version > self.current_version {
            return Err(DagMlError::RuntimeValidation(format!(
                "{owner} uses future schema_version {version}, current readable {} for {}",
                self.current_version, self.artifact
            )));
        }
        if version != self.current_version && !self.automatic_migrations.contains_key(&version) {
            return Err(DagMlError::RuntimeValidation(format!(
                "{owner} uses schema_version {version}, but {} declares no automatic migration to current version {}",
                self.artifact, self.current_version
            )));
        }
        Ok(())
    }
}

pub fn execution_bundle_schema_migration_policy() -> SchemaMigrationPolicy {
    SchemaMigrationPolicy {
        artifact: "execution_bundle".to_string(),
        current_version: EXECUTION_BUNDLE_SCHEMA_VERSION,
        min_readable_version: MIN_READABLE_EXECUTION_BUNDLE_SCHEMA_VERSION,
        min_writable_version: MIN_WRITABLE_EXECUTION_BUNDLE_SCHEMA_VERSION,
        automatic_migrations: BTreeMap::new(),
    }
}

pub fn prediction_cache_payload_schema_migration_policy() -> SchemaMigrationPolicy {
    SchemaMigrationPolicy {
        artifact: "prediction_cache_payload".to_string(),
        current_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        min_readable_version: MIN_READABLE_PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        min_writable_version: MIN_WRITABLE_PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        automatic_migrations: BTreeMap::new(),
    }
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
pub struct BundlePredictionBlockCacheRecord {
    #[serde(default)]
    pub prediction_id: Option<String>,
    #[serde(default)]
    pub fold_id: Option<FoldId>,
    pub row_count: usize,
    pub sample_ids: Vec<SampleId>,
    pub content_fingerprint: String,
}

impl BundlePredictionBlockCacheRecord {
    pub fn validate(&self) -> Result<()> {
        if let Some(prediction_id) = &self.prediction_id {
            validate_non_empty("prediction_id", prediction_id)?;
        }
        if self.row_count == 0 {
            return Err(DagMlError::RuntimeValidation(
                "prediction block cache record has zero rows".to_string(),
            ));
        }
        if self.row_count != self.sample_ids.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction block cache record row_count {} does not match {} sample ids",
                self.row_count,
                self.sample_ids.len()
            )));
        }
        validate_unique_ids("sample id", &self.sample_ids)?;
        validate_fingerprint("prediction block cache content", &self.content_fingerprint)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundlePredictionCacheRecord {
    pub requirement_key: String,
    pub cache_id: String,
    pub format: String,
    pub partition: PredictionPartition,
    #[serde(default)]
    pub fold_ids: Vec<FoldId>,
    pub sample_ids: Vec<SampleId>,
    pub prediction_width: usize,
    pub target_names: Vec<String>,
    pub block_count: usize,
    pub row_count: usize,
    pub content_fingerprint: String,
    #[serde(default)]
    pub blocks: Vec<BundlePredictionBlockCacheRecord>,
}

impl BundlePredictionCacheRecord {
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("requirement_key", &self.requirement_key)?;
        validate_non_empty("cache_id", &self.cache_id)?;
        validate_non_empty("format", &self.format)?;
        if self.format != BUNDLE_PREDICTION_CACHE_FORMAT {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` uses unsupported format `{}`",
                self.cache_id, self.format
            )));
        }
        if self.partition != PredictionPartition::Validation {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` must cache validation OOF predictions",
                self.cache_id
            )));
        }
        validate_unique_ids("fold id", &self.fold_ids)?;
        validate_unique_ids("sample id", &self.sample_ids)?;
        if self.prediction_width == 0 {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` has zero prediction width",
                self.cache_id
            )));
        }
        if self.row_count != self.sample_ids.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` row_count does not match unique sample ids",
                self.cache_id
            )));
        }
        if self.target_names.len() != self.prediction_width {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` target name count does not match prediction width",
                self.cache_id
            )));
        }
        for target_name in &self.target_names {
            validate_non_empty("target_name", target_name)?;
        }
        if self.block_count == 0 || self.block_count != self.blocks.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` block_count does not match block records",
                self.cache_id
            )));
        }
        let block_row_count = self
            .blocks
            .iter()
            .map(|block| block.row_count)
            .sum::<usize>();
        if self.row_count == 0 || self.row_count != block_row_count {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` row_count does not match block records",
                self.cache_id
            )));
        }
        validate_fingerprint("prediction cache content", &self.content_fingerprint)?;
        for block in &self.blocks {
            block.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundlePredictionCachePayload {
    pub requirement_key: String,
    pub cache_id: String,
    pub format: String,
    pub partition: PredictionPartition,
    pub block_count: usize,
    pub row_count: usize,
    pub content_fingerprint: String,
    #[serde(default)]
    pub blocks: Vec<PredictionBlock>,
}

impl BundlePredictionCachePayload {
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("requirement_key", &self.requirement_key)?;
        validate_non_empty("cache_id", &self.cache_id)?;
        validate_non_empty("format", &self.format)?;
        if self.format != BUNDLE_PREDICTION_CACHE_FORMAT {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload `{}` uses unsupported format `{}`",
                self.cache_id, self.format
            )));
        }
        if self.partition != PredictionPartition::Validation {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload `{}` must cache validation OOF predictions",
                self.cache_id
            )));
        }
        if self.block_count == 0 || self.block_count != self.blocks.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload `{}` block_count does not match blocks",
                self.cache_id
            )));
        }
        let mut row_count = 0usize;
        let mut sample_ids = BTreeSet::new();
        for block in &self.blocks {
            block.validate_shape()?;
            if block.partition != self.partition {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache payload `{}` contains a block from partition {:?}",
                    self.cache_id, block.partition
                )));
            }
            for sample_id in &block.sample_ids {
                if !sample_ids.insert(sample_id) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "prediction cache payload `{}` contains duplicate sample `{}`",
                        self.cache_id, sample_id
                    )));
                }
            }
            row_count += block.sample_ids.len();
        }
        if self.row_count == 0 || self.row_count != row_count {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload `{}` row_count does not match blocks",
                self.cache_id
            )));
        }
        validate_fingerprint(
            "prediction cache payload content",
            &self.content_fingerprint,
        )?;
        let actual_fingerprint = stable_json_fingerprint(&self.blocks)?;
        if actual_fingerprint != self.content_fingerprint {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload `{}` content fingerprint does not match blocks",
                self.cache_id
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundlePredictionCachePayloadSet {
    pub bundle_id: BundleId,
    #[serde(default = "default_prediction_cache_payload_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub caches: Vec<BundlePredictionCachePayload>,
}

impl BundlePredictionCachePayloadSet {
    pub fn validate(&self) -> Result<()> {
        prediction_cache_payload_schema_migration_policy().validate_read_version(
            self.schema_version,
            &format!(
                "prediction cache payload set for bundle `{}`",
                self.bundle_id
            ),
        )?;
        let mut requirement_keys = BTreeSet::new();
        let mut cache_ids = BTreeSet::new();
        for payload in &self.caches {
            payload.validate()?;
            if !requirement_keys.insert(payload.requirement_key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache payload set for bundle `{}` has duplicate requirement `{}`",
                    self.bundle_id, payload.requirement_key
                )));
            }
            if !cache_ids.insert(payload.cache_id.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache payload set for bundle `{}` has duplicate cache id `{}`",
                    self.bundle_id, payload.cache_id
                )));
            }
        }
        Ok(())
    }

    pub fn validate_against_bundle(&self, bundle: &ExecutionBundle) -> Result<()> {
        self.validate()?;
        bundle.validate()?;
        if self.bundle_id != bundle.bundle_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload set bundle `{}` does not match bundle `{}`",
                self.bundle_id, bundle.bundle_id
            )));
        }
        if self.caches.len() != bundle.prediction_caches.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload set for bundle `{}` has {} payload(s) for {} cache record(s)",
                self.bundle_id,
                self.caches.len(),
                bundle.prediction_caches.len()
            )));
        }
        let records_by_requirement = bundle
            .prediction_caches
            .iter()
            .map(|record| (record.requirement_key.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        let payloads_by_requirement = self
            .caches
            .iter()
            .map(|payload| (payload.requirement_key.as_str(), payload))
            .collect::<BTreeMap<_, _>>();
        for (requirement_key, record) in records_by_requirement {
            let payload = payloads_by_requirement
                .get(requirement_key)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "prediction cache payload set for bundle `{}` is missing requirement `{}`",
                        self.bundle_id, requirement_key
                    ))
                })?;
            validate_prediction_cache_payload_matches_record(payload, record)?;
        }
        for requirement_key in payloads_by_requirement.keys() {
            if !bundle
                .prediction_caches
                .iter()
                .any(|record| record.requirement_key.as_str() == *requirement_key)
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache payload set for bundle `{}` contains unknown requirement `{}`",
                    self.bundle_id, requirement_key
                )));
            }
        }
        Ok(())
    }
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
    pub prediction_caches: Vec<BundlePredictionCacheRecord>,
    #[serde(default)]
    pub data_requirements: Vec<BundleDataRequirement>,
    #[serde(default)]
    pub unsafe_flags: BTreeSet<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl ExecutionBundle {
    pub fn validate(&self) -> Result<()> {
        execution_bundle_schema_migration_policy()
            .validate_read_version(self.schema_version, &format!("bundle `{}`", self.bundle_id))?;
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
        let mut prediction_cache_keys = BTreeMap::new();
        for cache in &self.prediction_caches {
            cache.validate()?;
            let requirement = prediction_keys.get(&cache.requirement_key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "prediction cache `{}` references unknown prediction requirement `{}`",
                    cache.cache_id, cache.requirement_key
                ))
            })?;
            validate_prediction_cache_matches_requirement(cache, requirement)?;
            if prediction_cache_keys
                .insert(cache.requirement_key.clone(), cache)
                .is_some()
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` has duplicate prediction cache for requirement `{}`",
                    self.bundle_id, cache.requirement_key
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
                if !prediction_cache_keys.contains_key(key) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "refit artifact `{}` references prediction requirement `{key}` without a prediction cache record",
                        artifact.artifact.id
                    )));
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
        self.validate_selections_against_plan(plan)?;
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
            let edge = plan
                .graph_plan
                .graph
                .edges
                .iter()
                .find(|edge| {
                    edge.source.node_id == requirement.producer_node
                    && edge.source.port_name == requirement.source_port
                    && edge.target.node_id == requirement.consumer_node
                    && edge.target.port_name == requirement.target_port
                    && edge.contract.requires_oof
                })
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "bundle `{}` prediction requirement `{}` does not match an OOF edge in the plan",
                        self.bundle_id,
                        requirement.key()
                    ))
                })?;
            let cache = self
                .prediction_caches
                .iter()
                .find(|cache| cache.requirement_key == requirement.key());
            validate_prediction_requirement_against_plan(self, plan, edge, requirement, cache)?;
        }
        Ok(())
    }

    fn validate_selections_against_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        if self.selections.is_empty() {
            return Ok(());
        }
        let artifact_node_ids = self
            .refit_artifacts
            .iter()
            .map(|artifact| artifact.node_id.clone())
            .collect::<BTreeSet<_>>();
        let required_metric_level = plan.campaign.aggregation_policy.selection_metric_level;
        for (selection_key, decision) in &self.selections {
            match decision.metric_level {
                Some(metric_level) if metric_level == required_metric_level => {}
                Some(metric_level) => {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "bundle `{}` selection `{selection_key}` metric_level {:?} does not match campaign selection_metric_level {:?}",
                        self.bundle_id, metric_level, required_metric_level
                    )));
                }
                None => {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "bundle `{}` selection `{selection_key}` is missing metric_level for campaign selection_metric_level {:?}",
                        self.bundle_id, required_metric_level
                    )));
                }
            }
            let selected_candidate_id = decision.selected_candidate_id.as_str();
            if let Ok(selected_node_id) = NodeId::new(selected_candidate_id) {
                if let Some(node_plan) = plan.node_plans.get(&selected_node_id) {
                    if node_plan.supported_phases.contains(&Phase::Refit)
                        && !artifact_node_ids.contains(&node_plan.node_id)
                    {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "bundle `{}` selection `{selection_key}` chose refittable node `{}` without a matching refit artifact",
                            self.bundle_id, node_plan.node_id
                        )));
                    }
                    continue;
                }
            }
            if VariantId::new(selected_candidate_id).is_ok()
                && plan
                    .variants
                    .iter()
                    .any(|variant| variant.variant_id.as_str() == selected_candidate_id)
            {
                continue;
            }
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` selection `{selection_key}` chose unknown candidate `{selected_candidate_id}` for plan `{}`",
                self.bundle_id, plan.id
            )));
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

fn validate_prediction_requirement_against_plan(
    bundle: &ExecutionBundle,
    plan: &ExecutionPlan,
    edge: &crate::graph::EdgeSpec,
    requirement: &BundlePredictionRequirement,
    cache: Option<&BundlePredictionCacheRecord>,
) -> Result<()> {
    if !edge.contract.requires_fold_alignment {
        return Ok(());
    }
    let fold_set = plan.fold_set.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "bundle `{}` prediction requirement `{}` needs fold alignment but plan `{}` has no fold set",
            bundle.bundle_id,
            requirement.key(),
            plan.id
        ))
    })?;
    let expected_fold_ids = fold_set
        .folds
        .iter()
        .map(|fold| fold.fold_id.clone())
        .collect::<BTreeSet<_>>();
    let requirement_fold_ids = requirement
        .fold_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if requirement_fold_ids != expected_fold_ids {
        return Err(DagMlError::RuntimeValidation(format!(
            "bundle `{}` prediction requirement `{}` fold ids do not match plan fold set",
            bundle.bundle_id,
            requirement.key()
        )));
    }
    let expected_sample_ids = fold_set.sample_ids.iter().cloned().collect::<BTreeSet<_>>();
    let requirement_sample_ids = requirement
        .sample_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if requirement_sample_ids != expected_sample_ids {
        return Err(DagMlError::RuntimeValidation(format!(
            "bundle `{}` prediction requirement `{}` sample ids do not match plan fold set",
            bundle.bundle_id,
            requirement.key()
        )));
    }
    if let Some(cache) = cache {
        validate_prediction_cache_blocks_match_fold_set(bundle, requirement, cache, fold_set)?;
    }
    Ok(())
}

fn validate_prediction_cache_blocks_match_fold_set(
    bundle: &ExecutionBundle,
    requirement: &BundlePredictionRequirement,
    cache: &BundlePredictionCacheRecord,
    fold_set: &crate::fold::FoldSet,
) -> Result<()> {
    let folds = fold_set
        .folds
        .iter()
        .map(|fold| (&fold.fold_id, fold))
        .collect::<BTreeMap<_, _>>();
    let expected_fold_ids = fold_set
        .folds
        .iter()
        .map(|fold| fold.fold_id.clone())
        .collect::<BTreeSet<_>>();
    let mut covered_fold_ids = BTreeSet::new();
    let mut covered_sample_ids = BTreeSet::new();
    for block in &cache.blocks {
        let fold_id = block.fold_id.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "bundle `{}` prediction cache `{}` has an OOF block without a fold id",
                bundle.bundle_id, cache.cache_id
            ))
        })?;
        covered_fold_ids.insert(fold_id.clone());
        let fold = folds.get(fold_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "bundle `{}` prediction cache `{}` references unknown fold `{fold_id}`",
                bundle.bundle_id, cache.cache_id
            ))
        })?;
        let block_samples = block.sample_ids.iter().cloned().collect::<BTreeSet<_>>();
        let expected_samples = fold
            .validation_sample_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if block_samples != expected_samples {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` prediction cache `{}` block for fold `{fold_id}` does not match validation samples for requirement `{}`",
                bundle.bundle_id,
                cache.cache_id,
                requirement.key()
            )));
        }
        for sample_id in block_samples {
            if !covered_sample_ids.insert(sample_id.clone()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "bundle `{}` prediction cache `{}` has duplicate OOF sample `{sample_id}`",
                    bundle.bundle_id, cache.cache_id
                )));
            }
        }
    }
    if covered_fold_ids != expected_fold_ids {
        return Err(DagMlError::RuntimeValidation(format!(
            "bundle `{}` prediction cache `{}` does not cover all folds for requirement `{}`",
            bundle.bundle_id,
            cache.cache_id,
            requirement.key()
        )));
    }
    let expected_sample_ids = fold_set.sample_ids.iter().cloned().collect::<BTreeSet<_>>();
    if covered_sample_ids != expected_sample_ids {
        return Err(DagMlError::RuntimeValidation(format!(
            "bundle `{}` prediction cache `{}` does not cover the full OOF sample universe for requirement `{}`",
            bundle.bundle_id,
            cache.cache_id,
            requirement.key()
        )));
    }
    Ok(())
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
    build_execution_bundle_with_prediction_contracts(
        bundle_id,
        plan,
        selected_variant_id,
        selections,
        refit_artifacts,
        prediction_requirements,
        Vec::new(),
    )
}

pub fn build_execution_bundle_with_prediction_contracts(
    bundle_id: BundleId,
    plan: &ExecutionPlan,
    selected_variant_id: Option<VariantId>,
    selections: BTreeMap<String, SelectionDecision>,
    refit_artifacts: Vec<RefitArtifactRecord>,
    prediction_requirements: Vec<BundlePredictionRequirement>,
    prediction_caches: Vec<BundlePredictionCacheRecord>,
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
        prediction_caches,
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

pub fn build_prediction_cache_record(
    requirement: &BundlePredictionRequirement,
    blocks: &[PredictionBlock],
) -> Result<BundlePredictionCacheRecord> {
    let selected = select_prediction_cache_blocks(requirement, blocks)?;
    build_prediction_cache_record_from_selected(requirement, &selected)
}

pub fn build_prediction_cache_payload(
    requirement: &BundlePredictionRequirement,
    blocks: &[PredictionBlock],
) -> Result<BundlePredictionCachePayload> {
    let selected = select_prediction_cache_blocks(requirement, blocks)?;
    let payload = BundlePredictionCachePayload {
        requirement_key: requirement.key(),
        cache_id: format!("prediction-cache:{}", requirement.key()),
        format: BUNDLE_PREDICTION_CACHE_FORMAT.to_string(),
        partition: requirement.partition.clone(),
        block_count: selected.len(),
        row_count: selected.iter().map(|block| block.sample_ids.len()).sum(),
        content_fingerprint: stable_json_fingerprint(&selected)?,
        blocks: selected,
    };
    payload.validate()?;
    let record = build_prediction_cache_record(requirement, &payload.blocks)?;
    validate_prediction_cache_payload_matches_record(&payload, &record)?;
    Ok(payload)
}

pub fn validate_prediction_cache_payload_matches_record(
    payload: &BundlePredictionCachePayload,
    record: &BundlePredictionCacheRecord,
) -> Result<()> {
    payload.validate()?;
    record.validate()?;
    if payload.requirement_key != record.requirement_key
        || payload.cache_id != record.cache_id
        || payload.format != record.format
        || payload.partition != record.partition
        || payload.block_count != record.block_count
        || payload.row_count != record.row_count
        || payload.content_fingerprint != record.content_fingerprint
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache payload `{}` does not match cache record `{}`",
            payload.cache_id, record.cache_id
        )));
    }
    let block_records = payload
        .blocks
        .iter()
        .map(|block| {
            Ok(BundlePredictionBlockCacheRecord {
                prediction_id: block.prediction_id.clone(),
                fold_id: block.fold_id.clone(),
                row_count: block.sample_ids.len(),
                sample_ids: block.sample_ids.clone(),
                content_fingerprint: stable_json_fingerprint(block)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if block_records != record.blocks {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache payload `{}` block fingerprints do not match cache record",
            payload.cache_id
        )));
    }
    Ok(())
}

fn select_prediction_cache_blocks(
    requirement: &BundlePredictionRequirement,
    blocks: &[PredictionBlock],
) -> Result<Vec<PredictionBlock>> {
    requirement.validate()?;
    let mut selected = blocks
        .iter()
        .filter(|block| {
            block.producer_node == requirement.producer_node
                && block.partition == requirement.partition
        })
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache requirement `{}` has no matching prediction blocks",
            requirement.key()
        )));
    }
    selected.sort_by(|left, right| {
        (
            left.fold_id.as_ref().map(ToString::to_string),
            left.prediction_id.clone(),
        )
            .cmp(&(
                right.fold_id.as_ref().map(ToString::to_string),
                right.prediction_id.clone(),
            ))
    });
    Ok(selected)
}

fn build_prediction_cache_record_from_selected(
    requirement: &BundlePredictionRequirement,
    selected: &[PredictionBlock],
) -> Result<BundlePredictionCacheRecord> {
    requirement.validate()?;
    if selected.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache requirement `{}` has no matching prediction blocks",
            requirement.key()
        )));
    }
    let mut fold_ids = BTreeSet::new();
    let mut sample_ids = BTreeSet::new();
    let mut target_names: Option<Vec<String>> = None;
    let mut prediction_width: Option<usize> = None;
    let mut row_count = 0usize;
    let mut block_records = Vec::new();
    for block in selected {
        if block.producer_node != requirement.producer_node
            || block.partition != requirement.partition
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` contains a block outside the requirement scope",
                requirement.key()
            )));
        }
        let width = block.validate_shape()?;
        if prediction_width.is_some_and(|expected| expected != width) {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` has inconsistent prediction width",
                requirement.key()
            )));
        }
        prediction_width = Some(width);
        let block_target_names = normalized_prediction_targets(block, width);
        if target_names
            .as_ref()
            .is_some_and(|expected| expected != &block_target_names)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` has inconsistent target names",
                requirement.key()
            )));
        }
        target_names = Some(block_target_names);
        if let Some(fold_id) = &block.fold_id {
            fold_ids.insert(fold_id.clone());
        }
        sample_ids.extend(block.sample_ids.iter().cloned());
        row_count += block.sample_ids.len();
        block_records.push(BundlePredictionBlockCacheRecord {
            prediction_id: block.prediction_id.clone(),
            fold_id: block.fold_id.clone(),
            row_count: block.sample_ids.len(),
            sample_ids: block.sample_ids.clone(),
            content_fingerprint: stable_json_fingerprint(block)?,
        });
    }

    let record = BundlePredictionCacheRecord {
        requirement_key: requirement.key(),
        cache_id: format!("prediction-cache:{}", requirement.key()),
        format: BUNDLE_PREDICTION_CACHE_FORMAT.to_string(),
        partition: requirement.partition.clone(),
        fold_ids: fold_ids.into_iter().collect(),
        sample_ids: sample_ids.into_iter().collect(),
        prediction_width: prediction_width.unwrap_or_default(),
        target_names: target_names.unwrap_or_default(),
        block_count: block_records.len(),
        row_count,
        content_fingerprint: stable_json_fingerprint(selected)?,
        blocks: block_records,
    };
    validate_prediction_cache_matches_requirement(&record, requirement)?;
    record.validate()?;
    Ok(record)
}

fn validate_prediction_cache_matches_requirement(
    cache: &BundlePredictionCacheRecord,
    requirement: &BundlePredictionRequirement,
) -> Result<()> {
    if cache.requirement_key != requirement.key()
        || cache.partition != requirement.partition
        || cache.fold_ids != requirement.fold_ids
        || cache.sample_ids != requirement.sample_ids
        || cache.prediction_width != requirement.prediction_width
        || cache.target_names != requirement.target_names
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache `{}` does not match requirement `{}`",
            cache.cache_id,
            requirement.key()
        )));
    }
    Ok(())
}

fn normalized_prediction_targets(block: &PredictionBlock, width: usize) -> Vec<String> {
    if block.target_names.is_empty() {
        (0..width).map(|index| format!("p{index}")).collect()
    } else {
        block.target_names.clone()
    }
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
        self.validate_for_bundle_with_prediction_cache_store(bundle, false)
    }

    pub fn validate_for_bundle_with_prediction_cache_store(
        &self,
        bundle: &ExecutionBundle,
        prediction_cache_available: bool,
    ) -> Result<()> {
        self.validate_for_bundle_internal(bundle, prediction_cache_available)
    }

    pub fn validate_for_bundle_with_prediction_cache_payloads(
        &self,
        bundle: &ExecutionBundle,
        prediction_cache_payloads: Option<&BundlePredictionCachePayloadSet>,
    ) -> Result<()> {
        if let Some(payloads) = prediction_cache_payloads {
            payloads.validate_against_bundle(bundle)?;
        }
        self.validate_for_bundle_internal(bundle, prediction_cache_payloads.is_some())
    }

    fn validate_for_bundle_internal(
        &self,
        bundle: &ExecutionBundle,
        prediction_cache_available: bool,
    ) -> Result<()> {
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
        if self.phase == Phase::Refit && !bundle.prediction_requirements.is_empty() {
            if prediction_cache_available {
                return self.validate_data_envelope_keys(bundle);
            }
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` cannot replay REFIT because it depends on {} OOF prediction requirement(s) but stores only prediction cache manifests",
                bundle.bundle_id,
                bundle.prediction_requirements.len()
            )));
        }
        self.validate_data_envelope_keys(bundle)
    }

    fn validate_data_envelope_keys(&self, bundle: &ExecutionBundle) -> Result<()> {
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

    fn branch_merge_selection_decisions() -> BTreeMap<String, SelectionDecision> {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/bundle/selection_decisions_branch_merge.json"
        ))
        .unwrap()
    }

    fn refit_artifact(
        plan: &ExecutionPlan,
        node_id: &str,
        data_requirement_keys: Vec<String>,
        prediction_requirement_keys: Vec<String>,
    ) -> RefitArtifactRecord {
        let node_id = NodeId::new(node_id).unwrap();
        let node_plan = plan.node_plans.get(&node_id).unwrap();
        RefitArtifactRecord {
            node_id: node_plan.node_id.clone(),
            controller_id: node_plan.controller_id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new(format!("artifact:{}:refit", node_plan.node_id)).unwrap(),
                kind: "mock_model".to_string(),
                controller_id: node_plan.controller_id.clone(),
                size_bytes: Some(128),
            },
            params_fingerprint: node_plan.params_fingerprint.clone(),
            data_requirement_keys,
            prediction_requirement_keys,
        }
    }

    fn branch_merge_samples() -> Vec<SampleId> {
        vec![
            SampleId::new("sample:1").unwrap(),
            SampleId::new("sample:2").unwrap(),
            SampleId::new("sample:3").unwrap(),
            SampleId::new("sample:4").unwrap(),
        ]
    }

    fn branch_merge_requirement(
        producer_node: &str,
        target_port: &str,
    ) -> BundlePredictionRequirement {
        BundlePredictionRequirement {
            producer_node: NodeId::new(producer_node).unwrap(),
            source_port: "oof".to_string(),
            consumer_node: NodeId::new("merge:stack.pred_plus_original.meta:ridge").unwrap(),
            target_port: target_port.to_string(),
            partition: PredictionPartition::Validation,
            fold_ids: vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
            ],
            sample_ids: branch_merge_samples(),
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        }
    }

    fn branch_merge_prediction_blocks(producer_node: &str, offset: f64) -> Vec<PredictionBlock> {
        let producer_node = NodeId::new(producer_node).unwrap();
        let samples = branch_merge_samples();
        vec![
            PredictionBlock {
                prediction_id: Some(format!("prediction:{producer_node}:fold0")),
                producer_node: producer_node.clone(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                sample_ids: samples[0..2].to_vec(),
                values: vec![vec![offset + 0.1], vec![offset + 0.2]],
                target_names: vec!["y".to_string()],
            },
            PredictionBlock {
                prediction_id: Some(format!("prediction:{producer_node}:fold1")),
                producer_node,
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:1").unwrap()),
                sample_ids: samples[2..4].to_vec(),
                values: vec![vec![offset + 0.3], vec![offset + 0.4]],
                target_names: vec!["y".to_string()],
            },
        ]
    }

    fn decision() -> SelectionDecision {
        select_candidate(
            &SelectionPolicy {
                id: "select:merge".to_string(),
                metric: SelectionMetric {
                    name: "rmse".to_string(),
                    objective: MetricObjective::Minimize,
                },
                required_metric_level: Some(crate::policy::PredictionLevel::Sample),
                require_finite: true,
            },
            &[
                CandidateScore {
                    candidate_id: "model:base".to_string(),
                    metrics: BTreeMap::from([("rmse".to_string(), 1.0)]),
                    metadata: BTreeMap::from([(
                        "metric_level".to_string(),
                        serde_json::Value::String("sample".to_string()),
                    )]),
                },
                CandidateScore {
                    candidate_id: "model:other".to_string(),
                    metrics: BTreeMap::from([("rmse".to_string(), 2.0)]),
                    metadata: BTreeMap::from([(
                        "metric_level".to_string(),
                        serde_json::Value::String("sample".to_string()),
                    )]),
                },
            ],
        )
        .unwrap()
    }

    fn selected_model_base_decision() -> SelectionDecision {
        decision()
    }

    fn model_base_refit_artifact(plan: &ExecutionPlan) -> RefitArtifactRecord {
        let model_plan = plan
            .node_plans
            .get(&NodeId::new("model:base").unwrap())
            .unwrap();
        RefitArtifactRecord {
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
        }
    }

    #[test]
    fn builds_bundle_from_execution_plan() {
        let plan = plan();
        let artifact = model_base_refit_artifact(&plan);

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
    fn bundle_selections_must_match_plan_and_refit_artifacts() {
        let plan = plan();
        let artifact = model_base_refit_artifact(&plan);
        let valid = build_execution_bundle(
            BundleId::new("bundle:selected.model").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::from([("model".to_string(), selected_model_base_decision())]),
            vec![artifact.clone()],
        )
        .unwrap();
        valid.validate_against_plan(&plan).unwrap();

        assert!(build_execution_bundle(
            BundleId::new("bundle:selected.model.missing.artifact").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::from([("model".to_string(), selected_model_base_decision())]),
            Vec::new(),
        )
        .is_err());

        let mut missing_level = selected_model_base_decision();
        missing_level.metric_level = None;
        assert!(build_execution_bundle(
            BundleId::new("bundle:selected.missing.level").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::from([("model".to_string(), missing_level)]),
            vec![artifact.clone()],
        )
        .is_err());

        let mut wrong_level = selected_model_base_decision();
        wrong_level.metric_level = Some(crate::policy::PredictionLevel::Target);
        assert!(build_execution_bundle(
            BundleId::new("bundle:selected.wrong.level").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::from([("model".to_string(), wrong_level)]),
            vec![artifact.clone()],
        )
        .is_err());

        let mut unknown = selected_model_base_decision();
        unknown.selected_candidate_id = "model:missing".to_string();
        unknown.ranked_candidates[0].candidate_id = "model:missing".to_string();
        assert!(build_execution_bundle(
            BundleId::new("bundle:selected.unknown").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::from([("model".to_string(), unknown)]),
            vec![artifact],
        )
        .is_err());
    }

    #[test]
    fn branch_merge_bundle_links_selected_refits_and_fold_aligned_oof_caches() {
        let plan = branch_merge_plan();
        let b0_requirement = branch_merge_requirement("branch:b0.model:ridge", "b0_oof");
        let b1_requirement = branch_merge_requirement("branch:b1.model:rf", "b1_oof");
        let b0_cache = build_prediction_cache_record(
            &b0_requirement,
            &branch_merge_prediction_blocks("branch:b0.model:ridge", 0.0),
        )
        .unwrap();
        let b1_cache = build_prediction_cache_record(
            &b1_requirement,
            &branch_merge_prediction_blocks("branch:b1.model:rf", 1.0),
        )
        .unwrap();
        let b0_artifact = refit_artifact(
            &plan,
            "branch:b0.model:ridge",
            vec!["branch:b0.model:ridge.x".to_string()],
            Vec::new(),
        );
        let b1_artifact = refit_artifact(
            &plan,
            "branch:b1.model:rf",
            vec!["branch:b1.model:rf.x".to_string()],
            Vec::new(),
        );
        let merge_artifact = refit_artifact(
            &plan,
            "merge:stack.pred_plus_original.meta:ridge",
            vec!["merge:stack.pred_plus_original.meta:ridge.x_original".to_string()],
            vec![b0_requirement.key(), b1_requirement.key()],
        );

        let bundle = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:branch.merge.selected.refit").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            branch_merge_selection_decisions(),
            vec![
                b0_artifact.clone(),
                b1_artifact.clone(),
                merge_artifact.clone(),
            ],
            vec![b0_requirement.clone(), b1_requirement.clone()],
            vec![b0_cache.clone(), b1_cache.clone()],
        )
        .unwrap();
        bundle.validate_against_plan(&plan).unwrap();
        assert_eq!(bundle.selections.len(), 3);
        assert_eq!(bundle.prediction_requirements.len(), 2);
        assert_eq!(
            bundle.refit_artifacts[2].data_requirement_keys,
            vec!["merge:stack.pred_plus_original.meta:ridge.x_original"]
        );
        assert_eq!(
            bundle.refit_artifacts[2].prediction_requirement_keys,
            vec![
                "branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof",
                "branch:b1.model:rf.oof->merge:stack.pred_plus_original.meta:ridge.b1_oof",
            ]
        );

        assert!(build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:branch.merge.missing.branch.refit").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            branch_merge_selection_decisions(),
            vec![b0_artifact.clone(), merge_artifact.clone()],
            vec![b0_requirement.clone(), b1_requirement.clone()],
            vec![b0_cache.clone(), b1_cache.clone()],
        )
        .is_err());

        let mut misaligned_cache = b0_cache;
        misaligned_cache.blocks[0].sample_ids = vec![
            SampleId::new("sample:1").unwrap(),
            SampleId::new("sample:3").unwrap(),
        ];
        let error = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:branch.merge.misaligned.oof.cache").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            branch_merge_selection_decisions(),
            vec![b0_artifact, b1_artifact, merge_artifact],
            vec![b0_requirement, b1_requirement],
            vec![misaligned_cache, b1_cache],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("does not match validation samples"),
            "unexpected fold-alignment error: {error}"
        );
    }

    #[test]
    fn prediction_requirements_are_typed_and_validate_against_oof_edges() {
        let plan = branch_merge_plan();
        let meta_plan = plan
            .node_plans
            .get(&NodeId::new("merge:stack.pred_plus_original.meta:ridge").unwrap())
            .unwrap();
        let producer_node = NodeId::new("branch:b0.model:ridge").unwrap();
        let fold0 = FoldId::new("fold:0").unwrap();
        let fold1 = FoldId::new("fold:1").unwrap();
        let samples = [
            SampleId::new("sample:1").unwrap(),
            SampleId::new("sample:2").unwrap(),
            SampleId::new("sample:3").unwrap(),
            SampleId::new("sample:4").unwrap(),
        ];
        let requirement = BundlePredictionRequirement {
            producer_node: producer_node.clone(),
            source_port: "oof".to_string(),
            consumer_node: meta_plan.node_id.clone(),
            target_port: "b0_oof".to_string(),
            partition: PredictionPartition::Validation,
            fold_ids: vec![fold0.clone(), fold1.clone()],
            sample_ids: samples.to_vec(),
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let prediction_blocks = vec![
            PredictionBlock {
                prediction_id: Some("prediction:branch:b0.fold0".to_string()),
                producer_node: producer_node.clone(),
                partition: PredictionPartition::Validation,
                fold_id: Some(fold0),
                sample_ids: samples[0..2].to_vec(),
                values: vec![vec![0.1], vec![0.2]],
                target_names: vec!["y".to_string()],
            },
            PredictionBlock {
                prediction_id: Some("prediction:branch:b0.fold1".to_string()),
                producer_node: producer_node.clone(),
                partition: PredictionPartition::Validation,
                fold_id: Some(fold1),
                sample_ids: samples[2..4].to_vec(),
                values: vec![vec![0.3], vec![0.4]],
                target_names: vec!["y".to_string()],
            },
        ];
        let cache = build_prediction_cache_record(&requirement, &prediction_blocks).unwrap();
        let payload = build_prediction_cache_payload(&requirement, &prediction_blocks).unwrap();
        validate_prediction_cache_payload_matches_record(&payload, &cache).unwrap();
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

        assert!(build_execution_bundle_with_prediction_requirements(
            BundleId::new("bundle:typed.prediction.requirement.without.cache").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            vec![artifact.clone()],
            vec![requirement.clone()],
        )
        .is_err());

        let bundle = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:typed.prediction.requirement").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            vec![artifact],
            vec![requirement],
            vec![cache],
        )
        .unwrap();
        bundle.validate_against_plan(&plan).unwrap();
        assert_eq!(bundle.prediction_requirements.len(), 1);
        assert_eq!(bundle.prediction_caches.len(), 1);
        assert_eq!(
            bundle.refit_artifacts[0].prediction_requirement_keys,
            vec!["branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"]
        );
        let payload_set = BundlePredictionCachePayloadSet {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            caches: vec![payload],
        };
        payload_set.validate_against_bundle(&bundle).unwrap();
        let refit_replay_request = ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            data_envelope_keys: bundle
                .data_requirements
                .iter()
                .map(BundleDataRequirement::key)
                .collect(),
        };
        refit_replay_request
            .validate_for_bundle_with_prediction_cache_payloads(&bundle, Some(&payload_set))
            .unwrap();
        let mut tampered_payload_set = payload_set.clone();
        tampered_payload_set.caches[0].blocks[0].values[0][0] = 99.0;
        assert!(tampered_payload_set
            .validate_against_bundle(&bundle)
            .is_err());
        let mut missing_payload_set = payload_set.clone();
        missing_payload_set.caches.clear();
        assert!(missing_payload_set
            .validate_against_bundle(&bundle)
            .is_err());
        assert!(refit_replay_request.validate_for_bundle(&bundle).is_err());

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
            BTreeMap::new(),
            Vec::new(),
        )
        .unwrap();
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
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
            BTreeMap::new(),
            Vec::new(),
        )
        .unwrap();
        bundle.schema_version = EXECUTION_BUNDLE_SCHEMA_VERSION + 1;

        assert!(bundle.validate().is_err());

        bundle.schema_version = 0;
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn schema_migration_policy_is_explicit_and_refuses_implicit_migrations() {
        let bundle_policy = execution_bundle_schema_migration_policy();
        assert_eq!(
            bundle_policy.current_version,
            EXECUTION_BUNDLE_SCHEMA_VERSION
        );
        assert_eq!(
            bundle_policy.min_readable_version,
            MIN_READABLE_EXECUTION_BUNDLE_SCHEMA_VERSION
        );
        assert!(bundle_policy.automatic_migrations.is_empty());
        bundle_policy
            .validate_read_version(EXECUTION_BUNDLE_SCHEMA_VERSION, "bundle `current`")
            .unwrap();
        assert!(bundle_policy
            .validate_read_version(EXECUTION_BUNDLE_SCHEMA_VERSION + 1, "bundle `future`")
            .is_err());
        assert!(bundle_policy
            .validate_read_version(0, "bundle `zero`")
            .is_err());

        let mut future_policy = SchemaMigrationPolicy {
            artifact: "execution_bundle".to_string(),
            current_version: 2,
            min_readable_version: 1,
            min_writable_version: 2,
            automatic_migrations: BTreeMap::new(),
        };
        assert!(future_policy
            .validate_read_version(1, "bundle `old-without-migration`")
            .is_err());
        future_policy.automatic_migrations.insert(1, 2);
        future_policy
            .validate_read_version(1, "bundle `old-with-migration`")
            .unwrap();
    }

    #[test]
    fn prediction_cache_payload_schema_policy_rejects_unsupported_versions() {
        let policy = prediction_cache_payload_schema_migration_policy();
        assert_eq!(
            policy.current_version,
            PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION
        );
        assert!(policy.automatic_migrations.is_empty());

        let mut payload_set = BundlePredictionCachePayloadSet {
            bundle_id: BundleId::new("bundle:payload.schema").unwrap(),
            schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            caches: Vec::new(),
        };
        payload_set.validate().unwrap();

        payload_set.schema_version = PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION + 1;
        assert!(payload_set.validate().is_err());

        payload_set.schema_version = 0;
        assert!(payload_set.validate().is_err());
    }

    #[test]
    fn replay_request_requires_predict_explain_or_refit_phase() {
        let bundle = build_execution_bundle(
            BundleId::new("bundle:demo").unwrap(),
            &plan(),
            None,
            BTreeMap::new(),
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
        ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
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
