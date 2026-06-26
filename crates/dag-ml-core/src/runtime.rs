use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::aggregation::{
    aggregate_observation_predictions, aggregate_sample_predictions_by_unit,
    reduce_predictions_across_branches, reduce_proba_mean_across_branches,
    AggregatedPredictionBlock, AggregationControllerInput, AggregationControllerOutput,
    AggregationControllerResult, AggregationControllerTask, ObservationPredictionBlock,
    PredictionUnitId,
};
use crate::bundle::{
    build_aggregated_prediction_cache_payload, build_prediction_cache_payload,
    bundle_prediction_requirement_key, validate_prediction_cache_payload_matches_record,
    BundlePredictionCachePayload, BundlePredictionCachePayloadSet, BundlePredictionCacheRecord,
    BundlePredictionRequirement, ExecutionBundle, RefitArtifactRecord, ReplayPhaseRequest,
};
use crate::campaign::stable_json_fingerprint;
use crate::controller::{capabilities_support_fit_influence, ControllerCapability};
use crate::data::{
    DataBinding, DataRequestPartition, ExternalDataPlanEnvelope, RepresentationCompatibilityReport,
    RepresentationPlan, RepresentationReplayManifest,
};
use crate::error::{DagMlError, Result};
use crate::fold::{FoldAssignment, FoldPartitionMode, FoldSet};
use crate::generation::{GenerationChoice, VariantPlan};
use crate::graph::{EdgeSpec, PortKind};
use crate::ids::{
    ArtifactId, BranchId, BundleId, ControllerId, FoldId, LineageId, NodeId, RunId, SampleId,
    VariantId,
};
use crate::metrics::{
    cross_fold_validation_reports, reassemble_merge_targets, score_regression_aggregated_block,
    score_regression_prediction_block, RegressionMetricKind, RegressionMetricReport,
    RegressionTargetBlock, RegressionTargetRecord, ScoreSet, SCORE_SET_SCHEMA_VERSION,
};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::phase::Phase;
use crate::plan::{CampaignSpec, ExecutionPlan, NodePlan};
use crate::policy::{
    AggregationPolicy, FitInfluencePolicy, PredictionLevel, ShapeDelta, ShapeDeltaKind,
};
use crate::relation::SampleRelationSet;
use crate::rng::SeedContext;
use crate::selection::{select_candidate, CandidateScore, SelectionMetric, SelectionPolicy};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleKind {
    Data,
    DataView,
    Model,
    Artifact,
    Prediction,
    Relation,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct HandleRef {
    pub handle: u64,
    pub kind: HandleKind,
    pub owner_controller: ControllerId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactBackend {
    Joblib,
    Torch,
    Tensorflow,
    Onnx,
    Safetensors,
    Json,
    Raw,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub kind: String,
    pub controller_id: ControllerId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<ArtifactBackend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_fingerprint: Option<String>,
    pub size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_version: Option<String>,
}

impl ArtifactRef {
    pub fn validate(&self) -> Result<()> {
        if self.kind.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` has empty kind",
                self.id
            )));
        }
        validate_artifact_optional_text("uri", &self.uri, &self.id)?;
        validate_artifact_optional_text("plugin", &self.plugin, &self.id)?;
        validate_artifact_optional_text("plugin_version", &self.plugin_version, &self.id)?;
        if self.plugin_version.is_some() && self.plugin.is_none() {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` has plugin_version without plugin",
                self.id
            )));
        }
        if let Some(content_fingerprint) = &self.content_fingerprint {
            validate_runtime_fingerprint("artifact content", content_fingerprint)?;
        }
        if self.uri.is_some() && self.backend.is_none() {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` has uri without backend",
                self.id
            )));
        }
        if self.uri.is_some() && self.content_fingerprint.is_none() {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` has uri without content_fingerprint",
                self.id
            )));
        }
        Ok(())
    }

    /// Validate that the artifact carries portable metadata: a backend, a safe
    /// relative URI and a content fingerprint. Legacy artifacts that only carry
    /// inline metadata stay readable through [`ArtifactRef::validate`] but are
    /// refused here so persisted manifests can be moved with their payloads.
    pub fn validate_portable(&self) -> Result<()> {
        self.validate()?;
        let Some(uri) = self.uri.as_deref() else {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is not portable: requires backend, uri and content_fingerprint",
                self.id
            )));
        };
        // `validate` already guarantees that a present URI implies a backend and
        // a 64-hex content fingerprint, so confirming the URI is enough here.
        validate_relative_artifact_uri(&self.id, uri)
    }
}

pub fn refit_artifact_input_key(artifact_id: &ArtifactId) -> String {
    format!("artifact:{artifact_id}")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactMaterializationRequest {
    pub run_id: RunId,
    pub bundle_id: BundleId,
    pub node_id: NodeId,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub controller_id: ControllerId,
    pub artifact: ArtifactRef,
    pub params_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactHandleRecord {
    pub handle: HandleRef,
    pub node_id: NodeId,
    pub controller_id: ControllerId,
    pub artifact: ArtifactRef,
    pub params_fingerprint: String,
}

impl ArtifactHandleRecord {
    pub fn validate(&self) -> Result<()> {
        self.artifact.validate()?;
        if !matches!(self.handle.kind, HandleKind::Model | HandleKind::Artifact) {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered with non-artifact/model handle kind {:?}",
                self.artifact.id, self.handle.kind
            )));
        }
        if self.handle.owner_controller != self.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` handle owner `{}` does not match controller `{}`",
                self.artifact.id, self.handle.owner_controller, self.controller_id
            )));
        }
        if self.artifact.controller_id != self.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` controller `{}` does not match record controller `{}`",
                self.artifact.id, self.artifact.controller_id, self.controller_id
            )));
        }
        if self.params_fingerprint.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` has empty params fingerprint",
                self.artifact.id
            )));
        }
        Ok(())
    }
}

pub trait RuntimeArtifactStore {
    fn materialize(&self, request: &ArtifactMaterializationRequest) -> Result<HandleRef>;
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryArtifactStore {
    records: BTreeMap<ArtifactId, ArtifactHandleRecord>,
    refit_artifacts: BTreeMap<ArtifactId, RefitArtifactRecord>,
}

impl InMemoryArtifactStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, artifact: &RefitArtifactRecord, handle: HandleRef) -> Result<()> {
        artifact.validate()?;
        let record = ArtifactHandleRecord {
            handle,
            node_id: artifact.node_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
        };
        record.validate()?;
        if self.records.contains_key(&record.artifact.id)
            || self.refit_artifacts.contains_key(&record.artifact.id)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate artifact handle for `{}`",
                artifact.artifact.id
            )));
        }
        let previous_record = self.records.insert(record.artifact.id.clone(), record);
        debug_assert!(previous_record.is_none());
        let previous_artifact = self
            .refit_artifacts
            .insert(artifact.artifact.id.clone(), artifact.clone());
        debug_assert!(previous_artifact.is_none());
        Ok(())
    }

    pub fn capture_refit_artifacts(
        &mut self,
        task: &NodeTask,
        result: &NodeResult,
    ) -> Result<Vec<RefitArtifactRecord>> {
        if task.phase != Phase::Refit {
            return Err(DagMlError::RuntimeValidation(format!(
                "cannot capture refit artifacts from phase {:?}",
                task.phase
            )));
        }
        let mut records = Vec::new();
        for artifact in &result.artifacts {
            let handle = result.artifact_handles.get(&artifact.id).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` without artifact handle",
                    task.node_plan.node_id, artifact.id
                ))
            })?;
            let record = RefitArtifactRecord {
                node_id: task.node_plan.node_id.clone(),
                controller_id: task.node_plan.controller_id.clone(),
                artifact: artifact.clone(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_requirement_keys: task
                    .node_plan
                    .data_bindings
                    .iter()
                    .map(|binding| format!("{}.{}", binding.node_id, binding.input_name))
                    .collect(),
                // Only the Validation-OOF meta-feature inputs become replay
                // prediction-cache requirements. The off-fold (REFIT/PREDICT)
                // test/predict base predictions are recomputed each phase, not
                // replayed from cache, and they share the same producer/port as the
                // Validation OOF input — so including them would duplicate the
                // requirement key. They are excluded here (partition != Validation).
                prediction_requirement_keys: task
                    .prediction_inputs
                    .values()
                    .filter(|spec| spec.partition == PredictionPartition::Validation)
                    .map(|spec| {
                        bundle_prediction_requirement_key(
                            &spec.producer_node,
                            &spec.source_port,
                            &task.node_plan.node_id,
                            &spec.target_port,
                        )
                    })
                    .collect(),
            };
            self.register(&record, handle.clone())?;
            records.push(record);
        }
        Ok(records)
    }

    pub fn get(&self, artifact_id: &ArtifactId) -> Option<&ArtifactHandleRecord> {
        self.records.get(artifact_id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn refit_artifacts(&self) -> Vec<RefitArtifactRecord> {
        self.refit_artifacts.values().cloned().collect()
    }
}

impl RuntimeArtifactStore for InMemoryArtifactStore {
    fn materialize(&self, request: &ArtifactMaterializationRequest) -> Result<HandleRef> {
        let record = self.records.get(&request.artifact.id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "artifact store is missing refit artifact `{}` for bundle `{}`",
                request.artifact.id, request.bundle_id
            ))
        })?;
        if record.node_id != request.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered for node `{}` but requested for `{}`",
                request.artifact.id, record.node_id, request.node_id
            )));
        }
        if record.controller_id != request.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered for controller `{}` but requested for `{}`",
                request.artifact.id, record.controller_id, request.controller_id
            )));
        }
        if record.artifact != request.artifact {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` metadata does not match bundle record",
                request.artifact.id
            )));
        }
        if record.params_fingerprint != request.params_fingerprint {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` params fingerprint does not match bundle record",
                request.artifact.id
            )));
        }
        record.validate()?;
        Ok(record.handle.clone())
    }
}

pub const FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const FILE_ARTIFACT_MANIFEST_FILE: &str = "artifact_manifest.json";

fn default_file_artifact_manifest_schema_version() -> u32 {
    FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION
}

/// One persisted artifact entry. Mirrors the bundle [`RefitArtifactRecord`]
/// identity (node, controller, artifact and params fingerprint) while requiring
/// the [`ArtifactRef`] to be portable so the manifest stays movable with its
/// payloads.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FileArtifactManifestEntry {
    pub node_id: NodeId,
    pub controller_id: ControllerId,
    pub artifact: ArtifactRef,
    pub params_fingerprint: String,
}

impl FileArtifactManifestEntry {
    fn from_refit_record(record: &RefitArtifactRecord) -> Result<Self> {
        let entry = Self {
            node_id: record.node_id.clone(),
            controller_id: record.controller_id.clone(),
            artifact: record.artifact.clone(),
            params_fingerprint: record.params_fingerprint.clone(),
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<()> {
        self.artifact.validate_portable()?;
        if self.artifact.controller_id != self.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact manifest entry `{}` controller `{}` does not match artifact controller `{}`",
                self.artifact.id, self.controller_id, self.artifact.controller_id
            )));
        }
        validate_runtime_fingerprint("artifact manifest params", &self.params_fingerprint)
    }

    fn matches_refit_record(&self, record: &RefitArtifactRecord) -> bool {
        self.node_id == record.node_id
            && self.controller_id == record.controller_id
            && self.artifact == record.artifact
            && self.params_fingerprint == record.params_fingerprint
    }
}

/// Versioned, file-backed artifact manifest. This is a manifest/portability
/// layer only: it records portable [`ArtifactRef`] metadata for a bundle's
/// refit artifacts. It does not deserialize ML objects or materialize artifact
/// payloads; payload stores remain future work.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FileArtifactManifest {
    pub bundle_id: BundleId,
    #[serde(default = "default_file_artifact_manifest_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub artifacts: Vec<FileArtifactManifestEntry>,
}

impl FileArtifactManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "file artifact manifest for bundle `{}` uses unsupported schema_version {}, expected {}",
                self.bundle_id, self.schema_version, FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION
            )));
        }
        let mut artifact_ids = BTreeSet::new();
        let mut uris = BTreeSet::new();
        for entry in &self.artifacts {
            entry.validate()?;
            if !artifact_ids.insert(entry.artifact.id.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "file artifact manifest for bundle `{}` has duplicate artifact id `{}`",
                    self.bundle_id, entry.artifact.id
                )));
            }
            // `entry.validate` guarantees a portable URI is present.
            if let Some(uri) = entry.artifact.uri.as_deref() {
                if !uris.insert(uri) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "file artifact manifest for bundle `{}` has duplicate artifact uri `{}`",
                        self.bundle_id, uri
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn validate_against_bundle(&self, bundle: &ExecutionBundle) -> Result<()> {
        self.validate()?;
        bundle.validate()?;
        if self.bundle_id != bundle.bundle_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "file artifact manifest bundle `{}` does not match bundle `{}`",
                self.bundle_id, bundle.bundle_id
            )));
        }
        if self.artifacts.len() != bundle.refit_artifacts.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "file artifact manifest for bundle `{}` has {} artifact(s) for {} bundle refit artifact(s)",
                self.bundle_id,
                self.artifacts.len(),
                bundle.refit_artifacts.len()
            )));
        }
        let entries_by_id = self
            .artifacts
            .iter()
            .map(|entry| (entry.artifact.id.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        for record in &bundle.refit_artifacts {
            let entry = entries_by_id
                .get(record.artifact.id.as_str())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "file artifact manifest for bundle `{}` is missing refit artifact `{}`",
                        self.bundle_id, record.artifact.id
                    ))
                })?;
            if !entry.matches_refit_record(record) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "file artifact manifest entry `{}` does not match bundle refit artifact",
                    entry.artifact.id
                )));
            }
        }
        Ok(())
    }
}

/// File-backed artifact manifest store rooted at a directory.
///
/// This is a portability/manifest layer: [`FileArtifactManifestStore::write`]
/// serializes portable artifact references from a validated bundle and
/// [`FileArtifactManifestStore::open`] reloads and revalidates them against the
/// bundle. It never reads, writes or deserializes artifact payloads.
#[derive(Clone, Debug)]
pub struct FileArtifactManifestStore {
    root: PathBuf,
    manifest: FileArtifactManifest,
}

impl FileArtifactManifestStore {
    pub fn write(root: impl AsRef<Path>, bundle: &ExecutionBundle) -> Result<FileArtifactManifest> {
        bundle.validate()?;
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "failed to create artifact manifest store `{}`: {err}",
                root.display()
            ))
        })?;
        let mut entries = Vec::with_capacity(bundle.refit_artifacts.len());
        for record in &bundle.refit_artifacts {
            entries.push(FileArtifactManifestEntry::from_refit_record(record)?);
        }
        entries.sort_by(|left, right| left.artifact.id.cmp(&right.artifact.id));
        let manifest = FileArtifactManifest {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            artifacts: entries,
        };
        manifest.validate_against_bundle(bundle)?;
        write_runtime_json(
            &root.join(FILE_ARTIFACT_MANIFEST_FILE),
            &manifest,
            "artifact manifest",
        )?;
        Ok(manifest)
    }

    pub fn open(root: impl Into<PathBuf>, bundle: &ExecutionBundle) -> Result<Self> {
        bundle.validate()?;
        let root = root.into();
        let manifest: FileArtifactManifest =
            read_runtime_json(&root.join(FILE_ARTIFACT_MANIFEST_FILE), "artifact manifest")?;
        manifest.validate_against_bundle(bundle)?;
        Ok(Self { root, manifest })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &FileArtifactManifest {
        &self.manifest
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactPayloadMaterializationRecord {
    pub run_id: RunId,
    pub bundle_id: BundleId,
    pub node_id: NodeId,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub artifact_id: ArtifactId,
    pub payload_uri: String,
    pub content_fingerprint: String,
    pub size_bytes: u64,
    pub handle: HandleRef,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactPayloadMetadata {
    uri: String,
    content_fingerprint: String,
    size_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct FileArtifactPayloadStore {
    root: PathBuf,
    manifest: FileArtifactManifest,
    records_by_artifact_id: BTreeMap<ArtifactId, RefitArtifactRecord>,
    materialization_records: RefCell<Vec<ArtifactPayloadMaterializationRecord>>,
}

impl FileArtifactPayloadStore {
    pub fn write_from_source(
        output_root: impl AsRef<Path>,
        source_root: impl AsRef<Path>,
        bundle: &ExecutionBundle,
    ) -> Result<Self> {
        bundle.validate()?;
        let output_root = output_root.as_ref();
        let source_root = source_root.as_ref();
        fs::create_dir_all(output_root).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "failed to create artifact payload store `{}`: {err}",
                output_root.display()
            ))
        })?;
        for record in &bundle.refit_artifacts {
            record.artifact.validate_portable()?;
            validate_artifact_payload_file(source_root, &record.artifact)?;
            let source_path = artifact_payload_path(source_root, &record.artifact)?;
            let output_path = artifact_payload_path(output_root, &record.artifact)?;
            if let Some(parent) = output_path.parent() {
                fs::create_dir_all(parent).map_err(|err| {
                    DagMlError::RuntimeValidation(format!(
                        "failed to create artifact payload directory `{}`: {err}",
                        parent.display()
                    ))
                })?;
            }
            if source_path != output_path {
                fs::copy(&source_path, &output_path).map_err(|err| {
                    DagMlError::RuntimeValidation(format!(
                        "failed to copy artifact payload `{}` from {} to {}: {err}",
                        record.artifact.id,
                        source_path.display(),
                        output_path.display()
                    ))
                })?;
            }
        }
        FileArtifactManifestStore::write(output_root, bundle)?;
        Self::open(output_root.to_path_buf(), bundle)
    }

    pub fn open(root: impl Into<PathBuf>, bundle: &ExecutionBundle) -> Result<Self> {
        bundle.validate()?;
        let root = root.into();
        let manifest_store = FileArtifactManifestStore::open(root.clone(), bundle)?;
        let records_by_artifact_id = bundle
            .refit_artifacts
            .iter()
            .cloned()
            .map(|record| (record.artifact.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let store = Self {
            root,
            manifest: manifest_store.manifest().clone(),
            records_by_artifact_id,
            materialization_records: RefCell::new(Vec::new()),
        };
        store.validate_payloads()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &FileArtifactManifest {
        &self.manifest
    }

    pub fn payload_count(&self) -> usize {
        self.manifest.artifacts.len()
    }

    pub fn materialization_records(&self) -> Vec<ArtifactPayloadMaterializationRecord> {
        self.materialization_records.borrow().clone()
    }

    pub fn validate_payloads(&self) -> Result<()> {
        self.manifest.validate()?;
        for entry in &self.manifest.artifacts {
            let record = self
                .records_by_artifact_id
                .get(&entry.artifact.id)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "artifact payload store for bundle `{}` has no bundle record for `{}`",
                        self.manifest.bundle_id, entry.artifact.id
                    ))
                })?;
            if !entry.matches_refit_record(record) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "artifact payload store entry `{}` does not match bundle refit artifact",
                    entry.artifact.id
                )));
            }
            validate_artifact_payload_file(&self.root, &entry.artifact)?;
        }
        Ok(())
    }
}

impl RuntimeArtifactStore for FileArtifactPayloadStore {
    fn materialize(&self, request: &ArtifactMaterializationRequest) -> Result<HandleRef> {
        request.artifact.validate_portable()?;
        let record = self
            .records_by_artifact_id
            .get(&request.artifact.id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "artifact payload store is missing refit artifact `{}` for bundle `{}`",
                    request.artifact.id, request.bundle_id
                ))
            })?;
        if record.node_id != request.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered for node `{}` but requested for `{}`",
                request.artifact.id, record.node_id, request.node_id
            )));
        }
        if record.controller_id != request.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered for controller `{}` but requested for `{}`",
                request.artifact.id, record.controller_id, request.controller_id
            )));
        }
        if record.artifact != request.artifact {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` metadata does not match bundle record",
                request.artifact.id
            )));
        }
        if record.params_fingerprint != request.params_fingerprint {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` params fingerprint does not match bundle record",
                request.artifact.id
            )));
        }
        let metadata = validate_artifact_payload_file(&self.root, &request.artifact)?;
        let fingerprint = stable_json_fingerprint(&(
            &request.run_id,
            &request.bundle_id,
            &request.node_id,
            request.phase,
            &request.variant_id,
            &request.artifact.id,
            &metadata.content_fingerprint,
            &request.params_fingerprint,
        ))?;
        let handle = HandleRef {
            handle: u64::from_str_radix(&fingerprint[..16], 16)
                .expect("sha256 hex prefix should fit into u64"),
            kind: HandleKind::Artifact,
            owner_controller: request.controller_id.clone(),
        };
        self.materialization_records
            .borrow_mut()
            .push(ArtifactPayloadMaterializationRecord {
                run_id: request.run_id.clone(),
                bundle_id: request.bundle_id.clone(),
                node_id: request.node_id.clone(),
                phase: request.phase,
                variant_id: request.variant_id.clone(),
                artifact_id: request.artifact.id.clone(),
                payload_uri: metadata.uri,
                content_fingerprint: metadata.content_fingerprint,
                size_bytes: metadata.size_bytes,
                handle: handle.clone(),
            });
        Ok(handle)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineageRecord {
    pub record_id: LineageId,
    pub run_id: RunId,
    pub node_id: NodeId,
    pub phase: Phase,
    pub controller_id: ControllerId,
    pub controller_version: String,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub branch_path: Vec<BranchId>,
    #[serde(default)]
    pub input_lineage: Vec<LineageId>,
    #[serde(default)]
    pub artifact_refs: Vec<ArtifactRef>,
    pub params_fingerprint: String,
    pub data_model_shape_fingerprint: Option<String>,
    pub aggregation_policy_fingerprint: Option<String>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub unsafe_flags: BTreeSet<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
}

impl LineageRecord {
    pub fn validate(&self) -> Result<()> {
        if self.params_fingerprint.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage `{}` has empty params fingerprint",
                self.record_id
            )));
        }
        for artifact in &self.artifact_refs {
            artifact.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryLineageRecorder {
    records: BTreeMap<LineageId, LineageRecord>,
}

impl InMemoryLineageRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: LineageRecord) -> Result<()> {
        record.validate()?;
        if self
            .records
            .insert(record.record_id.clone(), record)
            .is_some()
        {
            return Err(DagMlError::RuntimeValidation(
                "duplicate lineage record id".to_string(),
            ));
        }
        Ok(())
    }

    pub fn get(&self, id: &LineageId) -> Option<&LineageRecord> {
        self.records.get(id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> impl Iterator<Item = &LineageRecord> {
        self.records.values()
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryPredictionStore {
    blocks: Vec<PredictionBlock>,
}

impl InMemoryPredictionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, block: PredictionBlock) -> Result<()> {
        block.validate_content()?;
        self.blocks.push(block);
        Ok(())
    }

    pub fn blocks(&self) -> &[PredictionBlock] {
        &self.blocks
    }

    pub fn find(
        &self,
        producer_node: Option<&NodeId>,
        phase_partition: Option<&crate::oof::PredictionPartition>,
        fold_id: Option<&FoldId>,
    ) -> Vec<&PredictionBlock> {
        self.blocks
            .iter()
            .filter(|block| {
                producer_node.is_none_or(|node_id| &block.producer_node == node_id)
                    && phase_partition.is_none_or(|partition| &block.partition == partition)
                    && fold_id.is_none_or(|requested| block.fold_id.as_ref() == Some(requested))
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryAggregatedPredictionStore {
    blocks: Vec<AggregatedPredictionBlock>,
}

impl InMemoryAggregatedPredictionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, block: AggregatedPredictionBlock) -> Result<()> {
        block.validate_shape()?;
        self.blocks.push(block);
        Ok(())
    }

    pub fn blocks(&self) -> &[AggregatedPredictionBlock] {
        &self.blocks
    }

    pub fn find(
        &self,
        producer_node: Option<&NodeId>,
        phase_partition: Option<&PredictionPartition>,
        fold_id: Option<&FoldId>,
        prediction_level: Option<PredictionLevel>,
    ) -> Vec<&AggregatedPredictionBlock> {
        self.blocks
            .iter()
            .filter(|block| {
                producer_node.is_none_or(|node_id| &block.producer_node == node_id)
                    && phase_partition.is_none_or(|partition| &block.partition == partition)
                    && fold_id.is_none_or(|requested| block.fold_id.as_ref() == Some(requested))
                    && prediction_level.is_none_or(|level| block.level == level)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionCacheMaterializationRequest {
    pub run_id: RunId,
    pub bundle_id: BundleId,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub requirement: BundlePredictionRequirement,
    pub cache: BundlePredictionCacheRecord,
    pub producer_controller_id: ControllerId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionCacheMaterializationRecord {
    pub run_id: RunId,
    pub bundle_id: BundleId,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub requirement_key: String,
    pub cache_id: String,
    pub handle: HandleRef,
}

pub trait RuntimePredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> Result<Vec<PredictionBlock>>;
    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> Result<Vec<AggregatedPredictionBlock>> {
        Err(DagMlError::RuntimeValidation(format!(
            "prediction cache store does not support aggregated requirement `{requirement_key}`"
        )))
    }
    fn materialize(&self, request: &PredictionCacheMaterializationRequest) -> Result<HandleRef>;
}

pub const FILE_PREDICTION_CACHE_STORE_SCHEMA_VERSION: u32 = 1;
pub const FILE_PREDICTION_CACHE_MANIFEST_FILE: &str = "prediction_cache_manifest.json";

fn default_file_prediction_cache_store_schema_version() -> u32 {
    FILE_PREDICTION_CACHE_STORE_SCHEMA_VERSION
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePredictionCacheEntry {
    pub requirement_key: String,
    pub cache_id: String,
    pub file_name: String,
    #[serde(default = "default_runtime_prediction_level")]
    pub prediction_level: PredictionLevel,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unit_ids: Vec<PredictionUnitId>,
    pub block_count: usize,
    pub row_count: usize,
    pub content_fingerprint: String,
}

impl FilePredictionCacheEntry {
    pub fn validate(&self) -> Result<()> {
        validate_runtime_non_empty("requirement_key", &self.requirement_key)?;
        validate_runtime_non_empty("cache_id", &self.cache_id)?;
        validate_runtime_non_empty("file_name", &self.file_name)?;
        validate_prediction_cache_file_name(&self.file_name)?;
        if self.block_count == 0 {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache `{}` has zero block_count",
                self.cache_id
            )));
        }
        if self.row_count == 0 {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache `{}` has zero row_count",
                self.cache_id
            )));
        }
        if self.prediction_level != PredictionLevel::Sample && self.unit_ids.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache `{}` has no aggregated unit ids",
                self.cache_id
            )));
        }
        if self
            .unit_ids
            .iter()
            .any(|unit_id| unit_id.level() != self.prediction_level)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache `{}` has unit ids outside {:?}",
                self.cache_id, self.prediction_level
            )));
        }
        validate_runtime_fingerprint("prediction cache content", &self.content_fingerprint)
    }

    fn from_payload(payload: &crate::bundle::BundlePredictionCachePayload) -> Result<Self> {
        Ok(Self {
            requirement_key: payload.requirement_key.clone(),
            cache_id: payload.cache_id.clone(),
            file_name: prediction_cache_payload_file_name(payload)?,
            prediction_level: payload.prediction_level,
            unit_ids: payload
                .aggregated_blocks
                .iter()
                .flat_map(|block| block.unit_ids.iter().cloned())
                .collect(),
            block_count: payload.block_count,
            row_count: payload.row_count,
            content_fingerprint: payload.content_fingerprint.clone(),
        })
    }

    fn matches_record(&self, record: &BundlePredictionCacheRecord) -> bool {
        self.requirement_key == record.requirement_key
            && self.cache_id == record.cache_id
            && self.prediction_level == record.prediction_level
            && self.unit_ids == record.unit_ids
            && self.block_count == record.block_count
            && self.row_count == record.row_count
            && self.content_fingerprint == record.content_fingerprint
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePredictionCacheManifest {
    pub bundle_id: BundleId,
    #[serde(default = "default_file_prediction_cache_store_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub caches: Vec<FilePredictionCacheEntry>,
}

impl FilePredictionCacheManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != FILE_PREDICTION_CACHE_STORE_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache manifest for bundle `{}` uses unsupported schema_version {}, expected {}",
                self.bundle_id,
                self.schema_version,
                FILE_PREDICTION_CACHE_STORE_SCHEMA_VERSION
            )));
        }
        let mut requirement_keys = BTreeSet::new();
        let mut cache_ids = BTreeSet::new();
        let mut file_names = BTreeSet::new();
        for entry in &self.caches {
            entry.validate()?;
            if !requirement_keys.insert(entry.requirement_key.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "file prediction cache manifest for bundle `{}` has duplicate requirement `{}`",
                    self.bundle_id, entry.requirement_key
                )));
            }
            if !cache_ids.insert(entry.cache_id.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "file prediction cache manifest for bundle `{}` has duplicate cache id `{}`",
                    self.bundle_id, entry.cache_id
                )));
            }
            if !file_names.insert(entry.file_name.as_str()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "file prediction cache manifest for bundle `{}` has duplicate file `{}`",
                    self.bundle_id, entry.file_name
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
                "file prediction cache manifest bundle `{}` does not match bundle `{}`",
                self.bundle_id, bundle.bundle_id
            )));
        }
        if self.caches.len() != bundle.prediction_caches.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache manifest for bundle `{}` has {} cache(s) for {} bundle cache record(s)",
                self.bundle_id,
                self.caches.len(),
                bundle.prediction_caches.len()
            )));
        }
        let entries_by_requirement = self
            .caches
            .iter()
            .map(|entry| (entry.requirement_key.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        for record in &bundle.prediction_caches {
            let entry = entries_by_requirement
                .get(record.requirement_key.as_str())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "file prediction cache manifest for bundle `{}` is missing requirement `{}`",
                        self.bundle_id, record.requirement_key
                    ))
                })?;
            if !entry.matches_record(record) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "file prediction cache manifest entry `{}` does not match bundle cache record",
                    entry.cache_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FilePredictionCacheStore {
    root: PathBuf,
    manifest: FilePredictionCacheManifest,
    records_by_requirement: BTreeMap<String, BundlePredictionCacheRecord>,
    materialization_records: RefCell<Vec<PredictionCacheMaterializationRecord>>,
}

impl FilePredictionCacheStore {
    pub fn write_payload_set(
        root: impl AsRef<Path>,
        bundle: &ExecutionBundle,
        payloads: &BundlePredictionCachePayloadSet,
    ) -> Result<FilePredictionCacheManifest> {
        payloads.validate_against_bundle(bundle)?;
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "failed to create prediction cache store `{}`: {err}",
                root.display()
            ))
        })?;

        let mut entries = Vec::new();
        let records_by_requirement = bundle
            .prediction_caches
            .iter()
            .map(|record| (record.requirement_key.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        for payload in &payloads.caches {
            let record = records_by_requirement
                .get(payload.requirement_key.as_str())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "prediction cache payload `{}` references unknown requirement `{}`",
                        payload.cache_id, payload.requirement_key
                    ))
                })?;
            validate_prediction_cache_payload_matches_record(payload, record)?;
            let entry = FilePredictionCacheEntry::from_payload(payload)?;
            let payload_path = root.join(&entry.file_name);
            write_runtime_json(&payload_path, payload, "prediction cache payload")?;
            entries.push(entry);
        }
        entries.sort_by(|left, right| left.requirement_key.cmp(&right.requirement_key));
        let manifest = FilePredictionCacheManifest {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: FILE_PREDICTION_CACHE_STORE_SCHEMA_VERSION,
            caches: entries,
        };
        manifest.validate_against_bundle(bundle)?;
        write_runtime_json(
            &root.join(FILE_PREDICTION_CACHE_MANIFEST_FILE),
            &manifest,
            "prediction cache manifest",
        )?;
        Ok(manifest)
    }

    pub fn open(root: impl Into<PathBuf>, bundle: &ExecutionBundle) -> Result<Self> {
        bundle.validate()?;
        let root = root.into();
        let manifest: FilePredictionCacheManifest = read_runtime_json(
            &root.join(FILE_PREDICTION_CACHE_MANIFEST_FILE),
            "prediction cache manifest",
        )?;
        manifest.validate_against_bundle(bundle)?;
        let records_by_requirement = bundle
            .prediction_caches
            .iter()
            .cloned()
            .map(|record| (record.requirement_key.clone(), record))
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            root,
            manifest,
            records_by_requirement,
            materialization_records: RefCell::new(Vec::new()),
        })
    }

    pub fn manifest(&self) -> &FilePredictionCacheManifest {
        &self.manifest
    }

    pub fn materialization_records(&self) -> Vec<PredictionCacheMaterializationRecord> {
        self.materialization_records.borrow().clone()
    }

    fn payload_for_requirement(
        &self,
        requirement_key: &str,
    ) -> Result<crate::bundle::BundlePredictionCachePayload> {
        let entry = self
            .manifest
            .caches
            .iter()
            .find(|entry| entry.requirement_key == requirement_key)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "file prediction cache store is missing requirement `{requirement_key}`"
                ))
            })?;
        let record = self
            .records_by_requirement
            .get(requirement_key)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "file prediction cache store has no bundle record for requirement `{requirement_key}`"
                ))
            })?;
        let payload: crate::bundle::BundlePredictionCachePayload = read_runtime_json(
            &self.root.join(&entry.file_name),
            "prediction cache payload",
        )?;
        validate_prediction_cache_payload_matches_record(&payload, record)?;
        Ok(payload)
    }
}

impl RuntimePredictionCacheStore for FilePredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> Result<Vec<PredictionBlock>> {
        let payload = self.payload_for_requirement(requirement_key)?;
        if payload.prediction_level != PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache store requirement `{requirement_key}` contains {:?} predictions, not sample blocks",
                payload.prediction_level
            )));
        }
        Ok(payload.blocks)
    }

    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> Result<Vec<AggregatedPredictionBlock>> {
        let payload = self.payload_for_requirement(requirement_key)?;
        if payload.prediction_level == PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache store requirement `{requirement_key}` contains sample predictions, not aggregated blocks"
            )));
        }
        Ok(payload.aggregated_blocks)
    }

    fn materialize(&self, request: &PredictionCacheMaterializationRequest) -> Result<HandleRef> {
        request.requirement.validate()?;
        request.cache.validate()?;
        let requirement_key = request.requirement.key();
        let record = self
            .records_by_requirement
            .get(&requirement_key)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "file prediction cache store is missing requirement `{requirement_key}`"
                ))
            })?;
        if record != &request.cache {
            return Err(DagMlError::RuntimeValidation(format!(
                "file prediction cache materialization request for `{requirement_key}` does not match bundle cache record"
            )));
        }
        let payload = self.payload_for_requirement(&requirement_key)?;
        validate_prediction_cache_payload_matches_record(&payload, record)?;
        let fingerprint = stable_json_fingerprint(&(
            &request.run_id,
            &request.bundle_id,
            request.phase,
            &request.variant_id,
            &request.cache.requirement_key,
            &request.cache.cache_id,
            request.cache.prediction_level,
            &request.cache.content_fingerprint,
        ))?;
        let handle = HandleRef {
            handle: u64::from_str_radix(&fingerprint[..16], 16)
                .expect("sha256 hex prefix should fit into u64"),
            kind: HandleKind::Prediction,
            owner_controller: request.producer_controller_id.clone(),
        };
        self.materialization_records
            .borrow_mut()
            .push(PredictionCacheMaterializationRecord {
                run_id: request.run_id.clone(),
                bundle_id: request.bundle_id.clone(),
                phase: request.phase,
                variant_id: request.variant_id.clone(),
                requirement_key,
                cache_id: request.cache.cache_id.clone(),
                handle: handle.clone(),
            });
        Ok(handle)
    }
}

fn prediction_cache_payload_file_name(
    payload: &crate::bundle::BundlePredictionCachePayload,
) -> Result<String> {
    let fingerprint = stable_json_fingerprint(&(
        &payload.requirement_key,
        &payload.cache_id,
        payload.prediction_level,
        &payload.content_fingerprint,
        payload.block_count,
        payload.row_count,
    ))?;
    Ok(format!("prediction-cache-{}.json", &fingerprint[..16]))
}

fn validate_prediction_cache_file_name(file_name: &str) -> Result<()> {
    if file_name == "." || file_name == ".." || file_name.contains('/') || file_name.contains('\\')
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache file name `{file_name}` must be a plain file name"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColumnarPredictionCacheBlock {
    pub prediction_id: Option<String>,
    pub producer_node: NodeId,
    pub partition: PredictionPartition,
    pub fold_id: Option<FoldId>,
    pub prediction_level: PredictionLevel,
    pub unit_ids: Vec<PredictionUnitId>,
    pub sample_ids: Vec<SampleId>,
    pub target_names: Vec<String>,
    pub width: usize,
    pub columns: Vec<Vec<f64>>,
}

impl ColumnarPredictionCacheBlock {
    pub fn from_prediction_block(block: &PredictionBlock) -> Result<Self> {
        let width = block.validate_shape()?;
        let mut columns = vec![Vec::with_capacity(block.values.len()); width];
        for row in &block.values {
            for (column_idx, value) in row.iter().enumerate() {
                columns[column_idx].push(*value);
            }
        }
        Ok(Self {
            prediction_id: block.prediction_id.clone(),
            producer_node: block.producer_node.clone(),
            partition: block.partition.clone(),
            fold_id: block.fold_id.clone(),
            prediction_level: PredictionLevel::Sample,
            unit_ids: Vec::new(),
            sample_ids: block.sample_ids.clone(),
            target_names: block.target_names.clone(),
            width,
            columns,
        })
    }

    pub fn from_aggregated_prediction_block(block: &AggregatedPredictionBlock) -> Result<Self> {
        let width = block.validate_shape()?;
        if block.level == PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar aggregated prediction block for `{}` must use target/group level, got sample",
                block.producer_node
            )));
        }
        let mut columns = vec![Vec::with_capacity(block.values.len()); width];
        for row in &block.values {
            for (column_idx, value) in row.iter().enumerate() {
                columns[column_idx].push(*value);
            }
        }
        Ok(Self {
            prediction_id: block.prediction_id.clone(),
            producer_node: block.producer_node.clone(),
            partition: block.partition.clone(),
            fold_id: block.fold_id.clone(),
            prediction_level: block.level,
            unit_ids: block.unit_ids.clone(),
            sample_ids: Vec::new(),
            target_names: block.target_names.clone(),
            width,
            columns,
        })
    }

    pub fn row_count(&self) -> usize {
        match self.prediction_level {
            PredictionLevel::Sample => self.sample_ids.len(),
            PredictionLevel::Target | PredictionLevel::Group => self.unit_ids.len(),
            PredictionLevel::Observation => 0,
        }
    }

    pub fn value_count(&self) -> usize {
        self.columns.iter().map(Vec::len).sum()
    }

    pub fn validate(&self) -> Result<()> {
        match self.prediction_level {
            PredictionLevel::Observation => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "columnar prediction block for `{}` cannot store observation-level predictions",
                    self.producer_node
                )));
            }
            PredictionLevel::Sample => {
                if self.sample_ids.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "columnar sample prediction block for `{}` has no sample ids",
                        self.producer_node
                    )));
                }
                if !self.unit_ids.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "columnar sample prediction block for `{}` unexpectedly carries unit ids",
                        self.producer_node
                    )));
                }
            }
            PredictionLevel::Target | PredictionLevel::Group => {
                if !self.sample_ids.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "columnar aggregated prediction block for `{}` unexpectedly carries sample ids",
                        self.producer_node
                    )));
                }
                if self.unit_ids.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "columnar aggregated prediction block for `{}` has no unit ids",
                        self.producer_node
                    )));
                }
                if self
                    .unit_ids
                    .iter()
                    .any(|unit_id| unit_id.level() != self.prediction_level)
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "columnar aggregated prediction block for `{}` carries unit ids outside {:?}",
                        self.producer_node, self.prediction_level
                    )));
                }
            }
        }
        if self.width == 0 {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction block for `{}` has zero width",
                self.producer_node
            )));
        }
        if self.columns.len() != self.width {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction block for `{}` has {} column(s), expected {}",
                self.producer_node,
                self.columns.len(),
                self.width
            )));
        }
        for (column_idx, column) in self.columns.iter().enumerate() {
            if column.len() != self.row_count() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "columnar prediction block for `{}` column {} has {} value(s), expected {}",
                    self.producer_node,
                    column_idx,
                    column.len(),
                    self.row_count()
                )));
            }
        }
        if !self.target_names.is_empty() && self.target_names.len() != self.width {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction block for `{}` has {} target names for width {}",
                self.producer_node,
                self.target_names.len(),
                self.width
            )));
        }
        Ok(())
    }

    pub fn to_prediction_block(&self) -> Result<PredictionBlock> {
        self.validate()?;
        if self.prediction_level != PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction block for `{}` contains {:?} predictions, not sample predictions",
                self.producer_node, self.prediction_level
            )));
        }
        let values = (0..self.row_count())
            .map(|row_idx| {
                self.columns
                    .iter()
                    .map(|column| column[row_idx])
                    .collect::<Vec<_>>()
            })
            .collect();
        let block = PredictionBlock {
            prediction_id: self.prediction_id.clone(),
            producer_node: self.producer_node.clone(),
            partition: self.partition.clone(),
            fold_id: self.fold_id.clone(),
            sample_ids: self.sample_ids.clone(),
            values,
            target_names: self.target_names.clone(),
        };
        block.validate_shape()?;
        Ok(block)
    }

    pub fn to_aggregated_prediction_block(&self) -> Result<AggregatedPredictionBlock> {
        self.validate()?;
        if self.prediction_level == PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction block for `{}` contains sample predictions, not aggregated predictions",
                self.producer_node
            )));
        }
        let values = (0..self.row_count())
            .map(|row_idx| {
                self.columns
                    .iter()
                    .map(|column| column[row_idx])
                    .collect::<Vec<_>>()
            })
            .collect();
        let block = AggregatedPredictionBlock {
            prediction_id: self.prediction_id.clone(),
            producer_node: self.producer_node.clone(),
            partition: self.partition.clone(),
            fold_id: self.fold_id.clone(),
            level: self.prediction_level,
            unit_ids: self.unit_ids.clone(),
            values,
            target_names: self.target_names.clone(),
        };
        block.validate_shape()?;
        Ok(block)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnarPredictionCacheManifest {
    pub requirement_key: String,
    pub cache_id: String,
    pub prediction_level: PredictionLevel,
    pub block_count: usize,
    pub row_count: usize,
    pub prediction_width: usize,
    pub value_count: usize,
    pub estimated_value_bytes: usize,
    pub content_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq)]
struct ColumnarPredictionCacheEntry {
    cache: BundlePredictionCacheRecord,
    blocks: Vec<ColumnarPredictionCacheBlock>,
}

impl ColumnarPredictionCacheEntry {
    fn from_payload(
        payload: BundlePredictionCachePayload,
        cache: BundlePredictionCacheRecord,
    ) -> Result<Self> {
        validate_prediction_cache_payload_matches_record(&payload, &cache)?;
        let blocks = match payload.prediction_level {
            PredictionLevel::Sample => payload
                .blocks
                .iter()
                .map(ColumnarPredictionCacheBlock::from_prediction_block)
                .collect::<Result<Vec<_>>>()?,
            PredictionLevel::Target | PredictionLevel::Group => payload
                .aggregated_blocks
                .iter()
                .map(ColumnarPredictionCacheBlock::from_aggregated_prediction_block)
                .collect::<Result<Vec<_>>>()?,
            PredictionLevel::Observation => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "columnar prediction cache payload `{}` cannot use observation-level predictions",
                    payload.cache_id
                )));
            }
        };
        let entry = Self { cache, blocks };
        entry.validate()?;
        Ok(entry)
    }

    fn validate(&self) -> Result<()> {
        self.cache.validate()?;
        if self.blocks.len() != self.cache.block_count {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache `{}` has {} block(s), expected {}",
                self.cache.cache_id,
                self.blocks.len(),
                self.cache.block_count
            )));
        }
        let mut row_count = 0usize;
        let mut value_count = 0usize;
        for block in &self.blocks {
            block.validate()?;
            if block.prediction_level != self.cache.prediction_level {
                return Err(DagMlError::RuntimeValidation(format!(
                    "columnar prediction cache `{}` contains a {:?} block, expected {:?}",
                    self.cache.cache_id, block.prediction_level, self.cache.prediction_level
                )));
            }
            if block.partition != self.cache.partition {
                return Err(DagMlError::RuntimeValidation(format!(
                    "columnar prediction cache `{}` contains a block from partition {:?}",
                    self.cache.cache_id, block.partition
                )));
            }
            row_count += block.row_count();
            value_count += block.value_count();
        }
        if row_count != self.cache.row_count {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache `{}` has {} row(s), expected {}",
                self.cache.cache_id, row_count, self.cache.row_count
            )));
        }
        let expected_values = self
            .cache
            .row_count
            .checked_mul(self.cache.prediction_width)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "columnar prediction cache `{}` value count overflow",
                    self.cache.cache_id
                ))
            })?;
        if value_count != expected_values {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache `{}` has {} value(s), expected {}",
                self.cache.cache_id, value_count, expected_values
            )));
        }
        Ok(())
    }

    fn to_blocks(&self) -> Result<Vec<PredictionBlock>> {
        self.validate()?;
        self.blocks
            .iter()
            .map(ColumnarPredictionCacheBlock::to_prediction_block)
            .collect()
    }

    fn to_aggregated_blocks(&self) -> Result<Vec<AggregatedPredictionBlock>> {
        self.validate()?;
        self.blocks
            .iter()
            .map(ColumnarPredictionCacheBlock::to_aggregated_prediction_block)
            .collect()
    }

    fn validate_against_cache_record(&self, cache: &BundlePredictionCacheRecord) -> Result<()> {
        if &self.cache != cache {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache materialization request for `{}` does not match bundle cache record",
                cache.requirement_key
            )));
        }
        let (blocks, aggregated_blocks) = match self.cache.prediction_level {
            PredictionLevel::Sample => (self.to_blocks()?, Vec::new()),
            PredictionLevel::Target | PredictionLevel::Group => {
                (Vec::new(), self.to_aggregated_blocks()?)
            }
            PredictionLevel::Observation => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "columnar prediction cache `{}` cannot materialize observation-level predictions",
                    self.cache.cache_id
                )));
            }
        };
        let payload = BundlePredictionCachePayload {
            requirement_key: self.cache.requirement_key.clone(),
            cache_id: self.cache.cache_id.clone(),
            format: self.cache.format.clone(),
            partition: self.cache.partition.clone(),
            prediction_level: self.cache.prediction_level,
            block_count: self.cache.block_count,
            row_count: self.cache.row_count,
            content_fingerprint: self.cache.content_fingerprint.clone(),
            blocks,
            aggregated_blocks,
        };
        validate_prediction_cache_payload_matches_record(&payload, cache)
    }

    fn manifest(&self) -> ColumnarPredictionCacheManifest {
        let value_count = self
            .blocks
            .iter()
            .map(ColumnarPredictionCacheBlock::value_count)
            .sum::<usize>();
        ColumnarPredictionCacheManifest {
            requirement_key: self.cache.requirement_key.clone(),
            cache_id: self.cache.cache_id.clone(),
            prediction_level: self.cache.prediction_level,
            block_count: self.cache.block_count,
            row_count: self.cache.row_count,
            prediction_width: self.cache.prediction_width,
            value_count,
            estimated_value_bytes: value_count * std::mem::size_of::<f64>(),
            content_fingerprint: self.cache.content_fingerprint.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ColumnarPredictionCacheStore {
    entries: BTreeMap<String, ColumnarPredictionCacheEntry>,
    materialization_records: RefCell<Vec<PredictionCacheMaterializationRecord>>,
}

impl ColumnarPredictionCacheStore {
    pub fn from_payloads(
        bundle: &ExecutionBundle,
        payloads: BundlePredictionCachePayloadSet,
    ) -> Result<Self> {
        payloads.validate_against_bundle(bundle)?;
        let records_by_requirement = bundle
            .prediction_caches
            .iter()
            .cloned()
            .map(|cache| (cache.requirement_key.clone(), cache))
            .collect::<BTreeMap<_, _>>();
        let mut entries = BTreeMap::new();
        for payload in payloads.caches {
            let cache = records_by_requirement
                .get(&payload.requirement_key)
                .cloned()
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "columnar prediction cache payload `{}` references unknown requirement `{}`",
                        payload.cache_id, payload.requirement_key
                    ))
                })?;
            let requirement_key = payload.requirement_key.clone();
            let previous = entries.insert(
                requirement_key,
                ColumnarPredictionCacheEntry::from_payload(payload, cache)?,
            );
            debug_assert!(previous.is_none());
        }
        Ok(Self {
            entries,
            materialization_records: RefCell::new(Vec::new()),
        })
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn manifests(&self) -> Vec<ColumnarPredictionCacheManifest> {
        self.entries
            .values()
            .map(ColumnarPredictionCacheEntry::manifest)
            .collect()
    }

    pub fn materialization_records(&self) -> Vec<PredictionCacheMaterializationRecord> {
        self.materialization_records.borrow().clone()
    }
}

impl RuntimePredictionCacheStore for ColumnarPredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> Result<Vec<PredictionBlock>> {
        let entry = self.entries.get(requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "columnar prediction cache store is missing requirement `{requirement_key}`"
            ))
        })?;
        if entry.cache.prediction_level != PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache store requirement `{requirement_key}` contains {:?} predictions, not sample blocks",
                entry.cache.prediction_level
            )));
        }
        entry.validate_against_cache_record(&entry.cache)?;
        entry.to_blocks()
    }

    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> Result<Vec<AggregatedPredictionBlock>> {
        let entry = self.entries.get(requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "columnar prediction cache store is missing requirement `{requirement_key}`"
            ))
        })?;
        if entry.cache.prediction_level == PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache store requirement `{requirement_key}` contains sample predictions, not aggregated blocks"
            )));
        }
        entry.validate_against_cache_record(&entry.cache)?;
        entry.to_aggregated_blocks()
    }

    fn materialize(&self, request: &PredictionCacheMaterializationRequest) -> Result<HandleRef> {
        request.requirement.validate()?;
        request.cache.validate()?;
        let requirement_key = request.requirement.key();
        if requirement_key != request.cache.requirement_key {
            return Err(DagMlError::RuntimeValidation(format!(
                "columnar prediction cache materialization request for `{}` uses cache `{}` with mismatched requirement `{}`",
                requirement_key, request.cache.cache_id, request.cache.requirement_key
            )));
        }
        let entry = self.entries.get(&requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "columnar prediction cache store is missing requirement `{requirement_key}`"
            ))
        })?;
        entry.validate_against_cache_record(&request.cache)?;
        let fingerprint = stable_json_fingerprint(&(
            &request.run_id,
            &request.bundle_id,
            request.phase,
            &request.variant_id,
            &request.cache.requirement_key,
            &request.cache.cache_id,
            request.cache.prediction_level,
            &request.cache.content_fingerprint,
        ))?;
        let handle = HandleRef {
            handle: u64::from_str_radix(&fingerprint[..16], 16)
                .expect("sha256 hex prefix should fit into u64"),
            kind: HandleKind::Prediction,
            owner_controller: request.producer_controller_id.clone(),
        };
        self.materialization_records
            .borrow_mut()
            .push(PredictionCacheMaterializationRecord {
                run_id: request.run_id.clone(),
                bundle_id: request.bundle_id.clone(),
                phase: request.phase,
                variant_id: request.variant_id.clone(),
                requirement_key,
                cache_id: request.cache.cache_id.clone(),
                handle: handle.clone(),
            });
        Ok(handle)
    }
}

fn validate_runtime_non_empty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(DagMlError::RuntimeValidation(format!("{label} is empty")));
    }
    Ok(())
}

fn validate_artifact_optional_text(
    label: &str,
    value: &Option<String>,
    artifact_id: &ArtifactId,
) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact `{artifact_id}` has empty {label}"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact `{artifact_id}` has control characters in {label}"
        )));
    }
    Ok(())
}

fn artifact_payload_path(root: &Path, artifact: &ArtifactRef) -> Result<PathBuf> {
    artifact.validate_portable()?;
    let uri = artifact
        .uri
        .as_deref()
        .expect("portable artifact validation requires uri");
    Ok(root.join(uri))
}

fn validate_artifact_payload_file(
    root: &Path,
    artifact: &ArtifactRef,
) -> Result<ArtifactPayloadMetadata> {
    artifact.validate_portable()?;
    let uri = artifact
        .uri
        .as_deref()
        .expect("portable artifact validation requires uri")
        .to_string();
    let path = artifact_payload_path(root, artifact)?;
    validate_payload_path_stays_within_root(root, &path, artifact)?;
    let metadata = fs::metadata(&path).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to stat artifact payload `{}` at {}: {err}",
            artifact.id,
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact payload `{}` at {} is not a regular file",
            artifact.id,
            path.display()
        )));
    }
    let size_bytes = metadata.len();
    if let Some(expected_size) = artifact.size_bytes {
        if expected_size != size_bytes {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact payload `{}` size mismatch: expected {}, got {}",
                artifact.id, expected_size, size_bytes
            )));
        }
    }
    let content_fingerprint =
        sha256_file_hex(&path, &format!("artifact payload `{}`", artifact.id))?;
    let expected_fingerprint = artifact
        .content_fingerprint
        .as_deref()
        .expect("portable artifact validation requires content_fingerprint");
    if !content_fingerprint.eq_ignore_ascii_case(expected_fingerprint) {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact payload `{}` content fingerprint mismatch",
            artifact.id
        )));
    }
    Ok(ArtifactPayloadMetadata {
        uri,
        content_fingerprint,
        size_bytes,
    })
}

fn validate_payload_path_stays_within_root(
    root: &Path,
    path: &Path,
    artifact: &ArtifactRef,
) -> Result<()> {
    let root = fs::canonicalize(root).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to canonicalize artifact payload root `{}`: {err}",
            root.display()
        ))
    })?;
    let path = fs::canonicalize(path).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to canonicalize artifact payload `{}` at {}: {err}",
            artifact.id,
            path.display()
        ))
    })?;
    if !path.starts_with(&root) {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact payload `{}` resolves outside store root `{}`",
            artifact.id,
            root.display()
        )));
    }
    Ok(())
}

fn sha256_file_hex(path: &Path, label: &str) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to open {label} at {}: {err}",
            path.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "failed to read {label} at {}: {err}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(bytes_to_hex(&hasher.finalize()))
}

#[cfg(test)]
fn sha256_bytes_hex(bytes: &[u8]) -> String {
    bytes_to_hex(&Sha256::digest(bytes))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

/// Deterministic path safety for relative artifact URIs. Rejects empty values,
/// control characters, absolute paths (POSIX root, Windows root or drive
/// prefix), URI schemes such as `http://`, `s3://` or `file://` (any colon in
/// the leading path segment) and any `..` traversal component. Parsing is
/// platform-independent so portable manifests validate identically everywhere;
/// it adds no dependency.
fn validate_relative_artifact_uri(artifact_id: &ArtifactId, uri: &str) -> Result<()> {
    if uri.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact `{artifact_id}` has empty uri"
        )));
    }
    if uri.chars().any(char::is_control) {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact `{artifact_id}` uri has control characters"
        )));
    }
    if uri.starts_with('/') || uri.starts_with('\\') {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact `{artifact_id}` uri `{uri}` must be a relative path"
        )));
    }
    let mut prefix = uri.chars();
    if let (Some(drive), Some(':')) = (prefix.next(), prefix.next()) {
        if drive.is_ascii_alphabetic() {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{artifact_id}` uri `{uri}` must be a relative path"
            )));
        }
    }
    // Reject URI schemes (`http://`, `s3://`, `file://`, ...) and any other
    // colon in the leading path segment. A scheme always places a colon in the
    // first segment, so a strictly relative artifact path never carries one.
    let first_segment = uri.split(['/', '\\']).next().unwrap_or(uri);
    if first_segment.contains(':') {
        return Err(DagMlError::RuntimeValidation(format!(
            "artifact `{artifact_id}` uri `{uri}` must not include a scheme or colon in its first path segment"
        )));
    }
    for segment in uri.split(['/', '\\']) {
        if segment == ".." {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{artifact_id}` uri `{uri}` must not contain `..` components"
            )));
        }
    }
    Ok(())
}

fn validate_runtime_fingerprint(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::RuntimeValidation(format!(
            "{label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

fn read_runtime_json<T: serde::de::DeserializeOwned>(path: &Path, label: &str) -> Result<T> {
    let data = fs::read(path).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to read {label} at {}: {err}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&data).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to parse {label} at {}: {err}",
            path.display()
        ))
    })
}

fn write_runtime_json<T: Serialize>(path: &Path, value: &T, label: &str) -> Result<()> {
    let mut data = serde_json::to_vec_pretty(value).map_err(|err| {
        DagMlError::RuntimeValidation(format!("failed to serialize {label}: {err}"))
    })?;
    data.push(b'\n');
    fs::write(path, data).map_err(|err| {
        DagMlError::RuntimeValidation(format!(
            "failed to write {label} at {}: {err}",
            path.display()
        ))
    })
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryPredictionCacheStore {
    payloads: BTreeMap<String, crate::bundle::BundlePredictionCachePayload>,
    materialization_records: RefCell<Vec<PredictionCacheMaterializationRecord>>,
}

impl InMemoryPredictionCacheStore {
    pub fn from_payloads(
        bundle: &ExecutionBundle,
        payloads: BundlePredictionCachePayloadSet,
    ) -> Result<Self> {
        payloads.validate_against_bundle(bundle)?;
        Ok(Self {
            payloads: payloads
                .caches
                .into_iter()
                .map(|payload| (payload.requirement_key.clone(), payload))
                .collect(),
            materialization_records: RefCell::new(Vec::new()),
        })
    }

    pub fn payload_count(&self) -> usize {
        self.payloads.len()
    }

    pub fn materialization_records(&self) -> Vec<PredictionCacheMaterializationRecord> {
        self.materialization_records.borrow().clone()
    }
}

impl RuntimePredictionCacheStore for InMemoryPredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> Result<Vec<PredictionBlock>> {
        let payload = self.payloads.get(requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache store is missing requirement `{requirement_key}`"
            ))
        })?;
        payload.validate()?;
        if payload.prediction_level != PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache store requirement `{requirement_key}` contains {:?} predictions, not sample blocks",
                payload.prediction_level
            )));
        }
        Ok(payload.blocks.clone())
    }

    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> Result<Vec<AggregatedPredictionBlock>> {
        let payload = self.payloads.get(requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache store is missing requirement `{requirement_key}`"
            ))
        })?;
        payload.validate()?;
        if payload.prediction_level == PredictionLevel::Sample {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache store requirement `{requirement_key}` contains sample predictions, not aggregated blocks"
            )));
        }
        Ok(payload.aggregated_blocks.clone())
    }

    fn materialize(&self, request: &PredictionCacheMaterializationRequest) -> Result<HandleRef> {
        request.requirement.validate()?;
        request.cache.validate()?;
        if request.requirement.key() != request.cache.requirement_key {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache materialization request for `{}` uses cache `{}` with mismatched requirement `{}`",
                request.requirement.key(),
                request.cache.cache_id,
                request.cache.requirement_key
            )));
        }
        let payload = self
            .payloads
            .get(&request.cache.requirement_key)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "prediction cache store is missing requirement `{}`",
                    request.cache.requirement_key
                ))
            })?;
        validate_prediction_cache_payload_matches_record(payload, &request.cache)?;
        let fingerprint = stable_json_fingerprint(&(
            &request.run_id,
            &request.bundle_id,
            request.phase,
            &request.variant_id,
            &request.cache.requirement_key,
            &request.cache.cache_id,
            request.cache.prediction_level,
            &request.cache.content_fingerprint,
        ))?;
        let handle = HandleRef {
            handle: u64::from_str_radix(&fingerprint[..16], 16)
                .expect("sha256 hex prefix should fit into u64"),
            kind: HandleKind::Prediction,
            owner_controller: request.producer_controller_id.clone(),
        };
        self.materialization_records
            .borrow_mut()
            .push(PredictionCacheMaterializationRecord {
                run_id: request.run_id.clone(),
                bundle_id: request.bundle_id.clone(),
                phase: request.phase,
                variant_id: request.variant_id.clone(),
                requirement_key: request.cache.requirement_key.clone(),
                cache_id: request.cache.cache_id.clone(),
                handle: handle.clone(),
            });
        Ok(handle)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionInputSpec {
    pub producer_node: NodeId,
    pub source_port: String,
    pub target_port: String,
    pub partition: PredictionPartition,
    #[serde(default = "default_runtime_prediction_level")]
    pub prediction_level: PredictionLevel,
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub fold_ids: Vec<FoldId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unit_ids: Vec<PredictionUnitId>,
    #[serde(default)]
    pub sample_ids: Vec<SampleId>,
    /// Per-sample OOF prediction rows, aligned 1:1 with `sample_ids`
    /// (width == `prediction_width`). Sourced only from Validation OOF blocks
    /// so a host can build a stacking meta-feature matrix during FIT_CV/REFIT.
    #[serde(default)]
    pub values: Vec<Vec<f64>>,
    pub prediction_width: usize,
    #[serde(default)]
    pub target_names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactInputSpec {
    pub node_id: NodeId,
    pub controller_id: ControllerId,
    pub artifact: ArtifactRef,
    pub params_fingerprint: String,
    #[serde(default)]
    pub data_requirement_keys: Vec<String>,
    #[serde(default)]
    pub prediction_requirement_keys: Vec<String>,
}

impl ArtifactInputSpec {
    fn from_refit_record(record: &RefitArtifactRecord) -> Result<Self> {
        record.validate()?;
        Ok(Self {
            node_id: record.node_id.clone(),
            controller_id: record.controller_id.clone(),
            artifact: record.artifact.clone(),
            params_fingerprint: record.params_fingerprint.clone(),
            data_requirement_keys: record.data_requirement_keys.clone(),
            prediction_requirement_keys: record.prediction_requirement_keys.clone(),
        })
    }
}

fn default_runtime_prediction_level() -> PredictionLevel {
    PredictionLevel::Sample
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeTask {
    pub run_id: RunId,
    pub node_plan: NodePlan,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    #[serde(default)]
    pub variant: Option<VariantExecutionSpec>,
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub branch_path: Vec<BranchId>,
    #[serde(default)]
    pub input_handles: BTreeMap<String, HandleRef>,
    #[serde(default)]
    pub data_views: BTreeMap<String, DataProviderViewSpec>,
    #[serde(default)]
    pub prediction_inputs: BTreeMap<String, PredictionInputSpec>,
    #[serde(default)]
    pub artifact_inputs: BTreeMap<String, ArtifactInputSpec>,
    /// Nested (inner) CV fold set for this node in the current outer fold, built
    /// by the runtime from the outer fold's training samples when an effective
    /// `inner_cv` policy applies (FIT_CV only). `None` otherwise. Leakage-safe by
    /// construction (inner ⊆ outer-train); see [`crate::fold::NestedCvSpec`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_fold_set: Option<FoldSet>,
    #[serde(default, skip_serializing_if = "FitInfluenceTask::is_default")]
    pub fit_influence: FitInfluenceTask,
    pub seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitInfluenceMechanism {
    UniformRows,
    SampleWeights,
    RowResampling,
    BackendLossWeights,
    ScorerOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitInfluenceTask {
    pub requested_policy: FitInfluencePolicy,
    pub effective_policy: FitInfluencePolicy,
    pub mechanism: FitInfluenceMechanism,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub row_weights: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl Default for FitInfluenceTask {
    fn default() -> Self {
        Self {
            requested_policy: FitInfluencePolicy::UniformRows,
            effective_policy: FitInfluencePolicy::UniformRows,
            mechanism: FitInfluenceMechanism::UniformRows,
            row_weights: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

impl FitInfluenceTask {
    fn is_default(&self) -> bool {
        self == &Self::default()
    }

    pub fn diagnostic(&self) -> FitInfluenceDiagnostic {
        FitInfluenceDiagnostic {
            requested_policy: self.requested_policy,
            effective_policy: self.effective_policy,
            mechanism: self.mechanism,
            fallback_used: !self.warnings.is_empty(),
            row_weight_count: self.row_weights.len(),
            warnings: self.warnings.clone(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self
            .row_weights
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0)
        {
            return Err(DagMlError::RuntimeValidation(
                "fit influence row_weights must be finite and > 0".to_string(),
            ));
        }
        if self
            .warnings
            .iter()
            .any(|warning| warning.trim().is_empty())
        {
            return Err(DagMlError::RuntimeValidation(
                "fit influence warnings must not be empty".to_string(),
            ));
        }
        match self.effective_policy {
            FitInfluencePolicy::EqualSampleInfluence | FitInfluencePolicy::BackendLossWeight
                if self.row_weights.is_empty() =>
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "fit influence {:?} requires row_weights",
                    self.effective_policy
                )));
            }
            _ => {}
        }
        if self.requested_policy == FitInfluencePolicy::StrictWeightSupport
            && self.effective_policy == FitInfluencePolicy::UniformRows
        {
            return Err(DagMlError::RuntimeValidation(
                "strict fit influence cannot fall back to uniform_rows".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitInfluenceDiagnostic {
    pub requested_policy: FitInfluencePolicy,
    pub effective_policy: FitInfluencePolicy,
    pub mechanism: FitInfluenceMechanism,
    #[serde(default)]
    pub fallback_used: bool,
    #[serde(default)]
    pub row_weight_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl FitInfluenceDiagnostic {
    pub fn validate(&self, task: &NodeTask) -> Result<()> {
        if self.requested_policy != task.fit_influence.requested_policy {
            return Err(DagMlError::RuntimeValidation(format!(
                "fit influence diagnostic requested_policy {:?} does not match task {:?}",
                self.requested_policy, task.fit_influence.requested_policy
            )));
        }
        if self.effective_policy != task.fit_influence.effective_policy {
            return Err(DagMlError::RuntimeValidation(format!(
                "fit influence diagnostic effective_policy {:?} does not match task {:?}",
                self.effective_policy, task.fit_influence.effective_policy
            )));
        }
        if self.mechanism != task.fit_influence.mechanism {
            return Err(DagMlError::RuntimeValidation(format!(
                "fit influence diagnostic mechanism {:?} does not match task {:?}",
                self.mechanism, task.fit_influence.mechanism
            )));
        }
        if self.row_weight_count != task.fit_influence.row_weights.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "fit influence diagnostic row_weight_count {} does not match task {}",
                self.row_weight_count,
                task.fit_influence.row_weights.len()
            )));
        }
        if self
            .warnings
            .iter()
            .any(|warning| warning.trim().is_empty())
        {
            return Err(DagMlError::RuntimeValidation(
                "fit influence diagnostic warnings must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariantExecutionSpec {
    pub variant_id: VariantId,
    #[serde(default)]
    pub choices: BTreeMap<String, GenerationChoice>,
    pub fingerprint: String,
    pub seed: Option<u64>,
}

impl VariantExecutionSpec {
    pub fn from_plan(variant: &VariantPlan) -> Self {
        Self {
            variant_id: variant.variant_id.clone(),
            choices: variant.choices.clone(),
            fingerprint: variant.fingerprint.clone(),
            seed: variant.seed,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.fingerprint.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "variant `{}` has an empty fingerprint in task context",
                self.variant_id
            )));
        }
        for (dimension_name, choice) in &self.choices {
            if dimension_name.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "variant `{}` has an empty generation dimension name",
                    self.variant_id
                )));
            }
            if choice.label.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "variant `{}` has an empty choice label for dimension `{dimension_name}`",
                    self.variant_id
                )));
            }
            for override_spec in &choice.param_overrides {
                if override_spec.params.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "variant `{}` has an empty param override for node `{}`",
                        self.variant_id, override_spec.node_id
                    )));
                }
                for param_key in override_spec.params.keys() {
                    if param_key.trim().is_empty() {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "variant `{}` has an empty param override key for node `{}`",
                            self.variant_id, override_spec.node_id
                        )));
                    }
                }
            }
        }
        self.param_overrides_by_node()?;
        Ok(())
    }

    pub fn effective_params_for_node(
        &self,
        node_id: &NodeId,
        base_params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<BTreeMap<String, serde_json::Value>> {
        let overrides_by_node = self.param_overrides_by_node()?;
        let Some(overrides) = overrides_by_node.get(node_id) else {
            return Ok(base_params.clone());
        };
        let mut params = base_params.clone();
        params.extend(overrides.clone());
        Ok(params)
    }

    fn param_overrides_by_node(
        &self,
    ) -> Result<BTreeMap<NodeId, BTreeMap<String, serde_json::Value>>> {
        let mut overrides = BTreeMap::<NodeId, BTreeMap<String, serde_json::Value>>::new();
        let mut owners = BTreeMap::<(NodeId, String), String>::new();
        for (dimension_name, choice) in &self.choices {
            for override_spec in &choice.param_overrides {
                for (param_key, value) in &override_spec.params {
                    let owner_key = (override_spec.node_id.clone(), param_key.clone());
                    if let Some(previous) =
                        owners.insert(owner_key, format!("{dimension_name}:{}", choice.label))
                    {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "variant `{}` has conflicting generation overrides for `{}.{}` from `{previous}` and `{}:{}`",
                            self.variant_id,
                            override_spec.node_id,
                            param_key,
                            dimension_name,
                            choice.label
                        )));
                    }
                    overrides
                        .entry(override_spec.node_id.clone())
                        .or_default()
                        .insert(param_key.clone(), value.clone());
                }
            }
        }
        Ok(overrides)
    }
}

/// An EXPLAIN-phase output block (ADR-12 explain contract). Explanations are a
/// node *output* returned in the [`NodeResult`] — like predictions, they cross as
/// data, not as an opaque host handle. The `payload` shape is controller-defined
/// (e.g. per-feature importances); the core does not interpret it. Explanations
/// are only valid in the `EXPLAIN` phase.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplanationBlock {
    /// Node whose model the explanation describes (must equal the producing node).
    pub producer_node: NodeId,
    /// Stable explanation method identifier, e.g. `shap`, `permutation_importance`.
    pub method: String,
    /// Optional target/output name the explanation pertains to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_name: Option<String>,
    /// Controller-defined explanation payload as canonical JSON.
    pub payload: serde_json::Value,
}

impl ExplanationBlock {
    /// Validate the intrinsic shape of the explanation block (method/target_name
    /// non-empty). Producer identity is checked against the node in
    /// [`NodeResult::validate_for_task`].
    pub fn validate(&self) -> Result<()> {
        if self.method.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "explanation method must be a non-empty identifier".to_string(),
            ));
        }
        if let Some(name) = &self.target_name {
            if name.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(
                    "explanation target_name must be non-empty when present".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeResult {
    pub node_id: NodeId,
    #[serde(default)]
    pub outputs: BTreeMap<String, HandleRef>,
    #[serde(default)]
    pub predictions: Vec<PredictionBlock>,
    #[serde(default)]
    pub observation_predictions: Vec<ObservationPredictionBlock>,
    #[serde(default)]
    pub aggregated_predictions: Vec<AggregatedPredictionBlock>,
    #[serde(default)]
    pub explanations: Vec<ExplanationBlock>,
    #[serde(default)]
    pub shape_deltas: Vec<ShapeDelta>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default)]
    pub artifact_handles: BTreeMap<ArtifactId, HandleRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fit_influence_diagnostics: Vec<FitInfluenceDiagnostic>,
    /// Optional ground-truth targets the host controller emits alongside predictions so the core
    /// can score natively (the runtime never sees feature matrices; `y_true` is data-tier and may
    /// cross the ABI per the ownership table). Each block is identity-keyed by `unit_ids`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regression_targets: Vec<RegressionTargetBlock>,
    pub lineage: LineageRecord,
}

impl NodeResult {
    pub fn validate_for_task(&self, task: &NodeTask) -> Result<()> {
        if self.node_id != task.node_plan.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "task for `{}` returned result for `{}`",
                task.node_plan.node_id, self.node_id
            )));
        }
        if self.lineage.node_id != task.node_plan.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for task `{}` references node `{}`",
                task.node_plan.node_id, self.lineage.node_id
            )));
        }
        if self.lineage.phase != task.phase {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has phase {:?}, expected {:?}",
                task.node_plan.node_id, self.lineage.phase, task.phase
            )));
        }
        if self.lineage.run_id != task.run_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has run `{}`, expected `{}`",
                task.node_plan.node_id, self.lineage.run_id, task.run_id
            )));
        }
        if self.lineage.controller_id != task.node_plan.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has controller `{}`, expected `{}`",
                task.node_plan.node_id, self.lineage.controller_id, task.node_plan.controller_id
            )));
        }
        if self.lineage.controller_version != task.node_plan.controller_version {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has controller version `{}`, expected `{}`",
                task.node_plan.node_id,
                self.lineage.controller_version,
                task.node_plan.controller_version
            )));
        }
        if self.lineage.variant_id != task.variant_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has variant {:?}, expected {:?}",
                task.node_plan.node_id, self.lineage.variant_id, task.variant_id
            )));
        }
        if let Some(variant) = &task.variant {
            variant.validate()?;
            if Some(&variant.variant_id) != task.variant_id.as_ref() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "task for node `{}` has variant context `{}` but variant_id {:?}",
                    task.node_plan.node_id, variant.variant_id, task.variant_id
                )));
            }
        }
        if self.lineage.fold_id != task.fold_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has fold {:?}, expected {:?}",
                task.node_plan.node_id, self.lineage.fold_id, task.fold_id
            )));
        }
        if self.lineage.branch_path != task.branch_path {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has branch path {:?}, expected {:?}",
                task.node_plan.node_id, self.lineage.branch_path, task.branch_path
            )));
        }
        if self.lineage.seed != task.seed {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has seed {:?}, expected {:?}",
                task.node_plan.node_id, self.lineage.seed, task.seed
            )));
        }
        if self.lineage.params_fingerprint != task.node_plan.params_fingerprint {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has params fingerprint `{}`, expected `{}`",
                task.node_plan.node_id,
                self.lineage.params_fingerprint,
                task.node_plan.params_fingerprint
            )));
        }
        task.fit_influence.validate()?;
        for diagnostic in &self.fit_influence_diagnostics {
            diagnostic.validate(task)?;
        }
        validate_lineage_shape_fingerprints(&self.lineage, task)?;
        if !self.explanations.is_empty() && task.phase != Phase::Explain {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` returned explanations outside the EXPLAIN phase",
                task.node_plan.node_id
            )));
        }
        for explanation in &self.explanations {
            explanation.validate()?;
            if explanation.producer_node != self.node_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` returned an explanation produced by `{}`",
                    self.node_id, explanation.producer_node
                )));
            }
        }
        for (port, handle) in &self.outputs {
            if handle.owner_controller != task.node_plan.controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` output `{port}` is owned by `{}`, expected `{}`",
                    task.node_plan.node_id, handle.owner_controller, task.node_plan.controller_id
                )));
            }
        }
        let mut artifact_ids = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_ids.insert(artifact.id.clone()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted duplicate artifact `{}`",
                    task.node_plan.node_id, artifact.id
                )));
            }
            if artifact.controller_id != task.node_plan.controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` for controller `{}`, expected `{}`",
                    task.node_plan.node_id,
                    artifact.id,
                    artifact.controller_id,
                    task.node_plan.controller_id
                )));
            }
            let handle = self.artifact_handles.get(&artifact.id).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` without artifact handle",
                    task.node_plan.node_id, artifact.id
                ))
            })?;
            if !matches!(handle.kind, HandleKind::Model | HandleKind::Artifact) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` with non-artifact/model handle kind {:?}",
                    task.node_plan.node_id, artifact.id, handle.kind
                )));
            }
            if handle.owner_controller != task.node_plan.controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` owned by `{}`, expected `{}`",
                    task.node_plan.node_id,
                    artifact.id,
                    handle.owner_controller,
                    task.node_plan.controller_id
                )));
            }
        }
        for artifact_id in self.artifact_handles.keys() {
            if !self
                .artifacts
                .iter()
                .any(|artifact| &artifact.id == artifact_id)
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact handle for undeclared artifact `{artifact_id}`",
                    task.node_plan.node_id
                )));
            }
        }
        for artifact in &self.artifacts {
            if !self
                .lineage
                .artifact_refs
                .iter()
                .any(|lineage_artifact| lineage_artifact == artifact)
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` without matching lineage artifact ref",
                    task.node_plan.node_id, artifact.id
                )));
            }
        }
        for artifact in &self.lineage.artifact_refs {
            if !self
                .artifacts
                .iter()
                .any(|emitted_artifact| emitted_artifact == artifact)
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` lineage references undeclared artifact `{}`",
                    task.node_plan.node_id, artifact.id
                )));
            }
        }
        for prediction in &self.predictions {
            prediction.validate_shape()?;
            if prediction.producer_node != task.node_plan.node_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted prediction for producer `{}`",
                    task.node_plan.node_id, prediction.producer_node
                )));
            }
            validate_prediction_scope(prediction, task)?;
        }
        for prediction in &self.observation_predictions {
            prediction.validate_shape()?;
            if prediction.producer_node != task.node_plan.node_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted observation prediction for producer `{}`",
                    task.node_plan.node_id, prediction.producer_node
                )));
            }
            validate_observation_prediction_scope(prediction, task)?;
        }
        for prediction in &self.aggregated_predictions {
            prediction.validate_shape()?;
            if prediction.producer_node != task.node_plan.node_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted aggregated prediction for producer `{}`",
                    task.node_plan.node_id, prediction.producer_node
                )));
            }
            validate_aggregated_prediction_scope(prediction, task)?;
        }
        for delta in &self.shape_deltas {
            delta.validate()?;
            if delta.node_id != task.node_plan.node_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted shape delta for `{}`",
                    task.node_plan.node_id, delta.node_id
                )));
            }
            validate_shape_delta_for_task(delta, task)?;
        }
        for target in &self.regression_targets {
            target.validate_shape()?;
        }
        self.lineage.validate()
    }
}

fn validate_lineage_shape_fingerprints(lineage: &LineageRecord, task: &NodeTask) -> Result<()> {
    let Some(shape_plan) = &task.node_plan.shape_plan else {
        if lineage.data_model_shape_fingerprint.is_some()
            || lineage.aggregation_policy_fingerprint.is_some()
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` carries shape fingerprints but the node has no shape plan",
                task.node_plan.node_id
            )));
        }
        return Ok(());
    };

    if let Some(actual) = &lineage.data_model_shape_fingerprint {
        let expected = stable_json_fingerprint(shape_plan)?;
        if actual != &expected {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has data/model shape fingerprint `{actual}`, expected `{expected}`",
                task.node_plan.node_id
            )));
        }
    }
    if let Some(actual) = &lineage.aggregation_policy_fingerprint {
        let expected = stable_json_fingerprint(&shape_plan.aggregation_policy)?;
        if actual != &expected {
            return Err(DagMlError::RuntimeValidation(format!(
                "lineage for node `{}` has aggregation policy fingerprint `{actual}`, expected `{expected}`",
                task.node_plan.node_id
            )));
        }
    }
    Ok(())
}

fn validate_shape_delta_for_task(delta: &ShapeDelta, task: &NodeTask) -> Result<()> {
    let Some(shape_plan) = &task.node_plan.shape_plan else {
        return Ok(());
    };
    if delta.kind == ShapeDeltaKind::Feature {
        if let Some(expected) = &shape_plan.feature_schema_fingerprint {
            if &delta.before_fingerprint != expected {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted feature shape delta from `{}`, expected current schema `{expected}`",
                    task.node_plan.node_id, delta.before_fingerprint
                )));
            }
        }
    }
    Ok(())
}

fn validate_prediction_scope(prediction: &PredictionBlock, task: &NodeTask) -> Result<()> {
    if prediction.partition != PredictionPartition::Validation {
        return Ok(());
    }
    if prediction.fold_id != task.fold_id {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` emitted validation predictions for fold {:?}, expected {:?}",
            task.node_plan.node_id, prediction.fold_id, task.fold_id
        )));
    }
    if task.phase == Phase::FitCv
        && task.fold_id.is_some()
        && (!task.node_plan.data_bindings.is_empty() || !task.data_views.is_empty())
    {
        let validation_sample_ids = validation_view_sample_ids(task).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "node `{}` emitted validation predictions without a fold-validation data view",
                task.node_plan.node_id
            ))
        })?;
        for sample_id in &prediction.sample_ids {
            if !validation_sample_ids.contains(sample_id) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted validation prediction for sample `{}` outside its validation view",
                    task.node_plan.node_id, sample_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_observation_prediction_scope(
    prediction: &ObservationPredictionBlock,
    task: &NodeTask,
) -> Result<()> {
    if prediction.partition != PredictionPartition::Validation {
        return Ok(());
    }
    if prediction.fold_id != task.fold_id {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` emitted observation validation predictions for fold {:?}, expected {:?}",
            task.node_plan.node_id, prediction.fold_id, task.fold_id
        )));
    }
    Ok(())
}

fn validate_aggregated_prediction_scope(
    prediction: &AggregatedPredictionBlock,
    task: &NodeTask,
) -> Result<()> {
    if prediction.partition != PredictionPartition::Validation {
        return Ok(());
    }
    if prediction.fold_id != task.fold_id {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` emitted aggregated validation predictions for fold {:?}, expected {:?}",
            task.node_plan.node_id, prediction.fold_id, task.fold_id
        )));
    }
    // Sample-level aggregated validation units must stay inside this fold's
    // validation view, mirroring `validate_prediction_scope`. Target / group
    // units are checked against their relation set in the aggregation path.
    if prediction.level == PredictionLevel::Sample
        && task.phase == Phase::FitCv
        && task.fold_id.is_some()
        && (!task.node_plan.data_bindings.is_empty() || !task.data_views.is_empty())
    {
        if let Some(validation_sample_ids) = validation_view_sample_ids(task) {
            for unit_id in &prediction.unit_ids {
                if let PredictionUnitId::Sample(sample_id) = unit_id {
                    if !validation_sample_ids.contains(sample_id) {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "node `{}` emitted aggregated validation prediction for sample `{}` outside its validation view",
                            task.node_plan.node_id, sample_id
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validation_view_sample_ids(task: &NodeTask) -> Option<BTreeSet<SampleId>> {
    let mut sample_ids = BTreeSet::new();
    for view in task
        .data_views
        .values()
        .filter(|view| view.partition == DataRequestPartition::FoldValidation)
    {
        if let Some(view_sample_ids) = &view.sample_ids {
            sample_ids.extend(view_sample_ids.iter().cloned());
        }
    }
    (!sample_ids.is_empty()).then_some(sample_ids)
}

fn fit_influence_task_for_node(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    data_views: &BTreeMap<String, DataProviderViewSpec>,
) -> Result<FitInfluenceTask> {
    let manifest = plan
        .controller_manifests
        .get(&node_plan.controller_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "node `{}` references missing controller manifest `{}`",
                node_plan.node_id, node_plan.controller_id
            ))
        })?;
    let Some(model_input_spec) = manifest.model_input_spec()? else {
        return Ok(FitInfluenceTask::default());
    };
    let Some(requested_policy) = model_input_spec.fit_influence_policy else {
        return Ok(FitInfluenceTask::default());
    };
    resolve_fit_influence_task(
        requested_policy,
        &node_plan.controller_capabilities,
        data_views,
    )
}

fn resolve_fit_influence_task(
    requested_policy: FitInfluencePolicy,
    capabilities: &BTreeSet<ControllerCapability>,
    data_views: &BTreeMap<String, DataProviderViewSpec>,
) -> Result<FitInfluenceTask> {
    let row_weights = equal_sample_influence_weights(data_views);
    match requested_policy {
        FitInfluencePolicy::UniformRows => Ok(FitInfluenceTask {
            requested_policy,
            effective_policy: FitInfluencePolicy::UniformRows,
            mechanism: FitInfluenceMechanism::UniformRows,
            row_weights: Vec::new(),
            warnings: Vec::new(),
        }),
        FitInfluencePolicy::ScorerOnly => Ok(FitInfluenceTask {
            requested_policy,
            effective_policy: FitInfluencePolicy::ScorerOnly,
            mechanism: FitInfluenceMechanism::ScorerOnly,
            row_weights: Vec::new(),
            warnings: Vec::new(),
        }),
        FitInfluencePolicy::EqualSampleInfluence => {
            require_fit_influence_support(capabilities, requested_policy)?;
            let weights = row_weights.ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "equal_sample_influence requires task row sample ids".to_string(),
                )
            })?;
            Ok(FitInfluenceTask {
                requested_policy,
                effective_policy: FitInfluencePolicy::EqualSampleInfluence,
                mechanism: FitInfluenceMechanism::SampleWeights,
                row_weights: weights,
                warnings: Vec::new(),
            })
        }
        FitInfluencePolicy::ResampleEqualized => {
            require_fit_influence_support(capabilities, requested_policy)?;
            Ok(FitInfluenceTask {
                requested_policy,
                effective_policy: FitInfluencePolicy::ResampleEqualized,
                mechanism: FitInfluenceMechanism::RowResampling,
                row_weights: Vec::new(),
                warnings: Vec::new(),
            })
        }
        FitInfluencePolicy::BackendLossWeight => {
            require_fit_influence_support(capabilities, requested_policy)?;
            let weights = row_weights.ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "backend_loss_weight requires task row sample ids".to_string(),
                )
            })?;
            Ok(FitInfluenceTask {
                requested_policy,
                effective_policy: FitInfluencePolicy::BackendLossWeight,
                mechanism: FitInfluenceMechanism::BackendLossWeights,
                row_weights: weights,
                warnings: Vec::new(),
            })
        }
        FitInfluencePolicy::StrictWeightSupport => {
            require_fit_influence_support(capabilities, requested_policy)?;
            strict_fit_influence_task(capabilities, row_weights, requested_policy)
        }
        FitInfluencePolicy::Auto => Ok(auto_fit_influence_task(capabilities, row_weights)),
    }
}

fn require_fit_influence_support(
    capabilities: &BTreeSet<ControllerCapability>,
    policy: FitInfluencePolicy,
) -> Result<()> {
    if capabilities_support_fit_influence(capabilities, policy) {
        return Ok(());
    }
    Err(DagMlError::RuntimeValidation(format!(
        "controller capabilities do not support requested fit influence policy {:?}",
        policy
    )))
}

fn strict_fit_influence_task(
    capabilities: &BTreeSet<ControllerCapability>,
    row_weights: Option<Vec<f64>>,
    requested_policy: FitInfluencePolicy,
) -> Result<FitInfluenceTask> {
    if capabilities.contains(&ControllerCapability::SupportsBackendLossWeights) {
        let weights = row_weights.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "strict_weight_support with backend loss weights requires task row sample ids"
                    .to_string(),
            )
        })?;
        return Ok(FitInfluenceTask {
            requested_policy,
            effective_policy: FitInfluencePolicy::BackendLossWeight,
            mechanism: FitInfluenceMechanism::BackendLossWeights,
            row_weights: weights,
            warnings: Vec::new(),
        });
    }
    if capabilities.contains(&ControllerCapability::SupportsSampleWeights) {
        let weights = row_weights.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "strict_weight_support with sample weights requires task row sample ids"
                    .to_string(),
            )
        })?;
        return Ok(FitInfluenceTask {
            requested_policy,
            effective_policy: FitInfluencePolicy::EqualSampleInfluence,
            mechanism: FitInfluenceMechanism::SampleWeights,
            row_weights: weights,
            warnings: Vec::new(),
        });
    }
    Ok(FitInfluenceTask {
        requested_policy,
        effective_policy: FitInfluencePolicy::ResampleEqualized,
        mechanism: FitInfluenceMechanism::RowResampling,
        row_weights: Vec::new(),
        warnings: Vec::new(),
    })
}

fn auto_fit_influence_task(
    capabilities: &BTreeSet<ControllerCapability>,
    row_weights: Option<Vec<f64>>,
) -> FitInfluenceTask {
    if capabilities.contains(&ControllerCapability::SupportsSampleWeights) {
        if let Some(weights) = row_weights.clone() {
            return FitInfluenceTask {
                requested_policy: FitInfluencePolicy::Auto,
                effective_policy: FitInfluencePolicy::EqualSampleInfluence,
                mechanism: FitInfluenceMechanism::SampleWeights,
                row_weights: weights,
                warnings: Vec::new(),
            };
        }
    }
    if capabilities.contains(&ControllerCapability::SupportsRowResampling) {
        return FitInfluenceTask {
            requested_policy: FitInfluencePolicy::Auto,
            effective_policy: FitInfluencePolicy::ResampleEqualized,
            mechanism: FitInfluenceMechanism::RowResampling,
            row_weights: Vec::new(),
            warnings: Vec::new(),
        };
    }
    if capabilities.contains(&ControllerCapability::SupportsBackendLossWeights) {
        if let Some(weights) = row_weights {
            return FitInfluenceTask {
                requested_policy: FitInfluencePolicy::Auto,
                effective_policy: FitInfluencePolicy::BackendLossWeight,
                mechanism: FitInfluenceMechanism::BackendLossWeights,
                row_weights: weights,
                warnings: Vec::new(),
            };
        }
    }
    FitInfluenceTask {
        requested_policy: FitInfluencePolicy::Auto,
        effective_policy: FitInfluencePolicy::UniformRows,
        mechanism: FitInfluenceMechanism::UniformRows,
        row_weights: Vec::new(),
        warnings: vec![
            "auto fit influence fell back to uniform_rows because no supported weighting capability was usable".to_string(),
        ],
    }
}

fn equal_sample_influence_weights(
    data_views: &BTreeMap<String, DataProviderViewSpec>,
) -> Option<Vec<f64>> {
    let row_sample_ids = data_views
        .values()
        .filter(|view| {
            matches!(
                view.partition,
                DataRequestPartition::FoldTrain | DataRequestPartition::FullTrain
            )
        })
        .filter_map(|view| view.sample_ids.as_ref())
        .find(|sample_ids| !sample_ids.is_empty())
        .or_else(|| {
            data_views
                .values()
                .filter_map(|view| view.sample_ids.as_ref())
                .find(|sample_ids| !sample_ids.is_empty())
        })?;
    let mut counts = BTreeMap::<&SampleId, usize>::new();
    for sample_id in row_sample_ids {
        *counts.entry(sample_id).or_default() += 1;
    }
    Some(
        row_sample_ids
            .iter()
            .map(|sample_id| 1.0 / *counts.get(sample_id).expect("counted sample id") as f64)
            .collect(),
    )
}

fn record_fit_influence_diagnostic(task: &NodeTask, result: &mut NodeResult) {
    if task.fit_influence.is_default() || !result.fit_influence_diagnostics.is_empty() {
        return;
    }
    result
        .fit_influence_diagnostics
        .push(task.fit_influence.diagnostic());
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataMaterializationRequest {
    pub run_id: RunId,
    pub node_id: NodeId,
    pub input_name: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    pub binding: crate::data::DataBinding,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataProviderViewSpec {
    #[serde(default)]
    pub sample_ids: Option<Vec<SampleId>>,
    pub partition: DataRequestPartition,
    #[serde(default)]
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub source_ids: Option<Vec<String>>,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    pub include_augmented: bool,
    pub include_excluded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_view: Option<crate::data::BranchViewPlan>,
    #[serde(default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

pub const DATA_OUTPUT_PROVENANCE_KEY: &str = "dag_ml_output";
pub const DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const DATA_OUTPUT_PROVENANCE_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/data_output_provenance.v1.schema.json";
pub const NODE_TASK_SCHEMA_VERSION: u32 = 1;
pub const NODE_TASK_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/node_task.v1.schema.json";
pub const NODE_RESULT_SCHEMA_VERSION: u32 = 1;
pub const NODE_RESULT_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/node_result.v1.schema.json";

fn default_data_output_provenance_schema_version() -> u32 {
    DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION
}

impl DataProviderViewSpec {
    pub fn validate(&self) -> Result<()> {
        validate_optional_ids("sample id", &self.sample_ids)?;
        validate_optional_strings("source id", &self.source_ids)?;
        validate_optional_strings("column", &self.columns)?;
        match self.partition {
            DataRequestPartition::FoldTrain | DataRequestPartition::FoldValidation => {
                if self.sample_ids.is_some() && self.fold_id.is_none() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "data provider view {:?} with explicit sample ids requires a fold id",
                        self.partition
                    )));
                }
            }
            DataRequestPartition::FullTrain | DataRequestPartition::Predict => {
                if self.fold_id.is_some() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "data provider view {:?} must not carry a fold id",
                        self.partition
                    )));
                }
            }
        }
        for key in self.extra.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(
                    "data provider view extra contains an empty key".to_string(),
                ));
            }
        }
        if let Some(branch_view) = &self.branch_view {
            branch_view.validate()?;
        }
        self.output_provenance()?;
        Ok(())
    }

    pub fn output_provenance(&self) -> Result<Option<DataOutputProvenance>> {
        let Some(value) = self.extra.get(DATA_OUTPUT_PROVENANCE_KEY) else {
            return Ok(None);
        };
        let provenance: DataOutputProvenance = serde_json::from_value(value.clone())?;
        provenance.validate()?;
        Ok(Some(provenance))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataOutputProvenance {
    #[serde(default = "default_data_output_provenance_schema_version")]
    pub schema_version: u32,
    pub producer_node: NodeId,
    pub producer_port: String,
    pub producer_phase: Phase,
    #[serde(default)]
    pub variant_id: Option<VariantId>,
    #[serde(default)]
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub shape_plan_fingerprint: Option<String>,
    #[serde(default)]
    pub aggregation_policy_fingerprint: Option<String>,
    #[serde(default)]
    pub feature_namespace: Option<String>,
    #[serde(default)]
    pub feature_schema_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_plan: Option<RepresentationPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_replay_manifest: Option<RepresentationReplayManifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_compatibility: Option<RepresentationCompatibilityReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation_delta_fingerprint: Option<String>,
    #[serde(default)]
    pub shape_deltas: Vec<ShapeDelta>,
}

impl DataOutputProvenance {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "data output provenance for `{}` uses unsupported schema_version {}, expected {}",
                self.producer_node, self.schema_version, DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION
            )));
        }
        if self.producer_port.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "data output provenance for `{}` has empty producer_port",
                self.producer_node
            )));
        }
        validate_optional_fingerprint(
            "shape_plan_fingerprint",
            &self.shape_plan_fingerprint,
            &self.producer_node,
        )?;
        validate_optional_fingerprint(
            "aggregation_policy_fingerprint",
            &self.aggregation_policy_fingerprint,
            &self.producer_node,
        )?;
        validate_optional_fingerprint(
            "feature_schema_fingerprint",
            &self.feature_schema_fingerprint,
            &self.producer_node,
        )?;
        validate_optional_fingerprint(
            "relation_delta_fingerprint",
            &self.relation_delta_fingerprint,
            &self.producer_node,
        )?;
        if let Some(representation_plan) = &self.representation_plan {
            representation_plan.validate().map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` has invalid representation_plan: {error}",
                    self.producer_node
                ))
            })?;
        }
        if let Some(replay_manifest) = &self.representation_replay_manifest {
            replay_manifest.validate().map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` has invalid representation_replay_manifest: {error}",
                    self.producer_node
                ))
            })?;
        }
        if let Some(report) = &self.representation_compatibility {
            report.validate().map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` has invalid representation_compatibility: {error}",
                    self.producer_node
                ))
            })?;
        }
        if self
            .feature_namespace
            .as_ref()
            .is_some_and(|namespace| namespace.trim().is_empty())
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "data output provenance for `{}` has empty feature_namespace",
                self.producer_node
            )));
        }
        for delta in &self.shape_deltas {
            delta.validate()?;
            if delta.node_id != self.producer_node {
                return Err(DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` contains shape delta for `{}`",
                    self.producer_node, delta.node_id
                )));
            }
        }
        if let Some(feature_schema_fingerprint) = &self.feature_schema_fingerprint {
            if let Some(last_feature_delta) = self
                .shape_deltas
                .iter()
                .rev()
                .find(|delta| delta.kind == ShapeDeltaKind::Feature)
            {
                if &last_feature_delta.after_fingerprint != feature_schema_fingerprint {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "data output provenance for `{}` has feature_schema_fingerprint `{feature_schema_fingerprint}` but last feature delta ends at `{}`",
                        self.producer_node, last_feature_delta.after_fingerprint
                    )));
                }
            }
        }
        Ok(())
    }
}

fn validate_optional_fingerprint(
    label: &str,
    fingerprint: &Option<String>,
    producer_node: &NodeId,
) -> Result<()> {
    let Some(fingerprint) = fingerprint else {
        return Ok(());
    };
    if fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::RuntimeValidation(format!(
            "data output provenance for `{producer_node}` has invalid {label}"
        )));
    }
    Ok(())
}

fn validate_optional_ids<T>(label: &str, values: &Option<Vec<T>>) -> Result<()>
where
    T: Ord + ToString,
{
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "data provider view {label} list is empty"
        )));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider view has duplicate {label} `{}`",
                value.to_string()
            )));
        }
    }
    Ok(())
}

fn validate_optional_strings(label: &str, values: &Option<Vec<String>>) -> Result<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "data provider view {label} list is empty"
        )));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider view contains an empty {label}"
            )));
        }
        if !seen.insert(value.as_str()) {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider view has duplicate {label} `{value}`"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataViewRequest {
    pub run_id: RunId,
    pub node_id: NodeId,
    pub input_name: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    pub binding: crate::data::DataBinding,
    pub data_handle: HandleRef,
    pub view: DataProviderViewSpec,
}

pub trait RuntimeDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> Result<HandleRef>;
    fn make_view(&self, request: &DataViewRequest) -> Result<HandleRef>;
    fn coordinator_relations(&self, _binding: &DataBinding) -> Result<Option<SampleRelationSet>> {
        Ok(None)
    }
}

pub trait RuntimeController: Send + Sync {
    fn controller_id(&self) -> &ControllerId;
    fn invoke(&self, task: &NodeTask) -> Result<NodeResult>;

    fn invoke_aggregation(
        &self,
        task: &AggregationControllerTask,
    ) -> Result<AggregationControllerResult> {
        Err(DagMlError::RuntimeValidation(format!(
            "runtime controller `{}` does not implement aggregation task `{}`",
            self.controller_id(),
            task.task_id
        )))
    }
}

pub struct BundleReplayExecution<'a> {
    pub plan: &'a ExecutionPlan,
    pub bundle: &'a ExecutionBundle,
    pub replay_request: &'a ReplayPhaseRequest,
    pub prediction_cache_store: Option<&'a dyn RuntimePredictionCacheStore>,
    pub controllers: &'a RuntimeControllerRegistry,
    pub data_provider: &'a dyn RuntimeDataProvider,
    pub artifact_store: &'a dyn RuntimeArtifactStore,
    pub data_envelopes: &'a BTreeMap<String, ExternalDataPlanEnvelope>,
}

#[derive(Default)]
pub struct RuntimeControllerRegistry {
    controllers: BTreeMap<ControllerId, Box<dyn RuntimeController>>,
}

impl RuntimeControllerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, controller: Box<dyn RuntimeController>) -> Result<()> {
        let id = controller.controller_id().clone();
        if self.controllers.insert(id.clone(), controller).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate runtime controller `{id}`"
            )));
        }
        Ok(())
    }

    pub fn get(&self, controller_id: &ControllerId) -> Option<&dyn RuntimeController> {
        self.controllers.get(controller_id).map(Box::as_ref)
    }
}

pub fn dispatch_custom_observation_aggregation(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    task_id: impl Into<String>,
    block: ObservationPredictionBlock,
    relations: SampleRelationSet,
    policy: AggregationPolicy,
    requested_sample_order: Vec<SampleId>,
) -> Result<PredictionBlock> {
    let controller_id = custom_aggregation_controller_id(&policy)?;
    ensure_aggregation_controller_capability(plan, controller_id)?;
    let task = AggregationControllerTask {
        schema_version: crate::aggregation::AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
        task_id: task_id.into(),
        controller_id: controller_id.clone(),
        policy,
        reduction_plan: None,
        input: AggregationControllerInput::ObservationToSample {
            block,
            relations,
            requested_sample_order,
        },
    };
    let result = dispatch_custom_aggregation_task(controllers, &task)?;
    match result.output {
        AggregationControllerOutput::Sample { block } => Ok(block),
        AggregationControllerOutput::Unit { .. } => Err(DagMlError::RuntimeValidation(format!(
            "aggregation controller task `{}` returned unit output for observation input",
            task.task_id
        ))),
    }
}

pub fn dispatch_custom_sample_aggregation(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    task_id: impl Into<String>,
    block: PredictionBlock,
    relations: SampleRelationSet,
    policy: AggregationPolicy,
    requested_unit_order: Vec<PredictionUnitId>,
) -> Result<AggregatedPredictionBlock> {
    let controller_id = custom_aggregation_controller_id(&policy)?;
    ensure_aggregation_controller_capability(plan, controller_id)?;
    let task = AggregationControllerTask {
        schema_version: crate::aggregation::AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
        task_id: task_id.into(),
        controller_id: controller_id.clone(),
        policy,
        reduction_plan: None,
        input: AggregationControllerInput::SampleToUnit {
            block,
            relations,
            requested_unit_order,
        },
    };
    let result = dispatch_custom_aggregation_task(controllers, &task)?;
    match result.output {
        AggregationControllerOutput::Unit { block } => Ok(block),
        AggregationControllerOutput::Sample { .. } => Err(DagMlError::RuntimeValidation(format!(
            "aggregation controller task `{}` returned sample output for sample input",
            task.task_id
        ))),
    }
}

pub fn dispatch_custom_aggregation_task(
    controllers: &RuntimeControllerRegistry,
    task: &AggregationControllerTask,
) -> Result<AggregationControllerResult> {
    task.validate()?;
    let controller = controllers.get(&task.controller_id).ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "aggregation runtime controller `{}` is not registered",
            task.controller_id
        ))
    })?;
    let result = controller.invoke_aggregation(task)?;
    result.validate_for_task(task)?;
    Ok(result)
}

fn custom_aggregation_controller_id(policy: &AggregationPolicy) -> Result<&ControllerId> {
    policy.validate()?;
    policy
        .custom_controller
        .as_ref()
        .map(|controller| &controller.controller_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "custom aggregation dispatch requires a custom_controller policy".to_string(),
            )
        })
}

fn ensure_aggregation_controller_capability(
    plan: &ExecutionPlan,
    controller_id: &ControllerId,
) -> Result<()> {
    let manifest = plan
        .controller_manifests
        .get(controller_id)
        .ok_or_else(|| {
            DagMlError::Planning(format!(
                "missing aggregation controller manifest `{controller_id}`"
            ))
        })?;
    if !manifest
        .capabilities
        .contains(&ControllerCapability::AggregatesPredictions)
    {
        return Err(DagMlError::Planning(format!(
            "aggregation controller `{controller_id}` must declare aggregates_predictions"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RunContext {
    pub run_id: RunId,
    pub root_seed: Option<u64>,
    pub variant_id: Option<VariantId>,
    pub prediction_store: InMemoryPredictionStore,
    pub aggregated_prediction_store: InMemoryAggregatedPredictionStore,
    pub lineage: InMemoryLineageRecorder,
    /// Native per-fold/per-partition score reports collected during the run (when the host emits
    /// `regression_targets`).
    pub score_collector: Vec<RegressionMetricReport>,
    /// Per-fold `y_true` records, kept so cross-fold ensembles (the OOF average) can be scored.
    pub regression_target_records: Vec<RegressionTargetRecord>,
}

impl RunContext {
    pub fn new(run_id: RunId, root_seed: Option<u64>) -> Self {
        Self {
            run_id,
            root_seed,
            variant_id: None,
            prediction_store: InMemoryPredictionStore::new(),
            aggregated_prediction_store: InMemoryAggregatedPredictionStore::new(),
            lineage: InMemoryLineageRecorder::new(),
            score_collector: Vec::new(),
            regression_target_records: Vec::new(),
        }
    }

    /// Score the cross-fold OOF average from the collected per-fold validation predictions + targets
    /// and append the reports (one per producer, `fold_id = "avg"`) to the score collector. Call
    /// after FIT_CV; a no-op when nothing was scored or no producer has more than one fold.
    ///
    /// `partition_mode` is the campaign's [`FoldPartitionMode`]: `Partition` (KFold) requires a unique
    /// per-producer OOF set, while `Resampled` (ShuffleSplit / repeated CV) permits a sample to be
    /// validated in multiple folds (averaged when scored). Pass the plan's
    /// [`fold_set`](ExecutionPlan::fold_set) mode (default `Partition` when there is no fold set).
    pub fn collect_cross_fold_validation_scores(
        &mut self,
        partition_mode: FoldPartitionMode,
    ) -> Result<()> {
        let reports = cross_fold_validation_reports(
            self.prediction_store.blocks(),
            &self.regression_target_records,
            SCORE_METRICS,
            partition_mode,
        )?;
        self.score_collector.extend(reports);
        Ok(())
    }

    /// Build a [`ScoreSet`] from the collected reports (or `None` if scoring was off / produced
    /// nothing), e.g. to attach to the [`ExecutionBundle`](crate::bundle::ExecutionBundle).
    pub fn build_score_set(
        &self,
        plan_id: impl Into<String>,
        selection_metric: Option<String>,
    ) -> Option<ScoreSet> {
        if self.score_collector.is_empty() {
            return None;
        }
        Some(ScoreSet {
            schema_version: SCORE_SET_SCHEMA_VERSION,
            plan_id: plan_id.into(),
            selection_metric,
            reports: self.score_collector.clone(),
        })
    }
}

/// Pick the best variant of a multi-variant plan by its cross-validation score, natively.
///
/// "Option A": each variant is scored with its OWN single-variant FIT_CV — the plan is cloned with
/// `variants = vec![variant]` so the existing per-producer cross-fold OOF averaging
/// ([`RunContext::collect_cross_fold_validation_scores`]) is unambiguous (one variant in scope, so a
/// validation `PredictionBlock` belongs to exactly one variant). The OOF-average report per variant
/// becomes a [`CandidateScore`], and [`select_candidate`] ranks them by `selection_metric` (the
/// metric's [`objective`](RegressionMetricKind::objective) drives the direction — RMSE minimizes,
/// accuracy maximizes). The winning candidate id maps back to its [`VariantId`].
///
/// Native scoring is opt-in: it only happens when the host emits `regression_targets`. So this
/// returns `Ok(None)` when NO variant produced a cross-fold OOF average (scoring is off, the normal
/// case today) — the caller should then fall back to its default variant, behaving exactly as before.
/// When EVERY variant scored, it returns `Ok(Some(best))`. A partially-scored set (some variants
/// scored, others not) is an inconsistent host and is rejected so variants are never ranked unfairly.
///
/// `run_single_variant_fit_cv` runs FIT_CV for the single-variant plan into the supplied context
/// (the caller supplies the scheduler/data-provider wiring); this keeps the selection logic free of
/// host runtime details and unit-testable with mock controllers. Cloning a one-variant plan is
/// valid: `node_plans`/`fold_set` are plan-level (not keyed per variant) and variant params are
/// applied per-node at task build time, so the per-variant CV is isolated.
pub fn select_best_variant_by_cv<F>(
    plan: &ExecutionPlan,
    run_id: &RunId,
    root_seed: Option<u64>,
    selection_metric: RegressionMetricKind,
    mut run_single_variant_fit_cv: F,
) -> Result<Option<VariantId>>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
{
    plan.validate()?;
    if plan.variants.is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "cannot select a variant for a plan with no variants".to_string(),
        ));
    }

    let mut candidates: Vec<CandidateScore> = Vec::with_capacity(plan.variants.len());
    // Tracks whether ANY variant emitted scores at all (host targets present), so an empty candidate
    // set can be told apart from "scoring genuinely off" (no targets) — see the post-loop branch.
    let mut any_scores_seen = false;
    for variant in &plan.variants {
        let single_variant_plan = ExecutionPlan {
            variants: vec![variant.clone()],
            ..plan.clone()
        };
        let mut ctx = RunContext::new(run_id.clone(), root_seed);
        ctx.variant_id = Some(variant.variant_id.clone());
        run_single_variant_fit_cv(&single_variant_plan, &mut ctx)?;
        ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(plan))?;
        if !ctx.score_collector.is_empty() {
            any_scores_seen = true;
        }
        // `cross_fold_validation_reports` emits one cross-fold OOF average PER producer. Native SELECT
        // ranks a variant by a single score, so a multi-producer DAG is ambiguous and refused rather
        // than silently ranked on whichever producer happened to be first (an explicit score-target
        // producer is a future extension).
        let avg_reports = ctx
            .score_collector
            .iter()
            .filter(|report| {
                report.partition == PredictionPartition::Validation
                    && report
                        .fold_id
                        .as_ref()
                        .is_some_and(|fold| fold.as_str() == "avg")
            })
            .collect::<Vec<_>>();
        match avg_reports.as_slice() {
            [] => {}
            [report] => candidates.push(
                (*report)
                    .clone()
                    .into_candidate_score(variant.variant_id.as_str())?,
            ),
            _ => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "variant `{}` produced {} cross-fold OOF averages (multiple prediction producers); native SELECT needs a single score target",
                    variant.variant_id,
                    avg_reports.len()
                )));
            }
        }
    }

    if candidates.is_empty() {
        if any_scores_seen {
            // Targets WERE emitted, but no producer yielded a cross-fold average (e.g. a single fold,
            // where the average is skipped). We cannot rank — surface it instead of falling back.
            return Err(DagMlError::RuntimeValidation(
                "variants produced scores but no cross-fold OOF average; cannot rank — need >=2 folds or an explicit score target".to_string(),
            ));
        }
        // Native scoring is genuinely off (no host targets) — let the caller keep its default variant.
        return Ok(None);
    }
    if candidates.len() != plan.variants.len() {
        return Err(DagMlError::RuntimeValidation(format!(
            "native variant SELECT scored only {} of {} variants; cannot rank variants fairly",
            candidates.len(),
            plan.variants.len()
        )));
    }

    let policy = SelectionPolicy {
        id: format!("select:variant:{}", selection_metric.name()),
        metric: SelectionMetric {
            name: selection_metric.name().to_string(),
            objective: selection_metric.objective(),
        },
        required_metric_level: None,
        require_finite: true,
        evaluation_scope: None,
        refit_slot_plan: None,
        stacking_fit_contract: None,
        reduction_id: None,
    };
    let decision = select_candidate(&policy, &candidates)?;
    let selected = VariantId::new(decision.selected_candidate_id).map_err(|error| {
        DagMlError::RuntimeValidation(format!("selected variant id is invalid: {error}"))
    })?;
    Ok(Some(selected))
}

#[derive(Clone, Debug, Default)]
pub struct SequentialScheduler;

#[derive(Clone, Debug)]
pub struct ParallelScheduler {
    max_workers: usize,
}

impl ParallelScheduler {
    pub fn new(max_workers: usize) -> Result<Self> {
        if max_workers == 0 {
            return Err(DagMlError::RuntimeValidation(
                "parallel scheduler max_workers must be at least 1".to_string(),
            ));
        }
        Ok(Self { max_workers })
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
    }
}

#[derive(Clone, Debug)]
struct PhaseScope {
    phase: Phase,
    variant_id: Option<VariantId>,
    variant: Option<VariantExecutionSpec>,
    fold_id: Option<FoldId>,
    seed_root: Option<u64>,
}

#[derive(Clone, Debug)]
struct ReplayPredictionCacheContract {
    requirement: BundlePredictionRequirement,
    cache: BundlePredictionCacheRecord,
}

struct MaterializedReplayArtifacts {
    handles: BTreeMap<NodeId, BTreeMap<String, HandleRef>>,
    inputs: BTreeMap<NodeId, BTreeMap<String, ArtifactInputSpec>>,
}

#[derive(Default)]
struct PhaseScopeResources<'a> {
    data_provider: Option<&'a dyn RuntimeDataProvider>,
    replay_artifact_handles: Option<&'a BTreeMap<NodeId, BTreeMap<String, HandleRef>>>,
    replay_artifact_inputs: Option<&'a BTreeMap<NodeId, BTreeMap<String, ArtifactInputSpec>>>,
    replay_bundle_id: Option<&'a BundleId>,
    data_envelopes: Option<&'a BTreeMap<String, ExternalDataPlanEnvelope>>,
    prediction_cache_store: Option<&'a dyn RuntimePredictionCacheStore>,
    prediction_cache_contracts: Option<&'a BTreeMap<String, ReplayPredictionCacheContract>>,
    artifact_store: Option<&'a mut InMemoryArtifactStore>,
}

impl SequentialScheduler {
    pub fn execute_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources::default(),
        )
    }

    pub fn execute_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(data_provider),
                ..Default::default()
            },
        )
    }

    pub fn execute_campaign_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
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
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources::default(),
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
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
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider_and_artifact_store(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        artifact_store: &mut InMemoryArtifactStore,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
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
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        artifact_store: Some(&mut *artifact_store),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_bundle_replay(
        &self,
        replay: BundleReplayExecution<'_>,
        ctx: &mut RunContext,
    ) -> Result<Vec<NodeResult>> {
        replay.bundle.validate_against_plan(replay.plan)?;
        replay
            .replay_request
            .validate_for_bundle_with_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store.is_some(),
            )?;
        replay
            .bundle
            .validate_replay_envelopes(replay.data_envelopes)?;
        let prediction_cache_contracts = if replay.replay_request.phase == Phase::Refit {
            Some(replay_prediction_cache_contracts(replay.bundle)?)
        } else {
            None
        };
        if replay.replay_request.phase == Phase::Refit {
            preload_replay_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store,
                ctx,
            )?;
        }
        let replay_artifacts = materialize_replay_artifact_handles(
            replay.plan,
            replay.bundle,
            replay.replay_request,
            replay.artifact_store,
            ctx,
        )?;
        let selected_variant = replay
            .bundle
            .selected_variant_id
            .as_ref()
            .map(|selected| {
                replay
                    .plan
                    .variants
                    .iter()
                    .find(|variant| &variant.variant_id == selected)
                    .map(VariantExecutionSpec::from_plan)
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "bundle `{}` selected unknown variant `{selected}`",
                            replay.bundle.bundle_id
                        ))
                    })
            })
            .transpose()?;
        let seed_root = selected_variant
            .as_ref()
            .and_then(|variant| variant.seed)
            .or(ctx.root_seed);

        self.execute_phase_scope(
            replay.plan,
            replay.controllers,
            ctx,
            PhaseScope {
                phase: replay.replay_request.phase,
                variant_id: replay.bundle.selected_variant_id.clone(),
                variant: selected_variant,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(replay.data_provider),
                replay_artifact_handles: Some(&replay_artifacts.handles),
                replay_artifact_inputs: Some(&replay_artifacts.inputs),
                replay_bundle_id: Some(&replay.bundle.bundle_id),
                data_envelopes: Some(replay.data_envelopes),
                prediction_cache_store: replay.prediction_cache_store,
                prediction_cache_contracts: prediction_cache_contracts.as_ref(),
                ..Default::default()
            },
        )
    }

    fn execute_phase_scope(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        scope: PhaseScope,
        mut resources: PhaseScopeResources<'_>,
    ) -> Result<Vec<NodeResult>> {
        let _phase_span = crate::observability::phase_span(
            ctx.run_id.as_str(),
            plan.id.as_str(),
            scope.phase.as_str(),
            scope.variant_id.as_ref().map(VariantId::as_str),
            scope.fold_id.as_ref().map(FoldId::as_str),
        )
        .entered();
        let mut results = Vec::new();
        let mut output_handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
        let mut output_data_views =
            BTreeMap::<NodeId, BTreeMap<String, DataProviderViewSpec>>::new();
        let mut input_lineage = BTreeMap::<NodeId, LineageId>::new();

        for level in plan.node_parallel_levels_for_phase(scope.phase)? {
            for node_id in &level {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
                // Cross-branch merge reassembly (concat or late-fusion) is a
                // scheduler/runtime handler, not a controller call: it reads the
                // upstream branch OOF blocks from the prediction store and emits
                // one merged per-sample OOF block. Intercept it before the
                // controller path (and before the `requires_oof` edge collection,
                // which is a stacking contract the branch inputs do not satisfy).
                if let Some(reduction) = merge_reduction_mode(plan, node_plan) {
                    if let Some(result) =
                        reassemble_branch_merge(plan, node_plan, ctx, &scope, reduction)?
                    {
                        for prediction in &result.predictions {
                            ctx.prediction_store.append(prediction.clone())?;
                        }
                        apply_result_scoring(
                            &result,
                            &mut ctx.score_collector,
                            &mut ctx.regression_target_records,
                        )?;
                        ctx.lineage.record(result.lineage.clone())?;
                        output_handles.insert(node_id.clone(), result.outputs.clone());
                        input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
                        results.push(result);
                    }
                    continue;
                }
                let controller = controllers.get(&node_plan.controller_id).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "runtime controller `{}` is not registered",
                        node_plan.controller_id
                    ))
                })?;
                let collected_inputs = collect_input_handles(
                    plan,
                    node_plan,
                    &output_handles,
                    &output_data_views,
                    &resources,
                    ctx,
                    &scope,
                )?;
                let mut input_handles = collected_inputs.handles;
                let mut artifact_inputs = BTreeMap::new();
                if let Some(node_artifact_handles) = resources
                    .replay_artifact_handles
                    .and_then(|handles| handles.get(node_id))
                {
                    for (key, handle) in node_artifact_handles {
                        if input_handles.insert(key.clone(), handle.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact input `{key}`"
                            )));
                        }
                    }
                }
                if let Some(node_artifact_inputs) = resources
                    .replay_artifact_inputs
                    .and_then(|inputs| inputs.get(node_id))
                {
                    for (key, spec) in node_artifact_inputs {
                        if artifact_inputs.insert(key.clone(), spec.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact metadata `{key}`"
                            )));
                        }
                    }
                }
                let task_node_plan = effective_node_plan_for_scope(node_plan, &scope)?;
                let inner_fold_set = inner_fold_set_for_scope(
                    &plan.campaign,
                    plan.fold_set.as_ref(),
                    node_plan,
                    &scope,
                )?;
                let fit_influence = fit_influence_task_for_node(
                    plan,
                    &task_node_plan,
                    &collected_inputs.data_views,
                )?;
                let task = NodeTask {
                    inner_fold_set,
                    run_id: ctx.run_id.clone(),
                    node_plan: task_node_plan.clone(),
                    phase: scope.phase,
                    variant_id: scope.variant_id.clone(),
                    variant: scope.variant.clone(),
                    fold_id: scope.fold_id.clone(),
                    branch_path: Vec::new(),
                    input_handles,
                    data_views: collected_inputs.data_views,
                    prediction_inputs: collected_inputs.prediction_inputs,
                    artifact_inputs,
                    fit_influence,
                    seed: derive_task_seed(
                        scope.seed_root,
                        scope.variant_id.as_ref(),
                        scope.fold_id.as_ref(),
                        &task_node_plan,
                        scope.phase,
                    ),
                };
                let _node_span = crate::observability::node_span(
                    task.run_id.as_str(),
                    plan.id.as_str(),
                    task.phase.as_str(),
                    task.node_plan.node_id.as_str(),
                    task.node_plan.controller_id.as_str(),
                )
                .entered();
                let mut result = controller.invoke(&task)?;
                record_fit_influence_diagnostic(&task, &mut result);
                result.validate_for_task(&task)?;
                apply_result_prediction_aggregation(
                    plan,
                    controllers,
                    &task,
                    &mut result,
                    &resources,
                )?;
                attach_coordinator_input_lineage(
                    &mut result,
                    plan,
                    &task.node_plan.node_id,
                    &input_lineage,
                )?;
                if let Some(store) = resources.artifact_store.as_deref_mut() {
                    if scope.phase == Phase::Refit {
                        store.capture_refit_artifacts(&task, &result)?;
                    }
                }
                for prediction in &result.predictions {
                    ctx.prediction_store.append(prediction.clone())?;
                }
                for prediction in &result.aggregated_predictions {
                    ctx.aggregated_prediction_store.append(prediction.clone())?;
                }
                apply_result_scoring(
                    &result,
                    &mut ctx.score_collector,
                    &mut ctx.regression_target_records,
                )?;
                ctx.lineage.record(result.lineage.clone())?;
                let data_views = derive_output_data_views(plan, &task, &result)?;
                output_handles.insert(node_id.clone(), result.outputs.clone());
                output_data_views.insert(node_id.clone(), data_views);
                input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
                results.push(result);
            }
        }

        Ok(results)
    }
}

impl ParallelScheduler {
    pub fn execute_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources::default(),
        )
    }

    pub fn execute_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(data_provider),
                ..Default::default()
            },
        )
    }

    pub fn execute_campaign_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
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
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources::default(),
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
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
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider_and_artifact_store(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        artifact_store: &mut InMemoryArtifactStore,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
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
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        artifact_store: Some(&mut *artifact_store),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_bundle_replay(
        &self,
        replay: BundleReplayExecution<'_>,
        ctx: &mut RunContext,
    ) -> Result<Vec<NodeResult>> {
        replay.bundle.validate_against_plan(replay.plan)?;
        replay
            .replay_request
            .validate_for_bundle_with_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store.is_some(),
            )?;
        replay
            .bundle
            .validate_replay_envelopes(replay.data_envelopes)?;
        let prediction_cache_contracts = if replay.replay_request.phase == Phase::Refit {
            Some(replay_prediction_cache_contracts(replay.bundle)?)
        } else {
            None
        };
        if replay.replay_request.phase == Phase::Refit {
            preload_replay_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store,
                ctx,
            )?;
        }
        let replay_artifacts = materialize_replay_artifact_handles(
            replay.plan,
            replay.bundle,
            replay.replay_request,
            replay.artifact_store,
            ctx,
        )?;
        let selected_variant = replay
            .bundle
            .selected_variant_id
            .as_ref()
            .map(|selected| {
                replay
                    .plan
                    .variants
                    .iter()
                    .find(|variant| &variant.variant_id == selected)
                    .map(VariantExecutionSpec::from_plan)
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "bundle `{}` selected unknown variant `{selected}`",
                            replay.bundle.bundle_id
                        ))
                    })
            })
            .transpose()?;
        let seed_root = selected_variant
            .as_ref()
            .and_then(|variant| variant.seed)
            .or(ctx.root_seed);

        self.execute_phase_scope(
            replay.plan,
            replay.controllers,
            ctx,
            PhaseScope {
                phase: replay.replay_request.phase,
                variant_id: replay.bundle.selected_variant_id.clone(),
                variant: selected_variant,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(replay.data_provider),
                replay_artifact_handles: Some(&replay_artifacts.handles),
                replay_artifact_inputs: Some(&replay_artifacts.inputs),
                replay_bundle_id: Some(&replay.bundle.bundle_id),
                data_envelopes: Some(replay.data_envelopes),
                prediction_cache_store: replay.prediction_cache_store,
                prediction_cache_contracts: prediction_cache_contracts.as_ref(),
                ..Default::default()
            },
        )
    }

    fn execute_phase_scope(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        scope: PhaseScope,
        mut resources: PhaseScopeResources<'_>,
    ) -> Result<Vec<NodeResult>> {
        // Hold the phase span on the scheduler thread, and clone it into each
        // worker so worker-thread telemetry nests under the phase (tracing spans
        // are thread-local and do not auto-propagate across `thread::scope`).
        let phase_span = crate::observability::phase_span(
            ctx.run_id.as_str(),
            plan.id.as_str(),
            scope.phase.as_str(),
            scope.variant_id.as_ref().map(VariantId::as_str),
            scope.fold_id.as_ref().map(FoldId::as_str),
        );
        let _phase_entered = phase_span.clone().entered();
        // Borrowed for the `thread::scope` below; workers join before it ends.
        let plan_id = plan.id.as_str();
        plan.validate_parallel_controller_capabilities(self.max_workers, scope.phase)?;
        let mut results = Vec::new();
        let mut output_handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
        let mut output_data_views =
            BTreeMap::<NodeId, BTreeMap<String, DataProviderViewSpec>>::new();
        let mut input_lineage = BTreeMap::<NodeId, LineageId>::new();

        for level in plan.node_parallel_levels_for_phase(scope.phase)? {
            let mut prepared = Vec::<PreparedNodeTask>::new();
            // Cross-branch merge nodes (concat or late-fusion) are not controller
            // tasks: they read the upstream branch OOF blocks from the prediction
            // store and reassemble them on the scheduler thread (no worker), AFTER
            // this level's worker tasks have populated the store. They are in a
            // later level than their branches, so the store already holds the
            // branch OOF by the time we reassemble — see `reassemble_branch_merge`.
            let mut merge_nodes = Vec::<(NodeId, MergeReduction)>::new();
            for node_id in &level {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
                if let Some(reduction) = merge_reduction_mode(plan, node_plan) {
                    merge_nodes.push((node_id.clone(), reduction));
                    continue;
                }
                let collected_inputs = collect_input_handles(
                    plan,
                    node_plan,
                    &output_handles,
                    &output_data_views,
                    &resources,
                    ctx,
                    &scope,
                )?;
                let mut input_handles = collected_inputs.handles;
                let mut artifact_inputs = BTreeMap::new();
                if let Some(node_artifact_handles) = resources
                    .replay_artifact_handles
                    .and_then(|handles| handles.get(node_id))
                {
                    for (key, handle) in node_artifact_handles {
                        if input_handles.insert(key.clone(), handle.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact input `{key}`"
                            )));
                        }
                    }
                }
                if let Some(node_artifact_inputs) = resources
                    .replay_artifact_inputs
                    .and_then(|inputs| inputs.get(node_id))
                {
                    for (key, spec) in node_artifact_inputs {
                        if artifact_inputs.insert(key.clone(), spec.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact metadata `{key}`"
                            )));
                        }
                    }
                }
                let task_node_plan = effective_node_plan_for_scope(node_plan, &scope)?;
                let inner_fold_set = inner_fold_set_for_scope(
                    &plan.campaign,
                    plan.fold_set.as_ref(),
                    node_plan,
                    &scope,
                )?;
                let fit_influence = fit_influence_task_for_node(
                    plan,
                    &task_node_plan,
                    &collected_inputs.data_views,
                )?;
                prepared.push(PreparedNodeTask {
                    node_id: node_id.clone(),
                    task: NodeTask {
                        inner_fold_set,
                        run_id: ctx.run_id.clone(),
                        node_plan: task_node_plan.clone(),
                        phase: scope.phase,
                        variant_id: scope.variant_id.clone(),
                        variant: scope.variant.clone(),
                        fold_id: scope.fold_id.clone(),
                        branch_path: Vec::new(),
                        input_handles,
                        data_views: collected_inputs.data_views,
                        prediction_inputs: collected_inputs.prediction_inputs,
                        artifact_inputs,
                        fit_influence,
                        seed: derive_task_seed(
                            scope.seed_root,
                            scope.variant_id.as_ref(),
                            scope.fold_id.as_ref(),
                            &task_node_plan,
                            scope.phase,
                        ),
                    },
                });
            }

            for chunk in prepared.chunks(self.max_workers) {
                let chunk_results =
                    std::thread::scope(|thread_scope| -> Result<Vec<NodeResult>> {
                        let mut handles = Vec::with_capacity(chunk.len());
                        for prepared_task in chunk {
                            let controller = controllers
                                .get(&prepared_task.task.node_plan.controller_id)
                                .ok_or_else(|| {
                                    DagMlError::RuntimeValidation(format!(
                                        "runtime controller `{}` is not registered",
                                        prepared_task.task.node_plan.controller_id
                                    ))
                                })?;
                            let worker_span = phase_span.clone();
                            handles.push(thread_scope.spawn(move || {
                                let _worker_span = worker_span.entered();
                                let _node_span = crate::observability::node_span(
                                    prepared_task.task.run_id.as_str(),
                                    plan_id,
                                    prepared_task.task.phase.as_str(),
                                    prepared_task.task.node_plan.node_id.as_str(),
                                    prepared_task.task.node_plan.controller_id.as_str(),
                                )
                                .entered();
                                let mut result = controller.invoke(&prepared_task.task)?;
                                record_fit_influence_diagnostic(&prepared_task.task, &mut result);
                                result.validate_for_task(&prepared_task.task)?;
                                Ok(result)
                            }));
                        }
                        handles
                            .into_iter()
                            .map(|handle| {
                                handle.join().map_err(|_| {
                                    DagMlError::RuntimeValidation(
                                        "parallel scheduler worker panicked".to_string(),
                                    )
                                })?
                            })
                            .collect()
                    })?;

                for (prepared_task, mut result) in chunk.iter().zip(chunk_results) {
                    apply_result_prediction_aggregation(
                        plan,
                        controllers,
                        &prepared_task.task,
                        &mut result,
                        &resources,
                    )?;
                    attach_coordinator_input_lineage(
                        &mut result,
                        plan,
                        &prepared_task.task.node_plan.node_id,
                        &input_lineage,
                    )?;
                    if let Some(store) = resources.artifact_store.as_deref_mut() {
                        if scope.phase == Phase::Refit {
                            store.capture_refit_artifacts(&prepared_task.task, &result)?;
                        }
                    }
                    for prediction in &result.predictions {
                        ctx.prediction_store.append(prediction.clone())?;
                    }
                    for prediction in &result.aggregated_predictions {
                        ctx.aggregated_prediction_store.append(prediction.clone())?;
                    }
                    apply_result_scoring(
                        &result,
                        &mut ctx.score_collector,
                        &mut ctx.regression_target_records,
                    )?;
                    ctx.lineage.record(result.lineage.clone())?;
                    let data_views = derive_output_data_views(plan, &prepared_task.task, &result)?;
                    output_handles.insert(prepared_task.node_id.clone(), result.outputs.clone());
                    output_data_views.insert(prepared_task.node_id.clone(), data_views);
                    input_lineage.insert(
                        prepared_task.node_id.clone(),
                        result.lineage.record_id.clone(),
                    );
                    results.push(result);
                }
            }

            // Reassemble any cross-branch merge nodes in this level now that the
            // level's worker tasks have populated the prediction store. Merge nodes
            // sit in a later level than the branches they consume, so the upstream
            // branch OOF is already present.
            for (node_id, reduction) in &merge_nodes {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
                if let Some(result) =
                    reassemble_branch_merge(plan, node_plan, ctx, &scope, *reduction)?
                {
                    for prediction in &result.predictions {
                        ctx.prediction_store.append(prediction.clone())?;
                    }
                    apply_result_scoring(
                        &result,
                        &mut ctx.score_collector,
                        &mut ctx.regression_target_records,
                    )?;
                    ctx.lineage.record(result.lineage.clone())?;
                    output_handles.insert(node_id.clone(), result.outputs.clone());
                    input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
                    results.push(result);
                }
            }
        }

        Ok(results)
    }
}

struct PreparedNodeTask {
    node_id: NodeId,
    task: NodeTask,
}

fn attach_coordinator_input_lineage(
    result: &mut NodeResult,
    plan: &ExecutionPlan,
    node_id: &NodeId,
    upstream_lineage: &BTreeMap<NodeId, LineageId>,
) -> Result<()> {
    let inferred = inferred_input_lineage_for_node(plan, node_id, upstream_lineage);
    if result.lineage.input_lineage.is_empty() {
        result.lineage.input_lineage = inferred;
        return Ok(());
    }

    let declared = result
        .lineage
        .input_lineage
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if declared != inferred {
        return Err(DagMlError::RuntimeValidation(format!(
            "lineage for node `{}` declared input lineage {:?}, expected {:?}",
            result.node_id, declared, inferred
        )));
    }
    result.lineage.input_lineage = declared;
    Ok(())
}

fn inferred_input_lineage_for_node(
    plan: &ExecutionPlan,
    node_id: &NodeId,
    upstream_lineage: &BTreeMap<NodeId, LineageId>,
) -> Vec<LineageId> {
    plan.graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| &edge.target.node_id == node_id && edge.contract.propagates_lineage)
        .filter_map(|edge| upstream_lineage.get(&edge.source.node_id).cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

const SCORE_METRICS: &[RegressionMetricKind] = &[
    RegressionMetricKind::Mse,
    RegressionMetricKind::Rmse,
    RegressionMetricKind::Mae,
    RegressionMetricKind::R2,
    RegressionMetricKind::Accuracy,
];

/// True when a Sample-level target block covers EXACTLY the prediction block's samples — the pairing
/// dag-ml's scoring requires (target units == prediction units). Lets one result carry several
/// sample-level blocks (e.g. refit's final-train + final-test), each with its own y_true.
fn sample_targets_match_block(block: &PredictionBlock, targets: &RegressionTargetBlock) -> bool {
    if targets.level != PredictionLevel::Sample || targets.unit_ids.len() != block.sample_ids.len()
    {
        return false;
    }
    let predicted: BTreeSet<&SampleId> = block.sample_ids.iter().collect();
    targets.unit_ids.iter().all(|unit| match unit {
        PredictionUnitId::Sample(sample_id) => predicted.contains(sample_id),
        _ => false,
    })
}

/// Score a result's prediction blocks against the host-supplied `regression_targets` and push the
/// reports into the collector. Native scoring is gated purely on the host emitting targets: a run
/// that emits no `regression_targets` (every existing run) collects nothing, so behavior is
/// unchanged and the campaign fingerprint is untouched. Each Sample prediction block is paired with
/// the target block covering exactly its samples; unmatched blocks are unscored.
fn apply_result_scoring(
    result: &NodeResult,
    collector: &mut Vec<RegressionMetricReport>,
    target_records: &mut Vec<RegressionTargetRecord>,
) -> Result<()> {
    if result.regression_targets.is_empty() {
        return Ok(());
    }
    for block in &result.predictions {
        if let Some(targets) = result
            .regression_targets
            .iter()
            .find(|targets| sample_targets_match_block(block, targets))
        {
            let mut report = score_regression_prediction_block(block, targets, SCORE_METRICS)?;
            report.variant_id = result.lineage.variant_id.clone();
            collector.push(report);
            // Retain y_true (tagged with its variant/fold/partition) so the OOF average can be
            // scored later, per-variant.
            target_records.push(RegressionTargetRecord {
                producer_node: block.producer_node.clone(),
                variant_id: result.lineage.variant_id.clone(),
                partition: block.partition.clone(),
                fold_id: block.fold_id.clone(),
                block: targets.clone(),
            });
        }
    }
    for block in &result.aggregated_predictions {
        if let Some(targets) = result
            .regression_targets
            .iter()
            .find(|targets| targets.level == block.level)
        {
            let mut report = score_regression_aggregated_block(block, targets, SCORE_METRICS)?;
            report.variant_id = result.lineage.variant_id.clone();
            collector.push(report);
        }
    }
    Ok(())
}

fn apply_result_prediction_aggregation(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    task: &NodeTask,
    result: &mut NodeResult,
    resources: &PhaseScopeResources<'_>,
) -> Result<()> {
    let has_observation_predictions = !result.observation_predictions.is_empty();
    let has_sample_predictions = !result.predictions.is_empty();
    if !has_observation_predictions && !has_sample_predictions {
        return Ok(());
    }
    let Some(shape_plan) = &task.node_plan.shape_plan else {
        if !has_observation_predictions {
            return Ok(());
        }
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` emitted observation predictions but has no data/model shape plan for aggregation",
            task.node_plan.node_id
        )));
    };
    let policy = &shape_plan.aggregation_policy;
    if !policy.store_aggregated_predictions {
        return Ok(());
    }
    if policy.aggregation_level == PredictionLevel::Observation {
        return Ok(());
    }
    if !has_observation_predictions && policy.aggregation_level == PredictionLevel::Sample {
        return Ok(());
    }

    let mut derived_sample_blocks = Vec::new();
    if !result.observation_predictions.is_empty() {
        let relations = coordinator_relations_for_task(task, resources)?;
        let sample_policy = observation_to_sample_policy(policy);
        for block in result.observation_predictions.clone() {
            let requested_sample_order =
                requested_sample_order_for_observation_block(plan, task, &block, &relations)?;
            let sample_block =
                if sample_policy.method == crate::policy::AggregationMethod::CustomController {
                    dispatch_custom_observation_aggregation(
                        plan,
                        controllers,
                        aggregation_task_id(
                            task,
                            &block.producer_node,
                            block.fold_id.as_ref(),
                            "obs_to_sample",
                        ),
                        block,
                        relations.clone(),
                        sample_policy.clone(),
                        requested_sample_order,
                    )?
                } else {
                    aggregate_observation_predictions(
                        &block,
                        &relations,
                        &sample_policy,
                        &requested_sample_order,
                    )?
                };
            derived_sample_blocks.push(sample_block);
        }
    }

    if policy.aggregation_level == PredictionLevel::Sample {
        result.predictions.extend(derived_sample_blocks);
        result.validate_for_task(task)?;
        return Ok(());
    }

    if !result.aggregated_predictions.is_empty() {
        // The controller emitted aggregated blocks itself, bypassing native
        // aggregation. They must still MATCH the node's aggregation policy
        // level — otherwise a block aggregated at the wrong unit level would be
        // accepted and scored against a mismatched policy.
        for block in &result.aggregated_predictions {
            if block.level != policy.aggregation_level {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted aggregated predictions at level {:?} but its aggregation policy is {:?}",
                    task.node_plan.node_id, block.level, policy.aggregation_level
                )));
            }
        }
        result.validate_for_task(task)?;
        return Ok(());
    }

    let relations = coordinator_relations_for_task(task, resources)?;
    let sample_blocks = result
        .predictions
        .iter()
        .cloned()
        .chain(derived_sample_blocks)
        .collect::<Vec<_>>();
    for block in sample_blocks {
        let requested_unit_order =
            requested_unit_order_for_sample_block(policy.aggregation_level, &relations, &block)?;
        let aggregated = if policy.method == crate::policy::AggregationMethod::CustomController {
            dispatch_custom_sample_aggregation(
                plan,
                controllers,
                aggregation_task_id(
                    task,
                    &block.producer_node,
                    block.fold_id.as_ref(),
                    "sample_to_unit",
                ),
                block,
                relations.clone(),
                policy.clone(),
                requested_unit_order,
            )?
        } else {
            aggregate_sample_predictions_by_unit(&block, &relations, policy, &requested_unit_order)?
        };
        result.aggregated_predictions.push(aggregated);
    }
    result.validate_for_task(task)
}

fn observation_to_sample_policy(policy: &AggregationPolicy) -> AggregationPolicy {
    let mut sample_policy = policy.clone();
    sample_policy.aggregation_level = PredictionLevel::Sample;
    sample_policy
}

fn coordinator_relations_for_task(
    task: &NodeTask,
    resources: &PhaseScopeResources<'_>,
) -> Result<SampleRelationSet> {
    coordinator_relations_for_node(&task.node_plan, resources)?.ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "node `{}` needs coordinator relations for prediction aggregation but no matching data provider/envelope carries relations",
            task.node_plan.node_id
        ))
    })
}

fn coordinator_relations_for_edge(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    resources: &PhaseScopeResources<'_>,
) -> Result<SampleRelationSet> {
    let target_plan = plan.node_plans.get(&edge.target.node_id).ok_or_else(|| {
        DagMlError::Planning(format!(
            "OOF edge target node `{}` has no node plan",
            edge.target.node_id
        ))
    })?;
    if let Some(relations) = coordinator_relations_for_node(target_plan, resources)? {
        return Ok(relations);
    }

    let source_plan = plan.node_plans.get(&edge.source.node_id).ok_or_else(|| {
        DagMlError::Planning(format!(
            "OOF edge source node `{}` has no node plan",
            edge.source.node_id
        ))
    })?;
    if let Some(relations) = coordinator_relations_for_node(source_plan, resources)? {
        return Ok(relations);
    }

    Err(DagMlError::RuntimeValidation(format!(
        "edge `{}.{}` -> `{}.{}` needs coordinator relations for aggregated OOF validation but neither endpoint has a relation-carrying data binding",
        edge.source.node_id,
        edge.source.port_name,
        edge.target.node_id,
        edge.target.port_name
    )))
}

fn coordinator_relations_for_node(
    node_plan: &NodePlan,
    resources: &PhaseScopeResources<'_>,
) -> Result<Option<SampleRelationSet>> {
    let mut selected: Option<SampleRelationSet> = None;
    for binding in &node_plan.data_bindings {
        if !binding.require_relations && binding.relation_fingerprint.is_none() {
            continue;
        }
        let relations = if let Some(envelopes) = resources.data_envelopes {
            let key = format!("{}.{}", binding.node_id, binding.input_name);
            match envelopes.get(&key) {
                Some(envelope) => {
                    binding.validate_envelope(envelope)?;
                    envelope.coordinator_relations.clone()
                }
                None => None,
            }
        } else if let Some(data_provider) = resources.data_provider {
            data_provider.coordinator_relations(binding)?
        } else {
            None
        };
        let Some(relations) = relations else {
            // A binding that REQUIRES relations must resolve them. Silently
            // defaulting to empty exclusions (no excluded samples) would let a
            // leakage / branch / exclusion / aggregation policy run without the
            // relation set it depends on, so refuse instead of degrading.
            if binding.require_relations {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` binding `{}` requires coordinator relations but none were resolved",
                    node_plan.node_id, binding.input_name
                )));
            }
            continue;
        };
        if let Some(previous) = &selected {
            if previous != &relations {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` has multiple non-identical coordinator relation sets",
                    node_plan.node_id
                )));
            }
        } else {
            selected = Some(relations);
        }
    }
    Ok(selected)
}

fn requested_sample_order_for_observation_block(
    plan: &ExecutionPlan,
    task: &NodeTask,
    block: &ObservationPredictionBlock,
    relations: &SampleRelationSet,
) -> Result<Vec<SampleId>> {
    if block.partition == PredictionPartition::Validation {
        if let Some(sample_ids) = validation_view_sample_ids(task) {
            return Ok(sample_ids.into_iter().collect());
        }
        if let (Some(fold_set), Some(fold_id)) = (plan.fold_set.as_ref(), block.fold_id.as_ref()) {
            if let Some(fold) = fold_set.folds.iter().find(|fold| &fold.fold_id == fold_id) {
                return Ok(fold.validation_sample_ids.clone());
            }
        }
    }
    first_seen_samples_for_observations(block, relations)
}

fn first_seen_samples_for_observations(
    block: &ObservationPredictionBlock,
    relations: &SampleRelationSet,
) -> Result<Vec<SampleId>> {
    let mut seen = BTreeSet::new();
    let mut sample_order = Vec::new();
    for observation_id in &block.observation_ids {
        let sample_id = relations
            .sample_for_observation(observation_id)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "observation prediction `{observation_id}` has no sample relation"
                ))
            })?;
        if seen.insert(sample_id.clone()) {
            sample_order.push(sample_id.clone());
        }
    }
    Ok(sample_order)
}

fn requested_unit_order_for_sample_block(
    level: PredictionLevel,
    relations: &SampleRelationSet,
    block: &PredictionBlock,
) -> Result<Vec<PredictionUnitId>> {
    let mut seen = BTreeSet::new();
    let mut unit_order = Vec::new();
    for sample_id in &block.sample_ids {
        let unit_id = match level {
            PredictionLevel::Sample => PredictionUnitId::Sample(sample_id.clone()),
            PredictionLevel::Target => relations
                .target_for_sample(sample_id)
                .cloned()
                .map(PredictionUnitId::Target)
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "sample `{sample_id}` is missing target id for target aggregation"
                    ))
                })?,
            PredictionLevel::Group => relations
                .group_for_sample(sample_id)
                .cloned()
                .map(PredictionUnitId::Group)
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "sample `{sample_id}` is missing group id for group aggregation"
                    ))
                })?,
            PredictionLevel::Observation => {
                return Err(DagMlError::OofValidation(
                    "sample prediction aggregation cannot output observation-level predictions"
                        .to_string(),
                ));
            }
        };
        if seen.insert(unit_id.clone()) {
            unit_order.push(unit_id);
        }
    }
    Ok(unit_order)
}

fn aggregation_task_id(
    task: &NodeTask,
    producer_node: &NodeId,
    fold_id: Option<&FoldId>,
    stage: &str,
) -> String {
    let fold = fold_id
        .map(ToString::to_string)
        .unwrap_or_else(|| "nofold".to_string());
    format!(
        "aggregation:{}:{}:{}:{}:{}",
        task.run_id, task.node_plan.node_id, producer_node, fold, stage
    )
}

fn collect_input_handles(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    output_handles: &BTreeMap<NodeId, BTreeMap<String, HandleRef>>,
    output_data_views: &BTreeMap<NodeId, BTreeMap<String, DataProviderViewSpec>>,
    resources: &PhaseScopeResources<'_>,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<CollectedInputs> {
    let mut inputs = BTreeMap::new();
    let mut data_views = BTreeMap::new();
    let mut prediction_inputs = BTreeMap::new();
    let training_oof_edges = incoming_training_oof_edges(plan, node_plan, scope)?;
    let training_oof_sources = training_oof_edges
        .iter()
        .map(|edge| edge.source.node_id.clone())
        .collect::<BTreeSet<_>>();
    let bound_data_inputs = node_plan
        .data_bindings
        .iter()
        .map(|binding| binding.input_name.clone())
        .collect::<BTreeSet<_>>();
    // Only forward upstream handles for ports this node DECLARES an edge to.
    // A controller must never see a handle outside its declared port contract,
    // so a sibling consumer of the same producer cannot expose extra ports here.
    let declared_source_ports = plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node_plan.node_id)
        .map(|edge| (edge.source.node_id.clone(), edge.source.port_name.clone()))
        .collect::<BTreeSet<_>>();
    for upstream in &node_plan.input_nodes {
        if training_oof_sources.contains(upstream) {
            continue;
        }
        if let Some(handles) = output_handles.get(upstream) {
            for (port, handle) in handles {
                if !declared_source_ports.contains(&(upstream.clone(), port.clone())) {
                    continue;
                }
                inputs.insert(format!("{upstream}.{port}"), handle.clone());
            }
        }
    }
    for edge in plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node_plan.node_id)
        .filter(|edge| edge.contract.kind == PortKind::Data && !edge.contract.requires_oof)
    {
        if bound_data_inputs.contains(&edge.target.port_name) {
            continue;
        }
        let Some(handles) = output_handles.get(&edge.source.node_id) else {
            continue;
        };
        let Some(handle) = handles.get(&edge.source.port_name) else {
            continue;
        };
        let key = data_view_key(&edge.target.port_name);
        if inputs.insert(key.clone(), handle.clone()).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` received duplicate data edge input `{key}`",
                node_plan.node_id
            )));
        }
        if let Some(source_views) = output_data_views.get(&edge.source.node_id) {
            if let Some(view) = source_views.get(&edge.source.port_name) {
                if data_views.insert(key.clone(), view.clone()).is_some() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate data edge view `{key}`",
                        node_plan.node_id
                    )));
                }
            }
            let source_validation_key = validation_data_view_key(&edge.source.port_name);
            if let Some(view) = source_views.get(&source_validation_key) {
                let validation_key = format!("{key}:validation");
                if data_views
                    .insert(validation_key.clone(), view.clone())
                    .is_some()
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate data edge validation view `{validation_key}`",
                        node_plan.node_id
                    )));
                }
            }
        }
    }
    for edge in training_oof_edges {
        let key = format!("{}.{}", edge.source.node_id, edge.source.port_name);
        let input = collect_oof_prediction_input(plan, edge, ctx, scope, resources)?;
        if inputs.insert(key.clone(), input.handle).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` received duplicate OOF prediction input `{key}`",
                node_plan.node_id
            )));
        }
        if prediction_inputs.insert(key.clone(), input.spec).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` received duplicate OOF prediction spec `{key}`",
                node_plan.node_id
            )));
        }
    }
    // REFIT / PREDICT: deliver each base producer's off-fold (test / predict)
    // predictions to the stacking meta-node as a SEPARATE prediction input (suffixed
    // `:test` / `:predict`) so the host meta-model predicts from them. The FIT_CV
    // Validation-OOF input above is the meta-features the meta-model trains on; this
    // off-fold input is used ONLY for REFIT/PREDICT scoring/prediction, never FIT_CV
    // training — keeping the leakage invariant intact.
    if matches!(scope.phase, Phase::Refit | Phase::Predict) {
        let off_fold_suffix = scope.phase.as_str().to_ascii_lowercase();
        for edge in incoming_oof_edges(plan, node_plan)? {
            let Some(input) = collect_off_fold_prediction_input(plan, edge, ctx, scope)? else {
                continue;
            };
            let key = format!(
                "{}.{}:{off_fold_suffix}",
                edge.source.node_id, edge.source.port_name
            );
            if inputs.insert(key.clone(), input.handle).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate off-fold prediction input `{key}`",
                    node_plan.node_id
                )));
            }
            if prediction_inputs.insert(key.clone(), input.spec).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate off-fold prediction spec `{key}`",
                    node_plan.node_id
                )));
            }
        }
    }
    if !node_plan.data_bindings.is_empty() && resources.data_provider.is_none() {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` requires {} data binding(s) but no runtime data provider is registered",
            node_plan.node_id,
            node_plan.data_bindings.len()
        )));
    }
    if let Some(data_provider) = resources.data_provider {
        // Samples excluded from training (sample-local) for this node, derived
        // from its coordinator relations. Used to filter FIT view specs so the
        // spec, the materialized view, and fit-influence row_weights agree.
        let excluded_samples = coordinator_relations_for_node(node_plan, resources)?
            .map(|relations| relations.excluded_sample_ids())
            .unwrap_or_default();
        for binding in &node_plan.data_bindings {
            let materialized = data_provider.materialize(&DataMaterializationRequest {
                run_id: ctx.run_id.clone(),
                node_id: node_plan.node_id.clone(),
                input_name: binding.input_name.clone(),
                phase: scope.phase,
                variant_id: scope.variant_id.clone(),
                fold_id: scope.fold_id.clone(),
                binding: binding.clone(),
            })?;
            let branch_view_for_node = branch_view_from_node_metadata(plan, &node_plan.node_id)?;
            let view = data_view_for_scope(
                binding,
                plan.fold_set.as_ref(),
                scope,
                branch_view_for_node.as_ref(),
                &excluded_samples,
            )?;
            let key = data_view_key(&binding.input_name);
            let view_handle = make_data_view_handle(
                data_provider,
                ctx,
                node_plan,
                scope,
                binding,
                &materialized,
                &view,
            )?;
            if data_views.insert(key.clone(), view).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate data view `{key}`",
                    node_plan.node_id
                )));
            }
            if inputs.insert(key.clone(), view_handle).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate data input `{key}`",
                    node_plan.node_id
                )));
            }

            if let Some(validation_view) = validation_data_view_for_scope(
                binding,
                plan.fold_set.as_ref(),
                scope,
                branch_view_for_node.as_ref(),
                &excluded_samples,
            )? {
                let validation_key = format!("{key}:validation");
                let validation_handle = make_data_view_handle(
                    data_provider,
                    ctx,
                    node_plan,
                    scope,
                    binding,
                    &materialized,
                    &validation_view,
                )?;
                if data_views
                    .insert(validation_key.clone(), validation_view)
                    .is_some()
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate validation data view `{validation_key}`",
                        node_plan.node_id
                    )));
                }
                if inputs
                    .insert(validation_key.clone(), validation_handle)
                    .is_some()
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate validation data input `{validation_key}`",
                        node_plan.node_id
                    )));
                }
            }
        }
    }
    Ok(CollectedInputs {
        handles: inputs,
        data_views,
        prediction_inputs,
    })
}

fn effective_node_plan_for_scope(node_plan: &NodePlan, scope: &PhaseScope) -> Result<NodePlan> {
    let Some(variant) = &scope.variant else {
        return Ok(node_plan.clone());
    };
    let params = variant.effective_params_for_node(&node_plan.node_id, &node_plan.params)?;
    if params == node_plan.params {
        return Ok(node_plan.clone());
    }
    let mut node_plan = node_plan.clone();
    node_plan.params = params;
    node_plan.params_fingerprint = stable_json_fingerprint(&node_plan.params)?;
    Ok(node_plan)
}

fn incoming_oof_edges<'a>(
    plan: &'a ExecutionPlan,
    node_plan: &NodePlan,
) -> Result<Vec<&'a EdgeSpec>> {
    plan.graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node_plan.node_id && edge.contract.requires_oof)
        .map(|edge| {
            if edge.contract.kind != PortKind::Prediction {
                return Err(DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` requires OOF but is not a prediction edge",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
            Ok(edge)
        })
        .collect()
}

fn incoming_training_oof_edges<'a>(
    plan: &'a ExecutionPlan,
    node_plan: &NodePlan,
    scope: &PhaseScope,
) -> Result<Vec<&'a EdgeSpec>> {
    if !scope.phase.is_training() {
        return Ok(Vec::new());
    }
    incoming_oof_edges(plan, node_plan)
}

/// The base producer's off-fold (test / predict) predictions delivered to a
/// stacking meta-node as a SEPARATE prediction input in REFIT / PREDICT, so the
/// host meta-model can predict from them. This is the prediction-stacking analogue
/// of the concat/fusion off-fold reassembly: the FIT_CV `requires_oof` path stays
/// Validation-OOF-only (the meta-features the meta-model trains on), and these
/// test/predict base predictions are a distinct input used ONLY in REFIT/PREDICT
/// scoring — never in FIT_CV training.
///
/// Reads the base producer's `fold_id == None` block in the phase-expected
/// partition (`Test` in REFIT / `Final` in PREDICT) scoped to the active variant,
/// and builds a [`CollectedPredictionInput`] (a prediction handle + a
/// [`PredictionInputSpec`] carrying its per-sample `values`), mirroring the
/// FIT_CV OOF input so the host adapter sees a handle alongside the spec. Returns
/// `None` when the base produced no such block (a phase with no base prediction).
///
/// LEAKAGE INVARIANT: never reads a `Validation` block, so the Validation-OOF
/// meta-features are untouched. Only runs in REFIT/PREDICT (the caller guards it),
/// and the phase-expected-partition filter keeps a stale `Final`/`Train` block
/// from a prior phase in the same context out of the meta-features.
fn collect_off_fold_prediction_input(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<Option<CollectedPredictionInput>> {
    let expected_partition = expected_off_fold_partition(scope.phase);
    let blocks: Vec<&PredictionBlock> = ctx
        .prediction_store
        .find(Some(&edge.source.node_id), Some(&expected_partition), None)
        .into_iter()
        .filter(|block| block.fold_id.is_none())
        .collect();
    if blocks.is_empty() {
        return Ok(None);
    }
    if blocks.len() > 1 {
        return Err(DagMlError::OofValidation(format!(
            "meta node `{}` found {} off-fold ({expected_partition:?}) blocks for base `{}`: the run context mixes several variants — predict each variant in its own context (native SELECT does this)",
            edge.target.node_id,
            blocks.len(),
            edge.source.node_id,
        )));
    }
    let block = blocks[0];
    let width = block.validate_shape()?;
    let target_names = if block.target_names.is_empty() {
        (0..width).map(|index| format!("p{index}")).collect()
    } else {
        block.target_names.clone()
    };
    let source_plan = plan
        .node_plans
        .get(&edge.source.node_id)
        .expect("edge source has a node plan");
    let handle = HandleRef {
        handle: deterministic_oof_handle(plan, edge, ctx, scope)?,
        kind: HandleKind::Prediction,
        owner_controller: source_plan.controller_id.clone(),
    };
    Ok(Some(CollectedPredictionInput {
        handle,
        spec: PredictionInputSpec {
            producer_node: edge.source.node_id.clone(),
            source_port: edge.source.port_name.clone(),
            target_port: edge.target.port_name.clone(),
            partition: block.partition.clone(),
            prediction_level: PredictionLevel::Sample,
            fold_id: None,
            fold_ids: Vec::new(),
            unit_ids: block
                .sample_ids
                .iter()
                .cloned()
                .map(PredictionUnitId::Sample)
                .collect(),
            sample_ids: block.sample_ids.clone(),
            values: block.values.clone(),
            prediction_width: width,
            target_names,
        },
    }))
}

struct CollectedPredictionInput {
    handle: HandleRef,
    spec: PredictionInputSpec,
}

fn collect_oof_prediction_input(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
) -> Result<CollectedPredictionInput> {
    if scope.phase == Phase::Refit {
        if let Some(contract) = replay_prediction_cache_contract_for_edge(resources, edge) {
            if contract.requirement.prediction_level != PredictionLevel::Sample {
                let source_plan = plan
                    .node_plans
                    .get(&edge.source.node_id)
                    .expect("edge source has a node plan");
                let handle = materialize_oof_prediction_handle(
                    plan,
                    edge,
                    ctx,
                    scope,
                    resources,
                    &source_plan.controller_id,
                )?;
                return Ok(CollectedPredictionInput {
                    handle,
                    spec: prediction_input_spec_from_requirement(&contract.requirement, scope)?,
                });
            }
        }
    }
    let source_plan = plan
        .node_plans
        .get(&edge.source.node_id)
        .expect("edge source has a node plan");
    let prediction_level = oof_prediction_level_for_source(source_plan);
    if prediction_level != PredictionLevel::Sample {
        let blocks = match scope.phase {
            Phase::FitCv => validate_fit_cv_aggregated_oof_edge(
                plan,
                edge,
                ctx,
                scope,
                resources,
                prediction_level,
            )?,
            Phase::Refit => {
                validate_refit_aggregated_oof_edge(plan, edge, ctx, resources, prediction_level)?
            }
            _ => Vec::new(),
        };
        let handle = materialize_oof_prediction_handle(
            plan,
            edge,
            ctx,
            scope,
            resources,
            &source_plan.controller_id,
        )?;
        return Ok(CollectedPredictionInput {
            handle,
            spec: aggregated_prediction_input_spec(edge, scope, prediction_level, &blocks)?,
        });
    }
    let blocks = match scope.phase {
        Phase::FitCv => validate_fit_cv_oof_edge(plan, edge, ctx, scope)?,
        Phase::Refit => validate_refit_oof_edge(plan, edge, ctx)?,
        _ => Vec::new(),
    };
    let handle = materialize_oof_prediction_handle(
        plan,
        edge,
        ctx,
        scope,
        resources,
        &source_plan.controller_id,
    )?;
    Ok(CollectedPredictionInput {
        handle,
        spec: prediction_input_spec(edge, scope, &blocks)?,
    })
}

fn oof_prediction_level_for_source(source_plan: &NodePlan) -> PredictionLevel {
    source_plan
        .shape_plan
        .as_ref()
        .map(|shape_plan| shape_plan.aggregation_policy.aggregation_level)
        .unwrap_or(PredictionLevel::Sample)
}

fn replay_prediction_cache_contract_for_edge<'a>(
    resources: &'a PhaseScopeResources<'_>,
    edge: &EdgeSpec,
) -> Option<&'a ReplayPredictionCacheContract> {
    let contracts = resources.prediction_cache_contracts?;
    let key = bundle_prediction_requirement_key(
        &edge.source.node_id,
        &edge.source.port_name,
        &edge.target.node_id,
        &edge.target.port_name,
    );
    contracts.get(&key)
}

fn materialize_oof_prediction_handle(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
    producer_controller_id: &ControllerId,
) -> Result<HandleRef> {
    if scope.phase == Phase::Refit {
        if let (Some(store), Some(bundle_id), Some(contracts)) = (
            resources.prediction_cache_store,
            resources.replay_bundle_id,
            resources.prediction_cache_contracts,
        ) {
            let key = bundle_prediction_requirement_key(
                &edge.source.node_id,
                &edge.source.port_name,
                &edge.target.node_id,
                &edge.target.port_name,
            );
            let contract = contracts.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "replay prediction cache store cannot materialize missing requirement `{key}`"
                ))
            })?;
            let handle = store.materialize(&PredictionCacheMaterializationRequest {
                run_id: ctx.run_id.clone(),
                bundle_id: bundle_id.clone(),
                phase: scope.phase,
                variant_id: scope.variant_id.clone(),
                requirement: contract.requirement.clone(),
                cache: contract.cache.clone(),
                producer_controller_id: producer_controller_id.clone(),
            })?;
            if handle.kind != HandleKind::Prediction {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store materialized requirement `{key}` as {:?}",
                    handle.kind
                )));
            }
            if &handle.owner_controller != producer_controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store materialized requirement `{key}` for controller `{}`, expected `{}`",
                    handle.owner_controller, producer_controller_id
                )));
            }
            return Ok(handle);
        }
    }
    Ok(HandleRef {
        handle: deterministic_oof_handle(plan, edge, ctx, scope)?,
        kind: HandleKind::Prediction,
        owner_controller: producer_controller_id.clone(),
    })
}

fn validate_fit_cv_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
    scope: &PhaseScope,
) -> Result<Vec<&'a PredictionBlock>> {
    let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` requires OOF predictions but FIT_CV has no fold scope",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        ))
    })?;
    let blocks = ctx.prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        Some(fold_id),
    );
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, Some(fold_id)));
    }
    // MANDATORY exact OOF coverage (spec rule 3 + audit R-P0-2): a `requires_oof` stacking edge that
    // reaches here must have exactly one validation prediction per fold-validation sample, exact and
    // unique. This was previously gated by `requires_fold_alignment` — making completeness conditional,
    // so an edge that left the flag unset (a future builder or adversarial JSON) admitted blocks that
    // merely *exist*. The branch-merge concat partition exception ("unless an explicit aggregation
    // policy says otherwise"), where a branch legitimately covers only its partition, is intercepted
    // before this code path (the separation-merge handler) and so is never over-rejected here.
    let fold_set = required_fold_set_for_oof(plan, edge)?;
    validate_oof_blocks_match_fold(edge, fold_set, fold_id, &blocks)?;
    Ok(blocks)
}

fn validate_refit_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
) -> Result<Vec<&'a PredictionBlock>> {
    let blocks = ctx.prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        None,
    );
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, None));
    }
    // MANDATORY exact OOF coverage — see `validate_fit_cv_oof_edge`. The branch-merge concat partition
    // exception is handled by the separation-merge handler, which never reaches this stacking path.
    let fold_set = required_fold_set_for_oof(plan, edge)?;
    validate_oof_blocks_cover_fold_set(edge, fold_set, &blocks)?;
    Ok(blocks)
}

fn validate_fit_cv_aggregated_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
    prediction_level: PredictionLevel,
) -> Result<Vec<&'a AggregatedPredictionBlock>> {
    let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` requires aggregated OOF predictions but FIT_CV has no fold scope",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        ))
    })?;
    let blocks = ctx.aggregated_prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        Some(fold_id),
        Some(prediction_level),
    );
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, Some(fold_id)));
    }
    validate_aggregated_blocks_basic(edge, prediction_level, &blocks)?;
    // MANDATORY exact aggregated-OOF coverage — see `validate_fit_cv_oof_edge` (audit R-P0-2). The
    // concat-merge partition exception is intercepted by the separation-merge handler upstream.
    let fold_set = required_fold_set_for_oof(plan, edge)?;
    let relations = coordinator_relations_for_edge(plan, edge, resources)?;
    validate_aggregated_oof_blocks_match_fold(
        edge,
        fold_set,
        &relations,
        prediction_level,
        fold_id,
        &blocks,
    )?;
    Ok(blocks)
}

fn validate_refit_aggregated_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
    resources: &PhaseScopeResources<'_>,
    prediction_level: PredictionLevel,
) -> Result<Vec<&'a AggregatedPredictionBlock>> {
    let blocks = ctx.aggregated_prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        None,
        Some(prediction_level),
    );
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, None));
    }
    validate_aggregated_blocks_basic(edge, prediction_level, &blocks)?;
    // MANDATORY exact aggregated-OOF coverage — see `validate_fit_cv_oof_edge` (audit R-P0-2). The
    // concat-merge partition exception is intercepted by the separation-merge handler upstream.
    let fold_set = required_fold_set_for_oof(plan, edge)?;
    let relations = coordinator_relations_for_edge(plan, edge, resources)?;
    validate_aggregated_oof_blocks_cover_fold_set(
        edge,
        fold_set,
        &relations,
        prediction_level,
        &blocks,
    )?;
    Ok(blocks)
}

fn validate_aggregated_blocks_basic(
    edge: &EdgeSpec,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<()> {
    for block in blocks {
        block.validate_shape()?;
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected non-validation aggregated predictions",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if block.level != prediction_level {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected {:?} aggregated predictions, expected {:?}",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                block.level,
                prediction_level
            )));
        }
    }
    Ok(())
}

fn prediction_input_spec(
    edge: &EdgeSpec,
    scope: &PhaseScope,
    blocks: &[&PredictionBlock],
) -> Result<PredictionInputSpec> {
    let sample_ids = collect_unique_oof_samples(edge, blocks)?
        .into_iter()
        .collect::<Vec<_>>();
    let fold_ids = blocks
        .iter()
        .filter_map(|block| block.fold_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    // Validation OOF rows keyed by sample, so the meta-node host can build a
    // stacking feature matrix in FIT_CV/REFIT. Blocks are Validation-only (the
    // leakage guards in `validate_fit_cv_oof_edge` / `validate_refit_oof_edge`
    // and `collect_unique_oof_samples` already refused any Train partition).
    let mut rows_by_sample: BTreeMap<&SampleId, &[f64]> = BTreeMap::new();
    let mut prediction_width = None;
    let mut target_names = None;
    for block in blocks {
        let width = block.validate_shape()?;
        for (sample_id, row) in block.sample_ids.iter().zip(block.values.iter()) {
            rows_by_sample.insert(sample_id, row.as_slice());
        }
        let block_target_names = if block.target_names.is_empty() {
            (0..width)
                .map(|index| format!("p{index}"))
                .collect::<Vec<_>>()
        } else {
            block.target_names.clone()
        };
        if prediction_width.is_some_and(|expected| expected != width) {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` OOF prediction width is not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if target_names
            .as_ref()
            .is_some_and(|expected| expected != &block_target_names)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` OOF target names are not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        prediction_width = Some(width);
        target_names = Some(block_target_names);
    }
    let values = sample_ids
        .iter()
        .map(|sample_id| {
            rows_by_sample
                .get(sample_id)
                .map(|row| row.to_vec())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "edge `{}.{}` -> `{}.{}` has no OOF prediction row for sample `{sample_id}`",
                        edge.source.node_id,
                        edge.source.port_name,
                        edge.target.node_id,
                        edge.target.port_name
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PredictionInputSpec {
        producer_node: edge.source.node_id.clone(),
        source_port: edge.source.port_name.clone(),
        target_port: edge.target.port_name.clone(),
        partition: PredictionPartition::Validation,
        prediction_level: PredictionLevel::Sample,
        fold_id: scope.fold_id.clone(),
        fold_ids,
        unit_ids: sample_ids
            .iter()
            .cloned()
            .map(PredictionUnitId::Sample)
            .collect(),
        sample_ids,
        values,
        prediction_width: prediction_width.unwrap_or_default(),
        target_names: target_names.unwrap_or_default(),
    })
}

fn aggregated_prediction_input_spec(
    edge: &EdgeSpec,
    scope: &PhaseScope,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<PredictionInputSpec> {
    let unit_ids = collect_unique_aggregated_oof_units(edge, prediction_level, blocks)?
        .into_iter()
        .collect::<Vec<_>>();
    let fold_ids = blocks
        .iter()
        .filter_map(|block| block.fold_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut prediction_width = None;
    let mut target_names = None;
    for block in blocks {
        let width = block.validate_shape()?;
        let block_target_names = if block.target_names.is_empty() {
            (0..width)
                .map(|index| format!("p{index}"))
                .collect::<Vec<_>>()
        } else {
            block.target_names.clone()
        };
        if prediction_width.is_some_and(|expected| expected != width) {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` aggregated OOF prediction width is not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if target_names
            .as_ref()
            .is_some_and(|expected| expected != &block_target_names)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` aggregated OOF target names are not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        prediction_width = Some(width);
        target_names = Some(block_target_names);
    }
    Ok(PredictionInputSpec {
        producer_node: edge.source.node_id.clone(),
        source_port: edge.source.port_name.clone(),
        target_port: edge.target.port_name.clone(),
        partition: PredictionPartition::Validation,
        prediction_level,
        fold_id: scope.fold_id.clone(),
        fold_ids,
        unit_ids,
        sample_ids: Vec::new(),
        // Aggregated (unit-level) OOF crosses as opaque handle, not per-sample rows.
        values: Vec::new(),
        prediction_width: prediction_width.unwrap_or_default(),
        target_names: target_names.unwrap_or_default(),
    })
}

fn prediction_input_spec_from_requirement(
    requirement: &BundlePredictionRequirement,
    scope: &PhaseScope,
) -> Result<PredictionInputSpec> {
    requirement.validate()?;
    Ok(PredictionInputSpec {
        producer_node: requirement.producer_node.clone(),
        source_port: requirement.source_port.clone(),
        target_port: requirement.target_port.clone(),
        partition: requirement.partition.clone(),
        prediction_level: requirement.prediction_level,
        fold_id: scope.fold_id.clone(),
        fold_ids: requirement.fold_ids.clone(),
        unit_ids: requirement.unit_ids.clone(),
        sample_ids: requirement.sample_ids.clone(),
        // Replay-cache requirement: OOF rows are materialized by the host via the
        // prediction-cache handle, not carried inline in the spec.
        values: Vec::new(),
        prediction_width: requirement.prediction_width,
        target_names: requirement.target_names.clone(),
    })
}

fn missing_oof_edge_error(edge: &EdgeSpec, fold_id: Option<&FoldId>) -> DagMlError {
    DagMlError::RuntimeValidation(format!(
        "edge `{}.{}` -> `{}.{}` requires OOF validation predictions from `{}`{}",
        edge.source.node_id,
        edge.source.port_name,
        edge.target.node_id,
        edge.target.port_name,
        edge.source.node_id,
        fold_id
            .map(|fold_id| format!(" for fold `{fold_id}`"))
            .unwrap_or_default()
    ))
}

/// The OOF [`FoldPartitionMode`] for a plan: its fold set's mode, or `Partition` (the clean-OOF
/// default) when the plan carries no fold set. Used to make the cross-fold scoring gate mode-aware so
/// `Resampled` (ShuffleSplit / repeated CV) campaigns, where a sample is validated in several folds,
/// are not rejected by the `Partition` exactly-once uniqueness rule.
pub fn plan_oof_partition_mode(plan: &ExecutionPlan) -> FoldPartitionMode {
    plan.fold_set
        .as_ref()
        .map(|fold_set| fold_set.partition_mode)
        .unwrap_or_default()
}

fn required_fold_set_for_oof<'a>(plan: &'a ExecutionPlan, edge: &EdgeSpec) -> Result<&'a FoldSet> {
    plan.fold_set.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` requires fold-aligned OOF predictions but the plan has no fold set",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name
        ))
    })
}

fn validate_oof_blocks_match_fold(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    fold_id: &FoldId,
    blocks: &[&PredictionBlock],
) -> Result<()> {
    let fold = fold_set
        .folds
        .iter()
        .find(|fold| &fold.fold_id == fold_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
    let actual = collect_unique_oof_samples(edge, blocks)?;
    let expected = fold
        .validation_sample_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` OOF predictions do not match validation samples for fold `{fold_id}`",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name
        )));
    }
    Ok(())
}

fn validate_oof_blocks_cover_fold_set(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    blocks: &[&PredictionBlock],
) -> Result<()> {
    let folds = fold_set
        .folds
        .iter()
        .map(|fold| (&fold.fold_id, fold))
        .collect::<BTreeMap<_, _>>();
    let mut all_samples = BTreeSet::new();
    for block in blocks {
        let fold_id = block.fold_id.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` has OOF predictions without a fold id",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
        let fold = folds.get(fold_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
        let block_samples = collect_unique_oof_samples(edge, &[*block])?;
        let expected = fold
            .validation_sample_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if block_samples != expected {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` OOF predictions do not match validation samples for fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        for sample_id in block_samples {
            // Partition is a clean OOF set: a sample covered by two folds is a duplicated fold or a
            // mixed-variant context. Resampled (ShuffleSplit / repeated CV) legitimately validates a
            // sample in several folds and averages its predictions, so the across-fold duplicate is
            // expected; the per-fold match above + per-block uniqueness (`collect_unique_oof_samples`)
            // still hold, and the universe-coverage check below still requires every sample at least
            // once.
            if !all_samples.insert(sample_id.clone())
                && fold_set.partition_mode == FoldPartitionMode::Partition
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate OOF prediction for sample `{sample_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }
    let expected_all = fold_set.sample_ids.iter().cloned().collect::<BTreeSet<_>>();
    if all_samples != expected_all {
        return Err(DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` OOF predictions do not cover the refit sample universe",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        )));
    }
    Ok(())
}

fn validate_aggregated_oof_blocks_match_fold(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    fold_id: &FoldId,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<()> {
    let fold = fold_set
        .folds
        .iter()
        .find(|fold| &fold.fold_id == fold_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
    validate_aggregated_fold_unit_safety(edge, relations, prediction_level, fold)?;
    for block in blocks {
        if block.fold_id.as_ref() != Some(fold_id) {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected aggregated OOF predictions outside fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
    }
    let actual = collect_unique_aggregated_oof_units(edge, prediction_level, blocks)?;
    let expected = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold.validation_sample_ids,
    )?;
    if actual != expected {
        return Err(DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` aggregated OOF predictions do not match {:?} validation units for fold `{fold_id}`",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name,
            prediction_level
        )));
    }
    Ok(())
}

fn validate_aggregated_oof_blocks_cover_fold_set(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<()> {
    let folds = fold_set
        .folds
        .iter()
        .map(|fold| (fold.fold_id.clone(), fold))
        .collect::<BTreeMap<_, _>>();
    let mut blocks_by_fold = BTreeMap::<FoldId, Vec<&AggregatedPredictionBlock>>::new();
    for block in blocks {
        let fold_id = block.fold_id.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` has aggregated OOF predictions without a fold id",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
        if !folds.contains_key(fold_id) {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        blocks_by_fold
            .entry(fold_id.clone())
            .or_default()
            .push(*block);
    }
    for fold_id in folds.keys() {
        if !blocks_by_fold.contains_key(fold_id) {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` is missing aggregated OOF predictions for fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
    }

    let mut all_units = BTreeSet::new();
    for (fold_id, fold_blocks) in blocks_by_fold {
        let fold = folds.get(&fold_id).expect("fold id was validated above");
        validate_aggregated_fold_unit_safety(edge, relations, prediction_level, fold)?;
        let fold_units = collect_unique_aggregated_oof_units(edge, prediction_level, &fold_blocks)?;
        let expected = expected_prediction_units_for_samples(
            edge,
            relations,
            prediction_level,
            &fold.validation_sample_ids,
        )?;
        if fold_units != expected {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` aggregated OOF predictions do not match {:?} validation units for fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                prediction_level
            )));
        }
        for unit_id in fold_units {
            // See `validate_oof_blocks_cover_fold_set`: Partition forbids a unit covered by two folds;
            // Resampled (ShuffleSplit / repeated CV) validates a unit in several folds and averages it,
            // so the across-fold duplicate is allowed while the universe-coverage check below still
            // requires every unit at least once.
            if !all_units.insert(unit_id.clone())
                && fold_set.partition_mode == FoldPartitionMode::Partition
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate aggregated OOF prediction for unit `{unit_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }

    let expected_all = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold_set.sample_ids,
    )?;
    if all_units != expected_all {
        return Err(DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` aggregated OOF predictions do not cover the refit {:?} unit universe",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name,
            prediction_level
        )));
    }
    Ok(())
}

fn validate_aggregated_fold_unit_safety(
    edge: &EdgeSpec,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    fold: &FoldAssignment,
) -> Result<()> {
    let train_units = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold.train_sample_ids,
    )?;
    let validation_units = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold.validation_sample_ids,
    )?;
    if let Some(unit_id) = train_units.intersection(&validation_units).next() {
        return Err(DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` fold `{}` has {:?} unit `{unit_id}` in both train and validation partitions",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name,
            fold.fold_id,
            prediction_level
        )));
    }
    Ok(())
}

fn collect_unique_oof_samples(
    edge: &EdgeSpec,
    blocks: &[&PredictionBlock],
) -> Result<BTreeSet<SampleId>> {
    let mut samples = BTreeSet::new();
    for block in blocks {
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected non-validation predictions",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        for sample_id in &block.sample_ids {
            if !samples.insert(sample_id.clone()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate OOF prediction for sample `{sample_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }
    Ok(samples)
}

fn collect_unique_aggregated_oof_units(
    edge: &EdgeSpec,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<BTreeSet<PredictionUnitId>> {
    let mut unit_ids = BTreeSet::new();
    for block in blocks {
        block.validate_shape()?;
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected non-validation aggregated predictions",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if block.level != prediction_level {
            return Err(DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected {:?} aggregated predictions, expected {:?}",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                block.level,
                prediction_level
            )));
        }
        for unit_id in &block.unit_ids {
            if !unit_ids.insert(unit_id.clone()) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate aggregated OOF prediction for unit `{unit_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }
    Ok(unit_ids)
}

fn expected_prediction_units_for_samples(
    edge: &EdgeSpec,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    sample_ids: &[SampleId],
) -> Result<BTreeSet<PredictionUnitId>> {
    sample_ids
        .iter()
        .map(|sample_id| prediction_unit_for_sample(edge, relations, prediction_level, sample_id))
        .collect()
}

fn prediction_unit_for_sample(
    edge: &EdgeSpec,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    sample_id: &SampleId,
) -> Result<PredictionUnitId> {
    match prediction_level {
        PredictionLevel::Sample => Ok(PredictionUnitId::Sample(sample_id.clone())),
        PredictionLevel::Target => relations
            .target_for_sample(sample_id)
            .cloned()
            .map(PredictionUnitId::Target)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` needs target-level OOF predictions but sample `{sample_id}` has no target relation",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                ))
            }),
        PredictionLevel::Group => relations
            .group_for_sample(sample_id)
            .cloned()
            .map(PredictionUnitId::Group)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` needs group-level OOF predictions but sample `{sample_id}` has no group relation",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                ))
            }),
        PredictionLevel::Observation => Err(DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` cannot consume observation-level OOF predictions from sample folds",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        ))),
    }
}

fn deterministic_oof_handle(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<u64> {
    let fingerprint = stable_json_fingerprint(&(
        &plan.id,
        &ctx.run_id,
        &edge.source.node_id,
        &edge.source.port_name,
        &edge.target.node_id,
        &edge.target.port_name,
        scope.phase,
        &scope.variant_id,
        &scope.fold_id,
    ))?;
    Ok(u64::from_str_radix(&fingerprint[..16], 16).expect("sha256 hex prefix should fit into u64"))
}

struct CollectedInputs {
    handles: BTreeMap<String, HandleRef>,
    data_views: BTreeMap<String, DataProviderViewSpec>,
    prediction_inputs: BTreeMap<String, PredictionInputSpec>,
}

fn data_view_key(input_name: &str) -> String {
    format!("data:{input_name}")
}

fn validation_data_view_key(input_name: &str) -> String {
    format!("{input_name}:validation")
}

fn derive_output_data_views(
    plan: &ExecutionPlan,
    task: &NodeTask,
    result: &NodeResult,
) -> Result<BTreeMap<String, DataProviderViewSpec>> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == task.node_plan.node_id)
        .expect("execution plan was validated");
    let mut views = BTreeMap::new();
    for port in node
        .ports
        .outputs
        .iter()
        .filter(|port| port.kind == PortKind::Data)
    {
        let Some(handle) = result.outputs.get(&port.name) else {
            continue;
        };
        if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` emitted data output `{}` with non-data/data-view handle kind {:?}",
                task.node_plan.node_id, port.name, handle.kind
            )));
        }
        if let Some(view) = primary_output_data_view(task) {
            views.insert(
                port.name.clone(),
                output_data_view_for_port(task, result, &port.name, view)?,
            );
        }
        if let Some(validation_view) = validation_output_data_view(task) {
            views.insert(
                validation_data_view_key(&port.name),
                output_data_view_for_port(task, result, &port.name, validation_view)?,
            );
        }
    }
    Ok(views)
}

fn output_data_view_for_port(
    task: &NodeTask,
    result: &NodeResult,
    port_name: &str,
    base_view: &DataProviderViewSpec,
) -> Result<DataProviderViewSpec> {
    let mut view = base_view.clone();
    if let Some(upstream_provenance) = view.extra.remove(DATA_OUTPUT_PROVENANCE_KEY) {
        let provenance: DataOutputProvenance =
            serde_json::from_value(upstream_provenance).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` cannot propagate data output `{port_name}` because upstream data output provenance is invalid JSON: {error}",
                    task.node_plan.node_id
                ))
            })?;
        provenance.validate().map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "node `{}` cannot propagate data output `{port_name}` because upstream data output provenance is invalid: {error}",
                task.node_plan.node_id
            ))
        })?;
    }
    let shape_deltas = result
        .shape_deltas
        .iter()
        .filter(|delta| delta.node_id == task.node_plan.node_id)
        .cloned()
        .collect::<Vec<_>>();
    let mut provenance = DataOutputProvenance {
        schema_version: DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION,
        producer_node: task.node_plan.node_id.clone(),
        producer_port: port_name.to_string(),
        producer_phase: task.phase,
        variant_id: task.variant_id.clone(),
        fold_id: task.fold_id.clone(),
        shape_plan_fingerprint: None,
        aggregation_policy_fingerprint: None,
        feature_namespace: None,
        feature_schema_fingerprint: None,
        representation_plan: None,
        representation_replay_manifest: None,
        representation_compatibility: None,
        relation_delta_fingerprint: None,
        shape_deltas,
    };
    if let Some(shape_plan) = &task.node_plan.shape_plan {
        provenance.shape_plan_fingerprint = Some(stable_json_fingerprint(shape_plan)?);
        provenance.aggregation_policy_fingerprint =
            Some(stable_json_fingerprint(&shape_plan.aggregation_policy)?);
        provenance.feature_namespace = shape_plan.feature_namespace.clone();
        provenance.feature_schema_fingerprint =
            output_feature_schema_fingerprint(shape_plan, result);
    }
    provenance.validate()?;

    view.extra.insert(
        DATA_OUTPUT_PROVENANCE_KEY.to_string(),
        serde_json::to_value(provenance)?,
    );
    view.validate()?;
    Ok(view)
}

fn output_feature_schema_fingerprint(
    shape_plan: &crate::policy::DataModelShapePlan,
    result: &NodeResult,
) -> Option<String> {
    result
        .shape_deltas
        .iter()
        .rev()
        .find(|delta| delta.kind == ShapeDeltaKind::Feature)
        .map(|delta| delta.after_fingerprint.clone())
        .or_else(|| shape_plan.feature_schema_fingerprint.clone())
}

fn primary_output_data_view(task: &NodeTask) -> Option<&DataProviderViewSpec> {
    task.data_views
        .values()
        .find(|view| view.partition != DataRequestPartition::FoldValidation)
        .or_else(|| task.data_views.values().next())
}

fn validation_output_data_view(task: &NodeTask) -> Option<&DataProviderViewSpec> {
    task.data_views
        .values()
        .find(|view| view.partition == DataRequestPartition::FoldValidation)
}

fn make_data_view_handle(
    data_provider: &dyn RuntimeDataProvider,
    ctx: &RunContext,
    node_plan: &NodePlan,
    scope: &PhaseScope,
    binding: &DataBinding,
    data_handle: &HandleRef,
    view: &DataProviderViewSpec,
) -> Result<HandleRef> {
    view.validate()?;
    let view_handle = data_provider.make_view(&DataViewRequest {
        run_id: ctx.run_id.clone(),
        node_id: node_plan.node_id.clone(),
        input_name: binding.input_name.clone(),
        phase: scope.phase,
        variant_id: scope.variant_id.clone(),
        fold_id: scope.fold_id.clone(),
        binding: binding.clone(),
        data_handle: data_handle.clone(),
        view: view.clone(),
    })?;
    // A data view is delivered to the controller as a data input, so the
    // provider must return a data-bearing handle. Refuse a model / artifact /
    // prediction / relation handle masquerading as a view across the ABI.
    if !matches!(view_handle.kind, HandleKind::Data | HandleKind::DataView) {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` data view `{}` resolved to a non-data/data-view handle kind {:?}",
            node_plan.node_id, binding.input_name, view_handle.kind
        )));
    }
    Ok(view_handle)
}

fn data_view_for_scope(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
    branch_view: Option<&crate::data::BranchViewPlan>,
    excluded_samples: &BTreeSet<SampleId>,
) -> Result<DataProviderViewSpec> {
    let partition = data_partition_for_scope(binding, scope);
    // During FIT_CV and REFIT this primary view IS the training input; during
    // PREDICT/EXPLAIN (and the planning phases) it is a non-fit read.
    let role = match scope.phase {
        Phase::FitCv | Phase::Refit => DataViewRole::Fit,
        _ => DataViewRole::NonFit,
    };
    data_view_for_partition(
        binding,
        fold_set,
        scope,
        partition,
        branch_view,
        role,
        excluded_samples,
    )
}

fn validation_data_view_for_scope(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
    branch_view: Option<&crate::data::BranchViewPlan>,
    excluded_samples: &BTreeSet<SampleId>,
) -> Result<Option<DataProviderViewSpec>> {
    if scope.phase != Phase::FitCv || scope.fold_id.is_none() {
        return Ok(None);
    }
    let partition = binding.view_policy.predict_partition;
    if partition == data_partition_for_scope(binding, scope) {
        return Ok(None);
    }
    // This is the validation companion read, never the training input.
    data_view_for_partition(
        binding,
        fold_set,
        scope,
        partition,
        branch_view,
        DataViewRole::NonFit,
        excluded_samples,
    )
    .map(Some)
}

/// The native cross-branch reduction a `PredictionJoin` merge node performs in
/// the scheduler, decoded from its DSL `merge_mode` metadata. These are the
/// merge kinds the scheduler reassembles itself (no controller call); any other
/// `merge_mode` (e.g. the default stacking semantics) is NOT a native reduction:
/// it stays an ordinary controller node joined through the `requires_oof` edge
/// path. So stacking (predictions-as-meta-features, a meta-model node) is out of
/// scope here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MergeReduction {
    /// Separation-branch *concat* reassembly: N branches each cover a DISJOINT
    /// partition of the fold validation set; the merge concatenates them by
    /// `sample_id` into one full-fold OOF block (overlap is an error). DSL
    /// `merge_mode == "concat"`.
    Concat,
    /// Duplication-branch *late-fusion* averaging (regression / value mean): N
    /// models on the FULL data; the merge averages each sample's branch
    /// predictions over the branches that covered it (asymmetric-coverage safe).
    /// DSL `merge_mode == "fusion"`.
    Fusion,
    /// Duplication-branch *probability-mean* fusion (classification): like
    /// [`MergeReduction::Fusion`] but each branch row is a per-class probability
    /// vector, averaged and renormalized to a valid distribution. DSL
    /// `merge_mode == "fusion_proba_mean"`.
    FusionProbaMean,
}

/// Decode the native cross-branch reduction `node_plan` performs, if any. A node
/// is a native reduction merge iff it is a `PredictionJoin` whose graph
/// `merge_mode` metadata names one of the reductions above; otherwise `None`
/// (the node takes the ordinary controller / `requires_oof` path).
fn merge_reduction_mode(plan: &ExecutionPlan, node_plan: &NodePlan) -> Option<MergeReduction> {
    if node_plan.kind != crate::graph::NodeKind::PredictionJoin {
        return None;
    }
    match plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == node_plan.node_id)
        .and_then(|node| node.metadata.get("merge_mode"))
        .and_then(serde_json::Value::as_str)
    {
        Some("concat") => Some(MergeReduction::Concat),
        Some("fusion") => Some(MergeReduction::Fusion),
        Some("fusion_proba_mean") => Some(MergeReduction::FusionProbaMean),
        _ => None,
    }
}

/// Reassemble the native cross-branch reduction `node_plan` performs (concat or
/// late-fusion averaging), dispatching on the decoded [`MergeReduction`]. Both
/// schedulers call this for any node `merge_reduction_mode` matched, so the
/// scheduler dispatch stays a single branch.
fn reassemble_branch_merge(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    ctx: &RunContext,
    scope: &PhaseScope,
    reduction: MergeReduction,
) -> Result<Option<NodeResult>> {
    // FIT_CV is the OOF (Validation, per-fold) reassembly the merge handlers were
    // built for. REFIT and PREDICT have no fold scope: the base branches predicted
    // the held-out test set (REFIT) / new data (PREDICT) into the prediction store
    // under a non-Validation partition (`Test` / `Final`) with `fold_id == None`,
    // and `reassemble_branch_merge_off_fold` reassembles THOSE into one scored
    // test/predict block under the merge producer. The Validation OOF path is left
    // untouched, so the FIT_CV meta-features stay Validation-only (the leakage
    // invariant): the test/predict reassembly only ever reads non-fold blocks.
    if scope.phase != Phase::FitCv {
        return reassemble_branch_merge_off_fold(node_plan, ctx, scope, reduction);
    }
    match reduction {
        MergeReduction::Concat => reassemble_separation_merge(plan, node_plan, ctx, scope),
        MergeReduction::Fusion | MergeReduction::FusionProbaMean => {
            reassemble_fusion_merge(plan, node_plan, ctx, scope, reduction)
        }
    }
}

/// The off-fold prediction partition a given non-FIT_CV phase consumes / produces:
/// REFIT predicts the held-out test set (`Test`); PREDICT new data (`Final`). Any
/// other phase that reaches the off-fold path defaults to `Test`. This pins the
/// off-fold reads (merge reassembly + stacking meta-feature delivery) to exactly
/// one partition so a stale block from a prior phase in the same `RunContext` is
/// never consumed. FIT_CV never uses this (it is Validation/per-fold).
fn expected_off_fold_partition(phase: Phase) -> PredictionPartition {
    match phase {
        Phase::Predict => PredictionPartition::Final,
        _ => PredictionPartition::Test,
    }
}

/// The non-FIT_CV (REFIT / PREDICT) analogue of [`reassemble_branch_merge`]:
/// reassemble the base branches' off-fold (test / predict) predictions into one
/// scored block under the merge producer, so concat/fusion merges produce a
/// `best_rmse` and can predict — not just FIT_CV Validation OOF.
///
/// In REFIT each base branch predicts the held-out TEST set; in PREDICT, new
/// data. Those base blocks are stored with `fold_id == None` and a non-Validation
/// partition (`Test` for REFIT, `Final` for PREDICT). This handler reads exactly
/// those `fold_id == None`, non-`Validation` branch blocks (scoped to the active
/// variant), reassembles them — concat keeps disjoint partitions (overlap is an
/// error), fusion averages each sample over the branches that covered it — and
/// emits a block carrying the SAME partition the branches used, with reassembled
/// `y_true` so `apply_result_scoring` scores it. The universe is the UNION of
/// branch coverage (there is no fold validation set to define it off-fold).
///
/// LEAKAGE INVARIANT: this path is a NO-OP in FIT_CV (the caller routes FIT_CV to
/// the Validation OOF handlers) and only ever reads `fold_id == None`,
/// non-`Validation` blocks. The FIT_CV Validation-OOF meta-features are never
/// touched, and a Validation block (whether OOF or accidentally off-fold) is
/// never reassembled here.
fn reassemble_branch_merge_off_fold(
    node_plan: &NodePlan,
    ctx: &RunContext,
    scope: &PhaseScope,
    reduction: MergeReduction,
) -> Result<Option<NodeResult>> {
    let variant_id = scope.variant_id.clone();
    // The phase pins EXACTLY which off-fold partition to consume: REFIT predicts
    // the held-out test set (`Test`), PREDICT new data (`Final`). Filtering by the
    // phase-expected partition (not just "non-Validation") keeps a stale `Final`
    // (from a prior REFIT/PREDICT in the same context) or any `Train` block out of
    // a REFIT merge, and lets a PREDICT-after-REFIT in one context pick `Final`
    // cleanly without tripping the multi-block "mixes variants" guard.
    let expected_partition = expected_off_fold_partition(scope.phase);

    // Gather each branch's off-fold (test / predict) block: `fold_id == None`,
    // partition == the phase-expected partition, scoped to the active variant. A
    // branch may emit none (a modelless / sparse branch); coverage is the union of
    // what is present.
    let mut branch_blocks: Vec<PredictionBlock> = Vec::new();
    let mut partition: Option<PredictionPartition> = None;
    let mut by_sample_target: BTreeMap<SampleId, Vec<f64>> = BTreeMap::new();
    let mut target_block_names: Option<Vec<String>> = None;

    for branch_id in &node_plan.input_nodes {
        let blocks: Vec<&PredictionBlock> = ctx
            .prediction_store
            .find(Some(branch_id), Some(&expected_partition), None)
            .into_iter()
            .filter(|block| block.fold_id.is_none())
            .collect();
        if blocks.is_empty() {
            continue;
        }
        if blocks.len() > 1 {
            return Err(DagMlError::OofValidation(format!(
                "merge node `{}` found {} off-fold ({expected_partition:?}) blocks for branch `{branch_id}`: the run context mixes several variants — reassemble each variant in its own context (native SELECT does this)",
                node_plan.node_id,
                blocks.len(),
            )));
        }
        let block = blocks[0];
        block.validate_shape()?;
        match &partition {
            None => partition = Some(block.partition.clone()),
            Some(existing) if existing != &block.partition => {
                return Err(DagMlError::OofValidation(format!(
                    "merge node `{}` received mismatched off-fold partitions ({existing:?} vs {:?}) from branch `{branch_id}`",
                    node_plan.node_id, block.partition
                )));
            }
            _ => {}
        }
        branch_blocks.push(block.clone());

        // Reassemble this branch's off-fold y_true (same phase-expected partition /
        // variant, no fold). The branches predict the SAME samples, so a per-sample
        // insert is correct (concat partitions are disjoint; fusion targets are
        // identical).
        for record in &ctx.regression_target_records {
            if &record.producer_node != branch_id
                || record.fold_id.is_some()
                || record.partition != expected_partition
                || record.variant_id != variant_id
            {
                continue;
            }
            if target_block_names.is_none() && !record.block.target_names.is_empty() {
                target_block_names = Some(record.block.target_names.clone());
            }
            for (unit_id, row) in record.block.unit_ids.iter().zip(&record.block.values) {
                let PredictionUnitId::Sample(sample_id) = unit_id else {
                    continue;
                };
                by_sample_target.insert(sample_id.clone(), row.clone());
            }
        }
    }

    // No branch produced an off-fold block: nothing to reassemble (a modelless
    // merge, or a phase where the branches do not predict).
    if branch_blocks.is_empty() {
        return Ok(None);
    }
    let partition = partition.expect("at least one branch block present");

    let reassembled = match reduction {
        MergeReduction::Concat => reassemble_off_fold_concat(&branch_blocks, &node_plan.node_id)?,
        MergeReduction::Fusion => {
            reduce_predictions_across_branches(&branch_blocks, None, &node_plan.node_id)?
        }
        MergeReduction::FusionProbaMean => {
            reduce_proba_mean_across_branches(&branch_blocks, &node_plan.node_id)?
        }
    };

    // Deterministic order: emit samples sorted by id (no fold order to follow
    // off-fold). Targets are emitted only when EVERY merged sample has a y_true
    // row, so `apply_result_scoring` pairs the block 1:1 with its targets.
    let mut sample_ids: Vec<SampleId> = reassembled.sample_ids.clone();
    sample_ids.sort();
    let by_sample: BTreeMap<&SampleId, &Vec<f64>> = reassembled
        .sample_ids
        .iter()
        .zip(&reassembled.values)
        .collect();
    let values: Vec<Vec<f64>> = sample_ids
        .iter()
        .map(|sample_id| by_sample[sample_id].clone())
        .collect();

    let regression_targets = reassemble_merge_targets(
        &node_plan.node_id,
        &sample_ids,
        &mut by_sample_target,
        target_block_names.unwrap_or_default(),
    )?
    .into_iter()
    .collect();

    // Lineage links every contributing branch (for this variant, off-fold).
    let branch_inputs: BTreeSet<&NodeId> = node_plan.input_nodes.iter().collect();
    let mut input_lineage: Vec<LineageId> = Vec::new();
    for record in ctx.lineage.records() {
        if branch_inputs.contains(&record.node_id)
            && record.phase == scope.phase
            && record.fold_id.is_none()
            && record.variant_id == variant_id
        {
            input_lineage.push(record.record_id.clone());
        }
    }

    let variant_suffix = variant_id
        .as_ref()
        .map(|variant| format!(":{variant}"))
        .unwrap_or_default();
    let phase_label = scope.phase.as_str();
    let merged = PredictionBlock {
        prediction_id: Some(format!(
            "merge:{}:{phase_label}{variant_suffix}",
            node_plan.node_id
        )),
        producer_node: node_plan.node_id.clone(),
        partition,
        fold_id: None,
        sample_ids,
        values,
        target_names: reassembled.target_names.clone(),
    };
    merged.validate_shape()?;

    let lineage = LineageRecord {
        record_id: LineageId::new(format!(
            "lineage:{}:{phase_label}{variant_suffix}",
            node_plan.node_id
        ))?,
        run_id: ctx.run_id.clone(),
        node_id: node_plan.node_id.clone(),
        phase: scope.phase,
        controller_id: node_plan.controller_id.clone(),
        controller_version: node_plan.controller_version.clone(),
        variant_id,
        fold_id: None,
        branch_path: Vec::new(),
        input_lineage,
        artifact_refs: Vec::new(),
        params_fingerprint: node_plan.params_fingerprint.clone(),
        data_model_shape_fingerprint: None,
        aggregation_policy_fingerprint: None,
        seed: None,
        unsafe_flags: BTreeSet::new(),
        metrics: BTreeMap::new(),
    };

    Ok(Some(NodeResult {
        node_id: node_plan.node_id.clone(),
        outputs: BTreeMap::new(),
        predictions: vec![merged],
        observation_predictions: Vec::new(),
        aggregated_predictions: Vec::new(),
        explanations: Vec::new(),
        shape_deltas: Vec::new(),
        artifacts: Vec::new(),
        artifact_handles: BTreeMap::new(),
        fit_influence_diagnostics: Vec::new(),
        regression_targets,
        lineage,
    }))
}

/// Concatenate disjoint off-fold (test / predict) branch blocks by `sample_id`
/// into one block under `merge_node`. The off-fold analogue of the concat half of
/// [`reassemble_separation_merge`]: branches cover DISJOINT partitions of the
/// universe (separation never shares a sample), so an overlapping sample is a hard
/// error. Width, target names and partition must agree across branches. Unlike the
/// FIT_CV concat there is no fold validation set to check completeness against —
/// the universe is simply the disjoint union of the branch coverage.
fn reassemble_off_fold_concat(
    branch_blocks: &[PredictionBlock],
    merge_node: &NodeId,
) -> Result<PredictionBlock> {
    let first = branch_blocks
        .first()
        .expect("at least one branch block present");
    let width = first.validate_shape()?;
    let target_names = if first.target_names.is_empty() {
        (0..width).map(|idx| format!("p{idx}")).collect::<Vec<_>>()
    } else {
        first.target_names.clone()
    };
    let mut by_sample: BTreeMap<SampleId, Vec<f64>> = BTreeMap::new();
    for block in branch_blocks {
        let block_width = block.validate_shape()?;
        if block_width != width {
            return Err(DagMlError::OofValidation(format!(
                "merge node `{merge_node}` received mismatched off-fold prediction widths ({width} vs {block_width})"
            )));
        }
        let block_targets = if block.target_names.is_empty() {
            (0..block_width).map(|idx| format!("p{idx}")).collect()
        } else {
            block.target_names.clone()
        };
        if block_targets != target_names {
            return Err(DagMlError::OofValidation(format!(
                "merge node `{merge_node}` received inconsistent off-fold target names across branches"
            )));
        }
        for (sample_id, row) in block.sample_ids.iter().zip(&block.values) {
            if by_sample.insert(sample_id.clone(), row.clone()).is_some() {
                return Err(DagMlError::OofValidation(format!(
                    "merge node `{merge_node}` received overlapping off-fold branch predictions: sample `{sample_id}` is covered by more than one partition"
                )));
            }
        }
    }
    let sample_ids: Vec<SampleId> = by_sample.keys().cloned().collect();
    let values: Vec<Vec<f64>> = sample_ids
        .iter()
        .map(|sample_id| by_sample[sample_id].clone())
        .collect();
    Ok(PredictionBlock {
        prediction_id: None,
        producer_node: merge_node.clone(),
        partition: first.partition.clone(),
        fold_id: None,
        sample_ids,
        values,
        target_names,
    })
}

/// Reassemble the per-partition OOF blocks of a separation branch into ONE
/// per-sample OOF block (and its targets) for a concat merge node.
///
/// Slice 3 of native branch support. The fan-out (Slice 2) turns one separation
/// criterion into N branch model nodes; each branch's FIT_CV emits a `Validation`
/// `PredictionBlock` covering ONLY its partition's slice of the current fold's
/// validation set. Nothing reassembled them, so a separation branch could not
/// produce a scored full-universe result. This handler is the reassembly: it
/// reads the merge node's upstream branch OOF blocks (and the per-partition
/// `y_true` the branch models emitted) from the run context, validates them,
/// concatenates by `sample_id`, and emits one merged `Validation` block — with
/// its reassembled targets — whose producer is the merge node.
///
/// Validation is *partition-aware on the inputs* but *full-fold on the output*:
///   - each branch input legitimately covers a SUBSET (its partition) of the
///     fold validation set — never the full set (that is the stacking contract,
///     not concat), so the normal full-fold OOF edge validation does not apply
///     to the inputs;
///   - the inputs must be DISJOINT by sample (separation partitions never share
///     a sample) — an overlap is a hard error;
///   - the reassembled OUTPUT must cover the fold's full validation set, each
///     sample present exactly once (the union of the partitions). This is the
///     completeness the rest of the OOF machinery expects of a producer.
///
/// Scoring (so a separation branch yields a scored full-universe result):
///   - the merged `NodeResult.regression_targets` are reassembled from each
///     branch's per-partition `y_true` (the records `apply_result_scoring`
///     collected from the branch FIT_CV results). This makes the per-fold
///     `apply_result_scoring` score the MERGE producer, AND attributes target
///     records to the merge node so the cross-fold OOF average
///     (`cross_fold_validation_reports`) scores the merge like a normal model.
///     When the branches emit NO targets (mock controllers), the merge emits no
///     targets and stays unscored — exactly as an unscored model node would.
///
/// Variant scoping: a branch model that ALSO carries a generator/sweep produces
/// one block per variant in the same run context. Blocks carry no variant tag,
/// so reads are scoped to the active variant via `scope.variant_id`: a branch's
/// per-fold target records are filtered by variant, and more than one
/// `Validation` block for a (branch, fold) — which only arises when several
/// variants accumulate in one context (the unsupported direct multi-variant
/// path; SELECT isolates each variant in its own context) — is a hard error
/// rather than a silent cross-variant mix. The emitted block id and lineage
/// record id are variant-distinguished so per-variant merges never collide.
///
/// Runs once per fold scope (the campaign phase loops folds, so the handler
/// reassembles within the current fold's validation universe). An empty fold
/// scope (`scope.fold_id == None`) yields no merged block.
fn reassemble_separation_merge(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<Option<NodeResult>> {
    // Concat reassembly is an OOF (validation) operation; it only runs inside a
    // FIT_CV fold scope. Other phases have no per-fold OOF to reassemble.
    let Some(fold_id) = scope.fold_id.clone() else {
        return Ok(None);
    };
    let fold = plan
        .fold_set
        .as_ref()
        .and_then(|fold_set| fold_set.folds.iter().find(|fold| fold.fold_id == fold_id))
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "merge node `{}` references unknown fold `{fold_id}`",
                node_plan.node_id
            ))
        })?;
    let expected: BTreeSet<&SampleId> = fold.validation_sample_ids.iter().collect();

    // A genuinely empty fold (no validation samples) has nothing to reassemble.
    // This is the ONLY skip: any NON-empty fold always runs the union/coverage
    // check below, so an all-empty or partial set of branch inputs surfaces the
    // missing samples as an error instead of silently dropping the merge output.
    if expected.is_empty() {
        return Ok(None);
    }

    // Concatenate every branch's validation OOF block for this fold, keyed by
    // sample id, refusing any sample that two branches both claim. Each branch
    // contributes at most one block per fold (per-variant isolation); two blocks
    // for one (branch, fold) means several variants are mixed in this context.
    let variant_id = scope.variant_id.clone();
    let mut by_sample: BTreeMap<SampleId, Vec<f64>> = BTreeMap::new();
    let mut by_sample_target: BTreeMap<SampleId, Vec<f64>> = BTreeMap::new();
    let mut target_names: Option<Vec<String>> = None;
    let mut target_block_names: Option<Vec<String>> = None;
    let mut width: Option<usize> = None;

    for branch_id in &node_plan.input_nodes {
        let blocks = ctx.prediction_store.find(
            Some(branch_id),
            Some(&PredictionPartition::Validation),
            Some(&fold_id),
        );
        if blocks.is_empty() {
            // The branch had an empty partition ∩ fold (skipped, no OOF block).
            // That is legitimate for a sparse partition; coverage is rechecked
            // against the full fold validation set below.
            continue;
        }
        if blocks.len() > 1 {
            return Err(DagMlError::OofValidation(format!(
                "merge node `{}` found {} validation blocks for branch `{branch_id}` in fold `{fold_id}`: the run context mixes several variants — reassemble each variant in its own context (native SELECT does this)",
                node_plan.node_id,
                blocks.len()
            )));
        }
        let block = blocks[0];
        let block_width = block.validate_shape()?;
        match width {
            None => width = Some(block_width),
            Some(existing) if existing != block_width => {
                return Err(DagMlError::OofValidation(format!(
                    "merge node `{}` received mismatched prediction widths ({existing} vs {block_width}) from branch `{branch_id}`",
                    node_plan.node_id
                )));
            }
            _ => {}
        }
        let block_targets = if block.target_names.is_empty() {
            (0..block_width).map(|idx| format!("p{idx}")).collect()
        } else {
            block.target_names.clone()
        };
        match &target_names {
            None => target_names = Some(block_targets),
            Some(existing) if existing != &block_targets => {
                return Err(DagMlError::OofValidation(format!(
                    "merge node `{}` received inconsistent target names across branches",
                    node_plan.node_id
                )));
            }
            _ => {}
        }
        for (sample_id, values) in block.sample_ids.iter().zip(block.values.iter()) {
            if !expected.contains(sample_id) {
                return Err(DagMlError::OofValidation(format!(
                    "merge node `{}` branch `{branch_id}` emitted sample `{sample_id}` outside fold `{fold_id}` validation set",
                    node_plan.node_id
                )));
            }
            if by_sample
                .insert(sample_id.clone(), values.clone())
                .is_some()
            {
                return Err(DagMlError::OofValidation(format!(
                    "merge node `{}` received overlapping branch predictions: sample `{sample_id}` is covered by more than one partition",
                    node_plan.node_id
                )));
            }
        }

        // Reassemble this branch's per-partition y_true (the records collected
        // from the branch FIT_CV result), scoped to the active variant, so the
        // merge producer can be scored per-fold and cross-fold.
        for record in &ctx.regression_target_records {
            if &record.producer_node != branch_id
                || record.partition != PredictionPartition::Validation
                || record.fold_id.as_ref() != Some(&fold_id)
                || record.variant_id != variant_id
            {
                continue;
            }
            if target_block_names.is_none() && !record.block.target_names.is_empty() {
                target_block_names = Some(record.block.target_names.clone());
            }
            for (unit_id, row) in record.block.unit_ids.iter().zip(&record.block.values) {
                let PredictionUnitId::Sample(sample_id) = unit_id else {
                    continue;
                };
                by_sample_target.insert(sample_id.clone(), row.clone());
            }
        }
    }

    // Full-fold output completeness: the union of partitions must be exactly the
    // fold validation set — no missing sample, none extra. A NON-empty fold with
    // no (or partial) branch inputs lands here and reports the missing samples.
    let covered: BTreeSet<&SampleId> = by_sample.keys().collect();
    if covered != expected {
        let missing: Vec<String> = expected
            .difference(&covered)
            .map(|sample| sample.to_string())
            .collect();
        return Err(DagMlError::OofValidation(format!(
            "merge node `{}` reassembled OOF does not cover fold `{fold_id}` validation set (missing {} sample(s): {})",
            node_plan.node_id,
            missing.len(),
            missing.join(", ")
        )));
    }

    // Deterministic order: emit samples in the fold's declared validation order.
    let sample_ids: Vec<SampleId> = fold.validation_sample_ids.clone();
    let values: Vec<Vec<f64>> = sample_ids
        .iter()
        .map(|sample_id| by_sample.remove(sample_id).expect("sample covered"))
        .collect();
    let target_names = target_names.unwrap_or_default();

    // Reassembled targets: emit a 1:1 target block only when EVERY merged sample
    // has a target row (so `apply_result_scoring` pairs block↔targets exactly). The
    // central R-P1-9 gate makes PARTIAL coverage (some branches emitted y_true,
    // others not) a hard error instead of a silent no-score; no branch emitting
    // targets stays the legitimate unscored case.
    let regression_targets = reassemble_merge_targets(
        &node_plan.node_id,
        &sample_ids,
        &mut by_sample_target,
        target_block_names.unwrap_or_default(),
    )?
    .into_iter()
    .collect();

    // Lineage links every contributing branch (for this variant + fold), so the
    // merge is fully traceable.
    let branch_inputs: BTreeSet<&NodeId> = node_plan.input_nodes.iter().collect();
    let mut input_lineage: Vec<LineageId> = Vec::new();
    for record in ctx.lineage.records() {
        if branch_inputs.contains(&record.node_id)
            && record.phase == scope.phase
            && record.fold_id.as_ref() == Some(&fold_id)
            && record.variant_id == variant_id
        {
            input_lineage.push(record.record_id.clone());
        }
    }

    // Variant-distinguish the emitted id + lineage id so per-variant merges in
    // one context never collide (an empty suffix for the common single-variant
    // case keeps ids stable).
    let variant_suffix = variant_id
        .as_ref()
        .map(|variant| format!(":{variant}"))
        .unwrap_or_default();
    let merged = PredictionBlock {
        prediction_id: Some(format!(
            "merge:{}:{fold_id}{variant_suffix}",
            node_plan.node_id
        )),
        producer_node: node_plan.node_id.clone(),
        partition: PredictionPartition::Validation,
        fold_id: Some(fold_id.clone()),
        sample_ids,
        values,
        target_names,
    };
    merged.validate_shape()?;

    let lineage = LineageRecord {
        record_id: LineageId::new(format!(
            "lineage:{}:{fold_id}{variant_suffix}",
            node_plan.node_id
        ))?,
        run_id: ctx.run_id.clone(),
        node_id: node_plan.node_id.clone(),
        phase: scope.phase,
        controller_id: node_plan.controller_id.clone(),
        controller_version: node_plan.controller_version.clone(),
        variant_id,
        fold_id: Some(fold_id),
        branch_path: Vec::new(),
        input_lineage,
        artifact_refs: Vec::new(),
        params_fingerprint: node_plan.params_fingerprint.clone(),
        data_model_shape_fingerprint: None,
        aggregation_policy_fingerprint: None,
        seed: None,
        unsafe_flags: BTreeSet::new(),
        metrics: BTreeMap::new(),
    };

    Ok(Some(NodeResult {
        node_id: node_plan.node_id.clone(),
        outputs: BTreeMap::new(),
        predictions: vec![merged],
        observation_predictions: Vec::new(),
        aggregated_predictions: Vec::new(),
        explanations: Vec::new(),
        shape_deltas: Vec::new(),
        artifacts: Vec::new(),
        artifact_handles: BTreeMap::new(),
        fit_influence_diagnostics: Vec::new(),
        regression_targets,
        lineage,
    }))
}

/// Average (fuse) the per-branch OOF blocks of a *duplication* branch into ONE
/// per-sample OOF block (and its targets) for a late-fusion merge node.
///
/// The cross-branch analogue of [`reassemble_separation_merge`] for the
/// duplication shape (`[[A], [B]]`, the default branch mode): instead of N
/// branches each covering a DISJOINT partition (concat), N models are fit on the
/// FULL data and the merge AVERAGES their held-out predictions per sample. This
/// is distinct from concat (disjoint reassembly) and from stacking (a meta-model
/// node). [`MergeReduction::Fusion`] averages raw values
/// ([`reduce_predictions_across_branches`]); [`MergeReduction::FusionProbaMean`]
/// averages per-class probability rows and renormalizes
/// ([`reduce_proba_mean_across_branches`]).
///
/// LEAKAGE INVARIANT: fusion averages each branch's HELD-OUT predictions — the
/// `Validation` OOF block of the *current fold* (per fold, never train). It reads
/// exactly the same partition/fold-scoped `Validation` blocks the concat handler
/// reads and emits a `Validation` block under the merge producer, so the
/// CV-scored output is built from out-of-fold predictions only; train
/// predictions never enter the average.
///
/// Asymmetric coverage: a branch that did not predict a sample (a modelless or
/// sparse branch emits no row for it) simply does not contribute — the reducers
/// average each sample over exactly the branches that covered it, never a fixed
/// denominator. The union of branch coverage must still equal the fold
/// validation set (full-fold output completeness); a non-empty fold with missing
/// samples is a hard error, exactly as in concat.
///
/// Targets, variant scoping, lineage and emitted ids mirror
/// [`reassemble_separation_merge`]; the only difference is the value reduction
/// (average vs concatenate) and that branches legitimately OVERLAP on samples
/// (the whole point of fusion) rather than being rejected for overlap.
fn reassemble_fusion_merge(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    ctx: &RunContext,
    scope: &PhaseScope,
    reduction: MergeReduction,
) -> Result<Option<NodeResult>> {
    // Fusion averaging is an OOF (validation) operation; it only runs inside a
    // FIT_CV fold scope. Other phases have no per-fold OOF to fuse.
    let Some(fold_id) = scope.fold_id.clone() else {
        return Ok(None);
    };
    let fold = plan
        .fold_set
        .as_ref()
        .and_then(|fold_set| fold_set.folds.iter().find(|fold| fold.fold_id == fold_id))
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "fusion merge node `{}` references unknown fold `{fold_id}`",
                node_plan.node_id
            ))
        })?;
    let expected: BTreeSet<&SampleId> = fold.validation_sample_ids.iter().collect();

    // A genuinely empty fold (no validation samples) has nothing to fuse.
    if expected.is_empty() {
        return Ok(None);
    }

    // Gather each branch's Validation OOF block for this fold, scoped to the
    // active variant (one block per (branch, fold); >1 means several variants are
    // mixed in this context). Branches legitimately overlap on samples — that is
    // what fusion averages — so unlike concat we collect, not deduplicate.
    let variant_id = scope.variant_id.clone();
    let mut branch_blocks: Vec<PredictionBlock> = Vec::new();
    let mut by_sample_target: BTreeMap<SampleId, Vec<f64>> = BTreeMap::new();
    let mut target_block_names: Option<Vec<String>> = None;

    for branch_id in &node_plan.input_nodes {
        let blocks = ctx.prediction_store.find(
            Some(branch_id),
            Some(&PredictionPartition::Validation),
            Some(&fold_id),
        );
        if blocks.is_empty() {
            // Modelless / sparse branch: no OOF block this fold. Coverage is
            // rechecked against the full fold validation set below.
            continue;
        }
        if blocks.len() > 1 {
            return Err(DagMlError::OofValidation(format!(
                "fusion merge node `{}` found {} validation blocks for branch `{branch_id}` in fold `{fold_id}`: the run context mixes several variants — reassemble each variant in its own context (native SELECT does this)",
                node_plan.node_id,
                blocks.len()
            )));
        }
        let block = blocks[0];
        block.validate_shape()?;
        for sample_id in &block.sample_ids {
            if !expected.contains(sample_id) {
                return Err(DagMlError::OofValidation(format!(
                    "fusion merge node `{}` branch `{branch_id}` emitted sample `{sample_id}` outside fold `{fold_id}` validation set",
                    node_plan.node_id
                )));
            }
        }
        branch_blocks.push(block.clone());

        // Reassemble this branch's per-sample y_true (scoped to the active
        // variant). Each branch predicts the SAME samples, so its targets are the
        // sample's fold-independent ground truth — identical across branches, so a
        // plain per-sample insert (last write wins) is correct.
        for record in &ctx.regression_target_records {
            if &record.producer_node != branch_id
                || record.partition != PredictionPartition::Validation
                || record.fold_id.as_ref() != Some(&fold_id)
                || record.variant_id != variant_id
            {
                continue;
            }
            if target_block_names.is_none() && !record.block.target_names.is_empty() {
                target_block_names = Some(record.block.target_names.clone());
            }
            for (unit_id, row) in record.block.unit_ids.iter().zip(&record.block.values) {
                let PredictionUnitId::Sample(sample_id) = unit_id else {
                    continue;
                };
                by_sample_target.insert(sample_id.clone(), row.clone());
            }
        }
    }

    // Average the branch blocks per sample (over covering branches only). The
    // reducer keys by sample_id, validates uniform width/target-names/partition,
    // and produces the merge producer's fused block.
    let fused = match reduction {
        MergeReduction::Fusion => {
            reduce_predictions_across_branches(&branch_blocks, None, &node_plan.node_id)?
        }
        MergeReduction::FusionProbaMean => {
            reduce_proba_mean_across_branches(&branch_blocks, &node_plan.node_id)?
        }
        MergeReduction::Concat => unreachable!("concat is handled by reassemble_separation_merge"),
    };

    // Full-fold output completeness: the union of branch coverage must be exactly
    // the fold validation set — no missing sample, none extra.
    let covered: BTreeSet<&SampleId> = fused.sample_ids.iter().collect();
    if covered != expected {
        let missing: Vec<String> = expected
            .difference(&covered)
            .map(|sample| sample.to_string())
            .collect();
        return Err(DagMlError::OofValidation(format!(
            "fusion merge node `{}` fused OOF does not cover fold `{fold_id}` validation set (missing {} sample(s): {})",
            node_plan.node_id,
            missing.len(),
            missing.join(", ")
        )));
    }

    // Deterministic order: emit samples in the fold's declared validation order,
    // carrying the fused values for each.
    let fused_by_sample: BTreeMap<&SampleId, &Vec<f64>> =
        fused.sample_ids.iter().zip(&fused.values).collect();
    let sample_ids: Vec<SampleId> = fold.validation_sample_ids.clone();
    let values: Vec<Vec<f64>> = sample_ids
        .iter()
        .map(|sample_id| fused_by_sample[sample_id].clone())
        .collect();
    let target_names = fused.target_names.clone();

    // Reassembled targets: emit a 1:1 target block only when EVERY merged sample
    // has a target row. The central R-P1-9 gate turns PARTIAL coverage into a hard
    // error (never a silent no-score); no branch emitting targets is the legitimate
    // unscored case.
    let regression_targets = reassemble_merge_targets(
        &node_plan.node_id,
        &sample_ids,
        &mut by_sample_target,
        target_block_names.unwrap_or_default(),
    )?
    .into_iter()
    .collect();

    // Lineage links every contributing branch (for this variant + fold).
    let branch_inputs: BTreeSet<&NodeId> = node_plan.input_nodes.iter().collect();
    let mut input_lineage: Vec<LineageId> = Vec::new();
    for record in ctx.lineage.records() {
        if branch_inputs.contains(&record.node_id)
            && record.phase == scope.phase
            && record.fold_id.as_ref() == Some(&fold_id)
            && record.variant_id == variant_id
        {
            input_lineage.push(record.record_id.clone());
        }
    }

    let variant_suffix = variant_id
        .as_ref()
        .map(|variant| format!(":{variant}"))
        .unwrap_or_default();
    let merged = PredictionBlock {
        prediction_id: Some(format!(
            "merge:{}:{fold_id}{variant_suffix}",
            node_plan.node_id
        )),
        producer_node: node_plan.node_id.clone(),
        partition: PredictionPartition::Validation,
        fold_id: Some(fold_id.clone()),
        sample_ids,
        values,
        target_names,
    };
    merged.validate_shape()?;

    let lineage = LineageRecord {
        record_id: LineageId::new(format!(
            "lineage:{}:{fold_id}{variant_suffix}",
            node_plan.node_id
        ))?,
        run_id: ctx.run_id.clone(),
        node_id: node_plan.node_id.clone(),
        phase: scope.phase,
        controller_id: node_plan.controller_id.clone(),
        controller_version: node_plan.controller_version.clone(),
        variant_id,
        fold_id: Some(fold_id),
        branch_path: Vec::new(),
        input_lineage,
        artifact_refs: Vec::new(),
        params_fingerprint: node_plan.params_fingerprint.clone(),
        data_model_shape_fingerprint: None,
        aggregation_policy_fingerprint: None,
        seed: None,
        unsafe_flags: BTreeSet::new(),
        metrics: BTreeMap::new(),
    };

    Ok(Some(NodeResult {
        node_id: node_plan.node_id.clone(),
        outputs: BTreeMap::new(),
        predictions: vec![merged],
        observation_predictions: Vec::new(),
        aggregated_predictions: Vec::new(),
        explanations: Vec::new(),
        shape_deltas: Vec::new(),
        artifacts: Vec::new(),
        artifact_handles: BTreeMap::new(),
        fit_influence_diagnostics: Vec::new(),
        regression_targets,
        lineage,
    }))
}

/// Extract the `BranchViewPlan` that the DSL compiler stashed in the graph
/// node's metadata under `dsl_branch_view_plan`, if any. Returns `None` when
/// the node was not produced by a separation branch; returns `Err` when the
/// stored value cannot be deserialized as a `BranchViewPlan`. This is the
/// scheduler-side bridge that activates the BranchView wiring at runtime;
/// without it, every `DataProviderViewSpec.branch_view` would stay `None`
/// even when the DSL compiled `branch_view_plans` into the campaign.
fn branch_view_from_node_metadata(
    plan: &ExecutionPlan,
    node_id: &NodeId,
) -> Result<Option<crate::data::BranchViewPlan>> {
    let node = match plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| &node.id == node_id)
    {
        Some(node) => node,
        None => return Ok(None),
    };
    let Some(value) = node.metadata.get("dsl_branch_view_plan") else {
        return Ok(None);
    };
    let plan: crate::data::BranchViewPlan =
        serde_json::from_value(value.clone()).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "node `{node_id}` carries malformed `dsl_branch_view_plan` metadata: {error}"
            ))
        })?;
    plan.validate()?;
    Ok(Some(plan))
}

/// Whether a data view is the FIT (training) input for its scope, or a
/// non-fit (validation / predict / explain) read.
///
/// Exclusion is keyed off this role, not the partition name: `exclude` drops
/// outlier samples from any TRAINING read (even an unsafe
/// `fit_partition=fold_validation` one), while genuine validation/predict reads
/// keep excluded samples so OOF/test coverage stays complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DataViewRole {
    Fit,
    NonFit,
}

fn data_view_for_partition(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
    partition: DataRequestPartition,
    branch_view: Option<&crate::data::BranchViewPlan>,
    role: DataViewRole,
    excluded_samples: &BTreeSet<SampleId>,
) -> Result<DataProviderViewSpec> {
    let fold = fold_for_scope(fold_set, scope.fold_id.as_ref())?;
    let mut sample_ids = sample_ids_for_partition(partition, fold_set, fold);
    // FIT role: enforce exclusion at the SPEC level (sample-local), so the
    // spec, the materialized view, and `equal_sample_influence_weights`
    // row_weights all agree on the same training rows. The policy escape hatch
    // `include_excluded` (+ `allow_excluded_rows`) keeps excluded rows when a
    // user explicitly opts in.
    if role == DataViewRole::Fit
        && !binding.view_policy.include_excluded
        && !excluded_samples.is_empty()
    {
        if let Some(ids) = sample_ids.as_mut() {
            ids.retain(|sample_id| !excluded_samples.contains(sample_id));
        }
    }
    if binding.view_policy.require_sample_ids
        && matches!(
            partition,
            DataRequestPartition::FoldTrain | DataRequestPartition::FoldValidation
        )
        && scope.fold_id.is_some()
        && sample_ids.as_ref().is_none_or(Vec::is_empty)
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "data binding `{}` on `{}` requires sample ids for {:?}",
            binding.input_name, binding.node_id, partition
        )));
    }
    let include_augmented = match partition {
        DataRequestPartition::FoldTrain | DataRequestPartition::FullTrain => {
            binding.view_policy.include_augmented_train
        }
        DataRequestPartition::FoldValidation | DataRequestPartition::Predict => {
            binding.view_policy.include_augmented_validation
        }
    };
    // Exclusion is keyed off the FIT role, not the partition name. A fit
    // (training) read drops excluded rows by default (the policy escape hatch
    // `include_excluded` + `allow_excluded_rows` can keep them); a genuine
    // validation/predict read always retains them so they are still validated
    // and predicted. `filter_relations` honors this `include_excluded` flag as
    // defense-in-depth, but the filtered spec sample_ids above are authoritative.
    let include_excluded = match role {
        DataViewRole::Fit => binding.view_policy.include_excluded,
        DataViewRole::NonFit => true,
    };
    let mut extra = BTreeMap::new();
    extra.insert(
        "feature_set_id".to_string(),
        serde_json::Value::String(binding.feature_set_id().to_string()),
    );
    if !binding.view_policy.unsafe_flags.is_empty() {
        extra.insert(
            "unsafe_flags".to_string(),
            serde_json::Value::Array(
                binding
                    .view_policy
                    .unsafe_flags
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    let view = DataProviderViewSpec {
        sample_ids,
        partition,
        fold_id: match partition {
            DataRequestPartition::FoldTrain | DataRequestPartition::FoldValidation => {
                scope.fold_id.clone()
            }
            DataRequestPartition::FullTrain | DataRequestPartition::Predict => None,
        },
        source_ids: (!binding.source_ids.is_empty()).then(|| binding.source_ids.clone()),
        columns: None,
        include_augmented,
        include_excluded,
        branch_view: branch_view.cloned(),
        extra,
    };
    view.validate()?;
    Ok(view)
}

fn data_partition_for_scope(binding: &DataBinding, scope: &PhaseScope) -> DataRequestPartition {
    match scope.phase {
        Phase::FitCv => binding.view_policy.fit_partition,
        Phase::Refit => DataRequestPartition::FullTrain,
        Phase::Predict | Phase::Explain if scope.fold_id.is_none() => DataRequestPartition::Predict,
        Phase::Predict | Phase::Explain => binding.view_policy.predict_partition,
        Phase::Compile | Phase::Plan | Phase::Select => DataRequestPartition::FullTrain,
    }
}

fn fold_for_scope<'a>(
    fold_set: Option<&'a FoldSet>,
    fold_id: Option<&FoldId>,
) -> Result<Option<&'a FoldAssignment>> {
    let Some(fold_id) = fold_id else {
        return Ok(None);
    };
    let fold_set = fold_set.ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "fold `{fold_id}` requested but execution plan has no fold set"
        ))
    })?;
    fold_set
        .folds
        .iter()
        .find(|fold| &fold.fold_id == fold_id)
        .map(Some)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "fold `{fold_id}` requested but is not present in fold set `{}`",
                fold_set.id
            ))
        })
}

/// Build the inner (nested) `FoldSet` for `node_plan` in `scope`, when an
/// effective inner-CV policy applies. Gated to FIT_CV with an outer fold in
/// scope; returns `Ok(None)` otherwise (no inner CV, or no outer fold to nest
/// within). The inner folds are built from the outer fold's TRAINING samples
/// only, so they are a subset of outer-train by construction (no leakage).
fn inner_fold_set_for_scope(
    campaign: &CampaignSpec,
    outer_fold_set: Option<&FoldSet>,
    node_plan: &NodePlan,
    scope: &PhaseScope,
) -> Result<Option<FoldSet>> {
    if scope.phase != Phase::FitCv {
        return Ok(None);
    }
    let Some(spec) =
        crate::fold::resolve_inner_cv(node_plan.inner_cv.as_ref(), campaign.inner_cv.as_ref())
    else {
        return Ok(None);
    };
    // Nested CV needs an outer fold to nest within. `fold_for_scope` yields
    // `None` only when there is no outer fold in scope (skip), and errors if a
    // fold was requested but is missing from the fold set.
    let Some(outer) = fold_for_scope(outer_fold_set, scope.fold_id.as_ref())? else {
        return Ok(None);
    };
    let outer_groups = &outer_fold_set
        .expect("fold_for_scope returned a fold, so the outer fold set is present")
        .sample_groups;
    Ok(Some(spec.build_inner_fold_set(outer, outer_groups)?))
}

fn sample_ids_for_partition(
    partition: DataRequestPartition,
    fold_set: Option<&FoldSet>,
    fold: Option<&FoldAssignment>,
) -> Option<Vec<SampleId>> {
    match partition {
        DataRequestPartition::FoldTrain => fold.map(|fold| fold.train_sample_ids.clone()),
        DataRequestPartition::FoldValidation => fold.map(|fold| fold.validation_sample_ids.clone()),
        DataRequestPartition::FullTrain => fold_set.map(|fold_set| {
            // R-P2-22 invariant (REFIT-EXCLUDES-TEST): the REFIT final-fit boundary
            // (COORDINATOR_SPEC §REFIT.1) is the selected *training universe*, EXCLUDING
            // held-out test samples. REFIT resolves to `FullTrain`, whose universe is
            // exactly `fold_set.sample_ids` — the pool the splitter partitioned into
            // train/validation folds. The held-out TEST partition is never passed to the
            // splitter: it is a SEPARATE, host-resolved request (`DataRequestPartition::Predict`,
            // sample_ids `None`) and so cannot appear in `fold_set.sample_ids` by construction.
            // This defense-in-depth assertion names that invariant: every FullTrain sample must
            // be accounted for by some fold's train∪validation set (`FoldSet::validate()` only
            // guarantees the ⊆ direction), so an out-of-fold (test/leakage) sample can never
            // silently enter the refit universe.
            debug_assert!(
                {
                    let in_a_fold: BTreeSet<&SampleId> = fold_set
                        .folds
                        .iter()
                        .flat_map(|fold| {
                            fold.train_sample_ids
                                .iter()
                                .chain(fold.validation_sample_ids.iter())
                        })
                        .collect();
                    fold_set
                        .sample_ids
                        .iter()
                        .all(|sample_id| in_a_fold.contains(sample_id))
                },
                "REFIT FullTrain universe must be fully fold-accounted (train∪validation); a sample outside every fold would be a held-out/test leakage into the refit training set"
            );
            fold_set.sample_ids.clone()
        }),
        DataRequestPartition::Predict => None,
    }
}

fn preload_replay_prediction_cache_store(
    bundle: &ExecutionBundle,
    prediction_cache_store: Option<&dyn RuntimePredictionCacheStore>,
    ctx: &mut RunContext,
) -> Result<()> {
    if bundle.prediction_requirements.is_empty() {
        return Ok(());
    }
    let store = prediction_cache_store.ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "bundle `{}` cannot preload OOF prediction caches without a prediction cache store",
            bundle.bundle_id
        ))
    })?;
    if !ctx.prediction_store.blocks().is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "bundle `{}` cannot preload OOF prediction caches into a non-empty prediction store",
            bundle.bundle_id
        )));
    }
    let contracts = replay_prediction_cache_contracts(bundle)?;
    for contract in contracts.values() {
        if contract.requirement.prediction_level == PredictionLevel::Sample {
            let blocks = store.load_blocks(&contract.cache.requirement_key)?;
            if blocks.iter().any(|block| {
                block.producer_node != contract.requirement.producer_node
                    || block.partition != contract.requirement.partition
            }) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store returned blocks outside requirement `{}`",
                    contract.cache.requirement_key
                )));
            }
            let payload = build_prediction_cache_payload(&contract.requirement, &blocks)?;
            validate_prediction_cache_payload_matches_record(&payload, &contract.cache)?;
            for block in &payload.blocks {
                ctx.prediction_store.append(block.clone())?;
            }
        } else {
            let blocks = store.load_aggregated_blocks(&contract.cache.requirement_key)?;
            if blocks.iter().any(|block| {
                block.producer_node != contract.requirement.producer_node
                    || block.partition != contract.requirement.partition
                    || block.level != contract.requirement.prediction_level
            }) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store returned aggregated blocks outside requirement `{}`",
                    contract.cache.requirement_key
                )));
            }
            let payload =
                build_aggregated_prediction_cache_payload(&contract.requirement, &blocks)?;
            validate_prediction_cache_payload_matches_record(&payload, &contract.cache)?;
        }
    }
    Ok(())
}

fn replay_prediction_cache_contracts(
    bundle: &ExecutionBundle,
) -> Result<BTreeMap<String, ReplayPredictionCacheContract>> {
    bundle.validate()?;
    let requirements = bundle
        .prediction_requirements
        .iter()
        .map(|requirement| (requirement.key(), requirement))
        .collect::<BTreeMap<_, _>>();
    let mut contracts = BTreeMap::new();
    for cache in &bundle.prediction_caches {
        let requirement = requirements.get(&cache.requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` references unknown prediction requirement `{}`",
                cache.cache_id, cache.requirement_key
            ))
        })?;
        contracts.insert(
            cache.requirement_key.clone(),
            ReplayPredictionCacheContract {
                requirement: (*requirement).clone(),
                cache: cache.clone(),
            },
        );
    }
    Ok(contracts)
}

fn materialize_replay_artifact_handles(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    replay_request: &ReplayPhaseRequest,
    artifact_store: &dyn RuntimeArtifactStore,
    ctx: &RunContext,
) -> Result<MaterializedReplayArtifacts> {
    let mut handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
    let mut inputs = BTreeMap::<NodeId, BTreeMap<String, ArtifactInputSpec>>::new();
    for artifact in &bundle.refit_artifacts {
        artifact.validate()?;
        let node_plan = plan.node_plans.get(&artifact.node_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "bundle `{}` artifact references unknown node `{}`",
                bundle.bundle_id, artifact.node_id
            ))
        })?;
        if !node_plan.supported_phases.contains(&replay_request.phase) {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` artifact node `{}` does not support replay phase {:?}",
                bundle.bundle_id, artifact.node_id, replay_request.phase
            )));
        }
        let handle = artifact_store.materialize(&ArtifactMaterializationRequest {
            run_id: ctx.run_id.clone(),
            bundle_id: bundle.bundle_id.clone(),
            node_id: artifact.node_id.clone(),
            phase: replay_request.phase,
            variant_id: bundle.selected_variant_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
        })?;
        if !matches!(handle.kind, HandleKind::Model | HandleKind::Artifact) {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` materialized as unsupported handle kind {:?}",
                artifact.artifact.id, handle.kind
            )));
        }
        if handle.owner_controller != artifact.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` handle owner `{}` does not match controller `{}`",
                artifact.artifact.id, handle.owner_controller, artifact.controller_id
            )));
        }
        let key = refit_artifact_input_key(&artifact.artifact.id);
        if handles
            .entry(artifact.node_id.clone())
            .or_default()
            .insert(key.clone(), handle)
            .is_some()
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate replay artifact input `{key}` for node `{}`",
                artifact.node_id
            )));
        }
        if inputs
            .entry(artifact.node_id.clone())
            .or_default()
            .insert(key.clone(), ArtifactInputSpec::from_refit_record(artifact)?)
            .is_some()
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate replay artifact metadata `{key}` for node `{}`",
                artifact.node_id
            )));
        }
    }
    Ok(MaterializedReplayArtifacts { handles, inputs })
}

fn derive_task_seed(
    root_seed: Option<u64>,
    variant_id: Option<&VariantId>,
    fold_id: Option<&FoldId>,
    node_plan: &NodePlan,
    phase: Phase,
) -> Option<u64> {
    root_seed.map(|root| {
        let mut context = SeedContext::root(root);
        if let Some(variant_id) = variant_id {
            context = context.child(format!("variant:{variant_id}"));
        }
        if let Some(fold_id) = fold_id {
            context = context.child(format!("fold:{fold_id}"));
        }
        context
            .child(format!("node:{}", node_plan.node_id))
            .child(format!("phase:{phase:?}"))
            .derive_u64("task")
    })
}

#[cfg(test)]
mod explain_contract_tests {
    use super::*;

    fn block(method: &str) -> ExplanationBlock {
        ExplanationBlock {
            producer_node: NodeId::new("model:base").unwrap(),
            method: method.to_string(),
            target_name: Some("y".to_string()),
            payload: serde_json::json!({"feature_importance": [0.5, 0.3, 0.2]}),
        }
    }

    #[test]
    fn validates_well_formed_explanation() {
        assert!(block("shap").validate().is_ok());
    }

    #[test]
    fn rejects_empty_method() {
        assert!(block("  ").validate().is_err());
    }

    #[test]
    fn rejects_empty_target_name() {
        let mut b = block("shap");
        b.target_name = Some(String::new());
        assert!(b.validate().is_err());
    }

    #[test]
    fn round_trips_through_json() {
        let b = block("permutation_importance");
        let json = serde_json::to_string(&b).expect("serialize");
        let parsed: ExplanationBlock = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, b);
        // `target_name` is omitted when absent.
        let mut without = block("shap");
        without.target_name = None;
        let json = serde_json::to_string(&without).expect("serialize");
        assert!(!json.contains("target_name"));
    }
}

#[cfg(test)]
mod tests;
