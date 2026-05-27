use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::aggregation::{AggregatedPredictionBlock, PredictionUnitId};
use crate::bundle::{
    build_aggregated_prediction_cache_payload, build_prediction_cache_payload,
    bundle_prediction_requirement_key, validate_prediction_cache_payload_matches_record,
    BundlePredictionCachePayload, BundlePredictionCachePayloadSet, BundlePredictionCacheRecord,
    BundlePredictionRequirement, ExecutionBundle, RefitArtifactRecord, ReplayPhaseRequest,
};
use crate::campaign::stable_json_fingerprint;
use crate::data::{DataBinding, DataRequestPartition, ExternalDataPlanEnvelope};
use crate::error::{DagMlError, Result};
use crate::fold::{FoldAssignment, FoldSet};
use crate::generation::{GenerationChoice, VariantPlan};
use crate::graph::{EdgeSpec, PortKind};
use crate::ids::{
    ArtifactId, BranchId, BundleId, ControllerId, FoldId, LineageId, NodeId, RunId, SampleId,
    VariantId,
};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::phase::Phase;
use crate::plan::{ExecutionPlan, NodePlan};
use crate::policy::{PredictionLevel, ShapeDelta, ShapeDeltaKind};
use crate::rng::SeedContext;

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
                prediction_requirement_keys: task
                    .prediction_inputs
                    .values()
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
        block.validate_shape()?;
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
    pub seed: Option<u64>,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeResult {
    pub node_id: NodeId,
    #[serde(default)]
    pub outputs: BTreeMap<String, HandleRef>,
    #[serde(default)]
    pub predictions: Vec<PredictionBlock>,
    #[serde(default)]
    pub shape_deltas: Vec<ShapeDelta>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default)]
    pub artifact_handles: BTreeMap<ArtifactId, HandleRef>,
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
        validate_lineage_shape_fingerprints(&self.lineage, task)?;
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
}

pub trait RuntimeController: Send + Sync {
    fn controller_id(&self) -> &ControllerId;
    fn invoke(&self, task: &NodeTask) -> Result<NodeResult>;
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

#[derive(Clone, Debug)]
pub struct RunContext {
    pub run_id: RunId,
    pub root_seed: Option<u64>,
    pub variant_id: Option<VariantId>,
    pub prediction_store: InMemoryPredictionStore,
    pub lineage: InMemoryLineageRecorder,
}

impl RunContext {
    pub fn new(run_id: RunId, root_seed: Option<u64>) -> Self {
        Self {
            run_id,
            root_seed,
            variant_id: None,
            prediction_store: InMemoryPredictionStore::new(),
            lineage: InMemoryLineageRecorder::new(),
        }
    }
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
                let task = NodeTask {
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
                    seed: derive_task_seed(
                        scope.seed_root,
                        scope.variant_id.as_ref(),
                        scope.fold_id.as_ref(),
                        &task_node_plan,
                        scope.phase,
                    ),
                };
                let mut result = controller.invoke(&task)?;
                result.validate_for_task(&task)?;
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
        plan.validate_parallel_controller_capabilities(self.max_workers, scope.phase)?;
        let mut results = Vec::new();
        let mut output_handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
        let mut output_data_views =
            BTreeMap::<NodeId, BTreeMap<String, DataProviderViewSpec>>::new();
        let mut input_lineage = BTreeMap::<NodeId, LineageId>::new();

        for level in plan.node_parallel_levels_for_phase(scope.phase)? {
            let mut prepared = Vec::<PreparedNodeTask>::new();
            for node_id in &level {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
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
                prepared.push(PreparedNodeTask {
                    node_id: node_id.clone(),
                    task: NodeTask {
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
                            handles.push(thread_scope.spawn(move || {
                                let result = controller.invoke(&prepared_task.task)?;
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
    for upstream in &node_plan.input_nodes {
        if training_oof_sources.contains(upstream) {
            continue;
        }
        if let Some(handles) = output_handles.get(upstream) {
            for (port, handle) in handles {
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
    if !node_plan.data_bindings.is_empty() && resources.data_provider.is_none() {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` requires {} data binding(s) but no runtime data provider is registered",
            node_plan.node_id,
            node_plan.data_bindings.len()
        )));
    }
    if let Some(data_provider) = resources.data_provider {
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
            let view = data_view_for_scope(binding, plan.fold_set.as_ref(), scope)?;
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

            if let Some(validation_view) =
                validation_data_view_for_scope(binding, plan.fold_set.as_ref(), scope)?
            {
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

fn incoming_training_oof_edges<'a>(
    plan: &'a ExecutionPlan,
    node_plan: &NodePlan,
    scope: &PhaseScope,
) -> Result<Vec<&'a EdgeSpec>> {
    if !scope.phase.is_training() {
        return Ok(Vec::new());
    }
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
    let blocks = match scope.phase {
        Phase::FitCv => validate_fit_cv_oof_edge(plan, edge, ctx, scope)?,
        Phase::Refit => validate_refit_oof_edge(plan, edge, ctx)?,
        _ => Vec::new(),
    };
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
    Ok(CollectedPredictionInput {
        handle,
        spec: prediction_input_spec(edge, scope, &blocks)?,
    })
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
    if edge.contract.requires_fold_alignment {
        let fold_set = required_fold_set_for_oof(plan, edge)?;
        validate_oof_blocks_match_fold(edge, fold_set, fold_id, &blocks)?;
    }
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
    if edge.contract.requires_fold_alignment {
        let fold_set = required_fold_set_for_oof(plan, edge)?;
        validate_oof_blocks_cover_fold_set(edge, fold_set, &blocks)?;
    }
    Ok(blocks)
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
            if !all_samples.insert(sample_id.clone()) {
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
    if view.extra.contains_key(DATA_OUTPUT_PROVENANCE_KEY) {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` cannot propagate data output `{port_name}` because input view metadata already contains reserved key `{DATA_OUTPUT_PROVENANCE_KEY}`",
            task.node_plan.node_id
        )));
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
    data_provider.make_view(&DataViewRequest {
        run_id: ctx.run_id.clone(),
        node_id: node_plan.node_id.clone(),
        input_name: binding.input_name.clone(),
        phase: scope.phase,
        variant_id: scope.variant_id.clone(),
        fold_id: scope.fold_id.clone(),
        binding: binding.clone(),
        data_handle: data_handle.clone(),
        view: view.clone(),
    })
}

fn data_view_for_scope(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
) -> Result<DataProviderViewSpec> {
    let partition = data_partition_for_scope(binding, scope);
    data_view_for_partition(binding, fold_set, scope, partition)
}

fn validation_data_view_for_scope(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
) -> Result<Option<DataProviderViewSpec>> {
    if scope.phase != Phase::FitCv || scope.fold_id.is_none() {
        return Ok(None);
    }
    let partition = binding.view_policy.predict_partition;
    if partition == data_partition_for_scope(binding, scope) {
        return Ok(None);
    }
    data_view_for_partition(binding, fold_set, scope, partition).map(Some)
}

fn data_view_for_partition(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
    partition: DataRequestPartition,
) -> Result<DataProviderViewSpec> {
    let fold = fold_for_scope(fold_set, scope.fold_id.as_ref())?;
    let sample_ids = sample_ids_for_partition(partition, fold_set, fold);
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
        include_excluded: binding.view_policy.include_excluded,
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

fn sample_ids_for_partition(
    partition: DataRequestPartition,
    fold_set: Option<&FoldSet>,
    fold: Option<&FoldAssignment>,
) -> Option<Vec<SampleId>> {
    match partition {
        DataRequestPartition::FoldTrain => fold.map(|fold| fold.train_sample_ids.clone()),
        DataRequestPartition::FoldValidation => fold.map(|fold| fold.validation_sample_ids.clone()),
        DataRequestPartition::FullTrain => fold_set.map(|fold_set| fold_set.sample_ids.clone()),
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
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use super::*;
    use crate::aggregation::{aggregate_observation_predictions, ObservationPredictionBlock};
    use crate::bundle::{
        build_aggregated_prediction_cache_payload, build_aggregated_prediction_cache_record,
        build_execution_bundle, build_execution_bundle_with_prediction_contracts,
        build_prediction_cache_payload, build_prediction_cache_record,
        BundlePredictionCachePayloadSet, BundlePredictionRequirement, RefitArtifactRecord,
        ReplayPhaseRequest, PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
    };
    use crate::controller::{
        ArtifactPolicy, ControllerCapability, ControllerFitScope, ControllerManifest,
        ControllerRegistry, RngPolicy,
    };
    use crate::data::{DataViewPolicy, ExternalDataPlanEnvelope, InMemoryDataProvider};
    use crate::fold::{FoldAssignment, FoldSet};
    use crate::generation::{
        GenerationChoice, GenerationDimension, GenerationSpec, GenerationStrategy,
    };
    use crate::graph::{
        EdgeContract, EdgeSpec, GraphInterface, GraphSpec, NodeKind, NodeSpec, PortCardinality,
        PortKind, PortRef, PortSchema, PortSpec,
    };
    use crate::ids::{
        ArtifactId, ControllerId, FoldId, GroupId, NodeId, ObservationId, SampleId, TargetId,
    };
    use crate::oof::{PredictionBlock, PredictionPartition};
    use crate::plan::{build_execution_plan, CampaignSpec, SplitInvocation};
    use crate::policy::{
        AggregationPolicy, DataModelShapePlan, FitBoundary, Granularity, LeakageUnitPolicy,
        ShapeDelta, ShapeDeltaKind, SplitUnit,
    };
    use crate::relation::{SampleRelation, SampleRelationSet};
    use serde_json::json;

    struct MockController {
        id: ControllerId,
        handle: u64,
        emit_prediction: bool,
    }

    struct VariantProbeController {
        id: ControllerId,
        handle: u64,
        variants: Arc<Mutex<Vec<Option<VariantExecutionSpec>>>>,
        node_plans: Arc<Mutex<Vec<NodePlan>>>,
    }

    impl RuntimeController for VariantProbeController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            self.variants.lock().unwrap().push(task.variant.clone());
            self.node_plans.lock().unwrap().push(task.node_plan.clone());
            let variant_label = task
                .variant_id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "base".to_string());
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([(
                    "out".to_string(),
                    HandleRef {
                        handle: self.handle,
                        kind: HandleKind::Data,
                        owner_controller: self.id.clone(),
                    },
                )]),
                predictions: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:{}:{:?}:{variant_label}",
                        task.node_plan.node_id, task.phase
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    struct ShapeDataController {
        id: ControllerId,
        handle: u64,
        before_feature_schema: String,
        after_feature_schema: String,
    }

    impl RuntimeController for ShapeDataController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            let shape_plan = task.node_plan.shape_plan.as_ref().ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "shape data controller `{}` expected a shape plan",
                    task.node_plan.node_id
                ))
            })?;
            let output = HandleRef {
                handle: self.handle,
                kind: HandleKind::Data,
                owner_controller: self.id.clone(),
            };
            let shape_delta = ShapeDelta {
                node_id: task.node_plan.node_id.clone(),
                kind: ShapeDeltaKind::Feature,
                before_fingerprint: self.before_feature_schema.clone(),
                after_fingerprint: self.after_feature_schema.clone(),
                metadata: BTreeMap::from([(
                    "feature_namespace".to_string(),
                    serde_json::Value::String("augmented.noise".to_string()),
                )]),
            };
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([("x_out".to_string(), output)]),
                predictions: Vec::new(),
                shape_deltas: vec![shape_delta],
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:{}:{:?}:{}:shape",
                        task.node_plan.node_id,
                        task.phase,
                        task.fold_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "nofold".to_string())
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: Some(stable_json_fingerprint(shape_plan)?),
                    aggregation_policy_fingerprint: Some(stable_json_fingerprint(
                        &shape_plan.aggregation_policy,
                    )?),
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    struct DataViewProbeController {
        id: ControllerId,
        observed_views: Arc<Mutex<Vec<BTreeMap<String, DataProviderViewSpec>>>>,
        prediction_sample_ids: Option<Vec<SampleId>>,
    }

    impl RuntimeController for DataViewProbeController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            self.observed_views
                .lock()
                .unwrap()
                .push(task.data_views.clone());
            let prediction_sample_ids = self.prediction_sample_ids.clone().unwrap_or_else(|| {
                validation_view_sample_ids(task)
                    .map(|ids| ids.into_iter().collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![SampleId::new("s1").unwrap()])
            });
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([(
                    "oof".to_string(),
                    HandleRef {
                        handle: 44,
                        kind: HandleKind::Prediction,
                        owner_controller: self.id.clone(),
                    },
                )]),
                predictions: vec![PredictionBlock {
                    prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                    producer_node: task.node_plan.node_id.clone(),
                    partition: PredictionPartition::Validation,
                    fold_id: task.fold_id.clone(),
                    sample_ids: prediction_sample_ids.clone(),
                    values: vec![vec![1.0]; prediction_sample_ids.len()],
                    target_names: vec!["y".to_string()],
                }],
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:{}:{:?}:{}:probe",
                        task.node_plan.node_id,
                        task.phase,
                        task.fold_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "nofold".to_string())
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    impl RuntimeController for MockController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            for binding in &task.node_plan.data_bindings {
                let key = format!("data:{}", binding.input_name);
                let handle = task.input_handles.get(&key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive data handle `{key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-data/data-view handle for `{key}`",
                        task.node_plan.node_id
                    )));
                }
                if !task.data_views.contains_key(&key) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive data view spec for `{key}`",
                        task.node_plan.node_id
                    )));
                }
                if task.phase == Phase::FitCv && task.fold_id.is_some() {
                    let validation_key = format!("{key}:validation");
                    let validation_view = task.data_views.get(&validation_key).ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "node `{}` did not receive validation data view spec for `{validation_key}`",
                            task.node_plan.node_id
                        ))
                    })?;
                    if validation_view.partition != DataRequestPartition::FoldValidation {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "node `{}` received non-validation data view for `{validation_key}`",
                            task.node_plan.node_id
                        )));
                    }
                }
            }
            let variant_label = task
                .variant_id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "base".to_string());
            let fold_label = task
                .fold_id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "nofold".to_string());
            let output = HandleRef {
                handle: self.handle,
                kind: HandleKind::Data,
                owner_controller: self.id.clone(),
            };
            let prediction_output = HandleRef {
                handle: self.handle,
                kind: HandleKind::Prediction,
                owner_controller: self.id.clone(),
            };
            let prediction_sample_ids = validation_view_sample_ids(task)
                .map(|ids| ids.into_iter().collect::<Vec<_>>())
                .unwrap_or_else(|| vec![SampleId::new("s1").unwrap()]);
            let predictions = self
                .emit_prediction
                .then(|| PredictionBlock {
                    prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                    producer_node: task.node_plan.node_id.clone(),
                    partition: PredictionPartition::Validation,
                    fold_id: task.fold_id.clone(),
                    sample_ids: prediction_sample_ids.clone(),
                    values: vec![vec![1.0]; prediction_sample_ids.len()],
                    target_names: vec!["y".to_string()],
                })
                .into_iter()
                .collect::<Vec<_>>();
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([
                    ("out".to_string(), output.clone()),
                    ("x".to_string(), output.clone()),
                    ("x_out".to_string(), output),
                    ("pred".to_string(), prediction_output.clone()),
                    ("oof".to_string(), prediction_output),
                ]),
                predictions,
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:{}:{:?}:{variant_label}:{fold_label}",
                        task.node_plan.node_id, task.phase
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    struct ReplayMockController {
        id: ControllerId,
        handle: u64,
        require_artifact: bool,
        emit_prediction: bool,
        emit_refit_artifact: bool,
    }

    impl RuntimeController for ReplayMockController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            for binding in &task.node_plan.data_bindings {
                let key = format!("data:{}", binding.input_name);
                let handle = task.input_handles.get(&key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive data handle `{key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-data/data-view handle for `{key}`",
                        task.node_plan.node_id
                    )));
                }
                if !task.data_views.contains_key(&key) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive data view spec for `{key}`",
                        task.node_plan.node_id
                    )));
                }
                if task.phase == Phase::FitCv && task.fold_id.is_some() {
                    let validation_key = format!("{key}:validation");
                    let validation_view = task.data_views.get(&validation_key).ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "node `{}` did not receive validation data view spec for `{validation_key}`",
                            task.node_plan.node_id
                        ))
                    })?;
                    if validation_view.partition != DataRequestPartition::FoldValidation {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "node `{}` received non-validation data view for `{validation_key}`",
                            task.node_plan.node_id
                        )));
                    }
                }
            }
            if self.require_artifact {
                let artifact_id = ArtifactId::new("artifact:model:base:refit").unwrap();
                let key = refit_artifact_input_key(&artifact_id);
                let handle = task.input_handles.get(&key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive refit artifact handle `{key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if handle.kind != HandleKind::Model {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-model refit handle for `{key}`",
                        task.node_plan.node_id
                    )));
                }
                let artifact_input = task.artifact_inputs.get(&key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive refit artifact metadata `{key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if artifact_input.artifact.id != artifact_id
                    || artifact_input.node_id != task.node_plan.node_id
                    || artifact_input.controller_id != task.node_plan.controller_id
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received mismatched refit artifact metadata `{key}`",
                        task.node_plan.node_id
                    )));
                }
            }

            let output = HandleRef {
                handle: self.handle,
                kind: HandleKind::Data,
                owner_controller: self.id.clone(),
            };
            let predictions = self
                .emit_prediction
                .then(|| PredictionBlock {
                    prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                    producer_node: task.node_plan.node_id.clone(),
                    partition: PredictionPartition::Final,
                    fold_id: None,
                    sample_ids: vec![SampleId::new("sample:mock").unwrap()],
                    values: vec![vec![self.handle as f64]],
                    target_names: vec!["y".to_string()],
                })
                .into_iter()
                .collect::<Vec<_>>();
            let artifacts = if self.emit_refit_artifact && task.phase == Phase::Refit {
                vec![ArtifactRef {
                    id: ArtifactId::new(format!("artifact:{}:refit", task.node_plan.node_id))
                        .unwrap(),
                    kind: "mock_model".to_string(),
                    controller_id: self.id.clone(),
                    backend: None,
                    uri: None,
                    content_fingerprint: None,
                    size_bytes: Some(128),
                    plugin: None,
                    plugin_version: None,
                }]
            } else {
                Vec::new()
            };
            let artifact_handles = artifacts
                .iter()
                .map(|artifact| {
                    (
                        artifact.id.clone(),
                        HandleRef {
                            handle: self.handle + 10_000,
                            kind: HandleKind::Model,
                            owner_controller: self.id.clone(),
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>();
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([("out".to_string(), output)]),
                predictions,
                shape_deltas: Vec::new(),
                artifacts: artifacts.clone(),
                artifact_handles,
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:replay:{}:{:?}",
                        task.node_plan.node_id, task.phase
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: artifacts,
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    #[derive(Clone, Copy)]
    enum OofSampleMode {
        Aligned,
        Swapped,
    }

    struct OofEdgeController {
        id: ControllerId,
        base_partition: Option<PredictionPartition>,
        sample_mode: OofSampleMode,
    }

    impl RuntimeController for OofEdgeController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            if task.node_plan.node_id.as_str() == "model:meta" {
                let handle = task.input_handles.get("model:base.pred").ok_or_else(|| {
                    DagMlError::RuntimeValidation(
                        "meta node did not receive OOF prediction input".to_string(),
                    )
                })?;
                if handle.kind != HandleKind::Prediction {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "meta node received {:?} instead of OOF prediction input",
                        handle.kind
                    )));
                }
                let prediction_input =
                    task.prediction_inputs
                        .get("model:base.pred")
                        .ok_or_else(|| {
                            DagMlError::RuntimeValidation(
                                "meta node did not receive OOF prediction input spec".to_string(),
                            )
                        })?;
                if prediction_input.producer_node.as_str() != "model:base"
                    || prediction_input.partition != PredictionPartition::Validation
                    || prediction_input.prediction_level != PredictionLevel::Sample
                    || prediction_input.prediction_width != 1
                {
                    return Err(DagMlError::RuntimeValidation(
                        "meta node received invalid OOF prediction input spec".to_string(),
                    ));
                }
                if task.phase == Phase::FitCv {
                    if prediction_input.fold_id != task.fold_id {
                        return Err(DagMlError::RuntimeValidation(
                            "meta node received OOF prediction spec for the wrong fold".to_string(),
                        ));
                    }
                    if prediction_input.sample_ids != aligned_validation_samples(task) {
                        return Err(DagMlError::RuntimeValidation(
                            "meta node received OOF prediction spec for wrong samples".to_string(),
                        ));
                    }
                }
                if task.phase == Phase::Refit
                    && (prediction_input.fold_id.is_some()
                        || prediction_input.fold_ids
                            != vec![
                                FoldId::new("fold:0").unwrap(),
                                FoldId::new("fold:1").unwrap(),
                            ]
                        || prediction_input.sample_ids
                            != vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()])
                {
                    return Err(DagMlError::RuntimeValidation(
                        "meta node received invalid refit OOF coverage spec".to_string(),
                    ));
                }
            }

            let predictions = if task.node_plan.node_id.as_str() == "model:base" {
                self.base_partition
                    .clone()
                    .map(|partition| {
                        let sample_ids = match self.sample_mode {
                            OofSampleMode::Aligned => aligned_validation_samples(task),
                            OofSampleMode::Swapped => swapped_validation_samples(task),
                        };
                        let fold_id = matches!(
                            partition,
                            PredictionPartition::Train | PredictionPartition::Validation
                        )
                        .then(|| task.fold_id.clone())
                        .flatten();
                        PredictionBlock {
                            prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                            producer_node: task.node_plan.node_id.clone(),
                            partition,
                            fold_id,
                            sample_ids: sample_ids.clone(),
                            values: vec![vec![0.5]; sample_ids.len()],
                            target_names: vec!["y".to_string()],
                        }
                    })
                    .into_iter()
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            let handle_id = if task.node_plan.node_id.as_str() == "model:base" {
                101
            } else {
                202
            };
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([(
                    "pred".to_string(),
                    HandleRef {
                        handle: handle_id,
                        kind: HandleKind::Data,
                        owner_controller: self.id.clone(),
                    },
                )]),
                predictions,
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:oof:{}:{}",
                        task.node_plan.node_id,
                        task.fold_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "nofold".to_string())
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    struct ExpectedRefitOofController {
        id: ControllerId,
        expected_fold_ids: Vec<FoldId>,
        expected_sample_ids: Vec<SampleId>,
        expected_target_names: Vec<String>,
    }

    impl RuntimeController for ExpectedRefitOofController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            if task.node_plan.node_id.as_str() == "model:meta" && task.phase == Phase::Refit {
                let handle = task.input_handles.get("model:base.pred").ok_or_else(|| {
                    DagMlError::RuntimeValidation(
                        "meta node did not receive grouped OOF prediction input".to_string(),
                    )
                })?;
                if handle.kind != HandleKind::Prediction {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "meta node received {:?} instead of grouped OOF prediction input",
                        handle.kind
                    )));
                }
                let prediction_input =
                    task.prediction_inputs
                        .get("model:base.pred")
                        .ok_or_else(|| {
                            DagMlError::RuntimeValidation(
                                "meta node did not receive grouped OOF prediction input spec"
                                    .to_string(),
                            )
                        })?;
                if prediction_input.producer_node.as_str() != "model:base"
                    || prediction_input.source_port != "pred"
                    || prediction_input.target_port != "pred"
                    || prediction_input.partition != PredictionPartition::Validation
                    || prediction_input.prediction_level != PredictionLevel::Sample
                    || prediction_input.fold_id.is_some()
                    || prediction_input.fold_ids != self.expected_fold_ids
                    || prediction_input.sample_ids != self.expected_sample_ids
                    || prediction_input.prediction_width != 1
                    || prediction_input.target_names != self.expected_target_names
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "meta node received invalid grouped refit OOF spec: {:?}",
                        prediction_input
                    )));
                }
            }

            let handle_id = if task.node_plan.node_id.as_str() == "model:base" {
                303
            } else {
                404
            };
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([(
                    "pred".to_string(),
                    HandleRef {
                        handle: handle_id,
                        kind: HandleKind::Prediction,
                        owner_controller: self.id.clone(),
                    },
                )]),
                predictions: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:grouped-oof:{}:{:?}",
                        task.node_plan.node_id, task.phase
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    fn aligned_validation_samples(task: &NodeTask) -> Vec<SampleId> {
        match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
            Some("fold:0") => vec![SampleId::new("s1").unwrap()],
            Some("fold:1") => vec![SampleId::new("s2").unwrap()],
            _ => vec![SampleId::new("s1").unwrap()],
        }
    }

    fn swapped_validation_samples(task: &NodeTask) -> Vec<SampleId> {
        match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
            Some("fold:0") => vec![SampleId::new("s2").unwrap()],
            Some("fold:1") => vec![SampleId::new("s1").unwrap()],
            _ => vec![SampleId::new("s2").unwrap()],
        }
    }

    fn temp_prediction_cache_dir(label: &str) -> PathBuf {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{label}_{}_{}", std::process::id(), suffix))
    }

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

    fn controller_manifest(id: &str, kind: NodeKind) -> ControllerManifest {
        let mut capabilities = BTreeSet::from([
            ControllerCapability::Deterministic,
            ControllerCapability::ThreadSafe,
            ControllerCapability::ProcessSafe,
        ]);
        if kind == NodeKind::Model {
            capabilities.insert(ControllerCapability::EmitsPredictions);
            capabilities.insert(ControllerCapability::ConsumesOofPredictions);
            capabilities.insert(ControllerCapability::EmitsArtifacts);
            capabilities.insert(ControllerCapability::Stateful);
        }
        ControllerManifest {
            controller_id: ControllerId::new(id).unwrap(),
            controller_version: "0.1.0".to_string(),
            operator_kind: kind,
            priority: 0,
            supported_phases: BTreeSet::from([Phase::FitCv]),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            data_requirements: None,
            capabilities,
            fit_scope: ControllerFitScope::FoldTrain,
            rng_policy: RngPolicy::UsesCoreSeed,
            artifact_policy: ArtifactPolicy::Serializable,
        }
    }

    fn simple_graph() -> GraphSpec {
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

    fn independent_parallel_graph() -> GraphSpec {
        GraphSpec {
            id: "g:parallel".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node(
                    "transform:left",
                    NodeKind::Transform,
                    vec![],
                    vec![port("x", PortKind::Data)],
                ),
                node(
                    "transform:right",
                    NodeKind::Transform,
                    vec![],
                    vec![port("x", PortKind::Data)],
                ),
            ],
            edges: Vec::new(),
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        }
    }

    fn parallel_stress_graph() -> GraphSpec {
        const WIDTH: usize = 6;

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut merge_inputs = Vec::new();
        for idx in 0..WIDTH {
            let transform_id = format!("transform:stress.{idx}");
            let model_id = format!("model:stress.{idx}");
            let merge_port = format!("pred{idx}");
            nodes.push(node(
                &transform_id,
                NodeKind::Transform,
                vec![],
                vec![port("x", PortKind::Data)],
            ));
            nodes.push(node(
                &model_id,
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ));
            merge_inputs.push(port(&merge_port, PortKind::Prediction));
            edges.push(EdgeSpec {
                source: PortRef {
                    node_id: NodeId::new(transform_id).unwrap(),
                    port_name: "x".to_string(),
                },
                target: PortRef {
                    node_id: NodeId::new(&model_id).unwrap(),
                    port_name: "x".to_string(),
                },
                contract: EdgeContract {
                    kind: PortKind::Data,
                    representation: None,
                    requires_oof: false,
                    requires_fold_alignment: false,
                    propagates_lineage: true,
                },
            });
            edges.push(EdgeSpec {
                source: PortRef {
                    node_id: NodeId::new(model_id).unwrap(),
                    port_name: "pred".to_string(),
                },
                target: PortRef {
                    node_id: NodeId::new("merge:stress").unwrap(),
                    port_name: merge_port,
                },
                contract: EdgeContract {
                    kind: PortKind::Prediction,
                    representation: None,
                    requires_oof: false,
                    requires_fold_alignment: true,
                    propagates_lineage: true,
                },
            });
        }
        nodes.push(node(
            "merge:stress",
            NodeKind::MixedJoin,
            merge_inputs,
            vec![port("merged", PortKind::Data)],
        ));

        GraphSpec {
            id: "g:parallel.stress".to_string(),
            interface: GraphInterface::default(),
            nodes,
            edges,
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        }
    }

    fn oof_edge_graph() -> GraphSpec {
        GraphSpec {
            id: "g:oof.edge".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node(
                    "model:base",
                    NodeKind::Model,
                    vec![],
                    vec![port("pred", PortKind::Prediction)],
                ),
                node(
                    "model:meta",
                    NodeKind::Model,
                    vec![port("pred", PortKind::Prediction)],
                    vec![port("pred", PortKind::Prediction)],
                ),
            ],
            edges: vec![EdgeSpec {
                source: PortRef {
                    node_id: NodeId::new("model:base").unwrap(),
                    port_name: "pred".to_string(),
                },
                target: PortRef {
                    node_id: NodeId::new("model:meta").unwrap(),
                    port_name: "pred".to_string(),
                },
                contract: EdgeContract {
                    kind: PortKind::Prediction,
                    representation: None,
                    requires_oof: true,
                    requires_fold_alignment: true,
                    propagates_lineage: true,
                },
            }],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        }
    }

    fn runtime_controllers() -> RuntimeControllerRegistry {
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(MockController {
                id: ControllerId::new("controller:transform").unwrap(),
                handle: 1,
                emit_prediction: false,
            }))
            .unwrap();
        controllers
            .register(Box::new(MockController {
                id: ControllerId::new("controller:model").unwrap(),
                handle: 2,
                emit_prediction: true,
            }))
            .unwrap();
        controllers
    }

    fn oof_edge_runtime_controllers(
        base_partition: Option<PredictionPartition>,
        sample_mode: OofSampleMode,
    ) -> RuntimeControllerRegistry {
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(OofEdgeController {
                id: ControllerId::new("controller:model").unwrap(),
                base_partition,
                sample_mode,
            }))
            .unwrap();
        controllers
    }

    fn expected_refit_oof_runtime_controllers(
        expected_fold_ids: Vec<FoldId>,
        expected_sample_ids: Vec<SampleId>,
        expected_target_names: Vec<String>,
    ) -> RuntimeControllerRegistry {
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ExpectedRefitOofController {
                id: ControllerId::new("controller:model").unwrap(),
                expected_fold_ids,
                expected_sample_ids,
                expected_target_names,
            }))
            .unwrap();
        controllers
    }

    fn replay_runtime_controllers() -> RuntimeControllerRegistry {
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:transform.mock").unwrap(),
                handle: 11,
                require_artifact: false,
                emit_prediction: false,
                emit_refit_artifact: false,
            }))
            .unwrap();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:model.mock").unwrap(),
                handle: 22,
                require_artifact: true,
                emit_prediction: true,
                emit_refit_artifact: false,
            }))
            .unwrap();
        controllers
    }

    fn two_fold_set() -> FoldSet {
        FoldSet {
            id: "outer".to_string(),
            sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
            folds: vec![
                FoldAssignment {
                    fold_id: FoldId::new("fold:0").unwrap(),
                    train_sample_ids: vec![SampleId::new("s2").unwrap()],
                    validation_sample_ids: vec![SampleId::new("s1").unwrap()],
                    metadata: BTreeMap::new(),
                },
                FoldAssignment {
                    fold_id: FoldId::new("fold:1").unwrap(),
                    train_sample_ids: vec![SampleId::new("s1").unwrap()],
                    validation_sample_ids: vec![SampleId::new("s2").unwrap()],
                    metadata: BTreeMap::new(),
                },
            ],
            sample_groups: BTreeMap::new(),
        }
    }

    fn three_fold_stress_set() -> FoldSet {
        let samples = (0..6)
            .map(|idx| SampleId::new(format!("s{idx}")).unwrap())
            .collect::<Vec<_>>();
        let folds = (0..3)
            .map(|fold_idx| {
                let validation_sample_ids = samples
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, sample_id)| {
                        (idx % 3 == fold_idx).then_some(sample_id.clone())
                    })
                    .collect::<Vec<_>>();
                let train_sample_ids = samples
                    .iter()
                    .filter(|sample_id| !validation_sample_ids.contains(sample_id))
                    .cloned()
                    .collect::<Vec<_>>();
                FoldAssignment {
                    fold_id: FoldId::new(format!("fold:{fold_idx}")).unwrap(),
                    train_sample_ids,
                    validation_sample_ids,
                    metadata: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        FoldSet {
            id: "outer:stress".to_string(),
            sample_ids: samples,
            folds,
            sample_groups: BTreeMap::new(),
        }
    }

    fn grouped_repetition_fold_set() -> FoldSet {
        let s1 = SampleId::new("s1").unwrap();
        let s1_rep = SampleId::new("s1_rep").unwrap();
        let s2 = SampleId::new("s2").unwrap();
        let s3 = SampleId::new("s3").unwrap();
        FoldSet {
            id: "outer:grouped-repetition".to_string(),
            sample_ids: vec![s1.clone(), s1_rep.clone(), s2.clone(), s3.clone()],
            folds: vec![
                FoldAssignment {
                    fold_id: FoldId::new("fold:0").unwrap(),
                    train_sample_ids: vec![s2.clone(), s3.clone()],
                    validation_sample_ids: vec![s1.clone(), s1_rep.clone()],
                    metadata: BTreeMap::new(),
                },
                FoldAssignment {
                    fold_id: FoldId::new("fold:1").unwrap(),
                    train_sample_ids: vec![s1.clone(), s1_rep.clone(), s3.clone()],
                    validation_sample_ids: vec![s2.clone()],
                    metadata: BTreeMap::new(),
                },
                FoldAssignment {
                    fold_id: FoldId::new("fold:2").unwrap(),
                    train_sample_ids: vec![s1.clone(), s1_rep.clone(), s2.clone()],
                    validation_sample_ids: vec![s3.clone()],
                    metadata: BTreeMap::new(),
                },
            ],
            sample_groups: BTreeMap::from([
                (s1, GroupId::new("group:product1").unwrap()),
                (s1_rep, GroupId::new("group:product1").unwrap()),
                (s2, GroupId::new("group:product2").unwrap()),
                (s3, GroupId::new("group:product3").unwrap()),
            ]),
        }
    }

    fn grouped_leakage_policy() -> LeakageUnitPolicy {
        LeakageUnitPolicy {
            split_unit: SplitUnit::Group,
            require_group_ids: true,
            ..LeakageUnitPolicy::default()
        }
    }

    fn sample_relation(
        observation_id: &str,
        sample_id: &str,
        target_id: &str,
        group_id: &str,
        origin_sample_id: Option<&str>,
        is_augmented: bool,
    ) -> SampleRelation {
        SampleRelation {
            observation_id: ObservationId::new(observation_id).unwrap(),
            sample_id: SampleId::new(sample_id).unwrap(),
            target_id: Some(TargetId::new(target_id).unwrap()),
            group_id: Some(GroupId::new(group_id).unwrap()),
            origin_sample_id: origin_sample_id.map(|value| SampleId::new(value).unwrap()),
            source_id: Some("nir".to_string()),
            is_augmented,
        }
    }

    fn grouped_repetition_relations() -> SampleRelationSet {
        SampleRelationSet {
            records: vec![
                sample_relation(
                    "obs:s1:a",
                    "s1",
                    "target:product1",
                    "group:product1",
                    None,
                    false,
                ),
                sample_relation(
                    "obs:s1:b",
                    "s1",
                    "target:product1",
                    "group:product1",
                    None,
                    false,
                ),
                sample_relation(
                    "obs:s1:aug0",
                    "s1",
                    "target:product1",
                    "group:product1",
                    Some("s1"),
                    true,
                ),
                sample_relation(
                    "obs:s1rep:a",
                    "s1_rep",
                    "target:product1",
                    "group:product1",
                    None,
                    false,
                ),
                sample_relation(
                    "obs:s2:a",
                    "s2",
                    "target:product2",
                    "group:product2",
                    None,
                    false,
                ),
                sample_relation(
                    "obs:s2:b",
                    "s2",
                    "target:product2",
                    "group:product2",
                    None,
                    false,
                ),
                sample_relation(
                    "obs:s3:a",
                    "s3",
                    "target:product3",
                    "group:product3",
                    None,
                    false,
                ),
            ],
        }
    }

    fn grouped_oof_campaign(fold_set: FoldSet) -> CampaignSpec {
        let leakage_policy = grouped_leakage_policy();
        CampaignSpec {
            id: "campaign:oof.grouped-repetition".to_string(),
            root_seed: Some(11),
            leakage_policy: leakage_policy.clone(),
            aggregation_policy: AggregationPolicy::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer.grouped-repetition".to_string(),
                controller_id: None,
                leakage_policy,
                params: BTreeMap::new(),
                fold_set: Some(fold_set),
            }),
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn data_binding(node_id: &NodeId) -> crate::data::DataBinding {
        crate::data::DataBinding {
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
            feature_set_id: Some("x".to_string()),
            source_ids: vec!["nir".to_string()],
            require_relations: true,
            view_policy: Default::default(),
            metadata: BTreeMap::new(),
        }
    }

    fn oof_edge_campaign() -> CampaignSpec {
        CampaignSpec {
            id: "campaign:oof.edge".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer".to_string(),
                controller_id: None,
                leakage_policy: Default::default(),
                params: BTreeMap::new(),
                fold_set: Some(two_fold_set()),
            }),
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn parallel_stress_campaign() -> CampaignSpec {
        CampaignSpec {
            id: "campaign:parallel.stress".to_string(),
            root_seed: Some(31),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:parallel.stress".to_string(),
                controller_id: None,
                leakage_policy: Default::default(),
                params: BTreeMap::new(),
                fold_set: Some(three_fold_stress_set()),
            }),
            generation: GenerationSpec {
                strategy: GenerationStrategy::Cartesian,
                dimensions: vec![GenerationDimension {
                    name: "model_family".to_string(),
                    choices: ["linear", "tree", "kernel"]
                        .into_iter()
                        .enumerate()
                        .map(|(rank, label)| GenerationChoice {
                            label: label.to_string(),
                            value: json!(label),
                            param_overrides: (0..6)
                                .map(|idx| crate::generation::GenerationParamOverride {
                                    node_id: NodeId::new(format!("model:stress.{idx}")).unwrap(),
                                    params: BTreeMap::from([
                                        ("family".to_string(), json!(label)),
                                        ("variant_rank".to_string(), json!(rank)),
                                    ]),
                                })
                                .collect(),
                        })
                        .collect(),
                }],
                max_variants: Some(3),
            },
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn parallel_stress_manifests() -> crate::controller::ControllerRegistry {
        let mut registry = manifests();
        registry
            .register(controller_manifest(
                "controller:mixed_join",
                NodeKind::MixedJoin,
            ))
            .unwrap();
        registry
    }

    fn manifests() -> crate::controller::ControllerRegistry {
        let mut manifests = crate::controller::ControllerRegistry::new();
        manifests
            .register(controller_manifest(
                "controller:transform",
                NodeKind::Transform,
            ))
            .unwrap();
        manifests
            .register(controller_manifest("controller:model", NodeKind::Model))
            .unwrap();
        manifests
    }

    fn oof_edge_manifests(phases: BTreeSet<Phase>) -> crate::controller::ControllerRegistry {
        let mut manifest = controller_manifest("controller:model", NodeKind::Model);
        manifest.supported_phases = phases;
        let mut manifests = crate::controller::ControllerRegistry::new();
        manifests.register(manifest).unwrap();
        manifests
    }

    fn fixture_plan(plan_id: &str) -> ExecutionPlan {
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
        build_execution_plan(plan_id, graph, campaign, &registry).unwrap()
    }

    fn replay_bundle(plan: &ExecutionPlan) -> crate::bundle::ExecutionBundle {
        let model_plan = plan
            .node_plans
            .get(&NodeId::new("model:base").unwrap())
            .unwrap();
        build_execution_bundle(
            crate::ids::BundleId::new("bundle:replay").unwrap(),
            plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            vec![RefitArtifactRecord {
                node_id: model_plan.node_id.clone(),
                controller_id: model_plan.controller_id.clone(),
                artifact: ArtifactRef {
                    id: ArtifactId::new("artifact:model:base:refit").unwrap(),
                    kind: "mock_model".to_string(),
                    controller_id: model_plan.controller_id.clone(),
                    backend: None,
                    uri: None,
                    content_fingerprint: None,
                    size_bytes: Some(128),
                    plugin: None,
                    plugin_version: None,
                },
                params_fingerprint: model_plan.params_fingerprint.clone(),
                data_requirement_keys: vec!["model:base.x".to_string()],
                prediction_requirement_keys: Vec::new(),
            }],
        )
        .unwrap()
    }

    fn replay_request(bundle: &crate::bundle::ExecutionBundle, phase: Phase) -> ReplayPhaseRequest {
        ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase,
            data_envelope_keys: vec!["model:base.x".to_string()],
        }
    }

    fn replay_envelopes() -> BTreeMap<String, ExternalDataPlanEnvelope> {
        BTreeMap::from([(
            "model:base.x".to_string(),
            serde_json::from_str(include_str!(
                "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
            ))
            .unwrap(),
        )])
    }

    fn replay_data_provider() -> InMemoryDataProvider {
        InMemoryDataProvider::with_envelope(
            ControllerId::new("controller:data.provider").unwrap(),
            replay_envelopes().remove("model:base.x").unwrap(),
        )
        .unwrap()
    }

    fn replay_artifact_store(bundle: &crate::bundle::ExecutionBundle) -> InMemoryArtifactStore {
        let mut store = InMemoryArtifactStore::new();
        let artifact = &bundle.refit_artifacts[0];
        store
            .register(
                artifact,
                HandleRef {
                    handle: 9001,
                    kind: HandleKind::Model,
                    owner_controller: artifact.controller_id.clone(),
                },
            )
            .unwrap();
        store
    }

    #[test]
    fn sequential_scheduler_invokes_mock_controllers_in_topological_order() {
        let plan = build_execution_plan(
            "plan:fitcv",
            simple_graph(),
            CampaignSpec {
                id: "campaign:fitcv".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let controllers = runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:1").unwrap(), Some(11));

        let results = SequentialScheduler
            .execute_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(ctx.lineage.len(), 2);
        assert_eq!(ctx.prediction_store.blocks().len(), 1);
        assert_eq!(results[1].node_id.as_str(), "model:pls");
        let transform_lineage = ctx
            .lineage
            .records()
            .find(|record| record.node_id.as_str() == "transform:snv")
            .expect("transform lineage exists");
        let model_lineage = ctx
            .lineage
            .records()
            .find(|record| record.node_id.as_str() == "model:pls")
            .expect("model lineage exists");
        assert_eq!(
            model_lineage.input_lineage,
            vec![transform_lineage.record_id.clone()]
        );
    }

    #[test]
    fn parallel_scheduler_invokes_independent_level_concurrently() {
        struct ConcurrencyProbeController {
            id: ControllerId,
            active: Arc<AtomicUsize>,
            max_active: Arc<AtomicUsize>,
        }

        impl RuntimeController for ConcurrencyProbeController {
            fn controller_id(&self) -> &ControllerId {
                &self.id
            }

            fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                let mut observed = self.max_active.load(Ordering::SeqCst);
                while active > observed
                    && self
                        .max_active
                        .compare_exchange(observed, active, Ordering::SeqCst, Ordering::SeqCst)
                        .is_err()
                {
                    observed = self.max_active.load(Ordering::SeqCst);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(NodeResult {
                    node_id: task.node_plan.node_id.clone(),
                    outputs: BTreeMap::from([(
                        "x".to_string(),
                        HandleRef {
                            handle: task.node_plan.node_id.as_str().len() as u64,
                            kind: HandleKind::Data,
                            owner_controller: self.id.clone(),
                        },
                    )]),
                    predictions: Vec::new(),
                    shape_deltas: Vec::new(),
                    artifacts: Vec::new(),
                    artifact_handles: BTreeMap::new(),
                    lineage: LineageRecord {
                        record_id: LineageId::new(format!(
                            "lineage:parallel:{}",
                            task.node_plan.node_id
                        ))
                        .unwrap(),
                        run_id: task.run_id.clone(),
                        node_id: task.node_plan.node_id.clone(),
                        phase: task.phase,
                        controller_id: self.id.clone(),
                        controller_version: task.node_plan.controller_version.clone(),
                        variant_id: task.variant_id.clone(),
                        fold_id: task.fold_id.clone(),
                        branch_path: task.branch_path.clone(),
                        input_lineage: Vec::new(),
                        artifact_refs: Vec::new(),
                        params_fingerprint: task.node_plan.params_fingerprint.clone(),
                        data_model_shape_fingerprint: None,
                        aggregation_policy_fingerprint: None,
                        seed: task.seed,
                        unsafe_flags: BTreeSet::new(),
                        metrics: BTreeMap::new(),
                    },
                })
            }
        }

        assert!(ParallelScheduler::new(0).is_err());
        let plan = build_execution_plan(
            "plan:parallel",
            independent_parallel_graph(),
            CampaignSpec {
                id: "campaign:parallel".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ConcurrencyProbeController {
                id: ControllerId::new("controller:transform").unwrap(),
                active: Arc::clone(&active),
                max_active: Arc::clone(&max_active),
            }))
            .unwrap();
        let mut ctx = RunContext::new(RunId::new("run:parallel").unwrap(), Some(11));

        let results = ParallelScheduler::new(2)
            .unwrap()
            .execute_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(ctx.lineage.len(), 2);
        assert!(max_active.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn parallel_campaign_scheduler_stress_matches_sequential_across_variants_and_folds() {
        struct StressProbeController {
            id: ControllerId,
            active: Arc<AtomicUsize>,
            max_active: Arc<AtomicUsize>,
            invocations: Arc<Mutex<Vec<String>>>,
            pause: bool,
        }

        impl RuntimeController for StressProbeController {
            fn controller_id(&self) -> &ControllerId {
                &self.id
            }

            fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
                assert_stress_inputs(task)?;
                let task_key = stress_task_key(task);
                self.invocations.lock().unwrap().push(task_key.clone());
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                update_max_active(&self.max_active, active);
                if self.pause {
                    std::thread::sleep(std::time::Duration::from_millis(8));
                }
                self.active.fetch_sub(1, Ordering::SeqCst);

                let (output_name, output_kind) = match &task.node_plan.kind {
                    NodeKind::Model => ("pred", HandleKind::Prediction),
                    NodeKind::MixedJoin => ("merged", HandleKind::Data),
                    _ => ("x", HandleKind::Data),
                };
                let prediction_value = (stable_test_handle(&task_key) % 10_000) as f64 / 100.0;
                let predictions = matches!(&task.node_plan.kind, NodeKind::Model)
                    .then(|| {
                        let sample_ids = stress_validation_samples(task.fold_id.as_ref());
                        PredictionBlock {
                            prediction_id: Some(format!(
                                "prediction:{}:{}:{}",
                                task.node_plan.node_id,
                                task.variant_id
                                    .as_ref()
                                    .map(ToString::to_string)
                                    .unwrap_or_else(|| "variant:base".to_string()),
                                task.fold_id
                                    .as_ref()
                                    .map(ToString::to_string)
                                    .unwrap_or_else(|| "nofold".to_string())
                            )),
                            producer_node: task.node_plan.node_id.clone(),
                            partition: PredictionPartition::Validation,
                            fold_id: task.fold_id.clone(),
                            values: sample_ids
                                .iter()
                                .enumerate()
                                .map(|(idx, _)| vec![prediction_value + idx as f64])
                                .collect(),
                            sample_ids,
                            target_names: vec!["y".to_string()],
                        }
                    })
                    .into_iter()
                    .collect::<Vec<_>>();
                Ok(NodeResult {
                    node_id: task.node_plan.node_id.clone(),
                    outputs: BTreeMap::from([(
                        output_name.to_string(),
                        HandleRef {
                            handle: stable_test_handle(&task_key),
                            kind: output_kind,
                            owner_controller: self.id.clone(),
                        },
                    )]),
                    predictions,
                    shape_deltas: Vec::new(),
                    artifacts: Vec::new(),
                    artifact_handles: BTreeMap::new(),
                    lineage: LineageRecord {
                        record_id: LineageId::new(format!(
                            "lineage:stress:{}:{}:{}",
                            task.node_plan.node_id,
                            task.variant_id
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_else(|| "variant:base".to_string()),
                            task.fold_id
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_else(|| "nofold".to_string())
                        ))
                        .unwrap(),
                        run_id: task.run_id.clone(),
                        node_id: task.node_plan.node_id.clone(),
                        phase: task.phase,
                        controller_id: self.id.clone(),
                        controller_version: task.node_plan.controller_version.clone(),
                        variant_id: task.variant_id.clone(),
                        fold_id: task.fold_id.clone(),
                        branch_path: task.branch_path.clone(),
                        input_lineage: Vec::new(),
                        artifact_refs: Vec::new(),
                        params_fingerprint: task.node_plan.params_fingerprint.clone(),
                        data_model_shape_fingerprint: None,
                        aggregation_policy_fingerprint: None,
                        seed: task.seed,
                        unsafe_flags: BTreeSet::new(),
                        metrics: BTreeMap::new(),
                    },
                })
            }
        }

        fn stress_runtime_controllers(
            active: Arc<AtomicUsize>,
            max_active: Arc<AtomicUsize>,
            invocations: Arc<Mutex<Vec<String>>>,
            pause: bool,
        ) -> RuntimeControllerRegistry {
            let mut controllers = RuntimeControllerRegistry::new();
            for id in [
                "controller:transform",
                "controller:model",
                "controller:mixed_join",
            ] {
                controllers
                    .register(Box::new(StressProbeController {
                        id: ControllerId::new(id).unwrap(),
                        active: Arc::clone(&active),
                        max_active: Arc::clone(&max_active),
                        invocations: Arc::clone(&invocations),
                        pause,
                    }))
                    .unwrap();
            }
            controllers
        }

        fn update_max_active(max_active: &AtomicUsize, active: usize) {
            let mut observed = max_active.load(Ordering::SeqCst);
            while active > observed
                && max_active
                    .compare_exchange(observed, active, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
            {
                observed = max_active.load(Ordering::SeqCst);
            }
        }

        fn stress_task_key(task: &NodeTask) -> String {
            format!(
                "{}|{}|{}|{}|{}",
                task.node_plan.node_id,
                task.variant_id
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "variant:base".to_string()),
                task.fold_id
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "nofold".to_string()),
                task.seed
                    .map(|seed| seed.to_string())
                    .unwrap_or_else(|| "noseed".to_string()),
                task.node_plan.params_fingerprint,
            )
        }

        fn stable_test_handle(label: &str) -> u64 {
            label
                .bytes()
                .fold(14_695_981_039_346_656_037, |hash, byte| {
                    (hash ^ byte as u64).wrapping_mul(1_099_511_628_211)
                })
        }

        fn stress_validation_samples(fold_id: Option<&FoldId>) -> Vec<SampleId> {
            match fold_id.map(FoldId::as_str) {
                Some("fold:0") => vec![SampleId::new("s0").unwrap(), SampleId::new("s3").unwrap()],
                Some("fold:1") => vec![SampleId::new("s1").unwrap(), SampleId::new("s4").unwrap()],
                Some("fold:2") => vec![SampleId::new("s2").unwrap(), SampleId::new("s5").unwrap()],
                _ => vec![SampleId::new("s0").unwrap()],
            }
        }

        fn assert_stress_inputs(task: &NodeTask) -> Result<()> {
            let node_id = task.node_plan.node_id.as_str();
            if node_id.starts_with("transform:stress.") && !task.input_handles.is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "source node `{node_id}` received unexpected inputs"
                )));
            }
            if node_id.starts_with("model:stress.")
                && !task
                    .input_handles
                    .keys()
                    .any(|key| key.starts_with("transform:stress.") && key.ends_with(".x"))
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "model node `{node_id}` did not receive its transform input"
                )));
            }
            if node_id == "merge:stress" {
                let model_inputs = task
                    .input_handles
                    .keys()
                    .filter(|key| key.starts_with("model:stress.") && key.ends_with(".pred"))
                    .count();
                if model_inputs != 6 {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "merge node received {model_inputs} model inputs, expected 6"
                    )));
                }
            }
            Ok(())
        }

        fn lineage_records(ctx: &RunContext) -> Vec<LineageRecord> {
            ctx.lineage.records().cloned().collect::<Vec<_>>()
        }

        let plan = build_execution_plan(
            "plan:parallel.stress",
            parallel_stress_graph(),
            parallel_stress_campaign(),
            &parallel_stress_manifests(),
        )
        .unwrap();
        let levels = plan.node_parallel_levels_for_phase(Phase::FitCv).unwrap();
        assert_eq!(
            levels.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![6, 6, 1]
        );
        assert_eq!(plan.variants.len(), 3);
        assert_eq!(plan.fold_set.as_ref().unwrap().folds.len(), 3);

        let sequential_active = Arc::new(AtomicUsize::new(0));
        let sequential_max_active = Arc::new(AtomicUsize::new(0));
        let sequential_invocations = Arc::new(Mutex::new(Vec::new()));
        let sequential_controllers = stress_runtime_controllers(
            Arc::clone(&sequential_active),
            Arc::clone(&sequential_max_active),
            Arc::clone(&sequential_invocations),
            false,
        );
        let mut sequential_ctx =
            RunContext::new(RunId::new("run:parallel.stress").unwrap(), Some(31));
        let sequential_results = SequentialScheduler
            .execute_campaign_phase(
                &plan,
                &sequential_controllers,
                &mut sequential_ctx,
                Phase::FitCv,
            )
            .unwrap();

        let parallel_active = Arc::new(AtomicUsize::new(0));
        let parallel_max_active = Arc::new(AtomicUsize::new(0));
        let parallel_invocations = Arc::new(Mutex::new(Vec::new()));
        let parallel_controllers = stress_runtime_controllers(
            Arc::clone(&parallel_active),
            Arc::clone(&parallel_max_active),
            Arc::clone(&parallel_invocations),
            true,
        );
        let mut parallel_ctx =
            RunContext::new(RunId::new("run:parallel.stress").unwrap(), Some(31));
        let parallel_results = ParallelScheduler::new(4)
            .unwrap()
            .execute_campaign_phase(
                &plan,
                &parallel_controllers,
                &mut parallel_ctx,
                Phase::FitCv,
            )
            .unwrap();

        assert_eq!(sequential_results.len(), 117);
        assert_eq!(parallel_results, sequential_results);
        assert_eq!(
            parallel_ctx.prediction_store.blocks(),
            sequential_ctx.prediction_store.blocks()
        );
        assert_eq!(
            lineage_records(&parallel_ctx),
            lineage_records(&sequential_ctx)
        );
        assert_eq!(parallel_ctx.prediction_store.blocks().len(), 54);
        assert_eq!(parallel_ctx.lineage.len(), 117);
        assert_eq!(
            parallel_results
                .iter()
                .filter_map(|result| result.lineage.seed)
                .collect::<BTreeSet<_>>()
                .len(),
            parallel_results.len()
        );
        assert_eq!(
            parallel_invocations
                .lock()
                .unwrap()
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>(),
            sequential_invocations
                .lock()
                .unwrap()
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
        );
        let observed_parallelism = parallel_max_active.load(Ordering::SeqCst);
        assert!((2..=4).contains(&observed_parallelism));
        assert_eq!(parallel_active.load(Ordering::SeqCst), 0);
        assert_eq!(sequential_max_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn campaign_scheduler_expands_variants_and_cv_folds() {
        let plan = build_execution_plan(
            "plan:campaign",
            simple_graph(),
            CampaignSpec {
                id: "campaign:fitcv".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: Some(SplitInvocation {
                    id: "split:outer".to_string(),
                    controller_id: None,
                    leakage_policy: Default::default(),
                    params: BTreeMap::new(),
                    fold_set: Some(two_fold_set()),
                }),
                generation: GenerationSpec {
                    strategy: GenerationStrategy::Cartesian,
                    dimensions: vec![GenerationDimension {
                        name: "model_family".to_string(),
                        choices: vec![
                            GenerationChoice {
                                label: "pls".to_string(),
                                value: json!("pls"),
                                param_overrides: Vec::new(),
                            },
                            GenerationChoice {
                                label: "rf".to_string(),
                                value: json!("rf"),
                                param_overrides: Vec::new(),
                            },
                        ],
                    }],
                    max_variants: Some(2),
                },
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let controllers = runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:campaign").unwrap(), Some(11));

        let results = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        assert_eq!(results.len(), 8);
        assert_eq!(ctx.lineage.len(), 8);
        assert_eq!(ctx.prediction_store.blocks().len(), 4);
        assert!(ctx
            .lineage
            .records()
            .all(|record| record.variant_id.is_some() && record.fold_id.is_some()));
        assert_eq!(
            ctx.lineage
                .records()
                .filter_map(|record| record.seed)
                .collect::<BTreeSet<_>>()
                .len(),
            8
        );
    }

    #[test]
    fn node_tasks_expose_generation_variant_context() {
        let plan = build_execution_plan(
            "plan:generation.task.context",
            simple_graph(),
            CampaignSpec {
                id: "campaign:generation.task.context".to_string(),
                root_seed: Some(23),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: GenerationSpec {
                    strategy: GenerationStrategy::Cartesian,
                    dimensions: vec![GenerationDimension {
                        name: "model_family".to_string(),
                        choices: vec![
                            GenerationChoice {
                                label: "pls".to_string(),
                                value: json!("pls"),
                                param_overrides: vec![crate::generation::GenerationParamOverride {
                                    node_id: NodeId::new("model:pls").unwrap(),
                                    params: BTreeMap::from([(
                                        "n_components".to_string(),
                                        json!(4),
                                    )]),
                                }],
                            },
                            GenerationChoice {
                                label: "rf".to_string(),
                                value: json!("rf"),
                                param_overrides: vec![crate::generation::GenerationParamOverride {
                                    node_id: NodeId::new("model:pls").unwrap(),
                                    params: BTreeMap::from([("trees".to_string(), json!(64))]),
                                }],
                            },
                        ],
                    }],
                    max_variants: Some(2),
                },
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let observed_variants = Arc::new(Mutex::new(Vec::new()));
        let observed_node_plans = Arc::new(Mutex::new(Vec::new()));
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(MockController {
                id: ControllerId::new("controller:transform").unwrap(),
                handle: 1,
                emit_prediction: false,
            }))
            .unwrap();
        controllers
            .register(Box::new(VariantProbeController {
                id: ControllerId::new("controller:model").unwrap(),
                handle: 2,
                variants: Arc::clone(&observed_variants),
                node_plans: Arc::clone(&observed_node_plans),
            }))
            .unwrap();
        let mut ctx = RunContext::new(RunId::new("run:generation.task.context").unwrap(), Some(23));

        let results = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        assert_eq!(results.len(), 4);
        let observed = observed_variants.lock().unwrap();
        assert_eq!(observed.len(), 2);
        let mut labels = BTreeSet::new();
        for variant in observed.iter().map(|variant| variant.as_ref().unwrap()) {
            variant.validate().unwrap();
            let expected = plan
                .variants
                .iter()
                .find(|planned| planned.variant_id == variant.variant_id)
                .unwrap();
            assert_eq!(variant.choices, expected.choices);
            assert_eq!(variant.fingerprint, expected.fingerprint);
            assert_eq!(variant.seed, expected.seed);
            labels.insert(variant.choices["model_family"].label.as_str());
        }
        assert_eq!(labels, BTreeSet::from(["pls", "rf"]));
        let observed_plans = observed_node_plans.lock().unwrap();
        assert_eq!(observed_plans.len(), 2);
        let base_plan = plan
            .node_plans
            .get(&NodeId::new("model:pls").unwrap())
            .unwrap();
        assert!(observed_plans
            .iter()
            .all(|node_plan| node_plan.params_fingerprint != base_plan.params_fingerprint));
        assert!(observed_plans
            .iter()
            .any(|node_plan| node_plan.params.get("n_components") == Some(&json!(4))));
        assert!(observed_plans
            .iter()
            .any(|node_plan| node_plan.params.get("trees") == Some(&json!(64))));
    }

    #[test]
    fn requires_oof_prediction_edge_supplies_validated_prediction_handle() {
        let plan = build_execution_plan(
            "plan:oof.edge.success",
            oof_edge_graph(),
            oof_edge_campaign(),
            &manifests(),
        )
        .unwrap();
        let controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Aligned,
        );
        let mut ctx = RunContext::new(RunId::new("run:oof.edge.success").unwrap(), Some(11));

        let results = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(ctx.prediction_store.blocks().len(), 2);
        assert_eq!(
            results
                .iter()
                .filter(|result| result.node_id.as_str() == "model:meta")
                .count(),
            2
        );
    }

    #[test]
    fn requires_oof_prediction_edge_rejects_missing_validation_predictions() {
        let plan = build_execution_plan(
            "plan:oof.edge.missing",
            oof_edge_graph(),
            oof_edge_campaign(),
            &manifests(),
        )
        .unwrap();
        let controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);
        let mut ctx = RunContext::new(RunId::new("run:oof.edge.missing").unwrap(), Some(11));

        let error = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires OOF validation predictions"));
        assert!(error.contains("model:base"));
    }

    #[test]
    fn requires_oof_prediction_edge_rejects_train_predictions_as_features() {
        let plan = build_execution_plan(
            "plan:oof.edge.train",
            oof_edge_graph(),
            oof_edge_campaign(),
            &manifests(),
        )
        .unwrap();
        let controllers =
            oof_edge_runtime_controllers(Some(PredictionPartition::Train), OofSampleMode::Aligned);
        let mut ctx = RunContext::new(RunId::new("run:oof.edge.train").unwrap(), Some(11));

        let error = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires OOF validation predictions"));
    }

    #[test]
    fn requires_oof_prediction_edge_rejects_fold_misalignment() {
        let plan = build_execution_plan(
            "plan:oof.edge.misaligned",
            oof_edge_graph(),
            oof_edge_campaign(),
            &manifests(),
        )
        .unwrap();
        let controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Swapped,
        );
        let mut ctx = RunContext::new(RunId::new("run:oof.edge.misaligned").unwrap(), Some(11));

        let error = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .unwrap_err()
            .to_string();

        assert!(error.contains("do not match validation samples"));
    }

    #[test]
    fn requires_oof_prediction_edge_refit_uses_cv_oof_coverage() {
        let plan = build_execution_plan(
            "plan:oof.edge.refit",
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let fit_controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Aligned,
        );
        let mut ctx = RunContext::new(RunId::new("run:oof.edge.refit").unwrap(), Some(11));
        SequentialScheduler
            .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
            .unwrap();
        assert_eq!(ctx.prediction_store.blocks().len(), 2);

        let refit_controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);
        let refit_results = SequentialScheduler
            .execute_campaign_phase(&plan, &refit_controllers, &mut ctx, Phase::Refit)
            .unwrap();

        assert_eq!(refit_results.len(), 2);
        assert_eq!(
            refit_results
                .iter()
                .filter(|result| result.node_id.as_str() == "model:meta")
                .count(),
            1
        );
    }

    #[test]
    fn refit_oof_accepts_grouped_repeated_aggregation_and_refuses_origin_leakage() {
        let fold_set = grouped_repetition_fold_set();
        let relations = grouped_repetition_relations();
        let leakage_policy = grouped_leakage_policy();
        relations
            .validate_against_fold_set(&fold_set, &leakage_policy)
            .unwrap();

        let mut leaky_relations = relations.clone();
        leaky_relations.records.push(sample_relation(
            "obs:s1:leaky_aug",
            "s1",
            "target:product1",
            "group:product1",
            Some("s2"),
            true,
        ));
        let leak_error = leaky_relations
            .validate_against_fold_set(&fold_set, &leakage_policy)
            .unwrap_err()
            .to_string();
        assert!(
            leak_error.contains("leaks origin sample"),
            "unexpected leakage error: {leak_error}"
        );

        let plan = build_execution_plan(
            "plan:oof.edge.grouped-repetition.refit",
            oof_edge_graph(),
            grouped_oof_campaign(fold_set.clone()),
            &oof_edge_manifests(BTreeSet::from([Phase::Refit])),
        )
        .unwrap();
        let mut ctx = RunContext::new(
            RunId::new("run:oof.edge.grouped-repetition.refit").unwrap(),
            Some(11),
        );

        let fold0 = aggregate_observation_predictions(
            &ObservationPredictionBlock {
                prediction_id: Some("pred:model:base:fold0:obs".to_string()),
                producer_node: NodeId::new("model:base").unwrap(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                observation_ids: vec![
                    ObservationId::new("obs:s1:a").unwrap(),
                    ObservationId::new("obs:s1:b").unwrap(),
                    ObservationId::new("obs:s1rep:a").unwrap(),
                ],
                values: vec![vec![1.0], vec![3.0], vec![4.0]],
                weights: Vec::new(),
                target_names: vec!["y".to_string()],
            },
            &relations,
            &AggregationPolicy::default(),
            &[
                SampleId::new("s1").unwrap(),
                SampleId::new("s1_rep").unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(fold0.values, vec![vec![2.0], vec![4.0]]);
        ctx.prediction_store.append(fold0).unwrap();

        let fold1 = aggregate_observation_predictions(
            &ObservationPredictionBlock {
                prediction_id: Some("pred:model:base:fold1:obs".to_string()),
                producer_node: NodeId::new("model:base").unwrap(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:1").unwrap()),
                observation_ids: vec![
                    ObservationId::new("obs:s2:a").unwrap(),
                    ObservationId::new("obs:s2:b").unwrap(),
                ],
                values: vec![vec![10.0], vec![14.0]],
                weights: Vec::new(),
                target_names: vec!["y".to_string()],
            },
            &relations,
            &AggregationPolicy::default(),
            &[SampleId::new("s2").unwrap()],
        )
        .unwrap();
        assert_eq!(fold1.values, vec![vec![12.0]]);
        ctx.prediction_store.append(fold1).unwrap();

        let fold2 = aggregate_observation_predictions(
            &ObservationPredictionBlock {
                prediction_id: Some("pred:model:base:fold2:obs".to_string()),
                producer_node: NodeId::new("model:base").unwrap(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:2").unwrap()),
                observation_ids: vec![ObservationId::new("obs:s3:a").unwrap()],
                values: vec![vec![20.0]],
                weights: Vec::new(),
                target_names: vec!["y".to_string()],
            },
            &relations,
            &AggregationPolicy::default(),
            &[SampleId::new("s3").unwrap()],
        )
        .unwrap();
        assert_eq!(fold2.values, vec![vec![20.0]]);
        ctx.prediction_store.append(fold2).unwrap();
        assert_eq!(ctx.prediction_store.blocks().len(), 3);

        let controllers = expected_refit_oof_runtime_controllers(
            vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
                FoldId::new("fold:2").unwrap(),
            ],
            vec![
                SampleId::new("s1").unwrap(),
                SampleId::new("s1_rep").unwrap(),
                SampleId::new("s2").unwrap(),
                SampleId::new("s3").unwrap(),
            ],
            vec!["y".to_string()],
        );
        let refit_results = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::Refit)
            .unwrap();

        assert_eq!(refit_results.len(), 2);
        assert_eq!(
            refit_results
                .iter()
                .filter(|result| result.node_id.as_str() == "model:meta")
                .count(),
            1
        );
    }

    #[test]
    fn in_memory_prediction_cache_store_loads_and_materializes_oof_payloads() {
        let plan = build_execution_plan(
            "plan:oof.edge.cache.store",
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let fit_controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Aligned,
        );
        let mut ctx = RunContext::new(RunId::new("run:oof.edge.cache.store").unwrap(), Some(11));
        SequentialScheduler
            .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Sample,
            fold_ids: vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
            ],
            unit_ids: Vec::new(),
            sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let cache =
            build_prediction_cache_record(&requirement, ctx.prediction_store.blocks()).unwrap();
        let payload =
            build_prediction_cache_payload(&requirement, ctx.prediction_store.blocks()).unwrap();
        let bundle = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:oof.edge.cache.store").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            Vec::new(),
            vec![requirement.clone()],
            vec![cache.clone()],
        )
        .unwrap();
        let payload_set = BundlePredictionCachePayloadSet {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            caches: vec![payload],
        };
        let store = InMemoryPredictionCacheStore::from_payloads(&bundle, payload_set).unwrap();
        assert_eq!(store.payload_count(), 1);
        assert_eq!(store.load_blocks(&requirement.key()).unwrap().len(), 2);

        ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            data_envelope_keys: Vec::new(),
        }
        .validate_for_bundle_with_prediction_cache_store(&bundle, true)
        .unwrap();

        let handle = store
            .materialize(&PredictionCacheMaterializationRequest {
                run_id: RunId::new("run:oof.edge.cache.store.replay").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                phase: Phase::Refit,
                variant_id: bundle.selected_variant_id.clone(),
                requirement: requirement.clone(),
                cache,
                producer_controller_id: ControllerId::new("controller:model").unwrap(),
            })
            .unwrap();
        assert_eq!(handle.kind, HandleKind::Prediction);
        assert_eq!(
            handle.owner_controller,
            ControllerId::new("controller:model").unwrap()
        );
        let records = store.materialization_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].requirement_key, requirement.key());
        assert_eq!(records[0].handle, handle);
    }

    #[test]
    fn prediction_cache_stores_load_and_materialize_aggregated_payloads() {
        let plan = build_execution_plan(
            "plan:oof.edge.aggregated.cache.store",
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let target_a = PredictionUnitId::Target(TargetId::new("target:a").unwrap());
        let target_b = PredictionUnitId::Target(TargetId::new("target:b").unwrap());
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Target,
            fold_ids: vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
            ],
            unit_ids: vec![target_a.clone(), target_b.clone()],
            sample_ids: Vec::new(),
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let aggregated_blocks = vec![
            AggregatedPredictionBlock {
                prediction_id: Some("prediction:model:base.target.fold0".to_string()),
                producer_node: requirement.producer_node.clone(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                level: PredictionLevel::Target,
                unit_ids: vec![target_a],
                values: vec![vec![0.5]],
                target_names: vec!["y".to_string()],
            },
            AggregatedPredictionBlock {
                prediction_id: Some("prediction:model:base.target.fold1".to_string()),
                producer_node: requirement.producer_node.clone(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:1").unwrap()),
                level: PredictionLevel::Target,
                unit_ids: vec![target_b],
                values: vec![vec![0.7]],
                target_names: vec!["y".to_string()],
            },
        ];
        let cache =
            build_aggregated_prediction_cache_record(&requirement, &aggregated_blocks).unwrap();
        let payload =
            build_aggregated_prediction_cache_payload(&requirement, &aggregated_blocks).unwrap();
        let bundle = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:aggregated.prediction.cache").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            Vec::new(),
            vec![requirement.clone()],
            vec![cache.clone()],
        )
        .unwrap();
        let payload_set = BundlePredictionCachePayloadSet {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            caches: vec![payload.clone()],
        };

        let in_memory =
            InMemoryPredictionCacheStore::from_payloads(&bundle, payload_set.clone()).unwrap();
        assert!(in_memory.load_blocks(&requirement.key()).is_err());
        assert_eq!(
            in_memory
                .load_aggregated_blocks(&requirement.key())
                .unwrap(),
            aggregated_blocks
        );
        let handle = in_memory
            .materialize(&PredictionCacheMaterializationRequest {
                run_id: RunId::new("run:oof.edge.aggregated.cache.store.replay").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                phase: Phase::Refit,
                variant_id: bundle.selected_variant_id.clone(),
                requirement: requirement.clone(),
                cache: cache.clone(),
                producer_controller_id: ControllerId::new("controller:model").unwrap(),
            })
            .unwrap();
        assert_eq!(handle.kind, HandleKind::Prediction);

        let columnar =
            ColumnarPredictionCacheStore::from_payloads(&bundle, payload_set.clone()).unwrap();
        assert_eq!(columnar.entry_count(), 1);
        let manifest = columnar.manifests();
        assert_eq!(manifest.len(), 1);
        assert_eq!(manifest[0].prediction_level, PredictionLevel::Target);
        assert_eq!(manifest[0].value_count, 2);
        assert!(columnar.load_blocks(&requirement.key()).is_err());
        assert_eq!(
            columnar.load_aggregated_blocks(&requirement.key()).unwrap(),
            aggregated_blocks
        );
        let columnar_handle = columnar
            .materialize(&PredictionCacheMaterializationRequest {
                run_id: RunId::new("run:oof.edge.aggregated.columnar.cache.store.replay").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                phase: Phase::Refit,
                variant_id: bundle.selected_variant_id.clone(),
                requirement: requirement.clone(),
                cache: cache.clone(),
                producer_controller_id: ControllerId::new("controller:model").unwrap(),
            })
            .unwrap();
        assert_eq!(columnar_handle.kind, HandleKind::Prediction);

        let root = temp_prediction_cache_dir("dag_ml_aggregated_prediction_cache_store");
        let manifest =
            FilePredictionCacheStore::write_payload_set(&root, &bundle, &payload_set).unwrap();
        assert_eq!(manifest.caches[0].prediction_level, PredictionLevel::Target);
        assert_eq!(manifest.caches[0].unit_ids, requirement.unit_ids);
        let file_store = FilePredictionCacheStore::open(root.clone(), &bundle).unwrap();
        assert_eq!(
            file_store
                .load_aggregated_blocks(&requirement.key())
                .unwrap(),
            aggregated_blocks
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn columnar_prediction_cache_block_round_trips_multi_target_rows() {
        let block = PredictionBlock {
            prediction_id: Some("pred:wide".to_string()),
            producer_node: NodeId::new("model:wide").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
            values: vec![vec![1.0, 10.0], vec![2.0, 20.0]],
            target_names: vec!["y0".to_string(), "y1".to_string()],
        };

        let columnar = ColumnarPredictionCacheBlock::from_prediction_block(&block).unwrap();
        assert_eq!(columnar.width, 2);
        assert_eq!(columnar.row_count(), 2);
        assert_eq!(columnar.value_count(), 4);
        assert_eq!(columnar.columns, vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        assert_eq!(columnar.to_prediction_block().unwrap(), block);
    }

    #[test]
    fn columnar_prediction_cache_block_round_trips_aggregated_units() {
        let block = AggregatedPredictionBlock {
            prediction_id: Some("pred:target".to_string()),
            producer_node: NodeId::new("model:target").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            level: PredictionLevel::Target,
            unit_ids: vec![
                PredictionUnitId::Target(TargetId::new("target:a").unwrap()),
                PredictionUnitId::Target(TargetId::new("target:b").unwrap()),
            ],
            values: vec![vec![1.0, 10.0], vec![2.0, 20.0]],
            target_names: vec!["y0".to_string(), "y1".to_string()],
        };

        let columnar =
            ColumnarPredictionCacheBlock::from_aggregated_prediction_block(&block).unwrap();
        assert_eq!(columnar.prediction_level, PredictionLevel::Target);
        assert_eq!(columnar.row_count(), 2);
        assert_eq!(columnar.value_count(), 4);
        assert_eq!(columnar.columns, vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        assert!(columnar.to_prediction_block().is_err());
        assert_eq!(columnar.to_aggregated_prediction_block().unwrap(), block);
    }

    #[test]
    fn columnar_prediction_cache_store_loads_and_materializes_oof_payloads() {
        let plan = build_execution_plan(
            "plan:oof.edge.columnar.cache.store",
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let fit_controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Aligned,
        );
        let mut ctx = RunContext::new(
            RunId::new("run:oof.edge.columnar.cache.store").unwrap(),
            Some(11),
        );
        SequentialScheduler
            .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Sample,
            fold_ids: vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
            ],
            unit_ids: Vec::new(),
            sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let cache =
            build_prediction_cache_record(&requirement, ctx.prediction_store.blocks()).unwrap();
        let payload =
            build_prediction_cache_payload(&requirement, ctx.prediction_store.blocks()).unwrap();
        let bundle = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:oof.edge.columnar.cache.store").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            Vec::new(),
            vec![requirement.clone()],
            vec![cache.clone()],
        )
        .unwrap();
        let payload_set = BundlePredictionCachePayloadSet {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            caches: vec![payload],
        };
        let store = ColumnarPredictionCacheStore::from_payloads(&bundle, payload_set).unwrap();
        assert_eq!(store.entry_count(), 1);
        let manifest = store.manifests();
        assert_eq!(manifest.len(), 1);
        assert_eq!(manifest[0].requirement_key, requirement.key());
        assert_eq!(manifest[0].prediction_level, PredictionLevel::Sample);
        assert_eq!(manifest[0].value_count, 2);
        assert_eq!(manifest[0].estimated_value_bytes, 16);
        assert_eq!(store.load_blocks(&requirement.key()).unwrap().len(), 2);

        let handle = store
            .materialize(&PredictionCacheMaterializationRequest {
                run_id: RunId::new("run:oof.edge.columnar.cache.store.replay").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                phase: Phase::Refit,
                variant_id: bundle.selected_variant_id.clone(),
                requirement: requirement.clone(),
                cache,
                producer_controller_id: ControllerId::new("controller:model").unwrap(),
            })
            .unwrap();
        assert_eq!(handle.kind, HandleKind::Prediction);
        assert_eq!(
            handle.owner_controller,
            ControllerId::new("controller:model").unwrap()
        );
        let records = store.materialization_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].requirement_key, requirement.key());
        assert_eq!(records[0].handle, handle);
    }

    #[test]
    fn file_prediction_cache_store_round_trips_oof_payloads_and_detects_tampering() {
        let plan = build_execution_plan(
            "plan:oof.edge.file.cache.store",
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let fit_controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Aligned,
        );
        let mut ctx = RunContext::new(
            RunId::new("run:oof.edge.file.cache.store").unwrap(),
            Some(11),
        );
        SequentialScheduler
            .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
            .unwrap();

        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Sample,
            fold_ids: vec![
                FoldId::new("fold:0").unwrap(),
                FoldId::new("fold:1").unwrap(),
            ],
            unit_ids: Vec::new(),
            sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let cache =
            build_prediction_cache_record(&requirement, ctx.prediction_store.blocks()).unwrap();
        let payload =
            build_prediction_cache_payload(&requirement, ctx.prediction_store.blocks()).unwrap();
        let bundle = build_execution_bundle_with_prediction_contracts(
            BundleId::new("bundle:oof.edge.file.cache.store").unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            Vec::new(),
            vec![requirement.clone()],
            vec![cache.clone()],
        )
        .unwrap();
        let payload_set = BundlePredictionCachePayloadSet {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            caches: vec![payload],
        };
        let root = temp_prediction_cache_dir("dag_ml_file_prediction_cache_store");

        let manifest =
            FilePredictionCacheStore::write_payload_set(&root, &bundle, &payload_set).unwrap();
        assert_eq!(manifest.caches.len(), 1);
        assert_eq!(manifest.caches[0].prediction_level, PredictionLevel::Sample);
        assert!(root.join(FILE_PREDICTION_CACHE_MANIFEST_FILE).exists());
        assert!(root.join(&manifest.caches[0].file_name).exists());

        let store = FilePredictionCacheStore::open(root.clone(), &bundle).unwrap();
        assert_eq!(store.manifest().caches, manifest.caches);
        assert_eq!(store.load_blocks(&requirement.key()).unwrap().len(), 2);
        let handle = store
            .materialize(&PredictionCacheMaterializationRequest {
                run_id: RunId::new("run:oof.edge.file.cache.store.replay").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                phase: Phase::Refit,
                variant_id: bundle.selected_variant_id.clone(),
                requirement: requirement.clone(),
                cache: cache.clone(),
                producer_controller_id: ControllerId::new("controller:model").unwrap(),
            })
            .unwrap();
        assert_eq!(handle.kind, HandleKind::Prediction);
        assert_eq!(store.materialization_records().len(), 1);

        let payload_path = root.join(&manifest.caches[0].file_name);
        let mut tampered: serde_json::Value =
            serde_json::from_slice(&fs::read(&payload_path).unwrap()).unwrap();
        tampered["blocks"][0]["values"][0][0] = json!(123456.0);
        fs::write(&payload_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        let err = store.load_blocks(&requirement.key()).unwrap_err();
        assert!(
            err.to_string().contains("content fingerprint"),
            "unexpected tamper error: {err}"
        );

        let _ = fs::remove_dir_all(root);
    }

    fn portable_artifact_bundle(plan: &ExecutionPlan) -> crate::bundle::ExecutionBundle {
        let model_plan = plan
            .node_plans
            .get(&NodeId::new("model:base").unwrap())
            .unwrap();
        let content_fingerprint = "a".repeat(64);
        build_execution_bundle(
            crate::ids::BundleId::new("bundle:artifact.manifest").unwrap(),
            plan,
            Some(plan.variants[0].variant_id.clone()),
            BTreeMap::new(),
            vec![RefitArtifactRecord {
                node_id: model_plan.node_id.clone(),
                controller_id: model_plan.controller_id.clone(),
                artifact: ArtifactRef {
                    id: ArtifactId::new("artifact:model:base:refit").unwrap(),
                    kind: "mock_model".to_string(),
                    controller_id: model_plan.controller_id.clone(),
                    backend: Some(ArtifactBackend::Joblib),
                    uri: Some(format!("artifacts/{content_fingerprint}.joblib")),
                    content_fingerprint: Some(content_fingerprint),
                    size_bytes: Some(128),
                    plugin: Some("dagml.mock".to_string()),
                    plugin_version: Some("1.0.0".to_string()),
                },
                params_fingerprint: model_plan.params_fingerprint.clone(),
                data_requirement_keys: vec!["model:base.x".to_string()],
                prediction_requirement_keys: Vec::new(),
            }],
        )
        .unwrap()
    }

    fn portable_artifact_bundle_with_payload(
        plan: &ExecutionPlan,
        payload: &[u8],
    ) -> crate::bundle::ExecutionBundle {
        let mut bundle = portable_artifact_bundle(plan);
        let content_fingerprint = sha256_bytes_hex(payload);
        let artifact = &mut bundle.refit_artifacts[0].artifact;
        artifact.uri = Some(format!("artifacts/{content_fingerprint}.joblib"));
        artifact.content_fingerprint = Some(content_fingerprint);
        artifact.size_bytes = Some(payload.len() as u64);
        bundle.validate().unwrap();
        bundle
    }

    fn write_artifact_payload(root: &Path, bundle: &ExecutionBundle, payload: &[u8]) -> PathBuf {
        let uri = bundle.refit_artifacts[0].artifact.uri.as_deref().unwrap();
        let path = root.join(uri);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, payload).unwrap();
        path
    }

    #[test]
    fn artifact_ref_validate_portable_rejects_unsafe_uris_and_legacy() {
        let content_fingerprint = "c".repeat(64);
        let base = ArtifactRef {
            id: ArtifactId::new("artifact:model:portable").unwrap(),
            kind: "model".to_string(),
            controller_id: ControllerId::new("controller:sklearn").unwrap(),
            backend: Some(ArtifactBackend::Joblib),
            uri: Some(format!("artifacts/{content_fingerprint}.joblib")),
            content_fingerprint: Some(content_fingerprint),
            size_bytes: Some(4096),
            plugin: Some("dagml.sklearn".to_string()),
            plugin_version: Some("1.0.0".to_string()),
        };
        base.validate_portable().unwrap();

        // Legacy artifact: still passes `validate` but is refused as non-portable.
        let legacy = ArtifactRef {
            backend: None,
            uri: None,
            content_fingerprint: None,
            ..base.clone()
        };
        legacy.validate().unwrap();
        assert!(legacy
            .validate_portable()
            .unwrap_err()
            .to_string()
            .contains("not portable"));

        let mut absolute = base.clone();
        absolute.uri = Some("/etc/passwd".to_string());
        assert!(absolute
            .validate_portable()
            .unwrap_err()
            .to_string()
            .contains("must be a relative path"));

        let mut traversal = base.clone();
        traversal.uri = Some("artifacts/../../secret.joblib".to_string());
        assert!(traversal
            .validate_portable()
            .unwrap_err()
            .to_string()
            .contains("`..`"));

        let mut drive = base.clone();
        drive.uri = Some("C:\\models\\model.joblib".to_string());
        assert!(drive
            .validate_portable()
            .unwrap_err()
            .to_string()
            .contains("must be a relative path"));

        // URI schemes and any other colon in the leading path segment are
        // rejected: a strictly relative artifact path never carries a scheme.
        for scheme_uri in [
            "http://example.com/model.joblib",
            "s3://bucket/model.joblib",
            "file:///models/model.joblib",
            "weird:thing/model.joblib",
        ] {
            let mut scheme = base.clone();
            scheme.uri = Some(scheme_uri.to_string());
            let err = scheme.validate_portable().unwrap_err().to_string();
            assert!(
                err.contains("first path segment"),
                "unexpected scheme error for `{scheme_uri}`: {err}"
            );
        }

        // A colon outside the first segment is allowed (not a scheme/drive).
        let mut later_colon = base;
        later_colon.uri = Some("artifacts/model:v1.joblib".to_string());
        later_colon.validate_portable().unwrap();
    }

    #[test]
    fn file_artifact_manifest_round_trips_portable_artifacts() {
        let plan = fixture_plan("plan:artifact.manifest.round.trip");
        let bundle = portable_artifact_bundle(&plan);
        let root = temp_prediction_cache_dir("dag_ml_file_artifact_manifest");

        let manifest = FileArtifactManifestStore::write(&root, &bundle).unwrap();
        assert_eq!(
            manifest.schema_version,
            FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(manifest.artifacts.len(), 1);
        assert_eq!(
            manifest.artifacts[0].artifact.id,
            ArtifactId::new("artifact:model:base:refit").unwrap()
        );
        assert_eq!(
            manifest.artifacts[0].artifact.backend,
            Some(ArtifactBackend::Joblib)
        );
        assert_eq!(
            manifest.artifacts[0].node_id,
            bundle.refit_artifacts[0].node_id
        );
        assert!(root.join(FILE_ARTIFACT_MANIFEST_FILE).exists());

        let store = FileArtifactManifestStore::open(root.clone(), &bundle).unwrap();
        assert_eq!(store.root(), root.as_path());
        assert_eq!(store.manifest().bundle_id, bundle.bundle_id);
        assert_eq!(store.manifest().artifacts, manifest.artifacts);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_artifact_manifest_refuses_legacy_non_portable_artifacts() {
        let plan = fixture_plan("plan:artifact.manifest.legacy");
        // `replay_bundle` carries a legacy artifact (no backend/uri/content fingerprint).
        let bundle = replay_bundle(&plan);
        let root = temp_prediction_cache_dir("dag_ml_file_artifact_manifest_legacy");

        let err = FileArtifactManifestStore::write(&root, &bundle).unwrap_err();
        assert!(
            err.to_string().contains("not portable"),
            "unexpected legacy error: {err}"
        );
        assert!(!root.join(FILE_ARTIFACT_MANIFEST_FILE).exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_artifact_manifest_open_refuses_tampered_entries() {
        let plan = fixture_plan("plan:artifact.manifest.tampered");
        let bundle = portable_artifact_bundle(&plan);
        let root = temp_prediction_cache_dir("dag_ml_file_artifact_manifest_tampered");

        FileArtifactManifestStore::write(&root, &bundle).unwrap();
        let manifest_path = root.join(FILE_ARTIFACT_MANIFEST_FILE);
        let mut tampered: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let tampered_fingerprint = "b".repeat(64);
        tampered["artifacts"][0]["params_fingerprint"] = json!(tampered_fingerprint);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&tampered).unwrap(),
        )
        .unwrap();

        let err = FileArtifactManifestStore::open(root.clone(), &bundle).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match bundle refit artifact"),
            "unexpected tamper error: {err}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_artifact_payload_store_validates_payloads_and_materializes_handles() {
        let plan = fixture_plan("plan:artifact.payload.round.trip");
        let payload = b"portable dag-ml artifact payload\n";
        let bundle = portable_artifact_bundle_with_payload(&plan, payload);
        let source_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_source");
        let store_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_store");
        let source_path = write_artifact_payload(&source_root, &bundle, payload);

        let store = FileArtifactPayloadStore::write_from_source(&store_root, &source_root, &bundle)
            .unwrap();
        assert_eq!(store.root(), store_root.as_path());
        assert_eq!(store.payload_count(), 1);
        assert!(store_root
            .join(bundle.refit_artifacts[0].artifact.uri.as_deref().unwrap())
            .exists());
        assert!(source_path.exists());
        assert_eq!(store.manifest().bundle_id, bundle.bundle_id);

        let artifact = &bundle.refit_artifacts[0];
        let handle = store
            .materialize(&ArtifactMaterializationRequest {
                run_id: RunId::new("run:artifact.payload.materialize").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                node_id: artifact.node_id.clone(),
                phase: Phase::Predict,
                variant_id: bundle.selected_variant_id.clone(),
                controller_id: artifact.controller_id.clone(),
                artifact: artifact.artifact.clone(),
                params_fingerprint: artifact.params_fingerprint.clone(),
            })
            .unwrap();
        assert_eq!(handle.kind, HandleKind::Artifact);
        assert_eq!(handle.owner_controller, artifact.controller_id);
        let records = store.materialization_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].artifact_id, artifact.artifact.id);
        assert_eq!(records[0].size_bytes, payload.len() as u64);
        assert_eq!(
            records[0].content_fingerprint,
            artifact.artifact.content_fingerprint.clone().unwrap()
        );

        let reopened = FileArtifactPayloadStore::open(store_root.clone(), &bundle).unwrap();
        reopened.validate_payloads().unwrap();

        let _ = fs::remove_dir_all(source_root);
        let _ = fs::remove_dir_all(store_root);
    }

    #[test]
    fn file_artifact_payload_store_refuses_tampered_payloads() {
        let plan = fixture_plan("plan:artifact.payload.tampered");
        let payload = b"portable dag-ml artifact payload\n";
        let bundle = portable_artifact_bundle_with_payload(&plan, payload);
        let source_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_source_tamper");
        let store_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_store_tamper");
        write_artifact_payload(&source_root, &bundle, payload);
        FileArtifactPayloadStore::write_from_source(&store_root, &source_root, &bundle).unwrap();

        let payload_path =
            store_root.join(bundle.refit_artifacts[0].artifact.uri.as_deref().unwrap());
        fs::write(&payload_path, vec![b'x'; payload.len()]).unwrap();
        let err = FileArtifactPayloadStore::open(store_root.clone(), &bundle).unwrap_err();
        assert!(
            err.to_string().contains("content fingerprint mismatch"),
            "unexpected tamper error: {err}"
        );

        let _ = fs::remove_dir_all(source_root);
        let _ = fs::remove_dir_all(store_root);
    }

    #[test]
    fn requires_oof_prediction_edge_refit_rejects_incomplete_oof_coverage() {
        let plan = build_execution_plan(
            "plan:oof.edge.refit.incomplete",
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let mut ctx = RunContext::new(
            RunId::new("run:oof.edge.refit.incomplete").unwrap(),
            Some(11),
        );
        ctx.prediction_store
            .append(PredictionBlock {
                prediction_id: Some("pred:model:base:fold0".to_string()),
                producer_node: NodeId::new("model:base").unwrap(),
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                sample_ids: vec![SampleId::new("s1").unwrap()],
                values: vec![vec![0.5]],
                target_names: vec!["y".to_string()],
            })
            .unwrap();
        let controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);

        let error = SequentialScheduler
            .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::Refit)
            .unwrap_err()
            .to_string();

        assert!(error.contains("do not cover the refit sample universe"));
    }

    #[test]
    fn data_bindings_require_runtime_provider_and_materialize_handles() {
        let model_id = NodeId::new("model:pls").unwrap();
        let plan = build_execution_plan(
            "plan:data",
            simple_graph(),
            CampaignSpec {
                id: "campaign:data".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::from([(model_id.clone(), vec![data_binding(&model_id)])]),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let controllers = runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:data").unwrap(), Some(11));

        assert!(SequentialScheduler
            .execute_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
            .is_err());

        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        let provider = InMemoryDataProvider::with_envelope(
            ControllerId::new("controller:data.provider").unwrap(),
            envelope,
        )
        .unwrap();
        let mut ctx = RunContext::new(RunId::new("run:data.provider").unwrap(), Some(11));
        let results = SequentialScheduler
            .execute_phase_with_data_provider(
                &plan,
                &controllers,
                &provider,
                &mut ctx,
                Phase::FitCv,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(provider.handle_records().len(), 1);
        assert_eq!(provider.view_records().len(), 1);
        assert_eq!(provider.handle_records()[0].input_name, "x");
        assert_eq!(provider.handle_records()[0].relation_record_count, Some(4));
        assert_eq!(provider.view_records()[0].handle.kind, HandleKind::DataView);
        assert_eq!(
            provider.view_records()[0].parent_handle,
            provider.handle_records()[0].handle
        );
    }

    #[test]
    fn campaign_data_bindings_create_fold_train_views() {
        let model_id = NodeId::new("model:pls").unwrap();
        let plan = build_execution_plan(
            "plan:data.folds",
            simple_graph(),
            CampaignSpec {
                id: "campaign:data.folds".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: Some(SplitInvocation {
                    id: "split:outer".to_string(),
                    controller_id: None,
                    leakage_policy: Default::default(),
                    params: BTreeMap::new(),
                    fold_set: Some(two_fold_set()),
                }),
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::from([(
                    model_id,
                    vec![data_binding(&NodeId::new("model:pls").unwrap())],
                )]),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        let provider = InMemoryDataProvider::with_envelope(
            ControllerId::new("controller:data.provider").unwrap(),
            envelope,
        )
        .unwrap();
        let controllers = runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:data.folds").unwrap(), Some(11));

        let results = SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &controllers,
                &provider,
                &mut ctx,
                Phase::FitCv,
            )
            .unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(provider.handle_records().len(), 2);
        let views = provider.view_records();
        assert_eq!(views.len(), 4);
        assert!(views
            .iter()
            .all(|view| view.handle.kind == HandleKind::DataView));
        let train_views = views
            .iter()
            .filter(|view| view.view.partition == DataRequestPartition::FoldTrain)
            .collect::<Vec<_>>();
        let validation_views = views
            .iter()
            .filter(|view| view.view.partition == DataRequestPartition::FoldValidation)
            .collect::<Vec<_>>();
        assert_eq!(train_views.len(), 2);
        assert_eq!(validation_views.len(), 2);
        assert_eq!(
            train_views[0].view.sample_ids,
            Some(vec![SampleId::new("s2").unwrap()])
        );
        assert_eq!(
            validation_views[0].view.sample_ids,
            Some(vec![SampleId::new("s1").unwrap()])
        );
        assert_eq!(
            train_views[1].view.sample_ids,
            Some(vec![SampleId::new("s1").unwrap()])
        );
        assert_eq!(
            validation_views[1].view.sample_ids,
            Some(vec![SampleId::new("s2").unwrap()])
        );
    }

    #[test]
    fn data_edges_propagate_fold_views_from_data_producing_nodes() {
        let augment_id = NodeId::new("augment:noise").unwrap();
        let model_id = NodeId::new("model:branch").unwrap();
        let before_feature_schema = "a".repeat(64);
        let after_feature_schema = "b".repeat(64);
        let shape_plan = DataModelShapePlan {
            node_id: augment_id.clone(),
            input_granularity: Granularity::Sample,
            target_granularity: Granularity::Sample,
            fit_rows: FitBoundary::FoldTrain,
            predict_rows: FitBoundary::FoldValidation,
            feature_namespace: Some("augmented.noise".to_string()),
            feature_schema_fingerprint: Some(before_feature_schema.clone()),
            target_space: "raw".to_string(),
            aggregation_policy: AggregationPolicy::default(),
            augmentation_policy: Default::default(),
            selection_policy: Default::default(),
        };
        let shape_plan_fingerprint = stable_json_fingerprint(&shape_plan).unwrap();
        let graph = GraphSpec {
            id: "g:data.edge.views".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node(
                    augment_id.as_str(),
                    NodeKind::Augmentation,
                    vec![port("x", PortKind::Data)],
                    vec![port("x_out", PortKind::Data)],
                ),
                node(
                    model_id.as_str(),
                    NodeKind::Model,
                    vec![port("x", PortKind::Data)],
                    vec![port("oof", PortKind::Prediction)],
                ),
            ],
            edges: vec![EdgeSpec {
                source: PortRef {
                    node_id: augment_id.clone(),
                    port_name: "x_out".to_string(),
                },
                target: PortRef {
                    node_id: model_id.clone(),
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
        };
        let mut manifest_registry = ControllerRegistry::new();
        manifest_registry
            .register(controller_manifest(
                "controller:augmentation",
                NodeKind::Augmentation,
            ))
            .unwrap();
        manifest_registry
            .register(controller_manifest(
                "controller:model.probe",
                NodeKind::Model,
            ))
            .unwrap();
        let plan = build_execution_plan(
            "plan:data.edge.views",
            graph,
            CampaignSpec {
                id: "campaign:data.edge.views".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: Some(SplitInvocation {
                    id: "split:outer".to_string(),
                    controller_id: None,
                    leakage_policy: Default::default(),
                    params: BTreeMap::new(),
                    fold_set: Some(two_fold_set()),
                }),
                generation: Default::default(),
                shape_plans: BTreeMap::from([(augment_id.clone(), shape_plan)]),
                data_bindings: BTreeMap::from([(
                    augment_id.clone(),
                    vec![data_binding(&augment_id)],
                )]),
                metadata: BTreeMap::new(),
            },
            &manifest_registry,
        )
        .unwrap();
        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        let provider = InMemoryDataProvider::with_envelope(
            ControllerId::new("controller:data.provider").unwrap(),
            envelope,
        )
        .unwrap();
        let observed_views = Arc::new(Mutex::new(Vec::new()));
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ShapeDataController {
                id: ControllerId::new("controller:augmentation").unwrap(),
                handle: 3,
                before_feature_schema: before_feature_schema.clone(),
                after_feature_schema: after_feature_schema.clone(),
            }))
            .unwrap();
        controllers
            .register(Box::new(DataViewProbeController {
                id: ControllerId::new("controller:model.probe").unwrap(),
                observed_views: observed_views.clone(),
                prediction_sample_ids: None,
            }))
            .unwrap();
        let mut ctx = RunContext::new(RunId::new("run:data.edge.views").unwrap(), Some(11));

        let results = SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &controllers,
                &provider,
                &mut ctx,
                Phase::FitCv,
            )
            .unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(provider.view_records().len(), 4);
        let observed_views = observed_views.lock().unwrap();
        assert_eq!(observed_views.len(), 2);
        for views in observed_views.iter() {
            let primary = views.get("data:x").expect("primary propagated data view");
            let validation = views
                .get("data:x:validation")
                .expect("validation propagated data view");
            for view in [primary, validation] {
                let provenance = view
                    .output_provenance()
                    .unwrap()
                    .expect("output data provenance metadata");
                assert_eq!(
                    provenance.producer_node,
                    NodeId::new("augment:noise").unwrap()
                );
                assert_eq!(provenance.producer_port, "x_out");
                assert_eq!(
                    provenance.shape_plan_fingerprint,
                    Some(shape_plan_fingerprint.clone())
                );
                assert_eq!(
                    provenance.feature_schema_fingerprint,
                    Some(after_feature_schema.clone())
                );
                assert_eq!(provenance.shape_deltas.len(), 1);
            }
        }
        let samples_by_fold = ctx
            .prediction_store
            .blocks()
            .iter()
            .filter(|block| block.producer_node == model_id)
            .map(|block| {
                (
                    block.fold_id.as_ref().unwrap().to_string(),
                    block.sample_ids.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            samples_by_fold["fold:0"],
            vec![SampleId::new("s1").unwrap()]
        );
        assert_eq!(
            samples_by_fold["fold:1"],
            vec![SampleId::new("s2").unwrap()]
        );

        let mut bad_controllers = RuntimeControllerRegistry::new();
        bad_controllers
            .register(Box::new(ShapeDataController {
                id: ControllerId::new("controller:augmentation").unwrap(),
                handle: 5,
                before_feature_schema,
                after_feature_schema,
            }))
            .unwrap();
        bad_controllers
            .register(Box::new(DataViewProbeController {
                id: ControllerId::new("controller:model.probe").unwrap(),
                observed_views: Arc::new(Mutex::new(Vec::new())),
                prediction_sample_ids: Some(vec![SampleId::new("s-outside").unwrap()]),
            }))
            .unwrap();
        let mut bad_ctx = RunContext::new(
            RunId::new("run:data.edge.views.bad-prediction").unwrap(),
            Some(11),
        );
        let error = SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &bad_controllers,
                &provider,
                &mut bad_ctx,
                Phase::FitCv,
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("outside its validation view"),
            "unexpected propagated-view validation error: {error}"
        );
    }

    #[test]
    fn data_provider_view_validates_typed_output_provenance() {
        let producer = NodeId::new("augment:noise").unwrap();
        let before_feature_schema = "a".repeat(64);
        let after_feature_schema = "b".repeat(64);
        let provenance = DataOutputProvenance {
            schema_version: DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION,
            producer_node: producer.clone(),
            producer_port: "x_out".to_string(),
            producer_phase: Phase::FitCv,
            variant_id: Some(VariantId::new("variant:base").unwrap()),
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            shape_plan_fingerprint: Some("c".repeat(64)),
            aggregation_policy_fingerprint: Some("d".repeat(64)),
            feature_namespace: Some("augmented.noise".to_string()),
            feature_schema_fingerprint: Some(after_feature_schema.clone()),
            shape_deltas: vec![ShapeDelta {
                node_id: producer.clone(),
                kind: ShapeDeltaKind::Feature,
                before_fingerprint: before_feature_schema,
                after_fingerprint: after_feature_schema,
                metadata: BTreeMap::new(),
            }],
        };
        let mut view = DataProviderViewSpec {
            sample_ids: Some(vec![SampleId::new("s1").unwrap()]),
            partition: DataRequestPartition::FoldTrain,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            source_ids: None,
            columns: None,
            include_augmented: true,
            include_excluded: false,
            extra: BTreeMap::from([(
                DATA_OUTPUT_PROVENANCE_KEY.to_string(),
                serde_json::to_value(&provenance).unwrap(),
            )]),
        };

        assert_eq!(view.output_provenance().unwrap(), Some(provenance.clone()));
        view.validate().unwrap();

        let mut empty_port = provenance.clone();
        empty_port.producer_port.clear();
        view.extra.insert(
            DATA_OUTPUT_PROVENANCE_KEY.to_string(),
            serde_json::to_value(empty_port).unwrap(),
        );
        let error = view.validate().unwrap_err().to_string();
        assert!(
            error.contains("empty producer_port"),
            "unexpected empty-port provenance error: {error}"
        );

        let mut wrong_delta_node = provenance.clone();
        wrong_delta_node.shape_deltas[0].node_id = NodeId::new("augment:other").unwrap();
        view.extra.insert(
            DATA_OUTPUT_PROVENANCE_KEY.to_string(),
            serde_json::to_value(wrong_delta_node).unwrap(),
        );
        let error = view.validate().unwrap_err().to_string();
        assert!(
            error.contains("contains shape delta"),
            "unexpected wrong-delta-node provenance error: {error}"
        );

        let mut wrong_feature_fingerprint = provenance.clone();
        wrong_feature_fingerprint.feature_schema_fingerprint = Some("e".repeat(64));
        view.extra.insert(
            DATA_OUTPUT_PROVENANCE_KEY.to_string(),
            serde_json::to_value(wrong_feature_fingerprint).unwrap(),
        );
        let error = view.validate().unwrap_err().to_string();
        assert!(
            error.contains("last feature delta"),
            "unexpected feature-fingerprint provenance error: {error}"
        );

        let mut unsupported_schema = provenance;
        unsupported_schema.schema_version = DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION + 1;
        view.extra.insert(
            DATA_OUTPUT_PROVENANCE_KEY.to_string(),
            serde_json::to_value(unsupported_schema).unwrap(),
        );
        let error = view.validate().unwrap_err().to_string();
        assert!(
            error.contains("unsupported schema_version"),
            "unexpected provenance schema-version error: {error}"
        );
    }

    #[test]
    fn published_data_output_provenance_schema_declares_current_version() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/data_output_provenance.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["properties"]["schema_version"]["const"].as_u64(),
            Some(u64::from(DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION))
        );
        assert_eq!(schema["$id"], DATA_OUTPUT_PROVENANCE_SCHEMA_ID);
        let required = schema["required"].as_array().unwrap();
        assert!(required
            .iter()
            .any(|field| field.as_str() == Some("schema_version")));
        assert!(required
            .iter()
            .any(|field| field.as_str() == Some("producer_node")));
    }

    #[test]
    fn published_node_task_and_result_schemas_declare_current_contracts() {
        let task_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/node_task.schema.json"
        ))
        .unwrap();
        let result_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/node_result.schema.json"
        ))
        .unwrap();

        assert_eq!(task_schema["$id"], NODE_TASK_SCHEMA_ID);
        assert_eq!(result_schema["$id"], NODE_RESULT_SCHEMA_ID);
        assert!(task_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("node_plan")));
        assert!(result_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("lineage")));
    }

    #[test]
    fn published_node_task_result_fixtures_validate_current_contract() {
        let task: NodeTask = serde_json::from_str(include_str!(
            "../../../examples/fixtures/runtime/node_task_transform_scale.json"
        ))
        .unwrap();
        let result: NodeResult = serde_json::from_str(include_str!(
            "../../../examples/fixtures/runtime/node_result_transform_scale.json"
        ))
        .unwrap();

        result.validate_for_task(&task).unwrap();
        assert_eq!(
            task.node_plan.node_id,
            NodeId::new("transform:scale").unwrap()
        );
        assert_eq!(result.outputs.len(), 1);
    }

    #[test]
    fn campaign_data_bindings_require_unsafe_flags_for_full_train_cv_views() {
        let model_id = NodeId::new("model:pls").unwrap();
        let mut unsafe_binding = data_binding(&model_id);
        unsafe_binding.view_policy.fit_partition = DataRequestPartition::FullTrain;
        unsafe_binding.view_policy.unsafe_flags =
            BTreeSet::from([DataViewPolicy::ALLOW_FIT_CV_FULL_TRAIN_VIEW.to_string()]);

        let mut unsafe_campaign = oof_edge_campaign();
        unsafe_campaign.data_bindings =
            BTreeMap::from([(model_id.clone(), vec![unsafe_binding.clone()])]);
        let plan = build_execution_plan(
            "plan:data.full-train.unsafe",
            simple_graph(),
            unsafe_campaign,
            &manifests(),
        )
        .unwrap();

        let mut missing_flag = unsafe_binding;
        missing_flag.view_policy.unsafe_flags.clear();
        let mut invalid_campaign = oof_edge_campaign();
        invalid_campaign.data_bindings = BTreeMap::from([(model_id.clone(), vec![missing_flag])]);
        assert!(build_execution_plan(
            "plan:data.full-train.missing-flag",
            simple_graph(),
            invalid_campaign,
            &manifests(),
        )
        .is_err());

        let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        let provider = InMemoryDataProvider::with_envelope(
            ControllerId::new("controller:data.provider").unwrap(),
            envelope,
        )
        .unwrap();
        let controllers = runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:data.full-train.unsafe").unwrap(), Some(11));

        SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &controllers,
                &provider,
                &mut ctx,
                Phase::FitCv,
            )
            .unwrap();

        let full_train_ids = plan.fold_set.as_ref().unwrap().sample_ids.clone();
        let views = provider.view_records();
        let full_train_views = views
            .iter()
            .filter(|view| view.view.partition == DataRequestPartition::FullTrain)
            .collect::<Vec<_>>();
        let validation_views = views
            .iter()
            .filter(|view| view.view.partition == DataRequestPartition::FoldValidation)
            .collect::<Vec<_>>();
        assert_eq!(full_train_views.len(), 2);
        assert_eq!(validation_views.len(), 2);
        assert!(full_train_views.iter().all(|view| {
            view.view.sample_ids == Some(full_train_ids.clone())
                && view.view.fold_id.is_none()
                && view.view.extra["unsafe_flags"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|flag| flag.as_str() == Some(DataViewPolicy::ALLOW_FIT_CV_FULL_TRAIN_VIEW))
        }));
        assert!(validation_views
            .iter()
            .all(|view| !view.view.include_augmented));
    }

    #[test]
    fn campaign_refit_data_bindings_create_full_train_views() {
        let plan = fixture_plan("plan:refit.views");
        let provider = replay_data_provider();
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:transform.mock").unwrap(),
                handle: 11,
                require_artifact: false,
                emit_prediction: false,
                emit_refit_artifact: false,
            }))
            .unwrap();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:model.mock").unwrap(),
                handle: 22,
                require_artifact: false,
                emit_prediction: true,
                emit_refit_artifact: false,
            }))
            .unwrap();
        let mut ctx = RunContext::new(RunId::new("run:refit.views").unwrap(), Some(11));
        ctx.variant_id = Some(plan.variants[0].variant_id.clone());

        let results = SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &controllers,
                &provider,
                &mut ctx,
                Phase::Refit,
            )
            .unwrap();

        assert!(!results.is_empty());
        let views = provider.view_records();
        assert_eq!(views.len(), 1);
        let full_train_ids = plan.fold_set.as_ref().unwrap().sample_ids.clone();
        assert!(views.iter().all(|view| {
            view.view.partition == DataRequestPartition::FullTrain
                && view.view.sample_ids == Some(full_train_ids.clone())
                && view.fold_id.is_none()
        }));
    }

    #[test]
    fn campaign_refit_captures_emitted_artifact_handles() {
        let plan = fixture_plan("plan:refit.artifact.capture");
        let provider = replay_data_provider();
        let mut artifact_store = InMemoryArtifactStore::new();
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:transform.mock").unwrap(),
                handle: 11,
                require_artifact: false,
                emit_prediction: false,
                emit_refit_artifact: false,
            }))
            .unwrap();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:model.mock").unwrap(),
                handle: 22,
                require_artifact: false,
                emit_prediction: true,
                emit_refit_artifact: true,
            }))
            .unwrap();
        let mut ctx = RunContext::new(RunId::new("run:refit.artifact.capture").unwrap(), Some(11));
        ctx.variant_id = Some(plan.variants[0].variant_id.clone());

        let results = SequentialScheduler
            .execute_campaign_phase_with_data_provider_and_artifact_store(
                &plan,
                &controllers,
                &provider,
                &mut artifact_store,
                &mut ctx,
                Phase::Refit,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results
                .iter()
                .filter(|result| !result.artifacts.is_empty())
                .count(),
            1
        );
        assert_eq!(artifact_store.len(), 1);
        let records = artifact_store.refit_artifacts();
        assert_eq!(records.len(), 1);
        let artifact = &records[0];
        artifact.validate().unwrap();
        assert_eq!(artifact.node_id.as_str(), "model:base");
        assert_eq!(artifact.controller_id.as_str(), "controller:model.mock");
        assert_eq!(artifact.artifact.id.as_str(), "artifact:model:base:refit");
        assert_eq!(artifact.data_requirement_keys, vec!["model:base.x"]);

        let handle = artifact_store
            .materialize(&ArtifactMaterializationRequest {
                run_id: ctx.run_id.clone(),
                bundle_id: crate::ids::BundleId::new("bundle:refit.capture").unwrap(),
                node_id: artifact.node_id.clone(),
                phase: Phase::Predict,
                variant_id: ctx.variant_id.clone(),
                controller_id: artifact.controller_id.clone(),
                artifact: artifact.artifact.clone(),
                params_fingerprint: artifact.params_fingerprint.clone(),
            })
            .unwrap();
        assert_eq!(
            handle,
            HandleRef {
                handle: 10_022,
                kind: HandleKind::Model,
                owner_controller: ControllerId::new("controller:model.mock").unwrap(),
            }
        );
    }

    #[test]
    fn parallel_campaign_refit_captures_emitted_artifact_handles() {
        let plan = fixture_plan("plan:parallel.refit.artifact.capture");
        let provider = replay_data_provider();
        let mut artifact_store = InMemoryArtifactStore::new();
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:transform.mock").unwrap(),
                handle: 11,
                require_artifact: false,
                emit_prediction: false,
                emit_refit_artifact: false,
            }))
            .unwrap();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:model.mock").unwrap(),
                handle: 22,
                require_artifact: false,
                emit_prediction: true,
                emit_refit_artifact: true,
            }))
            .unwrap();
        let mut ctx = RunContext::new(
            RunId::new("run:parallel.refit.artifact.capture").unwrap(),
            Some(11),
        );
        ctx.variant_id = Some(plan.variants[0].variant_id.clone());

        let results = ParallelScheduler::new(2)
            .unwrap()
            .execute_campaign_phase_with_data_provider_and_artifact_store(
                &plan,
                &controllers,
                &provider,
                &mut artifact_store,
                &mut ctx,
                Phase::Refit,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(artifact_store.len(), 1);
        assert_eq!(
            artifact_store.refit_artifacts()[0].artifact.id.as_str(),
            "artifact:model:base:refit"
        );
    }

    #[test]
    fn node_result_validation_rejects_external_conformance_mismatches() {
        let plan = build_execution_plan(
            "plan:result.validation",
            simple_graph(),
            CampaignSpec {
                id: "campaign:result.validation".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let node_plan = plan
            .node_plans
            .get(&NodeId::new("model:pls").unwrap())
            .unwrap()
            .clone();
        let task = NodeTask {
            run_id: RunId::new("run:result.validation").unwrap(),
            node_plan: node_plan.clone(),
            phase: Phase::FitCv,
            variant_id: None,
            variant: None,
            fold_id: None,
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
            prediction_inputs: BTreeMap::new(),
            artifact_inputs: BTreeMap::new(),
            seed: Some(99),
        };
        let controller = MockController {
            id: node_plan.controller_id.clone(),
            handle: 2,
            emit_prediction: false,
        };
        let result = controller.invoke(&task).unwrap();
        result.validate_for_task(&task).unwrap();

        let mut bad_controller = result.clone();
        bad_controller.lineage.controller_id = ControllerId::new("controller:wrong").unwrap();
        assert!(bad_controller
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("controller"));

        let mut bad_params = result.clone();
        bad_params.lineage.params_fingerprint = "wrong".to_string();
        assert!(bad_params
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("params fingerprint"));

        let mut bad_output_owner = result.clone();
        bad_output_owner
            .outputs
            .get_mut("out")
            .unwrap()
            .owner_controller = ControllerId::new("controller:wrong").unwrap();
        assert!(bad_output_owner
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("output `out`"));
    }

    #[test]
    fn node_result_validation_checks_shape_fingerprints_and_feature_deltas() {
        let model_id = NodeId::new("model:pls").unwrap();
        let initial_feature_schema = "a".repeat(64);
        let updated_feature_schema = "b".repeat(64);
        let shape_plan = DataModelShapePlan {
            node_id: model_id.clone(),
            input_granularity: Granularity::Sample,
            target_granularity: Granularity::Sample,
            fit_rows: FitBoundary::FoldTrain,
            predict_rows: FitBoundary::FoldValidation,
            feature_namespace: Some("raw.x".to_string()),
            feature_schema_fingerprint: Some(initial_feature_schema.clone()),
            target_space: "raw".to_string(),
            aggregation_policy: AggregationPolicy::default(),
            augmentation_policy: Default::default(),
            selection_policy: Default::default(),
        };
        let plan = build_execution_plan(
            "plan:result.validation.shape",
            simple_graph(),
            CampaignSpec {
                id: "campaign:result.validation.shape".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::from([(model_id.clone(), shape_plan.clone())]),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let node_plan = plan.node_plans.get(&model_id).unwrap().clone();
        let task = NodeTask {
            run_id: RunId::new("run:result.validation.shape").unwrap(),
            node_plan: node_plan.clone(),
            phase: Phase::FitCv,
            variant_id: None,
            variant: None,
            fold_id: None,
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
            prediction_inputs: BTreeMap::new(),
            artifact_inputs: BTreeMap::new(),
            seed: Some(99),
        };
        let controller = MockController {
            id: node_plan.controller_id.clone(),
            handle: 2,
            emit_prediction: false,
        };
        let mut result = controller.invoke(&task).unwrap();
        result.lineage.data_model_shape_fingerprint =
            Some(stable_json_fingerprint(&shape_plan).unwrap());
        result.lineage.aggregation_policy_fingerprint =
            Some(stable_json_fingerprint(&shape_plan.aggregation_policy).unwrap());
        result.shape_deltas = vec![ShapeDelta {
            node_id: model_id.clone(),
            kind: ShapeDeltaKind::Feature,
            before_fingerprint: initial_feature_schema.clone(),
            after_fingerprint: updated_feature_schema.clone(),
            metadata: BTreeMap::from([(
                "feature_namespace".to_string(),
                serde_json::Value::String("selected.x".to_string()),
            )]),
        }];
        result.validate_for_task(&task).unwrap();

        let mut wrong_shape_fingerprint = result.clone();
        wrong_shape_fingerprint.lineage.data_model_shape_fingerprint = Some("0".repeat(64));
        assert!(wrong_shape_fingerprint
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("data/model shape fingerprint"));

        let mut wrong_feature_delta = result.clone();
        wrong_feature_delta.shape_deltas[0].before_fingerprint = "c".repeat(64);
        assert!(wrong_feature_delta
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("expected current schema"));

        let mut unchanged_delta = result;
        unchanged_delta.shape_deltas[0].after_fingerprint = initial_feature_schema;
        assert!(unchanged_delta
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("does not change fingerprint"));
    }

    #[test]
    fn node_result_validation_rejects_bad_artifact_handles() {
        let plan = build_execution_plan(
            "plan:result.validation.artifacts",
            simple_graph(),
            CampaignSpec {
                id: "campaign:result.validation.artifacts".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: None,
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let node_plan = plan
            .node_plans
            .get(&NodeId::new("model:pls").unwrap())
            .unwrap()
            .clone();
        let task = NodeTask {
            run_id: RunId::new("run:result.validation.artifacts").unwrap(),
            node_plan: node_plan.clone(),
            phase: Phase::Refit,
            variant_id: None,
            variant: None,
            fold_id: None,
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
            prediction_inputs: BTreeMap::new(),
            artifact_inputs: BTreeMap::new(),
            seed: Some(99),
        };
        let controller = MockController {
            id: node_plan.controller_id.clone(),
            handle: 2,
            emit_prediction: false,
        };
        let base = controller.invoke(&task).unwrap();
        let artifact = ArtifactRef {
            id: ArtifactId::new("artifact:model:pls:refit").unwrap(),
            kind: "mock_model".to_string(),
            controller_id: node_plan.controller_id.clone(),
            backend: None,
            uri: None,
            content_fingerprint: None,
            size_bytes: Some(128),
            plugin: None,
            plugin_version: None,
        };
        let handle = HandleRef {
            handle: 77,
            kind: HandleKind::Model,
            owner_controller: node_plan.controller_id.clone(),
        };
        let mut valid = base.clone();
        valid.artifacts = vec![artifact.clone()];
        valid
            .artifact_handles
            .insert(artifact.id.clone(), handle.clone());
        valid.lineage.artifact_refs = vec![artifact.clone()];
        valid.validate_for_task(&task).unwrap();

        let mut missing_handle = valid.clone();
        missing_handle.artifact_handles.clear();
        assert!(missing_handle
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("without artifact handle"));

        let mut wrong_kind = valid.clone();
        wrong_kind
            .artifact_handles
            .get_mut(&artifact.id)
            .unwrap()
            .kind = HandleKind::Data;
        assert!(wrong_kind
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("non-artifact/model handle kind"));

        let mut wrong_owner = valid.clone();
        wrong_owner
            .artifact_handles
            .get_mut(&artifact.id)
            .unwrap()
            .owner_controller = ControllerId::new("controller:wrong").unwrap();
        assert!(wrong_owner
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("owned by"));

        let mut undeclared_handle = base.clone();
        undeclared_handle.artifact_handles.insert(
            ArtifactId::new("artifact:model:pls:extra").unwrap(),
            handle.clone(),
        );
        assert!(undeclared_handle
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("undeclared artifact"));

        let mut missing_lineage_ref = valid;
        missing_lineage_ref.lineage.artifact_refs.clear();
        assert!(missing_lineage_ref
            .validate_for_task(&task)
            .unwrap_err()
            .to_string()
            .contains("lineage artifact ref"));
    }

    #[test]
    fn artifact_ref_validates_portable_metadata() {
        let content_fingerprint = "a".repeat(64);
        let artifact = ArtifactRef {
            id: ArtifactId::new("artifact:model:portable").unwrap(),
            kind: "model".to_string(),
            controller_id: ControllerId::new("controller:sklearn").unwrap(),
            backend: Some(ArtifactBackend::Joblib),
            uri: Some(format!("artifacts/{content_fingerprint}.joblib")),
            content_fingerprint: Some(content_fingerprint.clone()),
            size_bytes: Some(4096),
            plugin: Some("dagml.sklearn".to_string()),
            plugin_version: Some("1.0.0".to_string()),
        };

        artifact.validate().unwrap();
        let encoded = serde_json::to_value(&artifact).unwrap();
        assert_eq!(encoded["backend"].as_str(), Some("joblib"));
        assert_eq!(
            encoded["content_fingerprint"].as_str(),
            Some(content_fingerprint.as_str())
        );

        let legacy: ArtifactRef = serde_json::from_value(serde_json::json!({
            "id": "artifact:model:legacy",
            "kind": "mock_model",
            "controller_id": "controller:mock",
            "size_bytes": 128
        }))
        .unwrap();
        assert_eq!(legacy.backend, None);
        assert_eq!(legacy.content_fingerprint, None);
        legacy.validate().unwrap();
    }

    #[test]
    fn artifact_ref_rejects_invalid_portable_metadata() {
        let mut artifact = ArtifactRef {
            id: ArtifactId::new("artifact:model:portable").unwrap(),
            kind: "model".to_string(),
            controller_id: ControllerId::new("controller:sklearn").unwrap(),
            backend: Some(ArtifactBackend::Joblib),
            uri: Some("artifacts/model.joblib".to_string()),
            content_fingerprint: Some("b".repeat(64)),
            size_bytes: Some(4096),
            plugin: Some("dagml.sklearn".to_string()),
            plugin_version: Some("1.0.0".to_string()),
        };
        artifact.validate().unwrap();

        let mut bad_fingerprint = artifact.clone();
        bad_fingerprint.content_fingerprint = Some("not-a-digest".to_string());
        assert!(bad_fingerprint
            .validate()
            .unwrap_err()
            .to_string()
            .contains("artifact content fingerprint"));

        let mut missing_backend = artifact.clone();
        missing_backend.backend = None;
        assert!(missing_backend
            .validate()
            .unwrap_err()
            .to_string()
            .contains("uri without backend"));

        let mut missing_fingerprint = artifact.clone();
        missing_fingerprint.content_fingerprint = None;
        assert!(missing_fingerprint
            .validate()
            .unwrap_err()
            .to_string()
            .contains("uri without content_fingerprint"));

        artifact.plugin = None;
        assert!(artifact
            .validate()
            .unwrap_err()
            .to_string()
            .contains("plugin_version without plugin"));
    }

    #[test]
    fn node_result_validation_rejects_predictions_outside_validation_view() {
        let model_id = NodeId::new("model:pls").unwrap();
        let plan = build_execution_plan(
            "plan:result.validation.samples",
            simple_graph(),
            CampaignSpec {
                id: "campaign:result.validation.samples".to_string(),
                root_seed: Some(11),
                leakage_policy: Default::default(),
                aggregation_policy: Default::default(),
                split_invocation: Some(SplitInvocation {
                    id: "split:outer".to_string(),
                    controller_id: None,
                    leakage_policy: Default::default(),
                    params: BTreeMap::new(),
                    fold_set: Some(two_fold_set()),
                }),
                generation: Default::default(),
                shape_plans: BTreeMap::new(),
                data_bindings: BTreeMap::from([(model_id.clone(), vec![data_binding(&model_id)])]),
                metadata: BTreeMap::new(),
            },
            &manifests(),
        )
        .unwrap();
        let node_plan = plan.node_plans.get(&model_id).unwrap().clone();
        let task = NodeTask {
            run_id: RunId::new("run:result.validation.samples").unwrap(),
            node_plan: node_plan.clone(),
            phase: Phase::FitCv,
            variant_id: Some(VariantId::new("variant:base").unwrap()),
            variant: None,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::from([(
                "data:x:validation".to_string(),
                DataProviderViewSpec {
                    sample_ids: Some(vec![SampleId::new("s1").unwrap()]),
                    partition: DataRequestPartition::FoldValidation,
                    fold_id: Some(FoldId::new("fold:0").unwrap()),
                    source_ids: None,
                    columns: None,
                    include_augmented: false,
                    include_excluded: false,
                    extra: BTreeMap::new(),
                },
            )]),
            prediction_inputs: BTreeMap::new(),
            artifact_inputs: BTreeMap::new(),
            seed: Some(99),
        };
        let result = NodeResult {
            node_id: model_id.clone(),
            outputs: BTreeMap::from([(
                "out".to_string(),
                HandleRef {
                    handle: 7,
                    kind: HandleKind::Data,
                    owner_controller: node_plan.controller_id.clone(),
                },
            )]),
            predictions: vec![PredictionBlock {
                prediction_id: Some("pred:bad.sample".to_string()),
                producer_node: model_id,
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                sample_ids: vec![SampleId::new("s2").unwrap()],
                values: vec![vec![1.0]],
                target_names: vec!["y".to_string()],
            }],
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new("lineage:bad.sample").unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: task.node_plan.controller_id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        };

        assert!(result.validate_for_task(&task).is_err());
    }

    #[test]
    fn in_memory_artifact_store_resolves_bundle_artifacts() {
        let plan = fixture_plan("plan:replay.artifacts");
        let bundle = replay_bundle(&plan);
        let artifact = &bundle.refit_artifacts[0];
        let mut store = InMemoryArtifactStore::new();
        let handle = HandleRef {
            handle: 77,
            kind: HandleKind::Model,
            owner_controller: artifact.controller_id.clone(),
        };
        store.register(artifact, handle.clone()).unwrap();

        let resolved = store
            .materialize(&ArtifactMaterializationRequest {
                run_id: RunId::new("run:replay.artifacts").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                node_id: artifact.node_id.clone(),
                phase: Phase::Predict,
                variant_id: bundle.selected_variant_id.clone(),
                controller_id: artifact.controller_id.clone(),
                artifact: artifact.artifact.clone(),
                params_fingerprint: artifact.params_fingerprint.clone(),
            })
            .unwrap();

        assert_eq!(resolved, handle);
        assert_eq!(store.len(), 1);
        assert!(InMemoryArtifactStore::new()
            .materialize(&ArtifactMaterializationRequest {
                run_id: RunId::new("run:replay.artifacts").unwrap(),
                bundle_id: bundle.bundle_id.clone(),
                node_id: artifact.node_id.clone(),
                phase: Phase::Predict,
                variant_id: bundle.selected_variant_id.clone(),
                controller_id: artifact.controller_id.clone(),
                artifact: artifact.artifact.clone(),
                params_fingerprint: artifact.params_fingerprint.clone(),
            })
            .is_err());
    }

    #[test]
    fn bundle_replay_invokes_predict_with_data_and_refit_artifact_handles() {
        let plan = fixture_plan("plan:replay.predict");
        let bundle = replay_bundle(&plan);
        let request = replay_request(&bundle, Phase::Predict);
        let envelopes = replay_envelopes();
        let provider = replay_data_provider();
        let store = replay_artifact_store(&bundle);
        let controllers = replay_runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:replay.predict").unwrap(), Some(11));

        let results = SequentialScheduler
            .execute_bundle_replay(
                BundleReplayExecution {
                    plan: &plan,
                    bundle: &bundle,
                    replay_request: &request,
                    prediction_cache_store: None,
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &store,
                    data_envelopes: &envelopes,
                },
                &mut ctx,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(provider.handle_records().len(), 1);
        assert_eq!(provider.view_records().len(), 1);
        assert_eq!(
            provider.view_records()[0].view.partition,
            DataRequestPartition::Predict
        );
        assert_eq!(ctx.prediction_store.blocks().len(), 1);
        assert_eq!(
            ctx.prediction_store.blocks()[0].partition,
            PredictionPartition::Final
        );
        assert!(ctx
            .lineage
            .records()
            .any(|record| record.node_id.as_str() == "model:base"
                && record.phase == Phase::Predict
                && record.variant_id == bundle.selected_variant_id));

        let provider = replay_data_provider();
        let mut ctx = RunContext::new(RunId::new("run:parallel.replay.predict").unwrap(), Some(11));
        let results = ParallelScheduler::new(2)
            .unwrap()
            .execute_bundle_replay(
                BundleReplayExecution {
                    plan: &plan,
                    bundle: &bundle,
                    replay_request: &request,
                    prediction_cache_store: None,
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &store,
                    data_envelopes: &envelopes,
                },
                &mut ctx,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(provider.handle_records().len(), 1);
        assert_eq!(provider.view_records().len(), 1);
        assert_eq!(
            provider.view_records()[0].view.partition,
            DataRequestPartition::Predict
        );
        assert_eq!(ctx.prediction_store.blocks().len(), 1);
    }

    #[test]
    fn bundle_replay_rejects_missing_artifact_unsupported_phase_and_bad_envelope() {
        let plan = fixture_plan("plan:replay.reject");
        let bundle = replay_bundle(&plan);
        let request = replay_request(&bundle, Phase::Predict);
        let envelopes = replay_envelopes();
        let provider = replay_data_provider();
        let controllers = replay_runtime_controllers();
        let mut ctx = RunContext::new(RunId::new("run:replay.reject").unwrap(), Some(11));

        assert!(SequentialScheduler
            .execute_bundle_replay(
                BundleReplayExecution {
                    plan: &plan,
                    bundle: &bundle,
                    replay_request: &request,
                    prediction_cache_store: None,
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &InMemoryArtifactStore::new(),
                    data_envelopes: &envelopes,
                },
                &mut ctx,
            )
            .is_err());

        let store = replay_artifact_store(&bundle);
        assert!(SequentialScheduler
            .execute_bundle_replay(
                BundleReplayExecution {
                    plan: &plan,
                    bundle: &bundle,
                    replay_request: &replay_request(&bundle, Phase::FitCv),
                    prediction_cache_store: None,
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &store,
                    data_envelopes: &envelopes,
                },
                &mut ctx,
            )
            .is_err());

        let mut bad_envelopes = replay_envelopes();
        bad_envelopes
            .get_mut("model:base.x")
            .unwrap()
            .schema_fingerprint = "0".repeat(64);
        assert!(SequentialScheduler
            .execute_bundle_replay(
                BundleReplayExecution {
                    plan: &plan,
                    bundle: &bundle,
                    replay_request: &request,
                    prediction_cache_store: None,
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &store,
                    data_envelopes: &bad_envelopes,
                },
                &mut ctx,
            )
            .is_err());
    }
}
