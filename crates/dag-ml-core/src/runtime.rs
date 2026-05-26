use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bundle::{ExecutionBundle, RefitArtifactRecord, ReplayPhaseRequest};
use crate::data::{DataBinding, DataRequestPartition, ExternalDataPlanEnvelope};
use crate::error::{DagMlError, Result};
use crate::fold::{FoldAssignment, FoldSet};
use crate::ids::{
    ArtifactId, BranchId, BundleId, ControllerId, FoldId, LineageId, NodeId, RunId, SampleId,
    VariantId,
};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::phase::Phase;
use crate::plan::{ExecutionPlan, NodePlan};
use crate::policy::ShapeDelta;
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub kind: String,
    pub controller_id: ControllerId,
    pub size_bytes: Option<u64>,
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
pub struct NodeTask {
    pub run_id: RunId,
    pub node_plan: NodePlan,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub branch_path: Vec<BranchId>,
    #[serde(default)]
    pub input_handles: BTreeMap<String, HandleRef>,
    #[serde(default)]
    pub data_views: BTreeMap<String, DataProviderViewSpec>,
    pub seed: Option<u64>,
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
        }
        self.lineage.validate()
    }
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
        && !task.node_plan.data_bindings.is_empty()
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

pub trait RuntimeController {
    fn controller_id(&self) -> &ControllerId;
    fn invoke(&self, task: &NodeTask) -> Result<NodeResult>;
}

pub struct BundleReplayExecution<'a> {
    pub plan: &'a ExecutionPlan,
    pub bundle: &'a ExecutionBundle,
    pub replay_request: &'a ReplayPhaseRequest,
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
struct PhaseScope {
    phase: Phase,
    variant_id: Option<VariantId>,
    fold_id: Option<FoldId>,
    seed_root: Option<u64>,
}

#[derive(Default)]
struct PhaseScopeResources<'a> {
    data_provider: Option<&'a dyn RuntimeDataProvider>,
    replay_artifact_handles: Option<&'a BTreeMap<NodeId, BTreeMap<String, HandleRef>>>,
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
        replay.replay_request.validate_for_bundle(replay.bundle)?;
        replay
            .bundle
            .validate_replay_envelopes(replay.data_envelopes)?;
        let replay_artifacts = materialize_replay_artifact_handles(
            replay.plan,
            replay.bundle,
            replay.replay_request,
            replay.artifact_store,
            ctx,
        )?;
        let seed_root = replay
            .bundle
            .selected_variant_id
            .as_ref()
            .and_then(|selected| {
                replay
                    .plan
                    .variants
                    .iter()
                    .find(|variant| &variant.variant_id == selected)
                    .and_then(|variant| variant.seed)
            })
            .or(ctx.root_seed);

        self.execute_phase_scope(
            replay.plan,
            replay.controllers,
            ctx,
            PhaseScope {
                phase: replay.replay_request.phase,
                variant_id: replay.bundle.selected_variant_id.clone(),
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(replay.data_provider),
                replay_artifact_handles: Some(&replay_artifacts),
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
        let mut input_lineage = BTreeMap::<NodeId, LineageId>::new();

        for node_id in &plan.graph_plan.topological_order {
            let node_plan = plan
                .node_plans
                .get(node_id)
                .expect("execution plan was validated");
            if !node_plan.supported_phases.contains(&scope.phase) {
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
                resources.data_provider,
                ctx,
                &scope,
            )?;
            let mut input_handles = collected_inputs.handles;
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
            let task = NodeTask {
                run_id: ctx.run_id.clone(),
                node_plan: node_plan.clone(),
                phase: scope.phase,
                variant_id: scope.variant_id.clone(),
                fold_id: scope.fold_id.clone(),
                branch_path: Vec::new(),
                input_handles,
                data_views: collected_inputs.data_views,
                seed: derive_task_seed(
                    scope.seed_root,
                    scope.variant_id.as_ref(),
                    scope.fold_id.as_ref(),
                    node_plan,
                    scope.phase,
                ),
            };
            let result = controller.invoke(&task)?;
            result.validate_for_task(&task)?;
            if let Some(store) = resources.artifact_store.as_deref_mut() {
                if scope.phase == Phase::Refit {
                    store.capture_refit_artifacts(&task, &result)?;
                }
            }
            for prediction in &result.predictions {
                ctx.prediction_store.append(prediction.clone())?;
            }
            ctx.lineage.record(result.lineage.clone())?;
            output_handles.insert(node_id.clone(), result.outputs.clone());
            input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
            results.push(result);
        }

        Ok(results)
    }
}

fn collect_input_handles(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    output_handles: &BTreeMap<NodeId, BTreeMap<String, HandleRef>>,
    data_provider: Option<&dyn RuntimeDataProvider>,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<CollectedInputs> {
    let mut inputs = BTreeMap::new();
    let mut data_views = BTreeMap::new();
    for upstream in &node_plan.input_nodes {
        if let Some(handles) = output_handles.get(upstream) {
            for (port, handle) in handles {
                inputs.insert(format!("{upstream}.{port}"), handle.clone());
            }
        }
    }
    if !node_plan.data_bindings.is_empty() && data_provider.is_none() {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` requires {} data binding(s) but no runtime data provider is registered",
            node_plan.node_id,
            node_plan.data_bindings.len()
        )));
    }
    if let Some(data_provider) = data_provider {
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
            let key = format!("data:{}", binding.input_name);
            let view_handle = make_data_view_handle(
                data_provider,
                ctx,
                node_plan,
                scope,
                binding,
                &materialized,
                &view,
            )?;
            data_views.insert(key.clone(), view);
            inputs.insert(key, view_handle);

            if let Some(validation_view) =
                validation_data_view_for_scope(binding, plan.fold_set.as_ref(), scope)?
            {
                let validation_key = format!("data:{}:validation", binding.input_name);
                let validation_handle = make_data_view_handle(
                    data_provider,
                    ctx,
                    node_plan,
                    scope,
                    binding,
                    &materialized,
                    &validation_view,
                )?;
                data_views.insert(validation_key.clone(), validation_view);
                inputs.insert(validation_key, validation_handle);
            }
        }
    }
    Ok(CollectedInputs {
        handles: inputs,
        data_views,
    })
}

struct CollectedInputs {
    handles: BTreeMap<String, HandleRef>,
    data_views: BTreeMap<String, DataProviderViewSpec>,
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
    Ok(DataProviderViewSpec {
        sample_ids,
        partition,
        fold_id: scope.fold_id.clone(),
        source_ids: (!binding.source_ids.is_empty()).then(|| binding.source_ids.clone()),
        columns: None,
        include_augmented,
        include_excluded: binding.view_policy.include_excluded,
        extra,
    })
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

fn materialize_replay_artifact_handles(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    replay_request: &ReplayPhaseRequest,
    artifact_store: &dyn RuntimeArtifactStore,
    ctx: &RunContext,
) -> Result<BTreeMap<NodeId, BTreeMap<String, HandleRef>>> {
    let mut handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
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
    }
    Ok(handles)
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
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::bundle::{build_execution_bundle, RefitArtifactRecord, ReplayPhaseRequest};
    use crate::controller::{
        ArtifactPolicy, ControllerCapability, ControllerFitScope, ControllerManifest,
        ControllerRegistry, RngPolicy,
    };
    use crate::data::{ExternalDataPlanEnvelope, InMemoryDataProvider};
    use crate::fold::{FoldAssignment, FoldSet};
    use crate::generation::{
        GenerationChoice, GenerationDimension, GenerationSpec, GenerationStrategy,
    };
    use crate::graph::{
        EdgeContract, EdgeSpec, GraphInterface, GraphSpec, NodeKind, NodeSpec, PortCardinality,
        PortKind, PortRef, PortSchema, PortSpec,
    };
    use crate::ids::{ArtifactId, ControllerId, FoldId, NodeId, SampleId};
    use crate::oof::{PredictionBlock, PredictionPartition};
    use crate::plan::{build_execution_plan, CampaignSpec, SplitInvocation};
    use serde_json::json;

    struct MockController {
        id: ControllerId,
        handle: u64,
        emit_prediction: bool,
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
                outputs: BTreeMap::from([("out".to_string(), output)]),
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
                    size_bytes: Some(128),
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
        ControllerManifest {
            controller_id: ControllerId::new(id).unwrap(),
            controller_version: "0.1.0".to_string(),
            operator_kind: kind,
            priority: 0,
            supported_phases: BTreeSet::from([Phase::FitCv]),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            data_requirements: None,
            capabilities: BTreeSet::from([ControllerCapability::Deterministic]),
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
                    size_bytes: Some(128),
                },
                params_fingerprint: model_plan.params_fingerprint.clone(),
                data_requirement_keys: vec!["model:base.x".to_string()],
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
                "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
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
                            },
                            GenerationChoice {
                                label: "rf".to_string(),
                                value: json!("rf"),
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
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
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
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
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
            fold_id: None,
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
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
            fold_id: None,
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
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
            size_bytes: Some(128),
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
