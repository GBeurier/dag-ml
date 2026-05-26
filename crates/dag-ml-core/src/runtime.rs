use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bundle::{ExecutionBundle, RefitArtifactRecord, ReplayPhaseRequest};
use crate::data::ExternalDataPlanEnvelope;
use crate::error::{DagMlError, Result};
use crate::ids::{
    ArtifactId, BranchId, BundleId, ControllerId, FoldId, LineageId, NodeId, RunId, VariantId,
};
use crate::oof::PredictionBlock;
use crate::phase::Phase;
use crate::plan::{ExecutionPlan, NodePlan};
use crate::policy::ShapeDelta;
use crate::rng::SeedContext;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleKind {
    Data,
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
        if self
            .records
            .insert(record.artifact.id.clone(), record)
            .is_some()
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate artifact handle for `{}`",
                artifact.artifact.id
            )));
        }
        Ok(())
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
        for artifact in &self.artifacts {
            if artifact.controller_id != task.node_plan.controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted artifact `{}` for controller `{}`, expected `{}`",
                    task.node_plan.node_id,
                    artifact.id,
                    artifact.controller_id,
                    task.node_plan.controller_id
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

pub trait RuntimeDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> Result<HandleRef>;
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
            None,
            None,
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
            Some(data_provider),
            None,
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
                    None,
                    None,
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
                    Some(data_provider),
                    None,
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
            Some(replay.data_provider),
            Some(&replay_artifacts),
        )
    }

    fn execute_phase_scope(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        scope: PhaseScope,
        data_provider: Option<&dyn RuntimeDataProvider>,
        replay_artifact_handles: Option<&BTreeMap<NodeId, BTreeMap<String, HandleRef>>>,
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
            let mut input_handles =
                collect_input_handles(node_plan, &output_handles, data_provider, ctx, &scope)?;
            if let Some(node_artifact_handles) =
                replay_artifact_handles.and_then(|handles| handles.get(node_id))
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
    node_plan: &NodePlan,
    output_handles: &BTreeMap<NodeId, BTreeMap<String, HandleRef>>,
    data_provider: Option<&dyn RuntimeDataProvider>,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<BTreeMap<String, HandleRef>> {
    let mut inputs = BTreeMap::new();
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
            let handle = data_provider.materialize(&DataMaterializationRequest {
                run_id: ctx.run_id.clone(),
                node_id: node_plan.node_id.clone(),
                input_name: binding.input_name.clone(),
                phase: scope.phase,
                variant_id: scope.variant_id.clone(),
                fold_id: scope.fold_id.clone(),
                binding: binding.clone(),
            })?;
            inputs.insert(format!("data:{}", binding.input_name), handle);
        }
    }
    Ok(inputs)
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
                if handle.kind != HandleKind::Data {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-data handle for `{key}`",
                        task.node_plan.node_id
                    )));
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
            let predictions = self
                .emit_prediction
                .then(|| PredictionBlock {
                    prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                    producer_node: task.node_plan.node_id.clone(),
                    partition: PredictionPartition::Validation,
                    fold_id: None,
                    sample_ids: vec![SampleId::new("s1").unwrap()],
                    values: vec![vec![1.0]],
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
                if handle.kind != HandleKind::Data {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-data handle for `{key}`",
                        task.node_plan.node_id
                    )));
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
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([("out".to_string(), output)]),
                predictions,
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
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
            }))
            .unwrap();
        controllers
            .register(Box::new(ReplayMockController {
                id: ControllerId::new("controller:model.mock").unwrap(),
                handle: 22,
                require_artifact: true,
                emit_prediction: true,
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
        assert_eq!(provider.handle_records()[0].input_name, "x");
        assert_eq!(provider.handle_records()[0].relation_record_count, Some(4));
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
