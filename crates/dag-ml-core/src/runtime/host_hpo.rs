//! Nonportable host-optimizer search with scheduler-owned candidate execution.
//!
//! The host proposes parameter values and receives native scores only. It never
//! supplies predictions, folds, or scalar scores through the tuner interface.
//! This contract deliberately has no Methods ABI or N4MOPT checkpoint claim.

use super::*;
use crate::realization_evidence as realization;

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostHpoFoldReduction {
    Mean,
    Best,
    RobustBest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoParameterBinding {
    pub node_id: NodeId,
    pub param_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoSearchRequest {
    pub target_node: NodeId,
    pub trial_budget: u32,
    pub metric: RegressionMetricKind,
    pub direction: crate::selection::MetricObjective,
    pub optimizer_descriptor: BTreeMap<String, serde_json::Value>,
    /// Consecutive trial budgets for a phased optimizer. Empty means one
    /// implicit phase. The budgets must sum to `trial_budget`; core owns the
    /// phase boundary while the proposal source owns sampler-specific state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phase_trial_budgets: Vec<u32>,
    /// Ask the host optimizer for a pruning decision after each native fold.
    #[serde(default, skip_serializing_if = "is_false")]
    pub progressive_pruning: bool,
    /// None preserves the original global OOF objective. Fold reductions are
    /// explicit selection evidence, never synthetic OOF ScoreSet reports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fold_score_reduction: Option<HostHpoFoldReduction>,
    /// Public proposal paths mapped to operator-local parameters. Empty retains
    /// the original single-target routing and its serialized fingerprints.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameter_bindings: BTreeMap<String, HostHpoParameterBinding>,
    /// Compiler-owned immutable recipes. Absent preserves fixed-topology wire
    /// bytes, objective hashes and schema-1 checkpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_catalogue: Option<HostHpoStructuralCatalogue>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostHpoMethodsBuildProfile {
    schema_version: u32,
    blas: bool,
    openmp: bool,
    cuda: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostHpoParallelExecutionProfile {
    schema_version: u32,
    profile: String,
    workers: usize,
    cpu_threads: u32,
    gpu_devices: Vec<String>,
    methods_build: HostHpoMethodsBuildProfile,
}

impl HostHpoSearchRequest {
    fn validated_phase_budgets(&self) -> Result<()> {
        if !self.phase_trial_budgets.is_empty()
            && (self.phase_trial_budgets.contains(&0)
                || self
                    .phase_trial_budgets
                    .iter()
                    .try_fold(0u32, |n, budget| n.checked_add(*budget))
                    != Some(self.trial_budget))
        {
            return Err(DagMlError::RuntimeValidation(
                "phase trial budgets must be positive and sum to the trial budget without overflow"
                    .into(),
            ));
        }
        Ok(())
    }

    fn validate_browser_execution(&self) -> Result<()> {
        if self.structural_catalogue.as_ref().is_some_and(|catalogue| {
            catalogue
                .entries
                .iter()
                .any(|entry| entry.graph.metadata.contains_key("python_torch_profile"))
        }) {
            return Err(DagMlError::RuntimeValidation(
                "Python Torch topology requires native serial host execution".into(),
            ));
        }
        if self
            .structural_catalogue
            .as_ref()
            .is_some_and(|catalogue| catalogue.topology_contract.is_some())
        {
            return Err(DagMlError::RuntimeValidation(
                "topology HPO currently requires serial candidate execution".into(),
            ));
        }
        if self.optimizer_descriptor.contains_key("parallel_execution") {
            return Err(DagMlError::RuntimeValidation(
                "parallel_execution is a native-only profile and is not supported by browser HPO"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Admit only the declared bounded sequential-numerics profile for typed
    /// Methods recipes. The host attests its actual library build; core signs
    /// that declaration into the objective/checkpoint and enforces its budget.
    /// Historical serial descriptors and generic host parallelism stay intact.
    pub fn validate_parallel_execution(
        &self,
        workers: usize,
    ) -> Result<Option<crate::training::TrainingResourceLimits>> {
        if workers > 1
            && self.structural_catalogue.as_ref().is_some_and(|catalogue| {
                catalogue
                    .entries
                    .iter()
                    .any(|entry| entry.graph.metadata.contains_key("python_torch_profile"))
            })
        {
            return Err(DagMlError::RuntimeValidation(
                "Python Torch topology requires serial candidate execution".into(),
            ));
        }
        if workers > 1
            && self.structural_catalogue.as_ref().is_some_and(|catalogue| {
                catalogue.entries.iter().any(|entry| {
                    entry.graph.nodes.iter().any(|node| {
                        node.operator.as_ref().is_some_and(|operator| {
                            matches!(
                                operator.get("type").and_then(serde_json::Value::as_str),
                                Some(
                                    crate::METHODS_RAW_CLASSIFIER | crate::METHODS_META_CLASSIFIER
                                )
                            )
                        })
                    })
                })
            })
        {
            return Err(DagMlError::RuntimeValidation("Methods classification requires serial candidate execution regardless of the objective metric".into()));
        }
        let typed_methods = self.structural_catalogue.as_ref().is_some_and(|catalogue| {
            catalogue.topology_contract.is_some()
                || catalogue.entries.iter().any(|entry| {
                    entry.graph.nodes.iter().any(|node| {
                        node.operator
                            .as_ref()
                            .and_then(|operator| operator.get("type"))
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|kind| {
                                matches!(
                                    kind,
                                    "N4mMultimodalPipeline" | crate::METHODS_RAW_CLASSIFIER
                                )
                            })
                    })
                })
        });
        let declaration = self.optimizer_descriptor.get("parallel_execution");
        if typed_methods || declaration.is_some() {
            let declared_workers = match self.optimizer_descriptor.get("n_jobs") {
                None => Some(1),
                Some(value) => value.as_u64().and_then(|value| usize::try_from(value).ok()),
            };
            if declared_workers != Some(workers) {
                return Err(DagMlError::RuntimeValidation(
                    "typed host HPO n_jobs must match the actual worker budget".into(),
                ));
            }
            if workers > 1 && declaration.is_none() {
                return Err(DagMlError::RuntimeValidation(
                    "typed parallel host HPO requires a sequential Methods build profile".into(),
                ));
            }
        }
        let Some(declaration) = declaration else {
            return Ok(None);
        };
        let profile: HostHpoParallelExecutionProfile = serde_json::from_value(declaration.clone())
            .map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "invalid typed host HPO parallel_execution declaration: {error}"
                ))
            })?;
        if profile.schema_version != 1
            || profile.profile != "methods_sequential_cpu_v1"
            || !(2..=4).contains(&workers)
            || profile.workers != workers
            || profile.cpu_threads != 1
            || !profile.gpu_devices.is_empty()
            || profile.methods_build.schema_version != 1
            || profile.methods_build.blas
            || profile.methods_build.openmp
            || profile.methods_build.cuda
            || !matches!(
                self.metric,
                RegressionMetricKind::Mse
                    | RegressionMetricKind::Rmse
                    | RegressionMetricKind::Mae
                    | RegressionMetricKind::R2
            )
            || self
                .optimizer_descriptor
                .get("sampler")
                .and_then(serde_json::Value::as_str)
                != Some("random")
            || self
                .optimizer_descriptor
                .get("pruner")
                .is_some_and(|value| !value.is_null() && value.as_str() != Some("none"))
            || self.progressive_pruning
            || self
                .optimizer_descriptor
                .contains_key("generated_view_mode")
        {
            return Err(DagMlError::RuntimeValidation(
                "typed parallel host HPO requires regression, workers 2..4, random sampling, no pruning, and the sequential Methods CPU profile".into(),
            ));
        }
        Ok(Some(crate::training::TrainingResourceLimits {
            cpu_threads: 1,
            memory_bytes: None,
            gpu_devices: Vec::new(),
            wall_time_ms: None,
        }))
    }

    /// Inspect the actual graph as well as the signed catalogue. A regression
    /// metric or an absent catalogue cannot relabel a classifier as parallel-safe.
    pub fn validate_parallel_execution_for_plan(
        &self,
        workers: usize,
        plan: &ExecutionPlan,
    ) -> Result<Option<crate::training::TrainingResourceLimits>> {
        if workers > 1 && crate::python_torch_profile::has_torch_profile(plan) {
            return Err(DagMlError::RuntimeValidation(
                "Python Torch topology requires serial candidate execution".into(),
            ));
        }
        if workers > 1
            && plan.graph_plan.graph.nodes.iter().any(|node| {
                node.operator.as_ref().is_some_and(|operator| {
                    matches!(
                        operator.get("type").and_then(serde_json::Value::as_str),
                        Some(crate::METHODS_RAW_CLASSIFIER | crate::METHODS_META_CLASSIFIER)
                    )
                })
            })
        {
            return Err(DagMlError::RuntimeValidation("Methods classification requires serial candidate execution regardless of the objective metric".into()));
        }
        self.validate_parallel_execution(workers)
    }

    fn phase_index(&self, trial_index: u32) -> Option<u32> {
        if self.phase_trial_budgets.is_empty() {
            return None;
        }
        let mut end = 0u32;
        for (index, budget) in self.phase_trial_budgets.iter().enumerate() {
            end = end.checked_add(*budget)?;
            if trial_index < end {
                return Some(index as u32);
            }
        }
        None
    }

    pub(super) fn validate_parameter_bindings(&self, plan: &ExecutionPlan) -> Result<()> {
        if self.direction != self.metric.objective() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO direction must match metric objective".into(),
            ));
        }
        self.validated_phase_budgets()?;
        crate::methods_classification::validate_methods_classifier_search(plan, self)?;
        crate::python_torch_profile::validate_torch_search(plan, self)?;
        if let Some(catalogue) = &self.structural_catalogue {
            if catalogue.topology_contract.is_some() && self.progressive_pruning {
                return Err(DagMlError::RuntimeValidation(
                    "topology HPO does not support progressive pruning of nested stacking".into(),
                ));
            }
            if !self.parameter_bindings.is_empty()
                || catalogue
                    .entries
                    .first()
                    .is_none_or(|entry| entry.target_node != self.target_node)
            {
                return Err(DagMlError::RuntimeValidation(
                    "structural HPO requires its first recipe target and recipe-local bindings"
                        .into(),
                ));
            }
            return catalogue.validate_for_plan(plan);
        }
        let mut destinations = BTreeSet::new();
        for (path, binding) in &self.parameter_bindings {
            if path.trim().is_empty() || binding.param_path.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO parameter binding paths must be nonempty".into(),
                ));
            }
            if plan.node_plans.get(&binding.node_id).is_none_or(|node| {
                !matches!(
                    node.kind,
                    NodeKind::Model | NodeKind::Transform | NodeKind::YTransform
                ) || !node.supported_phases.contains(&Phase::FitCv)
            }) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "host HPO parameter binding `{path}` requires an existing FIT_CV model or transform node: `{}`",
                    binding.node_id
                )));
            }
            if !destinations.insert((&binding.node_id, &binding.param_path)) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "host HPO parameter bindings collide at `{}.{}`",
                    binding.node_id, binding.param_path
                )));
            }
        }
        Ok(())
    }

    pub(super) fn parameter_overrides(
        &self,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Vec<crate::generation::GenerationParamOverride>> {
        if let Some(catalogue) = &self.structural_catalogue {
            let recipe = catalogue.recipe(params)?;
            let active = params
                .iter()
                .filter(|(path, _)| *path != &catalogue.selector_path)
                .map(|(path, value)| (path.clone(), value.clone()))
                .collect();
            let mut routed = self.clone();
            routed.structural_catalogue = None;
            routed.target_node = recipe.target_node.clone();
            routed.parameter_bindings = recipe.parameter_bindings.clone();
            if recipe.parameter_bindings.is_empty() {
                return Ok(Vec::new());
            }
            return routed.parameter_overrides(&active);
        }
        if params.is_empty() || params.keys().any(|key| key.trim().is_empty()) {
            return Err(DagMlError::RuntimeValidation(
                "host HPO proposal parameters must be nonempty".into(),
            ));
        }
        if self.parameter_bindings.is_empty() {
            return Ok(vec![crate::generation::GenerationParamOverride {
                node_id: self.target_node.clone(),
                params: params.clone(),
            }]);
        }
        let mut grouped: BTreeMap<NodeId, BTreeMap<String, serde_json::Value>> = BTreeMap::new();
        for (path, value) in params {
            let binding = self.parameter_bindings.get(path).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "host HPO proposed parameter `{path}` has no parameter binding"
                ))
            })?;
            grouped
                .entry(binding.node_id.clone())
                .or_default()
                .insert(binding.param_path.clone(), value.clone());
        }
        Ok(grouped
            .into_iter()
            .map(|(node_id, params)| crate::generation::GenerationParamOverride { node_id, params })
            .collect())
    }

    pub(super) fn target_for_plan<'a>(&'a self, plan: &ExecutionPlan) -> Result<&'a NodeId> {
        let Some(catalogue) = &self.structural_catalogue else {
            return Ok(&self.target_node);
        };
        let recipe_id = plan
            .variants
            .first()
            .and_then(|variant| variant.choices.get("host_hpo"))
            .and_then(|choice| choice.value.get("recipe_id"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "structural HPO candidate lacks recipe identity".into(),
                )
            })?;
        let recipe = catalogue
            .entries
            .iter()
            .find(|entry| entry.recipe_id.as_str() == recipe_id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation("structural HPO candidate recipe is unknown".into())
            })?;
        if !plan.node_plans.contains_key(&recipe.target_node) {
            return Err(DagMlError::RuntimeValidation(
                "structural HPO target is outside candidate plan".into(),
            ));
        }
        Ok(&recipe.target_node)
    }
}

pub trait HostHpoProposalSource {
    fn ask(&mut self, trial_index: u32) -> Result<Option<BTreeMap<String, serde_json::Value>>>;
    /// Receive the core-owned phase for a candidate. Existing one-phase
    /// sources remain source-compatible through the default implementation.
    fn ask_in_phase(
        &mut self,
        trial_index: u32,
        _phase_index: Option<u32>,
    ) -> Result<Option<BTreeMap<String, serde_json::Value>>> {
        self.ask(trial_index)
    }
    fn tell(&mut self, trial_index: u32, score: f64) -> Result<()>;
    /// Return true to prune this trial after a report-grade intermediate score.
    fn report_intermediate(&mut self, _trial_index: u32, _step: u32, _score: f64) -> Result<bool> {
        Ok(false)
    }
    fn pruned(&mut self, _trial_index: u32) -> Result<()> {
        Ok(())
    }
    /// Terminalize a failed candidate before pairing durable optimizer state.
    fn fail(&mut self, _trial_index: u32, _error: &str) -> Result<()> {
        Ok(())
    }
    /// Receive the planned candidate of a trial. Only the capture bridge uses it.
    fn candidate_planned(&mut self, _trial_index: u32, _plan: &ExecutionPlan) -> Result<()> {
        Ok(())
    }
    /// Observe the coordinator stage of a failing edge. Only the capture bridge
    /// retains it; every other source returns the error unchanged.
    fn stage_failure(
        &mut self,
        _trial_index: u32,
        _stage: realization::FailureStage,
        error: DagMlError,
    ) -> DagMlError {
        error
    }
}

/// Creates a fresh data-provider namespace for one candidate. Implementations
/// must preserve the attested input envelope while keeping handle maps local
/// to the candidate; the returned provider may move to a worker thread.
pub trait HostHpoCandidateProviderFactory: Send + Sync {
    fn create(&self, trial_index: u32) -> Result<Box<dyn RuntimeDataProvider + Send>>;

    /// Snapshot the cumulative generated views after a parallel window joins.
    /// Every terminal in that window receives the same snapshot, even when
    /// candidates finished out of order or one failed after reading views.
    fn batch_generated_view_manifest(
        &self,
        _has_success: bool,
    ) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }
}

fn host_hpo_provider_manifest(
    provider: &dyn RuntimeDataProvider,
) -> Result<Option<serde_json::Value>> {
    let manifest = provider.generated_view_manifest()?;
    if provider.generated_views_enabled() && manifest.is_none() {
        return Err(DagMlError::RuntimeValidation(
            "generated host HPO provider produced no view manifest".into(),
        ));
    }
    Ok(manifest)
}

/// Build a controller registry with candidate-local host callback state.
pub trait HostHpoCandidateControllerFactory: Send + Sync {
    fn create(&self, trial_index: u32) -> Result<RuntimeControllerRegistry>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoTrialEvidence {
    pub trial_index: u32,
    pub params: BTreeMap<String, serde_json::Value>,
    pub score: f64,
    pub variant_id: VariantId,
    pub scores: ScoreSet,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub objective_fold_scores: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_view_manifest: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostHpoSearchResult {
    pub profile: String,
    pub portable: bool,
    pub request_fingerprint: String,
    pub graph_fingerprint: String,
    pub controller_fingerprint: String,
    pub campaign_fingerprint: String,
    pub fold_set_fingerprint: String,
    pub trials: Vec<HostHpoTrialEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pruned_trials: Vec<HostHpoPrunedTrialEvidence>,
    pub selected_trial_index: u32,
    pub selected_params: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoPrunedTrialEvidence {
    pub trial_index: u32,
    pub params: BTreeMap<String, serde_json::Value>,
    pub variant_id: VariantId,
    pub scores: ScoreSet,
    pub intermediate_scores: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_view_manifest: Option<serde_json::Value>,
}

enum HostHpoEvaluation {
    Complete(HostHpoTrialEvidence, crate::selection::CandidateScore),
    Pruned(HostHpoPrunedTrialEvidence),
}

enum ParallelHostHpoEvent {
    Intermediate {
        trial_index: u32,
        step: u32,
        score: f64,
        reply: std::sync::mpsc::Sender<Result<bool>>,
    },
    Completed {
        trial_index: u32,
        evaluated: Box<Result<HostHpoEvaluation>>,
    },
}

/// Operational stopping state; cancellation never requests a REFIT.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostHpoSearchStatus {
    Running,
    Completed,
    Cancelled,
    Exhausted,
    Failed,
}

/// A terminal candidate. Failed fits carry no manufactured score.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostHpoTerminalTrial {
    Complete {
        evidence: HostHpoTrialEvidence,
    },
    Pruned {
        evidence: HostHpoPrunedTrialEvidence,
    },
    Failed {
        trial_index: u32,
        params: BTreeMap<String, serde_json::Value>,
        variant_id: VariantId,
        error: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        generated_view_manifest: Option<serde_json::Value>,
    },
}

impl HostHpoTerminalTrial {
    pub fn trial_index(&self) -> u32 {
        match self {
            Self::Complete { evidence } => evidence.trial_index,
            Self::Pruned { evidence } => evidence.trial_index,
            Self::Failed { trial_index, .. } => *trial_index,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCheckpointBinding {
    pub objective_fingerprint: String,
    pub graph_fingerprint: String,
    pub controller_fingerprint: String,
    pub campaign_fingerprint: String,
    pub fold_set_fingerprint: String,
    /// Canonical host-data envelope including schemas, identities and relations.
    /// Host content-version descriptors belong in that envelope or the request.
    pub data_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_catalogue_fingerprint: Option<String>,
}

/// Native score evidence, independent of opaque host optimizer bytes.
/// Producers atomically pair this document with their optimizer checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCheckpoint {
    pub schema_version: u32,
    pub binding: HostHpoCheckpointBinding,
    pub trials: Vec<HostHpoTerminalTrial>,
    pub fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoInterruptedTrial {
    pub trial_index: u32,
    pub params: BTreeMap<String, serde_json::Value>,
}

fn canonical_checkpoint_trials(trials: &[HostHpoTerminalTrial]) -> Vec<HostHpoTerminalTrial> {
    let mut trials = trials.to_vec();
    for trial in &mut trials {
        let (params, manifest) = match trial {
            HostHpoTerminalTrial::Complete { evidence } => {
                (&mut evidence.params, &mut evidence.generated_view_manifest)
            }
            HostHpoTerminalTrial::Pruned { evidence } => {
                (&mut evidence.params, &mut evidence.generated_view_manifest)
            }
            HostHpoTerminalTrial::Failed {
                params,
                generated_view_manifest,
                ..
            } => (params, generated_view_manifest),
        };
        for value in params.values_mut() {
            value.sort_all_objects();
        }
        if let Some(value) = manifest {
            value.sort_all_objects();
        }
    }
    trials
}

impl HostHpoCheckpoint {
    fn seal(&mut self) -> Result<()> {
        // Only dynamic JSON objects use canonical key ordering. Keep the
        // typed schema field order and every scientific array order unchanged,
        // matching checkpoints written before serde_json/preserve_order.
        let trials = canonical_checkpoint_trials(&self.trials);
        self.fingerprint = stable_json_fingerprint(&(self.schema_version, &self.binding, &trials))?;
        Ok(())
    }

    pub fn verify_seal(&self) -> Result<()> {
        let mut sealed = self.clone();
        sealed.seal()?;
        if sealed.fingerprint != self.fingerprint {
            return Err(DagMlError::RuntimeValidation(
                "host HPO checkpoint integrity fingerprint mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Recover a trial prepared before an optimizer transition, then account
    /// for any candidates that were still in flight when the host stopped.
    /// In-flight candidates have no score evidence and become failed trials.
    pub fn recover_interrupted_trials(
        mut self,
        prepared: Option<Self>,
        interrupted: Vec<HostHpoInterruptedTrial>,
    ) -> Result<Self> {
        self.verify_seal()?;
        if let Some(prepared) = prepared {
            prepared.verify_seal()?;
            if prepared.schema_version != self.schema_version
                || prepared.binding != self.binding
                || prepared.trials.len() != self.trials.len() + 1
                || stable_json_fingerprint(&canonical_checkpoint_trials(
                    &prepared.trials[..self.trials.len()],
                ))? != stable_json_fingerprint(&canonical_checkpoint_trials(&self.trials))?
                || prepared
                    .trials
                    .last()
                    .map(HostHpoTerminalTrial::trial_index)
                    != Some(self.trials.len() as u32)
            {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO prepared terminal does not extend the paired checkpoint".into(),
                ));
            }
            self = prepared;
        }
        for orphan in interrupted {
            if orphan.trial_index != self.trials.len() as u32
                || orphan.params.keys().any(|key| key.trim().is_empty())
            {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO interrupted trial indices must extend checkpoint contiguously".into(),
                ));
            }
            self.trials.push(HostHpoTerminalTrial::Failed {
                trial_index: orphan.trial_index,
                params: orphan.params,
                variant_id: VariantId::new(format!("host_hpo:trial:{:010}", orphan.trial_index))?,
                error: "interrupted_before_native_evaluation".into(),
                generated_view_manifest: None,
            });
        }
        self.seal()?;
        Ok(self)
    }
}

/// Return false only to request cancellation at this completed-trial boundary.
pub trait HostHpoProgress {
    /// Persist a sealed prospective terminal checkpoint before the host
    /// optimizer is told. A durable host can recover a crash between the
    /// optimizer transition and the ordinary checkpoint publication.
    fn prepare_terminal(
        &mut self,
        _checkpoint: &HostHpoCheckpoint,
        _status: HostHpoSearchStatus,
    ) -> Result<()> {
        Ok(())
    }

    fn checkpoint(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> Result<bool>;
}

pub struct HostHpoResumeOptions {
    pub data_fingerprint: String,
    pub checkpoint: Option<HostHpoCheckpoint>,
}

impl HostHpoResumeOptions {
    pub fn from_envelope(
        envelope: &crate::data::ExternalDataPlanEnvelope,
        checkpoint: Option<HostHpoCheckpoint>,
    ) -> Result<Self> {
        envelope.validate()?;
        Ok(Self {
            data_fingerprint: stable_json_fingerprint(envelope)?,
            checkpoint,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HostHpoSearchOutcome {
    /// Absent when cancelled/exhausted before any successful candidate.
    #[serde(flatten)]
    pub result: Option<HostHpoSearchResult>,
    pub status: HostHpoSearchStatus,
    pub checkpoint: Option<HostHpoCheckpoint>,
}

/// One candidate that a browser may evaluate in an isolated Web Worker.
/// The worker receives a complete candidate plan, never an unbound parameter
/// map; the coordinator retains the checkpoint and orders terminalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoWorkerTask {
    pub trial_index: u32,
    pub phase_index: Option<u32>,
    pub params: BTreeMap<String, serde_json::Value>,
    pub candidate_plan: ExecutionPlan,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoWorkerWindow {
    pub window_id: String,
    pub checkpoint: HostHpoCheckpoint,
    pub tasks: Vec<HostHpoWorkerTask>,
    pub exhausted: bool,
}

fn host_hpo_worker_candidate_plan(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    checkpoint: &HostHpoCheckpoint,
    trial_index: u32,
    params: &BTreeMap<String, serde_json::Value>,
) -> Result<ExecutionPlan> {
    host_hpo_candidate_plan(
        plan,
        request,
        trial_index,
        params,
        Some(&checkpoint.binding.objective_fingerprint),
    )
}

fn host_hpo_candidate_plan(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    trial_index: u32,
    params: &BTreeMap<String, serde_json::Value>,
    objective_fingerprint: Option<&str>,
) -> Result<ExecutionPlan> {
    let param_overrides = request.parameter_overrides(params)?;
    let mut candidate_plan = match &request.structural_catalogue {
        Some(catalogue) => catalogue.candidate_plan(plan, params)?,
        None => plan.clone(),
    };
    let base_fingerprint = candidate_plan.variants[0].fingerprint.clone();
    let objective_fingerprint = match objective_fingerprint {
        Some(fingerprint) => fingerprint.to_owned(),
        None => stable_json_fingerprint(request)?,
    };
    let mut variant = candidate_plan.variants[0].clone();
    variant.variant_id = VariantId::new(format!("host_hpo:trial:{trial_index:010}"))?;
    variant.choices.insert(
        "host_hpo".into(),
        GenerationChoice {
            label: format!("trial:{trial_index}"),
            value: if let Some(catalogue) = &request.structural_catalogue {
                serde_json::json!({"trial_index": trial_index,
                    "recipe_id": catalogue.recipe(params)?.recipe_id,
                    "catalogue_fingerprint": catalogue.catalogue_fingerprint,
                    "objective_fingerprint": objective_fingerprint})
            } else {
                serde_json::json!({"trial_index": trial_index,
                    "objective_fingerprint": objective_fingerprint})
            },
            param_overrides,
            active_subsequence: None,
        },
    );
    variant.fingerprint =
        stable_json_fingerprint(&(&base_fingerprint, &variant.choices, &objective_fingerprint))?;
    candidate_plan.variants = vec![variant];
    candidate_plan.validate()?;
    crate::methods_classification::validate_methods_classification_plan(&candidate_plan)?;
    Ok(candidate_plan)
}

fn validate_structural_worker_task(
    task: &HostHpoWorkerTask,
    request: &HostHpoSearchRequest,
) -> Result<()> {
    let Some(catalogue) = &request.structural_catalogue else {
        return Ok(());
    };
    let mut registry = crate::ControllerRegistry::new();
    for manifest in task.candidate_plan.controller_manifests.values() {
        registry.register(manifest.clone())?;
    }
    let compiled = crate::compile_pipeline_dsl_with_generation_and_controller_registry(
        &catalogue.source_dsl,
        &registry,
    )?;
    let union = crate::build_execution_plan(
        task.candidate_plan.id.clone(),
        compiled.graph,
        compiled.campaign_template,
        &registry,
    )?;
    request.validate_parameter_bindings(&union)?;
    let expected = host_hpo_candidate_plan(
        &union,
        request,
        task.trial_index,
        &task.params,
        Some(&host_hpo_objective_fingerprint(request)?),
    )?;
    if stable_json_fingerprint(&expected)? != stable_json_fingerprint(&task.candidate_plan)? {
        return Err(DagMlError::RuntimeValidation(
            "structural HPO worker candidate declaration/plan mismatch".into(),
        ));
    }
    Ok(())
}

/// Prepare one phase-bounded browser worker window. This is intentionally a
/// pure coordinator step: no operator callback or worker result is accepted
/// here, and existing sequential HPO remains unchanged. The window is tied
/// to the validated native checkpoint, data fingerprint, and candidate plans.
pub fn prepare_host_hpo_worker_window(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    options: &HostHpoResumeOptions,
    proposals: &mut dyn HostHpoProposalSource,
    max_workers: usize,
) -> Result<HostHpoWorkerWindow> {
    request.validate_browser_execution()?;
    plan.validate()?;
    if max_workers < 2 || request.trial_budget == 0 || request.optimizer_descriptor.is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO worker windows require at least two workers, a positive budget and an optimizer descriptor".into(),
        ));
    }
    if !request.phase_trial_budgets.is_empty()
        && (request.phase_trial_budgets.contains(&0)
            || request
                .phase_trial_budgets
                .iter()
                .try_fold(0u32, |sum, count| sum.checked_add(*count))
                != Some(request.trial_budget))
    {
        return Err(DagMlError::RuntimeValidation(
            "host HPO phase budgets must be positive and sum to trial_budget".into(),
        ));
    }
    if plan.fold_set.is_none()
        || plan.variants.len() != 1
        || !plan.variants[0].choices.is_empty()
        || plan
            .node_plans
            .get(&request.target_node)
            .is_none_or(|node| node.kind != NodeKind::Model)
    {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO requires explicit folds, one concrete base variant and a model target"
                .into(),
        ));
    }
    request.validate_parameter_bindings(plan)?;
    let checkpoint = prepare_host_hpo_checkpoint(plan, request, options)?;
    let first = checkpoint.trials.len() as u32;
    let phase_index = request.phase_index(first);
    let mut tasks = Vec::new();
    let mut exhausted = false;
    for trial_index in first..request.trial_budget {
        if tasks.len() == max_workers || request.phase_index(trial_index) != phase_index {
            break;
        }
        let Some(params) = proposals.ask_in_phase(trial_index, phase_index)? else {
            exhausted = true;
            break;
        };
        let candidate_plan =
            host_hpo_worker_candidate_plan(plan, request, &checkpoint, trial_index, &params)
                .and_then(|candidate_plan| {
                    proposals.candidate_planned(trial_index, &candidate_plan)?;
                    Ok(candidate_plan)
                })
                .map_err(|error| {
                    proposals.stage_failure(
                        trial_index,
                        realization::FailureStage::CandidatePlanning,
                        error,
                    )
                })?;
        tasks.push(HostHpoWorkerTask {
            trial_index,
            phase_index,
            params,
            candidate_plan,
        });
    }
    let window_id = stable_json_fingerprint(&(
        &checkpoint.fingerprint,
        &options.data_fingerprint,
        &tasks,
        exhausted,
    ))?;
    Ok(HostHpoWorkerWindow {
        window_id,
        checkpoint,
        tasks,
        exhausted,
    })
}

/// Execute exactly one candidate within a worker-owned controller/provider
/// namespace. Core, not the optimizer or main-thread JavaScript, constructs
/// the fold reports and objective score.
pub fn evaluate_host_hpo_worker_task(
    task: &HostHpoWorkerTask,
    request: &HostHpoSearchRequest,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<HostHpoTrialEvidence> {
    request.validate_browser_execution()?;
    validate_structural_worker_task(task, request)?;
    task.candidate_plan.validate()?;
    let [variant] = task.candidate_plan.variants.as_slice() else {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO worker task requires exactly one candidate variant".into(),
        ));
    };
    if variant.variant_id.as_str() != format!("host_hpo:trial:{:010}", task.trial_index)
        || task.phase_index != request.phase_index(task.trial_index)
    {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO worker task trial/phase identity mismatch".into(),
        ));
    }
    let mut context = RunContext::new(
        RunId::new(format!("run:host_hpo:{}", task.trial_index))?,
        variant.seed.or(task.candidate_plan.campaign.root_seed),
    );
    context.variant_id = Some(variant.variant_id.clone());
    SequentialScheduler.execute_campaign_phase_with_data_provider(
        &task.candidate_plan,
        controllers,
        provider,
        &mut context,
        Phase::FitCv,
    )?;
    context.collect_cross_fold_validation_scores_for_target(
        plan_oof_partition_mode(&task.candidate_plan),
        request.metric,
        Some(&request.target_node),
    )?;
    let scores = context
        .build_score_set(task.candidate_plan.id.clone(), None)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation("browser HPO worker lost native scores".into())
        })?;
    let (score, objective_fold_scores, _) =
        host_hpo_score(&task.candidate_plan, request, &variant.variant_id, &scores)?;
    Ok(HostHpoTrialEvidence {
        trial_index: task.trial_index,
        params: task.params.clone(),
        score,
        variant_id: variant.variant_id.clone(),
        scores,
        objective_fold_scores,
        generated_view_manifest: host_hpo_provider_manifest(provider)?,
    })
}

/// One native FIT_CV fold evaluated by a candidate-local browser worker.
/// This is an intermediate score, not a terminal or aggregate OOF score.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoWorkerFoldResult {
    pub trial_index: u32,
    pub fold_index: u32,
    pub fold_id: FoldId,
    pub score: f64,
    pub scores: ScoreSet,
    pub data_fingerprint: String,
}

/// Execute one fold and return native report evidence before the next fold can
/// be scheduled. A worker never makes the pruning decision itself.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_host_hpo_worker_fold(
    task: &HostHpoWorkerTask,
    request: &HostHpoSearchRequest,
    fold_index: u32,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
    data_fingerprint: &str,
) -> Result<HostHpoWorkerFoldResult> {
    request.validate_browser_execution()?;
    validate_structural_worker_task(task, request)?;
    task.candidate_plan.validate()?;
    let [variant] = task.candidate_plan.variants.as_slice() else {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO worker fold requires one candidate variant".into(),
        ));
    };
    if variant.variant_id.as_str() != format!("host_hpo:trial:{:010}", task.trial_index)
        || task.phase_index != request.phase_index(task.trial_index)
    {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO worker fold trial/phase identity mismatch".into(),
        ));
    }
    let (scores, score) = SequentialScheduler.execute_host_hpo_worker_fold(
        &task.candidate_plan,
        controllers,
        provider,
        request,
        fold_index as usize,
    )?;
    let fold_id = task
        .candidate_plan
        .fold_set
        .as_ref()
        .expect("validated FoldSet")
        .folds[fold_index as usize]
        .fold_id
        .clone();
    Ok(HostHpoWorkerFoldResult {
        trial_index: task.trial_index,
        fold_index,
        fold_id,
        score,
        scores,
        data_fingerprint: data_fingerprint.to_owned(),
    })
}

/// Reject an altered worker fold before any optimizer feedback. The evidence
/// must contain exactly the declared native target report for this fold.
pub fn validate_host_hpo_worker_fold_result(
    task: &HostHpoWorkerTask,
    request: &HostHpoSearchRequest,
    data_fingerprint: &str,
    result: &HostHpoWorkerFoldResult,
) -> Result<()> {
    let target_node = request.target_for_plan(&task.candidate_plan)?;
    let fold = task
        .candidate_plan
        .fold_set
        .as_ref()
        .and_then(|set| set.folds.get(result.fold_index as usize))
        .ok_or_else(|| DagMlError::RuntimeValidation("worker fold index is out of range".into()))?;
    if result.trial_index != task.trial_index
        || result.fold_id != fold.fold_id
        || result.data_fingerprint != data_fingerprint
        || result.scores.plan_id != task.candidate_plan.id
        || !result.score.is_finite()
        || result.scores.reports.iter().any(|report| {
            report
                .fold_id
                .as_ref()
                .is_some_and(|id| id != &fold.fold_id)
                || report
                    .variant_id
                    .as_ref()
                    .is_some_and(|id| id != &task.candidate_plan.variants[0].variant_id)
        })
    {
        return Err(DagMlError::RuntimeValidation(
            "worker fold identity, data or score provenance mismatch".into(),
        ));
    }
    result.scores.validate()?;
    let reports = result
        .scores
        .reports
        .iter()
        .filter(|report| {
            report.producer_node == *target_node
                && report_matches_statistical_policy(report, &task.candidate_plan)
                && report.partition == PredictionPartition::Validation
                && report.fold_id.as_ref() == Some(&fold.fold_id)
        })
        .collect::<Vec<_>>();
    let [report] = reports.as_slice() else {
        return Err(DagMlError::RuntimeValidation(
            "worker fold lacks exactly one native target score".into(),
        ));
    };
    if host_hpo_metric(report, request.metric)? != result.score {
        return Err(DagMlError::RuntimeValidation(
            "worker fold score differs from native target report".into(),
        ));
    }
    Ok(())
}

/// Validate a contiguous prefix of fold feedback and calculate the exact
/// scalar passed to the optimizer's pruning callback.
pub fn host_hpo_worker_intermediate_score(
    task: &HostHpoWorkerTask,
    request: &HostHpoSearchRequest,
    data_fingerprint: &str,
    folds: &[HostHpoWorkerFoldResult],
) -> Result<f64> {
    if folds.is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "worker pruning feedback requires at least one fold".into(),
        ));
    }
    let mut scores = BTreeMap::new();
    for (index, result) in folds.iter().enumerate() {
        if result.fold_index as usize != index {
            return Err(DagMlError::RuntimeValidation(
                "worker pruning feedback has a noncontiguous fold prefix".into(),
            ));
        }
        validate_host_hpo_worker_fold_result(task, request, data_fingerprint, result)?;
        scores.insert(result.fold_id.as_str().to_owned(), result.score);
    }
    reduce_host_hpo_fold_scores(
        &scores,
        request
            .fold_score_reduction
            .unwrap_or(HostHpoFoldReduction::Best),
        request.direction,
    )
}

pub fn host_hpo_worker_pruned_evidence(
    task: &HostHpoWorkerTask,
    request: &HostHpoSearchRequest,
    data_fingerprint: &str,
    folds: &[HostHpoWorkerFoldResult],
) -> Result<HostHpoPrunedTrialEvidence> {
    host_hpo_worker_intermediate_score(task, request, data_fingerprint, folds)?;
    let mut scores = folds[0].scores.clone();
    for fold in &folds[1..] {
        scores.reports.extend(fold.scores.reports.iter().cloned());
    }
    scores.validate()?;
    Ok(HostHpoPrunedTrialEvidence {
        trial_index: task.trial_index,
        params: task.params.clone(),
        variant_id: task.candidate_plan.variants[0].variant_id.clone(),
        scores,
        intermediate_scores: folds.iter().map(|fold| fold.score).collect(),
        generated_view_manifest: None,
    })
}

/// A worker may finish in any order; the coordinator processes these keyed
/// results only after every launched candidate has terminalized.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostHpoWorkerResult {
    Complete {
        evidence: HostHpoTrialEvidence,
        data_fingerprint: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        fold_evidence: Vec<HostHpoWorkerFoldResult>,
    },
    Pruned {
        evidence: HostHpoPrunedTrialEvidence,
        data_fingerprint: String,
        fold_evidence: Vec<HostHpoWorkerFoldResult>,
    },
    Failed {
        trial_index: u32,
        error: String,
    },
}

impl HostHpoWorkerResult {
    fn trial_index(&self) -> u32 {
        match self {
            Self::Complete { evidence, .. } => evidence.trial_index,
            Self::Pruned { evidence, .. } => evidence.trial_index,
            Self::Failed { trial_index, .. } => *trial_index,
        }
    }
}

/// Validate all worker evidence, then publish terminal transitions in trial
/// order. A worker's scalar score is recomputed from its native ScoreSet; an
/// unmatched or altered candidate plan, trial ID, sample fold, or checkpoint
/// is rejected before any optimizer `tell`.
#[allow(clippy::too_many_arguments)]
pub fn complete_host_hpo_worker_window(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    options: &HostHpoResumeOptions,
    window: HostHpoWorkerWindow,
    results: Vec<HostHpoWorkerResult>,
    proposals: &mut dyn HostHpoProposalSource,
    progress: &mut dyn HostHpoProgress,
) -> Result<HostHpoSearchOutcome> {
    request.validate_browser_execution()?;
    let expected_checkpoint = prepare_host_hpo_checkpoint(plan, request, options)?;
    if stable_json_fingerprint(&expected_checkpoint)?
        != stable_json_fingerprint(&window.checkpoint)?
        || window.window_id
            != stable_json_fingerprint(&(
                &window.checkpoint.fingerprint,
                &options.data_fingerprint,
                &window.tasks,
                window.exhausted,
            ))?
        || results.len() != window.tasks.len()
    {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO window checkpoint, identity or result count mismatch".into(),
        ));
    }
    let first = window.checkpoint.trials.len() as u32;
    let phase = request.phase_index(first);
    for (offset, task) in window.tasks.iter().enumerate() {
        let expected = host_hpo_worker_candidate_plan(
            plan,
            request,
            &window.checkpoint,
            task.trial_index,
            &task.params,
        )?;
        if task.trial_index != first + offset as u32
            || task.phase_index != phase
            || task.phase_index != request.phase_index(task.trial_index)
            || stable_json_fingerprint(&task.candidate_plan)? != stable_json_fingerprint(&expected)?
        {
            return Err(DagMlError::RuntimeValidation(
                "browser HPO window candidate plan or phase mismatch".into(),
            ));
        }
    }
    let mut keyed = BTreeMap::new();
    for result in results {
        if keyed.insert(result.trial_index(), result).is_some() {
            return Err(DagMlError::RuntimeValidation(
                "browser HPO window has duplicate worker trial result".into(),
            ));
        }
    }
    if keyed.len() != window.tasks.len()
        || window
            .tasks
            .iter()
            .any(|task| !keyed.contains_key(&task.trial_index))
    {
        return Err(DagMlError::RuntimeValidation(
            "browser HPO window is missing a worker trial result".into(),
        ));
    }
    // Validate every worker response before mutating optimizer state.
    for task in &window.tasks {
        let target_node = request.target_for_plan(&task.candidate_plan)?;
        match keyed
            .get(&task.trial_index)
            .expect("checked result coverage")
        {
            HostHpoWorkerResult::Complete {
                evidence,
                data_fingerprint,
                fold_evidence,
            } => {
                if evidence.params != task.params
                    || evidence.variant_id != task.candidate_plan.variants[0].variant_id
                    || data_fingerprint != &options.data_fingerprint
                {
                    return Err(DagMlError::RuntimeValidation(
                        "browser HPO worker evidence proposal/variant mismatch".into(),
                    ));
                }
                host_hpo_candidate(plan, request, evidence)?;
                if request.progressive_pruning {
                    if fold_evidence.len()
                        != plan
                            .fold_set
                            .as_ref()
                            .expect("validated FoldSet")
                            .folds
                            .len()
                    {
                        return Err(DagMlError::RuntimeValidation(
                            "completed prunable worker lacks all fold feedback".into(),
                        ));
                    }
                    host_hpo_worker_intermediate_score(
                        task,
                        request,
                        &options.data_fingerprint,
                        fold_evidence,
                    )?;
                    for fold in fold_evidence {
                        let reports = evidence
                            .scores
                            .reports
                            .iter()
                            .filter(|report| {
                                report.producer_node == *target_node
                                    && report_matches_statistical_policy(report, plan)
                                    && report.partition == PredictionPartition::Validation
                                    && report.fold_id.as_ref() == Some(&fold.fold_id)
                            })
                            .collect::<Vec<_>>();
                        let [report] = reports.as_slice() else {
                            return Err(DagMlError::RuntimeValidation(
                                "completed worker lacks a feedback fold report".into(),
                            ));
                        };
                        if host_hpo_metric(report, request.metric)? != fold.score {
                            return Err(DagMlError::RuntimeValidation(
                                "completed worker changed a reported fold score".into(),
                            ));
                        }
                    }
                } else if !fold_evidence.is_empty() {
                    return Err(DagMlError::RuntimeValidation(
                        "nonprunable worker unexpectedly supplied fold feedback".into(),
                    ));
                }
            }
            HostHpoWorkerResult::Pruned {
                evidence,
                data_fingerprint,
                fold_evidence,
            } => {
                if !request.progressive_pruning
                    || data_fingerprint != &options.data_fingerprint
                    || evidence.params != task.params
                    || evidence.variant_id != task.candidate_plan.variants[0].variant_id
                {
                    return Err(DagMlError::RuntimeValidation(
                        "pruned worker proposal or data identity mismatch".into(),
                    ));
                }
                let observed = host_hpo_worker_pruned_evidence(
                    task,
                    request,
                    &options.data_fingerprint,
                    fold_evidence,
                )?;
                if stable_json_fingerprint(&observed)? != stable_json_fingerprint(evidence)? {
                    return Err(DagMlError::RuntimeValidation(
                        "pruned worker terminal differs from observed fold feedback".into(),
                    ));
                }
                if evidence.intermediate_scores.is_empty()
                    || evidence.intermediate_scores.len()
                        > plan
                            .fold_set
                            .as_ref()
                            .expect("validated FoldSet")
                            .folds
                            .len()
                {
                    return Err(DagMlError::RuntimeValidation(
                        "pruned worker has an invalid fold prefix".into(),
                    ));
                }
                evidence.scores.validate()?;
                for (index, score) in evidence.intermediate_scores.iter().enumerate() {
                    let fold = &plan.fold_set.as_ref().expect("validated FoldSet").folds[index];
                    let reports = evidence
                        .scores
                        .reports
                        .iter()
                        .filter(|report| {
                            report.producer_node == *target_node
                                && report_matches_statistical_policy(report, plan)
                                && report.partition == PredictionPartition::Validation
                                && report.fold_id.as_ref() == Some(&fold.fold_id)
                        })
                        .collect::<Vec<_>>();
                    let [report] = reports.as_slice() else {
                        return Err(DagMlError::RuntimeValidation(
                            "pruned worker lacks a native fold report".into(),
                        ));
                    };
                    if host_hpo_metric(report, request.metric)? != *score {
                        return Err(DagMlError::RuntimeValidation(
                            "pruned worker fold feedback differs from native score".into(),
                        ));
                    }
                }
            }
            HostHpoWorkerResult::Failed { error, .. } if error.trim().is_empty() => {
                return Err(DagMlError::RuntimeValidation(
                    "browser HPO worker failure requires an error".into(),
                ));
            }
            HostHpoWorkerResult::Failed { .. } => {}
        }
    }
    let mut checkpoint = window.checkpoint;
    let mut cancel_requested = false;
    let mut failed = false;
    for task in &window.tasks {
        let result = keyed
            .remove(&task.trial_index)
            .expect("checked result coverage");
        let terminal = match &result {
            HostHpoWorkerResult::Complete { evidence, .. } => HostHpoTerminalTrial::Complete {
                evidence: evidence.clone(),
            },
            HostHpoWorkerResult::Pruned { evidence, .. } => HostHpoTerminalTrial::Pruned {
                evidence: evidence.clone(),
            },
            HostHpoWorkerResult::Failed { error, .. } => {
                failed = true;
                HostHpoTerminalTrial::Failed {
                    trial_index: task.trial_index,
                    params: task.params.clone(),
                    variant_id: task.candidate_plan.variants[0].variant_id.clone(),
                    error: error.clone(),
                    generated_view_manifest: None,
                }
            }
        };
        let status = if failed {
            HostHpoSearchStatus::Failed
        } else if task.trial_index + 1 == request.trial_budget {
            HostHpoSearchStatus::Completed
        } else {
            HostHpoSearchStatus::Running
        };
        let mut prepared = checkpoint.clone();
        prepared.trials.push(terminal);
        prepared.seal()?;
        prepare_host_hpo_checkpoint(
            plan,
            request,
            &HostHpoResumeOptions {
                data_fingerprint: options.data_fingerprint.clone(),
                checkpoint: Some(prepared.clone()),
            },
        )?;
        progress.prepare_terminal(&prepared, status)?;
        match result {
            HostHpoWorkerResult::Complete { evidence, .. } => {
                proposals.tell(task.trial_index, evidence.score)?;
            }
            HostHpoWorkerResult::Pruned { .. } => {
                proposals.pruned(task.trial_index)?;
            }
            HostHpoWorkerResult::Failed { error, .. } => {
                proposals.fail(task.trial_index, &error)?;
            }
        }
        checkpoint = prepared;
        if !progress.checkpoint(&checkpoint, status)? && status == HostHpoSearchStatus::Running {
            cancel_requested = true;
        }
    }
    let status = if failed {
        HostHpoSearchStatus::Failed
    } else if cancel_requested {
        HostHpoSearchStatus::Cancelled
    } else if window.exhausted {
        HostHpoSearchStatus::Exhausted
    } else if checkpoint.trials.len() == request.trial_budget as usize {
        HostHpoSearchStatus::Completed
    } else {
        HostHpoSearchStatus::Running
    };
    if matches!(
        status,
        HostHpoSearchStatus::Cancelled | HostHpoSearchStatus::Exhausted
    ) {
        progress.checkpoint(&checkpoint, status)?;
    }
    let trials = checkpoint
        .trials
        .iter()
        .filter_map(|trial| match trial {
            HostHpoTerminalTrial::Complete { evidence } => Some(evidence.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let pruned_trials = checkpoint
        .trials
        .iter()
        .filter_map(|trial| match trial {
            HostHpoTerminalTrial::Pruned { evidence } => Some(evidence.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let result = if trials.is_empty() {
        None
    } else {
        let candidates = trials
            .iter()
            .map(|trial| host_hpo_candidate(plan, request, trial))
            .collect::<Result<Vec<_>>>()?;
        let policy = SelectionPolicy {
            requested_rank: None,
            id: "select:host_hpo".into(),
            metric: SelectionMetric {
                name: request.metric.name().into(),
                objective: request.direction,
            },
            required_metric_level: None,
            require_finite: true,
            evaluation_scope: None,
            refit_slot_plan: None,
            stacking_fit_contract: None,
            reduction_id: None,
        };
        let selected = select_candidate(&policy, &candidates)?;
        let winner = trials
            .iter()
            .find(|trial| trial.variant_id.as_str() == selected.selected_candidate_id)
            .expect("selected observed candidate");
        Some(HostHpoSearchResult {
            profile: "host_optimizer_search_v1".into(),
            portable: false,
            request_fingerprint: stable_json_fingerprint(request)?,
            graph_fingerprint: plan.graph_fingerprint.clone(),
            controller_fingerprint: plan.controller_fingerprint.clone(),
            campaign_fingerprint: stable_json_fingerprint(&plan.campaign)?,
            fold_set_fingerprint: stable_json_fingerprint(
                plan.fold_set.as_ref().expect("validated host HPO folds"),
            )?,
            selected_trial_index: winner.trial_index,
            selected_params: winner.params.clone(),
            trials,
            pruned_trials,
        })
    };
    Ok(HostHpoSearchOutcome {
        result,
        status,
        checkpoint: Some(checkpoint),
    })
}

impl SequentialScheduler {
    /// Execute candidate FIT_CV only; the caller's outer scope owns final fitting.
    /// Every trial gets an isolated context and the same already-attested folds.
    /// A controller failure propagates immediately: no retry or replacement score.
    pub fn execute_host_hpo_search(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
    ) -> Result<HostHpoSearchResult> {
        self.execute_host_hpo_search_inner(
            plan,
            controllers,
            provider,
            None,
            None,
            request,
            proposals,
            None,
        )?
        .result
        .ok_or_else(|| DagMlError::RuntimeValidation("host HPO has no successful candidate".into()))
    }

    /// Resume terminal native evidence and publish progress between trials.
    /// The budget is a total across calls; SELECT includes historical winners.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_resumable_host_hpo_search(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        options: &HostHpoResumeOptions,
        progress: &mut dyn HostHpoProgress,
    ) -> Result<HostHpoSearchOutcome> {
        self.execute_host_hpo_search_inner(
            plan,
            controllers,
            provider,
            None,
            None,
            request,
            proposals,
            Some((options, progress)),
        )
    }

    /// Candidate-local provider variant of the durable host search. The
    /// factory is invoked only for trials that are actually evaluated, never
    /// for terminal trials restored from a checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_resumable_host_hpo_search_with_provider_factory(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        options: &HostHpoResumeOptions,
        progress: &mut dyn HostHpoProgress,
    ) -> Result<HostHpoSearchOutcome> {
        self.execute_host_hpo_search_inner(
            plan,
            controllers,
            provider,
            Some(provider_factory),
            None,
            request,
            proposals,
            Some((options, progress)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_resumable_host_hpo_search_with_candidate_factories(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        controller_factory: &dyn HostHpoCandidateControllerFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        options: &HostHpoResumeOptions,
        progress: &mut dyn HostHpoProgress,
    ) -> Result<HostHpoSearchOutcome> {
        self.execute_host_hpo_search_inner(
            plan,
            controllers,
            provider,
            Some(provider_factory),
            Some(controller_factory),
            request,
            proposals,
            Some((options, progress)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_host_hpo_search_inner(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        provider_factory: Option<&dyn HostHpoCandidateProviderFactory>,
        controller_factory: Option<&dyn HostHpoCandidateControllerFactory>,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        mut durable: Option<(&HostHpoResumeOptions, &mut dyn HostHpoProgress)>,
    ) -> Result<HostHpoSearchOutcome> {
        request.validate_parallel_execution_for_plan(1, plan)?;
        plan.validate()?;
        if request.trial_budget == 0 || request.optimizer_descriptor.is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires a positive budget and explicit optimizer descriptor".into(),
            ));
        }
        if !request.phase_trial_budgets.is_empty()
            && (request.phase_trial_budgets.contains(&0)
                || request
                    .phase_trial_budgets
                    .iter()
                    .try_fold(0u32, |total, budget| total.checked_add(*budget))
                    != Some(request.trial_budget))
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO phase budgets must be positive and sum to trial_budget".into(),
            ));
        }
        let folds = plan.fold_set.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation("host HPO requires explicit evaluation folds".into())
        })?;
        if plan.variants.len() != 1
            || !plan.variants[0].choices.is_empty()
            || plan
                .node_plans
                .get(&request.target_node)
                .is_none_or(|node| node.kind != NodeKind::Model)
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires one concrete base variant and a model target".into(),
            ));
        }
        request.validate_parameter_bindings(plan)?;
        let mut checkpoint = durable
            .as_ref()
            .map(|(options, _)| prepare_host_hpo_checkpoint(plan, request, options))
            .transpose()?;
        let mut trials = checkpoint
            .as_ref()
            .map(|checkpoint| {
                checkpoint
                    .trials
                    .iter()
                    .filter_map(|trial| match trial {
                        HostHpoTerminalTrial::Complete { evidence } => Some(evidence.clone()),
                        HostHpoTerminalTrial::Pruned { .. }
                        | HostHpoTerminalTrial::Failed { .. } => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut pruned_trials = checkpoint
            .as_ref()
            .map(|checkpoint| {
                checkpoint
                    .trials
                    .iter()
                    .filter_map(|trial| match trial {
                        HostHpoTerminalTrial::Pruned { evidence } => Some(evidence.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut candidates = trials
            .iter()
            .map(|trial| host_hpo_candidate(plan, request, trial))
            .collect::<Result<Vec<_>>>()?;
        let history_len = checkpoint
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.trials.len() as u32);
        let mut status = if history_len == request.trial_budget {
            HostHpoSearchStatus::Completed
        } else {
            HostHpoSearchStatus::Running
        };
        if let (Some(checkpoint), Some((_, progress))) = (&checkpoint, &mut durable) {
            if !progress.checkpoint(checkpoint, status)? && status == HostHpoSearchStatus::Running {
                status = HostHpoSearchStatus::Cancelled;
            }
        }
        for trial_index in history_len..request.trial_budget {
            if status == HostHpoSearchStatus::Cancelled {
                break;
            }
            let Some(params) =
                proposals.ask_in_phase(trial_index, request.phase_index(trial_index))?
            else {
                status = HostHpoSearchStatus::Exhausted;
                break;
            };
            let candidate_plan = host_hpo_candidate_plan(
                plan,
                request,
                trial_index,
                &params,
                checkpoint
                    .as_ref()
                    .map(|saved| saved.binding.objective_fingerprint.as_str()),
            )
            .and_then(|candidate_plan| {
                proposals.candidate_planned(trial_index, &candidate_plan)?;
                Ok(candidate_plan)
            })
            .map_err(|error| {
                proposals.stage_failure(
                    trial_index,
                    realization::FailureStage::CandidatePlanning,
                    error,
                )
            })?;
            let variant = candidate_plan.variants[0].clone();
            let mut context = RunContext::new(
                RunId::new(format!("run:host_hpo:{trial_index}")).map_err(|error| {
                    proposals.stage_failure(
                        trial_index,
                        realization::FailureStage::RunContextCreation,
                        error,
                    )
                })?,
                variant.seed.or(plan.campaign.root_seed),
            );
            context.variant_id = Some(variant.variant_id.clone());
            context.resource_limits = crate::python_torch_profile::torch_resources(&candidate_plan);
            let candidate_provider = provider_factory
                .map(|factory| factory.create(trial_index))
                .transpose()
                .map_err(|error| {
                    proposals.stage_failure(
                        trial_index,
                        realization::FailureStage::ProviderCreation,
                        error,
                    )
                })?;
            let provider = candidate_provider
                .as_deref()
                .map(|provider| provider as &dyn RuntimeDataProvider)
                .unwrap_or(provider);
            let candidate_controllers = controller_factory
                .map(|factory| factory.create(trial_index))
                .transpose()
                .map_err(|error| {
                    proposals.stage_failure(
                        trial_index,
                        realization::FailureStage::ControllerCreation,
                        error,
                    )
                })?;
            let controllers = candidate_controllers.as_ref().unwrap_or(controllers);
            let evaluated: Result<HostHpoEvaluation> = (|| {
                if request.progressive_pruning {
                    let mut intermediates = Vec::new();
                    let pruned = self.execute_host_hpo_candidate_fit_cv(
                        &candidate_plan,
                        controllers,
                        provider,
                        &mut context,
                        request,
                        &mut |step, score| {
                            intermediates.push(score);
                            let fold_scores = folds
                                .folds
                                .iter()
                                .zip(&intermediates)
                                .map(|(fold, value)| (fold.fold_id.as_str().to_owned(), *value))
                                .collect::<BTreeMap<_, _>>();
                            let aggregate = reduce_host_hpo_fold_scores(
                                &fold_scores,
                                request
                                    .fold_score_reduction
                                    .unwrap_or(HostHpoFoldReduction::Best),
                                request.direction,
                            )?;
                            proposals.report_intermediate(trial_index, step, aggregate)
                        },
                    )?;
                    if pruned {
                        let scores =
                            context
                                .build_score_set(plan.id.clone(), None)
                                .ok_or_else(|| {
                                    DagMlError::RuntimeValidation(
                                        "pruned host HPO trial lost native fold score evidence"
                                            .into(),
                                    )
                                })?;
                        return Ok(HostHpoEvaluation::Pruned(HostHpoPrunedTrialEvidence {
                            trial_index,
                            params: params.clone(),
                            variant_id: variant.variant_id.clone(),
                            scores,
                            intermediate_scores: intermediates,
                            generated_view_manifest: host_hpo_provider_manifest(provider)?,
                        }));
                    }
                } else {
                    self.execute_campaign_phase_with_data_provider(
                        &candidate_plan,
                        controllers,
                        provider,
                        &mut context,
                        Phase::FitCv,
                    )?;
                }
                context.collect_cross_fold_validation_scores_for_target(
                    plan_oof_partition_mode(plan),
                    request.metric,
                    Some(&request.target_node),
                )?;
                let scores = context
                    .build_score_set(plan.id.clone(), None)
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation("host HPO lost native score evidence".into())
                    })?;
                let (score, objective_fold_scores, candidate) =
                    host_hpo_score(&candidate_plan, request, &variant.variant_id, &scores)?;
                Ok(HostHpoEvaluation::Complete(
                    HostHpoTrialEvidence {
                        trial_index,
                        params: params.clone(),
                        score,
                        variant_id: variant.variant_id.clone(),
                        scores,
                        objective_fold_scores,
                        generated_view_manifest: host_hpo_provider_manifest(provider)?,
                    },
                    candidate,
                ))
            })();
            let evaluated = match evaluated {
                Ok(result) => result,
                Err(error) => {
                    let error = proposals.stage_failure(
                        trial_index,
                        realization::FailureStage::Evaluation,
                        error,
                    );
                    if let (Some(checkpoint), Some((_, progress))) = (&mut checkpoint, &mut durable)
                    {
                        let mut prepared = checkpoint.clone();
                        prepared.trials.push(HostHpoTerminalTrial::Failed {
                            trial_index,
                            params: params.clone(),
                            variant_id: variant.variant_id.clone(),
                            error: error.to_string(),
                            generated_view_manifest: provider
                                .generated_view_manifest_on_failure()?,
                        });
                        prepared.seal()?;
                        progress.prepare_terminal(&prepared, HostHpoSearchStatus::Failed)?;
                        proposals.fail(trial_index, &error.to_string())?;
                        *checkpoint = prepared;
                        progress.checkpoint(checkpoint, HostHpoSearchStatus::Failed)?;
                    }
                    return Err(error);
                }
            };
            if let HostHpoEvaluation::Pruned(evidence) = evaluated {
                status = if trial_index + 1 == request.trial_budget {
                    HostHpoSearchStatus::Completed
                } else {
                    HostHpoSearchStatus::Running
                };
                let prospective =
                    if let (Some(checkpoint), Some((_, progress))) = (&checkpoint, &mut durable) {
                        let mut prepared = checkpoint.clone();
                        prepared.trials.push(HostHpoTerminalTrial::Pruned {
                            evidence: evidence.clone(),
                        });
                        prepared.seal()?;
                        progress.prepare_terminal(&prepared, status)?;
                        Some(prepared)
                    } else {
                        None
                    };
                proposals.pruned(trial_index)?;
                pruned_trials.push(evidence.clone());
                if let (Some(checkpoint), Some((_, progress))) = (&mut checkpoint, &mut durable) {
                    *checkpoint = prospective.expect("durable search prepared terminal trial");
                    if !progress.checkpoint(checkpoint, status)?
                        && status == HostHpoSearchStatus::Running
                    {
                        status = HostHpoSearchStatus::Cancelled;
                    }
                }
                continue;
            }
            let HostHpoEvaluation::Complete(evidence, candidate) = evaluated else {
                unreachable!("pruned host HPO candidate was handled above")
            };
            status = if trial_index + 1 == request.trial_budget {
                HostHpoSearchStatus::Completed
            } else {
                HostHpoSearchStatus::Running
            };
            let prospective =
                if let (Some(checkpoint), Some((_, progress))) = (&checkpoint, &mut durable) {
                    let mut prepared = checkpoint.clone();
                    prepared.trials.push(HostHpoTerminalTrial::Complete {
                        evidence: evidence.clone(),
                    });
                    prepared.seal()?;
                    progress.prepare_terminal(&prepared, status)?;
                    Some(prepared)
                } else {
                    None
                };
            proposals.tell(trial_index, evidence.score)?;
            candidates.push(candidate);
            trials.push(evidence.clone());
            if let (Some(checkpoint), Some((_, progress))) = (&mut checkpoint, &mut durable) {
                *checkpoint = prospective.expect("durable search prepared terminal trial");
                if !progress.checkpoint(checkpoint, status)?
                    && status == HostHpoSearchStatus::Running
                {
                    status = HostHpoSearchStatus::Cancelled;
                }
            }
        }
        if matches!(
            status,
            HostHpoSearchStatus::Cancelled | HostHpoSearchStatus::Exhausted
        ) {
            if let (Some(checkpoint), Some((_, progress))) = (&checkpoint, &mut durable) {
                progress.checkpoint(checkpoint, status)?;
            }
        }
        if candidates.is_empty() {
            return Ok(HostHpoSearchOutcome {
                result: None,
                status,
                checkpoint,
            });
        }
        let policy = SelectionPolicy {
            requested_rank: None,
            id: "select:host_hpo".into(),
            metric: SelectionMetric {
                name: request.metric.name().into(),
                objective: request.direction,
            },
            required_metric_level: None,
            require_finite: true,
            evaluation_scope: None,
            refit_slot_plan: None,
            stacking_fit_contract: None,
            reduction_id: None,
        };
        let selected = select_candidate(&policy, &candidates)?;
        let winner = trials
            .iter()
            .find(|trial| trial.variant_id.as_str() == selected.selected_candidate_id)
            .expect("selection returns an observed candidate");
        Ok(HostHpoSearchOutcome {
            result: Some(HostHpoSearchResult {
                profile: "host_optimizer_search_v1".into(),
                portable: false,
                request_fingerprint: stable_json_fingerprint(request)?,
                graph_fingerprint: plan.graph_fingerprint.clone(),
                controller_fingerprint: plan.controller_fingerprint.clone(),
                campaign_fingerprint: stable_json_fingerprint(&plan.campaign)?,
                fold_set_fingerprint: stable_json_fingerprint(folds)?,
                selected_trial_index: winner.trial_index,
                selected_params: winner.params.clone(),
                trials,
                pruned_trials,
            }),
            status,
            checkpoint,
        })
    }

    /// Non-durable search with a separate provider and handle namespace for
    /// every evaluated candidate.
    pub fn execute_host_hpo_search_with_provider_factory(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
    ) -> Result<HostHpoSearchResult> {
        self.execute_host_hpo_search_inner(
            plan,
            controllers,
            provider,
            Some(provider_factory),
            None,
            request,
            proposals,
            None,
        )?
        .result
        .ok_or_else(|| DagMlError::RuntimeValidation("host HPO has no successful candidate".into()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_host_hpo_search_with_candidate_factories(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        controller_factory: &dyn HostHpoCandidateControllerFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
    ) -> Result<HostHpoSearchResult> {
        self.execute_host_hpo_search_inner(
            plan,
            controllers,
            provider,
            Some(provider_factory),
            Some(controller_factory),
            request,
            proposals,
            None,
        )?
        .result
        .ok_or_else(|| DagMlError::RuntimeValidation("host HPO has no successful candidate".into()))
    }

    /// Evaluate bounded candidate windows while the coordinator alone owns all
    /// optimizer transitions. Intermediate fold reports cross a channel back
    /// to that coordinator; no optimizer object enters a worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_parallel_host_hpo_search_with_candidate_factories(
        &self,
        plan: &ExecutionPlan,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        controller_factory: &dyn HostHpoCandidateControllerFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        max_parallel_trials: usize,
    ) -> Result<HostHpoSearchResult> {
        self.execute_parallel_host_hpo_search_inner(
            plan,
            provider_factory,
            controller_factory,
            request,
            proposals,
            max_parallel_trials,
            None,
        )?
        .result
        .ok_or_else(|| DagMlError::RuntimeValidation("host HPO has no successful candidate".into()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_resumable_parallel_host_hpo_search_with_candidate_factories(
        &self,
        plan: &ExecutionPlan,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        controller_factory: &dyn HostHpoCandidateControllerFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        max_parallel_trials: usize,
        options: &HostHpoResumeOptions,
        progress: &mut dyn HostHpoProgress,
    ) -> Result<HostHpoSearchOutcome> {
        self.execute_parallel_host_hpo_search_inner(
            plan,
            provider_factory,
            controller_factory,
            request,
            proposals,
            max_parallel_trials,
            Some((options, progress)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_parallel_host_hpo_search_inner(
        &self,
        plan: &ExecutionPlan,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        controller_factory: &dyn HostHpoCandidateControllerFactory,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        max_parallel_trials: usize,
        mut durable: Option<(&HostHpoResumeOptions, &mut dyn HostHpoProgress)>,
    ) -> Result<HostHpoSearchOutcome> {
        if max_parallel_trials <= 1 {
            return Err(DagMlError::RuntimeValidation(
                "parallel host HPO requires at least two workers".into(),
            ));
        }
        let parallel_resources =
            request.validate_parallel_execution_for_plan(max_parallel_trials, plan)?;
        plan.validate()?;
        if request.trial_budget == 0 || request.optimizer_descriptor.is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires a positive budget and explicit optimizer descriptor".into(),
            ));
        }
        if !request.phase_trial_budgets.is_empty()
            && (request.phase_trial_budgets.contains(&0)
                || request
                    .phase_trial_budgets
                    .iter()
                    .try_fold(0u32, |sum, count| sum.checked_add(*count))
                    != Some(request.trial_budget))
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO phase budgets must be positive and sum to trial_budget".into(),
            ));
        }
        let folds = plan.fold_set.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation("host HPO requires explicit evaluation folds".into())
        })?;
        if plan.variants.len() != 1
            || !plan.variants[0].choices.is_empty()
            || plan
                .node_plans
                .get(&request.target_node)
                .is_none_or(|node| node.kind != NodeKind::Model)
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires one concrete base variant and a model target".into(),
            ));
        }
        request.validate_parameter_bindings(plan)?;
        let mut checkpoint = durable
            .as_ref()
            .map(|(options, _)| prepare_host_hpo_checkpoint(plan, request, options))
            .transpose()?;
        let mut trials = checkpoint
            .as_ref()
            .map(|saved| {
                saved
                    .trials
                    .iter()
                    .filter_map(|trial| {
                        if let HostHpoTerminalTrial::Complete { evidence } = trial {
                            Some(evidence.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut pruned_trials = checkpoint
            .as_ref()
            .map(|saved| {
                saved
                    .trials
                    .iter()
                    .filter_map(|trial| {
                        if let HostHpoTerminalTrial::Pruned { evidence } = trial {
                            Some(evidence.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut candidates = trials
            .iter()
            .map(|trial| host_hpo_candidate(plan, request, trial))
            .collect::<Result<Vec<_>>>()?;
        let mut next = checkpoint
            .as_ref()
            .map_or(0, |saved| saved.trials.len() as u32);
        let mut status = if next == request.trial_budget {
            HostHpoSearchStatus::Completed
        } else {
            HostHpoSearchStatus::Running
        };
        if let (Some(saved), Some((_, progress))) = (&checkpoint, &mut durable) {
            if !progress.checkpoint(saved, status)? && status == HostHpoSearchStatus::Running {
                status = HostHpoSearchStatus::Cancelled;
            }
        }
        while next < request.trial_budget && status == HostHpoSearchStatus::Running {
            let phase = request.phase_index(next);
            let mut pending = Vec::new();
            let mut exhausted = false;
            while pending.len() < max_parallel_trials
                && next < request.trial_budget
                && request.phase_index(next) == phase
            {
                let Some(params) = proposals.ask_in_phase(next, phase)? else {
                    exhausted = true;
                    break;
                };
                let candidate_plan = host_hpo_candidate_plan(
                    plan,
                    request,
                    next,
                    &params,
                    checkpoint
                        .as_ref()
                        .map(|saved| saved.binding.objective_fingerprint.as_str()),
                )
                .and_then(|candidate_plan| {
                    proposals.candidate_planned(next, &candidate_plan)?;
                    Ok(candidate_plan)
                })
                .map_err(|error| {
                    proposals.stage_failure(
                        next,
                        realization::FailureStage::CandidatePlanning,
                        error,
                    )
                })?;
                let variant = candidate_plan.variants[0].clone();
                let mut context = RunContext::new(
                    RunId::new(format!("run:host_hpo:{next}")).map_err(|error| {
                        proposals.stage_failure(
                            next,
                            realization::FailureStage::RunContextCreation,
                            error,
                        )
                    })?,
                    variant.seed.or(plan.campaign.root_seed),
                );
                context.resource_limits = parallel_resources.clone();
                context.variant_id = Some(variant.variant_id.clone());
                pending.push((
                    next,
                    params,
                    variant,
                    candidate_plan,
                    context,
                    provider_factory.create(next).map_err(|error| {
                        proposals.stage_failure(
                            next,
                            realization::FailureStage::ProviderCreation,
                            error,
                        )
                    })?,
                    controller_factory.create(next).map_err(|error| {
                        proposals.stage_failure(
                            next,
                            realization::FailureStage::ControllerCreation,
                            error,
                        )
                    })?,
                ));
                next += 1;
            }
            let pending_params = pending
                .iter()
                .map(|(index, params, ..)| (*index, params.clone()))
                .collect::<BTreeMap<_, _>>();
            let count = pending.len();
            if count == 0 {
                status = HostHpoSearchStatus::Exhausted;
                break;
            }
            let (events_tx, events_rx) = std::sync::mpsc::channel();
            let mut completed = std::thread::scope(|scope| {
                let handles = pending.into_iter().map(|(trial_index, params, variant,
                    candidate_plan, mut context, provider, controllers)| {
                    let tx = events_tx.clone();
                    let handle = scope.spawn(move || {
                        let evaluated: Result<HostHpoEvaluation> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if request.progressive_pruning {
                                let mut intermediates = Vec::new();
                                let pruned = self.execute_host_hpo_candidate_fit_cv(
                                    &candidate_plan, &controllers, provider.as_ref(), &mut context,
                                    request, &mut |step, score| {
                                        intermediates.push(score);
                                        let fold_scores = folds.folds.iter().zip(&intermediates)
                                            .map(|(fold, value)| (fold.fold_id.as_str().to_owned(), *value))
                                            .collect::<BTreeMap<_, _>>();
                                        let aggregate = reduce_host_hpo_fold_scores(
                                            &fold_scores,
                                            request.fold_score_reduction.unwrap_or(HostHpoFoldReduction::Best),
                                            request.direction,
                                        )?;
                                        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
                                        tx.send(ParallelHostHpoEvent::Intermediate {
                                            trial_index, step, score: aggregate, reply: reply_tx,
                                        }).map_err(|_| DagMlError::RuntimeValidation(
                                            "host HPO pruning coordinator disconnected".into()))?;
                                        reply_rx.recv().map_err(|_| DagMlError::RuntimeValidation(
                                            "host HPO pruning coordinator dropped feedback".into()))?
                                    },
                                )?;
                                if pruned {
                                    let scores = context.build_score_set(plan.id.clone(), None)
                                        .ok_or_else(|| DagMlError::RuntimeValidation(
                                            "pruned host HPO trial lost native fold score evidence".into()))?;
                                    return Ok(HostHpoEvaluation::Pruned(HostHpoPrunedTrialEvidence {
                                        trial_index, params, variant_id: variant.variant_id,
                                        scores, intermediate_scores: intermediates,
                                        generated_view_manifest: host_hpo_provider_manifest(provider.as_ref())?,
                                    }));
                                }
                            } else {
                                self.execute_campaign_phase_with_data_provider(&candidate_plan,
                                    &controllers, provider.as_ref(), &mut context, Phase::FitCv)?;
                            }
                            context.collect_cross_fold_validation_scores_for_target(plan_oof_partition_mode(plan), request.metric, Some(&request.target_node))?;
                            let scores = context.build_score_set(plan.id.clone(), None)
                                .ok_or_else(|| DagMlError::RuntimeValidation(
                                    "host HPO lost native score evidence".into()))?;
                            let (score, objective_fold_scores, candidate) =
                                host_hpo_score(&candidate_plan, request, &variant.variant_id, &scores)?;
                            Ok(HostHpoEvaluation::Complete(HostHpoTrialEvidence {
                                trial_index, params, score, variant_id: variant.variant_id,
                                scores, objective_fold_scores,
                                generated_view_manifest: host_hpo_provider_manifest(provider.as_ref())?,
                            }, candidate))
                        })).unwrap_or_else(|_| Err(DagMlError::RuntimeValidation(
                            "host HPO candidate worker panicked".into())));
                        let _ = tx.send(ParallelHostHpoEvent::Completed {
                            trial_index,
                            evaluated: Box::new(evaluated),
                        });
                    });
                    (trial_index, handle)
                }).collect::<Vec<_>>();
                drop(events_tx);
                let mut completed = BTreeMap::new();
                // Pruners observe a stable trial/step order even when workers
                // finish at different times. Each worker waits for its reply.
                let mut active_trial = next - count as u32;
                let mut buffered: BTreeMap<u32, std::collections::VecDeque<ParallelHostHpoEvent>> =
                    BTreeMap::new();
                while completed.len() < count {
                    let Ok(event) = events_rx.recv() else {
                        break;
                    };
                    let trial_index = match &event {
                        ParallelHostHpoEvent::Intermediate { trial_index, .. }
                        | ParallelHostHpoEvent::Completed { trial_index, .. } => *trial_index,
                    };
                    buffered.entry(trial_index).or_default().push_back(event);
                    while let Some(event) = buffered
                        .get_mut(&active_trial)
                        .and_then(|queue| queue.pop_front())
                    {
                        match event {
                            ParallelHostHpoEvent::Intermediate {
                                trial_index,
                                step,
                                score,
                                reply,
                            } => {
                                let decision =
                                    proposals.report_intermediate(trial_index, step, score);
                                let _ = reply.send(decision);
                            }
                            ParallelHostHpoEvent::Completed {
                                trial_index,
                                evaluated,
                            } => {
                                completed.insert(trial_index, *evaluated);
                                active_trial += 1;
                            }
                        }
                    }
                }
                for (trial_index, handle) in handles {
                    if handle.join().is_err() {
                        completed.insert(
                            trial_index,
                            Err(DagMlError::RuntimeValidation(
                                "host HPO candidate worker panicked".into(),
                            )),
                        );
                    }
                }
                completed
            });
            let batch_manifest = provider_factory
                .batch_generated_view_manifest(completed.values().any(Result::is_ok))?;
            let mut first_error = None;
            let mut cancel_requested = false;
            for trial_index in next - count as u32..next {
                let evaluated = completed.remove(&trial_index).unwrap_or_else(|| {
                    Err(DagMlError::RuntimeValidation(
                        "host HPO candidate worker returned no result".into(),
                    ))
                });
                let evaluated = evaluated.map(|outcome| match outcome {
                    HostHpoEvaluation::Complete(mut evidence, candidate) => {
                        if let Some(manifest) = &batch_manifest {
                            evidence.generated_view_manifest = Some(manifest.clone());
                        }
                        HostHpoEvaluation::Complete(evidence, candidate)
                    }
                    HostHpoEvaluation::Pruned(mut evidence) => {
                        if let Some(manifest) = &batch_manifest {
                            evidence.generated_view_manifest = Some(manifest.clone());
                        }
                        HostHpoEvaluation::Pruned(evidence)
                    }
                });
                let evaluated = evaluated.map_err(|error| {
                    proposals.stage_failure(
                        trial_index,
                        realization::FailureStage::Evaluation,
                        error,
                    )
                });
                let trial_status = if evaluated.is_err() || first_error.is_some() {
                    HostHpoSearchStatus::Failed
                } else if trial_index + 1 == request.trial_budget {
                    HostHpoSearchStatus::Completed
                } else {
                    HostHpoSearchStatus::Running
                };
                let prospective = if let Some(saved) = &checkpoint {
                    let terminal = match &evaluated {
                        Ok(HostHpoEvaluation::Complete(evidence, _)) => {
                            HostHpoTerminalTrial::Complete {
                                evidence: evidence.clone(),
                            }
                        }
                        Ok(HostHpoEvaluation::Pruned(evidence)) => HostHpoTerminalTrial::Pruned {
                            evidence: evidence.clone(),
                        },
                        Err(error) => HostHpoTerminalTrial::Failed {
                            trial_index,
                            params: pending_params
                                .get(&trial_index)
                                .expect("pending proposal")
                                .clone(),
                            variant_id: VariantId::new(format!(
                                "host_hpo:trial:{trial_index:010}"
                            ))?,
                            error: error.to_string(),
                            generated_view_manifest: batch_manifest.clone(),
                        },
                    };
                    let mut prepared = saved.clone();
                    prepared.trials.push(terminal);
                    prepared.seal()?;
                    if let Some((_, progress)) = &mut durable {
                        progress.prepare_terminal(&prepared, trial_status)?;
                    }
                    Some(prepared)
                } else {
                    None
                };
                match evaluated {
                    Ok(HostHpoEvaluation::Complete(evidence, candidate)) => {
                        proposals.tell(trial_index, evidence.score)?;
                        candidates.push(candidate);
                        trials.push(evidence);
                    }
                    Ok(HostHpoEvaluation::Pruned(evidence)) => {
                        proposals.pruned(trial_index)?;
                        pruned_trials.push(evidence);
                    }
                    Err(error) => {
                        proposals.fail(trial_index, &error.to_string())?;
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
                checkpoint = prospective;
                if let (Some(saved), Some((_, progress))) = (&mut checkpoint, &mut durable) {
                    if !progress.checkpoint(saved, trial_status)?
                        && trial_status == HostHpoSearchStatus::Running
                    {
                        cancel_requested = true;
                    }
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
            if cancel_requested {
                status = HostHpoSearchStatus::Cancelled;
            } else if exhausted {
                status = HostHpoSearchStatus::Exhausted;
            } else if next == request.trial_budget {
                status = HostHpoSearchStatus::Completed;
            }
        }
        if matches!(
            status,
            HostHpoSearchStatus::Cancelled | HostHpoSearchStatus::Exhausted
        ) {
            if let (Some(saved), Some((_, progress))) = (&checkpoint, &mut durable) {
                progress.checkpoint(saved, status)?;
            }
        }
        if candidates.is_empty() {
            return Ok(HostHpoSearchOutcome {
                result: None,
                status,
                checkpoint,
            });
        }
        let policy = SelectionPolicy {
            requested_rank: None,
            id: "select:host_hpo".into(),
            metric: SelectionMetric {
                name: request.metric.name().into(),
                objective: request.direction,
            },
            required_metric_level: None,
            require_finite: true,
            evaluation_scope: None,
            refit_slot_plan: None,
            stacking_fit_contract: None,
            reduction_id: None,
        };
        let selected = select_candidate(&policy, &candidates)?;
        let winner = trials
            .iter()
            .find(|trial| trial.variant_id.as_str() == selected.selected_candidate_id)
            .expect("selection returns an observed candidate");
        Ok(HostHpoSearchOutcome {
            result: Some(HostHpoSearchResult {
                profile: "host_optimizer_search_v1".into(),
                portable: false,
                request_fingerprint: stable_json_fingerprint(request)?,
                graph_fingerprint: plan.graph_fingerprint.clone(),
                controller_fingerprint: plan.controller_fingerprint.clone(),
                campaign_fingerprint: stable_json_fingerprint(&plan.campaign)?,
                fold_set_fingerprint: stable_json_fingerprint(folds)?,
                selected_trial_index: winner.trial_index,
                selected_params: winner.params.clone(),
                trials,
                pruned_trials,
            }),
            status,
            checkpoint,
        })
    }
}

fn host_hpo_objective_fingerprint(request: &HostHpoSearchRequest) -> Result<String> {
    let mut value = serde_json::to_value(request)?;
    let object = value.as_object_mut().expect("request is an object");
    object.remove("trial_budget");
    if let Some(serde_json::Value::Object(descriptor)) = object.get_mut("optimizer_descriptor") {
        for key in ["n_trials", "trial_budget", "resume", "storage"] {
            descriptor.remove(key);
        }
    }
    // With serde_json/preserve_order, Map::remove swaps the last entry into
    // the removed slot. Resume-only keys must not reorder the scientific
    // objective or change its identity. Preserve the canonical object order
    // of the historical BTreeMap-backed Python binding, including nested maps.
    value.sort_all_objects();
    stable_json_fingerprint(&value)
}

/// Verify a resume checkpoint against the exact plan, request and data
/// envelope before a host invokes any external data-view recheck callback.
pub fn prepare_host_hpo_checkpoint(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    options: &HostHpoResumeOptions,
) -> Result<HostHpoCheckpoint> {
    request.validate_parameter_bindings(plan)?;
    if options.data_fingerprint.trim().is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "durable host HPO requires a data fingerprint".into(),
        ));
    }
    let binding = HostHpoCheckpointBinding {
        objective_fingerprint: host_hpo_objective_fingerprint(request)?,
        graph_fingerprint: plan.graph_fingerprint.clone(),
        controller_fingerprint: plan.controller_fingerprint.clone(),
        campaign_fingerprint: stable_json_fingerprint(&plan.campaign)?,
        fold_set_fingerprint: stable_json_fingerprint(&plan.fold_set)?,
        data_fingerprint: options.data_fingerprint.clone(),
        structural_catalogue_fingerprint: request
            .structural_catalogue
            .as_ref()
            .map(|catalogue| catalogue.catalogue_fingerprint.clone()),
    };
    let mut checkpoint = options.checkpoint.clone().unwrap_or(HostHpoCheckpoint {
        schema_version: 1,
        binding: binding.clone(),
        trials: Vec::new(),
        fingerprint: String::new(),
    });
    if options.checkpoint.is_none() {
        checkpoint.seal()?;
    }
    if checkpoint.schema_version != 1 || checkpoint.binding != binding {
        return Err(DagMlError::RuntimeValidation(
            "host HPO checkpoint objective/graph/controller/data/fold binding mismatch".into(),
        ));
    }
    let expected = checkpoint.fingerprint.clone();
    checkpoint.seal()?;
    if checkpoint.fingerprint != expected {
        return Err(DagMlError::RuntimeValidation(
            "host HPO checkpoint integrity fingerprint mismatch".into(),
        ));
    }
    if checkpoint.trials.len() > request.trial_budget as usize {
        return Err(DagMlError::RuntimeValidation(
            "host HPO total budget is smaller than checkpoint terminal history".into(),
        ));
    }
    for (index, trial) in checkpoint.trials.iter().enumerate() {
        if trial.trial_index() as usize != index {
            return Err(DagMlError::RuntimeValidation(
                "host HPO checkpoint trial indices must be unique and contiguous".into(),
            ));
        }
        let interrupted_without_params = matches!(trial, HostHpoTerminalTrial::Failed {
            params, error, ..
        } if params.is_empty() && error == "interrupted_before_native_evaluation");
        let (params, variant_id) = match trial {
            HostHpoTerminalTrial::Complete { evidence } => {
                host_hpo_candidate(plan, request, evidence)?;
                (&evidence.params, &evidence.variant_id)
            }
            HostHpoTerminalTrial::Pruned { evidence } => {
                let target_node = match &request.structural_catalogue {
                    Some(catalogue) => &catalogue.recipe(&evidence.params)?.target_node,
                    None => &request.target_node,
                };
                evidence.scores.validate()?;
                if evidence.scores.plan_id != plan.id
                    || evidence.intermediate_scores.is_empty()
                    || evidence.intermediate_scores.len()
                        > plan
                            .fold_set
                            .as_ref()
                            .expect("validated FoldSet")
                            .folds
                            .len()
                    || evidence
                        .intermediate_scores
                        .iter()
                        .any(|value| !value.is_finite())
                    || evidence.scores.reports.iter().any(|report| {
                        report
                            .variant_id
                            .as_ref()
                            .is_some_and(|id| id != &evidence.variant_id)
                    })
                {
                    return Err(DagMlError::RuntimeValidation(
                        "host HPO checkpoint pruned trial has invalid native intermediate evidence"
                            .into(),
                    ));
                }
                for (fold, recorded) in plan
                    .fold_set
                    .as_ref()
                    .expect("validated FoldSet")
                    .folds
                    .iter()
                    .zip(&evidence.intermediate_scores)
                {
                    let reports = evidence
                        .scores
                        .reports
                        .iter()
                        .filter(|report| {
                            report.producer_node == *target_node
                                && report_matches_statistical_policy(report, plan)
                                && report.partition == PredictionPartition::Validation
                                && report.fold_id.as_ref() == Some(&fold.fold_id)
                        })
                        .collect::<Vec<_>>();
                    let [report] = reports.as_slice() else {
                        return Err(DagMlError::RuntimeValidation(
                            "host HPO pruned checkpoint lacks one native score per observed fold"
                                .into(),
                        ));
                    };
                    if host_hpo_metric(report, request.metric)? != *recorded {
                        return Err(DagMlError::RuntimeValidation(
                            "host HPO pruned checkpoint intermediate differs from native score"
                                .into(),
                        ));
                    }
                }
                (&evidence.params, &evidence.variant_id)
            }
            HostHpoTerminalTrial::Failed {
                params,
                variant_id,
                error,
                ..
            } => {
                if error.trim().is_empty() {
                    return Err(DagMlError::RuntimeValidation(
                        "host HPO failed trial requires error evidence".into(),
                    ));
                }
                (params, variant_id)
            }
        };
        if (params.is_empty() && !interrupted_without_params)
            || params.keys().any(|key| key.trim().is_empty())
            || variant_id.as_str() != format!("host_hpo:trial:{index:010}")
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO checkpoint has invalid parameter/variant identity".into(),
            ));
        }
        if !interrupted_without_params {
            request.parameter_overrides(params)?;
        }
    }
    Ok(checkpoint)
}

pub(super) fn host_hpo_candidate(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    trial: &HostHpoTrialEvidence,
) -> Result<crate::selection::CandidateScore> {
    if trial.trial_index >= request.trial_budget
        || trial.variant_id.as_str() != format!("host_hpo:trial:{:010}", trial.trial_index)
    {
        return Err(DagMlError::RuntimeValidation(
            "host HPO trial/variant identity mismatch".into(),
        ));
    }
    trial.scores.validate()?;
    if trial.scores.plan_id != plan.id
        || trial.scores.reports.iter().any(|report| {
            report
                .variant_id
                .as_ref()
                .is_some_and(|id| id != &trial.variant_id)
        })
    {
        return Err(DagMlError::RuntimeValidation(
            "host HPO checkpoint score plan/variant mismatch".into(),
        ));
    }
    let candidate_plan = if request.structural_catalogue.is_some() {
        host_hpo_candidate_plan(plan, request, trial.trial_index, &trial.params, None)?
    } else {
        plan.clone()
    };
    if let Some(catalogue) = &request.structural_catalogue {
        let target = request.target_for_plan(&candidate_plan)?;
        let mut producers = BTreeSet::from([target.clone()]);
        if catalogue.schema_version == 2 {
            // Nested late-fusion execution retains native branch scores as
            // evidence. Admit only the selected signed graph's upstream
            // closure; host_hpo_score still requires the sink objective.
            loop {
                let before = producers.len();
                for edge in &candidate_plan.graph_plan.graph.edges {
                    if producers.contains(&edge.target.node_id)
                        && candidate_plan.node_plans.contains_key(&edge.source.node_id)
                    {
                        producers.insert(edge.source.node_id.clone());
                    }
                }
                if producers.len() == before {
                    break;
                }
            }
            producers.retain(|id| {
                candidate_plan.node_plans.get(id).is_some_and(|node| {
                    node.kind == NodeKind::Model
                        && node
                            .controller_capabilities
                            .contains(&ControllerCapability::EmitsPredictions)
                })
            });
        }
        if trial
            .scores
            .reports
            .iter()
            .any(|report| !producers.contains(&report.producer_node))
        {
            return Err(DagMlError::RuntimeValidation(
                "structural HPO score producer is outside selected recipe".into(),
            ));
        }
    }
    let (score, fold_scores, candidate) =
        host_hpo_score(&candidate_plan, request, &trial.variant_id, &trial.scores)?;
    if !trial.score.is_finite()
        || score != trial.score
        || fold_scores != trial.objective_fold_scores
    {
        return Err(DagMlError::RuntimeValidation(
            "host HPO checkpoint scalar/fold scores differ from native reports".into(),
        ));
    }
    Ok(candidate)
}

fn host_hpo_score(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    variant: &VariantId,
    scores: &ScoreSet,
) -> Result<(f64, BTreeMap<String, f64>, crate::selection::CandidateScore)> {
    let target_node = request.target_for_plan(plan)?;
    let folds = plan.fold_set.as_ref().expect("validated host HPO FoldSet");
    let reports = scores
        .reports
        .iter()
        .filter(|report| {
            report.producer_node == *target_node
                && report_matches_statistical_policy(report, plan)
                && report.partition == PredictionPartition::Validation
                && report.fold_id.as_ref().is_some_and(|fold| {
                    if request.fold_score_reduction.is_some() {
                        folds.folds.iter().any(|item| &item.fold_id == fold)
                    } else {
                        fold.as_str() == "avg"
                            || (folds.folds.len() == 1 && fold == &folds.folds[0].fold_id)
                    }
                })
        })
        .collect::<Vec<_>>();
    let mut fold_scores = BTreeMap::new();
    if let Some(reduction) = request.fold_score_reduction {
        for report in &reports {
            let fold = report.fold_id.as_ref().expect("filtered fold");
            if fold_scores
                .insert(
                    fold.as_str().to_owned(),
                    host_hpo_metric(report, request.metric)?,
                )
                .is_some()
            {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO has ambiguous fold score producers".into(),
                ));
            }
        }
        if fold_scores.len() != folds.folds.len() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires every declared fold's native score".into(),
            ));
        }
        let score = reduce_host_hpo_fold_scores(&fold_scores, reduction, request.direction)?;
        let candidate = crate::selection::CandidateScore {
            candidate_id: variant.as_str().to_owned(),
            metrics: BTreeMap::from([(request.metric.name().to_owned(), score)]),
            metadata: BTreeMap::from([
                (
                    "host_hpo_fold_score_reduction".into(),
                    serde_json::to_value(reduction)?,
                ),
                (
                    "objective_fold_scores".into(),
                    serde_json::to_value(&fold_scores)?,
                ),
            ]),
        };
        Ok((score, fold_scores, candidate))
    } else {
        let [report] = reports.as_slice() else {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires exactly one native target OOF report".into(),
            ));
        };
        Ok((
            host_hpo_metric(report, request.metric)?,
            fold_scores,
            (*report).clone().into_candidate_score(variant.as_str())?,
        ))
    }
}

fn host_hpo_metric(
    report: &crate::metrics::RegressionMetricReport,
    metric: RegressionMetricKind,
) -> Result<f64> {
    report
        .metrics
        .get(metric.name())
        .copied()
        .filter(|score| score.is_finite())
        .ok_or_else(|| {
            DagMlError::RuntimeValidation("host HPO has no finite requested metric".into())
        })
}

fn reduce_host_hpo_fold_scores(
    scores: &BTreeMap<String, f64>,
    reduction: HostHpoFoldReduction,
    direction: crate::selection::MetricObjective,
) -> Result<f64> {
    if scores.is_empty() || scores.values().any(|score| !score.is_finite()) {
        return Err(DagMlError::RuntimeValidation(
            "host HPO fold reduction requires finite observed scores".into(),
        ));
    }
    let values = scores.values().copied();
    let score = match reduction {
        HostHpoFoldReduction::Mean => values.map(|value| value / scores.len() as f64).sum(),
        HostHpoFoldReduction::Best | HostHpoFoldReduction::RobustBest => match direction {
            crate::selection::MetricObjective::Minimize => values.fold(f64::INFINITY, f64::min),
            crate::selection::MetricObjective::Maximize => values.fold(f64::NEG_INFINITY, f64::max),
        },
    };
    if !score.is_finite() {
        return Err(DagMlError::RuntimeValidation(
            "host HPO fold reduction overflowed".into(),
        ));
    }
    Ok(score)
}

fn report_matches_statistical_policy(
    report: &crate::metrics::RegressionMetricReport,
    plan: &ExecutionPlan,
) -> bool {
    let policy = &plan.campaign.aggregation_policy;
    policy.grouping_key.is_none()
        || (report.level == policy.selection_metric_level
            && report.grouping_key == policy.grouping_key)
}

/// Exact diagnostic for pending siblings of a candidate-planning failure.
/// Other failure edges keep their actual stage and cause.
pub const HOST_HPO_PLANNING_SIBLING_DIAGNOSTIC: &str =
    "not evaluated; owning window planning failed";

const CAPTURE_REQUEST_SCHEMA_VERSION: u32 = 2;
const CAPTURE_CHECKPOINT_SCHEMA_VERSION: u32 = 2;
const PROPOSAL_EVIDENCE_SCHEMA_VERSION: u32 = 1;

fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::deserialize(deserializer)
}

fn capture_hex(text: &str) -> String {
    crate::canonical::bytes_to_hex(text.as_bytes())
}

fn is_capture_hex(text: &str) -> bool {
    text.len() % 2 == 0
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_nonempty_capture_hex(text: &str) -> bool {
    !text.is_empty() && is_capture_hex(text)
}

fn decode_capture_hex(text: &str) -> Option<Vec<u8>> {
    if !is_capture_hex(text) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).ok())
        .collect()
}

fn is_terminal_status(status: Option<realization::TrialStatus>) -> bool {
    matches!(
        status,
        Some(
            realization::TrialStatus::Completed
                | realization::TrialStatus::Failed
                | realization::TrialStatus::Pruned
                | realization::TrialStatus::Cancelled
        )
    )
}

fn failure_stage_name(stage: realization::FailureStage) -> &'static str {
    match stage {
        realization::FailureStage::Ask => "ask",
        realization::FailureStage::ParameterResolution => "parameter_resolution",
        realization::FailureStage::ProposalConversion => "proposal_conversion",
        realization::FailureStage::CandidatePlanning => "candidate_planning",
        realization::FailureStage::RunContextCreation => "run_context_creation",
        realization::FailureStage::ProviderCreation => "provider_creation",
        realization::FailureStage::ControllerCreation => "controller_creation",
        realization::FailureStage::Evaluation => "evaluation",
        realization::FailureStage::Cleanup => "cleanup",
        realization::FailureStage::OrphanRecovery => "orphan_recovery",
    }
}

/// Whether a failed stage may have consumed owner randomness.
fn stage_consumption(stage: realization::FailureStage) -> realization::Consumption {
    match stage {
        realization::FailureStage::CandidatePlanning
        | realization::FailureStage::RunContextCreation => realization::Consumption::None,
        _ => realization::Consumption::Unknown,
    }
}

fn host_hpo_predecessor_refused(code: &str, detail: impl std::fmt::Display) -> DagMlError {
    DagMlError::HostHpoPredecessorRefused(format!("{code}: {detail}"))
}

fn evidence_refusal(error: &realization::CaptureError) -> DagMlError {
    DagMlError::RuntimeValidation(format!("realization evidence refused: {error}"))
}

fn proposal_scope(trial_index: u32) -> String {
    format!("proposal:{trial_index:010}")
}

fn terminal_scope(trial_index: u32) -> String {
    format!("terminal:{trial_index:010}")
}

fn visit_id(trial_index: u32) -> String {
    format!("visit:{trial_index:010}")
}

fn trial_id(trial_index: u32) -> String {
    format!("trial:{trial_index:010}")
}

/// A seed declared before any ask. `requested = None` is absence, not zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoSeedDeclaration {
    pub seed_id: String,
    #[serde(deserialize_with = "required_option")]
    pub requested: Option<realization::SeedValue>,
}

/// Capture declaration and failed-attempt provenance of the capture-gated route.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCaptureDeclaration {
    pub capture_id: String,
    pub scope_id: String,
    pub search_scope_id: String,
    pub declaration_text: String,
    pub source_digest: String,
    pub node_ids: Vec<String>,
    pub seed_declarations: Vec<HostHpoSeedDeclaration>,
    pub failed_attempts: Vec<realization::FailedAttempt>,
}

/// Selects capture cleanup and evidence. `request` is the unchanged v1 document,
/// so scientific objective identity does not depend on capture lineage.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCaptureSearchRequest {
    pub schema_version: u32,
    pub request: HostHpoSearchRequest,
    pub capture: HostHpoCaptureDeclaration,
}

impl HostHpoCaptureSearchRequest {
    /// Admit the declaration, prior attempts and seed declarations through the
    /// closed realization recorder, without writing any evidence.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CAPTURE_REQUEST_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(
                "captured host HPO requires request schema version 2".into(),
            ));
        }
        let mut recorder = realization::Recorder::new(
            std::io::sink(),
            &self.capture.capture_id,
            realization::Limits::default(),
        )
        .map_err(|error| capture_opening_refusal(&error, false))?;
        for event in capture_opening_events(&self.capture) {
            let provenance = matches!(event, realization::Event::CaptureProvenance { .. });
            recorder
                .record(&event)
                .map_err(|error| capture_opening_refusal(&error, provenance))?;
        }
        Ok(())
    }
}

fn capture_opening_refusal(error: &realization::CaptureError, provenance: bool) -> DagMlError {
    if provenance {
        host_hpo_predecessor_refused("failed_attempt", error)
    } else {
        DagMlError::RuntimeValidation(format!("host HPO capture declaration refused: {error}"))
    }
}

/// One located proposal value. `value_hex` is the UTF-8 JSON text of
/// `params[parameter]`, so restoration never reconstructs a choice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoLocatedChoice {
    pub choice_id: String,
    pub parameter: String,
    pub node_id: String,
    pub source_path_hex: Vec<String>,
    pub value_hex: String,
    pub semantics: realization::ChoiceSemantics,
    #[serde(deserialize_with = "required_option")]
    pub seed_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoSeedParent {
    pub seed_id: String,
    pub value_slot: realization::ValueSlot,
}

/// Effective seed fact. In a proposal it lists a seed first used there; in an
/// attempt it also covers earlier seeds the attempt references, with their first
/// use inside that attempt, so replay needs no reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoSeedResolution {
    pub seed_id: String,
    pub effective: realization::SeedValue,
    pub resolution: realization::ResolutionKind,
    #[serde(deserialize_with = "required_option")]
    pub parent: Option<HostHpoSeedParent>,
    #[serde(deserialize_with = "required_option")]
    pub policy: Option<realization::Policy>,
    pub first_actual_use_id: String,
}

/// A real draw from a seed that is not itself a located choice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoSeedUse {
    pub use_id: String,
    pub seed_id: String,
    pub operation: realization::Policy,
}

/// Closed result of one captured ask. `Failed` is how failures in stages only
/// the host bridge can see reach the coordinator: `owned_attempt = None` is an
/// authoritative observation that no attempt exists, while an unknown attempt
/// keeps `ownership = unknown`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostHpoProposalOutcome {
    Proposed {
        owner_ref_hex: String,
        owner_evidence_hex: String,
        params: BTreeMap<String, serde_json::Value>,
        located_choices: Vec<HostHpoLocatedChoice>,
        seed_resolutions: Vec<HostHpoSeedResolution>,
        seed_uses: Vec<HostHpoSeedUse>,
    },
    /// A struct variant, because serde ignores unknown fields on an internally
    /// tagged unit variant.
    Exhausted {},
    Failed {
        failure_stage: realization::FailureStage,
        causal_diagnostic_hex: String,
        consumption: realization::Consumption,
        #[serde(deserialize_with = "required_option")]
        owned_attempt: Option<realization::OwnedAttempt>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoProposalEvidence {
    pub schema_version: u32,
    pub outcome: HostHpoProposalOutcome,
}

impl HostHpoProposalEvidence {
    /// Check the envelope and bind located values to the proposed parameters.
    /// Seed and node facts are judged by the closed recorder, not here.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != PROPOSAL_EVIDENCE_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(
                "attempt_evidence: unsupported proposal evidence schema version".into(),
            ));
        }
        match &self.outcome {
            HostHpoProposalOutcome::Proposed {
                owner_ref_hex,
                owner_evidence_hex,
                params,
                located_choices,
                ..
            } => {
                if !is_nonempty_capture_hex(owner_ref_hex) {
                    return Err(DagMlError::RuntimeValidation(
                        "attempt_evidence: proposal has no owner reference".into(),
                    ));
                }
                validate_located_values(owner_evidence_hex, located_choices, params).map_err(
                    |detail| DagMlError::RuntimeValidation(format!("attempt_evidence: {detail}")),
                )
            }
            HostHpoProposalOutcome::Exhausted {} => Ok(()),
            HostHpoProposalOutcome::Failed {
                failure_stage,
                causal_diagnostic_hex,
                ..
            } => {
                let bridge_stage = matches!(
                    failure_stage,
                    realization::FailureStage::Ask
                        | realization::FailureStage::ParameterResolution
                        | realization::FailureStage::ProposalConversion
                );
                if !bridge_stage || !is_capture_hex(causal_diagnostic_hex) {
                    return Err(DagMlError::RuntimeValidation(
                        "attempt_evidence: proposal failure names a coordinator stage or an invalid diagnostic"
                            .into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

fn duplicate_seed_id(seeds: &[HostHpoSeedResolution]) -> Option<&str> {
    let mut seen = BTreeSet::new();
    seeds
        .iter()
        .map(|seed| seed.seed_id.as_str())
        .find(|seed_id| !seen.insert(*seed_id))
}

fn validate_located_values(
    owner_evidence_hex: &str,
    choices: &[HostHpoLocatedChoice],
    params: &BTreeMap<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    if !is_nonempty_capture_hex(owner_evidence_hex) {
        return Err("owner evidence is missing".into());
    }
    let mut located = BTreeSet::new();
    for choice in choices {
        let value = decode_capture_hex(&choice.value_hex)
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        if value.is_none()
            || value.as_ref() != params.get(&choice.parameter)
            || !located.insert(choice.parameter.as_str())
        {
            return Err(format!(
                "located choice `{}` does not bind exactly one proposal value",
                choice.choice_id
            ));
        }
    }
    if located.len() != params.len() {
        return Err("every proposal value requires a located choice".into());
    }
    Ok(())
}

/// Observed owner proposal state. Reading it must not draw or mutate.
///
/// All resume safety rests on `owner_checkpoint_fingerprint`. It fills the
/// existing pair slot and must be a canonical 64-hex TCV1 digest of everything
/// that can change a future proposal: trial rows and status, sampler and RNG
/// state, pending parameters and owner configuration. A digest of rows alone is
/// insufficient wherever that state can move independently of them. An owner
/// that cannot expose and persist it reports `None`, which refuses qualified
/// resume; it must never guess or restore a seed to fill the gap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoProposalState {
    /// `None` means no pairable owner state exists, which refuses resume.
    #[serde(deserialize_with = "required_option")]
    pub owner_checkpoint_fingerprint: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub structural_binding: Option<serde_json::Value>,
    /// Every trial the owner holds, terminal or still running.
    pub owner_trial_count: u64,
}

fn is_owner_state_digest(value: &str) -> bool {
    value.len() == 64 && is_capture_hex(value)
}

fn validate_owner_state(state: &HostHpoProposalState) -> Result<()> {
    if state
        .owner_checkpoint_fingerprint
        .as_deref()
        .is_some_and(|digest| !is_owner_state_digest(digest))
    {
        return Err(host_hpo_predecessor_refused(
            "owner_state_invalid",
            "owner state must be a canonical 64-hex digest",
        ));
    }
    Ok(())
}

/// Capture-gated proposal producer. Trials are addressed by the owner reference
/// the owner returned, never by a requested index.
pub trait HostHpoCaptureProposalSource {
    /// Read current owner state with no draw and no transition.
    fn proposal_state(&mut self) -> Result<HostHpoProposalState>;
    /// Register the owner trial before validating anything about it, and report
    /// a failure after a confirmed ask through a `Failed` outcome. A raised error
    /// keeps ownership unknown.
    ///
    /// A logical seed id has one immutable value across capture lineages. An
    /// owner that resumed may omit a seed the restored publication resolved, and
    /// the coordinator carries the sealed fact forward with no draw; if it
    /// restates that seed, the facts must match exactly.
    fn ask_evidence(
        &mut self,
        trial_index: u32,
        phase_index: Option<u32>,
    ) -> Result<HostHpoProposalEvidence>;
    fn tell(&mut self, owner_ref_hex: &str, score: f64) -> Result<()>;
    fn report_intermediate(
        &mut self,
        _owner_ref_hex: &str,
        _step: u32,
        _score: f64,
    ) -> Result<bool> {
        Ok(false)
    }
    fn pruned(&mut self, owner_ref_hex: &str) -> Result<()>;
    /// Query owner status first and transition only a RUNNING trial through the
    /// genuine Failed API. A terminal trial returns its observation unchanged.
    /// `stage = None` is an edge with no known stage. `diagnostic_hex` is the
    /// owner's own text, which may be truncated; return it as observed, or null.
    fn fail(
        &mut self,
        owner_ref_hex: &str,
        stage: Option<realization::FailureStage>,
        diagnostic: &str,
    ) -> Result<realization::OwnedAttempt>;
}

/// Identity of the planned candidate, taken from the actual candidate plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCandidateEvidence {
    pub owner_variant_id: String,
    pub owner_fingerprint: String,
    pub active_node_ids: Vec<String>,
    pub parameters_hex: String,
}

fn candidate_parameters_hex(params: &BTreeMap<String, serde_json::Value>) -> Result<String> {
    Ok(crate::canonical::bytes_to_hex(&crate::stable_json::to_vec(
        params,
    )?))
}

fn candidate_evidence(
    plan: &ExecutionPlan,
    params: &BTreeMap<String, serde_json::Value>,
) -> Result<HostHpoCandidateEvidence> {
    let variant = &plan.variants[0];
    Ok(HostHpoCandidateEvidence {
        owner_variant_id: variant.variant_id.as_str().to_owned(),
        owner_fingerprint: variant.fingerprint.clone(),
        active_node_ids: plan
            .node_plans
            .keys()
            .map(|node| node.as_str().to_owned())
            .collect(),
        parameters_hex: candidate_parameters_hex(params)?,
    })
}

/// Real per-attempt evidence: everything the closed recorder needs to replay
/// this attempt, in the order it was recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoAttemptEvidence {
    pub trial_index: u32,
    /// The capture that asked this trial. A replayed attempt keeps it.
    pub capture_id: String,
    pub scope_id: String,
    /// The Search scope its live seed facts were recorded in.
    pub search_scope_id: String,
    pub owner_ref_hex: String,
    pub owner_evidence_hex: String,
    #[serde(deserialize_with = "required_option")]
    pub phase_index: Option<u32>,
    pub seed_declarations: Vec<HostHpoSeedDeclaration>,
    pub seed_resolutions: Vec<HostHpoSeedResolution>,
    pub seed_uses: Vec<HostHpoSeedUse>,
    pub located_choices: Vec<HostHpoLocatedChoice>,
    /// Null is the honest `not_materialized` state.
    #[serde(deserialize_with = "required_option")]
    pub candidate: Option<HostHpoCandidateEvidence>,
    #[serde(deserialize_with = "required_option")]
    pub failure_stage: Option<realization::FailureStage>,
    #[serde(deserialize_with = "required_option")]
    pub causal_diagnostic_hex: Option<String>,
    /// The capture that observed the terminal transition; null while pending.
    #[serde(deserialize_with = "required_option")]
    pub terminal_capture_id: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub terminal_scope_id: Option<String>,
    pub owner_observation: realization::OwnedAttempt,
}

/// An asked trial still RUNNING at publication. It replays from these sealed
/// values, so resuming it never reaches the owner's ask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoPendingAttempt {
    pub params: BTreeMap<String, serde_json::Value>,
    pub attempt: HostHpoAttemptEvidence,
}

/// Sealed v1 checkpoint plus the evidence needed to restore each attempt.
/// An aborted window publishes none of these.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCaptureCheckpoint {
    pub schema_version: u32,
    pub checkpoint: HostHpoCheckpoint,
    pub source_digest: String,
    pub attempts: Vec<HostHpoAttemptEvidence>,
    pub pending_attempts: Vec<HostHpoPendingAttempt>,
    pub fingerprint: String,
}

impl HostHpoCaptureCheckpoint {
    fn seal(&mut self) -> Result<()> {
        self.fingerprint = stable_json_fingerprint(&(
            self.schema_version,
            &self.checkpoint.fingerprint,
            &self.source_digest,
            &self.attempts,
            &self.pending_attempts,
        ))?;
        Ok(())
    }

    pub fn verify_seal(&self) -> Result<()> {
        let mut sealed = self.clone();
        sealed.seal()?;
        if sealed.fingerprint != self.fingerprint {
            return Err(host_hpo_predecessor_refused(
                "checkpoint_seal",
                "captured checkpoint integrity fingerprint mismatch",
            ));
        }
        Ok(())
    }
}

/// TCV1 pair over observed owner state and the sealed checkpoint. This is the
/// existing `pair_fingerprint` preimage, with no new hash dimension. An absent
/// `structural_binding` is omitted from the preimage, not encoded as null, so
/// another producer must drop the key to reach the same digest.
pub fn host_hpo_capture_pair_fingerprint(
    checkpoint: &HostHpoCaptureCheckpoint,
    owner_checkpoint_fingerprint: &str,
    structural_binding: Option<&serde_json::Value>,
) -> Result<String> {
    let typed = |raw: String| {
        crate::canonical::parse_typed_json(&raw)
            .map_err(|error| DagMlError::RuntimeValidation(format!("pair canonical: {error}")))
    };
    let mut entries = vec![
        (
            "checkpoint_fingerprint".to_owned(),
            crate::canonical::TypedCanonicalValue::String(owner_checkpoint_fingerprint.to_owned()),
        ),
        (
            "native_checkpoint".to_owned(),
            typed(serde_json::to_string(checkpoint)?)?,
        ),
    ];
    if let Some(binding) = structural_binding {
        entries.push((
            "structural_binding".to_owned(),
            typed(serde_json::to_string(binding)?)?,
        ));
    }
    crate::canonical::tcv1_sha256(&crate::canonical::TypedCanonicalValue::Object(entries))
        .map_err(|error| DagMlError::RuntimeValidation(format!("pair canonical: {error}")))
}

/// One publication: a sealed checkpoint paired with the owner state observed
/// after the last transition. A null pair is honest and refuses resume.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoCapturePublication {
    pub checkpoint: HostHpoCaptureCheckpoint,
    #[serde(deserialize_with = "required_option")]
    pub owner_checkpoint_fingerprint: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub structural_binding: Option<serde_json::Value>,
    #[serde(deserialize_with = "required_option")]
    pub pair_fingerprint: Option<String>,
}

/// Failed-attempt provenance derived from a sealed publication, so a Failed
/// prefix stays resumable after the original report is gone.
pub fn host_hpo_capture_prefix_provenance(
    publication: &HostHpoCapturePublication,
) -> Result<Vec<realization::FailedAttempt>> {
    let Some(pair_fingerprint) = &publication.pair_fingerprint else {
        return Err(host_hpo_predecessor_refused(
            "pair_unavailable",
            "this publication has no owner pair",
        ));
    };
    let checkpoint = &publication.checkpoint;
    Ok(checkpoint
        .attempts
        .iter()
        .filter(|attempt| {
            attempt.owner_observation.status == Some(realization::TrialStatus::Failed)
        })
        .map(|attempt| realization::FailedAttempt {
            failed_capture_id: attempt.terminal_capture_id.clone(),
            failed_scope_id: attempt.terminal_scope_id.clone(),
            failure_stage: attempt.failure_stage,
            causal_diagnostic_hex: attempt.causal_diagnostic_hex.clone().unwrap_or_default(),
            owned_attempts: vec![attempt.owner_observation.clone()],
            trusted_prefix_ref: Some(realization::PrefixRef {
                owner_ref_hex: attempt.owner_ref_hex.clone(),
                checkpoint_fingerprint: checkpoint.fingerprint.clone(),
                pair_fingerprint: Some(pair_fingerprint.clone()),
            }),
        })
        .collect())
}

/// Durable publication of the capture route, with the ordinary ordering:
/// prepare the terminal, transition the owner, then publish. A prepared
/// checkpoint is prospective and never resumable.
pub trait HostHpoCaptureProgress {
    fn prepare_terminal(
        &mut self,
        _checkpoint: &HostHpoCaptureCheckpoint,
        _status: HostHpoSearchStatus,
    ) -> Result<()> {
        Ok(())
    }

    fn checkpoint(
        &mut self,
        publication: &HostHpoCapturePublication,
        status: HostHpoSearchStatus,
    ) -> Result<bool>;
}

pub struct HostHpoCaptureResumeOptions {
    pub data_fingerprint: String,
    pub publication: Option<HostHpoCapturePublication>,
}

impl HostHpoCaptureResumeOptions {
    fn ordinary(&self) -> HostHpoResumeOptions {
        HostHpoResumeOptions {
            data_fingerprint: self.data_fingerprint.clone(),
            checkpoint: self
                .publication
                .as_ref()
                .map(|publication| publication.checkpoint.checkpoint.clone()),
        }
    }
}

/// The real completion record and the commitment `verify_stream` checks it with.
#[derive(Clone, Debug, Serialize)]
pub struct HostHpoCaptureCompletion {
    pub completion: realization::Completion,
    pub expected_commitment: realization::ExpectedCohortCommitment,
}

/// A returned search outcome. `completion` is present once the stream closed
/// with every owner trial accounted for. A `Running` browser window has none
/// yet, and neither does a stream that `uncaptured_pending` lists trials for.
#[derive(Clone, Debug, Serialize)]
pub struct HostHpoCaptureOutcome {
    pub outcome: HostHpoSearchOutcome,
    pub publication: Option<HostHpoCapturePublication>,
    pub attempts: Vec<HostHpoAttemptEvidence>,
    pub failed_attempts: Vec<realization::FailedAttempt>,
    pub completion: Option<HostHpoCaptureCompletion>,
    /// Trials the owner still holds RUNNING that no stream closed, by index.
    pub uncaptured_pending: Vec<u32>,
    /// How the stream ended when it could not complete, including whether its
    /// prefix reached the sink.
    pub refusal: Option<crate::error::HostHpoCaptureEvidence>,
    /// Non-fatal failures, such as an owner state read that left a null pair.
    pub secondary_errors: Vec<crate::error::DagMlErrorDescriptor>,
    pub evidence_errors: Vec<crate::error::DagMlErrorDescriptor>,
}

/// How a returned status closed the stream.
#[derive(Default)]
struct HostHpoCaptureClosure {
    completion: Option<HostHpoCaptureCompletion>,
    uncaptured_pending: Vec<u32>,
    refusal: Option<crate::error::HostHpoCaptureEvidence>,
}

fn capture_opening_events(capture: &HostHpoCaptureDeclaration) -> Vec<realization::Event> {
    let mut events = vec![
        realization::Event::Declaration {
            scope_id: capture.scope_id.clone(),
            scope_kind: realization::ScopeKind::Declaration,
            parent_scope_id: None,
            declaration_text: capture.declaration_text.clone(),
            source_digest: capture.source_digest.clone(),
            node_ids: capture.node_ids.clone(),
        },
        realization::Event::CaptureProvenance {
            scope_id: capture.scope_id.clone(),
            failed_attempts: capture.failed_attempts.clone(),
        },
        realization::Event::ScopeOpen {
            scope_id: capture.search_scope_id.clone(),
            scope_kind: realization::ScopeKind::Search,
            parent_scope_id: capture.scope_id.clone(),
            declaration_ref: declaration_ref(capture),
        },
    ];
    events.extend(
        capture
            .seed_declarations
            .iter()
            .map(|seed| realization::Event::SeedDeclared {
                scope_id: capture.search_scope_id.clone(),
                seed_id: seed.seed_id.clone(),
                requested: seed.requested.clone(),
                provenance: attempt_provenance(None, None, ""),
            }),
    );
    events
}

fn declaration_ref(capture: &HostHpoCaptureDeclaration) -> realization::DeclarationRef {
    realization::DeclarationRef {
        scope_id: capture.scope_id.clone(),
        source_digest: capture.source_digest.clone(),
    }
}

/// `checkpoint_ref = None` is a new observation in this capture lineage.
fn attempt_provenance(
    attempt: Option<&HostHpoAttemptEvidence>,
    checkpoint_ref: Option<&str>,
    original_id: &str,
) -> realization::Provenance {
    match (attempt, checkpoint_ref) {
        (Some(attempt), Some(checkpoint_ref)) => realization::Provenance {
            observation: realization::Origin::Restored,
            original_capture_id: Some(attempt.capture_id.clone()),
            original_scope_id: Some(proposal_scope(attempt.trial_index)),
            original_id: Some(original_id.to_owned()),
            checkpoint_ref: Some(checkpoint_ref.to_owned()),
        },
        _ => realization::Provenance {
            observation: realization::Origin::New,
            original_capture_id: None,
            original_scope_id: None,
            original_id: None,
            checkpoint_ref: None,
        },
    }
}

fn candidate_event(
    attempt: &HostHpoAttemptEvidence,
    candidate: &HostHpoCandidateEvidence,
    checkpoint_ref: Option<&str>,
) -> realization::Event {
    let visit = visit_id(attempt.trial_index);
    realization::Event::Candidate {
        scope_id: proposal_scope(attempt.trial_index),
        provenance: attempt_provenance(Some(attempt), checkpoint_ref, &visit),
        visit_id: visit,
        owner_variant_id: candidate.owner_variant_id.clone(),
        owner_fingerprint: candidate.owner_fingerprint.clone(),
        active_node_ids: candidate.active_node_ids.clone(),
        parameters_hex: candidate.parameters_hex.clone(),
    }
}

/// The one builder of an attempt's pre-evaluation facts. The live path and the
/// restored dry replay both feed this to the closed recorder, so seed, first-use
/// and candidate rules have a single judge. Live seeds resolve in the Search
/// scope; restored seeds are restated inside their Proposal scope, because a
/// new draw on a restored resolution is refused.
fn attempt_opening_events(
    capture: &HostHpoCaptureDeclaration,
    attempt: &HostHpoAttemptEvidence,
    checkpoint_ref: Option<&str>,
    fresh_seed_ids: &[String],
) -> Vec<realization::Event> {
    let index = attempt.trial_index;
    let proposal = proposal_scope(index);
    let restored = checkpoint_ref.is_some();
    let seed_scope = if restored {
        proposal.clone()
    } else {
        capture.search_scope_id.clone()
    };
    let seed_ref = |seed_id: &str| realization::Ref {
        scope_id: seed_scope.clone(),
        id: seed_id.to_owned(),
    };
    let provenance = |id: &str| attempt_provenance(Some(attempt), checkpoint_ref, id);
    // Live seed facts were recorded in the Search scope, not the Proposal.
    let seed_provenance = |id: &str| {
        let mut provenance = provenance(id);
        if restored {
            provenance.original_scope_id = Some(attempt.search_scope_id.clone());
        }
        provenance
    };
    let mut events = vec![realization::Event::ScopeOpen {
        scope_id: proposal.clone(),
        scope_kind: realization::ScopeKind::Proposal,
        parent_scope_id: capture.search_scope_id.clone(),
        declaration_ref: declaration_ref(capture),
    }];
    if restored {
        events.extend(attempt.seed_declarations.iter().map(|seed| {
            realization::Event::SeedDeclared {
                scope_id: proposal.clone(),
                seed_id: seed.seed_id.clone(),
                requested: seed.requested.clone(),
                provenance: seed_provenance(&seed.seed_id),
            }
        }));
    }
    for seed in &attempt.seed_resolutions {
        if restored || fresh_seed_ids.contains(&seed.seed_id) {
            events.push(realization::Event::EffectiveSeedResolved {
                scope_id: seed_scope.clone(),
                seed_id: seed.seed_id.clone(),
                declared_ref: seed_ref(&seed.seed_id),
                effective: seed.effective.clone(),
                resolution: seed.resolution,
                parent: seed.parent.as_ref().map(|parent| realization::ParentSeed {
                    seed_ref: seed_ref(&parent.seed_id),
                    value_slot: parent.value_slot,
                }),
                policy: seed.policy.clone(),
                first_actual_use: realization::Ref {
                    scope_id: proposal.clone(),
                    id: seed.first_actual_use_id.clone(),
                },
                provenance: seed_provenance(&seed.seed_id),
            });
        }
    }
    for seed_use in &attempt.seed_uses {
        events.push(realization::Event::SeedUse {
            scope_id: proposal.clone(),
            use_id: seed_use.use_id.clone(),
            seed_ref: seed_ref(&seed_use.seed_id),
            operation: seed_use.operation.clone(),
            provenance: provenance(&seed_use.use_id),
        });
    }
    for choice in &attempt.located_choices {
        events.push(realization::Event::Choice {
            scope_id: proposal.clone(),
            choice_id: choice.choice_id.clone(),
            node_id: choice.node_id.clone(),
            source_path_hex: choice.source_path_hex.clone(),
            value_hex: choice.value_hex.clone(),
            semantics: choice.semantics,
            seed_ref: choice.seed_id.as_deref().map(seed_ref),
            provenance: provenance(&choice.choice_id),
        });
    }
    let visit = visit_id(index);
    let trial = trial_id(index);
    let visit_ref = realization::Ref {
        scope_id: proposal.clone(),
        id: visit.clone(),
    };
    events.push(realization::Event::Visit {
        scope_id: proposal.clone(),
        visit_id: visit.clone(),
        choice_ids: attempt
            .located_choices
            .iter()
            .map(|choice| choice.choice_id.clone())
            .collect(),
        provenance: provenance(&visit),
    });
    let restored_candidate = attempt.candidate.as_ref().filter(|_| restored);
    if let Some(candidate) = restored_candidate {
        events.push(candidate_event(attempt, candidate, checkpoint_ref));
    }
    let origin = if restored {
        realization::Origin::Restored
    } else {
        realization::Origin::New
    };
    events.push(realization::Event::TrialVisit {
        scope_id: proposal.clone(),
        trial_id: trial.clone(),
        visit_ref: visit_ref.clone(),
        owner_trial_ref_hex: attempt.owner_ref_hex.clone(),
        origin,
        candidate_ref: restored_candidate.map(|_| visit_ref),
        owner_status: attempt.owner_observation.status,
        owner_evidence_hex: attempt.owner_evidence_hex.clone(),
        provenance: provenance(&trial),
    });
    if restored {
        events.push(realization::Event::TrialResume {
            scope_id: proposal.clone(),
            trial_ref: realization::Ref {
                scope_id: proposal,
                id: trial.clone(),
            },
            observed_status: attempt.owner_observation.status,
            owner_evidence_hex: attempt.owner_evidence_hex.clone(),
            provenance: provenance(&trial),
        });
    }
    events
}

fn known_observation(value_hex: Option<String>) -> realization::Observation {
    realization::Observation {
        availability: if value_hex.is_some() {
            realization::Availability::Known
        } else {
            realization::Availability::Unknown
        },
        value_hex,
    }
}

/// Close one attempt that reached a terminal status.
fn attempt_terminal_events(
    capture: &HostHpoCaptureDeclaration,
    attempt: &HostHpoAttemptEvidence,
    status: realization::TrialStatus,
    mode: realization::OutcomeMode,
) -> Vec<realization::Event> {
    let index = attempt.trial_index;
    let proposal = proposal_scope(index);
    let terminal = terminal_scope(index);
    vec![
        realization::Event::ScopeOpen {
            scope_id: terminal.clone(),
            scope_kind: realization::ScopeKind::Terminal,
            parent_scope_id: proposal.clone(),
            declaration_ref: declaration_ref(capture),
        },
        realization::Event::TrialOutcome {
            scope_id: terminal.clone(),
            trial_ref: realization::Ref {
                scope_id: proposal.clone(),
                id: trial_id(index),
            },
            outcome: realization::Outcome {
                status,
                mode,
                stage: known_observation(
                    attempt
                        .failure_stage
                        .map(|stage| capture_hex(failure_stage_name(stage))),
                ),
                reason: known_observation(attempt.causal_diagnostic_hex.clone()),
                failure: None,
            },
            owner_evidence_hex: attempt.owner_evidence_hex.clone(),
        },
        realization::Event::ScopeClose { scope_id: terminal },
        realization::Event::ScopeClose { scope_id: proposal },
    ]
}

/// The complete restored stream of one publication, shared by the dry replay
/// and the real recorder.
fn restored_terminal_events(
    capture: &HostHpoCaptureDeclaration,
    checkpoint: &HostHpoCaptureCheckpoint,
) -> Result<Vec<realization::Event>> {
    let mut events = Vec::new();
    for attempt in &checkpoint.attempts {
        let status = attempt.owner_observation.status.ok_or_else(|| {
            host_hpo_predecessor_refused("attempt_terminal", "restored status is missing")
        })?;
        events.extend(attempt_opening_events(
            capture,
            attempt,
            Some(&checkpoint.fingerprint),
            &[],
        ));
        events.extend(attempt_terminal_events(
            capture,
            attempt,
            status,
            realization::OutcomeMode::RestoredObservation,
        ));
    }
    Ok(events)
}

fn dry_replay_restored(
    capture: &HostHpoCaptureDeclaration,
    checkpoint: &HostHpoCaptureCheckpoint,
) -> Result<()> {
    let refused =
        |error: realization::CaptureError| host_hpo_predecessor_refused("restored_evidence", error);
    let mut recorder = realization::Recorder::new(
        std::io::sink(),
        &capture.capture_id,
        realization::Limits::default(),
    )
    .map_err(refused)?;
    let mut events = capture_opening_events(capture);
    events.extend(restored_terminal_events(capture, checkpoint)?);
    for pending in &checkpoint.pending_attempts {
        events.extend(attempt_opening_events(
            capture,
            &pending.attempt,
            Some(&checkpoint.fingerprint),
            &[],
        ));
    }
    for event in &events {
        recorder.record(event).map_err(refused)?;
    }
    Ok(())
}

fn seed_fact<'a>(
    known: &'a BTreeMap<String, HostHpoSeedResolution>,
    fresh: &'a [HostHpoSeedResolution],
    seed_id: &str,
) -> Option<(&'a HostHpoSeedResolution, bool)> {
    fresh
        .iter()
        .find(|seed| seed.seed_id == seed_id)
        .map(|seed| (seed, true))
        .or_else(|| known.get(seed_id).map(|seed| (seed, false)))
}

/// Seeds an attempt references, parents first. A seed resolved by an earlier
/// proposal takes the first event here that reaches it; a fresh seed keeps the
/// first use the host declared, so a phantom one still refuses.
fn seed_closure(
    known: &BTreeMap<String, HostHpoSeedResolution>,
    fresh: &[HostHpoSeedResolution],
    seed_uses: &[HostHpoSeedUse],
    choices: &[HostHpoLocatedChoice],
) -> Vec<HostHpoSeedResolution> {
    let references = seed_uses
        .iter()
        .map(|seed_use| (seed_use.seed_id.as_str(), seed_use.use_id.as_str()))
        .chain(choices.iter().filter_map(|choice| {
            choice
                .seed_id
                .as_deref()
                .map(|seed_id| (seed_id, choice.choice_id.as_str()))
        }));
    let mut closure: Vec<HostHpoSeedResolution> = Vec::new();
    for (seed_id, event_id) in references {
        let mut chain: Vec<HostHpoSeedResolution> = Vec::new();
        let mut current = seed_fact(known, fresh, seed_id);
        while let Some((seed, is_fresh)) = current {
            if closure
                .iter()
                .chain(chain.iter())
                .any(|seen| seen.seed_id == seed.seed_id)
            {
                break;
            }
            let mut restated = seed.clone();
            if !is_fresh {
                restated.first_actual_use_id = event_id.to_owned();
            }
            chain.push(restated);
            current = seed
                .parent
                .as_ref()
                .filter(|parent| parent.value_slot == realization::ValueSlot::Effective)
                .and_then(|parent| seed_fact(known, fresh, &parent.seed_id));
        }
        chain.reverse();
        closure.extend(chain);
    }
    for seed in fresh {
        if !closure.iter().any(|seen| seen.seed_id == seed.seed_id) {
            closure.push(seed.clone());
        }
    }
    closure
}

/// Judge one proposal's seed, first-use and node facts through the closed
/// recorder before they can poison the real stream. Its restated shape closes
/// the Proposal scope, which is where an unused resolution refuses.
fn dry_check_attempt(
    capture: &HostHpoCaptureDeclaration,
    attempt: &HostHpoAttemptEvidence,
) -> std::result::Result<(), realization::CaptureError> {
    let mut recorder = realization::Recorder::new(
        std::io::sink(),
        &capture.capture_id,
        realization::Limits::default(),
    )?;
    for event in capture_opening_events(capture) {
        recorder.record(&event)?;
    }
    for event in attempt_opening_events(capture, attempt, Some("00"), &[]) {
        if !matches!(
            event,
            realization::Event::Visit { .. }
                | realization::Event::Candidate { .. }
                | realization::Event::TrialVisit { .. }
                | realization::Event::TrialResume { .. }
        ) {
            recorder.record(&event)?;
        }
    }
    recorder.record(&realization::Event::ScopeClose {
        scope_id: proposal_scope(attempt.trial_index),
    })
}

#[allow(clippy::too_many_arguments)]
fn proposal_attempt(
    capture: &HostHpoCaptureDeclaration,
    known: &BTreeMap<String, HostHpoSeedResolution>,
    trial_index: u32,
    phase_index: Option<u32>,
    owner_ref_hex: String,
    owner_evidence_hex: String,
    located_choices: Vec<HostHpoLocatedChoice>,
    seed_resolutions: &[HostHpoSeedResolution],
    seed_uses: Vec<HostHpoSeedUse>,
) -> HostHpoAttemptEvidence {
    HostHpoAttemptEvidence {
        trial_index,
        capture_id: capture.capture_id.clone(),
        scope_id: capture.scope_id.clone(),
        search_scope_id: capture.search_scope_id.clone(),
        owner_observation: realization::OwnedAttempt {
            owner_ref_hex: Some(owner_ref_hex.clone()),
            ownership: realization::Ownership::Confirmed,
            status: Some(realization::TrialStatus::Running),
            cleanup: realization::CleanupState::NotAttempted,
            diagnostic_hex: None,
        },
        owner_ref_hex,
        owner_evidence_hex,
        phase_index,
        seed_declarations: capture.seed_declarations.clone(),
        seed_resolutions: seed_closure(known, seed_resolutions, &seed_uses, &located_choices),
        seed_uses,
        located_choices,
        candidate: None,
        failure_stage: None,
        causal_diagnostic_hex: None,
        terminal_capture_id: None,
        terminal_scope_id: None,
    }
}

/// Judge a first proposal against a capture declaration through the closed
/// recorder, with no owner call. Codes are the recorder's own.
pub fn validate_host_hpo_capture_proposal(
    request: &HostHpoCaptureSearchRequest,
    proposal: &HostHpoProposalEvidence,
) -> Result<()> {
    request.validate()?;
    proposal.validate()?;
    if let HostHpoProposalOutcome::Proposed {
        owner_ref_hex,
        owner_evidence_hex,
        located_choices,
        seed_resolutions,
        seed_uses,
        ..
    } = &proposal.outcome
    {
        if let Some(seed_id) = duplicate_seed_id(seed_resolutions) {
            return Err(DagMlError::RuntimeValidation(format!(
                "attempt_evidence: seed_resolution_conflict: seed `{seed_id}` appears twice"
            )));
        }
        let attempt = proposal_attempt(
            &request.capture,
            &BTreeMap::new(),
            0,
            None,
            owner_ref_hex.clone(),
            owner_evidence_hex.clone(),
            located_choices.clone(),
            seed_resolutions,
            seed_uses.clone(),
        );
        dry_check_attempt(&request.capture, &attempt)
            .map_err(|error| DagMlError::RuntimeValidation(format!("attempt_evidence: {error}")))?;
    }
    Ok(())
}

/// Known limit: a publication does not name its publisher. A capture whose
/// publication holds only replayed attempts that never reached a terminal leaves
/// no sealed trace of its id, so reusing that id cannot be detected here.
fn validate_restored_attempt(
    capture_id: &str,
    attempt: &HostHpoAttemptEvidence,
    request: &HostHpoSearchRequest,
    owner_refs: &mut BTreeSet<String>,
) -> Result<()> {
    let observation = &attempt.owner_observation;
    if attempt.capture_id == capture_id
        || attempt.terminal_capture_id.as_deref() == Some(capture_id)
        || !is_nonempty_capture_hex(&attempt.owner_ref_hex)
        || !owner_refs.insert(attempt.owner_ref_hex.clone())
        || observation.owner_ref_hex.as_deref() != Some(attempt.owner_ref_hex.as_str())
        || observation.ownership != realization::Ownership::Confirmed
        || attempt.phase_index != request.phase_index(attempt.trial_index)
    {
        return Err(host_hpo_predecessor_refused(
            "attempt_identity",
            format!(
                "trial {} has no unique owned identity in a prior lineage",
                attempt.trial_index
            ),
        ));
    }
    Ok(())
}

fn same_seed_fact(left: &HostHpoSeedResolution, right: &HostHpoSeedResolution) -> bool {
    left.effective == right.effective
        && left.resolution == right.resolution
        && left.parent == right.parent
        && left.policy == right.policy
}

/// A logical seed id has one value across lineages. Restored seeds are restated
/// per Proposal, so the recorder cannot see two attempts disagreeing, and it
/// cannot compare a new declaration with a sealed one. First use stays local.
fn validate_restored_seed_facts(
    capture: &HostHpoCaptureDeclaration,
    restored: &HostHpoCaptureCheckpoint,
) -> Result<()> {
    let conflict = |seed_id: &str| {
        host_hpo_predecessor_refused(
            "seed_conflict",
            format!("seed `{seed_id}` is restated with a different value"),
        )
    };
    let mut declared: BTreeMap<&str, &Option<realization::SeedValue>> = capture
        .seed_declarations
        .iter()
        .map(|seed| (seed.seed_id.as_str(), &seed.requested))
        .collect();
    let mut resolved: BTreeMap<&str, &HostHpoSeedResolution> = BTreeMap::new();
    let attempts = restored.attempts.iter().chain(
        restored
            .pending_attempts
            .iter()
            .map(|pending| &pending.attempt),
    );
    for attempt in attempts {
        for seed in &attempt.seed_declarations {
            let known = declared
                .entry(seed.seed_id.as_str())
                .or_insert(&seed.requested);
            if *known != &seed.requested {
                return Err(conflict(&seed.seed_id));
            }
        }
        for seed in &attempt.seed_resolutions {
            let known = resolved.entry(seed.seed_id.as_str()).or_insert(seed);
            if !same_seed_fact(known, seed) {
                return Err(conflict(&seed.seed_id));
            }
        }
    }
    Ok(())
}

fn validate_restored_attempts(
    capture_id: &str,
    request: &HostHpoSearchRequest,
    restored: &HostHpoCaptureCheckpoint,
) -> Result<()> {
    let trials = &restored.checkpoint.trials;
    if restored.attempts.len() != trials.len()
        || restored
            .pending_attempts
            .iter()
            .enumerate()
            .any(|(offset, pending)| pending.attempt.trial_index as usize != trials.len() + offset)
    {
        return Err(host_hpo_predecessor_refused(
            "attempt_coverage",
            "every restored trial requires its attempt, and pending attempts must follow them",
        ));
    }
    if restored
        .pending_attempts
        .iter()
        .any(|pending| pending.attempt.trial_index >= request.trial_budget)
    {
        return Err(host_hpo_predecessor_refused(
            "pending_outside_budget",
            "this budget could never serve a pending attempt the owner still holds",
        ));
    }
    let mut owner_refs = BTreeSet::new();
    for (attempt, trial) in restored.attempts.iter().zip(trials) {
        if attempt.trial_index != trial.trial_index() {
            return Err(host_hpo_predecessor_refused(
                "attempt_identity",
                "attempt evidence is bound to another trial",
            ));
        }
        validate_restored_attempt(capture_id, attempt, request, &mut owner_refs)?;
        let (params, variant_id, status, cause) = match trial {
            HostHpoTerminalTrial::Complete { evidence } => (
                &evidence.params,
                &evidence.variant_id,
                realization::TrialStatus::Completed,
                None,
            ),
            HostHpoTerminalTrial::Pruned { evidence } => (
                &evidence.params,
                &evidence.variant_id,
                realization::TrialStatus::Pruned,
                None,
            ),
            HostHpoTerminalTrial::Failed {
                params,
                variant_id,
                error,
                ..
            } => (
                params,
                variant_id,
                realization::TrialStatus::Failed,
                Some(capture_hex(error)),
            ),
        };
        validate_located_values(
            &attempt.owner_evidence_hex,
            &attempt.located_choices,
            params,
        )
        .map_err(|detail| host_hpo_predecessor_refused("attempt_evidence", detail))?;
        let failed = status == realization::TrialStatus::Failed;
        let observation = &attempt.owner_observation;
        let candidate_bound = match &attempt.candidate {
            Some(candidate) => {
                candidate.owner_variant_id == variant_id.as_str()
                    && candidate.parameters_hex == candidate_parameters_hex(params)?
            }
            None => failed && attempt.failure_stage != Some(realization::FailureStage::Evaluation),
        };
        // The owner's own diagnostic may honestly differ, for example by
        // truncation; only the coordinator's cause binds to the trial error.
        if observation.status != Some(status)
            || (failed && observation.cleanup != realization::CleanupState::Confirmed)
            || failed != attempt.failure_stage.is_some()
            || attempt.causal_diagnostic_hex != cause
            || attempt.terminal_capture_id.is_none()
            || attempt.terminal_scope_id.is_none()
            || !candidate_bound
        {
            return Err(host_hpo_predecessor_refused(
                "attempt_terminal",
                format!(
                    "trial {} lacks its actual terminal evidence",
                    attempt.trial_index
                ),
            ));
        }
    }
    for pending in &restored.pending_attempts {
        let attempt = &pending.attempt;
        validate_restored_attempt(capture_id, attempt, request, &mut owner_refs)?;
        validate_located_values(
            &attempt.owner_evidence_hex,
            &attempt.located_choices,
            &pending.params,
        )
        .map_err(|detail| host_hpo_predecessor_refused("attempt_evidence", detail))?;
        if attempt.owner_observation.status != Some(realization::TrialStatus::Running)
            || attempt.failure_stage.is_some()
            || attempt.causal_diagnostic_hex.is_some()
            || attempt.terminal_capture_id.is_some()
            || attempt.terminal_scope_id.is_some()
        {
            return Err(host_hpo_predecessor_refused(
                "attempt_terminal",
                format!(
                    "pending trial {} is not a running attempt",
                    attempt.trial_index
                ),
            ));
        }
    }
    Ok(())
}

/// Known limit: a prior attempt of unknown ownership refuses even once the pair
/// verifies, so a host that reports a raised ask is refused while one that omits
/// it passes on the digest alone. This stays conservative until Root decides
/// otherwise, and it is safe only if the owner digest covers all proposal state.
fn validate_prior_attempts(
    capture: &HostHpoCaptureDeclaration,
    publication: &HostHpoCapturePublication,
) -> Result<()> {
    let checkpoint = &publication.checkpoint;
    let required = host_hpo_capture_prefix_provenance(publication)?;
    for failed in &capture.failed_attempts {
        let Some(prefix) = &failed.trusted_prefix_ref else {
            if failed.owned_attempts.is_empty() {
                continue;
            }
            return Err(host_hpo_predecessor_refused(
                "unbound_prior_attempts",
                "failed attempts without a trusted prefix cannot pair with this publication",
            ));
        };
        if prefix.checkpoint_fingerprint != checkpoint.fingerprint {
            return Err(host_hpo_predecessor_refused(
                "prefix_mismatch",
                "trusted prefix does not name the restored checkpoint",
            ));
        }
        if prefix.pair_fingerprint != publication.pair_fingerprint {
            return Err(host_hpo_predecessor_refused(
                "pair_mismatch",
                "trusted prefix names another owner pair",
            ));
        }
        if !checkpoint
            .attempts
            .iter()
            .any(|attempt| attempt.owner_ref_hex == prefix.owner_ref_hex)
        {
            return Err(host_hpo_predecessor_refused(
                "prefix_owner_mismatch",
                "trusted prefix owner reference is not a restored attempt",
            ));
        }
    }
    for failed in &capture.failed_attempts {
        for attempt in &failed.owned_attempts {
            if attempt.ownership != realization::Ownership::Confirmed
                || !is_terminal_status(attempt.status)
            {
                return Err(host_hpo_predecessor_refused(
                    "prior_attempt_uncertain",
                    "a prior attempt has unknown ownership or no terminal observation",
                ));
            }
            if !checkpoint.attempts.iter().any(|restored| {
                attempt.owner_ref_hex.as_deref() == Some(restored.owner_ref_hex.as_str())
                    && restored.owner_observation.status == attempt.status
            }) {
                return Err(host_hpo_predecessor_refused(
                    "prior_attempt_outside_prefix",
                    "a prior attempt mutated owner state outside the restored prefix",
                ));
            }
        }
    }
    if let Some(missing) = required
        .iter()
        .find(|failed| !capture.failed_attempts.contains(failed))
    {
        return Err(host_hpo_predecessor_refused(
            "unpaired_failed_trial",
            format!(
                "failed attempt of capture {:?} lacks its exact derived provenance",
                missing.failed_capture_id
            ),
        ));
    }
    if capture
        .failed_attempts
        .iter()
        .any(|failed| !failed.owned_attempts.is_empty() && !required.contains(failed))
    {
        return Err(host_hpo_predecessor_refused(
            "prior_attempt_outside_prefix",
            "extra failed-attempt provenance may not carry owned attempts",
        ));
    }
    Ok(())
}

/// Pure cross-binding of capture lineage, observed owner state, a restored
/// publication and prior failed attempts, in the order a resume must be judged.
/// Detail strings start with a stable refusal code. `observed = None` checks
/// self-consistency only, before an owner read is available.
pub fn validate_host_hpo_capture_lineage(
    request: &HostHpoCaptureSearchRequest,
    publication: Option<&HostHpoCapturePublication>,
    observed: Option<&HostHpoProposalState>,
) -> Result<()> {
    request.validate()?;
    if let Some(state) = observed {
        validate_owner_state(state)?;
    }
    let capture = &request.capture;
    if capture
        .failed_attempts
        .iter()
        .any(|failed| failed.failed_capture_id.as_deref() == Some(capture.capture_id.as_str()))
    {
        return Err(host_hpo_predecessor_refused(
            "capture_lineage_reused",
            "a resumed capture requires a new capture_id",
        ));
    }
    let Some(publication) = publication else {
        if capture
            .failed_attempts
            .iter()
            .any(|failed| failed.trusted_prefix_ref.is_some())
        {
            return Err(host_hpo_predecessor_refused(
                "prefix_mismatch",
                "a trusted prefix requires its restored publication",
            ));
        }
        if observed.is_some_and(|state| state.owner_trial_count != 0)
            || capture
                .failed_attempts
                .iter()
                .any(|failed| !failed.owned_attempts.is_empty())
        {
            return Err(host_hpo_predecessor_refused(
                "fresh_store_required",
                "a capture without a publication requires an empty owner store",
            ));
        }
        return Ok(());
    };
    let restored = &publication.checkpoint;
    if restored.schema_version != CAPTURE_CHECKPOINT_SCHEMA_VERSION {
        return Err(host_hpo_predecessor_refused(
            "checkpoint_version",
            "captured resume requires checkpoint schema version 2",
        ));
    }
    if restored.source_digest != capture.source_digest {
        return Err(host_hpo_predecessor_refused(
            "declaration_mismatch",
            "restored checkpoint belongs to another declaration",
        ));
    }
    restored.verify_seal()?;
    restored
        .checkpoint
        .verify_seal()
        .map_err(|error| host_hpo_predecessor_refused("checkpoint_seal", error))?;
    let (Some(pair), Some(stored_owner)) = (
        &publication.pair_fingerprint,
        &publication.owner_checkpoint_fingerprint,
    ) else {
        return Err(host_hpo_predecessor_refused(
            "pair_unavailable",
            "this publication has no owner pair",
        ));
    };
    if !is_owner_state_digest(stored_owner) {
        return Err(host_hpo_predecessor_refused(
            "owner_state_invalid",
            "stored owner state must be a canonical 64-hex digest",
        ));
    }
    let stored = host_hpo_capture_pair_fingerprint(
        restored,
        stored_owner,
        publication.structural_binding.as_ref(),
    )?;
    if &stored != pair {
        return Err(host_hpo_predecessor_refused(
            "pair_mismatch",
            "stored pair does not cover this publication",
        ));
    }
    if let Some(state) = observed {
        let Some(owner) = &state.owner_checkpoint_fingerprint else {
            return Err(host_hpo_predecessor_refused(
                "pair_unavailable",
                "the owner has no pairable state",
            ));
        };
        let current =
            host_hpo_capture_pair_fingerprint(restored, owner, state.structural_binding.as_ref())?;
        if &current != pair {
            return Err(host_hpo_predecessor_refused(
                "pair_mismatch",
                "observed owner state does not pair with this publication",
            ));
        }
        let expected = restored.checkpoint.trials.len() + restored.pending_attempts.len();
        if state.owner_trial_count != expected as u64 {
            return Err(host_hpo_predecessor_refused(
                "owner_row_mismatch",
                "the owner holds a different number of trials",
            ));
        }
    }
    validate_restored_attempts(&capture.capture_id, &request.request, restored)?;
    validate_restored_seed_facts(capture, restored)?;
    validate_prior_attempts(capture, publication)?;
    dry_replay_restored(capture, restored)
}

fn validate_restored_candidates(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    restored: &HostHpoCaptureCheckpoint,
) -> Result<()> {
    let objective = restored.checkpoint.binding.objective_fingerprint.as_str();
    let terminal = restored.attempts.iter().zip(&restored.checkpoint.trials);
    let mut bound = Vec::new();
    for (attempt, trial) in terminal {
        let params = match trial {
            HostHpoTerminalTrial::Complete { evidence } => &evidence.params,
            HostHpoTerminalTrial::Pruned { evidence } => &evidence.params,
            HostHpoTerminalTrial::Failed { params, .. } => params,
        };
        bound.push((attempt, params));
    }
    bound.extend(
        restored
            .pending_attempts
            .iter()
            .map(|pending| (&pending.attempt, &pending.params)),
    );
    for (attempt, params) in bound {
        if let Some(candidate) = &attempt.candidate {
            let planned = host_hpo_candidate_plan(
                plan,
                request,
                attempt.trial_index,
                params,
                Some(objective),
            )
            .and_then(|plan| candidate_evidence(&plan, params))
            .map_err(|error| host_hpo_predecessor_refused("candidate_identity", error))?;
            if &planned != candidate {
                return Err(host_hpo_predecessor_refused(
                    "candidate_identity",
                    format!(
                        "trial {} candidate differs from its plan",
                        attempt.trial_index
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Complete qualified-resume validation. It calls no proposal, progress or
/// recovery operation, so hosts can run it before any owner mutation.
pub fn validate_host_hpo_capture_resume(
    plan: &ExecutionPlan,
    request: &HostHpoCaptureSearchRequest,
    options: Option<&HostHpoCaptureResumeOptions>,
    observed: Option<&HostHpoProposalState>,
) -> Result<()> {
    let publication = options.and_then(|options| options.publication.as_ref());
    validate_host_hpo_capture_lineage(request, publication, observed)?;
    if let Some(node) = plan.node_plans.keys().find(|node| {
        !request
            .capture
            .node_ids
            .iter()
            .any(|id| id == node.as_str())
    }) {
        return Err(DagMlError::RuntimeValidation(format!(
            "declaration_node_mismatch: plan node `{node}` is not declared"
        )));
    }
    if let Some(options) = options {
        prepare_host_hpo_checkpoint(plan, &request.request, &options.ordinary()).map_err(
            |error| {
                if publication.is_some() {
                    host_hpo_predecessor_refused("checkpoint_binding", error)
                } else {
                    error
                }
            },
        )?;
    }
    if let Some(publication) = publication {
        validate_restored_candidates(plan, &request.request, &publication.checkpoint)?;
    }
    Ok(())
}

#[derive(Clone)]
struct HostHpoCapturePrimary {
    trial_index: Option<u32>,
    stage: Option<realization::FailureStage>,
    descriptor: crate::error::DagMlErrorDescriptor,
    causal_diagnostic_hex: String,
}

impl HostHpoCapturePrimary {
    fn new(
        trial_index: Option<u32>,
        stage: Option<realization::FailureStage>,
        error: &DagMlError,
        causal_diagnostic_hex: Option<String>,
    ) -> Self {
        let descriptor = error.descriptor();
        Self {
            trial_index,
            stage,
            causal_diagnostic_hex: causal_diagnostic_hex
                .unwrap_or_else(|| capture_hex(&descriptor.message)),
            descriptor,
        }
    }
}

#[derive(Default)]
struct HostHpoCaptureLedger {
    /// Confirmed attempts not yet known terminal. Only these may be cleaned.
    pending: BTreeMap<u32, HostHpoPendingAttempt>,
    /// Sealed pending attempts not yet served again.
    retained: BTreeMap<u32, HostHpoPendingAttempt>,
    settled: BTreeMap<u32, HostHpoAttemptEvidence>,
    /// Checkpoint reference of each pending attempt replayed from a publication.
    origins: BTreeMap<u32, String>,
    /// Attempts whose Proposal scope is open in the stream.
    open_proposals: BTreeSet<u32>,
    /// The actual failing stage of a trial, where it is known.
    staged: BTreeMap<u32, realization::FailureStage>,
    /// Seeds resolved in this lineage's Search scope.
    resolved: BTreeMap<String, HostHpoSeedResolution>,
    /// Sealed seed facts of the restored publication, carried forward with no
    /// draw when this lineage first references them.
    restored_seeds: BTreeMap<String, HostHpoSeedResolution>,
    owner_refs: BTreeSet<String>,
    /// Failure and cleanup observations for the abort report.
    observations: Vec<realization::OwnedAttempt>,
    primary: Option<HostHpoCapturePrimary>,
    secondary_errors: Vec<crate::error::DagMlErrorDescriptor>,
    cleanup_errors: Vec<crate::error::DagMlErrorDescriptor>,
    operation_recorded: bool,
    owner_state: Option<HostHpoProposalState>,
    published: Option<HostHpoCapturePublication>,
    terminal: bool,
}

impl HostHpoCaptureLedger {
    /// Refuse a proposal's own seed facts before they reach the live recorder.
    fn seed_restatement_refusal(&self, seeds: &[HostHpoSeedResolution]) -> Option<String> {
        if let Some(seed_id) = duplicate_seed_id(seeds) {
            return Some(format!(
                "seed_resolution_conflict: seed `{seed_id}` appears twice"
            ));
        }
        seeds.iter().find_map(|seed| {
            let seed_id = &seed.seed_id;
            if self.resolved.contains_key(seed_id) {
                return Some(format!(
                    "seed_resolution_conflict: seed `{seed_id}` is already resolved"
                ));
            }
            self.restored_seeds
                .get(seed_id)
                .filter(|restored| !same_seed_fact(restored, seed))
                .map(|_| format!("seed_conflict: seed `{seed_id}` differs from its restored value"))
        })
    }

    fn record_unowned(&mut self, owner_ref_hex: Option<String>, diagnostic: &str) {
        self.observations.push(realization::OwnedAttempt {
            owner_ref_hex,
            ownership: realization::Ownership::Unknown,
            status: None,
            cleanup: realization::CleanupState::NotAttempted,
            diagnostic_hex: Some(capture_hex(diagnostic)),
        });
    }

    fn close_transitioned(
        &mut self,
        capture: &HostHpoCaptureDeclaration,
        trial_index: u32,
        status: realization::TrialStatus,
    ) -> Option<HostHpoAttemptEvidence> {
        let mut attempt = self.pending.remove(&trial_index)?.attempt;
        attempt.owner_observation.status = Some(status);
        attempt.terminal_capture_id = Some(capture.capture_id.clone());
        attempt.terminal_scope_id = Some(capture.scope_id.clone());
        self.settled.insert(trial_index, attempt.clone());
        Some(attempt)
    }

    /// Retain what the owner actually reported, keeping an unusable reply and a
    /// raised cleanup error uncertain rather than treating either as cleaned.
    fn observe(
        &mut self,
        owner_ref_hex: &str,
        diagnostic: &str,
        observed: Result<realization::OwnedAttempt>,
    ) -> realization::OwnedAttempt {
        let uncertain = |cleanup, detail: &str| realization::OwnedAttempt {
            owner_ref_hex: Some(owner_ref_hex.to_owned()),
            ownership: realization::Ownership::Confirmed,
            status: None,
            cleanup,
            diagnostic_hex: Some(capture_hex(detail)),
        };
        let mut attempt = match observed {
            Ok(attempt) => attempt,
            Err(error) => {
                self.cleanup_errors.push(error.descriptor());
                return uncertain(realization::CleanupState::Failed, &error.to_string());
            }
        };
        let confirmed = attempt.cleanup == realization::CleanupState::Confirmed;
        if attempt.owner_ref_hex.as_deref() != Some(owner_ref_hex)
            || attempt.ownership != realization::Ownership::Confirmed
            || (confirmed && !is_terminal_status(attempt.status))
            || attempt
                .diagnostic_hex
                .as_deref()
                .is_some_and(|diagnostic| !is_capture_hex(diagnostic))
        {
            let error = DagMlError::RuntimeValidation(format!(
                "owner returned an unusable cleanup observation for attempt `{owner_ref_hex}`"
            ));
            self.cleanup_errors.push(error.descriptor());
            return uncertain(realization::CleanupState::Unknown, &error.to_string());
        }
        if confirmed
            && attempt.status == Some(realization::TrialStatus::Failed)
            && attempt.diagnostic_hex.is_none()
        {
            attempt.diagnostic_hex = Some(capture_hex(diagnostic));
        }
        attempt
    }
}

/// One capture lineage writing one realization evidence stream. It carries the
/// owned-attempt ledger across a browser worker window. Once the stream ends,
/// by completion or abort, the session accepts no further work.
pub struct HostHpoCaptureSession<'w> {
    capture: HostHpoCaptureDeclaration,
    durable: bool,
    /// The publication this lineage restored. Every restated fact cites it,
    /// even after this capture publishes its own checkpoints.
    restored_checkpoint_ref: Option<String>,
    recorder: RefCell<realization::Recorder<Box<dyn std::io::Write + 'w>>>,
    /// The closed recorder poisons itself on refusal, so only its first error
    /// is meaningful; later records are skipped rather than repeated.
    evidence_failed: std::cell::Cell<bool>,
    evidence_errors: RefCell<Vec<crate::error::DagMlErrorDescriptor>>,
    /// Stable closed-gate code of the first refusal.
    refusal_code: RefCell<Option<String>>,
    ledger: RefCell<HostHpoCaptureLedger>,
}

struct HostHpoCaptureBridge<'a, 'w> {
    session: &'a HostHpoCaptureSession<'w>,
    source: &'a mut dyn HostHpoCaptureProposalSource,
}

impl HostHpoCaptureBridge<'_, '_> {
    fn pending_owner(&self, trial_index: u32) -> Result<String> {
        self.session
            .ledger
            .borrow()
            .pending
            .get(&trial_index)
            .map(|pending| pending.attempt.owner_ref_hex.clone())
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "host HPO capture trial {trial_index} is not a pending owned attempt"
                ))
            })
    }

    /// Record the terminal scope, then read the owner state the next
    /// publication pairs with. Recording first means a refusal blocks the seal.
    fn after_transition(
        &mut self,
        attempt: &HostHpoAttemptEvidence,
        status: realization::TrialStatus,
    ) -> Result<()> {
        self.session.emit_all(&attempt_terminal_events(
            &self.session.capture,
            attempt,
            status,
            realization::OutcomeMode::Transition,
        ))?;
        self.session
            .ledger
            .borrow_mut()
            .open_proposals
            .remove(&attempt.trial_index);
        self.session.refresh_owner_state(&mut *self.source);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bridge_failure(
        &self,
        trial_index: u32,
        phase_index: Option<u32>,
        validated: Result<()>,
        failure_stage: realization::FailureStage,
        causal_diagnostic_hex: String,
        consumption: realization::Consumption,
        owned_attempt: Option<realization::OwnedAttempt>,
    ) -> DagMlError {
        let session = self.session;
        if let Some(attempt) = owned_attempt {
            let live = attempt.ownership == realization::Ownership::Confirmed
                && matches!(
                    attempt.status,
                    None | Some(realization::TrialStatus::Running)
                );
            let mut ledger = session.ledger.borrow_mut();
            match attempt.owner_ref_hex.clone() {
                Some(owner_ref_hex)
                    if live
                        && is_nonempty_capture_hex(&owner_ref_hex)
                        && ledger.owner_refs.insert(owner_ref_hex.clone()) =>
                {
                    let attempt = proposal_attempt(
                        &session.capture,
                        &BTreeMap::new(),
                        trial_index,
                        phase_index,
                        owner_ref_hex,
                        String::new(),
                        Vec::new(),
                        &[],
                        Vec::new(),
                    );
                    ledger.pending.insert(
                        trial_index,
                        HostHpoPendingAttempt {
                            params: BTreeMap::new(),
                            attempt,
                        },
                    );
                }
                _ => ledger.observations.push(attempt),
            }
        }
        match validated {
            Ok(()) => {
                let cause = decode_capture_hex(&causal_diagnostic_hex).unwrap_or_default();
                let error = DagMlError::RuntimeValidation(format!(
                    "host HPO proposal failed: {}",
                    String::from_utf8_lossy(&cause)
                ));
                session.stage_failure(
                    trial_index,
                    failure_stage,
                    &error,
                    consumption,
                    Some(causal_diagnostic_hex),
                );
                error
            }
            Err(error) => {
                session.stage_failure(
                    trial_index,
                    realization::FailureStage::ProposalConversion,
                    &error,
                    realization::Consumption::Unknown,
                    None,
                );
                error
            }
        }
    }

    fn replay_retained(
        &self,
        trial_index: u32,
    ) -> Option<Result<BTreeMap<String, serde_json::Value>>> {
        let session = self.session;
        let mut ledger = session.ledger.borrow_mut();
        let pending = ledger.retained.remove(&trial_index)?;
        let checkpoint_ref = session.restored_checkpoint_ref.clone()?;
        let events = attempt_opening_events(
            &session.capture,
            &pending.attempt,
            Some(&checkpoint_ref),
            &[],
        );
        let params = pending.params.clone();
        ledger.origins.insert(trial_index, checkpoint_ref);
        ledger.pending.insert(trial_index, pending);
        if let Err(error) = session.emit_all(&events) {
            return Some(Err(error));
        }
        ledger.open_proposals.insert(trial_index);
        Some(Ok(params))
    }
}

impl HostHpoProposalSource for HostHpoCaptureBridge<'_, '_> {
    fn ask(&mut self, trial_index: u32) -> Result<Option<BTreeMap<String, serde_json::Value>>> {
        self.ask_in_phase(trial_index, None)
    }

    fn ask_in_phase(
        &mut self,
        trial_index: u32,
        phase_index: Option<u32>,
    ) -> Result<Option<BTreeMap<String, serde_json::Value>>> {
        if self.session.ledger.borrow().terminal {
            return Err(DagMlError::RuntimeValidation(
                "host HPO capture session is terminal and accepts no further ask".into(),
            ));
        }
        if let Some(replayed) = self.replay_retained(trial_index) {
            return replayed.map(Some);
        }
        let proposal = match self.source.ask_evidence(trial_index, phase_index) {
            Ok(proposal) => proposal,
            Err(error) => {
                // No returned reference: the requested index proves nothing.
                self.session
                    .ledger
                    .borrow_mut()
                    .record_unowned(None, &error.to_string());
                self.session.stage_failure(
                    trial_index,
                    realization::FailureStage::Ask,
                    &error,
                    realization::Consumption::Unknown,
                    None,
                );
                return Err(error);
            }
        };
        let validated = proposal.validate();
        match proposal.outcome {
            HostHpoProposalOutcome::Exhausted {} => {
                validated?;
                // Exhausting the owner may move it, and the final publication
                // must pair with the state it actually ended in.
                self.session.refresh_owner_state(&mut *self.source);
                Ok(None)
            }
            HostHpoProposalOutcome::Failed {
                failure_stage,
                causal_diagnostic_hex,
                consumption,
                owned_attempt,
            } => Err(self.bridge_failure(
                trial_index,
                phase_index,
                validated,
                failure_stage,
                causal_diagnostic_hex,
                consumption,
                owned_attempt,
            )),
            HostHpoProposalOutcome::Proposed {
                owner_ref_hex,
                owner_evidence_hex,
                params,
                located_choices,
                seed_resolutions,
                seed_uses,
            } => {
                let session = self.session;
                let mut ledger = session.ledger.borrow_mut();
                let usable = is_nonempty_capture_hex(&owner_ref_hex);
                if !usable || !ledger.owner_refs.insert(owner_ref_hex.clone()) {
                    let error = DagMlError::RuntimeValidation(format!(
                        "host HPO proposal {trial_index} has an unusable or duplicate owner reference"
                    ));
                    ledger.record_unowned(usable.then_some(owner_ref_hex), &error.to_string());
                    drop(ledger);
                    session.stage_failure(
                        trial_index,
                        realization::FailureStage::ProposalConversion,
                        &error,
                        realization::Consumption::Unknown,
                        None,
                    );
                    return Err(error);
                }
                let mut known = ledger.restored_seeds.clone();
                known.extend(ledger.resolved.clone());
                let attempt = proposal_attempt(
                    &session.capture,
                    &known,
                    trial_index,
                    phase_index,
                    owner_ref_hex,
                    owner_evidence_hex,
                    located_choices,
                    &seed_resolutions,
                    seed_uses,
                );
                // Ownership is registered before any further validation.
                ledger.pending.insert(
                    trial_index,
                    HostHpoPendingAttempt {
                        params: params.clone(),
                        attempt: attempt.clone(),
                    },
                );
                // The recorder would refuse this live and lose the stream.
                let refused = ledger.seed_restatement_refusal(&seed_resolutions);
                drop(ledger);
                let checked = validated.and_then(|()| match refused {
                    Some(detail) => Err(DagMlError::RuntimeValidation(format!(
                        "attempt_evidence: {detail}"
                    ))),
                    None => dry_check_attempt(&session.capture, &attempt).map_err(|error| {
                        DagMlError::RuntimeValidation(format!("attempt_evidence: {error}"))
                    }),
                });
                if let Err(error) = checked {
                    session.stage_failure(
                        trial_index,
                        realization::FailureStage::ProposalConversion,
                        &error,
                        realization::Consumption::Unknown,
                        None,
                    );
                    return Err(error);
                }
                // First reference in this lineage, whether the owner stated the
                // seed or the restored publication supplied it.
                let fresh = {
                    let ledger = session.ledger.borrow();
                    attempt
                        .seed_resolutions
                        .iter()
                        .filter(|seed| !ledger.resolved.contains_key(&seed.seed_id))
                        .map(|seed| seed.seed_id.clone())
                        .collect::<Vec<_>>()
                };
                session.emit_all(&attempt_opening_events(
                    &session.capture,
                    &attempt,
                    None,
                    &fresh,
                ))?;
                let mut ledger = session.ledger.borrow_mut();
                ledger.open_proposals.insert(trial_index);
                for seed in &attempt.seed_resolutions {
                    if fresh.contains(&seed.seed_id) {
                        ledger.resolved.insert(seed.seed_id.clone(), seed.clone());
                    }
                }
                Ok(Some(params))
            }
        }
    }

    fn candidate_planned(&mut self, trial_index: u32, plan: &ExecutionPlan) -> Result<()> {
        let session = self.session;
        let mut ledger = session.ledger.borrow_mut();
        let origin = ledger.origins.get(&trial_index).cloned();
        let pending = ledger.pending.get_mut(&trial_index).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "host HPO capture trial {trial_index} is not a pending owned attempt"
            ))
        })?;
        let candidate = candidate_evidence(plan, &pending.params)?;
        if let Some(sealed) = &pending.attempt.candidate {
            if sealed != &candidate {
                return Err(DagMlError::RuntimeValidation(format!(
                    "host HPO capture trial {trial_index} replanned a different candidate"
                )));
            }
            // A sealed candidate was already restated with its restored trial.
            return Ok(());
        }
        pending.attempt.candidate = Some(candidate.clone());
        let attempt = pending.attempt.clone();
        drop(ledger);
        let proposal = proposal_scope(trial_index);
        session.emit_all(&[
            candidate_event(&attempt, &candidate, origin.as_deref()),
            realization::Event::TrialBind {
                scope_id: proposal.clone(),
                trial_ref: realization::Ref {
                    scope_id: proposal.clone(),
                    id: trial_id(trial_index),
                },
                candidate_ref: realization::Ref {
                    scope_id: proposal,
                    id: visit_id(trial_index),
                },
            },
        ])
    }

    fn tell(&mut self, trial_index: u32, score: f64) -> Result<()> {
        let owner_ref_hex = self.pending_owner(trial_index)?;
        self.source.tell(&owner_ref_hex, score)?;
        let closed = self.session.ledger.borrow_mut().close_transitioned(
            &self.session.capture,
            trial_index,
            realization::TrialStatus::Completed,
        );
        match closed {
            Some(attempt) => self.after_transition(&attempt, realization::TrialStatus::Completed),
            None => Ok(()),
        }
    }

    fn report_intermediate(&mut self, trial_index: u32, step: u32, score: f64) -> Result<bool> {
        let owner_ref_hex = self.pending_owner(trial_index)?;
        self.source.report_intermediate(&owner_ref_hex, step, score)
    }

    fn pruned(&mut self, trial_index: u32) -> Result<()> {
        let owner_ref_hex = self.pending_owner(trial_index)?;
        self.source.pruned(&owner_ref_hex)?;
        let closed = self.session.ledger.borrow_mut().close_transitioned(
            &self.session.capture,
            trial_index,
            realization::TrialStatus::Pruned,
        );
        match closed {
            Some(attempt) => self.after_transition(&attempt, realization::TrialStatus::Pruned),
            None => Ok(()),
        }
    }

    fn fail(&mut self, trial_index: u32, error: &str) -> Result<()> {
        let (stage, pending) = {
            let mut ledger = self.session.ledger.borrow_mut();
            // Native edges stage first. A returned worker failure may not have
            // a stage at all, and then none is claimed.
            let stage = ledger.staged.get(&trial_index).copied();
            if ledger.primary.is_none() {
                ledger.primary = Some(HostHpoCapturePrimary::new(
                    Some(trial_index),
                    stage,
                    &DagMlError::RuntimeValidation(error.to_owned()),
                    Some(capture_hex(error)),
                ));
            }
            let pending = ledger.pending.remove(&trial_index).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "host HPO capture trial {trial_index} is not a pending owned attempt"
                ))
            })?;
            (stage, pending)
        };
        let mut attempt = pending.attempt;
        let observed = self.source.fail(&attempt.owner_ref_hex, stage, error);
        let observation = {
            let mut ledger = self.session.ledger.borrow_mut();
            let observation = ledger.observe(&attempt.owner_ref_hex, error, observed);
            ledger.observations.push(observation.clone());
            observation
        };
        if observation.status != Some(realization::TrialStatus::Failed)
            || observation.cleanup != realization::CleanupState::Confirmed
        {
            self.session.record_operation(
                realization::FailureStage::Cleanup,
                Some(trial_index),
                &capture_hex(error),
                Some(attempt.owner_ref_hex.clone()),
                realization::Consumption::Unknown,
            );
            return Err(DagMlError::RuntimeValidation(format!(
                "host HPO capture could not confirm Failed terminalization of trial {trial_index}"
            )));
        }
        attempt.failure_stage = stage;
        attempt.causal_diagnostic_hex = Some(capture_hex(error));
        attempt.terminal_capture_id = Some(self.session.capture.capture_id.clone());
        attempt.terminal_scope_id = Some(self.session.capture.scope_id.clone());
        attempt.owner_observation = observation;
        self.session
            .ledger
            .borrow_mut()
            .settled
            .insert(trial_index, attempt.clone());
        self.after_transition(&attempt, realization::TrialStatus::Failed)
    }

    fn stage_failure(
        &mut self,
        trial_index: u32,
        stage: realization::FailureStage,
        error: DagMlError,
    ) -> DagMlError {
        self.session
            .stage_failure(trial_index, stage, &error, stage_consumption(stage), None);
        error
    }
}

struct HostHpoCapturePublisher<'a, 'w> {
    session: &'a HostHpoCaptureSession<'w>,
    progress: &'a mut dyn HostHpoCaptureProgress,
}

impl HostHpoCapturePublisher<'_, '_> {
    fn document(&self, checkpoint: &HostHpoCheckpoint) -> Result<HostHpoCaptureCheckpoint> {
        let ledger = self.session.ledger.borrow();
        let attempts = checkpoint
            .trials
            .iter()
            .map(|trial| {
                let trial_index = trial.trial_index();
                ledger
                    .settled
                    .get(&trial_index)
                    .or_else(|| ledger.pending.get(&trial_index).map(|p| &p.attempt))
                    .cloned()
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "host HPO capture trial {trial_index} has no attempt evidence"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let terminal = checkpoint.trials.len() as u32;
        let pending_attempts = ledger
            .pending
            .iter()
            .chain(ledger.retained.iter())
            .filter(|(trial_index, _)| **trial_index >= terminal)
            .map(|(trial_index, pending)| (*trial_index, pending.clone()))
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect();
        let mut document = HostHpoCaptureCheckpoint {
            schema_version: CAPTURE_CHECKPOINT_SCHEMA_VERSION,
            checkpoint: checkpoint.clone(),
            source_digest: self.session.capture.source_digest.clone(),
            attempts,
            pending_attempts,
            fingerprint: String::new(),
        };
        document.seal()?;
        Ok(document)
    }

    fn publication(&self, checkpoint: &HostHpoCheckpoint) -> Result<HostHpoCapturePublication> {
        let document = self.document(checkpoint)?;
        let state = self.session.ledger.borrow().owner_state.clone();
        let owner_checkpoint_fingerprint = state
            .as_ref()
            .and_then(|state| state.owner_checkpoint_fingerprint.clone());
        let structural_binding = state.and_then(|state| state.structural_binding);
        let pair_fingerprint = owner_checkpoint_fingerprint
            .as_deref()
            .map(|owner| {
                host_hpo_capture_pair_fingerprint(&document, owner, structural_binding.as_ref())
            })
            .transpose()?;
        Ok(HostHpoCapturePublication {
            checkpoint: document,
            owner_checkpoint_fingerprint,
            structural_binding,
            pair_fingerprint,
        })
    }
}

impl HostHpoProgress for HostHpoCapturePublisher<'_, '_> {
    fn prepare_terminal(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> Result<()> {
        let document = self.document(checkpoint)?;
        self.progress.prepare_terminal(&document, status)
    }

    fn checkpoint(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> Result<bool> {
        let publication = self.publication(checkpoint)?;
        let proceed = self.progress.checkpoint(&publication, status)?;
        self.session.ledger.borrow_mut().published = Some(publication);
        Ok(proceed)
    }
}

impl<'w> HostHpoCaptureSession<'w> {
    /// Observe the owner once and validate the complete predecessor before any
    /// ask, then start the evidence stream. `options = None` is non-durable.
    pub fn open(
        plan: &ExecutionPlan,
        request: &HostHpoCaptureSearchRequest,
        options: Option<&HostHpoCaptureResumeOptions>,
        source: &mut dyn HostHpoCaptureProposalSource,
        sink: Box<dyn std::io::Write + 'w>,
    ) -> Result<Self> {
        let observed = source
            .proposal_state()
            .map_err(|error| host_hpo_predecessor_refused("owner_state_unavailable", error))?;
        validate_host_hpo_capture_resume(plan, request, options, Some(&observed))?;
        let capture = request.capture.clone();
        let refused = |error: realization::CaptureError| evidence_refusal(&error);
        let mut recorder =
            realization::Recorder::new(sink, &capture.capture_id, realization::Limits::default())
                .map_err(refused)?;
        let mut events = capture_opening_events(&capture);
        let mut ledger = HostHpoCaptureLedger::default();
        let mut restored_checkpoint_ref = None;
        if let Some(publication) = options.and_then(|options| options.publication.as_ref()) {
            let checkpoint = &publication.checkpoint;
            events.extend(restored_terminal_events(&capture, checkpoint)?);
            let pending_attempts = checkpoint.pending_attempts.iter().map(|p| &p.attempt);
            for attempt in checkpoint.attempts.iter().chain(pending_attempts) {
                ledger.owner_refs.insert(attempt.owner_ref_hex.clone());
                for seed in &attempt.seed_resolutions {
                    ledger
                        .restored_seeds
                        .entry(seed.seed_id.clone())
                        .or_insert_with(|| seed.clone());
                }
            }
            for attempt in &checkpoint.attempts {
                ledger.settled.insert(attempt.trial_index, attempt.clone());
            }
            for pending in &checkpoint.pending_attempts {
                ledger
                    .retained
                    .insert(pending.attempt.trial_index, pending.clone());
            }
            ledger.published = Some(publication.clone());
            restored_checkpoint_ref = Some(checkpoint.fingerprint.clone());
        }
        for event in &events {
            recorder.record(event).map_err(refused)?;
        }
        ledger.owner_state = Some(observed);
        Ok(Self {
            capture,
            durable: options.is_some(),
            restored_checkpoint_ref,
            recorder: RefCell::new(recorder),
            evidence_failed: std::cell::Cell::new(false),
            evidence_errors: RefCell::new(Vec::new()),
            refusal_code: RefCell::new(None),
            ledger: RefCell::new(ledger),
        })
    }

    fn evidence_refused(&self, error: &realization::CaptureError) -> DagMlError {
        self.evidence_failed.set(true);
        self.refusal_code
            .borrow_mut()
            .get_or_insert_with(|| error.code().to_owned());
        let error = evidence_refusal(error);
        self.evidence_errors.borrow_mut().push(error.descriptor());
        error
    }

    fn emit(&self, event: &realization::Event) -> Result<()> {
        if self.evidence_failed.get() {
            return Err(DagMlError::RuntimeValidation(
                "realization evidence stopped at an earlier refusal".into(),
            ));
        }
        self.recorder
            .borrow_mut()
            .record(event)
            .map_err(|error| self.evidence_refused(&error))
    }

    fn emit_all(&self, events: &[realization::Event]) -> Result<()> {
        events.iter().try_for_each(|event| self.emit(event))
    }

    /// A failed read leaves no pair, so the prefix is honestly not resumable.
    fn refresh_owner_state(&self, source: &mut dyn HostHpoCaptureProposalSource) {
        if !self.durable {
            return;
        }
        let observed = source
            .proposal_state()
            .and_then(|state| validate_owner_state(&state).map(|()| state));
        let mut ledger = self.ledger.borrow_mut();
        match observed {
            Ok(state) => ledger.owner_state = Some(state),
            Err(error) => {
                ledger.owner_state = None;
                ledger.secondary_errors.push(error.descriptor());
            }
        }
    }

    /// `OperationFailed` flushes, so the cause survives a later cleanup crash.
    /// A recording failure is kept as evidence and never interrupts the caller.
    fn record_operation(
        &self,
        stage: realization::FailureStage,
        trial_index: Option<u32>,
        diagnostic_hex: &str,
        owner_ref_hex: Option<String>,
        consumption: realization::Consumption,
    ) {
        let mut ledger = self.ledger.borrow_mut();
        let scope_id = trial_index
            .filter(|index| ledger.open_proposals.contains(index))
            .map_or_else(|| self.capture.search_scope_id.clone(), proposal_scope);
        let subject = trial_index.map_or_else(|| "search".to_owned(), |index| index.to_string());
        let ownership = if owner_ref_hex.is_some() {
            realization::Ownership::Confirmed
        } else {
            realization::Ownership::Unknown
        };
        let recorded = self.emit(&realization::Event::OperationFailed {
            scope_id,
            operation_id: format!("op:{}:{subject}", failure_stage_name(stage)),
            stage,
            diagnostic_hex: diagnostic_hex.to_owned(),
            owner_ref_hex,
            ownership,
            consumption,
        });
        ledger.operation_recorded |= recorded.is_ok();
    }

    fn stage_failure(
        &self,
        trial_index: u32,
        stage: realization::FailureStage,
        error: &DagMlError,
        consumption: realization::Consumption,
        causal_diagnostic_hex: Option<String>,
    ) {
        let (diagnostic_hex, owner_ref_hex) = {
            let mut ledger = self.ledger.borrow_mut();
            let refused = self.evidence_errors.borrow().contains(&error.descriptor());
            let current = HostHpoCapturePrimary::new(
                Some(trial_index),
                (!refused).then_some(stage),
                error,
                causal_diagnostic_hex,
            );
            let diagnostic_hex = current.causal_diagnostic_hex.clone();
            if !refused {
                ledger.staged.insert(trial_index, stage);
            }
            match &ledger.primary {
                Some(primary) if primary.descriptor != current.descriptor => {
                    ledger.secondary_errors.push(current.descriptor);
                }
                Some(_) => {}
                None => ledger.primary = Some(current),
            }
            let owner_ref_hex = ledger
                .pending
                .get(&trial_index)
                .map(|pending| pending.attempt.owner_ref_hex.clone());
            (diagnostic_hex, owner_ref_hex)
        };
        self.record_operation(
            stage,
            Some(trial_index),
            &diagnostic_hex,
            owner_ref_hex,
            consumption,
        );
    }

    fn trusted_prefix_ref(ledger: &HostHpoCaptureLedger) -> Option<realization::PrefixRef> {
        let published = ledger.published.as_ref()?;
        let checkpoint = &published.checkpoint;
        checkpoint
            .attempts
            .last()
            .map(|attempt| realization::PrefixRef {
                owner_ref_hex: attempt.owner_ref_hex.clone(),
                checkpoint_fingerprint: checkpoint.fingerprint.clone(),
                pair_fingerprint: published.pair_fingerprint.clone(),
            })
    }

    fn flush_status(&self) -> crate::error::HostHpoCaptureFlush {
        match self.recorder.borrow().refusal_flush_status() {
            realization::RefusalFlushStatus::NotAttempted => {
                crate::error::HostHpoCaptureFlush::NotAttempted
            }
            realization::RefusalFlushStatus::Flushed => crate::error::HostHpoCaptureFlush::Flushed,
            realization::RefusalFlushStatus::Failed(error) => {
                self.evidence_errors
                    .borrow_mut()
                    .push(evidence_refusal(error).descriptor());
                crate::error::HostHpoCaptureFlush::Failed
            }
        }
    }

    fn failure_report(
        &self,
        disposition: crate::error::HostHpoCaptureDisposition,
        primary: HostHpoCapturePrimary,
    ) -> DagMlError {
        let flush = self.flush_status();
        let mut ledger = self.ledger.borrow_mut();
        let failed_attempt = realization::FailedAttempt {
            failed_capture_id: Some(self.capture.capture_id.clone()),
            failed_scope_id: Some(self.capture.scope_id.clone()),
            failure_stage: primary.stage,
            causal_diagnostic_hex: primary.causal_diagnostic_hex,
            owned_attempts: std::mem::take(&mut ledger.observations),
            trusted_prefix_ref: Self::trusted_prefix_ref(&ledger),
        };
        DagMlError::HostHpoCaptureAbort(Box::new(crate::error::HostHpoCaptureAbortReport {
            disposition,
            primary_error: primary.descriptor,
            failed_attempt,
            secondary_errors: std::mem::take(&mut ledger.secondary_errors),
            cleanup_errors: std::mem::take(&mut ledger.cleanup_errors),
            evidence_errors: self.evidence_errors.take(),
            evidence: crate::error::HostHpoCaptureEvidence {
                refusal_code: self.refusal_code.take(),
                flush,
            },
            uncaptured_pending: ledger.retained.keys().copied().collect(),
        }))
    }

    /// Terminalize one pending attempt during an abort. Only a transition this
    /// cleanup confirmed closes its trial; anything else stays open and named.
    fn clean_attempt(
        &self,
        source: &mut dyn HostHpoCaptureProposalSource,
        primary: &HostHpoCapturePrimary,
        trial_index: u32,
        mut attempt: HostHpoAttemptEvidence,
    ) {
        let sibling = primary.stage == Some(realization::FailureStage::CandidatePlanning)
            && primary.trial_index != Some(trial_index);
        let diagnostic = if sibling {
            HOST_HPO_PLANNING_SIBLING_DIAGNOSTIC
        } else {
            primary.descriptor.message.as_str()
        };
        let observed = source.fail(&attempt.owner_ref_hex, primary.stage, diagnostic);
        let (observation, visited) = {
            let mut ledger = self.ledger.borrow_mut();
            let observation = ledger.observe(&attempt.owner_ref_hex, diagnostic, observed);
            ledger.observations.push(observation.clone());
            (observation, ledger.open_proposals.contains(&trial_index))
        };
        if !visited {
            return;
        }
        if observation.status == Some(realization::TrialStatus::Failed)
            && observation.cleanup == realization::CleanupState::Confirmed
        {
            attempt.failure_stage = primary.stage;
            attempt.causal_diagnostic_hex = Some(capture_hex(diagnostic));
            attempt.owner_observation = observation;
            let events = attempt_terminal_events(
                &self.capture,
                &attempt,
                realization::TrialStatus::Failed,
                realization::OutcomeMode::Transition,
            );
            if self.emit_all(&events).is_ok() {
                self.ledger.borrow_mut().open_proposals.remove(&trial_index);
            }
        } else {
            self.record_operation(
                realization::FailureStage::Cleanup,
                Some(trial_index),
                &capture_hex(diagnostic),
                Some(attempt.owner_ref_hex),
                realization::Consumption::Unknown,
            );
        }
    }

    /// End a stream that must not complete. `finish` is called once, as the
    /// refusal flush; an open Search or a recorded failure makes it refuse.
    fn refuse_stream(&self, primary: &HostHpoCapturePrimary) {
        let closable = {
            let ledger = self.ledger.borrow();
            ledger.operation_recorded
                && ledger.open_proposals.is_empty()
                && ledger.retained.is_empty()
        };
        if closable {
            // A refusal here is already retained by `emit`.
            let _closed = self.emit_all(&[
                realization::Event::SearchClose {
                    scope_id: self.capture.search_scope_id.clone(),
                    reason: realization::StopReason::Failed,
                    unvisited: None,
                    basis: realization::StopBasisKind::OperationError,
                    evidence: known_observation(Some(capture_hex(&primary.descriptor.message))),
                },
                realization::Event::ScopeClose {
                    scope_id: self.capture.scope_id.clone(),
                },
            ]);
        }
        self.restate_retained();
        self.finish_refused();
    }

    /// Show trials the owner still holds RUNNING without touching them. Their
    /// open trials keep the stream from closing, which is the honest result.
    fn restate_retained(&self) {
        let Some(checkpoint_ref) = self.restored_checkpoint_ref.as_deref() else {
            return;
        };
        let restated: Vec<realization::Event> = self
            .ledger
            .borrow()
            .retained
            .values()
            .flat_map(|pending| {
                attempt_opening_events(&self.capture, &pending.attempt, Some(checkpoint_ref), &[])
            })
            .collect();
        // A refusal here is already retained by `emit`.
        let _restated = self.emit_all(&restated);
    }

    fn finish_refused(&self) {
        if self.evidence_failed.get() {
            return;
        }
        match self.recorder.borrow_mut().finish() {
            Err(error) => {
                self.refusal_code
                    .borrow_mut()
                    .get_or_insert_with(|| error.code().to_owned());
            }
            Ok(_) => self.evidence_errors.borrow_mut().push(
                DagMlError::RuntimeValidation(
                    "an aborted capture reached completion; discard this stream".into(),
                )
                .descriptor(),
            ),
        }
    }

    /// Handle a propagated error: owner cleanup always runs, a recorder or sink
    /// failure cannot stop it, and nothing here seals or completes.
    fn abort(
        &self,
        error: DagMlError,
        source: &mut dyn HostHpoCaptureProposalSource,
    ) -> DagMlError {
        let (primary, pending) = {
            let mut ledger = self.ledger.borrow_mut();
            ledger.terminal = true;
            let returned = error.descriptor();
            let primary = match ledger.primary.clone() {
                Some(primary) => {
                    if returned != primary.descriptor
                        && !self.evidence_errors.borrow().contains(&returned)
                    {
                        ledger.secondary_errors.push(returned);
                    }
                    primary
                }
                None => HostHpoCapturePrimary::new(None, None, &error, None),
            };
            (primary, std::mem::take(&mut ledger.pending))
        };
        let cleaned = !pending.is_empty();
        for (trial_index, pending) in pending {
            self.clean_attempt(source, &primary, trial_index, pending.attempt);
        }
        // Cleanup moves the owner beyond the publication, so a prefix only stands
        // if nothing was cleaned and every observation is inside it.
        let published_failure = !cleaned && {
            let ledger = self.ledger.borrow();
            ledger.published.as_ref().is_some_and(|last| {
                let attempts = &last.checkpoint.attempts;
                let failure_published = primary.trial_index.is_some_and(|index| {
                    attempts.iter().any(|attempt| {
                        attempt.trial_index == index
                            && attempt.owner_observation.status
                                == Some(realization::TrialStatus::Failed)
                    })
                });
                failure_published
                    && ledger.observations.iter().all(|observed| {
                        attempts.iter().any(|attempt| {
                            observed.owner_ref_hex.as_deref()
                                == Some(attempt.owner_ref_hex.as_str())
                                && attempt.owner_observation.status == observed.status
                        })
                    })
            })
        };
        self.refuse_stream(&primary);
        let disposition = if published_failure {
            crate::error::HostHpoCaptureDisposition::PublishedFailedPrefix
        } else {
            crate::error::HostHpoCaptureDisposition::Aborted
        };
        self.failure_report(disposition, primary)
    }

    /// A trial the owner still holds RUNNING cannot be closed in a completed
    /// stream. Restate unserved attempts so the prefix shows them, then refuse,
    /// and report what actually reached the sink.
    fn refuse_uncaptured(&self) -> crate::error::HostHpoCaptureEvidence {
        self.restate_retained();
        self.finish_refused();
        let flush = self.flush_status();
        crate::error::HostHpoCaptureEvidence {
            refusal_code: self.refusal_code.take(),
            flush,
        }
    }

    /// Close and finish the stream of a returned status. A `Running` browser
    /// window keeps it open. Trials the owner still holds that no stream closed
    /// mean there is no completion.
    fn close_stream(&self, status: HostHpoSearchStatus) -> Result<HostHpoCaptureClosure> {
        let (reason, basis, evidence) = match status {
            HostHpoSearchStatus::Running => return Ok(HostHpoCaptureClosure::default()),
            HostHpoSearchStatus::Completed => (
                realization::StopReason::CountLimit,
                realization::StopBasisKind::OwnerStatus,
                None,
            ),
            HostHpoSearchStatus::Exhausted => (
                realization::StopReason::OwnerExhausted,
                realization::StopBasisKind::OwnerStatus,
                None,
            ),
            HostHpoSearchStatus::Cancelled => (
                realization::StopReason::Cancelled,
                realization::StopBasisKind::OwnerDecision,
                None,
            ),
            HostHpoSearchStatus::Failed => (
                realization::StopReason::Failed,
                realization::StopBasisKind::OwnerStatus,
                self.ledger
                    .borrow()
                    .primary
                    .as_ref()
                    .map(|primary| primary.causal_diagnostic_hex.clone()),
            ),
        };
        let uncaptured = {
            let mut ledger = self.ledger.borrow_mut();
            ledger.terminal = true;
            ledger
                .pending
                .keys()
                .chain(ledger.retained.keys())
                .copied()
                .collect::<BTreeSet<u32>>()
        };
        if !uncaptured.is_empty() {
            return Ok(HostHpoCaptureClosure {
                completion: None,
                uncaptured_pending: uncaptured.into_iter().collect(),
                refusal: Some(self.refuse_uncaptured()),
            });
        }
        self.emit_all(&[
            realization::Event::SearchClose {
                scope_id: self.capture.search_scope_id.clone(),
                reason,
                unvisited: None,
                basis,
                evidence: known_observation(evidence),
            },
            realization::Event::ScopeClose {
                scope_id: self.capture.scope_id.clone(),
            },
        ])?;
        let finished = self
            .recorder
            .borrow_mut()
            .finish()
            .map_err(|error| self.evidence_refused(&error))?;
        Ok(HostHpoCaptureClosure {
            completion: Some(HostHpoCaptureCompletion {
                completion: finished.completion().clone(),
                expected_commitment: finished.expected_commitment().clone(),
            }),
            ..HostHpoCaptureClosure::default()
        })
    }

    /// A returned outcome settles its failures, so they cannot be reported as
    /// the cause of a later abort in the same session.
    fn outcome(
        &self,
        outcome: HostHpoSearchOutcome,
        closure: HostHpoCaptureClosure,
    ) -> HostHpoCaptureOutcome {
        let mut ledger = self.ledger.borrow_mut();
        ledger.primary = None;
        ledger.observations.clear();
        let failed_attempts = ledger
            .published
            .as_ref()
            .and_then(|published| host_hpo_capture_prefix_provenance(published).ok())
            .unwrap_or_default();
        HostHpoCaptureOutcome {
            outcome,
            publication: ledger.published.clone(),
            attempts: ledger.settled.values().cloned().collect(),
            failed_attempts,
            completion: closure.completion,
            uncaptured_pending: closure.uncaptured_pending,
            refusal: closure.refusal,
            secondary_errors: std::mem::take(&mut ledger.secondary_errors),
            evidence_errors: self.evidence_errors.take(),
        }
    }

    fn conclude(
        &self,
        result: Result<HostHpoSearchOutcome>,
        source: &mut dyn HostHpoCaptureProposalSource,
    ) -> Result<HostHpoCaptureOutcome> {
        let outcome = result.map_err(|error| self.abort(error, source))?;
        match self.close_stream(outcome.status) {
            Ok(closure) => Ok(self.outcome(outcome, closure)),
            Err(error) => {
                let primary = HostHpoCapturePrimary::new(None, None, &error, None);
                Err(self.failure_report(
                    crate::error::HostHpoCaptureDisposition::EvidenceIncomplete,
                    primary,
                ))
            }
        }
    }

    fn run<F>(
        &self,
        options: Option<&HostHpoCaptureResumeOptions>,
        source: &mut dyn HostHpoCaptureProposalSource,
        progress: Option<&mut dyn HostHpoCaptureProgress>,
        search: F,
    ) -> Result<HostHpoCaptureOutcome>
    where
        F: FnOnce(
            &mut dyn HostHpoProposalSource,
            Option<(&HostHpoResumeOptions, &mut dyn HostHpoProgress)>,
        ) -> Result<HostHpoSearchOutcome>,
    {
        let ordinary = options.map(HostHpoCaptureResumeOptions::ordinary);
        let mut bridge = HostHpoCaptureBridge {
            session: self,
            source: &mut *source,
        };
        let mut publisher = progress.map(|progress| HostHpoCapturePublisher {
            session: self,
            progress,
        });
        let durable = ordinary
            .as_ref()
            .zip(publisher.as_mut())
            .map(|(options, publisher)| (options, publisher as &mut dyn HostHpoProgress));
        let result = search(&mut bridge as &mut dyn HostHpoProposalSource, durable);
        self.conclude(result, source)
    }

    /// `window = None` opens one; a window to complete must own exactly the
    /// pending attempts, and may be empty when the search has nothing to ask.
    fn admit_window(
        &self,
        request: &HostHpoCaptureSearchRequest,
        options: &HostHpoCaptureResumeOptions,
        window: Option<&HostHpoWorkerWindow>,
    ) -> Result<()> {
        let ledger = self.ledger.borrow();
        let published = ledger
            .published
            .as_ref()
            .map(|publication| publication.checkpoint.fingerprint.as_str());
        let supplied = options
            .publication
            .as_ref()
            .map(|publication| publication.checkpoint.fingerprint.as_str());
        let window_bound = match window {
            Some(window) => ledger
                .pending
                .keys()
                .copied()
                .eq(window.tasks.iter().map(|task| task.trial_index)),
            None => ledger.pending.is_empty(),
        };
        if !self.durable
            || ledger.terminal
            || !window_bound
            || request.capture.capture_id != self.capture.capture_id
            || supplied != published
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO capture window requires this open durable session, one window at a time and its published checkpoint"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Capture-gated [`prepare_host_hpo_worker_window`]. Asked trials stay
    /// pending in this session until their window completes, and a second
    /// window cannot open meanwhile.
    pub fn prepare_worker_window(
        &self,
        plan: &ExecutionPlan,
        request: &HostHpoCaptureSearchRequest,
        options: &HostHpoCaptureResumeOptions,
        source: &mut dyn HostHpoCaptureProposalSource,
        max_workers: usize,
    ) -> Result<HostHpoWorkerWindow> {
        self.admit_window(request, options, None)?;
        let ordinary = options.ordinary();
        let mut bridge = HostHpoCaptureBridge {
            session: self,
            source: &mut *source,
        };
        let result = prepare_host_hpo_worker_window(
            plan,
            &request.request,
            &ordinary,
            &mut bridge,
            max_workers,
        );
        result.map_err(|error| self.abort(error, source))
    }

    /// Capture-gated [`complete_host_hpo_worker_window`]. The v1 worker result
    /// stays stageless, so `failure_stages` carries the actual stage of each
    /// failure a worker could name. A failed trial left out of it claims no
    /// stage: it completes honestly but cannot resume, because qualified resume
    /// needs the actual stage. A returned `Failed` status is an observed outcome
    /// that completes the stream; only a propagated error aborts.
    #[allow(clippy::too_many_arguments)]
    pub fn complete_worker_window(
        &self,
        plan: &ExecutionPlan,
        request: &HostHpoCaptureSearchRequest,
        options: &HostHpoCaptureResumeOptions,
        window: HostHpoWorkerWindow,
        results: Vec<HostHpoWorkerResult>,
        failure_stages: BTreeMap<u32, realization::FailureStage>,
        source: &mut dyn HostHpoCaptureProposalSource,
        progress: &mut dyn HostHpoCaptureProgress,
    ) -> Result<HostHpoCaptureOutcome> {
        self.admit_window(request, options, Some(&window))?;
        for (trial_index, stage) in &failure_stages {
            let reported_failed = results.iter().any(|result| {
                matches!(
                    result,
                    HostHpoWorkerResult::Failed { trial_index: failed, .. } if failed == trial_index
                )
            });
            let worker_stage = matches!(
                stage,
                realization::FailureStage::RunContextCreation
                    | realization::FailureStage::ProviderCreation
                    | realization::FailureStage::ControllerCreation
                    | realization::FailureStage::Evaluation
            );
            if !reported_failed || !worker_stage {
                return Err(DagMlError::RuntimeValidation(format!(
                    "worker failure stage for trial {trial_index} is not a worker stage of a reported failure"
                )));
            }
        }
        self.ledger.borrow_mut().staged.extend(failure_stages);
        let ordinary = options.ordinary();
        let mut bridge = HostHpoCaptureBridge {
            session: self,
            source: &mut *source,
        };
        let mut publisher = HostHpoCapturePublisher {
            session: self,
            progress,
        };
        let result = complete_host_hpo_worker_window(
            plan,
            &request.request,
            &ordinary,
            window,
            results,
            &mut bridge,
            &mut publisher,
        );
        self.conclude(result, source)
    }
}

impl SequentialScheduler {
    /// Capture-gated serial search, durable or not. It writes one evidence
    /// stream to `sink`; a returned status finishes it with a real completion,
    /// and a propagated failure terminalizes pending owned attempts and returns
    /// a typed abort with no completion.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_captured_host_hpo_search(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        provider_factory: Option<&dyn HostHpoCandidateProviderFactory>,
        controller_factory: Option<&dyn HostHpoCandidateControllerFactory>,
        request: &HostHpoCaptureSearchRequest,
        source: &mut dyn HostHpoCaptureProposalSource,
        sink: Box<dyn std::io::Write + '_>,
        durable: Option<(
            &HostHpoCaptureResumeOptions,
            &mut dyn HostHpoCaptureProgress,
        )>,
    ) -> Result<HostHpoCaptureOutcome> {
        let (options, progress) = durable.unzip();
        let session = HostHpoCaptureSession::open(plan, request, options, &mut *source, sink)?;
        session.run(options, source, progress, |proposals, durable| {
            self.execute_host_hpo_search_inner(
                plan,
                controllers,
                provider,
                provider_factory,
                controller_factory,
                &request.request,
                proposals,
                durable,
            )
        })
    }

    /// Capture-gated parallel windows. Asks and owner transitions stay on the
    /// coordinator thread, so the session never crosses into a worker.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_captured_parallel_host_hpo_search(
        &self,
        plan: &ExecutionPlan,
        provider_factory: &dyn HostHpoCandidateProviderFactory,
        controller_factory: &dyn HostHpoCandidateControllerFactory,
        request: &HostHpoCaptureSearchRequest,
        source: &mut dyn HostHpoCaptureProposalSource,
        sink: Box<dyn std::io::Write + '_>,
        max_parallel_trials: usize,
        durable: Option<(
            &HostHpoCaptureResumeOptions,
            &mut dyn HostHpoCaptureProgress,
        )>,
    ) -> Result<HostHpoCaptureOutcome> {
        let (options, progress) = durable.unzip();
        let session = HostHpoCaptureSession::open(plan, request, options, &mut *source, sink)?;
        session.run(options, source, progress, |proposals, durable| {
            self.execute_parallel_host_hpo_search_inner(
                plan,
                provider_factory,
                controller_factory,
                &request.request,
                proposals,
                max_parallel_trials,
                durable,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_json_object_order_preserves_legacy_seal_and_rejects_content_tamper() {
        let mut checkpoint: HostHpoCheckpoint = serde_json::from_str(
            r#"{"schema_version":1,"binding":{"objective_fingerprint":"objective",
            "graph_fingerprint":"graph","controller_fingerprint":"controllers",
            "campaign_fingerprint":"campaign","fold_set_fingerprint":"folds",
            "data_fingerprint":"data"},"trials":[{"state":"failed","trial_index":0,
            "params":{"model":{"z":2,"a":1}},"variant_id":"trial:0","error":"fit failed",
            "generated_view_manifest":{"views":[{"sample_ids":["s2","s1"],
            "content":{"z":"sha-z","a":"sha-a"}}],"schema_version":1}}],"fingerprint":""}"#,
        )
        .unwrap();
        checkpoint.seal().unwrap();
        assert_eq!(
            checkpoint.fingerprint,
            "c012ba760617bcc51bb630baa6341292eab941691f2ab3b0dadb6657e1e3485f"
        );
        // Legacy typed struct field order; only JSON object keys are sorted.
        let legacy: HostHpoCheckpoint = serde_json::from_str(
            r#"{"schema_version":1,"binding":{"objective_fingerprint":"objective",
            "graph_fingerprint":"graph","controller_fingerprint":"controllers",
            "campaign_fingerprint":"campaign","fold_set_fingerprint":"folds",
            "data_fingerprint":"data"},"trials":[{"state":"failed","trial_index":0,
            "params":{"model":{"a":1,"z":2}},"variant_id":"trial:0","error":"fit failed",
            "generated_view_manifest":{"schema_version":1,"views":[{"content":{"a":"sha-a","z":"sha-z"},
            "sample_ids":["s2","s1"]}]}}],"fingerprint":""}"#,
        ).unwrap();
        assert_eq!(
            checkpoint.fingerprint,
            stable_json_fingerprint(&(legacy.schema_version, &legacy.binding, &legacy.trials,))
                .unwrap()
        );
        let mut sorted_wire = serde_json::to_value(&checkpoint).unwrap();
        sorted_wire.sort_all_objects(); // SDK checkpoint json.dump(sort_keys=True).
        let restored: HostHpoCheckpoint = serde_json::from_value(sorted_wire.clone()).unwrap();
        restored.verify_seal().unwrap();
        assert_eq!(restored.fingerprint, checkpoint.fingerprint);
        let mut changed = sorted_wire.clone();
        changed["trials"][0]["generated_view_manifest"]["views"][0]["content"]["a"] =
            serde_json::json!("foreign-content");
        assert!(serde_json::from_value::<HostHpoCheckpoint>(changed)
            .unwrap()
            .verify_seal()
            .is_err());
        let mut changed = sorted_wire;
        changed["trials"][0]["generated_view_manifest"]["views"][0]["sample_ids"] =
            serde_json::json!(["s1", "s2"]);
        assert!(serde_json::from_value::<HostHpoCheckpoint>(changed)
            .unwrap()
            .verify_seal()
            .is_err());

        // Recovery must compare canonical prospective prefixes too, without
        // changing the original checkpoint or accepting a tampered prefix.
        let mut prepared = checkpoint.clone();
        prepared.trials.push(HostHpoTerminalTrial::Failed {
            trial_index: 1,
            params: BTreeMap::new(),
            variant_id: VariantId::new("trial:1").unwrap(),
            error: "fit failed".into(),
            generated_view_manifest: None,
        });
        prepared.seal().unwrap();
        assert_eq!(
            restored
                .clone()
                .recover_interrupted_trials(Some(prepared), Vec::new())
                .unwrap()
                .trials
                .len(),
            2
        );
        let mut malformed = restored.clone();
        if let HostHpoTerminalTrial::Failed { params, .. } = &mut malformed.trials[0] {
            params.insert("extra".into(), serde_json::json!(1));
        }
        assert!(malformed
            .recover_interrupted_trials(None, Vec::new())
            .is_err());
    }

    #[test]
    fn objective_identity_ignores_resume_controls_and_json_object_order() {
        let original: HostHpoSearchRequest = serde_json::from_str(
            r#"{"target_node":"model:base","trial_budget":2,"metric":"rmse","direction":"minimize",
            "optimizer_descriptor":{"sampler":"random","n_trials":2,"seed":7,"storage":"first",
            "model_params":{"n_components":[1,2],"alpha":[0,1]}}}"#,
        )
        .unwrap();
        let resumed: HostHpoSearchRequest = serde_json::from_str(
            r#"{"target_node":"model:base","trial_budget":3,"metric":"rmse","direction":"minimize",
            "optimizer_descriptor":{"resume":true,"storage":"second","n_trials":3,"seed":7,
            "model_params":{"alpha":[0,1],"n_components":[1,2]},"sampler":"random"}}"#,
        )
        .unwrap();
        let expected = host_hpo_objective_fingerprint(&original).unwrap();
        // SHA-256 of the legacy recursively key-sorted JSON objective; this
        // remains the same with and without serde_json's preserve_order feature.
        assert_eq!(
            expected,
            "db558102f5eb6cebc7047a8447d430e35573df3124d2009d63e4df9e958fb37c"
        );
        assert_eq!(expected, host_hpo_objective_fingerprint(&resumed).unwrap());
        let mut changed = resumed;
        changed
            .optimizer_descriptor
            .get_mut("model_params")
            .unwrap()["n_components"] = serde_json::json!([1, 3]);
        assert_ne!(expected, host_hpo_objective_fingerprint(&changed).unwrap());
    }

    #[cfg(dag_ml_workspace_contract_fixtures)]
    #[test]
    fn torch_effective_plan_cannot_bypass_serial_admission_with_a_regression_metric() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../examples/fixtures/training/python_training_smoke.v1.json"
        ))
        .unwrap();
        let training: crate::TrainingRequest =
            serde_json::from_value(fixture["request"].clone()).unwrap();
        let mut manifests = crate::ControllerRegistry::new();
        for manifest in training.controller_manifests {
            manifests.register(manifest).unwrap();
        }
        let mut plan = crate::build_execution_plan(
            "plan:torch.refusal",
            training.graph,
            training.campaign,
            &manifests,
        )
        .unwrap();
        let request: HostHpoSearchRequest = serde_json::from_value(serde_json::json!({
            "target_node":"model:base","trial_budget":2,"metric":"rmse","direction":"minimize",
            "optimizer_descriptor":{"sampler":"random","n_jobs":2}}))
        .unwrap();
        assert!(
            request
                .validate_parallel_execution_for_plan(2, &plan)
                .is_ok(),
            "historical generic parallel route unchanged"
        );
        plan.node_plans
            .values_mut()
            .find(|node| node.kind == crate::NodeKind::Model)
            .unwrap()
            .params
            .insert(
                "factory_path".into(),
                serde_json::json!(crate::python_torch_profile::TORCH_FACTORY),
            );
        assert!(request
            .validate_parallel_execution_for_plan(2, &plan)
            .unwrap_err()
            .to_string()
            .contains("serial"));
        assert!(
            crate::python_torch_profile::validate_torch_plan(&plan).is_err(),
            "known factory without signed metadata must refuse"
        );
    }

    #[test]
    fn fold_reduction_is_not_global_oof_and_honors_metric_direction() {
        use crate::selection::MetricObjective::{Maximize, Minimize};
        let scores = BTreeMap::from([("fold0".into(), 1.0), ("fold1".into(), 3.0)]);
        assert_eq!(
            reduce_host_hpo_fold_scores(&scores, HostHpoFoldReduction::Mean, Minimize).unwrap(),
            2.0
        );
        assert_ne!(
            2.0,
            5_f64.sqrt(),
            "mean per-fold RMSE must not become pooled OOF RMSE"
        );
        for reduction in [HostHpoFoldReduction::Best, HostHpoFoldReduction::RobustBest] {
            assert_eq!(
                reduce_host_hpo_fold_scores(&scores, reduction, Minimize).unwrap(),
                1.0
            );
            assert_eq!(
                reduce_host_hpo_fold_scores(&scores, reduction, Maximize).unwrap(),
                3.0
            );
        }
        assert!(reduce_host_hpo_fold_scores(
            &BTreeMap::new(),
            HostHpoFoldReduction::Mean,
            Minimize
        )
        .is_err());
        let invalid = BTreeMap::from([("fold0".into(), f64::INFINITY)]);
        assert!(
            reduce_host_hpo_fold_scores(&invalid, HostHpoFoldReduction::RobustBest, Minimize)
                .is_err(),
            "failed trials cannot become synthetic penalties or disappear from a fold reduction"
        );
    }
}
