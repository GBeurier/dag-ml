//! Nonportable host-optimizer search with scheduler-owned candidate execution.
//!
//! The host proposes parameter values and receives native scores only. It never
//! supplies predictions, folds, or scalar scores through the tuner interface.
//! This contract deliberately has no Methods ABI or N4MOPT checkpoint claim.

use super::*;

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
}

impl HostHpoSearchRequest {
    fn phase_index(&self, trial_index: u32) -> Option<u32> {
        if self.phase_trial_budgets.is_empty() {
            return None;
        }
        let mut end = 0;
        for (index, budget) in self.phase_trial_budgets.iter().enumerate() {
            end += budget;
            if trial_index < end {
                return Some(index as u32);
            }
        }
        None
    }

    fn validate_parameter_bindings(&self, plan: &ExecutionPlan) -> Result<()> {
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

    fn parameter_overrides(
        &self,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Vec<crate::generation::GenerationParamOverride>> {
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostHpoTrialEvidence {
    pub trial_index: u32,
    pub params: BTreeMap<String, serde_json::Value>,
    pub score: f64,
    pub variant_id: VariantId,
    pub scores: ScoreSet,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub objective_fold_scores: BTreeMap<String, f64>,
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
pub struct HostHpoPrunedTrialEvidence {
    pub trial_index: u32,
    pub params: BTreeMap<String, serde_json::Value>,
    pub variant_id: VariantId,
    pub scores: ScoreSet,
    pub intermediate_scores: Vec<f64>,
}

enum HostHpoEvaluation {
    Complete(HostHpoTrialEvidence, crate::selection::CandidateScore),
    Pruned(HostHpoPrunedTrialEvidence),
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

impl HostHpoCheckpoint {
    fn seal(&mut self) -> Result<()> {
        self.fingerprint =
            stable_json_fingerprint(&(self.schema_version, &self.binding, &self.trials))?;
        Ok(())
    }
}

/// Return false only to request cancellation at this completed-trial boundary.
pub trait HostHpoProgress {
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
        self.execute_host_hpo_search_inner(plan, controllers, provider, request, proposals, None)?
            .result
            .ok_or_else(|| {
                DagMlError::RuntimeValidation("host HPO has no successful candidate".into())
            })
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
            request,
            proposals,
            Some((options, progress)),
        )
    }

    fn execute_host_hpo_search_inner(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        provider: &dyn RuntimeDataProvider,
        request: &HostHpoSearchRequest,
        proposals: &mut dyn HostHpoProposalSource,
        mut durable: Option<(&HostHpoResumeOptions, &mut dyn HostHpoProgress)>,
    ) -> Result<HostHpoSearchOutcome> {
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
            let param_overrides = request.parameter_overrides(&params)?;
            let mut variant = plan.variants[0].clone();
            variant.variant_id = VariantId::new(format!("host_hpo:trial:{trial_index:010}"))?;
            variant.choices.insert(
                "host_hpo".into(),
                GenerationChoice {
                    label: format!("trial:{trial_index}"),
                    value: serde_json::json!({"trial_index": trial_index}),
                    param_overrides,
                    active_subsequence: None,
                },
            );
            variant.fingerprint = if let Some(checkpoint) = &checkpoint {
                stable_json_fingerprint(&(
                    &plan.variants[0].fingerprint,
                    &variant.choices,
                    &checkpoint.binding.objective_fingerprint,
                ))?
            } else {
                stable_json_fingerprint(&(
                    &plan.variants[0].fingerprint,
                    &variant.choices,
                    request,
                ))?
            };
            let mut candidate_plan = plan.clone();
            candidate_plan.variants = vec![variant.clone()];
            candidate_plan.validate()?;
            let mut context = RunContext::new(
                RunId::new(format!("run:host_hpo:{trial_index}"))?,
                variant.seed.or(plan.campaign.root_seed),
            );
            context.variant_id = Some(variant.variant_id.clone());
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
                context.collect_cross_fold_validation_scores(plan_oof_partition_mode(plan))?;
                let scores = context
                    .build_score_set(plan.id.clone(), None)
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation("host HPO lost native score evidence".into())
                    })?;
                let (score, objective_fold_scores, candidate) =
                    host_hpo_score(plan, request, &variant.variant_id, &scores)?;
                Ok(HostHpoEvaluation::Complete(
                    HostHpoTrialEvidence {
                        trial_index,
                        params: params.clone(),
                        score,
                        variant_id: variant.variant_id.clone(),
                        scores,
                        objective_fold_scores,
                    },
                    candidate,
                ))
            })();
            let evaluated = match evaluated {
                Ok(result) => result,
                Err(error) => {
                    if let (Some(checkpoint), Some((_, progress))) = (&mut checkpoint, &mut durable)
                    {
                        proposals.fail(trial_index, &error.to_string())?;
                        checkpoint.trials.push(HostHpoTerminalTrial::Failed {
                            trial_index,
                            params,
                            variant_id: variant.variant_id,
                            error: error.to_string(),
                        });
                        checkpoint.seal()?;
                        progress.checkpoint(checkpoint, HostHpoSearchStatus::Failed)?;
                    }
                    return Err(error);
                }
            };
            if let HostHpoEvaluation::Pruned(evidence) = evaluated {
                proposals.pruned(trial_index)?;
                pruned_trials.push(evidence.clone());
                status = if trial_index + 1 == request.trial_budget {
                    HostHpoSearchStatus::Completed
                } else {
                    HostHpoSearchStatus::Running
                };
                if let (Some(checkpoint), Some((_, progress))) = (&mut checkpoint, &mut durable) {
                    checkpoint
                        .trials
                        .push(HostHpoTerminalTrial::Pruned { evidence });
                    checkpoint.seal()?;
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
            proposals.tell(trial_index, evidence.score)?;
            candidates.push(candidate);
            trials.push(evidence.clone());
            status = if trial_index + 1 == request.trial_budget {
                HostHpoSearchStatus::Completed
            } else {
                HostHpoSearchStatus::Running
            };
            if let (Some(checkpoint), Some((_, progress))) = (&mut checkpoint, &mut durable) {
                checkpoint
                    .trials
                    .push(HostHpoTerminalTrial::Complete { evidence });
                checkpoint.seal()?;
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
    stable_json_fingerprint(&value)
}

fn prepare_host_hpo_checkpoint(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    options: &HostHpoResumeOptions,
) -> Result<HostHpoCheckpoint> {
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
        let (params, variant_id) = match trial {
            HostHpoTerminalTrial::Complete { evidence } => {
                host_hpo_candidate(plan, request, evidence)?;
                (&evidence.params, &evidence.variant_id)
            }
            HostHpoTerminalTrial::Pruned { evidence } => {
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
                            report.producer_node == request.target_node
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
        if params.is_empty()
            || params.keys().any(|key| key.trim().is_empty())
            || variant_id.as_str() != format!("host_hpo:trial:{index:010}")
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO checkpoint has invalid parameter/variant identity".into(),
            ));
        }
        request.parameter_overrides(params)?;
    }
    Ok(checkpoint)
}

fn host_hpo_candidate(
    plan: &ExecutionPlan,
    request: &HostHpoSearchRequest,
    trial: &HostHpoTrialEvidence,
) -> Result<crate::selection::CandidateScore> {
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
    let (score, fold_scores, candidate) =
        host_hpo_score(plan, request, &trial.variant_id, &trial.scores)?;
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
    let folds = plan.fold_set.as_ref().expect("validated host HPO FoldSet");
    let reports = scores
        .reports
        .iter()
        .filter(|report| {
            report.producer_node == request.target_node
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

#[cfg(test)]
mod tests {
    use super::*;

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
