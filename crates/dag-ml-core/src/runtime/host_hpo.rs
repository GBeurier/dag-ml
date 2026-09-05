//! Nonportable host-optimizer search with scheduler-owned candidate execution.
//!
//! The host proposes parameter values and receives native scores only. It never
//! supplies predictions, folds, or scalar scores through the tuner interface.
//! This contract deliberately has no Methods ABI or N4MOPT checkpoint claim.

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHpoSearchRequest {
    pub target_node: NodeId,
    pub trial_budget: u32,
    pub metric: RegressionMetricKind,
    pub direction: crate::selection::MetricObjective,
    pub optimizer_descriptor: BTreeMap<String, serde_json::Value>,
}

pub trait HostHpoProposalSource {
    fn ask(&mut self, trial_index: u32) -> Result<Option<BTreeMap<String, serde_json::Value>>>;
    fn tell(&mut self, trial_index: u32, score: f64) -> Result<()>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostHpoTrialEvidence {
    pub trial_index: u32,
    pub params: BTreeMap<String, serde_json::Value>,
    pub score: f64,
    pub variant_id: VariantId,
    pub scores: ScoreSet,
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
    pub selected_trial_index: u32,
    pub selected_params: BTreeMap<String, serde_json::Value>,
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
        plan.validate()?;
        if request.trial_budget == 0 || request.optimizer_descriptor.is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO requires a positive budget and explicit optimizer descriptor".into(),
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
        let mut trials = Vec::new();
        let mut candidates = Vec::new();
        for trial_index in 0..request.trial_budget {
            let Some(params) = proposals.ask(trial_index)? else {
                break;
            };
            if params.is_empty() || params.keys().any(|key| key.trim().is_empty()) {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO proposal parameters must be nonempty".into(),
                ));
            }
            let mut variant = plan.variants[0].clone();
            variant.variant_id = VariantId::new(format!("host_hpo:trial:{trial_index:010}"))?;
            variant.choices.insert(
                "host_hpo".into(),
                GenerationChoice {
                    label: format!("trial:{trial_index}"),
                    value: serde_json::json!({"trial_index": trial_index}),
                    param_overrides: vec![crate::generation::GenerationParamOverride {
                        node_id: request.target_node.clone(),
                        params: params.clone(),
                    }],
                    active_subsequence: None,
                },
            );
            variant.fingerprint = stable_json_fingerprint(&(
                &plan.variants[0].fingerprint,
                &variant.choices,
                request,
            ))?;
            let mut candidate_plan = plan.clone();
            candidate_plan.variants = vec![variant.clone()];
            candidate_plan.validate()?;
            let mut context = RunContext::new(
                RunId::new(format!("run:host_hpo:{trial_index}"))?,
                variant.seed.or(plan.campaign.root_seed),
            );
            context.variant_id = Some(variant.variant_id.clone());
            self.execute_campaign_phase_with_data_provider(
                &candidate_plan,
                controllers,
                provider,
                &mut context,
                Phase::FitCv,
            )?;
            context.collect_cross_fold_validation_scores(plan_oof_partition_mode(plan))?;
            let reports = context
                .score_collector
                .iter()
                .filter(|report| {
                    report.producer_node == request.target_node
                        && report.partition == PredictionPartition::Validation
                        && report.fold_id.as_ref().is_some_and(|fold| {
                            fold.as_str() == "avg"
                                || (folds.folds.len() == 1 && fold == &folds.folds[0].fold_id)
                        })
                })
                .collect::<Vec<_>>();
            let [report] = reports.as_slice() else {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO requires exactly one native target OOF report".into(),
                ));
            };
            let score = *report
                .metrics
                .get(request.metric.name())
                .filter(|score| score.is_finite())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation("host HPO has no finite requested metric".into())
                })?;
            candidates.push(
                (*report)
                    .clone()
                    .into_candidate_score(variant.variant_id.as_str())?,
            );
            let scores = context
                .build_score_set(plan.id.clone(), None)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation("host HPO lost native score evidence".into())
                })?;
            proposals.tell(trial_index, score)?;
            trials.push(HostHpoTrialEvidence {
                trial_index,
                params,
                score,
                variant_id: variant.variant_id,
                scores,
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
        Ok(HostHpoSearchResult {
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
        })
    }
}
