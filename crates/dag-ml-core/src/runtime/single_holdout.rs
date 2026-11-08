//! Explicit single-resampled-holdout bank execution. Legacy CV selection keeps
//! its cross-fold-average requirement; this operation never fabricates an avg.
use super::*;
use crate::metrics::{PopulationRmseRatiosReport, RegressionReportProfile};

pub const SINGLE_HOLDOUT_BANK_PROFILE: &str = "single_resampled_holdout_bank.v1";
pub const PIPELINE_STEPS_V1_PARAM: &str = "pipeline_steps_v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleHoldoutBankRequest {
    pub schema_version: u32,
    pub profile: String,
    pub report_profile: RegressionReportProfile,
    pub failure_group_dimension: String,
    pub target_node: NodeId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoldoutVariantIdentity {
    pub ordinal: String,
    pub variant_id: VariantId,
    pub chain_label: String,
    /// Effective parameters for the requested target, including the recipe key.
    pub params: BTreeMap<String, serde_json::Value>,
    /// Optional PLS convenience metadata; never required by bank selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_components: Option<u64>,
    /// Effective Methods steps, including the scalar model parameters.
    pub recipe: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoldoutVariantEvidence {
    pub identity: HoldoutVariantIdentity,
    pub validation_predictions: Option<PredictionBlock>,
    pub validation_targets: Option<RegressionTargetBlock>,
    pub report: Option<PopulationRmseRatiosReport>,
    /// Native RMSE report on the actual validation fold, never `avg`.
    pub selection_report: Option<RegressionMetricReport>,
    pub failure: Option<String>,
    /// False for every member of a failed chain, including completed variants.
    pub eligible: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoldoutChainIssue {
    pub chain_label: String,
    pub failures: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoldoutBankWinner {
    pub identity: HoldoutVariantIdentity,
    /// Concrete graph with every variant parameter frozen by the owner. Build
    /// a no-splitter plan against the train-only envelope before initial REFIT.
    pub selected_graph: crate::graph::GraphSpec,
    /// The selected closure's complete effective parameter map by native node ID.
    pub params_by_node: BTreeMap<NodeId, BTreeMap<String, serde_json::Value>>,
    pub training_sample_ids: Vec<SampleId>,
    pub validation_sample_ids: Vec<SampleId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoldoutBankStatus {
    Selected,
    Refused,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SingleHoldoutBankResult {
    pub schema_version: u32,
    pub profile: String,
    pub status: HoldoutBankStatus,
    pub ordinal_variant_mapping: Vec<HoldoutVariantIdentity>,
    pub variants: Vec<HoldoutVariantEvidence>,
    pub chain_issues: Vec<HoldoutChainIssue>,
    pub selected: Option<HoldoutBankWinner>,
    pub decision: Option<SelectionDecision>,
    pub refusal: Option<String>,
}

/// One owner operation owns the entire bank loop and native SELECT. Controllers
/// only execute individual NodeTasks through the existing scheduler.
#[allow(clippy::too_many_arguments)]
pub fn execute_single_holdout_bank(
    plan: &ExecutionPlan,
    request: &SingleHoldoutBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    data_provider: &dyn RuntimeDataProvider,
) -> Result<SingleHoldoutBankResult> {
    select_single_holdout_bank_with_runner(plan, request, run_id, root_seed, |candidate, ctx| {
        SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                candidate,
                controllers,
                data_provider,
                ctx,
                Phase::FitCv,
            )
            .map(|_| ())
    })
}

fn refusal(message: impl Into<String>) -> DagMlError {
    DagMlError::RuntimeValidation(message.into())
}

/// Validate the reserved recipe at the adapter boundary. Methods still owns
/// method/role/parameter capability validation and all numerical operations.
fn effective_pipeline_recipe(
    operator: &serde_json::Value,
    params: &BTreeMap<String, serde_json::Value>,
) -> Result<Vec<serde_json::Value>> {
    if operator.get("type").and_then(serde_json::Value::as_str) != Some("N4mRolePipeline") {
        return Err(refusal("holdout bank target requires N4mRolePipeline"));
    }
    let steps = params
        .get(PIPELINE_STEPS_V1_PARAM)
        .and_then(serde_json::Value::as_array)
        .filter(|steps| !steps.is_empty())
        .ok_or_else(|| refusal("holdout bank requires a nonempty pipeline_steps_v1 recipe"))?;
    if let Some(fixed) = operator.get("steps") {
        if fixed != &serde_json::Value::Array(steps.clone()) {
            return Err(refusal(
                "pipeline_steps_v1 conflicts with fixed operator steps",
            ));
        }
    }
    let mut effective = steps.clone();
    for step in &effective {
        let object = step
            .as_object()
            .ok_or_else(|| refusal("recipe step must be an object"))?;
        let class = object.get("class").and_then(serde_json::Value::as_str);
        let method = object.get("methodId").and_then(serde_json::Value::as_str);
        if !matches!((class, method), (Some(id), None) | (None, Some(id)) if !id.is_empty())
            || object
                .keys()
                .any(|key| !matches!(key.as_str(), "class" | "methodId" | "params"))
            || object.get("params").is_some_and(|value| !value.is_object())
        {
            return Err(refusal(
                "recipe steps use one nonempty class/methodId and optional params object",
            ));
        }
        if object
            .get("params")
            .and_then(|value| value.get(PIPELINE_STEPS_V1_PARAM))
            .is_some()
        {
            return Err(refusal("pipeline_steps_v1 is reserved for node parameters"));
        }
    }
    let last = effective
        .last_mut()
        .expect("validated nonempty recipe")
        .as_object_mut()
        .expect("validated step");
    let model_params = last
        .entry("params")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .expect("validated params");
    model_params.extend(
        params
            .iter()
            .filter(|(key, _)| key.as_str() != PIPELINE_STEPS_V1_PARAM)
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    Ok(effective)
}

fn bank_identities(
    plan: &ExecutionPlan,
    request: &SingleHoldoutBankRequest,
) -> Result<Vec<HoldoutVariantIdentity>> {
    if let Some(dimension) = plan
        .campaign
        .generation
        .dimensions
        .iter()
        .find(|dimension| dimension.name == request.failure_group_dimension)
    {
        let mut labels = BTreeSet::new();
        for choice in &dimension.choices {
            if !labels.insert(choice.label.as_str()) {
                return Err(refusal(format!(
                    "holdout bank failure-group dimension `{}` has duplicate chain label `{}`",
                    dimension.name, choice.label,
                )));
            }
        }
    }
    plan.validate()?;
    if request.schema_version != 1 || request.profile != SINGLE_HOLDOUT_BANK_PROFILE {
        return Err(refusal("unsupported single holdout bank schema/profile"));
    }
    let fold_set = plan
        .fold_set
        .as_ref()
        .ok_or_else(|| refusal("holdout bank requires exactly one Resampled fold"))?;
    if fold_set.partition_mode != FoldPartitionMode::Resampled || fold_set.folds.len() != 1 {
        return Err(refusal("holdout bank requires exactly one Resampled fold"));
    }
    let generation = &plan.campaign.generation;
    if generation.strategy != GenerationStrategy::Cartesian
        || generation.dimensions.len() != 2
        || generation.dimensions[0].name != request.failure_group_dimension
        || !generation.constraints.is_empty()
    {
        return Err(refusal("holdout bank requires Cartesian chain-first, model-parameters-second dimensions without pruning"));
    }
    // Re-enumerate at the public boundary, never trust plan vector order or IDs
    // as an ordering convention. Ordinary plan validation also binds identities.
    let variants = enumerate_variants(generation, plan.campaign.root_seed)?;
    if plan.variants != variants {
        return Err(refusal(
            "holdout bank variants must match native enumeration order",
        ));
    }
    let node = plan
        .node_plans
        .get(&request.target_node)
        .ok_or_else(|| refusal("holdout bank score target is absent"))?;
    if node.kind != NodeKind::Model || node.controller_version != "1.1.0" {
        return Err(refusal(
            "holdout bank requires the pipeline_steps_v1 controller version 1.1.0",
        ));
    }
    let graph_node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == request.target_node)
        .expect("validated node plan");
    let operator = graph_node
        .operator
        .as_ref()
        .ok_or_else(|| refusal("holdout bank requires a native pipeline operator"))?;
    for (index, dimension) in generation.dimensions.iter().enumerate() {
        for choice in &dimension.choices {
            let [binding] = choice.param_overrides.as_slice() else {
                return Err(refusal(
                    "each bank axis choice must override the target node once",
                ));
            };
            if binding.node_id != request.target_node {
                return Err(refusal("bank axes must override only the target node"));
            }
            if index == 0 {
                if binding.params.len() != 1
                    || !binding.params.contains_key(PIPELINE_STEPS_V1_PARAM)
                {
                    return Err(refusal("chain axis must override only pipeline_steps_v1"));
                }
            } else if binding.params.is_empty()
                || binding.params.contains_key(PIPELINE_STEPS_V1_PARAM)
            {
                return Err(refusal("model parameter axis requires nonempty target parameters excluding pipeline_steps_v1"));
            }
        }
    }
    let width = 5usize.max(variants.len().saturating_sub(1).to_string().len());
    variants
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            let params = variant.effective_params_for_node(&request.target_node, &node.params)?;
            let n_components = params
                .get("n_components")
                .and_then(serde_json::Value::as_u64);
            Ok(HoldoutVariantIdentity {
                ordinal: format!("ord:{index:0width$}"),
                variant_id: variant.variant_id.clone(),
                chain_label: variant.choices[&request.failure_group_dimension]
                    .label
                    .clone(),
                n_components,
                recipe: effective_pipeline_recipe(operator, &params)?,
                params,
            })
        })
        .collect()
}

pub(super) fn select_single_holdout_bank_with_runner<F>(
    plan: &ExecutionPlan,
    request: &SingleHoldoutBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    mut run: F,
) -> Result<SingleHoldoutBankResult>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
{
    let identities = bank_identities(plan, request)?;
    let fold = &plan.fold_set.as_ref().expect("validated fold set").folds[0];
    let mut variants = Vec::with_capacity(identities.len());
    let mut issues = BTreeMap::<String, BTreeMap<String, String>>::new();
    for (variant, identity) in plan.variants.iter().zip(&identities) {
        let candidate = ExecutionPlan {
            variants: vec![variant.clone()],
            ..plan.clone()
        };
        let mut ctx = RunContext::new(run_id.clone(), root_seed);
        ctx.variant_id = Some(variant.variant_id.clone());
        // Discard the entire transient context after any failure. It never
        // participates in a different candidate or downstream training.
        let scored = run(&candidate, &mut ctx).and_then(|()| {
            ctx.collect_cross_fold_validation_scores(FoldPartitionMode::Resampled)?;
            super::scoring::single_holdout_validation_evidence(
                &ctx,
                &request.target_node,
                fold,
                request.report_profile,
            )
        });
        match scored {
            Ok((prediction, truth, report, selection_report)) => {
                variants.push(HoldoutVariantEvidence {
                    identity: identity.clone(),
                    validation_predictions: Some(prediction),
                    validation_targets: Some(truth),
                    report: Some(report),
                    selection_report: Some(selection_report),
                    failure: None,
                    eligible: true,
                })
            }
            Err(error) => {
                let message = error.to_string();
                issues
                    .entry(identity.chain_label.clone())
                    .or_default()
                    .insert(identity.ordinal.clone(), message.clone());
                variants.push(HoldoutVariantEvidence {
                    identity: identity.clone(),
                    validation_predictions: None,
                    validation_targets: None,
                    report: None,
                    selection_report: None,
                    failure: Some(message),
                    eligible: false,
                });
            }
        }
    }
    let mut candidates = Vec::new();
    for evidence in &mut variants {
        evidence.eligible &= !issues.contains_key(&evidence.identity.chain_label);
        if evidence.eligible {
            // Only RMSE crosses into CandidateScore. Report-only ratios, bias
            // and the profile-specific R² cannot become selectable objectives.
            let mut report = evidence
                .selection_report
                .clone()
                .expect("completed evidence");
            report.metrics.retain(|name, _| name == "rmse");
            candidates.push(report.into_candidate_score(&evidence.identity.ordinal)?);
        }
    }
    let (selected, decision) = if candidates.is_empty() {
        (None, None)
    } else {
        let policy = SelectionPolicy {
            id: "select:single_resampled_holdout_bank.v1:rmse".into(),
            metric: SelectionMetric {
                name: "rmse".into(),
                objective: crate::selection::MetricObjective::Minimize,
            },
            required_metric_level: Some(PredictionLevel::Sample),
            require_finite: true,
            evaluation_scope: None,
            refit_slot_plan: None,
            requested_rank: None,
            stacking_fit_contract: None,
            reduction_id: None,
        };
        let decision = select_candidate(&policy, &candidates)?;
        let winner = identities
            .iter()
            .find(|identity| identity.ordinal == decision.selected_candidate_id)
            .expect("selected native ordinal");
        let variant = plan
            .variants
            .iter()
            .find(|variant| variant.variant_id == winner.variant_id)
            .expect("selected native variant");
        let mut selected_graph = plan.graph_plan.graph.clone();
        selected_graph.search_space_fingerprint = None;
        for node in &mut selected_graph.nodes {
            node.params = variant.effective_params_for_node(&node.id, &node.params)?;
        }
        let params_by_node = selected_graph
            .nodes
            .iter()
            .map(|node| (node.id.clone(), node.params.clone()))
            .collect();
        (
            Some(HoldoutBankWinner {
                identity: winner.clone(),
                selected_graph,
                params_by_node,
                training_sample_ids: fold.train_sample_ids.clone(),
                validation_sample_ids: fold.validation_sample_ids.clone(),
            }),
            Some(decision),
        )
    };
    let status = if selected.is_some() {
        HoldoutBankStatus::Selected
    } else {
        HoldoutBankStatus::Refused
    };
    Ok(SingleHoldoutBankResult {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_BANK_PROFILE.into(),
        status,
        ordinal_variant_mapping: identities,
        variants,
        chain_issues: plan.campaign.generation.dimensions[0]
            .choices
            .iter()
            .filter_map(|choice| {
                issues
                    .remove(&choice.label)
                    .map(|failures| HoldoutChainIssue {
                        chain_label: choice.label.clone(),
                        failures,
                    })
            })
            .collect(),
        selected,
        decision,
        refusal: (status == HoldoutBankStatus::Refused)
            .then(|| "all chains failed; no eligible holdout winner".into()),
    })
}
