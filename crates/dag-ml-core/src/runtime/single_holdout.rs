//! Explicit single-resampled-holdout bank execution. Legacy CV selection keeps
//! its cross-fold-average requirement; this operation never fabricates an avg.
use super::*;
use crate::metrics::{PopulationRmseRatiosReport, RegressionReportProfile};

pub const SINGLE_HOLDOUT_BANK_PROFILE: &str = "single_resampled_holdout_bank.v1";
pub const PIPELINE_STEPS_V1_PARAM: &str = "pipeline_steps_v1";
/// Classification sibling: native accuracy, maximized, over ClassLabel ids.
pub const SINGLE_HOLDOUT_BANK_V2_PROFILE: &str = "single_resampled_holdout_bank.v2";

/// Shared request of linear and graph classification banks. Accuracy is the
/// closed metric; there is no report profile or host-selected objective.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationBankRequest {
    pub schema_version: u32,
    pub profile: String,
    pub failure_group_dimension: String,
    pub target_node: NodeId,
}

/// An evaluated class outside the fitted classifier domain (its
/// native estimator or `RolePipeline::classes()`, after intermediate roles).
/// Its samples stay evaluated and always score as errors; nothing is dropped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnseenValidationClass {
    pub class_id: i64,
    pub count: usize,
    /// Validation fold order.
    pub sample_ids: Vec<SampleId>,
}

/// The fitted class domain recorded by the target's classifier controller.
fn take_fitted_classes(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    target: &NodeId,
    phase: Phase,
    fold_id: Option<&FoldId>,
) -> Result<Vec<i64>> {
    let node = plan
        .node_plans
        .get(target)
        .ok_or_else(|| refusal("classification bank score target is absent"))?;
    controllers
        .get(&node.controller_id)
        .ok_or_else(|| refusal("classification bank controller is not registered"))?
        .take_fitted_class_domain(target, phase, fold_id)?
        .ok_or_else(|| refusal("classifier controller reported no fitted class domain"))
}

/// Closed bank metric. v1 keeps RMSE and its regression evidence exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankMetric {
    Rmse,
    GraphRmse,
    Accuracy,
    GraphAccuracy,
}
impl BankMetric {
    fn name(self) -> &'static str {
        match self {
            Self::Rmse | Self::GraphRmse => "rmse",
            Self::Accuracy | Self::GraphAccuracy => "accuracy",
        }
    }
    fn objective(self) -> crate::selection::MetricObjective {
        match self {
            Self::Rmse | Self::GraphRmse => crate::selection::MetricObjective::Minimize,
            Self::Accuracy | Self::GraphAccuracy => crate::selection::MetricObjective::Maximize,
        }
    }
    fn is_graph(self) -> bool {
        matches!(self, Self::GraphRmse | Self::GraphAccuracy)
    }
    fn is_classification(self) -> bool {
        matches!(self, Self::Accuracy | Self::GraphAccuracy)
    }
    fn controller_admitted(self, id: &ControllerId, version: &str) -> bool {
        match self {
            Self::Rmse => version == "1.1.0",
            Self::GraphRmse => {
                id.as_str()
                    == crate::n4m_roles::N4mRole::Regressor
                        .controller_id()
                        .expect("regressor id")
                    && version.starts_with("n4m-abi-")
            }
            Self::GraphAccuracy => {
                id.as_str()
                    == crate::n4m_roles::N4mRole::Classifier
                        .controller_id()
                        .expect("classifier id")
                    && version.starts_with("n4m-abi-")
            }
            Self::Accuracy => {
                id.as_str() == crate::PORTABLE_CLASSIFIER_CONTROLLER
                    && version == crate::PORTABLE_CLASSIFIER_VERSION
            }
        }
    }
    fn controller_refusal(self) -> &'static str {
        match self {
            Self::GraphRmse => "graph bank requires the native per-estimator regressor controller",
            Self::Rmse => "holdout bank requires the pipeline_steps_v1 controller version 1.1.0",
            Self::GraphAccuracy => "graph classification bank requires the native per-estimator classifier controller",
            Self::Accuracy => {
                "classification bank requires the pipeline_steps_classifier controller version 1.0.0"
            }
        }
    }
    fn training_profile(self) -> &'static str {
        match self {
            Self::Rmse => TRAINING_BANK_PROFILE,
            Self::GraphRmse => TRAINING_GRAPH_BANK_PROFILE,
            Self::Accuracy => TRAINING_BANK_V2_PROFILE,
            Self::GraphAccuracy => TRAINING_GRAPH_CLASSIFICATION_BANK_PROFILE,
        }
    }
    fn training_policy_id(self) -> &'static str {
        match self {
            Self::Rmse => "select:training_bank.v1:rmse",
            Self::GraphRmse => "select:training_graph_bank.v1:rmse",
            Self::Accuracy => "select:training_bank.v2:accuracy",
            Self::GraphAccuracy => "select:training_graph_classification_bank.v1:accuracy",
        }
    }
}

#[derive(Clone, Copy)]
enum HoldoutScoring {
    Rmse(RegressionReportProfile),
    GraphRmse(RegressionReportProfile),
    Accuracy,
    GraphAccuracy,
}
impl HoldoutScoring {
    fn metric(self) -> BankMetric {
        match self {
            Self::Rmse(_) => BankMetric::Rmse,
            Self::GraphRmse(_) => BankMetric::GraphRmse,
            Self::Accuracy => BankMetric::Accuracy,
            Self::GraphAccuracy => BankMetric::GraphAccuracy,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Rmse(_) => SINGLE_HOLDOUT_BANK_PROFILE,
            Self::GraphRmse(_) => SINGLE_HOLDOUT_GRAPH_BANK_PROFILE,
            Self::Accuracy => SINGLE_HOLDOUT_BANK_V2_PROFILE,
            Self::GraphAccuracy => SINGLE_HOLDOUT_GRAPH_CLASSIFICATION_BANK_PROFILE,
        }
    }
    fn policy_id(self) -> &'static str {
        match self {
            Self::Rmse(_) => "select:single_resampled_holdout_bank.v1:rmse",
            Self::GraphRmse(_) => "select:single_resampled_holdout_graph_bank.v1:rmse",
            Self::Accuracy => "select:single_resampled_holdout_bank.v2:accuracy",
            Self::GraphAccuracy => {
                "select:single_resampled_holdout_graph_classification_bank.v1:accuracy"
            }
        }
    }
}

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
    /// Classification banks only: validation rows outside this candidate's fitted class domain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unseen_validation_classes: Option<Vec<UnseenValidationClass>>,
    /// Graph profile only: join node ID -> ordered branch column blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_blocks: Option<BTreeMap<String, serde_json::Value>>,
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
    /// Classification banks only (absent from regression bytes): validation rows whose class is
    /// outside the selected classifier's fitted domain, ascending by class id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unseen_validation_classes: Option<Vec<UnseenValidationClass>>,
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

/// Classification bank v2 over the same private bank core and scheduler.
/// Selects native accuracy on the actual Resampled validation fold.
#[allow(clippy::too_many_arguments)]
pub fn execute_single_holdout_bank_v2(
    plan: &ExecutionPlan,
    request: &ClassificationBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    data_provider: &dyn RuntimeDataProvider,
) -> Result<SingleHoldoutBankResult> {
    if request.schema_version != 1 || request.profile != SINGLE_HOLDOUT_BANK_V2_PROFILE {
        return Err(refusal("unsupported single holdout bank v2 schema/profile"));
    }
    select_classification_holdout_with_runner(
        plan,
        request,
        run_id,
        root_seed,
        |candidate, ctx| {
            SequentialScheduler
                .execute_campaign_phase_with_data_provider(
                    candidate,
                    controllers,
                    data_provider,
                    ctx,
                    Phase::FitCv,
                )
                .map(|_| ())
        },
        |fold| {
            take_fitted_classes(
                plan,
                controllers,
                &request.target_node,
                Phase::FitCv,
                Some(fold),
            )
        },
    )
}

fn refusal(message: impl Into<String>) -> DagMlError {
    DagMlError::RuntimeValidation(message.into())
}

/// Validate the reserved recipe at the adapter boundary. Methods still owns
/// method/role/parameter capability validation and all numerical operations.
pub(crate) fn effective_pipeline_recipe(
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
    bank_identities_for_scope(plan, request, true)
}

fn bank_identities_for_scope(
    plan: &ExecutionPlan,
    request: &SingleHoldoutBankRequest,
    require_holdout: bool,
) -> Result<Vec<HoldoutVariantIdentity>> {
    bank_identities_core(
        plan,
        &request.failure_group_dimension,
        &request.target_node,
        require_holdout,
        (request.schema_version != 1 || request.profile != SINGLE_HOLDOUT_BANK_PROFILE)
            .then_some("unsupported single holdout bank schema/profile"),
        BankMetric::Rmse,
    )
}

/// Shared admission of every bank profile, in the original v1 check order.
fn bank_identities_core(
    plan: &ExecutionPlan,
    failure_group_dimension: &str,
    target_node: &NodeId,
    require_holdout: bool,
    profile_refusal: Option<&'static str>,
    metric: BankMetric,
) -> Result<Vec<HoldoutVariantIdentity>> {
    if let Some(dimension) = plan
        .campaign
        .generation
        .dimensions
        .iter()
        .find(|dimension| dimension.name == failure_group_dimension)
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
    if let Some(message) = profile_refusal {
        return Err(refusal(message));
    }
    if require_holdout {
        let fold_set = plan
            .fold_set
            .as_ref()
            .ok_or_else(|| refusal("holdout bank requires exactly one Resampled fold"))?;
        if fold_set.partition_mode != FoldPartitionMode::Resampled || fold_set.folds.len() != 1 {
            return Err(refusal("holdout bank requires exactly one Resampled fold"));
        }
    } else if plan.fold_set.is_some() || plan.campaign.split_invocation.is_some() {
        return Err(refusal("training bank forbids folds and split invocations"));
    }
    let generation = &plan.campaign.generation;
    if generation.strategy != GenerationStrategy::Cartesian
        || generation.dimensions.len() != 2
        || generation.dimensions[0].name != failure_group_dimension
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
        .get(target_node)
        .ok_or_else(|| refusal("holdout bank score target is absent"))?;
    if node.kind != NodeKind::Model
        || !metric.controller_admitted(&node.controller_id, &node.controller_version)
    {
        return Err(refusal(metric.controller_refusal()));
    }
    let graph_node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| &node.id == target_node)
        .expect("validated node plan");
    let operator = graph_node
        .operator
        .as_ref()
        .ok_or_else(|| refusal("holdout bank requires a native pipeline operator"))?;
    let graph_profile = metric.is_graph();
    if graph_profile && generation.dimensions[0].choices.len() != 1 {
        return Err(refusal("graph bank requires one fixed chain choice"));
    }
    for (index, dimension) in generation.dimensions.iter().enumerate() {
        for choice in &dimension.choices {
            let [binding] = choice.param_overrides.as_slice() else {
                return Err(refusal(
                    "each bank axis choice must override the target node once",
                ));
            };
            if &binding.node_id != target_node {
                return Err(refusal("bank axes must override only the target node"));
            }
            if index == 0 {
                if graph_profile {
                    if binding.params.len() != 1
                        || binding.params.get(crate::METHODS_ESTIMATOR_METHOD_PARAM)
                            != node.params.get(crate::METHODS_ESTIMATOR_METHOD_PARAM)
                    {
                        return Err(refusal(
                            "graph bank chain must fix only the terminal method_id",
                        ));
                    }
                } else if binding.params.len() != 1
                    || !binding.params.contains_key(PIPELINE_STEPS_V1_PARAM)
                {
                    return Err(refusal("chain axis must override only pipeline_steps_v1"));
                }
            } else if binding.params.is_empty()
                || binding.params.contains_key(PIPELINE_STEPS_V1_PARAM)
                || (graph_profile
                    && binding
                        .params
                        .contains_key(crate::METHODS_ESTIMATOR_METHOD_PARAM))
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
            let params = variant.effective_params_for_node(target_node, &node.params)?;
            let n_components = params
                .get("n_components")
                .and_then(serde_json::Value::as_u64);
            Ok(HoldoutVariantIdentity {
                ordinal: format!("ord:{index:0width$}"),
                variant_id: variant.variant_id.clone(),
                chain_label: variant.choices[failure_group_dimension].label.clone(),
                n_components,
                recipe: if graph_profile {
                    effective_graph_recipe(plan, &params)?
                } else {
                    effective_pipeline_recipe(operator, &params)?
                },
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
    run: F,
) -> Result<SingleHoldoutBankResult>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
{
    let identities = bank_identities(plan, request)?;
    select_holdout_with_identities(
        plan,
        identities,
        &request.target_node,
        HoldoutScoring::Rmse(request.report_profile),
        run_id,
        root_seed,
        run,
        None,
    )
}

/// `fitted` returns the fitted class domain of the candidate that just ran
/// on `fold` (the controller's `RolePipeline::classes()`).
pub(super) fn select_classification_holdout_with_runner<F, D>(
    plan: &ExecutionPlan,
    request: &ClassificationBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    run: F,
    mut fitted: D,
) -> Result<SingleHoldoutBankResult>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
    D: FnMut(&FoldId) -> Result<Vec<i64>>,
{
    let metric = match request.profile.as_str() {
        SINGLE_HOLDOUT_BANK_V2_PROFILE => BankMetric::Accuracy,
        SINGLE_HOLDOUT_GRAPH_CLASSIFICATION_BANK_PROFILE => BankMetric::GraphAccuracy,
        _ => return Err(refusal("unsupported single holdout bank v2 schema/profile")),
    };
    let identities = bank_identities_core(
        plan,
        &request.failure_group_dimension,
        &request.target_node,
        true,
        (request.schema_version != 1)
            .then_some("unsupported single holdout bank v2 schema/profile"),
        metric,
    )?;
    select_holdout_with_identities(
        plan,
        identities,
        &request.target_node,
        if metric.is_graph() {
            HoldoutScoring::GraphAccuracy
        } else {
            HoldoutScoring::Accuracy
        },
        run_id,
        root_seed,
        run,
        Some(&mut fitted),
    )
}

fn select_holdout_with_identities<F>(
    plan: &ExecutionPlan,
    identities: Vec<HoldoutVariantIdentity>,
    target_node: &NodeId,
    scoring: HoldoutScoring,
    run_id: &RunId,
    root_seed: Option<u64>,
    mut run: F,
    mut fitted: Option<&mut dyn FnMut(&FoldId) -> Result<Vec<i64>>>,
) -> Result<SingleHoldoutBankResult>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
{
    let metric = scoring.metric();
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
        let scored = run(&candidate, &mut ctx).and_then(|()| match scoring {
            HoldoutScoring::Rmse(profile) | HoldoutScoring::GraphRmse(profile) => {
                ctx.collect_cross_fold_validation_scores(FoldPartitionMode::Resampled)?;
                super::scoring::single_holdout_validation_evidence(&ctx, target_node, fold, profile)
                    .map(|(prediction, truth, report, selection)| {
                        (prediction, truth, Some(report), selection, None)
                    })
            }
            HoldoutScoring::Accuracy | HoldoutScoring::GraphAccuracy => {
                ctx.collect_cross_fold_validation_scores_for_target(
                    FoldPartitionMode::Resampled,
                    RegressionMetricKind::Accuracy,
                    Some(target_node),
                )?;
                // Taken once per completed candidate, from its own fit.
                let domain = fitted
                    .as_deref_mut()
                    .expect("the classification bank supplies its fitted domain");
                let classes = domain(&fold.fold_id)?;
                super::scoring::single_holdout_classification_evidence(
                    &ctx,
                    target_node,
                    fold,
                    &classes,
                )
                .map(|(prediction, truth, selection, unseen)| {
                    (prediction, truth, None, selection, Some(unseen))
                })
            }
        });
        match scored {
            Ok((prediction, truth, report, selection_report, unseen)) => {
                variants.push(HoldoutVariantEvidence {
                    identity: identity.clone(),
                    validation_predictions: Some(prediction),
                    validation_targets: Some(truth),
                    report,
                    selection_report: Some(selection_report),
                    failure: None,
                    eligible: true,
                    unseen_validation_classes: unseen,
                    feature_blocks: None,
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
                    unseen_validation_classes: None,
                    feature_blocks: None,
                });
            }
        }
    }
    let mut candidates = Vec::new();
    for evidence in &mut variants {
        evidence.eligible &= !issues.contains_key(&evidence.identity.chain_label);
        if evidence.eligible {
            // Only the closed bank metric crosses into CandidateScore. Report-only
            // ratios, bias and the profile-specific R² cannot become objectives.
            let mut report = evidence
                .selection_report
                .clone()
                .expect("completed evidence");
            report.metrics.retain(|name, _| name == metric.name());
            candidates.push(report.into_candidate_score(&evidence.identity.ordinal)?);
        }
    }
    let (selected, decision) = if candidates.is_empty() {
        (None, None)
    } else {
        let policy = SelectionPolicy {
            id: scoring.policy_id().into(),
            metric: SelectionMetric {
                name: metric.name().into(),
                objective: metric.objective(),
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
    // The selected classifier's own fitted-domain report.
    let unseen_validation_classes = decision.as_ref().and_then(|decision| {
        variants
            .iter()
            .find(|evidence| evidence.identity.ordinal == decision.selected_candidate_id)
            .and_then(|evidence| evidence.unseen_validation_classes.clone())
    });
    Ok(SingleHoldoutBankResult {
        schema_version: 1,
        profile: scoring.profile().into(),
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
        unseen_validation_classes,
    })
}

pub const TRAINING_BANK_PROFILE: &str = "training_bank.v1";
/// Classification training sibling: resubstitution accuracy, maximized.
pub const TRAINING_BANK_V2_PROFILE: &str = "training_bank.v2";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingBankRequest {
    pub schema_version: u32,
    pub profile: String,
    pub failure_group_dimension: String,
    pub target_node: NodeId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingVariantEvidence {
    pub identity: HoldoutVariantIdentity,
    pub predictions: PredictionBlock,
    pub targets: RegressionTargetBlock,
    pub report: RegressionMetricReport,
    /// Classification banks only: evaluated rows outside this candidate's fitted class domain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unseen_validation_classes: Option<Vec<UnseenValidationClass>>,
    /// Graph profile only: join node ID -> ordered branch column blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_blocks: Option<BTreeMap<String, serde_json::Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingBankResult {
    pub schema_version: u32,
    pub profile: String,
    pub scope: crate::EvaluationScope,
    pub independent_test: bool,
    pub variants: Vec<TrainingVariantEvidence>,
    pub selected: HoldoutVariantIdentity,
    pub decision: SelectionDecision,
    pub training_sample_ids: Vec<SampleId>,
    pub selected_package: crate::InitialFullRefitPackage,
    /// Classification banks only (absent from regression bytes): evaluated rows outside the
    /// selected classifier's fitted domain, e.g. a class every row of which a
    /// training sample filter removed before the classifier fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unseen_validation_classes: Option<Vec<UnseenValidationClass>>,
}

/// Small no-split sibling: native REFIT and resubstitution, never synthetic OOF.
/// Candidate identity/ordering uses the same admission as bank v1. Any failed
/// candidate refuses this single chain; no successful prefix becomes a winner.
#[allow(clippy::too_many_arguments)]
pub fn execute_training_bank(
    plan: &ExecutionPlan,
    request: &TrainingBankRequest,
    envelope: &ExternalDataPlanEnvelope,
    sample_ids: &[SampleId],
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<TrainingBankResult> {
    if request.schema_version != 1 || request.profile != TRAINING_BANK_PROFILE {
        return Err(refusal("unsupported training bank schema/profile"));
    }
    let bank = SingleHoldoutBankRequest {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_BANK_PROFILE.into(),
        report_profile: RegressionReportProfile::PopulationRmseRatiosV1,
        failure_group_dimension: request.failure_group_dimension.clone(),
        target_node: request.target_node.clone(),
    };
    let identities = bank_identities_for_scope(plan, &bank, false)?;
    training_bank_with_identities(
        plan,
        identities,
        &request.target_node,
        BankMetric::Rmse,
        envelope,
        sample_ids,
        run_id,
        root_seed,
        controllers,
        provider,
    )
}

/// Classification training bank v2: native REFIT and resubstitution accuracy
/// on all rows (Train scope), never synthetic OOF and no extra winner fit.
#[allow(clippy::too_many_arguments)]
pub fn execute_training_bank_v2(
    plan: &ExecutionPlan,
    request: &ClassificationBankRequest,
    envelope: &ExternalDataPlanEnvelope,
    sample_ids: &[SampleId],
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<TrainingBankResult> {
    if request.schema_version != 1 || request.profile != TRAINING_BANK_V2_PROFILE {
        return Err(refusal("unsupported training bank v2 schema/profile"));
    }
    let identities = bank_identities_core(
        plan,
        &request.failure_group_dimension,
        &request.target_node,
        false,
        None,
        BankMetric::Accuracy,
    )?;
    training_bank_with_identities(
        plan,
        identities,
        &request.target_node,
        BankMetric::Accuracy,
        envelope,
        sample_ids,
        run_id,
        root_seed,
        controllers,
        provider,
    )
}

#[allow(clippy::too_many_arguments)]
fn training_bank_with_identities(
    plan: &ExecutionPlan,
    identities: Vec<HoldoutVariantIdentity>,
    target_node: &NodeId,
    metric: BankMetric,
    envelope: &ExternalDataPlanEnvelope,
    sample_ids: &[SampleId],
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<TrainingBankResult> {
    let mut variants = Vec::new();
    let mut scores = Vec::new();
    let mut packages = BTreeMap::new();
    let mut registry = crate::ControllerRegistry::new();
    for manifest in plan.controller_manifests.values() {
        registry.register(manifest.clone())?;
    }
    for (variant, identity) in plan.variants.iter().zip(&identities) {
        let mut graph = plan.graph_plan.graph.clone();
        graph.search_space_fingerprint = None;
        for node in &mut graph.nodes {
            node.params = variant.effective_params_for_node(&node.id, &node.params)?;
        }
        let mut campaign = plan.campaign.clone();
        campaign.generation = GenerationSpec::default();
        let candidate = crate::build_execution_plan(
            format!("{}:{}", plan.id, identity.ordinal),
            graph,
            campaign,
            &registry,
        )?;
        let execution = crate::execute_initial_full_refit(crate::InitialFullRefitExecutionInput {
            package_id: format!("package:{}:{}", run_id, identity.ordinal),
            run_id: run_id.clone(),
            plan: &candidate,
            training_envelope: envelope,
            training_sample_ids: sample_ids,
            controllers,
            data_provider: provider,
            root_seed,
            resource_limits: None,
            scheduler: crate::InitialRefitScheduler::Sequential,
        })?;
        let predictions = execution
            .results
            .iter()
            .flat_map(|r| &r.predictions)
            .filter(|p| {
                &p.producer_node == target_node && p.partition == PredictionPartition::Final
            })
            .collect::<Vec<_>>();
        let [prediction] = predictions.as_slice() else {
            return Err(refusal(
                "training bank requires one final resubstitution block",
            ));
        };
        if prediction.sample_ids.iter().collect::<BTreeSet<_>>()
            != sample_ids.iter().collect::<BTreeSet<_>>()
        {
            return Err(refusal(
                "training bank prediction identities differ from fit population",
            ));
        }
        let targets = execution
            .results
            .iter()
            .filter(|r| &r.node_id == target_node)
            .flat_map(|r| &r.regression_targets)
            .collect::<Vec<_>>();
        let [target] = targets.as_slice() else {
            return Err(refusal(
                "training bank requires one resubstitution target block",
            ));
        };
        let reports = execution
            .scores
            .as_ref()
            .ok_or_else(|| refusal("training bank requires native scores"))?
            .reports
            .iter()
            .filter(|r| {
                &r.producer_node == target_node
                    && r.partition == PredictionPartition::Final
                    && r.level == PredictionLevel::Sample
            })
            .collect::<Vec<_>>();
        let [report] = reports.as_slice() else {
            return Err(refusal(
                "training bank requires one native resubstitution report",
            ));
        };
        let mut report = (*report).clone();
        report.metrics.retain(|key, _| key == metric.name());
        scores.push(report.clone().into_candidate_score(&identity.ordinal)?);
        let unseen_validation_classes = if metric.is_classification() {
            let classes =
                take_fitted_classes(&candidate, controllers, target_node, Phase::Refit, None)?;
            Some(super::scoring::unseen_classes(
                &target_block_aligned_to_samples(sample_ids, target),
                &classes,
            )?)
        } else {
            None
        };
        variants.push(TrainingVariantEvidence {
            identity: identity.clone(),
            predictions: (*prediction).clone(),
            targets: (*target).clone(),
            report,
            unseen_validation_classes,
            feature_blocks: metric
                .is_graph()
                .then(|| feature_blocks(&execution.results)),
        });
        packages.insert(identity.ordinal.clone(), execution.package);
    }
    let decision = select_training_scores_for(&scores, metric)?;
    let selected = identities
        .into_iter()
        .find(|i| i.ordinal == decision.selected_candidate_id)
        .ok_or_else(|| refusal("training selection ordinal missing"))?;
    let selected_package = packages
        .remove(&selected.ordinal)
        .expect("selected candidate package");
    let unseen_validation_classes = variants
        .iter()
        .find(|evidence| evidence.identity.ordinal == selected.ordinal)
        .and_then(|evidence| evidence.unseen_validation_classes.clone());
    Ok(TrainingBankResult {
        schema_version: 1,
        profile: metric.training_profile().into(),
        scope: crate::EvaluationScope::Train,
        independent_test: false,
        variants,
        selected,
        decision,
        training_sample_ids: sample_ids.to_vec(),
        selected_package,
        unseen_validation_classes,
    })
}

#[cfg(test)]
fn select_training_scores(scores: &[CandidateScore]) -> Result<SelectionDecision> {
    select_training_scores_for(scores, BankMetric::Rmse)
}

fn select_training_scores_for(
    scores: &[CandidateScore],
    metric: BankMetric,
) -> Result<SelectionDecision> {
    let policy = SelectionPolicy {
        id: metric.training_policy_id().into(),
        metric: SelectionMetric {
            name: metric.name().into(),
            objective: metric.objective(),
        },
        required_metric_level: Some(PredictionLevel::Sample),
        require_finite: true,
        evaluation_scope: None,
        refit_slot_plan: None,
        requested_rank: None,
        stacking_fit_contract: None,
        reduction_id: None,
    };
    let mut decision = select_candidate(&policy, scores)?;
    decision.evaluation_scope = Some(crate::EvaluationScope::Train);
    Ok(decision)
}

#[cfg(test)]
mod training_contract_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn training_admission_has_no_folds_and_reuses_bank_ordinals() {
        let admission=crate::admit_portable_linear(&json!({"pipeline":[{"model":{"class":"n4m.PLS"},"param":"n_components","_range_":[1,3,1]}]})).unwrap();
        let mut spec = crate::HostControllerSpec::new(
            crate::PORTABLE_STEPS_CONTROLLER,
            crate::PORTABLE_STEPS_VERSION,
            crate::NodeKind::Model,
        );
        spec.output_ports=Some(serde_json::from_value(json!([{"name":"oof","kind":"prediction","representation":null,"cardinality":"one","description":""}])).unwrap());
        let mut registry = crate::ControllerRegistry::new();
        registry.register(spec.derive().unwrap()).unwrap();
        let hash = "0".repeat(64);
        let binding=serde_json::from_value(json!({"node_id":"model:portable","input_name":"x","request_id":"test:opaque",
            "schema_fingerprint":hash,"plan_fingerprint":hash,"relation_fingerprint":hash,"output_representation":"tabular_numeric","source_ids":["x"],"require_relations":true})).unwrap();
        let compiled =
            crate::compile_portable_linear(&admission, binding, None, &registry).unwrap();
        let plan = crate::build_execution_plan(
            "plan:training.test",
            compiled.graph,
            compiled.campaign_template,
            &registry,
        )
        .unwrap();
        assert!(plan.fold_set.is_none());
        assert!(plan.campaign.split_invocation.is_none());
        let request = SingleHoldoutBankRequest {
            schema_version: 1,
            profile: SINGLE_HOLDOUT_BANK_PROFILE.into(),
            report_profile: RegressionReportProfile::PopulationRmseRatiosV1,
            failure_group_dimension: "chain".into(),
            target_node: NodeId::new("model:portable").unwrap(),
        };
        let identities = bank_identities_for_scope(&plan, &request, false).unwrap();
        assert_eq!(identities.len(), 3);
        assert_eq!(identities[0].ordinal, "ord:00000");
        assert_eq!(identities[2].recipe[0]["params"]["n_components"], 3);
        assert!(
            bank_identities(&plan, &request).is_err(),
            "bank v1 must still require its real holdout"
        );
    }
    #[test]
    fn native_training_select_keeps_first_tie_and_training_scope() {
        let candidate = |ordinal: &str| CandidateScore {
            candidate_id: ordinal.into(),
            metrics: BTreeMap::from([("rmse".into(), 1.0)]),
            metadata: BTreeMap::from([("metric_level".into(), json!("sample"))]),
        };
        let decision =
            select_training_scores(&[candidate("ord:00000"), candidate("ord:00001")]).unwrap();
        assert_eq!(decision.selected_candidate_id, "ord:00000");
        assert_eq!(
            decision.evaluation_scope,
            Some(crate::EvaluationScope::Train)
        );
        let mut invalid = candidate("ord:00000");
        invalid.metrics.insert("rmse".into(), f64::INFINITY);
        assert!(select_training_scores(&[invalid]).is_err());
    }
    #[test]
    fn training_v2_admits_only_the_classifier_and_maximizes_accuracy_first_tie() {
        let admission = crate::admit_portable_pipeline(&json!({"pipeline":[{"class":"n4m.SNV"},
            {"model":{"class":"n4m.SparsePLSDA"},"param":"n_components","_range_":[1,3,1]}]}))
        .unwrap();
        assert_eq!(admission.profile, crate::PORTABLE_CLASSIFICATION_PROFILE);
        let mut spec = crate::HostControllerSpec::new(
            crate::PORTABLE_CLASSIFIER_CONTROLLER,
            crate::PORTABLE_CLASSIFIER_VERSION,
            crate::NodeKind::Model,
        );
        spec.output_ports=Some(serde_json::from_value(json!([{"name":"oof","kind":"prediction","representation":null,"cardinality":"one","description":""}])).unwrap());
        let mut registry = crate::ControllerRegistry::new();
        registry.register(spec.derive().unwrap()).unwrap();
        let hash = "0".repeat(64);
        let binding=serde_json::from_value(json!({"node_id":"model:portable","input_name":"x","request_id":"test:opaque",
            "schema_fingerprint":hash,"plan_fingerprint":hash,"relation_fingerprint":hash,"output_representation":"tabular_numeric","source_ids":["x"],"require_relations":true})).unwrap();
        let compiled =
            crate::compile_portable_linear(&admission, binding, None, &registry).unwrap();
        let plan = crate::build_execution_plan(
            "plan:training.classification.test",
            compiled.graph,
            compiled.campaign_template,
            &registry,
        )
        .unwrap();
        let target = NodeId::new("model:portable").unwrap();
        let identities =
            bank_identities_core(&plan, "chain", &target, false, None, BankMetric::Accuracy)
                .unwrap();
        assert_eq!(identities.len(), 3);
        assert_eq!(
            identities[1].recipe[1]["methodId"],
            "models.sparse.sparse_pls_da"
        );
        assert_eq!(identities[1].recipe[1]["params"]["n_components"], 2);
        assert_eq!(identities[1].recipe[1]["params"]["sparsity_lambda"], 0.05);
        let error = bank_identities_core(&plan, "chain", &target, false, None, BankMetric::Rmse)
            .unwrap_err()
            .to_string();
        assert!(error.contains("controller version 1.1.0"), "{error}");
        let score = |ordinal: &str, accuracy: f64| CandidateScore {
            candidate_id: ordinal.into(),
            metrics: BTreeMap::from([("accuracy".into(), accuracy)]),
            metadata: BTreeMap::from([("metric_level".into(), json!("sample"))]),
        };
        let decision = select_training_scores_for(
            &[
                score("ord:00000", 0.5),
                score("ord:00001", 0.9),
                score("ord:00002", 0.9),
            ],
            BankMetric::Accuracy,
        )
        .unwrap();
        assert_eq!(decision.selected_candidate_id, "ord:00001");
        assert_eq!(decision.policy_id, "select:training_bank.v2:accuracy");
        assert_eq!(
            decision.evaluation_scope,
            Some(crate::EvaluationScope::Train)
        );
        let graph = select_training_scores_for(
            &[
                score("ord:00000", 0.5),
                score("ord:00001", 0.9),
                score("ord:00002", 0.9),
            ],
            BankMetric::GraphAccuracy,
        )
        .unwrap();
        assert_eq!(graph.selected_candidate_id, "ord:00001");
        assert_eq!(
            graph.policy_id,
            "select:training_graph_classification_bank.v1:accuracy"
        );
        assert_eq!(graph.objective, crate::selection::MetricObjective::Maximize);
        assert_eq!(graph.evaluation_scope, Some(crate::EvaluationScope::Train));
    }
}

pub const SINGLE_HOLDOUT_GRAPH_BANK_PROFILE: &str = "single_resampled_holdout_graph_bank.v1";
pub const TRAINING_GRAPH_BANK_PROFILE: &str = "training_graph_bank.v1";

fn effective_graph_recipe(
    plan: &ExecutionPlan,
    params: &BTreeMap<String, serde_json::Value>,
) -> Result<Vec<serde_json::Value>> {
    let mut recipe: Vec<serde_json::Value> = serde_json::from_value(
        plan.graph_plan
            .graph
            .metadata
            .get(crate::PORTABLE_GRAPH_RECIPE_PARAM)
            .cloned()
            .ok_or_else(|| refusal("graph bank requires canonical portable graph recipe"))?,
    )?;
    let terminal = recipe
        .last_mut()
        .ok_or_else(|| refusal("graph recipe requires a terminal model"))?;
    if terminal["methodId"] != params[crate::METHODS_ESTIMATOR_METHOD_PARAM] {
        return Err(refusal("graph recipe terminal disagrees with native model"));
    }
    terminal["params"] = serde_json::to_value(
        params
            .iter()
            .filter(|(key, _)| key.as_str() != crate::METHODS_ESTIMATOR_METHOD_PARAM)
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    )?;
    Ok(recipe)
}

fn feature_blocks(results: &[NodeResult]) -> BTreeMap<String, serde_json::Value> {
    results
        .iter()
        .filter(|result| {
            result.lineage.controller_id.as_str() == crate::METHODS_FEATURE_JOIN_CONTROLLER
        })
        .filter_map(|result| {
            result
                .shape_deltas
                .iter()
                .find_map(|delta| delta.metadata.get("feature_blocks"))
                .map(|blocks| (result.node_id.to_string(), blocks.clone()))
        })
        .collect()
}

/// Graph admission differs only in target/axes; the bank loop, scoring, chain
/// failure grouping, native ordinals and first-tie SELECT are shared.
#[allow(clippy::too_many_arguments)]
pub fn execute_single_holdout_graph_bank(
    plan: &ExecutionPlan,
    request: &SingleHoldoutBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<SingleHoldoutBankResult> {
    let mut blocks = BTreeMap::new();
    let mut bank = select_single_holdout_graph_bank_with_runner(
        plan,
        request,
        run_id,
        root_seed,
        |candidate, ctx| {
            let results = SequentialScheduler.execute_campaign_phase_with_data_provider(
                candidate,
                controllers,
                provider,
                ctx,
                Phase::FitCv,
            )?;
            let variant = ctx
                .variant_id
                .clone()
                .ok_or_else(|| refusal("graph bank requires a candidate identity"))?;
            blocks.insert(variant, feature_blocks(&results));
            Ok(())
        },
    )?;
    for entry in &mut bank.variants {
        entry.feature_blocks = blocks.remove(&entry.identity.variant_id);
    }
    Ok(bank)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_training_graph_bank(
    plan: &ExecutionPlan,
    request: &TrainingBankRequest,
    envelope: &ExternalDataPlanEnvelope,
    sample_ids: &[SampleId],
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<TrainingBankResult> {
    let identities = bank_identities_core(
        plan,
        &request.failure_group_dimension,
        &request.target_node,
        false,
        (request.schema_version != 1 || request.profile != TRAINING_GRAPH_BANK_PROFILE)
            .then_some("unsupported training graph bank schema/profile"),
        BankMetric::GraphRmse,
    )?;
    training_bank_with_identities(
        plan,
        identities,
        &request.target_node,
        BankMetric::GraphRmse,
        envelope,
        sample_ids,
        run_id,
        root_seed,
        controllers,
        provider,
    )
}

pub(super) fn select_single_holdout_graph_bank_with_runner<F>(
    plan: &ExecutionPlan,
    request: &SingleHoldoutBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    run: F,
) -> Result<SingleHoldoutBankResult>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
{
    let identities = bank_identities_core(
        plan,
        &request.failure_group_dimension,
        &request.target_node,
        true,
        (request.schema_version != 1 || request.profile != SINGLE_HOLDOUT_GRAPH_BANK_PROFILE)
            .then_some("unsupported single holdout graph bank schema/profile"),
        BankMetric::GraphRmse,
    )?;
    select_holdout_with_identities(
        plan,
        identities,
        &request.target_node,
        HoldoutScoring::GraphRmse(request.report_profile),
        run_id,
        root_seed,
        run,
        None,
    )
}

pub const SINGLE_HOLDOUT_GRAPH_CLASSIFICATION_BANK_PROFILE: &str =
    "single_resampled_holdout_graph_classification_bank.v1";
pub const TRAINING_GRAPH_CLASSIFICATION_BANK_PROFILE: &str =
    "training_graph_classification_bank.v1";

/// Graph accuracy uses the same classification runner/domain closure and
/// selection loop as linear classification; only target/axes/evidence differ.
#[allow(clippy::too_many_arguments)]
pub fn execute_single_holdout_graph_classification_bank(
    plan: &ExecutionPlan,
    request: &ClassificationBankRequest,
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<SingleHoldoutBankResult> {
    if request.schema_version != 1
        || request.profile != SINGLE_HOLDOUT_GRAPH_CLASSIFICATION_BANK_PROFILE
    {
        return Err(refusal(
            "unsupported single holdout graph classification bank schema/profile",
        ));
    }
    let mut blocks = BTreeMap::new();
    let mut bank = select_classification_holdout_with_runner(
        plan,
        request,
        run_id,
        root_seed,
        |candidate, ctx| {
            let results = SequentialScheduler.execute_campaign_phase_with_data_provider(
                candidate,
                controllers,
                provider,
                ctx,
                Phase::FitCv,
            )?;
            let variant = ctx
                .variant_id
                .clone()
                .ok_or_else(|| refusal("graph bank requires a candidate identity"))?;
            blocks.insert(variant, feature_blocks(&results));
            Ok(())
        },
        |fold| {
            take_fitted_classes(
                plan,
                controllers,
                &request.target_node,
                Phase::FitCv,
                Some(fold),
            )
        },
    )?;
    for entry in &mut bank.variants {
        entry.feature_blocks = blocks.remove(&entry.identity.variant_id);
    }
    Ok(bank)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_training_graph_classification_bank(
    plan: &ExecutionPlan,
    request: &ClassificationBankRequest,
    envelope: &ExternalDataPlanEnvelope,
    sample_ids: &[SampleId],
    run_id: &RunId,
    root_seed: Option<u64>,
    controllers: &RuntimeControllerRegistry,
    provider: &dyn RuntimeDataProvider,
) -> Result<TrainingBankResult> {
    let identities = bank_identities_core(
        plan,
        &request.failure_group_dimension,
        &request.target_node,
        false,
        (request.schema_version != 1
            || request.profile != TRAINING_GRAPH_CLASSIFICATION_BANK_PROFILE)
            .then_some("unsupported training graph classification bank schema/profile"),
        BankMetric::GraphAccuracy,
    )?;
    training_bank_with_identities(
        plan,
        identities,
        &request.target_node,
        BankMetric::GraphAccuracy,
        envelope,
        sample_ids,
        run_id,
        root_seed,
        controllers,
        provider,
    )
}
