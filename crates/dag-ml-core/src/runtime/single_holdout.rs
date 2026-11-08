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

pub const TRAINING_BANK_PROFILE: &str = "training_bank.v1";

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
                p.producer_node == request.target_node && p.partition == PredictionPartition::Final
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
            .filter(|r| r.node_id == request.target_node)
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
                r.producer_node == request.target_node
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
        report.metrics.retain(|key, _| key == "rmse");
        scores.push(report.clone().into_candidate_score(&identity.ordinal)?);
        variants.push(TrainingVariantEvidence {
            identity: identity.clone(),
            predictions: (*prediction).clone(),
            targets: (*target).clone(),
            report,
        });
        packages.insert(identity.ordinal.clone(), execution.package);
    }
    let decision = select_training_scores(&scores)?;
    let selected = identities
        .into_iter()
        .find(|i| i.ordinal == decision.selected_candidate_id)
        .ok_or_else(|| refusal("training selection ordinal missing"))?;
    let selected_package = packages
        .remove(&selected.ordinal)
        .expect("selected candidate package");
    Ok(TrainingBankResult {
        schema_version: 1,
        profile: TRAINING_BANK_PROFILE.into(),
        scope: crate::EvaluationScope::Train,
        independent_test: false,
        variants,
        selected,
        decision,
        training_sample_ids: sample_ids.to_vec(),
        selected_package,
    })
}

fn select_training_scores(scores: &[CandidateScore]) -> Result<SelectionDecision> {
    let policy = SelectionPolicy {
        id: "select:training_bank.v1:rmse".into(),
        metric: SelectionMetric {
            name: "rmse".into(),
            objective: crate::MetricObjective::Minimize,
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
}
