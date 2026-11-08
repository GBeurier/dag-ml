//! Bounded selector tests; production-path WASM/Methods bank parity belongs to Root.
use super::*;
use crate::metrics::RegressionReportProfile;
use crate::runtime::single_holdout::select_single_holdout_bank_with_runner;

fn bank() -> (ExecutionPlan, SingleHoldoutBankRequest) {
    let mut graph = simple_graph();
    graph.nodes.retain(|node| node.kind == NodeKind::Model);
    graph.edges.clear();
    graph.nodes[0].ports.inputs.clear();
    graph.nodes[0].operator = Some(json!({"type":"N4mRolePipeline"}));
    let target = graph.nodes[0].id.clone();
    let mut manifest = controller_manifest("controller:model", NodeKind::Model);
    manifest.controller_version = "1.1.0".into();
    let mut registry = crate::controller::ControllerRegistry::new();
    registry.register(manifest).unwrap();
    let mut campaign = single_fold_variant_scoring_campaign(vec![("unused", 1.0)]);
    let dimension =
        |name: &str, key: &str, choices: Vec<(&str, serde_json::Value)>| GenerationDimension {
            name: name.into(),
            choices: choices
                .into_iter()
                .map(|(label, value)| GenerationChoice {
                    label: label.into(),
                    value: json!(label),
                    active_subsequence: None,
                    param_overrides: vec![crate::generation::GenerationParamOverride {
                        node_id: target.clone(),
                        params: BTreeMap::from([(key.into(), value)]),
                    }],
                })
                .collect(),
        };
    campaign.generation.dimensions = vec![
        dimension(
            "chain",
            PIPELINE_STEPS_V1_PARAM,
            vec![
                ("z:first", json!([{"class":"n4m:models.pls","params":{}}])),
                ("a:second", json!([{"methodId":"models.pls","params":{}}])),
            ],
        ),
        dimension(
            "opaque component labels",
            "n_components",
            vec![("z:first", json!(2)), ("a:second", json!(1))],
        ),
    ];
    campaign.generation.max_variants = Some(4);
    let plan = build_execution_plan("plan:holdout.bank", graph, campaign, &registry).unwrap();
    let request = SingleHoldoutBankRequest {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_BANK_PROFILE.into(),
        report_profile: RegressionReportProfile::PopulationRmseRatiosV1,
        failure_group_dimension: "chain".into(),
        target_node: target,
    };
    (plan, request)
}

fn emit(candidate: &ExecutionPlan, ctx: &mut RunContext, value: f64) -> Result<()> {
    assert!(
        ctx.prediction_store.blocks().is_empty(),
        "each variant needs an isolated context"
    );
    assert!(ctx.score_collector.is_empty());
    let fold = &candidate.fold_set.as_ref().unwrap().folds[0];
    let node = candidate.graph_plan.graph.nodes[0].id.clone();
    let block = PredictionBlock {
        prediction_id: None,
        producer_node: node.clone(),
        producer_port: None,
        partition: PredictionPartition::Validation,
        fold_id: Some(fold.fold_id.clone()),
        sample_ids: fold.validation_sample_ids.clone(),
        values: vec![vec![value]; fold.validation_sample_ids.len()],
        target_names: vec!["y".into()],
    };
    let truth = RegressionTargetBlock {
        level: PredictionLevel::Sample,
        unit_ids: block
            .sample_ids
            .iter()
            .cloned()
            .map(PredictionUnitId::Sample)
            .collect(),
        values: vec![vec![0.0]; block.sample_ids.len()],
        validity_masks: None,
        target_names: vec!["y".into()],
    };
    let mut report =
        score_regression_prediction_block(&block, &truth, &[RegressionMetricKind::Rmse])?;
    report.variant_id = ctx.variant_id.clone();
    ctx.score_collector.push(report);
    ctx.regression_target_records.push(RegressionTargetRecord {
        producer_node: node,
        producer_port: None,
        variant_id: ctx.variant_id.clone(),
        partition: PredictionPartition::Validation,
        fold_id: block.fold_id.clone(),
        block: truth,
    });
    ctx.prediction_store.append(block)
}

#[test]
fn ordinal_enumeration_ties_and_selected_closure_are_exact() {
    let (plan, request) = bank();
    let mut calls = 0;
    let result = select_single_holdout_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:bank").unwrap(),
        Some(7),
        |candidate, ctx| {
            calls += 1;
            emit(candidate, ctx, 1.0)
        },
    )
    .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(result.status, HoldoutBankStatus::Selected);
    let native = enumerate_variants(&plan.campaign.generation, plan.campaign.root_seed).unwrap();
    for (index, identity) in result.ordinal_variant_mapping.iter().enumerate() {
        assert_eq!(identity.ordinal, format!("ord:{index:05}"));
        assert_eq!(identity.variant_id, native[index].variant_id);
    }
    let winner = result.selected.as_ref().unwrap();
    assert_eq!(winner.identity.ordinal, "ord:00000");
    assert_eq!(winner.identity.chain_label, "z:first");
    assert_eq!(winner.identity.n_components, Some(2));
    assert_eq!(
        winner.training_sample_ids,
        plan.fold_set.as_ref().unwrap().folds[0].train_sample_ids
    );
    assert_eq!(
        winner.validation_sample_ids,
        plan.fold_set.as_ref().unwrap().folds[0].validation_sample_ids
    );
    assert_eq!(
        winner.selected_graph.nodes[0].params["n_components"],
        json!(2)
    );
    assert!(winner.selected_graph.nodes[0]
        .params
        .contains_key(PIPELINE_STEPS_V1_PARAM));
    assert_eq!(
        winner.identity.recipe[0]["params"]["n_components"],
        json!(2)
    );
    assert!(winner.identity.recipe[0]["params"]
        .get(PIPELINE_STEPS_V1_PARAM)
        .is_none());
    assert_eq!(result.decision.unwrap().selected_candidate_id, "ord:00000");
    assert!(result
        .variants
        .iter()
        .all(|item| item.eligible && item.report.as_ref().unwrap().metrics["rmse"] == 1.0));
}

#[test]
fn a_failed_component_excludes_the_whole_chain_and_never_mixes_predictions() {
    let (plan, request) = bank();
    let mut index = 0;
    let result = select_single_holdout_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:chain.fail").unwrap(),
        Some(7),
        |candidate, ctx| {
            let current = index;
            index += 1;
            if current == 1 {
                return Err(DagMlError::RuntimeValidation(
                    "component fit refused".into(),
                ));
            }
            emit(
                candidate,
                ctx,
                if current == 0 { 0.0 } else { current as f64 },
            )
        },
    )
    .unwrap();
    assert_eq!(result.selected.unwrap().identity.ordinal, "ord:00002");
    assert_eq!(result.chain_issues.len(), 1);
    assert_eq!(result.chain_issues[0].chain_label, "z:first");
    assert_eq!(result.chain_issues[0].failures.len(), 1);
    assert!(!result.variants[0].eligible);
    assert!(!result.variants[1].eligible);
    assert!(result.variants[1].validation_predictions.is_none());
    assert_eq!(
        result.variants[2]
            .validation_predictions
            .as_ref()
            .unwrap()
            .values,
        vec![vec![2.0]]
    );
    assert_eq!(
        result.variants[3]
            .validation_predictions
            .as_ref()
            .unwrap()
            .values,
        vec![vec![3.0]]
    );
}

#[test]
fn all_failed_or_missing_evidence_refuses_without_a_winner() {
    let (plan, request) = bank();
    // A callback that emits nothing must not salvage missing scoring/OOF evidence.
    let result = select_single_holdout_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:missing").unwrap(),
        None,
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(result.status, HoldoutBankStatus::Refused);
    assert!(result.selected.is_none() && result.decision.is_none() && result.refusal.is_some());
    assert_eq!(result.chain_issues.len(), 2);
    assert_eq!(result.chain_issues[0].chain_label, "z:first");
    assert_eq!(result.chain_issues[1].chain_label, "a:second");
    assert!(result
        .variants
        .iter()
        .all(|item| !item.eligible && item.failure.is_some()));
    let result = select_single_holdout_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:all.fail").unwrap(),
        None,
        |_, _| Err(DagMlError::RuntimeValidation("operator refused".into())),
    )
    .unwrap();
    assert_eq!(result.status, HoldoutBankStatus::Refused);
    assert!(result.selected.is_none());
}

#[test]
fn wrong_fold_shape_and_reordered_bank_refuse_before_callbacks() {
    let (plan, request) = bank();
    let mut campaign = plan.campaign.clone();
    let fold_set = campaign
        .split_invocation
        .as_mut()
        .unwrap()
        .fold_set
        .as_mut()
        .unwrap();
    let mut extra = fold_set.folds[0].clone();
    extra.fold_id = FoldId::new("fold:extra").unwrap();
    fold_set.folds.push(extra);
    let wrong = build_execution_plan(
        "plan:wrong.holdout",
        plan.graph_plan.graph.clone(),
        campaign,
        &manifests_for(&plan),
    )
    .unwrap();
    assert!(select_single_holdout_bank_with_runner(
        &wrong,
        &request,
        &RunId::new("run:shape").unwrap(),
        None,
        |_, _| panic!("must refuse before fitting")
    )
    .unwrap_err()
    .to_string()
    .contains("exactly one Resampled fold"));
    let mut reordered = plan.clone();
    reordered.variants.reverse();
    assert!(select_single_holdout_bank_with_runner(
        &reordered,
        &request,
        &RunId::new("run:order").unwrap(),
        None,
        |_, _| panic!("must refuse before fitting")
    )
    .is_err());
    // Preserve the existing default-profile single-fold-average refusal test.
}

fn manifests_for(plan: &ExecutionPlan) -> crate::controller::ControllerRegistry {
    let mut registry = crate::controller::ControllerRegistry::new();
    for manifest in plan.controller_manifests.values() {
        registry.register(manifest.clone()).unwrap();
    }
    registry
}

#[test]
fn zero_rmse_winner_keeps_undefined_reports_and_generic_params() {
    let (base, request) = bank();
    let mut campaign = base.campaign.clone();
    for choice in &mut campaign.generation.dimensions[0].choices {
        choice.param_overrides[0].params.insert(
            PIPELINE_STEPS_V1_PARAM.into(),
            json!([{"class":"n4m:models.regularized.ridge","params":{}}]),
        );
    }
    for (choice, alpha) in campaign.generation.dimensions[1]
        .choices
        .iter_mut()
        .zip([0.1, 1.0])
    {
        choice.param_overrides[0].params = BTreeMap::from([("alpha".into(), json!(alpha))]);
    }
    let plan = build_execution_plan(
        "plan:holdout.generic.zero",
        base.graph_plan.graph.clone(),
        campaign,
        &manifests_for(&base),
    )
    .unwrap();
    let mut calls = 0;
    let result = select_single_holdout_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:holdout.generic.zero").unwrap(),
        Some(7),
        |candidate, ctx| {
            let ordinal = calls;
            calls += 1;
            // The last candidate is perfect on the constant validation target.
            emit(candidate, ctx, if ordinal == 3 { 0.0 } else { 1.0 })
        },
    )
    .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(result.status, HoldoutBankStatus::Selected);
    assert!(result.chain_issues.is_empty() && result.refusal.is_none());
    assert!(result
        .variants
        .iter()
        .all(|evidence| evidence.eligible && evidence.failure.is_none()));
    let winner = result.selected.as_ref().unwrap();
    assert_eq!(winner.identity.ordinal, "ord:00003");
    assert_eq!(winner.identity.params["alpha"], json!(1.0));
    assert!(winner.identity.n_components.is_none());
    assert!(!winner.identity.params.contains_key("n_components"));
    assert!(result
        .ordinal_variant_mapping
        .iter()
        .all(|identity| identity.n_components.is_none()));
    assert_eq!(result.ordinal_variant_mapping[3], winner.identity);
    assert_eq!(
        winner.params_by_node[&request.target_node],
        winner.identity.params
    );
    assert_eq!(
        winner.selected_graph.nodes[0].params,
        winner.identity.params
    );
    assert_eq!(winner.identity.recipe[0]["params"]["alpha"], json!(1.0));
    let decision = result.decision.as_ref().unwrap();
    assert_eq!(decision.selected_candidate_id, "ord:00003");
    assert_eq!(decision.selected_score.to_bits(), 0.0_f64.to_bits());
    let perfect = &result.variants[3];
    let report = perfect.report.as_ref().unwrap();
    report.validate().unwrap();
    assert_eq!(report.metrics["rmse"].to_bits(), 0.0_f64.to_bits());
    assert_eq!(report.metrics["bias"], 0.0);
    assert_eq!(
        report.undefined_metrics,
        BTreeMap::from([
            ("r2".into(), "constant_target".into()),
            ("rpd".into(), "zero_rmse".into()),
            ("rpiq".into(), "zero_rmse".into()),
        ])
    );
    assert!(["r2", "rpd", "rpiq"]
        .iter()
        .all(|name| !report.metrics.contains_key(*name)));
    assert_eq!(
        perfect.selection_report.as_ref().unwrap().metrics["rmse"],
        0.0
    );
    assert!(perfect.validation_predictions.is_some() && perfect.validation_targets.is_some());
    let encoded = serde_json::to_value(winner).unwrap();
    assert!(encoded["identity"].get("n_components").is_none());
    assert_eq!(
        encoded["params_by_node"][request.target_node.as_str()]["alpha"],
        json!(1.0)
    );
}

#[test]
fn duplicate_failure_group_labels_refuse_before_callbacks() {
    let (mut plan, request) = bank();
    // Mutate at the runtime boundary: rebuilding would already refuse the
    // duplicate through generic generation validation. The bank must name the
    // ambiguous failure-group label before plan validation or controller work.
    let choices = &mut plan.campaign.generation.dimensions[0].choices;
    choices[1].label = choices[0].label.clone();
    let error = select_single_holdout_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:duplicate.chain.labels").unwrap(),
        Some(7),
        |_, _| panic!("duplicate failure-group labels must refuse before controller callbacks"),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        DagMlError::RuntimeValidation(
            "holdout bank failure-group dimension `chain` has duplicate chain label `z:first`"
                .into(),
        )
        .to_string()
    );
}
