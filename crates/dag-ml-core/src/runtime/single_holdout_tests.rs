//! Bounded selector tests; production-path WASM/Methods bank parity belongs to Root.
use super::*;
use crate::metrics::RegressionReportProfile;
use crate::runtime::single_holdout::{
    select_classification_holdout_with_runner, select_single_holdout_bank_with_runner,
};

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

// ---- Classification bank v2: accuracy, maximized, over ClassLabel ids ----

const V2_CLASSIFIER: &str = "controller:methods.native.pipeline_steps_classifier";

fn classification_bank() -> (ExecutionPlan, ClassificationBankRequest) {
    let (base, v1) = bank();
    let mut manifest = controller_manifest(V2_CLASSIFIER, NodeKind::Model);
    manifest.controller_version = "1.0.0".into();
    let mut registry = crate::controller::ControllerRegistry::new();
    registry.register(manifest).unwrap();
    let plan = build_execution_plan(
        "plan:classification.bank",
        base.graph_plan.graph.clone(),
        base.campaign.clone(),
        &registry,
    )
    .unwrap();
    let request = ClassificationBankRequest {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_BANK_V2_PROFILE.into(),
        failure_group_dimension: v1.failure_group_dimension,
        target_node: v1.target_node,
    };
    (plan, request)
}

/// One validation (s1) and one training (s2) row: predicted and observed ids.
fn emit_labels(
    candidate: &ExecutionPlan,
    ctx: &mut RunContext,
    predicted: f64,
    validation_truth: f64,
    training_truth: f64,
) -> Result<()> {
    let fold = &candidate.fold_set.as_ref().unwrap().folds[0];
    let node = candidate.graph_plan.graph.nodes[0].id.clone();
    let targets = |ids: &[SampleId], value: f64| RegressionTargetBlock {
        level: PredictionLevel::Sample,
        unit_ids: ids.iter().cloned().map(PredictionUnitId::Sample).collect(),
        values: vec![vec![value]; ids.len()],
        validity_masks: None,
        target_names: vec!["y".into()],
    };
    let block = PredictionBlock {
        prediction_id: None,
        producer_node: node.clone(),
        producer_port: None,
        partition: PredictionPartition::Validation,
        fold_id: Some(fold.fold_id.clone()),
        sample_ids: fold.validation_sample_ids.clone(),
        values: vec![vec![predicted]; fold.validation_sample_ids.len()],
        target_names: vec!["y".into()],
    };
    let truth = targets(&fold.validation_sample_ids, validation_truth);
    let mut report =
        score_regression_prediction_block(&block, &truth, &[RegressionMetricKind::Accuracy])?;
    report.variant_id = ctx.variant_id.clone();
    ctx.score_collector.push(report);
    for (partition, targets) in [
        (PredictionPartition::Validation, truth),
        (
            PredictionPartition::Train,
            targets(&fold.train_sample_ids, training_truth),
        ),
    ] {
        ctx.regression_target_records.push(RegressionTargetRecord {
            producer_node: node.clone(),
            producer_port: None,
            variant_id: ctx.variant_id.clone(),
            partition,
            fold_id: block.fold_id.clone(),
            block: targets,
        });
    }
    ctx.prediction_store.append(block)
}

#[test]
fn v2_maximizes_native_accuracy_and_keeps_the_first_ordinal_tie() {
    let (plan, request) = classification_bank();
    let mut calls = 0;
    let result = select_classification_holdout_with_runner(
        &plan,
        &request,
        &RunId::new("run:classification.bank").unwrap(),
        Some(7),
        |candidate, ctx| {
            let ordinal = calls;
            calls += 1;
            // Ordinals 1 and 2 are both correct; the first of them must win.
            let predicted = if matches!(ordinal, 1 | 2) { 0.0 } else { 1.0 };
            emit_labels(candidate, ctx, predicted, 0.0, 0.0)
        },
        |_| Ok(vec![0]),
    )
    .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(result.profile, SINGLE_HOLDOUT_BANK_V2_PROFILE);
    assert_eq!(result.status, HoldoutBankStatus::Selected);
    let decision = result.decision.as_ref().unwrap();
    assert_eq!(decision.selected_candidate_id, "ord:00001");
    assert_eq!(
        decision.policy_id,
        "select:single_resampled_holdout_bank.v2:accuracy"
    );
    assert_eq!(decision.metric_name, "accuracy");
    assert_eq!(
        decision.objective,
        crate::selection::MetricObjective::Maximize
    );
    assert_eq!(decision.selected_score, 1.0);
    for (evidence, expected) in result.variants.iter().zip([0.0, 1.0, 1.0, 0.0]) {
        let selection = evidence.selection_report.as_ref().unwrap();
        assert_eq!(
            selection.metrics.keys().collect::<Vec<_>>(),
            vec!["accuracy"]
        );
        assert_eq!(selection.metrics["accuracy"], expected);
        assert!(evidence.report.is_none(), "no regression report in v2");
    }
    assert_eq!(result.unseen_validation_classes, Some(Vec::new()));
    let winner = result.selected.as_ref().unwrap();
    assert_eq!(winner.identity.ordinal, "ord:00001");
    assert_eq!(
        winner.training_sample_ids,
        plan.fold_set.as_ref().unwrap().folds[0].train_sample_ids
    );
}

#[test]
fn v2_reports_and_scores_a_validation_class_absent_from_the_fit_population() {
    let (plan, request) = classification_bank();
    // Class 2 exists only in validation; the fitted classifier learned class 0.
    let result = select_classification_holdout_with_runner(
        &plan,
        &request,
        &RunId::new("run:classification.unseen").unwrap(),
        None,
        |candidate, ctx| emit_labels(candidate, ctx, 0.0, 2.0, 0.0),
        |_| Ok(vec![0]),
    )
    .unwrap();
    assert_eq!(result.status, HoldoutBankStatus::Selected);
    assert!(result.variants.iter().all(|evidence| {
        evidence.eligible
            && evidence.selection_report.as_ref().unwrap().metrics["accuracy"] == 0.0
            && evidence.validation_targets.as_ref().unwrap().values == vec![vec![2.0]]
    }));
    let unseen = result.unseen_validation_classes.as_ref().unwrap();
    assert_eq!(
        unseen,
        &vec![UnseenValidationClass {
            class_id: 2,
            count: 1,
            sample_ids: vec![SampleId::new("s1").unwrap()],
        }]
    );
    assert_eq!(result.decision.unwrap().selected_candidate_id, "ord:00000");
    assert!(result
        .variants
        .iter()
        .all(|evidence| evidence.unseen_validation_classes.as_ref() == Some(unseen)));
}

#[test]
fn v2_unseen_report_uses_the_fitted_domain_not_raw_fit_labels() {
    let (plan, request) = classification_bank();
    // Raw fit rows carry class 2, but a training sample filter removed them
    // before the classifier fit: the learned domain is only class 0.
    let mut domains = Vec::new();
    let result = select_classification_holdout_with_runner(
        &plan,
        &request,
        &RunId::new("run:classification.filtered").unwrap(),
        None,
        |candidate, ctx| emit_labels(candidate, ctx, 0.0, 2.0, 2.0),
        |fold| {
            domains.push(fold.clone());
            Ok(vec![0])
        },
    )
    .unwrap();
    // Taken once per completed candidate, for its actual fold.
    assert_eq!(domains.len(), 4);
    assert!(domains
        .iter()
        .all(|fold| fold == &plan.fold_set.as_ref().unwrap().folds[0].fold_id));
    assert_eq!(
        result.unseen_validation_classes,
        Some(vec![UnseenValidationClass {
            class_id: 2,
            count: 1,
            sample_ids: vec![SampleId::new("s1").unwrap()],
        }])
    );
    // A domain that does contain the validation class reports nothing, while
    // the row still scores as an error when mispredicted.
    let result = select_classification_holdout_with_runner(
        &plan,
        &request,
        &RunId::new("run:classification.seen").unwrap(),
        None,
        |candidate, ctx| emit_labels(candidate, ctx, 0.0, 2.0, 0.0),
        |_| Ok(vec![0, 2]),
    )
    .unwrap();
    assert_eq!(result.unseen_validation_classes, Some(Vec::new()));
    assert!(result.variants.iter().all(|evidence| evidence
        .selection_report
        .as_ref()
        .unwrap()
        .metrics["accuracy"]
        == 0.0));
}

#[test]
fn v2_profile_and_classifier_identity_are_closed_and_v1_bytes_are_unchanged() {
    let (plan, mut request) = classification_bank();
    request.profile = SINGLE_HOLDOUT_BANK_PROFILE.into();
    let error = select_classification_holdout_with_runner(
        &plan,
        &request,
        &RunId::new("run:v2.profile").unwrap(),
        None,
        |_, _| panic!("must refuse before fitting"),
        |_| panic!("must refuse before fitting"),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("unsupported single holdout bank v2 schema/profile"));
    // The regression 1.1.0 plan is not a classification bank.
    let (regression, v1) = bank();
    let v2 = ClassificationBankRequest {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_BANK_V2_PROFILE.into(),
        failure_group_dimension: v1.failure_group_dimension.clone(),
        target_node: v1.target_node.clone(),
    };
    let error = select_classification_holdout_with_runner(
        &regression,
        &v2,
        &RunId::new("run:v2.controller").unwrap(),
        None,
        |_, _| panic!("must refuse before fitting"),
        |_| panic!("must refuse before fitting"),
    )
    .unwrap_err();
    assert!(error.to_string().contains(
        "classification bank requires the pipeline_steps_classifier controller version 1.0.0"
    ));
    // And bank v1 still refuses the classifier controller version.
    let error = select_single_holdout_bank_with_runner(
        &plan,
        &v1,
        &RunId::new("run:v1.on.classifier").unwrap(),
        None,
        |_, _| panic!("must refuse before fitting"),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("holdout bank requires the pipeline_steps_v1 controller version 1.1.0"));
    // v1 evidence keeps its RMSE selection and serializes without the v2 field.
    let result = select_single_holdout_bank_with_runner(
        &regression,
        &v1,
        &RunId::new("run:v1.bytes").unwrap(),
        Some(7),
        |candidate, ctx| emit(candidate, ctx, 1.0),
    )
    .unwrap();
    assert_eq!(result.profile, SINGLE_HOLDOUT_BANK_PROFILE);
    assert!(result.unseen_validation_classes.is_none());
    let encoded = serde_json::to_value(&result).unwrap();
    assert!(encoded.get("unseen_validation_classes").is_none());
    assert!(encoded["variants"]
        .as_array()
        .unwrap()
        .iter()
        .all(|evidence| evidence.get("unseen_validation_classes").is_none()));
    assert_eq!(
        encoded["decision"]["policy_id"],
        "select:single_resampled_holdout_bank.v1:rmse"
    );
}

#[test]
fn feature_graph_compile_and_bank_keep_declared_ports_native_ordinals_and_first_tie() {
    let (legacy, _) = bank();
    let mut registry = crate::ControllerRegistry::new();
    for (id, kind) in [
        ("controller:n4m.transformer", NodeKind::Transform),
        ("controller:n4m.regressor", NodeKind::Model),
        (
            crate::METHODS_FEATURE_JOIN_CONTROLLER,
            NodeKind::FeatureJoin,
        ),
    ] {
        let mut spec = crate::HostControllerSpec::new(
            id,
            if kind == NodeKind::FeatureJoin {
                "1.0.0"
            } else {
                "n4m-abi-2.14.0"
            },
            kind.clone(),
        );
        if kind == NodeKind::FeatureJoin {
            spec.output_ports =
                Some(crate::manifest_kind_template(&NodeKind::Transform).output_ports);
        }
        registry.register(spec.derive().unwrap()).unwrap();
    }
    let admitted = crate::admit_portable_pipeline(&json!([
        {"class":"n4m.KennardStone"},
        {"branch":[{"name":"z_first","steps":[{"class":"n4m.SNV"}]},
            {"name":"a_second","steps":[{"class":"n4m.MSC"}]}]}, {"merge":"features"},
        {"model":{"class":"n4m.PLS"},"param":"n_components","_range_":[1,2,1]}]))
    .unwrap();
    let binding = serde_json::from_value(json!({"node_id":"model:portable","input_name":"x","request_id":"io:test",
        "schema_fingerprint":"a".repeat(64),"plan_fingerprint":"b".repeat(64),"relation_fingerprint":"c".repeat(64),
        "output_representation":"tabular_numeric","feature_set_id":"x","source_ids":["nir"],"require_relations":false})).unwrap();
    let compiled =
        crate::compile_portable_linear(&admitted, binding, legacy.fold_set.clone(), &registry)
            .unwrap();
    assert_eq!(
        compiled
            .graph
            .nodes
            .iter()
            .map(|node| node.kind.clone())
            .collect::<Vec<_>>(),
        vec![
            NodeKind::Transform,
            NodeKind::Transform,
            NodeKind::FeatureJoin,
            NodeKind::Model
        ]
    );
    let join = &compiled.graph.nodes[2];
    assert_eq!(
        join.params["branch_inputs_v1"],
        json!([
        {"branch_id":"z_first","input":"z_first_x"},{"branch_id":"a_second","input":"a_second_x"}])
    );
    assert_eq!(
        compiled.data_bindings.len(),
        2,
        "each branch root reads the same provider population"
    );
    let plan = build_execution_plan(
        "plan:graph.bank",
        compiled.graph,
        compiled.campaign_template,
        &registry,
    )
    .unwrap();
    let target = NodeId::new("model:portable").unwrap();
    let request = SingleHoldoutBankRequest {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_GRAPH_BANK_PROFILE.into(),
        report_profile: RegressionReportProfile::PopulationRmseRatiosV1,
        failure_group_dimension: "chain".into(),
        target_node: target.clone(),
    };
    let mut calls = 0;
    let result = crate::runtime::single_holdout::select_single_holdout_graph_bank_with_runner(
        &plan,
        &request,
        &RunId::new("run:graph.tie").unwrap(),
        Some(7),
        |candidate, ctx| {
            calls += 1;
            // Reuse the existing score emitter while retaining the real graph.
            let mut scored = candidate.clone();
            scored.graph_plan.graph.nodes.swap(0, 3);
            emit(&scored, ctx, 1.0)
        },
    )
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(result.profile, SINGLE_HOLDOUT_GRAPH_BANK_PROFILE);
    assert_eq!(
        result.selected.as_ref().unwrap().identity.ordinal,
        "ord:00000"
    );
    assert_eq!(
        result.decision.as_ref().unwrap().selected_candidate_id,
        "ord:00000"
    );
    for (index, entry) in result.variants.iter().enumerate() {
        assert_eq!(entry.identity.ordinal, format!("ord:{index:05}"));
        assert_eq!(entry.identity.variant_id, plan.variants[index].variant_id);
        assert_eq!(entry.identity.recipe[0], admitted.recipe[0]);
        assert_eq!(
            entry.identity.recipe.last().unwrap()["params"]["n_components"],
            json!(index + 1)
        );
        assert!(entry.identity.recipe[0].get("feature_blocks").is_none());
        assert_eq!(
            entry.selection_report.as_ref().unwrap().metrics["rmse"],
            1.0
        );
    }
    assert!(
        crate::runtime::single_holdout::select_single_holdout_bank_with_runner(
            &plan,
            &SingleHoldoutBankRequest {
                profile: SINGLE_HOLDOUT_BANK_PROFILE.into(),
                ..request
            },
            &RunId::new("run:graph.not.v1").unwrap(),
            None,
            |_, _| panic!("wrong profile must refuse before execution")
        )
        .is_err()
    );
}

#[test]
fn graph_accuracy_uses_fitted_domain_first_tie_and_canonical_graph_identity() {
    let (legacy, _) = bank();
    let mut registry = crate::ControllerRegistry::new();
    for (id, kind) in [
        ("controller:n4m.transformer", NodeKind::Transform),
        ("controller:n4m.classifier", NodeKind::Model),
        (
            crate::METHODS_FEATURE_JOIN_CONTROLLER,
            NodeKind::FeatureJoin,
        ),
    ] {
        let mut spec = crate::HostControllerSpec::new(
            id,
            if kind == NodeKind::FeatureJoin {
                "1.0.0"
            } else {
                "n4m-abi-2.14.0"
            },
            kind.clone(),
        );
        if kind == NodeKind::FeatureJoin {
            spec.output_ports =
                Some(crate::manifest_kind_template(&NodeKind::Transform).output_ports);
        }
        registry.register(spec.derive().unwrap()).unwrap();
    }
    let admission = crate::admit_portable_pipeline(&json!([
        {"class":"n4m.KennardStone"},
        {"branch":[{"name":"z_msc","steps":[{"class":"n4m.MSC"}]},
            {"name":"a_snv","steps":[{"class":"n4m.SNV"}]}]}, {"merge":"features"},
        {"model":{"class":"n4m.SparsePLSDA"},"param":"n_components","_range_":[1,2,1]}
    ]))
    .unwrap();
    let binding = serde_json::from_value(json!({"node_id":"model:portable","input_name":"x","request_id":"io:classgraph",
        "schema_fingerprint":"a".repeat(64),"plan_fingerprint":"b".repeat(64),"relation_fingerprint":"c".repeat(64),
        "output_representation":"tabular_numeric","feature_set_id":"x","source_ids":["nir"],"require_relations":false})).unwrap();
    let compiled =
        crate::compile_portable_linear(&admission, binding, legacy.fold_set.clone(), &registry)
            .unwrap();
    let plan = build_execution_plan(
        "plan:graph.classification",
        compiled.graph,
        compiled.campaign_template,
        &registry,
    )
    .unwrap();
    let request = ClassificationBankRequest {
        schema_version: 1,
        profile: SINGLE_HOLDOUT_GRAPH_CLASSIFICATION_BANK_PROFILE.into(),
        failure_group_dimension: "chain".into(),
        target_node: NodeId::new("model:portable").unwrap(),
    };
    for (validation_truth, expected_accuracy, expected_unseen) in
        [(0.0, 1.0, vec![]), (2.0, 0.0, vec![2])]
    {
        let mut calls = 0;
        let mut domains = 0;
        let result = select_classification_holdout_with_runner(
            &plan,
            &request,
            &RunId::new("run:graph.classification").unwrap(),
            Some(7),
            |candidate, ctx| {
                calls += 1;
                let mut scored = candidate.clone();
                scored.graph_plan.graph.nodes.swap(0, 3);
                emit_labels(&scored, ctx, 0.0, validation_truth, 2.0)
            },
            |fold| {
                domains += 1;
                assert_eq!(fold, &plan.fold_set.as_ref().unwrap().folds[0].fold_id);
                // The fitted domain, not raw training labels (2), is authoritative.
                Ok(vec![0])
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(domains, calls);
        assert_eq!(
            result.profile,
            SINGLE_HOLDOUT_GRAPH_CLASSIFICATION_BANK_PROFILE
        );
        let decision = result.decision.as_ref().unwrap();
        assert_eq!(decision.selected_candidate_id, "ord:00000");
        assert_eq!(
            decision.policy_id,
            "select:single_resampled_holdout_graph_classification_bank.v1:accuracy"
        );
        assert_eq!(
            decision.objective,
            crate::selection::MetricObjective::Maximize
        );
        assert_eq!(decision.selected_score, expected_accuracy);
        assert_eq!(
            result
                .unseen_validation_classes
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| c.class_id)
                .collect::<Vec<_>>(),
            expected_unseen
        );
        for (i, evidence) in result.variants.iter().enumerate() {
            assert_eq!(evidence.identity.variant_id, plan.variants[i].variant_id);
            assert_eq!(evidence.identity.recipe[0], admission.recipe[0]);
            assert_eq!(
                evidence.selection_report.as_ref().unwrap().metrics["accuracy"],
                expected_accuracy
            );
            assert!(evidence.report.is_none());
            assert_eq!(
                evidence.validation_predictions.as_ref().unwrap().sample_ids,
                plan.fold_set.as_ref().unwrap().folds[0].validation_sample_ids
            );
        }
    }
    let mut wrong = request;
    wrong.profile = SINGLE_HOLDOUT_BANK_V2_PROFILE.into();
    assert!(select_classification_holdout_with_runner(
        &plan,
        &wrong,
        &RunId::new("run:graph.wrong.profile").unwrap(),
        None,
        |_, _| panic!("profile conflict must refuse before fitting"),
        |_| panic!("profile conflict must refuse before fitting")
    )
    .is_err());
}
