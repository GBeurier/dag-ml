// Auto-split from the former monolithic `runtime.rs` (pure refactor).
use super::*;

pub(crate) const SCORE_METRICS: &[RegressionMetricKind] = &[
    RegressionMetricKind::Mse,
    RegressionMetricKind::Rmse,
    RegressionMetricKind::Mae,
    RegressionMetricKind::R2,
    RegressionMetricKind::Accuracy,
    RegressionMetricKind::BalancedAccuracy,
];

/// True when a Sample-level target block covers EXACTLY the prediction block's samples — the pairing
/// dag-ml's scoring requires (target units == prediction units). Lets one result carry several
/// sample-level blocks (e.g. refit's final-train + final-test), each with its own y_true.
pub(crate) fn sample_targets_match_block(
    block: &PredictionBlock,
    targets: &RegressionTargetBlock,
) -> bool {
    if targets.level != PredictionLevel::Sample || targets.unit_ids.len() != block.sample_ids.len()
    {
        return false;
    }
    let predicted: BTreeSet<&SampleId> = block.sample_ids.iter().collect();
    targets.unit_ids.iter().all(|unit| match unit {
        PredictionUnitId::Sample(sample_id) => predicted.contains(sample_id),
        _ => false,
    })
}

/// Score a result's prediction blocks against the host-supplied `regression_targets` and push the
/// reports into the collector. Native scoring is gated purely on the host emitting targets: a run
/// that emits no `regression_targets` (every existing run) collects nothing, so behavior is
/// unchanged and the campaign fingerprint is untouched. Each Sample prediction block is paired with
/// the target block covering exactly its samples; unmatched blocks are unscored.
pub(crate) fn apply_result_scoring(
    result: &NodeResult,
    collector: &mut Vec<RegressionMetricReport>,
    target_records: &mut Vec<RegressionTargetRecord>,
) -> Result<()> {
    if result.regression_targets.is_empty() {
        return Ok(());
    }
    for block in &result.predictions {
        if let Some(targets) = result
            .regression_targets
            .iter()
            .find(|targets| sample_targets_match_block(block, targets))
        {
            let mut report = score_regression_prediction_block(block, targets, SCORE_METRICS)?;
            report.variant_id = result.lineage.variant_id.clone();
            collector.push(report);
            // Retain y_true (tagged with its variant/fold/partition) so the OOF average can be
            // scored later, per-variant.
            target_records.push(RegressionTargetRecord {
                producer_node: block.producer_node.clone(),
                producer_port: block.producer_port.clone(),
                variant_id: result.lineage.variant_id.clone(),
                partition: block.partition.clone(),
                fold_id: block.fold_id.clone(),
                block: targets.clone(),
            });
        }
    }
    for block in &result.aggregated_predictions {
        if let Some(targets) = result
            .regression_targets
            .iter()
            .find(|targets| targets.level == block.level)
        {
            let mut report = score_regression_aggregated_block(block, targets, SCORE_METRICS)?;
            report.variant_id = result.lineage.variant_id.clone();
            collector.push(report);
        }
    }
    Ok(())
}

pub(crate) fn apply_result_prediction_aggregation(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    task: &NodeTask,
    result: &mut NodeResult,
    resources: &PhaseScopeResources<'_>,
) -> Result<()> {
    let has_observation_predictions = !result.observation_predictions.is_empty();
    let has_sample_predictions = !result.predictions.is_empty();
    if !has_observation_predictions && !has_sample_predictions {
        return Ok(());
    }
    let Some(shape_plan) = &task.node_plan.shape_plan else {
        if !has_observation_predictions {
            return Ok(());
        }
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` emitted observation predictions but has no data/model shape plan for aggregation",
            task.node_plan.node_id
        )));
    };
    let policy = &shape_plan.aggregation_policy;
    if !policy.store_aggregated_predictions {
        return Ok(());
    }
    if policy.aggregation_level == PredictionLevel::Observation {
        return Ok(());
    }
    if !has_observation_predictions && policy.aggregation_level == PredictionLevel::Sample {
        return Ok(());
    }

    let mut derived_sample_blocks = Vec::new();
    if !result.observation_predictions.is_empty() {
        let relations = coordinator_relations_for_task(task, resources)?;
        let sample_policy = observation_to_sample_policy(policy);
        for block in result.observation_predictions.clone() {
            let requested_sample_order =
                requested_sample_order_for_observation_block(plan, task, &block, &relations)?;
            let sample_block =
                if sample_policy.method == crate::policy::AggregationMethod::CustomController {
                    dispatch_custom_observation_aggregation(
                        plan,
                        controllers,
                        aggregation_task_id(
                            task,
                            &block.producer_node,
                            block.fold_id.as_ref(),
                            "obs_to_sample",
                        ),
                        block,
                        relations.clone(),
                        sample_policy.clone(),
                        requested_sample_order,
                    )?
                } else {
                    aggregate_observation_predictions(
                        &block,
                        &relations,
                        &sample_policy,
                        &requested_sample_order,
                    )?
                };
            derived_sample_blocks.push(sample_block);
        }
    }

    if policy.aggregation_level == PredictionLevel::Sample {
        result.predictions.extend(derived_sample_blocks);
        result.validate_for_task(task)?;
        return Ok(());
    }

    if !result.aggregated_predictions.is_empty() {
        // The controller emitted aggregated blocks itself, bypassing native
        // aggregation. They must still MATCH the node's aggregation policy
        // level — otherwise a block aggregated at the wrong unit level would be
        // accepted and scored against a mismatched policy.
        for block in &result.aggregated_predictions {
            if block.level != policy.aggregation_level {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` emitted aggregated predictions at level {:?} but its aggregation policy is {:?}",
                    task.node_plan.node_id, block.level, policy.aggregation_level
                )));
            }
        }
        result.validate_for_task(task)?;
        return Ok(());
    }

    let relations = coordinator_relations_for_task(task, resources)?;
    let sample_blocks = result
        .predictions
        .iter()
        .cloned()
        .chain(derived_sample_blocks)
        .collect::<Vec<_>>();
    for block in sample_blocks {
        let requested_unit_order =
            requested_unit_order_for_sample_block(policy.aggregation_level, &relations, &block)?;
        let aggregated = if policy.method == crate::policy::AggregationMethod::CustomController {
            dispatch_custom_sample_aggregation(
                plan,
                controllers,
                aggregation_task_id(
                    task,
                    &block.producer_node,
                    block.fold_id.as_ref(),
                    "sample_to_unit",
                ),
                block,
                relations.clone(),
                policy.clone(),
                requested_unit_order,
            )?
        } else {
            aggregate_sample_predictions_by_unit(&block, &relations, policy, &requested_unit_order)?
        };
        result.aggregated_predictions.push(aggregated);
    }
    result.validate_for_task(task)
}

pub(crate) fn observation_to_sample_policy(policy: &AggregationPolicy) -> AggregationPolicy {
    let mut sample_policy = policy.clone();
    sample_policy.aggregation_level = PredictionLevel::Sample;
    sample_policy
}

pub(crate) fn coordinator_relations_for_task(
    task: &NodeTask,
    resources: &PhaseScopeResources<'_>,
) -> Result<SampleRelationSet> {
    coordinator_relations_for_node(&task.node_plan, resources)?.ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "node `{}` needs coordinator relations for prediction aggregation but no matching data provider/envelope carries relations",
            task.node_plan.node_id
        ))
    })
}

pub(crate) fn coordinator_relations_for_edge(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    resources: &PhaseScopeResources<'_>,
) -> Result<SampleRelationSet> {
    let target_plan = plan.node_plans.get(&edge.target.node_id).ok_or_else(|| {
        DagMlError::Planning(format!(
            "OOF edge target node `{}` has no node plan",
            edge.target.node_id
        ))
    })?;
    if let Some(relations) = coordinator_relations_for_node(target_plan, resources)? {
        return Ok(relations);
    }

    let source_plan = plan.node_plans.get(&edge.source.node_id).ok_or_else(|| {
        DagMlError::Planning(format!(
            "OOF edge source node `{}` has no node plan",
            edge.source.node_id
        ))
    })?;
    if let Some(relations) = coordinator_relations_for_node(source_plan, resources)? {
        return Ok(relations);
    }

    Err(DagMlError::RuntimeValidation(format!(
        "edge `{}.{}` -> `{}.{}` needs coordinator relations for aggregated OOF validation but neither endpoint has a relation-carrying data binding",
        edge.source.node_id,
        edge.source.port_name,
        edge.target.node_id,
        edge.target.port_name
    )))
}

pub(crate) fn coordinator_relations_for_node(
    node_plan: &NodePlan,
    resources: &PhaseScopeResources<'_>,
) -> Result<Option<SampleRelationSet>> {
    let mut selected: Option<SampleRelationSet> = None;
    for binding in &node_plan.data_bindings {
        if !binding.require_relations && binding.relation_fingerprint.is_none() {
            continue;
        }
        let relations = if let Some(envelopes) = resources.data_envelopes {
            let key = data_binding_requirement_key(&binding.node_id, &binding.input_name);
            match envelopes.get(&key) {
                Some(envelope) => {
                    binding.validate_envelope(envelope)?;
                    envelope.coordinator_relations.clone()
                }
                None => None,
            }
        } else if let Some(data_provider) = resources.data_provider {
            data_provider.coordinator_relations(binding)?
        } else {
            None
        };
        let Some(relations) = relations else {
            // A binding that REQUIRES relations must resolve them. Silently
            // defaulting to empty exclusions (no excluded samples) would let a
            // leakage / branch / exclusion / aggregation policy run without the
            // relation set it depends on, so refuse instead of degrading.
            if binding.require_relations {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` binding `{}` requires coordinator relations but none were resolved",
                    node_plan.node_id, binding.input_name
                )));
            }
            continue;
        };
        if let Some(previous) = &selected {
            if previous != &relations {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` has multiple non-identical coordinator relation sets",
                    node_plan.node_id
                )));
            }
        } else {
            selected = Some(relations);
        }
    }
    Ok(selected)
}

pub(crate) fn requested_sample_order_for_observation_block(
    plan: &ExecutionPlan,
    task: &NodeTask,
    block: &ObservationPredictionBlock,
    relations: &SampleRelationSet,
) -> Result<Vec<SampleId>> {
    if block.partition == PredictionPartition::Validation {
        if let Some(sample_ids) = validation_view_sample_ids(task) {
            return Ok(sample_ids.into_iter().collect());
        }
        if let (Some(fold_set), Some(fold_id)) = (plan.fold_set.as_ref(), block.fold_id.as_ref()) {
            if let Some(fold) = fold_set.folds.iter().find(|fold| &fold.fold_id == fold_id) {
                return Ok(fold.validation_sample_ids.clone());
            }
        }
    }
    first_seen_samples_for_observations(block, relations)
}

pub(crate) fn first_seen_samples_for_observations(
    block: &ObservationPredictionBlock,
    relations: &SampleRelationSet,
) -> Result<Vec<SampleId>> {
    let mut seen = BTreeSet::new();
    let mut sample_order = Vec::new();
    for observation_id in &block.observation_ids {
        let sample_id = relations
            .sample_for_observation(observation_id)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "observation prediction `{observation_id}` has no sample relation"
                ))
            })?;
        if seen.insert(sample_id.clone()) {
            sample_order.push(sample_id.clone());
        }
    }
    Ok(sample_order)
}

pub(crate) fn requested_unit_order_for_sample_block(
    level: PredictionLevel,
    relations: &SampleRelationSet,
    block: &PredictionBlock,
) -> Result<Vec<PredictionUnitId>> {
    let mut seen = BTreeSet::new();
    let mut unit_order = Vec::new();
    for sample_id in &block.sample_ids {
        let unit_id = match level {
            PredictionLevel::Sample => PredictionUnitId::Sample(sample_id.clone()),
            PredictionLevel::Target => relations
                .target_for_sample(sample_id)
                .cloned()
                .map(PredictionUnitId::Target)
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "sample `{sample_id}` is missing target id for target aggregation"
                    ))
                })?,
            PredictionLevel::Group => relations
                .group_for_sample(sample_id)
                .cloned()
                .map(PredictionUnitId::Group)
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "sample `{sample_id}` is missing group id for group aggregation"
                    ))
                })?,
            PredictionLevel::Observation => {
                return Err(DagMlError::OofValidation(
                    "sample prediction aggregation cannot output observation-level predictions"
                        .to_string(),
                ));
            }
        };
        if seen.insert(unit_id.clone()) {
            unit_order.push(unit_id);
        }
    }
    Ok(unit_order)
}

pub(crate) fn aggregation_task_id(
    task: &NodeTask,
    producer_node: &NodeId,
    fold_id: Option<&FoldId>,
    stage: &str,
) -> String {
    let fold = fold_id
        .map(ToString::to_string)
        .unwrap_or_else(|| "nofold".to_string());
    format!(
        "aggregation:{}:{}:{}:{}:{}",
        task.run_id, task.node_plan.node_id, producer_node, fold, stage
    )
}
