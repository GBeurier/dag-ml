// Auto-split from the former monolithic `runtime.rs` (pure refactor).
use super::*;

/// Materialize one prediction-to-Data join in graph-edge order. The scheduler
/// supplies the required training identities from the current fold scope; a
/// host cannot silently use the outer validation rows as fitting features.
pub(crate) fn join_prediction_feature_specs(
    join_node: &NodeId,
    sources: &[&PredictionInputSpec],
    required_samples: &[SampleId],
) -> Result<crate::oof::OofMatrix> {
    join_prediction_feature_specs_for_partition(
        join_node,
        sources,
        required_samples,
        PredictionPartition::Validation,
    )
}

fn join_prediction_feature_specs_for_partition(
    join_node: &NodeId,
    sources: &[&PredictionInputSpec],
    required_samples: &[SampleId],
    expected_partition: PredictionPartition,
) -> Result<crate::oof::OofMatrix> {
    if sources.is_empty() || required_samples.is_empty() {
        return Err(DagMlError::OofValidation(format!(
            "prediction feature join `{join_node}` needs sources and training samples"
        )));
    }
    let required = required_samples.iter().collect::<BTreeSet<_>>();
    if required.len() != required_samples.len() {
        return Err(DagMlError::OofValidation(format!(
            "prediction feature join `{join_node}` repeats a training sample"
        )));
    }
    let mut columns = Vec::new();
    let mut rows = vec![Vec::new(); required_samples.len()];
    let mut seen_ports = BTreeSet::new();
    for source in sources {
        if source.partition != expected_partition
            || source.prediction_level != PredictionLevel::Sample
        {
            let label = if expected_partition == PredictionPartition::Validation {
                "validation OOF"
            } else {
                "final prediction"
            };
            return Err(DagMlError::OofValidation(format!(
                "prediction feature join `{join_node}` requires sample-level {label} rows from `{}.{}`",
                source.producer_node, source.source_port,
            )));
        }
        if !seen_ports.insert((&source.producer_node, &source.source_port)) {
            return Err(DagMlError::OofValidation(format!(
                "prediction feature join `{join_node}` repeats source `{}.{}`",
                source.producer_node, source.source_port
            )));
        }
        if source.prediction_width == 0
            || source.sample_ids.len() != source.values.len()
            || (!source.target_names.is_empty()
                && source.target_names.len() != source.prediction_width)
        {
            return Err(DagMlError::OofValidation(format!(
                "prediction feature join `{join_node}` received malformed rows from `{}.{}`",
                source.producer_node, source.source_port
            )));
        }
        let by_id = source
            .sample_ids
            .iter()
            .zip(&source.values)
            .collect::<BTreeMap<_, _>>();
        if source.sample_ids.len() != required.len()
            || by_id.len() != required.len()
            || by_id.keys().copied().collect::<BTreeSet<_>>() != required
        {
            return Err(DagMlError::OofValidation(format!(
                "prediction feature join `{join_node}` source `{}.{}` does not exactly cover the training samples",
                source.producer_node, source.source_port
            )));
        }
        for (index, sample) in required_samples.iter().enumerate() {
            let values = by_id.get(sample).expect("exact coverage checked");
            if values.len() != source.prediction_width
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(DagMlError::OofValidation(format!(
                    "prediction feature join `{join_node}` source `{}.{}` has invalid numeric rows",
                    source.producer_node, source.source_port
                )));
            }
            rows[index].extend(values.iter().copied());
        }
        for column in 0..source.prediction_width {
            let target = source
                .target_names
                .get(column)
                .cloned()
                .unwrap_or_else(|| format!("p{column}"));
            columns.push(format!(
                "{}.{}__{target}",
                source.producer_node, source.source_port
            ));
        }
    }
    if columns.iter().collect::<BTreeSet<_>>().len() != columns.len() {
        return Err(DagMlError::OofValidation(format!(
            "prediction feature join `{join_node}` repeats a feature column"
        )));
    }
    Ok(crate::oof::OofMatrix {
        sample_ids: required_samples.to_vec(),
        columns,
        values: rows,
    })
}

pub(crate) fn prediction_feature_matrix_for_task(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    prediction_inputs: &BTreeMap<String, PredictionInputSpec>,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
) -> Result<Option<crate::oof::OofMatrix>> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == node_plan.node_id)
        .expect("validated node plan");
    if node.kind != NodeKind::PredictionJoin
        || node
            .metadata
            .get("prediction_feature_execution")
            .and_then(serde_json::Value::as_str)
            != Some("native_oof_v1")
    {
        return Ok(None);
    }
    if !node
        .ports
        .outputs
        .iter()
        .any(|port| port.kind == PortKind::Data)
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction feature join `{}` must have a Data output",
            node.id
        )));
    }
    if !scope.phase.is_training() && scope.phase != Phase::Predict {
        return Ok(None);
    }
    let (required_samples, suffix, expected_partition) = if scope.phase == Phase::Predict {
        let first = prediction_inputs
            .iter()
            .find(|(key, _)| key.ends_with(":predict"))
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "prediction feature join `{}` has no final prediction inputs",
                    node.id
                ))
            })?;
        (
            first.1.sample_ids.clone(),
            ":predict",
            PredictionPartition::Final,
        )
    } else {
        let fold_set = resources
            .fold_set_override
            .or(plan.fold_set.as_ref())
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "prediction feature join `{}` requires a scoped fold set",
                    node.id
                ))
            })?;
        let required = if scope.phase == Phase::FitCv {
            let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "prediction feature join `{}` requires a current FIT_CV fold",
                    node.id
                ))
            })?;
            fold_set
                .folds
                .iter()
                .find(|fold| &fold.fold_id == fold_id)
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "prediction feature join `{}` has unknown scoped fold `{fold_id}`",
                        node.id
                    ))
                })?
                .train_sample_ids
                .clone()
        } else {
            fold_set.sample_ids.clone()
        };
        (required, "", PredictionPartition::Validation)
    };
    let sources = plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node.id && edge.contract.requires_oof)
        .map(|edge| {
            let key = format!("{}.{}{suffix}", edge.source.node_id, edge.source.port_name);
            prediction_inputs.get(&key).ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "prediction feature join `{}` is missing OOF input `{key}`",
                    node.id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if expected_partition == PredictionPartition::Validation {
        join_prediction_feature_specs(&node.id, &sources, &required_samples).map(Some)
    } else {
        join_prediction_feature_specs_for_partition(
            &node.id,
            &sources,
            &required_samples,
            expected_partition,
        )
        .map(Some)
    }
}

pub(crate) fn prediction_feature_off_fold_matrix_for_task(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    prediction_inputs: &BTreeMap<String, PredictionInputSpec>,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
) -> Result<Option<crate::oof::OofMatrix>> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == node_plan.node_id)
        .expect("validated node plan");
    if node.kind != NodeKind::PredictionJoin
        || node
            .metadata
            .get("prediction_feature_execution")
            .and_then(serde_json::Value::as_str)
            != Some("native_oof_v1")
    {
        return Ok(None);
    }
    let (suffix, expected_partition, required_samples) = match scope.phase {
        Phase::FitCv => {
            let fold_set = resources
                .fold_set_override
                .or(plan.fold_set.as_ref())
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "prediction feature join `{}` requires a scoped fold set",
                        node.id
                    ))
                })?;
            let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "prediction feature join `{}` requires a current FIT_CV fold",
                    node.id
                ))
            })?;
            let fold = fold_set
                .folds
                .iter()
                .find(|fold| &fold.fold_id == fold_id)
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "prediction feature join `{}` has unknown scoped fold `{fold_id}`",
                        node.id
                    ))
                })?;
            (
                ":outer",
                PredictionPartition::Validation,
                fold.validation_sample_ids.clone(),
            )
        }
        Phase::Refit => {
            let Some(first) = prediction_inputs
                .iter()
                .find(|(key, _)| key.ends_with(":refit"))
            else {
                return Ok(None);
            };
            (
                ":refit",
                PredictionPartition::Test,
                first.1.sample_ids.clone(),
            )
        }
        _ => return Ok(None),
    };
    let sources = plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node.id && edge.contract.requires_oof)
        .map(|edge| {
            let key = format!("{}.{}{suffix}", edge.source.node_id, edge.source.port_name);
            prediction_inputs.get(&key).ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "prediction feature join `{}` is missing off-fold input `{key}`",
                    node.id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    join_prediction_feature_specs_for_partition(
        &node.id,
        &sources,
        &required_samples,
        expected_partition,
    )
    .map(Some)
}

/// Reduce per-branch model probabilities before they reach a stacking controller.
/// The selector contract is compiled by the DSL and interpreted here, on both
/// the nested OOF and off-fold paths, so every host language sees the same
/// identity-keyed meta-feature matrix.
pub(crate) fn apply_stacking_prediction_aggregations(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    prediction_inputs: &mut BTreeMap<String, PredictionInputSpec>,
    scores: &[RegressionMetricReport],
    candidate_score_folds: Option<&BTreeSet<FoldId>>,
    candidate_variant: Option<&VariantId>,
) -> Result<()> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == node_plan.node_id)
        .expect("validated node plan");
    // Mixed/prediction joins also carry DSL `selectors`, but their selection
    // contract is interpreted by their own controller. This reduction belongs
    // only to a compiled merge_model stacking node.
    if node.kind != crate::graph::NodeKind::Model || !node.metadata.contains_key("merge_mode") {
        return Ok(());
    }
    let Some(value) = node.metadata.get("selectors") else {
        return Ok(());
    };
    let selectors: Vec<crate::dsl::PipelineDslMergeSelector> =
        serde_json::from_value(value.clone()).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "stacking node `{}` has invalid selectors: {error}",
                node_plan.node_id
            ))
        })?;
    if selectors.is_empty() {
        return Ok(());
    }
    let mut reduced = BTreeMap::new();
    for selector in &selectors {
        let branch = selector.branch.as_deref();
        let mut by_suffix: BTreeMap<String, Vec<(&String, &PredictionInputSpec)>> = BTreeMap::new();
        for (key, spec) in prediction_inputs.iter() {
            let source = plan
                .graph_plan
                .graph
                .nodes
                .iter()
                .find(|source| source.id == spec.producer_node)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "stacking input `{key}` names missing producer `{}`",
                        spec.producer_node
                    ))
                })?;
            if branch.is_some_and(|branch| {
                source
                    .metadata
                    .get("dsl_branch")
                    .and_then(serde_json::Value::as_str)
                    != Some(branch)
            }) || selector
                .model
                .as_ref()
                .is_some_and(|model| model != &spec.producer_node)
            {
                continue;
            }
            let prefix = format!("{}.{}", spec.producer_node, spec.source_port);
            let suffix = key.strip_prefix(&prefix).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "stacking input `{key}` does not match its producer `{prefix}`"
                ))
            })?;
            if !matches!(suffix, "" | ":outer" | ":refit" | ":predict" | ":test") {
                return Err(DagMlError::RuntimeValidation(format!(
                    "stacking input `{key}` has unsupported suffix `{suffix}`"
                )));
            }
            by_suffix
                .entry(suffix.to_string())
                .or_default()
                .push((key, spec));
        }
        if by_suffix.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "stacking node `{}` has no predictions for selector {:?}",
                node_plan.node_id, selector
            )));
        }
        let branch = branch.unwrap_or("selected");
        let expected_models = by_suffix.values().map(Vec::len).max().unwrap_or(0);
        let virtual_producer = NodeId::new(format!("{}.branch.{}", node_plan.node_id, branch))?;
        for (suffix, inputs) in by_suffix {
            if inputs.len() != expected_models {
                return Err(DagMlError::OofValidation(format!(
                    "stacking branch `{branch}` has incomplete `{suffix}` prediction coverage across models"
                )));
            }
            let first = inputs[0].1;
            if inputs
                .iter()
                .any(|(_, spec)| spec.fold_ids != first.fold_ids)
            {
                return Err(DagMlError::OofValidation(format!(
                    "stacking branch `{branch}` models differ in OOF fold identities"
                )));
            }
            let inputs = select_stacking_inputs(
                plan,
                inputs,
                selector,
                scores,
                candidate_score_folds,
                candidate_variant,
            )?;
            if selector.aggregate.is_none() {
                for (key, spec) in inputs {
                    if reduced.insert(key.clone(), spec.clone()).is_some() {
                        return Err(DagMlError::RuntimeValidation(format!(
                            "stacking selector duplicates input `{key}`"
                        )));
                    }
                }
                continue;
            }
            let blocks = inputs
                .iter()
                .map(|(_, spec)| PredictionBlock {
                    prediction_id: None,
                    producer_node: spec.producer_node.clone(),
                    producer_port: Some(spec.source_port.clone()),
                    partition: spec.partition.clone(),
                    fold_id: spec.fold_id.clone(),
                    sample_ids: spec.sample_ids.clone(),
                    values: spec.values.clone(),
                    target_names: spec.target_names.clone(),
                })
                .collect::<Vec<_>>();
            let aggregate = match selector.aggregate.as_deref() {
                Some("proba_mean") => {
                    crate::aggregation::reduce_proba_mean_within_branch(&blocks, &virtual_producer)?
                }
                Some("mean") => {
                    crate::aggregation::reduce_mean_within_branch(&blocks, None, &virtual_producer)?
                }
                Some("weighted_mean") => {
                    let weights = stacking_model_weights(
                        &blocks,
                        scores,
                        selector.metric.as_deref().unwrap_or("rmse"),
                    );
                    crate::aggregation::reduce_mean_within_branch(
                        &blocks,
                        weights.as_deref(),
                        &virtual_producer,
                    )?
                }
                _ => {
                    return Err(DagMlError::RuntimeValidation(
                        "unsupported stacking aggregation".to_string(),
                    ))
                }
            };
            let mut spec = first.clone();
            spec.producer_node = virtual_producer.clone();
            spec.source_port = "oof".to_string();
            spec.target_port = format!("{branch}_oof");
            spec.sample_ids = aggregate.sample_ids;
            spec.values = aggregate.values;
            spec.prediction_width = spec.values.first().map_or(0, Vec::len);
            spec.target_names = aggregate.target_names;
            spec.unit_ids = spec
                .sample_ids
                .iter()
                .cloned()
                .map(PredictionUnitId::Sample)
                .collect();
            let key = format!("{virtual_producer}.oof{suffix}");
            if reduced.insert(key.clone(), spec).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "stacking aggregation produced duplicate input `{key}`"
                )));
            }
        }
    }
    *prediction_inputs = reduced;
    Ok(())
}

/// A closed producer selection request shared by native stacking and archive
/// producers across language bindings. Only validation scores can rank models.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackingProducerSelectionRequest {
    pub producer_nodes: Vec<NodeId>,
    pub select: serde_json::Value,
    pub metric: String,
    pub reports: Vec<RegressionMetricReport>,
    /// Report-grade fold scope used for candidate ranking (outer CV folds for REFIT/replay).
    #[serde(default)]
    pub fold_ids: Vec<FoldId>,
    #[serde(default)]
    pub variant_id: Option<VariantId>,
    #[serde(default)]
    pub producer_classes: BTreeMap<NodeId, String>,
}

/// Choose the one CV estimator whose held-out prediction supplies a stacking
/// test feature. Fold identity and ranking live in the core; hosts retain the
/// corresponding estimator and apply it to the prediction cohort.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackingFoldSelectionRequest {
    pub producer_node: NodeId,
    pub fold_ids: Vec<FoldId>,
    pub metric: String,
    pub reports: Vec<RegressionMetricReport>,
}

impl StackingFoldSelectionRequest {
    pub fn selected_fold_id(&self) -> Result<FoldId> {
        if self.fold_ids.is_empty()
            || self.fold_ids.iter().collect::<BTreeSet<_>>().len() != self.fold_ids.len()
        {
            return Err(DagMlError::RuntimeValidation(
                "stacking fold selection needs distinct fold ids".to_string(),
            ));
        }
        let kind = RegressionMetricKind::from_name(&self.metric).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "stacking fold selection has unsupported metric `{}`",
                self.metric
            ))
        })?;
        let mut ranked = self
            .fold_ids
            .iter()
            .enumerate()
            .filter_map(|(index, fold_id)| {
                self.reports
                    .iter()
                    .find(|report| {
                        report.producer_node == self.producer_node
                            && report.partition == PredictionPartition::Validation
                            && report.fold_id.as_ref() == Some(fold_id)
                            && report
                                .metrics
                                .get(&self.metric)
                                .is_some_and(|score| score.is_finite())
                    })
                    .and_then(|report| report.metrics.get(&self.metric))
                    .map(|score| (index, *score))
            })
            .collect::<Vec<_>>();
        if ranked.is_empty() {
            return Ok(self.fold_ids[0].clone());
        }
        let higher_better = kind.objective() == crate::selection::MetricObjective::Maximize;
        ranked.sort_by(|left, right| {
            let order = if higher_better {
                right.1.total_cmp(&left.1)
            } else {
                left.1.total_cmp(&right.1)
            };
            order.then_with(|| left.0.cmp(&right.0))
        });
        Ok(self.fold_ids[ranked[0].0].clone())
    }

    /// Objective-aware fold weights in the declared fold order. A missing
    /// validation report contributes zero; absent usable evidence falls back
    /// to a uniform mean, never to a test-derived weight.
    pub fn normalized_weights(&self) -> Result<Vec<f64>> {
        if self.fold_ids.is_empty()
            || self.fold_ids.iter().collect::<BTreeSet<_>>().len() != self.fold_ids.len()
        {
            return Err(DagMlError::RuntimeValidation(
                "stacking fold weights need distinct fold ids".to_string(),
            ));
        }
        let kind = RegressionMetricKind::from_name(&self.metric).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "stacking fold weights have unsupported metric `{}`",
                self.metric
            ))
        })?;
        let higher_better = kind.objective() == crate::selection::MetricObjective::Maximize;
        let mut weights = self
            .fold_ids
            .iter()
            .map(|fold_id| {
                self.reports
                    .iter()
                    .find(|report| {
                        report.producer_node == self.producer_node
                            && report.partition == PredictionPartition::Validation
                            && report.fold_id.as_ref() == Some(fold_id)
                    })
                    .and_then(|report| report.metrics.get(&self.metric))
                    .filter(|score| score.is_finite())
                    .map_or(0.0, |score| {
                        if higher_better {
                            score.max(0.0)
                        } else if *score >= 0.0 {
                            1.0 / (score + 1e-10)
                        } else {
                            score.abs()
                        }
                    })
            })
            .collect::<Vec<_>>();
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        } else {
            weights.fill(1.0 / self.fold_ids.len() as f64);
        }
        Ok(weights)
    }
}

impl StackingProducerSelectionRequest {
    pub fn selected_producer_nodes(&self) -> Result<Vec<NodeId>> {
        if self.producer_nodes.is_empty()
            || self.producer_nodes.iter().collect::<BTreeSet<_>>().len()
                != self.producer_nodes.len()
        {
            return Err(DagMlError::RuntimeValidation(
                "stacking selection needs distinct producer nodes".to_string(),
            ));
        }
        if let Some(config) = self
            .select
            .as_object()
            .filter(|config| config.contains_key("fold_candidates_top_k"))
        {
            return self.selected_fold_candidate_nodes(config);
        }
        if let Some(config) = self
            .select
            .as_object()
            .and_then(|config| config.get("diverse_fold_candidates"))
        {
            return self.selected_diverse_candidate_nodes(config);
        }
        let limit = match &self.select {
            serde_json::Value::String(mode) if mode == "all" => {
                return Ok(self.producer_nodes.clone())
            }
            serde_json::Value::String(mode) if mode == "best" => 1,
            serde_json::Value::Object(config)
                if config.len() == 1 && config.contains_key("models") =>
            {
                let models = config["models"].as_array().ok_or_else(|| {
                    DagMlError::RuntimeValidation(
                        "stacking models must be a non-empty array of producer node ids"
                            .to_string(),
                    )
                })?;
                let selected = models
                    .iter()
                    .map(|value| value.as_str().and_then(|id| NodeId::new(id).ok()))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(
                            "stacking models must contain valid producer node ids".to_string(),
                        )
                    })?;
                if selected.is_empty()
                    || selected.iter().collect::<BTreeSet<_>>().len() != selected.len()
                    || selected
                        .iter()
                        .any(|node| !self.producer_nodes.contains(node))
                {
                    return Err(DagMlError::RuntimeValidation(
                        "stacking models must be distinct producers in the selection scope"
                            .to_string(),
                    ));
                }
                return Ok(selected);
            }
            serde_json::Value::Object(config) if config.len() == 1 => config
                .get("top_k")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(
                        "stacking top_k must be a positive integer".to_string(),
                    )
                })?,
            _ => {
                return Err(DagMlError::RuntimeValidation(
                    "stacking select must be all, best or an object with top_k or models"
                        .to_string(),
                ))
            }
        };
        if limit == 0 || limit > self.producer_nodes.len() || self.metric.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "stacking selection needs a valid top_k and metric".to_string(),
            ));
        }
        let mut ranked = self
            .producer_nodes
            .iter()
            .enumerate()
            .filter_map(|(index, producer)| {
                self.reports
                    .iter()
                    .find(|report| {
                        report.producer_node == *producer
                            && report.partition == PredictionPartition::Validation
                            && report.fold_id.is_some()
                            && report
                                .metrics
                                .get(&self.metric)
                                .is_some_and(|score| score.is_finite())
                    })
                    .and_then(|report| report.metrics.get(&self.metric))
                    .map(|score| (index, *score))
            })
            .collect::<Vec<_>>();
        if ranked.is_empty() {
            return Ok(self.producer_nodes.iter().take(limit).cloned().collect());
        }
        let higher_better = crate::metrics::RegressionMetricKind::from_name(&self.metric)
            .is_some_and(|kind| kind.objective() == crate::selection::MetricObjective::Maximize)
            || matches!(self.metric.as_str(), "f1" | "auc");
        ranked.sort_by(|left, right| {
            let order = if higher_better {
                right.1.total_cmp(&left.1)
            } else {
                left.1.total_cmp(&right.1)
            };
            order.then_with(|| left.0.cmp(&right.0))
        });
        Ok(ranked
            .into_iter()
            .take(limit)
            .map(|(index, _)| self.producer_nodes[index].clone())
            .collect())
    }

    fn selected_fold_candidate_nodes(
        &self,
        config: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Vec<NodeId>> {
        let limit = config
            .get("fold_candidates_top_k")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "stacking fold_candidates_top_k must be a positive integer".to_string(),
                )
            })?;
        let kind = RegressionMetricKind::from_name(&self.metric).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "stacking fold candidate selection has unsupported metric `{}`",
                self.metric
            ))
        })?;
        let ascending = config
            .get("ascending")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(kind.objective() != crate::selection::MetricObjective::Maximize);
        let allowed = self.fold_ids.iter().collect::<BTreeSet<_>>();
        let producers = self.producer_nodes.iter().collect::<BTreeSet<_>>();
        let mut candidates = self
            .reports
            .iter()
            .enumerate()
            .filter_map(|(index, report)| {
                let fold = report.fold_id.as_ref()?;
                let score = *report.metrics.get(&self.metric)?;
                (report.partition == PredictionPartition::Validation
                    && report.level == PredictionLevel::Sample
                    && score.is_finite()
                    && producers.contains(&report.producer_node)
                    && (allowed.is_empty() || allowed.contains(fold))
                    && self
                        .variant_id
                        .as_ref()
                        .is_none_or(|variant| report.variant_id.as_ref() == Some(variant)))
                .then_some((index, &report.producer_node, score))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "stacking fold candidate selection has no validation `{}` scores in its fold scope",
                self.metric,
            )));
        }
        candidates.sort_by(|left, right| {
            (if ascending {
                left.2.total_cmp(&right.2)
            } else {
                right.2.total_cmp(&left.2)
            })
            .then_with(|| left.0.cmp(&right.0))
        });
        let mut seen = BTreeSet::new();
        Ok(candidates
            .into_iter()
            .take(limit)
            .filter_map(|(_, producer, _)| {
                seen.insert((*producer).clone())
                    .then_some((*producer).clone())
            })
            .collect())
    }

    fn selected_diverse_candidate_nodes(&self, config: &serde_json::Value) -> Result<Vec<NodeId>> {
        let max_per_class = config
            .get("max_per_class")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "stacking diverse_fold_candidates needs a positive max_per_class".to_string(),
                )
            })?;
        let kind = RegressionMetricKind::from_name(&self.metric).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "stacking diverse fold selection has unsupported metric `{}`",
                self.metric
            ))
        })?;
        let ascending = kind.objective() != crate::selection::MetricObjective::Maximize;
        let allowed = self.fold_ids.iter().collect::<BTreeSet<_>>();
        let producers = self.producer_nodes.iter().collect::<BTreeSet<_>>();
        let mut classes: BTreeMap<&str, Vec<(usize, &NodeId, f64)>> = BTreeMap::new();
        for (index, report) in self.reports.iter().enumerate() {
            let Some(fold) = report.fold_id.as_ref() else {
                continue;
            };
            let Some(score) = report.metrics.get(&self.metric).copied() else {
                continue;
            };
            if report.partition != PredictionPartition::Validation
                || report.level != PredictionLevel::Sample
                || !score.is_finite()
                || !producers.contains(&report.producer_node)
                || (!allowed.is_empty() && !allowed.contains(fold))
                || self
                    .variant_id
                    .as_ref()
                    .is_some_and(|variant| report.variant_id.as_ref() != Some(variant))
            {
                continue;
            }
            let class = self
                .producer_classes
                .get(&report.producer_node)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "stacking diverse fold selection lacks class for producer `{}`",
                        report.producer_node
                    ))
                })?;
            classes
                .entry(class.as_str())
                .or_default()
                .push((index, &report.producer_node, score));
        }
        if classes.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "stacking diverse fold selection has no validation `{}` scores in its fold scope",
                self.metric,
            )));
        }
        let preferred = config
            .get("preferred_classes")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "stacking diverse fold selection needs preferred_classes strings".to_string(),
                )
            })?;
        let mut class_order = preferred
            .iter()
            .filter_map(serde_json::Value::as_str)
            .filter(|class| classes.contains_key(class))
            .collect::<Vec<_>>();
        for class in classes.keys() {
            if !class_order.contains(class) {
                class_order.push(class);
            }
        }
        let mut selected = BTreeSet::new();
        for class in class_order {
            let candidates = classes
                .get_mut(class)
                .expect("class order came from groups");
            candidates.sort_by(|left, right| {
                (if ascending {
                    left.2.total_cmp(&right.2)
                } else {
                    right.2.total_cmp(&left.2)
                })
                .then_with(|| left.0.cmp(&right.0))
            });
            for (_, producer, _) in candidates.iter().take(max_per_class) {
                selected.insert((*producer).clone());
            }
        }
        Ok(self
            .producer_nodes
            .iter()
            .filter(|producer| selected.contains(*producer))
            .cloned()
            .collect())
    }
}

fn select_stacking_inputs<'a>(
    plan: &ExecutionPlan,
    inputs: Vec<(&'a String, &'a PredictionInputSpec)>,
    selector: &crate::dsl::PipelineDslMergeSelector,
    scores: &[RegressionMetricReport],
    candidate_score_folds: Option<&BTreeSet<FoldId>>,
    candidate_variant: Option<&VariantId>,
) -> Result<Vec<(&'a String, &'a PredictionInputSpec)>> {
    let Some(select) = &selector.select else {
        return Ok(inputs);
    };
    let request = StackingProducerSelectionRequest {
        producer_nodes: inputs
            .iter()
            .map(|(_, input)| input.producer_node.clone())
            .collect(),
        select: select.clone(),
        metric: selector
            .metric
            .clone()
            .unwrap_or_else(|| "rmse".to_string()),
        reports: scores.to_vec(),
        fold_ids: candidate_score_folds
            .map_or_else(Vec::new, |folds| folds.iter().cloned().collect()),
        variant_id: candidate_variant.cloned(),
        producer_classes: if select
            .as_object()
            .is_some_and(|value| value.contains_key("diverse_fold_candidates"))
        {
            inputs
                .iter()
                .map(|(_, input)| {
                    let node = plan
                        .graph_plan
                        .graph
                        .nodes
                        .iter()
                        .find(|node| node.id == input.producer_node)
                        .ok_or_else(|| {
                            DagMlError::RuntimeValidation(format!(
                                "stacking diverse fold selection lacks producer `{}`",
                                input.producer_node,
                            ))
                        })?;
                    let class = node
                        .operator
                        .as_ref()
                        .and_then(|operator| {
                            operator
                                .get("class")
                                .and_then(serde_json::Value::as_str)
                                .or_else(|| operator.as_str())
                        })
                        .and_then(|name| name.rsplit('.').next())
                        .filter(|name| !name.is_empty())
                        .ok_or_else(|| {
                            DagMlError::RuntimeValidation(format!(
                                "stacking diverse fold selection needs operator class for `{}`",
                                input.producer_node,
                            ))
                        })?;
                    Ok((input.producer_node.clone(), class.to_string()))
                })
                .collect::<Result<BTreeMap<_, _>>>()?
        } else {
            BTreeMap::new()
        },
    };
    let selected = request.selected_producer_nodes()?;
    Ok(selected
        .iter()
        .filter_map(|producer| {
            inputs
                .iter()
                .find(|(_, input)| &input.producer_node == producer)
                .copied()
        })
        .collect())
}

/// Legacy weighted mean uses inverse validation error and falls back to equal
/// weights when any model lacks a score. The first report for each producer is
/// chosen consistently so refit and replay do not depend on later score rows.
fn stacking_model_weights(
    blocks: &[PredictionBlock],
    scores: &[RegressionMetricReport],
    metric: &str,
) -> Option<Vec<f64>> {
    let higher_better = crate::metrics::RegressionMetricKind::from_name(metric)
        .is_some_and(|kind| kind.objective() == crate::selection::MetricObjective::Maximize);
    let weights = blocks
        .iter()
        .map(|block| {
            let score = scores
                .iter()
                .find(|report| {
                    report.producer_node == block.producer_node
                        && report.partition == PredictionPartition::Validation
                        && report.fold_id.is_some()
                        && report.metrics.contains_key(metric)
                })
                .and_then(|report| report.metrics.get(metric));
            match score {
                Some(score) if score.is_finite() && higher_better => score.max(0.0),
                Some(score) if score.is_finite() && *score >= 0.0 => 1.0 / (score + 1e-10),
                Some(score) if score.is_finite() => score.abs(),
                _ => 0.0,
            }
        })
        .collect::<Vec<_>>();
    weights
        .iter()
        .any(|weight| *weight > 0.0)
        .then_some(weights)
}

pub(crate) fn effective_node_plan_for_scope(
    node_plan: &NodePlan,
    scope: &PhaseScope,
) -> Result<NodePlan> {
    let Some(variant) = &scope.variant else {
        return Ok(node_plan.clone());
    };
    let params = variant.effective_params_for_node(&node_plan.node_id, &node_plan.params)?;
    if params == node_plan.params {
        return Ok(node_plan.clone());
    }
    let mut node_plan = node_plan.clone();
    node_plan.params = params;
    node_plan.params_fingerprint = stable_json_fingerprint(&node_plan.params)?;
    Ok(node_plan)
}

pub(crate) fn incoming_oof_edges<'a>(
    plan: &'a ExecutionPlan,
    node_plan: &NodePlan,
) -> Result<Vec<&'a EdgeSpec>> {
    plan.graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node_plan.node_id && edge.contract.requires_oof)
        .map(|edge| {
            if edge.contract.kind != PortKind::Prediction {
                return Err(DagMlError::RuntimeValidation(format!(
                    "edge `{}.{}` -> `{}.{}` requires OOF but is not a prediction edge",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
            Ok(edge)
        })
        .collect()
}

pub(crate) fn incoming_training_oof_edges<'a>(
    plan: &'a ExecutionPlan,
    node_plan: &NodePlan,
    scope: &PhaseScope,
) -> Result<Vec<&'a EdgeSpec>> {
    if !scope.phase.is_training() {
        return Ok(Vec::new());
    }
    incoming_oof_edges(plan, node_plan)
}

/// Validate that a graph edge's source port is a prediction output. Stored block
/// provenance itself is checked by [`prediction_block_matches_edge_source_port`]
/// and [`aggregated_prediction_block_matches_edge_source_port`], which can use
/// the R1 `producer_port` field while preserving the legacy single-port fallback.
pub(crate) fn validate_oof_source_port_provenance(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
) -> Result<()> {
    let source = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == edge.source.node_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "OOF edge source node `{}` is absent from the execution graph",
                edge.source.node_id
            ))
        })?;
    let mut prediction_ports = source
        .ports
        .outputs
        .iter()
        .filter(|port| port.kind == PortKind::Prediction)
        .map(|port| port.name.as_str())
        .collect::<Vec<_>>();
    prediction_ports.sort_unstable();
    if !prediction_ports
        .iter()
        .any(|port| *port == edge.source.port_name)
    {
        return Err(DagMlError::OofValidation(format!(
            "OOF edge source `{}.{}` is not a declared Prediction output; declared prediction ports are {:?}",
            edge.source.node_id,
            edge.source.port_name,
            prediction_ports
        )));
    }
    Ok(())
}

fn legacy_absent_port_matches_single_prediction_output(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    block_kind: &str,
) -> Result<bool> {
    let source = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == edge.source.node_id)
        .ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "OOF edge source node `{}` is absent from the execution graph",
                edge.source.node_id
            ))
        })?;
    let mut prediction_ports = source
        .ports
        .outputs
        .iter()
        .filter(|port| port.kind == PortKind::Prediction)
        .map(|port| port.name.as_str())
        .collect::<Vec<_>>();
    prediction_ports.sort_unstable();
    if prediction_ports.len() == 1 {
        return Ok(prediction_ports[0] == edge.source.port_name);
    }
    Err(DagMlError::OofValidation(format!(
        "cannot bind legacy {block_kind} without producer_port to `{}.{}`: producer `{}` exposes {} Prediction output ports {:?}; multi-output producers must store producer_port explicitly",
        edge.source.node_id,
        edge.source.port_name,
        edge.source.node_id,
        prediction_ports.len(),
        prediction_ports
    )))
}

pub(crate) fn producer_port_matches_edge_source_port(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    producer_node: &NodeId,
    producer_port: Option<&str>,
    block_kind: &str,
) -> Result<bool> {
    match producer_port {
        Some(port) if port.trim().is_empty() => Err(DagMlError::OofValidation(format!(
            "{block_kind} from `{producer_node}` has blank producer_port"
        ))),
        Some(port) => Ok(port == edge.source.port_name),
        None => legacy_absent_port_matches_single_prediction_output(plan, edge, block_kind),
    }
}

pub(crate) fn prediction_block_matches_edge_source_port(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    block: &PredictionBlock,
) -> Result<bool> {
    producer_port_matches_edge_source_port(
        plan,
        edge,
        &block.producer_node,
        block.producer_port.as_deref(),
        "prediction block",
    )
}

pub(crate) fn aggregated_prediction_block_matches_edge_source_port(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    block: &AggregatedPredictionBlock,
) -> Result<bool> {
    producer_port_matches_edge_source_port(
        plan,
        edge,
        &block.producer_node,
        block.producer_port.as_deref(),
        "aggregated prediction block",
    )
}

pub(crate) fn filter_prediction_blocks_for_edge_source_port<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    blocks: impl IntoIterator<Item = &'a PredictionBlock>,
) -> Result<Vec<&'a PredictionBlock>> {
    blocks
        .into_iter()
        .filter_map(
            |block| match prediction_block_matches_edge_source_port(plan, edge, block) {
                Ok(true) => Some(Ok(block)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

pub(crate) fn filter_aggregated_prediction_blocks_for_edge_source_port<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    blocks: impl IntoIterator<Item = &'a AggregatedPredictionBlock>,
) -> Result<Vec<&'a AggregatedPredictionBlock>> {
    blocks
        .into_iter()
        .filter_map(|block| {
            match aggregated_prediction_block_matches_edge_source_port(plan, edge, block) {
                Ok(true) => Some(Ok(block)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

/// The base producer's off-fold (test / predict) predictions delivered to a
/// stacking meta-node as a SEPARATE prediction input in REFIT / PREDICT, so the
/// host meta-model can predict from them. This is the prediction-stacking analogue
/// of the concat/fusion off-fold reassembly: the FIT_CV `requires_oof` path stays
/// Validation-OOF-only (the meta-features the meta-model trains on), and these
/// test/predict base predictions are a distinct input used ONLY in REFIT/PREDICT
/// scoring — never in FIT_CV training.
///
/// Reads the base producer's `fold_id == None` block in the phase-expected
/// partition (`Test` in REFIT / `Final` in PREDICT) scoped to the active variant,
/// and builds a [`CollectedPredictionInput`] (a prediction handle + a
/// [`PredictionInputSpec`] carrying its per-sample `values`), mirroring the
/// FIT_CV OOF input so the host adapter sees a handle alongside the spec. Returns
/// `None` when the base produced no such block (a phase with no base prediction).
///
/// LEAKAGE INVARIANT: never reads a `Validation` block, so the Validation-OOF
/// meta-features are untouched. Only runs in REFIT/PREDICT (the caller guards it),
/// and the phase-expected-partition filter keeps a stale `Final`/`Train` block
/// from a prior phase in the same context out of the meta-features.
pub(crate) fn collect_off_fold_prediction_input(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<Option<CollectedPredictionInput>> {
    validate_oof_source_port_provenance(plan, edge)?;
    let expected_partition = expected_off_fold_partition(scope.phase);
    let target = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == edge.target.node_id);
    let aggregation = if scope.phase == Phase::Refit {
        target
            .and_then(|node| node.metadata.get("stacking_test_aggregation"))
            .and_then(serde_json::Value::as_str)
    } else {
        None
    };
    let fold_request = if matches!(aggregation, Some("best" | "weighted")) {
        let fold_set = plan.fold_set.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "stacking fold test aggregation requires a CV fold set".to_string(),
            )
        })?;
        let metric = target
            .and_then(|node| node.metadata.get("stacking_test_metric"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("rmse");
        Some(StackingFoldSelectionRequest {
            producer_node: edge.source.node_id.clone(),
            fold_ids: fold_set
                .folds
                .iter()
                .map(|fold| fold.fold_id.clone())
                .collect(),
            metric: metric.to_string(),
            reports: if ctx.score_collector.is_empty() {
                ctx.stacking_weight_scores.clone()
            } else {
                ctx.score_collector.clone()
            },
        })
    } else {
        None
    };
    let best_fold = if aggregation == Some("best") {
        Some(
            fold_request
                .as_ref()
                .expect("best has fold request")
                .selected_fold_id()?,
        )
    } else {
        None
    };
    let raw_blocks: Vec<&PredictionBlock> = ctx
        .prediction_store
        .find(Some(&edge.source.node_id), Some(&expected_partition), None)
        .into_iter()
        .filter(|block| {
            if let Some(fold) = best_fold.as_ref() {
                block.fold_id.as_ref() == Some(fold)
            } else if aggregation == Some("weighted") {
                fold_request.as_ref().is_some_and(|request| {
                    block
                        .fold_id
                        .as_ref()
                        .is_some_and(|fold| request.fold_ids.contains(fold))
                })
            } else {
                block.fold_id.is_none()
            }
        })
        .collect();
    let selected_blocks =
        filter_prediction_blocks_for_edge_source_port(plan, edge, raw_blocks.clone())?;
    if !raw_blocks.is_empty() && selected_blocks.is_empty() {
        return Err(DagMlError::OofValidation(format!(
            "meta node `{}` found off-fold ({expected_partition:?}) blocks for producer `{}` but none for source port `{}`",
            edge.target.node_id, edge.source.node_id, edge.source.port_name
        )));
    }
    if selected_blocks.is_empty() {
        return Ok(None);
    }
    let weighted_block = if aggregation == Some("weighted") {
        let request = fold_request.as_ref().expect("weighted has fold request");
        let weights = request.normalized_weights()?;
        let mut fold_blocks = Vec::with_capacity(request.fold_ids.len());
        for fold_id in &request.fold_ids {
            let matching = selected_blocks
                .iter()
                .filter(|block| block.fold_id.as_ref() == Some(fold_id))
                .collect::<Vec<_>>();
            if matching.len() != 1 {
                return Err(DagMlError::OofValidation(format!(
                    "stacking weighted test aggregation requires one block for fold `{fold_id}` of `{}`; found {}",
                    edge.source.node_id,
                    matching.len()
                )));
            }
            let mut block = (**matching[0]).clone();
            block.fold_id = None;
            fold_blocks.push(block);
        }
        Some(crate::aggregation::reduce_mean_within_branch(
            &fold_blocks,
            Some(&weights),
            &edge.source.node_id,
        )?)
    } else {
        None
    };
    let blocks: Vec<&PredictionBlock> = weighted_block
        .as_ref()
        .map_or(selected_blocks.clone(), |block| vec![block]);
    if blocks.len() > 1 {
        return Err(DagMlError::OofValidation(format!(
            "meta node `{}` found {} off-fold ({expected_partition:?}) blocks for base `{}`: the run context mixes several variants — predict each variant in its own context (native SELECT does this)",
            edge.target.node_id,
            blocks.len(),
            edge.source.node_id,
        )));
    }
    let block = blocks[0];
    let width = block.validate_shape()?;
    let target_names = if block.target_names.is_empty() {
        (0..width).map(|index| format!("p{index}")).collect()
    } else {
        block.target_names.clone()
    };
    let source_plan = plan
        .node_plans
        .get(&edge.source.node_id)
        .expect("edge source has a node plan");
    let handle = HandleRef {
        handle: deterministic_oof_handle(plan, edge, ctx, scope)?,
        kind: HandleKind::Prediction,
        owner_controller: source_plan.controller_id.clone(),
    };
    Ok(Some(CollectedPredictionInput {
        handle,
        spec: PredictionInputSpec {
            producer_node: edge.source.node_id.clone(),
            source_port: edge.source.port_name.clone(),
            target_port: edge.target.port_name.clone(),
            partition: block.partition.clone(),
            prediction_level: PredictionLevel::Sample,
            fold_id: None,
            fold_ids: Vec::new(),
            unit_ids: block
                .sample_ids
                .iter()
                .cloned()
                .map(PredictionUnitId::Sample)
                .collect(),
            sample_ids: block.sample_ids.clone(),
            values: block.values.clone(),
            prediction_width: width,
            target_names,
        },
    }))
}

/// Deliver the producer's held-out Test prediction for the *current* CV fold.
/// This is a separate, explicitly requested prediction input for evaluation of
/// a downstream learner; it must never be mixed into its Validation OOF fit
/// matrix. Exact fold matching also excludes nested and prior-fold evidence.
pub(crate) fn collect_cv_fold_test_prediction_input(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<Option<CollectedPredictionInput>> {
    validate_oof_source_port_provenance(plan, edge)?;
    if scope.phase != Phase::FitCv {
        return Err(DagMlError::RuntimeValidation(
            "CV fold Test prediction input requires FIT_CV".to_string(),
        ));
    }
    let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation("CV fold Test prediction input requires a fold".to_string())
    })?;
    let raw_blocks = ctx
        .prediction_store
        .find(
            Some(&edge.source.node_id),
            Some(&PredictionPartition::Test),
            None,
        )
        .into_iter()
        .filter(|block| block.fold_id.as_ref() == Some(fold_id))
        .collect::<Vec<_>>();
    let blocks = filter_prediction_blocks_for_edge_source_port(plan, edge, raw_blocks.clone())?;
    if !raw_blocks.is_empty() && blocks.is_empty() {
        return Err(DagMlError::OofValidation(format!(
            "meta node `{}` found CV fold Test blocks for producer `{}` but none for source port `{}`",
            edge.target.node_id, edge.source.node_id, edge.source.port_name
        )));
    }
    let Some(block) = blocks.first() else {
        return Ok(None);
    };
    if blocks.len() != 1 {
        return Err(DagMlError::OofValidation(format!(
            "meta node `{}` requires one Test block for fold `{fold_id}` of `{}`; found {}",
            edge.target.node_id,
            edge.source.node_id,
            blocks.len()
        )));
    }
    let width = block.validate_shape()?;
    let source_plan = plan
        .node_plans
        .get(&edge.source.node_id)
        .expect("edge source has a node plan");
    Ok(Some(CollectedPredictionInput {
        handle: HandleRef {
            handle: deterministic_cv_fold_test_handle(plan, edge, ctx, scope)?,
            kind: HandleKind::Prediction,
            owner_controller: source_plan.controller_id.clone(),
        },
        spec: PredictionInputSpec {
            producer_node: edge.source.node_id.clone(),
            source_port: edge.source.port_name.clone(),
            target_port: edge.target.port_name.clone(),
            partition: PredictionPartition::Test,
            prediction_level: PredictionLevel::Sample,
            fold_id: Some(fold_id.clone()),
            fold_ids: Vec::new(),
            unit_ids: block
                .sample_ids
                .iter()
                .cloned()
                .map(PredictionUnitId::Sample)
                .collect(),
            sample_ids: block.sample_ids.clone(),
            values: block.values.clone(),
            prediction_width: width,
            target_names: if block.target_names.is_empty() {
                (0..width).map(|index| format!("p{index}")).collect()
            } else {
                block.target_names.clone()
            },
        },
    }))
}

fn deterministic_cv_fold_test_handle(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<u64> {
    let fingerprint = stable_json_fingerprint(&(
        &plan.id,
        &ctx.run_id,
        &edge.source.node_id,
        &edge.source.port_name,
        &edge.target.node_id,
        &edge.target.port_name,
        scope.phase,
        &scope.variant_id,
        &scope.fold_id,
        "cv-fold-test",
    ))?;
    Ok(u64::from_str_radix(&fingerprint[..16], 16).expect("sha256 hex prefix should fit into u64"))
}

pub(crate) struct CollectedPredictionInput {
    pub(crate) handle: HandleRef,
    pub(crate) spec: PredictionInputSpec,
}

pub(crate) fn collect_oof_prediction_input(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
) -> Result<Option<CollectedPredictionInput>> {
    validate_oof_source_port_provenance(plan, edge)?;
    if scope.phase == Phase::Refit {
        if let Some(contract) = replay_prediction_cache_contract_for_edge(resources, edge) {
            if contract.requirement.prediction_level != PredictionLevel::Sample {
                let source_plan = plan
                    .node_plans
                    .get(&edge.source.node_id)
                    .expect("edge source has a node plan");
                let handle = materialize_oof_prediction_handle(
                    plan,
                    edge,
                    ctx,
                    scope,
                    resources,
                    &source_plan.controller_id,
                )?;
                return Ok(Some(CollectedPredictionInput {
                    handle,
                    spec: prediction_input_spec_from_requirement(&contract.requirement, scope)?,
                }));
            }
        }
    }
    let source_plan = plan
        .node_plans
        .get(&edge.source.node_id)
        .expect("edge source has a node plan");
    let prediction_level = oof_prediction_level_for_source(source_plan);
    if prediction_level != PredictionLevel::Sample {
        let blocks = match scope.phase {
            Phase::FitCv => validate_fit_cv_aggregated_oof_edge(
                plan,
                edge,
                ctx,
                scope,
                resources,
                prediction_level,
            )?,
            Phase::Refit => {
                validate_refit_aggregated_oof_edge(plan, edge, ctx, resources, prediction_level)?
            }
            _ => Vec::new(),
        };
        let handle = materialize_oof_prediction_handle(
            plan,
            edge,
            ctx,
            scope,
            resources,
            &source_plan.controller_id,
        )?;
        return Ok(Some(CollectedPredictionInput {
            handle,
            spec: aggregated_prediction_input_spec(edge, scope, prediction_level, &blocks)?,
        }));
    }
    let blocks = match scope.phase {
        Phase::FitCv => Some(validate_fit_cv_oof_edge(
            plan,
            edge,
            ctx,
            scope,
            resources.fold_set_override,
        )?),
        Phase::Refit => validate_refit_oof_edge(plan, edge, ctx)?,
        _ => Some(Vec::new()),
    };
    let Some(blocks) = blocks else {
        return Ok(None);
    };
    let handle = materialize_oof_prediction_handle(
        plan,
        edge,
        ctx,
        scope,
        resources,
        &source_plan.controller_id,
    )?;
    Ok(Some(CollectedPredictionInput {
        handle,
        spec: prediction_input_spec(
            edge,
            scope,
            &blocks,
            scope.phase == Phase::Refit
                && plan_oof_partition_mode(plan) == FoldPartitionMode::Resampled,
        )?,
    }))
}

pub(crate) fn oof_prediction_level_for_source(source_plan: &NodePlan) -> PredictionLevel {
    source_plan
        .shape_plan
        .as_ref()
        .map(|shape_plan| shape_plan.aggregation_policy.aggregation_level)
        .unwrap_or(PredictionLevel::Sample)
}

pub(crate) fn replay_prediction_cache_contract_for_edge<'a>(
    resources: &'a PhaseScopeResources<'_>,
    edge: &EdgeSpec,
) -> Option<&'a ReplayPredictionCacheContract> {
    let contracts = resources.prediction_cache_contracts?;
    let key = bundle_prediction_requirement_key(
        &edge.source.node_id,
        &edge.source.port_name,
        &edge.target.node_id,
        &edge.target.port_name,
    );
    contracts.get(&key)
}

pub(crate) fn materialize_oof_prediction_handle(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
    producer_controller_id: &ControllerId,
) -> Result<HandleRef> {
    if scope.phase == Phase::Refit {
        if let (Some(store), Some(bundle_id), Some(contracts)) = (
            resources.prediction_cache_store,
            resources.replay_bundle_id,
            resources.prediction_cache_contracts,
        ) {
            let key = bundle_prediction_requirement_key(
                &edge.source.node_id,
                &edge.source.port_name,
                &edge.target.node_id,
                &edge.target.port_name,
            );
            let contract = contracts.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "replay prediction cache store cannot materialize missing requirement `{key}`"
                ))
            })?;
            let handle = store.materialize(&PredictionCacheMaterializationRequest {
                run_id: ctx.run_id.clone(),
                bundle_id: bundle_id.clone(),
                phase: scope.phase,
                variant_id: scope.variant_id.clone(),
                requirement: contract.requirement.clone(),
                cache: contract.cache.clone(),
                producer_controller_id: producer_controller_id.clone(),
            })?;
            if handle.kind != HandleKind::Prediction {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store materialized requirement `{key}` as {:?}",
                    handle.kind
                )));
            }
            if &handle.owner_controller != producer_controller_id {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store materialized requirement `{key}` for controller `{}`, expected `{}`",
                    handle.owner_controller, producer_controller_id
                )));
            }
            return Ok(handle);
        }
    }
    Ok(HandleRef {
        handle: deterministic_oof_handle(plan, edge, ctx, scope)?,
        kind: HandleKind::Prediction,
        owner_controller: producer_controller_id.clone(),
    })
}

pub(crate) fn validate_fit_cv_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
    scope: &PhaseScope,
    fold_set_override: Option<&FoldSet>,
) -> Result<Vec<&'a PredictionBlock>> {
    let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` requires OOF predictions but FIT_CV has no fold scope",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        ))
    })?;
    let blocks = ctx.prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        Some(fold_id),
    );
    let blocks = filter_prediction_blocks_for_edge_source_port(plan, edge, blocks)?;
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, Some(fold_id)));
    }
    // MANDATORY exact OOF coverage (spec rule 3 + audit R-P0-2): a `requires_oof` stacking edge that
    // reaches here must have exactly one validation prediction per fold-validation sample, exact and
    // unique. This was previously gated by `requires_fold_alignment` — making completeness conditional,
    // so an edge that left the flag unset (a future builder or adversarial JSON) admitted blocks that
    // merely *exist*. The branch-merge concat partition exception ("unless an explicit aggregation
    // policy says otherwise"), where a branch legitimately covers only its partition, is intercepted
    // before this code path (the separation-merge handler) and so is never over-rejected here.
    let fold_set = match fold_set_override {
        Some(folds) => folds,
        None => required_fold_set_for_oof(plan, edge)?,
    };
    validate_oof_blocks_match_fold(edge, fold_set, fold_id, &blocks)?;
    Ok(blocks)
}

pub(crate) fn validate_refit_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
) -> Result<Option<Vec<&'a PredictionBlock>>> {
    let contract = stacking_oof_refit_contract_for_edge(plan, edge)?;
    let blocks = ctx.prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        None,
    );
    // Nested execution retains outer evaluation OOF, per-outer inner OOF,
    // and optionally separately declared REFIT OOF. Select exactly the
    // declared REFIT pool (outer OOF by default), never combine evidence
    // classes or average duplicate/foreign folds to manufacture coverage.
    let target_is_prediction_feature_join = plan.graph_plan.graph.nodes.iter().any(|node| {
        node.id == edge.target.node_id
            && node.kind == NodeKind::PredictionJoin
            && node
                .metadata
                .get("prediction_feature_execution")
                .and_then(serde_json::Value::as_str)
                == Some("native_oof_v1")
    });
    let nested = if is_nested_stacking_meta_node(plan, &edge.target.node_id)? {
        nested_stacking_campaign_plan_for_node(plan, edge.target.node_id.clone())?
    } else if target_is_prediction_feature_join {
        let mut owner = None;
        for level in plan.node_parallel_levels_for_phase(Phase::FitCv)? {
            for node_id in level {
                if is_nested_stacking_meta_node(plan, &node_id)? {
                    let campaign = nested_stacking_campaign_plan_for_node(plan, node_id)?
                        .expect("validated nested meta node");
                    if campaign.base_node_ids.contains(&edge.target.node_id) {
                        owner = Some(campaign);
                        break;
                    }
                }
            }
            if owner.is_some() {
                break;
            }
        }
        owner
    } else {
        None
    };
    let refit_fold_set = nested
        .as_ref()
        .and_then(|nested| nested.refit_fold_set.as_ref());
    let fold_set = match refit_fold_set {
        Some(folds) => folds,
        None => required_fold_set_for_oof(plan, edge)?,
    };
    let blocks = if nested.is_some() {
        let refit_fold_ids = fold_set
            .folds
            .iter()
            .map(|fold| fold.fold_id.clone())
            .collect::<BTreeSet<_>>();
        blocks
            .into_iter()
            .filter(|block| {
                block
                    .fold_id
                    .as_ref()
                    .is_some_and(|fold_id| refit_fold_ids.contains(fold_id))
            })
            .collect::<Vec<_>>()
    } else {
        blocks
    };
    let blocks = filter_prediction_blocks_for_edge_source_port(plan, edge, blocks)?;
    // No validation OOF at all, under the default full-coverage policy, means the CV phase was never
    // run for this producer (e.g. a direct REFIT without a prior FIT_CV). Report it as a missing-OOF
    // edge — matching `validate_fit_cv_oof_edge` and `validate_refit_aggregated_oof_edge`, which both
    // guard `blocks.is_empty()` up front — rather than routing an empty set through the partial-coverage
    // contract validator, which would mislabel "no OOF at all" as `partial_oof_without_policy`. The
    // explicit `cv_only` / `skip_refit_on_incomplete_oof` policies still legitimately skip REFIT with
    // zero OOF, so this guard is scoped to `RequireFullCoverage`.
    if blocks.is_empty() && contract.policy == StackingOofRefitPolicy::RequireFullCoverage {
        return Err(missing_oof_edge_error(edge, None));
    }
    // MANDATORY exact OOF coverage — see `validate_fit_cv_oof_edge`. The branch-merge concat partition
    // exception is handled by the separation-merge handler, which never reaches this stacking path.
    let decision = crate::oof::validate_stacking_oof_refit_contract(
        &edge.source.node_id,
        &blocks,
        fold_set,
        &contract,
    )?;
    match decision {
        StackingOofRefitDecision::RefitAllowed(_) => Ok(Some(blocks)),
        StackingOofRefitDecision::SkipRefit(_) => Ok(None),
    }
}

pub(crate) fn stacking_oof_refit_contract_for_edge(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
) -> Result<StackingOofRefitContract> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == edge.target.node_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "edge `{}.{}` -> `{}.{}` targets unknown node `{}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                edge.target.node_id
            ))
        })?;
    StackingOofRefitContract::from_metadata(&node.metadata).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "node `{}` carries invalid stacking OOF refit contract: {}",
            node.id, error
        ))
    })
}

pub(crate) fn validate_fit_cv_aggregated_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
    scope: &PhaseScope,
    resources: &PhaseScopeResources<'_>,
    prediction_level: PredictionLevel,
) -> Result<Vec<&'a AggregatedPredictionBlock>> {
    let fold_id = scope.fold_id.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` requires aggregated OOF predictions but FIT_CV has no fold scope",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        ))
    })?;
    let blocks = ctx.aggregated_prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        Some(fold_id),
        Some(prediction_level),
    );
    let blocks = filter_aggregated_prediction_blocks_for_edge_source_port(plan, edge, blocks)?;
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, Some(fold_id)));
    }
    validate_aggregated_blocks_basic(edge, prediction_level, &blocks)?;
    // MANDATORY exact aggregated-OOF coverage — see `validate_fit_cv_oof_edge` (audit R-P0-2). The
    // concat-merge partition exception is intercepted by the separation-merge handler upstream.
    let fold_set = required_fold_set_for_oof(plan, edge)?;
    let relations = coordinator_relations_for_edge(plan, edge, resources)?;
    validate_aggregated_oof_blocks_match_fold(
        edge,
        fold_set,
        &relations,
        prediction_level,
        fold_id,
        &blocks,
    )?;
    Ok(blocks)
}

pub(crate) fn validate_refit_aggregated_oof_edge<'a>(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &'a RunContext,
    resources: &PhaseScopeResources<'_>,
    prediction_level: PredictionLevel,
) -> Result<Vec<&'a AggregatedPredictionBlock>> {
    let blocks = ctx.aggregated_prediction_store.find(
        Some(&edge.source.node_id),
        Some(&PredictionPartition::Validation),
        None,
        Some(prediction_level),
    );
    let blocks = filter_aggregated_prediction_blocks_for_edge_source_port(plan, edge, blocks)?;
    if blocks.is_empty() {
        return Err(missing_oof_edge_error(edge, None));
    }
    validate_aggregated_blocks_basic(edge, prediction_level, &blocks)?;
    // MANDATORY exact aggregated-OOF coverage — see `validate_fit_cv_oof_edge` (audit R-P0-2). The
    // concat-merge partition exception is intercepted by the separation-merge handler upstream.
    let fold_set = required_fold_set_for_oof(plan, edge)?;
    let relations = coordinator_relations_for_edge(plan, edge, resources)?;
    validate_aggregated_oof_blocks_cover_fold_set(
        edge,
        fold_set,
        &relations,
        prediction_level,
        &blocks,
    )?;
    Ok(blocks)
}

pub(crate) fn validate_aggregated_blocks_basic(
    edge: &EdgeSpec,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<()> {
    for block in blocks {
        block.validate_shape()?;
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected non-validation aggregated predictions",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if block.level != prediction_level {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected {:?} aggregated predictions, expected {:?}",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                block.level,
                prediction_level
            )));
        }
    }
    Ok(())
}

pub(crate) fn prediction_input_spec(
    edge: &EdgeSpec,
    scope: &PhaseScope,
    blocks: &[&PredictionBlock],
    allow_cross_fold_duplicates: bool,
) -> Result<PredictionInputSpec> {
    let fold_ids = blocks
        .iter()
        .filter_map(|block| block.fold_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    // Validation OOF rows keyed by sample, so the meta-node host can build a
    // stacking feature matrix in FIT_CV/REFIT. Blocks are Validation-only (the
    // leakage guards in `validate_fit_cv_oof_edge` / `validate_refit_oof_edge`
    // already refused any Train partition). Partition fold sets keep one row per
    // sample; Resampled REFIT may average repeated validation rows after the
    // contract validator has accepted that multiplicity.
    let mut rows_by_sample: BTreeMap<&SampleId, Vec<&[f64]>> = BTreeMap::new();
    let mut prediction_width = None;
    let mut target_names = None;
    for block in blocks {
        let width = block.validate_shape()?;
        for (sample_id, row) in block.sample_ids.iter().zip(block.values.iter()) {
            let rows = rows_by_sample.entry(sample_id).or_default();
            if !allow_cross_fold_duplicates && !rows.is_empty() {
                return Err(DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate OOF prediction for sample `{sample_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
            rows.push(row.as_slice());
        }
        let block_target_names = if block.target_names.is_empty() {
            (0..width)
                .map(|index| format!("p{index}"))
                .collect::<Vec<_>>()
        } else {
            block.target_names.clone()
        };
        if prediction_width.is_some_and(|expected| expected != width) {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` OOF prediction width is not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if target_names
            .as_ref()
            .is_some_and(|expected| expected != &block_target_names)
        {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` OOF target names are not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        prediction_width = Some(width);
        target_names = Some(block_target_names);
    }
    let sample_ids = rows_by_sample
        .keys()
        .map(|sample_id| (*sample_id).clone())
        .collect::<Vec<_>>();
    let values = sample_ids
        .iter()
        .map(|sample_id| {
            rows_by_sample
                .get(sample_id)
                .map(|rows| average_prediction_rows(rows, prediction_width.unwrap_or_default()))
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "edge `{}.{}` -> `{}.{}` has no OOF prediction row for sample `{sample_id}`",
                        edge.source.node_id,
                        edge.source.port_name,
                        edge.target.node_id,
                        edge.target.port_name
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PredictionInputSpec {
        producer_node: edge.source.node_id.clone(),
        source_port: edge.source.port_name.clone(),
        target_port: edge.target.port_name.clone(),
        partition: PredictionPartition::Validation,
        prediction_level: PredictionLevel::Sample,
        fold_id: scope.fold_id.clone(),
        fold_ids,
        unit_ids: sample_ids
            .iter()
            .cloned()
            .map(PredictionUnitId::Sample)
            .collect(),
        sample_ids,
        values,
        prediction_width: prediction_width.unwrap_or_default(),
        target_names: target_names.unwrap_or_default(),
    })
}

fn average_prediction_rows(rows: &[&[f64]], width: usize) -> Vec<f64> {
    if rows.len() == 1 {
        return rows[0].to_vec();
    }
    let mut averaged = vec![0.0; width];
    for row in rows {
        for (index, value) in row.iter().enumerate() {
            averaged[index] += value;
        }
    }
    let denominator = rows.len() as f64;
    for value in &mut averaged {
        *value /= denominator;
    }
    averaged
}

pub(crate) fn aggregated_prediction_input_spec(
    edge: &EdgeSpec,
    scope: &PhaseScope,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<PredictionInputSpec> {
    let unit_ids = collect_unique_aggregated_oof_units(edge, prediction_level, blocks)?
        .into_iter()
        .collect::<Vec<_>>();
    let fold_ids = blocks
        .iter()
        .filter_map(|block| block.fold_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut prediction_width = None;
    let mut target_names = None;
    for block in blocks {
        let width = block.validate_shape()?;
        let block_target_names = if block.target_names.is_empty() {
            (0..width)
                .map(|index| format!("p{index}"))
                .collect::<Vec<_>>()
        } else {
            block.target_names.clone()
        };
        if prediction_width.is_some_and(|expected| expected != width) {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` aggregated OOF prediction width is not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if target_names
            .as_ref()
            .is_some_and(|expected| expected != &block_target_names)
        {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` aggregated OOF target names are not stable across folds",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        prediction_width = Some(width);
        target_names = Some(block_target_names);
    }
    Ok(PredictionInputSpec {
        producer_node: edge.source.node_id.clone(),
        source_port: edge.source.port_name.clone(),
        target_port: edge.target.port_name.clone(),
        partition: PredictionPartition::Validation,
        prediction_level,
        fold_id: scope.fold_id.clone(),
        fold_ids,
        unit_ids,
        sample_ids: Vec::new(),
        // Aggregated (unit-level) OOF crosses as opaque handle, not per-sample rows.
        values: Vec::new(),
        prediction_width: prediction_width.unwrap_or_default(),
        target_names: target_names.unwrap_or_default(),
    })
}

pub(crate) fn prediction_input_spec_from_requirement(
    requirement: &BundlePredictionRequirement,
    scope: &PhaseScope,
) -> Result<PredictionInputSpec> {
    requirement.validate()?;
    Ok(PredictionInputSpec {
        producer_node: requirement.producer_node.clone(),
        source_port: requirement.source_port.clone(),
        target_port: requirement.target_port.clone(),
        partition: requirement.partition.clone(),
        prediction_level: requirement.prediction_level,
        fold_id: scope.fold_id.clone(),
        fold_ids: requirement.fold_ids.clone(),
        unit_ids: requirement.unit_ids.clone(),
        sample_ids: requirement.sample_ids.clone(),
        // Replay-cache requirement: OOF rows are materialized by the host via the
        // prediction-cache handle, not carried inline in the spec.
        values: Vec::new(),
        prediction_width: requirement.prediction_width,
        target_names: requirement.target_names.clone(),
    })
}

pub(crate) fn missing_oof_edge_error(edge: &EdgeSpec, fold_id: Option<&FoldId>) -> DagMlError {
    DagMlError::OofValidation(format!(
        "edge `{}.{}` -> `{}.{}` requires OOF validation predictions from `{}`{}",
        edge.source.node_id,
        edge.source.port_name,
        edge.target.node_id,
        edge.target.port_name,
        edge.source.node_id,
        fold_id
            .map(|fold_id| format!(" for fold `{fold_id}`"))
            .unwrap_or_default()
    ))
}

/// The OOF [`FoldPartitionMode`] for a plan: its fold set's mode, or `Partition` (the clean-OOF
/// default) when the plan carries no fold set. Used to make the cross-fold scoring gate mode-aware so
/// `Resampled` (ShuffleSplit / repeated CV) campaigns, where a sample is validated in several folds,
/// are not rejected by the `Partition` exactly-once uniqueness rule.
pub fn plan_oof_partition_mode(plan: &ExecutionPlan) -> FoldPartitionMode {
    plan.fold_set
        .as_ref()
        .map(|fold_set| fold_set.partition_mode)
        .unwrap_or_default()
}

pub(crate) fn required_fold_set_for_oof<'a>(
    plan: &'a ExecutionPlan,
    edge: &EdgeSpec,
) -> Result<&'a FoldSet> {
    plan.fold_set.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "edge `{}.{}` -> `{}.{}` requires fold-aligned OOF predictions but the plan has no fold set",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name
        ))
    })
}

pub(crate) fn validate_oof_blocks_match_fold(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    fold_id: &FoldId,
    blocks: &[&PredictionBlock],
) -> Result<()> {
    let fold = fold_set
        .folds
        .iter()
        .find(|fold| &fold.fold_id == fold_id)
        .ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
    let actual = collect_unique_oof_samples(edge, blocks)?;
    let expected = fold
        .validation_sample_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(DagMlError::OofValidation(format!(
            "edge `{}.{}` -> `{}.{}` OOF predictions do not match validation samples for fold `{fold_id}`",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_oof_blocks_cover_fold_set(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    blocks: &[&PredictionBlock],
) -> Result<()> {
    let folds = fold_set
        .folds
        .iter()
        .map(|fold| (&fold.fold_id, fold))
        .collect::<BTreeMap<_, _>>();
    let mut all_samples = BTreeSet::new();
    for block in blocks {
        let fold_id = block.fold_id.as_ref().ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` has OOF predictions without a fold id",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
        let fold = folds.get(fold_id).ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
        let block_samples = collect_unique_oof_samples(edge, &[*block])?;
        let expected = fold
            .validation_sample_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if block_samples != expected {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` OOF predictions do not match validation samples for fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        for sample_id in block_samples {
            // Partition is a clean OOF set: a sample covered by two folds is a duplicated fold or a
            // mixed-variant context. Resampled (ShuffleSplit / repeated CV) legitimately validates a
            // sample in several folds and averages its predictions, so the across-fold duplicate is
            // expected; the per-fold match above + per-block uniqueness (`collect_unique_oof_samples`)
            // still hold, and the universe-coverage check below still requires every sample at least
            // once.
            if !all_samples.insert(sample_id.clone())
                && fold_set.partition_mode == FoldPartitionMode::Partition
            {
                return Err(DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate OOF prediction for sample `{sample_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }
    let expected_all = fold_set.sample_ids.iter().cloned().collect::<BTreeSet<_>>();
    if all_samples != expected_all {
        return Err(DagMlError::OofValidation(format!(
            "edge `{}.{}` -> `{}.{}` OOF predictions do not cover the refit sample universe",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        )));
    }
    Ok(())
}

pub(crate) fn validate_aggregated_oof_blocks_match_fold(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    fold_id: &FoldId,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<()> {
    let fold = fold_set
        .folds
        .iter()
        .find(|fold| &fold.fold_id == fold_id)
        .ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
    validate_aggregated_fold_unit_safety(edge, relations, prediction_level, fold)?;
    for block in blocks {
        if block.fold_id.as_ref() != Some(fold_id) {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected aggregated OOF predictions outside fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
    }
    let actual = collect_unique_aggregated_oof_units(edge, prediction_level, blocks)?;
    let expected = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold.validation_sample_ids,
    )?;
    if actual != expected {
        return Err(DagMlError::OofValidation(format!(
            "edge `{}.{}` -> `{}.{}` aggregated OOF predictions do not match {:?} validation units for fold `{fold_id}`",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name,
            prediction_level
        )));
    }
    Ok(())
}

pub(crate) fn validate_aggregated_oof_blocks_cover_fold_set(
    edge: &EdgeSpec,
    fold_set: &FoldSet,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<()> {
    let folds = fold_set
        .folds
        .iter()
        .map(|fold| (fold.fold_id.clone(), fold))
        .collect::<BTreeMap<_, _>>();
    let mut blocks_by_fold = BTreeMap::<FoldId, Vec<&AggregatedPredictionBlock>>::new();
    for block in blocks {
        let fold_id = block.fold_id.as_ref().ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` has aggregated OOF predictions without a fold id",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            ))
        })?;
        if !folds.contains_key(fold_id) {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` references unknown fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        blocks_by_fold
            .entry(fold_id.clone())
            .or_default()
            .push(*block);
    }
    for fold_id in folds.keys() {
        if !blocks_by_fold.contains_key(fold_id) {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` is missing aggregated OOF predictions for fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
    }

    let mut all_units = BTreeSet::new();
    for (fold_id, fold_blocks) in blocks_by_fold {
        let fold = folds.get(&fold_id).expect("fold id was validated above");
        validate_aggregated_fold_unit_safety(edge, relations, prediction_level, fold)?;
        let fold_units = collect_unique_aggregated_oof_units(edge, prediction_level, &fold_blocks)?;
        let expected = expected_prediction_units_for_samples(
            edge,
            relations,
            prediction_level,
            &fold.validation_sample_ids,
        )?;
        if fold_units != expected {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` aggregated OOF predictions do not match {:?} validation units for fold `{fold_id}`",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                prediction_level
            )));
        }
        for unit_id in fold_units {
            // See `validate_oof_blocks_cover_fold_set`: Partition forbids a unit covered by two folds;
            // Resampled (ShuffleSplit / repeated CV) validates a unit in several folds and averages it,
            // so the across-fold duplicate is allowed while the universe-coverage check below still
            // requires every unit at least once.
            if !all_units.insert(unit_id.clone())
                && fold_set.partition_mode == FoldPartitionMode::Partition
            {
                return Err(DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate aggregated OOF prediction for unit `{unit_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }

    let expected_all = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold_set.sample_ids,
    )?;
    if all_units != expected_all {
        return Err(DagMlError::OofValidation(format!(
            "edge `{}.{}` -> `{}.{}` aggregated OOF predictions do not cover the refit {:?} unit universe",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name,
            prediction_level
        )));
    }
    Ok(())
}

pub(crate) fn validate_aggregated_fold_unit_safety(
    edge: &EdgeSpec,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    fold: &FoldAssignment,
) -> Result<()> {
    let train_units = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold.train_sample_ids,
    )?;
    let validation_units = expected_prediction_units_for_samples(
        edge,
        relations,
        prediction_level,
        &fold.validation_sample_ids,
    )?;
    if let Some(unit_id) = train_units.intersection(&validation_units).next() {
        return Err(DagMlError::OofValidation(format!(
            "edge `{}.{}` -> `{}.{}` fold `{}` has {:?} unit `{unit_id}` in both train and validation partitions",
            edge.source.node_id,
            edge.source.port_name,
            edge.target.node_id,
            edge.target.port_name,
            fold.fold_id,
            prediction_level
        )));
    }
    Ok(())
}

pub(crate) fn collect_unique_oof_samples(
    edge: &EdgeSpec,
    blocks: &[&PredictionBlock],
) -> Result<BTreeSet<SampleId>> {
    let mut samples = BTreeSet::new();
    for block in blocks {
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected non-validation predictions",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        for sample_id in &block.sample_ids {
            if !samples.insert(sample_id.clone()) {
                return Err(DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate OOF prediction for sample `{sample_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }
    Ok(samples)
}

pub(crate) fn collect_unique_aggregated_oof_units(
    edge: &EdgeSpec,
    prediction_level: PredictionLevel,
    blocks: &[&AggregatedPredictionBlock],
) -> Result<BTreeSet<PredictionUnitId>> {
    let mut unit_ids = BTreeSet::new();
    for block in blocks {
        block.validate_shape()?;
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected non-validation aggregated predictions",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name
            )));
        }
        if block.level != prediction_level {
            return Err(DagMlError::OofValidation(format!(
                "edge `{}.{}` -> `{}.{}` selected {:?} aggregated predictions, expected {:?}",
                edge.source.node_id,
                edge.source.port_name,
                edge.target.node_id,
                edge.target.port_name,
                block.level,
                prediction_level
            )));
        }
        for unit_id in &block.unit_ids {
            if !unit_ids.insert(unit_id.clone()) {
                return Err(DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has duplicate aggregated OOF prediction for unit `{unit_id}`",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }
        }
    }
    Ok(unit_ids)
}

pub(crate) fn expected_prediction_units_for_samples(
    edge: &EdgeSpec,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    sample_ids: &[SampleId],
) -> Result<BTreeSet<PredictionUnitId>> {
    sample_ids
        .iter()
        .map(|sample_id| prediction_unit_for_sample(edge, relations, prediction_level, sample_id))
        .collect()
}

pub(crate) fn prediction_unit_for_sample(
    edge: &EdgeSpec,
    relations: &SampleRelationSet,
    prediction_level: PredictionLevel,
    sample_id: &SampleId,
) -> Result<PredictionUnitId> {
    match prediction_level {
        PredictionLevel::Sample => Ok(PredictionUnitId::Sample(sample_id.clone())),
        PredictionLevel::Target => relations
            .target_for_sample(sample_id)
            .cloned()
            .map(PredictionUnitId::Target)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` needs target-level OOF predictions but sample `{sample_id}` has no target relation",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                ))
            }),
        PredictionLevel::Group => relations
            .group_for_sample(sample_id)
            .cloned()
            .map(PredictionUnitId::Group)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "edge `{}.{}` -> `{}.{}` needs group-level OOF predictions but sample `{sample_id}` has no group relation",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                ))
            }),
        PredictionLevel::Observation => Err(DagMlError::OofValidation(format!(
            "edge `{}.{}` -> `{}.{}` cannot consume observation-level OOF predictions from sample folds",
            edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name
        ))),
    }
}

pub(crate) fn deterministic_oof_handle(
    plan: &ExecutionPlan,
    edge: &EdgeSpec,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<u64> {
    let fingerprint = stable_json_fingerprint(&(
        &plan.id,
        &ctx.run_id,
        &edge.source.node_id,
        &edge.source.port_name,
        &edge.target.node_id,
        &edge.target.port_name,
        scope.phase,
        &scope.variant_id,
        &scope.fold_id,
    ))?;
    Ok(u64::from_str_radix(&fingerprint[..16], 16).expect("sha256 hex prefix should fit into u64"))
}
