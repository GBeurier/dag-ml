use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::aggregation::{
    reduce_predictions_across_folds, AggregatedPredictionBlock, PredictionUnitId,
};
use crate::error::{DagMlError, Result};
use crate::ids::{FoldId, NodeId, VariantId};
use crate::oof::{validate_producer_oof_coverage, PredictionBlock, PredictionPartition};
use crate::policy::PredictionLevel;
use crate::selection::{CandidateScore, MetricObjective};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionMetricKind {
    Mse,
    Rmse,
    Mae,
    R2,
    /// Classification accuracy: fraction of predictions whose label matches the target (integer
    /// label encoding, matched within 0.5). Meaningless on continuous regression targets (≈0) but
    /// always emitted so the host can score classification natively without a separate code path.
    Accuracy,
}

impl RegressionMetricKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::Rmse => "rmse",
            Self::Mae => "mae",
            Self::R2 => "r2",
            Self::Accuracy => "accuracy",
        }
    }

    pub fn objective(self) -> MetricObjective {
        match self {
            Self::Mse | Self::Rmse | Self::Mae => MetricObjective::Minimize,
            Self::R2 | Self::Accuracy => MetricObjective::Maximize,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionTargetBlock {
    pub level: PredictionLevel,
    pub unit_ids: Vec<PredictionUnitId>,
    pub values: Vec<Vec<f64>>,
    #[serde(default)]
    pub target_names: Vec<String>,
}

impl RegressionTargetBlock {
    pub fn validate_shape(&self) -> Result<usize> {
        if self.unit_ids.len() != self.values.len() {
            return Err(DagMlError::OofValidation(format!(
                "target block has {} unit ids but {} target rows",
                self.unit_ids.len(),
                self.values.len()
            )));
        }
        if self
            .unit_ids
            .iter()
            .any(|unit_id| unit_id.level() != self.level)
        {
            return Err(DagMlError::OofValidation(format!(
                "target block contains units outside level {:?}",
                self.level
            )));
        }
        let unique = self.unit_ids.iter().collect::<BTreeSet<_>>();
        if unique.len() != self.unit_ids.len() {
            return Err(DagMlError::OofValidation(
                "target block contains duplicate unit ids".to_string(),
            ));
        }
        let width = self.values.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(DagMlError::OofValidation(
                "target block has empty target rows".to_string(),
            ));
        }
        if self.values.iter().any(|row| row.len() != width) {
            return Err(DagMlError::OofValidation(
                "target block has ragged target rows".to_string(),
            ));
        }
        if self.values.iter().flatten().any(|value| !value.is_finite()) {
            return Err(DagMlError::OofValidation(
                "target block contains non-finite values".to_string(),
            ));
        }
        if !self.target_names.is_empty() && self.target_names.len() != width {
            return Err(DagMlError::OofValidation(format!(
                "target block has {} target names for width {}",
                self.target_names.len(),
                width
            )));
        }
        Ok(width)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionMetricReport {
    #[serde(default)]
    pub prediction_id: Option<String>,
    pub producer_node: NodeId,
    /// Variant this score belongs to — distinguishes per-variant candidates when a generated
    /// campaign scores several variants, so native SELECT can pick the best. Skipped (None) for
    /// single-variant runs, so existing fixtures/fingerprints are byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant_id: Option<VariantId>,
    pub partition: PredictionPartition,
    pub fold_id: Option<FoldId>,
    pub level: PredictionLevel,
    pub row_count: usize,
    pub target_width: usize,
    #[serde(default)]
    pub target_names: Vec<String>,
    pub metrics: BTreeMap<String, f64>,
}

impl RegressionMetricReport {
    pub fn validate(&self) -> Result<()> {
        if self.row_count == 0 {
            return Err(DagMlError::OofValidation(
                "regression metric report has zero rows".to_string(),
            ));
        }
        if self.target_width == 0 {
            return Err(DagMlError::OofValidation(
                "regression metric report has zero target width".to_string(),
            ));
        }
        if !self.target_names.is_empty() && self.target_names.len() != self.target_width {
            return Err(DagMlError::OofValidation(format!(
                "regression metric report has {} target names for width {}",
                self.target_names.len(),
                self.target_width
            )));
        }
        if self.metrics.is_empty() {
            return Err(DagMlError::OofValidation(
                "regression metric report has no metrics".to_string(),
            ));
        }
        for (name, value) in &self.metrics {
            if name.trim().is_empty() {
                return Err(DagMlError::OofValidation(
                    "regression metric report contains an empty metric name".to_string(),
                ));
            }
            if !value.is_finite() {
                return Err(DagMlError::OofValidation(format!(
                    "regression metric `{name}` is not finite"
                )));
            }
        }
        Ok(())
    }

    pub fn into_candidate_score(self, candidate_id: impl Into<String>) -> Result<CandidateScore> {
        self.validate()?;
        let mut metadata = BTreeMap::from([
            (
                "producer_node".to_string(),
                serde_json::json!(self.producer_node),
            ),
            ("partition".to_string(), serde_json::json!(self.partition)),
            (
                "metric_level".to_string(),
                serde_json::json!(prediction_level_name(self.level)),
            ),
            ("row_count".to_string(), serde_json::json!(self.row_count)),
            (
                "target_width".to_string(),
                serde_json::json!(self.target_width),
            ),
        ]);
        if let Some(prediction_id) = self.prediction_id {
            metadata.insert(
                "prediction_id".to_string(),
                serde_json::json!(prediction_id),
            );
        }
        if let Some(fold_id) = self.fold_id {
            metadata.insert("fold_id".to_string(), serde_json::json!(fold_id));
        }
        if let Some(variant_id) = self.variant_id {
            metadata.insert("variant_id".to_string(), serde_json::json!(variant_id));
        }
        if !self.target_names.is_empty() {
            metadata.insert(
                "target_names".to_string(),
                serde_json::json!(self.target_names),
            );
        }
        let score = CandidateScore {
            candidate_id: candidate_id.into(),
            metrics: self.metrics,
            metadata,
        };
        score.validate()?;
        Ok(score)
    }
}

pub fn regression_report_to_candidate_score(
    candidate_id: impl Into<String>,
    report: RegressionMetricReport,
) -> Result<CandidateScore> {
    report.into_candidate_score(candidate_id)
}

pub fn score_regression_prediction_block(
    predictions: &PredictionBlock,
    targets: &RegressionTargetBlock,
    metrics: &[RegressionMetricKind],
) -> Result<RegressionMetricReport> {
    let width = validate_sample_prediction_block(predictions)?;
    let prediction_units = predictions
        .sample_ids
        .iter()
        .cloned()
        .map(PredictionUnitId::Sample)
        .collect::<Vec<_>>();
    score_regression_rows(
        PredictionRows {
            level: PredictionLevel::Sample,
            unit_ids: &prediction_units,
            values: &predictions.values,
            target_names: &predictions.target_names,
            width,
            origin: PredictionReportOrigin {
                prediction_id: predictions.prediction_id.clone(),
                producer_node: predictions.producer_node.clone(),
                partition: predictions.partition.clone(),
                fold_id: predictions.fold_id.clone(),
            },
        },
        targets,
        metrics,
    )
}

pub fn score_regression_aggregated_block(
    predictions: &AggregatedPredictionBlock,
    targets: &RegressionTargetBlock,
    metrics: &[RegressionMetricKind],
) -> Result<RegressionMetricReport> {
    let width = predictions.validate_shape()?;
    score_regression_rows(
        PredictionRows {
            level: predictions.level,
            unit_ids: &predictions.unit_ids,
            values: &predictions.values,
            target_names: &predictions.target_names,
            width,
            origin: PredictionReportOrigin {
                prediction_id: predictions.prediction_id.clone(),
                producer_node: predictions.producer_node.clone(),
                partition: predictions.partition.clone(),
                fold_id: predictions.fold_id.clone(),
            },
        },
        targets,
        metrics,
    )
}

/// Current on-disk schema version for [`ScoreSet`].
pub const SCORE_SET_SCHEMA_VERSION: u32 = 1;

fn default_score_set_schema_version() -> u32 {
    SCORE_SET_SCHEMA_VERSION
}

/// A persisted collection of per-block regression metric reports — the native, cross-language
/// score record produced by a run (one report per `(producer_node, partition, fold_id, level)`).
///
/// Unlike the prediction-*value* cache (which is Validation-only and leakage-gated), scores are
/// scalars derived from `y_true` and carry no feature data, so they are safe to persist for
/// every partition (train / validation / test / final). This is the score half of "dag-ml owns
/// prediction/score persistence natively" — the Python (or any host) `RunResult` reads these
/// scalars by identity, with no recomputation.
/// Identity of a score report within a [`ScoreSet`] — unique per variant, partition, fold and level.
type ScoreReportKey = (
    NodeId,
    Option<VariantId>,
    PredictionPartition,
    Option<FoldId>,
    PredictionLevel,
);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreSet {
    #[serde(default = "default_score_set_schema_version")]
    pub schema_version: u32,
    pub plan_id: String,
    /// The metric SELECT optimized for this run (e.g. `"rmse"`), if a selection ran. Metadata only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_metric: Option<String>,
    pub reports: Vec<RegressionMetricReport>,
}

impl ScoreSet {
    /// Validate the version, plan id, every report, and report-key uniqueness.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version == 0 || self.schema_version > SCORE_SET_SCHEMA_VERSION {
            return Err(DagMlError::OofValidation(format!(
                "score set schema version {} is unsupported (current {SCORE_SET_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.plan_id.trim().is_empty() {
            return Err(DagMlError::OofValidation(
                "score set has an empty plan_id".to_string(),
            ));
        }
        let mut seen: BTreeSet<ScoreReportKey> = BTreeSet::new();
        for report in &self.reports {
            report.validate()?;
            let key = (
                report.producer_node.clone(),
                report.variant_id.clone(),
                report.partition.clone(),
                report.fold_id.clone(),
                report.level,
            );
            if !seen.insert(key) {
                return Err(DagMlError::OofValidation(format!(
                    "score set has a duplicate report for node `{}` partition {:?} fold {:?} level {:?}",
                    report.producer_node, report.partition, report.fold_id, report.level
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct PredictionReportOrigin {
    prediction_id: Option<String>,
    producer_node: NodeId,
    partition: PredictionPartition,
    fold_id: Option<FoldId>,
}

#[derive(Clone, Debug)]
struct PredictionRows<'a> {
    level: PredictionLevel,
    unit_ids: &'a [PredictionUnitId],
    values: &'a [Vec<f64>],
    target_names: &'a [String],
    width: usize,
    origin: PredictionReportOrigin,
}

fn score_regression_rows(
    predictions: PredictionRows<'_>,
    targets: &RegressionTargetBlock,
    metrics: &[RegressionMetricKind],
) -> Result<RegressionMetricReport> {
    if metrics.is_empty() {
        return Err(DagMlError::OofValidation(
            "no regression metrics requested".to_string(),
        ));
    }
    let mut requested_metrics = BTreeSet::new();
    for metric in metrics {
        if !requested_metrics.insert(*metric) {
            return Err(DagMlError::OofValidation(format!(
                "duplicate regression metric `{}` requested",
                metric.name()
            )));
        }
    }

    let target_width = targets.validate_shape()?;
    if predictions.width != target_width {
        return Err(DagMlError::OofValidation(format!(
            "prediction width {} does not match target width {target_width}",
            predictions.width
        )));
    }
    if predictions.level != targets.level {
        return Err(DagMlError::OofValidation(format!(
            "prediction level {:?} does not match target level {:?}",
            predictions.level, targets.level
        )));
    }
    if !predictions.target_names.is_empty()
        && !targets.target_names.is_empty()
        && predictions.target_names != targets.target_names
    {
        return Err(DagMlError::OofValidation(
            "prediction target names do not match target block names".to_string(),
        ));
    }

    let target_by_unit = targets
        .unit_ids
        .iter()
        .zip(targets.values.iter().map(Vec::as_slice))
        .collect::<BTreeMap<_, _>>();
    let mut aligned_predictions = Vec::with_capacity(predictions.unit_ids.len());
    let mut aligned_targets = Vec::with_capacity(predictions.unit_ids.len());
    for (unit_id, prediction_row) in predictions.unit_ids.iter().zip(predictions.values.iter()) {
        let target_row = target_by_unit.get(unit_id).ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "prediction unit `{unit_id}` is missing from target block"
            ))
        })?;
        aligned_predictions.push(prediction_row.as_slice());
        aligned_targets.push(*target_row);
    }
    if aligned_predictions.len() != target_by_unit.len() {
        return Err(DagMlError::OofValidation(
            "target block contains units not present in predictions".to_string(),
        ));
    }

    let target_names = if !predictions.target_names.is_empty() {
        predictions.target_names.to_vec()
    } else {
        targets.target_names.clone()
    };
    let metric_suffixes = target_metric_names(predictions.width, &target_names);
    let mut values = BTreeMap::new();
    for metric in metrics {
        let per_target = compute_metric_per_target(
            *metric,
            predictions.width,
            &aligned_predictions,
            &aligned_targets,
        );
        values.insert(metric.name().to_string(), macro_mean(&per_target));
        for (name, value) in metric_suffixes.iter().zip(per_target) {
            values.insert(format!("{}:{name}", metric.name()), value);
        }
    }

    let report = RegressionMetricReport {
        prediction_id: predictions.origin.prediction_id,
        producer_node: predictions.origin.producer_node,
        variant_id: None,
        partition: predictions.origin.partition,
        fold_id: predictions.origin.fold_id,
        level: predictions.level,
        row_count: predictions.unit_ids.len(),
        target_width: predictions.width,
        target_names,
        metrics: values,
    };
    report.validate()?;
    Ok(report)
}

fn validate_sample_prediction_block(block: &PredictionBlock) -> Result<usize> {
    block.validate_content()
}

fn compute_metric_per_target(
    metric: RegressionMetricKind,
    width: usize,
    predictions: &[&[f64]],
    targets: &[&[f64]],
) -> Vec<f64> {
    (0..width)
        .map(|target_idx| match metric {
            RegressionMetricKind::Mse => {
                predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(prediction, target)| {
                        let error = prediction[target_idx] - target[target_idx];
                        error * error
                    })
                    .sum::<f64>()
                    / predictions.len() as f64
            }
            RegressionMetricKind::Rmse => (predictions
                .iter()
                .zip(targets.iter())
                .map(|(prediction, target)| {
                    let error = prediction[target_idx] - target[target_idx];
                    error * error
                })
                .sum::<f64>()
                / predictions.len() as f64)
                .sqrt(),
            RegressionMetricKind::Mae => {
                predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(prediction, target)| (prediction[target_idx] - target[target_idx]).abs())
                    .sum::<f64>()
                    / predictions.len() as f64
            }
            RegressionMetricKind::R2 => r2_for_target(target_idx, predictions, targets),
            RegressionMetricKind::Accuracy => {
                predictions
                    .iter()
                    .zip(targets.iter())
                    .filter(|(prediction, target)| {
                        (prediction[target_idx] - target[target_idx]).abs() < 0.5
                    })
                    .count() as f64
                    / predictions.len() as f64
            }
        })
        .collect()
}

fn r2_for_target(target_idx: usize, predictions: &[&[f64]], targets: &[&[f64]]) -> f64 {
    let mean = targets.iter().map(|row| row[target_idx]).sum::<f64>() / targets.len() as f64;
    let ss_res = predictions
        .iter()
        .zip(targets.iter())
        .map(|(prediction, target)| {
            let error = prediction[target_idx] - target[target_idx];
            error * error
        })
        .sum::<f64>();
    let ss_tot = targets
        .iter()
        .map(|target| {
            let centered = target[target_idx] - mean;
            centered * centered
        })
        .sum::<f64>();
    if ss_tot == 0.0 {
        if ss_res == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

fn macro_mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn target_metric_names(width: usize, target_names: &[String]) -> Vec<String> {
    if target_names.is_empty() {
        (0..width).map(|idx| format!("target_{idx}")).collect()
    } else {
        target_names.to_vec()
    }
}

fn prediction_level_name(level: PredictionLevel) -> &'static str {
    match level {
        PredictionLevel::Observation => "observation",
        PredictionLevel::Sample => "sample",
        PredictionLevel::Target => "target",
        PredictionLevel::Group => "group",
    }
}

/// A host-emitted `y_true` block tagged with the prediction it scores (producer/partition/fold), so
/// the runtime can aggregate ground truth across folds to score cross-fold ensembles natively.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionTargetRecord {
    pub producer_node: NodeId,
    /// Variant that produced the scored block — lets the cross-fold OOF average be computed
    /// per-variant (for native SELECT) without tagging every PredictionBlock with a variant.
    pub variant_id: Option<VariantId>,
    pub partition: PredictionPartition,
    pub fold_id: Option<FoldId>,
    pub block: RegressionTargetBlock,
}

/// Combine a producer's per-fold VALIDATION `y_true` into one block (dedup by unit id — a sample's
/// ground truth is fold-independent), aligned to the producer's OOF samples.
///
/// Defense-in-depth (audit R-P0-1): records are grouped only by `producer_node`, but each carries a
/// `variant_id`. A sample's ground truth is variant-independent, so the same unit seen again must
/// carry the SAME `y_true`. If two records (e.g. from two variants sharing one context) disagree on a
/// unit's target, the ground truth has been mixed and the combined block would silently score against
/// a corrupted reference — that is refused rather than keeping whichever value happened to be first.
fn combine_validation_targets(
    producer: &NodeId,
    records: &[RegressionTargetRecord],
) -> Result<RegressionTargetBlock> {
    let mut seen: BTreeMap<PredictionUnitId, Vec<f64>> = BTreeMap::new();
    let mut unit_ids = Vec::new();
    let mut values = Vec::new();
    let mut target_names = Vec::new();
    for record in records {
        if &record.producer_node != producer || record.partition != PredictionPartition::Validation
        {
            continue;
        }
        if target_names.is_empty() {
            target_names = record.block.target_names.clone();
        }
        for (unit_id, row) in record.block.unit_ids.iter().zip(&record.block.values) {
            match seen.get(unit_id) {
                None => {
                    seen.insert(unit_id.clone(), row.clone());
                    unit_ids.push(unit_id.clone());
                    values.push(row.clone());
                }
                Some(existing) if existing != row => {
                    return Err(DagMlError::OofValidation(format!(
                        "producer `{producer}` has conflicting ground truth for unit `{unit_id:?}` across validation records — the y_true reference is mixed (e.g. several variants in one context); refusing to score against a corrupted reference"
                    )));
                }
                Some(_) => {}
            }
        }
    }
    Ok(RegressionTargetBlock {
        level: PredictionLevel::Sample,
        unit_ids,
        values,
        target_names,
    })
}

/// Score the cross-fold OOF average per producer: concatenate each producer's per-fold VALIDATION
/// predictions (disjoint OOF samples) into one block and score it against the combined `y_true`.
/// Yields one report per producer with `fold_id = "avg"` — nirs4all's `cv_best_score` row. The
/// per-fold join is identity-keyed; producers with a single fold are skipped (nothing to ensemble).
pub fn cross_fold_validation_reports(
    prediction_blocks: &[PredictionBlock],
    target_records: &[RegressionTargetRecord],
    metrics: &[RegressionMetricKind],
) -> Result<Vec<RegressionMetricReport>> {
    let mut producers: Vec<NodeId> = Vec::new();
    let mut by_producer: BTreeMap<NodeId, Vec<PredictionBlock>> = BTreeMap::new();
    for block in prediction_blocks {
        if block.partition != PredictionPartition::Validation {
            continue;
        }
        if !by_producer.contains_key(&block.producer_node) {
            producers.push(block.producer_node.clone());
        }
        by_producer
            .entry(block.producer_node.clone())
            .or_default()
            .push(block.clone());
    }
    let mut reports = Vec::new();
    for producer in &producers {
        let blocks = &by_producer[producer];
        if blocks.len() < 2 {
            continue;
        }
        // Mandatory OOF coverage gate (spec rule 3): the producer's per-fold validation blocks must
        // be UNIQUE — exactly one validation prediction per sample. A sample appearing in two of this
        // producer's blocks would otherwise be silently averaged twice by `reduce_predictions_across_folds`,
        // mixing a duplicated fold or — since `PredictionBlock` carries no variant tag — two variants'
        // OOF in a shared context (audit R-P0-1). This is the scoring-path analogue of the runtime merge
        // handler's "mixes several variants" guard, so cross-variant CV scores can NEVER mix here.
        let block_refs = blocks.iter().collect::<Vec<_>>();
        validate_producer_oof_coverage(producer, &block_refs, None)?;
        let targets = combine_validation_targets(producer, target_records)?;
        if targets.unit_ids.is_empty() {
            // No y_true was emitted for this producer (e.g. mock controllers) — nothing to score.
            continue;
        }
        let average = reduce_predictions_across_folds(blocks, None, "avg")?;
        reports.push(score_regression_prediction_block(
            &average, &targets, metrics,
        )?);
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{FoldId, GroupId, NodeId, SampleId, TargetId};
    use crate::oof::PredictionPartition;

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn sample_unit(value: &str) -> PredictionUnitId {
        PredictionUnitId::Sample(sid(value))
    }

    fn target_unit(value: &str) -> PredictionUnitId {
        PredictionUnitId::Target(TargetId::new(value).unwrap())
    }

    fn group_unit(value: &str) -> PredictionUnitId {
        PredictionUnitId::Group(GroupId::new(value).unwrap())
    }

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "expected {right}, got {left}");
    }

    #[test]
    fn metric_objectives_match_selection_direction() {
        assert_eq!(
            RegressionMetricKind::Rmse.objective(),
            MetricObjective::Minimize
        );
        assert_eq!(
            RegressionMetricKind::Mae.objective(),
            MetricObjective::Minimize
        );
        assert_eq!(
            RegressionMetricKind::Mse.objective(),
            MetricObjective::Minimize
        );
        assert_eq!(
            RegressionMetricKind::R2.objective(),
            MetricObjective::Maximize
        );
    }

    #[test]
    fn scores_sample_predictions_and_exports_candidate_metrics() {
        let predictions = PredictionBlock {
            prediction_id: Some("pred:sample".to_string()),
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            sample_ids: vec![sid("sample:1"), sid("sample:2")],
            values: vec![vec![2.0], vec![4.0]],
            target_names: vec!["y".to_string()],
        };
        let targets = RegressionTargetBlock {
            level: PredictionLevel::Sample,
            unit_ids: vec![sample_unit("sample:2"), sample_unit("sample:1")],
            values: vec![vec![5.0], vec![1.0]],
            target_names: vec!["y".to_string()],
        };

        let report = score_regression_prediction_block(
            &predictions,
            &targets,
            &[
                RegressionMetricKind::Rmse,
                RegressionMetricKind::Mae,
                RegressionMetricKind::R2,
            ],
        )
        .unwrap();

        assert_eq!(report.level, PredictionLevel::Sample);
        assert_close(report.metrics["rmse"], 1.0);
        assert_close(report.metrics["rmse:y"], 1.0);
        assert_close(report.metrics["mae"], 1.0);
        assert_close(report.metrics["r2"], 0.75);
        let candidate = regression_report_to_candidate_score("model:pls", report).unwrap();
        assert_eq!(candidate.metrics["rmse"], 1.0);
        assert_eq!(candidate.metadata["metric_level"], "sample");
        assert_eq!(candidate.metadata["producer_node"], "model:pls");
        assert_eq!(candidate.metadata["partition"], "validation");
        assert_eq!(candidate.metadata["prediction_id"], "pred:sample");
        assert_eq!(candidate.metadata["target_names"], serde_json::json!(["y"]));
    }

    #[test]
    fn scores_target_and_group_prediction_blocks() {
        let predictions = AggregatedPredictionBlock {
            prediction_id: Some("pred:target".to_string()),
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            level: PredictionLevel::Target,
            unit_ids: vec![target_unit("target:a"), target_unit("target:b")],
            values: vec![vec![1.0, 10.0], vec![3.0, 30.0]],
            target_names: vec!["y1".to_string(), "y2".to_string()],
        };
        let targets = RegressionTargetBlock {
            level: PredictionLevel::Target,
            unit_ids: vec![target_unit("target:b"), target_unit("target:a")],
            values: vec![vec![2.0, 28.0], vec![2.0, 12.0]],
            target_names: vec!["y1".to_string(), "y2".to_string()],
        };
        let report = score_regression_aggregated_block(
            &predictions,
            &targets,
            &[RegressionMetricKind::Mse, RegressionMetricKind::Rmse],
        )
        .unwrap();

        assert_eq!(report.level, PredictionLevel::Target);
        assert_close(report.metrics["mse:y1"], 1.0);
        assert_close(report.metrics["mse:y2"], 4.0);
        assert_close(report.metrics["mse"], 2.5);
        assert_close(report.metrics["rmse:y1"], 1.0);
        assert_close(report.metrics["rmse:y2"], 2.0);
        assert_close(report.metrics["rmse"], 1.5);

        let group_predictions = AggregatedPredictionBlock {
            prediction_id: Some("pred:group".to_string()),
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            level: PredictionLevel::Group,
            unit_ids: vec![group_unit("group:a")],
            values: vec![vec![3.0]],
            target_names: vec!["y".to_string()],
        };
        let group_targets = RegressionTargetBlock {
            level: PredictionLevel::Group,
            unit_ids: vec![group_unit("group:a")],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        };
        let group_report = score_regression_aggregated_block(
            &group_predictions,
            &group_targets,
            &[RegressionMetricKind::Mae],
        )
        .unwrap();
        assert_eq!(group_report.level, PredictionLevel::Group);
        assert_close(group_report.metrics["mae"], 2.0);
    }

    #[test]
    fn refuses_metric_alignment_and_contract_mismatches() {
        let predictions = AggregatedPredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            level: PredictionLevel::Target,
            unit_ids: vec![target_unit("target:a")],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        };
        let missing_target = RegressionTargetBlock {
            level: PredictionLevel::Target,
            unit_ids: vec![target_unit("target:b")],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        };
        assert!(score_regression_aggregated_block(
            &predictions,
            &missing_target,
            &[RegressionMetricKind::Rmse],
        )
        .is_err());

        let wrong_level = RegressionTargetBlock {
            level: PredictionLevel::Group,
            unit_ids: vec![group_unit("group:a")],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        };
        assert!(score_regression_aggregated_block(
            &predictions,
            &wrong_level,
            &[RegressionMetricKind::Rmse],
        )
        .is_err());

        assert!(score_regression_aggregated_block(&predictions, &missing_target, &[]).is_err());
        assert!(score_regression_aggregated_block(
            &predictions,
            &RegressionTargetBlock {
                level: PredictionLevel::Target,
                unit_ids: vec![target_unit("target:a")],
                values: vec![vec![1.0]],
                target_names: vec!["other".to_string()],
            },
            &[RegressionMetricKind::Rmse],
        )
        .is_err());
        assert!(score_regression_aggregated_block(
            &predictions,
            &RegressionTargetBlock {
                level: PredictionLevel::Target,
                unit_ids: vec![target_unit("target:a")],
                values: vec![vec![1.0]],
                target_names: vec!["y".to_string()],
            },
            &[RegressionMetricKind::Rmse, RegressionMetricKind::Rmse],
        )
        .is_err());
    }

    #[test]
    fn refuses_duplicate_and_non_finite_sample_predictions() {
        let targets = RegressionTargetBlock {
            level: PredictionLevel::Sample,
            unit_ids: vec![sample_unit("sample:1")],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        };
        let mut predictions = PredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            sample_ids: vec![sid("sample:1")],
            values: vec![vec![f64::INFINITY]],
            target_names: vec!["y".to_string()],
        };
        assert!(score_regression_prediction_block(
            &predictions,
            &targets,
            &[RegressionMetricKind::Rmse],
        )
        .is_err());

        predictions.values = vec![vec![1.0], vec![1.0]];
        predictions.sample_ids = vec![sid("sample:1"), sid("sample:1")];
        assert!(score_regression_prediction_block(
            &predictions,
            &targets,
            &[RegressionMetricKind::Rmse],
        )
        .is_err());
    }

    #[test]
    fn constant_target_r2_is_finite_and_deterministic() {
        let targets = RegressionTargetBlock {
            level: PredictionLevel::Sample,
            unit_ids: vec![sample_unit("sample:1"), sample_unit("sample:2")],
            values: vec![vec![2.0], vec![2.0]],
            target_names: vec!["y".to_string()],
        };
        let exact_predictions = PredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:exact").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            sample_ids: vec![sid("sample:1"), sid("sample:2")],
            values: vec![vec![2.0], vec![2.0]],
            target_names: vec!["y".to_string()],
        };
        let exact_report = score_regression_prediction_block(
            &exact_predictions,
            &targets,
            &[RegressionMetricKind::R2],
        )
        .unwrap();
        assert_close(exact_report.metrics["r2"], 1.0);

        let off_predictions = PredictionBlock {
            values: vec![vec![2.0], vec![3.0]],
            ..exact_predictions
        };
        let off_report = score_regression_prediction_block(
            &off_predictions,
            &targets,
            &[RegressionMetricKind::R2],
        )
        .unwrap();
        assert_close(off_report.metrics["r2"], 0.0);
    }

    fn score_report(
        partition: PredictionPartition,
        fold: Option<&str>,
        rmse: f64,
    ) -> RegressionMetricReport {
        RegressionMetricReport {
            prediction_id: None,
            producer_node: NodeId::new("model:compat.0").unwrap(),
            variant_id: None,
            partition,
            fold_id: fold.map(|value| FoldId::new(value).unwrap()),
            level: PredictionLevel::Sample,
            row_count: 10,
            target_width: 1,
            target_names: vec!["y".to_string()],
            metrics: BTreeMap::from([("rmse".to_string(), rmse), ("r2".to_string(), 0.5)]),
        }
    }

    #[test]
    fn score_set_round_trips_validates_and_rejects_duplicates() {
        let set = ScoreSet {
            schema_version: SCORE_SET_SCHEMA_VERSION,
            plan_id: "plan:demo".to_string(),
            selection_metric: Some("rmse".to_string()),
            reports: vec![
                score_report(PredictionPartition::Validation, Some("avg"), 18.75),
                score_report(PredictionPartition::Test, Some("final"), 13.28),
            ],
        };
        set.validate().unwrap();

        // JSON round-trip is lossless.
        let json = serde_json::to_string(&set).unwrap();
        let back: ScoreSet = serde_json::from_str(&json).unwrap();
        assert_eq!(back, set);

        // schema_version defaults when omitted (forward-compatible read).
        let parsed: ScoreSet =
            serde_json::from_value(serde_json::json!({"plan_id": "p", "reports": []})).unwrap();
        assert_eq!(parsed.schema_version, SCORE_SET_SCHEMA_VERSION);

        // Duplicate (producer_node, partition, fold_id, level) is rejected.
        let dup = ScoreSet {
            reports: vec![
                score_report(PredictionPartition::Test, Some("final"), 1.0),
                score_report(PredictionPartition::Test, Some("final"), 2.0),
            ],
            ..set.clone()
        };
        assert!(dup.validate().is_err());

        // Empty plan_id is rejected.
        let blank = ScoreSet {
            plan_id: "  ".to_string(),
            reports: vec![score_report(PredictionPartition::Test, Some("final"), 1.0)],
            ..set
        };
        assert!(blank.validate().is_err());
    }
}
