use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::aggregation::{AggregatedPredictionBlock, PredictionUnitId};
use crate::error::{DagMlError, Result};
use crate::oof::PredictionBlock;
use crate::policy::PredictionLevel;
use crate::selection::{CandidateScore, MetricObjective};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionMetricKind {
    Mse,
    Rmse,
    Mae,
    R2,
}

impl RegressionMetricKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::Rmse => "rmse",
            Self::Mae => "mae",
            Self::R2 => "r2",
        }
    }

    pub fn objective(self) -> MetricObjective {
        match self {
            Self::Mse | Self::Rmse | Self::Mae => MetricObjective::Minimize,
            Self::R2 => MetricObjective::Maximize,
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
                "metric_level".to_string(),
                serde_json::json!(prediction_level_name(self.level)),
            ),
            ("row_count".to_string(), serde_json::json!(self.row_count)),
            (
                "target_width".to_string(),
                serde_json::json!(self.target_width),
            ),
        ]);
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
        PredictionLevel::Sample,
        &prediction_units,
        &predictions.values,
        &predictions.target_names,
        width,
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
        predictions.level,
        &predictions.unit_ids,
        &predictions.values,
        &predictions.target_names,
        width,
        targets,
        metrics,
    )
}

fn score_regression_rows(
    prediction_level: PredictionLevel,
    prediction_unit_ids: &[PredictionUnitId],
    prediction_values: &[Vec<f64>],
    prediction_target_names: &[String],
    width: usize,
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
    if width != target_width {
        return Err(DagMlError::OofValidation(format!(
            "prediction width {width} does not match target width {target_width}"
        )));
    }
    if prediction_level != targets.level {
        return Err(DagMlError::OofValidation(format!(
            "prediction level {:?} does not match target level {:?}",
            prediction_level, targets.level
        )));
    }
    if !prediction_target_names.is_empty()
        && !targets.target_names.is_empty()
        && prediction_target_names != targets.target_names
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
    let mut aligned_predictions = Vec::with_capacity(prediction_unit_ids.len());
    let mut aligned_targets = Vec::with_capacity(prediction_unit_ids.len());
    for (unit_id, prediction_row) in prediction_unit_ids.iter().zip(prediction_values.iter()) {
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

    let target_names = if !prediction_target_names.is_empty() {
        prediction_target_names.to_vec()
    } else {
        targets.target_names.clone()
    };
    let metric_suffixes = target_metric_names(width, &target_names);
    let mut values = BTreeMap::new();
    for metric in metrics {
        let per_target =
            compute_metric_per_target(*metric, width, &aligned_predictions, &aligned_targets);
        values.insert(metric.name().to_string(), macro_mean(&per_target));
        for (name, value) in metric_suffixes.iter().zip(per_target) {
            values.insert(format!("{}:{name}", metric.name()), value);
        }
    }

    let report = RegressionMetricReport {
        level: prediction_level,
        row_count: prediction_unit_ids.len(),
        target_width: width,
        target_names,
        metrics: values,
    };
    report.validate()?;
    Ok(report)
}

fn validate_sample_prediction_block(block: &PredictionBlock) -> Result<usize> {
    let width = block.validate_shape()?;
    if block
        .values
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(DagMlError::OofValidation(format!(
            "producer `{}` emitted non-finite sample prediction values",
            block.producer_node
        )));
    }
    let unique = block.sample_ids.iter().collect::<BTreeSet<_>>();
    if unique.len() != block.sample_ids.len() {
        return Err(DagMlError::OofValidation(format!(
            "producer `{}` emitted duplicate sample predictions",
            block.producer_node
        )));
    }
    Ok(width)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{GroupId, NodeId, SampleId, TargetId};
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
}
