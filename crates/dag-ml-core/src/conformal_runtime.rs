//! Durable, identity-keyed wiring for the native split-conformal kernel.
//!
//! This module deliberately accepts point predictions and truth only.  The
//! scheduler/controller boundary remains unchanged: a host supplies the
//! ordinary PREDICT result and DAG-ML validates its stable sample identities,
//! calibrates with [`crate::conformal`], and can apply the persisted record on
//! another ordinary PREDICT result.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::canonical::parse_typed_json;
use crate::conformal::{
    apply_split_absolute_residual, finite_sample_conformal_rank, split_absolute_residual_quantiles,
    ConformalMultiTargetPolicy, ConformalSmallSamplePolicy, RegressionConformalInterval,
    SplitConformalQuantile,
};
use crate::error::{DagMlError, Result};
use crate::ids::SampleId;
use crate::oof::PredictionBlock;

/// V1 did not bind calibration to the training/replay provenance closure.  It
/// is deliberately not accepted: callers must migrate to this closed V2 form.
pub const CONFORMAL_RUNTIME_SCHEMA_VERSION: u32 = 2;

/// Relation-derived calibration cohort. Physical and origin identities are
/// both retained so a relation-expanded training cohort cannot be bypassed by
/// presenting only one namespace. The attachment boundary derives and checks
/// these fields from an authoritative [`crate::relation::SampleRelationSet`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalCalibrationCohort {
    pub role: String,
    pub physical_sample_ids: Vec<SampleId>,
    pub origin_sample_ids: Vec<SampleId>,
    pub target_names: Vec<String>,
    pub manifest_fingerprint: String,
}

/// Complete, canonical provenance closure supplied by the replay boundary.
/// These are not optional hints: attached calibration checks every member
/// against the exact source outcome and replay before it is persisted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalCalibrationContext {
    pub predictor_binding_fingerprint: String,
    pub source_training_outcome_fingerprint: String,
    pub calibration_replay_outcome_fingerprint: String,
    pub data_identities_fingerprint: String,
    pub fold_set_fingerprint: String,
    pub training_influence_fingerprint: String,
    pub relation_fingerprint: String,
    pub calibration_cohort: ConformalCalibrationCohort,
    pub context_fingerprint: String,
}

/// Closed, self-fingerprinted split-conformal state retained beside a bundle.
/// `sample_ids` is the calibration order, not an interchangeable set: this
/// makes accidental positional joins fail before residuals are calculated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalCalibration {
    pub schema_version: u32,
    pub binding_id: String,
    pub target_names: Vec<String>,
    pub sample_ids: Vec<SampleId>,
    pub coverages: Vec<f64>,
    pub multi_target_policy: ConformalMultiTargetPolicy,
    pub small_sample_policy: ConformalSmallSamplePolicy,
    pub quantiles: Vec<SplitConformalQuantile>,
    pub context: ConformalCalibrationContext,
    pub calibration_fingerprint: String,
}

/// Typed reference retained by portable execution bundles.  It contains no
/// host object or duplicate algorithm state; the complete state stays in the
/// matching `TrainingOutcome`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalCalibrationRef {
    pub schema_version: u32,
    pub binding_id: String,
    pub calibration_fingerprint: String,
}

/// Identity-preserving interval result for one replayed point block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalIntervalBlock {
    pub schema_version: u32,
    pub binding_id: String,
    pub sample_ids: Vec<SampleId>,
    pub intervals: Vec<RegressionConformalInterval>,
    pub calibration_fingerprint: String,
    pub point_prediction_fingerprint: String,
}

/// Truth supplied by the data layer for a calibration replay.  It carries the
/// same stable physical sample ids as the point block so a host can never
/// smuggle a positional `y_true` matrix across a reordered replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalCalibrationTruth {
    pub sample_ids: Vec<SampleId>,
    pub values: Vec<Vec<f64>>,
}

impl ConformalIntervalBlock {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CONFORMAL_RUNTIME_SCHEMA_VERSION
            || self.binding_id.trim().is_empty()
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal interval block has an unsupported version or empty binding id"
                    .to_string(),
            ));
        }
        validate_unique_samples(&self.sample_ids)?;
        if self.intervals.is_empty()
            || self
                .intervals
                .iter()
                .any(|interval| interval.cells.len() != self.sample_ids.len())
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal interval block does not cover its exact sample ids".to_string(),
            ));
        }
        validate_sha256(&self.calibration_fingerprint)?;
        validate_sha256(&self.point_prediction_fingerprint)
    }
}

impl ConformalCalibration {
    #[allow(clippy::too_many_arguments)]
    pub fn calibrate_with_truth(
        binding_id: impl Into<String>,
        target_names: Vec<String>,
        predictions: &PredictionBlock,
        truth: &ConformalCalibrationTruth,
        context: ConformalCalibrationContext,
        coverages: Vec<f64>,
        multi_target_policy: ConformalMultiTargetPolicy,
        small_sample_policy: ConformalSmallSamplePolicy,
    ) -> Result<Self> {
        predictions.validate_content()?;
        validate_identity_aligned_truth(predictions, truth)?;
        context.validate_for_truth(truth, &target_names)?;
        if target_names.len() != predictions.values[0].len()
            || (!predictions.target_names.is_empty() && predictions.target_names != target_names)
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal target order does not match the point prediction binding".to_string(),
            ));
        }
        let residuals = predictions
            .values
            .iter()
            .zip(&truth.values)
            .map(|(prediction, actual)| {
                prediction
                    .iter()
                    .zip(actual)
                    .map(|(point, value)| (point - value).abs())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quantiles = split_absolute_residual_quantiles(
            &residuals,
            &coverages,
            multi_target_policy,
            small_sample_policy,
        )
        .map_err(|error| {
            DagMlError::RuntimeValidation(format!("conformal calibration failed: {error}"))
        })?;
        let mut calibration = Self {
            schema_version: CONFORMAL_RUNTIME_SCHEMA_VERSION,
            binding_id: binding_id.into(),
            target_names,
            sample_ids: predictions.sample_ids.clone(),
            coverages,
            multi_target_policy,
            small_sample_policy,
            quantiles,
            context,
            calibration_fingerprint: String::new(),
        };
        calibration.calibration_fingerprint = calibration.compute_fingerprint()?;
        calibration.validate()?;
        Ok(calibration)
    }

    pub fn reference(&self) -> Result<ConformalCalibrationRef> {
        self.validate()?;
        Ok(ConformalCalibrationRef {
            schema_version: CONFORMAL_RUNTIME_SCHEMA_VERSION,
            binding_id: self.binding_id.clone(),
            calibration_fingerprint: self.calibration_fingerprint.clone(),
        })
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        fingerprint_without(self, "calibration_fingerprint", "conformal calibration")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let raw = parse_typed_json(json)
            .and_then(|value| value.fingerprint_without("calibration_fingerprint"))
            .map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "conformal calibration is not strict TCV1 JSON: {error}"
                ))
            })?;
        let calibration: Self = serde_json::from_str(json)?;
        if calibration.calibration_fingerprint != raw {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration fingerprint does not match original TCV1 JSON".to_string(),
            ));
        }
        calibration.validate()?;
        Ok(calibration)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CONFORMAL_RUNTIME_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "conformal calibration has unsupported schema_version {}",
                self.schema_version
            )));
        }
        if self.binding_id.trim().is_empty() || self.target_names.is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration requires a binding id and target names".to_string(),
            ));
        }
        validate_unique_samples(&self.sample_ids)?;
        self.context.validate_for_calibration(self)?;
        if self.coverages.is_empty() || self.quantiles.len() != self.coverages.len() {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration coverages and quantiles must have equal non-zero length"
                    .to_string(),
            ));
        }
        if self
            .quantiles
            .iter()
            .zip(&self.coverages)
            .any(|(quantile, coverage)| quantile.coverage.to_bits() != coverage.to_bits())
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration quantile coverage order does not match coverages"
                    .to_string(),
            ));
        }
        let sample_count = u64::try_from(self.sample_ids.len()).map_err(|_| {
            DagMlError::RuntimeValidation(
                "conformal calibration sample count exceeds u64".to_string(),
            )
        })?;
        for (index, (coverage, quantile)) in self.coverages.iter().zip(&self.quantiles).enumerate()
        {
            let expected =
                finite_sample_conformal_rank(sample_count, *coverage).map_err(|error| {
                    DagMlError::RuntimeValidation(format!(
                        "invalid conformal rank at coverage {index}: {error}"
                    ))
                })?;
            if quantile.rank != expected {
                return Err(DagMlError::RuntimeValidation(format!(
                    "conformal quantile rank at coverage {index} does not match sample count and coverage"
                )));
            }
        }
        // The kernel validates coverage ordering, radius shape, and nestedness
        // before application; applying to one finite dummy row is a compact
        // validation that does not introduce another conformal algorithm.
        apply_split_absolute_residual(
            &[vec![0.0; self.target_names.len()]],
            &self.quantiles,
            self.multi_target_policy,
        )
        .map_err(|error| {
            DagMlError::RuntimeValidation(format!("invalid conformal quantiles: {error}"))
        })?;
        validate_sha256(&self.calibration_fingerprint)?;
        if self.calibration_fingerprint != self.compute_fingerprint()? {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration fingerprint does not match TCV1 content".to_string(),
            ));
        }
        Ok(())
    }

    pub fn apply(&self, predictions: &PredictionBlock) -> Result<ConformalIntervalBlock> {
        self.validate()?;
        predictions.validate_content()?;
        if predictions.target_names != self.target_names {
            return Err(DagMlError::RuntimeValidation(
                "conformal application target order does not match calibration".to_string(),
            ));
        }
        let intervals = apply_split_absolute_residual(
            &predictions.values,
            &self.quantiles,
            self.multi_target_policy,
        )
        .map_err(|error| {
            DagMlError::RuntimeValidation(format!("conformal application failed: {error}"))
        })?;
        Ok(ConformalIntervalBlock {
            schema_version: CONFORMAL_RUNTIME_SCHEMA_VERSION,
            binding_id: self.binding_id.clone(),
            sample_ids: predictions.sample_ids.clone(),
            intervals,
            calibration_fingerprint: self.calibration_fingerprint.clone(),
            point_prediction_fingerprint: point_prediction_fingerprint_for_runtime(predictions)?,
        })
    }
}

impl ConformalCalibrationContext {
    pub fn compute_fingerprint(&self) -> Result<String> {
        fingerprint_without(self, "context_fingerprint", "conformal calibration context")
    }

    pub fn validate_for_truth(
        &self,
        truth: &ConformalCalibrationTruth,
        target_names: &[String],
    ) -> Result<()> {
        self.validate()?;
        if self.calibration_cohort.physical_sample_ids != truth.sample_ids
            || self.calibration_cohort.target_names != target_names
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration cohort must exactly bind truth sample ids and targets"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        for value in [
            &self.predictor_binding_fingerprint,
            &self.source_training_outcome_fingerprint,
            &self.calibration_replay_outcome_fingerprint,
            &self.data_identities_fingerprint,
            &self.fold_set_fingerprint,
            &self.training_influence_fingerprint,
            &self.relation_fingerprint,
            &self.context_fingerprint,
        ] {
            validate_sha256(value)?;
        }
        self.calibration_cohort.validate()?;
        if self.context_fingerprint != self.compute_fingerprint()? {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration context fingerprint does not match TCV1 content".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_for_calibration(&self, calibration: &ConformalCalibration) -> Result<()> {
        self.validate_for_truth(
            &ConformalCalibrationTruth {
                sample_ids: calibration.sample_ids.clone(),
                values: vec![vec![0.0]; calibration.sample_ids.len()],
            },
            &calibration.target_names,
        )
    }
}

impl ConformalCalibrationCohort {
    pub fn compute_fingerprint(&self) -> Result<String> {
        fingerprint_without(self, "manifest_fingerprint", "conformal calibration cohort")
    }

    pub fn validate(&self) -> Result<()> {
        validate_sha256(&self.manifest_fingerprint)?;
        if self.role != "calibration" || self.target_names.is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration context requires calibration cohort role and targets"
                    .to_string(),
            ));
        }
        validate_unique_samples(&self.physical_sample_ids)?;
        if self.origin_sample_ids.iter().collect::<BTreeSet<_>>().len()
            != self.origin_sample_ids.len()
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration origin sample ids must be unique".to_string(),
            ));
        }
        if self.manifest_fingerprint != self.compute_fingerprint()? {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration cohort fingerprint does not match TCV1 content".to_string(),
            ));
        }
        Ok(())
    }
}

impl ConformalIntervalBlock {
    /// Validate interval closure against the actual point block and quantiles;
    /// a matching hash alone is never treated as sufficient.
    pub fn validate_against(
        &self,
        calibration: &ConformalCalibration,
        predictions: &PredictionBlock,
    ) -> Result<()> {
        self.validate()?;
        calibration.validate()?;
        if self.binding_id != calibration.binding_id
            || self.calibration_fingerprint != calibration.calibration_fingerprint
            || self.sample_ids != predictions.sample_ids
            || self.point_prediction_fingerprint
                != point_prediction_fingerprint_for_runtime(predictions)?
        {
            return Err(DagMlError::RuntimeValidation("conformal interval block is not bound to its calibration and point prediction block".to_string()));
        }
        let expected = calibration.apply(predictions)?;
        if self != &expected {
            return Err(DagMlError::RuntimeValidation(
                "conformal interval bounds do not close over point predictions and quantiles"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl ConformalCalibrationRef {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CONFORMAL_RUNTIME_SCHEMA_VERSION
            || self.binding_id.trim().is_empty()
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration reference has an unsupported version or empty binding id"
                    .to_string(),
            ));
        }
        validate_sha256(&self.calibration_fingerprint)
    }

    pub fn validate_against(&self, calibration: &ConformalCalibration) -> Result<()> {
        self.validate()?;
        calibration.validate()?;
        if self.schema_version != CONFORMAL_RUNTIME_SCHEMA_VERSION
            || self.binding_id != calibration.binding_id
            || self.calibration_fingerprint != calibration.calibration_fingerprint
        {
            return Err(DagMlError::RuntimeValidation(
                "conformal calibration reference does not match calibration state".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_identity_aligned_truth(
    predictions: &PredictionBlock,
    truth: &ConformalCalibrationTruth,
) -> Result<()> {
    if predictions.sample_ids != truth.sample_ids
        || predictions.values.len() != truth.values.len()
        || truth.values.is_empty()
        || truth
            .values
            .iter()
            .any(|row| row.len() != predictions.values[0].len())
        || truth
            .values
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(DagMlError::RuntimeValidation(
            "conformal truth must be finite and exactly row/target aligned by sample id"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_unique_samples(sample_ids: &[SampleId]) -> Result<()> {
    if sample_ids.is_empty() || sample_ids.iter().collect::<BTreeSet<_>>().len() != sample_ids.len()
    {
        return Err(DagMlError::RuntimeValidation(
            "conformal calibration requires non-empty unique sample ids".to_string(),
        ));
    }
    Ok(())
}

fn fingerprint_without<T: Serialize>(value: &T, field: &str, label: &str) -> Result<String> {
    let json = serde_json::to_string(value)?;
    parse_typed_json(&json)
        .and_then(|typed| typed.fingerprint_without(field))
        .map_err(|error| DagMlError::RuntimeValidation(format!("{label} is outside TCV1: {error}")))
}

pub(crate) fn point_prediction_fingerprint_for_runtime(
    predictions: &PredictionBlock,
) -> Result<String> {
    predictions.validate_content()?;
    fingerprint_without(predictions, "prediction_id", "conformal point prediction")
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DagMlError::RuntimeValidation(
            "conformal calibration fingerprint must be lowercase SHA-256".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::NodeId;
    use crate::oof::PredictionPartition;

    fn block(ids: &[&str], values: &[f64]) -> PredictionBlock {
        PredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:regressor").unwrap(),
            producer_port: Some("prediction".to_string()),
            partition: PredictionPartition::Validation,
            fold_id: None,
            sample_ids: ids.iter().map(|id| SampleId::new(*id).unwrap()).collect(),
            values: values.iter().map(|value| vec![*value]).collect(),
            target_names: vec!["y".to_string()],
        }
    }

    fn context(ids: Vec<SampleId>, targets: Vec<String>) -> ConformalCalibrationContext {
        let mut cohort = ConformalCalibrationCohort {
            role: "calibration".to_string(),
            physical_sample_ids: ids.clone(),
            origin_sample_ids: ids,
            target_names: targets,
            manifest_fingerprint: String::new(),
        };
        cohort.manifest_fingerprint = cohort.compute_fingerprint().unwrap();
        let mut context = ConformalCalibrationContext {
            predictor_binding_fingerprint: "1".repeat(64),
            source_training_outcome_fingerprint: "2".repeat(64),
            calibration_replay_outcome_fingerprint: "3".repeat(64),
            data_identities_fingerprint: "4".repeat(64),
            fold_set_fingerprint: "5".repeat(64),
            training_influence_fingerprint: "6".repeat(64),
            relation_fingerprint: "7".repeat(64),
            calibration_cohort: cohort,
            context_fingerprint: String::new(),
        };
        context.context_fingerprint = context.compute_fingerprint().unwrap();
        context
    }

    #[test]
    fn calibration_round_trips_and_application_preserves_replay_ids() {
        let calibration = ConformalCalibration::calibrate_with_truth(
            "output:main",
            vec!["y".to_string()],
            &block(&["s1", "s2", "s3"], &[1.0, 3.0, 5.0]),
            &ConformalCalibrationTruth {
                sample_ids: vec![
                    SampleId::new("s1").unwrap(),
                    SampleId::new("s2").unwrap(),
                    SampleId::new("s3").unwrap(),
                ],
                values: vec![vec![0.0], vec![2.0], vec![4.0]],
            },
            context(
                vec![
                    SampleId::new("s1").unwrap(),
                    SampleId::new("s2").unwrap(),
                    SampleId::new("s3").unwrap(),
                ],
                vec!["y".to_string()],
            ),
            vec![0.5],
            ConformalMultiTargetPolicy::Marginal,
            ConformalSmallSamplePolicy::Error,
        )
        .unwrap();
        let json = serde_json::to_string(&calibration).unwrap();
        let loaded = ConformalCalibration::from_json(&json).unwrap();
        let replay = block(&["new:2", "new:1"], &[10.0, 20.0]);
        let intervals = loaded.apply(&replay).unwrap();
        assert_eq!(intervals.sample_ids, replay.sample_ids);
        assert_eq!(intervals.intervals.len(), 1);
        let cell = intervals.intervals[0].cells[0][0];
        assert_eq!(cell.endpoints(), (Some(9.0), Some(11.0)));
    }

    #[test]
    fn calibration_refuses_order_and_tamper() {
        let prediction = block(&["s1", "s2"], &[1.0, 2.0]);
        assert!(ConformalCalibration::calibrate_with_truth(
            "output:main",
            vec!["y".to_string()],
            &prediction,
            &ConformalCalibrationTruth {
                sample_ids: vec![SampleId::new("s2").unwrap(), SampleId::new("s1").unwrap()],
                values: vec![vec![1.0], vec![0.0]],
            },
            context(prediction.sample_ids.clone(), vec!["y".to_string()]),
            vec![0.5],
            ConformalMultiTargetPolicy::Marginal,
            ConformalSmallSamplePolicy::Error,
        )
        .is_err());
        assert!(ConformalCalibration::calibrate_with_truth(
            "output:main",
            vec!["wrong".to_string()],
            &prediction,
            &ConformalCalibrationTruth {
                sample_ids: prediction.sample_ids.clone(),
                values: vec![vec![0.0], vec![1.0]]
            },
            context(prediction.sample_ids.clone(), vec!["wrong".to_string()]),
            vec![0.5],
            ConformalMultiTargetPolicy::Marginal,
            ConformalSmallSamplePolicy::Error
        )
        .is_err());
        let calibration = ConformalCalibration::calibrate_with_truth(
            "output:main",
            vec!["y".to_string()],
            &prediction,
            &ConformalCalibrationTruth {
                sample_ids: prediction.sample_ids.clone(),
                values: vec![vec![0.0], vec![1.0]],
            },
            context(prediction.sample_ids.clone(), vec!["y".to_string()]),
            vec![0.5],
            ConformalMultiTargetPolicy::Marginal,
            ConformalSmallSamplePolicy::Error,
        )
        .unwrap();
        let mut value = serde_json::to_value(calibration).unwrap();
        value["quantiles"][0]["rank"] = serde_json::json!(1);
        let mut resigned: ConformalCalibration = serde_json::from_value(value.clone()).unwrap();
        resigned.calibration_fingerprint = resigned.compute_fingerprint().unwrap();
        value = serde_json::to_value(resigned).unwrap();
        assert!(ConformalCalibration::from_json(&value.to_string()).is_err());
    }

    #[test]
    fn v2_context_is_required_and_interval_bounds_close_over_points() {
        let prediction = block(&["cal:1", "cal:2"], &[3.0, 7.0]);
        let truth = ConformalCalibrationTruth {
            sample_ids: prediction.sample_ids.clone(),
            values: vec![vec![2.0], vec![5.0]],
        };
        let calibration = ConformalCalibration::calibrate_with_truth(
            "output:main",
            vec!["y".to_string()],
            &prediction,
            &truth,
            context(prediction.sample_ids.clone(), vec!["y".to_string()]),
            vec![0.5],
            ConformalMultiTargetPolicy::Marginal,
            ConformalSmallSamplePolicy::Error,
        )
        .unwrap();
        let replay = block(&["replay:1"], &[10.0]);
        let mut intervals = calibration.apply(&replay).unwrap();
        intervals.validate_against(&calibration, &replay).unwrap();
        intervals.intervals[0].coverage = 0.8;
        assert!(intervals.validate_against(&calibration, &replay).is_err());

        let mut v1 = serde_json::to_value(&calibration).unwrap();
        v1["schema_version"] = serde_json::json!(1);
        assert!(ConformalCalibration::from_json(&v1.to_string()).is_err());
        let mut missing_context = serde_json::to_value(&calibration).unwrap();
        missing_context.as_object_mut().unwrap().remove("context");
        assert!(ConformalCalibration::from_json(&missing_context.to_string()).is_err());
    }
}
