//! Opt-in reporting definitions, separate from the selectable metric catalog.
use super::*;

pub const POPULATION_RMSE_RATIOS_PROFILE: &str = "regression_report.population_rmse_ratios.v1";

/// Report-only profile. Undefined diagnostics carry reasons instead of numbers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegressionReportProfile {
    #[serde(rename = "regression_report.population_rmse_ratios.v1")]
    PopulationRmseRatiosV1,
}

impl RegressionReportProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PopulationRmseRatiosV1 => POPULATION_RMSE_RATIOS_PROFILE,
        }
    }
}

/// Profile reports use the finite owner ScoreSet representation. The alias
/// preserves the named Rust return type; diagnostics live in metrics/reasons.
pub type PopulationRmseRatiosReport = RegressionMetricReport;

/// Score one complete scalar target, joining by identity and retaining prediction
/// order for every sum. Missing/nonfinite cells are refused, never filtered.
/// Ratios and bias have no selectable metric references or objective directions.
pub fn score_regression_prediction_block_with_profile(
    predictions: &PredictionBlock,
    targets: &RegressionTargetBlock,
    profile: RegressionReportProfile,
) -> Result<PopulationRmseRatiosReport> {
    let width = predictions.validate_content()?;
    let target_width = targets.validate_shape()?;
    targets.require_complete_targets("population RMSE ratio report")?;
    if width != 1 || target_width != 1 || targets.level != PredictionLevel::Sample {
        return Err(DagMlError::OofValidation(
            "population RMSE ratio report requires one sample-level scalar target".into(),
        ));
    }
    if !predictions.target_names.is_empty()
        && !targets.target_names.is_empty()
        && predictions.target_names != targets.target_names
    {
        return Err(DagMlError::OofValidation(
            "report target names differ".into(),
        ));
    }
    let by_unit = targets
        .unit_ids
        .iter()
        .zip(&targets.values)
        .collect::<BTreeMap<_, _>>();
    if by_unit.len() != predictions.sample_ids.len() {
        return Err(DagMlError::OofValidation(
            "report target identities differ".into(),
        ));
    }
    let actual = predictions
        .sample_ids
        .iter()
        .map(|id| {
            by_unit
                .get(&PredictionUnitId::Sample(id.clone()))
                .map(|row| row[0])
                .ok_or_else(|| {
                    DagMlError::OofValidation(format!("report target missing sample `{id}`"))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let n = actual.len();
    let mut total = 0.0;
    for value in &actual {
        total += value;
    }
    let mean = total / n as f64;
    let mut sse = 0.0;
    let mut sst = 0.0;
    let mut residual_sum = 0.0;
    for (row, value) in predictions.values.iter().zip(&actual) {
        let residual = row[0] - value;
        sse += residual * residual;
        let centered = value - mean;
        sst += centered * centered;
        residual_sum += residual;
    }
    if !mean.is_finite() || !sse.is_finite() || !sst.is_finite() {
        return Err(DagMlError::OofValidation(
            "population report moments are not finite".into(),
        ));
    }
    let rmse = (sse / n as f64).sqrt();
    let mut sorted = actual;
    sorted.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        let pos = (n - 1) as f64 * q;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    };
    let mut metrics = BTreeMap::from([
        ("rmse".to_string(), rmse),
        ("bias".to_string(), residual_sum / n as f64),
    ]);
    let mut undefined_metrics = BTreeMap::new();
    if sst > 0.0 {
        metrics.insert("r2".into(), 1.0 - sse / sst);
    } else {
        undefined_metrics.insert("r2".into(), "constant_target".into());
    }
    if rmse > 0.0 {
        metrics.insert("rpd".into(), (sst / n as f64).sqrt() / rmse);
        metrics.insert("rpiq".into(), (quantile(0.75) - quantile(0.25)) / rmse);
    } else {
        undefined_metrics.insert("rpd".into(), "zero_rmse".into());
        undefined_metrics.insert("rpiq".into(), "zero_rmse".into());
    }
    let report = RegressionMetricReport {
        target_domain: None,
        grouping_key: None,
        prediction_id: predictions.prediction_id.clone(),
        producer_node: predictions.producer_node.clone(),
        producer_port: predictions.producer_port.clone(),
        variant_id: None,
        variant_label: None,
        partition: predictions.partition.clone(),
        fold_id: predictions.fold_id.clone(),
        level: PredictionLevel::Sample,
        row_count: n,
        target_width: 1,
        target_names: if predictions.target_names.is_empty() {
            targets.target_names.clone()
        } else {
            predictions.target_names.clone()
        },
        metrics,
        metric_profile: Some(profile.as_str().into()),
        undefined_metrics,
    };
    report.validate()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks(actual: &[f64], predicted: &[f64]) -> (PredictionBlock, RegressionTargetBlock) {
        let samples = (0..actual.len())
            .map(|i| crate::ids::SampleId::new(format!("s{i}")).unwrap())
            .collect::<Vec<_>>();
        (
            PredictionBlock {
                prediction_id: None,
                producer_node: NodeId::new("model:report").unwrap(),
                producer_port: None,
                partition: PredictionPartition::Validation,
                fold_id: Some(FoldId::new("holdout").unwrap()),
                sample_ids: samples.clone(),
                values: predicted.iter().map(|v| vec![*v]).collect(),
                target_names: vec!["y".into()],
            },
            RegressionTargetBlock {
                level: PredictionLevel::Sample,
                unit_ids: samples.into_iter().map(PredictionUnitId::Sample).collect(),
                values: actual.iter().map(|v| vec![*v]).collect(),
                validity_masks: None,
                target_names: vec!["y".into()],
            },
        )
    }

    #[test]
    fn amended_literal_population_reports() {
        // Literal accepted amendment cases. Historical Quali zeros are retained
        // below as a frozen named test baseline, never used as profile values.
        let cases: &[(&[f64], &[f64], serde_json::Value, serde_json::Value)] = &[
            (
                &[0.0, 2.0],
                &[0.0, 2.0],
                serde_json::json!({"rmse":0.0,"bias":0.0,"r2":1.0}),
                serde_json::json!({"rpd":"zero_rmse","rpiq":"zero_rmse"}),
            ),
            (
                &[2.0, 2.0],
                &[2.0, 2.0],
                serde_json::json!({"rmse":0.0,"bias":0.0}),
                serde_json::json!({"r2":"constant_target","rpd":"zero_rmse","rpiq":"zero_rmse"}),
            ),
            (
                &[2.0, 2.0],
                &[1.0, 3.0],
                serde_json::json!({"rmse":1.0,"bias":0.0,"rpd":0.0,"rpiq":0.0}),
                serde_json::json!({"r2":"constant_target"}),
            ),
            (
                &[0.0, 2.0],
                &[1.0, 3.0],
                serde_json::json!({"rmse":1.0,"bias":1.0,"r2":0.0,"rpd":1.0,"rpiq":1.0}),
                serde_json::json!({}),
            ),
        ];
        let frozen_historical_quali_r2_rpd_rpiq = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ];
        for (index, (actual, predicted, expected, undefined)) in cases.iter().enumerate() {
            let (block, truth) = blocks(actual, predicted);
            let report = score_regression_prediction_block_with_profile(
                &block,
                &truth,
                RegressionReportProfile::PopulationRmseRatiosV1,
            )
            .unwrap();
            assert_eq!(serde_json::to_value(&report.metrics).unwrap(), *expected);
            assert_eq!(
                serde_json::to_value(&report.undefined_metrics).unwrap(),
                *undefined
            );
            assert_eq!(
                report.metric_profile.as_deref(),
                Some(POPULATION_RMSE_RATIOS_PROFILE)
            );
            for (column, metric) in ["r2", "rpd", "rpiq"].iter().enumerate() {
                if report.undefined_metrics.contains_key(*metric) {
                    assert!(!report.metrics.contains_key(*metric));
                    assert_eq!(frozen_historical_quali_r2_rpd_rpiq[index][column], 0.0);
                }
            }
            let mut shuffled = truth.clone();
            shuffled.unit_ids.reverse();
            shuffled.values.reverse();
            assert_eq!(
                report,
                score_regression_prediction_block_with_profile(
                    &block,
                    &shuffled,
                    RegressionReportProfile::PopulationRmseRatiosV1
                )
                .unwrap()
            );
            let legacy =
                score_regression_prediction_block(&block, &truth, &[RegressionMetricKind::Rmse])
                    .unwrap();
            assert_eq!(
                legacy.metrics["rmse"].to_bits(),
                report.metrics["rmse"].to_bits()
            );
            // A report is directly serializable in the existing owner ScoreSet.
            let set = ScoreSet {
                schema_version: LEGACY_SCORE_SET_SCHEMA_VERSION,
                plan_id: "plan:profile.literal".into(),
                selection_metric: Some("rmse".into()),
                reports: vec![report],
            };
            set.validate().unwrap();
            let json = serde_json::to_string(&set).unwrap();
            assert!(!json.contains("NaN") && !json.contains("Infinity"));
            let back: ScoreSet = serde_json::from_str(&json).unwrap();
            back.validate().unwrap();
            assert_eq!(set, back);
        }
        // Existing built-in constant-target R² remains 1 for a perfect fit.
        let (block, truth) = blocks(&[2.0, 2.0], &[2.0, 2.0]);
        assert_eq!(
            score_regression_prediction_block(&block, &truth, &[RegressionMetricKind::R2])
                .unwrap()
                .metrics["r2"],
            1.0
        );
    }

    #[test]
    fn population_report_tiny_positive_interpolation_and_nonfinite() {
        // Exact binary residual: no epsilon clamp, positive RMSE, large finite ratios.
        let tiny = 9.094947017729282e-13;
        let (block, truth) = blocks(&[0.0, 2.0], &[tiny, 2.0 + tiny]);
        let report = score_regression_prediction_block_with_profile(
            &block,
            &truth,
            RegressionReportProfile::PopulationRmseRatiosV1,
        )
        .unwrap();
        assert_eq!(report.metrics["rmse"].to_bits(), tiny.to_bits());
        assert_eq!(report.metrics["rpd"], 1099511627776.0);
        assert_eq!(report.metrics["rpiq"], 1099511627776.0);
        assert!(report.undefined_metrics.is_empty());
        // Population SD (not sample SD) and linear quartiles for two/four rows.
        let (block, truth) = blocks(&[0.0, 0.0, 4.0, 4.0], &[2.0, 2.0, 6.0, 6.0]);
        let report = score_regression_prediction_block_with_profile(
            &block,
            &truth,
            RegressionReportProfile::PopulationRmseRatiosV1,
        )
        .unwrap();
        assert_eq!(report.metrics["rpd"], 1.0);
        assert_eq!(report.metrics["rpiq"], 2.0);
        let (mut block, mut truth) = blocks(&[0.0, 2.0], &[1.0, 3.0]);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            block.values[0][0] = value;
            assert!(score_regression_prediction_block_with_profile(
                &block,
                &truth,
                RegressionReportProfile::PopulationRmseRatiosV1
            )
            .is_err());
            block.values[0][0] = 1.0;
            truth.values[0][0] = value;
            assert!(score_regression_prediction_block_with_profile(
                &block,
                &truth,
                RegressionReportProfile::PopulationRmseRatiosV1
            )
            .is_err());
            truth.values[0][0] = 0.0;
        }
        truth.validity_masks = Some(vec![vec![false], vec![true]]);
        assert!(score_regression_prediction_block_with_profile(
            &block,
            &truth,
            RegressionReportProfile::PopulationRmseRatiosV1
        )
        .is_err());
    }

    #[test]
    fn profile_validation_selection_and_legacy_bytes() {
        const LEGACY: &str = r#"{"prediction_id":null,"producer_node":"model:report","partition":"validation","fold_id":"holdout","level":"sample","row_count":2,"target_width":1,"target_names":["y"],"metrics":{"r2":1.0,"rmse":0.0}}"#;
        let legacy: RegressionMetricReport = serde_json::from_str(LEGACY).unwrap();
        legacy.validate().unwrap();
        assert_eq!(serde_json::to_string(&legacy).unwrap(), LEGACY);
        let (block, truth) = blocks(&[2.0, 2.0], &[2.0, 2.0]);
        let report = score_regression_prediction_block_with_profile(
            &block,
            &truth,
            RegressionReportProfile::PopulationRmseRatiosV1,
        )
        .unwrap();
        let mut invalid = report.clone();
        invalid.metrics.insert("rpd".into(), 0.0);
        assert!(invalid.validate().is_err());
        let mut invalid = report.clone();
        invalid.metric_profile = None;
        assert!(invalid.validate().is_err());
        let mut invalid = report.clone();
        invalid.metric_profile = Some("unknown.v1".into());
        assert!(invalid.validate().is_err());
        for (metric, reason) in [
            ("", "zero_rmse"),
            ("unknown", "zero_rmse"),
            ("r2", ""),
            ("r2", "unknown"),
            ("r2", "zero_rmse"),
            ("rpd", "constant_target"),
        ] {
            let mut invalid = report.clone();
            invalid
                .undefined_metrics
                .insert(metric.into(), reason.into());
            assert!(invalid.validate().is_err());
        }
        let mut invalid = report.clone();
        invalid.metrics.remove("rmse");
        assert!(invalid.validate().is_err());
        let mut invalid = report.clone();
        invalid.metrics.insert("rmse".into(), f64::INFINITY);
        assert!(invalid.validate().is_err());
        let candidate = report
            .clone()
            .into_candidate_score("candidate:profile")
            .unwrap();
        assert_eq!(candidate.metrics, BTreeMap::from([("rmse".into(), 0.0)]));
        for metric in ["r2", "rpd", "rpiq", "bias"] {
            let policy: crate::selection::SelectionPolicy =
                serde_json::from_value(serde_json::json!({
                    "id":"select:profile", "metric":{"name":metric,"objective":"maximize"}
                }))
                .unwrap();
            assert!(
                crate::selection::select_candidate(&policy, std::slice::from_ref(&candidate))
                    .is_err()
            );
            let set = ScoreSet {
                schema_version: LEGACY_SCORE_SET_SCHEMA_VERSION,
                plan_id: "plan:profile".into(),
                selection_metric: Some(metric.into()),
                reports: vec![report.clone()],
            };
            assert!(set.validate().is_err());
        }
        let policy: crate::selection::SelectionPolicy = serde_json::from_value(serde_json::json!({
            "id":"select:profile", "metric":{"name":"rmse","objective":"minimize"}
        }))
        .unwrap();
        assert_eq!(
            crate::selection::select_candidate(&policy, &[candidate])
                .unwrap()
                .selected_score,
            0.0
        );
    }
}
