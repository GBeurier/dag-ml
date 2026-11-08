//! Default-gate contracts for the DAG-owned shared numerical fixture.
use std::collections::{BTreeMap, BTreeSet};

use dag_ml_core::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE: &str = include_str!("fixtures/shared_scoring_numerical.v1.json");
const FIXTURE_SHA256: &str = "bb6e5650505f44d9e529d46f3f051d848d9a35fbc4552ea72e785d1ccd4e9d94";

#[derive(Deserialize)]
struct Fixture {
    profile: String,
    schema_version: u32,
    tolerance: Tolerance,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Tolerance {
    absolute: f64,
    relative: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    #[serde(default)]
    n_outputs: usize,
    #[serde(default)]
    y_true: Vec<f64>,
    #[serde(default)]
    y_pred: Vec<f64>,
    validity_mask: Option<Vec<bool>>,
    target_row_order: Option<Vec<usize>>,
    provider_error_contains: Option<String>,
    class_labels: Option<Vec<f64>>,
    #[serde(default)]
    expected: BTreeMap<String, serde_json::Value>,
    applicability: BTreeMap<String, String>,
    base_case: Option<String>,
    operation: Option<String>,
    metric: Option<String>,
    input_field: Option<String>,
    input_token: Option<String>,
    error_contains: Option<String>,
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(fixture.profile, "dagml.shared-scoring.numerical.v1");
    assert_eq!(fixture.schema_version, 1);
    fixture
}

#[test]
fn shared_scoring_fixture_is_pinned_and_matches_local_canonical_source() {
    assert_eq!(
        format!("{:x}", Sha256::digest(FIXTURE.as_bytes())),
        FIXTURE_SHA256
    );
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/fixtures/criteria/shared_scoring_numerical.v1.json");
    match std::fs::read(&source) {
        Ok(bytes) => assert_eq!(bytes, FIXTURE.as_bytes(), "canonical fixture drift"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("Canonical fixture comparison: UNAVAILABLE ({}); packaged isolated tests use the pinned crate-local fixture", source.display());
        }
        Err(error) => panic!(
            "cannot read canonical fixture {}: {error}",
            source.display()
        ),
    }
    let fixture = fixture();
    let implementations = [
        "dag_coordinator",
        "dag_builtin_provider",
        "methods_scalar_abi",
        "methods_stats_abi",
        "methods_scalar_per_output_adapter",
        "methods_stats_per_output_adapter",
    ];
    let known_keys = implementations.into_iter().collect::<BTreeSet<_>>();
    let raw: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let declared_keys = raw["implementations"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        declared_keys, known_keys,
        "implementation declarations drift"
    );
    let supported = "supported";
    let rejects = "rejects";
    let na = "not_applicable";
    let statuses = BTreeSet::from([supported, rejects, na]);
    assert_eq!(
        raw["applicability_statuses"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        statuses,
        "applicability status declarations drift"
    );
    let mut kinds = BTreeSet::new();
    for case in &fixture.cases {
        assert_eq!(
            case.applicability
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            declared_keys,
            "{}: applicability must contain exactly all six declared keys",
            case.name
        );
        for (implementation, status) in &case.applicability {
            assert!(
                statuses.contains(status.as_str()),
                "{}: unknown status {implementation}={status}",
                case.name
            );
        }
        kinds.insert(case.kind.as_str());
        // Pin semantics by case kind/operation so valid-looking statuses cannot
        // silently disable an executable path or claim unsupported ABI coverage.
        let expected = match case.kind.as_str() {
            "regression" => {
                let adapter = case.n_outputs > 1 || case.validity_mask.is_some();
                let direct_status = if adapter { na } else { supported };
                let adapter_status = if adapter { supported } else { na };
                [
                    supported,
                    if case.validity_mask.is_some() {
                        rejects
                    } else {
                        supported
                    },
                    direct_status,
                    direct_status,
                    adapter_status,
                    adapter_status,
                ]
            }
            "classification" => [supported, supported, na, na, na, na],
            "refusal" => match case.operation.as_deref() {
                Some("missing_target_identity") => [rejects, na, na, na, na, na],
                Some("sample_weights") => [na, rejects, na, na, na, na],
                Some("nonfinite_input") => [rejects, rejects, rejects, rejects, na, na],
                operation => panic!("{}: unknown refusal operation {operation:?}", case.name),
            },
            kind => panic!("{}: unknown case kind {kind}", case.name),
        };
        for (implementation, expected_status) in implementations.into_iter().zip(expected) {
            assert_eq!(
                case.applicability[implementation], expected_status,
                "{}: incorrect applicability for {implementation}",
                case.name
            );
        }
    }
    assert_eq!(
        kinds,
        BTreeSet::from(["regression", "classification", "refusal"])
    );
}

fn close(fixture: &Fixture, actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs()
                <= fixture.tolerance.absolute + fixture.tolerance.relative * expected.abs(),
        "{context}: {actual} != {expected}"
    );
}

fn blocks(case: &Case) -> (PredictionBlock, RegressionTargetBlock) {
    let ids = (0..case.y_true.len() / case.n_outputs)
        .map(|i| SampleId::new(format!("sample:{i}")).unwrap())
        .collect::<Vec<_>>();
    let names = (0..case.n_outputs)
        .map(|i| format!("output:{i}"))
        .collect::<Vec<_>>();
    let order = case
        .target_row_order
        .clone()
        .unwrap_or_else(|| (0..ids.len()).collect());
    let predictions = PredictionBlock {
        prediction_id: Some("prediction:contract".into()),
        producer_node: NodeId::new("model:contract").unwrap(),
        producer_port: Some("prediction".into()),
        partition: PredictionPartition::Validation,
        fold_id: Some(FoldId::new("fold:0").unwrap()),
        sample_ids: ids.clone(),
        values: case
            .y_pred
            .chunks(case.n_outputs)
            .map(<[_]>::to_vec)
            .collect(),
        target_names: names.clone(),
    };
    let targets = RegressionTargetBlock {
        level: PredictionLevel::Sample,
        unit_ids: order
            .iter()
            .map(|&i| PredictionUnitId::Sample(ids[i].clone()))
            .collect(),
        values: order
            .iter()
            .map(|&i| case.y_true[i * case.n_outputs..(i + 1) * case.n_outputs].to_vec())
            .collect(),
        validity_masks: case.validity_mask.as_ref().map(|mask| {
            order
                .iter()
                .map(|&i| mask[i * case.n_outputs..(i + 1) * case.n_outputs].to_vec())
                .collect()
        }),
        target_names: names,
    };
    (predictions, targets)
}

fn task(case: &Case, metric: RegressionMetricKind) -> MetricEvaluationTask {
    let (predictions, _) = blocks(case);
    let classification = case.kind == "classification";
    MetricEvaluationTask::new(
        format!("contract:{}:{}", case.name, metric.name()),
        builtin_metric_reference(metric).unwrap(),
        if classification {
            LearningTaskKind::MulticlassClassification
        } else {
            LearningTaskKind::Regression
        },
        if classification {
            PredictionKind::ClassLabel
        } else {
            PredictionKind::RegressionPoint
        },
        MetricEvaluationScope {
            producer_node: predictions.producer_node,
            producer_port: predictions.producer_port,
            prediction_id: predictions.prediction_id,
            variant_id: None,
            partition: predictions.partition,
            fold_id: predictions.fold_id,
            level: PredictionLevel::Sample,
        },
        predictions
            .sample_ids
            .into_iter()
            .map(MetricUnitId::Sample)
            .collect(),
        predictions.values,
        case.y_true
            .chunks(case.n_outputs)
            .map(<[_]>::to_vec)
            .collect(),
        predictions.target_names,
        None,
        None,
        None,
    )
    .unwrap()
}

fn metrics(case: &Case) -> Vec<RegressionMetricKind> {
    if case.kind == "classification" {
        vec![
            RegressionMetricKind::Accuracy,
            RegressionMetricKind::BalancedAccuracy,
            RegressionMetricKind::F1,
        ]
    } else {
        vec![
            RegressionMetricKind::Rmse,
            RegressionMetricKind::Mae,
            RegressionMetricKind::R2,
        ]
    }
}

#[test]
fn metric_shared_fixture_coordinator_and_registry_decomposition() {
    let fixture = fixture();
    let registry = builtin_metric_registry().unwrap();
    for case in fixture.cases.iter().filter(|case| case.kind != "refusal") {
        let (predictions, targets) = blocks(case);
        let report =
            score_regression_prediction_block(&predictions, &targets, &metrics(case)).unwrap();
        assert_eq!(report.row_count, predictions.sample_ids.len());
        for metric in metrics(case) {
            let expected = case.expected[metric.name()].as_f64().unwrap();
            close(
                &fixture,
                report.metrics[metric.name()],
                expected,
                &case.name,
            );
            if case.applicability["dag_builtin_provider"] == "supported" {
                let task = task(case, metric);
                assert_eq!(
                    task.metric.spec.decomposition,
                    MetricDecomposition::PerOutput
                );
                assert_eq!(task.metric.spec.reduction, MetricReduction::Mean);
                let evaluated = registry.evaluate(&task).unwrap();
                close(&fixture, evaluated.aggregate, expected, &case.name);
                assert_eq!(evaluated.result.values.len(), case.n_outputs);
                for (column, value) in evaluated.result.values.iter().enumerate() {
                    let output = format!("output:{column}");
                    assert_eq!(value.output_id.as_deref(), Some(output.as_str()));
                    let component = case
                        .expected
                        .get(&format!("{}_per_output", metric.name()))
                        .map_or(expected, |values| values[column].as_f64().unwrap());
                    close(&fixture, value.value, component, &case.name);
                }
            }
            if let Some(values) = case.expected.get(&format!("{}_per_output", metric.name())) {
                for column in 0..case.n_outputs {
                    close(
                        &fixture,
                        report.metrics[&format!("{}:output:{column}", metric.name())],
                        values[column].as_f64().unwrap(),
                        &case.name,
                    );
                }
            }
        }
        if let Some(flattened) = case.expected.get("flattened_rmse") {
            // This adapter pools observed cells only, deliberately changing the reduction.
            let mut pooled = task(&fixture.cases[0], RegressionMetricKind::Rmse);
            let observed = |values: &[f64]| {
                values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| case.validity_mask.as_ref().is_none_or(|mask| mask[*i]))
                    .map(|(_, &v)| vec![v])
                    .collect::<Vec<_>>()
            };
            pooled.predictions = observed(&case.y_pred);
            pooled.targets = observed(&case.y_true);
            pooled.unit_ids = (0..pooled.targets.len())
                .map(|i| MetricUnitId::Sample(SampleId::new(format!("cell:{i}")).unwrap()))
                .collect();
            pooled.task_fingerprint = pooled.compute_fingerprint().unwrap();
            let flat = registry.evaluate(&pooled).unwrap().aggregate;
            close(&fixture, flat, flattened.as_f64().unwrap(), &case.name);
            assert!((flat - report.metrics["rmse"]).abs() > 0.01);
        }
        if let Some(mask) = &case.validity_mask {
            assert_eq!(case.applicability["dag_builtin_provider"], "rejects");
            let mut raw_task = task(case, RegressionMetricKind::Rmse);
            raw_task.missing_mask = Some(mask.chunks(case.n_outputs).map(<[_]>::to_vec).collect());
            let error = registry.evaluate(&raw_task).unwrap_err();
            assert!(error
                .to_string()
                .contains(case.provider_error_contains.as_ref().unwrap()));
            let mut hidden = targets.clone();
            for (row, mask) in hidden
                .values
                .iter_mut()
                .zip(hidden.validity_masks.as_ref().unwrap())
            {
                for (value, observed) in row.iter_mut().zip(mask) {
                    if !observed {
                        *value = 1e100;
                    }
                }
            }
            assert_eq!(
                report,
                score_regression_prediction_block(&predictions, &hidden, &metrics(case)).unwrap()
            );
        }
    }
}

#[test]
fn metric_shared_fixture_class_identity_and_probability_column_order() {
    let fixture = fixture();
    for case in fixture
        .cases
        .iter()
        .filter(|case| case.kind == "classification")
    {
        let (mut predictions, targets) = blocks(case);
        let labels = case.class_labels.as_ref().unwrap();
        let mut probabilities = ClassificationProbabilityBlock {
            producer_node: predictions.producer_node.clone(),
            producer_port: predictions.producer_port.clone(),
            partition: predictions.partition.clone(),
            fold_id: predictions.fold_id.clone(),
            sample_ids: predictions.sample_ids.clone(),
            class_labels: labels.clone(),
            values: case
                .y_pred
                .iter()
                .map(|pred| {
                    labels
                        .iter()
                        .map(|label| if label == pred { 0.8 } else { 0.1 })
                        .collect()
                })
                .collect(),
        };
        predictions.values.fill(vec![0.1]); // Projected feature is not the class label.
        for _ in 0..2 {
            let report = score_prediction_with_class_probabilities(
                &predictions,
                &probabilities,
                &targets,
                &metrics(case),
            )
            .unwrap();
            for metric in metrics(case) {
                close(
                    &fixture,
                    report.metrics[metric.name()],
                    case.expected[metric.name()].as_f64().unwrap(),
                    &case.name,
                );
            }
            probabilities.class_labels.reverse();
            for row in &mut probabilities.values {
                row.reverse();
            }
            probabilities.sample_ids.reverse();
            probabilities.values.reverse();
        }
    }
}

#[test]
fn metric_shared_fixture_rejects_invalid_inputs() {
    let fixture = fixture();
    let registry = builtin_metric_registry().unwrap();
    for case in fixture.cases.iter().filter(|case| case.kind == "refusal") {
        let base = fixture
            .cases
            .iter()
            .find(|base| Some(&base.name) == case.base_case.as_ref())
            .unwrap();
        let expected_error = case.error_contains.as_ref().unwrap();
        let (mut predictions, mut targets) = blocks(base);
        match case.operation.as_deref().unwrap() {
            "missing_target_identity" => {
                assert_eq!(case.applicability["dag_coordinator"], "rejects");
                targets.unit_ids[0] =
                    PredictionUnitId::Sample(SampleId::new("sample:absent").unwrap());
                let error =
                    score_regression_prediction_block(&predictions, &targets, &metrics(base))
                        .unwrap_err();
                assert!(
                    error.to_string().contains(expected_error),
                    "{}: {error}",
                    case.name
                );
            }
            "sample_weights" => {
                assert_eq!(case.applicability["dag_builtin_provider"], "rejects");
                let mut task = task(
                    base,
                    RegressionMetricKind::from_name(case.metric.as_ref().unwrap()).unwrap(),
                );
                task.sample_weights = Some(vec![1.0; task.unit_ids.len()]);
                let error = registry.evaluate(&task).unwrap_err();
                assert!(
                    error.to_string().contains(expected_error),
                    "{}: {error}",
                    case.name
                );
            }
            "nonfinite_input" => {
                assert_eq!(case.applicability["dag_coordinator"], "rejects");
                assert_eq!(case.applicability["dag_builtin_provider"], "rejects");
                let value = match case.input_token.as_deref().unwrap() {
                    "NaN" => f64::NAN,
                    "+Inf" => f64::INFINITY,
                    "-Inf" => f64::NEG_INFINITY,
                    token => panic!("unexpected token: {token}"),
                };
                for metric in metrics(base) {
                    let mut task = task(base, metric);
                    if case.input_field.as_deref() == Some("y_true") {
                        task.targets[0][0] = value;
                    } else {
                        task.predictions[0][0] = value;
                    }
                    let error = registry.evaluate(&task).unwrap_err();
                    assert!(
                        error.to_string().contains(expected_error),
                        "{}: {error}",
                        case.name
                    );
                }
                if case.input_field.as_deref() == Some("y_true") {
                    targets.values[0][0] = value;
                } else {
                    predictions.values[0][0] = value;
                }
                let error =
                    score_regression_prediction_block(&predictions, &targets, &metrics(base))
                        .unwrap_err();
                assert!(
                    error.to_string().contains(expected_error),
                    "{}: {error}",
                    case.name
                );
            }
            operation => panic!("unhandled operation: {operation}"),
        }
    }
}
