//! Validated reader for the generic native dag-ml results directory.
//!
//! A results directory contains a schema-versioned manifest, the authoritative
//! dag-ml `ScoreSet`, and a columnar projection of prediction rows.  This crate
//! owns only that generic prediction/score surface: it does not know about a
//! host application, does not execute models, and deliberately never opens the
//! optional artifact subtree.
//!
//! The V2 layout uses fixed filenames.  Even though the manifest repeats those
//! names for human inspection, this reader rejects a mismatch and only opens
//! the fixed paths.  That keeps a corrupt manifest from redirecting a results
//! read outside its run directory.

use std::fs::File;
use std::io;
use std::path::Path;

use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, LargeListArray, LargeStringArray, ListArray,
    RecordBatch, StringArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Schema version currently understood by this reader.
pub const NATIVE_RESULTS_SCHEMA_VERSION: i64 = 2;
/// The manifest filename in a native results directory.
pub const MANIFEST_FILENAME: &str = "manifest.json";
/// The ScoreSet filename in a native results directory.
pub const SCORE_SET_FILENAME: &str = "score_set.json";
/// The prediction projection filename in a native results directory.
pub const PREDICTIONS_FILENAME: &str = "predictions.parquet";

const NATIVE_ENGINE: &str = "dag-ml";

/// Errors emitted while opening or validating native results.
#[derive(Debug, Error)]
pub enum NativeResultsError {
    /// The run directory or one of its fixed files could not be opened.
    #[error("native results I/O failed: {0}")]
    Io(#[from] io::Error),
    /// On-disk data was not a valid native results V2 payload.
    #[error("native results validation failed: {0}")]
    Validation(String),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, NativeResultsError>;

/// One queryable prediction row from the Parquet projection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NativePredictionRow {
    /// Dataset identifier supplied by the caller.
    pub dataset: String,
    /// Configuration identifier supplied by the caller.
    pub config_name: String,
    /// Stable configuration/variant identifier.
    pub variant_id: String,
    /// Model identifier supplied by the caller.
    pub model_name: String,
    /// Prediction partition (for example validation or test).
    pub partition: String,
    /// Fold identifier, if the producer supplied one.
    pub fold_id: String,
    /// Refit context, if the producer supplied one.
    pub refit_context: String,
    /// Original sample indices for array-backed rows.
    pub sample_indices: Vec<i64>,
    /// Flattened true targets.
    pub y_true: Vec<f64>,
    /// Flattened predictions.
    pub y_pred: Vec<f64>,
    /// Flattened probabilities, when the task provides them.
    pub y_proba: Vec<f64>,
    /// Shape paired with `y_true`.
    pub y_true_shape: Vec<i64>,
    /// Shape paired with `y_pred`.
    pub y_pred_shape: Vec<i64>,
    /// Shape paired with `y_proba`.
    pub y_proba_shape: Vec<i64>,
    /// Optional per-sample weights.
    pub weights: Vec<f64>,
    /// Whether the row carries direct prediction arrays.
    pub arrays_present: bool,
    /// Optional validation score.
    pub val_score: Option<f64>,
    /// Optional test score.
    pub test_score: Option<f64>,
    /// Optional train score.
    pub train_score: Option<f64>,
    /// Generic score record preserved as a JSON object.
    pub scores: Map<String, Value>,
    /// Metric name for this row.
    pub metric: String,
    /// Task type for this row.
    pub task_type: String,
    /// Number of target columns.
    pub target_width: i64,
    /// Descriptive names for each target column.
    pub target_names: Vec<String>,
}

/// Validated native results, ready for a language binding or a host query API.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NativeResultsView {
    /// The additive manifest, preserved without re-authoring it.
    pub manifest: Value,
    /// The authoritative dag-ml ScoreSet, preserved without re-authoring it.
    pub score_set: Value,
    /// Queryable prediction rows from the columnar projection.
    pub predictions: Vec<NativePredictionRow>,
}

/// Read a native results V2 directory.
///
/// Only `manifest.json`, `score_set.json`, and `predictions.parquet` are
/// opened.  In particular, artifact references are metadata only: this reader
/// never resolves a URI, deserializes an artifact, or executes user payloads.
pub fn read_native_results(run_dir: impl AsRef<Path>) -> Result<NativeResultsView> {
    let run_dir = run_dir.as_ref();
    let manifest = read_json(run_dir.join(MANIFEST_FILENAME), "manifest")?;
    validate_manifest(&manifest)?;

    let score_set = read_json(run_dir.join(SCORE_SET_FILENAME), "score set")?;
    if !score_set.is_object() {
        return Err(validation("score_set.json must contain a JSON object"));
    }
    let actual_hash = score_set_hash(&score_set);
    let expected_hash = required_string(
        required_object(&manifest, "manifest")?,
        "score_set_hash",
        "manifest",
    )?;
    if actual_hash != expected_hash {
        return Err(validation(
            "manifest score_set_hash does not match score_set.json",
        ));
    }

    let predictions = read_prediction_rows(&run_dir.join(PREDICTIONS_FILENAME))?;
    Ok(NativeResultsView {
        manifest,
        score_set,
        predictions,
    })
}

fn read_json(path: impl AsRef<Path>, label: &str) -> Result<Value> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| validation(format!("{label} is not valid JSON: {error}")))
}

fn validate_manifest(manifest: &Value) -> Result<()> {
    let manifest = required_object(manifest, "manifest")?;
    let schema_version = required_i64(manifest, "schema_version", "manifest")?;
    if schema_version != NATIVE_RESULTS_SCHEMA_VERSION {
        return Err(validation(format!(
            "unsupported manifest schema_version {schema_version}; expected {NATIVE_RESULTS_SCHEMA_VERSION}",
        )));
    }
    let engine = required_string(manifest, "engine", "manifest")?;
    if engine != NATIVE_ENGINE {
        return Err(validation(format!(
            "manifest engine must be {NATIVE_ENGINE:?}, got {engine:?}",
        )));
    }

    let files = manifest
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| validation("manifest.files must be a JSON object"))?;
    require_fixed_filename(files, "score_set", SCORE_SET_FILENAME)?;
    require_fixed_filename(files, "predictions", PREDICTIONS_FILENAME)?;
    let hash = required_string(manifest, "score_set_hash", "manifest")?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(validation(
            "manifest.score_set_hash must be a SHA-256 hex digest",
        ));
    }
    Ok(())
}

fn require_fixed_filename(files: &Map<String, Value>, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(files, key, "manifest.files")?;
    if actual != expected {
        return Err(validation(format!(
            "manifest.files.{key} must be {expected:?}",
        )));
    }
    Ok(())
}

fn read_prediction_rows(path: &Path) -> Result<Vec<NativePredictionRow>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| validation(format!("predictions.parquet cannot be opened: {error}")))?;
    let reader = builder
        .build()
        .map_err(|error| validation(format!("predictions.parquet cannot be read: {error}")))?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|error| {
            validation(format!("predictions.parquet has an invalid batch: {error}"))
        })?;
        for row in 0..batch.num_rows() {
            rows.push(read_prediction_row(&batch, row)?);
        }
    }
    Ok(rows)
}

fn read_prediction_row(batch: &RecordBatch, row: usize) -> Result<NativePredictionRow> {
    let y_true = required_list_f64(batch, "y_true", row)?;
    let y_pred = required_list_f64(batch, "y_pred", row)?;
    let y_proba = required_list_f64(batch, "y_proba", row)?;
    let y_true_shape = required_list_i64(batch, "y_true_shape", row)?;
    let y_pred_shape = required_list_i64(batch, "y_pred_shape", row)?;
    let y_proba_shape = required_list_i64(batch, "y_proba_shape", row)?;
    let arrays_present = required_bool(batch, "arrays_present", row)?;
    if arrays_present == y_pred.is_empty() {
        return Err(validation(
            "prediction row arrays_present must agree with whether y_pred is present",
        ));
    }
    validate_shape("y_true", &y_true, &y_true_shape)?;
    validate_shape("y_pred", &y_pred, &y_pred_shape)?;
    validate_shape("y_proba", &y_proba, &y_proba_shape)?;

    let target_width = required_i64_column(batch, "target_width", row)?;
    if target_width < 1 {
        return Err(validation("prediction row target_width must be positive"));
    }
    let target_names = parse_string_list(
        &required_string_column(batch, "target_names", row)?,
        "prediction row target_names",
    )?;
    if i64::try_from(target_names.len()).ok() != Some(target_width) {
        return Err(validation(
            "prediction row target_names length must equal target_width",
        ));
    }
    if y_true_shape.len() >= 2 && y_true_shape[1] != target_width {
        return Err(validation(
            "prediction row y_true target dimension must equal target_width",
        ));
    }

    let scores = parse_object(
        &required_string_column(batch, "scores", row)?,
        "prediction row scores",
    )?;

    Ok(NativePredictionRow {
        dataset: required_string_column(batch, "dataset", row)?,
        config_name: required_string_column(batch, "config_name", row)?,
        variant_id: required_string_column(batch, "variant_id", row)?,
        model_name: required_string_column(batch, "model_name", row)?,
        partition: required_string_column(batch, "partition", row)?,
        fold_id: required_string_column(batch, "fold_id", row)?,
        refit_context: required_string_column(batch, "refit_context", row)?,
        sample_indices: required_list_i64(batch, "sample_indices", row)?,
        y_true,
        y_pred,
        y_proba,
        y_true_shape,
        y_pred_shape,
        y_proba_shape,
        weights: required_list_f64(batch, "weights", row)?,
        arrays_present,
        val_score: optional_f64_column(batch, "val_score", row)?,
        test_score: optional_f64_column(batch, "test_score", row)?,
        train_score: optional_f64_column(batch, "train_score", row)?,
        scores,
        metric: required_string_column(batch, "metric", row)?,
        task_type: required_string_column(batch, "task_type", row)?,
        target_width,
        target_names,
    })
}

fn validate_shape(field: &str, values: &[f64], shape: &[i64]) -> Result<()> {
    if values.is_empty() && shape.is_empty() {
        return Ok(());
    }
    if values.is_empty() || shape.is_empty() {
        return Err(validation(format!(
            "prediction row {field} and {field}_shape must both be empty or both be populated",
        )));
    }
    let expected_len = shape.iter().try_fold(1_usize, |product, dimension| {
        let dimension = usize::try_from(*dimension)
            .ok()
            .filter(|dimension| *dimension > 0)
            .ok_or_else(|| {
                validation(format!(
                    "prediction row {field}_shape must contain positive dimensions"
                ))
            })?;
        product.checked_mul(dimension).ok_or_else(|| {
            validation(format!(
                "prediction row {field}_shape product overflows usize"
            ))
        })
    })?;
    if expected_len != values.len() {
        return Err(validation(format!(
            "prediction row {field} length does not match {field}_shape",
        )));
    }
    Ok(())
}

fn required_string_column(batch: &RecordBatch, name: &str, row: usize) -> Result<String> {
    let column = batch.column_by_name(name).ok_or_else(|| {
        validation(format!(
            "predictions.parquet is missing required column {name}"
        ))
    })?;
    if column.is_null(row) {
        return Err(validation(format!(
            "prediction column {name} must not contain nulls"
        )));
    }
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row).to_owned());
    }
    if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row).to_owned());
    }
    Err(validation(format!(
        "predictions.parquet column {name} must be a UTF-8 string"
    )))
}

fn required_bool(batch: &RecordBatch, name: &str, row: usize) -> Result<bool> {
    let values = required_column::<BooleanArray>(batch, name)?;
    if values.is_null(row) {
        return Err(validation(format!(
            "prediction column {name} must not contain nulls"
        )));
    }
    Ok(values.value(row))
}

fn required_i64_column(batch: &RecordBatch, name: &str, row: usize) -> Result<i64> {
    let values = required_column::<Int64Array>(batch, name)?;
    if values.is_null(row) {
        return Err(validation(format!(
            "prediction column {name} must not contain nulls"
        )));
    }
    Ok(values.value(row))
}

fn optional_f64_column(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<f64>> {
    let values = required_column::<Float64Array>(batch, name)?;
    Ok((!values.is_null(row)).then(|| values.value(row)))
}

fn required_list_f64(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<f64>> {
    let values = required_list_value(batch, name, row)?;
    let values = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| validation(format!("prediction column {name} must be list<float64>")))?;
    if values.null_count() != 0 {
        return Err(validation(format!(
            "prediction column {name} list values must not contain nulls",
        )));
    }
    Ok((0..values.len()).map(|index| values.value(index)).collect())
}

fn required_list_i64(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<i64>> {
    let values = required_list_value(batch, name, row)?;
    let values = values
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| validation(format!("prediction column {name} must be list<int64>")))?;
    if values.null_count() != 0 {
        return Err(validation(format!(
            "prediction column {name} list values must not contain nulls",
        )));
    }
    Ok((0..values.len()).map(|index| values.value(index)).collect())
}

fn required_list_value(
    batch: &RecordBatch,
    name: &str,
    row: usize,
) -> Result<arrow_array::ArrayRef> {
    let column = batch.column_by_name(name).ok_or_else(|| {
        validation(format!(
            "predictions.parquet is missing required column {name}"
        ))
    })?;
    if column.is_null(row) {
        return Err(validation(format!(
            "prediction column {name} must not contain nulls"
        )));
    }
    if let Some(list) = column.as_any().downcast_ref::<ListArray>() {
        return Ok(list.value(row));
    }
    if let Some(list) = column.as_any().downcast_ref::<LargeListArray>() {
        return Ok(list.value(row));
    }
    Err(validation(format!(
        "predictions.parquet column {name} must be a list array"
    )))
}

fn required_column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let column = batch.column_by_name(name).ok_or_else(|| {
        validation(format!(
            "predictions.parquet is missing required column {name}"
        ))
    })?;
    column.as_any().downcast_ref::<T>().ok_or_else(|| {
        validation(format!(
            "predictions.parquet column {name} has an unexpected Arrow type",
        ))
    })
}

fn parse_object(text: &str, label: &str) -> Result<Map<String, Value>> {
    serde_json::from_str::<Value>(text)
        .map_err(|error| validation(format!("{label} is not valid JSON: {error}")))?
        .as_object()
        .cloned()
        .ok_or_else(|| validation(format!("{label} must be a JSON object")))
}

fn parse_string_list(text: &str, label: &str) -> Result<Vec<String>> {
    serde_json::from_str::<Value>(text)
        .map_err(|error| validation(format!("{label} is not valid JSON: {error}")))?
        .as_array()
        .ok_or_else(|| validation(format!("{label} must be a JSON array")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| validation(format!("{label} must contain only strings")))
        })
        .collect()
}

fn required_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| validation(format!("{label} must be a JSON object")))
}

fn required_string<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| validation(format!("{label}.{key} must be a string")))
}

fn required_i64(object: &Map<String, Value>, key: &str, label: &str) -> Result<i64> {
    object
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| validation(format!("{label}.{key} must be an integer")))
}

fn score_set_hash(score_set: &Value) -> String {
    let canonical = canonical_json(score_set);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => {
            serde_json::to_string(value).expect("string serialization cannot fail")
        }
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let quoted_key =
                        serde_json::to_string(key).expect("string serialization cannot fail");
                    let value = values
                        .get(key)
                        .expect("object keys must resolve to their values");
                    format!("{quoted_key}:{}", canonical_json(value))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

fn validation(message: impl Into<String>) -> NativeResultsError {
    NativeResultsError::Validation(message.into())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder};
    use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use serde_json::{json, Value};

    use super::{read_native_results, score_set_hash, NativeResultsError, PREDICTIONS_FILENAME};

    #[test]
    fn reads_a_valid_v2_native_results_directory() {
        let fixture = Fixture::new();
        let view = read_native_results(fixture.path()).expect("valid results should read");

        assert_eq!(view.score_set["variants"]["base"]["score"], 0.42);
        assert_eq!(view.predictions.len(), 1);
        assert_eq!(view.predictions[0].y_pred, vec![0.1, 0.9]);
        assert_eq!(view.predictions[0].target_names, vec!["y"]);
        assert_eq!(view.predictions[0].val_score, Some(0.42));
    }

    #[test]
    fn refuses_a_score_set_hash_mismatch() {
        let fixture = Fixture::new();
        fs::write(fixture.path().join("score_set.json"), r#"{"changed":true}"#)
            .expect("fixture score set should be writable");

        assert_validation(read_native_results(fixture.path()), "score_set_hash");
    }

    #[test]
    fn refuses_manifest_redirects_instead_of_following_them() {
        let fixture = Fixture::new();
        let mut manifest = fixture.manifest();
        manifest["files"]["predictions"] = json!("../outside.parquet");
        fixture.write_manifest(&manifest);

        assert_validation(
            read_native_results(fixture.path()),
            "manifest.files.predictions",
        );
    }

    #[test]
    fn refuses_missing_or_wrong_prediction_columns() {
        let fixture = Fixture::new();
        write_predictions(&fixture.path().join(PREDICTIONS_FILENAME), false, false);
        assert_validation(
            read_native_results(fixture.path()),
            "missing required column target_names",
        );

        write_predictions(&fixture.path().join(PREDICTIONS_FILENAME), true, true);
        assert_validation(read_native_results(fixture.path()), "unexpected Arrow type");
    }

    #[test]
    fn refuses_malformed_row_json() {
        let fixture = Fixture::new();
        write_predictions_with_scores(&fixture.path().join(PREDICTIONS_FILENAME), "not-json");

        assert_validation(
            read_native_results(fixture.path()),
            "scores is not valid JSON",
        );
    }

    #[test]
    #[ignore = "requires DAG_ML_RESULTS_PYTHON_FIXTURE written by the Python producer"]
    fn reads_a_python_produced_results_directory() {
        let path = std::env::var("DAG_ML_RESULTS_PYTHON_FIXTURE")
            .expect("the ignored interoperability test requires DAG_ML_RESULTS_PYTHON_FIXTURE");
        let view = read_native_results(path).expect("the Python producer output must be readable");

        assert!(view.score_set.is_object());
        assert!(!view.predictions.is_empty());
    }

    fn assert_validation(result: super::Result<super::NativeResultsView>, expected: &str) {
        let error = result.expect_err("fixture must be rejected");
        let NativeResultsError::Validation(message) = error else {
            panic!("expected a validation error");
        };
        assert!(
            message.contains(expected),
            "{message} did not contain {expected}"
        );
    }

    struct Fixture {
        directory: std::path::PathBuf,
        score_set: Value,
    }

    impl Fixture {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time must be after epoch")
                .as_nanos();
            let directory = std::env::temp_dir().join(format!("dag-ml-results-{nonce}"));
            fs::create_dir(&directory).expect("fixture directory should be created");
            let score_set = json!({"variants": {"base": {"score": 0.42}}, "plan_id": "p1"});
            let fixture = Self {
                directory,
                score_set,
            };
            fixture.write_manifest(&fixture.manifest());
            fs::write(
                fixture.path().join("score_set.json"),
                super::canonical_json(&fixture.score_set),
            )
            .expect("fixture score set should be written");
            write_predictions(&fixture.path().join(PREDICTIONS_FILENAME), true, false);
            fixture
        }

        fn path(&self) -> &std::path::Path {
            &self.directory
        }

        fn manifest(&self) -> Value {
            json!({
                "schema_version": 2,
                "engine": "dag-ml",
                "score_set_hash": score_set_hash(&self.score_set),
                "files": {
                    "score_set": "score_set.json",
                    "predictions": "predictions.parquet",
                },
                "artifacts": [{"uri": "../never-opened.joblib"}],
            })
        }

        fn write_manifest(&self, manifest: &Value) {
            fs::write(
                self.path().join("manifest.json"),
                serde_json::to_vec(manifest).expect("fixture manifest should serialize"),
            )
            .expect("fixture manifest should be written");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    fn write_predictions(
        path: &std::path::Path,
        include_target_names: bool,
        wrong_target_width: bool,
    ) {
        write_prediction_batch(
            path,
            include_target_names,
            wrong_target_width,
            r#"{"rmse":0.1}"#,
        );
    }

    fn write_predictions_with_scores(path: &std::path::Path, scores: &str) {
        write_prediction_batch(path, true, false, scores);
    }

    fn write_prediction_batch(
        path: &std::path::Path,
        include_target_names: bool,
        wrong_target_width: bool,
        scores: &str,
    ) {
        let mut fields = vec![
            Field::new("dataset", DataType::Utf8, false),
            Field::new("config_name", DataType::Utf8, false),
            Field::new("variant_id", DataType::Utf8, false),
            Field::new("model_name", DataType::Utf8, false),
            Field::new("partition", DataType::Utf8, false),
            Field::new("fold_id", DataType::Utf8, false),
            Field::new("refit_context", DataType::Utf8, false),
            list_field("sample_indices", DataType::Int64),
            list_field("y_true", DataType::Float64),
            list_field("y_pred", DataType::Float64),
            list_field("y_proba", DataType::Float64),
            list_field("y_true_shape", DataType::Int64),
            list_field("y_pred_shape", DataType::Int64),
            list_field("y_proba_shape", DataType::Int64),
            list_field("weights", DataType::Float64),
            Field::new("arrays_present", DataType::Boolean, false),
            Field::new("val_score", DataType::Float64, true),
            Field::new("test_score", DataType::Float64, true),
            Field::new("train_score", DataType::Float64, true),
            Field::new("scores", DataType::Utf8, false),
            Field::new("metric", DataType::Utf8, false),
            Field::new("task_type", DataType::Utf8, false),
            Field::new(
                "target_width",
                if wrong_target_width {
                    DataType::Utf8
                } else {
                    DataType::Int64
                },
                false,
            ),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["dataset"])),
            Arc::new(StringArray::from(vec!["base"])),
            Arc::new(StringArray::from(vec!["base"])),
            Arc::new(StringArray::from(vec!["model"])),
            Arc::new(StringArray::from(vec!["validation"])),
            Arc::new(StringArray::from(vec!["0"])),
            Arc::new(StringArray::from(vec![""])),
            list_i64(&[0, 1]),
            list_f64(&[1.0, 2.0]),
            list_f64(&[0.1, 0.9]),
            list_f64(&[]),
            list_i64(&[2]),
            list_i64(&[2]),
            list_i64(&[]),
            list_f64(&[]),
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(Float64Array::from(vec![Some(0.42)])),
            Arc::new(Float64Array::from(vec![Option::<f64>::None])),
            Arc::new(Float64Array::from(vec![Option::<f64>::None])),
            Arc::new(StringArray::from(vec![scores])),
            Arc::new(StringArray::from(vec!["rmse"])),
            Arc::new(StringArray::from(vec!["regression"])),
        ];
        if wrong_target_width {
            columns.push(Arc::new(StringArray::from(vec!["1"])));
        } else {
            columns.push(Arc::new(Int64Array::from(vec![1])));
        }
        if include_target_names {
            fields.push(Field::new("target_names", DataType::Utf8, false));
            columns.push(Arc::new(StringArray::from(vec![r#"["y"]"#])));
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
            .expect("fixture record batch should be valid");
        let file = fs::File::create(path).expect("fixture parquet should be created");
        let mut writer =
            ArrowWriter::try_new(file, schema, None).expect("fixture writer should open");
        writer.write(&batch).expect("fixture batch should write");
        writer.close().expect("fixture writer should close");
    }

    fn list_field(name: &str, value_type: DataType) -> Field {
        Field::new(
            name,
            DataType::List(Arc::new(Field::new("item", value_type, true))),
            false,
        )
    }

    fn list_f64(values: &[f64]) -> ArrayRef {
        let mut builder = ListBuilder::new(Float64Builder::new());
        for value in values {
            builder.values().append_value(*value);
        }
        builder.append(true);
        Arc::new(builder.finish())
    }

    fn list_i64(values: &[i64]) -> ArrayRef {
        let mut builder = ListBuilder::new(Int64Builder::new());
        for value in values {
            builder.values().append_value(*value);
        }
        builder.append(true);
        Arc::new(builder.finish())
    }
}
