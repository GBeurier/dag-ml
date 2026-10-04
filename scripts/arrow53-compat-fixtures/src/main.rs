//! Test-only, standalone producer. Never link candidate Arrow 59 crates here.
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use serde_json::json;
use sha2::{Digest, Sha256};

// Struct-serialization field order is the existing PredictionBlock contract.
const BLOCK: &str = r#"{"prediction_id":"pred:53","producer_node":"model:53","producer_port":"pred","partition":"validation","fold_id":"fold:0","sample_ids":["sample:11","sample:7"],"values":[[1.25,-2.5],[3.75,4.5]],"target_names":["target:a","target:b"]}"#;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: fixture-producer NEW_OUTPUT_DIRECTORY")?,
    );
    if args.next().is_some() {
        return Err("exactly one new output directory is required".into());
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    // Refuse existing directories, including empty ones; never refresh fixtures.
    fs::create_dir(&output)?;
    fs::create_dir(output.join("native_results"))?;
    write_ipc(&output)?;
    write_results(&output.join("native_results"))?;

    let mut files = BTreeMap::new();
    for name in [
        "cache.arrow",
        "native_results/predictions.parquet",
        "native_results/manifest.json",
        "native_results/score_set.json",
    ] {
        files.insert(name, hash(&fs::read(output.join(name))?));
    }
    let source = Path::new(env!("CARGO_MANIFEST_DIR"));
    let provenance = json!({
        "schema_version": 1,
        "producer": "dag-ml-arrow53-fixture-producer",
        "dependencies": {"arrow-array": "53.4.1", "arrow-ipc": "53.4.1",
            "arrow-schema": "53.4.1", "parquet": "53.4.1"},
        "producer_source_sha256": hash(include_bytes!("main.rs")),
        "producer_manifest_sha256": hash(include_bytes!("../Cargo.toml")),
        "producer_lock_sha256": hash(&fs::read(source.join("Cargo.lock"))?),
        "producer_executable_sha256": hash(&fs::read(std::env::current_exe()?)?),
        "ipc_content_fingerprint": hash(format!("[{BLOCK}]").as_bytes()),
        "files": files,
        "note": "Actual Rust 53 writer bytes; metadata HashMap order is not a byte-stability contract. Freeze this single generation with its build/log/lock provenance."
    });
    fs::write(
        output.join("provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    println!("{}", output.display());
    Ok(())
}

fn write_ipc(output: &Path) -> Result<(), Box<dyn Error>> {
    let mut metadata = HashMap::new();
    for (key, value) in [
        ("format", "v2".to_owned()),
        ("requirement_key", "requirement:53".to_owned()),
        ("cache_id", "cache:53".to_owned()),
        (
            "cache_namespace_fingerprints",
            format!("[\"{}\"]", "a".repeat(64)),
        ),
        ("partition", "\"validation\"".to_owned()),
        ("prediction_level", "\"sample\"".to_owned()),
        ("content_fingerprint", hash(format!("[{BLOCK}]").as_bytes())),
        ("block_count", "1".to_owned()),
        ("row_count", "2".to_owned()),
    ] {
        metadata.insert(format!("dag_ml.prediction_cache.{key}"), value);
    }
    metadata.insert(
        "fixture.producer".to_owned(),
        "rust-arrow-ipc-53.4.1".to_owned(),
    );
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("block_kind", DataType::Utf8, false),
            Field::new("payload_json", DataType::Utf8, false),
        ],
        metadata,
    ));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["sample"])),
            Arc::new(StringArray::from(vec![BLOCK])),
        ],
    )?;
    let mut writer = StreamWriter::try_new(fs::File::create(output.join("cache.arrow"))?, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}

fn list_field(name: &str, item: DataType) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", item, true))),
        false,
    )
}

fn floats(rows: &[&[f64]]) -> ArrayRef {
    let mut builder = ListBuilder::new(Float64Builder::new());
    for row in rows {
        for value in *row {
            builder.values().append_value(*value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn integers(rows: &[&[i64]]) -> ArrayRef {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for row in rows {
        for value in *row {
            builder.values().append_value(*value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn ids() -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in [&["sample:11", "sample:7"][..], &["sample:21"][..]] {
        for id in row {
            builder.values().append_value(id);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn write_results(output: &Path) -> Result<(), Box<dyn Error>> {
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (name, values) in [
        ("dataset", ["dataset:53", "dataset:53"]),
        ("config_name", ["base:53", "base:53"]),
        ("variant_id", ["variant:53", "variant:53"]),
        ("model_name", ["model:53", "model:53"]),
        ("partition", ["validation", "test"]),
        ("fold_id", ["fold:0", ""]),
        ("refit_context", ["", "final"]),
        ("scores", [r#"{"rmse":0.125}"#, r#"{"rmse":0.25}"#]),
        ("metric", ["rmse", "rmse"]),
        ("task_type", ["regression", "regression"]),
        (
            "target_names",
            [r#"["target:a","target:b"]"#, r#"["target:a","target:b"]"#],
        ),
    ] {
        fields.push(Field::new(name, DataType::Utf8, false));
        columns.push(Arc::new(StringArray::from(values.to_vec())));
    }
    for (name, column) in [
        ("y_true", floats(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0]])),
        ("y_pred", floats(&[&[1.25, -2.5, 3.75, 4.5], &[5.5, 6.5]])),
        ("y_proba", floats(&[&[], &[]])),
        ("weights", floats(&[&[0.5, 1.5], &[]])),
    ] {
        fields.push(list_field(name, DataType::Float64));
        columns.push(column);
    }
    for (name, column) in [
        ("sample_indices", integers(&[&[11, 7], &[21]])),
        ("y_true_shape", integers(&[&[2, 2], &[1, 2]])),
        ("y_pred_shape", integers(&[&[2, 2], &[1, 2]])),
        ("y_proba_shape", integers(&[&[], &[]])),
    ] {
        fields.push(list_field(name, DataType::Int64));
        columns.push(column);
    }
    fields.push(list_field("sample_ids", DataType::Utf8));
    columns.push(ids());
    fields.push(Field::new("arrays_present", DataType::Boolean, false));
    columns.push(Arc::new(BooleanArray::from(vec![true, true])));
    fields.push(Field::new("target_width", DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from(vec![2, 2])));
    for (name, values) in [
        ("val_score", vec![Some(0.125), None]),
        ("test_score", vec![None, Some(0.25)]),
        ("train_score", vec![None, None]),
    ] {
        fields.push(Field::new(name, DataType::Float64, true));
        columns.push(Arc::new(Float64Array::from(values)));
    }
    let metadata = HashMap::from([(
        "fixture.producer".to_owned(),
        "rust-parquet-53.4.1".to_owned(),
    )]);
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let mut writer = ArrowWriter::try_new(
        fs::File::create(output.join("predictions.parquet"))?,
        schema,
        None,
    )?;
    writer.write(&batch)?;
    writer.close()?;
    let scores = br#"{"plan_id":"plan:53","variants":{"variant:53":{"score":0.125}}}"#;
    fs::write(output.join("score_set.json"), scores)?;
    let manifest = json!({"schema_version":2,"engine":"dag-ml","score_set_hash":hash(scores),
        "files":{"score_set":"score_set.json","predictions":"predictions.parquet"},
        "artifacts":[{"uri":"../never-opened-model.bin"}],"fixture_producer":"rust-parquet-53.4.1"});
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
