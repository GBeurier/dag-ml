//! Apache Arrow IPC codec for `BundlePredictionCachePayload`.
//!
//! The dag-ml core already ships JSON-payload-backed file and columnar
//! prediction cache stores. This crate adds the production Arrow IPC
//! path that STATUS.md flagged as missing for non-sample aggregated
//! prediction blocks.
//!
//! The on-wire format is an Arrow IPC stream containing a single
//! `RecordBatch`:
//!
//! - schema metadata carries the payload-level fields
//!   (`requirement_key`, `cache_id`, `partition`, `prediction_level`,
//!   `content_fingerprint`, `block_count`, `row_count`, codec
//!   version);
//! - each row is one block, with `block_kind` distinguishing sample
//!   blocks (`PredictionBlock`) from aggregated blocks
//!   (`AggregatedPredictionBlock`) and `payload_json` carrying the
//!   serde-canonical JSON for that block. JSON is intentional: it
//!   preserves serde-`Eq` round-tripping for both block shapes (which
//!   differ only in the unit-identification surface) without forcing
//!   a wide-format columnar schema that would have to carry every
//!   variant column even when most rows leave them null.
//!
//! Reading is the inverse: deserialize each row's JSON back into the
//! correct block variant based on `block_kind`. The codec validates
//! the resulting payload through `BundlePredictionCachePayload::
//! validate` so a corrupt stream cannot silently produce a payload
//! the runtime would reject downstream.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};

use dag_ml_core::aggregation::AggregatedPredictionBlock;
use dag_ml_core::bundle::BundlePredictionCachePayload;
use dag_ml_core::error::{DagMlError, Result};
use dag_ml_core::oof::PredictionBlock;

/// Codec version stamped into the Arrow schema metadata. Bump if the
/// row layout or metadata key set changes in a way readers must
/// reject as unsupported.
pub const CODEC_VERSION: &str = "v1";
const BLOCK_KIND_SAMPLE: &str = "sample";
const BLOCK_KIND_AGGREGATED: &str = "aggregated";

/// Metadata keys placed on the Arrow stream schema. They are exposed
/// publicly so that downstream tooling (CLI dumps, dashboards) can
/// inspect a cache file without re-deserializing the body.
pub const METADATA_KEY_FORMAT: &str = "dag_ml.prediction_cache.format";
pub const METADATA_KEY_REQUIREMENT_KEY: &str = "dag_ml.prediction_cache.requirement_key";
pub const METADATA_KEY_CACHE_ID: &str = "dag_ml.prediction_cache.cache_id";
pub const METADATA_KEY_PARTITION: &str = "dag_ml.prediction_cache.partition";
pub const METADATA_KEY_PREDICTION_LEVEL: &str = "dag_ml.prediction_cache.prediction_level";
pub const METADATA_KEY_CONTENT_FINGERPRINT: &str = "dag_ml.prediction_cache.content_fingerprint";
pub const METADATA_KEY_BLOCK_COUNT: &str = "dag_ml.prediction_cache.block_count";
pub const METADATA_KEY_ROW_COUNT: &str = "dag_ml.prediction_cache.row_count";

fn cache_schema(payload: &BundlePredictionCachePayload) -> Result<Schema> {
    let mut metadata = HashMap::new();
    metadata.insert(METADATA_KEY_FORMAT.to_string(), CODEC_VERSION.to_string());
    metadata.insert(
        METADATA_KEY_REQUIREMENT_KEY.to_string(),
        payload.requirement_key.clone(),
    );
    metadata.insert(METADATA_KEY_CACHE_ID.to_string(), payload.cache_id.clone());
    metadata.insert(
        METADATA_KEY_PARTITION.to_string(),
        serde_json::to_string(&payload.partition).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize partition for Arrow metadata: {error}"
            ))
        })?,
    );
    metadata.insert(
        METADATA_KEY_PREDICTION_LEVEL.to_string(),
        serde_json::to_string(&payload.prediction_level).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize prediction_level for Arrow metadata: {error}"
            ))
        })?,
    );
    metadata.insert(
        METADATA_KEY_CONTENT_FINGERPRINT.to_string(),
        payload.content_fingerprint.clone(),
    );
    metadata.insert(
        METADATA_KEY_BLOCK_COUNT.to_string(),
        payload.block_count.to_string(),
    );
    metadata.insert(
        METADATA_KEY_ROW_COUNT.to_string(),
        payload.row_count.to_string(),
    );

    let fields = vec![
        Field::new("block_kind", DataType::Utf8, false),
        Field::new("payload_json", DataType::Utf8, false),
    ];
    Ok(Schema::new_with_metadata(fields, metadata))
}

fn build_record_batch(
    payload: &BundlePredictionCachePayload,
    schema: Schema,
) -> Result<RecordBatch> {
    let mut kinds: Vec<&str> =
        Vec::with_capacity(payload.blocks.len() + payload.aggregated_blocks.len());
    let mut bodies: Vec<String> = Vec::with_capacity(kinds.capacity());
    for block in &payload.blocks {
        kinds.push(BLOCK_KIND_SAMPLE);
        bodies.push(serde_json::to_string(block).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize sample prediction block: {error}"
            ))
        })?);
    }
    for block in &payload.aggregated_blocks {
        kinds.push(BLOCK_KIND_AGGREGATED);
        bodies.push(serde_json::to_string(block).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize aggregated prediction block: {error}"
            ))
        })?);
    }

    let kind_array = StringArray::from(kinds);
    let body_array = StringArray::from(bodies);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(kind_array), Arc::new(body_array)],
    )
    .map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "failed to assemble Arrow RecordBatch for prediction cache: {error}"
        ))
    })
}

/// Serialize a `BundlePredictionCachePayload` to an Arrow IPC stream.
/// The output is a self-contained byte buffer suitable for writing to
/// disk, sending over a socket, or wrapping in a bundle artifact.
pub fn predictions_to_arrow_ipc(payload: &BundlePredictionCachePayload) -> Result<Vec<u8>> {
    payload.validate()?;
    let schema = cache_schema(payload)?;
    let batch = build_record_batch(payload, schema.clone())?;
    let mut buffer: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to create Arrow IPC writer for prediction cache: {error}"
            ))
        })?;
        writer.write(&batch).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to write Arrow batch for prediction cache: {error}"
            ))
        })?;
        writer.finish().map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to finalize Arrow IPC stream for prediction cache: {error}"
            ))
        })?;
    }
    Ok(buffer)
}

fn parse_metadata(metadata: &HashMap<String, String>, key: &str) -> Result<String> {
    metadata.get(key).cloned().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "Arrow prediction cache stream missing metadata key `{key}`"
        ))
    })
}

fn parse_metadata_json<T>(metadata: &HashMap<String, String>, key: &str) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let raw = parse_metadata(metadata, key)?;
    serde_json::from_str(&raw).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "Arrow prediction cache metadata `{key}` is not valid JSON: {error}"
        ))
    })
}

fn parse_usize_metadata(metadata: &HashMap<String, String>, key: &str) -> Result<usize> {
    let raw = parse_metadata(metadata, key)?;
    raw.parse::<usize>().map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "Arrow prediction cache metadata `{key}` is not a valid usize: {error}"
        ))
    })
}

/// Deserialize a `BundlePredictionCachePayload` from an Arrow IPC
/// stream produced by `predictions_to_arrow_ipc`. The reader walks
/// the single batch and reconstructs both sample blocks and
/// aggregated blocks, then runs the payload through `validate` so
/// any drift between the metadata and the rows is caught.
pub fn predictions_from_arrow_ipc(bytes: &[u8]) -> Result<BundlePredictionCachePayload> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "failed to open Arrow IPC stream for prediction cache: {error}"
        ))
    })?;
    let schema = reader.schema();
    let metadata = schema.metadata.clone();

    let format = parse_metadata(&metadata, METADATA_KEY_FORMAT)?;
    if format != CODEC_VERSION {
        return Err(DagMlError::RuntimeValidation(format!(
            "Arrow prediction cache uses codec version `{format}`, expected `{CODEC_VERSION}`"
        )));
    }
    let requirement_key = parse_metadata(&metadata, METADATA_KEY_REQUIREMENT_KEY)?;
    let cache_id = parse_metadata(&metadata, METADATA_KEY_CACHE_ID)?;
    let partition = parse_metadata_json(&metadata, METADATA_KEY_PARTITION)?;
    let prediction_level = parse_metadata_json(&metadata, METADATA_KEY_PREDICTION_LEVEL)?;
    let content_fingerprint = parse_metadata(&metadata, METADATA_KEY_CONTENT_FINGERPRINT)?;
    let block_count = parse_usize_metadata(&metadata, METADATA_KEY_BLOCK_COUNT)?;
    let row_count = parse_usize_metadata(&metadata, METADATA_KEY_ROW_COUNT)?;

    let mut blocks: Vec<PredictionBlock> = Vec::new();
    let mut aggregated_blocks: Vec<AggregatedPredictionBlock> = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to read Arrow batch from prediction cache: {error}"
            ))
        })?;
        let kind_array = batch
            .column_by_name("block_kind")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "Arrow prediction cache batch missing `block_kind` column".to_string(),
                )
            })?;
        let body_array = batch
            .column_by_name("payload_json")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "Arrow prediction cache batch missing `payload_json` column".to_string(),
                )
            })?;
        for row in 0..batch.num_rows() {
            let kind = kind_array.value(row);
            let body = body_array.value(row);
            match kind {
                BLOCK_KIND_SAMPLE => {
                    let block: PredictionBlock = serde_json::from_str(body).map_err(|error| {
                        DagMlError::RuntimeValidation(format!(
                            "Arrow prediction cache sample block at row {row} is not valid JSON: {error}"
                        ))
                    })?;
                    blocks.push(block);
                }
                BLOCK_KIND_AGGREGATED => {
                    let block: AggregatedPredictionBlock = serde_json::from_str(body)
                        .map_err(|error| {
                            DagMlError::RuntimeValidation(format!(
                                "Arrow prediction cache aggregated block at row {row} is not valid JSON: {error}"
                            ))
                        })?;
                    aggregated_blocks.push(block);
                }
                other => {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "Arrow prediction cache row {row} carries unknown block_kind `{other}`"
                    )));
                }
            }
        }
    }

    let payload = BundlePredictionCachePayload {
        requirement_key,
        cache_id,
        format: dag_ml_core::bundle::BUNDLE_PREDICTION_CACHE_FORMAT.to_string(),
        partition,
        prediction_level,
        block_count,
        row_count,
        content_fingerprint,
        blocks,
        aggregated_blocks,
    };
    payload.validate()?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    use dag_ml_core::aggregation::PredictionUnitId;
    use dag_ml_core::ids::{FoldId, NodeId, SampleId, TargetId};
    use dag_ml_core::oof::PredictionPartition;
    use dag_ml_core::policy::PredictionLevel;
    use serde::Serialize;
    use sha2::{Digest, Sha256};

    fn fingerprint<T: Serialize + ?Sized>(value: &T) -> String {
        let json = serde_json::to_vec(value).expect("canonical json");
        let digest = Sha256::digest(json);
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write;
            write!(&mut out, "{byte:02x}").expect("writing to string cannot fail");
        }
        out
    }

    fn sample_block() -> PredictionBlock {
        PredictionBlock {
            prediction_id: Some("pred:1".to_string()),
            producer_node: NodeId::new("model:ridge").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: vec![
                SampleId::new("S001").unwrap(),
                SampleId::new("S002").unwrap(),
            ],
            values: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            target_names: vec!["y0".to_string(), "y1".to_string()],
        }
    }

    fn aggregated_block() -> AggregatedPredictionBlock {
        AggregatedPredictionBlock {
            prediction_id: Some("pred:agg:1".to_string()),
            producer_node: NodeId::new("model:ridge").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            level: PredictionLevel::Target,
            unit_ids: vec![
                PredictionUnitId::Target(TargetId::new("target:a").unwrap()),
                PredictionUnitId::Target(TargetId::new("target:b").unwrap()),
            ],
            values: vec![vec![10.0], vec![20.0]],
            target_names: vec!["y0".to_string()],
        }
    }

    fn sample_payload() -> BundlePredictionCachePayload {
        let blocks = vec![sample_block()];
        BundlePredictionCachePayload {
            requirement_key: "requirement:sample".to_string(),
            cache_id: "cache:sample".to_string(),
            format: dag_ml_core::bundle::BUNDLE_PREDICTION_CACHE_FORMAT.to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Sample,
            block_count: blocks.len(),
            row_count: blocks.iter().map(|block| block.sample_ids.len()).sum(),
            content_fingerprint: fingerprint(&blocks),
            blocks,
            aggregated_blocks: Vec::new(),
        }
    }

    fn aggregated_payload() -> BundlePredictionCachePayload {
        let aggregated_blocks = vec![aggregated_block()];
        BundlePredictionCachePayload {
            requirement_key: "requirement:agg".to_string(),
            cache_id: "cache:agg".to_string(),
            format: dag_ml_core::bundle::BUNDLE_PREDICTION_CACHE_FORMAT.to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Target,
            block_count: aggregated_blocks.len(),
            row_count: aggregated_blocks
                .iter()
                .map(|block| block.unit_ids.len())
                .sum(),
            content_fingerprint: fingerprint(&aggregated_blocks),
            blocks: Vec::new(),
            aggregated_blocks,
        }
    }

    #[test]
    fn arrow_ipc_round_trips_sample_blocks_only() {
        let payload = sample_payload();
        let bytes = predictions_to_arrow_ipc(&payload).expect("encode");
        let decoded = predictions_from_arrow_ipc(&bytes).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn arrow_ipc_round_trips_aggregated_blocks_only() {
        let payload = aggregated_payload();
        let bytes = predictions_to_arrow_ipc(&payload).expect("encode");
        let decoded = predictions_from_arrow_ipc(&bytes).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn arrow_ipc_rejects_unknown_codec_version() {
        // Construct an Arrow IPC stream directly with a non-`v1`
        // codec version (instead of fragile byte-scanning the
        // encoded stream) so the test cannot accidentally corrupt
        // the wrong bytes if the literal `v1` happens to appear
        // elsewhere in the IPC framing.
        let payload = aggregated_payload();
        let mut schema = cache_schema(&payload).expect("schema");
        let mut bad_metadata = schema.metadata.clone();
        bad_metadata.insert(METADATA_KEY_FORMAT.to_string(), "v9".to_string());
        schema = Schema::new_with_metadata(schema.fields.clone(), bad_metadata);
        let batch = build_record_batch(&payload, schema.clone()).expect("batch");
        let mut buffer: Vec<u8> = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).expect("writer");
            writer.write(&batch).expect("write batch");
            writer.finish().expect("finish stream");
        }
        let err = predictions_from_arrow_ipc(&buffer).unwrap_err();
        match err {
            DagMlError::RuntimeValidation(message) => {
                assert!(
                    message.contains("codec version") && message.contains("v9"),
                    "unexpected: {message}"
                );
            }
            other => panic!("expected RuntimeValidation, got {other:?}"),
        }
    }

    #[test]
    fn arrow_ipc_refuses_payload_that_fails_validate() {
        let mut payload = sample_payload();
        // Force a block_count drift that `validate()` will reject so
        // the encoder propagates the validation error rather than
        // silently writing an invalid stream.
        payload.block_count = 99;
        let err = predictions_to_arrow_ipc(&payload).unwrap_err();
        match err {
            DagMlError::RuntimeValidation(_) => {}
            other => panic!("expected RuntimeValidation, got {other:?}"),
        }
    }
}
