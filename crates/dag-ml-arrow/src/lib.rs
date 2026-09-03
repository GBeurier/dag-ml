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
//!   (`requirement_key`, `cache_id`, cache namespace fingerprints,
//!   `partition`, `prediction_level`, `content_fingerprint`,
//!   `block_count`, `row_count`, codec version);
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

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};

use dag_ml_core::aggregation::AggregatedPredictionBlock;
use dag_ml_core::bundle::{
    BundlePredictionCachePayload, BundlePredictionCachePayloadSet, ExecutionBundle,
};
use dag_ml_core::error::{DagMlError, Result};
use dag_ml_core::ids::BundleId;
use dag_ml_core::oof::PredictionBlock;
use dag_ml_core::runtime::{
    ColumnarPredictionCacheStore, HandleRef, PredictionCacheMaterializationRecord,
    PredictionCacheMaterializationRequest, RuntimePredictionCacheStore,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Codec version stamped into the Arrow schema metadata. Bump if the
/// row layout or metadata key set changes in a way readers must
/// reject as unsupported.
pub const CODEC_VERSION: &str = "v2";
pub const LEGACY_CODEC_VERSION: &str = "v1";
const BLOCK_KIND_SAMPLE: &str = "sample";
const BLOCK_KIND_AGGREGATED: &str = "aggregated";

/// Metadata keys placed on the Arrow stream schema. They are exposed
/// publicly so that downstream tooling (CLI dumps, dashboards) can
/// inspect a cache file without re-deserializing the body.
pub const METADATA_KEY_FORMAT: &str = "dag_ml.prediction_cache.format";
pub const METADATA_KEY_REQUIREMENT_KEY: &str = "dag_ml.prediction_cache.requirement_key";
pub const METADATA_KEY_CACHE_ID: &str = "dag_ml.prediction_cache.cache_id";
pub const METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS: &str =
    "dag_ml.prediction_cache.cache_namespace_fingerprints";
pub const METADATA_KEY_PARTITION: &str = "dag_ml.prediction_cache.partition";
pub const METADATA_KEY_PREDICTION_LEVEL: &str = "dag_ml.prediction_cache.prediction_level";
pub const METADATA_KEY_CONTENT_FINGERPRINT: &str = "dag_ml.prediction_cache.content_fingerprint";
pub const METADATA_KEY_BLOCK_COUNT: &str = "dag_ml.prediction_cache.block_count";
pub const METADATA_KEY_ROW_COUNT: &str = "dag_ml.prediction_cache.row_count";

/// Manifest name used by the durable Arrow IPC prediction-cache store.
///
/// It is deliberately distinct from the core JSON store manifest so callers
/// can reject an ambiguous directory before reading any cache payload.
pub const ARROW_PREDICTION_CACHE_MANIFEST_FILE: &str = "prediction_cache_arrow_manifest.json";
pub const ARROW_PREDICTION_CACHE_STORE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrowPredictionCacheEntry {
    pub requirement_key: String,
    pub cache_id: String,
    pub file_name: String,
    pub content_fingerprint: String,
    pub ipc_fingerprint: String,
}

impl ArrowPredictionCacheEntry {
    fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("requirement_key", self.requirement_key.as_str()),
            ("cache_id", self.cache_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "Arrow prediction cache manifest {label} is empty"
                )));
            }
        }
        if self.file_name == "."
            || self.file_name == ".."
            || self.file_name.contains('/')
            || self.file_name.contains('\\')
            || !self.file_name.ends_with(".arrow")
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "Arrow prediction cache file name `{}` must be a plain .arrow file name",
                self.file_name
            )));
        }
        validate_sha256("Arrow prediction cache content", &self.content_fingerprint)?;
        validate_sha256("Arrow prediction cache IPC", &self.ipc_fingerprint)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrowPredictionCacheManifest {
    pub schema_version: u32,
    pub bundle_id: BundleId,
    pub caches: Vec<ArrowPredictionCacheEntry>,
}

impl ArrowPredictionCacheManifest {
    pub fn validate_against_bundle(&self, bundle: &ExecutionBundle) -> Result<()> {
        if self.schema_version != ARROW_PREDICTION_CACHE_STORE_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "Arrow prediction cache manifest uses unsupported schema_version {}",
                self.schema_version
            )));
        }
        bundle.validate()?;
        if self.bundle_id != bundle.bundle_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "Arrow prediction cache manifest bundle `{}` does not match `{}`",
                self.bundle_id, bundle.bundle_id
            )));
        }
        if self.caches.len() != bundle.prediction_caches.len() {
            return Err(DagMlError::RuntimeValidation(format!(
                "Arrow prediction cache manifest has {} entries for {} bundle cache records",
                self.caches.len(),
                bundle.prediction_caches.len()
            )));
        }

        let mut requirement_keys = BTreeSet::new();
        let mut cache_ids = BTreeSet::new();
        let mut file_names = BTreeSet::new();
        for entry in &self.caches {
            entry.validate()?;
            if !requirement_keys.insert(entry.requirement_key.as_str())
                || !cache_ids.insert(entry.cache_id.as_str())
                || !file_names.insert(entry.file_name.as_str())
            {
                return Err(DagMlError::RuntimeValidation(
                    "Arrow prediction cache manifest contains duplicate identities".to_string(),
                ));
            }
            let record = bundle
                .prediction_caches
                .iter()
                .find(|record| record.requirement_key == entry.requirement_key)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "Arrow prediction cache manifest contains unknown requirement `{}`",
                        entry.requirement_key
                    ))
                })?;
            if record.cache_id != entry.cache_id
                || record.content_fingerprint != entry.content_fingerprint
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "Arrow prediction cache manifest entry `{}` does not match its bundle record",
                    entry.requirement_key
                )));
            }
        }
        Ok(())
    }
}

/// Durable Arrow IPC implementation of the runtime prediction-cache store.
///
/// `open` validates the manifest, every IPC byte fingerprint, Arrow metadata,
/// and the complete payload set against the freshly deserialized execution
/// bundle. It retains decoded, core-owned columnar values only: no file or
/// host handle survives construction.
#[derive(Clone, Debug)]
pub struct ArrowPredictionCacheStore {
    root: PathBuf,
    manifest: ArrowPredictionCacheManifest,
    inner: ColumnarPredictionCacheStore,
    materialization_records: RefCell<Vec<PredictionCacheMaterializationRecord>>,
}

impl ArrowPredictionCacheStore {
    pub fn write_payload_set(
        root: impl AsRef<Path>,
        bundle: &ExecutionBundle,
        payloads: &BundlePredictionCachePayloadSet,
    ) -> Result<ArrowPredictionCacheManifest> {
        payloads.validate_against_bundle(bundle)?;
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to create Arrow prediction cache store `{}`: {error}",
                root.display()
            ))
        })?;

        let mut caches = Vec::with_capacity(payloads.caches.len());
        for payload in &payloads.caches {
            let bytes = predictions_to_arrow_ipc(payload)?;
            let ipc_fingerprint = sha256_hex(&bytes);
            let file_name = format!("prediction-cache-{}.arrow", &ipc_fingerprint[..16]);
            fs::write(root.join(&file_name), &bytes).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to write Arrow prediction cache `{}`: {error}",
                    root.join(&file_name).display()
                ))
            })?;
            caches.push(ArrowPredictionCacheEntry {
                requirement_key: payload.requirement_key.clone(),
                cache_id: payload.cache_id.clone(),
                file_name,
                content_fingerprint: payload.content_fingerprint.clone(),
                ipc_fingerprint,
            });
        }
        caches.sort_by(|left, right| left.requirement_key.cmp(&right.requirement_key));
        let manifest = ArrowPredictionCacheManifest {
            schema_version: ARROW_PREDICTION_CACHE_STORE_SCHEMA_VERSION,
            bundle_id: bundle.bundle_id.clone(),
            caches,
        };
        manifest.validate_against_bundle(bundle)?;
        write_json(
            &root.join(ARROW_PREDICTION_CACHE_MANIFEST_FILE),
            &manifest,
            "Arrow prediction cache manifest",
        )?;
        Ok(manifest)
    }

    pub fn open(root: impl Into<PathBuf>, bundle: &ExecutionBundle) -> Result<Self> {
        let root = root.into();
        let manifest: ArrowPredictionCacheManifest = read_json(
            &root.join(ARROW_PREDICTION_CACHE_MANIFEST_FILE),
            "Arrow prediction cache manifest",
        )?;
        manifest.validate_against_bundle(bundle)?;

        let mut payloads_by_requirement = BTreeMap::new();
        for entry in &manifest.caches {
            let path = root.join(&entry.file_name);
            let bytes = fs::read(&path).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to read Arrow prediction cache `{}`: {error}",
                    path.display()
                ))
            })?;
            let actual_ipc_fingerprint = sha256_hex(&bytes);
            if actual_ipc_fingerprint != entry.ipc_fingerprint {
                return Err(DagMlError::RuntimeValidation(format!(
                    "Arrow prediction cache `{}` IPC fingerprint does not match its manifest",
                    entry.requirement_key
                )));
            }
            let payload = predictions_from_arrow_ipc(&bytes)?;
            if payload.requirement_key != entry.requirement_key
                || payload.cache_id != entry.cache_id
                || payload.content_fingerprint != entry.content_fingerprint
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "Arrow prediction cache `{}` metadata does not match its manifest",
                    entry.requirement_key
                )));
            }
            if payloads_by_requirement
                .insert(entry.requirement_key.clone(), payload)
                .is_some()
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "Arrow prediction cache store repeats requirement `{}`",
                    entry.requirement_key
                )));
            }
        }
        let payloads = BundlePredictionCachePayloadSet {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: bundle.schema_version,
            caches: payloads_by_requirement.into_values().collect(),
        };
        let inner = ColumnarPredictionCacheStore::from_payloads(bundle, payloads)?;
        Ok(Self {
            root,
            manifest,
            inner,
            materialization_records: RefCell::new(Vec::new()),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &ArrowPredictionCacheManifest {
        &self.manifest
    }

    pub fn materialization_records(&self) -> Vec<PredictionCacheMaterializationRecord> {
        self.materialization_records.borrow().clone()
    }
}

impl RuntimePredictionCacheStore for ArrowPredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> Result<Vec<PredictionBlock>> {
        self.inner.load_blocks(requirement_key)
    }

    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> Result<Vec<AggregatedPredictionBlock>> {
        self.inner.load_aggregated_blocks(requirement_key)
    }

    fn materialize(&self, request: &PredictionCacheMaterializationRequest) -> Result<HandleRef> {
        let handle = self.inner.materialize(request)?;
        let record = self
            .inner
            .materialization_records()
            .into_iter()
            .last()
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "Arrow prediction cache materialization did not record its handle".to_string(),
                )
            })?;
        self.materialization_records.borrow_mut().push(record);
        Ok(handle)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::RuntimeValidation(format!(
            "{label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, label: &str) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "failed to read {label} at `{}`: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "failed to parse {label} at `{}`: {error}",
            path.display()
        ))
    })
}

fn write_json<T: Serialize>(path: &Path, value: &T, label: &str) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        DagMlError::RuntimeValidation(format!("failed to serialize {label}: {error}"))
    })?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "failed to write {label} at `{}`: {error}",
            path.display()
        ))
    })
}

fn cache_schema(payload: &BundlePredictionCachePayload) -> Result<Schema> {
    let mut metadata = HashMap::new();
    metadata.insert(METADATA_KEY_FORMAT.to_string(), CODEC_VERSION.to_string());
    metadata.insert(
        METADATA_KEY_REQUIREMENT_KEY.to_string(),
        payload.requirement_key.clone(),
    );
    metadata.insert(METADATA_KEY_CACHE_ID.to_string(), payload.cache_id.clone());
    metadata.insert(
        METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS.to_string(),
        serde_json::to_string(&payload.cache_namespace_fingerprints).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize cache namespace fingerprints for Arrow metadata: {error}"
            ))
        })?,
    );
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
    metadata.insert(
        METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS.to_string(),
        serde_json::to_string(&payload.cache_namespace_fingerprints).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize cache namespace fingerprints for Arrow metadata: {error}"
            ))
        })?,
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
    if !matches!(format.as_str(), LEGACY_CODEC_VERSION | CODEC_VERSION) {
        return Err(DagMlError::RuntimeValidation(format!(
            "Arrow prediction cache uses codec version `{format}`, expected `{LEGACY_CODEC_VERSION}` or `{CODEC_VERSION}`"
        )));
    }
    let requirement_key = parse_metadata(&metadata, METADATA_KEY_REQUIREMENT_KEY)?;
    let cache_id = parse_metadata(&metadata, METADATA_KEY_CACHE_ID)?;
    let partition = parse_metadata_json(&metadata, METADATA_KEY_PARTITION)?;
    let prediction_level = parse_metadata_json(&metadata, METADATA_KEY_PREDICTION_LEVEL)?;
    let content_fingerprint = parse_metadata(&metadata, METADATA_KEY_CONTENT_FINGERPRINT)?;
    let block_count = parse_usize_metadata(&metadata, METADATA_KEY_BLOCK_COUNT)?;
    let row_count = parse_usize_metadata(&metadata, METADATA_KEY_ROW_COUNT)?;
    // This key was added after the initial v1 Arrow IPC payload. Its absence
    // is the historical empty namespace list, so old cache members remain
    // readable while new D10-enriched payloads round-trip losslessly.
    let cache_namespace_fingerprints = metadata
        .get(METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS)
        .map(|raw| {
            serde_json::from_str(raw).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "Arrow prediction cache metadata `{METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS}` is not valid JSON: {error}"
                ))
            })
        })
        .transpose()?
        .unwrap_or_default();

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
        cache_namespace_fingerprints,
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
            producer_port: Some("pred".to_string()),
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
            producer_port: Some("pred".to_string()),
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
            cache_namespace_fingerprints: vec!["a".repeat(64)],
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
            cache_namespace_fingerprints: Vec::new(),
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
    fn arrow_ipc_round_trips_cache_namespace_fingerprints() {
        let mut payload = sample_payload();
        payload.cache_namespace_fingerprints = vec!["a".repeat(64)];
        let bytes = predictions_to_arrow_ipc(&payload).expect("encode");
        let decoded = predictions_from_arrow_ipc(&bytes).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn arrow_ipc_reads_legacy_v1_without_cache_namespace_fingerprints() {
        let payload = sample_payload();
        let mut expected = payload.clone();
        expected.cache_namespace_fingerprints.clear();
        let mut schema = cache_schema(&payload).expect("schema");
        let mut legacy_metadata = schema.metadata.clone();
        legacy_metadata.insert(
            METADATA_KEY_FORMAT.to_string(),
            LEGACY_CODEC_VERSION.to_string(),
        );
        legacy_metadata.remove(METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS);
        schema = Schema::new_with_metadata(schema.fields.clone(), legacy_metadata);
        let batch = build_record_batch(&payload, schema.clone()).expect("batch");
        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).expect("writer");
            writer.write(&batch).expect("write batch");
            writer.finish().expect("finish stream");
        }
        let decoded = predictions_from_arrow_ipc(&buffer).expect("decode legacy v1");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn arrow_ipc_rejects_unknown_codec_version() {
        // Construct an Arrow IPC stream directly with an unsupported
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
