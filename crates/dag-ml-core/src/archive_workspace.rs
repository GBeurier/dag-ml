//! Archive V2 replay-member assembly owned by DAG-ML.
//!
//! This module deliberately does not write ZIPs or read archives.  It turns a
//! fully validated native training result into the exact DAG-ML document bytes
//! and manifest references required by ADR-23; `nirs4all-core` remains the
//! sole owner of bounded archive storage and inventory validation.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::canonical::parse_typed_json;
use crate::error::{DagMlError, Result};
use crate::runtime::ArtifactBackend;
use crate::training::{ArtifactLoadMode, FittedArtifactMode, PortablePredictorPackage};
use crate::training_runtime::TrainingOutcome;

pub const ARCHIVE_V2_PACKAGE_MEMBER: &str = "dagml/portable_predictor_package.json";
pub const ARCHIVE_V2_GRAPH_MEMBER: &str = "dagml/graph.json";
pub const ARCHIVE_V2_BUNDLE_MEMBER: &str = "dagml/execution_bundle.json";
pub const ARCHIVE_V2_OUTCOME_MEMBER: &str = "dagml/training_outcome.json";
pub const ARCHIVE_V2_CACHE_MEMBER: &str = "dagml/prediction_cache_payload_set.json";
pub const ARCHIVE_V2_SCORE_MEMBER: &str = "dagml/score_set.json";

const PACKAGE_SCHEMA: &str =
    "https://github.com/GBeurier/dag-ml/schemas/portable_predictor_package.v2.schema.json";
const GRAPH_SCHEMA: &str = "https://github.com/GBeurier/dag-ml/schemas/graph_spec.v1.schema.json";
const BUNDLE_SCHEMA: &str =
    "https://github.com/GBeurier/dag-ml/schemas/execution_bundle.v2.schema.json";
const OUTCOME_SCHEMA: &str =
    "https://github.com/GBeurier/dag-ml/schemas/training_outcome.v2.schema.json";
const CACHE_SCHEMA: &str =
    "https://github.com/GBeurier/dag-ml/schemas/prediction_cache_payload_set.v2.schema.json";
const SCORE_SCHEMA: &str = "https://github.com/GBeurier/dag-ml/schemas/score_set.v2.schema.json";

/// Exact bytes and manifest handed to the Core Archive V2 writer.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveV2ReplayPayloads {
    pub manifest: Value,
    pub members: BTreeMap<String, Vec<u8>>,
}

/// Assemble the strict ADR-23 P0 replay closure from real DAG-ML contracts.
///
/// This fails closed instead of creating cache/score placeholders or changing
/// an artifact URI.  In particular, portable packages that are valid for a
/// host-sidecar deployment are intentionally not Archive V2 P0 candidates.
pub fn build_archive_v2_native_portable_payloads(
    archive_id: impl Into<String>,
    outcome: &TrainingOutcome,
    package: &PortablePredictorPackage,
) -> Result<ArchiveV2ReplayPayloads> {
    outcome.validate()?;
    package.validate()?;
    let archive_id = archive_id.into();
    if archive_id.is_empty()
        || archive_id.len() > 128
        || !archive_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
    {
        return refuse("archive V2 archive_id is not a portable identifier");
    }
    if package.schema_version != 2 || outcome.schema_version != 2 {
        return refuse("Archive V2 requires Package and TrainingOutcome schema V2");
    }
    if package.fitted_artifact_mode != FittedArtifactMode::PortableRequired
        || package
            .artifact_bindings
            .iter()
            .any(|binding| binding.load_mode != ArtifactLoadMode::NativePortable)
    {
        return refuse("Archive V2 P0 refuses host-sidecar package artifacts");
    }
    if package.training_outcome != outcome.to_reference()?
        || package.execution_bundle != outcome.execution_bundle
        || package.effective_plan != outcome.effective_plan
        || package.template.graph != outcome.effective_plan.graph_plan.graph
    {
        return refuse("Archive V2 package does not exactly cross-link its TrainingOutcome");
    }
    let caches = outcome.portable_prediction_caches.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation(
            "Archive V2 requires retained prediction-cache payload set; it will not synthesize one"
                .to_string(),
        )
    })?;
    caches.validate_against_bundle(&outcome.execution_bundle)?;
    if caches.schema_version != 2 || outcome.score_set.schema_version != 2 {
        return refuse("Archive V2 requires V2 prediction-cache and score-set companions");
    }

    let mut members = BTreeMap::new();
    insert_json(&mut members, ARCHIVE_V2_PACKAGE_MEMBER, package)?;
    insert_json(
        &mut members,
        ARCHIVE_V2_GRAPH_MEMBER,
        &package.template.graph,
    )?;
    insert_json(
        &mut members,
        ARCHIVE_V2_BUNDLE_MEMBER,
        &package.execution_bundle,
    )?;
    insert_json(&mut members, ARCHIVE_V2_OUTCOME_MEMBER, outcome)?;
    insert_json(&mut members, ARCHIVE_V2_CACHE_MEMBER, caches)?;
    insert_json(&mut members, ARCHIVE_V2_SCORE_MEMBER, &outcome.score_set)?;

    let mut n4mm = Vec::new();
    for record in &package.execution_bundle.refit_artifacts {
        let artifact = &record.artifact;
        if artifact.kind != "n4m_model"
            || artifact.backend != Some(ArtifactBackend::Raw)
            || artifact.plugin.is_some()
            || artifact.plugin_version.is_some()
        {
            return refuse("Archive V2 P0 accepts only raw plugin-free n4m_model refit artifacts");
        }
        let path = artifact.uri.as_deref().ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "Archive V2 P0 N4MM artifact has no archive member URI".to_string(),
            )
        })?;
        if !safe_n4mm_path(path) {
            return refuse("Archive V2 P0 N4MM URI must be a safe methods/*.n4mm path");
        }
        let bytes = package
            .execution_bundle
            .raw_artifact_payloads
            .get(&artifact.id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "Archive V2 P0 lacks raw N4MM payload `{}`",
                    artifact.id
                ))
            })?
            .clone();
        if artifact.size_bytes != Some(bytes.len() as u64) {
            return refuse("Archive V2 P0 N4MM size does not match raw payload");
        }
        let raw = sha256(&bytes);
        if artifact.content_fingerprint.as_deref() != Some(raw.as_str()) {
            return refuse("Archive V2 P0 N4MM raw SHA-256 does not match artifact fingerprint");
        }
        if members.insert(path.to_owned(), bytes).is_some() {
            return refuse("Archive V2 P0 N4MM paths must be unique");
        }
        n4mm.push(json!({
            "artifact_id": artifact.id,
            "kind": "N4MM",
            "owner": "nirs4all-methods",
            "format_version": 1,
            "abi_major": 2,
            "member_path": path,
            "raw_sha256": raw,
            "semantic_fingerprint": raw,
            "semantic_profile": "n4mm_raw_sha256"
        }));
    }
    if n4mm.is_empty()
        || package.execution_bundle.raw_artifact_payloads.len() != n4mm.len()
        || package.artifact_bindings.len() != n4mm.len()
    {
        return refuse("Archive V2 P0 N4MM members must exactly cover all package refit artifacts");
    }

    let package_semantic = package.package_fingerprint.clone();
    let mut manifest = json!({
        "schema_version": 2,
        "profile": "nirs4all.archive_workspace.v2",
        "archive_id": archive_id,
        "persistence_kind": "n4a_archive",
        "writer": {"product_aggregate_owner": "nirs4all-core", "canonical_writer_id": "nirs4all-core.archive_workspace_writer.v2"},
        "reader_dispatch": {
            "archive_v2": {"accepted_versions": [2], "future_versions": "refuse", "dispatch_before_extraction": true},
            "archive_v1": {"accepted_versions": [1], "read_mode": "immutable_dual_read", "mutation": "never_in_place"},
            "legacy_n4a": {"form": "historical_n4a_zip", "manifest_member": "manifest.json", "reader_id": "nirs4all.pipeline.bundle.loader.BundleLoader", "maximum_bundle_format_version": "1.0", "migration_direction": "legacy_to_v1_copy_on_write_only"}
        },
        "physical_profile": {"container": "zip", "manifest_member": "manifest.json", "regular_files_only": true, "limits": {"max_entries": 256, "max_total_uncompressed_bytes": 536870912_u64, "max_member_uncompressed_bytes": 134217728_u64, "max_compression_ratio": 100}},
        "replay": {
            "portable_predictor_package": dag_ref(ARCHIVE_V2_PACKAGE_MEMBER, PACKAGE_SCHEMA, 2, true, "dagml_tcv1", package_semantic),
            "training_artifacts": {
                "graph": dag_ref(ARCHIVE_V2_GRAPH_MEMBER, GRAPH_SCHEMA, 1, false, "dagml_historical_serde_json_v1", historical_fingerprint(members.get(ARCHIVE_V2_GRAPH_MEMBER).expect("inserted graph"))),
                "execution_bundle": dag_ref(ARCHIVE_V2_BUNDLE_MEMBER, BUNDLE_SCHEMA, 2, true, "dagml_tcv1", tcv1_bytes(members.get(ARCHIVE_V2_BUNDLE_MEMBER).expect("inserted bundle"))?),
                "training_outcome": dag_ref(ARCHIVE_V2_OUTCOME_MEMBER, OUTCOME_SCHEMA, 2, true, "dagml_tcv1", outcome.outcome_fingerprint.clone()),
                "prediction_cache_payload_set": dag_ref(ARCHIVE_V2_CACHE_MEMBER, CACHE_SCHEMA, 2, true, "dagml_historical_serde_json_v1", historical_fingerprint(members.get(ARCHIVE_V2_CACHE_MEMBER).expect("inserted cache"))),
                "score_set": dag_ref(ARCHIVE_V2_SCORE_MEMBER, SCORE_SCHEMA, 2, true, "dagml_historical_serde_json_v1", historical_fingerprint(members.get(ARCHIVE_V2_SCORE_MEMBER).expect("inserted scores")))
            },
            "future_artifacts": []
        },
        "payloads": {"methods": {"n4mm": n4mm, "n4mopt": []}, "n4d_aggregate_reference": null, "conformal": null, "robustness": null, "host_artifacts": []},
        "member_inventory": [],
        "migration_provenance": null,
        "security": {"integrity_profile": "sha256_raw_member_inventory_v2", "signature": null},
        "workspace": null
    });
    let inventory = members
        .iter()
        .map(|(path, bytes)| {
            let (semantic_profile, semantic_fingerprint) = if path == ARCHIVE_V2_PACKAGE_MEMBER {
                ("dagml_tcv1", package.package_fingerprint.clone())
            } else if path.ends_with(".n4mm") {
                ("n4mm_raw_sha256", sha256(bytes))
            } else if path == ARCHIVE_V2_BUNDLE_MEMBER {
                ("dagml_tcv1", tcv1_bytes(bytes).expect("serialized TCV1 document"))
            } else if path == ARCHIVE_V2_OUTCOME_MEMBER {
                ("dagml_tcv1", outcome.outcome_fingerprint.clone())
            } else {
                ("dagml_historical_serde_json_v1", historical_fingerprint(bytes))
            };
            json!({"path": path, "regular_file": true, "raw_sha256": sha256(bytes), "uncompressed_size_bytes": bytes.len(), "semantic_fingerprint": semantic_fingerprint, "semantic_profile": semantic_profile})
        })
        .collect::<Vec<_>>();
    manifest["member_inventory"] = Value::Array(inventory);
    bind_raw_hashes(&mut manifest, &members);
    Ok(ArchiveV2ReplayPayloads { manifest, members })
}

fn insert_json<T: serde::Serialize>(
    members: &mut BTreeMap<String, Vec<u8>>,
    path: &str,
    value: &T,
) -> Result<()> {
    members.insert(path.to_owned(), serde_json::to_vec(value)?);
    Ok(())
}

fn dag_ref(
    path: &str,
    schema_id: &str,
    schema_version: u64,
    producer_port_required: bool,
    semantic_profile: &str,
    semantic_fingerprint: String,
) -> Value {
    let mut reference = json!({
        "owner": "dag-ml",
        "schema_id": schema_id,
        "schema_version": schema_version,
        "member_path": path,
        "raw_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "semantic_fingerprint": semantic_fingerprint,
        "semantic_profile": semantic_profile
    });
    if producer_port_required {
        reference["producer_port_required"] = Value::Bool(true);
    }
    reference
}

fn tcv1_bytes(bytes: &[u8]) -> Result<String> {
    parse_typed_json(std::str::from_utf8(bytes).map_err(|error| {
        DagMlError::RuntimeValidation(format!("Archive V2 DAG-ML JSON was not UTF-8: {error}"))
    })?)
    .map_err(|error| {
        DagMlError::RuntimeValidation(format!("Archive V2 DAG-ML JSON was not TCV1: {error}"))
    })?
    .fingerprint()
    .map_err(|error| {
        DagMlError::RuntimeValidation(format!("Archive V2 TCV1 fingerprint failed: {error}"))
    })
}

fn historical_fingerprint(bytes: &[u8]) -> String {
    sha256(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Core recomputes these during its final atomic write.  Binding them here as
/// well keeps the DAG-ML handoff self-consistent for callers that validate the
/// manifest before handing its bytes to Core.
fn bind_raw_hashes(value: &mut Value, members: &BTreeMap<String, Vec<u8>>) {
    match value {
        Value::Object(object) => {
            if let Some(path) = object.get("member_path").and_then(Value::as_str) {
                if let Some(bytes) = members.get(path) {
                    object.insert("raw_sha256".to_string(), Value::String(sha256(bytes)));
                }
            }
            for child in object.values_mut() {
                bind_raw_hashes(child, members);
            }
        }
        Value::Array(items) => {
            for item in items {
                bind_raw_hashes(item, members);
            }
        }
        _ => {}
    }
}

fn safe_n4mm_path(path: &str) -> bool {
    path.starts_with("methods/")
        && path.ends_with(".n4mm")
        && path.len() <= 512
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn refuse<T>(message: &str) -> Result<T> {
    Err(DagMlError::RuntimeValidation(message.to_string()))
}
