use std::fs;
use std::io::Cursor;
use std::path::Path;

use arrow_ipc::reader::StreamReader;
use dag_ml_arrow::predictions_from_arrow_ipc;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[test]
fn arrow59_public_decoder_reads_frozen_rust53_ipc() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/arrow53");
    let bytes = fs::read(root.join("cache.arrow"))
        .expect("mandatory versioned Rust53 IPC fixture absent; see scripts/arrow53-compat-fixtures/README.md");
    let provenance: Value = serde_json::from_slice(
        &fs::read(root.join("provenance.json"))
            .expect("mandatory Rust53 fixture provenance absent"),
    )
    .unwrap();
    assert_eq!(provenance["producer"], "dag-ml-arrow53-fixture-producer");
    for dependency in ["arrow-array", "arrow-ipc", "arrow-schema", "parquet"] {
        assert_eq!(provenance["dependencies"][dependency], "53.4.1");
    }
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        provenance["files"]["cache.arrow"].as_str().unwrap()
    );
    let reader = StreamReader::try_new(Cursor::new(&bytes), None).unwrap();
    assert_eq!(
        reader.schema().metadata()["fixture.producer"],
        "rust-arrow-ipc-53.4.1"
    );

    let payload = predictions_from_arrow_ipc(&bytes)
        .expect("public candidate reader must accept old Rust53 bytes");
    assert_eq!(payload.requirement_key, "requirement:53");
    assert_eq!(payload.cache_id, "cache:53");
    assert_eq!(payload.block_count, 1);
    assert_eq!(payload.row_count, 2);
    assert_eq!(payload.cache_namespace_fingerprints, vec!["a".repeat(64)]);
    assert!(payload.aggregated_blocks.is_empty());
    let block = &payload.blocks[0];
    assert_eq!(
        block
            .sample_ids
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        vec!["sample:11", "sample:7"]
    );
    assert_eq!(block.values, vec![vec![1.25, -2.5], vec![3.75, 4.5]]);
    assert_eq!(block.target_names, vec!["target:a", "target:b"]);
    assert_eq!(block.producer_node.as_str(), "model:53");
    assert_eq!(block.producer_port.as_deref(), Some("pred"));
    assert_eq!(block.fold_id.as_ref().unwrap().as_str(), "fold:0");
    assert_eq!(
        payload.content_fingerprint,
        provenance["ipc_content_fingerprint"].as_str().unwrap()
    );
    assert_eq!(
        payload.content_fingerprint,
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&payload.blocks).unwrap())
        )
    );
}
