//! Export exact direct owner outputs for a container consumer, without archives.
use std::{collections::BTreeMap, fs, path::PathBuf};

use dag_ml_core::build_training_artifact_facets_from_bytes;
use serde_json::json;
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: artifact_facets OUTCOME_JSON PACKAGE_JSON NEW_OUTPUT_DIRECTORY".into());
    }
    let export =
        build_training_artifact_facets_from_bytes(&fs::read(&args[0])?, &fs::read(&args[1])?)?;
    let report = export.validate()?;
    let directory = PathBuf::from(&args[2]);
    // An immutable capture: refuse to overwrite an existing destination.
    fs::create_dir(&directory)?;
    fs::write(
        directory.join("facets.json"),
        serde_json::to_vec_pretty(&export.graph)?,
    )?;
    fs::write(
        directory.join("links.json"),
        serde_json::to_vec_pretty(&json!({
            "report_kind": "dagml.facet-link-diagnostics.v1",
            "authority": "diagnostic_only",
            "notice": "Non-authoritative diagnostics; recreate owner proofs from original bytes before use.",
            "links": report,
        }))?,
    )?;
    let inventory = export
        .members
        .iter()
        .map(|(path, bytes)| {
            (
                path,
                json!({"byte_length":bytes.len(), "sha256":format!("{:x}",Sha256::digest(bytes))}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    fs::write(
        directory.join("inventory.json"),
        serde_json::to_vec_pretty(&inventory)?,
    )?;
    for (path, bytes) in export.members {
        let path = directory.join(path);
        fs::create_dir_all(path.parent().ok_or("member has no parent directory")?)?;
        fs::write(path, bytes)?;
    }
    Ok(())
}
