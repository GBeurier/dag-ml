//! Bounded JSON bridge to the current candidate Core archive implementation.
//! Only generated in-process conformance fixtures cross stdin. This helper
//! accepts no paths and only reads archives it just wrote in a private tempdir.
use std::collections::BTreeMap;
use std::io::{Read, Write};

use nirs4all_archive_core::{
    load_archive, load_archive_v2, write_archive_v2, ArchivePayload, ArchiveV2WriteRequest,
    LoadedArchive,
};
use serde::Deserialize;
use serde_json::{json, Value};

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    path: String,
    bytes: Vec<u8>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    manifest: Value,
    payloads: Vec<Payload>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err("current oracle request exceeds 16 MiB".into());
    }
    let request: Request = serde_json::from_slice(&bytes)?;
    if request.schema_version != 1 || request.payloads.len() > 256 {
        return Err("unsupported current oracle protocol or payload count".into());
    }
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("generated-conformance.n4a");
    let payloads = request
        .payloads
        .into_iter()
        .map(|payload| ArchivePayload {
            path: payload.path,
            bytes: payload.bytes,
        })
        .collect::<Vec<_>>();
    let names = payloads
        .iter()
        .map(|payload| payload.path.clone())
        .collect::<Vec<_>>();
    let written = write_archive_v2(
        &path,
        ArchiveV2WriteRequest {
            manifest: request.manifest,
            payloads,
        },
    )?;
    let loaded = load_archive_v2(&path)?;
    let reference = |value: &nirs4all_archive_core::ArchiveV2Reference| {
        json!({
            "archive_id": value.archive_id(), "archive_sha256": value.archive_sha256(),
            "schema_version": value.schema_version(), "profile": value.profile(),
        })
    };
    let members = names
        .iter()
        .map(|name| {
            loaded
                .member(name)
                .map(|value| (name.clone(), value.to_vec()))
                .map_err(|error| format!("Core did not return requested member {name}: {error}"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    // Invoke the public typed accessor too, rather than treating every member
    // as a valid portable predictor package solely because its path exists.
    let portable_package = loaded.portable_predictor_package()?.to_vec();
    // Preserve the existing negative witness through the same exact Core API.
    // Flip the stored Package V2 member bytes, then demand load_archive_v2 fails.
    let mut tampered = std::fs::read(&path)?;
    let offset = tampered
        .windows(portable_package.len())
        .position(|window| window == portable_package.as_slice())
        .ok_or("stored ZIP does not contain exact Package V2 bytes")?;
    tampered[offset] ^= 1;
    let tampered_path = directory.path().join("tampered-conformance.n4a");
    std::fs::write(&tampered_path, tampered)?;
    let tampered_member_rejected = load_archive_v2(&tampered_path).is_err();
    if !tampered_member_rejected {
        return Err("current Core accepted a flipped Package V2 member".into());
    }
    let response = json!({
        "schema_version": 1, "oracle_package": "nirs4all", "oracle_version": "0.4.6",
        "writer_reference": reference(&written), "loaded_reference": reference(loaded.reference()),
        "loaded_is_v2": matches!(load_archive(&path)?, LoadedArchive::V2(_)),
        "members": members, "portable_package": portable_package,
        "tampered_member_rejected": tampered_member_rejected,
    });
    let response = serde_json::to_vec(&response)?;
    if response.len() as u64 > MAX_INPUT_BYTES {
        return Err("current oracle response exceeds 16 MiB".into());
    }
    std::io::stdout().write_all(&response)?;
    Ok(())
}
