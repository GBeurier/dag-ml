use dagml_canonical_rust_oracle::{
    lowercase_hex, parse_strict_json, restricted_jcs_bytes, restricted_jcs_fingerprint,
    tcv1_preimage, tcv1_sha256,
};
use sha2::{Digest, Sha256};
use std::io::{self, Read};

fn main() {
    if let Err(error) = run() {
        eprintln!("canonical oracle error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.len() != 2 || !matches!(arguments[1].as_str(), "tcv1" | "jcs") {
        return Err("usage: dagml-canonical-rust-oracle <tcv1|jcs> < document.json".to_string());
    }

    let mut document = String::new();
    io::stdin()
        .read_to_string(&mut document)
        .map_err(|error| format!("stdin is not strict UTF-8: {error}"))?;
    let value = parse_strict_json(&document)?;

    match arguments[1].as_str() {
        "tcv1" => {
            let preimage = tcv1_preimage(&value)?;
            println!("canonical_hex={}", lowercase_hex(&preimage));
            println!("sha256={}", tcv1_sha256(&value)?);
        }
        "jcs" => {
            let canonical = restricted_jcs_bytes(&value)?;
            println!("canonical_hex={}", lowercase_hex(&canonical));
            println!("sha256={}", lowercase_hex(&Sha256::digest(&canonical)));
            println!("fingerprint={}", restricted_jcs_fingerprint(&value)?);
        }
        _ => unreachable!(),
    }
    Ok(())
}
