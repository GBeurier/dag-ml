# Cross-language canonical fingerprint oracle

This directory contains an independent, test-only Rust implementation of the two
canonical profiles that must never be confused:

- DAG-ML Typed Canonical Value v1 (TCV1): NFC-normalized text, UTF-8 object-key
  order, typed integer/binary64 encoding, and the `DAGML-TCV1\0` domain prefix;
- the restricted RFC 8785/JCS domain used by `OrderedSearchSpaceSpec` V1: no
  Unicode normalization, UTF-16 object-key order, safe non-negative structural
  integers, and binary64-derived values carried as canonical strings.

The Rust oracle parses JSON itself. It therefore preserves the normative
difference between an integer token such as `2` and a binary64 token such as
`2.0`, rejects duplicate members and unpaired surrogates, and does not import
the DAG-ML production crates or the Python reference oracle.

`golden/tcv1_jcs_cross_language.v1.json` pins key-order, NFC, signed-zero,
integer-versus-float, subnormal, normal, 2^53, and maximum-finite vectors. The
Python test compares every TCV1 preimage and digest byte-for-byte with
`parity.conformal.oracle`, while an independent Python restricted-JCS renderer
checks the Rust canonical bytes.

Run the isolated gates from the repository root:

```bash
CARGO_TARGET_DIR=/tmp/dagml-canonical-rust-target \
  cargo test --locked --manifest-path parity/canonical/rust-oracle/Cargo.toml
python3 -m pytest parity/canonical/tests/test_rust_oracle_parity.py -q
```

The crate is deliberately outside the production Cargo workspace. Its locked
dependencies are test-only; `unicode-normalization` supplies the complete NFC
tables instead of a fixture-specific approximation. After dependencies have
been cached, both commands can be run with `CARGO_NET_OFFLINE=true`.
