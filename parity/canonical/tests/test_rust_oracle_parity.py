"""Cross-language TCV1/restricted-JCS parity against the Python oracle."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import load_json, tcv1_preimage, tcv1_sha256  # noqa: E402

GOLDEN = ROOT / "parity" / "canonical" / "golden" / "tcv1_jcs_cross_language.v1.json"
MANIFEST = ROOT / "parity" / "canonical" / "rust-oracle" / "Cargo.toml"
BINARY_NAME = "dagml-canonical-rust-oracle"


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    return load_json(GOLDEN)


@pytest.fixture(scope="session")
def rust_oracle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("canonical-rust-target")
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--locked",
            "--manifest-path",
            str(MANIFEST),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    binary = target / "debug" / f"{BINARY_NAME}{suffix}"
    assert binary.is_file(), f"Rust oracle binary was not built at {binary}"
    return binary


def run_oracle(binary: Path, profile: str, document_json: str) -> dict[str, str]:
    process = subprocess.run(
        [str(binary), profile],
        input=document_json,
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in process.stdout.splitlines())


def python_restricted_jcs(value: Any) -> bytes:
    """Independent Python rendering of the OrderedSearchSpaceSpec V1 domain."""

    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    if isinstance(value, int):
        if not 0 <= value <= 2**53 - 1:
            raise ValueError("restricted JCS structural integer is outside 0..2^53-1")
        return str(value).encode()
    if isinstance(value, float):
        raise ValueError("restricted JCS represents binary64-derived values as strings")
    if isinstance(value, list):
        return b"[" + b",".join(python_restricted_jcs(item) for item in value) + b"]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        members = (
            python_restricted_jcs(key) + b":" + python_restricted_jcs(value[key])
            for key in keys
        )
        return b"{" + b",".join(members) + b"}"
    raise TypeError(f"unsupported restricted JCS value: {type(value).__name__}")


def test_golden_covers_all_profile_discriminators(golden: dict[str, Any]) -> None:
    tcv1_ids = {vector["id"] for vector in golden["tcv1_vectors"]}
    jcs_ids = {vector["id"] for vector in golden["restricted_jcs_vectors"]}
    assert {
        "utf8_key_order_differs_from_utf16",
        "unicode_nfc",
        "negative_zero",
        "integer_two",
        "binary64_integral_two",
        "binary64_min_subnormal",
        "binary64_largest_subnormal",
        "binary64_min_normal",
        "binary64_two_pow_53",
        "binary64_max_finite",
    } <= tcv1_ids
    assert {
        "utf16_key_order_differs_from_utf8",
        "unicode_nfc_is_not_applied",
        "binary64_labels_are_strings",
    } <= jcs_ids


def test_rust_tcv1_matches_python_oracle_byte_for_byte(
    golden: dict[str, Any], rust_oracle: Path
) -> None:
    for vector in golden["tcv1_vectors"]:
        value = json.loads(vector["document_json"])
        python_preimage = tcv1_preimage(value).hex()
        python_digest = tcv1_sha256(value)
        rust = run_oracle(rust_oracle, "tcv1", vector["document_json"])

        assert python_preimage == vector["expected_preimage_hex"], vector["id"]
        assert python_digest == vector["expected_sha256"], vector["id"]
        assert rust["canonical_hex"] == python_preimage, vector["id"]
        assert rust["sha256"] == python_digest, vector["id"]

        if "equivalent_json" in vector:
            equivalent_value = json.loads(vector["equivalent_json"])
            equivalent_rust = run_oracle(rust_oracle, "tcv1", vector["equivalent_json"])
            assert tcv1_preimage(equivalent_value).hex() == python_preimage
            assert equivalent_rust == rust


def test_unicode_nfc_expected_bytes_are_composed_utf8(
    golden: dict[str, Any], rust_oracle: Path
) -> None:
    vector = next(
        vector for vector in golden["tcv1_vectors"] if vector["id"] == "unicode_nfc"
    )
    expected = bytes.fromhex(vector["expected_preimage_hex"])
    assert expected.endswith(b"S\x00\x00\x00\x00\x00\x00\x00\x02\xc3\xa9")
    assert b"e\xcc\x81" not in expected
    assert (
        run_oracle(rust_oracle, "tcv1", vector["document_json"])["canonical_hex"]
        == expected.hex()
    )


def test_rust_tcv1_matches_python_on_nested_and_boundary_matrix(
    rust_oracle: Path,
) -> None:
    documents = [
        None,
        False,
        True,
        -(2**63),
        -1,
        0,
        2**64 - 1,
        -0.0,
        5e-324,
        0.1,
        1.7976931348623157e308,
        "e\u0301",
        [None, False, -1, 2.0, "é"],
        {
            "": [2**64 - 1, -0.0],
            "𐀀": {"nested": "e\u0301", "empty": {}},
        },
    ]
    for value in documents:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        rust = run_oracle(rust_oracle, "tcv1", raw)
        assert rust["canonical_hex"] == tcv1_preimage(value).hex(), raw
        assert rust["sha256"] == tcv1_sha256(value), raw


def test_binary64_vectors_pin_bits_labels_and_integer_type(
    golden: dict[str, Any], rust_oracle: Path
) -> None:
    vectors = {vector["id"]: vector for vector in golden["tcv1_vectors"]}
    binary64_vectors = [
        vector for vector in vectors.values() if "binary64_be_hex" in vector
    ]
    labels = []
    for vector in binary64_vectors:
        value = json.loads(vector["document_json"])
        assert isinstance(value, float), vector["id"]
        assert struct.pack(">d", value).hex() == vector["binary64_be_hex"]
        labels.append(vector["binary64_label"])

        rust = run_oracle(rust_oracle, "tcv1", vector["document_json"])
        canonical_binary64 = (
            "0000000000000000" if value == 0.0 else vector["binary64_be_hex"]
        )
        assert rust["canonical_hex"].endswith(f"44{canonical_binary64}")

    assert len(labels) == len(set(labels)), "binary64 wire labels must be unique"
    assert (
        vectors["integer_two"]["expected_sha256"]
        != vectors["binary64_integral_two"]["expected_sha256"]
    )


def test_restricted_jcs_vectors_are_stable_and_keep_binary64_labels_as_text(
    golden: dict[str, Any], rust_oracle: Path
) -> None:
    by_id = {vector["id"]: vector for vector in golden["restricted_jcs_vectors"]}
    for vector in by_id.values():
        rust = run_oracle(rust_oracle, "jcs", vector["document_json"])
        python_canonical = python_restricted_jcs(json.loads(vector["document_json"]))
        assert rust["canonical_hex"] == vector["expected_canonical_hex"], vector["id"]
        assert bytes.fromhex(rust["canonical_hex"]) == python_canonical, vector["id"]
        assert rust["sha256"] == vector["expected_sha256"], vector["id"]
        assert rust["fingerprint"] == vector["expected_fingerprint"], vector["id"]
        assert (
            hashlib.sha256(bytes.fromhex(rust["canonical_hex"])).hexdigest()
            == rust["sha256"]
        )

        if "equivalent_json" in vector:
            equivalent = run_oracle(rust_oracle, "jcs", vector["equivalent_json"])
            assert equivalent == rust
        if "non_equivalent_json" in vector:
            non_equivalent = run_oracle(
                rust_oracle, "jcs", vector["non_equivalent_json"]
            )
            assert non_equivalent["sha256"] == vector["non_equivalent_sha256"]
            assert non_equivalent["sha256"] != rust["sha256"]

    label_document = json.loads(by_id["binary64_labels_are_strings"]["document_json"])
    assert all(isinstance(label, str) for label in label_document["labels"])
    assert label_document["labels"] == [
        "0",
        "-0",
        "4.9406564584124654e-324",
        "2.2250738585072009e-308",
        "2.2250738585072014e-308",
        "1.0000000000000001e-05",
        "9007199254740992",
        "-9007199254740992",
        "1.7976931348623157e+308",
    ]


def test_utf8_and_utf16_profiles_are_observably_disjoint(
    golden: dict[str, Any], rust_oracle: Path
) -> None:
    raw = '{"\\ue000":1,"\\ud800\\udc00":2}'
    tcv1 = bytes.fromhex(run_oracle(rust_oracle, "tcv1", raw)["canonical_hex"])
    jcs = bytes.fromhex(run_oracle(rust_oracle, "jcs", raw)["canonical_hex"])

    assert tcv1.index("".encode()) < tcv1.index("𐀀".encode())
    assert jcs.decode() == '{"𐀀":2,"":1}'

    tcv1_vector = next(
        vector
        for vector in golden["tcv1_vectors"]
        if vector["id"] == "utf8_key_order_differs_from_utf16"
    )
    jcs_vector = next(
        vector
        for vector in golden["restricted_jcs_vectors"]
        if vector["id"] == "utf16_key_order_differs_from_utf8"
    )
    assert tcv1_vector["expected_sha256"] != jcs_vector["expected_sha256"]


def test_rust_oracle_fails_closed_on_invalid_domain_vectors(
    golden: dict[str, Any], rust_oracle: Path
) -> None:
    for vector in golden["invalid_vectors"]:
        process = subprocess.run(
            [str(rust_oracle), vector["profile"]],
            input=vector["document_json"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 2, vector["id"]
        assert vector["error_contains"] in process.stderr, vector["id"]
