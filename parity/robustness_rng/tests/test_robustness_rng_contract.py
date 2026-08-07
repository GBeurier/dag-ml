"""Strict tests for the independent robustness counter/RNG oracle."""

from __future__ import annotations

import ast
import hashlib
import math
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from parity.robustness_rng.oracle import (  # noqa: E402
    PHILOX_ALGORITHM,
    PORTABLE_INTEGER_MAX,
    PROFILE_NAME,
    TARGET_KINDS,
    OracleError,
    derive_key_words,
    derive_robustness_block,
    load_json,
    philox4x32_10,
)

GOLDEN = ROOT / "parity" / "robustness_rng" / "golden" / "philox4x32_10_counter.v1.json"
ORACLE_SOURCE = ROOT / "parity" / "robustness_rng" / "oracle.py"
SCENARIO_SCHEMA = ROOT / "docs" / "contracts" / "robustness_scenario_spec.schema.json"


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    return load_json(GOLDEN)


def _hex_words(words: tuple[int, ...]) -> list[str]:
    return [f"{word:08x}" for word in words]


def test_profile_metadata_pins_every_cross_language_choice(
    golden: dict[str, Any],
) -> None:
    profile = golden["profile"]
    assert golden["fixture_id"] == "dag-ml.robustness-counter.philox4x32-10.v1"
    assert golden["schema_version"] == 1
    assert profile == {
        "counter_profile": PROFILE_NAME,
        "algorithm": PHILOX_ALGORITHM,
        "algorithm_version": 1,
        "portable_integer_range": "0..2^53-1",
        "key_derivation": "uint64-seed-as-two-little-endian-u32",
        "key_word_mapping": "key[0]=seed low 32 bits; key[1]=seed bits 32..63",
        "counter_fields": [
            "scenario_fingerprint",
            "severity_binary64",
            "unit_id",
            "target_kind",
            "target_id",
            "draw_index",
        ],
        "counter_derivation": "sha256-tcv1-first128",
        "counter_preimage": "DAGML-TCV1 NUL-prefixed exact counter_fields payload",
        "counter_word_order": "four consecutive big-endian u32 words",
        "text_normalization": "TCV1 NFC",
        "negative_zero": "severity binary64 -0 canonicalizes to +0",
        "target_semantics": (
            "source/node require non-empty target_id; global requires empty target_id"
        ),
    }


def test_profile_metadata_matches_robustness_scenario_schema(
    golden: dict[str, Any],
) -> None:
    profile = golden["profile"]
    rng_contract = load_json(SCENARIO_SCHEMA)["$defs"]["rng_derivation"]
    properties = rng_contract["properties"]

    assert properties["algorithm"]["const"] == profile["algorithm"]
    assert properties["algorithm_version"]["const"] == profile["algorithm_version"]
    assert properties["counter_profile"]["const"] == profile["counter_profile"]
    assert properties["counter_derivation"]["const"] == profile["counter_derivation"]
    assert properties["counter_fields"]["const"] == profile["counter_fields"]
    assert properties["key_derivation"]["const"] == profile["key_derivation"]
    assert set(properties["target_kind"]["enum"]) == TARGET_KINDS
    assert properties["seed"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": PORTABLE_INTEGER_MAX,
    }


def test_philox_matches_random123_known_answer_vectors(
    golden: dict[str, Any],
) -> None:
    assert len(golden["philox_known_answer_vectors"]) == 3
    for vector in golden["philox_known_answer_vectors"]:
        counter = tuple(int(word, 16) for word in vector["counter_words_hex"])
        key = tuple(int(word, 16) for word in vector["key_words_hex"])
        assert _hex_words(philox4x32_10(counter, key)) == vector["output_words_hex"], (
            vector["id"]
        )


def test_golden_vectors_pin_tcv1_sha_counter_key_and_output(
    golden: dict[str, Any],
) -> None:
    for vector in golden["vectors"]:
        actual = derive_robustness_block(**vector["input"])
        expected = vector["expected"]
        assert actual == expected, vector["id"]

        preimage = bytes.fromhex(expected["tcv1_preimage_hex"])
        digest = hashlib.sha256(preimage).digest()
        assert preimage.startswith(b"DAGML-TCV1\0A")
        assert digest.hex() == expected["counter_sha256"]
        assert digest[:16].hex() == expected["counter_128_hex"]
        assert list(struct.unpack(">4I", digest[:16])) == expected["counter_words"]
        assert [int(word, 16) for word in expected["counter_words_hex"]] == expected[
            "counter_words"
        ]
        assert [int(word, 16) for word in expected["key_words_hex"]] == expected[
            "key_words"
        ]
        assert [int(word, 16) for word in expected["output_words_hex"]] == expected[
            "output_words"
        ]
        assert "".join(expected["output_words_hex"]) == expected["output_128_hex"]


def test_vectors_cover_source_node_global_unicode_zero_and_draw_index(
    golden: dict[str, Any],
) -> None:
    vectors = {vector["id"]: vector for vector in golden["vectors"]}
    assert {vector["input"]["target_kind"] for vector in vectors.values()} == {
        "source",
        "node",
        "global",
    }

    source_0 = vectors["source_draw_0"]
    source_1 = vectors["source_draw_1"]
    assert source_0["input"] | {"draw_index": 1} == source_1["input"]
    assert (
        source_0["expected"]["counter_128_hex"]
        != source_1["expected"]["counter_128_hex"]
    )
    assert (
        source_0["expected"]["output_128_hex"] != source_1["expected"]["output_128_hex"]
    )

    unicode_vector = vectors["node_unicode_nfc"]
    assert (
        unicode_vector["input"]["unit_id"]
        != unicode_vector["equivalent_input"]["unit_id"]
    )
    assert (
        derive_robustness_block(**unicode_vector["equivalent_input"])
        == unicode_vector["expected"]
    )
    assert unicode_vector["expected"]["normalized_payload"][2] == "sample:é:α"

    zero_vector = vectors["global_negative_zero"]
    assert math.copysign(1.0, zero_vector["input"]["severity"]) == -1.0
    assert (
        derive_robustness_block(**zero_vector["equivalent_input"])
        == zero_vector["expected"]
    )
    normalized_zero = zero_vector["expected"]["normalized_payload"][1]
    assert normalized_zero == 0.0
    assert math.copysign(1.0, normalized_zero) == 1.0


def test_portable_seed_key_mapping_is_little_limb_first() -> None:
    assert derive_key_words(0) == (0, 0)
    assert derive_key_words(17) == (17, 0)
    assert derive_key_words(2**32) == (0, 1)
    assert derive_key_words(PORTABLE_INTEGER_MAX) == (0xFFFFFFFF, 0x001FFFFF)


def test_each_identity_axis_domain_separates_the_counter() -> None:
    common = {
        "seed": 7,
        "scenario_fingerprint": "a" * 64,
        "severity": 0.5,
        "unit_id": "unit:1",
        "target_id": "nir",
        "draw_index": 3,
    }
    inputs = [
        common | {"target_kind": "source"},
        common | {"target_kind": "source", "scenario_fingerprint": "b" * 64},
        common | {"target_kind": "source", "severity": 0.6},
        common | {"target_kind": "source", "unit_id": "unit:2"},
        common | {"target_kind": "node"},
        common | {"target_kind": "source", "target_id": "nir.secondary"},
        common | {"target_kind": "source", "draw_index": 4},
        common | {"target_kind": "global", "target_id": ""},
    ]
    counters = {derive_robustness_block(**item)["counter_128_hex"] for item in inputs}
    assert len(counters) == len(inputs)

    base = derive_robustness_block(**inputs[0])
    other_seed = derive_robustness_block(**(inputs[0] | {"seed": 8}))
    assert other_seed["counter_128_hex"] == base["counter_128_hex"]
    assert other_seed["key_words"] != base["key_words"]
    assert other_seed["output_128_hex"] != base["output_128_hex"]


def test_maximum_portable_draw_index_is_accepted() -> None:
    result = derive_robustness_block(
        seed=PORTABLE_INTEGER_MAX,
        scenario_fingerprint="f" * 64,
        severity=5e-324,
        unit_id="unit:max",
        target_kind="global",
        target_id="",
        draw_index=PORTABLE_INTEGER_MAX,
    )
    assert result["normalized_payload"][-1] == PORTABLE_INTEGER_MAX
    assert result["key_words"] == [0xFFFFFFFF, 0x001FFFFF]


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_nonfinite_severity_refuses(nonfinite: float) -> None:
    with pytest.raises(OracleError, match="severity must be finite binary64"):
        derive_robustness_block(
            seed=0,
            scenario_fingerprint="a" * 64,
            severity=nonfinite,
            unit_id="unit:1",
            target_kind="global",
            target_id="",
            draw_index=0,
        )


def test_committed_invalid_vectors_fail_closed(golden: dict[str, Any]) -> None:
    assert len(golden["invalid_vectors"]) >= 10
    for vector in golden["invalid_vectors"]:
        invalid_input = vector["input"].copy()
        synthesis = vector.get("input_synthesis")
        if synthesis is not None:
            assert synthesis == {
                "field": "unit_id",
                "operation": "unpaired_utf16_high_surrogate",
                "code_unit": 0xD800,
            }
            invalid_input[synthesis["field"]] = chr(synthesis["code_unit"])
        with pytest.raises(OracleError) as error:
            derive_robustness_block(**invalid_input)
        assert vector["error_contains"] in str(error.value), vector["id"]


def test_json_loader_refuses_duplicates_overflow_and_invalid_utf8(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(OracleError, match="duplicate key"):
        load_json(duplicate)

    overflow = tmp_path / "overflow.json"
    overflow.write_text("1e400", encoding="utf-8")
    with pytest.raises(OracleError, match="non-finite JSON number"):
        load_json(overflow)

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b'"\xff"')
    with pytest.raises(OracleError, match="not strict UTF-8 JSON"):
        load_json(invalid_utf8)


@pytest.mark.parametrize(
    ("counter", "key", "error"),
    [
        ((0, 0, 0), (0, 0), "counter must contain exactly 4 words"),
        ((0, 0, 0, 0), (0,), "key must contain exactly 2 words"),
        ((0, 0, 0, True), (0, 0), "counter[3] must be an integer"),
        ((0, 0, 0, 2**32), (0, 0), "counter[3] must be a u32"),
        ((0, 0, 0, 0), (0, -1), "key[1] must be a u32"),
    ],
)
def test_philox_word_domain_refuses(
    counter: tuple[Any, ...], key: tuple[Any, ...], error: str
) -> None:
    with pytest.raises(
        OracleError, match=error.replace("[", r"\[").replace("]", r"\]")
    ):
        philox4x32_10(counter, key)


def test_non_sequence_philox_inputs_and_unhashable_target_kind_refuse() -> None:
    with pytest.raises(OracleError, match="counter must be a word sequence"):
        philox4x32_10(7, (0, 0))  # type: ignore[arg-type]

    with pytest.raises(
        OracleError, match="target_kind must be source, node, or global"
    ):
        derive_robustness_block(
            seed=0,
            scenario_fingerprint="a" * 64,
            severity=0.0,
            unit_id="unit:1",
            target_kind=["global"],
            target_id="",
            draw_index=0,
        )


def test_oracle_has_no_production_or_parity_dependency() -> None:
    tree = ast.parse(ORACLE_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
        "hashlib",
        "json",
        "math",
        "re",
        "struct",
        "unicodedata",
        "pathlib",
        "typing",
    }
