from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import fingerprint_without  # noqa: E402
from parity.criteria.oracle import VALIDATORS, CriteriaContractError  # noqa: E402
from scripts.validate_contracts import build_local_schema_registry  # noqa: E402


FIXTURE = ROOT / "examples/fixtures/criteria/criteria_contracts.v1.json"
PACK = ROOT / "docs/contracts/criteria_conformance_pack.v1.json"
SCHEMA_IDS = {
    "loss_spec": "https://github.com/GBeurier/dag-ml/schemas/loss_spec.v1.schema.json",
    "metric_spec": "https://github.com/GBeurier/dag-ml/schemas/metric_spec.v1.schema.json",
    "implementation_descriptor": "https://github.com/GBeurier/dag-ml/schemas/implementation_descriptor.v1.schema.json",
    "training_loss_role": "https://github.com/GBeurier/dag-ml/schemas/training_loss_role.v1.schema.json",
    "metric_role": "https://github.com/GBeurier/dag-ml/schemas/metric_role.v1.schema.json",
}
VALID_CONTRACTS = {
    "loss_spec": "loss_spec",
    "metric_spec": "metric_spec",
    "loss_implementation": "implementation_descriptor",
    "metric_implementation": "implementation_descriptor",
    "training_loss_role": "training_loss_role",
    "metric_role": "metric_role",
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def schema_errors(contract: str, document: dict[str, Any]) -> list[Any]:
    registry, schemas = build_local_schema_registry()
    validator = Draft202012Validator(schemas[SCHEMA_IDS[contract]], registry=registry)
    return list(validator.iter_errors(document))


def test_valid_fixture_matches_all_schemas_and_independent_semantics() -> None:
    fixture = load(FIXTURE)
    for key, contract in VALID_CONTRACTS.items():
        document = fixture["valid"][key]
        assert schema_errors(contract, document) == [], key
        VALIDATORS[contract](document)


def test_positive_fingerprints_are_frozen_tcv1_values() -> None:
    valid = load(FIXTURE)["valid"]
    for key, field in (
        ("loss_spec", "spec_fingerprint"),
        ("metric_spec", "spec_fingerprint"),
        ("loss_implementation", "descriptor_fingerprint"),
        ("metric_implementation", "descriptor_fingerprint"),
    ):
        assert valid[key][field] == fingerprint_without(valid[key], field), key


def test_required_negative_fixture_cases_are_rejected() -> None:
    cases = load(FIXTURE)["invalid"]
    assert {case["id"] for case in cases} == {
        "host_local_descriptor_without_registry_key",
        "loss_mismatched_fingerprint",
        "loss_nested_callable_payload",
        "loss_unknown_task",
        "loss_unversioned_id",
        "loss_weighted_without_weight_input",
        "metric_without_objective",
        "selection_metric_skips_missing_values",
    }
    for case in cases:
        errors = schema_errors(case["contract"], case["document"])
        try:
            VALIDATORS[case["contract"]](case["document"])
        except (CriteriaContractError, KeyError):
            semantic_rejected = True
        else:
            semantic_rejected = False
        assert errors or semantic_rejected, case["id"]


def test_nested_executable_payload_is_rejected_semantically_case_insensitively() -> None:
    document = load(FIXTURE)["valid"]["loss_spec"]
    document["parameters"] = {"nested": [{"CallAble": "not serialized code"}]}
    document["spec_fingerprint"] = fingerprint_without(document, "spec_fingerprint")
    with pytest.raises(CriteriaContractError, match="CallAble"):
        VALIDATORS["loss_spec"](document)


def test_conformance_pack_is_exact_and_self_fingerprinted() -> None:
    pack = load(PACK)
    assert pack["pack_checksum"] == fingerprint_without(pack, "pack_checksum")
    paths = [artifact["path"] for artifact in pack["artifacts"]]
    assert paths == sorted(set(paths))
    for artifact in pack["artifacts"]:
        digest = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
        assert digest == artifact["sha256"], artifact["path"]
    assert pack["required_negative_cases"] == [
        case["id"] for case in load(FIXTURE)["invalid"]
    ]
