from __future__ import annotations

import copy
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
from parity.criteria.oracle import (  # noqa: E402
    VALIDATORS,
    CriteriaContractError,
    validate_metric_evaluation_result,
)
from scripts.validate_contracts import build_local_schema_registry  # noqa: E402


FIXTURE = ROOT / "examples/fixtures/criteria/criteria_contracts.v1.json"
PACK = ROOT / "docs/contracts/criteria_conformance_pack.v1.json"
PROVIDER_FIXTURE = ROOT / "examples/fixtures/criteria/metric_provider_contracts.v1.json"
R_FIXTURE = ROOT / "bindings/r/inst/extdata/r_local_implementations.v1.json"
MATLAB_FIXTURE = ROOT / "bindings/matlab/fixtures/matlab_local_implementations.v1.json"
SCHEMA_IDS = {
    "loss_spec": "https://github.com/GBeurier/dag-ml/schemas/loss_spec.v1.schema.json",
    "metric_spec": "https://github.com/GBeurier/dag-ml/schemas/metric_spec.v1.schema.json",
    "implementation_descriptor": "https://github.com/GBeurier/dag-ml/schemas/implementation_descriptor.v1.schema.json",
    "training_loss_role": "https://github.com/GBeurier/dag-ml/schemas/training_loss_role.v1.schema.json",
    "loss_execution_attestation": "https://github.com/GBeurier/dag-ml/schemas/loss_execution_attestation.v1.schema.json",
    "metric_role": "https://github.com/GBeurier/dag-ml/schemas/metric_role.v1.schema.json",
    "metric_evaluation_task": "https://github.com/GBeurier/dag-ml/schemas/metric_evaluation_task.v1.schema.json",
    "metric_evaluation_result": "https://github.com/GBeurier/dag-ml/schemas/metric_evaluation_result.v1.schema.json",
    "node_task": "https://github.com/GBeurier/dag-ml/schemas/node_task.v1.schema.json",
}
VALID_CONTRACTS = {
    "loss_spec": "loss_spec",
    "metric_spec": "metric_spec",
    "loss_implementation": "implementation_descriptor",
    "metric_implementation": "implementation_descriptor",
    "training_loss_role": "training_loss_role",
    "loss_execution_attestation": "loss_execution_attestation",
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
        ("loss_execution_attestation", "attestation_fingerprint"),
    ):
        assert valid[key][field] == fingerprint_without(valid[key], field), key


def test_required_negative_fixture_cases_are_rejected() -> None:
    cases = load(FIXTURE)["invalid"]
    assert {case["id"] for case in cases} == {
        "host_local_descriptor_without_registry_key",
        "loss_mismatched_fingerprint",
        "loss_nested_callable_payload",
        "loss_c1_control_id",
        "loss_leading_zero_version",
        "loss_uppercase_callable_payload",
        "loss_unknown_task",
        "loss_unversioned_id",
        "loss_weighted_without_weight_input",
        "loss_attestation_wrong_phase",
        "metric_without_objective",
        "selection_metric_skips_missing_values",
    }
    schema_required = {
        "loss_c1_control_id",
        "loss_leading_zero_version",
        "loss_uppercase_callable_payload",
        "loss_attestation_wrong_phase",
    }
    for case in cases:
        errors = schema_errors(case["contract"], case["document"])
        if case["id"] in schema_required:
            assert errors, f"schema accepted parity case {case['id']}"
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


def test_large_decimal_version_is_accepted_by_schema_and_semantics() -> None:
    document = load(FIXTURE)["valid"]["loss_spec"]
    document["loss_id"] = "example.loss.asymmetric@4294967296"
    document["spec_fingerprint"] = fingerprint_without(document, "spec_fingerprint")
    assert schema_errors("loss_spec", document) == []
    VALIDATORS["loss_spec"](document)


def test_metric_provider_fixture_has_independent_task_result_and_refusal_parity() -> None:
    fixture = load(PROVIDER_FIXTURE)
    task = fixture["valid"]["task"]
    result = fixture["valid"]["result"]
    assert schema_errors("metric_evaluation_task", task) == []
    assert schema_errors("metric_evaluation_result", result) == []
    VALIDATORS["metric_evaluation_task"](task)
    assert validate_metric_evaluation_result(result, task) == fixture["valid"]["aggregate"]

    for case in fixture["invalid"]:
        errors = schema_errors(case["contract"], case["document"])
        try:
            if case["contract"] == "metric_evaluation_task":
                VALIDATORS[case["contract"]](case["document"])
            else:
                validate_metric_evaluation_result(case["document"], task)
        except (CriteriaContractError, KeyError):
            semantic_rejected = True
        else:
            semantic_rejected = False
        assert errors or semantic_rejected, case["id"]

    nonfinite = copy.deepcopy(result)
    nonfinite["values"][0]["value"] = float("nan")
    with pytest.raises(CriteriaContractError, match="non-finite provider value"):
        validate_metric_evaluation_result(nonfinite, task)


@pytest.mark.parametrize("fixture_path", [R_FIXTURE, MATLAB_FIXTURE])
def test_host_local_registry_fixture_uses_native_node_task_requirements(
    fixture_path: Path,
) -> None:
    fixture = load(fixture_path)
    assert (
        schema_errors(
            "implementation_descriptor",
            fixture["loss_reference"]["implementation"],
        )
        == []
    )
    assert (
        schema_errors(
            "implementation_descriptor",
            fixture["metric_reference"]["implementation"],
        )
        == []
    )
    assert schema_errors("training_loss_role", fixture["training_loss_role"]) == []
    for phase in ("FIT_CV", "REFIT"):
        task = fixture["tasks"][phase]
        assert schema_errors("node_task", task) == []
        requirement = task["required_loss_attestations"][0]
        assert schema_errors("loss_execution_attestation", requirement) == []
        VALIDATORS["loss_execution_attestation"](requirement)
        assert requirement["attestation_fingerprint"] == fingerprint_without(
            requirement, "attestation_fingerprint"
        )


def test_conformance_pack_is_exact_and_self_fingerprinted() -> None:
    pack = load(PACK)
    assert pack["pack_checksum"] == fingerprint_without(pack, "pack_checksum")
    paths = [artifact["path"] for artifact in pack["artifacts"]]
    assert paths == sorted(set(paths))
    for artifact in pack["artifacts"]:
        digest = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
        assert digest == artifact["sha256"], artifact["path"]
    assert pack["required_negative_cases"] == [
        case["id"]
        for case in load(FIXTURE)["invalid"] + load(PROVIDER_FIXTURE)["invalid"]
    ]
    assert pack["runtime_only_negative_cases"] == ["provider_result_non_finite"]
