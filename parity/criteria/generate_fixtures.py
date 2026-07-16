#!/usr/bin/env python3
"""Regenerate criterion fingerprints and the exact L1 conformance pack."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import fingerprint_without  # noqa: E402


FIXTURE = ROOT / "examples/fixtures/criteria/criteria_contracts.v1.json"
PROVIDER_FIXTURE = ROOT / "examples/fixtures/criteria/metric_provider_contracts.v1.json"
PACK = ROOT / "docs/contracts/criteria_conformance_pack.v1.json"
ARTIFACTS = {
    "crates/dag-ml-cli/src/main.rs": "cli_validator",
    "crates/dag-ml-cli/tests/cli_contracts.rs": "cli_test",
    "crates/dag-ml-core/src/criteria.rs": "rust_contract",
    "crates/dag-ml-core/src/metric_provider.rs": "rust_provider_contract",
    "crates/dag-ml-core/src/metrics.rs": "native_metric_adapter",
    "docs/CRITERIA_CONTRACTS.md": "contract_documentation",
    "docs/contracts/implementation_descriptor.schema.json": "schema",
    "docs/contracts/loss_execution_attestation.schema.json": "schema",
    "docs/contracts/loss_spec.schema.json": "schema",
    "docs/contracts/metric_evaluation_result.schema.json": "schema",
    "docs/contracts/metric_evaluation_task.schema.json": "schema",
    "docs/contracts/metric_role.schema.json": "schema",
    "docs/contracts/metric_spec.schema.json": "schema",
    "docs/contracts/training_loss_role.schema.json": "schema",
    "examples/fixtures/criteria/criteria_contracts.v1.json": "fixture",
    "examples/fixtures/criteria/metric_provider_contracts.v1.json": "fixture",
    "parity/criteria/oracle.py": "independent_validator",
    "parity/criteria/generate_fixtures.py": "generator",
    "parity/criteria/tests/test_criteria_contracts.py": "parity_test",
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provider_fixture(valid: dict[str, Any]) -> dict[str, Any]:
    scope = {
        "producer_node": "model:custom",
        "producer_port": "prediction",
        "prediction_id": "prediction:validation",
        "partition": "validation",
        "fold_id": "fold:0",
        "level": "sample",
    }
    task = {
        "schema_version": 1,
        "request_id": "metric-request:bias",
        "metric": {
            "spec": copy.deepcopy(valid["metric_spec"]),
            "implementation": copy.deepcopy(valid["metric_implementation"]),
        },
        "task_kind": "regression",
        "prediction_kind": "regression_point",
        "scope": scope,
        "unit_ids": [
            {"level": "sample", "id": "sample:0"},
            {"level": "sample", "id": "sample:1"},
        ],
        "predictions": [[2.0], [5.0]],
        "targets": [[1.0], [3.0]],
        "output_ids": ["target"],
        "task_fingerprint": "",
    }
    task["task_fingerprint"] = fingerprint_without(task, "task_fingerprint")
    result = {
        "schema_version": 1,
        "request_id": task["request_id"],
        "semantic_id": task["metric"]["spec"]["metric_id"],
        "semantic_fingerprint": task["metric"]["spec"]["spec_fingerprint"],
        "implementation_fingerprint": task["metric"]["implementation"][
            "implementation_fingerprint"
        ],
        "descriptor_fingerprint": task["metric"]["implementation"][
            "descriptor_fingerprint"
        ],
        "scope": copy.deepcopy(scope),
        "values": [{"value": 1.5}],
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = fingerprint_without(result, "result_fingerprint")

    missing_objective = copy.deepcopy(task)
    missing_objective["metric"]["spec"].pop("objective")
    wrong_scope = copy.deepcopy(result)
    wrong_scope["scope"]["partition"] = "test"
    wrong_scope["result_fingerprint"] = fingerprint_without(
        wrong_scope, "result_fingerprint"
    )
    wrong_coverage = copy.deepcopy(result)
    wrong_coverage["values"] = []
    wrong_coverage["result_fingerprint"] = fingerprint_without(
        wrong_coverage, "result_fingerprint"
    )
    wrong_implementation = copy.deepcopy(result)
    wrong_implementation["implementation_fingerprint"] = "0" * 64
    wrong_implementation["result_fingerprint"] = fingerprint_without(
        wrong_implementation, "result_fingerprint"
    )
    return {
        "profile": "dagml.metric-provider.contracts.v1",
        "canonicalization": "TCV1-unicode-17.0.0",
        "valid": {"task": task, "result": result, "aggregate": 1.5},
        "invalid": [
            {
                "id": "custom_metric_without_objective",
                "contract": "metric_evaluation_task",
                "document": missing_objective,
            },
            {
                "id": "provider_result_wrong_scope",
                "contract": "metric_evaluation_result",
                "document": wrong_scope,
            },
            {
                "id": "provider_result_wrong_coverage",
                "contract": "metric_evaluation_result",
                "document": wrong_coverage,
            },
            {
                "id": "provider_result_mismatched_implementation_fingerprint",
                "contract": "metric_evaluation_result",
                "document": wrong_implementation,
            },
        ],
    }


def main() -> None:
    fixture = load(FIXTURE)
    valid = fixture["valid"]
    for key in ("loss_spec", "metric_spec"):
        valid[key]["spec_fingerprint"] = fingerprint_without(valid[key], "spec_fingerprint")
    valid["loss_implementation"]["semantic_fingerprint"] = valid["loss_spec"]["spec_fingerprint"]
    valid["metric_implementation"]["semantic_fingerprint"] = valid["metric_spec"]["spec_fingerprint"]
    for key in ("loss_implementation", "metric_implementation"):
        valid[key]["descriptor_fingerprint"] = fingerprint_without(
            valid[key], "descriptor_fingerprint"
        )
    valid["training_loss_role"]["loss"] = {
        "spec": valid["loss_spec"],
        "implementation": valid["loss_implementation"],
    }
    valid["loss_execution_attestation"] = {
        "schema_version": 1,
        "node_id": valid["training_loss_role"]["node_id"],
        "output_id": valid["training_loss_role"]["output_id"],
        "phase": "FIT_CV",
        "loss_id": valid["loss_spec"]["loss_id"],
        "semantic_fingerprint": valid["loss_spec"]["spec_fingerprint"],
        "implementation_fingerprint": valid["loss_implementation"][
            "implementation_fingerprint"
        ],
        "descriptor_fingerprint": valid["loss_implementation"][
            "descriptor_fingerprint"
        ],
        "effective_parameters": copy.deepcopy(valid["loss_spec"]["parameters"]),
        "reduction": valid["loss_spec"]["reduction"],
        "attestation_fingerprint": "",
    }
    valid["loss_execution_attestation"]["attestation_fingerprint"] = fingerprint_without(
        valid["loss_execution_attestation"], "attestation_fingerprint"
    )
    wrong_phase = copy.deepcopy(valid["loss_execution_attestation"])
    wrong_phase["phase"] = "PREDICT"
    wrong_phase["attestation_fingerprint"] = fingerprint_without(
        wrong_phase, "attestation_fingerprint"
    )
    fixture["invalid"] = [
        case
        for case in fixture["invalid"]
        if case["id"] != "loss_attestation_wrong_phase"
    ] + [
        {
            "id": "loss_attestation_wrong_phase",
            "contract": "loss_execution_attestation",
            "document": wrong_phase,
        }
    ]
    valid["metric_role"]["metric"] = {
        "spec": valid["metric_spec"],
        "implementation": valid["metric_implementation"],
    }
    write(FIXTURE, fixture)
    provider_fixture = build_provider_fixture(valid)
    write(PROVIDER_FIXTURE, provider_fixture)

    pack: dict[str, Any] = {
        "pack_id": "dag-ml.criteria-conformance.v1",
        "schema_version": 1,
        "hash_algorithm": "sha256-file-bytes",
        "fingerprint_profile": "DAGML-TCV1-unicode-17.0.0",
        "artifacts": [
            {"path": path, "sha256": sha256(ROOT / path), "kind": kind}
            for path, kind in sorted(ARTIFACTS.items())
        ],
        "required_negative_cases": [
            case["id"]
            for case in fixture["invalid"] + provider_fixture["invalid"]
        ],
        "runtime_only_negative_cases": ["provider_result_non_finite"],
        "pack_checksum": "",
    }
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    write(PACK, pack)


if __name__ == "__main__":
    main()
