#!/usr/bin/env python3
"""Regenerate criterion fingerprints and the exact L1 conformance pack."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import fingerprint_without  # noqa: E402


FIXTURE = ROOT / "examples/fixtures/criteria/criteria_contracts.v1.json"
PACK = ROOT / "docs/contracts/criteria_conformance_pack.v1.json"
ARTIFACTS = {
    "crates/dag-ml-cli/src/main.rs": "cli_validator",
    "crates/dag-ml-cli/tests/cli_contracts.rs": "cli_test",
    "crates/dag-ml-core/src/criteria.rs": "rust_contract",
    "docs/contracts/implementation_descriptor.schema.json": "schema",
    "docs/contracts/loss_spec.schema.json": "schema",
    "docs/contracts/metric_role.schema.json": "schema",
    "docs/contracts/metric_spec.schema.json": "schema",
    "docs/contracts/training_loss_role.schema.json": "schema",
    "examples/fixtures/criteria/criteria_contracts.v1.json": "fixture",
    "parity/criteria/oracle.py": "independent_validator",
    "parity/criteria/generate_fixtures.py": "generator",
    "parity/criteria/tests/test_criteria_contracts.py": "parity_test",
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    valid["metric_role"]["metric"] = {
        "spec": valid["metric_spec"],
        "implementation": valid["metric_implementation"],
    }
    write(FIXTURE, fixture)

    pack: dict[str, Any] = {
        "pack_id": "dag-ml.criteria-conformance.v1",
        "schema_version": 1,
        "hash_algorithm": "sha256-file-bytes",
        "fingerprint_profile": "DAGML-TCV1-unicode-17.0.0",
        "artifacts": [
            {"path": path, "sha256": sha256(ROOT / path), "kind": kind}
            for path, kind in sorted(ARTIFACTS.items())
        ],
        "required_negative_cases": [case["id"] for case in fixture["invalid"]],
        "pack_checksum": "",
    }
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    write(PACK, pack)


if __name__ == "__main__":
    main()
