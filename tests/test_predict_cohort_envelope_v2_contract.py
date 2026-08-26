"""Closed wire-contract checks for the ADR-26 PREDICT cohort envelope."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[1]
V1_SCHEMA = ROOT / "docs/contracts/coordinator_data_plan_envelope.schema.json"
V2_SCHEMA = ROOT / "docs/contracts/coordinator_data_plan_envelope.v2.schema.json"
V1_FIXTURE = ROOT / "examples/fixtures/data/coordinator_data_plan_envelope_nir.json"


def _sha(char: str) -> str:
    return char * 64


def _external_test_cohort() -> dict[str, object]:
    relations = {
        "records": [
            {
                "observation_id": "obs.holdout.1",
                "sample_id": "sample:holdout:1",
                "origin_sample_id": "sample:holdout:1",
                "target_id": "classification:y",
            },
            {
                "observation_id": "obs.holdout.2",
                "sample_id": "sample:holdout:2",
                "origin_sample_id": "sample:holdout:2",
                "target_id": "classification:y",
            },
        ]
    }
    return {
        "role": "external_test",
        "physical_sample_ids": ["sample:holdout:1", "sample:holdout:2"],
        "origin_sample_ids": ["sample:holdout:1", "sample:holdout:2"],
        "target_names": ["classification:y"],
        "relation_fingerprint": _sha("a"),
        "relations": relations,
        "data_content_fingerprint": _sha("b"),
        "target_content_fingerprint": _sha("c"),
        "cohort_fingerprint": _sha("d"),
    }


def _v2_document() -> dict[str, object]:
    document = json.loads(V1_FIXTURE.read_text())
    document["schema_version"] = 2
    document["predict_cohort"] = _external_test_cohort()
    return document


def _validator(path: Path) -> Draft202012Validator:
    schema = json.loads(path.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_v2_schema_accepts_closed_external_test_and_rejects_v1_mixing() -> None:
    v1 = _validator(V1_SCHEMA)
    v2 = _validator(V2_SCHEMA)
    document = _v2_document()

    v2.validate(document)
    with pytest.raises(ValidationError):
        v1.validate(document)


def test_v2_schema_refuses_unknown_or_inference_target() -> None:
    validator = _validator(V2_SCHEMA)

    unknown = _v2_document()
    unknown["predict_cohort"]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(ValidationError):
        validator.validate(unknown)

    inference = _v2_document()
    cohort = inference["predict_cohort"]
    assert isinstance(cohort, dict)
    cohort["role"] = "inference"
    cohort.pop("target_content_fingerprint")
    validator.validate(inference)

    target_bearing_inference = copy.deepcopy(inference)
    target_bearing_inference["predict_cohort"]["target_content_fingerprint"] = _sha("c")  # type: ignore[index]
    with pytest.raises(ValidationError):
        validator.validate(target_bearing_inference)
