"""Offline wire boundaries; native campaign tests attest scientific semantics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.validate_contracts import build_local_schema_registry

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/GBeurier/dag-ml/schemas/"
STATE_ID = BASE + "methods_hpo_fold_state.v1.schema.json"


@pytest.fixture(scope="module")
def contracts():
    # Resolve every external $ref offline, including the complete checkpoint,
    # proposal/report/candidate ledgers and the root coordinator relations.
    return build_local_schema_registry()


def _validator(contracts, fragment: str = "") -> Draft202012Validator:
    registry, _ = contracts
    return Draft202012Validator({"$ref": STATE_ID + fragment}, registry=registry)


@pytest.mark.parametrize("params", [{"n_components": 1}, {"scale": False}, {"n_components": 3, "scale": True}])
def test_public_winners_keep_searched_only_bool_and_integer_types(contracts, params) -> None:
    _validator(contracts, "#/$defs/winner_params").validate(params)


@pytest.mark.parametrize("params", [
    {}, {"scale": 0}, {"scale": "false"}, {"n_components": True},
    {"n_components": 0}, {"n_components": 1.5}, {"n_components": 2**31},
    {"scale_x": False}, {"scale": True, "train_params": {}},
])
def test_winner_wire_refuses_coercion_aliases_and_unsearched_controls(contracts, params) -> None:
    with pytest.raises(ValidationError):
        _validator(contracts, "#/$defs/winner_params").validate(params)


@pytest.mark.parametrize("refit", [False, True])
def test_phase_wire_and_parent_binding_are_distinct_from_lowercase_scope_ids(contracts, refit: bool) -> None:
    fragment = "#/properties/refit_scope/allOf/1" if refit else "#/properties/outer_scopes/items/allOf/1"
    validator = _validator(contracts, fragment)
    scope = {"phase": "REFIT" if refit else "FIT_CV", "outer_fold_id": None if refit else "fold0"}
    validator.validate(scope)
    for mutation in ("lowercase", "wrong_phase", "wrong_parent"):
        invalid = copy.deepcopy(scope)
        if mutation == "lowercase":
            invalid["phase"] = invalid["phase"].lower()
        elif mutation == "wrong_phase":
            invalid["phase"] = "FIT_CV" if refit else "REFIT"
        else:
            invalid["outer_fold_id"] = "fold0" if refit else None
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def _inner_folds():
    return {
        "id": "fold0.inner", "sample_ids": ["s0", "s1", "s2", "s3"],
        "sample_groups": {"s0": "g0", "s1": "g0", "s2": "g1", "s3": "g1"},
        "folds": [
            {"fold_id": "fold0.inner.fold0", "train_sample_ids": ["s2", "s3"], "validation_sample_ids": ["s0", "s1"], "metadata": {}},
            {"fold_id": "fold0.inner.fold1", "train_sample_ids": ["s0", "s1"], "validation_sample_ids": ["s2", "s3"], "metadata": {}},
        ],
    }


def test_inner_wire_preserves_declared_groups_and_default_partition_authority(contracts) -> None:
    validator = _validator(contracts, "#/$defs/inner_fold_set")
    folds = _inner_folds()
    validator.validate(folds)
    folds.update(partition_mode="partition", train_exclusion="relations")
    validator.validate(folds)


@pytest.mark.parametrize("mutation", ["resampled", "single_fold", "duplicate_universe", "unknown", "unknown_fold", "unknown_exclusion"])
def test_inner_wire_refuses_resampling_and_open_transport(contracts, mutation: str) -> None:
    folds = _inner_folds()
    if mutation == "resampled":
        folds["partition_mode"] = "resampled"
    elif mutation == "single_fold":
        folds["folds"].pop()
    elif mutation == "duplicate_universe":
        folds["sample_ids"].append("s0")
    elif mutation == "unknown":
        folds["inner_cv"] = 2
    elif mutation == "unknown_fold":
        folds["folds"][0]["parent_alias"] = "fold0"
    else:
        folds["train_exclusion"] = "unchecked"
    with pytest.raises(ValidationError):
        _validator(contracts, "#/$defs/inner_fold_set").validate(folds)


def test_v2_optional_state_links_are_closed_and_v1_reader_shape_is_preserved(contracts) -> None:
    registry, schemas = contracts
    for family in ("execution_bundle", "training_outcome"):
        schema = schemas[BASE + family + ".v2.schema.json"]
        field = schema["properties"]["methods_hpo_fold_state"]
        assert field["anyOf"][0]["$ref"] == STATE_ID
        Draft202012Validator(field, registry=registry).validate(None)
        with pytest.raises(ValidationError):
            Draft202012Validator(field, registry=registry).validate({"schema_version": 1})
        assert "methods_hpo_fold_state" not in schemas[BASE + family + ".v1.schema.json"]["properties"]
    schema = schemas[STATE_ID]
    assert schema["additionalProperties"] is False
    assert {"relations", "base_plan", "provenance", "outer_scopes", "refit_scope"} <= set(schema["required"])
    assert schema["$defs"]["scope"]["additionalProperties"] is False
    assert "resume_state" in schema["$defs"]["scope"]["required"]
    package = json.loads((ROOT / "examples/fixtures/training/portable_predictor_package.v1.json").read_text())
    Draft202012Validator(schemas[BASE + "portable_predictor_package.v1.schema.json"], registry=registry).validate(package)


def test_relation_transport_is_required_closed_and_keeps_group_origin_authority(contracts) -> None:
    validator = _validator(contracts, "#/properties/relations")
    relations = {"records": [{"observation_id": "obs:1", "sample_id": "s1", "origin_sample_id": "origin:1", "group_id": "g1", "excluded": False}]}
    validator.validate(relations)
    relations["records"][0]["guessed_origin"] = "origin:2"
    with pytest.raises(ValidationError):
        validator.validate(relations)
