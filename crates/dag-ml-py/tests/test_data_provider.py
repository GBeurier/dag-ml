"""Real native PLAN execution of host-owned finite dataset providers."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import dag_ml
import pytest


def _recipe() -> dict[str, Any]:
    return {
        "provider_id": "source:synthetic", "provider_version": "1.0",
        "params": {"sample_count": 12}, "seed": 31, "scope": "run",
        "finite": True, "learned": False, "context": {"purpose": "qualification"},
    }


class Provider:
    def __init__(self, handle: int = 1) -> None:
        self.handle = handle
        self.tasks: list[dict[str, Any]] = []
        self.dataset: dict[str, Any] | None = None

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        assert task["phase"] == "PLAN"
        assert task["node_plan"]["kind"] == "generator"
        assert task["node_plan"]["supported_phases"] == ["PLAN"]
        assert task["node_plan"]["data_bindings"] == []
        assert task["data_views"] == task["input_handles"] == {}
        assert task["fold_id"] is None and task["variant_id"] is None
        assert isinstance(task["seed"], int)
        rng = random.Random(task["seed"])
        features = [[rng.random(), rng.random()] for _ in range(task["node_plan"]["params"]["sample_count"])]
        self.dataset = {"X": features, "y": [3 * row[0] + row[1] for row in features]}
        digest = hashlib.sha256(json.dumps(self.dataset, sort_keys=True).encode()).hexdigest()
        return {
            "handle": {"handle": self.handle, "kind": "data", "owner_controller": task["node_plan"]["controller_id"]},
            "metadata": {"sample_count": len(features), "content_fingerprint": digest},
        }


def test_native_provider_materializes_xy_once_with_reproducible_lineage() -> None:
    first, second = Provider(1), Provider(99)
    a = dag_ml.execute_data_provider(_recipe(), first)
    b = dag_ml.execute_data_provider(_recipe(), second)
    assert len(first.tasks) == len(second.tasks) == 1
    assert first.dataset == second.dataset
    assert a["profile"] == "data_provider_prepare_v1"
    assert a["execution_fingerprint"] == b["execution_fingerprint"]
    assert a["handle"] != b["handle"]
    assert a["task_seed"] == a["lineage"]["seed"] == first.tasks[0]["seed"]
    assert a["lineage"]["phase"] == "PLAN"
    assert a["lineage"]["fold_id"] is None
    assert a["lineage"]["params_fingerprint"] == first.tasks[0]["node_plan"]["params_fingerprint"]
    assert set(a["metadata"]) == {"sample_count", "content_fingerprint"}
    assert "envelope" not in a and "fold_set" not in a
    assert "X" not in json.dumps(a) and '"y"' not in json.dumps(a)


@pytest.mark.parametrize("field,value", [
    ("scope", "fold"), ("scope", "epoch"), ("finite", False), ("learned", True),
    ("provider_id", " "), ("provider_version", ""), ("seed", -1),
    ("seed", True), ("params", {" ": 1}), ("context", {"": 1}), ("undeclared", 1),
])
def test_invalid_provider_recipe_is_refused_before_callback(field: str, value: Any) -> None:
    recipe = _recipe()
    recipe[field] = value
    provider = Provider()
    with pytest.raises(dag_ml.DagMlError):
        dag_ml.execute_data_provider(recipe, provider)
    assert provider.tasks == []


@pytest.mark.parametrize("field,value", [
    ("provider_id", "source:other"), ("provider_version", "2.0"), ("seed", 32),
    ("params", {"sample_count": 6}), ("context", {"purpose": "other"}),
])
def test_recipe_changes_are_visible_in_native_receipt(field: str, value: Any) -> None:
    recipe = _recipe()
    original = dag_ml.execute_data_provider(recipe, Provider())
    recipe[field] = value
    changed = dag_ml.execute_data_provider(recipe, Provider())
    assert original["recipe_fingerprint"] != changed["recipe_fingerprint"]
    assert original["execution_fingerprint"] != changed["execution_fingerprint"]
    if field == "seed":
        assert original["task_seed"] != changed["task_seed"]
        assert original["metadata"]["content_fingerprint"] != changed["metadata"]["content_fingerprint"]
    if field == "context":
        assert original["context_fingerprint"] != changed["context_fingerprint"]


@pytest.mark.parametrize("field,value", [("handle", 0), ("kind", "model"), ("kind", "data_view"), ("owner_controller", "forged")])
def test_native_provider_refuses_invalid_output_handle(field: str, value: Any) -> None:
    provider = Provider()

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        result = provider(task)
        result["handle"][field] = value
        return result

    with pytest.raises(dag_ml.DagMlRuntimeError, match="nonzero data handle"):
        dag_ml.execute_data_provider(_recipe(), callback)
    assert len(provider.tasks) == 1


def test_provider_errors_propagate_without_retry_or_fabricated_receipt() -> None:
    calls: list[dict[str, Any]] = []

    def failing(task: dict[str, Any]) -> dict[str, Any]:
        calls.append(task)
        raise ValueError("deliberate provider error")

    with pytest.raises(dag_ml.DagMlRuntimeError, match="deliberate provider error"):
        dag_ml.execute_data_provider(_recipe(), failing)
    assert len(calls) == 1
    with pytest.raises(dag_ml.DagMlRuntimeError, match="callable"):
        dag_ml.execute_data_provider(_recipe(), None)


def test_provider_callback_cannot_supply_its_own_lineage() -> None:
    provider = Provider()

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(provider(task))
        result["lineage"] = {"seed": 99}
        return result

    with pytest.raises(dag_ml.DagMlError, match="lineage"):
        dag_ml.execute_data_provider(_recipe(), callback)


def test_provider_receipt_matches_published_schema_and_recipe_defaults() -> None:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schema_dir = Path(__file__).resolve().parents[3] / "docs" / "contracts"
    schemas = [json.loads(path.read_text()) for path in schema_dir.glob("*.schema.json")]
    registry = Registry().with_resources((schema["$id"], Resource.from_contents(schema)) for schema in schemas)
    schema = json.loads((schema_dir / "data_provider_prepare.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    result = dag_ml.execute_data_provider(_recipe(), Provider())
    Draft202012Validator(schema, registry=registry).validate(result)
    Draft202012Validator({**schema, "$ref": "#/$defs/recipe"}, registry=registry).validate(_recipe())
    materialization = {"handle": result["handle"], "metadata": result["metadata"]}
    Draft202012Validator({**schema, "$ref": "#/$defs/materialization"}, registry=registry).validate(materialization)

    minimal = {"provider_id": "source:minimal", "provider_version": "1"}
    explicit = {**minimal, "params": {}, "seed": 0, "scope": "run", "finite": True, "learned": False, "context": {}}

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        return {"handle": {"handle": 1, "kind": "data", "owner_controller": task["node_plan"]["controller_id"]}}

    assert dag_ml.execute_data_provider(minimal, callback) == dag_ml.execute_data_provider(explicit, callback)
