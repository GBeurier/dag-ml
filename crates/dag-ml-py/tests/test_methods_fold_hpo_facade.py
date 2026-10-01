"""Fold-HPO metadata stays native-validated and callback-free."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import dag_ml
import pytest


def test_public_metadata_reader_has_no_execution_callbacks() -> None:
    assert list(inspect.signature(dag_ml.methods_hpo_fold_state_from_package).parameters) == ["package"]
    for name in (
        "MethodsFoldHpoState", "methods_hpo_fold_state_from_package",
        "validate_methods_hpo_fold_state_json", "methods_hpo_fold_state_from_package_json",
    ):
        assert name in dag_ml.__all__
    manifest = json.loads(dag_ml.contract_manifest_json())
    assert {"id": "methods_hpo_fold_state", "version": 1} in manifest["contracts"]
    assert "validate_methods_hpo_fold_state_json" in manifest["python_exports"]
    assert "methods_hpo_fold_state_from_package_json" in manifest["python_exports"]


def test_reader_forwards_package_and_returns_an_independent_validated_snapshot(tmp_path: Path) -> None:
    # A transport seam test, not synthetic proof of native state validity.
    # Full positive evidence is exercised by the SDK's mandatory native fold campaign.
    package = {"schema_version": 2, "execution_bundle": {"methods_hpo_fold_state": "native authority"}}
    state = {"outer_scopes": [{"winner_params": {"scale": False}}], "refit_scope": {"winner_params": {"n_components": 2}}}
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    with patch.object(dag_ml, "methods_hpo_fold_state_from_package_json", return_value=json.dumps(state)) as reader:
        with patch.object(dag_ml, "validate_methods_hpo_fold_state_json") as validator:
            result = dag_ml.methods_hpo_fold_state_from_package(path)
    reader.assert_called_once()
    assert json.loads(reader.call_args.args[0]) == package
    validator.assert_called_once_with(result.json())
    assert isinstance(result, dag_ml.MethodsFoldHpoState)
    first = result.to_dict()
    first["outer_scopes"][0]["winner_params"]["scale"] = True
    assert result.to_dict() == state


def test_native_reader_rejects_a_real_legacy_package_without_fold_state() -> None:
    root = Path(__file__).resolve().parents[3]
    package = root / "examples/fixtures/training/portable_predictor_package.v1.json"
    dag_ml.PortablePredictorPackage.from_path(package)
    with pytest.raises(dag_ml.DagMlError, match="no typed native fold HPO state"):
        dag_ml.methods_hpo_fold_state_from_package(package)


@pytest.mark.parametrize("value", [{}, {"schema_version": 1}, {"schema_version": 1, "unknown": True}, []])
def test_direct_native_state_validation_refuses_incomplete_or_unknown_wire(value: object) -> None:
    with pytest.raises(dag_ml.DagMlError):
        dag_ml.MethodsFoldHpoState(value)


def test_native_package_failure_is_preserved_without_state_fallback() -> None:
    error = dag_ml.DagMlValidationError("package authority rejected")
    with patch.object(dag_ml, "methods_hpo_fold_state_from_package_json", side_effect=error):
        with patch.object(dag_ml, "validate_methods_hpo_fold_state_json") as validator:
            with pytest.raises(dag_ml.DagMlValidationError, match="package authority rejected"):
                dag_ml.methods_hpo_fold_state_from_package({"untrusted": True})
    validator.assert_not_called()
