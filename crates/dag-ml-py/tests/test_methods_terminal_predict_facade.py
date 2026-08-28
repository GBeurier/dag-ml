"""Public Python surface tests for the strict native Methods terminal facade."""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dag_ml
import dag_ml._dag_ml as native_module


REPO = Path(__file__).resolve().parents[3]


class _FakeNativeTrainingResult:
    """Enough of the owning result API to verify facade wrapping."""

    is_attached = True


class _FakeNativeTerminalReceipt:
    """Native-shaped sealed receipt stand-in for the pure Python facade test."""

    terminal_run_id = "run:strict.methods:methods-terminal-predict"
    receipt_fingerprint = "f" * 64

    def __init__(self) -> None:
        self._json = json.dumps(
            {
                "schema_version": 1,
                "terminal_run_id": self.terminal_run_id,
                "terminal_node_id": "model:base",
                "terminal_port": "oof",
                "receipt_fingerprint": self.receipt_fingerprint,
                "refit_artifacts": ["artifact:methods-pls:model:base:refit"],
            }
        )

    def json(self) -> str:
        return self._json


class MethodsTerminalPredictFacadeTests(unittest.TestCase):
    def test_surface_is_callback_free_and_returns_owned_result_package_prediction_and_receipt(
        self,
    ) -> None:
        package_json = (
            REPO
            / "examples"
            / "fixtures"
            / "training"
            / "portable_predictor_package.v1.json"
        ).read_text(encoding="utf-8")
        captured: dict[str, Any] = {}

        def native(*args: str) -> dict[str, object]:
            captured["request"] = json.loads(args[0])
            captured["predict_envelope"] = json.loads(args[5])
            captured["predict_input"] = json.loads(args[6])
            captured["selector"] = json.loads(args[12])
            return {
                "training_result": _FakeNativeTrainingResult(),
                "portable_predictor_package_json": package_json,
                "terminal_prediction_json": json.dumps(
                    {
                        "partition": "final",
                        "sample_ids": ["sample:predict:1"],
                        "target_names": ["protein"],
                    }
                ),
                "terminal_receipt": _FakeNativeTerminalReceipt(),
            }

        signature = inspect.signature(dag_ml.execute_methods_cv_refit_terminal_predict)
        json_signature = inspect.signature(
            dag_ml.execute_methods_cv_refit_terminal_predict_json
        )
        self.assertNotIn("op_callback", signature.parameters)
        self.assertNotIn("callback", signature.parameters)
        self.assertNotIn("op_callback", json_signature.parameters)
        self.assertNotIn("callback", json_signature.parameters)
        self.assertIn("MethodsTerminalPredictionResult", dag_ml.__all__)
        self.assertIn("MethodsTerminalPredictionReceipt", dag_ml.__all__)
        self.assertIn("execute_methods_cv_refit_terminal_predict", dag_ml.__all__)

        with (
            patch.object(
                dag_ml,
                "_native_execute_methods_cv_refit_terminal_predict_json",
                side_effect=native,
            ),
            patch.object(
                dag_ml,
                "_NativeMethodsTerminalPredictionReceipt",
                _FakeNativeTerminalReceipt,
            ),
        ):
            result = dag_ml.execute_methods_cv_refit_terminal_predict(
                {"request": "strict"},
                {"model:base.x": {"envelope": "training"}},
                {"records": []},
                {"entries": []},
                {
                    "model:base.x": {
                        "sample_ids": ["sample:1"],
                        "x": [[1.0]],
                        "y": [[2.0]],
                        "target_names": ["protein"],
                    }
                },
                {"schema_version": 2, "predict_cohort": {"role": "inference"}},
                {
                    "sample_ids": ["sample:predict:1"],
                    "x": [[3.0]],
                    "target_names": ["protein"],
                },
                methods_library_path=Path("/opt/lib/libn4m.so"),
                outcome_id="outcome:strict.methods",
                run_id="run:strict.methods",
                bundle_id="bundle:strict.methods",
                package_id="package:strict.methods",
                terminal_node_id="model:base",
                terminal_port="oof",
            )

        self.assertIsInstance(result, dag_ml.MethodsTerminalPredictionResult)
        self.assertIsInstance(result.training_result, dag_ml.TrainingResult)
        self.assertTrue(result.training_result.is_attached)
        self.assertIsInstance(
            result.portable_predictor_package, dag_ml.PortablePredictorPackage
        )
        self.assertEqual(result.terminal_prediction["partition"], "final")
        self.assertEqual(result.terminal_prediction["sample_ids"], ["sample:predict:1"])
        self.assertEqual(result.terminal_receipt["terminal_node_id"], "model:base")
        self.assertEqual(result.terminal_receipt["terminal_port"], "oof")
        self.assertEqual(
            result.terminal_receipt.terminal_run_id,
            "run:strict.methods:methods-terminal-predict",
        )
        self.assertEqual(result.terminal_receipt.receipt_fingerprint, "f" * 64)
        with self.assertRaises(TypeError):
            result.terminal_receipt["terminal_port"] = "forged"  # type: ignore[index]
        with self.assertRaises(AttributeError):
            result.terminal_receipt = result.terminal_receipt
        with self.assertRaises(AttributeError):
            result.terminal_receipt.terminal_run_id = "run:forged"  # type: ignore[misc]
        snapshot = result.terminal_receipt.to_dict()
        snapshot["terminal_port"] = "forged"
        self.assertEqual(result.terminal_receipt["terminal_port"], "oof")
        with self.assertRaises(TypeError):
            dag_ml.MethodsTerminalPredictionReceipt(_FakeNativeTerminalReceipt())
        with self.assertRaises(TypeError):
            native_module.MethodsTerminalPredictionReceipt()
        with self.assertRaises(TypeError):
            dag_ml.MethodsTerminalPredictionResult(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                {},
                result.terminal_receipt,
            )
        self.assertEqual(captured["request"], {"request": "strict"})
        self.assertEqual(captured["selector"], {"node_id": "model:base", "port": "oof"})
        self.assertEqual(
            captured["predict_input"],
            {
                "sample_ids": ["sample:predict:1"],
                "x": [[3.0]],
                "target_names": ["protein"],
            },
        )


if __name__ == "__main__":
    unittest.main()
