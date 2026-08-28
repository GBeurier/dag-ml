"""Public Python surface tests for the strict native Methods terminal facade."""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dag_ml


REPO = Path(__file__).resolve().parents[3]


class _FakeNativeTrainingResult:
    """Enough of the owning result API to verify facade wrapping."""

    is_attached = True


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
                "terminal_receipt_json": json.dumps(
                    {
                        "schema_version": 1,
                        "terminal_node_id": "model:base",
                        "terminal_port": "oof",
                        "refit_artifacts": ["artifact:methods-pls:model:base:refit"],
                    }
                ),
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
        self.assertIn("execute_methods_cv_refit_terminal_predict", dag_ml.__all__)

        with patch.object(
            dag_ml,
            "_native_execute_methods_cv_refit_terminal_predict_json",
            side_effect=native,
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
        self.assertIsInstance(result.portable_predictor_package, dag_ml.PortablePredictorPackage)
        self.assertEqual(result.terminal_prediction["partition"], "final")
        self.assertEqual(result.terminal_prediction["sample_ids"], ["sample:predict:1"])
        self.assertEqual(result.terminal_receipt["terminal_node_id"], "model:base")
        self.assertEqual(result.terminal_receipt["terminal_port"], "oof")
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
