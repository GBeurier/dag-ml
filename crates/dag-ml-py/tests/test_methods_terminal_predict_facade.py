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


class MethodsTerminalPredictFacadeTests(unittest.TestCase):
    def test_surface_is_callback_free_and_forwards_the_native_result(self) -> None:
        captured: dict[str, Any] = {}
        native_result = object()

        def native(*args: str) -> object:
            captured["request"] = json.loads(args[0])
            captured["predict_envelope"] = json.loads(args[5])
            captured["predict_input"] = json.loads(args[6])
            captured["selector"] = json.loads(args[12])
            return native_result

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

        self.assertIs(result, native_result)
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

    def test_authoritative_result_and_receipt_are_native_nonconstructible_types(
        self,
    ) -> None:
        self.assertIs(
            dag_ml.MethodsTerminalPredictionReceipt,
            native_module.MethodsTerminalPredictionReceipt,
        )
        self.assertIs(
            dag_ml.MethodsTerminalPredictionResult,
            native_module.MethodsTerminalPredictionResult,
        )
        for native_type in (
            dag_ml.MethodsTerminalPredictionReceipt,
            dag_ml.MethodsTerminalPredictionResult,
        ):
            with self.assertRaises(TypeError):
                native_type()
            with self.assertRaises(TypeError):
                object.__new__(native_type)


if __name__ == "__main__":
    unittest.main()
