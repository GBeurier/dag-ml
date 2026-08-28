"""Facade-level contract for the closed in-process terminal PREDICT route."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import dag_ml


class TerminalPredictFacadeTests(unittest.TestCase):
    def test_facade_forwards_explicit_terminal_selector_and_mock_controller(self) -> None:
        callback_calls: list[dict[str, object]] = []

        def mock_controller(task: dict[str, object]) -> dict[str, object]:
            callback_calls.append(task)
            return {}

        captured: dict[str, object] = {}

        def native(
            dsl_json: str,
            envelope_json: str,
            manifests_json: str,
            callback: object,
            selection_metric: str,
            terminal_selector_json: str,
        ) -> str:
            captured.update(
                {
                    "dsl": json.loads(dsl_json),
                    "envelope": json.loads(envelope_json),
                    "manifests": json.loads(manifests_json),
                    "callback": callback,
                    "selection_metric": selection_metric,
                    "terminal_selector": json.loads(terminal_selector_json),
                }
            )
            return json.dumps(
                {
                    "execution_bundle": {"bundle_id": "bundle:terminal"},
                    "terminal_prediction": {"sample_ids": ["sample:holdout:1"]},
                    "terminal_receipt": {"cohort_fingerprint": "a" * 64},
                }
            )

        with patch.object(
            dag_ml, "_native_run_cv_refit_predict_in_process", side_effect=native
        ):
            result = dag_ml.run_cv_refit_predict_in_process(
                {"id": "dsl:terminal"},
                {"schema_version": 2, "predict_cohort": {"cohort_fingerprint": "a" * 64}},
                [{"controller_id": "controller:mock"}],
                mock_controller,
                "rmse",
                "model:terminal",
                "oof",
            )

        self.assertIn("run_cv_refit_predict_in_process", dag_ml.__all__)
        self.assertEqual(captured["dsl"], {"id": "dsl:terminal"})
        self.assertEqual(captured["selection_metric"], "rmse")
        self.assertEqual(
            captured["terminal_selector"], {"node_id": "model:terminal", "port": "oof"}
        )
        self.assertIs(captured["callback"], mock_controller)
        self.assertEqual(callback_calls, [])
        self.assertEqual(result["terminal_prediction"]["sample_ids"], ["sample:holdout:1"])


if __name__ == "__main__":
    unittest.main()
