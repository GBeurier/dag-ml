"""Facade-level contract for the closed in-process terminal PREDICT route."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dag_ml
import dag_ml._dag_ml as native_module


REPO = Path(__file__).resolve().parents[3]


def _terminal_dsl() -> dict[str, Any]:
    """Return the smallest closed terminal-PREDICT campaign fixture."""

    return {
        "id": "dsl:terminal.predict.python-source",
        "input": {"name": "x", "representation": "tabular_numeric"},
        "campaign_id": "campaign:terminal.predict.python-source",
        "root_seed": 7,
        "leakage_policy": {
            "split_unit": "sample",
            "forbid_origin_cross_fold": True,
            "allow_observation_split_with_shared_target": False,
            "require_group_ids": False,
            "unsafe_flags": [],
        },
        "aggregation_policy": {
            "aggregation_level": "sample",
            "method": "mean",
            "weights": "none",
            "emit_parallel_metrics": True,
            "selection_metric_level": "sample",
            "store_raw_predictions": True,
            "store_aggregated_predictions": True,
        },
        "split_invocation": {
            "id": "split:terminal.predict.python-source",
            "controller_id": None,
            "leakage_policy": {
                "split_unit": "sample",
                "forbid_origin_cross_fold": True,
                "allow_observation_split_with_shared_target": False,
                "require_group_ids": False,
                "unsafe_flags": [],
            },
            "params": {},
            "fold_set": {
                "id": "folds:terminal.predict.python-source",
                "sample_ids": ["sample:1", "sample:2"],
                "folds": [
                    {
                        "fold_id": "fold:0",
                        "train_sample_ids": ["sample:2"],
                        "validation_sample_ids": ["sample:1"],
                        "metadata": {},
                    },
                    {
                        "fold_id": "fold:1",
                        "train_sample_ids": ["sample:1"],
                        "validation_sample_ids": ["sample:2"],
                        "metadata": {},
                    },
                ],
                "sample_groups": {},
            },
        },
        "data_bindings": [
            {
                "node_id": "model:terminal",
                "input_name": "x",
                "request_id": "nir-to-tabular",
                "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
                "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
                "relation_fingerprint": "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10",
                "output_representation": "tabular_numeric",
                "feature_set_id": "x",
                "source_ids": ["nir"],
                "require_relations": True,
            }
        ],
        "steps": [
            {
                "kind": "model",
                "id": "model:terminal",
                "operator": {"type": "TerminalMock"},
                "params": {},
            }
        ],
    }


def _terminal_manifest() -> list[dict[str, Any]]:
    return [
        {
            "controller_id": "controller:model.terminal",
            "controller_version": "0.1.0",
            "operator_kind": "model",
            "priority": 0,
            "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
            "input_ports": [],
            "output_ports": [],
            "data_requirements": None,
            "capabilities": [
                "deterministic",
                "thread_safe",
                "process_safe",
                "emits_predictions",
                "emits_artifacts",
                "stateful",
            ],
            "fit_scope": "fold_train",
            "rng_policy": "uses_core_seed",
            "artifact_policy": "serializable",
        }
    ]


def _terminal_envelope() -> dag_ml.JsonContract:
    base = json.loads(
        (
            REPO
            / "crates"
            / "dag-ml-core"
            / "tests"
            / "fixtures"
            / "package"
            / "data"
            / "coordinator_data_plan_envelope_sample12.json"
        ).read_text(encoding="utf-8")
    )
    return dag_ml.attach_predict_cohort_to_envelope(
        base,
        {
            "role": "external_test",
            "relations": {
                "records": [
                    {
                        "observation_id": "obs.H001",
                        "sample_id": "sample:holdout:1",
                        "target_id": "target:holdout:1",
                        "group_id": "group:holdout",
                        "origin_sample_id": None,
                        "source_id": "nir",
                        "is_augmented": False,
                    },
                    {
                        "observation_id": "obs.H002",
                        "sample_id": "sample:holdout:2",
                        "target_id": "target:holdout:2",
                        "group_id": "group:holdout",
                        "origin_sample_id": None,
                        "source_id": "nir",
                        "is_augmented": False,
                    },
                ]
            },
            "target_names": ["y"],
            "data_content_fingerprint": "c" * 64,
            "target_content_fingerprint": "d" * 64,
        },
    )


class _RealTerminalCallback:
    """Minimal real callback proving native CV -> REFIT -> PREDICT replay."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._next_handle = 0
        self.saw_refit_artifact = False

    def _handle(self) -> int:
        self._next_handle += 1
        return self._next_handle

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
        phase = task["phase"]
        self.calls.append(phase)
        node_plan = task["node_plan"]
        node_id = node_plan["node_id"]
        if phase == "FIT_CV":
            sample_ids = task["data_views"]["data:x:validation"]["sample_ids"]
            partition = "validation"
            fold_id = task["fold_id"]
        else:
            view = task["data_views"]["data:x"]
            sample_ids = view["sample_ids"]
            partition = "final"
            fold_id = None
            if phase == "PREDICT":
                self._assert_terminal_predict_input(task, view)

        artifacts: list[dict[str, Any]] = []
        artifact_handles: dict[str, dict[str, Any]] = {}
        if phase == "REFIT":
            artifact = {
                "id": "artifact:model:terminal:refit",
                "kind": "python_terminal_smoke_model",
                "controller_id": node_plan["controller_id"],
                "backend": None,
                "uri": None,
                "content_fingerprint": None,
                "size_bytes": 1,
                "plugin": None,
                "plugin_version": None,
            }
            artifacts = [artifact]
            artifact_handles[artifact["id"]] = {
                "handle": self._handle(),
                "kind": "model",
                "owner_controller": node_plan["controller_id"],
            }

        predictions: list[dict[str, Any]] = []
        if phase in {"FIT_CV", "PREDICT"}:
            predictions.append(
                {
                    "prediction_id": f"prediction:{node_id}:{phase}:{fold_id or 'terminal'}",
                    "producer_node": node_id,
                    "producer_port": "oof",
                    "partition": partition,
                    "fold_id": fold_id,
                    "sample_ids": sample_ids,
                    "values": [[0.0] for _ in sample_ids],
                    "target_names": ["y"],
                }
            )

        regression_targets: list[dict[str, Any]] = []
        if phase == "FIT_CV":
            regression_targets = [
                {
                    "level": "sample",
                    "unit_ids": [
                        {"level": "sample", "id": sample_id}
                        for sample_id in sample_ids
                    ],
                    "values": [[0.0] for _ in sample_ids],
                    "target_names": ["y"],
                }
            ]

        return {
            "node_id": node_id,
            "outputs": {
                "oof": {
                    "handle": self._handle(),
                    "kind": "prediction",
                    "owner_controller": node_plan["controller_id"],
                }
            },
            "predictions": predictions,
            "observation_predictions": [],
            "aggregated_predictions": [],
            "explanations": [],
            "shape_deltas": [],
            "fit_influence_diagnostics": [],
            "artifacts": artifacts,
            "artifact_handles": artifact_handles,
            "regression_targets": regression_targets,
            "lineage": {
                "record_id": (
                    f"lineage:{node_id}:{phase}:"
                    f"{task['variant_id'] or 'base'}:{task['fold_id'] or 'terminal'}"
                ),
                "run_id": task["run_id"],
                "node_id": node_id,
                "phase": phase,
                "controller_id": node_plan["controller_id"],
                "controller_version": node_plan["controller_version"],
                "variant_id": task["variant_id"],
                "fold_id": task["fold_id"],
                "branch_path": task["branch_path"],
                "input_lineage": [],
                "artifact_refs": artifacts,
                "params_fingerprint": node_plan["params_fingerprint"],
                "data_model_shape_fingerprint": None,
                "aggregation_policy_fingerprint": None,
                "seed": task["seed"],
                "unsafe_flags": [],
                "metrics": {},
            },
        }

    def _assert_terminal_predict_input(
        self, task: dict[str, Any], view: dict[str, Any]
    ) -> None:
        if view["partition"] != "predict" or view["fold_id"] is not None:
            raise AssertionError("PREDICT did not receive the terminal cohort view")
        artifact_id = "artifact:model:terminal:refit"
        artifact_key = f"artifact:{artifact_id}"
        if task["artifact_inputs"][artifact_key]["artifact"]["id"] != artifact_id:
            raise AssertionError("PREDICT did not receive the REFIT artifact metadata")
        if task["input_handles"][artifact_key]["kind"] not in {"model", "artifact"}:
            raise AssertionError("PREDICT did not receive a REFIT model/artifact handle")
        self.saw_refit_artifact = True


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

    def test_source_extension_executes_closed_terminal_replay_and_preflights_selector(
        self,
    ) -> None:
        module_path = Path(native_module.__file__).resolve()
        self.assertTrue(
            module_path.is_relative_to(REPO),
            f"terminal smoke must load the tracked source extension, got {module_path}",
        )
        callback = _RealTerminalCallback()
        result = dag_ml.run_cv_refit_predict_in_process(
            _terminal_dsl(),
            _terminal_envelope(),
            _terminal_manifest(),
            callback,
            "rmse",
            "model:terminal",
            "oof",
        )
        self.assertEqual(
            result["terminal_prediction"]["sample_ids"],
            ["sample:holdout:1", "sample:holdout:2"],
        )
        self.assertEqual(result["terminal_prediction"]["partition"], "final")
        self.assertEqual(result["terminal_receipt"]["terminal_node_id"], "model:terminal")
        self.assertEqual(result["terminal_receipt"]["terminal_port"], "oof")
        self.assertEqual(
            callback.calls,
            ["FIT_CV", "FIT_CV", "REFIT", "PREDICT"],
        )
        self.assertTrue(callback.saw_refit_artifact)

        selector_calls: list[dict[str, Any]] = []

        def forbidden_callback(task: dict[str, Any]) -> dict[str, Any]:
            selector_calls.append(task)
            raise AssertionError("terminal selector must fail before callback execution")

        with self.assertRaises(dag_ml.DagMlError) as raised:
            dag_ml.run_cv_refit_predict_in_process(
                _terminal_dsl(),
                _terminal_envelope(),
                _terminal_manifest(),
                forbidden_callback,
                "rmse",
                "model:terminal",
                "missing",
            )
        self.assertIn("has no output port `missing`", str(raised.exception))
        self.assertEqual(selector_calls, [])


if __name__ == "__main__":
    unittest.main()
