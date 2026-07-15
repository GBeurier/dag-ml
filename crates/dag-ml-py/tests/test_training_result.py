"""Targeted Python tests for the owning native training surface."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import dag_ml

from parity.conformal.oracle import fingerprint_without


REPO = Path(__file__).resolve().parents[3]


class _FakeNativeTrainingResult:
    """Small facade probe; native execution itself is covered by Rust/PyO3 tests."""

    def __init__(self, outcome: dict[str, Any]) -> None:
        self._outcome = outcome
        self.is_attached = True
        self.process_local_artifact_count: int | None = 1
        self.process_local_data_handle_count: int | None = 4
        self.process_local_data_view_count: int | None = 8
        self.outcome_fingerprint = outcome["outcome_fingerprint"]

    def detach(self) -> bool:
        if not self.is_attached:
            return False
        self.is_attached = False
        self.process_local_artifact_count = None
        self.process_local_data_handle_count = None
        self.process_local_data_view_count = None
        return True

    def outcome_json(self) -> str:
        return json.dumps(self._outcome, separators=(",", ":"))

    def execution_bundle_json(self) -> str:
        return json.dumps(self._outcome["execution_bundle"], separators=(",", ":"))

    def score_set_json(self) -> str:
        return json.dumps(self._outcome["score_set"], separators=(",", ":"))

    def outputs_json(self) -> str:
        return json.dumps(self._outcome["outputs"], separators=(",", ":"))

    def artifacts_json(self) -> str:
        return json.dumps(
            self._outcome["execution_bundle"]["refit_artifacts"],
            separators=(",", ":"),
        )

    def portable_prediction_caches_json(self) -> str | None:
        payload = self._outcome["portable_prediction_caches"]
        return None if payload is None else json.dumps(payload, separators=(",", ":"))

    def replay_json(
        self,
        _request_json: str,
        _data_envelopes_json: str,
        _outcome_id: str,
        _run_id: str,
        _warnings_json: str,
        _diagnostics_json: str,
    ) -> str:
        raise RuntimeError("fake native replay is not implemented")


class _SuccessfulTrainingCallback:
    """Fixture controller that exercises the installed public training path."""

    def __init__(self, *, explicit_model_ports: bool = False) -> None:
        self.call_count = 0
        self._next_handle = 0
        self._explicit_model_ports = explicit_model_ports

    def _handle(self) -> int:
        self._next_handle += 1
        return self._next_handle

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        node_plan = task["node_plan"]
        node_id = node_plan["node_id"]
        phase = task["phase"]
        is_model = node_id == "model:base"
        fold_samples = {
            "fold:0": ["sample:1", "sample:2"],
            "fold:1": ["sample:3", "sample:4"],
        }
        sample_ids = fold_samples.get(
            task["fold_id"],
            ["sample:1", "sample:2", "sample:3", "sample:4"],
        )
        partition = "final" if phase in {"REFIT", "PREDICT", "EXPLAIN"} else "validation"
        predictions = []
        if is_model and phase in {"FIT_CV", "REFIT", "PREDICT", "EXPLAIN"}:
            predictions.append(
                {
                    "prediction_id": (
                        f"prediction:{node_id}:{phase}:{task['fold_id'] or 'full'}"
                    ),
                    "producer_node": node_id,
                    **(
                        {"producer_port": "oof"}
                        if self._explicit_model_ports
                        else {}
                    ),
                    "partition": partition,
                    "fold_id": task["fold_id"] if phase == "FIT_CV" else None,
                    "sample_ids": sample_ids,
                    "values": [[0.0] for _ in sample_ids],
                    "target_names": ["protein"],
                }
            )
        if self._explicit_model_ports and predictions:
            sibling = dict(predictions[0])
            sibling["prediction_id"] = f"{sibling['prediction_id']}:probability"
            sibling["producer_port"] = "probability"
            sibling["values"] = [
                [value + 100.0 for value in row] for row in sibling["values"]
            ]
            predictions.append(sibling)
        regression_targets = []
        if is_model and phase == "FIT_CV":
            regression_targets.append(
                {
                    "level": "sample",
                    "unit_ids": [
                        {"level": "sample", "id": sample_id} for sample_id in sample_ids
                    ],
                    "values": [[0.0] for _ in sample_ids],
                    "target_names": ["protein"],
                }
            )
        artifacts = []
        artifact_handles = {}
        if is_model and phase == "REFIT":
            artifact_id = "artifact:model.base.python.smoke"
            artifacts.append(
                {
                    "id": artifact_id,
                    "kind": "python_smoke_model",
                    "controller_id": node_plan["controller_id"],
                    "backend": None,
                    "uri": None,
                    "content_fingerprint": None,
                    "size_bytes": 8,
                    "plugin": None,
                    "plugin_version": None,
                }
            )
            artifact_handles[artifact_id] = {
                "handle": self._handle(),
                "kind": "artifact",
                "owner_controller": node_plan["controller_id"],
            }
        output_port = "oof" if is_model else "x_out"
        outputs = {
            output_port: {
                "handle": self._handle(),
                "kind": "prediction" if is_model else "data",
                "owner_controller": node_plan["controller_id"],
            }
        }
        if is_model and self._explicit_model_ports:
            outputs["probability"] = {
                "handle": self._handle(),
                "kind": "prediction",
                "owner_controller": node_plan["controller_id"],
            }
        explanations = []
        if is_model and phase == "EXPLAIN":
            explanations.append(
                {
                    "producer_node": node_id,
                    "producer_port": "oof",
                    "method": "shap",
                    "target_name": "protein",
                    "payload": {
                        "feature_names": ["feature:1"],
                        "values": [[0.0] for _ in sample_ids],
                    },
                }
            )
        return {
            "node_id": node_id,
            "outputs": outputs,
            "predictions": predictions,
            "observation_predictions": [],
            "aggregated_predictions": [],
            "explanations": explanations,
            "shape_deltas": [],
            "artifacts": artifacts,
            "artifact_handles": artifact_handles,
            "fit_influence_diagnostics": [],
            "regression_targets": regression_targets,
            "lineage": {
                "record_id": (
                    f"lineage:{node_id}:{phase}:"
                    f"{task['variant_id'] or 'base'}:{task['fold_id'] or 'full'}"
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


class TrainingResultTests(unittest.TestCase):
    def _predict_replay_request(
        self,
        outcome: dict[str, Any],
        data_envelopes: dict[str, Any],
    ) -> dict[str, Any]:
        request = {
            "schema_version": 1,
            "request_id": "replay:python.public.predict",
            "source_outcome_fingerprint": outcome["outcome_fingerprint"],
            "phase": "PREDICT",
            "data_envelope_keys": sorted(data_envelopes),
            "output_binding_ids": sorted(
                output["binding"]["binding_id"] for output in outcome["outputs"]
            ),
            "request_fingerprint": "0" * 64,
        }
        request["request_fingerprint"] = fingerprint_without(
            request, "request_fingerprint"
        )
        return request

    def _package_replay_request(
        self,
        package: dict[str, Any],
        data_envelopes: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        request = {
            "schema_version": 1,
            "request_id": f"replay:python.public.package.{phase.lower()}",
            "source_outcome_fingerprint": package["training_outcome"][
                "outcome_fingerprint"
            ],
            "phase": phase,
            "data_envelope_keys": sorted(data_envelopes),
            "output_binding_ids": sorted(
                binding["binding_id"] for binding in package["output_bindings"]
            ),
            "request_fingerprint": "0" * 64,
        }
        request["request_fingerprint"] = fingerprint_without(
            request, "request_fingerprint"
        )
        return request

    def _package_predict_replay_request(
        self,
        package: dict[str, Any],
        data_envelopes: dict[str, Any],
    ) -> dict[str, Any]:
        return self._package_replay_request(package, data_envelopes, phase="PREDICT")

    def _sidecar_artifact_handles(
        self, package: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        return {
            record["artifact"]["id"]: {
                "handle": 10_000 + index,
                "kind": "artifact",
                "owner_controller": record["controller_id"],
            }
            for index, record in enumerate(
                package["execution_bundle"]["refit_artifacts"], start=1
            )
        }

    def _with_explain_phase_support(self, request: dict[str, Any]) -> dict[str, Any]:
        explainable = json.loads(json.dumps(request))

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                if "PREDICT" in value and "EXPLAIN" not in value:
                    value.append("EXPLAIN")
                for child in value:
                    visit(child)

        visit(explainable)
        explainable["request_fingerprint"] = "0" * 64
        return dag_ml.sign_training_request(explainable).to_dict()

    def test_public_facade_executes_native_training_success(self) -> None:
        fixture = json.loads(
            (
                REPO / "examples/fixtures/training/python_training_smoke.v1.json"
            ).read_text(encoding="utf-8")
        )
        callback = _SuccessfulTrainingCallback()

        result = dag_ml.execute_training(
            fixture["request"],
            fixture["data_envelopes"],
            fixture["relations"],
            fixture["training_influence"],
            callback,
            outcome_id="outcome:python.public.smoke",
            run_id="run:python.public.smoke",
            bundle_id="bundle:python.public.smoke",
            diagnostics={"binding": "public_python_facade"},
        )

        outcome = result.outcome.to_dict()
        self.assertGreater(callback.call_count, 0)
        self.assertEqual(outcome["outcome_id"], "outcome:python.public.smoke")
        self.assertEqual(outcome["refit"]["status"], "completed")
        self.assertEqual(outcome["replayable_phases"], ["PREDICT"])
        self.assertTrue(result.score_set["reports"])
        self.assertEqual(len(result.outputs), 1)
        self.assertEqual(result.process_local_artifact_count, 1)
        package = result.export_portable_predictor_package(
            "predictor:python.public.package"
        ).to_dict()
        self.assertEqual(package["package_id"], "predictor:python.public.package")
        self.assertEqual(
            package["training_outcome"]["outcome_fingerprint"],
            outcome["outcome_fingerprint"],
        )
        self.assertEqual(
            len(package["artifact_bindings"]),
            len(outcome["execution_bundle"]["refit_artifacts"]),
        )
        self.assertTrue(
            all(
                binding["load_mode"] == "host_sidecar"
                for binding in package["artifact_bindings"]
            )
        )
        self.assertEqual(
            dag_ml.PortablePredictorPackage(
                result.portable_predictor_package_json(
                    "predictor:python.public.package.raw"
                )
            ).to_dict()["package_id"],
            "predictor:python.public.package.raw",
        )
        self.assertTrue(result.is_attached)
        self.assertTrue(result.detach())
        self.assertFalse(result.is_attached)
        self.assertIsNone(result.process_local_artifact_count)
        self.assertEqual(result.outcome.to_dict(), outcome)

    def test_public_facade_signs_training_request(self) -> None:
        fixture = json.loads(
            (
                REPO / "examples/fixtures/training/python_training_smoke.v1.json"
            ).read_text(encoding="utf-8")
        )
        unsigned = json.loads(json.dumps(fixture["request"]))
        unsigned["request_fingerprint"] = "0" * 64

        signed = dag_ml.sign_training_request(unsigned)

        self.assertEqual(
            signed.to_dict()["request_fingerprint"],
            fixture["request"]["request_fingerprint"],
        )
        self.assertEqual(signed.project().to_dict()["request_id"], unsigned["request_id"])

    def test_public_facade_filters_explicit_multi_prediction_port_outputs(self) -> None:
        fixture = json.loads(
            (
                REPO
                / "examples/fixtures/training/python_training_multiport_smoke.v1.json"
            ).read_text(encoding="utf-8")
        )
        callback = _SuccessfulTrainingCallback(explicit_model_ports=True)

        result = dag_ml.execute_training(
            fixture["request"],
            fixture["data_envelopes"],
            fixture["relations"],
            fixture["training_influence"],
            callback,
            outcome_id="outcome:python.public.multiport",
            run_id="run:python.public.multiport",
            bundle_id="bundle:python.public.multiport",
            diagnostics={"binding": "public_python_facade_multiport"},
        )

        outcome = result.outcome.to_dict()
        self.assertGreater(callback.call_count, 0)
        self.assertEqual(len(result.outputs), 1)
        output = result.outputs[0]
        self.assertEqual(output["binding"]["node_id"], "model:base")
        self.assertEqual(output["binding"]["port_name"], "oof")
        self.assertTrue(output["predictions"])
        self.assertTrue(
            all(
                block["producer_node"] == "model:base"
                and block.get("producer_port") == "oof"
                and block["partition"] == "final"
                and block["fold_id"] is None
                for block in output["predictions"]
            )
        )
        self.assertTrue(
            any(
                report["producer_node"] == "model:base"
                and report.get("producer_port") == "probability"
                for report in result.score_set["reports"]
            )
        )
        self.assertEqual(result.outcome.to_dict(), outcome)

    def test_public_facade_replays_attached_training_result_predict(self) -> None:
        fixture = json.loads(
            (
                REPO
                / "examples/fixtures/training/python_training_multiport_smoke.v1.json"
            ).read_text(encoding="utf-8")
        )
        callback = _SuccessfulTrainingCallback(explicit_model_ports=True)

        result = dag_ml.execute_training(
            fixture["request"],
            fixture["data_envelopes"],
            fixture["relations"],
            fixture["training_influence"],
            callback,
            outcome_id="outcome:python.public.replay.source",
            run_id="run:python.public.replay.source",
            bundle_id="bundle:python.public.replay.source",
        )
        source_outcome = result.outcome.to_dict()
        replay_request = self._predict_replay_request(
            source_outcome, fixture["data_envelopes"]
        )

        replay_outcome = result.replay(
            replay_request,
            fixture["data_envelopes"],
            outcome_id="outcome:python.public.replay.predict",
            run_id="run:python.public.replay.predict",
        ).to_dict()

        self.assertEqual(replay_outcome["phase"], "PREDICT")
        self.assertEqual(
            replay_outcome["source_training_outcome"]["outcome_fingerprint"],
            source_outcome["outcome_fingerprint"],
        )
        self.assertEqual(
            replay_outcome["replay_request_fingerprint"],
            replay_request["request_fingerprint"],
        )
        self.assertEqual(len(replay_outcome["outputs"]), 1)
        output = replay_outcome["outputs"][0]
        self.assertEqual(output["binding"]["port_name"], "oof")
        self.assertTrue(output["predictions"])
        self.assertTrue(
            all(
                block["producer_node"] == "model:base"
                and block.get("producer_port") == "oof"
                and block["partition"] == "final"
                and block["fold_id"] is None
                for block in output["predictions"]
            )
        )
        self.assertFalse(replay_outcome["explanations"])
        self.assertEqual(replay_outcome["prediction_cache_store"], False)

        self.assertTrue(result.detach())
        with self.assertRaisesRegex(dag_ml.DagMlError, "detached"):
            result.replay(
                replay_request,
                fixture["data_envelopes"],
                outcome_id="outcome:python.public.replay.detached",
                run_id="run:python.public.replay.detached",
            )

    def test_public_facade_replays_loaded_predictor_package_predict(self) -> None:
        fixture = json.loads(
            (
                REPO
                / "examples/fixtures/training/python_training_multiport_smoke.v1.json"
            ).read_text(encoding="utf-8")
        )
        training_callback = _SuccessfulTrainingCallback(explicit_model_ports=True)

        result = dag_ml.execute_training(
            fixture["request"],
            fixture["data_envelopes"],
            fixture["relations"],
            fixture["training_influence"],
            training_callback,
            outcome_id="outcome:python.public.package.source",
            run_id="run:python.public.package.source",
            bundle_id="bundle:python.public.package.source",
        )
        package = result.export_portable_predictor_package(
            "predictor:python.public.package.replay"
        ).to_dict()
        replay_request = self._package_predict_replay_request(
            package, fixture["data_envelopes"]
        )
        artifact_handles = self._sidecar_artifact_handles(package)

        self.assertTrue(result.detach())
        replay_callback = _SuccessfulTrainingCallback(explicit_model_ports=True)
        replay_outcome = dag_ml.replay_loaded_predictor_package(
            package,
            replay_request,
            fixture["data_envelopes"],
            artifact_handles,
            replay_callback,
            outcome_id="outcome:python.public.package.predict",
            run_id="run:python.public.package.predict",
        ).to_dict()

        self.assertGreater(replay_callback.call_count, 0)
        self.assertEqual(replay_outcome["phase"], "PREDICT")
        self.assertEqual(
            replay_outcome["source_training_outcome"],
            package["training_outcome"],
        )
        self.assertEqual(
            replay_outcome["replay_request_fingerprint"],
            replay_request["request_fingerprint"],
        )
        self.assertEqual(
            [output["binding"]["binding_id"] for output in replay_outcome["outputs"]],
            replay_request["output_binding_ids"],
        )
        output = replay_outcome["outputs"][0]
        self.assertEqual(output["binding"]["port_name"], "oof")
        self.assertTrue(output["predictions"])
        self.assertTrue(
            all(
                block["producer_node"] == "model:base"
                and block.get("producer_port") == "oof"
                and block["partition"] == "final"
                and block["fold_id"] is None
                for block in output["predictions"]
            )
        )
        self.assertFalse(replay_outcome["explanations"])
        self.assertEqual(replay_outcome["prediction_cache_store"], False)

    def test_public_facade_replays_loaded_predictor_package_explain(self) -> None:
        fixture = json.loads(
            (
                REPO
                / "examples/fixtures/training/python_training_multiport_smoke.v1.json"
            ).read_text(encoding="utf-8")
        )
        fixture["request"] = self._with_explain_phase_support(fixture["request"])
        training_callback = _SuccessfulTrainingCallback(explicit_model_ports=True)

        result = dag_ml.execute_training(
            fixture["request"],
            fixture["data_envelopes"],
            fixture["relations"],
            fixture["training_influence"],
            training_callback,
            outcome_id="outcome:python.public.package.explain.source",
            run_id="run:python.public.package.explain.source",
            bundle_id="bundle:python.public.package.explain.source",
        )
        self.assertEqual(result.outcome.to_dict()["replayable_phases"], ["PREDICT", "EXPLAIN"])
        package = result.export_portable_predictor_package(
            "predictor:python.public.package.explain"
        ).to_dict()
        replay_request = self._package_replay_request(
            package,
            fixture["data_envelopes"],
            phase="EXPLAIN",
        )
        artifact_handles = self._sidecar_artifact_handles(package)

        self.assertTrue(result.detach())
        replay_callback = _SuccessfulTrainingCallback(explicit_model_ports=True)
        replay_outcome = dag_ml.replay_loaded_predictor_package(
            package,
            replay_request,
            fixture["data_envelopes"],
            artifact_handles,
            replay_callback,
            outcome_id="outcome:python.public.package.explain",
            run_id="run:python.public.package.explain",
        ).to_dict()

        self.assertGreater(replay_callback.call_count, 0)
        self.assertEqual(replay_outcome["phase"], "EXPLAIN")
        self.assertEqual(
            replay_outcome["source_training_outcome"],
            package["training_outcome"],
        )
        self.assertEqual(
            replay_outcome["replay_request_fingerprint"],
            replay_request["request_fingerprint"],
        )
        self.assertTrue(replay_outcome["outputs"])
        self.assertTrue(replay_outcome["explanations"])
        self.assertTrue(
            all(
                block["producer_node"] == "model:base"
                and block.get("producer_port") == "oof"
                and block["partition"] == "final"
                and block["fold_id"] is None
                for output in replay_outcome["outputs"]
                for block in output["predictions"]
            )
        )
        self.assertEqual(replay_outcome["explanations"][0]["method"], "shap")
        self.assertEqual(replay_outcome["prediction_cache_store"], False)

    def test_facade_keeps_portable_views_after_explicit_detach(self) -> None:
        outcome = json.loads(
            (
                REPO / "examples/fixtures/training/training_outcome_refit.v1.json"
            ).read_text(encoding="utf-8")
        )
        result = dag_ml.TrainingResult(_FakeNativeTrainingResult(outcome))

        self.assertTrue(result.is_attached)
        self.assertEqual(result.process_local_artifact_count, 1)
        self.assertEqual(result.outcome.to_dict(), outcome)
        self.assertEqual(result.execution_bundle.to_dict(), outcome["execution_bundle"])
        self.assertEqual(result.score_set, outcome["score_set"])
        self.assertEqual(result.outputs, outcome["outputs"])
        self.assertEqual(
            result.artifacts, outcome["execution_bundle"]["refit_artifacts"]
        )

        self.assertTrue(result.detach())
        self.assertFalse(result.is_attached)
        self.assertIsNone(result.process_local_artifact_count)
        self.assertFalse(result.detach())
        self.assertEqual(result.outcome.to_dict(), outcome)

    def test_raw_entry_point_rejects_duplicate_envelope_keys(self) -> None:
        request = (
            REPO / "examples/fixtures/training/training_request_refit.v1.json"
        ).read_text(encoding="utf-8")
        with self.assertRaisesRegex(dag_ml.DagMlError, "duplicate JSON object key"):
            dag_ml.execute_training_json(
                request,
                '{"model:base.x":{},"model:base.x":{}}',
                '{"records":[]}',
                "{}",
                lambda _task: {},
                "outcome:duplicate",
                "run:duplicate",
                "bundle:duplicate",
            )


if __name__ == "__main__":
    unittest.main()
