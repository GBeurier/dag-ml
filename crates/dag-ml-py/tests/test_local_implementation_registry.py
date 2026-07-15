"""Python facade tests for process-local custom implementations."""

from __future__ import annotations

import gc
import json
import pickle
import unittest
import weakref
from copy import deepcopy
from pathlib import Path

import dag_ml


REPO = Path(__file__).resolve().parents[3]


class LocalImplementationRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = json.loads(
            (REPO / "examples/fixtures/criteria/criteria_contracts.v1.json").read_text()
        )
        cls.role = fixture["valid"]["training_loss_role"]
        cls.loss = cls.role["loss"]
        cls.binding_fixture = json.loads(
            (
                REPO / "examples/fixtures/criteria/python_local_implementations.v1.json"
            ).read_text()
        )

    def test_python_loss_is_local_resolvable_and_executable(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()

        def absolute_error(target: float, prediction: float) -> float:
            return abs(prediction - target)

        registry.register_loss(self.loss, absolute_error)
        resolved = registry.resolve_training_loss(self.role, "FIT_CV")

        self.assertIs(resolved, absolute_error)
        self.assertEqual(resolved(2.0, 5.5), 3.5)
        self.assertEqual(len(registry), 1)
        self.assertEqual(
            registry.descriptors()[0]["descriptor_fingerprint"],
            self.loss["implementation"]["descriptor_fingerprint"],
        )

    def test_registry_is_explicitly_process_local(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()
        registry.register_loss(
            self.loss, lambda target, prediction: prediction - target
        )

        with self.assertRaisesRegex(TypeError, "cannot be serialized"):
            pickle.dumps(registry)

    def test_registry_retains_and_clear_releases_python_callable(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()

        class LocalLoss:
            def __call__(self, target: float, prediction: float) -> float:
                return prediction - target

        implementation = LocalLoss()
        reference = weakref.ref(implementation)
        registry.register_loss(self.loss, implementation)
        del implementation
        gc.collect()
        self.assertIsNotNone(reference())

        registry.clear()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(len(registry), 0)

    def test_attestation_matches_refit_role(self) -> None:
        attestation = dag_ml.loss_execution_attestation(self.role, "REFIT")

        self.assertEqual(attestation["phase"], "REFIT")
        self.assertEqual(attestation["loss_id"], self.loss["spec"]["loss_id"])
        with self.assertRaises(dag_ml.DagMlValidationError):
            dag_ml.loss_execution_attestation(self.role, "PREDICT")

    def test_native_task_invokes_loss_in_cv_and_refit_before_attesting(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()
        calls: list[tuple[float, float]] = []

        def asymmetric_loss(target: float, prediction: float) -> float:
            calls.append((target, prediction))
            return (prediction - target) ** 2

        registry.register_loss(self.binding_fixture["loss_reference"], asymmetric_loss)
        for phase in ("FIT_CV", "REFIT"):
            invocation = registry.invoke_training_loss(
                self.binding_fixture["tasks"][phase], 2.0, 5.0
            )
            self.assertEqual(invocation["value"], 9.0)
            self.assertEqual(invocation["attestation"]["phase"], phase)
        self.assertEqual(calls, [(2.0, 5.0), (2.0, 5.0)])

    def test_failed_loss_does_not_return_an_attestation(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()

        def failing_loss(*_args: object) -> float:
            raise ValueError("local loss failed")

        registry.register_loss(self.binding_fixture["loss_reference"], failing_loss)
        with self.assertRaisesRegex(dag_ml.DagMlRuntimeError, "local loss failed"):
            registry.invoke_training_loss(
                self.binding_fixture["tasks"]["FIT_CV"], 2.0, 5.0
            )

    def test_tampered_native_loss_requirement_fails_before_callback(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()
        calls = 0

        def asymmetric_loss(*_args: object) -> float:
            nonlocal calls
            calls += 1
            return 0.0

        registry.register_loss(self.binding_fixture["loss_reference"], asymmetric_loss)
        task = deepcopy(self.binding_fixture["tasks"]["FIT_CV"])
        task["required_loss_attestations"] = []

        with self.assertRaises(dag_ml.DagMlRuntimeError):
            registry.invoke_training_loss(task, 2.0, 5.0)
        self.assertEqual(calls, 0)

    def test_new_process_registry_requires_explicit_reconstruction(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()
        with self.assertRaises(dag_ml.DagMlRuntimeError):
            registry.invoke_training_loss(
                self.binding_fixture["tasks"]["FIT_CV"], 2.0, 5.0
            )

    def test_typed_metric_task_executes_and_native_code_builds_result(self) -> None:
        registry = dag_ml.LocalImplementationRegistry()
        task = self.binding_fixture["metric_task"]

        def bias_metric(metric_task: dict[str, object]) -> list[dict[str, float]]:
            predictions = metric_task["predictions"]
            targets = metric_task["targets"]
            values = [
                prediction[0] - target[0]
                for prediction, target in zip(predictions, targets, strict=True)
            ]
            return [{"value": sum(values) / len(values)}]

        registry.register_metric(self.binding_fixture["metric_reference"], bias_metric)
        evaluation = registry.evaluate_metric(task)

        self.assertEqual(evaluation["aggregate"], 1.5)
        self.assertEqual(evaluation["result"]["request_id"], task["request_id"])
        self.assertEqual(
            evaluation["result"]["descriptor_fingerprint"],
            task["metric"]["implementation"]["descriptor_fingerprint"],
        )

    def test_metric_callback_exception_and_non_finite_value_fail_closed(self) -> None:
        task = self.binding_fixture["metric_task"]
        failing = dag_ml.LocalImplementationRegistry()
        failing.register_metric(
            self.binding_fixture["metric_reference"],
            lambda _task: (_ for _ in ()).throw(ValueError("metric failed")),
        )
        with self.assertRaisesRegex(dag_ml.DagMlRuntimeError, "metric failed"):
            failing.evaluate_metric(task)

        non_finite = dag_ml.LocalImplementationRegistry()
        non_finite.register_metric(
            self.binding_fixture["metric_reference"],
            lambda _task: [{"value": float("nan")}],
        )
        with self.assertRaises(dag_ml.DagMlRuntimeError):
            non_finite.evaluate_metric(task)


if __name__ == "__main__":
    unittest.main()
