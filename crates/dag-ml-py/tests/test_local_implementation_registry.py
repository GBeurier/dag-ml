"""Python facade tests for process-local custom implementations."""

from __future__ import annotations

import json
import pickle
import unittest
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
        registry.register_loss(self.loss, lambda target, prediction: prediction - target)

        with self.assertRaisesRegex(TypeError, "cannot be serialized"):
            pickle.dumps(registry)

    def test_attestation_matches_refit_role(self) -> None:
        attestation = dag_ml.loss_execution_attestation(self.role, "REFIT")

        self.assertEqual(attestation["phase"], "REFIT")
        self.assertEqual(attestation["loss_id"], self.loss["spec"]["loss_id"])
        with self.assertRaises(dag_ml.DagMlValidationError):
            dag_ml.loss_execution_attestation(self.role, "PREDICT")


if __name__ == "__main__":
    unittest.main()
