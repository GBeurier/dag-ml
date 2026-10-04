"""Parallel profile wire guards and the real native Python worker bridge.

These callbacks qualify orchestration, not Methods numerical thread behavior;
the SDK and Methods suites supply the actual native-model overlap/oracle cases.
"""

from __future__ import annotations

import copy
import json
import threading
import unittest

import dag_ml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from scripts.validate_archive_v2_contract import contract_schema_registry
from test_host_hpo_resume import _Operators, _Proposals, _request
from test_terminal_predict_facade import (
    REPO,
    _terminal_dsl,
    _terminal_envelope,
    _terminal_manifest,
)


def _parallel_request(budget: int, workers: int) -> dict:
    request = _request(budget)
    request["optimizer_descriptor"].update(
        n_jobs=workers,
        sampler="random",
        pruner=None,
        parallel_execution={
            "schema_version": 1,
            "profile": "methods_sequential_cpu_v1",
            "workers": workers,
            "cpu_threads": 1,
            "gpu_devices": [],
            "methods_build": {
                "schema_version": 1, "blas": False, "openmp": False, "cuda": False,
            },
        },
    )
    return request


class HostHpoParallelProfileTests(unittest.TestCase):
    def test_profile_schema_is_closed_and_legacy_descriptor_is_unchanged(self) -> None:
        schemas, registry = contract_schema_registry(REPO)
        validator = Draft202012Validator(
            schemas["https://github.com/GBeurier/dag-ml/schemas/host_hpo_search_request.v1.schema.json"],
            registry=registry,
        )
        validator.validate(_request(1))
        for workers in (2, 3, 4):
            validator.validate(_parallel_request(1, workers))
        mutations = (
            ("schema_version", 2), ("workers", 5), ("workers", True),
            ("profile", "accelerated"), ("cpu_threads", 2),
            ("gpu_devices", ["0"]), ("unknown", False),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                request = _parallel_request(1, 2)
                request["optimizer_descriptor"]["parallel_execution"][key] = value
                with self.assertRaises(ValidationError):
                    validator.validate(request)
        for key, value in (("schema_version", 2), ("blas", True), ("openmp", True),
                           ("cuda", True), ("blas", 0), ("unknown", False)):
            with self.subTest(build_key=key, value=value):
                request = _parallel_request(1, 2)
                request["optimizer_descriptor"]["parallel_execution"]["methods_build"][key] = value
                with self.assertRaises(ValidationError):
                    validator.validate(request)
        for key, value in (("n_jobs", -1), ("n_jobs", 3), ("sampler", "tpe"), ("pruner", "median"),
                           ("generated_view_mode", "checkpoint_manifest_v1")):
            request = _parallel_request(1, 2)
            request["optimizer_descriptor"][key] = value
            with self.assertRaises(ValidationError):
                validator.validate(request)
        request = _parallel_request(1, 2)
        request["progressive_pruning"] = True
        with self.assertRaises(ValidationError):
            validator.validate(request)
        request = _parallel_request(1, 2)
        request["metric"] = "accuracy"
        with self.assertRaises(ValidationError):
            validator.validate(request)

    def test_native_profile_refusal_precedes_every_host_callback(self) -> None:
        for mutation in ("blas", "cuda", "unknown", "budget", "serial", "pruner", "generated"):
            with self.subTest(mutation=mutation):
                request = _parallel_request(1, 2)
                descriptor = request["optimizer_descriptor"]
                profile = descriptor["parallel_execution"]
                if mutation in ("blas", "cuda"):
                    profile["methods_build"][mutation] = True
                elif mutation == "unknown":
                    profile["unknown"] = False
                elif mutation == "budget":
                    profile["workers"] = 3
                elif mutation == "serial":
                    descriptor["n_jobs"] = 1
                elif mutation == "pruner":
                    descriptor["pruner"] = "median"
                else:
                    descriptor["generated_view_mode"] = "checkpoint_manifest_v1"
                calls = []

                def forbidden(message):
                    calls.append(message)
                    raise AssertionError("profile refusal must precede host callbacks")

                with self.assertRaises(dag_ml.DagMlRuntimeError):
                    dag_ml.run_host_hpo_search_in_process(
                        _terminal_dsl(), _terminal_envelope(), _terminal_manifest(),
                        request, forbidden, forbidden,
                        candidate_callback_factory=forbidden, progress_callback=forbidden,
                    )
                self.assertEqual(calls, [])

    def test_partial_worker_windows_bind_resources_and_join_before_progress(self) -> None:
        for budget, workers in ((1, 4), (3, 2), (5, 4)):
            with self.subTest(budget=budget, workers=workers):
                lock = threading.Lock()
                barrier = threading.Barrier(min(budget, workers))
                active = 0
                maximum = 0
                candidates = {}

                class Operator(_Operators):
                    def __init__(self, index):
                        super().__init__()
                        self.index = index

                    def __call__(self, task):
                        nonlocal active, maximum
                        self_test.assertEqual(task["resources"], {"cpu_threads": 1, "gpu_devices": []})
                        with lock:
                            active += 1
                            maximum = max(maximum, active)
                        try:
                            if self.index < min(budget, workers) and not self.calls:
                                barrier.wait(timeout=5)
                            return super().__call__(task)
                        finally:
                            with lock:
                                active -= 1

                self_test = self

                def factory(index):
                    operator = Operator(index)
                    candidates[index] = operator
                    return operator

                def progress(message):
                    self.assertEqual(active, 0, "every admitted worker must join before progress")
                    return True

                proposals = _Proposals()
                fallback = _Operators()
                outcome = dag_ml.run_host_hpo_search_in_process(
                    _terminal_dsl(), _terminal_envelope(), _terminal_manifest(),
                    _parallel_request(budget, workers), fallback, proposals,
                    candidate_callback_factory=factory, progress_callback=progress,
                )
                self.assertEqual(outcome["status"], "completed")
                self.assertEqual(outcome["selected_trial_index"], 0)
                self.assertEqual(fallback.calls, [])
                self.assertEqual(sorted(candidates), list(range(budget)))
                self.assertGreaterEqual(maximum, min(budget, workers))
                self.assertEqual(active, 0)
                self.assertEqual([message["trial_index"] for message in proposals.calls
                                  if message["operation"] == "tell"], list(range(budget)))

    def test_changed_resource_profile_refuses_resume_without_callbacks_or_checkpoint_mutation(self) -> None:
        request = _parallel_request(3, 2)

        def stop(message):
            return len(message["checkpoint"]["trials"]) < 1

        stopped = dag_ml.run_host_hpo_search_in_process(
            _terminal_dsl(), _terminal_envelope(), _terminal_manifest(), request,
            _Operators(), _Proposals(), candidate_callback_factory=lambda _: _Operators(),
            progress_callback=stop,
        )
        self.assertEqual(stopped["status"], "cancelled")
        checkpoint = stopped["checkpoint"]
        self.assertEqual(len(checkpoint["trials"]), 2)
        before = json.dumps(checkpoint, sort_keys=True)
        changed = copy.deepcopy(request)
        changed["optimizer_descriptor"]["n_jobs"] = 3
        changed["optimizer_descriptor"]["parallel_execution"]["workers"] = 3
        calls = []

        def forbidden(message):
            calls.append(message)
            raise AssertionError("changed profile must refuse before callbacks")

        with self.assertRaises(dag_ml.DagMlRuntimeError):
            dag_ml.run_host_hpo_search_in_process(
                _terminal_dsl(), _terminal_envelope(), _terminal_manifest(), changed,
                forbidden, forbidden, candidate_callback_factory=forbidden,
                progress_callback=forbidden, resume_checkpoint=checkpoint,
            )
        self.assertEqual(calls, [])
        self.assertEqual(json.dumps(checkpoint, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
