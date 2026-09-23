"""Real extension checks for native host-search checkpoints and Python callbacks."""

from __future__ import annotations

import importlib.util
import json
import unittest
from hashlib import sha256
from typing import Any

import dag_ml
import dag_ml._dag_ml as native_module
from test_terminal_predict_facade import (
    REPO,
    _RealTerminalCallback,
    _terminal_dsl,
    _terminal_envelope,
    _terminal_manifest,
)


def _request(budget: int) -> dict[str, Any]:
    return {
        "target_node": "model:terminal",
        "trial_budget": budget,
        "metric": "rmse",
        "direction": "minimize",
        "optimizer_descriptor": {
            "owner": "deterministic-binding-test",
            "space": {"offset": [1.0, 2.0, 3.0]},
            "seed": 7,
        },
    }


class _Proposals:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, message: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append(message)
        if message["operation"] == "ask":
            return {"offset": float(message["trial_index"] + 1)}
        return None


class _Operators(_RealTerminalCallback):
    def __init__(self, *, fail_first: bool = False) -> None:
        super().__init__()
        self.fail_first = fail_first
        self.offsets: list[float] = []

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
        offset = task["node_plan"]["params"]["offset"]
        self.offsets.append(offset)
        if self.fail_first and offset == 1.0:
            raise ValueError("encoder failure for first candidate")
        result = super().__call__(task)
        for prediction in result["predictions"]:
            prediction["values"] = [[offset] for _ in prediction["sample_ids"]]
        return result


class HostHpoResumeTests(unittest.TestCase):
    def _run(
        self,
        budget: int,
        operator: _Operators,
        proposals: _Proposals,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dag_ml.run_host_hpo_search_in_process(
            _terminal_dsl(), _terminal_envelope(), _terminal_manifest(),
            _request(budget), operator, proposals, **kwargs,
        )

    def test_native_resume_preserves_scores_and_winner_without_replaying_trials(self) -> None:
        self.assertTrue(str(native_module.__file__).startswith(str(REPO)))
        operator, proposals = _Operators(), _Proposals()
        progress: list[dict[str, Any]] = []

        def stop_after_first(message: dict[str, Any]) -> bool:
            progress.append(message)
            return len(message["checkpoint"]["trials"]) < 1

        stopped = self._run(3, operator, proposals, progress_callback=stop_after_first)
        self.assertEqual(stopped["status"], "cancelled")
        self.assertEqual(operator.offsets, [1.0, 1.0])
        self.assertEqual([item["status"] for item in progress], ["running", "running", "cancelled"])
        checkpoint = json.loads(json.dumps(stopped["checkpoint"]))
        self.assertEqual(checkpoint, progress[-1]["checkpoint"])

        no_work_operator, no_work_proposals = _Operators(), _Proposals()
        completed = self._run(
            1, no_work_operator, no_work_proposals, resume_checkpoint=checkpoint,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(no_work_operator.offsets, [])
        self.assertEqual(no_work_proposals.calls, [])

        resumed_operator, resumed_proposals = _Operators(), _Proposals()
        resumed = self._run(
            3, resumed_operator, resumed_proposals, resume_checkpoint=checkpoint,
        )
        self.assertEqual(resumed_operator.offsets, [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(
            [(call["operation"], call["trial_index"]) for call in resumed_proposals.calls],
            [("ask", 1), ("tell", 1), ("ask", 2), ("tell", 2)],
        )
        self.assertEqual(resumed["selected_trial_index"], 0)
        self.assertEqual(resumed["trials"][0], stopped["trials"][0])
        uninterrupted = self._run(3, _Operators(), _Proposals(), progress_callback=lambda _: None)
        self.assertEqual(resumed, uninterrupted)
        self.assertEqual(operator.calls + resumed_operator.calls, ["FIT_CV"] * 6)

    def test_candidate_callback_factory_isolates_operator_state(self) -> None:
        fallback = _Operators()
        candidates: dict[int, _Operators] = {}

        def factory(trial_index: int) -> _Operators:
            self.assertNotIn(trial_index, candidates)
            operator = _Operators()
            candidates[trial_index] = operator
            return operator

        outcome = self._run(
            3, fallback, _Proposals(), candidate_callback_factory=factory,
        )
        self.assertEqual(outcome["selected_trial_index"], 0)
        self.assertEqual(fallback.calls, [])
        self.assertEqual(list(candidates), [0, 1, 2])
        self.assertEqual([operator.offsets for operator in candidates.values()],
                         [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    def test_python_operator_failure_is_checkpointed_before_error_then_skipped(self) -> None:
        operator, proposals = _Operators(fail_first=True), _Proposals()
        progress: list[dict[str, Any]] = []

        def capture(message: dict[str, Any]) -> bool:
            progress.append(message)
            return message["status"] != "failed"

        with self.assertRaisesRegex(dag_ml.DagMlRuntimeError, "encoder failure"):
            self._run(3, operator, proposals, progress_callback=capture)
        self.assertEqual(progress[-1]["status"], "failed")
        checkpoint = json.loads(json.dumps(progress[-1]["checkpoint"]))
        self.assertEqual(checkpoint["trials"][0]["state"], "failed")
        self.assertNotIn("score", checkpoint["trials"][0])
        self.assertEqual([item["operation"] for item in proposals.calls], ["ask", "fail"])
        resumed_operator, resumed_proposals = _Operators(), _Proposals()
        resumed = self._run(
            3, resumed_operator, resumed_proposals, resume_checkpoint=checkpoint,
        )
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed_operator.offsets, [2.0, 2.0, 3.0, 3.0])
        self.assertEqual([trial["trial_index"] for trial in resumed["trials"]], [1, 2])
        self.assertEqual(resumed["selected_trial_index"], 1)
        self.assertEqual(resumed["checkpoint"]["trials"][0], checkpoint["trials"][0])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "requires sklearn host operators")
class NestedHostHpoBindingTests(unittest.TestCase):
    def test_real_ridge_nested_oof_routes_three_nodes_and_resumes(self) -> None:
        import numpy as np
        from sklearn.linear_model import Ridge

        samples = [f"sample:{index}" for index in range(12)]
        dsl = _terminal_dsl()
        dsl["inner_cv"] = {"kind": "kfold", "n_splits": 2, "shuffle": False, "seed": 7}
        fold_set = dsl["split_invocation"]["fold_set"]
        fold_set["sample_ids"] = samples
        fold_set["folds"] = [
            {
                "fold_id": f"outer:{fold}",
                "train_sample_ids": [item for index, item in enumerate(samples) if index % 3 != fold],
                "validation_sample_ids": [item for index, item in enumerate(samples) if index % 3 == fold],
                "metadata": {},
            }
            for fold in range(3)
        ]
        envelope = _terminal_envelope().to_dict()
        envelope["schema_version"] = 1
        del envelope["predict_cohort"]
        envelope["coordinator_relations"]["records"] = [
            {
                "observation_id": f"obs:{sample}", "sample_id": sample,
                "target_id": f"target:{sample}", "group_id": None,
                "origin_sample_id": None, "source_id": "nir", "is_augmented": False,
            }
            for sample in samples
        ]
        envelope["relation_fingerprint"] = sha256(json.dumps(
            envelope["coordinator_relations"], sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        binding = dsl["data_bindings"][0]
        binding["relation_fingerprint"] = envelope["relation_fingerprint"]
        dsl["data_bindings"] = [{**binding, "node_id": f"model:{name}"} for name in ("a", "b")]
        dsl["steps"] = [
            {"kind": "branch", "branches": [
                {"id": name, "steps": [{
                    "kind": "model", "id": f"model:{name}",
                    "operator": {"type": "Ridge"}, "params": {"alpha": 1.0},
                }]}
                for name in ("a", "b")
            ]},
            {
                "kind": "merge_model", "id": "model:meta",
                "operator": {"type": "Ridge"}, "params": {"alpha": 1.0},
                "metadata": {"stacking_oof_execution": "nested_oof_v1"},
            },
        ]
        request = _request(3)
        request["target_node"] = "model:meta"
        request["fold_score_reduction"] = "mean"
        request["parameter_bindings"] = {
            f"{name}.alpha": {"node_id": f"model:{name}", "param_path": "alpha"}
            for name in ("a", "b", "meta")
        }
        request["optimizer_descriptor"]["space"] = {
            "a.alpha": [0.01, 1.0, 100.0], "b.alpha": [0.02, 2.0, 200.0],
            "meta.alpha": [0.03, 3.0, 300.0],
        }

        class Proposals(_Proposals):
            def __call__(self, message: dict[str, Any]) -> dict[str, Any] | None:
                self.calls.append(message)
                if message["operation"] == "ask":
                    alpha = [0.01, 1.0, 100.0][message["trial_index"]]
                    return {"a.alpha": alpha, "b.alpha": 2 * alpha, "meta.alpha": 3 * alpha}
                return None

        class Operators(_RealTerminalCallback):
            def __init__(self) -> None:
                super().__init__()
                self.fits: list[tuple[str, str, float]] = []

            def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
                assert task["phase"] == "FIT_CV"
                node = task["node_plan"]["node_id"]
                alpha = task["node_plan"]["params"]["alpha"]
                self.fits.append((task["variant_id"], node, alpha))

                def feature(ids: list[str]) -> Any:
                    return np.asarray([samples.index(item) / 5 - 1 for item in ids])[:, None]

                def target(ids: list[str]) -> Any:
                    x = feature(ids)[:, 0]
                    return x + 2 * x ** 2 + np.sin(x)

                if node == "model:meta":
                    inner = [value for key, value in sorted(task["prediction_inputs"].items()) if not key.endswith(":outer")]
                    outer = [value for key, value in sorted(task["prediction_inputs"].items()) if key.endswith(":outer")]
                    train_ids, valid_ids = inner[0]["sample_ids"], outer[0]["sample_ids"]
                    assert all(value["sample_ids"] == train_ids for value in inner)
                    assert all(value["sample_ids"] == valid_ids for value in outer)
                    assert all(task["fold_id"] not in value["fold_ids"] for value in inner)
                    x_train = np.column_stack([value["values"] for value in inner])
                    x_valid = np.column_stack([value["values"] for value in outer])
                    # The helper only constructs result metadata; the native OOF
                    # inputs above remain the exclusive meta fitting features.
                    task = {**task, "data_views": {"data:x:validation": {"sample_ids": valid_ids}}}
                else:
                    train_ids = task["data_views"]["data:x"]["sample_ids"]
                    valid_ids = task["data_views"]["data:x:validation"]["sample_ids"]
                    power = 1 if node == "model:a" else 2
                    x_train, x_valid = feature(train_ids) ** power, feature(valid_ids) ** power
                assert set(train_ids).isdisjoint(valid_ids)
                fitted = Ridge(alpha=alpha).fit(x_train, target(train_ids))
                result = super().__call__(task)
                result["predictions"][0]["values"] = fitted.predict(x_valid)[:, None].tolist()
                result["regression_targets"][0]["values"] = target(valid_ids)[:, None].tolist()
                return result

        def run(operator: Operators, proposals: Proposals, **kwargs: Any) -> dict[str, Any]:
            manifests = _terminal_manifest()
            manifests[0]["capabilities"].append("consumes_oof_predictions")
            return dag_ml.run_host_hpo_search_in_process(
                dsl, envelope, manifests, request, operator, proposals, **kwargs,
            )

        first_operator, first_proposals = Operators(), Proposals()
        stopped = run(first_operator, first_proposals, progress_callback=lambda message: len(message["checkpoint"]["trials"]) < 1)
        self.assertEqual(stopped["status"], "cancelled")
        self.assertEqual(len(first_operator.fits), 21)
        self.assertEqual({(node, alpha) for _, node, alpha in first_operator.fits}, {
            ("model:a", 0.01), ("model:b", 0.02), ("model:meta", 0.03),
        })
        checkpoint = json.loads(json.dumps(stopped["checkpoint"]))
        request["parameter_bindings"]["a.alpha"]["param_path"] = "tol"
        rejected_operator, rejected_proposals = Operators(), Proposals()
        with self.assertRaisesRegex(dag_ml.DagMlRuntimeError, "binding mismatch"):
            run(rejected_operator, rejected_proposals, resume_checkpoint=checkpoint)
        self.assertEqual(rejected_operator.fits, [])
        self.assertEqual(rejected_proposals.calls, [])
        request["parameter_bindings"]["a.alpha"]["param_path"] = "alpha"
        resumed_operator, resumed_proposals = Operators(), Proposals()
        resumed = run(resumed_operator, resumed_proposals, resume_checkpoint=checkpoint)
        self.assertEqual(len(resumed_operator.fits), 42)
        self.assertTrue(all(variant != "host_hpo:trial:0000000000" for variant, _, _ in resumed_operator.fits))
        self.assertEqual([call["trial_index"] for call in resumed_proposals.calls if call["operation"] == "ask"], [1, 2])
        self.assertEqual(resumed, run(Operators(), Proposals(), progress_callback=lambda _: None))
        self.assertEqual(resumed["trials"][0], stopped["trials"][0])
        self.assertEqual(len(resumed["trials"][0]["objective_fold_scores"]), 3)
        self.assertEqual(resumed["selected_trial_index"], min(resumed["trials"], key=lambda trial: trial["score"])["trial_index"])


if __name__ == "__main__":
    unittest.main()
