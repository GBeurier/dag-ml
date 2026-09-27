"""Callback-free CV/REFIT through the native Methods estimator controllers.

``run_cv_refit_methods_in_process`` runs the compat-DSL campaign of
``run_cv_refit_in_process`` with every node served by a native n4m role
controller. The tests need an explicit libn4m (``N4M_LIBRARY_PATH``); numeric
equality with direct n4m estimators is asserted by the dag-ml-core tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import unittest

import dag_ml
import dag_ml._dag_ml as native

LIBRARY = os.environ.get("N4M_LIBRARY_PATH")
FINGERPRINT = "a" * 64
N_SAMPLES = 12
N_FEATURES = 6


def _sample(index: int) -> str:
    return f"sample:{index:02d}"


def _envelope(*, content: bool = True, excluded: frozenset[int] = frozenset()) -> dict:
    relations = {
        "records": [
            {"observation_id": f"observation:{index:02d}", "sample_id": _sample(index)}
            | ({"excluded": True} if index in excluded else {})
            for index in range(N_SAMPLES)
        ]
    }
    envelope = {
        "schema_version": 1,
        "schema_fingerprint": FINGERPRINT,
        "plan_fingerprint": "b" * 64,
        "relation_fingerprint": dag_ml.sample_relation_set_fingerprint_json(json.dumps(relations)),
        "coordinator_relations": relations,
    }
    if content:
        envelope["data_content_fingerprint"] = "c" * 64
        envelope["target_content_fingerprint"] = "d" * 64
    return envelope


def _dsl(envelope: dict, steps: list[dict]) -> dict:
    ids = [_sample(index) for index in range(N_SAMPLES)]
    folds = []
    for fold in range(3):
        validation = ids[fold * 4:(fold + 1) * 4]
        folds.append({
            "fold_id": f"fold:{fold}",
            "train_sample_ids": [sample for sample in ids if sample not in validation],
            "validation_sample_ids": validation,
            "metadata": {},
        })
    return {
        "id": "methods-lane",
        "pipeline": steps,
        "split_invocation": {
            "id": "split:outer",
            "controller_id": None,
            "params": {"kind": "kfold", "n_splits": 3, "shuffle": False},
            "fold_set": {"id": "folds.outer", "sample_ids": ids, "folds": folds, "sample_groups": {}},
        },
        "data_bindings": [{
            "node_id": "transform:compat.0",
            "input_name": "x",
            "request_id": "nir-to-tabular",
            "schema_fingerprint": envelope["schema_fingerprint"],
            "plan_fingerprint": envelope["plan_fingerprint"],
            "relation_fingerprint": envelope["relation_fingerprint"],
            "output_representation": "tabular_numeric",
            "feature_set_id": "x",
            "source_ids": ["nir"],
            "require_relations": True,
        }],
    }


def _steps(model: str, **params: object) -> list[dict]:
    return [
        {"ref": "n4m:preprocessing.scatter.snv", "params": {"method_id": "preprocessing.scatter.snv"}},
        {"model": f"n4m:{model}", "params": {"method_id": model, **params}},
    ]


def _inputs(labels: bool = False) -> dict:
    x = [[math.sin(0.7 * row + 0.3 * col) + 0.1 * col for col in range(N_FEATURES)] for row in range(N_SAMPLES)]
    y = [[float(row % 3)] if labels else [0.5 * row + math.cos(row)] for row in range(N_SAMPLES)]
    return {"transform:compat.0.x": {
        "sample_ids": [_sample(index) for index in range(N_SAMPLES)],
        "x": x,
        "y": y,
        "target_names": ["y"],
    }}


def _run(dsl: dict, envelope: dict, inputs: dict, metric: str = "rmse", **options: object) -> tuple[dict, dict]:
    payload_json, payloads = native.run_cv_refit_methods_in_process(
        json.dumps(dsl), json.dumps(envelope), json.dumps(inputs), LIBRARY, metric, **options,
    )
    return json.loads(payload_json), payloads


@unittest.skipUnless(LIBRARY, "native Methods lane tests require N4M_LIBRARY_PATH")
class MethodsCvRefitInProcessTests(unittest.TestCase):
    def test_role_chain_trains_scores_and_exports_n4me_without_a_callback(self) -> None:
        envelope = _envelope()
        payload, payloads = _run(_dsl(envelope, _steps("models.pls.cppls", n_components=2)), envelope, _inputs())

        reports = {(report["partition"], report.get("fold_id")) for report in payload["scores"]["reports"]}
        for fold in ("fold:0", "fold:1", "fold:2"):
            for partition in ("validation", "train", "train_pool"):
                self.assertIn((partition, fold), reports)
        self.assertIn(("validation", "avg"), reports)
        self.assertIn(("final", None), reports)

        records = payload["refit_artifacts"]
        self.assertEqual(
            {(record["node_id"], record["controller_id"]) for record in records},
            {("transform:compat.0", "controller:n4m.transformer"), ("model:compat.1", "controller:n4m.regressor")},
        )
        self.assertEqual(set(payloads), {record["artifact"]["id"] for record in records})
        for record in records:
            artifact = record["artifact"]
            state = payloads[artifact["id"]]
            self.assertEqual(artifact["kind"], "n4m_estimator")
            self.assertEqual(artifact["content_fingerprint"], hashlib.sha256(state).hexdigest())
            self.assertEqual(artifact["native_estimator_descriptor"]["format"], "N4ME")

        # Identity-keyed OOF rows cover every sample exactly once.
        validation = [
            block
            for frame in payload["node_results"]
            for block in frame.get("predictions", [])
            if block["partition"] == "validation" and block["producer_node"] == "model:compat.1"
        ]
        self.assertEqual(sorted(sample for block in validation for sample in block["sample_ids"]), sorted(_inputs()["transform:compat.0.x"]["sample_ids"]))

    def test_parameter_sweep_selects_natively(self) -> None:
        envelope = _envelope()
        dsl = _dsl(envelope, _steps("models.pls.cppls", n_components=1))
        dsl["pipeline"][1]["generators"] = [{"kind": "range", "param": "n_components", "start": 1, "stop": 3, "step": 1}]
        payload, _ = _run(dsl, envelope, _inputs())
        self.assertEqual(len(payload["variant_catalog"]), 3)
        self.assertEqual(len(payload["selected_refit_variant_ids"]), 1)
        variants = {report.get("variant_id") for report in payload["scores"]["reports"] if report.get("fold_id") == "avg"}
        self.assertEqual(len(variants), 3)

    def test_classifier_scores_class_ids(self) -> None:
        envelope = _envelope()
        payload, _ = _run(
            _dsl(envelope, _steps("models.classification.pls_logistic", n_components=2)),
            envelope, _inputs(labels=True), metric="balanced_accuracy",
        )
        blocks = [
            block
            for frame in payload["node_results"]
            for block in frame.get("predictions", [])
            if block["producer_node"] == "model:compat.1"
        ]
        self.assertTrue(blocks)
        self.assertTrue(all(value[0] in (0.0, 1.0, 2.0) for block in blocks for value in block["values"]))
        probabilities = [
            block for frame in payload["node_results"] for block in frame.get("classification_probabilities", [])
        ]
        self.assertEqual({block["partition"] for block in probabilities}, {"train", "train_pool"})

    def test_host_operators_and_unattested_targets_are_refused(self) -> None:
        envelope = _envelope()
        steps = _steps("models.pls.cppls", n_components=2)
        steps[0] = {"class": "sklearn.preprocessing.StandardScaler"}
        with self.assertRaisesRegex(dag_ml.DagMlError, "controller"):
            _run(_dsl(envelope, steps), envelope, _inputs())

        unattested = _envelope(content=False)
        with self.assertRaisesRegex(dag_ml.DagMlError, "target-bound"):
            _run(_dsl(unattested, _steps("models.pls.cppls", n_components=2)), unattested, _inputs())

        # Inputs must exactly cover the plan's data bindings.
        with self.assertRaisesRegex(dag_ml.DagMlError, "exactly cover"):
            _run(_dsl(envelope, _steps("models.pls.cppls", n_components=2)), envelope, {})

    def test_fold_local_train_exclusion_trains_listed_rows_and_refit_excludes(self) -> None:
        # sample:05 is relation-excluded (the REFIT-level filter); the host's
        # fold-local filters kept it in fold:0 and fold:2's train lists.
        envelope = _envelope(excluded=frozenset({5}))

        def fit_rows(payload: dict, partition: str) -> dict:
            return {
                block.get("fold_id"): sorted(block["sample_ids"])
                for frame in payload["node_results"]
                for block in frame.get("predictions", [])
                if block["partition"] == partition and block["producer_node"] == "model:compat.1"
            }

        relations, _ = _run(_dsl(envelope, _steps("models.pls.cppls", n_components=2)), envelope, _inputs())
        fold_local_dsl = _dsl(envelope, _steps("models.pls.cppls", n_components=2))
        fold_local_dsl["split_invocation"]["fold_set"]["train_exclusion"] = "fold_local"
        fold_local, _ = _run(fold_local_dsl, envelope, _inputs())

        excluded = _sample(5)
        relation_train = fit_rows(relations, "train")
        local_train = fit_rows(fold_local, "train")
        for fold in ("fold:0", "fold:2"):
            self.assertNotIn(excluded, relation_train[fold])
            self.assertIn(excluded, local_train[fold])
            self.assertEqual(sorted(local_train[fold]), sorted([*relation_train[fold], excluded]))
        # Validation views are unchanged: the excluded sample is still validated in fold:1.
        self.assertEqual(fit_rows(relations, "validation"), fit_rows(fold_local, "validation"))
        self.assertIn(excluded, fit_rows(fold_local, "validation")["fold:1"])
        # The relation bit still removes it from the REFIT training cohort.
        self.assertEqual(fit_rows(relations, "final"), fit_rows(fold_local, "final"))
        self.assertNotIn(excluded, [sample for rows in fit_rows(fold_local, "final").values() for sample in rows])

    def test_contract_manifest_declares_the_lane(self) -> None:
        manifest = json.loads(dag_ml.contract_manifest_json())
        self.assertIn("run_cv_refit_methods_in_process", manifest["python_exports"])
        self.assertIn("execute_methods_cv_refit", manifest["capabilities"])


if __name__ == "__main__":
    unittest.main()
