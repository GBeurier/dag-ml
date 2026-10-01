"""Replay the exact WASM DAG/parameters through native Python callbacks.

The qualification replays recorded optimizer proposals to isolate operator and
scheduler parity. It does not substitute a Python CV loop or a new optimizer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import dag_ml
import numpy as np
from n4m.roles import RolePipeline


def main() -> None:
    fixture = json.loads(Path(sys.argv[1]).read_text())
    assert (
        dag_ml.sample_relation_set_fingerprint_json(
            json.dumps(fixture["envelope"]["coordinator_relations"])
        )
        == fixture["envelope"]["relation_fingerprint"]
    )
    samples = fixture["sampleIds"]
    sources = {
        name: np.asarray(rows, dtype=float)
        for name, rows in fixture["sourceRows"].items()
    }
    target = np.asarray(fixture["target"], dtype=float)
    indices = {sample: index for index, sample in enumerate(samples)}
    fits = []
    proposals = fixture["resumed"]["trials"]

    def optimizer(message):
        if message["operation"] == "ask":
            return proposals[message["trial_index"]]["params"]
        if message["operation"] == "report_intermediate":
            return False
        return None

    def operator(task):
        node = task["node_plan"]
        assert task["phase"] == "FIT_CV"
        if task["prediction_inputs"]:
            inner = [
                value
                for key, value in sorted(task["prediction_inputs"].items())
                if not key.endswith(":outer")
            ]
            outer = [
                value
                for key, value in sorted(task["prediction_inputs"].items())
                if key.endswith(":outer")
            ]
            train_ids, valid_ids = inner[0]["sample_ids"], outer[0]["sample_ids"]
            assert all(value["sample_ids"] == train_ids for value in inner)
            assert all(value["sample_ids"] == valid_ids for value in outer)
            assert all(value["partition"] == "validation" for value in inner + outer)
            assert all(task["fold_id"] not in value["fold_ids"] for value in inner)
            x_train = np.column_stack([value["values"] for value in inner])
            x_valid = np.column_stack([value["values"] for value in outer])
        else:
            train, validation = (
                task["data_views"]["data:x"],
                task["data_views"]["data:x:validation"],
            )
            (source,) = train["source_ids"]
            train_ids, valid_ids = train["sample_ids"], validation["sample_ids"]
            x_train = sources[source][[indices[sample] for sample in train_ids]]
            x_valid = sources[source][[indices[sample] for sample in valid_ids]]
        assert set(train_ids).isdisjoint(valid_ids)
        y_train = target[[indices[sample] for sample in train_ids]]
        fitted = RolePipeline.from_steps(
            [{"class": "n4m:models.regularized.ridge", "params": node["params"]}]
        )
        fitted.fit(x_train, y_train)
        prediction = (
            np.asarray(fitted.predict(x_valid), dtype=float).reshape(-1, 1).tolist()
        )
        fits.append(
            {
                "node": node["node_id"],
                "fold": task["fold_id"],
                "alpha": node["params"]["alpha"],
                "train_ids": train_ids,
                "validation_ids": valid_ids,
                "width": x_train.shape[1],
            }
        )
        return {
            "node_id": node["node_id"],
            "outputs": {},
            "predictions": [
                {
                    "producer_node": node["node_id"],
                    "partition": "validation",
                    "fold_id": task["fold_id"],
                    "sample_ids": valid_ids,
                    "values": prediction,
                    "target_names": ["y"],
                }
            ],
            "regression_targets": [
                {
                    "level": "sample",
                    "unit_ids": [
                        {"level": "sample", "id": sample} for sample in valid_ids
                    ],
                    "values": target[[indices[sample] for sample in valid_ids]][
                        :, None
                    ].tolist(),
                    "target_names": ["y"],
                }
            ],
            "lineage": {
                "record_id": ":".join(
                    [
                        "lineage:python-methods",
                        task["run_id"],
                        node["node_id"],
                        task["variant_id"],
                        task["fold_id"],
                    ]
                ),
                "run_id": task["run_id"],
                "node_id": node["node_id"],
                "phase": task["phase"],
                "controller_id": node["controller_id"],
                "controller_version": node["controller_version"],
                "variant_id": task["variant_id"],
                "fold_id": task["fold_id"],
                "branch_path": task["branch_path"],
                "input_lineage": [],
                "artifact_refs": [],
                "params_fingerprint": node["params_fingerprint"],
                "data_model_shape_fingerprint": None,
                "aggregation_policy_fingerprint": None,
                "seed": task["seed"],
                "unsafe_flags": [],
                "metrics": {},
                "loss_attestations": [],
                "early_stopping_records": [],
            },
        }

    result = dag_ml.run_host_hpo_search_in_process(
        fixture["dsl"],
        fixture["envelope"],
        [fixture["manifest"]],
        fixture["request"],
        operator,
        optimizer,
    )
    assert result["selected_trial_index"] == fixture["resumed"]["selected_trial_index"]
    for python, wasm in zip(result["trials"], proposals, strict=True):
        assert python["params"] == wasm["params"]
        np.testing.assert_allclose(
            python["score"], wasm["score"], rtol=1e-10, atol=1e-12
        )

        def reports(trial):
            return {
                (report["producer_node"], report["fold_id"]): report
                for report in trial["scores"]["reports"]
                if report["fold_id"] is not None
            }

        actual, expected = reports(python), reports(wasm)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert actual[key]["row_count"] == expected[key]["row_count"]
            np.testing.assert_allclose(
                actual[key]["metrics"]["rmse"],
                expected[key]["metrics"]["rmse"],
                rtol=1e-10,
                atol=1e-12,
            )
    assert len(fits) == 117  # 3 trials x 3 outer folds x (4 sources x 3 fits + 1 meta).
    assert sum(fit["node"] == "model:meta" and fit["width"] == 4 for fit in fits) == 9
    refit = RolePipeline.from_steps(
        [
            {
                "class": "n4m:models.regularized.ridge",
                "params": {"alpha": fixture["nativeRefit"]["alpha"]},
            },
        ]
    )
    refit.fit(sources["nir"], target)
    np.testing.assert_allclose(
        np.asarray(
            refit.predict(np.asarray([fixture["nativeRefit"]["heldoutRow"]]))
        ).reshape(-1, 1),
        fixture["nativeRefit"]["prediction"],
        rtol=1e-10,
        atol=1e-12,
    )
    # Import the actual five-model WASM package through the same native replay
    # API exposed to Python. Heavy model state stays in Methods N4ME binaries.
    archive = fixture["completeArchive"]
    package = json.loads(archive["packageJson"])
    states = {}
    operations = []
    next_handle = 0

    def artifact_bridge(message):
        nonlocal next_handle
        operation = message["operation"]
        operations.append(operation)
        if operation == "release":
            del states[message["handle"]["handle"]]
            return None
        assert operation == "hydrate"
        request = message["request"]
        payload = bytes(message["payload"])
        assert (
            hashlib.sha256(payload).hexdigest()
            == request["artifact"]["content_fingerprint"]
        )
        saved = json.loads(payload)
        assert saved["schema"] == "dagml.methods.regression.v1"
        assert saved["node_id"] == request["node_id"]
        assert saved["params_fingerprint"] == request["params_fingerprint"]
        model = RolePipeline.from_states(
            saved["steps"], saved["states"], feature_names=saved["feature_names"]
        )
        next_handle += 1
        states[next_handle] = (model, saved, request["artifact"])
        return {
            "handle": next_handle,
            "kind": "model",
            "owner_controller": request["controller_id"],
        }

    def predict_operator(task):
        assert task["phase"] == "PREDICT"
        operations.append("PREDICT")
        node = task["node_plan"]
        [(key, artifact)] = task["artifact_inputs"].items()
        handle = task["input_handles"][key]
        model, saved, expected_artifact = states[handle["handle"]]
        assert saved["node_id"] == node["node_id"]
        assert saved["params_fingerprint"] == node["params_fingerprint"]
        assert artifact["artifact"]["id"] == expected_artifact["id"]
        if task["prediction_inputs"]:
            inputs = [
                value
                for key, value in sorted(task["prediction_inputs"].items())
                if key.endswith(":predict")
            ]
            replay_ids = inputs[0]["sample_ids"]
            assert all(value["sample_ids"] == replay_ids for value in inputs)
            x = np.column_stack([value["values"] for value in inputs])
        else:
            [view] = task["data_views"].values()
            replay_ids = view["sample_ids"]
            [source] = view["source_ids"]
            x = np.asarray([archive["heldoutRows"][source]], dtype=float)
        values = np.asarray(model.predict(x), dtype=float).reshape(-1, 1).tolist()
        return {
            "node_id": node["node_id"],
            "outputs": {},
            "artifacts": [],
            "artifact_handles": {},
            "predictions": [
                {
                    "producer_node": node["node_id"],
                    "partition": "final",
                    "fold_id": None,
                    "sample_ids": replay_ids,
                    "values": values,
                    "target_names": ["y"],
                }
            ],
            "lineage": {
                "record_id": "lineage:python-replay:" + node["node_id"],
                "run_id": task["run_id"],
                "node_id": node["node_id"],
                "phase": "PREDICT",
                "controller_id": node["controller_id"],
                "controller_version": node["controller_version"],
                "variant_id": task["variant_id"],
                "fold_id": None,
                "branch_path": task["branch_path"],
                "input_lineage": [],
                "artifact_refs": [],
                "params_fingerprint": node["params_fingerprint"],
                "seed": task["seed"],
                "unsafe_flags": [],
                "metrics": {},
                "loss_attestations": [],
                "early_stopping_records": [],
            },
        }

    original_fit = RolePipeline.fit
    RolePipeline.fit = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Detached replay cannot fit")
    )
    try:
        replay = json.loads(
            dag_ml.replay_loaded_predictor_package_json(
                archive["packageJson"],
                archive["replayRequestJson"],
                json.dumps(archive["predictEnvelopes"]),
                "{}",
                predict_operator,
                outcome_id="outcome:python.four-source.replay",
                run_id="run:python.four-source.replay",
                artifact_callback=artifact_bridge,
            )
        )
    finally:
        RolePipeline.fit = original_fit
    assert not states
    assert (
        operations.count("hydrate")
        == operations.count("release")
        == operations.count("PREDICT")
        == 5
    )
    assert len(package["execution_bundle"]["refit_artifacts"]) == 5
    actual = replay["outputs"][0]["predictions"][0]
    expected = archive["replay"]["outputs"][0]["predictions"][0]
    assert actual["sample_ids"] == expected["sample_ids"]
    np.testing.assert_allclose(
        actual["values"], expected["values"], rtol=1e-10, atol=1e-12
    )
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(
            json.dumps(
                {
                    "result": result,
                    "fits": fits,
                    "completeArchiveReplay": replay,
                    "replayOperations": operations,
                },
                indent=2,
            )
            + "\n"
        )
    print(
        "PYTHON_WASM_MULTIMODAL_PARITY_OK",
        len(fits),
        result["selected_trial_index"],
        "five-model replay without fit",
    )


if __name__ == "__main__":
    main()
