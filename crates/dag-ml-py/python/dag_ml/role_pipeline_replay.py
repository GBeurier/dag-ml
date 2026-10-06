"""Frozen CPU consumer of Methods N4ME packages produced by the WASM controller.

DAG owns the replay, identities, artifact lifecycle and lineage. Methods owns
state import and predictions. This consumer intentionally has no FIT path.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def replay_browser_role_pipeline(package_json: str, envelope: Any, *, x: Any,
                                 sample_ids: list[str], source_id: str,
                                 run_id: str = "run:browser-package:cpu") -> dict[str, Any]:
    """Replay a validated initial-full-refit WASM PLS package on CPU, fit-free."""
    from n4m.roles import RolePipeline

    from dag_ml import InitialFullRefitPackage, replay_initial_full_refit_in_process

    package = InitialFullRefitPackage(package_json).to_dict()
    nodes = package["effective_plan"]["node_plans"]
    if len(nodes) != 1 or len(package["artifacts"]) != 1:
        raise ValueError("Browser replay requires one native Methods role pipeline")
    node = next(iter(nodes.values()))
    record = package["artifacts"][0]["record"]
    artifact = record["artifact"]
    owner = "controller:methods.wasm.regression"
    if (node["controller_id"] != owner or artifact["controller_id"] != owner
            or artifact["plugin"] != "dagml.methods.wasm.regression"
            or artifact["plugin_version"] != "1.0.0" or artifact["kind"] != "methods_role_pipeline"
            or artifact["backend"] != "raw" or package["artifacts"][0]["load_mode"] != "native_portable"):
        raise ValueError("Unsupported browser Methods controller or artifact owner")
    matrix = np.asarray(x, dtype=float)
    if (matrix.ndim != 2 or len(sample_ids) != matrix.shape[0] or not np.isfinite(matrix).all()
            or len(set(sample_ids)) != len(sample_ids)):
        raise ValueError("Finite identity-keyed prediction matrix required")
    positions = {sample: index for index, sample in enumerate(sample_ids)}
    models: dict[int, tuple[Any, dict[str, Any]]] = {}

    def artifact_callback(message: dict[str, Any]) -> Any:
        if message["operation"] == "release":
            handle = message["handle"]
            if handle["owner_controller"] != owner or handle["handle"] not in models:
                raise ValueError("Unknown browser model handle")
            del models[handle["handle"]]
            return None
        if message["operation"] != "hydrate":
            raise ValueError("Frozen browser consumer cannot export or fit artifacts")
        request, payload = message["request"], bytes(message["payload"])
        if (request["controller_id"] != owner or request["artifact"] != artifact
                or request["node_id"] != record["node_id"]
                or request["params_fingerprint"] != record["params_fingerprint"]
                or hashlib.sha256(payload).hexdigest() != artifact["content_fingerprint"]):
            raise ValueError("Browser model artifact binding mismatch")
        saved = json.loads(payload)
        if (saved["schema"] != "dagml.methods.regression.v1" or saved["node_id"] != node["node_id"]
                or saved["params_fingerprint"] != node["params_fingerprint"]
                or len(saved["steps"]) != len(saved["states"])):
            raise ValueError("Browser state differs from signed model parameters")
        expected = [
            {"methodId": "preprocessing.scatter.snv", "params": {"with_mean": True, "with_std": True, "ddof": 0}},
            {"methodId": "preprocessing.derivatives.savitzky_golay", "params": {"window_length": 5, "polyorder": 2, "deriv": 0, "delta": 1, "mode": "interp", "cval": 0}},
            {"methodId": "models.pls.pls_regression", "params": {"n_components": node["params"]["n_components"],
             "solver": "nipals", "center_x": True, "center_y": True, "scale_x": node["params"]["scale"], "scale_y": node["params"]["scale"]}},
        ]
        if saved["steps"] != expected:
            raise ValueError("Browser state recipe differs from native selected PLS controls")
        steps = [(step["methodId"], step["params"]) for step in saved["steps"]]
        model = RolePipeline.from_states(steps, [bytes(state) for state in saved["states"]], feature_names=saved["feature_names"])
        handle = len(models) + 1
        models[handle] = (model, saved)
        return {"handle": handle, "kind": "model", "owner_controller": owner}

    def operator(task: dict[str, Any]) -> dict[str, Any]:
        current = task["node_plan"]
        if task["phase"] != "PREDICT" or current != node:
            raise ValueError("Frozen browser consumer only executes the signed PREDICT node")
        entries = list(task["artifact_inputs"].items())
        if len(entries) != 1:
            raise ValueError("Exactly one frozen browser artifact required")
        key, binding = entries[0]
        handle = task["input_handles"][key]
        if handle["owner_controller"] != owner or binding["artifact"] != artifact:
            raise ValueError("Foreign browser predictor handle")
        model, saved = models[handle["handle"]]
        views = [view for key, view in task["data_views"].items() if not key.endswith(":validation")]
        if len(views) != 1 or views[0]["source_ids"] != [source_id]:
            raise ValueError("Browser predictor source binding differs")
        ids = views[0]["sample_ids"]
        values = matrix[[positions[sample] for sample in ids]]
        class NamedMatrix(np.ndarray):
            pass
        named = values.view(NamedMatrix)
        named.columns = saved["feature_names"]
        predicted = np.asarray(model.predict(named)).reshape(len(ids), -1)
        return {"node_id": current["node_id"], "outputs": {}, "predictions": [{
            "producer_node": current["node_id"], "partition": "final", "fold_id": None,
            "sample_ids": ids, "values": predicted.tolist(), "target_names": saved["target_names"]}],
            "regression_targets": [], "artifacts": [], "artifact_handles": {}, "lineage": {
                "record_id": f"lineage:browser-cpu:{task['run_id']}:{current['node_id']}",
                "run_id": task["run_id"], "node_id": current["node_id"], "phase": "PREDICT",
                "controller_id": owner, "controller_version": current["controller_version"],
                "variant_id": task["variant_id"], "fold_id": None, "branch_path": task["branch_path"],
                "input_lineage": [], "artifact_refs": [], "params_fingerprint": current["params_fingerprint"],
                "data_model_shape_fingerprint": None, "aggregation_policy_fingerprint": None,
                "seed": task["seed"], "unsafe_flags": [], "metrics": {}, "loss_attestations": [], "early_stopping_records": []}}

    try:
        return replay_initial_full_refit_in_process(package_json, envelope, operator, {},
                                                   [output["output_id"] for output in package["outputs"]], run_id,
                                                   artifact_callback=artifact_callback)
    finally:
        models.clear()
