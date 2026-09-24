#!/usr/bin/env python3
"""Persistent DAG-ML process adapter for a concrete portable Methods PLS model.

Set ``DAGML_METHODS_DATA_JSON`` to a mapping of sample IDs to ``{"x": [...],
"y": number}`` (``y`` is optional for prediction) and install ``pls4all``.
The only fitted state crossing the process boundary is Methods' N4MM bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from pls4all._context import Context
from pls4all._model import Model
from pls4all.sklearn import PLSRegression
from python_process_controller import (
    adapter_description,
    build_result,
    emit_ack,
    emit_error,
    emit_json,
    first_view_sample_ids,
    stable_handle,
)


def _mark(operation: str) -> None:
    path = os.environ.get("DAGML_METHODS_ARTIFACT_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(operation + "\n")


def _rows(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with open(os.environ["DAGML_METHODS_DATA_JSON"], encoding="utf-8") as stream:
        rows = json.load(stream)
    x = np.asarray([rows[sample_id]["x"] for sample_id in ids], dtype=np.float64)
    y = np.asarray([rows[sample_id].get("y", float("nan")) for sample_id in ids], dtype=np.float64)
    return x, y


def _model_result(task: dict, models: dict[int, tuple[Context, Model]], payloads: dict[str, bytes]) -> dict:
    node = task["node_plan"]
    node_id = node["node_id"]
    phase = task["phase"]
    ids = first_view_sample_ids(task, "full_train" if phase == "REFIT" else "predict")
    if not ids:
        raise ValueError(f"Methods PLS {phase} has no scheduler-selected sample IDs")
    x, y = _rows(ids)
    result = build_result(task)
    if phase == "REFIT":
        if not np.isfinite(y).all():
            raise ValueError("Methods PLS REFIT requires numeric targets")
        model = PLSRegression(n_components=int(node.get("params", {}).get("n_components", 2))).fit(x, y)
        payload = model._bundle_
        assert payload is not None
        artifact_id = f"artifact:{node_id}:refit"
        payloads[artifact_id] = payload
        digest = hashlib.sha256(payload).hexdigest()
        artifact = {
            "id": artifact_id, "kind": "n4m_model", "controller_id": node["controller_id"],
            "backend": "raw", "uri": f"artifacts/{digest}.n4mm",
            "content_fingerprint": digest, "size_bytes": len(payload),
            "plugin": "dagml.methods_pls_process", "plugin_version": "1.0.0",
        }
        result["artifacts"] = [artifact]
        result["artifact_handles"] = {artifact_id: {
            "handle": stable_handle(artifact_id), "kind": "model",
            "owner_controller": node["controller_id"],
        }}
        result["lineage"]["artifact_refs"] = [artifact]
        predictions = model.predict(x)
    elif phase == "PREDICT":
        artifact_handles = [handle for key, handle in task.get("input_handles", {}).items()
                            if key.startswith("artifact:")]
        if len(artifact_handles) != 1 or artifact_handles[0]["handle"] not in models:
            raise ValueError("Methods PLS PREDICT requires one hydrated N4MM handle")
        ctx, model = models[artifact_handles[0]["handle"]]
        predictions = model.predict(ctx, x)
    else:
        raise ValueError(f"unsupported Methods PLS phase {phase}")
    result["predictions"][0]["sample_ids"] = ids
    result["predictions"][0]["values"] = np.asarray(predictions).reshape(len(ids), -1).tolist()
    return result


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--describe":
        description = adapter_description("dag-ml-methods-pls-process", ["portable_artifact_bridge_v1"])
        description["supported_modes"] = ["jsonl"]
        emit_json(description)
        return
    if len(sys.argv) != 2 or sys.argv[1] != "--jsonl":
        raise SystemExit("Methods PLS adapter requires --jsonl")
    models: dict[int, tuple[Context, Model]] = {}
    payloads: dict[str, bytes] = {}
    next_handle = 1
    for line in sys.stdin:
        if not line.strip():
            continue
        frame = json.loads(line)
        kind = frame.get("type")
        if kind == "init":
            emit_ack("initialized")
        elif kind == "close":
            for ctx, model in models.values():
                model.close()
                ctx.close()
            emit_ack("closed")
            break
        elif kind == "task":
            try:
                result = _model_result(frame["task"], models, payloads)
                emit_json({"type": "result", "schema_version": 1, "result": result})
            except Exception as error:  # noqa: BLE001 - report native adapter errors as protocol frames
                emit_error("methods_pls_task_failed", str(error))
        elif kind == "portable_artifact":
            try:
                task = frame["task"]
                operation = task["operation"]
                if operation == "export_artifact_payload":
                    _mark("export")
                    result = {"operation": "exported_artifact_payload", "schema_version": 1,
                              "payload": list(payloads[task["artifact_id"]])}
                elif operation == "hydrate_artifact_payload":
                    _mark("hydrate")
                    ctx = Context()
                    model = Model.from_bytes(ctx, bytes(task["payload"]))
                    handle = next_handle
                    next_handle += 1
                    models[handle] = (ctx, model)
                    result = {"operation": "hydrated_artifact_payload", "schema_version": 1,
                              "handle": {"handle": handle, "kind": "model",
                                         "owner_controller": task["request"]["controller_id"]}}
                elif operation == "release_hydrated_artifact_payload":
                    _mark("release")
                    ctx, model = models.pop(task["handle"]["handle"])
                    model.close()
                    ctx.close()
                    result = {"operation": "released_hydrated_artifact_payload", "schema_version": 1}
                else:
                    raise ValueError(f"unsupported artifact operation {operation}")
                emit_json({"type": "portable_artifact", "schema_version": 1, "result": result})
            except Exception as error:  # noqa: BLE001 - report native adapter errors as protocol frames
                emit_error("methods_pls_artifact_failed", str(error))
        else:
            emit_error("unsupported_frame", f"unsupported frame {kind}")


if __name__ == "__main__":
    main()
