#!/usr/bin/env python3
"""Stateful sklearn process adapter for dag-ml coordinator smoke tests."""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
from typing import Any

PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION = 1
PROCESS_ADAPTER_PROTOCOL = "dag-ml-process-adapter"
PROCESS_ADAPTER_MODES = ["one_shot", "jsonl"]
PROCESS_ADAPTER_CAPABILITIES = [
    "control_frames_v1",
    "node_task_json_v1",
    "node_result_json_v1",
    "parallel_invocation_v1",
    "persistent_workers",
    "worker_env",
    "stateful_refit_artifacts",
    "sklearn_smoke",
]
PROCESS_ADAPTER_FRAME_SCHEMA_VERSION = 1


def emit_description() -> None:
    json.dump(
        {
            "schema_version": PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION,
            "protocol": PROCESS_ADAPTER_PROTOCOL,
            "adapter_id": "dag-ml-sklearn-process-controller",
            "supported_modes": PROCESS_ADAPTER_MODES,
            "capabilities": sorted(set(PROCESS_ADAPTER_CAPABILITIES)),
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_json(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_ack(status: str) -> None:
    emit_json(
        {
            "type": "ack",
            "schema_version": PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
            "status": status,
        }
    )


def emit_error(code: str, message: str, retryable: bool = False) -> None:
    emit_json(
        {
            "type": "error",
            "schema_version": PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        }
    )


# Keep the coordinator handshake cheap; runtime modes import sklearn below.
if len(sys.argv) > 1 and sys.argv[1] == "--describe":
    emit_description()
    raise SystemExit(0)


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODELS: dict[int, Pipeline] = {}


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def stable_handle(value: str) -> int:
    acc = 17
    for byte in value.encode("utf-8"):
        acc = ((acc * 31) + byte) & ((1 << 64) - 1)
    return acc


def content_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_scalar(sample_id: str) -> float:
    return (stable_handle(sample_id) % 10_000) / 10_000.0


def features(sample_ids: list[str]) -> np.ndarray:
    rows = []
    for sample_id in sample_ids:
        x = sample_scalar(sample_id)
        rows.append([x, x * x, math.sin(x * math.pi), math.cos(x * math.pi)])
    return np.asarray(rows, dtype=float)


def targets(sample_ids: list[str]) -> np.ndarray:
    x = np.asarray([sample_scalar(sample_id) for sample_id in sample_ids], dtype=float)
    return 1.7 * x - 0.3 * x * x + np.sin(x * math.pi) * 0.2


def prediction_partition(phase: str) -> str:
    if phase == "FIT_CV":
        return "validation"
    if phase in {"REFIT", "PREDICT", "EXPLAIN"}:
        return "final"
    return "test"


def require_data_handles(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    input_handles = task.get("input_handles", {})
    data_views = task.get("data_views", {})
    for binding in node_plan.get("data_bindings", []):
        key = f"data:{binding['input_name']}"
        handle = input_handles.get(key)
        if handle is None:
            fail(f"node `{node_plan['node_id']}` did not receive data handle `{key}`")
        if handle.get("kind") not in {"data", "data_view"}:
            fail(f"node `{node_plan['node_id']}` received non-data/data-view handle `{key}`")
        view = data_views.get(key)
        if view is None:
            fail(f"node `{node_plan['node_id']}` did not receive data view spec `{key}`")
        if task.get("phase") == "FIT_CV" and task.get("fold_id") is not None:
            if view.get("partition") != "fold_train":
                fail(f"node `{node_plan['node_id']}` received non-train fold view `{key}`")
            validation_key = f"{key}:validation"
            validation_view = data_views.get(validation_key)
            if validation_view is None or validation_view.get("partition") != "fold_validation":
                fail(f"node `{node_plan['node_id']}` did not receive validation view `{validation_key}`")
        if task.get("phase") == "REFIT" and view.get("partition") != "full_train":
            fail(f"node `{node_plan['node_id']}` received non-full-train refit view `{key}`")
        if task.get("phase") == "PREDICT" and view.get("partition") != "predict":
            fail(f"node `{node_plan['node_id']}` received non-predict replay view `{key}`")


def data_view(task: dict[str, Any], suffix: str = "") -> dict[str, Any] | None:
    bindings = task["node_plan"].get("data_bindings", [])
    if not bindings:
        return None
    input_name = bindings[0]["input_name"]
    return task.get("data_views", {}).get(f"data:{input_name}{suffix}")


def train_sample_ids(task: dict[str, Any]) -> list[str]:
    view = data_view(task)
    if view is None:
        return ["sample:train:0", "sample:train:1", "sample:train:2", "sample:train:3"]
    sample_ids = view.get("sample_ids")
    if not sample_ids:
        fail(f"node `{task['node_plan']['node_id']}` train view has no sample ids")
    return list(sample_ids)


def prediction_sample_ids(task: dict[str, Any]) -> list[str]:
    phase = task["phase"]
    if phase == "FIT_CV":
        validation = data_view(task, ":validation")
        sample_ids = None if validation is None else validation.get("sample_ids")
        if not sample_ids:
            fail(f"node `{task['node_plan']['node_id']}` validation view has no sample ids")
        return list(sample_ids)
    if phase == "REFIT":
        return train_sample_ids(task)
    view = data_view(task)
    sample_ids = None if view is None else view.get("sample_ids")
    return list(sample_ids) if sample_ids else ["sample:predict:0", "sample:predict:1"]


def make_estimator(seed: int | None) -> Pipeline:
    alpha = 1.0 if seed is None else 1.0 + ((seed % 17) / 100.0)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def replay_model(task: dict[str, Any]) -> Pipeline:
    artifact_handles = {
        key: handle
        for key, handle in task.get("input_handles", {}).items()
        if key.startswith("artifact:")
    }
    if not artifact_handles:
        fail(f"node `{task['node_plan']['node_id']}` did not receive replay artifact handle")
    key, handle = next(iter(artifact_handles.items()))
    if task["node_plan"]["node_id"] not in key:
        fail(f"node `{task['node_plan']['node_id']}` received artifact handle for another node `{key}`")
    if handle.get("kind") not in {"model", "artifact"}:
        fail(f"node `{task['node_plan']['node_id']}` received invalid artifact handle `{key}`")
    model = MODELS.get(int(handle["handle"]))
    if model is None:
        fail(f"node `{task['node_plan']['node_id']}` has no sklearn model for handle `{key}`")
    return model


def require_prediction_inputs(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    input_handles = task.get("input_handles", {})
    prediction_inputs = task.get("prediction_inputs", {})
    for key, spec in prediction_inputs.items():
        handle = input_handles.get(key)
        if handle is None:
            fail(f"node `{node_plan['node_id']}` did not receive prediction handle `{key}`")
        if handle.get("kind") != "prediction":
            fail(f"node `{node_plan['node_id']}` received non-prediction handle `{key}`")
        if spec.get("producer_node") not in key:
            fail(f"node `{node_plan['node_id']}` received mismatched prediction spec `{key}`")
        if spec.get("partition") != "validation":
            fail(f"node `{node_plan['node_id']}` received non-validation prediction spec `{key}`")
        if spec.get("prediction_level", "sample") != "sample":
            fail(f"node `{node_plan['node_id']}` received non-sample prediction spec `{key}`")
        if not spec.get("sample_ids"):
            fail(f"node `{node_plan['node_id']}` received prediction spec without samples `{key}`")
        if int(spec.get("prediction_width", 0)) <= 0:
            fail(f"node `{node_plan['node_id']}` received prediction spec without width `{key}`")
        if task.get("phase") == "FIT_CV":
            if spec.get("fold_id") != task.get("fold_id"):
                fail(f"node `{node_plan['node_id']}` received prediction spec for wrong fold `{key}`")
            validation_samples: set[str] = set()
            for view in task.get("data_views", {}).values():
                if view.get("partition") == "fold_validation":
                    validation_samples.update(view.get("sample_ids") or [])
            if validation_samples and set(spec.get("sample_ids") or []) != validation_samples:
                fail(f"node `{node_plan['node_id']}` received prediction spec for wrong samples `{key}`")
        if task.get("phase") == "REFIT" and spec.get("fold_id") is not None:
            fail(f"node `{node_plan['node_id']}` received fold-scoped prediction spec during REFIT `{key}`")


def require_variant_param_overrides(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    params = node_plan.get("params", {})
    variant = task.get("variant")
    if variant is None:
        return
    for dimension_name, choice in variant.get("choices", {}).items():
        choice_label = choice.get("label", "<unknown>")
        for override in choice.get("param_overrides", []):
            if override.get("node_id") != node_id:
                continue
            for key, value in override.get("params", {}).items():
                if params.get(key) != value:
                    fail(
                        f"node `{node_id}` missing generated param override "
                        f"`{dimension_name}.{choice_label}.{key}`"
                    )


def model_result(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase = task["phase"]
    node_id = task["node_plan"]["node_id"]
    controller_id = task["node_plan"]["controller_id"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"

    if phase == "PREDICT":
        estimator = replay_model(task)
    else:
        estimator = make_estimator(task.get("seed"))
        ids = train_sample_ids(task)
        estimator.fit(features(ids), targets(ids))

    pred_ids = prediction_sample_ids(task)
    values = [[float(value)] for value in estimator.predict(features(pred_ids))]
    predictions = [
        {
            "prediction_id": f"pred:{node_id}:{phase}:{variant_label}:{fold_label}",
            "producer_node": node_id,
            "partition": prediction_partition(phase),
            "fold_id": task.get("fold_id") if phase == "FIT_CV" else None,
            "sample_ids": pred_ids,
            "values": values,
            "target_names": ["y"],
        }
    ]

    artifacts = []
    artifact_handles = {}
    if phase == "REFIT":
        artifact_id = f"artifact:{node_id}:sklearn:refit"
        handle_value = stable_handle(f"{artifact_id}:{variant_label}")
        MODELS[handle_value] = estimator
        artifact = {
            "id": artifact_id,
            "kind": "sklearn_pipeline",
            "controller_id": controller_id,
            "backend": "joblib",
            "uri": f"artifacts/{content_fingerprint(artifact_id)}.joblib",
            "content_fingerprint": content_fingerprint(f"{artifact_id}:{variant_label}"),
            "size_bytes": 256,
            "plugin": "dagml.sklearn_process",
            "plugin_version": "1.0.0",
        }
        artifacts.append(artifact)
        artifact_handles[artifact_id] = {
            "handle": handle_value,
            "kind": "model",
            "owner_controller": controller_id,
        }

    return predictions, artifacts, artifact_handles


def build_result(task: dict[str, Any]) -> dict[str, Any]:
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    phase = task["phase"]
    controller_id = node_plan["controller_id"]
    variant_id = task.get("variant_id")
    fold_id = task.get("fold_id")
    variant_label = variant_id or "base"
    fold_label = fold_id or "nofold"
    handle_value = stable_handle(f"{node_id}:{phase}:{variant_label}:{fold_label}")

    predictions: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if node_plan.get("kind") == "model":
        predictions, artifacts, artifact_handles = model_result(task)

    metrics = {"sklearn_adapter": 1.0}
    if predictions:
        flat = [row[0] for row in predictions[0]["values"]]
        metrics["prediction_mean"] = float(sum(flat) / len(flat))
    worker_index = os.environ.get("DAG_ML_PROCESS_WORKER_INDEX")
    worker_count = os.environ.get("DAG_ML_PROCESS_WORKER_COUNT")
    if worker_index is not None:
        metrics["process_worker_index"] = float(worker_index)
    if worker_count is not None:
        metrics["process_worker_count"] = float(worker_count)

    return {
        "node_id": node_id,
        "outputs": {
            "out": {
                "handle": handle_value,
                "kind": "data",
                "owner_controller": controller_id,
            }
        },
        "predictions": predictions,
        "shape_deltas": [],
        "artifacts": artifacts,
        "artifact_handles": artifact_handles,
        "lineage": {
            "record_id": f"lineage:{node_id}:{phase}:{variant_label}:{fold_label}",
            "run_id": task["run_id"],
            "node_id": node_id,
            "phase": phase,
            "controller_id": controller_id,
            "controller_version": node_plan["controller_version"],
            "variant_id": variant_id,
            "fold_id": fold_id,
            "branch_path": task.get("branch_path", []),
            "input_lineage": [],
            "artifact_refs": artifacts,
            "params_fingerprint": node_plan["params_fingerprint"],
            "data_model_shape_fingerprint": None,
            "aggregation_policy_fingerprint": None,
            "seed": task.get("seed"),
            "unsafe_flags": [],
            "metrics": metrics,
        },
    }


def emit_result(task: dict[str, Any]) -> None:
    require_data_handles(task)
    require_prediction_inputs(task)
    require_variant_param_overrides(task)
    emit_json(build_result(task))


def emit_result_frame(task: dict[str, Any]) -> None:
    require_data_handles(task)
    require_prediction_inputs(task)
    require_variant_param_overrides(task)
    emit_json(
        {
            "type": "result",
            "schema_version": PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
            "result": build_result(task),
        }
    )


def run_jsonl() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid NodeTask JSON line: {exc}")
        if is_control_frame(payload):
            if not handle_control_frame(payload):
                break
            continue
        emit_result(payload)


def is_control_frame(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("type"), str)


def validate_frame_schema(frame: dict[str, Any]) -> bool:
    if frame.get("schema_version") != PROCESS_ADAPTER_FRAME_SCHEMA_VERSION:
        emit_error(
            "unsupported_frame_schema",
            f"unsupported frame schema version `{frame.get('schema_version')}`",
            retryable=False,
        )
        return False
    return True


def handle_control_frame(frame: dict[str, Any]) -> bool:
    if not validate_frame_schema(frame):
        return True
    frame_type = frame["type"]
    if frame_type == "init":
        write_lifecycle_marker("init", frame)
        emit_ack("initialized")
        return True
    if frame_type == "task":
        task = frame.get("task")
        if not isinstance(task, dict):
            emit_error("invalid_task_frame", "task frame is missing object field `task`")
            return True
        emit_result_frame(task)
        return True
    if frame_type == "close":
        write_lifecycle_marker("close", frame)
        emit_ack("closed")
        return False
    emit_error("unsupported_frame", f"unsupported frame type `{frame_type}`")
    return True


def write_lifecycle_marker(event: str, frame: dict[str, Any]) -> None:
    marker_dir = os.environ.get("DAG_ML_PROCESS_LIFECYCLE_MARKER_DIR")
    if not marker_dir:
        return
    os.makedirs(marker_dir, exist_ok=True)
    controller_id = frame.get("controller_id") or os.environ.get("DAG_ML_CONTROLLER_ID", "controller")
    worker_index = (
        frame.get("worker_index")
        if frame.get("worker_index") is not None
        else os.environ.get("DAG_ML_PROCESS_WORKER_INDEX", "0")
    )
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in f"{event}_{controller_id}_{worker_index}"
    )
    with open(os.path.join(marker_dir, f"{safe_name}.marker"), "a", encoding="utf-8") as marker:
        marker.write(event)
        marker.write("\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--describe":
        emit_description()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--jsonl":
        run_jsonl()
        return
    try:
        task = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        fail(f"invalid NodeTask JSON: {exc}")
    emit_result(task)


if __name__ == "__main__":
    main()
