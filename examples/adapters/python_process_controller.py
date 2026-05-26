#!/usr/bin/env python3
"""Minimal external controller adapter for dag-ml process replay smoke tests."""

from __future__ import annotations

import json
import sys
from typing import Any


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def stable_handle(value: str) -> int:
    acc = 17
    for byte in value.encode("utf-8"):
        acc = ((acc * 31) + byte) & ((1 << 64) - 1)
    return acc


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
        if task.get("phase") == "PREDICT" and task.get("fold_id") is None:
            if view.get("partition") != "predict":
                fail(f"node `{node_plan['node_id']}` received non-predict replay view `{key}`")
        if task.get("phase") == "FIT_CV" and task.get("fold_id") is not None:
            if view.get("partition") != "fold_train":
                fail(f"node `{node_plan['node_id']}` received non-train fold view `{key}`")
            if not view.get("sample_ids"):
                fail(f"node `{node_plan['node_id']}` received fold view without samples `{key}`")
            validation_key = f"{key}:validation"
            validation_view = data_views.get(validation_key)
            if validation_view is None:
                fail(
                    f"node `{node_plan['node_id']}` did not receive validation data view "
                    f"`{validation_key}`"
                )
            if validation_view.get("partition") != "fold_validation":
                fail(
                    f"node `{node_plan['node_id']}` received non-validation fold view "
                    f"`{validation_key}`"
                )
            if not validation_view.get("sample_ids"):
                fail(
                    f"node `{node_plan['node_id']}` received validation view without samples "
                    f"`{validation_key}`"
                )


def require_replay_artifact(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    if task.get("phase") != "PREDICT" or node_plan.get("kind") != "model":
        return
    artifact_handles = {
        key: handle
        for key, handle in task.get("input_handles", {}).items()
        if key.startswith("artifact:")
    }
    if not artifact_handles:
        fail(f"node `{node_plan['node_id']}` did not receive replay artifact handle")
    for key, handle in artifact_handles.items():
        if node_plan["node_id"] not in key:
            fail(f"node `{node_plan['node_id']}` received artifact handle for another node `{key}`")
        if handle.get("kind") not in {"model", "artifact"}:
            fail(f"node `{node_plan['node_id']}` received invalid artifact handle `{key}`")


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

    predictions = []
    if node_plan.get("kind") == "model":
        prediction_sample_ids = ["sample:process"]
        data_bindings = node_plan.get("data_bindings", [])
        if phase == "FIT_CV" and data_bindings:
            input_name = data_bindings[0]["input_name"]
            validation_view = task.get("data_views", {}).get(f"data:{input_name}:validation")
            if validation_view is not None:
                prediction_sample_ids = validation_view.get("sample_ids") or prediction_sample_ids
        predictions.append(
            {
                "prediction_id": f"pred:{node_id}:{phase}",
                "producer_node": node_id,
                "partition": prediction_partition(phase),
                "fold_id": fold_id,
                "sample_ids": prediction_sample_ids,
                "values": [[float(handle_value % 1_000_000)] for _ in prediction_sample_ids],
                "target_names": ["y"],
            }
        )

    artifacts = []
    artifact_handles = {}
    if phase == "REFIT" and node_plan.get("kind") == "model":
        artifact_id = f"artifact:{node_id}:refit"
        artifact = {
            "id": artifact_id,
            "kind": "mock_model",
            "controller_id": controller_id,
            "size_bytes": 128,
        }
        artifacts.append(artifact)
        artifact_handles[artifact_id] = {
            "handle": stable_handle(artifact_id),
            "kind": "model",
            "owner_controller": controller_id,
        }

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
            "metrics": {"adapter_smoke": 1.0},
        },
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--jsonl":
        run_jsonl()
        return
    try:
        task = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        fail(f"invalid NodeTask JSON: {exc}")
    emit_result(task)


def run_jsonl() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid NodeTask JSON line: {exc}")
        emit_result(task)


def emit_result(task: dict[str, Any]) -> None:
    require_data_handles(task)
    require_replay_artifact(task)
    require_prediction_inputs(task)
    json.dump(build_result(task), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
