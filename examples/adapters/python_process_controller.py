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
    for binding in node_plan.get("data_bindings", []):
        key = f"data:{binding['input_name']}"
        handle = input_handles.get(key)
        if handle is None:
            fail(f"node `{node_plan['node_id']}` did not receive data handle `{key}`")
        if handle.get("kind") != "data":
            fail(f"node `{node_plan['node_id']}` received non-data handle `{key}`")


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
        if handle.get("kind") not in {"model", "artifact"}:
            fail(f"node `{node_plan['node_id']}` received invalid artifact handle `{key}`")


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
        predictions.append(
            {
                "prediction_id": f"pred:{node_id}:{phase}",
                "producer_node": node_id,
                "partition": prediction_partition(phase),
                "fold_id": fold_id,
                "sample_ids": ["sample:process"],
                "values": [[float(handle_value % 1_000_000)]],
                "target_names": ["y"],
            }
        )

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
        "artifacts": [],
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
            "artifact_refs": [],
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
    json.dump(build_result(task), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
