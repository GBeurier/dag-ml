#!/usr/bin/env python3
"""Process adapter that hangs once per run/controller/worker before delegating.

This adapter is intentionally test-only. It exercises dag-ml's persistent
process timeout, worker restart, and retry path without changing controller
semantics after the first recovered failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any

import python_process_controller as base


def safe_marker_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def marker_path(task: dict[str, Any], behavior: str) -> Path:
    run_id = safe_marker_part(str(task.get("run_id", "run")))
    controller_id = safe_marker_part(os.environ.get("DAG_ML_CONTROLLER_ID", "controller"))
    worker_index = safe_marker_part(os.environ.get("DAG_ML_PROCESS_WORKER_INDEX", "0"))
    marker_root = Path(os.environ.get("DAG_ML_FLAKY_MARKER_DIR", tempfile.gettempdir()))
    marker_root.mkdir(parents=True, exist_ok=True)
    return marker_root / (
        f"dag_ml_flaky_process_{behavior}_{run_id}_{controller_id}_{worker_index}.marker"
    )


def maybe_hang_once(task: dict[str, Any]) -> None:
    marker = marker_path(task, "hang")
    if marker.exists():
        return
    marker.write_text("hung once\n", encoding="utf-8")
    time.sleep(float(os.environ.get("DAG_ML_FLAKY_SLEEP_SECONDS", "10.0")))


def maybe_emit_retryable_error_once(task: dict[str, Any]) -> bool:
    if os.environ.get("DAG_ML_FLAKY_ERROR_ONCE") not in {"1", "true", "yes"}:
        return False
    marker = marker_path(task, "retryable_error")
    if marker.exists():
        return False
    marker.write_text("errored once\n", encoding="utf-8")
    base.emit_error(
        "retryable_test_error",
        "retryable process adapter test error",
        retryable=True,
    )
    return True


def emit_result(task: dict[str, Any]) -> None:
    maybe_hang_once(task)
    base.emit_result(task)


def emit_result_frame(task: dict[str, Any]) -> None:
    if maybe_emit_retryable_error_once(task):
        return
    maybe_hang_once(task)
    base.emit_result_frame(task)


def run_jsonl() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            base.fail(f"invalid NodeTask JSON line: {exc}")
        if base.is_control_frame(payload):
            if not handle_control_frame(payload):
                break
            continue
        emit_result(payload)


def handle_control_frame(frame: dict[str, Any]) -> bool:
    if not base.validate_frame_schema(frame):
        return True
    frame_type = frame["type"]
    if frame_type == "task":
        task = frame.get("task")
        if not isinstance(task, dict):
            base.emit_error("invalid_task_frame", "task frame is missing object field `task`")
            return True
        emit_result_frame(task)
        return True
    return base.handle_control_frame(frame)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--describe":
        base.emit_description(
            adapter_id="dag-ml-flaky-process-controller",
            extra_capabilities=["test_flaky_timeout_once"],
        )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--jsonl":
        run_jsonl()
        return
    try:
        task = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        base.fail(f"invalid NodeTask JSON: {exc}")
    emit_result(task)


if __name__ == "__main__":
    main()
