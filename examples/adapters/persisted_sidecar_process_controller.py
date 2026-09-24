#!/usr/bin/env python3
"""Small host adapter proving sidecar-backed replay after a process restart.

The numerical value is intentionally trivial. DAG-ML owns artifact identity,
bundle validation, replay scheduling and predictions; this host owns only its
serialized model bytes and checks them against the signed artifact metadata.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import python_process_controller as reference


def sidecar_root() -> Path:
    configured = os.environ.get("DAG_ML_PROCESS_ARTIFACT_DIR")
    if not configured:
        raise ValueError("DAG_ML_PROCESS_ARTIFACT_DIR is required")
    root = Path(configured).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def sidecar_path(root: Path, uri: str) -> Path:
    relative = Path(uri)
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "artifacts":
        raise ValueError("artifact URI must be artifacts/<basename>")
    if relative.parts[1] in {".", ".."}:
        raise ValueError("artifact URI has invalid basename")
    return root / relative.parts[1]


def execute(task: dict) -> dict:
    reference.require_data_handles(task)
    reference.require_replay_artifact(task)
    reference.require_prediction_inputs(task)
    reference.require_variant_param_overrides(task)
    result = reference.build_result(task)
    root = sidecar_root()
    if task["phase"] == "REFIT":
        for artifact in result["artifacts"]:
            handle = result["artifact_handles"][artifact["id"]]
            value = result["predictions"][0]["values"][0][0]
            payload = {
                "artifact_id": artifact["id"],
                "controller_id": artifact["controller_id"],
                "handle": handle["handle"],
                "prediction_value": value,
            }
            data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            sidecar_path(root, artifact["uri"]).write_bytes(data)
            artifact["content_fingerprint"] = hashlib.sha256(data).hexdigest()
            artifact["size_bytes"] = len(data)
    elif task["phase"] == "PREDICT" and result["predictions"]:
        inputs = task.get("artifact_inputs", {})
        if len(inputs) != 1:
            raise ValueError("PREDICT requires exactly one host artifact input per model")
        key, spec = next(iter(inputs.items()))
        artifact = spec["artifact"]
        handle = task["input_handles"][key]
        data = sidecar_path(root, artifact["uri"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact["content_fingerprint"]:
            raise ValueError("host sidecar fingerprint mismatch")
        payload = json.loads(data)
        if (
            payload["artifact_id"] != artifact["id"]
            or payload["controller_id"] != spec["controller_id"]
            or payload["handle"] != handle["handle"]
        ):
            raise ValueError("host sidecar identity or handle mismatch")
        value = payload["prediction_value"]
        for block in result["predictions"]:
            block["values"] = [[value] for _ in block["sample_ids"]]
    return result


def main() -> None:
    if sys.argv[1:] == ["--describe"]:
        description = reference.adapter_description(
            adapter_id="dag-ml-persisted-sidecar-process-controller"
        )
        description["supported_modes"] = ["one_shot"]
        print(json.dumps(description), flush=True)
        return
    if len(sys.argv) != 1:
        raise SystemExit("one-shot NodeTask JSON expected on stdin")
    try:
        task = json.load(sys.stdin)
        print(json.dumps(execute(task)), flush=True)
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"host sidecar adapter failed: {error}") from error


if __name__ == "__main__":
    main()
