#!/usr/bin/env python3
"""Small FIT_CV process controller that emits native regression evidence."""

import json
import sys

from python_process_controller import build_result, emit_description


def main():
    if sys.argv[1:] == ["--describe"]:
        emit_description(adapter_id="dag-ml-hpo-process-example")
        return
    task = json.loads(sys.stdin.readline())
    result = build_result(task)
    if task["node_plan"]["kind"] == "model" and task["phase"] == "FIT_CV":
        fold = task["fold_id"]
        samples = ["sample:1", "sample:2"] if fold == "fold:0" else ["sample:3", "sample:4"]
        params = task["node_plan"].get("params", {})
        prediction = float(params.get("n_components", 1))
        result["predictions"][0]["sample_ids"] = samples
        result["predictions"][0]["values"] = [[prediction] for _ in samples]
        result["regression_targets"] = [{
            "level": "sample", "unit_ids": [{"level": "sample", "id": sample} for sample in samples],
            "values": [[0.0] for _ in samples], "target_names": ["y"],
        }]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
