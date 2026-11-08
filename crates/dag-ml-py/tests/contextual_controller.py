"""Deterministic synthetic callback for contextual owner tests; no native/IO claim."""
mode = "normal"
calls = []
next_handle = 10


def artifact(task):
    global next_handle
    calls.append(task["operation"])
    if task["operation"] == "release":
        return None
    assert task["operation"] == "hydrate"
    assert task["payload"]
    next_handle += 1
    return {"handle": next_handle, "kind": "artifact",
            "owner_controller": task["request"]["controller_id"]}


def callback(task):
    global next_handle
    calls.append(task["phase"])
    next_handle += 1
    node = task["node_plan"]
    model = node["kind"] == "model"
    port = "oof" if model else "x_out"
    targets = ["protein" if node["node_id"] == "model:base" else "fat"]
    ids = [f"sample:contextual:{i}" for i in range(11, 15)]
    if mode == "alien":
        ids[0] = "sample:alien"
    explanation = {"producer_node": node["node_id"], "producer_port": port,
                   "method": "synthetic", "target_name": targets[0],
                   "payload": {"importance": 0.0}}
    if mode == "target":
        explanation["target_name"] = "protein:a"
    if mode == "global":
        explanation["target_name"] = None
    if mode == "raw" or (mode == "unselected_raw" and node["node_id"] == "model:second"):
        explanation["payload"] = {"nested": [{"FeAtUrE_MaTrIx": [[0.0]]}]}
    return {"node_id": node["node_id"],
            "outputs": {port: {"handle": next_handle, "kind": "prediction" if model else "data",
                                "owner_controller": node["controller_id"]}},
            "predictions": [{"producer_node": node["node_id"], "producer_port": port,
                             "partition": "final", "fold_id": None, "sample_ids": ids,
                             "values": [[0.0] for _ in ids], "target_names": targets}]
                           if model and task["phase"] == "PREDICT" else [],
            "explanations": [explanation] if model and task["phase"] == "EXPLAIN" else [],
            "lineage": {"record_id": f"lineage:contextual:{next_handle}",
                        "run_id": task["run_id"], "node_id": node["node_id"],
                        "phase": task["phase"], "controller_id": node["controller_id"],
                        "controller_version": node["controller_version"],
                        "variant_id": task["variant_id"], "fold_id": task["fold_id"],
                        "branch_path": task["branch_path"], "input_lineage": [], "artifact_refs": [],
                        "params_fingerprint": node["params_fingerprint"], "seed": task.get("seed"),
                        "unsafe_flags": [], "metrics": {}, "loss_attestations": [],
                        "early_stopping_records": []}}
