#!/usr/bin/env python3
"""Run canonical SDK U07 through actual Methods hosts and native DAG phases.

The SDK capture owns fixture creation. This script only marshals raw buffers,
replays recorded proposals in the native HPO protocol and compares evidence.
It never schedules folds, learns features or manufactures scores.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import queue
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def compare_numbers(actual: Any, expected: Any, tolerance: float, label: str) -> None:
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " shape")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare_numbers(left, right, tolerance, f"{label}[{index}]")
    else:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), f"{label}: {actual!r} != {expected!r}")


def replay_prediction_values(outcome: dict[str, Any], sample_ids: list[str], target_names: list[str]) -> list[float]:
    """Validate one native prediction block and align its rows by exact identity."""
    import dag_ml

    require(isinstance(outcome, dict), "Replay output must be an object")
    outputs = outcome.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], dict), "Exactly one replay output required")
    output = outputs[0]
    binding = output.get("binding")
    require(isinstance(binding, dict) and isinstance(binding.get("node_id"), str) and bool(binding["node_id"].strip()), "Replay output binding requires a producer node")
    predictions = output.get("predictions")
    require(isinstance(predictions, list) and len(predictions) == 1 and isinstance(predictions[0], dict), "Exactly one replay prediction block required")
    require(output.get("observation_predictions", []) == [] and output.get("aggregated_predictions", []) == [], "Unexpected additional replay prediction blocks")
    prediction = predictions[0]
    require(isinstance(target_names, list) and len(target_names) == 1 and isinstance(target_names[0], str) and bool(target_names[0]), "Exactly one replay target required")
    require(prediction.get("target_names") == target_names, "Replay prediction target names changed")
    actual_ids, values = prediction.get("sample_ids"), prediction.get("values")
    require(isinstance(sample_ids, list) and isinstance(actual_ids, list) and isinstance(values, list) and len(values) == len(actual_ids), "Replay sample IDs and prediction rows must match")
    for row in values:
        require(isinstance(row, list) and len(row) == 1, "Replay prediction row must have width one")
        require(type(row[0]) in (int, float) and math.isfinite(row[0]), "Replay prediction values must be finite numbers")
    alignment = dag_ml.align_named_source_rows({
        "sample_ids": sample_ids,
        "required_source_ids": [binding["node_id"]],
        "sources": [{"source_id": binding["node_id"], "sample_ids": actual_ids}],
    })
    return [values[index][0] for index in alignment["sources"][0]["row_indices"]]


def compare_scores(actual: dict[str, Any], expected: dict[str, Any], tolerance: float) -> None:
    coordinates = ("producer_node", "producer_port", "variant_id", "partition", "fold_id", "level")
    def index(value: dict[str, Any]) -> dict[tuple[Any, ...], Any]:
        reports = value["reports"]
        result = {tuple(row.get(key) for key in coordinates): row for row in reports}
        require(len(result) == len(reports), "Native score coordinates must be unique")
        return result
    left, right = index(actual), index(expected)
    require(left.keys() == right.keys(), "Native score coordinates changed across hosts")
    for key, report in right.items():
        require(left[key]["row_count"] == report["row_count"] and left[key]["target_names"] == report["target_names"], "Native score row/target coverage changed")
        for metric, value in report["metrics"].items():
            compare_numbers(left[key]["metrics"][metric], value, tolerance, f"score {key}/{metric}")


def capture_sdk(output: Path, sdk_root: Path) -> dict[str, Any]:
    path = sdk_root / "tests/integration/api/test_methods_multimodal_u07.py"
    spec = importlib.util.spec_from_file_location("canonical_u07_qualification", path)
    require(spec is not None and spec.loader is not None, "SDK canonical qualification helper is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_qualification(output)


def replace_owner(value: Any, host: str) -> Any:
    if isinstance(value, list):
        return [replace_owner(child, host) for child in value]
    if isinstance(value, dict):
        return {key: replace_owner(child, host) for key, child in value.items()}
    return f"controller:methods.{host}.multimodal" if value == "controller:methods.python.multimodal" else value


def python_sources(sources: dict[str, Any]) -> dict[str, Any]:
    return {name: {"sample_ids": source["sample_ids"], "descriptor": source["descriptor"],
                   "values": np.asarray(source["rows"], dtype=np.dtype(source["descriptor"]["dtype"])) if name == "metadata" else np.asarray(source["data"], dtype=np.dtype(source["descriptor"]["dtype"])).reshape(source["shape"])}
            for name, source in sources.items()}


class ProcessWorker:
    """Persistent public process frames; all operator semantics stay in host."""
    def __init__(self, command: list[str], config: dict[str, Any], work: Path, host: str) -> None:
        self.host, self.calls, self.closed = host, [], False
        self.runtime: dict[str, Any] = {}
        self.audit_path = Path(config["audit_path"])
        self.stderr = (work / "worker.stderr.log").open("w")
        env = {**os.environ, "DAGML_METHODS_MULTIMODAL_CONFIG": str(work / "config.json")}
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1, env=env)
        self.lines: queue.Queue[str | None] = queue.Queue()
        def reader() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            response = self.frame("init", controller_id=config["controller_id"])
            require(response.get("status") == "initialized", "Real host initialization failed")
            self.runtime = response["runtime"]
        except BaseException:
            self.close(abort=True)
            raise

    def frame(self, kind: str, **values: Any) -> dict[str, Any]:
        require(self.process.poll() is None, "Host worker exited")
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"type": kind, "schema_version": 1, **values}, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=120)
        except queue.Empty as error:
            raise TimeoutError("Actual host worker timed out") from error
        require(line is not None, "Host worker EOF")
        reply = json.loads(line)
        require(reply["schema_version"] == 1, "Process schema mismatch")
        if reply["type"] == "error":
            raise RuntimeError(reply["error"]["message"])
        return reply

    def operator(self, task: dict[str, Any]) -> dict[str, Any]:
        result = self.frame("task", task=task)["result"]
        require(result["lineage"]["seed"] == task["seed"], "Host rounded the exact native u64 seed")
        self.calls.append({"operation": task["phase"], "node": task["node_plan"]["node_id"], "fold": task.get("fold_id"), "variant": task.get("variant_id"), "task": copy.deepcopy(task), "result": copy.deepcopy(result)})
        return result

    def artifact(self, message: dict[str, Any]) -> Any:
        operation = message["operation"]
        names = {"export": "export_artifact_payload", "hydrate": "hydrate_artifact_payload", "release": "release_hydrated_artifact_payload"}
        reply = self.frame("portable_artifact", task={**message, "operation": names[operation], "schema_version": 1})["result"]
        self.calls.append({"operation": operation})
        return reply.get("payload") if operation == "export" else reply.get("handle") if operation == "hydrate" else None

    def close(self, *, abort: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None and not abort:
                require(self.frame("close").get("status") == "closed", "Worker did not acknowledge cleanup")
        finally:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
            self.stderr.close()

    def __enter__(self) -> ProcessWorker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close(abort=args[0] is not None)


def prepare_worker(capture: dict[str, Any], work: Path, host: str, *, sources: Any = None, allow_fit: bool = True, producer_host: str | None = None) -> tuple[list[str], dict[str, Any]]:
    from dag_ml.multimodal_methods import manifest_for_host
    work.mkdir(parents=True, exist_ok=False)
    owner = host if producer_host is None else producer_host
    require(not allow_fit or owner == host, "Fitting cannot borrow another host owner")
    config = {"operators": capture["operators"], "node_params": capture["node_params"], "source_ids": capture["source_ids"], "sources": capture["sources"] if sources is None else sources,
              "targets": capture["targets"] if allow_fit else None, "target_names": capture["target_names"], "allow_fit": allow_fit, "controller_id": f"controller:methods.{owner}.multimodal", "manifest": manifest_for_host(owner), "trusted_manifest": manifest_for_host(owner), "audit_path": str(work / "lifecycle.jsonl")}
    write_json(work / "config.json", config)
    if host == "r":
        command = [os.environ["DAG_ML_RSCRIPT"], str(ROOT / "examples/adapters/r_methods_multimodal_adapter.R")]
    elif host == "octave":
        adapter = str(ROOT / "examples/adapters").replace("'", "''")
        command = [os.environ["DAG_ML_OCTAVE"], "--no-gui", "--quiet", "--eval", f"addpath('{adapter}'); octave_methods_multimodal_adapter();"]
    elif host == "wasm":
        command = [os.environ.get("DAG_ML_NODE", "node"), str(ROOT / "scripts/smoke_wasm_u07_methods_multimodal.mjs"), "--worker", str(work / "config.json"), os.environ["DAG_ML_METHODS_WASM_DIST"]]
    else:
        raise ValueError("Real process host required")
    return command, config


def qualify(capture: dict[str, Any], work: Path, host: str, tolerance: float = 1e-7) -> dict[str, Any]:
    import dag_ml
    from dag_ml.multimodal_methods import manifest_for_host
    require(math.isfinite(tolerance) and tolerance > 0, "Positive finite tolerance required")
    request = replace_owner(copy.deepcopy(capture["training_request"]), host)
    request["controller_manifests"] = [manifest_for_host(host)]
    signed = dag_ml.sign_training_request(request)
    inputs = capture["training_inputs"]
    work.mkdir(parents=True, exist_ok=False)
    command, config = prepare_worker(capture, work / "training", host)
    with ProcessWorker(command, config, work / "training", host) as worker:
        training = dag_ml.execute_training(signed, inputs["data_envelopes"], inputs["relations"], inputs["training_influence"], worker.operator,
                                         outcome_id=f"outcome:u07.{host}", run_id=f"run:u07.{host}", bundle_id=f"bundle:u07.{host}", artifact_callback=worker.artifact)
        outcome = training.outcome.to_dict()
        package = training.export_portable_predictor_package(f"package:u07.{host}", fitted_artifact_mode="portable_required", artifact_load_mode="native_portable").to_dict()
        training.detach()
        calls, runtime = worker.calls, worker.runtime
    compare_scores(outcome["score_set"], capture["training_outcome"]["score_set"], tolerance)
    require(len(package["artifact_bindings"]) == len(package["execution_bundle"]["refit_artifacts"]) == 1, "One complete early-fusion predictor required")
    require(package["execution_bundle"]["refit_artifacts"][0]["artifact"]["kind"] == "methods_multimodal_pipeline", "Historical projected fixture cannot qualify U07")
    require(outcome["selected_variant_id"] == capture["training_outcome"]["selected_variant_id"], "Native SELECT changed across hosts")
    manifest, members = dag_ml.build_archive_v2_native_portable_payloads(f"archive:u07.{host}", outcome, package)
    require(len(manifest["payloads"]["methods"]["multimodal_pipelines"]) == 1, "Archive must carry one complete native predictor")
    from nirs4all import write_portable_predictor_archive_v2, read_portable_predictor_archive_v2
    archive = work / "complete.n4a"
    write_portable_predictor_archive_v2(archive, outcome=outcome, package=package, archive_id=f"archive:u07.{host}")
    loaded = read_portable_predictor_archive_v2(archive).to_dict()
    search_args = replace_owner(copy.deepcopy(capture["search_request"]), host)
    search_args["controller_manifests"] = [manifest_for_host(host)]
    proposals = [trial["params"] for trial in capture["search_result"]["trials"]]
    require(len(proposals) == 8, "Canonical native eight-proposal search required")
    def optimizer(message: dict[str, Any]) -> Any:
        if message["operation"] == "ask":
            return proposals[message["trial_index"]]
        if message["operation"] == "report_intermediate":
            return False
        return None
    command, config = prepare_worker(capture, work / "hpo", host)
    with ProcessWorker(command, config, work / "hpo", host) as worker:
        search = dag_ml.run_host_hpo_search_in_process(search_args["dsl"], search_args["envelope"], search_args["controller_manifests"], search_args["request"], worker.operator, optimizer)
    require(search["selected_trial_index"] == capture["search_result"]["selected_trial_index"], "Native HPO selected another proposal")
    for trial, expected in zip(search["trials"], capture["search_result"]["trials"], strict=True):
        require(trial["params"] == expected["params"], "Host changed native proposals")
        compare_numbers(trial["score"], expected["score"], tolerance, "Native HPO objective")
        compare_scores(trial["scores"], expected["scores"], tolerance)
    prediction_ids = capture["prediction_dataset"]["sample_ids"]
    native_prediction_ids = next(iter(capture["prediction_envelopes"].values()))["predict_cohort"]["physical_sample_ids"]
    dag_ml.align_named_source_rows({
        "sample_ids": prediction_ids,
        "required_source_ids": ["prediction-cohort"],
        "sources": [{"source_id": "prediction-cohort", "sample_ids": native_prediction_ids}],
    })
    command, config = prepare_worker(capture, work / "replay", host, sources=capture["prediction_sources"], allow_fit=False)
    replay_request = copy.deepcopy(capture["prediction_request"])
    replay_request["source_outcome_fingerprint"] = loaded["training_outcome"]["outcome_fingerprint"]
    replay_request = dag_ml.sign_training_replay_request(replay_request)
    with ProcessWorker(command, config, work / "replay", host) as worker:
        replay = json.loads(dag_ml.replay_loaded_predictor_package_json(loaded, replay_request, capture["prediction_envelopes"], {}, worker.operator,
                            outcome_id=f"outcome:u07.{host}.replay", run_id=f"run:u07.{host}.replay", artifact_callback=worker.artifact, trusted_controller_manifests=[manifest_for_host(host)]))
        replay_calls = Counter(row["operation"] for row in worker.calls)
    require(replay_calls == Counter({"hydrate": 1, "PREDICT": 1, "release": 1}), "Fresh replay must hydrate/predict/release exactly one state with no fit or HPO")
    predictions = replay_prediction_values(replay, prediction_ids, capture["target_names"])
    compare_numbers(predictions, capture["expected_prediction"], tolerance, "Fresh raw-input replay")
    lifecycle = [json.loads(line) for line in Path(config["audit_path"]).read_text().splitlines()]
    require(Counter(row["operation"] for row in lifecycle) == Counter({"hydrate": 1, "PREDICT": 1, "release": 1, "dispose": 1}), "Actual native host leaked a replay state or fitted")
    # Consume the exact original Python producer archive in a new host process.
    # Signed owner, package, SELECT and learned state remain byte-identical.
    producer_package = read_portable_predictor_archive_v2(capture["archive"]).to_dict()
    command, config = prepare_worker(capture, work / "python-archive-replay", host, sources=capture["prediction_sources"], allow_fit=False, producer_host="python")
    with ProcessWorker(command, config, work / "python-archive-replay", host) as worker:
        cross_replay = json.loads(dag_ml.replay_loaded_predictor_package_json(producer_package, capture["prediction_request"], capture["prediction_envelopes"], {}, worker.operator,
                            outcome_id=f"outcome:u07.python-in-{host}", run_id=f"run:u07.python-in-{host}", artifact_callback=worker.artifact, trusted_controller_manifests=[manifest_for_host("python")]))
        cross_calls = Counter(row["operation"] for row in worker.calls)
        cross_runtime = worker.runtime
    require(cross_calls == Counter({"hydrate": 1, "PREDICT": 1, "release": 1}), "Producer archive replay must preserve one no-fit lifecycle")
    compare_numbers(replay_prediction_values(cross_replay, prediction_ids, capture["target_names"]), capture["expected_prediction"], tolerance, "Unchanged Python archive in another host")
    write_json(work / "training-outcome.json", outcome)
    write_json(work / "portable-package.json", package)
    write_json(work / "training-callbacks.json", calls)
    write_json(work / "hpo-result.json", search)
    receipt = {"host": host, "runtime": runtime, "training_callbacks": dict(Counter(row["operation"] for row in calls)), "replay_callbacks": dict(replay_calls), "producer_archive_replay": {"runtime": cross_runtime, "callbacks": dict(cross_calls), "source_archive_sha256": hashlib.sha256(Path(capture["archive"]).read_bytes()).hexdigest()},
               "state_count": 1, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "manifest": manifest, "member_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}, "source_shapes": {name: source["shape"] for name, source in capture["sources"].items()}}
    write_json(work / "qualification.json", receipt)
    return receipt


def main(default_host: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", choices=("r", "wasm", "octave"), default=default_host, required=default_host is None)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    if args.capture is None:
        require(args.sdk_root is not None, "Existing SDK U07 helper must supply the canonical fixture")
        capture = capture_sdk(args.output.parent / "canonical-sdk-capture", args.sdk_root)
    else:
        capture = json.loads(args.capture.read_text())
    print(json.dumps(qualify(capture, args.output, args.host, args.tolerance), allow_nan=False))


if __name__ == "__main__":
    main()
