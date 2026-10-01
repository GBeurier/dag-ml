#!/usr/bin/env python3
"""Qualify the public Octave Methods controller against the existing Node capture.

All fits and predictions run in the public n4m.RolePipeline Octave adapter. Rust owns
HPO, folds, nested OOF, selection, refit, signing and replay. Python only
marshals published process frames and compares evidence. The input capture
is emitted by smoke_wasm_multimodal_methods_hpo.mjs; no data or proposals are
generated here. Numeric projections do not qualify raw N-D encoders.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import zipfile
from collections import Counter
from pathlib import Path
from types import TracebackType
from typing import Any, Self

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("nir", "image", "series", "metadata")
NODES = tuple("model:" + name for name in (*SOURCES, "meta"))
CONTROLLER = "controller:methods.octave.regression"
# Current installed controller contract, independent of the archive/capture.
# The Octave constructor also compares it against its independently supplied trust.
OCTAVE_MANIFEST = {
    "controller_id": CONTROLLER, "controller_version": "1.0.0",
    "operator_kind": "model", "priority": 0,
    "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
    "input_ports": [{"name": "x", "kind": "data", "representation": "tabular_numeric",
                     "cardinality": "one", "description": ""}],
    "output_ports": [{"name": "y_hat", "kind": "prediction", "representation": None,
                      "cardinality": "one", "description": ""},
                     {"name": "model", "kind": "artifact", "representation": None,
                      "cardinality": "one", "description": ""}],
    "data_requirements": {"schema_version": 1, "default_fusion": None, "metadata": {},
                          "ports": [{"name": "x", "accepted_representations": ["tabular_numeric"],
                                     "accepted_types": ["table"], "rank": None,
                                     "multi_source": False, "optional": False, "metadata": {}}]},
    "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions",
                     "consumes_oof_predictions", "emits_artifacts", "stateful", "uses_core_rng"],
    "fit_scope": "fold_train", "rng_policy": "externally_deterministic",
    "artifact_policy": "serializable",
}



def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def compare_numeric(actual: Any, expected: Any, *, label: str, tolerance: float) -> None:
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " shape")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare_numeric(left, right, label=f"{label}[{index}]", tolerance=tolerance)
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool), label + " type")
        require(math.isfinite(actual) and math.isfinite(expected) and
                math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
                f"{label}: {actual!r} != {expected!r}")


def compare_scores(actual: dict[str, Any], expected: dict[str, Any], tolerance: float) -> None:
    fields = ("producer_node", "producer_port", "variant_id", "partition", "fold_id", "level")
    def indexed(scores: dict[str, Any]) -> dict[tuple[Any, ...], Any]:
        reports = scores["reports"]
        result = {tuple(report.get(field) for field in fields): report for report in reports}
        require(len(result) == len(reports), "Score report coordinates must be unique")
        return result
    left, right = indexed(actual), indexed(expected)
    require(left.keys() == right.keys(), "Octave/Node native score coordinate coverage differs")
    for key, report in right.items():
        require(left[key]["target_names"] == report["target_names"], "Score target order differs")
        require(left[key]["row_count"] == report["row_count"], "Score cohort size differs")
        for metric, value in report["metrics"].items():
            if metric.split(":", 1)[0] in {"rmse", "mse", "mae", "r2"}:
                compare_numeric(left[key]["metrics"][metric], value,
                                label=f"score {key}/{metric}", tolerance=tolerance)


def recorded_optimizer(proposals_path: Path) -> None:
    """Qualification fixture for the native CLI optimizer JSONL protocol."""
    proposals = json.loads(proposals_path.read_text())
    for line in sys.stdin:
        message = json.loads(line)
        operation = message["operation"]
        if operation == "init":
            response = {"prepared_checkpoint": None, "interrupted": []}
        elif operation == "ask":
            response = {"params": proposals[message["trial_index"]]}
        elif operation == "report_intermediate":
            response = {"prune": False}
        elif operation in {"tell", "pruned", "fail", "prepare_terminal", "checkpoint"}:
            response = {"ok": True}
        else:
            raise ValueError("Unsupported recorded optimizer operation: " + operation)
        print(json.dumps(response, allow_nan=False), flush=True)


class OctaveWorker:
    """Persistent published JSONL frames; no operator semantics in Python."""

    def __init__(self, prepared: dict[str, Any], timeout: float = 120) -> None:
        self.prepared, self.timeout = prepared, timeout
        self.calls: list[dict[str, Any]] = []
        self.refit_tasks: list[dict[str, Any]] = []
        self.hydrated: set[int] = set()
        self.stderr = Path(prepared["workdir"]) / "worker.stderr.log"
        self._stderr_stream = self.stderr.open("w")
        self.process = subprocess.Popen([prepared["adapter"], "--jsonl"], text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._stderr_stream, bufsize=1)
        self.lines: queue.Queue[str | None] = queue.Queue()
        def reader() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self._reader = threading.Thread(target=reader, daemon=True)
        self._reader.start()
        try:
            reply = self.frame("init", controller_id=CONTROLLER)
            require(reply.get("type") == "ack" and reply.get("status") == "initialized",
                    "Octave adapter did not acknowledge initialization")
            self.runtime = reply["runtime"]
            require("+abi.2.14." in self.runtime["methods_version"],
                    "Qualification must load the public Methods ABI 2.14 runtime")
            for field in ("mex", "role_pipeline"):
                path = Path(self.runtime[field])
                require(path.is_file(), "Missing actual Octave binding origin: " + str(path))
                self.runtime[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except BaseException as error:
            try:
                self.close(abort=True)
            except BaseException as cleanup_error:  # noqa: BLE001 - retain the initialization error.
                error.add_note(f"Worker initialization cleanup also failed: {cleanup_error}")
            raise

    def frame(self, kind: str, **payload: Any) -> dict[str, Any]:
        require(self.process.poll() is None, "Octave worker exited: " + self.stderr.read_text())
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"type": kind, "schema_version": 1, **payload},
                                           allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("Octave frame timed out: " + self.stderr.read_text()) from error
        require(line is not None, "Octave worker EOF: " + self.stderr.read_text())
        response = json.loads(line)
        require(response.get("schema_version") == 1, "Invalid Octave process response version")
        if response.get("type") == "error":
            raise RuntimeError(response["error"]["message"])
        return response

    def operator(self, task: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"operation": task["phase"], "node": task["node_plan"]["node_id"],
                           "fold": task.get("fold_id")})
        if task["phase"] == "REFIT":
            self.refit_tasks.append(copy.deepcopy(task))
        response = self.frame("task", task=task)
        require(response.get("type") == "result", "Invalid Octave NodeResult frame")
        return response["result"]

    def artifact(self, message: dict[str, Any]) -> Any:
        operation = message["operation"]
        operations = {"export": "export_artifact_payload", "hydrate": "hydrate_artifact_payload",
                      "release": "release_hydrated_artifact_payload"}
        self.calls.append({"operation": operation})
        task = {**message, "operation": operations[operation], "schema_version": 1}
        response = self.frame("portable_artifact", task=task)
        require(response.get("type") == "portable_artifact", "Invalid Octave artifact frame")
        result = response["result"]
        expected = {"export": "exported_artifact_payload", "hydrate": "hydrated_artifact_payload",
                    "release": "released_hydrated_artifact_payload"}[operation]
        require(result.get("operation") == expected and result.get("schema_version") == 1,
                "Invalid Octave artifact operation response")
        if operation == "export":
            return result["payload"]
        if operation == "hydrate":
            self.hydrated.add(result["handle"]["handle"])
            return result["handle"]
        self.hydrated.remove(message["handle"]["handle"])
        return None

    def close(self, *, abort: bool = False) -> None:
        failure: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            if self.process.poll() is None and not abort:
                reply = self.frame("close")
                require(reply.get("status") == "closed", "Octave close acknowledgement missing")
        except BaseException as error:  # noqa: BLE001 - defer re-raising until cleanup completes.
            failure = error
        finally:
            # A bad/missing acknowledgement must never bypass process cleanup.
            # Bound each attempt, including a worker that ignores EOF/SIGTERM.
            shutdown_timeout = min(10, self.timeout)
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except BaseException as error:  # noqa: BLE001 - remaining cleanup must still run.
                    cleanup_errors.append(error)
            for signal in (None, self.process.terminate, self.process.kill):
                try:
                    if signal is not None and self.process.poll() is None:
                        signal()
                    self.process.wait(timeout=shutdown_timeout)
                    break
                except subprocess.TimeoutExpired as error:
                    if signal == self.process.kill:
                        cleanup_errors.append(error)
                except BaseException as error:  # noqa: BLE001 - still attempt kill and reap.
                    cleanup_errors.append(error)
            self._reader.join(timeout=shutdown_timeout)
            for stream in (self.process.stdout, self._stderr_stream):
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as error:  # noqa: BLE001 - close the other stream as well.
                        cleanup_errors.append(error)
        if failure is not None:
            for error in cleanup_errors:
                failure.add_note(f"Worker shutdown cleanup also failed: {error}")
            raise failure.with_traceback(failure.__traceback__)
        if cleanup_errors:
            raise cleanup_errors[0]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        try:
            self.close(abort=self.process.poll() is not None)
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Worker shutdown also failed: {cleanup_error}")


def prepare_octave(octave: Path, work: Path, label: str, sources: dict[str, Any],
                   operators: dict[str, Any], targets: dict[str, Any] | None,
                   *, feature_permutation: bool = False) -> dict[str, Any]:
    """Prepare the public persistent controller; Methods validates recipes at init."""
    current_sources = copy.deepcopy(sources)
    if feature_permutation:
        for source in current_sources.values():
            names = source["feature_names"]
            require(len(names) > 1, "Feature permutation needs at least two columns")
            names[0], names[1] = names[1], names[0]
    workdir = work / (label + "-adapter")
    workdir.mkdir(parents=True, exist_ok=False)
    audit_path = workdir / "lifecycle.jsonl"
    config = {"sources": current_sources, "operators": operators,
              "manifest": copy.deepcopy(OCTAVE_MANIFEST),
              "trusted_manifest": copy.deepcopy(OCTAVE_MANIFEST),
              "target_names": ["y"], "targets": targets,
              "allow_fit": targets is not None, "audit_path": str(audit_path)}
    path = workdir / "config.json"
    write_json(path, config)
    methods = Path(os.environ.get("DAG_ML_OCTAVE_METHODS_PATH",
                                str(ROOT.parent / "nirs4all-methods/bindings/matlab"))).resolve()
    adapter = workdir / "run-octave-role-adapter"
    command = [str(octave), "--quiet", "--no-gui", "--no-init-file", "--eval",
               "addpath(getenv('DAGML_OCTAVE_ADAPTER_DIR')); octave_methods_roles_adapter"]
    adapter.write_text("#!/bin/sh\nset -eu\nexport DAGML_OCTAVE_ROLE_CONFIG=" + shlex.quote(str(path)) +
                       "\nexport DAGML_OCTAVE_ADAPTER_DIR=" + shlex.quote(str(ROOT / "examples/adapters")) +
                       "\nexport DAG_ML_OCTAVE_METHODS_PATH=" + shlex.quote(str(methods)) +
                       '\nif [ "${1:-}" = --describe ]; then export DAGML_OCTAVE_MODE=describe; fi\nexec ' +
                       shlex.join(command) + "\n")
    adapter.chmod(0o755)
    prepared = {"adapter": str(adapter), "manifest": copy.deepcopy(OCTAVE_MANIFEST),
                "workdir": str(workdir), "audit_path": str(audit_path), "config": str(path),
                "octave": str(octave), "methods_path": str(methods)}
    write_json(work / (label + "-prepared.json"), prepared)
    return prepared


def source_tables(capture: dict[str, Any], *, permuted: bool) -> dict[str, Any]:
    ids = capture["sampleIds"]
    result = {}
    for index, name in enumerate(SOURCES):
        permutation = list(range(len(ids)))
        if permuted:
            shift = index + 1
            permutation = permutation[shift:] + permutation[:shift]
            if index % 2:
                permutation.reverse()
        rows = capture["sourceRows"][name]
        result[name] = {"sample_ids": [ids[i] for i in permutation],
                        "rows": [rows[i] for i in permutation],
                        "feature_names": [str(i) for i in range(len(rows[0]))]}
    return result


def audit(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(prepared["audit_path"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_no_fit_replay(worker: OctaveWorker, prepared: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(item["operation"] for item in worker.calls)
    require(counts["hydrate"] == counts["PREDICT"] == counts["release"] == 5,
            "Replay must hydrate/predict/release each of five actual models")
    require(not counts["FIT_CV"] and not counts["REFIT"] and not counts["export"] and
            not worker.hydrated, "Replay retained a model or invoked training/export")
    events = audit(prepared)
    lifecycle = Counter(item["operation"] for item in events)
    require(lifecycle["fit"] == 0 and lifecycle["hydrate"] == lifecycle["PREDICT"] ==
            lifecycle["release"] == lifecycle["dispose"] == 5,
            "Actual Octave replay lifecycle did not release all five native states without fit")
    return {"callbacks": dict(counts), "octave_lifecycle": dict(lifecycle)}


def expect_refusal(call: Any, *, contains: str | None = None) -> str:
    try:
        call()
    except Exception as error:  # noqa: BLE001 - the qualification records native refusal evidence.
        message = str(error)
        if contains is not None:
            require(contains.lower() in message.lower(), "Unexpected refusal: " + message)
        return message
    raise AssertionError("Expected qualification refusal, but operation succeeded")


def executable(path: Path) -> Path:
    selected = str(path) if path.is_absolute() or path.parent != Path(".") else shutil.which(str(path))
    require(selected is not None, "A real Octave executable is required")
    return Path(selected).resolve(strict=True)


def train_from_python_api(node_capture: Path, octave: Path, workdir: Path, *, tolerance: float = 1e-8,
                          hpo_search: Any = None, execute_training: Any = None) -> dict[str, Any]:
    """Run the recorded native HPO and training via public phase APIs.

    SDK consumers inject nirs4all.run_host_hpo_search and nirs4all.execute_training.
    The returned public outcome/package are detached portable objects. This helper
    does not write the Core archive or replay it; the consumer owns those operations.
    """
    import dag_ml

    hpo_search = dag_ml.run_host_hpo_search_in_process if hpo_search is None else hpo_search
    execute_training = dag_ml.execute_training if execute_training is None else execute_training
    require(math.isfinite(tolerance) and tolerance > 0, "Tolerance must be finite and positive")
    octave = executable(octave)
    work = workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    capture = json.loads(node_capture.read_text())
    require(capture["manifest"] == {**OCTAVE_MANIFEST, "controller_id": "controller:methods.wasm.regression"},
            "Captured fixture differs from the independently trusted controller manifest")
    require(set(capture["sourceRows"]) == set(SOURCES) and len(capture["sampleIds"]) == 12 and
            len(capture["resumed"]["trials"]) == 3, "Expected the unchanged four-source qualification capture")
    proposals = [trial["params"] for trial in capture["resumed"]["trials"]]
    operators = {node: {"type": "n4m:models.regularized.ridge"} for node in NODES}
    original_sources = source_tables(capture, permuted=False)
    permuted_sources = source_tables(capture, permuted=True)
    ids = capture["sampleIds"]
    targets = {"sample_ids": list(reversed(ids)), "values": [[y] for y in reversed(capture["target"])]}
    dsl, envelope = copy.deepcopy(capture["dsl"]), copy.deepcopy(capture["envelope"])
    request = copy.deepcopy(capture["request"]); request["trial_budget"] = len(proposals)
    def proposal(message: dict[str, Any]) -> Any:
        if message["operation"] == "ask":
            return proposals[message["trial_index"]]
        if message["operation"] == "report_intermediate":
            return False
        return None
    prepared_hpo = prepare_octave(octave, work, "python-hpo", original_sources, operators, targets)
    with OctaveWorker(prepared_hpo) as worker:
        search = hpo_search(dsl, envelope, [OCTAVE_MANIFEST], request,
                                                      worker.operator, proposal)
    for label, result in (("Python", search),):
        require(len(result["trials"]) == 3 and result["selected_trial_index"] ==
                capture["resumed"]["selected_trial_index"], label + " Octave HPO selection differs")
        for trial, expected in zip(result["trials"], capture["resumed"]["trials"], strict=True):
            require(trial["params"] == expected["params"], "Recorded proposals were changed")
            compare_numeric(trial["score"], expected["score"],
                            label=label + " HPO objective", tolerance=tolerance)
            compare_scores(trial["scores"], expected["scores"], tolerance)
    write_json(work / "python-hpo-result.json", search)
    # No host fold/OOF loop: the same selected complete request is recompiled
    # against the current Octave manifest and executed by the native training core.
    complete = capture["completeArchive"]
    training_request = json.loads(complete["requestJson"])
    selected = search["trials"][search["selected_trial_index"]]["params"]
    selected_dsl = copy.deepcopy(dsl)
    for branch in selected_dsl["steps"][0]["branches"]:
        branch["steps"][0]["params"]["alpha"] = selected[branch["id"] + ".alpha"]
    selected_dsl["steps"][1]["params"]["alpha"] = selected["meta.alpha"]
    selected_compiled = dag_ml.compile_pipeline_dsl_artifact_with_controllers(
        selected_dsl, [OCTAVE_MANIFEST]).to_dict()
    training_request.update(graph=selected_compiled["graph"],
                            campaign=selected_compiled["campaign_template"],
                            controller_manifests=[OCTAVE_MANIFEST])
    signed_request = dag_ml.sign_training_request(training_request)
    (work / "octave-signed-training-request.json").write_text(signed_request.json() + "\n")
    node_outcome = json.loads(complete["outcomeJson"])
    # Influence is an immutable native-produced fixture with identical node,
    # fold and relation coordinates; Rust revalidates it for this projection.
    influence = node_outcome["training_influence"]
    prepared_training = prepare_octave(octave, work, "five-model-training", permuted_sources,
                                  operators, targets)
    with OctaveWorker(prepared_training) as worker:
        result = execute_training(signed_request, complete["trainingEnvelopes"],
                                        envelope["coordinator_relations"], influence, worker.operator,
                                        outcome_id="outcome:methods.octave.four-sources",
                                        run_id="run:methods.octave.four-sources.training",
                                        bundle_id="bundle:methods.octave.four-sources",
                                        artifact_callback=worker.artifact)
        try:
            public_outcome = result.outcome
            outcome = public_outcome.to_dict()
            package = result.export_portable_predictor_package(
                "package:methods.octave.four-sources", fitted_artifact_mode="portable_required",
                artifact_load_mode="native_portable")
            package_dict = package.to_dict()
            require(len(package_dict["artifact_bindings"]) == 5 and
                    len(package_dict["execution_bundle"]["raw_artifact_payloads"]) == 5,
                    "Complete Octave package did not capture five actual RAW estimators")
            require(set(package_dict["predictor_node_ids"]) == set(NODES), "Octave predictor closure differs")
            require(all(binding["load_mode"] == "native_portable" for binding in
                        package_dict["artifact_bindings"]), "Octave estimator is not portable")
            require(all(record["artifact"]["plugin"] == "dagml.methods.octave.regression" for record in
                        package_dict["execution_bundle"]["refit_artifacts"]), "Octave plugin identity changed")
            compare_scores(outcome["score_set"], node_outcome["score_set"], tolerance)
            refit_task = copy.deepcopy(worker.refit_tasks[0])
        finally:
            result.detach()
        training_calls = copy.deepcopy(worker.calls)
        training_runtime = copy.deepcopy(worker.runtime)
    replay_contract = json.loads(complete["replayRequestJson"])
    replay_contract["source_outcome_fingerprint"] = outcome["outcome_fingerprint"]
    replay_request = dag_ml.sign_training_replay_request(replay_contract)
    heldout = {name: {"rows": [complete["heldoutRows"][name]], "sample_ids": ["heldout-0"],
                      "feature_names": original_sources[name]["feature_names"]} for name in SOURCES}
    return {"search": search, "training_result": result, "outcome": public_outcome, "package": package,
            "prepared": prepared_training, "prepared_hpo": prepared_hpo, "training_calls": training_calls,
            "runtime": training_runtime, "signed_request": signed_request, "operators": operators,
            "heldout": heldout, "replay_envelopes": complete["predictEnvelopes"],
            "replay_request": replay_request, "unsigned_replay_request": replay_contract,
            "node_outcome": node_outcome, "capture": capture, "refit_task": refit_task}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    import dag_ml
    import dag_ml._dag_ml as dag_native
    import nirs4all
    import nirs4all_core

    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    capture = json.loads(args.node_capture.read_text())
    require(set(capture["sourceRows"]) == set(SOURCES) and len(capture["sampleIds"]) == 12,
            "Expected current four-source, twelve-sample Node capture")
    require(len(capture["resumed"]["trials"]) == 3,
            "Expected all three recorded Methods optimizer proposals")
    expected_controller = {**OCTAVE_MANIFEST, "controller_id": "controller:methods.wasm.regression"}
    require(capture["manifest"] == expected_controller,
            "Fresh Node fixture controller differs from current known contract")
    proposals = [trial["params"] for trial in capture["resumed"]["trials"]]
    operators = {node: {"type": "n4m:models.regularized.ridge"} for node in NODES}
    original_sources = source_tables(capture, permuted=False)
    permuted_sources = source_tables(capture, permuted=True)
    ids = capture["sampleIds"]
    targets = {"sample_ids": list(reversed(ids)),
               "values": [[value] for value in reversed(capture["target"])]}
    dsl, envelope = copy.deepcopy(capture["dsl"]), copy.deepcopy(capture["envelope"])
    compiled = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, [OCTAVE_MANIFEST]).to_dict()
    plan = dag_ml.build_execution_plan("plan:methods.four-sources", compiled["graph"],
                                       compiled["campaign_template"], [OCTAVE_MANIFEST]).to_dict()
    request = copy.deepcopy(capture["request"])
    request["trial_budget"] = len(proposals)
    prepared_cli = prepare_octave(args.octave, work, "cli-hpo", permuted_sources, operators, targets)
    for name, value in (("hpo-plan", plan), ("hpo-envelope", envelope),
                        ("hpo-request", request), ("recorded-proposals", proposals)):
        write_json(work / (name + ".json"), value)
    optimizer = work / "recorded-proposals-optimizer"
    optimizer.write_text("#!/bin/sh\nexec " + shlex.join([sys.executable, str(Path(__file__).resolve()),
                        "--recorded-optimizer", str(work / "recorded-proposals.json")]) + "\n")
    optimizer.chmod(0o755)
    # Preserve exact native input frames for cross-language boundary diagnosis.
    # tee does not parse, normalize or replace any task/seed byte.
    traced_adapter = work / "trace-public-octave-adapter"
    traced_adapter.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"--jsonl\" ]; then\n  tee -a " +
        shlex.quote(str(work / "cli-native-frames.jsonl")) + " | " +
        shlex.quote(prepared_cli["adapter"]) + ' "$@"\nelse\n  exec ' +
        shlex.quote(prepared_cli["adapter"]) + ' "$@"\nfi\n'
    )
    traced_adapter.chmod(0o755)
    command = [str(args.cli), "run-host-hpo", "--plan", str(work / "hpo-plan.json"),
               "--envelope", str(work / "hpo-envelope.json"), "--request", str(work / "hpo-request.json"),
               "--operator-adapter", str(traced_adapter), "--operator-persistent",
               "--optimizer-adapter", str(optimizer), "--parallel-trials", "1",
               "--adapter-timeout-ms", "120000", "--output", str(work / "cli-hpo-result.json")]
    cli = subprocess.run(command, text=True, capture_output=True, timeout=900, check=False)
    (work / "cli-hpo.stdout.log").write_text(cli.stdout)
    (work / "cli-hpo.stderr.log").write_text(cli.stderr)
    require(cli.returncode == 0, "Public native CLI Octave HPO failed: " + cli.stderr)
    cli_search = json.loads((work / "cli-hpo-result.json").read_text())
    trained = train_from_python_api(args.node_capture, args.octave, work / "python-api", tolerance=args.tolerance)
    search = trained["search"]
    require(len(cli_search["trials"]) == 3 and cli_search["selected_trial_index"] == search["selected_trial_index"],
            "CLI and public Python HPO selections differ")
    for trial, expected in zip(cli_search["trials"], capture["resumed"]["trials"], strict=True):
        require(trial["params"] == expected["params"], "CLI changed recorded proposals")
        compare_numeric(trial["score"], expected["score"], label="CLI HPO objective", tolerance=args.tolerance)
        compare_scores(trial["scores"], expected["scores"], args.tolerance)
    outcome, package = trained["outcome"].to_dict(), trained["package"]
    package_dict, complete = package.to_dict(), capture["completeArchive"]
    prepared_hpo, prepared_training = trained["prepared_hpo"], trained["prepared"]
    training_calls, refit_task = trained["training_calls"], trained["refit_task"]
    archive = work / "five-methods-octave-models.n4a"
    reference = nirs4all.write_portable_predictor_archive_v2(
        archive, archive_id="archive:methods.octave.four-sources", outcome=trained["outcome"], package=package)
    read_package = nirs4all.read_portable_predictor_archive_v2(archive)
    require(read_package.to_dict() == package_dict, "Core archive changed the captured package")
    payloads = nirs4all_core.read_archive_v2_payloads(str(archive))
    require(payloads["manifest"]["payloads"]["methods"]["n4mm"] == [] and
            len(payloads["manifest"]["payloads"]["methods"]["role_pipelines"]) == 5 and
            len(payloads["members"]) == 11, "Octave archive must retain pure five-role transport")
    replay_request = json.loads(complete["replayRequestJson"])
    replay_request["source_outcome_fingerprint"] = outcome["outcome_fingerprint"]
    replay_request = dag_ml.sign_training_replay_request(replay_request)
    (work / "octave-signed-replay-request.json").write_text(replay_request.json() + "\n")
    heldout = {name: {"rows": [complete["heldoutRows"][name]], "sample_ids": ["heldout-0"],
                      "feature_names": original_sources[name]["feature_names"]} for name in SOURCES}
    replay_envelopes = complete["predictEnvelopes"]
    def replay(worker: OctaveWorker, path: Path = archive, trusted: Any = None,
               op: Any = None, request_override: Any = None) -> Any:
        return nirs4all.replay_portable_predictor_archive_v2(
            path, replay_request if request_override is None else request_override,
            replay_envelopes, [OCTAVE_MANIFEST] if trusted is None else trusted,
            worker.operator if op is None else op, artifact_callback=worker.artifact,
            outcome_id="outcome:methods.octave.replay", run_id="run:methods.octave.replay")
    prepared_replay = prepare_octave(args.octave, work, "fresh-archive-replay", heldout, operators, None)
    with OctaveWorker(prepared_replay) as worker:
        prediction_outcome = replay(worker).to_dict()
        replay_proof = check_no_fit_replay(worker, prepared_replay)
        replay_runtime = copy.deepcopy(worker.runtime)
        before = len(worker.calls)
        bad_trust = [{**OCTAVE_MANIFEST, "controller_version": "999.0.0"}]
        trust_error = expect_refusal(lambda: replay(worker, trusted=bad_trust))
        bad_request = replay_request.to_dict()
        bad_request["phase"] = "REFIT"
        request_error = expect_refusal(lambda: replay(worker, request_override=bad_request))
        require(len(worker.calls) == before, "Untrusted/unsigned replay reached Octave callbacks")
        stored = dict(payloads["members"])
        role_path = payloads["manifest"]["payloads"]["methods"]["role_pipelines"][0]["member_path"]
        negative_errors = {}
        for mutation in ("tampered", "missing", "alias"):
            mutant = work / (mutation + ".n4a")
            with zipfile.ZipFile(archive) as source, zipfile.ZipFile(mutant, "w") as destination:
                for member in source.infolist():
                    content = source.read(member.filename)
                    name = member.filename
                    if name == role_path:
                        if mutation == "missing":
                            continue
                        if mutation == "alias":
                            name = "artifacts/alias.json"
                        else:
                            content = bytes([content[0] ^ 1]) + content[1:]
                    destination.writestr(name, content)
            negative_errors[mutation] = expect_refusal(lambda mutant=mutant: replay(worker, mutant))
            require(len(worker.calls) == before, "Invalid archive reached Octave callbacks")
        require(stored == payloads["members"], "Negative guards mutated original archive payloads")
    prediction = prediction_outcome["outputs"][0]["predictions"][0]
    expected_prediction = complete["replay"]["outputs"][0]["predictions"][0]
    require(prediction["sample_ids"] == expected_prediction["sample_ids"] and
            prediction["target_names"] == expected_prediction["target_names"], "Heldout identity/order differs")
    compare_numeric(prediction["values"], expected_prediction["values"],
                    label="five-model heldout Octave/Node replay", tolerance=args.tolerance)
    prepared_failed = prepare_octave(args.octave, work, "failed-replay", heldout, operators, None)
    with OctaveWorker(prepared_failed) as worker:
        def fail_predict(task: dict[str, Any]) -> Any:
            require(task["phase"] == "PREDICT", "Replay attempted fit")
            raise RuntimeError("Injected qualification prediction failure")
        failure_error = expect_refusal(lambda: replay(worker, op=fail_predict), contains="Injected qualification")
        counts = Counter(item["operation"] for item in worker.calls)
        lifecycle = Counter(item["operation"] for item in audit(prepared_failed))
        require(counts["hydrate"] == counts["release"] == lifecycle["dispose"] == 5 and
                not worker.hydrated and lifecycle["fit"] == 0,
                "Failed replay did not release all five actual Octave states")
        cleanup_proof = {"callbacks": dict(counts), "octave_lifecycle": dict(lifecycle)}
    prepared_features = prepare_octave(args.octave, work, "permuted-feature-order", heldout, operators,
                                  None, feature_permutation=True)
    with OctaveWorker(prepared_features) as worker:
        feature_error = expect_refusal(lambda: replay(worker), contains="feature order")
        lifecycle = Counter(item["operation"] for item in audit(prepared_features))
        require(lifecycle["fit"] == lifecycle["PREDICT"] == 0 and
                lifecycle["hydrate"] == lifecycle["dispose"] == 5 and not worker.hydrated,
                "Same-width feature-order refusal executed numerics or leaked native state")
    prepared_no_fit = prepare_octave(args.octave, work, "explicit-fit-refusal", heldout, operators, None)
    with OctaveWorker(prepared_no_fit) as worker:
        fit_error = expect_refusal(lambda: worker.operator(refit_task), contains="fitting is disabled")
        require(Counter(item["operation"] for item in audit(prepared_no_fit))["fit"] == 0,
                "allow_fit=FALSE permitted a numerical fit")
    # Native Octave FIT events contain the exact coordinator-selected cohort. Check
    # actual external folds, including inner-CV train rows, by stable identity.
    for prepared in (prepared_cli, prepared_hpo, prepared_training):
        events = audit(prepared)
        require(any(event["operation"] == "fit" for event in events), "Qualification did not fit actual Octave models")
        require(all(set(event.get("sample_ids") or []).issubset(ids) for event in events),
                "Heldout sample entered native training")
    proof = {"status": "passed", "node_capture": str(args.node_capture.resolve()),
             "node_capture_sha256": hashlib.sha256(args.node_capture.read_bytes()).hexdigest(),
             "cli_command": command, "cli_hpo_trials": 3, "python_hpo_trials": 3,
             "hpo_scores": {"node": [trial["score"] for trial in capture["resumed"]["trials"]],
                            "octave_cli": [trial["score"] for trial in cli_search["trials"]],
                            "octave_python": [trial["score"] for trial in search["trials"]]},
             "selected_trial_index": search["selected_trial_index"], "sources": list(SOURCES),
             "source_widths": {name: len(original_sources[name]["rows"][0]) for name in SOURCES},
             "native_nested_oof": True, "independent_source_and_target_row_permutations": True,
             "actual_raw_models": 5, "archive_members": 11, "archive": str(archive),
             "training_callbacks": training_calls,
             "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "archive_reference": reference,
             "package_fingerprint": package_dict["package_fingerprint"], "prediction": prediction,
             "fresh_octave_replay": replay_proof, "failed_replay_cleanup": cleanup_proof,
             "numerical_tolerance": args.tolerance,
             "negative_guards": {"trust": trust_error, "unsigned_phase": request_error,
                                 **negative_errors, "failure_cleanup": failure_error,
                                 "same_width_feature_order": feature_error, "fit_disabled": fit_error},
             "python_runtime": {"executable": sys.executable, "dag_ml": dag_ml.__file__,
                                "dag_ml_native": dag_native.__file__, "sdk": nirs4all.__file__,
                                "dag_ml_native_sha256": hashlib.sha256(
                                    Path(dag_native.__file__).read_bytes()).hexdigest(),
                                "core": nirs4all_core.__file__}, "octave_runtime": {**prepared_replay, **replay_runtime},
             "scope": "four numeric projections plus OOF meta Ridge; raw N-D encoders are unqualified"}
    write_json(work / "qualification-receipt.json", proof)
    write_json(work / "octave-training-outcome.json", outcome)
    write_json(work / "octave-portable-predictor-package.json", package_dict)
    write_json(work / "fresh-octave-replay-outcome.json", prediction_outcome)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-optimizer", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--node-capture", type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--octave", type=Path, default=Path(os.environ.get("DAG_ML_OCTAVE", "octave")))
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.recorded_optimizer is not None:
        recorded_optimizer(args.recorded_optimizer)
        return
    if args.node_capture is None or args.cli is None or args.workdir is None:
        parser.error("--node-capture, --cli and --workdir are required")
    require(args.tolerance > 0 and math.isfinite(args.tolerance), "Tolerance must be finite and positive")
    args.cli = args.cli.resolve(strict=True)
    args.octave = executable(args.octave)
    proof = run_gate(args)
    print(json.dumps({"status": proof["status"], "cli_hpo_trials": proof["cli_hpo_trials"],
                      "raw_models": proof["actual_raw_models"], "archive_sha256": proof["archive_sha256"],
                      "fresh_octave_replay": proof["fresh_octave_replay"],
                      "receipt": str(args.workdir.resolve() / "qualification-receipt.json")}))


if __name__ == "__main__":
    main()
