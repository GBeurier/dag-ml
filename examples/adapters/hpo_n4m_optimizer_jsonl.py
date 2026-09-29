#!/usr/bin/env python3
"""N4M-backed JSONL optimizer for the dag-ml-cli host HPO command.

The request's optimizer_descriptor.n4m declares an ordered search space,
sampler, pruner, seed and absolute state_path dedicated to one search.
DAG-ML still owns folds, scores, trial states and selection.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from n4m.model_selection import (
    Direction,
    Optimizer,
    Pruner,
    Sampler,
    SearchSpace,
    TrialStatus,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=".n4m-hpo-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _axis_value(trial: Any, axis: dict[str, Any]) -> Any:
    kind = axis["kind"]
    name = axis["name"]
    if kind == "int":
        return trial.get_int(name)
    if kind == "float":
        return trial.get_float(name)
    if kind == "categorical":
        index, _label = trial.get_category(name)
        return axis["choices"][index]
    raise ValueError(f"unsupported N4M axis kind {kind!r}")


def _record_params(record: Any) -> dict[str, Any]:
    return dict(record.params)


def _terminal_parts(
    terminal: dict[str, Any],
) -> tuple[int, dict[str, Any], str, float | None]:
    state = terminal["state"]
    evidence = terminal.get("evidence", terminal)
    return (
        evidence["trial_index"],
        evidence["params"],
        state,
        evidence.get("score") if state == "complete" else None,
    )


class N4MHostOptimizer:
    def __init__(self, event: dict[str, Any]) -> None:
        request = event["request"]
        self.config = request["optimizer_descriptor"]["n4m"]
        state_path = self.config.get("state_path") or os.environ.get(
            "DAGML_N4M_STATE_PATH"
        )
        if not state_path or not Path(state_path).is_absolute():
            raise ValueError("N4M state_path must be an absolute path")
        self.path = Path(state_path)
        self.axes = self.config["space"]
        if not isinstance(self.axes, list) or not self.axes:
            raise ValueError(
                "optimizer_descriptor.n4m.space must be a nonempty ordered axis list"
            )
        names = [axis["name"] for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("N4M axis names must be unique")
        self.contract = {
            "n4m": self.config,
            "metric": request["metric"],
            "direction": request["direction"],
            "target_node": request["target_node"],
        }
        self.space: SearchSpace | None = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                self.state.get("format") != "dagml.n4m.jsonl.v1"
                or self.state.get("contract") != self.contract
            ):
                raise ValueError("N4M state contract differs from the CLI request")
            raw = base64.b64decode(self.state["n4mopt_b64"], validate=True)
            if hashlib.sha256(raw).hexdigest() != self.state["n4mopt_sha256"]:
                raise ValueError("N4M checkpoint digest mismatch")
            self.optimizer = Optimizer.load(raw)
            self._recover(event["checkpoint"])
        else:
            if event["checkpoint"] is not None:
                raise ValueError(
                    "native DAG checkpoint exists without its paired N4M state"
                )
            self.space = SearchSpace()
            for axis in self.axes:
                kind, name = axis["kind"], axis["name"]
                if kind == "int":
                    self.space.add_int(
                        name,
                        axis["low"],
                        axis["high"],
                        axis.get("step", 1),
                        axis.get("log", False),
                    )
                elif kind == "float":
                    self.space.add_float(
                        name,
                        axis["low"],
                        axis["high"],
                        axis.get("step", 0.0),
                        axis.get("log", False),
                    )
                elif kind == "categorical":
                    self.space.add_categorical(name, axis["choices"])
                else:
                    raise ValueError(f"unsupported N4M axis kind {kind!r}")
            sampler = getattr(Sampler, self.config.get("sampler", "random").upper())
            pruner_name = self.config.get("pruner", "none").upper()
            pruner = getattr(
                Pruner, "ASHA" if pruner_name == "SUCCESSIVE_HALVING" else pruner_name
            )
            direction = getattr(Direction, request["direction"].upper())
            self.optimizer = Optimizer(
                self.space,
                sampler=sampler,
                pruner=pruner,
                direction=direction,
                seed=self.config.get("seed", 0),
                n_startup_trials=self.config.get("n_startup_trials", 10),
                max_resource=self.config.get("max_resource", 0),
                reduction_factor=self.config.get("reduction_factor", 0),
            )
            self.state = {
                "format": "dagml.n4m.jsonl.v1",
                "contract": self.contract,
                "committed_checkpoint": None,
                "previous_checkpoint": None,
                "prepared_checkpoint": None,
                "recovering_orphans": [],
            }
            self._save()
            self.prepared_on_init = None
            self.interrupted_on_init = []

    def _save(self) -> None:
        raw = self.optimizer.save()
        self.state["n4mopt_b64"] = base64.b64encode(raw).decode("ascii")
        self.state["n4mopt_sha256"] = hashlib.sha256(raw).hexdigest()
        _write_atomic(self.path, self.state)

    def _validate_history(self, checkpoint: dict[str, Any]) -> None:
        records = self.optimizer.get_trials()
        native = checkpoint["trials"]
        if len(records) < len(native):
            raise ValueError("DAG checkpoint has more trials than N4M")
        statuses = {
            "complete": TrialStatus.COMPLETED,
            "pruned": TrialStatus.PRUNED,
            "failed": TrialStatus.FAILED,
        }
        for index, terminal in enumerate(native):
            trial_id, params, state, score = _terminal_parts(terminal)
            record = records[index]
            if (
                trial_id != index
                or record.id != index
                or record.status != statuses[state]
                or _record_params(record) != params
                or (state == "complete" and record.score != score)
            ):
                raise ValueError("DAG and N4M terminal histories disagree")

    def _finish_prepared(self, prepared: dict[str, Any]) -> None:
        terminal = prepared["trials"][-1]
        index, params, state, score = _terminal_parts(terminal)
        record = self.optimizer.get_trials()[index]
        if _record_params(record) != params:
            raise ValueError("prepared DAG and N4M parameters disagree")
        if record.status == TrialStatus.RUNNING:
            if state == "complete":
                self.optimizer.tell(index, score)
            elif state == "failed":
                self.optimizer.tell_result(
                    index, TrialStatus.FAILED, error="recovered_prepared_failure"
                )
            else:
                raise ValueError("prepared pruned trial lacks N4M pruning transition")
        self._validate_history(prepared)

    def _recover(self, native: dict[str, Any] | None) -> None:
        committed = self.state["committed_checkpoint"]
        previous = self.state["previous_checkpoint"]
        prepared = self.state["prepared_checkpoint"]
        if native is None:
            if committed is not None or self.optimizer.get_trials():
                raise ValueError("N4M state exists without its native DAG checkpoint")
            self.prepared_on_init = None
            self.interrupted_on_init = []
            return
        if native == committed:
            recover = (
                prepared
                if prepared is not None
                and len(prepared["trials"]) == len(native["trials"]) + 1
                else None
            )
            interrupted = list(self.state["recovering_orphans"])
        elif native == previous and committed is not None:
            # The adapter acknowledged a checkpoint but the CLI crashed before
            # publishing the corresponding native file. Its terminal may be
            # followed by interrupted candidates recovered as failed.
            recover = (
                prepared
                if prepared is not None
                and len(prepared["trials"]) == len(native["trials"]) + 1
                and committed["trials"][: len(prepared["trials"])] == prepared["trials"]
                else None
            )
            base_count = (
                len(recover["trials"]) if recover is not None else len(native["trials"])
            )
            interrupted = []
            for terminal in committed["trials"][base_count:]:
                index, params, state, _ = _terminal_parts(terminal)
                if (
                    state != "failed"
                    or terminal.get("error") != "interrupted_before_native_evaluation"
                ):
                    raise ValueError(
                        "unpublished native checkpoint has an unexpected terminal trial"
                    )
                interrupted.append({"trial_index": index, "params": params})
        else:
            raise ValueError("native DAG checkpoint does not match paired N4M state")
        self._validate_history(native)
        if recover is not None:
            self._finish_prepared(recover)
        paired_count = (
            len(recover["trials"]) if recover is not None else len(native["trials"])
        )
        records = self.optimizer.get_trials()
        for record in records[paired_count + len(interrupted) :]:
            if (
                record.id != paired_count + len(interrupted)
                or record.status != TrialStatus.RUNNING
            ):
                raise ValueError("N4M has an unpaired non-running trial")
            params = _record_params(record)
            self.optimizer.tell_result(
                record.id,
                TrialStatus.FAILED,
                error="interrupted_before_native_evaluation",
            )
            interrupted.append({"trial_index": record.id, "params": params})
        self.state["recovering_orphans"] = interrupted
        self._save()
        self.prepared_on_init = recover
        self.interrupted_on_init = interrupted

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        operation = event["operation"]
        if operation == "ask":
            trial = self.optimizer.ask()
            if trial.id != event["trial_index"]:
                raise ValueError("DAG and N4M trial IDs disagree")
            params = {axis["name"]: _axis_value(trial, axis) for axis in self.axes}
            self._save()
            return {"params": params}
        if operation == "report_intermediate":
            prune = self.optimizer.tell_intermediate(
                event["trial_index"], event["step"], event["score"]
            )
            # Before a prospective DAG terminal, the persisted trial stays
            # RUNNING so a crash becomes an interrupted trial.
            return {"prune": prune}
        if operation == "prepare_terminal":
            self.state["prepared_checkpoint"] = event["checkpoint"]
            self._save()
            return {"ok": True}
        if operation == "tell":
            self.optimizer.tell(event["trial_index"], event["score"])
            self._save()
            return {"ok": True}
        if operation == "pruned":
            record = self.optimizer.get_trials()[event["trial_index"]]
            if record.status != TrialStatus.PRUNED:
                raise ValueError("DAG pruned a trial that N4M did not prune")
            self._save()
            return {"ok": True}
        if operation == "fail":
            self.optimizer.tell_result(
                event["trial_index"],
                TrialStatus.FAILED,
                error=str(event["error"])[:200],
            )
            self._save()
            return {"ok": True}
        if operation == "checkpoint":
            checkpoint = event["checkpoint"]
            self._validate_history(checkpoint)
            self.state["previous_checkpoint"] = self.state["committed_checkpoint"]
            self.state["committed_checkpoint"] = checkpoint
            self.state["recovering_orphans"] = []
            self._save()
            return {"ok": True}
        raise ValueError(f"unsupported HPO operation {operation!r}")

    def close(self) -> None:
        self.optimizer.close()
        if self.space is not None:
            self.space.close()


def main() -> None:
    optimizer: N4MHostOptimizer | None = None
    try:
        for line in sys.stdin:
            try:
                event = json.loads(line)
                if event["operation"] == "init":
                    if optimizer is not None:
                        raise ValueError("N4M adapter already initialized")
                    optimizer = N4MHostOptimizer(event)
                    reply = {
                        "prepared_checkpoint": optimizer.prepared_on_init,
                        "interrupted": optimizer.interrupted_on_init,
                    }
                elif optimizer is None:
                    raise ValueError("N4M adapter requires init before trial events")
                else:
                    reply = optimizer.handle(event)
            except Exception as error:  # noqa: BLE001 - return one JSON reply for every event
                reply = {"error": str(error)}
            print(_canonical(reply), flush=True)
    finally:
        if optimizer is not None:
            optimizer.close()


if __name__ == "__main__":
    main()
