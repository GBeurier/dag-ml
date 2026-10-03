"""Native Methods predictors behind DAG-owned grouped OOF topology execution.

Raw encoders use the existing multimodal controller. This adapter transports
native prediction rows to a native Ridge RolePipeline; it never chooses folds,
constructs OOF predictions, selects a recipe, or computes scores.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from .multimodal_methods import (
    MAX_PAYLOAD,
    MAX_STATE,
    SOURCE_ORDER,
    MethodsMultimodalController,
    _bounded_identifier,
    _strict_json,
    require,
)

META_CONTROLLER = "controller:methods.python.regression"
META_PLUGIN = "dagml.methods.python.regression"
META_KIND = "methods_role_pipeline"


def _meta_recipe(
    operator: dict[str, Any], params: dict[str, Any]
) -> list[dict[str, Any]]:
    require(
        set(operator) == {"type", "steps", "source_order"}
        and operator["type"] == "N4mRolePipeline",
        "Explicit OOF Ridge operator required",
    )
    order = operator["source_order"]
    require(
        isinstance(order, list)
        and 2 <= len(order) <= 4
        and len(set(order)) == len(order)
        and all(name in SOURCE_ORDER for name in order),
        "OOF Ridge requires distinct ordered raw-source branches",
    )
    steps = operator["steps"]
    require(
        isinstance(steps, list)
        and len(steps) == 1
        and set(steps[0]) == {"methodId", "params"}
        and steps[0]["methodId"] == "models.regularized.ridge",
        "One native Ridge meta-model required",
    )
    require(
        isinstance(params, dict) and set(params) <= {"alpha"},
        "Unsupported meta-model effective parameters",
    )
    result = copy.deepcopy(steps)
    result[0]["params"].update(params)
    values = result[0]["params"]
    require(
        set(values) == {"alpha", "center_x", "center_y", "scale_x"}
        and type(values["alpha"]) in (int, float)
        and np.isfinite(values["alpha"])
        and values["alpha"] >= 0
        and values["center_x"] is True
        and values["center_y"] is True
        and values["scale_x"] is False,
        "Closed ordinary Ridge parameters required",
    )
    return result


def _role_steps(recipe: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [(step["methodId"], copy.deepcopy(step["params"])) for step in recipe]


class MethodsOOFController:
    """Own only native meta-models fitted on scheduler-supplied OOF rows."""

    def __init__(
        self,
        *,
        operators: dict[str, Any],
        raw_operators: dict[str, Any],
        edges: list[dict[str, Any]] | None,
        targets: dict[str, Any] | None,
        target_names: tuple[str, ...],
        allow_fit: bool,
        node_params: dict[str, Any] | None = None,
    ) -> None:
        from . import derive_controller_manifest_json

        require(
            bool(operators)
            and tuple(target_names) == ("y",)
            and type(allow_fit) is bool
            and (targets is not None) == allow_fit,
            "Exact meta-model replay/target policy required",
        )
        self.operators = copy.deepcopy(operators)
        self.raw_operators = copy.deepcopy(raw_operators)
        self.edges = copy.deepcopy(edges)
        expected_params = {} if node_params is None else node_params
        require(
            isinstance(expected_params, dict)
            and set(expected_params) <= set(self.operators),
            "Foreign expected meta parameter node",
        )
        self.node_params = copy.deepcopy(expected_params)
        self.targets = targets
        self.allow_fit = allow_fit
        self.target_names = list(target_names)
        self.controller_id = META_CONTROLLER
        self.manifest = json.loads(
            derive_controller_manifest_json(
                json.dumps(
                    {
                        "controller_id": META_CONTROLLER,
                        "controller_version": "1.0.0",
                        "operator_kind": "model",
                        "added_capabilities": ["consumes_oof_predictions"],
                        "rng_policy": "externally_deterministic",
                    }
                )
            )
        )
        self.models: dict[int, dict[str, Any]] = {}
        self.artifacts: dict[str, bytes] = {}
        self.next_handle = 1
        self.audit: list[dict[str, Any]] = []
        self.closed = False
        for node_id, operator in self.operators.items():
            _meta_recipe(operator, self.node_params.get(node_id, {}))
        if edges is not None:
            for node_id in self.operators:
                self._input_nodes(node_id)

    def _event(self, operation: str, **fields: Any) -> None:
        self.audit.append(
            {"operation": operation, "controller_id": self.controller_id, **fields}
        )

    def _input_nodes(self, node_id: str) -> list[str]:
        require(
            self.edges is not None, "Signed prediction edges are required for execution"
        )
        incoming = [edge for edge in self.edges if edge["target"]["node_id"] == node_id]
        require(
            2 <= len(incoming) <= 4
            and all(
                edge["contract"]["kind"] == "prediction"
                and edge["contract"].get("requires_oof") is True
                for edge in incoming
            ),
            "Meta-model requires signed OOF prediction edges only",
        )
        sources: dict[str, str] = {}
        for edge in incoming:
            producer = edge["source"]["node_id"]
            require(producer in self.raw_operators, "Foreign OOF producer")
            selected = self.raw_operators[producer]["recipe"]["source_order"]
            require(
                len(selected) == 1 and selected[0] not in sources,
                "Late branches must select distinct single sources",
            )
            sources[selected[0]] = producer
        order = self.operators[node_id]["source_order"]
        require(
            set(order) == set(sources),
            "Declared meta branch order differs from signed edges",
        )
        return [sources[name] for name in order]

    def _predictions(
        self, task: dict[str, Any], partition: str
    ) -> tuple[list[str], np.ndarray, list[str]]:
        phase = task["phase"]
        suffix = {"FIT_CV": ":outer", "REFIT": ":refit", "PREDICT": ":predict"}[phase]
        if partition == "test":
            suffix = ":test"
        selected = []
        for key, block in task.get("prediction_inputs", {}).items():
            operational = any(
                key.endswith(end) for end in (":outer", ":refit", ":predict", ":test")
            )
            if (partition == "train" and not operational) or (
                partition != "train" and key.endswith(suffix)
            ):
                selected.append(block)
        expected = self._input_nodes(task["node_plan"]["node_id"])
        require(
            len(selected) == len(expected),
            "Complete native prediction columns required",
        )
        by_producer = {block["producer_node"]: block for block in selected}
        require(
            len(by_producer) == len(selected) and set(by_producer) == set(expected),
            "OOF producers differ from signed topology",
        )
        ids: list[str] | None = None
        arrays, names = [], []
        for producer in expected:
            block = by_producer[producer]
            assert self.edges is not None
            edge = next(
                edge
                for edge in self.edges
                if edge["target"]["node_id"] == task["node_plan"]["node_id"]
                and edge["source"]["node_id"] == producer
            )
            require(
                block["source_port"] == edge["source"]["port_name"]
                and block["target_port"] == edge["target"]["port_name"],
                "Prediction input ports differ from signed topology",
            )
            samples = block["sample_ids"]
            require(
                isinstance(samples, list)
                and samples
                and all(isinstance(sample, str) and sample for sample in samples)
                and len(set(samples)) == len(samples),
                "Native prediction identities must be complete and unique",
            )
            require(
                ids is None or samples == ids,
                "Native prediction columns must share exact row order",
            )
            ids = samples
            require(
                block.get("prediction_level", "sample") == "sample"
                and block["prediction_width"] == 1
                and block["target_names"] == self.target_names,
                "OOF Ridge requires one sample-level target column per branch",
            )
            training = partition == "train"
            require(
                not training or block["partition"] == "validation",
                "Meta fitting must consume validation OOF exclusively",
            )
            require(
                not training
                or phase != "FIT_CV"
                or task["fold_id"] not in block.get("fold_ids", []),
                "Outer validation fold cannot fit its meta-model",
            )
            if not training:
                allowed = (
                    {"validation"}
                    if phase == "FIT_CV" and partition != "test"
                    else {"test"}
                    if partition == "test"
                    else {"final", "test"}
                    if phase == "REFIT"
                    else {"final"}
                )
                require(
                    block["partition"] in allowed,
                    "Invalid native meta prediction partition",
                )
            values = np.asarray(block["values"], dtype=float)
            require(
                values.shape == (len(samples), 1) and np.isfinite(values).all(),
                "Finite native prediction matrix required",
            )
            arrays.append(values)
            names.append(f"{producer}/0")
        assert ids is not None
        return ids, np.concatenate(arrays, axis=1), names

    def _targets(self, ids: list[str]) -> np.ndarray:
        require(self.targets is not None, "Replay cannot read training targets")
        positions = {
            sample: index for index, sample in enumerate(self.targets["sample_ids"])
        }
        require(
            all(sample in positions for sample in ids), "Unknown OOF target identity"
        )
        values = np.asarray(self.targets["values"], dtype=float)
        selected = values[[positions[sample] for sample in ids]].reshape(len(ids), 1)
        require(np.isfinite(selected).all(), "Complete finite meta targets required")
        return selected

    @staticmethod
    def _frame(rows: tuple[list[str], np.ndarray, list[str]]) -> Any:
        # DataFrame carries column identities through the official native facade.
        import pandas as pd

        return pd.DataFrame(rows[1], columns=rows[2])

    def _keep(self, entry: dict[str, Any]) -> dict[str, Any]:
        handle = self.next_handle
        self.next_handle += 1
        self.models[handle] = entry
        return {
            "handle": handle,
            "kind": "model",
            "owner_controller": self.controller_id,
        }

    def _result(
        self,
        task: dict[str, Any],
        rows: tuple[list[str], np.ndarray, list[str]] | None,
        model: Any,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        node = task["node_plan"]
        refs = artifacts or []
        predictions, targets = [], []
        surfaces = []
        if rows is not None:
            partition = "validation" if task["phase"] == "FIT_CV" else "final"
            if task["phase"] == "REFIT":
                partitions = {
                    block["partition"]
                    for key, block in task.get("prediction_inputs", {}).items()
                    if key.endswith(":refit")
                }
                require(
                    len(partitions) == 1 and partitions <= {"test", "final"},
                    "REFIT prediction columns require one signed native partition",
                )
                partition = next(iter(partitions))
            surfaces.append((rows, partition))
        if task["phase"] != "PREDICT" and any(
            key.endswith(":test") for key in task.get("prediction_inputs", {})
        ):
            surfaces.append((self._predictions(task, "test"), "test"))
        for surface, partition in surfaces:
            values = np.asarray(
                model.predict(self._frame(surface)), dtype=float
            ).reshape(len(surface[0]), 1)
            require(
                np.isfinite(values).all(), "Finite native meta predictions required"
            )
            predictions.append(
                {
                    "producer_node": node["node_id"],
                    "partition": partition,
                    "fold_id": task.get("fold_id"),
                    "sample_ids": surface[0],
                    "values": values.tolist(),
                    "target_names": self.target_names,
                }
            )
            if self.targets is not None:
                targets.append(
                    {
                        "level": "sample",
                        "unit_ids": [
                            {"level": "sample", "id": sample} for sample in surface[0]
                        ],
                        "values": self._targets(surface[0]).tolist(),
                        "target_names": self.target_names,
                    }
                )
        self._event(
            task["phase"],
            node=node["node_id"],
            sample_ids=rows[0] if rows is not None else [],
        )
        return {
            "node_id": node["node_id"],
            "outputs": {},
            "artifacts": refs,
            "artifact_handles": {},
            "predictions": predictions,
            "regression_targets": targets,
            "lineage": {
                "record_id": _bounded_identifier(
                    "lineage:methods-oof",
                    task["run_id"],
                    node["node_id"],
                    task["phase"],
                    task.get("variant_id") or "base",
                    task.get("fold_id") or "full",
                ),
                "run_id": task["run_id"],
                "node_id": node["node_id"],
                "phase": task["phase"],
                "controller_id": self.controller_id,
                "controller_version": "1.0.0",
                "variant_id": task.get("variant_id"),
                "fold_id": task.get("fold_id"),
                "branch_path": task.get("branch_path", []),
                "input_lineage": [],
                "artifact_refs": refs,
                "params_fingerprint": node["params_fingerprint"],
                "data_model_shape_fingerprint": None,
                "aggregation_policy_fingerprint": None,
                "seed": task["seed"],
                "unsafe_flags": [],
                "metrics": {},
                "loss_attestations": [],
                "early_stopping_records": [],
            },
        }

    def operator(self, task: dict[str, Any]) -> dict[str, Any]:
        require(not self.closed, "Meta controller is closed")
        node, phase = task["node_plan"], task["phase"]
        require(
            node["kind"] == "model"
            and node["node_id"] in self.operators
            and node["controller_id"] == self.controller_id
            and node["controller_version"] == "1.0.0",
            "Foreign meta node owner",
        )
        require(
            phase in {"FIT_CV", "REFIT", "PREDICT"}
            and task.get("prediction_inputs")
            and not task.get("data_view_receipts")
            and not task.get("required_loss_attestations")
            and not task.get("residual_targets"),
            "OOF meta controller requires native prediction inputs only",
        )
        influence = task.get("fit_influence")
        require(
            not influence
            or influence.get("mechanism") == "uniform_rows"
            and not influence.get("row_weights"),
            "Meta influence must be uniform",
        )
        recipe = _meta_recipe(self.operators[node["node_id"]], node.get("params", {}))
        if phase == "PREDICT":
            require(
                len(task["artifact_inputs"]) == 1,
                "One meta predictor artifact required",
            )
            key, binding = next(iter(task["artifact_inputs"].items()))
            handle = task["input_handles"][key]
            require(
                handle.get("owner_controller") == self.controller_id
                and handle.get("kind") == "model",
                "Foreign meta predictor handle",
            )
            entry = self.models.get(handle["handle"])
            require(
                entry is not None
                and binding["artifact"] == entry["artifact"]
                and binding["node_id"] == node["node_id"]
                and binding["controller_id"] == self.controller_id
                and binding["params_fingerprint"] == node["params_fingerprint"]
                and entry["saved"]["params_fingerprint"] == node["params_fingerprint"]
                and entry["saved"]["node_id"] == node["node_id"]
                and entry["saved"]["steps"] == recipe,
                "Meta prediction binding mismatch",
            )
            rows = self._predictions(task, "predict")
            require(
                rows[2] == entry["saved"]["feature_names"],
                "Meta predictor feature order mismatch",
            )
            return self._result(task, rows, entry["model"])
        require(self.allow_fit, "Fitting is disabled for replay")
        from n4m.roles import RolePipeline

        train = self._predictions(task, "train")
        valid = (
            self._predictions(task, "outer")
            if phase == "FIT_CV"
            else self._predictions(task, "refit")
            if any(key.endswith(":refit") for key in task["prediction_inputs"])
            else None
        )
        require(
            phase != "FIT_CV"
            or valid is not None
            and not set(train[0]).intersection(valid[0]),
            "Meta train/validation overlap",
        )
        model = RolePipeline(_role_steps(recipe))
        retained = False
        try:
            model.fit(self._frame(train), self._targets(train[0]))
            self._event(
                "fit",
                node=node["node_id"],
                sample_ids=train[0],
                feature_names=train[2],
                fold=task.get("fold_id"),
                steps=copy.deepcopy(recipe),
            )
            if phase == "FIT_CV":
                return self._result(task, valid, model)
            exported = model.export_states()
            require(
                len(exported) == 1
                and exported[0][0] == "models.regularized.ridge"
                and not exported[0][2],
                "One portable Ridge state without training rows required",
            )
            saved = {
                "schema": "dagml.methods.regression.v1",
                "node_id": node["node_id"],
                "params_fingerprint": node["params_fingerprint"],
                "target_names": self.target_names,
                "steps": recipe,
                "feature_names": train[2],
                "states": [list(exported[0][1])],
            }
            raw = json.dumps(
                saved, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
            require(
                len(raw) <= MAX_PAYLOAD
                and exported[0][1].startswith(b"N4ME")
                and len(exported[0][1]) <= MAX_STATE,
                "Bounded native meta state required",
            )
            digest = hashlib.sha256(raw).hexdigest()
            artifact_id = _bounded_identifier(
                "artifact:methods.oof",
                task["run_id"],
                node["node_id"],
                task.get("variant_id") or "base",
                "refit",
            )
            require(artifact_id not in self.artifacts, "Duplicate meta REFIT artifact")
            artifact = {
                "id": artifact_id,
                "kind": META_KIND,
                "controller_id": self.controller_id,
                "backend": "raw",
                "uri": f"artifacts/{digest}.json",
                "content_fingerprint": digest,
                "size_bytes": len(raw),
                "plugin": META_PLUGIN,
                "plugin_version": "1.0.0",
            }
            result = self._result(task, valid, model, [artifact])
            result["artifact_handles"][artifact_id] = self._keep(
                {"model": model, "saved": saved, "artifact": artifact}
            )
            self.artifacts[artifact_id] = raw
            retained = True
            return result
        finally:
            if not retained:
                model.close()
                self._event("dispose", node=node["node_id"])

    def artifact(self, message: dict[str, Any]) -> Any:
        require(not self.closed, "Meta controller is closed")
        operation = message["operation"]
        if operation == "export":
            require(message["artifact_id"] in self.artifacts, "Unknown meta artifact")
            self._event("export")
            return list(self.artifacts[message["artifact_id"]])
        if operation == "release":
            handle = message["handle"]
            require(
                handle.get("kind") == "model"
                and handle.get("owner_controller") == self.controller_id
                and handle.get("handle") in self.models,
                "Unknown or foreign meta handle",
            )
            self.models.pop(handle["handle"])["model"].close()
            self._event("dispose")
            self._event("release")
            return None
        require(operation == "hydrate", "Unsupported meta artifact operation")
        request, payload = message["request"], message["payload"]
        require(
            isinstance(payload, (bytes, bytearray, list))
            and len(payload) <= MAX_PAYLOAD
            and all(type(byte) is int and 0 <= byte <= 255 for byte in payload),
            "Bounded exact meta payload required",
        )
        raw = bytes(payload)
        artifact = request["artifact"]
        digest = hashlib.sha256(raw).hexdigest()
        require(
            request["controller_id"] == self.controller_id
            and artifact["controller_id"] == self.controller_id
            and artifact["kind"] == META_KIND
            and artifact["backend"] == "raw"
            and artifact.get("plugin") == META_PLUGIN
            and artifact.get("plugin_version") == "1.0.0"
            and artifact.get("native_predictor_descriptor") is None
            and artifact.get("native_estimator_descriptor") is None
            and artifact["uri"] == f"artifacts/{digest}.json"
            and artifact["content_fingerprint"] == digest
            and artifact["size_bytes"] == len(raw),
            "Meta RAW owner or payload binding mismatch",
        )
        saved = _strict_json(raw)
        require(
            isinstance(saved, dict)
            and set(saved)
            == {
                "schema",
                "node_id",
                "params_fingerprint",
                "target_names",
                "steps",
                "feature_names",
                "states",
            }
            and saved["schema"] == "dagml.methods.regression.v1",
            "Closed meta RAW wrapper required",
        )
        node_id = saved["node_id"]
        require(
            node_id in self.operators
            and node_id == request["node_id"]
            and saved["params_fingerprint"] == request["params_fingerprint"]
            and saved["target_names"] == self.target_names,
            "Meta node/target binding mismatch",
        )
        require(
            saved["steps"]
            == _meta_recipe(self.operators[node_id], self.node_params.get(node_id, {})),
            "Saved meta recipe differs from signed selected parameters",
        )
        require(
            saved["feature_names"]
            == [f"{producer}/0" for producer in self._input_nodes(node_id)],
            "Saved meta feature identities differ from signed topology",
        )
        states = saved["states"]
        require(
            isinstance(states, list)
            and len(states) == 1
            and isinstance(states[0], list)
            and 4 <= len(states[0]) <= MAX_STATE
            and all(type(byte) is int and 0 <= byte <= 255 for byte in states[0])
            and bytes(states[0][:4]) == b"N4ME",
            "Bounded native N4ME meta state required",
        )
        from n4m.roles import RolePipeline

        model = RolePipeline.from_states(
            _role_steps(saved["steps"]),
            [bytes(states[0])],
            feature_names=saved["feature_names"],
        )
        try:
            handle = self._keep(
                {"model": model, "saved": saved, "artifact": copy.deepcopy(artifact)}
            )
        except BaseException:
            model.close()
            raise
        self._event("hydrate", node=node_id)
        return handle

    def close(self) -> None:
        if self.closed:
            return
        error: BaseException | None = None
        try:
            for entry in self.models.values():
                try:
                    entry["model"].close()
                    self._event("dispose")
                except BaseException as problem:
                    error = error or problem
        finally:
            self.models.clear()
            self.artifacts.clear()
            self.closed = True
        if error is not None:
            raise error


class MethodsTopologyController:
    """Dispatch raw and OOF nodes/handles to their distinct signed owners."""

    def __init__(
        self,
        *,
        operators: dict[str, Any],
        sources: dict[str, Any],
        targets: dict[str, Any] | None = None,
        target_names: tuple[str, ...] = ("y",),
        allow_fit: bool = True,
        source_ids: tuple[str, ...] = SOURCE_ORDER,
        node_params: dict[str, Any] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> None:
        raw = {
            node: operator
            for node, operator in operators.items()
            if operator.get("type") == "N4mMultimodalPipeline"
        }
        meta = {
            node: operator
            for node, operator in operators.items()
            if operator.get("type") == "N4mRolePipeline"
        }
        require(
            raw and len(raw) + len(meta) == len(operators),
            "Topology contains unsupported native operators",
        )
        expected_params = {} if node_params is None else node_params
        require(
            isinstance(expected_params, dict)
            and set(expected_params) <= set(operators),
            "Foreign expected topology parameter node",
        )
        raw_params = {
            node: params for node, params in expected_params.items() if node in raw
        }
        meta_params = {
            node: params for node, params in expected_params.items() if node in meta
        }
        self.raw = MethodsMultimodalController(
            operators=raw,
            sources=sources,
            targets=targets,
            target_names=target_names,
            allow_fit=allow_fit,
            source_ids=source_ids,
            node_params=raw_params,
        )
        try:
            self.meta = (
                MethodsOOFController(
                    operators=meta,
                    raw_operators=raw,
                    edges=edges,
                    targets=targets,
                    target_names=target_names,
                    allow_fit=allow_fit,
                    node_params=meta_params,
                )
                if meta
                else None
            )
        except BaseException:
            self.raw.close()
            raise
        self.manifest = self.raw.manifest
        self.manifests = (
            [self.manifest, self.meta.manifest]
            if self.meta is not None
            else [self.manifest]
        )
        self.closed = False

    @property
    def audit(self) -> list[dict[str, Any]]:
        return [*self.raw.audit, *(self.meta.audit if self.meta is not None else [])]

    def operator(self, task: dict[str, Any]) -> dict[str, Any]:
        require(not self.closed, "Topology controller is closed")
        owner = task["node_plan"]["controller_id"]
        if owner == self.raw.controller_id:
            return self.raw.operator(task)
        require(
            self.meta is not None and owner == self.meta.controller_id,
            "Foreign topology node owner",
        )
        return self.meta.operator(task)

    def artifact(self, message: dict[str, Any]) -> Any:
        require(not self.closed, "Topology controller is closed")
        operation = message["operation"]
        if operation == "export":
            artifact_id = message["artifact_id"]
            in_raw = artifact_id in self.raw.artifacts
            in_meta = self.meta is not None and artifact_id in self.meta.artifacts
            require(in_raw != in_meta, "Unknown or colliding topology artifact")
            controller = self.raw if in_raw else self.meta
        elif operation == "release":
            owner = message["handle"].get("owner_controller")
            controller = self.raw if owner == self.raw.controller_id else self.meta
            require(
                controller is not None and controller.controller_id == owner,
                "Foreign topology handle owner",
            )
        else:
            require(operation == "hydrate", "Unsupported topology artifact operation")
            owner = message["request"]["controller_id"]
            controller = self.raw if owner == self.raw.controller_id else self.meta
            require(
                controller is not None and controller.controller_id == owner,
                "Foreign topology artifact owner",
            )
        assert controller is not None
        return controller.artifact(message)

    def close(self) -> None:
        if self.closed:
            return
        try:
            try:
                self.raw.close()
            finally:
                if self.meta is not None:
                    self.meta.close()
        finally:
            self.closed = True
