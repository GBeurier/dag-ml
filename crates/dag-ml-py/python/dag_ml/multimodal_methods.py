"""Methods-owned fixed-shape U07 controller; DAG-ML owns the campaign."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise ImportError(
        "Python Methods multimodal execution requires dag-ml[multimodal] "
        "(nirs4all-methods, NumPy and scikit-learn)"
    ) from error

SCHEMA = "dagml.methods.multimodal.v1"
KIND = "methods_multimodal_pipeline"
SOURCE_ORDER = ("nir", "image", "series", "metadata")
MAX_PAYLOAD = 134_217_728
MAX_STATE = 67_108_864
PARAMETERS = {"model__alpha", "source_weights__image", "transformers__image__n_components"}
DECLARATIONS = {"recipe", "source_schemas"}


def _methods_pipeline() -> Any:
    """Require the optional published host facade before numerical execution."""
    try:
        from n4m import MultimodalPipeline
    except ImportError as error:
        raise ImportError(
            "Python Methods multimodal execution requires dag-ml[multimodal] "
            "(nirs4all-methods, NumPy and scikit-learn)"
        ) from error
    return MultimodalPipeline


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bounded_identifier(prefix: str, *coordinates: str) -> str:
    """Keep short identities and hash complete coordinates at the native limit."""
    parts = [prefix, *coordinates]
    identifier = ":".join(parts)
    if len(identifier.encode("utf-8")) <= 128:
        return identifier
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


def manifest_for_host(host: str = "python") -> dict[str, Any]:
    """The independently trusted raw-input contract; no numerical defaults."""
    require(host in {"python", "wasm", "r", "octave"}, "Unknown Methods multimodal host")
    return {
        "controller_id": f"controller:methods.{host}.multimodal", "controller_version": "1.0.0",
        "operator_kind": "model", "priority": 0, "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [{"name": "x", "kind": "data", "representation": "feature_block_set", "cardinality": "one", "description": ""}],
        "output_ports": [{"name": "y_hat", "kind": "prediction", "representation": None, "cardinality": "one", "description": ""},
                         {"name": "model", "kind": "artifact", "representation": None, "cardinality": "one", "description": ""}],
        "data_requirements": {"schema_version": 1, "default_fusion": {
            "mode": "dict_by_source", "alignment": "sample_id", "adapter_id": None, "params": {},
        }, "metadata": {}, "ports": [
            {"name": "x", "accepted_representations": ["feature_block_set"], "accepted_types": ["multi_block"],
             "rank": None, "multi_source": True, "optional": False, "metadata": {}}]},
        "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions", "emits_artifacts", "stateful", "uses_core_rng"],
        "fit_scope": "fold_train", "rng_policy": "externally_deterministic", "artifact_policy": "serializable",
    }


def recipe_for_node(operator: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Apply only declared parameter paths; Methods validates all numerics."""
    require(set(operator) == {"type", "recipe", "source_schemas"} and operator["type"] == "N4mMultimodalPipeline", "Expected explicit native multimodal operator")
    require(isinstance(params, dict) and set(params) <= PARAMETERS | DECLARATIONS, "Unsupported Methods multimodal effective parameters")
    declarations = set(params) & DECLARATIONS
    require(not declarations or declarations == DECLARATIONS, "Both immutable multimodal declarations are required")
    if declarations:
        require(set(params) <= DECLARATIONS | {"model__alpha"}, "Structural multimodal tuning permits alpha only")
        require(all(json.dumps(params[key], sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(operator[key], sort_keys=True, ensure_ascii=False, allow_nan=False) for key in DECLARATIONS), "Immutable multimodal declarations differ from the signed graph operator")
    recipe = copy.deepcopy(operator["recipe"])
    if "model__alpha" in params:
        recipe["model"]["params"]["alpha"] = params["model__alpha"]
    if "source_weights__image" in params:
        require("image" in recipe["source_order"], "Methods multimodal image weight is inactive")
        recipe["source_weights"]["image"] = params["source_weights__image"]
    if "transformers__image__n_components" in params:
        require("image" in recipe["source_order"], "Methods multimodal image PCA is inactive")
        recipe["encoders"]["image"]["n_components"] = params["transformers__image__n_components"]
    return recipe


def _strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in items:
            require(key not in value, "Duplicate multimodal JSON field")
            value[key] = child
        return value
    def constant(value: str) -> None:
        raise ValueError("Non-finite multimodal JSON: " + value)
    require(0 < len(raw) <= MAX_PAYLOAD, "Methods multimodal payload budget exceeded")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def validate_wrapper(saved: Any) -> None:
    """Bound transport before native import; Methods validates learned state."""
    require(isinstance(saved, dict) and set(saved) == {"schema", "node_id", "params_fingerprint", "target_names", "recipe", "source_schemas", "state"}, "Closed Methods multimodal wrapper required")
    require(saved["schema"] == SCHEMA, "Unknown Methods multimodal schema")
    require(isinstance(saved["target_names"], list) and len(saved["target_names"]) == 1 and isinstance(saved["target_names"][0], str) and 0 < len(saved["target_names"][0].encode()) <= 4096, "One named target is required")
    recipe, schemas = saved["recipe"], saved["source_schemas"]
    validate_recipe(recipe, schemas)
    require(isinstance(schemas, dict) and set(schemas) == set(SOURCE_ORDER), "Complete raw source schemas required")
    for schema in schemas.values():
        require(isinstance(schema, dict) and set(schema) == {"representation_id", "input_shape", "dtype", "identity"}, "Closed raw source schema required")
        require(isinstance(schema["identity"], str) and 0 < len(schema["identity"].encode()) <= 1_048_576, "Source identity budget exceeded")
        _strict_json(schema["identity"].encode())
    state = saved["state"]
    require(isinstance(state, list) and 28 <= len(state) <= MAX_STATE and all(type(byte) is int and 0 <= byte <= 255 for byte in state), "Bounded exact N4MF bytes required")
    header = bytes(state[:12])
    require(header[:4] == b"N4MF" and header[4:8] == (1).to_bytes(4, "little") and header[8:12] == (2).to_bytes(4, "little"), "Unsupported complete native state format")


def validate_recipe(recipe: Any, schemas: Any) -> None:
    """Closed declarative profile; all arithmetic stays in Methods."""
    require(isinstance(recipe, dict) and set(recipe) == {"schema_version", "fusion", "source_order", "encoders", "source_weights", "model"} and type(recipe["schema_version"]) is int and recipe["schema_version"] == 1 and recipe["fusion"] == "early", "Closed early-fusion recipe required")
    order = recipe["source_order"]
    require(isinstance(order, list) and 1 <= len(order) <= len(SOURCE_ORDER) and all(isinstance(name, str) and name in SOURCE_ORDER for name in order) and len(set(order)) == len(order), "Distinct ordered U07 source subset required")
    require(isinstance(schemas, dict) and set(schemas) == set(SOURCE_ORDER) and isinstance(recipe["encoders"], dict) and set(recipe["encoders"]) == set(order) and isinstance(recipe["source_weights"], dict) and set(recipe["source_weights"]) == set(order), "Selected encoder/weight recipe and complete raw source schemas required")
    for name, representation in zip(SOURCE_ORDER, ("signal_1d", "rgb_image", "series_mv", "tabular_mixed"), strict=True):
        schema = schemas[name]
        require(isinstance(schema, dict) and set(schema) == {"representation_id", "input_shape", "dtype", "identity"} and schema["representation_id"] == representation, "Closed raw source descriptor required")
        shape = schema["input_shape"]
        require(isinstance(shape, list) and 0 < len(shape) <= 7 and all(type(size) is int and 0 < size <= 2**63-1 for size in shape) and math.prod(shape) <= 1_048_576 and (name != "metadata" or shape == [2]), "Fixed positive source shape required")
        require(isinstance(schema["dtype"], str) and 0 < len(schema["dtype"].encode()) <= 128 and isinstance(schema["identity"], str) and 0 < len(schema["identity"].encode()) <= 1_048_576, "Source identity budget exceeded")
        _strict_json(schema["identity"].encode())
    if "nir" in order:
        nir = recipe["encoders"]["nir"]
        require(nir == {"kind":"standard_scaler", "with_mean":True, "with_std":True} and nir["with_mean"] is True and nir["with_std"] is True, "Closed scaler encoder pattern required")
    if "metadata" in order:
        metadata = recipe["encoders"]["metadata"]
        require(metadata == {"kind":"column_transformer", "numeric_columns":[0], "categorical_columns":[1], "with_mean":True, "with_std":True, "handle_unknown":"ignore", "sparse_output":False, "drop":None} and metadata["with_mean"] is True and metadata["with_std"] is True and metadata["sparse_output"] is False and type(metadata["numeric_columns"][0]) is int and type(metadata["categorical_columns"][0]) is int, "Closed mixed encoder pattern required")
    for name in ("image", "series"):
        if name not in order:
            continue
        encoder = recipe["encoders"][name]
        require(isinstance(encoder, dict) and set(encoder) == {"kind", "n_components", "whiten", "random_state"} and encoder["kind"] == "tensor_pca" and type(encoder["n_components"]) is int and 0 < encoder["n_components"] <= min(2**31-1, math.prod(schemas[name]["input_shape"])) and encoder["whiten"] is False and type(encoder["random_state"]) is int and 0 <= encoder["random_state"] <= 2**32-1, "Declared positive unwhitened PCA recipe required")
    require(all(type(value) in (int,float) and np.isfinite(value) and value >= 0 for value in recipe["source_weights"].values()), "Finite nonnegative source weights required")
    model = recipe["model"]
    require(isinstance(model, dict) and set(model) == {"method_id", "params"} and model["method_id"] == "models.regularized.ridge" and isinstance(model["params"], dict) and set(model["params"]) == {"alpha", "center_x", "center_y", "scale_x"}, "Closed native Ridge recipe required")
    params = model["params"]
    require(type(params["alpha"]) in (int,float) and np.isfinite(params["alpha"]) and params["alpha"] >= 0 and params["center_x"] is True and params["center_y"] is True and params["scale_x"] is False, "Native Ridge scaling differs from declared profile")


class MethodsMultimodalController:
    """One invocation-local Methods predictor behind standard native callbacks."""

    def __init__(self, *, operators: dict[str, Any], sources: dict[str, Any], targets: dict[str, Any] | None = None,
                 target_names: tuple[str, ...] = ("y",), allow_fit: bool = True, controller_id: str = "controller:methods.python.multimodal", source_ids: tuple[str, ...] = SOURCE_ORDER, node_params: dict[str, dict[str, Any]] | None = None) -> None:
        owners = {f"controller:methods.{host}.multimodal": host for host in ("python", "wasm", "r", "octave")}
        require(controller_id in owners and (not allow_fit or owners[controller_id] == "python"), "Exact closed multimodal producer owner required; fitting requires Python ownership")
        require(type(allow_fit) is bool and allow_fit == (targets is not None), "Replay must not receive fitting targets")
        require(set(sources) == set(SOURCE_ORDER) and bool(operators), "Complete canonical raw sources/operators required")
        require(len(target_names) == 1 and bool(target_names[0]), "One target required")
        self.controller_id, self.plugin = controller_id, f"dagml.methods.{owners[controller_id]}.multimodal"
        self.execution_host = "python"
        self.manifest = manifest_for_host(owners[controller_id])
        self.operators = copy.deepcopy(operators)
        self.node_params = copy.deepcopy(node_params or {})
        self.sources, self.targets = sources, targets
        self.target_names, self.allow_fit = list(target_names), allow_fit
        require(len(source_ids) == 4 and len(set(source_ids)) == 4 and all(isinstance(name, str) and name for name in source_ids), "Four exact native source IDs required")
        self.source_ids = list(source_ids)
        self.audit: list[dict[str, Any]] = []
        self.models: dict[int, dict[str, Any]] = {}
        self.artifacts: dict[str, bytes] = {}
        self.next_handle, self.closed = 1, False
        for name, source in sources.items():
            ids = source["sample_ids"]
            require(isinstance(ids, list) and len(ids) == len(set(ids)) and all(isinstance(sample, str) and sample for sample in ids), "Source sample IDs must be unique strings")
            require(len(source["values"]) == len(ids), "Source row/ID mismatch")
            values = np.asarray(source["values"])
            require(values.ndim >= 2 and values.size <= 16_777_216 and list(values.shape[1:]) == source["descriptor"]["input_shape"], "Current raw source shape or value budget mismatch")
            require(name == "metadata" or values.dtype in (np.dtype("float32"), np.dtype("float64")), "Numeric raw sources require float32/float64 arrays")
            if name == "metadata":
                require(values.ndim == 2 and values.shape[1] == 2, "Complete two-column raw metadata is required")
                require(np.isfinite(np.asarray(values[:, 0], dtype=float)).all() and all(isinstance(value, str) for value in values[:, 1]), "Complete finite numeric/raw-string metadata is required")
            else:
                require(np.isfinite(values).all(), "Complete finite raw sources are required, including excluded modalities")
        for operator in self.operators.values():
            recipe_for_node(operator, {})
            validate_recipe(operator["recipe"], operator["source_schemas"])
            require(operator["source_schemas"] == {name: sources[name]["descriptor"] for name in SOURCE_ORDER}, "Current raw source schemas differ from signed operator")
        require(set(self.node_params) <= set(self.operators), "Foreign expected parameter node")
        for node, params in self.node_params.items():
            validate_recipe(recipe_for_node(self.operators[node], params), self.operators[node]["source_schemas"])
        if targets is not None:
            require(len(targets["sample_ids"]) == len(set(targets["sample_ids"])) and len(targets["values"]) == len(targets["sample_ids"]), "Target sample IDs must be unique and complete")

    def _event(self, operation: str, **fields: Any) -> None:
        self.audit.append({"operation": operation, **fields})

    def _selected_schemas(self, node_id: str) -> dict[str, Any]:
        operator = self.operators[node_id]
        return {name: operator["source_schemas"][name] for name in operator["recipe"]["source_order"]}

    def _features(self, task: dict[str, Any], partition: str) -> tuple[list[str], dict[str, Any]]:
        views = [view for view in task.get("data_views", {}).values() if view["partition"] == partition]
        require(len(views) == 1, "One native raw multimodal view is required")
        view = views[0]
        ids = view["sample_ids"]
        require(isinstance(ids, list) and ids and len(ids) == len(set(ids)), "Native sample IDs must be complete and unique")
        require(view.get("source_ids") == self.source_ids, "Native raw source order mismatch")
        require(not view.get("include_augmented") and (partition in {"fold_validation", "predict"} or not view.get("include_excluded")) and not view.get("columns"), "Augmented/excluded fitting or column-selected raw inputs are unsupported")
        order = self.operators[task["node_plan"]["node_id"]]["recipe"]["source_order"]
        blocks = {}
        for name in SOURCE_ORDER:
            source = self.sources[name]
            positions = {sample: index for index, sample in enumerate(source["sample_ids"])}
            require(all(sample in positions for sample in ids), "Unknown source sample IDs")
            values = np.asarray(source["values"])
            require(list(values.shape[1:]) == source["descriptor"]["input_shape"], "Current raw source shape mismatch")
            if name in order:
                blocks[name] = values[[positions[sample] for sample in ids]]
        return ids, {name: blocks[name] for name in order}

    def _targets(self, ids: list[str]) -> np.ndarray:
        require(self.targets is not None, "Replay cannot read training targets")
        positions = {sample: index for index, sample in enumerate(self.targets["sample_ids"])}
        require(all(sample in positions for sample in ids), "Unknown target sample IDs")
        values = np.asarray(self.targets["values"], dtype=float)
        return values[[positions[sample] for sample in ids]].reshape(len(ids), 1)

    def _keep(self, entry: dict[str, Any]) -> dict[str, Any]:
        handle = self.next_handle
        self.next_handle += 1
        self.models[handle] = entry
        return {"handle": handle, "kind": "model", "owner_controller": self.controller_id}

    def _result(self, task: dict[str, Any], ids: list[str], blocks: dict[str, Any], model: Any,
                artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        node = task["node_plan"]
        refs = artifacts or []
        predicted = np.asarray(model.predict(blocks, source_schemas=self._selected_schemas(node["node_id"])), dtype=float).reshape(len(ids), 1)
        require(np.isfinite(predicted).all(), "Native multimodal predictions must be finite")
        result = {"node_id": node["node_id"], "outputs": {}, "artifacts": refs, "artifact_handles": {},
                  "predictions": [{"producer_node": node["node_id"], "partition": "validation" if task["phase"] == "FIT_CV" else "final",
                                   "fold_id": task.get("fold_id"), "sample_ids": ids, "values": predicted.tolist(), "target_names": self.target_names}],
                  "lineage": {"record_id": _bounded_identifier("lineage:methods-multimodal", task["run_id"], node["node_id"], task["phase"], task.get("variant_id") or "base", task.get("fold_id") or "full"),
                              "run_id": task["run_id"], "node_id": node["node_id"], "phase": task["phase"], "controller_id": self.controller_id,
                              "controller_version": "1.0.0", "variant_id": task.get("variant_id"), "fold_id": task.get("fold_id"), "branch_path": task.get("branch_path", []),
                              "input_lineage": [], "artifact_refs": refs, "params_fingerprint": node["params_fingerprint"], "data_model_shape_fingerprint": None,
                              "aggregation_policy_fingerprint": None, "seed": task["seed"], "unsafe_flags": [], "metrics": {}, "loss_attestations": [], "early_stopping_records": []}}
        if self.targets is not None:
            result["regression_targets"] = [{"level": "sample", "unit_ids": [{"level": "sample", "id": sample} for sample in ids], "values": self._targets(ids).tolist(), "target_names": self.target_names}]
        if task["phase"] in {"FIT_CV", "REFIT"} and any(view["partition"] == "predict" for view in task.get("data_views", {}).values()):
            test_ids, test_blocks = self._features(task, "predict")
            test_values = np.asarray(model.predict(test_blocks, source_schemas=self._selected_schemas(node["node_id"])), dtype=float).reshape(len(test_ids), 1)
            require(np.isfinite(test_values).all(), "Native test predictions must be finite")
            result["predictions"].append({"producer_node": node["node_id"], "partition": "test", "fold_id": task.get("fold_id"), "sample_ids": test_ids, "values": test_values.tolist(), "target_names": self.target_names})
            if self.targets is not None:
                result["regression_targets"].append({"level": "sample", "unit_ids": [{"level": "sample", "id": sample} for sample in test_ids], "values": self._targets(test_ids).tolist(), "target_names": self.target_names})
        self._event(task["phase"], node=node["node_id"], sample_ids=list(ids))
        return result

    def operator(self, task: dict[str, Any]) -> dict[str, Any]:
        require(not self.closed, "Methods multimodal controller is closed")
        node, phase = task["node_plan"], task["phase"]
        require(node["kind"] == "model" and node["controller_id"] == self.controller_id and node["controller_version"] == "1.0.0" and node["node_id"] in self.operators, "Foreign Methods multimodal node owner")
        require(phase in {"FIT_CV", "REFIT", "PREDICT"} and not task.get("prediction_inputs") and not task.get("data_view_receipts") and not task.get("required_loss_attestations") and not task.get("residual_targets"), "Unsupported multimodal phase or generated/OOF/residual/loss inputs")
        influence = task.get("fit_influence")
        require(not influence or influence.get("mechanism") == "uniform_rows" and not influence.get("row_weights"), "Nonuniform influence is outside U07")
        operator = self.operators[node["node_id"]]
        recipe = recipe_for_node(operator, node.get("params", {}))
        validate_recipe(recipe, operator["source_schemas"])
        if phase == "PREDICT":
            require(len(task["artifact_inputs"]) == 1, "PREDICT needs one attested complete predictor")
            key, artifact_input = next(iter(task["artifact_inputs"].items()))
            handle = task["input_handles"][key]
            require(handle.get("owner_controller") == self.controller_id and handle.get("kind") == "model", "Foreign predictor handle")
            entry = self.models.get(handle["handle"])
            require(entry is not None and entry["saved"]["node_id"] == node["node_id"] and entry["saved"]["params_fingerprint"] == node["params_fingerprint"] and entry["saved"]["recipe"] == recipe and entry["saved"]["source_schemas"] == operator["source_schemas"] and artifact_input["artifact"] == entry["artifact"] and artifact_input["params_fingerprint"] == node["params_fingerprint"] and artifact_input["controller_id"] == self.controller_id and artifact_input["node_id"] == node["node_id"], "PREDICT recipe, schema or artifact binding mismatch")
            ids, blocks = self._features(task, "predict")
            return self._result(task, ids, blocks, entry["model"])
        require(self.allow_fit, "Fitting is disabled for replay")
        MultimodalPipeline = _methods_pipeline()
        ids, blocks = self._features(task, "fold_train" if phase == "FIT_CV" else "full_train")
        valid_ids, valid = self._features(task, "fold_validation") if phase == "FIT_CV" else (ids, blocks)
        require(phase != "FIT_CV" or not set(ids).intersection(valid_ids), "Training/validation overlap")
        model = MultimodalPipeline(recipe, self._selected_schemas(node["node_id"]))
        retained = False
        try:
            model.fit(blocks, self._targets(ids))
            self._event("fit", node=node["node_id"], fold=task.get("fold_id"), sample_ids=list(ids), raw_shapes={name: list(value.shape) for name, value in blocks.items()}, source_order=list(recipe["source_order"]), source_weights=copy.deepcopy(recipe["source_weights"]), recipe=copy.deepcopy(recipe))
            if phase == "FIT_CV":
                return self._result(task, valid_ids, valid, model)
            saved = {"schema": SCHEMA, "node_id": node["node_id"], "params_fingerprint": node["params_fingerprint"], "target_names": self.target_names,
                     "recipe": recipe, "source_schemas": operator["source_schemas"], "state": list(model.export_state())}
            validate_wrapper(saved)
            payload = json.dumps(saved, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            require(len(payload) <= MAX_PAYLOAD, "Complete predictor payload budget exceeded")
            digest = hashlib.sha256(payload).hexdigest()
            artifact_id = _bounded_identifier("artifact:methods.multimodal", task["run_id"], node["node_id"], task.get("variant_id") or "base", "refit")
            require(artifact_id not in self.artifacts, "Duplicate complete REFIT predictor")
            artifact = {"id": artifact_id, "kind": KIND, "controller_id": self.controller_id, "backend": "raw", "uri": f"artifacts/{digest}.json",
                        "content_fingerprint": digest, "size_bytes": len(payload), "plugin": self.plugin, "plugin_version": "1.0.0"}
            result = self._result(task, valid_ids, valid, model, [artifact])
            handle = self._keep({"model": model, "saved": saved, "artifact": artifact})
            self.artifacts[artifact_id] = payload
            result["artifact_handles"][artifact_id] = handle
            retained = True
            return result
        finally:
            if not retained:
                model.close()
                self._event("dispose", node=node["node_id"])

    def artifact(self, message: dict[str, Any]) -> Any:
        require(not self.closed, "Methods multimodal controller is closed")
        operation = message["operation"]
        if operation == "export":
            require(message["artifact_id"] in self.artifacts, "Unknown complete predictor")
            self._event("export")
            return list(self.artifacts[message["artifact_id"]])
        if operation == "release":
            handle = message["handle"]
            require(handle.get("kind") == "model" and handle.get("owner_controller") == self.controller_id and handle.get("handle") in self.models, "Unknown or foreign predictor handle")
            entry = self.models.pop(handle["handle"])
            entry["model"].close()
            self._event("dispose")
            self._event("release")
            return None
        require(operation == "hydrate", "Unsupported complete predictor artifact operation")
        request = message["request"]
        values = message["payload"]
        require(isinstance(values, (bytes, bytearray, list)) and len(values) <= MAX_PAYLOAD and all(type(value) is int and 0 <= value <= 255 for value in values), "Exact bounded raw payload required")
        raw = bytes(values)
        artifact = request["artifact"]
        digest = hashlib.sha256(raw).hexdigest()
        require(request["controller_id"] == self.controller_id and artifact["controller_id"] == self.controller_id and artifact["kind"] == KIND and artifact["backend"] == "raw" and artifact.get("plugin") == self.plugin and artifact.get("plugin_version") == "1.0.0" and artifact.get("native_predictor_descriptor") is None and artifact.get("native_estimator_descriptor") is None and artifact["uri"] == f"artifacts/{digest}.json" and artifact["content_fingerprint"] == digest and artifact["size_bytes"] == len(raw), "Complete predictor RAW owner, plugin, hash or size mismatch")
        saved = _strict_json(raw)
        validate_wrapper(saved)
        validate_recipe(saved["recipe"], saved["source_schemas"])
        require(saved["node_id"] == request["node_id"] and saved["params_fingerprint"] == request["params_fingerprint"] and saved["target_names"] == self.target_names and saved["node_id"] in self.operators and saved["source_schemas"] == self.operators[saved["node_id"]]["source_schemas"], "Complete predictor node, target or source schema mismatch")
        require(saved["recipe"] == recipe_for_node(self.operators[saved["node_id"]], self.node_params.get(saved["node_id"], {})), "Complete predictor selected recipe mismatch before hydration")
        MultimodalPipeline = _methods_pipeline()
        model = MultimodalPipeline.from_state(bytes(saved["state"]), recipe=saved["recipe"], source_schemas=self._selected_schemas(saved["node_id"]))
        try:
            handle = self._keep({"model": model, "saved": saved, "artifact": copy.deepcopy(artifact)})
        except BaseException:
            model.close()
            raise
        self._event("hydrate")
        return handle

    def close(self) -> None:
        if self.closed:
            return
        failure: BaseException | None = None
        try:
            for entry in self.models.values():
                try:
                    entry["model"].close()
                    self._event("dispose")
                except BaseException as problem:
                    if failure is None:
                        failure = problem
        finally:
            self.models.clear()
            self.artifacts.clear()
            self.closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> MethodsMultimodalController:
        require(not self.closed, "Methods multimodal controller is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
