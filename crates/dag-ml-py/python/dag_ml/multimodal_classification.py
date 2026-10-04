"""Native four-source PLS-logistic classification; scheduling and scoring remain Rust-owned."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Self, cast

import numpy as np

from .multimodal_methods import (
    MAX_PAYLOAD,
    MAX_STATE,
    SOURCE_ORDER,
    _bounded_identifier,
    _strict_json,
    require,
)

METHOD = "models.classification.pls_logistic"
SCHEMA = "dagml.methods.multimodal.classification.v1"
KIND = "methods_multimodal_classifier_pipeline"
RAW_CONTROLLER = "controller:methods.python.multimodal.classification"
META_CONTROLLER = "controller:methods.python.classification"
META_PLUGIN = "dagml.methods.python.classification"
META_KIND = "methods_role_classifier_pipeline"
PARAMETERS = {"model__n_components", "model__max_iter"}
DECLARATIONS = {"recipe", "source_schemas", "classification"}


def validate_classification(value):
    require(isinstance(value, dict) and set(value) == {"schema_version", "class_labels", "label_names"} and type(value["schema_version"]) is int and value["schema_version"] == 1, "Closed classification vocabulary required")
    labels, names = value["class_labels"], value["label_names"]
    require(isinstance(labels, list) and 2 <= len(labels) <= 65_536 and all(type(label) is int for label in labels) and labels == list(range(len(labels))) and isinstance(names, list) and len(names) == len(labels), "Complete contiguous native class order required")
    strings = all(type(name) is str and len(name.encode()) <= 1_048_576 for name in names)
    integers = all(type(name) is int and -(1 << 63) <= name < (1 << 63) for name in names)
    require((strings or integers) and names == sorted(names) and len(set(names)) == len(names), "Sorted unique homogeneous string or int64 class names required")


def _classifier_params(params):
    require(isinstance(params, dict) and set(params) == {"n_components", "max_iter"} and all(type(value) is int and 0 < value <= 2**31-1 for value in params.values()), "Positive integer PLS-logistic parameters required")


def _classifier_working_set(classification, params, rows=0):
    _classifier_params(params)
    classes = len(classification["class_labels"])
    dimension = params["n_components"] + 1
    hessian_width = (classes - 1) * dimension
    require(max(rows * classes, rows * dimension, hessian_width * hessian_width) <= 16_777_216,
            "PLS-logistic working set exceeds the closed 16777216-element matrix limit")


def _checked_target_values(values, classification):
    array = np.asarray(values)
    require(array.ndim in (1, 2) and (array.ndim == 1 or array.shape[1] == 1) and array.dtype.kind in "iu" and array.size > 0 and np.all(array >= 0) and np.all(array < len(classification["class_labels"])), "Complete integral target IDs inside the signed vocabulary required")
    return array.astype(np.int64, copy=False)


def _checked_classes(model, classification):
    require(np.asarray(model.classes_).tolist() == classification["class_labels"], "Native class order differs from signed train vocabulary")


def manifest_for_host(host="python"):
    require(host == "python", "Classification currently requires the native Python host")
    from .multimodal_methods import manifest_for_host as raw_manifest
    manifest = raw_manifest(host)
    manifest["controller_id"] = RAW_CONTROLLER
    manifest["output_ports"].insert(1, {"name":"probabilities", "kind":"prediction", "representation":None, "cardinality":"one", "description":"Signed ordered class distributions"})
    return manifest


def recipe_for_node(operator, params):
    require(isinstance(operator, dict) and set(operator) == {"type", "recipe", "source_schemas", "classification"} and operator["type"] == "N4mMultimodalClassifierPipeline", "Explicit raw classifier operator required")
    validate_classification(operator["classification"])
    require(isinstance(params, dict) and set(params) <= PARAMETERS | DECLARATIONS, "Unsupported raw classifier effective parameters")
    for key in set(params) & DECLARATIONS:
        require(params[key] == operator[key], "Immutable classifier declarations differ from signed graph")
    recipe = copy.deepcopy(operator["recipe"])
    for key in set(params) & PARAMETERS:
        recipe["model"]["params"][key.removeprefix("model__")] = params[key]
    _classifier_working_set(operator["classification"], recipe["model"]["params"])
    return recipe


def validate_wrapper(saved):
    require(isinstance(saved, dict) and set(saved) == {"schema", "node_id", "params_fingerprint", "target_names", "recipe", "source_schemas", "classification", "state"} and saved["schema"] == SCHEMA and saved["target_names"] == ["y"], "Closed raw classification wrapper required")
    validate_classification(saved["classification"])
    validate_recipe(saved["recipe"], saved["source_schemas"])
    state = saved["state"]
    require(isinstance(state, list) and 28 <= len(state) <= MAX_STATE and all(type(byte) is int and 0 <= byte <= 255 for byte in state) and bytes(state[:20]) == b"N4MC" + (1).to_bytes(4,"little") + (2).to_bytes(4,"little") + (17).to_bytes(4,"little") + (0).to_bytes(4,"little"), "Bounded N4MC classifier state required")


def _classification_result(owner, task, surfaces, model, classification, refs, adapt):
    _checked_classes(model, classification)
    node = task["node_plan"]
    predictions, evidence, targets = [], [], []
    for ids, value, partition in surfaces:
        require(len(ids) * len(classification["class_labels"]) <= 16_777_216, "Native classification probability buffer exceeds the closed matrix budget")
        matrix, kwargs = adapt(value)
        labels = np.asarray(model.predict(matrix, **kwargs))
        probabilities = np.asarray(model.predict_proba(matrix, **kwargs), dtype=float)
        owner._event("predict", node=node["node_id"], fold=task.get("fold_id"), sample_ids=list(ids), partition=partition)
        count = len(classification["class_labels"])
        require(labels.shape == (len(ids),) and labels.dtype.kind in "iu" and probabilities.shape == (len(ids), count) and np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)) and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6), "Complete native labels and class distributions required")
        require(np.array_equal(labels, probabilities.argmax(axis=1)), "Native labels must match their probability argmax")
        common = {"producer_node":node["node_id"], "partition":partition, "fold_id":task.get("fold_id"), "sample_ids":list(ids)}
        predictions.extend([{**common, "producer_port":"y_hat", "values":labels.reshape(-1,1).tolist(), "target_names":["y"]}, {**common, "producer_port":"probabilities", "values":probabilities.tolist(), "target_names":[f"class:{i}" for i in classification["class_labels"]]}])
        if partition in {"validation", "test"} and task["phase"] != "PREDICT":
            evidence.append({**common, "producer_port":"y_hat", "class_labels":classification["class_labels"], "values":probabilities.tolist()})
        if owner.targets is not None:
            targets.append({"level":"sample", "unit_ids":[{"level":"sample", "id":sample} for sample in ids], "values":owner._targets(ids).reshape(-1,1).tolist(), "target_names":["y"]})
    owner._event(task["phase"], node=node["node_id"], sample_ids=[sample for ids, _, _ in surfaces for sample in ids])
    return {"node_id":node["node_id"], "outputs":{}, "artifacts":refs, "artifact_handles":{}, "predictions":predictions, "classification_probabilities":evidence, "regression_targets":targets,
      "lineage":{"record_id":_bounded_identifier("lineage:methods-classification", task["run_id"],node["node_id"],task["phase"],task.get("variant_id") or "base",task.get("fold_id") or "full"), "run_id":task["run_id"], "node_id":node["node_id"], "phase":task["phase"], "controller_id":owner.controller_id, "controller_version":"1.0.0", "variant_id":task.get("variant_id"), "fold_id":task.get("fold_id"), "branch_path":task.get("branch_path",[]), "input_lineage":[], "artifact_refs":refs, "params_fingerprint":node["params_fingerprint"], "data_model_shape_fingerprint":None, "aggregation_policy_fingerprint":None, "seed":task["seed"], "unsafe_flags":[], "metrics":{}, "loss_attestations":[], "early_stopping_records":[]}}

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
    require(isinstance(recipe["model"], dict) and set(recipe["model"]) == {"method_id", "params"}, "Closed classifier head required")
    _classifier_params(recipe["model"]["params"])
    require(set(recipe["model"]) == {"method_id", "params"} and recipe["model"]["method_id"] == METHOD, "Closed native PLS-logistic head required")

class MethodsMultimodalClassifierController:
    """One invocation-local Methods predictor behind standard native callbacks."""

    def __init__(self, *, operators: dict[str, Any], sources: dict[str, Any], targets: dict[str, Any] | None = None,
                 target_names: tuple[str, ...] = ("y",), allow_fit: bool = True, controller_id: str = "controller:methods.python.multimodal.classification", source_ids: tuple[str, ...] = SOURCE_ORDER, node_params: dict[str, dict[str, Any]] | None = None) -> None:
        owners = {f"controller:methods.{host}.multimodal.classification": host for host in ("python",)}
        require(controller_id in owners and (not allow_fit or owners[controller_id] == "python"), "Exact closed multimodal producer owner required; fitting requires Python ownership")
        require(type(allow_fit) is bool and allow_fit == (targets is not None), "Replay must not receive fitting targets")
        require(set(sources) == set(SOURCE_ORDER) and bool(operators), "Complete canonical raw sources/operators required")
        require(tuple(target_names) == ("y",), "One canonical classification target required")
        self.controller_id, self.plugin = controller_id, f"dagml.methods.{owners[controller_id]}.multimodal.classification"
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
            require(operator["classification"] == next(iter(self.operators.values()))["classification"], "Raw classifier vocabularies must be identical")
            validate_recipe(operator["recipe"], operator["source_schemas"])
            require(operator["source_schemas"] == {name: sources[name]["descriptor"] for name in SOURCE_ORDER}, "Current raw source schemas differ from signed operator")
        require(set(self.node_params) <= set(self.operators), "Foreign expected parameter node")
        for node, params in self.node_params.items():
            validate_recipe(recipe_for_node(self.operators[node], params), self.operators[node]["source_schemas"])
        if targets is not None:
            require(len(targets["sample_ids"]) == len(set(targets["sample_ids"])) and len(targets["values"]) == len(targets["sample_ids"]), "Target sample IDs must be unique and complete")
            _checked_target_values(targets["values"], next(iter(self.operators.values()))["classification"])

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
        targets = cast(dict[str, Any], self.targets)
        positions = {sample: index for index, sample in enumerate(targets["sample_ids"])}
        require(all(sample in positions for sample in ids), "Unknown target sample IDs")
        values = _checked_target_values(targets["values"], next(iter(self.operators.values()))["classification"])
        return values[[positions[sample] for sample in ids]].reshape(len(ids), 1)

    def _keep(self, entry: dict[str, Any]) -> dict[str, Any]:
        handle = self.next_handle
        self.next_handle += 1
        self.models[handle] = entry
        return {"handle": handle, "kind": "model", "owner_controller": self.controller_id}

    def _result(self, task, ids, blocks, model, artifacts=None):
        node = task["node_plan"]["node_id"]
        surfaces = [(ids, blocks, "validation" if task["phase"] == "FIT_CV" else "final")]
        if task["phase"] != "PREDICT" and any(view["partition"] == "predict" for view in task.get("data_views", {}).values()):
            test_ids, test_blocks = self._features(task, "predict")
            surfaces.append((test_ids, test_blocks, "test"))
        return _classification_result(self, task, surfaces, model, self.operators[node]["classification"], artifacts or [], lambda value: (value, {"source_schemas": self._selected_schemas(node)}))

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
            entry = cast(dict[str, Any], entry)
            ids, blocks = self._features(task, "predict")
            return self._result(task, ids, blocks, entry["model"])
        require(self.allow_fit, "Fitting is disabled for replay")
        from n4m import MultimodalClassifierPipeline
        ids, blocks = self._features(task, "fold_train" if phase == "FIT_CV" else "full_train")
        valid_ids, valid = self._features(task, "fold_validation") if phase == "FIT_CV" else (ids, blocks)
        require(phase != "FIT_CV" or not set(ids).intersection(valid_ids), "Training/validation overlap")
        _classifier_working_set(operator["classification"], recipe["model"]["params"], len(ids))
        model = MultimodalClassifierPipeline(recipe, self._selected_schemas(node["node_id"]))
        retained = False
        try:
            model.fit(blocks, self._targets(ids).reshape(-1)); _checked_classes(model, operator["classification"])
            self._event("fit", node=node["node_id"], fold=task.get("fold_id"), sample_ids=list(ids), raw_shapes={name: list(value.shape) for name, value in blocks.items()}, source_order=list(recipe["source_order"]), source_weights=copy.deepcopy(recipe["source_weights"]), recipe=copy.deepcopy(recipe))
            if phase == "FIT_CV":
                return self._result(task, valid_ids, valid, model)
            saved = {"schema": SCHEMA, "node_id": node["node_id"], "params_fingerprint": node["params_fingerprint"], "target_names": self.target_names,
                     "recipe": recipe, "source_schemas": operator["source_schemas"], "classification": operator["classification"], "state": list(model.export_state())}
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
        require(saved["classification"] == self.operators[saved["node_id"]]["classification"], "Saved class vocabulary differs from signed graph")
        require(saved["recipe"] == recipe_for_node(self.operators[saved["node_id"]], self.node_params.get(saved["node_id"], {})), "Complete predictor selected recipe mismatch before hydration")
        from n4m import MultimodalClassifierPipeline
        model = MultimodalClassifierPipeline.from_state(bytes(saved["state"]), recipe=saved["recipe"], source_schemas=self._selected_schemas(saved["node_id"]), class_names=saved["classification"]["class_labels"])
        try:
            _checked_classes(model, saved["classification"])
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
                except BaseException as problem:  # noqa: BLE001 - Dispose every owner before re-raising the first failure.
                    if failure is None:
                        failure = problem
        finally:
            self.models.clear()
            self.artifacts.clear()
            self.closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> Self:
        require(not self.closed, "Methods multimodal controller is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _meta_recipe(operator, params):
    require(isinstance(operator, dict) and set(operator) == {"type", "steps", "source_order", "classification"} and operator["type"] == "N4mRoleClassifierPipeline", "Explicit classifier OOF meta operator required")
    validate_classification(operator["classification"])
    order = operator["source_order"]
    require(isinstance(order, list) and 2 <= len(order) <= 4 and len(set(order)) == len(order) and all(name in SOURCE_ORDER for name in order), "Distinct ordered raw classifier branches required")
    steps = operator["steps"]
    require(isinstance(steps, list) and len(steps) == 1 and set(steps[0]) == {"methodId", "params"} and steps[0]["methodId"] == METHOD, "One PLS-logistic meta classifier required")
    require(isinstance(params, dict) and set(params) <= {"n_components", "max_iter"}, "Unsupported effective meta classifier parameter")
    recipe = copy.deepcopy(steps)
    recipe[0]["params"].update(params)
    _classifier_params(recipe[0]["params"])
    _classifier_working_set(operator["classification"], recipe[0]["params"])
    return recipe


def _role_steps(recipe):
    return [(step["methodId"], copy.deepcopy(step["params"])) for step in recipe]

class MethodsOOFClassifierController:
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
                        "output_ports": manifest_for_host()["output_ports"],
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
        edges = cast(list[dict[str, Any]], self.edges)
        incoming = [edge for edge in edges if edge["target"]["node_id"] == node_id]
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
            require(edge["source"]["port_name"] == "probabilities" and self.raw_operators[producer]["classification"] == self.operators[node_id]["classification"], "OOF class vocabulary/port differs from signed meta declaration")
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
        arrays = []
        names: list[str] = []
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
            require(len(samples) * len(expected) * len(self.operators[task["node_plan"]["node_id"]]["classification"]["class_labels"]) <= 16_777_216, "Native ordered classifier OOF matrix exceeds the closed buffer budget")
            require(
                block.get("prediction_level", "sample") == "sample"
                and block["prediction_width"] == len(self.operators[task["node_plan"]["node_id"]]["classification"]["class_labels"])
                and block["target_names"] == [f"class:{i}" for i in self.operators[task["node_plan"]["node_id"]]["classification"]["class_labels"]],
                "OOF classifiers require exact signed ordered class distributions per branch",
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
                values.shape == (len(samples), len(self.operators[task["node_plan"]["node_id"]]["classification"]["class_labels"])) and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)) and np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6),
                "Finite native prediction matrix required",
            )
            arrays.append(values)
            names.extend(f"{producer}/class:{i}" for i in self.operators[task["node_plan"]["node_id"]]["classification"]["class_labels"])
        assert ids is not None
        return ids, np.concatenate(arrays, axis=1), names

    def _targets(self, ids: list[str]) -> np.ndarray:
        require(self.targets is not None, "Replay cannot read training targets")
        targets = cast(dict[str, Any], self.targets)
        positions = {
            sample: index for index, sample in enumerate(targets["sample_ids"])
        }
        require(
            all(sample in positions for sample in ids), "Unknown OOF target identity"
        )
        values = _checked_target_values(targets["values"], next(iter(self.operators.values()))["classification"])
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
        declaration = self.operators[node["node_id"]]["classification"]
        return _classification_result(self, task, [(surface[0], surface, partition) for surface, partition in surfaces], model, declaration, refs, lambda value: (self._frame(value), {}))

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
            entry = cast(dict[str, Any], entry)
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
        _classifier_working_set(self.operators[node["node_id"]]["classification"], recipe[0]["params"], len(train[0]))
        model = RolePipeline(_role_steps(recipe))
        retained = False
        try:
            model.fit(self._frame(train), self._targets(train[0]).reshape(-1)); _checked_classes(model, self.operators[node["node_id"]]["classification"])
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
                and exported[0][0] == "models.classification.pls_logistic"
                and not exported[0][2],
                "One portable PLS-logistic state without training rows required",
            )
            saved = {
                "schema": "dagml.methods.classification.v1",
                "node_id": node["node_id"],
                "params_fingerprint": node["params_fingerprint"],
                "target_names": self.target_names,
                "classification": self.operators[node["node_id"]]["classification"],
                "source_order": self.operators[node["node_id"]]["source_order"],
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
                "classification",
                "source_order",
            }
            and saved["schema"] == "dagml.methods.classification.v1",
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
            saved["classification"] == self.operators[node_id]["classification"]
            and saved["source_order"] == self.operators[node_id]["source_order"]
            and saved["steps"]
            == _meta_recipe(self.operators[node_id], self.node_params.get(node_id, {})),
            "Saved meta recipe differs from signed selected parameters",
        )
        require(
            saved["feature_names"]
            == [f"{producer}/class:{i}" for producer in self._input_nodes(node_id) for i in self.operators[node_id]["classification"]["class_labels"]],
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
            class_names=saved["classification"]["class_labels"],
        )
        try:
            _checked_classes(model, saved["classification"])
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
                except BaseException as problem:  # noqa: BLE001 - Dispose every owner before re-raising the first failure.
                    error = error or problem
        finally:
            self.models.clear()
            self.artifacts.clear()
            self.closed = True
        if error is not None:
            raise error


class ClassificationTopologyController:
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
            if operator.get("type") == "N4mMultimodalClassifierPipeline"
        }
        meta = {
            node: operator
            for node, operator in operators.items()
            if operator.get("type") == "N4mRoleClassifierPipeline"
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
        self.raw = MethodsMultimodalClassifierController(
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
                MethodsOOFClassifierController(
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
        return cast(MethodsOOFClassifierController, self.meta).operator(task)

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

    def __enter__(self) -> Self:
        require(not self.closed, "Topology controller is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
