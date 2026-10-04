"""Real classifier state hydration and strict class-aware OOF transport.

These protocol witnesses do not replace the SDK's independent grouped numerical
oracles. No production host constructs folds, selects models or computes scores.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct

import numpy as np
import pytest
from dag_ml.multimodal_classification import (
    META_CONTROLLER,
    METHOD,
    RAW_CONTROLLER,
    ClassificationTopologyController,
    MethodsOOFClassifierController,
    _classifier_working_set,
    validate_classification,
)

from scripts.validate_classification_state import Reader
from scripts.validate_classification_state import classifier as validate_state


def declarations(classes=3):
    ids = [f"s{i}" for i in range(12)] + ["t0", "t1"]
    rng = np.random.default_rng(7103)
    arrays = {"nir": rng.normal(size=(14, 4)), "image": rng.normal(size=(14, 2, 2, 1)),
              "series": rng.normal(size=(14, 2, 2)),
              "metadata": np.array([[float(i), f"category:{i % 3}"] for i in range(14)], dtype=object)}
    sources = {name: {"sample_ids": ids, "values": values, "descriptor": {
        "representation_id": representation, "input_shape": list(values.shape[1:]), "dtype": "float64" if name != "metadata" else "object",
        "identity": json.dumps({"source": name})}}
        for (name, values), representation in zip(arrays.items(), ("signal_1d", "rgb_image", "series_mv", "tabular_mixed"), strict=True)}
    vocabulary = {"schema_version": 1, "class_labels": list(range(classes)), "label_names": [-17, 42, 901][:classes]}
    encoder = {"nir": {"kind": "standard_scaler", "with_mean": True, "with_std": True},
               "image": {"kind": "tensor_pca", "n_components": 1, "whiten": False, "random_state": 17},
               "series": {"kind": "tensor_pca", "n_components": 1, "whiten": False, "random_state": 17},
               "metadata": {"kind": "column_transformer", "numeric_columns": [0], "categorical_columns": [1], "with_mean": True, "with_std": True, "handle_unknown": "ignore", "sparse_output": False, "drop": None}}
    def raw(order):
        return {"type": "N4mMultimodalClassifierPipeline", "classification": copy.deepcopy(vocabulary),
                "source_schemas": {name: value["descriptor"] for name, value in sources.items()},
                "recipe": {"schema_version": 1, "fusion": "early", "source_order": order,
                           "encoders": {name: encoder[name] for name in order}, "source_weights": {name: 1.0 for name in order},
                           "model": {"method_id": METHOD, "params": {"n_components": 1, "max_iter": 200}}}}
    operators = {"early": raw(list(arrays)), **{f"raw:{name}": raw([name]) for name in arrays}}
    operators["meta"] = {"type": "N4mRoleClassifierPipeline", "source_order": ["image", "nir", "series", "metadata"],
                         "classification": copy.deepcopy(vocabulary), "steps": [{"methodId": METHOD, "params": {"n_components": 1, "max_iter": 200}}]}
    edges = [{"source": {"node_id": f"raw:{name}", "port_name": "probabilities"}, "target": {"node_id": "meta", "port_name": name},
              "contract": {"kind": "prediction", "requires_oof": True}} for name in arrays]
    targets = {"sample_ids": ids, "values": np.arange(14, dtype=np.int64) % classes}
    return operators, sources, targets, edges


def controller(allow_fit=True, classes=3, params=None):
    operators, sources, targets, edges = declarations(classes)
    return ClassificationTopologyController(operators=operators, sources=sources,
        targets=targets if allow_fit else None, allow_fit=allow_fit, edges=edges, node_params=params)


def meta_task(phase="REFIT", test=False, classes=3):
    rng = np.random.default_rng(17)
    suffix = ":predict" if phase == "PREDICT" else ""
    values = {name: rng.dirichlet(np.ones(classes), size=12).tolist() for name in ("nir", "image", "series", "metadata")}
    inputs = {name + suffix: {"producer_node": f"raw:{name}", "source_port": "probabilities", "target_port": name,
         "partition": "final" if phase == "PREDICT" else "validation", "sample_ids": [f"s{i}" for i in range(12)],
         "values": rows, "prediction_width": classes, "target_names": [f"class:{i}" for i in range(classes)], "fold_ids": ["inner0", "inner1"]}
         for name, rows in values.items()}
    if test:
        inputs.update({name + ":refit": {**copy.deepcopy(block), "partition": "test", "sample_ids": ["t0", "t1"], "values": rng.dirichlet(np.ones(classes), size=2).tolist(), "fold_ids": []}
                       for name, block in list(inputs.items())})
    return {"phase": phase, "node_plan": {"node_id": "meta", "kind": "model", "controller_id": META_CONTROLLER,
             "controller_version": "1.0.0", "params": {}, "params_fingerprint": "a" * 64},
            "prediction_inputs": inputs, "run_id": "run:classifier", "variant_id": "variant:classifier", "fold_id": None,
            "seed": 17, "branch_path": [], "data_views": {}, "artifact_inputs": {}, "input_handles": {}}


def raw_task(phase="REFIT", params=None):
    views = {"x": {"partition": "predict" if phase == "PREDICT" else "full_train", "sample_ids": [f"s{i}" for i in range(12)],
                   "source_ids": ["nir", "image", "series", "metadata"], "include_augmented": False, "include_excluded": False, "columns": []}}
    return {"phase": phase, "node_plan": {"node_id": "early", "kind": "model", "controller_id": RAW_CONTROLLER,
             "controller_version": "1.0.0", "params": params or {}, "params_fingerprint": "b" * 64},
            "data_views": views, "prediction_inputs": {}, "run_id": "run:raw-classifier", "variant_id": "variant:classifier", "fold_id": None,
            "seed": 17, "branch_path": [], "artifact_inputs": {}, "input_handles": {}}


def hydrate(producer, result):
    artifact = result["artifacts"][0]
    binding = {"artifact": artifact, "controller_id": artifact["controller_id"], "node_id": result["node_id"], "params_fingerprint": result["lineage"]["params_fingerprint"]}
    payload = producer.artifact({"operation": "export", "artifact_id": artifact["id"]})
    return binding, payload


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("heldout", [False, True])
def test_real_meta_state_preserves_class_order_refit_partition_and_target_free_replay(monkeypatch, classes, heldout):
    from n4m.roles import RolePipeline
    with controller(classes=classes) as producer, controller(False, classes=classes) as consumer:
        task = meta_task(test=heldout, classes=classes)
        result = producer.operator(task)
        assert all(block["partition"] == "test" and block["fold_id"] is None for block in result["predictions"])
        assert bool(result["predictions"]) is heldout
        assert not result["predictions"] or {block["producer_port"] for block in result["predictions"]} == {"y_hat", "probabilities"}
        binding, payload = hydrate(producer, result)
        saved = json.loads(bytes(payload))
        expected_names = [f"raw:{name}/class:{i}" for name in ["image", "nir", "series", "metadata"] for i in range(classes)]
        assert saved["feature_names"] == expected_names
        assert saved["classification"]["label_names"] == [-17, 42, 901][:classes]
        validate_state(bytes(saved["states"][0]), saved["steps"][0]["params"], len(expected_names), list(range(classes)))
        expected_rows = producer.meta._predictions(task, "train")
        fitted = next(iter(producer.meta.models.values()))["model"]
        expected_labels = fitted.predict(producer.meta._frame(expected_rows))
        expected_probabilities = fitted.predict_proba(producer.meta._frame(expected_rows))
        def forbidden(*args, **kwargs):
            raise AssertionError("Portable replay must never fit")
        monkeypatch.setattr(RolePipeline, "fit", forbidden)
        handle = consumer.artifact({"operation": "hydrate", "request": binding, "payload": payload})
        replay = meta_task("PREDICT", classes=classes)
        replay["artifact_inputs"], replay["input_handles"] = {"model": binding}, {"model": handle}
        actual = consumer.operator(replay)
        blocks = {block["producer_port"]: block for block in actual["predictions"]}
        assert set(blocks) == {"y_hat", "probabilities"}
        assert all(block["partition"] == "final" for block in blocks.values())
        np.testing.assert_array_equal(np.asarray(blocks["y_hat"]["values"])[:, 0], expected_labels)
        np.testing.assert_array_equal(blocks["probabilities"]["values"], expected_probabilities)
        assert actual["regression_targets"] == [] and actual["classification_probabilities"] == []
        consumer.artifact({"operation": "release", "handle": handle})
        assert not consumer.meta.models


@pytest.mark.parametrize("mutation", ["class_columns", "foreign", "train", "test", "outer", "row_order", "width", "probability"])
def test_classifier_oof_refusals_precede_fit(monkeypatch, mutation):
    from n4m.roles import RolePipeline
    task = meta_task("FIT_CV")
    task["fold_id"] = "outer"
    block = task["prediction_inputs"]["nir"]
    if mutation == "class_columns": block["target_names"].reverse()
    elif mutation == "foreign": block["producer_node"] = "foreign"
    elif mutation in {"train", "test"}: block["partition"] = mutation
    elif mutation == "outer": block["fold_ids"] = ["outer"]
    elif mutation == "row_order": block["sample_ids"].reverse()
    elif mutation == "width": block["prediction_width"] = 1
    else: block["values"][0][0] = 1.5
    calls = []
    monkeypatch.setattr(RolePipeline, "fit", lambda *args, **kwargs: calls.append(True))
    with controller() as owner:
        with pytest.raises(ValueError): owner.operator(task)
        assert calls == [] and owner.meta.models == {} and owner.meta.artifacts == {}


def test_raw_selected_counts_export_and_hydrate_preserve_n4mc_without_fit(monkeypatch):
    from n4m import MultimodalClassifierPipeline
    params = {"early": {"model__n_components": 2, "model__max_iter": 180}, "meta": {"n_components": 2, "max_iter": 180}}
    with controller(params=params) as producer, controller(False, params=params) as consumer:
        task = raw_task(params=params["early"])
        result = producer.operator(task)
        binding, payload = hydrate(producer, result)
        saved = json.loads(bytes(payload))
        assert bytes(saved["state"][:4]) == b"N4MC"
        assert saved["recipe"]["model"]["params"] == {"n_components": 2, "max_iter": 180}
        def forbidden(*args, **kwargs):
            raise AssertionError("Raw classifier replay must never fit")
        monkeypatch.setattr(MultimodalClassifierPipeline, "fit", forbidden)
        handle = consumer.artifact({"operation": "hydrate", "request": binding, "payload": payload})
        replay = raw_task("PREDICT", params=params["early"])
        replay["artifact_inputs"], replay["input_handles"] = {"model": binding}, {"model": handle}
        actual = consumer.operator(replay)
        assert actual["predictions"] == result["predictions"]
        assert not actual["regression_targets"]


@pytest.mark.parametrize("mutation", ["label_map", "steps", "branch_order", "features", "checksum"])
def test_resigned_meta_wrapper_refused_before_hydration(monkeypatch, mutation):
    from n4m.roles import RolePipeline
    with controller() as producer, controller(False) as consumer:
        result = producer.operator(meta_task())
        binding, payload = hydrate(producer, result)
        saved = json.loads(bytes(payload))
        if mutation == "label_map": saved["classification"]["label_names"] = ["a", "b", "c"]
        elif mutation == "steps": saved["steps"][0]["params"]["n_components"] = 2
        elif mutation == "branch_order": saved["source_order"].reverse()
        elif mutation == "features": saved["feature_names"].reverse()
        else: saved["states"][0][-1] ^= 1
        encoded = json.dumps(saved, separators=(",", ":"), ensure_ascii=False).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        binding["artifact"].update(uri=f"artifacts/{digest}.json", content_fingerprint=digest, size_bytes=len(encoded))
        if mutation != "checksum":
            monkeypatch.setattr(RolePipeline, "from_states", lambda *a, **kw: pytest.fail("Metadata must refuse before native import"))
        with pytest.raises((ValueError, RuntimeError)):
            consumer.artifact({"operation": "hydrate", "request": binding, "payload": list(encoded)})
        assert not consumer.meta.models


@pytest.mark.parametrize("names", [["", "named"], [-19, 45]])
def test_typed_original_label_maps_are_conserved(names):
    validate_classification({"schema_version": 1, "class_labels": [0, 1], "label_names": names})


def test_logistic_working_set_checks_hessian_and_design_before_allocation():
    vocabulary = {"class_labels": [0, 1]}
    _classifier_working_set(vocabulary, {"n_components": 4095, "max_iter": 30}, 1)
    for rows, classes, components in [(1, 2, 4096), (3072, 512, 128), (16_777_216 // 9 + 1, 2, 8)]:
        with pytest.raises(ValueError, match="PLS-logistic working set"):
            _classifier_working_set({"class_labels": list(range(classes))}, {"n_components": components, "max_iter": 30}, rows)


@pytest.mark.parametrize("owner", ["raw", "meta"])
def test_effective_dangerous_head_refuses_before_native_model_construction(monkeypatch, owner):
    import n4m
    import n4m.roles

    def forbidden(*args, **kwargs):
        pytest.fail("working-set refusal must precede native model construction")

    monkeypatch.setattr(n4m, "MultimodalClassifierPipeline", forbidden)
    monkeypatch.setattr(n4m.roles, "RolePipeline", forbidden)
    with controller(classes=2) as instance:
        task = meta_task(classes=2) if owner == "meta" else raw_task()
        task["node_plan"]["params"] = {"n_components" if owner == "meta" else "model__n_components": 4096}
        with pytest.raises(ValueError, match="PLS-logistic working set"):
            instance.operator(task)
        assert not instance.audit


@pytest.mark.parametrize("names", [[1, "2"], [False, True], ["b", "a"], [1, 1], [0, 2**63]])
def test_invalid_typed_vocabularies_refused(names):
    with pytest.raises(ValueError): validate_classification({"schema_version": 1, "class_labels": [0, 1], "label_names": names})


def test_unknown_parameter_nodes_and_unimplemented_hosts_refused():
    with pytest.raises(ValueError, match="Foreign expected topology parameter node"):
        controller(params={"foreign": {"n_components": 1}})
    operators, _, targets, edges = declarations()
    with pytest.raises(ValueError, match="Foreign expected meta parameter node"):
        MethodsOOFClassifierController(operators={"meta": operators["meta"]}, raw_operators={node: op for node, op in operators.items() if node.startswith("raw:")}, edges=edges,
            targets=targets, target_names=("y",), allow_fit=True, node_params={"early": {}})
    from dag_ml.multimodal_classification import manifest_for_host
    for host in ("r", "wasm", "octave"):
        with pytest.raises(ValueError, match="native Python host"):
            manifest_for_host(host)


def reseal_state(value):
    hashed = 0xcbf29ce484222325
    for byte in value[:-8]:
        hashed = ((hashed ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    value[-8:] = struct.pack("<Q", hashed)


@pytest.mark.parametrize("mutation", ["abi", "classes", "training_rows", "latent_solver", "trailing"])
def test_independent_parser_rejects_resealed_genuine_classifier_state(mutation):
    with controller() as producer:
        result = producer.operator(meta_task())
        _, payload = hydrate(producer, result)
        saved = json.loads(bytes(payload))
        state = bytearray(saved["states"][0])
        reader = Reader(state)
        reader.take(20)
        reader.text(True)
        for _ in range(reader.u32()):
            reader.text(True)
            reader.u32()
            count = reader.u64()
            reader.take(count * 8)
        caps_position = reader.pos
        reader.take(24)
        assert reader.u32() == 1 and reader.u32() == 0x31534c43
        block_size = reader.u64()
        block_position = reader.pos
        assert block_size > 16 + 3 * 8
        if mutation == "abi": state[12:16] = struct.pack("<I", 16)
        elif mutation == "classes": state[block_position+16:block_position+40] = struct.pack("<qqq", 2, 1, 0)
        elif mutation == "training_rows": state[caps_position:caps_position+8] = struct.pack("<Q", 156 | 512)
        elif mutation == "latent_solver":
            model_size = struct.unpack("<q", state[block_position+40:block_position+48])[0]
            model_position = block_position + 56
            model = bytearray(state[model_position:model_position+model_size])
            model[24:28] = struct.pack("<I", 0)
            reseal_state(model)
            state[model_position:model_position+model_size] = model
        else: state[-8:-8] = b"x"
        reseal_state(state)
        with pytest.raises(ValueError):
            validate_state(bytes(state), saved["steps"][0]["params"], len(saved["feature_names"]), [0, 1, 2])
