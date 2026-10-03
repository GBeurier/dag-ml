"""Real native meta-state replay and refusals at the prediction transport boundary."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from dag_ml.multimodal_topology import (
    META_CONTROLLER,
    MethodsOOFController,
    MethodsTopologyController,
)


def declarations():
    meta = {
        "type": "N4mRolePipeline",
        "source_order": ["image", "nir"],
        "steps": [
            {
                "methodId": "models.regularized.ridge",
                "params": {
                    "alpha": 0.4,
                    "center_x": True,
                    "center_y": True,
                    "scale_x": False,
                },
            }
        ],
    }
    raw = {
        "branch:nir": {"recipe": {"source_order": ["nir"]}},
        "branch:image": {"recipe": {"source_order": ["image"]}},
    }
    edges = [
        {
            "source": {"node_id": node, "port_name": "y_hat"},
            "target": {"node_id": "meta", "port_name": name},
            "contract": {"kind": "prediction", "requires_oof": True},
        }
        for node, name in [("branch:nir", "nir"), ("branch:image", "image")]
    ]
    return meta, raw, edges


def make_controller(*, allow_fit=True):
    meta, raw, edges = declarations()
    return MethodsOOFController(
        operators={"meta": meta},
        raw_operators=raw,
        edges=edges,
        targets={
            "sample_ids": ["a", "b", "c", "d", "e"],
            "values": [0.4, 0.7, 0.2, 1.1, 0.9],
        }
        if allow_fit
        else None,
        target_names=("y",),
        allow_fit=allow_fit,
        node_params={"meta": {"alpha": 0.4}},
    )


def task(phase="REFIT"):
    suffix = "" if phase != "PREDICT" else ":predict"
    blocks = {
        # Opposite map order from the declared meta feature order.
        "nir": ([0.1, 0.4, 0.9, 0.2, 0.8], "branch:nir"),
        "image": ([0.8, 0.2, 0.5, 0.7, 0.3], "branch:image"),
    }
    return {
        "phase": phase,
        "node_plan": {
            "node_id": "meta",
            "kind": "model",
            "controller_id": META_CONTROLLER,
            "controller_version": "1.0.0",
            "params": {"alpha": 0.4},
            "params_fingerprint": "a" * 64,
        },
        "prediction_inputs": {
            name + suffix: {
                "producer_node": producer,
                "partition": "validation" if phase != "PREDICT" else "final",
                "source_port": "y_hat",
                "target_port": name,
                "sample_ids": ["a", "b", "c", "d", "e"],
                "values": [[x] for x in values],
                "prediction_width": 1,
                "target_names": ["y"],
                "fold_ids": ["inner0", "inner1"],
            }
            for name, (values, producer) in blocks.items()
        },
        "run_id": "run:meta",
        "variant_id": "v1",
        "fold_id": None,
        "seed": 17,
        "branch_path": [],
        "data_views": {},
        "artifact_inputs": {},
        "input_handles": {},
    }


def test_native_ridge_meta_refit_export_hydrate_and_replay_without_fit(monkeypatch):
    from n4m.roles import RolePipeline

    producer = make_controller()
    consumer = make_controller(allow_fit=False)
    try:
        request = task()
        result = producer.operator(request)
        assert result["predictions"] == []
        assert result["regression_targets"] == []
        x = np.column_stack(([0.8, 0.2, 0.5, 0.7, 0.3], [0.1, 0.4, 0.9, 0.2, 0.8]))
        expected = Ridge(alpha=0.4).fit(x, [0.4, 0.7, 0.2, 1.1, 0.9]).predict(x)
        entry = next(iter(producer.models.values()))
        values = np.asarray(
            entry["model"].predict(
                producer._frame(producer._predictions(request, "train"))
            )
        ).reshape(-1, 1)
        np.testing.assert_allclose(values[:, 0], expected, atol=1e-10, rtol=1e-10)
        artifact = result["artifacts"][0]
        payload = producer.artifact(
            {"operation": "export", "artifact_id": artifact["id"]}
        )
        assert json.loads(bytes(payload))["feature_names"] == [
            "branch:image/0",
            "branch:nir/0",
        ]
        binding = {
            "artifact": artifact,
            "controller_id": META_CONTROLLER,
            "node_id": "meta",
            "params_fingerprint": "a" * 64,
        }

        def forbidden(*args, **kwargs):
            raise AssertionError("Replay must never fit")

        monkeypatch.setattr(RolePipeline, "fit", forbidden)
        handle = consumer.artifact(
            {"operation": "hydrate", "request": binding, "payload": payload}
        )
        replay = task("PREDICT")
        replay["artifact_inputs"] = {"model": binding}
        replay["input_handles"] = {"model": handle}
        actual = consumer.operator(replay)
        np.testing.assert_array_equal(actual["predictions"][0]["values"], values)
        consumer.artifact({"operation": "release", "handle": handle})
        assert not consumer.models
        assert [event["operation"] for event in consumer.audit] == [
            "hydrate",
            "PREDICT",
            "dispose",
            "release",
        ]
    finally:
        producer.close()
        consumer.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "producer",
        "source_port",
        "target_port",
        "row_order",
        "partition",
        "width",
        "nan",
        "missing",
        "overlap",
        "own_outer_fold",
    ],
)
def test_meta_invalid_oof_refused_before_native_fit(monkeypatch, mutation):
    from n4m.roles import RolePipeline

    controller = make_controller()
    request = task()
    first = request["prediction_inputs"]["nir"]
    if mutation == "producer":
        first["producer_node"] = "other-candidate:nir"
    elif mutation == "source_port":
        first["source_port"] = "other_output"
    elif mutation == "target_port":
        first["target_port"] = "other_input"
    elif mutation == "row_order":
        first["sample_ids"].reverse()
    elif mutation == "partition":
        first["partition"] = "train"
    elif mutation == "width":
        first["prediction_width"] = 2
    elif mutation == "nan":
        first["values"][0][0] = float("nan")
    elif mutation == "missing":
        request["prediction_inputs"].pop("image")
    else:
        request.update(phase="FIT_CV", fold_id="outer0")
        for key, value in list(request["prediction_inputs"].items()):
            request["prediction_inputs"][key + ":outer"] = copy.deepcopy(value)
        if mutation == "own_outer_fold":
            first["fold_ids"].append("outer0")
    called = []
    monkeypatch.setattr(
        RolePipeline, "fit", lambda *args, **kwargs: called.append(True)
    )
    try:
        with pytest.raises(ValueError):
            controller.operator(request)
        assert not called and not controller.models and not controller.artifacts
    finally:
        controller.close()


@pytest.mark.parametrize(
    "mutation", ["recipe", "features", "owner", "byte_bool", "checksum"]
)
def test_resigned_meta_payload_refusals_before_native_hydration(monkeypatch, mutation):
    from n4m.roles import RolePipeline

    producer, consumer = make_controller(), make_controller(allow_fit=False)
    try:
        result = producer.operator(task())
        artifact = copy.deepcopy(result["artifacts"][0])
        payload = producer.artifact(
            {"operation": "export", "artifact_id": artifact["id"]}
        )
        saved = json.loads(bytes(payload))
        if mutation == "recipe":
            saved["steps"][0]["params"]["alpha"] += 1
        elif mutation == "features":
            saved["feature_names"].reverse()
        elif mutation == "byte_bool":
            saved["states"][0][7] = True
        elif mutation == "checksum":
            saved["states"][0][-1] ^= 1
        else:
            artifact["plugin"] = "dagml.methods.wasm.regression"
        raw = json.dumps(saved, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        artifact.update(
            uri=f"artifacts/{digest}.json",
            content_fingerprint=digest,
            size_bytes=len(raw),
        )
        message = {
            "operation": "hydrate",
            "payload": list(raw),
            "request": {
                "artifact": artifact,
                "node_id": "meta",
                "controller_id": META_CONTROLLER,
                "params_fingerprint": "a" * 64,
            },
        }
        if mutation != "checksum":
            monkeypatch.setattr(
                RolePipeline,
                "from_states",
                lambda *args, **kwargs: pytest.fail(
                    "Invalid wrapper reached native hydration"
                ),
            )
        with pytest.raises(Exception):
            consumer.artifact(message)
        assert not consumer.models and not any(
            event["operation"] == "hydrate" for event in consumer.audit
        )
    finally:
        producer.close()
        consumer.close()


def test_refit_preserves_native_test_partition_and_never_fabricates_final():
    controller = make_controller()
    request = task()
    for key, block in list(request["prediction_inputs"].items()):
        request["prediction_inputs"][key + ":refit"] = {
            **copy.deepcopy(block),
            "partition": "test",
        }
    try:
        result = controller.operator(request)
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["partition"] == "test"
        assert result["predictions"][0]["sample_ids"] == ["a", "b", "c", "d", "e"]
        assert len(result["artifacts"]) == 1
    finally:
        controller.close()


def topology_declarations():
    meta, _, edges = declarations()
    ids = ["a", "b", "c", "d", "e"]
    arrays = {
        "nir": np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5], [0.2, 0.7], [0.8, 0.3]]),
        "image": np.arange(15, dtype=float).reshape(5, 1, 1, 3),
        "series": np.arange(10, dtype=float).reshape(5, 2, 1),
        "metadata": np.array([[str(i), str(i % 2)] for i in range(5)]),
    }
    representations = dict(
        zip(
            arrays,
            ("signal_1d", "rgb_image", "series_mv", "tabular_mixed"),
            strict=True,
        )
    )
    sources = {
        name: {
            "sample_ids": ids,
            "values": values,
            "descriptor": {
                "representation_id": representations[name],
                "input_shape": list(values.shape[1:]),
                "dtype": str(values.dtype),
                "identity": json.dumps({"source": name}),
            },
        }
        for name, values in arrays.items()
    }
    encoders = {
        "nir": {"kind": "standard_scaler", "with_mean": True, "with_std": True},
        "image": {
            "kind": "tensor_pca",
            "n_components": 1,
            "whiten": False,
            "random_state": 17,
        },
    }
    raw = {
        "branch:" + name: {
            "type": "N4mMultimodalPipeline",
            "source_schemas": {
                key: value["descriptor"] for key, value in sources.items()
            },
            "recipe": {
                "schema_version": 1,
                "fusion": "early",
                "source_order": [name],
                "encoders": {name: encoders[name]},
                "source_weights": {name: 1.0},
                "model": {
                    "method_id": "models.regularized.ridge",
                    "params": {
                        "alpha": 0.25,
                        "center_x": True,
                        "center_y": True,
                        "scale_x": False,
                    },
                },
            },
        }
        for name in encoders
    }
    return {**raw, "meta": meta}, sources, edges


def test_topology_selected_raw_and_meta_params_hydrate_without_fit(monkeypatch):
    from n4m import MultimodalPipeline
    from n4m.roles import RolePipeline

    operators, sources, edges = topology_declarations()
    selected = {
        "branch:nir": {"model__alpha": 1.8},
        "branch:image": {"model__alpha": 3.4},
        "meta": {"alpha": 2.2},
    }
    targets = {
        "sample_ids": ["a", "b", "c", "d", "e"],
        "values": [0.4, 0.7, 0.2, 1.1, 0.9],
    }
    producer = MethodsTopologyController(
        operators=operators,
        sources=sources,
        targets=targets,
        node_params=selected,
        edges=edges,
    )
    consumer = MethodsTopologyController(
        operators=operators,
        sources=sources,
        allow_fit=False,
        node_params=selected,
        edges=edges,
    )
    records = []
    try:
        assert producer.raw.node_params == {
            node: selected[node] for node in ("branch:nir", "branch:image")
        }
        assert producer.meta.node_params == {"meta": selected["meta"]}
        for node_id in ("branch:nir", "branch:image", "meta"):
            request = task()
            request["node_plan"].update(node_id=node_id, params=selected[node_id])
            if node_id != "meta":
                request["node_plan"]["controller_id"] = producer.raw.controller_id
                request["prediction_inputs"] = {}
                request["data_views"] = {
                    "x": {
                        "partition": "full_train",
                        "sample_ids": targets["sample_ids"],
                        "source_ids": list(sources),
                        "include_augmented": False,
                        "include_excluded": False,
                        "columns": [],
                    }
                }
            result = producer.operator(request)
            artifact = result["artifacts"][0]
            payload = producer.artifact(
                {"operation": "export", "artifact_id": artifact["id"]}
            )
            saved = json.loads(bytes(payload))
            effective = (
                saved["steps"][0]["params"]["alpha"]
                if node_id == "meta"
                else saved["recipe"]["model"]["params"]["alpha"]
            )
            assert (
                effective
                == selected[node_id]["alpha" if node_id == "meta" else "model__alpha"]
            )
            if node_id == "meta":
                x = np.column_stack(
                    ([0.8, 0.2, 0.5, 0.7, 0.3], [0.1, 0.4, 0.9, 0.2, 0.8])
                )
                expected = (
                    Ridge(alpha=2.2).fit(x, targets["values"]).predict(x).reshape(-1, 1)
                )
            else:
                expected = np.asarray(result["predictions"][0]["values"])
            binding = {
                "artifact": artifact,
                "controller_id": request["node_plan"]["controller_id"],
                "node_id": node_id,
                "params_fingerprint": "a" * 64,
            }
            records.append((request, binding, payload, expected))

        def forbidden(*args, **kwargs):
            raise AssertionError("Hydration and replay must never fit")

        monkeypatch.setattr(MultimodalPipeline, "fit", forbidden)
        monkeypatch.setattr(RolePipeline, "fit", forbidden)
        for request, binding, payload, expected in records:
            handle = consumer.artifact(
                {"operation": "hydrate", "request": binding, "payload": payload}
            )
            replay = copy.deepcopy(request)
            replay["phase"] = "PREDICT"
            if replay["node_plan"]["node_id"] == "meta":
                replay["prediction_inputs"] = task("PREDICT")["prediction_inputs"]
            else:
                replay["data_views"]["x"]["partition"] = "predict"
            replay["artifact_inputs"] = {"model": binding}
            replay["input_handles"] = {"model": handle}
            actual = consumer.operator(replay)
            if replay["node_plan"]["node_id"] == "meta":
                np.testing.assert_allclose(
                    actual["predictions"][0]["values"], expected, atol=1e-10, rtol=1e-10
                )
            else:
                np.testing.assert_array_equal(
                    actual["predictions"][0]["values"], expected
                )
            consumer.artifact({"operation": "release", "handle": handle})
        assert not consumer.raw.models and not consumer.meta.models
        assert (
            len([event for event in consumer.audit if event["operation"] == "hydrate"])
            == 3
        )
    finally:
        producer.close()
        consumer.close()


@pytest.mark.parametrize("owner", ["topology", "meta"])
def test_unknown_parameter_node_refused_by_its_constructor(owner):
    operators, sources, edges = topology_declarations()
    if owner == "topology":
        with pytest.raises(
            ValueError, match="Foreign expected topology parameter node"
        ):
            MethodsTopologyController(
                operators=operators,
                sources=sources,
                allow_fit=False,
                edges=edges,
                node_params={"foreign": {"alpha": 1.0}},
            )
    else:
        with pytest.raises(ValueError, match="Foreign expected meta parameter node"):
            MethodsOOFController(
                operators={"meta": operators["meta"]},
                raw_operators={
                    key: value for key, value in operators.items() if key != "meta"
                },
                edges=edges,
                targets=None,
                target_names=("y",),
                allow_fit=False,
                node_params={"branch:nir": {"model__alpha": 1.8}},
            )


@pytest.mark.parametrize(
    "params",
    [{"alpha": -1.0}, {"alpha": True}, {"alpha": float("nan")}, {"foreign": 1.0}],
)
def test_meta_effective_recipe_refused_before_native_hydration(monkeypatch, params):
    from n4m.roles import RolePipeline

    meta, raw, edges = declarations()
    monkeypatch.setattr(
        RolePipeline,
        "from_states",
        lambda *a, **k: pytest.fail("Invalid recipe reached hydration"),
    )
    with pytest.raises(ValueError, match="Ridge parameters|effective parameters"):
        MethodsOOFController(
            operators={"meta": meta},
            raw_operators=raw,
            edges=edges,
            targets=None,
            target_names=("y",),
            allow_fit=False,
            node_params={"meta": params},
        )
