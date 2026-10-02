"""Real portable controller/protocol tests, independent of transport witnesses."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from dag_ml.multimodal_methods import MethodsMultimodalController, manifest_for_host
from scripts.qualify_u07_methods_multimodal import ProcessWorker, prepare_worker, python_sources, qualify, replay_prediction_values


@pytest.fixture
def capture() -> dict:
    selected = os.environ.get("DAG_ML_U07_CAPTURE")
    if not selected:
        if os.environ.get("DAG_ML_REQUIRE_U07_NATIVE") == "1":
            pytest.fail("Mandatory actual canonical U07 capture is absent")
        pytest.skip("Requires the actual public SDK U07 native capture")
    value = json.loads(Path(selected).read_text())
    assert len(value["package"]["artifact_bindings"]) == 1
    assert len(value["search_result"]["trials"]) == 8
    return value


def controller(capture: dict, **kwargs) -> MethodsMultimodalController:
    return MethodsMultimodalController(operators=capture["operators"], sources=python_sources(capture["prediction_sources"]), targets=None,
        allow_fit=False, target_names=tuple(capture["target_names"]), source_ids=tuple(capture["source_ids"]), node_params=capture["node_params"], **kwargs)


def hydrate_message(capture: dict) -> dict:
    package = capture["package"]
    record = package["execution_bundle"]["refit_artifacts"][0]
    return {"operation": "hydrate", "request": {"artifact": copy.deepcopy(record["artifact"]), "controller_id": record["controller_id"], "node_id": record["node_id"], "params_fingerprint": record["params_fingerprint"]},
            "payload": copy.deepcopy(package["execution_bundle"]["raw_artifact_payloads"][record["artifact"]["id"]])}


def resign(message: dict, wrapper: dict) -> None:
    raw = json.dumps(wrapper, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    message["payload"] = list(raw)
    message["request"]["artifact"].update(uri=f"artifacts/{digest}.json", content_fingerprint=digest, size_bytes=len(raw))


def test_actual_python_complete_state_replays_without_fit(capture: dict) -> None:
    import dag_ml
    with controller(capture) as consumer:
        outcome = dag_ml.replay_loaded_predictor_package(capture["package"], capture["prediction_request"], capture["prediction_envelopes"], {}, consumer.operator,
            outcome_id="outcome:u07.controller", run_id="run:u07.controller", artifact_callback=consumer.artifact, trusted_controller_manifests=[manifest_for_host()]).to_dict()
        actual = replay_prediction_values(outcome, capture["prediction_sources"]["nir"]["sample_ids"], capture["target_names"])
        assert actual == pytest.approx(capture["expected_prediction"], rel=1e-7, abs=1e-7)
        assert [entry["operation"] for entry in consumer.audit] == ["hydrate", "PREDICT", "dispose", "release"]
        assert not consumer.models


@pytest.mark.parametrize("mutation", ["owner", "cross_pair", "unknown_plugin", "params", "shape", "axis", "recipe", "extra_field", "magic", "boolean_byte", "checksum"])
def test_actual_resigned_state_refusals_cleanup(capture: dict, mutation: str) -> None:
    message = hydrate_message(capture)
    wrapper = json.loads(bytes(message["payload"]))
    if mutation == "owner": message["request"]["controller_id"] = "controller:foreign"
    elif mutation == "cross_pair": message["request"]["artifact"]["plugin"] = "dagml.methods.r.multimodal"
    elif mutation == "unknown_plugin": message["request"]["artifact"]["plugin"] = "dagml.methods.unknown.multimodal"
    elif mutation == "params": wrapper["params_fingerprint"] = "0" * 64
    elif mutation == "shape": wrapper["source_schemas"]["image"]["input_shape"][0] += 1
    elif mutation == "axis": wrapper["source_schemas"]["nir"]["identity"] = '{"unit":"wrong"}'
    elif mutation == "recipe": wrapper["recipe"]["model"]["params"]["alpha"] += 1
    elif mutation == "extra_field": wrapper["pickle"] = "refused"
    elif mutation == "magic": wrapper["state"][0] = 0
    elif mutation == "boolean_byte": wrapper["state"][20] = True
    else: wrapper["state"][-1] ^= 1
    if mutation not in {"owner", "cross_pair", "unknown_plugin"}:
        resign(message, wrapper)
    with controller(capture) as consumer:
        with pytest.raises(Exception, match="(?i)owner|plugin|node|parameter|params|schema|shape|recipe|wrapper|byte|state|checksum|integrity|format|N4MF"):
            consumer.artifact(message)
        assert not consumer.models
        assert not consumer.artifacts
        assert not any(row["operation"] == "hydrate" for row in consumer.audit)


@pytest.mark.parametrize("owner", ["controller:foreign", "controller:methods.r.regression"])
def test_unknown_replay_owner_refused_without_native_import(capture: dict, owner: str) -> None:
    with pytest.raises(ValueError, match="closed multimodal producer"):
        controller(capture, controller_id=owner)


def test_current_trusted_manifest_refuses_before_host_hydration(capture: dict) -> None:
    import dag_ml
    with controller(capture) as consumer:
        manifest = manifest_for_host()
        manifest["controller_version"] = "1.0.1"
        with pytest.raises(Exception, match="(?i)controller|manifest|version"):
            dag_ml.replay_loaded_predictor_package(capture["package"], capture["prediction_request"], capture["prediction_envelopes"], {}, consumer.operator,
                outcome_id="outcome:u07.untrusted", run_id="run:u07.untrusted", artifact_callback=consumer.artifact, trusted_controller_manifests=[manifest])
        assert not consumer.audit


@pytest.mark.parametrize("host,environment", [("r", "DAG_ML_RSCRIPT"), ("wasm", "DAG_ML_METHODS_WASM_DIST"), ("octave", "DAG_ML_OCTAVE")])
def test_real_host_canonical_grid_hpo_and_unchanged_python_archive(capture: dict, tmp_path: Path, host: str, environment: str) -> None:
    if not os.environ.get(environment):
        if os.environ.get(f"DAG_ML_REQUIRE_U07_{host.upper()}") == "1":
            pytest.fail("Mandatory real host prerequisite absent: " + environment)
        pytest.skip("Actual host prerequisite absent: " + environment)
    receipt = qualify(capture, tmp_path / host, host)
    assert receipt["producer_archive_replay"]["runtime"]["execution_host"] == host
    assert receipt["producer_archive_replay"]["runtime"]["signed_controller"] == "controller:methods.python.multimodal"
    package = json.loads((tmp_path / host / "portable-package.json").read_text())
    payload = next(iter(package["execution_bundle"]["raw_artifact_payloads"].values()))
    saved = json.loads(bytes(payload))
    assert saved["recipe"]["encoders"]["metadata"]["drop"] is None



@pytest.mark.parametrize("host,environment", [("r", "DAG_ML_RSCRIPT"), ("wasm", "DAG_ML_METHODS_WASM_DIST"), ("octave", "DAG_ML_OCTAVE")])
def test_real_host_singleton_u64_keys_and_failed_request_cleanup(capture: dict, tmp_path: Path, host: str, environment: str) -> None:
    if not os.environ.get(environment):
        if os.environ.get(f"DAG_ML_REQUIRE_U07_{host.upper()}") == "1":
            pytest.fail("Mandatory actual host prerequisite absent: " + environment)
        pytest.skip("Actual host prerequisite absent: " + environment)
    command, config = prepare_worker(capture, tmp_path / host, host, sources=capture["prediction_sources"], allow_fit=False, producer_host="python")
    message = hydrate_message(capture)
    with ProcessWorker(command, config, tmp_path / host, host) as worker:
        handle = worker.artifact(message)
        record = capture["package"]["execution_bundle"]["refit_artifacts"][0]
        node_id = record["node_id"]
        node_plan = copy.deepcopy(capture["package"]["effective_plan"]["node_plans"][node_id])
        sample = capture["prediction_sources"]["nir"]["sample_ids"][0]
        task = {"run_id": "run:/singleton.v1", "phase": "PREDICT", "seed": 2**64-1, "variant_id": None, "fold_id": None, "branch_path": [], "node_plan": node_plan,
                "data_views": {"input:/x.v1": {"partition": "predict", "sample_ids": [sample], "source_ids": capture["source_ids"], "include_augmented": False, "include_excluded": False, "columns": []}},
                "artifact_inputs": {"artifact:/input.v1": record}, "input_handles": {"artifact:/input.v1": handle}, "prediction_inputs": {}, "data_view_receipts": {}, "required_loss_attestations": [], "residual_targets": None, "fit_influence": None}
        # A direct protocol witness uses actual native state; it does not claim
        # to be a separately signed scheduler-produced campaign.
        for seed in (2**64-1, None):
            task["seed"] = seed
            result = worker.operator(task)
            assert result["predictions"][0]["sample_ids"] == [sample]
            assert len(result["predictions"][0]["values"]) == len(result["predictions"][0]["values"][0]) == 1
            assert result["lineage"]["seed"] == seed
        for seed in (True, -1, 1.5, 2**64):
            invalid_seed = copy.deepcopy(task)
            invalid_seed["seed"] = seed
            with pytest.raises(RuntimeError, match="(?i)seed|u64"):
                worker.operator(invalid_seed)
        bad = copy.deepcopy(task)
        bad["input_handles"]["artifact:/input.v1"]["owner_controller"] = "controller:foreign"
        with pytest.raises(RuntimeError, match="(?i)owner|handle|binding|predict"):
            worker.operator(bad)
        worker.artifact({"operation": "release", "handle": handle})


def test_raw_qualification_transport_preserves_declared_dtypes() -> None:
    sources = {
        "nir": {"sample_ids": ["s:/0", "s:/1"], "descriptor": {"dtype": "float32"}, "shape": [2, 1], "data": [1.25, 2.5]},
        "metadata": {"sample_ids": ["s:/0", "s:/1"], "descriptor": {"dtype": "<U4"}, "shape": [2, 2], "rows": [["1", ""], ["2", "µ"]]},
    }
    transported = python_sources(sources)
    assert transported["nir"]["values"].dtype == np.dtype("float32")
    assert transported["nir"]["values"].tolist() == [[1.25], [2.5]]
    assert transported["metadata"]["values"].dtype == np.dtype("<U4")
    assert transported["metadata"]["values"].tolist() == [["1", ""], ["2", "µ"]]


def test_empty_raw_sample_id_remains_refused(capture: dict) -> None:
    changed = copy.deepcopy(capture)
    changed["prediction_sources"]["nir"]["sample_ids"][0] = ""
    with pytest.raises(ValueError, match="Sample|sample.*IDs|unique strings"):
        controller(changed)
