"""JSONL transport and real Octave state tests; no surrogate numerical controller."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import queue
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.qualify_multimodal_methods_hpo_octave import (
    CONTROLLER,
    OctaveWorker,
    audit,
    prepare_octave,
    require_role_abi,
)


@pytest.mark.parametrize("version", ["1.2.1+abi.2.14.0", "1.2.1+abi.2.17.0", "1.2.1+abi.2.18.1"])
def test_role_abi_accepts_additive_minor_compatibility(version):
    require_role_abi(version)


@pytest.mark.parametrize("version", ["1.2.1+abi.2.13.9", "1.2.1+abi.3.17.0",
                                     "1.2.1+abi.2.17.", "1.2.1+abi.2.17.0-extra", None, 217])
def test_role_abi_refuses_incompatible_or_malformed_runtime(version):
    with pytest.raises(AssertionError, match="ABI major 2, minor >= 14"):
        require_role_abi(version)


@pytest.mark.parametrize("payload_type", [bytes, bytearray, memoryview, list])
def test_hydration_frames_binary_payload_as_lossless_json_octets(tmp_path, payload_type):
    raw = bytes(range(256)) + "é猫".encode()
    payload = payload_type(raw)
    handle = {"handle": 73, "kind": "artifact", "owner_controller": CONTROLLER}
    worker = OctaveWorker.__new__(OctaveWorker)
    worker.calls = []
    worker.hydrated = set()
    worker.timeout = 1
    worker.stderr = tmp_path / "worker.stderr.log"
    worker.stderr.write_text("")
    worker.process = SimpleNamespace(poll=lambda: None, stdin=io.StringIO())
    worker.lines = queue.Queue()
    worker.lines.put(json.dumps({"type": "portable_artifact", "schema_version": 1,
                                "result": {"operation": "hydrated_artifact_payload",
                                           "schema_version": 1, "handle": handle}}))
    artifact = {"content_fingerprint": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    message = hydration(artifact, payload)
    original_request = copy.deepcopy(message["request"])

    assert worker.artifact(message) == handle

    # Exercise the real frame encoder, replacing only the external worker's acknowledgement.
    frame = json.loads(worker.process.stdin.getvalue())
    assert frame["type"] == "portable_artifact" and frame["schema_version"] == 1
    assert frame["task"]["operation"] == "hydrate_artifact_payload"
    assert frame["task"]["schema_version"] == 1
    assert frame["task"]["request"] == original_request
    assert frame["task"]["payload"] == list(raw)
    assert hashlib.sha256(bytes(frame["task"]["payload"])).hexdigest() == artifact["content_fingerprint"]
    assert message["operation"] == "hydrate" and message["payload"] is payload
    assert bytes(message["payload"]) == raw
    assert message["request"] == original_request
    assert worker.hydrated == {73} and worker.calls == [{"operation": "hydrate"}]


@pytest.fixture
def octave_inputs(tmp_path):
    octave = shutil.which(os.environ.get("DAG_ML_OCTAVE", "octave"))
    capture_path = os.environ.get("NIRS4ALL_OCTAVE_ROLE_NODE_CAPTURE")
    if not octave or not capture_path:
        if os.environ.get("DAGML_REQUIRE_OCTAVE_ROLES") == "1":
            pytest.fail("Real Octave and the unchanged Node qualification capture are required")
        pytest.skip("Opt-in native Octave qualification needs runtime and captured fixture")
    capture = json.loads(Path(capture_path).read_text())
    sample_ids = capture["sampleIds"]
    rows = capture["sourceRows"]["nir"]
    sources = {"sensor:nir.v1": {"sample_ids": sample_ids, "rows": rows,
               "feature_names": [f"signal:{i}.v1" for i in range(len(rows[0]))]}}
    operators = {"model:nir.v1": {"type": "n4m:models.regularized.ridge"}}
    targets = {"sample_ids": list(reversed(sample_ids)),
               "values": [[y] for y in reversed(capture["target"])]}
    return Path(octave), tmp_path, sources, operators, targets


def task(ids, phase="REFIT"):
    return {"node_plan": {"node_id": "model:nir.v1", "kind": "model",
            "controller_id": CONTROLLER, "controller_version": "1.0.0",
            "params": {"alpha": 0.05}, "params_fingerprint": "a" * 64},
            "phase": phase, "run_id": "run:protocol.v1", "fold_id": None,
            "variant_id": None, "branch_path": ["branch:nir.v1"], "seed": 2**64 - 7,
            "prediction_inputs": {}, "data_views": {"data:x": {
                "partition": "predict" if phase == "PREDICT" else "full_train",
                "sample_ids": ids, "source_ids": ["sensor:nir.v1"], "columns": None,
                "include_augmented": False, "include_excluded": False}},
            "artifact_inputs": {}, "input_handles": {}, "data_view_receipts": {},
            "required_loss_attestations": [], "residual_targets": None, "fit_influence": None}


def fit_export(inputs):
    octave, work, sources, operators, targets = inputs
    prepared = prepare_octave(octave, work, "training", sources, operators, targets)
    with OctaveWorker(prepared) as worker:
        native_task = task(targets["sample_ids"])
        for field in ("data_view_receipts", "required_loss_attestations", "residual_targets", "fit_influence"):
            native_task.pop(field)
        result = worker.operator(native_task)
        ref = result["artifacts"][0]
        raw = worker.artifact({"operation": "export", "artifact_id": ref["id"]})
    return result, ref, raw, prepared


def hydration(ref, raw):
    return {"operation": "hydrate", "request": {"controller_id": CONTROLLER,
            "node_id": "model:nir.v1", "params_fingerprint": "a" * 64, "artifact": ref},
            "payload": raw}


def reseal(ref, saved):
    raw = json.dumps(saved, ensure_ascii=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    ref = {**ref, "uri": f"artifacts/{digest}.json", "size_bytes": len(raw),
           "content_fingerprint": digest}
    return ref, list(raw)


def test_real_raw_singletons_exact_keys_seed_and_fresh_no_fit(octave_inputs):
    result, ref, raw, training = fit_export(octave_inputs)
    saved = json.loads(bytes(raw))
    assert set(saved) == {"schema", "node_id", "params_fingerprint", "target_names",
                          "steps", "feature_names", "states"}
    assert saved["schema"] == "dagml.methods.regression.v1"
    assert len(saved["states"]) == len(saved["steps"]) == 1
    assert bytes(saved["states"][0][:4]) == b"N4ME"
    assert saved["feature_names"][0] == "data:x/signal:0.v1"
    assert result["lineage"]["seed"] == 2**64 - 7
    assert list(result["artifact_handles"]) == [ref["id"]]
    assert ref["plugin"] == "dagml.methods.octave.regression"
    octave, work, sources, operators, _ = octave_inputs
    first_id = result["predictions"][0]["sample_ids"][0]
    source = sources["sensor:nir.v1"]
    index = source["sample_ids"].index(first_id)
    heldout = {"sensor:nir.v1": {**source, "sample_ids": [first_id], "rows": [source["rows"][index]]}}
    prepared = prepare_octave(octave, work, "fresh", heldout, operators, None)
    with OctaveWorker(prepared) as worker:
        handle = worker.artifact(hydration(ref, bytes(raw)))
        predict_task = task([first_id], "PREDICT")
        key = "artifact:state.v1"
        predict_task["input_handles"][key] = handle
        predict_task["artifact_inputs"][key] = {"node_id": "model:nir.v1", "controller_id": CONTROLLER,
                                               "params_fingerprint": "a" * 64, "artifact": ref}
        predicted = worker.operator(predict_task)
        assert predicted["predictions"][0]["sample_ids"] == [first_id]
        assert predicted["predictions"][0]["values"] == [result["predictions"][0]["values"][0]]
        assert predicted["lineage"]["seed"] == 2**64 - 7
        worker.artifact({"operation": "release", "handle": handle})
        assert not worker.hydrated
        assert Counter(row["operation"] for row in audit(prepared)) == {
            "hydrate": 1, "PREDICT": 1, "dispose": 1, "release": 1}
        with pytest.raises(RuntimeError, match="Unknown or released"):
            worker.artifact({"operation": "release", "handle": handle})
    assert Counter(row["operation"] for row in audit(training))["dispose"] == 1


@pytest.mark.parametrize("mutation", ["owner", "plugin", "node", "field", "state", "target"])
def test_real_hydration_refuses_foreign_tampered_and_incomplete_wrappers(octave_inputs, mutation):
    _, ref, raw, _ = fit_export(octave_inputs)
    saved = json.loads(bytes(raw))
    if mutation == "owner":
        ref["controller_id"] = "controller:methods.r.regression"
    elif mutation == "plugin":
        ref["plugin"] = "dagml.methods.octave.unknown"
    else:
        if mutation == "node": saved["node_id"] = "model:foreign.v1"
        elif mutation == "field": saved["untrusted"] = True
        elif mutation == "state": saved["states"][0] = list(b"N4MEinvalid-native-state")
        elif mutation == "target": saved["target_names"] = ["foreign"]
        ref, raw = reseal(ref, saved)
    octave, work, sources, operators, _ = octave_inputs
    prepared = prepare_octave(octave, work, "negative", sources, operators, None)
    with OctaveWorker(prepared) as worker:
        with pytest.raises(RuntimeError): worker.artifact(hydration(ref, raw))
        assert not worker.hydrated
        assert Counter(row["operation"] for row in audit(prepared))["hydrate"] == 0


@pytest.mark.parametrize("flag", ["include_excluded", "include_augmented"])
def test_real_fitting_view_refuses_excluded_or_augmented_rows(octave_inputs, flag):
    octave, work, sources, operators, targets = octave_inputs
    prepared = prepare_octave(octave, work, "excluded", sources, operators, targets)
    with OctaveWorker(prepared) as worker:
        bad = task(targets["sample_ids"]); bad["data_views"]["data:x"][flag] = True
        with pytest.raises(RuntimeError, match="Augmented or excluded fitting"):
            worker.operator(bad)
        assert not audit(prepared)


def test_real_prediction_refuses_recipe_feature_order_and_foreign_handles(octave_inputs):
    _, ref, raw, _ = fit_export(octave_inputs)
    octave, work, sources, operators, _ = octave_inputs
    prepared = prepare_octave(octave, work, "recipe-refusal", sources, operators, None)
    with OctaveWorker(prepared) as worker:
        handle = worker.artifact(hydration(ref, raw))
        current = task(sources["sensor:nir.v1"]["sample_ids"], "PREDICT")
        key = "artifact:state.v1"
        current["input_handles"][key] = handle
        current["artifact_inputs"][key] = {"node_id": "model:nir.v1", "controller_id": CONTROLLER,
                                          "params_fingerprint": "a" * 64, "artifact": ref}
        bad = copy.deepcopy(current); bad["node_plan"]["params"]["alpha"] = 2
        with pytest.raises(RuntimeError, match="recipe or parameters"):
            worker.operator(bad)
        bad = copy.deepcopy(current); bad["input_handles"][key]["owner_controller"] = "controller:foreign"
        with pytest.raises(RuntimeError, match="foreign hydrated"):
            worker.operator(bad)
        with pytest.raises(RuntimeError, match="Fitting is disabled"):
            worker.operator(task(sources["sensor:nir.v1"]["sample_ids"]))
        assert Counter(row["operation"] for row in audit(prepared))["PREDICT"] == 0
        worker.artifact({"operation": "release", "handle": handle})
    features = prepare_octave(octave, work, "feature-refusal", sources, operators, None,
                              feature_permutation=True)
    with OctaveWorker(features) as worker:
        handle = worker.artifact(hydration(ref, raw))
        current["input_handles"][key] = handle
        with pytest.raises(RuntimeError, match="feature order"):
            worker.operator(current)
        worker.artifact({"operation": "release", "handle": handle})
        assert Counter(row["operation"] for row in audit(features))["PREDICT"] == 0


@pytest.mark.parametrize("field,value", [
    ("data_view_receipts", {"unexpected": {"receipt": "generated"}}),
    ("required_loss_attestations", [{"loss_id": "custom"}]),
    ("residual_targets", {"sample_ids": ["foreign"]}),
    ("fit_influence", {"mechanism": "sample_weights", "row_weights": [1.0]}),
    ("fit_influence", {"mechanism": "uniform_rows", "target_row_weights": [[1.0]]}),
])
def test_real_omitted_defaults_keep_specialized_task_refusals(octave_inputs, field, value):
    octave, work, sources, operators, targets = octave_inputs
    prepared = prepare_octave(octave, work, "specialized-refusal", sources, operators, targets)
    with OctaveWorker(prepared) as worker:
        current = task(targets["sample_ids"])
        for optional in ("data_view_receipts", "required_loss_attestations", "residual_targets", "fit_influence"):
            current.pop(optional)
        current[field] = value
        with pytest.raises(RuntimeError, match="specialized controller"):
            worker.operator(current)
        assert not audit(prepared)
        assert not worker.hydrated


def test_real_persistent_worker_runs_two_cv_fits_then_selected_refit(octave_inputs):
    octave, work, sources, operators, targets = octave_inputs
    prepared = prepare_octave(octave, work, "persistent-cv-refit", sources, operators, targets)
    all_ids = targets["sample_ids"]
    middle = len(all_ids) // 2
    halves = [all_ids[:middle], all_ids[middle:]]
    assert min(map(len, halves)) >= 2
    with OctaveWorker(prepared) as worker:
        for index in range(2):
            train, valid = halves[index], halves[1 - index]
            current = task(train, "FIT_CV")
            current["fold_id"] = f"fold:persistent.{index}"
            current["data_views"]["data:x"]["partition"] = "fold_train"
            current["data_views"]["data:x:validation"] = {
                **current["data_views"]["data:x"],
                "partition": "fold_validation", "sample_ids": valid}
            out = worker.operator(current)
            assert out["predictions"][0]["sample_ids"] == valid
            assert out["predictions"][0]["partition"] == "validation"
            assert not out["artifacts"]
            assert not worker.hydrated
        final = worker.operator(task(all_ids))
        assert final["predictions"][0]["sample_ids"] == all_ids
        assert len(final["artifacts"]) == 1
        raw = worker.artifact({"operation": "export", "artifact_id": final["artifacts"][0]["id"]})
        assert bytes(json.loads(bytes(raw))["states"][0][:4]) == b"N4ME"
        fit_rows = [row["sample_ids"] for row in audit(prepared) if row["operation"] == "fit"]
        assert fit_rows == [halves[0], halves[1], all_ids]
    counts = Counter(row["operation"] for row in audit(prepared))
    assert counts["fit"] == counts["dispose"] == 3
    assert counts["release"] == 1
