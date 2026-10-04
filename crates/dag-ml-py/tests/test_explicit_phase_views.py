"""Real explicit-phase scheduling with host-owned fixed-table view receipts.

The fixture operator exercises the bridge contract and returns fixed outputs;
it is not a numerical training implementation. Tables, row selection and their
fingerprints are nevertheless real and precede every operator invocation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import unittest
from pathlib import Path
from typing import Any

import dag_ml
import dag_ml._dag_ml as native

REPO = Path(__file__).resolve().parents[3]
TRAIN_IDS = ["sample:zulu", "sample:alpha"]
PREDICT_IDS = ["heldout:delta", "heldout:omega"]
CONTROLLER_ID = "controller:fixture.explicit.named"
NODE_ID = "model:fixture.explicit.named"

# Physical storage differs between sources and from both native row scopes.
TABLES: dict[str, dict[str, Any]] = {
    "nir": {
        "sample_ids": ["sample:alpha", "heldout:omega", "sample:zulu", "heldout:delta"],
        "values": [[1., 2., 3., 4.], [31., 32., 33., 34.], [11., 12., 13., 14.], [21., 22., 23., 24.]],
    },
    "clinical": {
        "sample_ids": ["heldout:delta", "sample:zulu", "heldout:omega", "sample:alpha"],
        "values": [[400., 401., 402.], [200., 201., 202.], [300., 301., 302.], [100., 101., 102.]],
    },
}


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _selected_table(name: str, sample_ids: list[str]) -> list[list[float]]:
    source = TABLES[name]
    return [list(source["values"][source["sample_ids"].index(sample_id)]) for sample_id in sample_ids]


def _buffer_digest(port: dict[str, Any], sample_ids: list[str], values: list[list[float]]) -> str:
    """Hash the selected fixture buffer's actual dtype, shape, rows and bytes."""
    dtype = port["metadata"]["dtype"]
    width = port["metadata"]["feature_shape"][0]
    cells = [value for row in values for value in row]
    prefix = json.dumps([port["metadata"]["source_id"], sample_ids, dtype, [len(sample_ids), width]]).encode()
    return hashlib.sha256(prefix + b"\0" + struct.pack("<" + ("d" if dtype == "float64" else "f") * len(cells), *cells)).hexdigest()


def _inputs(phase: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    specification = json.loads((REPO / "examples/fixtures/data/model_input_spec_named_dense.json").read_text())
    relations = {"records": [{"observation_id": f"obs:{sample_id}", "sample_id": sample_id} for sample_id in reversed(TRAIN_IDS)]}
    envelope = {
        "schema_version": 1,
        "schema_fingerprint": _json_digest(specification),
        "plan_fingerprint": _json_digest({"fixed_sources": list(TABLES)}),
        "relation_fingerprint": dag_ml.sample_relation_set_fingerprint_json(json.dumps(relations)),
        "coordinator_relations": relations,
        "data_content_fingerprint": _json_digest(TABLES),
        "target_content_fingerprint": _json_digest({"sample:zulu": 2., "sample:alpha": 1.}),
    }
    if phase == "PREDICT":
        envelope = dag_ml.attach_predict_cohort_to_envelope(envelope, {
            "role": "inference",
            "relations": {"records": [
                {"observation_id": f"obs:{sample_id}", "sample_id": sample_id}
                for sample_id in reversed(PREDICT_IDS)
            ]},
            "target_names": ["y"],
            "data_content_fingerprint": _json_digest({name: _selected_table(name, PREDICT_IDS) for name in TABLES}),
            "target_content_fingerprint": None,
        }).to_dict()
    dsl = {
        "id": "dsl:fixture.explicit.named",
        "root_seed": 29,
        "steps": [{
            "kind": "model", "id": NODE_ID, "operator": {"type": "FixtureNamedOperator"},
            "params": {}, "model_input": specification,
            "metadata": {"controller_id": CONTROLLER_ID},
        }],
        "data_bindings": [{
            "node_id": NODE_ID, "input_name": port["name"], "request_id": f"fixed:{port['name']}",
            "schema_fingerprint": envelope["schema_fingerprint"], "plan_fingerprint": envelope["plan_fingerprint"],
            "relation_fingerprint": envelope["relation_fingerprint"], "output_representation": "tabular_numeric",
            "feature_set_id": f"features:{port['name']}", "source_ids": [port["metadata"]["source_id"]],
            "require_relations": True, "metadata": {"source_name": port["name"]},
        } for port in specification["ports"]],
    }
    # The real compiler determines the signatures; the fixture declares no
    # substitute positional X port or fabricated graph-node identity.
    node = dag_ml.compile_pipeline_dsl_artifact(dsl).graph.to_dict()["nodes"][0]
    manifest = {
        "controller_id": CONTROLLER_ID, "controller_version": "1", "operator_kind": "model", "priority": 0,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"], "input_ports": node["ports"]["inputs"], "output_ports": node["ports"]["outputs"],
        "data_requirements": specification,
        "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions", "emits_artifacts", "stateful"],
        "fit_scope": "fold_train", "rng_policy": "uses_core_seed", "artifact_policy": "serializable",
    }
    return dsl, envelope, [manifest], specification


class _FixedTableViews:
    """Select actual fixture buffers behind handles supplied by the scheduler."""

    def __init__(self, testcase: unittest.TestCase, specification: dict[str, Any], phase: str) -> None:
        self.testcase = testcase
        self.ports = {port["name"]: port for port in specification["ports"]}
        self.phase = phase
        self.records: dict[int, dict[str, Any]] = {}
        self.events: list[tuple[str, Any]] = []

    def __call__(self, call: dict[str, Any]) -> dict[str, Any]:
        request, handle = call["request"], call["handle"]
        name, view = request["input_name"], request["view"]
        port = self.ports[name]
        ids = TRAIN_IDS if self.phase == "REFIT" else PREDICT_IDS
        self.testcase.assertEqual(request["phase"], self.phase)
        self.testcase.assertIsNone(request["fold_id"])
        self.testcase.assertEqual(request["node_id"], NODE_ID)
        self.testcase.assertEqual(view["partition"], "full_train" if self.phase == "REFIT" else "predict")
        self.testcase.assertEqual(view["sample_ids"], ids)
        self.testcase.assertEqual(request["binding"]["source_ids"], [port["metadata"]["source_id"]])
        self.testcase.assertEqual(view["source_ids"], request["binding"]["source_ids"])
        self.testcase.assertEqual(handle["kind"], "data_view")
        self.testcase.assertEqual(request["data_handle"]["kind"], "data")
        self.testcase.assertNotEqual(handle["handle"], request["data_handle"]["handle"])
        self.testcase.assertNotIn(handle["handle"], self.records)
        self.testcase.assertTrue(request["view_key"])
        self.testcase.assertIsInstance(request["view_seed"], int)
        values = _selected_table(name, ids)
        receipt = {
            "handle": copy.deepcopy(handle), "view_key": request["view_key"], "sample_ids": list(ids),
            "schema_fingerprint": _json_digest(port), "content_fingerprint": _buffer_digest(port, ids, values),
        }
        self.records[handle["handle"]] = {"request": copy.deepcopy(request), "receipt": copy.deepcopy(receipt), "values": values}
        self.events.append(("view", handle["handle"]))
        return receipt


class _FixtureOperator:
    """Witness receipt consumption; static responses do not claim ML training."""

    def __init__(self, testcase: unittest.TestCase, views: _FixedTableViews) -> None:
        self.testcase, self.views = testcase, views
        self.calls: list[dict[str, Any]] = []

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(task))
        self.views.events.append(("operator", task["phase"]))
        ids = TRAIN_IDS if task["phase"] == "REFIT" else PREDICT_IDS
        receipts = task["data_view_receipts"]
        self.testcase.assertEqual(set(receipts), {"data:nir", "data:clinical"})
        self.testcase.assertEqual(set(receipts), set(task["data_views"]))
        consumed = {}
        for name in self.views.ports:
            key = f"data:{name}"
            receipt, handle = receipts[key], task["input_handles"][key]
            record = self.views.records[handle["handle"]]
            self.testcase.assertEqual(receipt, record["receipt"])
            self.testcase.assertEqual(record["request"]["view"], task["data_views"][key])
            self.testcase.assertEqual(receipt["sample_ids"], ids)
            self.testcase.assertEqual(record["values"], _selected_table(name, ids))
            self.testcase.assertLess(self.views.events.index(("view", handle["handle"])), len(self.views.events) - 1)
            model_call = {
                "operation": "fit" if task["phase"] == "REFIT" else "predict", "sample_ids": list(ids),
                "input_fingerprint": _buffer_digest(self.views.ports[name], ids, record["values"]),
            }
            if task["phase"] == "REFIT":
                model_call["target_fingerprint"] = _json_digest([[2.], [1.]])
            consumed[key] = {"receipt": copy.deepcopy(receipt), "read_batches": [list(ids)], "model_calls": [model_call]}
        node = task["node_plan"]
        artifact_id = "artifact:fixture.explicit.named"
        artifacts = [{"id": artifact_id, "kind": "fixture_operator", "controller_id": node["controller_id"],
                      "content_fingerprint": _json_digest({"fixed_fixture": True})}] if task["phase"] == "REFIT" else []
        return {
            "node_id": node["node_id"],
            "outputs": {"oof": {"handle": 100_001, "kind": "prediction", "owner_controller": node["controller_id"]}},
            "predictions": [{"prediction_id": "prediction:fixture.explicit.named", "producer_node": node["node_id"],
                             "producer_port": "oof", "partition": "final", "fold_id": None, "sample_ids": list(ids),
                             "values": [[0.] for _ in ids], "target_names": ["y"]}] if task["phase"] == "PREDICT" else [],
            "artifacts": artifacts,
            "artifact_handles": {artifact_id: {"handle": 100_002, "kind": "model", "owner_controller": node["controller_id"]}} if artifacts else {},
            "consumed_data_views": consumed,
            "lineage": {
                "record_id": f"lineage:fixture.explicit.named:{task['phase']}", "run_id": task["run_id"], "node_id": node["node_id"],
                "phase": task["phase"], "controller_id": node["controller_id"], "controller_version": node["controller_version"],
                "variant_id": task["variant_id"], "fold_id": task["fold_id"], "branch_path": task["branch_path"],
                "input_lineage": [], "artifact_refs": artifacts, "params_fingerprint": node["params_fingerprint"],
                "seed": task["seed"], "unsafe_flags": [], "metrics": {},
            },
        }


def _run(phase: str, dsl: dict[str, Any], envelope: dict[str, Any], manifests: list[dict[str, Any]], operator: Any, views: Any, resources: Any = None) -> dict[str, Any]:
    return json.loads(native.execute_phase_in_process(
        json.dumps(dsl), json.dumps(envelope), json.dumps(manifests), operator, phase,
        training_sample_ids=TRAIN_IDS if phase == "REFIT" else None, view_callback=views,
        resource_limits_json=json.dumps(resources) if resources is not None else None,
    ))


class ExplicitPhaseViewsTests(unittest.TestCase):
    def test_native_refit_and_predict_preserve_each_attested_named_buffer_and_order(self) -> None:
        for phase in ("REFIT", "PREDICT"):
            with self.subTest(phase=phase):
                dsl, envelope, manifests, specification = _inputs(phase)
                views = _FixedTableViews(self, specification, phase)
                operator = _FixtureOperator(self, views)
                outcome = _run(phase, dsl, envelope, manifests, operator, views)
                self.assertEqual(outcome["phase"], phase)
                self.assertEqual(len(operator.calls), 1)
                self.assertEqual(len(views.records), 2)
                self.assertEqual(views.events[-1], ("operator", phase))
                expected = {
                    "nir": [[11., 12., 13., 14.], [1., 2., 3., 4.]],
                    "clinical": [[200., 201., 202.], [100., 101., 102.]],
                } if phase == "REFIT" else {
                    "nir": [[21., 22., 23., 24.], [31., 32., 33., 34.]],
                    "clinical": [[400., 401., 402.], [300., 301., 302.]],
                }
                self.assertEqual({record["request"]["input_name"]: record["values"] for record in views.records.values()}, expected)
                self.assertEqual(len(outcome["node_results"]), 1)
                result = outcome["node_results"][0]
                self.assertEqual(set(result["consumed_data_views"]), {"data:nir", "data:clinical"})
                content = {entry["receipt"]["content_fingerprint"] for entry in result["consumed_data_views"].values()}
                self.assertEqual(len(content), 2)
                self.assertIsNone(outcome["effective_plan"]["fold_set"])
                if phase == "REFIT":
                    self.assertEqual(outcome["effective_plan"]["campaign"]["metadata"]["explicit_training_sample_ids"], TRAIN_IDS)
                    self.assertTrue(result["artifacts"])
                else:
                    self.assertEqual(result["predictions"][0]["sample_ids"], PREDICT_IDS)
                    self.assertFalse(result["artifacts"])
                    self.assertIsNone(outcome["scores"])

    def test_explicit_resource_limits_are_native_tasks_for_refit_and_predict(self) -> None:
        resources = {"cpu_threads": 1, "gpu_devices": [], "memory_bytes": 4096, "wall_time_ms": 1000}
        for phase in ("REFIT", "PREDICT"):
            with self.subTest(phase=phase):
                dsl, envelope, manifests, specification = _inputs(phase)
                views = _FixedTableViews(self, specification, phase)
                operator = _FixtureOperator(self, views)
                _run(phase, dsl, envelope, manifests, operator, views, resources)
                self.assertEqual(len(operator.calls), 1)
                self.assertEqual(operator.calls[0]["resources"], resources)

    def test_invalid_resource_limits_refuse_before_views_and_operator(self) -> None:
        invalid = (
            {"cpu_threads": 0, "gpu_devices": []},
            {"cpu_threads": 1, "gpu_devices": [], "memory_bytes": 0},
            {"cpu_threads": 1, "gpu_devices": [], "wall_time_ms": 0},
            {"cpu_threads": 1, "gpu_devices": ["gpu:0", "gpu:0"]},
            {"cpu_threads": 1, "gpu_devices": [], "unknown": True},
        )
        for phase in ("REFIT", "PREDICT"):
            for resources in invalid:
                with self.subTest(phase=phase, resources=resources):
                    dsl, envelope, manifests, specification = _inputs(phase)
                    views = _FixedTableViews(self, specification, phase)
                    operator = _FixtureOperator(self, views)
                    with self.assertRaises((native.DagMlCompatibilityError, native.DagMlRuntimeError, native.DagMlValidationError)):
                        _run(phase, dsl, envelope, manifests, operator, views, resources)
                    self.assertEqual(operator.calls, [])
                    self.assertEqual(views.events, [])

    def test_noncallable_materializer_is_refused_before_any_operator(self) -> None:
        for phase in ("REFIT", "PREDICT"):
            with self.subTest(phase=phase):
                dsl, envelope, manifests, specification = _inputs(phase)
                operator = _FixtureOperator(self, _FixedTableViews(self, specification, phase))
                with self.assertRaisesRegex(native.DagMlRuntimeError, "view_callback must be callable"):
                    _run(phase, dsl, envelope, manifests, operator, 42)
                self.assertEqual(operator.calls, [])

    def test_missing_or_tampered_receipt_is_refused_before_any_operator(self) -> None:
        for phase in ("REFIT", "PREDICT"):
            for mutation in ("none", "missing_digest", "handle", "view_key", "ordered_ids", "invalid_digest"):
                with self.subTest(phase=phase, mutation=mutation):
                    dsl, envelope, manifests, specification = _inputs(phase)
                    views = _FixedTableViews(self, specification, phase)
                    operator = _FixtureOperator(self, views)

                    def broken(call: dict[str, Any], mutation: str = mutation) -> Any:
                        receipt = views(call)  # The real fixture table was read first.
                        if mutation == "none":
                            return None
                        if mutation == "missing_digest":
                            receipt.pop("content_fingerprint")
                        elif mutation == "handle":
                            receipt["handle"]["handle"] += 1_000
                        elif mutation == "view_key":
                            receipt["view_key"] = "wrong-native-view-key"
                        elif mutation == "ordered_ids":
                            receipt["sample_ids"].reverse()
                        else:
                            receipt["content_fingerprint"] = "not-a-sha256-digest"
                        return receipt

                    # Malformed wire receipts fail deserialization; valid-shaped
                    # mismatches fail the native runtime contract. Neither may FIT.
                    with self.assertRaises((native.DagMlCompatibilityError, native.DagMlRuntimeError)):
                        _run(phase, dsl, envelope, manifests, operator, broken)
                    self.assertTrue(views.records)
                    self.assertEqual(operator.calls, [])
                    self.assertTrue(all(event[0] == "view" for event in views.events))


if __name__ == "__main__":
    unittest.main()
