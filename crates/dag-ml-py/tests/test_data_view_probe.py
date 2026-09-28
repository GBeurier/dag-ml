"""No-fit Python bridge qualification for DATA-PROV-01 view receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from parity.robustness_rng.oracle import tcv1_preimage

import dag_ml._dag_ml as native

FIXTURES = Path(__file__).resolve().parents[3] / "examples" / "fixtures" / "data"
SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "contracts" / "generated_view_manifest.v1.schema.json"


def _inputs() -> tuple[dict, dict]:
    envelope = json.loads((FIXTURES / "coordinator_data_plan_envelope_sample12.json").read_text())
    request = json.loads((FIXTURES / "data_view_request_v3.json").read_text())
    request["view"]["sample_ids"] = ["sample:1"]
    request["view_key"] = "view:v1:" + "e" * 64
    return envelope, request


def _receipt(call: dict) -> dict:
    request = call["request"]
    return {
        "handle": call["handle"],
        "view_key": request["view_key"],
        "sample_ids": request["view"]["sample_ids"],
        "schema_fingerprint": "a" * 64,
        "content_fingerprint": "b" * 64,
    }


class DataViewProbeTests(unittest.TestCase):
    def test_view_callback_receives_native_handle_and_ordered_ids_without_training(self) -> None:
        envelope, request = _inputs()
        calls: list[dict] = []

        def callback(call: dict) -> dict:
            calls.append(call)
            return _receipt(call)

        result = json.loads(native.probe_data_view_in_process(json.dumps(envelope), json.dumps(request), callback, True))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["request"]["data_handle"]["kind"], "data")
        self.assertEqual(calls[0]["handle"]["kind"], "data_view")
        self.assertNotEqual(calls[0]["request"]["data_handle"]["handle"], calls[0]["handle"]["handle"])
        self.assertEqual(result["handle"], calls[0]["handle"])
        self.assertEqual(result["sample_ids"], ["sample:1"])
        self.assertEqual(result["content_fingerprint"], "b" * 64)
        manifest = result["generated_view_manifest"]
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["fingerprint"]), 64)
        self.assertEqual(len(manifest["views"]), 1)
        self.assertEqual(manifest["views"][0]["view_key"], request["view_key"])
        self.assertEqual(manifest["views"][0]["view_seed"], request["view_seed"])
        self.assertEqual(manifest["views"][0]["view"]["sample_ids"], ["sample:1"])
        schema = json.loads(SCHEMA.read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
        preimage = tcv1_preimage(["generated-view-manifest-v1", manifest["views"]])
        self.assertEqual(manifest["fingerprint"], hashlib.sha256(preimage).hexdigest())

        def changed_content(call: dict) -> dict:
            return _receipt(call) | {"content_fingerprint": "c" * 64}

        changed = json.loads(native.probe_data_view_in_process(json.dumps(envelope), json.dumps(request), changed_content, True))
        self.assertNotEqual(manifest["fingerprint"], changed["generated_view_manifest"]["fingerprint"])

        float_request = copy.deepcopy(request)
        float_request["view"]["extra"]["epsilon"] = 1e-7
        with_float = json.loads(native.probe_data_view_in_process(json.dumps(envelope), json.dumps(float_request), callback, True))
        float_views = with_float["generated_view_manifest"]["views"]
        self.assertEqual(float_views[0]["view"]["extra"]["epsilon"], 1e-7)
        self.assertEqual(
            with_float["generated_view_manifest"]["fingerprint"],
            hashlib.sha256(tcv1_preimage(["generated-view-manifest-v1", float_views])).hexdigest(),
        )

    def test_probe_default_keeps_the_strict_receipt_shape(self) -> None:
        envelope, request = _inputs()
        result = json.loads(native.probe_data_view_in_process(
            json.dumps(envelope), json.dumps(request), _receipt,
        ))
        self.assertEqual(set(result), {
            "handle", "view_key", "sample_ids", "schema_fingerprint", "content_fingerprint",
        })

    def test_wrong_receipt_handle_key_ids_or_digest_is_refused(self) -> None:
        envelope, request = _inputs()
        for replacement in (
            {"handle": {"handle": 999, "kind": "data_view", "owner_controller": "controller:data.provider"}},
            {"view_key": "another-key"},
            {"sample_ids": ["sample:2"]},
            {"content_fingerprint": "not-a-digest"},
        ):
            with self.subTest(replacement=replacement):
                def callback(call: dict, replacement: dict = replacement) -> dict:
                    return _receipt(call) | replacement

                with self.assertRaisesRegex(native.DagMlRuntimeError, "receipt does not match"):
                    native.probe_data_view_in_process(json.dumps(envelope), json.dumps(request), callback)

    def test_invalid_cohort_or_binding_is_refused_before_callback(self) -> None:
        envelope, request = _inputs()
        for mutation, pattern in (
            ("unknown-id", "outside its attested cohort"),
            ("duplicate-id", "duplicate sample id"),
            ("wrong-binding", "no external data-plan envelope"),
        ):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(request)
                if mutation == "unknown-id":
                    changed["view"]["sample_ids"] = ["sample:outside"]
                elif mutation == "duplicate-id":
                    changed["view"]["sample_ids"] = ["sample:1", "sample:1"]
                else:
                    changed["binding"]["schema_fingerprint"] = "0" * 64
                calls: list[dict] = []
                with self.assertRaisesRegex(native.DagMlRuntimeError, pattern):
                    native.probe_data_view_in_process(
                        json.dumps(envelope), json.dumps(changed), lambda call, calls=calls: calls.append(call)
                    )
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
