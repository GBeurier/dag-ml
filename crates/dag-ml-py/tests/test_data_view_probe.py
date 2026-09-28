"""No-fit Python bridge qualification for DATA-PROV-01 view receipts."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import dag_ml._dag_ml as native

FIXTURES = Path(__file__).resolve().parents[3] / "examples" / "fixtures" / "data"


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

        result = json.loads(native.probe_data_view_in_process(json.dumps(envelope), json.dumps(request), callback))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["request"]["data_handle"]["kind"], "data")
        self.assertEqual(calls[0]["handle"]["kind"], "data_view")
        self.assertNotEqual(calls[0]["request"]["data_handle"]["handle"], calls[0]["handle"]["handle"])
        self.assertEqual(result["handle"], calls[0]["handle"])
        self.assertEqual(result["sample_ids"], ["sample:1"])
        self.assertEqual(result["content_fingerprint"], "b" * 64)

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
