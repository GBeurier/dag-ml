from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.validate_archive_v3_contract import (
    ARCHIVE_ROOT,
    FIXED_MEMBERS,
    POSITIVE_NAME,
    REFUSALS_NAME,
    REFUSAL_CATEGORIES,
    REQUIRED_REFUSAL_CASE_IDS,
    SCHEMA_NAME,
    ArchiveV3ContractError,
    apply_mutations,
    load_json,
    load_json_bytes,
    synthetic_member_bytes,
    synthetic_recipe,
    validate_archive_v3_contract,
    validate_archive_zip,
    validate_document,
    validate_recipe_binding,
    validate_refusal_cases,
    validate_schema,
    write_fixture_zip,
)


ROOT = Path(__file__).resolve().parents[1]
N4MM_MEMBER = "methods/synthetic.n4mm"


class ArchiveV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_json(ARCHIVE_ROOT / SCHEMA_NAME)
        cls.validator = validate_schema(cls.schema)
        cls.manifest = load_json(ARCHIVE_ROOT / "fixtures/positive" / POSITIVE_NAME)
        cls.refusals = load_json(ARCHIVE_ROOT / "fixtures/negative" / REFUSALS_NAME)
        cls.recipe = synthetic_recipe(ROOT)
        cls.payloads = synthetic_member_bytes(cls.recipe)

    def test_complete_contract_gate(self) -> None:
        validate_archive_v3_contract(ROOT)

    def test_positive_manifest_is_exact_and_recipe_bound(self) -> None:
        validate_recipe_binding(self.manifest, self.recipe, self.payloads)
        validate_document(self.manifest, self.payloads, self.validator)
        inventory = {entry["path"]: entry for entry in self.manifest["member_inventory"]}
        self.assertEqual(set(inventory), {*FIXED_MEMBERS, N4MM_MEMBER})
        for path, data in self.payloads.items():
            self.assertEqual(inventory[path]["raw_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(inventory[path]["uncompressed_size_bytes"], len(data))

    def test_synthetic_members_are_marked_and_cover_hostile_payload_bytes(self) -> None:
        marker = self.recipe["marker_utf8"].encode("utf-8")
        self.assertIn(b"NOT A DAG-ML DOCUMENT OR METHODS MODEL", marker)
        for data in self.payloads.values():
            self.assertTrue(data.startswith(marker))
        self.assertIn("日本語".encode(), self.payloads["dagml/portable_refit_package.json"])
        self.assertIn(b"\x00\x00NUL", self.payloads["dagml/graph.json"])
        with self.assertRaises(UnicodeDecodeError):
            self.payloads["dagml/portable_refit_execution_bundle.json"].decode("utf-8")
        self.assertIn(b"\r\n\xef\xbb\xbf", self.payloads["dagml/portable_refit_outcome.json"])
        self.assertTrue(self.payloads[N4MM_MEMBER].endswith(bytes(range(256))))
        # Unicode lives only in payload bytes: every manifest path and id is ASCII.
        manifest_text = json.dumps(self.manifest, ensure_ascii=False)
        self.assertTrue(manifest_text.isascii())

    def test_zip_round_trip_preserves_exact_payload_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "storage-only-v3.n4a"
            write_fixture_zip(path, self.manifest, self.payloads)
            validate_archive_zip(path, self.validator)
            with zipfile.ZipFile(path) as archive:
                for member, data in self.payloads.items():
                    self.assertEqual(archive.read(member), data)
                self.assertEqual(
                    load_json_bytes(archive.read("manifest.json"), "manifest.json"),
                    self.manifest,
                )

    def test_zip_gate_refuses_physical_divergence(self) -> None:
        def zip_with(payloads: dict[str, bytes], manifest: dict | None = None) -> Path:
            directory = Path(self._tmp.name)
            path = directory / f"case-{len(list(directory.iterdir()))}.n4a"
            write_fixture_zip(path, manifest or self.manifest, payloads)
            return path

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tampered = dict(self.payloads)
        tampered[N4MM_MEMBER] += b"\x00"
        with self.assertRaisesRegex(ArchiveV3ContractError, "^member_integrity_refusal:"):
            validate_archive_zip(zip_with(tampered), self.validator)
        extra = dict(self.payloads)
        extra["extra/undeclared.bin"] = b"x"
        with self.assertRaisesRegex(ArchiveV3ContractError, "^member_inventory_refusal:"):
            validate_archive_zip(zip_with(extra), self.validator)
        missing = dict(self.payloads)
        del missing["dagml/portable_refit_outcome.json"]
        with self.assertRaisesRegex(ArchiveV3ContractError, "^member_inventory_refusal:"):
            validate_archive_zip(zip_with(missing), self.validator)
        future = copy.deepcopy(self.manifest)
        future["schema_version"] = 4
        with self.assertRaisesRegex(ArchiveV3ContractError, "^schema_refusal:"):
            validate_archive_zip(zip_with(self.payloads, future), self.validator)

    def test_manifest_json_refuses_duplicate_members(self) -> None:
        with self.assertRaisesRegex(ArchiveV3ContractError, "^json_refusal:"):
            load_json_bytes(b'{"schema_version":3,"schema_version":4}', "manifest.json")

    def test_every_refusal_case_is_exercised_with_its_category(self) -> None:
        validate_refusal_cases(self.manifest, self.payloads, self.refusals, self.validator)
        cases = self.refusals["cases"]
        self.assertEqual({case["expected_error"] for case in cases}, REFUSAL_CATEGORIES)
        self.assertLessEqual(REQUIRED_REFUSAL_CASE_IDS, {case["id"] for case in cases})
        # Mutation vocabulary is the V2 `op`/`path`/`value` array shape.
        for case in cases:
            for mutation in case.get("mutations", []):
                self.assertIn(mutation["op"], {"add", "replace", "delete", "append"})
                self.assertIsInstance(mutation["path"], list)

    def test_requested_version_and_family_cases_are_present(self) -> None:
        by_id = {case["id"]: case for case in self.refusals["cases"]}
        versions = {
            by_id[case_id]["mutations"][0]["value"]
            for case_id in ("future_archive_v4_is_refused", "future_archive_v999_is_refused")
        }
        self.assertEqual(versions, {4, 999})

    def test_refusal_harness_rejects_wrong_category_noop_and_dropped_cases(self) -> None:
        wrong = copy.deepcopy(self.refusals)
        wrong["cases"][0]["expected_error"] = "semantic_refusal"
        with self.assertRaisesRegex(ArchiveV3ContractError, "expected semantic_refusal"):
            validate_refusal_cases(self.manifest, self.payloads, wrong, self.validator)

        noop = copy.deepcopy(self.refusals)
        noop["cases"][0]["mutations"] = [
            {"op": "replace", "path": ["schema_version"], "value": 3}
        ]
        with self.assertRaisesRegex(ArchiveV3ContractError, "no-op"):
            validate_refusal_cases(self.manifest, self.payloads, noop, self.validator)

        passing = copy.deepcopy(self.refusals)
        passing["cases"][0]["mutations"] = [
            {"op": "replace", "path": ["archive_id"], "value": "archive:other"}
        ]
        with self.assertRaisesRegex(ArchiveV3ContractError, "unexpectedly passed"):
            validate_refusal_cases(self.manifest, self.payloads, passing, self.validator)

        dropped = copy.deepcopy(self.refusals)
        dropped["cases"] = [
            case for case in dropped["cases"] if case["id"] != "future_archive_v999_is_refused"
        ]
        with self.assertRaisesRegex(ArchiveV3ContractError, "required Archive V3 refusal cases"):
            validate_refusal_cases(self.manifest, self.payloads, dropped, self.validator)

    def test_recipe_drift_is_detected(self) -> None:
        drifted = dict(self.payloads)
        drifted[N4MM_MEMBER] = drifted[N4MM_MEMBER][:-1] + b"\x00"
        with self.assertRaisesRegex(ArchiveV3ContractError, "^member_integrity_refusal:"):
            validate_document(self.manifest, drifted, self.validator)
        manifest = apply_mutations(
            self.manifest,
            [
                {
                    "op": "replace",
                    "path": ["member_inventory", 0, "semantic_fingerprint"],
                    "value": "1" * 64,
                }
            ],
        )
        with self.assertRaisesRegex(ArchiveV3ContractError, "^fixture_refusal:"):
            validate_recipe_binding(manifest, self.recipe, self.payloads)

    def test_future_methods_abi_minor_is_not_a_storage_refusal(self) -> None:
        # Replay owns "newer ABI minor than the runtime"; storage only bounds
        # the integer. A large but valid minor must stay storable and loadable.
        manifest = apply_mutations(
            self.manifest,
            [
                {
                    "op": "replace",
                    "path": ["payloads", "methods", "n4mm", 0, "abi_min_minor"],
                    "value": 99,
                }
            ],
        )
        validate_document(manifest, self.payloads, self.validator)

    def test_known_schema_gap_n4me_and_role_pipelines_are_not_in_the_dag_v3_schema(self) -> None:
        # Core additionally supports N4ME and RolePipeline families in V3. The
        # owner schema is N4MM-only; this pins the gap (reported, not widened).
        for family in ("n4me", "role_pipelines"):
            manifest = copy.deepcopy(self.manifest)
            manifest["payloads"]["methods"][family] = []
            with self.assertRaisesRegex(ArchiveV3ContractError, "^schema_refusal:"):
                validate_document(manifest, self.payloads, self.validator)

    def test_forward_incompatible_dispatch_and_schema_stay_closed(self) -> None:
        properties = self.schema["properties"]
        self.assertEqual(properties["schema_version"], {"const": 3})
        self.assertEqual(properties["profile"], {"const": "nirs4all.archive_workspace.v3"})
        self.assertFalse(self.schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
