from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.validate_archive_v2_contract import (
    ARCHIVE_ROOT,
    PACKAGE_MEMBER,
    PACKAGE_V1_SCHEMA_ID,
    PACKAGE_V2_SCHEMA_ID,
    ArchiveV2ContractError,
    canonical_portable_package_v2,
    contract_schema_registry,
    load_json,
    materialize_fixture,
    rebind_package_integrity,
    schema_validate,
    validate_archive_v2_contract,
    validate_archive_v2_payloads,
    validate_archive_zip,
    validate_package_portability,
    validate_package_schema_boundary,
    validate_schema,
    write_fixture_zip,
)


ROOT = Path(__file__).resolve().parents[1]

FROZEN_ARCHIVE_V1_SHA256 = {
    "README.md": "054131e92fa6160a677a3e50d295574d6be57643379b3e69ea77ffee0d7fbaa1",
    "THREAT_MODEL.md": "a7514441da79d1cb3a2f2f94a34bee51b9e5ff3da31f9490408041ca0f9250ae",
    "archive_workspace_manifest.v1.schema.json": "91daa7209843ab9043aa62a50200ff43b0f85f4c4e61ad8f73aa67b65a0a98dc",
    "fixtures/legacy/historical_n4a_manifest_v1.json": "10ba5276b2695215f71948aacfbf4c79322c6da8f481e6b8b6075846cb6279bb",
    "fixtures/negative/refusals.v1.json": "d6522da50d8debc87b5c824392d793fd8895e4822ebe80289199a246f502ded5",
    "fixtures/positive/portable_split_conformal.json": "79acef8a6bedee201c9e7be7a398bf7ec0ef6de2c75777824ec9ec0633b4c451",
    "fixtures/positive/workspace_n4d_host_sidecar.json": "6c3d678e955258a2f652c886d86358d4775dc98e9e85714166d88ad0e16b13ca",
}


class ArchiveV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_json(
            ARCHIVE_ROOT / "fixtures/positive/native_portable_replay.json"
        )
        cls.schema = load_json(
            ARCHIVE_ROOT / "archive_workspace_manifest.v2.schema.json"
        )
        cls.validator = validate_schema(cls.schema)
        cls.schemas, cls.registry = contract_schema_registry(ROOT)

    def test_complete_contract_gate(self) -> None:
        validate_archive_v2_contract(ROOT)

    def test_archive_v1_contract_bytes_remain_frozen(self) -> None:
        archive_v1 = ROOT / "docs/contracts/archive-v1"
        for relative, expected in FROZEN_ARCHIVE_V1_SHA256.items():
            with self.subTest(relative=relative):
                actual = hashlib.sha256((archive_v1 / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "scripts/validate_archive_v1_contract.py").read_bytes()
            ).hexdigest(),
            "b99ef21522c25326bbbabeef9e70b1a649fcaf85f6cf59bd3426f4b0cdcf5074",
        )

    def test_v2_manifest_is_closed_and_version_families_do_not_mix(self) -> None:
        schema_validate(self.manifest, self.validator)
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertEqual(self.manifest["profile"], "nirs4all.archive_workspace.v2")
        package = self.manifest["replay"]["portable_predictor_package"]
        self.assertEqual(package["schema_id"], PACKAGE_V2_SCHEMA_ID)
        self.assertEqual(package["schema_version"], 2)
        self.assertTrue(package["producer_port_required"])

        package_v1 = copy.deepcopy(self.manifest)
        package_v1["replay"]["portable_predictor_package"]["schema_id"] = (
            PACKAGE_V1_SCHEMA_ID
        )
        package_v1["replay"]["portable_predictor_package"]["schema_version"] = 1
        del package_v1["replay"]["portable_predictor_package"][
            "producer_port_required"
        ]
        with self.assertRaisesRegex(ArchiveV2ContractError, r"^schema_refusal:"):
            schema_validate(package_v1, self.validator)

    def test_package_v1_rejects_v2_null_keys_and_embeds_bundle_v1(self) -> None:
        validate_package_schema_boundary(ROOT)
        package_v1_schema = self.schemas[PACKAGE_V1_SCHEMA_ID]
        self.assertTrue(
            package_v1_schema["properties"]["execution_bundle"]["$ref"].endswith(
                "/execution_bundle.v1.schema.json"
            )
        )
        package_v1 = load_json(
            ROOT / "examples/fixtures/training/portable_predictor_package.v1.json"
        )
        validator = Draft202012Validator(package_v1_schema, registry=self.registry)
        self.assertEqual(list(validator.iter_errors(package_v1)), [])
        for field in ("conformal_calibration", "conformal_calibration_replay"):
            with self.subTest(field=field):
                mutated = copy.deepcopy(package_v1)
                mutated[field] = None
                self.assertTrue(list(validator.iter_errors(mutated)))
        bundle_v2 = copy.deepcopy(package_v1)
        bundle_v2["execution_bundle"]["schema_version"] = 2
        self.assertTrue(list(validator.iter_errors(bundle_v2)))

    def test_package_v2_schema_id_bundle_and_native_policy_are_exact(self) -> None:
        package_v2_schema = self.schemas[PACKAGE_V2_SCHEMA_ID]
        self.assertEqual(package_v2_schema["$id"], PACKAGE_V2_SCHEMA_ID)
        self.assertTrue(
            package_v2_schema["properties"]["execution_bundle"]["$ref"].endswith(
                "/execution_bundle.v2.schema.json"
            )
        )
        package = canonical_portable_package_v2(ROOT)
        errors = list(
            Draft202012Validator(
                package_v2_schema, registry=self.registry
            ).iter_errors(package)
        )
        self.assertEqual(errors, [])
        validate_package_portability(package)
        self.assertIsNone(package.get("conformal_calibration"))
        self.assertIsNone(package.get("conformal_calibration_replay"))
        raw_payloads = package["execution_bundle"]["raw_artifact_payloads"]
        self.assertEqual(
            set(raw_payloads),
            {binding["artifact_id"] for binding in package["artifact_bindings"]},
        )

    def test_package_v2_fixture_is_accepted_by_dagml_semantic_owner(self) -> None:
        fixture = (
            ROOT
            / "docs/contracts/archive-v2/fixtures/positive/"
            "portable_predictor_package.native_methods.v2.json"
        )
        result = subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "dag-ml-cli",
                "--",
                "validate-portable-predictor-package",
                str(fixture),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("valid portable predictor package", result.stdout)

    def test_dagml_semantic_owner_rejects_resigned_raw_payload_drift(self) -> None:
        from parity.conformal.oracle import fingerprint_without

        package = canonical_portable_package_v2(ROOT)
        artifact_id = "artifact:branch:b0.model:ridge:refit"
        package["execution_bundle"]["raw_artifact_payloads"][artifact_id][0] = 79
        package["package_fingerprint"] = fingerprint_without(
            package, "package_fingerprint"
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "resigned-raw-payload-drift.v2.json"
            fixture.write_text(
                json.dumps(package, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "dag-ml-cli",
                    "--",
                    "validate-portable-predictor-package",
                    str(fixture),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw artifact payload", result.stderr)

    def test_host_sidecar_is_refused_before_archive_write(self) -> None:
        document, payloads = materialize_fixture(self.manifest, ROOT)
        package = json.loads(payloads[PACKAGE_MEMBER])
        package["fitted_artifact_mode"] = "allow_host_sidecar"
        for binding in package["artifact_bindings"]:
            binding["load_mode"] = "host_sidecar"
        from parity.conformal.oracle import fingerprint_without

        package["package_fingerprint"] = fingerprint_without(
            package, "package_fingerprint"
        )
        payloads[PACKAGE_MEMBER] = json.dumps(
            package, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        rebind_package_integrity(document, payloads)
        with self.assertRaisesRegex(
            ArchiveV2ContractError, r"^host_artifact_refusal:"
        ):
            validate_archive_v2_payloads(
                document, payloads, self.validator, root=ROOT
            )

    def test_materialized_zip_binds_package_and_exact_n4mm_bytes(self) -> None:
        document, payloads = materialize_fixture(self.manifest, ROOT)
        package = json.loads(payloads[PACKAGE_MEMBER])
        for reference in document["payloads"]["methods"]["n4mm"]:
            self.assertEqual(
                payloads[reference["member_path"]],
                bytes(
                    package["execution_bundle"]["raw_artifact_payloads"]
                    [reference["artifact_id"]]
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "portable-v2.n4a"
            write_fixture_zip(archive, document, payloads)
            validate_archive_zip(archive, self.validator, root=ROOT)

            tampered = copy.deepcopy(payloads)
            tampered["methods/branch-b0-ridge.n4mm"] += b"tamper"
            tampered_archive = Path(directory) / "tampered-n4mm.n4a"
            write_fixture_zip(tampered_archive, document, tampered)
            with self.assertRaisesRegex(
                ArchiveV2ContractError, r"^member_integrity_refusal:"
            ):
                validate_archive_zip(tampered_archive, self.validator, root=ROOT)


if __name__ == "__main__":
    unittest.main()
