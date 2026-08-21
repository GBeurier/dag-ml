"""Mutation regression gate for the contract-only SAVE-001 profile."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.validate_archive_v1_contract import (
    ARCHIVE_ROOT,
    ArchiveContractError,
    apply_refusal_mutations,
    _set_path,
    _refs,
    classify_zip_dispatch,
    historical_legacy_fixture_bytes,
    historical_legacy_fixture_sha256,
    load_json,
    REQUIRED_REFUSAL_CASE_IDS,
    schema_validate,
    validate_archive_zip,
    validate_schema,
    validate_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_archive_v1_contract.py"


class ArchiveV1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = validate_schema(load_json(ARCHIVE_ROOT / "archive_workspace_manifest.v1.schema.json"))
        cls.workspace = load_json(ARCHIVE_ROOT / "fixtures/positive/workspace_n4d_host_sidecar.json")
        cls.archive = load_json(ARCHIVE_ROOT / "fixtures/positive/portable_split_conformal.json")

    def assert_refused(self, base: dict, path: list[object], value: object, prefix: str) -> None:
        mutated = copy.deepcopy(base)
        _set_path(mutated, path, value)
        with self.assertRaisesRegex(ArchiveContractError, rf"^{prefix}:"):
            schema_validate(mutated, self.validator)
            validate_semantics(mutated)

    def test_command_validates_schema_physical_profile_and_mutations(self) -> None:
        result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("physical profile, fixtures, and refusals", result.stdout)

    def test_reviewer_mutations_are_fail_closed(self) -> None:
        cases = [
            (self.workspace, ["schema_version"], 2, "schema_refusal"),
            (self.workspace, ["payloads", "host_artifacts", 0, "host_state"], "external_reference", "schema_refusal"),
            (self.workspace, ["payloads", "host_artifacts", 0, "load_policy"], "native_portable", "unsafe_host_load_refusal"),
            (self.workspace, ["member_inventory", 0, "path"], "C:/escape", "member_path_refusal"),
            (self.workspace, ["member_inventory", 0, "path"], "dagml\\escape", "member_path_refusal"),
            (self.workspace, ["member_inventory", 0, "path"], "dagml/../escape", "member_path_refusal"),
            (self.workspace, ["member_inventory", 0, "raw_sha256"], "b" * 64, "member_integrity_refusal"),
            (self.workspace, ["member_inventory", 0, "uncompressed_size_bytes"], 134217729, "budget_refusal"),
            (self.workspace, ["migration_provenance", "source_retained"], False, "schema_refusal"),
            (self.workspace, ["replay", "training_artifacts", "execution_bundle", "schema_version"], 1, "schema_refusal"),
            (self.workspace, ["replay", "training_artifacts", "training_outcome", "schema_version"], 1, "schema_refusal"),
            (self.workspace, ["replay", "training_artifacts", "prediction_cache_payload_set", "schema_version"], 1, "schema_refusal"),
            (self.workspace, ["replay", "training_artifacts", "score_set", "schema_version"], 1, "schema_refusal"),
            (self.workspace, ["replay", "training_artifacts", "execution_bundle", "producer_port_required"], False, "schema_refusal"),
            (self.archive, ["replay", "portable_predictor_package", "schema_id"], "https://example.invalid/v2", "schema_refusal"),
        ]
        for base, path, value, prefix in cases:
            with self.subTest(path=path):
                self.assert_refused(base, path, value, prefix)

    def test_legacy_migration_provenance_is_immutable(self) -> None:
        self.assertEqual(self.workspace["migration_provenance"]["legacy_format_version"], "1.0")
        self.assert_refused(self.workspace, ["migration_provenance", "copy_on_write"], False, "schema_refusal")
        self.assert_refused(self.workspace, ["migration_provenance", "source_raw_sha256"], None, "migration_refusal")

    def test_member_inventory_closure_and_workspace_snapshot_protocol(self) -> None:
        orphan = copy.deepcopy(self.workspace)
        orphan["member_inventory"].append({"path": "orphan.json", "regular_file": True, "raw_sha256": "c" * 64, "uncompressed_size_bytes": 1, "semantic_fingerprint": None, "semantic_profile": "none"})
        with self.assertRaisesRegex(ArchiveContractError, r"^member_inventory_refusal:"):
            schema_validate(orphan, self.validator)
            validate_semantics(orphan)
        self.assert_refused(self.workspace, ["workspace", "snapshot_protocol", "inventory_complete"], False, "schema_refusal")

    def test_aliases_profiles_host_ids_and_extensions_are_typed(self) -> None:
        alias = copy.deepcopy(self.workspace)
        package = alias["replay"]["portable_predictor_package"]
        outcome = alias["replay"]["training_artifacts"]["training_outcome"]
        for field in ("member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"):
            outcome[field] = package[field]
        with self.assertRaisesRegex(ArchiveContractError, r"^replay_alias_refusal:"):
            schema_validate(alias, self.validator)
            validate_semantics(alias)
        methods_alias = copy.deepcopy(self.archive)
        methods_alias["payloads"]["methods"]["n4mopt"][0]["member_path"] = methods_alias["payloads"]["methods"]["n4mm"][0]["member_path"]
        with self.assertRaisesRegex(ArchiveContractError, r"^methods_alias_refusal:"):
            schema_validate(methods_alias, self.validator)
            validate_semantics(methods_alias)
        self.assert_refused(self.workspace, ["replay", "training_artifacts", "graph", "semantic_profile"], "dagml_tcv1", "schema_refusal")
        self.assert_refused(self.workspace, ["replay", "training_artifacts", "graph", "semantic_fingerprint"], None, "schema_refusal")
        duplicate_host = copy.deepcopy(self.workspace)
        duplicate_host["payloads"]["host_artifacts"].append(copy.deepcopy(duplicate_host["payloads"]["host_artifacts"][0]))
        with self.assertRaisesRegex(ArchiveContractError, r"^host_artifact_refusal:"):
            schema_validate(duplicate_host, self.validator)
            validate_semantics(duplicate_host)
        extension = copy.deepcopy(self.workspace)
        extension["extensions"] = {"vendor.example": {"optional_hint": "ignored by V1 core"}}
        schema_validate(extension, self.validator)
        validate_semantics(extension)
        self.assert_refused(self.workspace, ["unexpected_core_field"], True, "schema_refusal")

    def test_runtime_replay_references_are_port_explicit_v2_without_downconversion(self) -> None:
        runtime = self.workspace["replay"]["training_artifacts"]
        expected = {
            "execution_bundle": "execution_bundle.v2.schema.json",
            "training_outcome": "training_outcome.v2.schema.json",
            "prediction_cache_payload_set": "prediction_cache_payload_set.v2.schema.json",
            "score_set": "score_set.v2.schema.json",
        }
        for name, schema_suffix in expected.items():
            with self.subTest(name=name):
                self.assertEqual(runtime[name]["schema_version"], 2)
                self.assertTrue(runtime[name]["producer_port_required"])
                self.assertTrue(runtime[name]["schema_id"].endswith(schema_suffix))
        self.assertEqual(self.workspace["replay"]["portable_predictor_package"]["schema_version"], 1)
        self.assertEqual(runtime["graph"]["schema_version"], 1)

    def _write_tiny_archive(
        self,
        target: Path,
        *,
        compression: int = zipfile.ZIP_STORED,
        extra: str | None = None,
        duplicate: str | None = None,
        non_regular: str | None = None,
        corrupt_member: str | None = None,
        truncate_member: str | None = None,
    ) -> None:
        document = copy.deepcopy(self.archive)
        payloads = {member["path"]: member["path"].encode("utf-8") for member in document["member_inventory"]}
        if compression == zipfile.ZIP_DEFLATED:
            payloads["dagml/portable_predictor_package.json"] = b"z" * 20_000
        raw_hashes = {path: hashlib.sha256(payload).hexdigest() for path, payload in payloads.items()}
        for member in document["member_inventory"]:
            member["raw_sha256"] = raw_hashes[member["path"]]
            member["uncompressed_size_bytes"] = len(payloads[member["path"]])
        for reference in _refs({"replay": document["replay"], "payloads": document["payloads"], "workspace": document["workspace"]}):
            reference["raw_sha256"] = raw_hashes[reference["member_path"]]
        with zipfile.ZipFile(target, "w", compression=compression) as archive:
            archive.writestr("manifest.json", json.dumps(document, separators=(",", ":")))
            for path, payload in payloads.items():
                if path == non_regular:
                    continue
                if path == corrupt_member:
                    payload = b"corrupt" if len(payload) != len(b"corrupt") else b"changed"
                if path == truncate_member:
                    payload = payload[:-1]
                archive.writestr(path, payload)
            if extra is not None:
                archive.writestr(extra, b"orphan")
            if duplicate is not None:
                archive.writestr(duplicate, payloads[duplicate])
            if non_regular is not None:
                info = zipfile.ZipInfo(non_regular)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b"target")

    def _write_materialized_archive(
        self,
        target: Path,
        document: dict,
        *,
        payload_overrides: dict[str, bytes] | None = None,
        preserve_inventory_fields: set[tuple[int, str]] | None = None,
    ) -> None:
        """Write one small regular-file ZIP while preserving the target mutation.

        The fixture manifests intentionally contain illustrative hashes and
        sizes.  A physical proof must instead use the actual ZIP bytes, except
        for the metadata field which is itself the refusal under test.
        """
        document = copy.deepcopy(document)
        payload_overrides = payload_overrides or {}
        preserve_inventory_fields = preserve_inventory_fields or set()
        payloads = {
            member["path"]: payload_overrides.get(member["path"], member["path"].encode("utf-8"))
            for member in document["member_inventory"]
        }
        raw_hashes = {path: hashlib.sha256(payload).hexdigest() for path, payload in payloads.items()}
        for index, member in enumerate(document["member_inventory"]):
            path = member["path"]
            if (index, "raw_sha256") not in preserve_inventory_fields:
                member["raw_sha256"] = raw_hashes[path]
            if (index, "uncompressed_size_bytes") not in preserve_inventory_fields:
                member["uncompressed_size_bytes"] = len(payloads[path])
        for reference in _refs({"replay": document["replay"], "payloads": document["payloads"], "workspace": document["workspace"]}):
            if reference["member_path"] in raw_hashes:
                reference["raw_sha256"] = raw_hashes[reference["member_path"]]
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(document, separators=(",", ":"), allow_nan=False))
            for path, payload in payloads.items():
                archive.writestr(path, payload)

    def _sqlite_snapshot_and_sidecars(self, directory: Path) -> dict[str, bytes]:
        """Capture genuine SQLite 3 database/WAL/SHM/rollback-journal bytes.

        All paths stay under TemporaryDirectory.  The returned bytes are copied
        before the connections close, so SQLite cleanup/checkpoint timing cannot
        affect the archive materialized by the caller.
        """
        database = directory / "store.sqlite"
        rollback = database.with_name(f"{database.name}-journal")
        wal = database.with_name(f"{database.name}-wal")
        shm = database.with_name(f"{database.name}-shm")
        connection = sqlite3.connect(database)
        try:
            connection.execute("CREATE TABLE proof (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO proof(value) VALUES ('committed')")
            connection.commit()
            self.assertEqual(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower(), "delete")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE proof SET value='uncommitted' WHERE id=1")
            self.assertTrue(rollback.is_file(), "SQLite did not materialize its rollback journal")
            rollback_bytes = rollback.read_bytes()
            self.assertTrue(rollback_bytes, "SQLite rollback journal was unexpectedly empty")
            connection.rollback()
            snapshot_bytes = database.read_bytes()
        finally:
            connection.close()

        writer = sqlite3.connect(database)
        reader = sqlite3.connect(database)
        try:
            self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower(), "wal")
            writer.execute("INSERT INTO proof(value) VALUES ('wal')")
            writer.commit()
            self.assertEqual(reader.execute("SELECT COUNT(*) FROM proof").fetchone()[0], 2)
            self.assertTrue(wal.is_file(), "SQLite did not materialize its WAL sidecar")
            self.assertTrue(shm.is_file(), "SQLite did not materialize its SHM sidecar")
            wal_bytes = wal.read_bytes()
            shm_bytes = shm.read_bytes()
            self.assertTrue(wal_bytes, "SQLite WAL was unexpectedly empty")
            self.assertTrue(shm_bytes, "SQLite SHM was unexpectedly empty")
        finally:
            reader.close()
            writer.close()
        self.assertTrue(snapshot_bytes.startswith(b"SQLite format 3\x00"))
        return {
            "workspace/store.sqlite": snapshot_bytes,
            "workspace/runs/run-a/store.sqlite-wal": wal_bytes,
            "workspace/runs/run-a/store.sqlite-shm": shm_bytes,
            "workspace/runs/run-a/store.sqlite-journal": rollback_bytes,
        }

    def _set_central_directory_size(self, target: Path, member_name: str, size: int) -> None:
        """Alter only a central-directory size field; payload bytes stay tiny."""
        payload = bytearray(target.read_bytes())
        cursor = 0
        while True:
            offset = payload.find(b"PK\x01\x02", cursor)
            self.assertNotEqual(offset, -1, f"missing central-directory member {member_name}")
            name_length, extra_length, comment_length = struct.unpack_from("<HHH", payload, offset + 28)
            name_start = offset + 46
            name_end = name_start + name_length
            if payload[name_start:name_end].decode("utf-8") == member_name:
                struct.pack_into("<I", payload, offset + 24, size)
                target.write_bytes(payload)
                return
            cursor = name_end + extra_length + comment_length

    def test_bounded_physical_zip_dispatch_and_central_directory_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive.n4a"
            self._write_tiny_archive(archive)
            self.assertEqual(classify_zip_dispatch(archive), "archive_v1")
            validate_archive_zip(archive, self.validator)
            orphan = root / "orphan.n4a"
            self._write_tiny_archive(orphan, extra="orphan.bin")
            with self.assertRaisesRegex(ArchiveContractError, r"^member_inventory_refusal:"):
                validate_archive_zip(orphan, self.validator)
            compressed = root / "compressed.n4a"
            self._write_tiny_archive(compressed, compression=zipfile.ZIP_DEFLATED)
            with self.assertRaisesRegex(ArchiveContractError, r"^compression_refusal:"):
                validate_archive_zip(compressed, self.validator)
            reserved = root / "reserved.n4a"
            self._write_tiny_archive(reserved, extra="CON.json")
            with self.assertRaisesRegex(ArchiveContractError, r"^member_path_refusal:"):
                validate_archive_zip(reserved, self.validator)
            duplicate = root / "duplicate.n4a"
            self._write_tiny_archive(duplicate, duplicate="dagml/portable_predictor_package.json")
            with self.assertRaisesRegex(ArchiveContractError, r"^member_inventory_refusal:"):
                validate_archive_zip(duplicate, self.validator)
            non_regular = root / "non-regular.n4a"
            self._write_tiny_archive(non_regular, non_regular="dagml/portable_predictor_package.json")
            with self.assertRaisesRegex(ArchiveContractError, r"^member_type_refusal:"):
                validate_archive_zip(non_regular, self.validator)
            hash_mismatch = root / "hash-mismatch.n4a"
            self._write_tiny_archive(hash_mismatch, corrupt_member="dagml/portable_predictor_package.json")
            with self.assertRaisesRegex(ArchiveContractError, r"^member_integrity_refusal:"):
                validate_archive_zip(hash_mismatch, self.validator)
            size_mismatch = root / "size-mismatch.n4a"
            self._write_tiny_archive(size_mismatch, truncate_member="dagml/portable_predictor_package.json")
            with self.assertRaisesRegex(ArchiveContractError, r"^member_integrity_refusal:"):
                validate_archive_zip(size_mismatch, self.validator)
            quota = root / "quota.n4a"
            self._write_tiny_archive(quota)
            self._set_central_directory_size(quota, "dagml/portable_predictor_package.json", 134_217_729)
            with self.assertRaisesRegex(ArchiveContractError, r"^budget_refusal:"):
                validate_archive_zip(quota, self.validator)
            legacy = root / "legacy.n4a"
            legacy.write_bytes(historical_legacy_fixture_bytes())
            self.assertEqual(classify_zip_dispatch(legacy), "legacy_n4a")

    def test_historical_legacy_fixture_is_loader_readable_and_provenance_bound(self) -> None:
        fixture = historical_legacy_fixture_bytes()
        self.assertEqual(
            hashlib.sha256(fixture).hexdigest(),
            historical_legacy_fixture_sha256(),
        )
        self.assertEqual(
            self.workspace["migration_provenance"]["source_raw_sha256"],
            historical_legacy_fixture_sha256(),
        )
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "historical-fixture.n4a"
            legacy.write_bytes(fixture)
            self.assertEqual(classify_zip_dispatch(legacy), "legacy_n4a")
            source_root = Path(os.environ.get("NIRS4ALL_LEGACY_REPO", ROOT.parent / "nirs4all"))
            self.assertTrue((source_root / "nirs4all/pipeline/bundle/loader.py").is_file())
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
            probe = (
                "from nirs4all.pipeline.bundle.loader import BundleLoader; "
                f"loader = BundleLoader({str(legacy)!r}); "
                "assert loader.metadata is not None; "
                "assert loader.metadata.bundle_format_version == '1.0'; "
                "assert loader.metadata.pipeline_uid == 'fixture:historical-loader'"
            )
            result = subprocess.run(
                [shutil.which("python3.11") or sys.executable, "-c", probe],
                cwd=source_root,
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_sqlite_live_transaction_and_temp_names_are_refused_as_ordinary_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sqlite_payloads = self._sqlite_snapshot_and_sidecars(Path(directory))
            for live_name in (
            "workspace/runs/run-a/store.sqlite-wal",
            "workspace/runs/run-a/store.sqlite-shm",
            "workspace/runs/run-a/store.sqlite-journal",
            "workspace/runs/run-a/store.sqlite-stmtjrnl",
            "workspace/runs/run-a/store.sqlite-mjA1B2C39FF",
            "workspace/runs/run-a/store.sqlite-mj A1B2C39FF",
            "workspace/runs/run-a/etilqs_abcdef",
            "workspace/runs/run-a/sqlite-tmp-abcdef",
            "workspace/runs/run-a/sqlite_temp_abcdef",
            ):
                with self.subTest(live_name=live_name):
                    document = copy.deepcopy(self.workspace)
                    document["member_inventory"].append({"path": live_name, "regular_file": True, "raw_sha256": "a" * 64, "uncompressed_size_bytes": 1, "semantic_fingerprint": None, "semantic_profile": "none"})
                    document["workspace"]["payload_inventory"].append({"kind": "ordinary", "run_id": "run:a", "member_path": live_name, "raw_sha256": "a" * 64, "semantic_fingerprint": None, "semantic_profile": "none"})
                    target = Path(directory) / f"{live_name.rsplit('/', 1)[-1]}.n4a"
                    payloads = dict(sqlite_payloads)
                    payloads[live_name] = payloads.get(live_name, b"sqlite-live-name-proof")
                    self._write_materialized_archive(target, document, payload_overrides=payloads)
                    with self.assertRaisesRegex(ArchiveContractError, r"^workspace_refusal:"):
                        validate_archive_zip(target, self.validator)

    def test_negative_refusal_manifest_is_frozen_and_each_invariant_refuses_physical_zip(self) -> None:
        refusals = load_json(ARCHIVE_ROOT / "fixtures/negative/refusals.v1.json")
        self.assertEqual({case["id"] for case in refusals["cases"]}, REQUIRED_REFUSAL_CASE_IDS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sqlite_payloads = self._sqlite_snapshot_and_sidecars(root)
            for case in refusals["cases"]:
                with self.subTest(case=case["id"]):
                    base = self.archive if case.get("base") == "portable_split_conformal.json" else self.workspace
                    mutated = apply_refusal_mutations(base, case)
                    protected: set[tuple[int, str]] = set()
                    for mutation in case.get("mutations", [{"mutation": case.get("mutation")}]):
                        path = mutation["mutation"]
                        if path[:1] == ["member_inventory"] and path[-1] in {"raw_sha256", "uncompressed_size_bytes"}:
                            protected.add((path[1], path[-1]))
                    payloads = sqlite_payloads if mutated["persistence_kind"] == "workspace_snapshot" else None
                    target = root / f"{case['id']}.n4a"
                    self._write_materialized_archive(
                        target,
                        mutated,
                        payload_overrides=payloads,
                        preserve_inventory_fields=protected,
                    )
                    with self.assertRaisesRegex(ArchiveContractError, rf"^{case['expected_error']}:"):
                        validate_archive_zip(target, self.validator)

    def test_workspace_inventory_signature_and_windows_path_refusals(self) -> None:
        workspace = copy.deepcopy(self.workspace)
        workspace["member_inventory"].append({"path": "workspace/untracked.bin", "regular_file": True, "raw_sha256": "f" * 64, "uncompressed_size_bytes": 1, "semantic_fingerprint": None, "semantic_profile": "none"})
        with self.assertRaisesRegex(ArchiveContractError, r"^member_inventory_refusal:"):
            schema_validate(workspace, self.validator)
            validate_semantics(workspace)
        self.assert_refused(self.workspace, ["workspace", "exclusions"], ["workspace/.session.lock"], "schema_refusal")
        self.assert_refused(self.workspace, ["member_inventory", 0, "path"], "CON.json", "member_path_refusal")
        malformed = copy.deepcopy(self.workspace)
        malformed["security"]["signature"] = {"status": "reserved_future_contract"}
        with self.assertRaisesRegex(ArchiveContractError, r"^schema_refusal:"):
            schema_validate(malformed, self.validator)
        reservation = copy.deepcopy(self.archive)
        reservation["security"]["signature"] = {
            "status": "reserved_future_contract",
            "manifest_sha256": "0" * 64,
            "canonical_profile": "archive_v1_manifest_json_canonical_v1",
            "preimage_rules": "utf8_json_sort_keys_compact_with_signature_null_v1",
            "algorithm": None,
            "key_id": None,
            "signature": None,
            "trust_root": None,
        }
        preimage = copy.deepcopy(reservation)
        preimage["security"]["signature"] = None
        reservation["security"]["signature"]["manifest_sha256"] = hashlib.sha256(
            json.dumps(preimage, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        schema_validate(reservation, self.validator)
        validate_semantics(reservation)


if __name__ == "__main__":
    unittest.main()
