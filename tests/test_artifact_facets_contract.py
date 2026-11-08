from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.validate_artifact_facets_contract import (
    DEFAULT_CAPTURE,
    DEFAULT_CAPTURES,
    NATIVE_ROLE_CAPTURE,
    SCHEMA_PATH,
    ArtifactFacetsContractError,
    build_validator,
    load_json,
    validate_artifact_facets_contract,
    validate_instance,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_artifact_facets_contract.py"


class ArtifactFacetsContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_json(SCHEMA_PATH, "schema")
        cls.validator = build_validator(cls.schema)
        cls.capture = load_json(DEFAULT_CAPTURE, "capture")
        cls.embedded = next(i for i, a in enumerate(cls.capture["artifacts"]) if a["storage"]["kind"] == "embedded")
        cls.reference = next(i for i, a in enumerate(cls.capture["artifacts"]) if a["storage"]["kind"] == "reference")

    def fresh(self) -> dict:
        return copy.deepcopy(self.capture)

    def problems(self, doc: object) -> list[str]:
        return validate_instance(self.validator, doc)

    def assertRefused(self, doc: object, fragment: str = "") -> None:
        found = self.problems(doc)
        self.assertTrue(found, "mutation was accepted")
        self.assertTrue(any(fragment in p for p in found), found)

    def art(self, doc: dict, idx: int | None = None) -> dict:
        return doc["artifacts"][self.embedded if idx is None else idx]

    def test_actual_capture_passes(self) -> None:
        self.assertEqual(self.problems(self.fresh()), [])
        validate_artifact_facets_contract()

    def test_required_fields_absent(self) -> None:
        for field in ("identity", "requirements"):
            doc = self.fresh()
            del self.art(doc)[field]
            self.assertRefused(doc, field)
        for field in ("artifacts", "links"):
            doc = self.fresh()
            del doc[field]
            self.assertRefused(doc, field)

    def test_requirements(self) -> None:
        doc = self.fresh()
        self.art(doc)["requirements"] = None
        self.assertEqual(self.problems(doc), [])
        doc = self.fresh()
        self.art(doc)["requirements"] = {"artifact_id": "recipe"}
        self.assertEqual(self.problems(doc), [])
        for bad in ({}, {"artifact_id": ""}, {"artifact_id": 3}, {"artifact_id": "x", "extra": 1}, "recipe", []):
            doc = self.fresh()
            self.art(doc)["requirements"] = bad
            self.assertRefused(doc)

    def test_reference_root_sha256(self) -> None:
        doc = self.fresh()
        self.art(doc, self.reference)["storage"].pop("root_sha256", None)
        self.assertEqual(self.problems(doc), [])
        doc = self.fresh()
        self.art(doc, self.reference)["storage"]["root_sha256"] = "ab" * 32
        self.assertEqual(self.problems(doc), [])
        for bad in (None, 7, "AB" * 32, "ab" * 31, "zz" * 32, ""):
            doc = self.fresh()
            self.art(doc, self.reference)["storage"]["root_sha256"] = bad
            self.assertRefused(doc)

    def test_forbidden_enums(self) -> None:
        doc = self.fresh()
        self.art(doc)["kind"] = "calibration"
        self.assertRefused(doc)
        doc = self.fresh()
        doc["links"][0]["relation"] = "calibrates"
        self.assertRefused(doc)

    def test_extra_fields(self) -> None:
        for mutate in (
            lambda d: d.__setitem__("extra", 1),
            lambda d: self.art(d).__setitem__("extra", 1),
            lambda d: self.art(d)["storage"].__setitem__("extra", 1),
            lambda d: self.art(d, self.reference)["storage"].__setitem__("extra", 1),
            lambda d: self.art(d)["identity"].__setitem__("extra", 1),
            lambda d: self.art(d)["content_type"].__setitem__("extra", 1),
            lambda d: d["links"][0].__setitem__("extra", 1),
        ):
            doc = self.fresh()
            mutate(doc)
            self.assertRefused(doc)

    def test_wrong_shapes(self) -> None:
        for mutate in (
            lambda d: d.__setitem__("artifacts", {}),
            lambda d: d.__setitem__("links", "none"),
            lambda d: d.__setitem__("artifacts", [1]),
            lambda d: self.art(d).__setitem__("storage", []),
            lambda d: self.art(d).__setitem__("identity", "x"),
            lambda d: self.art(d).__setitem__("id", 5),
            lambda d: self.art(d).__setitem__("owner", "   "),
            lambda d: self.art(d)["storage"].__setitem__("member_paths", "a"),
            lambda d: self.art(d)["storage"].__setitem__("member_paths", []),
            lambda d: self.art(d)["storage"].__setitem__("member_paths", ["a", "a"]),
        ):
            doc = self.fresh()
            mutate(doc)
            self.assertRefused(doc)
        self.assertRefused([])
        self.assertRefused("text")

    def test_versions(self) -> None:
        for bad in (True, False, "1", 1.5, 0, -1, 4294967296):
            for holder in ("content_type", "identity"):
                key = "version" if holder == "content_type" else "schema_version"
                doc = self.fresh()
                self.art(doc)[holder][key] = bad
                self.assertRefused(doc, key)
        doc = self.fresh()
        self.art(doc)["content_type"]["version"] = 4294967295
        self.assertEqual(self.problems(doc), [])

    def test_remote_reference_is_never_fetched(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.invalid/local.schema.json",
            "type": "object",
            "properties": {"a": {"$ref": "https://example.invalid/remote.schema.json"}},
        }
        validator = build_validator(schema)
        with mock.patch("urllib.request.urlopen") as urlopen, mock.patch("socket.create_connection") as conn:
            with self.assertRaises(ArtifactFacetsContractError):
                validate_instance(validator, {"a": 1})
            urlopen.assert_not_called()
            conn.assert_not_called()

    def test_local_fragment_reference_works(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.invalid/local.schema.json",
            "$defs": {"n": {"type": "integer"}},
            "properties": {"a": {"$ref": "#/$defs/n"}},
        }
        validator = build_validator(schema)
        self.assertEqual(validate_instance(validator, {"a": 1}), [])
        self.assertTrue(validate_instance(validator, {"a": "x"}))

    def test_malformed_schema_refused(self) -> None:
        with self.assertRaises(ArtifactFacetsContractError):
            build_validator({"type": 12})

    def test_error_names_paths(self) -> None:
        doc = self.fresh()
        self.art(doc, 0)["kind"] = "calibration"
        message = self.problems(doc)[0]
        self.assertIn("$.artifacts[0].kind", message)
        self.assertIn("violates schema $", message)


class ArtifactFacetsCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, check=False
        )

    def write(self, directory: str, text: str) -> str:
        path = Path(directory) / "capture.json"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_default_capture_exit_zero(self) -> None:
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_option_accepts_valid_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, DEFAULT_CAPTURE.read_text(encoding="utf-8"))
            self.assertEqual(self.run_cli("--capture", path).returncode, 0)

    def test_capture_option_refuses_mutated_copy(self) -> None:
        doc = load_json(DEFAULT_CAPTURE, "capture")
        doc["artifacts"][0]["kind"] = "calibration"
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli("--capture", self.write(tmp, json.dumps(doc)))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("$.artifacts[0].kind", result.stderr)

    def test_absent_and_malformed_capture_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            absent = self.run_cli("--capture", str(Path(tmp) / "missing.json"))
            malformed = self.run_cli("--capture", self.write(tmp, "{not json"))
        for result in (absent, malformed):
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAILED", result.stderr)

    def test_absent_and_malformed_schema_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ArtifactFacetsContractError):
                validate_artifact_facets_contract(DEFAULT_CAPTURE, Path(tmp) / "missing.schema.json")
            bad = Path(tmp) / "bad.schema.json"
            bad.write_text("{", encoding="utf-8")
            with self.assertRaises(ArtifactFacetsContractError):
                validate_artifact_facets_contract(DEFAULT_CAPTURE, bad)
            invalid = Path(tmp) / "invalid.schema.json"
            invalid.write_text('{"type": 12}', encoding="utf-8")
            with self.assertRaises(ArtifactFacetsContractError):
                validate_artifact_facets_contract(DEFAULT_CAPTURE, invalid)

    def test_missing_dependency_fails_nonzero(self) -> None:
        code = (
            "import runpy, sys; sys.modules['jsonschema'] = None; "
            f"sys.argv = [{str(SCRIPT)!r}]; "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing dependency", result.stderr)


class ArtifactFacetsDefaultCaptureSetTests(unittest.TestCase):
    """The default gate must validate both committed captures; a copied project keeps originals untouched."""

    def make_project(self, tmp: str, *, native: str | None) -> Path:
        """Copy the script, schema and host capture into a temp project; ``native`` is the native_role text or None."""
        project = Path(tmp)
        for source in (SCRIPT, SCHEMA_PATH, DEFAULT_CAPTURE):
            target = project / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        if native is not None:
            target = project / NATIVE_ROLE_CAPTURE.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(native, encoding="utf-8")
        return project

    def run_copied(self, project: Path, *args: str) -> subprocess.CompletedProcess[str]:
        script = project / SCRIPT.relative_to(ROOT)
        return subprocess.run([sys.executable, str(script), *args], cwd=project, capture_output=True, text=True, check=False)

    def test_default_set_is_exactly_the_two_committed_instances(self) -> None:
        self.assertEqual(DEFAULT_CAPTURES, (DEFAULT_CAPTURE, NATIVE_ROLE_CAPTURE))
        self.assertEqual(
            DEFAULT_CAPTURE.relative_to(ROOT).as_posix(),
            "crates/dag-ml-core/tests/fixtures/artifact_facets/capture/facets.json",
        )
        self.assertEqual(
            NATIVE_ROLE_CAPTURE.relative_to(ROOT).as_posix(),
            "crates/dag-ml-core/tests/fixtures/artifact_facets/native_role/capture/facets.json",
        )

    def test_default_helper_validates_both_instances(self) -> None:
        loaded: list[Path] = []
        real_load = load_json

        def spy(path: Path, label: str) -> object:
            loaded.append(path)
            return real_load(path, label)

        with mock.patch("scripts.validate_artifact_facets_contract.load_json", side_effect=spy):
            validated = validate_artifact_facets_contract()
        self.assertEqual(validated, list(DEFAULT_CAPTURES))
        self.assertCountEqual(loaded, [SCHEMA_PATH, *DEFAULT_CAPTURES])

    def test_default_cli_validates_and_names_both_captures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            native = NATIVE_ROLE_CAPTURE.read_text(encoding="utf-8")
            project = self.make_project(tmp, native=native)
            result = self.run_copied(project)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 capture(s)", result.stdout)
        self.assertIn(DEFAULT_CAPTURE.relative_to(ROOT).as_posix(), result.stdout)
        self.assertIn(NATIVE_ROLE_CAPTURE.relative_to(ROOT).as_posix(), result.stdout)

    def test_default_cli_fails_when_native_capture_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, native=None)
            result = self.run_copied(project)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stderr)
        self.assertIn("absent", result.stderr)
        self.assertIn("native_role", result.stderr)

    def test_default_cli_fails_when_native_capture_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, native="{not json")
            result = self.run_copied(project)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("malformed", result.stderr)

    def test_default_cli_fails_when_native_capture_violates_schema(self) -> None:
        doc = load_json(NATIVE_ROLE_CAPTURE, "capture")
        doc["artifacts"][0]["kind"] = "calibration"
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, native=json.dumps(doc))
            result = self.run_copied(project)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native_role", result.stderr)
        self.assertIn("$.artifacts[0].kind", result.stderr)

    def test_explicit_capture_selects_single_instance_without_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, native=None)
            host = project / DEFAULT_CAPTURE.relative_to(ROOT)
            result = self.run_copied(project, "--capture", str(host))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 capture(s)", result.stdout)
        self.assertNotIn("native_role", result.stdout)


if __name__ == "__main__":
    unittest.main()
