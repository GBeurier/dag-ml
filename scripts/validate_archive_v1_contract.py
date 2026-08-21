#!/usr/bin/env python3
"""Validate the contract-only SAVE-001 archive/workspace V1 freeze.

This is deliberately not a ZIP reader.  SAVE-002 must implement the physical
reader described by the profile; this gate makes the profile and its refusal
vocabulary executable without accepting, extracting, or loading an archive.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import stat
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "docs/contracts/archive-v1"
SCHEMA_NAME = "archive_workspace_manifest.v1.schema.json"
LEGACY_FIXTURE_NAME = "historical_n4a_manifest_v1.json"

# This set is deliberately frozen: adding a new fail-closed rule requires a
# corresponding executable mutation.  Keep it in sync with refusals.v1.json.
REQUIRED_REFUSAL_CASE_IDS = frozenset(
    {
        "future_v2_is_not_accepted",
        "implementation_status_is_not_wire",
        "external_reference_is_never_a_v1_host_state",
        "unsafe_joblib_needs_explicit_opt_in",
        "windows_separator_is_refused",
        "dotdot_path_is_refused",
        "inventory_hash_must_match_reference",
        "bounded_member_metadata_is_enforced",
        "legacy_import_retains_source",
        "legacy_source_hash_matches_fixture",
        "runtime_execution_bundle_must_be_v2",
        "runtime_training_outcome_must_be_v2",
        "runtime_prediction_cache_payload_set_must_be_v2",
        "runtime_score_set_must_be_v2",
        "runtime_v2_references_require_producer_port",
        "workspace_lock_is_not_snapshotted",
        "sqlite_wal_is_never_snapshotted",
        "sqlite_shm_is_never_snapshotted",
        "sqlite_rollback_journal_is_never_snapshotted",
        "sqlite_statement_journal_is_never_snapshotted",
        "sqlite_super_journal_is_never_snapshotted",
        "sqlite_temp_file_is_never_snapshotted",
        "portable_package_schema_is_not_arbitrary",
        "archive_cannot_embed_workspace",
    }
)

RUNTIME_V2_ARTIFACTS = {
    "execution_bundle": "https://github.com/GBeurier/dag-ml/schemas/execution_bundle.v2.schema.json",
    "training_outcome": "https://github.com/GBeurier/dag-ml/schemas/training_outcome.v2.schema.json",
    "prediction_cache_payload_set": "https://github.com/GBeurier/dag-ml/schemas/prediction_cache_payload_set.v2.schema.json",
    "score_set": "https://github.com/GBeurier/dag-ml/schemas/score_set.v2.schema.json",
}
SQLITE_SNAPSHOT_PATH = re.compile(r"^workspace/[A-Za-z0-9][A-Za-z0-9_.-]*\.sqlite$")
RUN_SCOPED_WORKSPACE_PATH = re.compile(r"^workspace/runs/[A-Za-z0-9][A-Za-z0-9_.-]*/.+$")


class ArchiveContractError(RuntimeError):
    """Stable SAVE-001 refusal (also the future reader error namespace)."""


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ArchiveContractError(f"{code}: {detail}")


def _reject_constant(token: str) -> None:
    raise ArchiveContractError(f"json_refusal: non-standard JSON constant `{token}`")


def load_json(path: Path) -> Any:
    def unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "json_refusal", f"duplicate member `{key}` in {path}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=unique_members, parse_constant=_reject_constant)
    except FileNotFoundError as exc:
        raise ArchiveContractError(f"json_refusal: missing file {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveContractError(f"json_refusal: invalid JSON in {path}: {exc}") from exc


def load_json_bytes(payload: bytes, label: str) -> Any:
    """Parse one ZIP dispatch member with the same strict JSON rules."""
    def unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "json_refusal", f"duplicate member `{key}` in {label}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_members, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveContractError(f"json_refusal: invalid JSON in {label}: {exc}") from exc


def historical_legacy_fixture_bytes(archive_root: Path = ARCHIVE_ROOT) -> bytes:
    """Return the retained, deterministic, loader-readable historical ZIP.

    The only payload is a public, synthetic historical manifest.  It contains
    no user data or serialized model and has a fixed ZIP timestamp/attributes,
    so its raw SHA-256 is a reproducible migration-provenance target.
    """
    source = archive_root / "fixtures/legacy" / LEGACY_FIXTURE_NAME
    try:
        manifest_bytes = source.read_bytes()
    except OSError as exc:
        raise ArchiveContractError(f"fixture_refusal: missing legacy fixture source {source}") from exc
    manifest = load_json_bytes(manifest_bytes, str(source))
    require(isinstance(manifest, dict), "fixture_refusal", "legacy fixture manifest must be an object")
    require(manifest.get("bundle_format_version") == "1.0", "fixture_refusal", "legacy fixture must retain bundle format 1.0")

    output = io.BytesIO()
    info = zipfile.ZipInfo("manifest.json", date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(info, manifest_bytes)
    return output.getvalue()


def historical_legacy_fixture_sha256(archive_root: Path = ARCHIVE_ROOT) -> str:
    """Return the raw-byte migration identity of the synthetic historical ZIP."""
    return hashlib.sha256(historical_legacy_fixture_bytes(archive_root)).hexdigest()


def _error_path(error: ValidationError) -> str:
    return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)


def validate_schema(schema: Any) -> Draft202012Validator:
    require(isinstance(schema, dict), "schema_refusal", "schema must be an object")
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "schema_refusal", "schema must declare Draft 2020-12")
    require(schema.get("$id") == "https://github.com/GBeurier/dag-ml/contracts/archive-v1/archive_workspace_manifest.v1.schema.json", "schema_refusal", "schema id drifted")
    require(schema.get("properties", {}).get("schema_version", {}).get("const") == 1, "schema_refusal", "V1 accepts only version 1")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ArchiveContractError(f"schema_refusal: {exc.message}") from exc
    return Draft202012Validator(schema)


def schema_validate(document: Any, validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        raise ArchiveContractError(f"schema_refusal: {_error_path(error)}: {error.message}")


def _safe_member_path(path: str) -> bool:
    if not path or path != unicodedata.normalize("NFC", path):
        return False
    if "\\" in path or ":" in path or any(ord(char) <= 0x1F or ord(char) == 0x7F for char in path):
        return False
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return False
    reserved = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE)
    return all(
        part not in {"", ".", ".."}
        and not part.endswith((".", " "))
        and not reserved.fullmatch(part)
        for part in path.split("/")
    )


def _refs(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"} <= value.keys():
            found.append(value)
        for nested in value.values():
            found.extend(_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_refs(nested))
    return found


def _closed_reference(reference: dict[str, Any], expected: set[str], label: str) -> None:
    require(set(reference) == expected, "typed_refusal", f"{label} has unknown or missing typed fields")


def _is_sqlite_live_state(path: str) -> bool:
    """Recognize SQLite files that can encode a live or incomplete transaction."""
    name = path.rsplit("/", 1)[-1].lower()
    return (
        name.endswith(("-wal", "-shm", "-journal", "-stmtjrnl"))
        or bool(re.search(r"-mj ?[0-9a-f]+$", name))
        or name.startswith(("etilqs_", "sqlite-tmp-", "sqlite_temp_"))
    )


def _is_allowed_workspace_payload_path(path: str, kind: str) -> bool:
    """Allow only the frozen SQLite root snapshot or run-scoped payloads."""
    return bool(SQLITE_SNAPSHOT_PATH.fullmatch(path)) if kind == "sqlite" else bool(RUN_SCOPED_WORKSPACE_PATH.fullmatch(path))


def validate_semantics(document: Any, *, legacy_fixture_sha256: str | None = None) -> None:
    """Enforce closure and physical-profile rules JSON Schema cannot express."""
    inventory = document["member_inventory"]
    limits = document["physical_profile"]["limits"]
    require(len(inventory) + 1 <= limits["max_entries"], "budget_refusal", "member count exceeds dispatch budget")
    paths: dict[str, dict[str, Any]] = {}
    total = 0
    for member in inventory:
        path = member["path"]
        require(_safe_member_path(path), "member_path_refusal", f"unsafe or non-canonical POSIX path `{path}`")
        require(path != "manifest.json", "member_inventory_refusal", "manifest.json is the dispatch member, not a self-hashed payload")
        require(path not in paths, "member_inventory_refusal", f"duplicate or normalization-colliding member `{path}`")
        require(member["regular_file"] is True, "member_type_refusal", f"member `{path}` is not a regular file")
        size = member["uncompressed_size_bytes"]
        require(size <= limits["max_member_uncompressed_bytes"], "budget_refusal", f"member `{path}` exceeds uncompressed budget")
        total += size
        paths[path] = member
    require(total <= limits["max_total_uncompressed_bytes"], "budget_refusal", "inventory exceeds total uncompressed budget")

    replay = document["replay"]
    required_replay_paths = [
        replay["portable_predictor_package"]["member_path"],
        *(reference["member_path"] for reference in replay["training_artifacts"].values()),
    ]
    require(
        len(required_replay_paths) == len(set(required_replay_paths)),
        "replay_alias_refusal",
        "portable package, graph, bundle, outcome, cache, and scores must use distinct members",
    )
    methods = document["payloads"]["methods"]
    method_paths = [reference["member_path"] for reference in methods["n4mm"] + methods["n4mopt"]]
    require(len(method_paths) == len(set(method_paths)), "methods_alias_refusal", "N4MM and N4MOPT members cannot alias")
    references = _refs({"replay": replay, "payloads": document["payloads"], "workspace": document["workspace"]})
    used: set[str] = set()
    for reference in references:
        path = reference["member_path"]
        require(_safe_member_path(path), "member_path_refusal", f"unsafe reference path `{path}`")
        member = paths.get(path)
        require(member is not None, "member_inventory_refusal", f"reference `{path}` is absent from closed inventory")
        require(member["raw_sha256"] == reference["raw_sha256"], "member_integrity_refusal", f"raw SHA-256 mismatch for `{path}`")
        require(member["semantic_profile"] == reference["semantic_profile"] and member["semantic_fingerprint"] == reference["semantic_fingerprint"], "member_semantic_refusal", f"semantic profile/fingerprint mismatch for `{path}`")
        used.add(path)
    require(set(paths) == used, "member_inventory_refusal", "closed inventory contains an unreferenced payload")

    dagml_v1_fields = {"owner", "schema_id", "schema_version", "member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"}
    dagml_v2_fields = dagml_v1_fields | {"producer_port_required"}
    _closed_reference(replay["portable_predictor_package"], dagml_v1_fields, "PortablePredictorPackage")
    for name, reference in replay["training_artifacts"].items():
        if name == "graph":
            _closed_reference(reference, dagml_v1_fields, f"training artifact {name}")
            continue
        _closed_reference(reference, dagml_v2_fields, f"training artifact {name}")
        require(name in RUNTIME_V2_ARTIFACTS, "replay_refusal", f"unknown mandatory runtime artifact `{name}`")
        require(reference["schema_version"] == 2, "replay_refusal", f"runtime artifact `{name}` must remain V2")
        require(reference["schema_id"] == RUNTIME_V2_ARTIFACTS[name], "replay_refusal", f"runtime artifact `{name}` has a mismatched V2 schema id")
        require(reference["producer_port_required"] is True, "producer_port_refusal", f"runtime artifact `{name}` must require producer_port")
    require(
        replay["training_artifacts"]["graph"]["semantic_profile"] == "dagml_historical_serde_json_v1",
        "semantic_profile_refusal",
        "graph must use the historical DAG-ML serde JSON fingerprint profile",
    )
    for label, reference in [("portable predictor package", replay["portable_predictor_package"]), *replay["training_artifacts"].items()]:
        require(reference["semantic_fingerprint"] is not None, "semantic_fingerprint_refusal", f"{label} requires a semantic fingerprint")
    methods_fields = {"kind", "owner", "format_version", "abi_major", "member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"}
    for reference in methods["n4mm"] + methods["n4mopt"]:
        _closed_reference(reference, methods_fields, "Methods reference")
        require(reference["semantic_fingerprint"] is not None, "semantic_fingerprint_refusal", "Methods references require semantic fingerprints")
    n4d = document["payloads"]["n4d_aggregate_reference"]
    if n4d is not None:
        _closed_reference(n4d, {"kind", "owner", "interpretation", "member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"}, "N4D aggregate reference")
    for label in ("conformal", "robustness"):
        reference = document["payloads"][label]
        if reference is not None:
            _closed_reference(reference, dagml_v1_fields, label)
    host_fields = {"artifact_id", "host_state", "serialization_backend", "load_policy", "controller_id", "controller_version", "plugin_id", "plugin_version", "runtime_id", "abi_id", "capability_id", "member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"}
    for reference in document["payloads"]["host_artifacts"]:
        _closed_reference(reference, host_fields, "host artifact")
    host_ids = [host["artifact_id"] for host in document["payloads"]["host_artifacts"]]
    require(len(host_ids) == len(set(host_ids)), "host_artifact_refusal", "host artifact ids must be unique")
    require(replay["portable_predictor_package"]["schema_version"] == 1, "replay_refusal", "PortablePredictorPackage must be exact V1")
    require(not any(item["affects_replay"] for item in replay["future_artifacts"]), "replay_refusal", "a deferred artifact cannot be required for replay")
    for host in document["payloads"]["host_artifacts"]:
        require(host["host_state"] != "external_reference", "external_reference_refusal", "external_reference is forbidden in .n4a V1")
        unsafe = host["serialization_backend"] in {"pickle", "joblib", "rds"}
        require(not unsafe or host["load_policy"] == "host_opt_in", "unsafe_host_load_refusal", "code-bearing host artifact requires host_opt_in")
        require(host["host_state"] != "native_portable" or host["load_policy"] == "native_portable", "host_artifact_policy_refusal", "native artifact requires native_portable policy")
    migration = document["migration_provenance"]
    if migration is not None and migration["legacy_format"] is not None:
        require(migration["source_raw_sha256"] is not None and migration["legacy_format_version"] is not None, "migration_refusal", "legacy import requires source hash and version")
        require(migration["copy_on_write"] and migration["source_retained"], "migration_refusal", "legacy import must retain immutable source copy")
        if legacy_fixture_sha256 is not None:
            require(
                migration["source_raw_sha256"] == legacy_fixture_sha256,
                "migration_refusal",
                "legacy migration source SHA-256 must equal the retained historical fixture bytes",
            )
    signature = document["security"]["signature"]
    if signature is not None:
        _closed_reference(signature, {"status", "manifest_sha256", "canonical_profile", "preimage_rules", "algorithm", "key_id", "signature", "trust_root"}, "signature reservation")
        require(all(signature[key] is None for key in ("algorithm", "key_id", "signature", "trust_root")), "signature_refusal", "SAVE-001 reserves metadata only and makes no completed-signature claim")
        preimage = copy.deepcopy(document)
        preimage["security"]["signature"] = None
        canonical = json.dumps(preimage, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        require(signature["manifest_sha256"] == hashlib.sha256(canonical).hexdigest(), "signature_refusal", "manifest SHA-256 does not match the reserved canonical preimage")

    workspace = document["workspace"]
    if document["persistence_kind"] == "workspace_snapshot":
        require(workspace is not None, "workspace_refusal", "workspace snapshot requires snapshot protocol")
        require(workspace["exclusions"] == ["workspace/.session.lock", "workspace/live-session/**"], "workspace_refusal", "live exclusions must be exact")
        snapshot = workspace["snapshot_protocol"]
        declared_runs = set(snapshot["run_ids"])
        payloads = workspace["payload_inventory"]
        payload_paths = [payload["member_path"] for payload in payloads]
        require(len(payload_paths) == len(set(payload_paths)), "workspace_refusal", "workspace payload inventory has duplicate members")
        for path in payload_paths:
            require(
                not _is_sqlite_live_state(path),
                "workspace_refusal",
                f"SQLite live transaction or temporary state `{path}` cannot be snapshotted",
            )
        sqlite_count = 0
        for payload in payloads:
            _closed_reference(payload, {"kind", "run_id", "member_path", "raw_sha256", "semantic_fingerprint", "semantic_profile"}, "workspace payload")
            path = payload["member_path"]
            require(path.startswith("workspace/"), "workspace_refusal", "workspace payload must live under workspace/")
            require(
                _is_allowed_workspace_payload_path(path, payload["kind"]),
                "workspace_refusal",
                f"workspace payload `{path}` is outside the frozen snapshot allowlist",
            )
            require(path not in workspace["exclusions"] and not path.startswith("workspace/live-session/"), "workspace_refusal", "live workspace state cannot be snapshotted")
            if payload["kind"] == "sqlite":
                sqlite_count += 1
                require(payload["run_id"] is None and path.endswith(".sqlite"), "workspace_refusal", "SQLite snapshot is workspace-wide")
            else:
                require(payload["run_id"] in declared_runs, "workspace_refusal", "workspace payload run_id is not declared by checkpoint")
            if payload["kind"] == "parquet":
                require(path.endswith(".parquet"), "workspace_refusal", "Parquet payload must use .parquet")
        require(sqlite_count == 1, "workspace_refusal", "workspace snapshot requires exactly one SQLite payload")
        actual_workspace_paths = {path for path in paths if path.startswith("workspace/")}
        require(actual_workspace_paths == set(payload_paths), "workspace_refusal", "workspace inventory omits or invents a workspace payload")
    else:
        require(workspace is None, "workspace_refusal", "n4a archive cannot embed a live workspace")


def _set_path(document: Any, path: list[Any], value: Any) -> None:
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value


def apply_refusal_mutations(document: Any, case: dict[str, Any]) -> Any:
    """Apply one frozen negative fixture case without validating it."""
    mutated = copy.deepcopy(document)
    case_id = case.get("id", "<unnamed>")
    mutations = case.get("mutations")
    if mutations is None:
        mutations = [{"mutation": case.get("mutation"), "value": case.get("value")}]
    require(isinstance(mutations, list) and mutations, "fixture_refusal", f"{case_id} mutations are required")
    for mutation in mutations:
        require(isinstance(mutation, dict), "fixture_refusal", f"{case_id} mutation must be an object")
        path = mutation.get("mutation")
        require(isinstance(path, list) and path, "fixture_refusal", f"{case_id} mutation path is invalid")
        operation = mutation.get("operation", "set")
        if operation == "set":
            _set_path(mutated, path, mutation.get("value"))
        elif operation == "append":
            target = mutated
            for component in path:
                target = target[component]
            require(isinstance(target, list), "fixture_refusal", f"{case_id} append target is not a list")
            target.append(mutation.get("value"))
        elif operation == "remove":
            target = mutated
            for component in path[:-1]:
                target = target[component]
            require(path[-1] in target, "fixture_refusal", f"{case_id} remove target is missing")
            del target[path[-1]]
        else:
            raise ArchiveContractError(f"fixture_refusal: {case_id} has unknown mutation operation `{operation}`")
    return mutated


def _expected(error: ArchiveContractError, code: str) -> bool:
    return str(error).startswith(f"{code}:")


def classify_zip_dispatch(path: Path) -> str:
    """Classify only the manifest shape; legacy payload loading stays external."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            manifests = [info for info in infos if info.filename == "manifest.json"]
            require(len(manifests) == 1, "dispatch_refusal", "ZIP must contain exactly one manifest.json")
            require(manifests[0].file_size <= 1_048_576, "budget_refusal", "dispatch manifest exceeds bootstrap budget")
            require(manifests[0].compress_size > 0 or manifests[0].file_size == 0, "compression_refusal", "dispatch manifest has invalid compressed size")
            if manifests[0].file_size:
                require(manifests[0].file_size / manifests[0].compress_size <= 100, "compression_refusal", "dispatch manifest exceeds bootstrap compression ratio")
            manifest = load_json_bytes(archive.read(manifests[0]), f"{path}!manifest.json")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveContractError(f"zip_refusal: cannot open ZIP {path}: {exc}") from exc
    if isinstance(manifest, dict) and manifest.get("profile") == "nirs4all.archive_workspace.v1":
        return "archive_v1"
    if isinstance(manifest, dict) and manifest.get("bundle_format_version") == "1.0":
        return "legacy_n4a"
    raise ArchiveContractError("dispatch_refusal: manifest is neither archive V1 nor the retained historical n4a shape")


def _validate_zip_info(info: zipfile.ZipInfo, limits: dict[str, int]) -> None:
    require(_safe_member_path(info.filename), "member_path_refusal", f"unsafe ZIP member path `{info.filename}`")
    require(not info.is_dir() and not (info.external_attr & 0x10), "member_type_refusal", f"ZIP member `{info.filename}` is a directory")
    mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(mode)
    require(file_type in {0, stat.S_IFREG}, "member_type_refusal", f"ZIP member `{info.filename}` is not a regular file")
    require(info.file_size <= limits["max_member_uncompressed_bytes"], "budget_refusal", f"ZIP member `{info.filename}` exceeds uncompressed budget")
    require(info.compress_size > 0 or info.file_size == 0, "compression_refusal", f"ZIP member `{info.filename}` has invalid compressed size")
    if info.file_size:
        require(info.file_size / info.compress_size <= limits["max_compression_ratio"], "compression_refusal", f"ZIP member `{info.filename}` exceeds compression ratio")


def validate_archive_zip(
    path: Path,
    validator: Draft202012Validator | None = None,
    *,
    legacy_fixture_sha256: str | None = None,
) -> None:
    """Validate a bounded physical fixture ZIP without extracting it to disk.

    This is an executable SAVE-001 contract harness, not the promised SAVE-002
    archive reader. It checks central-directory metadata before reading payloads.
    """
    validator = validator or validate_schema(load_json(ARCHIVE_ROOT / SCHEMA_NAME))
    # A physical archive must be held to the same immutable retained-source
    # identity as the JSON fixture gate.  Do not let the ZIP path silently
    # skip the migration-provenance proof.
    if legacy_fixture_sha256 is None:
        legacy_fixture_sha256 = historical_legacy_fixture_sha256()
    require(classify_zip_dispatch(path) == "archive_v1", "dispatch_refusal", "expected an archive V1 fixture")
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = load_json_bytes(archive.read("manifest.json"), f"{path}!manifest.json")
            schema_validate(manifest, validator)
            validate_semantics(manifest, legacy_fixture_sha256=legacy_fixture_sha256)
            limits = manifest["physical_profile"]["limits"]
            infos = archive.infolist()
            require(len(infos) <= limits["max_entries"], "budget_refusal", "ZIP entry count exceeds budget")
            names = [info.filename for info in infos]
            require(len(names) == len(set(names)), "member_inventory_refusal", "ZIP central directory has duplicate names")
            require(names.count("manifest.json") == 1, "dispatch_refusal", "ZIP has an invalid manifest member")
            for info in infos:
                _validate_zip_info(info, limits)
            total = sum(info.file_size for info in infos)
            require(total <= limits["max_total_uncompressed_bytes"], "budget_refusal", "ZIP exceeds total uncompressed budget")
            inventory = {member["path"]: member for member in manifest["member_inventory"]}
            require(set(names) == {"manifest.json", *inventory}, "member_inventory_refusal", "ZIP central directory differs from the closed inventory")
            for info in infos:
                if info.filename == "manifest.json":
                    continue
                expected = inventory[info.filename]
                require(info.file_size == expected["uncompressed_size_bytes"], "member_integrity_refusal", f"ZIP size mismatch for `{info.filename}`")
                require(hashlib.sha256(archive.read(info)).hexdigest() == expected["raw_sha256"], "member_integrity_refusal", f"ZIP SHA-256 mismatch for `{info.filename}`")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveContractError(f"zip_refusal: cannot validate ZIP {path}: {exc}") from exc


def validate_archive_v1_contract(root: Path = ROOT) -> None:
    archive_root = root / "docs/contracts/archive-v1"
    validator = validate_schema(load_json(archive_root / SCHEMA_NAME))
    positives = {path.name: load_json(path) for path in sorted((archive_root / "fixtures/positive").glob("*.json"))}
    require(set(positives) == {"portable_split_conformal.json", "workspace_n4d_host_sidecar.json"}, "fixture_refusal", "positive fixtures must freeze archive and workspace forms")
    legacy_fixture_sha256 = historical_legacy_fixture_sha256(archive_root)
    for document in positives.values():
        schema_validate(document, validator)
        validate_semantics(document, legacy_fixture_sha256=legacy_fixture_sha256)
    negatives = load_json(archive_root / "fixtures/negative/refusals.v1.json")
    cases = negatives.get("cases") if isinstance(negatives, dict) else None
    require(negatives.get("schema_version") == 1 and isinstance(cases, list) and cases, "fixture_refusal", "negative cases are required")
    ids: set[str] = set()
    for case in cases:
        case_id, base = case.get("id"), case.get("base", "workspace_n4d_host_sidecar.json")
        require(isinstance(case_id, str) and case_id not in ids and base in positives, "fixture_refusal", "negative case identity/base is invalid")
        ids.add(case_id)
        mutated = apply_refusal_mutations(positives[base], case)
        try:
            schema_validate(mutated, validator)
            validate_semantics(mutated, legacy_fixture_sha256=legacy_fixture_sha256)
        except ArchiveContractError as exc:
            require(_expected(exc, case["expected_error"]), "fixture_refusal", f"{case_id} expected {case['expected_error']}, got {exc}")
        else:
            raise ArchiveContractError(f"fixture_refusal: negative case {case_id} was accepted")
    require(ids == REQUIRED_REFUSAL_CASE_IDS, "fixture_refusal", "negative refusal fixture ids drifted from the frozen required set")


def main() -> int:
    try:
        validate_archive_v1_contract()
    except ArchiveContractError as exc:
        print(f"archive V1 contract validation failed: {exc}", file=sys.stderr)
        return 1
    print("validated SAVE-001 archive/workspace V1 schema, physical profile, fixtures, and refusals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
