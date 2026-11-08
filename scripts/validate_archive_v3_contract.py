#!/usr/bin/env python3
"""Validate the contract-only Archive V3 storage-only fixture corpus.

This is an independent contract gate, not a production archive reader or
writer.  It proves that the frozen V3 manifest schema, the manifest reference /
inventory closure and the raw member integrity rules are executable on a small
synthetic archive.  The synthetic members are opaque storage witnesses: they are
not DAG-ML documents and not executable Methods models.  DAG-ML semantic
validation of a real refit package is covered elsewhere.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "docs/contracts/archive-v3"
SCHEMA_NAME = "archive_workspace_manifest.v3.schema.json"
POSITIVE_NAME = "storage_only_native_refit.json"
SYNTHETIC_NAME = "synthetic_members.v3.json"
REFUSALS_NAME = "refusals.v3.json"
ARCHIVE_V3_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/contracts/archive-v3/"
    "archive_workspace_manifest.v3.schema.json"
)
SYNTHETIC_PROFILE = "nirs4all.archive_workspace.synthetic_members.v3"
REFUSALS_PROFILE = "nirs4all.archive_workspace.refusals.v3"

PACKAGE_MEMBER = "dagml/portable_refit_package.json"
GRAPH_MEMBER = "dagml/graph.json"
BUNDLE_MEMBER = "dagml/portable_refit_execution_bundle.json"
OUTCOME_MEMBER = "dagml/portable_refit_outcome.json"
FIXED_MEMBERS = (PACKAGE_MEMBER, GRAPH_MEMBER, BUNDLE_MEMBER, OUTCOME_MEMBER)
MANIFEST_MEMBER = "manifest.json"
SAFE_ZIP_NAME = re.compile(r"^(?!.*(?:^|/)\.{1,2}(?:/|$))[A-Za-z0-9._/-]+$")

MAX_ENTRIES = 256
MAX_MEMBER_BYTES = 134_217_728
MAX_TOTAL_BYTES = 536_870_912
MAX_COMPRESSION_RATIO = 100

REFUSAL_CATEGORIES = frozenset(
    {
        "schema_refusal",
        "semantic_refusal",
        "member_inventory_refusal",
        "member_integrity_refusal",
    }
)

# Cases whose removal would silently weaken the forward-compatibility gate.
REQUIRED_REFUSAL_CASE_IDS = frozenset(
    {
        "future_archive_v4_is_refused",
        "future_archive_v999_is_refused",
        "profile_v4_is_refused",
        "archive_v2_profile_with_v3_version_is_refused",
        "archive_v2_version_with_v3_profile_is_refused",
        "writer_v2_identity_is_refused",
        "predictor_package_v2_mixing_is_refused",
        "training_outcome_v2_mixing_is_refused",
        "execution_bundle_v2_mixing_is_refused",
        "reader_dispatch_v3_future_versions_must_refuse",
        "reader_dispatch_v2_must_be_never_in_place",
        "n4mm_abi_major_3_is_refused",
        "n4mm_abi_major_missing_is_refused",
        "n4mm_format_version_3_requires_abi_2_20",
        "n4mm_format_version_4_is_refused",
        "unknown_manifest_field_is_refused",
        "unknown_n4mm_field_is_refused",
        "package_missing_producer_port_is_refused",
        "bundle_wrong_producer_port_is_refused",
        "outcome_missing_producer_port_is_refused",
        "host_artifacts_are_refused",
        "n4mopt_is_refused",
        "archive_level_conformal_is_refused",
        "n4d_aggregate_reference_is_refused",
        "non_ascii_member_path_is_refused",
        "inventory_path_diverges_from_reference_is_refused",
        "unreferenced_inventory_member_is_refused",
        "duplicate_inventory_path_is_refused",
        "reference_hash_diverges_from_inventory_is_refused",
        "n4mm_duplicate_artifact_id_is_refused",
        "n4mm_pipeline_format_requires_abi_2_5",
        "inventory_size_mismatch_is_refused",
        "forged_consistent_raw_hash_is_refused",
        "package_payload_tamper_is_refused",
        "n4mm_payload_tamper_is_refused",
        "missing_zip_member_is_refused",
        "undeclared_zip_member_is_refused",
    }
)


class ArchiveV3ContractError(RuntimeError):
    """Stable Archive V3 contract refusal; the prefix is the error category."""


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ArchiveV3ContractError(f"{code}: {detail}")


def _reject_constant(token: str) -> None:
    raise ArchiveV3ContractError(f"json_refusal: non-standard JSON constant `{token}`")


def _unique_members(label: str) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(
                key not in result,
                "json_refusal",
                f"duplicate member `{key}` in {label}",
            )
            result[key] = value
        return result

    return hook


def load_json_bytes(payload: bytes, label: str) -> Any:
    """Load strict UTF-8 JSON with duplicate-name and non-finite refusal."""

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_members(label),
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveV3ContractError(f"json_refusal: invalid JSON in {label}: {exc}") from exc


def load_json(path: Path) -> Any:
    try:
        return load_json_bytes(path.read_bytes(), str(path))
    except FileNotFoundError as exc:
        raise ArchiveV3ContractError(f"json_refusal: missing file {path}") from exc


def _error_path(error: ValidationError) -> str:
    return "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )


def validate_schema(schema: Any) -> Draft202012Validator:
    require(isinstance(schema, dict), "schema_refusal", "schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "schema_refusal",
        "schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == ARCHIVE_V3_SCHEMA_ID,
        "schema_refusal",
        "Archive V3 schema id drifted",
    )
    properties = schema.get("properties", {})
    require(
        properties.get("schema_version", {}).get("const") == 3
        and properties.get("profile", {}).get("const")
        == "nirs4all.archive_workspace.v3",
        "schema_refusal",
        "Archive V3 must accept only exact version 3 / profile v3",
    )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ArchiveV3ContractError(f"schema_refusal: {exc.message}") from exc
    return Draft202012Validator(schema)


def schema_validate(document: Any, validator: Draft202012Validator) -> None:
    errors = sorted(
        validator.iter_errors(document), key=lambda error: list(error.absolute_path)
    )
    if errors:
        error = errors[0]
        raise ArchiveV3ContractError(
            f"schema_refusal: {_error_path(error)}: {error.message}"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_semantics(document: dict[str, Any]) -> None:
    """Check what the closed schema cannot express: reference/inventory closure."""

    inventory: dict[str, dict[str, Any]] = {}
    for entry in document["member_inventory"]:
        path = entry["path"]
        require(
            path not in inventory,
            "semantic_refusal",
            f"duplicate inventory path `{path}`",
        )
        inventory[path] = entry

    def bind(reference: dict[str, Any]) -> None:
        path = reference["member_path"]
        entry = inventory.get(path)
        require(
            entry is not None,
            "semantic_refusal",
            f"reference path `{path}` is absent from the inventory",
        )
        for field in ("raw_sha256", "semantic_fingerprint", "semantic_profile"):
            require(
                reference[field] == entry[field],
                "semantic_refusal",
                f"reference `{path}` {field} does not bind its inventory member",
            )

    replay = document["replay"]
    dag_references = [
        replay["portable_refit_package"],
        replay["refit_artifacts"]["graph"],
        replay["refit_artifacts"]["execution_bundle"],
        replay["refit_artifacts"]["refit_outcome"],
    ]
    declared = set()
    for reference in dag_references:
        bind(reference)
        declared.add(reference["member_path"])
    require(
        declared == set(FIXED_MEMBERS),
        "semantic_refusal",
        "DAG-ML references must cover the four fixed refit members exactly",
    )

    artifact_ids: set[str] = set()
    for reference in document["payloads"]["methods"]["n4mm"]:
        path = reference["member_path"]
        require(
            reference["artifact_id"] not in artifact_ids and path not in declared,
            "semantic_refusal",
            "N4MM artifact ids and member paths must be globally unique",
        )
        artifact_ids.add(reference["artifact_id"])
        declared.add(path)
        bind(reference)
        require(
            reference["semantic_fingerprint"] == reference["raw_sha256"],
            "semantic_refusal",
            f"N4MM `{path}` semantic fingerprint must equal its raw SHA-256",
        )
        if reference["format_version"] == 2:
            require(
                reference.get("abi_min_minor", -1) >= 5,
                "semantic_refusal",
                f"N4MM `{path}` pipeline format 2 requires Methods ABI 2.5+",
            )

    require(
        declared == set(inventory),
        "semantic_refusal",
        "member inventory is not closed over the declared members",
    )


def synthetic_recipe(root: Path = ROOT) -> dict[str, Any]:
    recipe = load_json(root / "docs/contracts/archive-v3/fixtures/positive" / SYNTHETIC_NAME)
    require(
        isinstance(recipe, dict) and recipe.get("profile") == SYNTHETIC_PROFILE,
        "fixture_refusal",
        "synthetic member recipe profile drifted",
    )
    return recipe


def synthetic_member_bytes(recipe: dict[str, Any]) -> dict[str, bytes]:
    """Return the exact opaque bytes: marker + hex body, per recipe member."""

    marker = recipe["marker_utf8"].encode("utf-8")
    members: dict[str, bytes] = {}
    for item in recipe["members"]:
        path = item["path"]
        require(path not in members, "fixture_refusal", f"duplicate recipe member `{path}`")
        body = bytes.fromhex(item["body_hex"])
        members[path] = marker + body
    return members


def synthetic_semantic_fingerprint(recipe: dict[str, Any], path: str) -> str:
    return _sha256(recipe["semantic_fingerprint_prefix_utf8"].encode("utf-8") + path.encode("ascii"))


def validate_recipe_binding(
    document: dict[str, Any], recipe: dict[str, Any], payloads: dict[str, bytes]
) -> None:
    """The positive manifest hashes are exactly the recipe-derived values."""

    for entry in document["member_inventory"]:
        path = entry["path"]
        expected_semantic = (
            entry["raw_sha256"]
            if entry["semantic_profile"] == "n4mm_raw_sha256"
            else synthetic_semantic_fingerprint(recipe, path)
        )
        require(
            entry["semantic_fingerprint"] == expected_semantic,
            "fixture_refusal",
            f"positive fixture semantic fingerprint for `{path}` drifted from the recipe",
        )
    require(
        set(payloads) == {entry["path"] for entry in document["member_inventory"]},
        "fixture_refusal",
        "synthetic recipe members must equal the positive inventory",
    )


def validate_payloads(document: dict[str, Any], payloads: dict[str, bytes]) -> None:
    """Closed member set plus exact size and raw SHA-256 for every member."""

    inventory_paths = {entry["path"] for entry in document["member_inventory"]}
    require(
        set(payloads) == inventory_paths,
        "member_inventory_refusal",
        "archive members do not equal the manifest closed inventory",
    )
    for entry in document["member_inventory"]:
        data = payloads[entry["path"]]
        require(
            len(data) == entry["uncompressed_size_bytes"],
            "member_integrity_refusal",
            f"size mismatch for `{entry['path']}`",
        )
        require(
            _sha256(data) == entry["raw_sha256"],
            "member_integrity_refusal",
            f"raw SHA-256 mismatch for `{entry['path']}`",
        )


def validate_document(
    document: Any,
    payloads: dict[str, bytes],
    validator: Draft202012Validator,
) -> None:
    schema_validate(document, validator)
    validate_semantics(document)
    validate_payloads(document, payloads)


def _regular_zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def write_fixture_zip(
    path: Path, document: dict[str, Any], payloads: dict[str, bytes]
) -> None:
    """Write a deterministic stored ZIP used only by contract tests."""

    manifest_bytes = json.dumps(
        document, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_regular_zip_info(MANIFEST_MEMBER), manifest_bytes)
        for member_path in sorted(payloads):
            archive.writestr(_regular_zip_info(member_path), payloads[member_path])


def validate_archive_zip(path: Path, validator: Draft202012Validator) -> None:
    """Validate one materialized Archive V3 without extracting it."""

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveV3ContractError(f"zip_refusal: invalid ZIP {path}: {exc}") from exc
    with archive:
        infos = archive.infolist()
        require(len(infos) <= MAX_ENTRIES, "budget_refusal", "entry budget exceeded")
        by_name: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in infos:
            name = info.filename
            require(
                name == MANIFEST_MEMBER or SAFE_ZIP_NAME.fullmatch(name) is not None,
                "member_path_refusal",
                f"unsafe central-directory path `{name}`",
            )
            require(name not in by_name, "member_inventory_refusal", f"duplicate ZIP member `{name}`")
            by_name[name] = info
            mode = (info.external_attr >> 16) & 0xFFFF
            require(
                info.create_system != 3 or stat.S_ISREG(mode),
                "member_type_refusal",
                f"ZIP member `{name}` is not regular",
            )
            require(info.file_size <= MAX_MEMBER_BYTES, "budget_refusal", f"`{name}` exceeds member budget")
            require(
                info.file_size / max(info.compress_size, 1) <= MAX_COMPRESSION_RATIO,
                "compression_refusal",
                f"`{name}` exceeds compression-ratio budget",
            )
            total += info.file_size
        require(total <= MAX_TOTAL_BYTES, "budget_refusal", "total uncompressed budget exceeded")
        require(MANIFEST_MEMBER in by_name, "dispatch_refusal", "ZIP lacks manifest.json")
        document = load_json_bytes(archive.read(by_name[MANIFEST_MEMBER]), MANIFEST_MEMBER)
        payloads = {
            name: archive.read(info)
            for name, info in by_name.items()
            if name != MANIFEST_MEMBER
        }
    validate_document(document, payloads, validator)


def apply_mutations(document: Any, mutations: list[dict[str, Any]]) -> Any:
    """Apply `replace` / `add` / `delete` / `append` mutations to a copy."""

    mutated = copy.deepcopy(document)
    for mutation in mutations:
        operation = mutation["op"]
        path = mutation["path"]
        require(bool(path), "fixture_refusal", "mutation path cannot be empty")
        parent = mutated
        for token in path[:-1]:
            parent = parent[token]
        token = path[-1]
        if operation in {"add", "replace"}:
            parent[token] = copy.deepcopy(mutation["value"])
        elif operation == "delete":
            del parent[token]
        elif operation == "append":
            parent[token].append(copy.deepcopy(mutation["value"]))
        else:
            raise ArchiveV3ContractError(
                f"fixture_refusal: unknown mutation operation `{operation}`"
            )
    return mutated


def apply_payload_mutations(
    payloads: dict[str, bytes], mutations: list[dict[str, Any]]
) -> dict[str, bytes]:
    mutated = dict(payloads)
    for mutation in mutations:
        operation = mutation["op"]
        member = mutation["member"]
        if operation == "append_hex":
            mutated[member] = mutated[member] + bytes.fromhex(mutation["value_hex"])
        elif operation == "drop":
            del mutated[member]
        elif operation == "rename":
            require(mutation["to"] not in mutated, "fixture_refusal", "rename target exists")
            mutated[mutation["to"]] = mutated.pop(member)
        elif operation == "add_hex":
            require(member not in mutated, "fixture_refusal", f"`{member}` already exists")
            mutated[member] = bytes.fromhex(mutation["value_hex"])
        else:
            raise ArchiveV3ContractError(
                f"fixture_refusal: unknown payload operation `{operation}`"
            )
    return mutated


def _expect_refusal(expected: str, operation: Any, case_id: str) -> None:
    try:
        operation()
    except ArchiveV3ContractError as exc:
        require(
            str(exc).startswith(f"{expected}:"),
            "fixture_refusal",
            f"{case_id} expected {expected}, received {exc}",
        )
    except Exception as exc:
        raise ArchiveV3ContractError(
            f"fixture_refusal: {case_id} raised unstable {type(exc).__name__}: {exc}"
        ) from exc
    else:
        raise ArchiveV3ContractError(
            f"fixture_refusal: {case_id} unexpectedly passed; expected {expected}"
        )


def validate_refusal_cases(
    manifest: dict[str, Any],
    payloads: dict[str, bytes],
    refusals: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    require(
        refusals.get("profile") == REFUSALS_PROFILE,
        "fixture_refusal",
        "Archive V3 refusal profile drifted",
    )
    cases = refusals.get("cases")
    require(isinstance(cases, list), "fixture_refusal", "refusal cases missing")
    ids = [case.get("id") for case in cases]
    require(len(ids) == len(set(ids)), "fixture_refusal", "refusal ids must be unique")
    require(
        REQUIRED_REFUSAL_CASE_IDS <= set(ids),
        "fixture_refusal",
        f"required Archive V3 refusal cases missing: {sorted(REQUIRED_REFUSAL_CASE_IDS - set(ids))}",
    )
    seen_categories: set[str] = set()
    for case in cases:
        case_id = case["id"]
        expected = case["expected_error"]
        require(
            case["base"] == "archive_v3" and expected in REFUSAL_CATEGORIES,
            "fixture_refusal",
            f"{case_id} has an unknown base or error category",
        )
        seen_categories.add(expected)
        mutations = case.get("mutations", [])
        payload_mutations = case.get("payload_mutations", [])
        require(
            bool(mutations) or bool(payload_mutations),
            "fixture_refusal",
            f"{case_id} must mutate something",
        )
        mutated = apply_mutations(manifest, mutations)
        mutated_payloads = apply_payload_mutations(payloads, payload_mutations)
        require(
            mutated != manifest or mutated_payloads != payloads,
            "fixture_refusal",
            f"{case_id} is a no-op",
        )
        _expect_refusal(
            expected,
            lambda document=mutated, data=mutated_payloads: validate_document(
                document, data, validator
            ),
            case_id,
        )
    require(
        seen_categories == REFUSAL_CATEGORIES,
        "fixture_refusal",
        "every refusal category must be exercised",
    )


def validate_archive_v3_contract(root: Path = ROOT) -> None:
    """Run the complete local Archive V3 storage contract gate."""

    archive_root = root / "docs/contracts/archive-v3"
    validator = validate_schema(load_json(archive_root / SCHEMA_NAME))
    manifest = load_json(archive_root / "fixtures/positive" / POSITIVE_NAME)
    refusals = load_json(archive_root / "fixtures/negative" / REFUSALS_NAME)
    recipe = synthetic_recipe(root)
    payloads = synthetic_member_bytes(recipe)
    validate_recipe_binding(manifest, recipe, payloads)
    validate_document(manifest, payloads, validator)
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "storage-only-v3.n4a"
        write_fixture_zip(archive_path, manifest, payloads)
        validate_archive_zip(archive_path, validator)
    validate_refusal_cases(manifest, payloads, refusals, validator)


def main() -> int:
    try:
        validate_archive_v3_contract(ROOT)
    except ArchiveV3ContractError as exc:
        print(f"Archive V3 contract validation failed: {exc}", file=sys.stderr)
        return 1
    print("Archive V3 contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
