#!/usr/bin/env python3
"""Validate the contract-only Archive V2 native-portable replay freeze.

This is an independent contract gate, not a production archive reader or
writer.  It materializes bounded synthetic ZIPs only to prove that the frozen
manifest, raw-member integrity and pre-write portability refusals are
executable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import stat
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import (  # noqa: E402
    ContractError,
    fingerprint_without,
    tcv1_sha256,
    validate_strict_json,
)
from parity.training.oracle import _serde_sha256  # noqa: E402
from parity.training.training_replay_oracle import replay_relation_fingerprint  # noqa: E402
from scripts.validate_archive_v1_contract import (  # noqa: E402
    ARCHIVE_ROOT as ARCHIVE_V1_ROOT,
    ArchiveContractError as ArchiveV1ContractError,
    load_json as load_v1_json,
    schema_validate as schema_validate_v1,
    validate_schema as validate_schema_v1,
)


from scripts.validate_classification_state import (
    RAW_KIND as CLASSIFIER_RAW_KIND, META_KIND as CLASSIFIER_META_KIND,
    RAW_PROFILE as CLASSIFIER_RAW_PROFILE, META_PROFILE as CLASSIFIER_META_PROFILE,
    validate_payload as validate_classifier_payload,
)

ARCHIVE_ROOT = ROOT / "docs/contracts/archive-v2"
SCHEMA_NAME = "archive_workspace_manifest.v2.schema.json"
POSITIVE_NAME = "native_portable_replay.json"
REFUSALS_NAME = "refusals.v2.json"
PACKAGE_MEMBER = "dagml/portable_predictor_package.json"
METHODS_ABI_MAJOR = 2
METHODS_HISTORICAL_UNKNOWN_MIN_MINOR = 2
METHODS_PLS_CONTROLLER_ID = "controller:methods.pls"
METHODS_RIDGE_CONTROLLER_ID = "controller:methods.ridge"
PACKAGE_V1_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "portable_predictor_package.v1.schema.json"
)
PACKAGE_V2_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "portable_predictor_package.v2.schema.json"
)
ARCHIVE_V2_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/contracts/archive-v2/"
    "archive_workspace_manifest.v2.schema.json"
)

REQUIRED_REFUSAL_CASE_IDS = frozenset(
    {
        "archive_v2_package_v1_mixing_is_refused",
        "archive_v1_package_v2_mixing_is_refused",
        "manifest_host_reference_is_refused",
        "package_host_sidecar_is_refused_pre_write",
        "package_reference_hash_mismatch_is_refused",
        "package_payload_tamper_is_refused",
        "n4mm_payload_tamper_is_refused",
        "n4mm_package_byte_divergence_is_refused",
        "n4mm_abi_mismatch_is_refused",
        "n4mm_abi_minor_mismatch_is_refused",
        "package_ridge_missing_abi_minor_is_refused",
        "package_python_plugin_fallback_is_refused",
        "package_raw_payload_semantic_mismatch_is_refused",
        "future_archive_v3_is_refused",
        "unknown_core_field_is_refused",
        "package_wrong_producer_port_policy_is_refused",
        "runtime_wrong_producer_port_policy_is_refused",
        "archive_level_conformal_sidecar_is_refused",
        "missing_native_n4mm_is_refused",
    }
)

RUNTIME_V2_ARTIFACTS = {
    "execution_bundle": (
        "https://github.com/GBeurier/dag-ml/schemas/"
        "execution_bundle.v2.schema.json"
    ),
    "training_outcome": (
        "https://github.com/GBeurier/dag-ml/schemas/"
        "training_outcome.v2.schema.json"
    ),
    "prediction_cache_payload_set": (
        "https://github.com/GBeurier/dag-ml/schemas/"
        "prediction_cache_payload_set.v2.schema.json"
    ),
    "score_set": (
        "https://github.com/GBeurier/dag-ml/schemas/score_set.v2.schema.json"
    ),
}


class ArchiveV2ContractError(RuntimeError):
    """Stable Archive V2 contract refusal."""


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ArchiveV2ContractError(f"{code}: {detail}")


def _reject_constant(token: str) -> None:
    raise ArchiveV2ContractError(
        f"json_refusal: non-standard JSON constant `{token}`"
    )


def load_json(path: Path) -> Any:
    """Load strict UTF-8 JSON with duplicate-name and non-finite refusal."""

    def unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(
                key not in result,
                "json_refusal",
                f"duplicate member `{key}` in {path}",
            )
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=unique_members,
                parse_constant=_reject_constant,
            )
    except FileNotFoundError as exc:
        raise ArchiveV2ContractError(f"json_refusal: missing file {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveV2ContractError(
            f"json_refusal: invalid JSON in {path}: {exc}"
        ) from exc


def load_json_bytes(payload: bytes, label: str) -> Any:
    """Load strict UTF-8 JSON from a bounded archive member."""

    def unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(
                key not in result,
                "json_refusal",
                f"duplicate member `{key}` in {label}",
            )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_members,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveV2ContractError(
            f"json_refusal: invalid JSON in {label}: {exc}"
        ) from exc


def _error_path(error: ValidationError) -> str:
    return "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )


def validate_schema(schema: Any) -> Draft202012Validator:
    require(isinstance(schema, dict), "schema_refusal", "schema must be an object")
    require(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema",
        "schema_refusal",
        "schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == ARCHIVE_V2_SCHEMA_ID,
        "schema_refusal",
        "Archive V2 schema id drifted",
    )
    require(
        schema.get("properties", {}).get("schema_version", {}).get("const") == 2,
        "schema_refusal",
        "Archive V2 must accept only version 2",
    )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ArchiveV2ContractError(f"schema_refusal: {exc.message}") from exc
    return Draft202012Validator(schema)


def schema_validate(document: Any, validator: Draft202012Validator) -> None:
    errors = sorted(
        validator.iter_errors(document), key=lambda error: list(error.absolute_path)
    )
    if errors:
        error = errors[0]
        raise ArchiveV2ContractError(
            f"schema_refusal: {_error_path(error)}: {error.message}"
        )


def contract_schema_registry(root: Path = ROOT) -> tuple[dict[str, Any], Registry]:
    """Load the local contract schema graph by canonical schema ID."""

    schemas: dict[str, Any] = {}
    registry = Registry()
    for path in sorted((root / "docs/contracts").glob("*.schema.json")):
        schema = load_json(path)
        schema_id = schema.get("$id") if isinstance(schema, dict) else None
        if not isinstance(schema_id, str):
            continue
        require(
            schema_id not in schemas,
            "schema_refusal",
            f"duplicate local schema id `{schema_id}`",
        )
        schemas[schema_id] = schema
        try:
            registry = registry.with_resource(
                schema_id, Resource.from_contents(schema)
            )
        except Exception as exc:
            raise ArchiveV2ContractError(
                f"schema_refusal: cannot register {path}: {exc}"
            ) from exc
    return schemas, registry


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
    registry: Registry,
    label: str,
) -> None:
    errors = sorted(
        Draft202012Validator(schema, registry=registry).iter_errors(instance),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        raise ArchiveV2ContractError(
            f"schema_refusal: {label}{_error_path(error)[1:]}: {error.message}"
        )


def _safe_member_path(path: str) -> bool:
    if not path or path != unicodedata.normalize("NFC", path):
        return False
    if (
        "\\" in path
        or ":" in path
        or any(ord(char) <= 0x1F or ord(char) == 0x7F for char in path)
    ):
        return False
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return False
    reserved = re.compile(
        r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE
    )
    return all(
        part not in {"", ".", ".."}
        and not part.endswith((".", " "))
        and not reserved.fullmatch(part)
        for part in path.split("/")
    )


def _refs(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {
            "member_path",
            "raw_sha256",
            "semantic_fingerprint",
            "semantic_profile",
        } <= value.keys():
            found.append(value)
        for nested in value.values():
            found.extend(_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_refs(nested))
    return found


def n4mm_abi_requirement(artifact: dict[str, Any]) -> tuple[int, int]:
    """Resolve payload capability, with narrow dual-read for old references."""

    controller = artifact.get("controller_id")
    major = artifact.get("abi_major")
    minor = artifact.get("abi_min_minor")
    require(
        (major is None) == (minor is None),
        "native_model_refusal",
        f"artifact `{artifact.get('id')}` must declare both ABI fields or neither",
    )
    if controller == METHODS_PLS_CONTROLLER_ID:
        expected = 0
    elif controller == METHODS_RIDGE_CONTROLLER_ID:
        expected = 3
    elif major is None:
        # The frozen pre-field Archive V2 fixture predates the two built-in
        # controller ids. Its conservative historical profile is ABI 2.2,
        # never the ABI 2.3 imported-linear capability.
        return METHODS_ABI_MAJOR, METHODS_HISTORICAL_UNKNOWN_MIN_MINOR
    else:
        expected = minor
    require(
        major == METHODS_ABI_MAJOR and minor == expected,
        "native_model_refusal",
        f"artifact `{artifact.get('id')}` has an invalid capability-derived ABI minimum",
    )
    return METHODS_ABI_MAJOR, expected


def n4mm_reference_abi_requirement(
    reference: dict[str, Any], artifact: dict[str, Any]
) -> tuple[int, int]:
    """Resolve an archive reference without widening the historical read."""

    minor = reference.get("abi_min_minor")
    if minor is not None:
        return reference["abi_major"], minor
    controller = artifact.get("controller_id")
    if (
        controller == METHODS_PLS_CONTROLLER_ID
        and artifact.get("abi_major") is None
        and artifact.get("abi_min_minor") is None
    ):
        return METHODS_ABI_MAJOR, 0
    # The frozen pre-field fixture used generic mock controller ids. Preserve
    # that exact 2.2 profile, but do not let absence stand in for Ridge 2.3.
    return METHODS_ABI_MAJOR, METHODS_HISTORICAL_UNKNOWN_MIN_MINOR


def validate_semantics(document: Any) -> None:
    """Enforce closed inventory and cross-reference invariants."""

    inventory = document["member_inventory"]
    limits = document["physical_profile"]["limits"]
    require(
        len(inventory) + 1 <= limits["max_entries"],
        "budget_refusal",
        "member count exceeds dispatch budget",
    )
    members: dict[str, dict[str, Any]] = {}
    total = 0
    for member in inventory:
        path = member["path"]
        require(
            _safe_member_path(path),
            "member_path_refusal",
            f"unsafe or non-canonical POSIX path `{path}`",
        )
        require(
            path != "manifest.json",
            "member_inventory_refusal",
            "manifest.json is dispatch metadata, not a self-hashed payload",
        )
        require(
            path not in members,
            "member_inventory_refusal",
            f"duplicate inventory member `{path}`",
        )
        require(
            member["regular_file"] is True,
            "member_type_refusal",
            f"member `{path}` is not regular",
        )
        size = member["uncompressed_size_bytes"]
        require(
            size <= limits["max_member_uncompressed_bytes"],
            "budget_refusal",
            f"member `{path}` exceeds per-member budget",
        )
        total += size
        members[path] = member
    require(
        total <= limits["max_total_uncompressed_bytes"],
        "budget_refusal",
        "inventory exceeds total budget",
    )

    replay = document["replay"]
    replay_paths = [
        replay["portable_predictor_package"]["member_path"],
        *(
            reference["member_path"]
            for reference in replay["training_artifacts"].values()
        ),
    ]
    require(
        len(replay_paths) == len(set(replay_paths)),
        "replay_alias_refusal",
        "package, graph, bundle, outcome, cache and scores must be distinct",
    )
    methods = document["payloads"]["methods"]
    for reference in methods["n4mm"] + methods["n4mopt"]:
        require(
            reference.get("abi_major") == METHODS_ABI_MAJOR,
            "native_model_refusal",
            "Methods reference must declare ABI major 2",
        )
        if "abi_min_minor" in reference:
            require(
                isinstance(reference["abi_min_minor"], int)
                and not isinstance(reference["abi_min_minor"], bool)
                and reference["abi_min_minor"] >= 0,
                "native_model_refusal",
                "Methods reference has an invalid ABI minimum minor",
            )
    role_pipelines = methods.get("role_pipelines", [])
    multimodal_pipelines = methods.get("multimodal_pipelines", [])
    method_paths = [
        reference["member_path"]
        for reference in methods["n4mm"] + methods["n4mopt"] + role_pipelines + multimodal_pipelines
    ]
    require(
        len(method_paths) == len(set(method_paths)),
        "methods_alias_refusal",
        "Methods payload members cannot alias",
    )
    n4mm_artifact_ids = [
        reference["artifact_id"] for reference in methods["n4mm"] + role_pipelines + multimodal_pipelines
    ]
    require(
        len(n4mm_artifact_ids) == len(set(n4mm_artifact_ids)),
        "native_model_refusal",
        "N4MM, RolePipeline and multimodal artifact ids must be unique",
    )
    require(
        bool(methods["n4mm"] or role_pipelines or multimodal_pipelines),
        "native_model_refusal",
        "Archive V2 native replay requires at least one N4MM or RolePipeline member",
    )

    references = _refs({"replay": replay, "payloads": document["payloads"]})
    used: set[str] = set()
    for reference in references:
        path = reference["member_path"]
        require(
            _safe_member_path(path),
            "member_path_refusal",
            f"unsafe reference path `{path}`",
        )
        member = members.get(path)
        require(
            member is not None,
            "member_inventory_refusal",
            f"reference `{path}` is absent from inventory",
        )
        require(
            member["raw_sha256"] == reference["raw_sha256"],
            "member_integrity_refusal",
            f"raw SHA-256 mismatch for `{path}`",
        )
        require(
            member["semantic_profile"] == reference["semantic_profile"]
            and member["semantic_fingerprint"]
            == reference["semantic_fingerprint"],
            "member_semantic_refusal",
            f"semantic identity mismatch for `{path}`",
        )
        used.add(path)
    require(
        set(members) == used,
        "member_inventory_refusal",
        "closed inventory contains an unreferenced payload",
    )

    package = replay["portable_predictor_package"]
    require(
        package["schema_id"] == PACKAGE_V2_SCHEMA_ID
        and package["schema_version"] == 2,
        "version_mixing_refusal",
        "Archive V2 requires exact PortablePredictorPackage V2",
    )
    require(
        package["producer_port_required"] is True,
        "producer_port_refusal",
        "PortablePredictorPackage V2 must require producer_port",
    )
    require(
        package["member_path"] == PACKAGE_MEMBER,
        "package_refusal",
        "PortablePredictorPackage V2 member path drifted",
    )
    graph = replay["training_artifacts"]["graph"]
    require(
        graph["schema_version"] == 1
        and graph["schema_id"].endswith("/graph_spec.v1.schema.json"),
        "replay_refusal",
        "Archive V2 retains exact GraphSpec V1",
    )
    for name, expected_schema_id in RUNTIME_V2_ARTIFACTS.items():
        reference = replay["training_artifacts"][name]
        require(
            reference["schema_version"] == 2
            and reference["schema_id"] == expected_schema_id,
            "version_mixing_refusal",
            f"Archive V2 requires exact {name} V2",
        )
        require(
            reference["producer_port_required"] is True,
            "producer_port_refusal",
            f"Archive V2 {name} must require producer_port",
        )
    require(
        document["payloads"]["conformal"] is None,
        "conformal_ownership_refusal",
        "Archive V2 conformal state belongs to the package",
    )
    require(
        document["payloads"]["host_artifacts"] == [],
        "host_artifact_refusal",
        "Archive V2 cannot inventory host artifacts",
    )


def canonical_portable_package_v2(root: Path = ROOT) -> dict[str, Any]:
    """Load the frozen schema-valid native Methods Package V2 fixture."""

    return load_json(
        root
        / "docs/contracts/archive-v2/fixtures/positive/"
        "portable_predictor_package.native_methods.v2.json"
    )


def canonical_portable_package_v2_bytes(root: Path = ROOT) -> bytes:
    package = canonical_portable_package_v2(root)
    return json.dumps(
        package,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_package_schema_boundary(root: Path = ROOT) -> None:
    """Prove exact package IDs and the closed Package V1/V2 boundary."""

    schemas, registry = contract_schema_registry(root)
    require(
        PACKAGE_V1_SCHEMA_ID in schemas,
        "schema_refusal",
        "missing PortablePredictorPackage V1 schema id",
    )
    require(
        PACKAGE_V2_SCHEMA_ID in schemas,
        "schema_refusal",
        "missing PortablePredictorPackage V2 schema id",
    )
    package_v1_schema = schemas[PACKAGE_V1_SCHEMA_ID]
    package_v2_schema = schemas[PACKAGE_V2_SCHEMA_ID]
    require(
        package_v2_schema.get("$id") == PACKAGE_V2_SCHEMA_ID,
        "schema_refusal",
        "PortablePredictorPackage V2 schema id must identify V2",
    )
    require(
        package_v1_schema["properties"]["execution_bundle"]["$ref"].endswith(
            "/execution_bundle.v1.schema.json"
        ),
        "schema_refusal",
        "PortablePredictorPackage V1 must require ExecutionBundle V1",
    )
    require(
        package_v2_schema["properties"]["execution_bundle"]["$ref"].endswith(
            "/execution_bundle.v2.schema.json"
        ),
        "schema_refusal",
        "PortablePredictorPackage V2 must require ExecutionBundle V2",
    )

    package_v1 = load_json(
        root / "examples/fixtures/training/portable_predictor_package.v1.json"
    )
    validate_instance(
        package_v1, package_v1_schema, registry, "PortablePredictorPackage V1"
    )
    for v2_only_key in (
        "conformal_calibration",
        "conformal_calibration_replay",
    ):
        mutated = copy.deepcopy(package_v1)
        mutated[v2_only_key] = None
        try:
            validate_instance(
                mutated,
                package_v1_schema,
                registry,
                f"PortablePredictorPackage V1 with {v2_only_key}",
            )
        except ArchiveV2ContractError as exc:
            require(
                str(exc).startswith("schema_refusal:"),
                "schema_refusal",
                f"unexpected Package V1 refusal for {v2_only_key}: {exc}",
            )
        else:
            raise ArchiveV2ContractError(
                "schema_refusal: PortablePredictorPackage V1 accepted "
                f"V2-only null key `{v2_only_key}`"
            )

    bundle_v2_in_v1 = copy.deepcopy(package_v1)
    bundle_v2_in_v1["execution_bundle"]["schema_version"] = 2
    try:
        validate_instance(
            bundle_v2_in_v1,
            package_v1_schema,
            registry,
            "PortablePredictorPackage V1 with ExecutionBundle V2",
        )
    except ArchiveV2ContractError as exc:
        require(
            str(exc).startswith("schema_refusal:"),
            "schema_refusal",
            f"unexpected Package V1/Bundle V2 refusal: {exc}",
        )
    else:
        raise ArchiveV2ContractError(
            "schema_refusal: PortablePredictorPackage V1 accepted ExecutionBundle V2"
        )

    package_v2 = canonical_portable_package_v2(root)
    validate_instance(
        package_v2, package_v2_schema, registry, "PortablePredictorPackage V2"
    )
    validate_package_portability(package_v2)


ROLE_PIPELINE_PROFILE = "dagml_methods_role_pipeline_raw_sha256"
MULTIMODAL_PIPELINE_PROFILE = "dagml_methods_multimodal_pipeline_raw_sha256"


def validate_raw_encoder_recipe(recipe: dict[str, Any], schemas: dict[str, Any]) -> None:
    """Shared closed encoder declarations; each codec validates its own head."""
    order = ["nir", "image", "series", "metadata"]
    require(isinstance(recipe, dict) and set(recipe) == {"schema_version", "fusion", "source_order", "encoders", "source_weights", "model"}
            and type(recipe["schema_version"]) is int and recipe["schema_version"] == 1 and recipe["fusion"] == "early"
            and isinstance(recipe["source_order"], list) and 1 <= len(recipe["source_order"]) <= len(order)
            and all(isinstance(name, str) and name in order for name in recipe["source_order"])
            and len(set(recipe["source_order"])) == len(recipe["source_order"])
            and isinstance(schemas, dict) and set(schemas) == set(order)
            and isinstance(recipe["encoders"], dict) and set(recipe["encoders"]) == set(recipe["source_order"])
            and isinstance(recipe["source_weights"], dict) and set(recipe["source_weights"]) == set(recipe["source_order"]),
            "native_model_refusal", "Distinct ordered U07 subset and complete raw source schemas required")
    for name, representation in zip(order, ("signal_1d", "rgb_image", "series_mv", "tabular_mixed"), strict=True):
        schema = schemas[name]
        require(isinstance(schema, dict) and set(schema) == {"representation_id", "input_shape", "dtype", "identity"}
                and schema["representation_id"] == representation and isinstance(schema["input_shape"], list)
                and 0 < len(schema["input_shape"]) <= 7 and all(type(size) is int and 0 < size <= 2**63-1 for size in schema["input_shape"])
                and math.prod(schema["input_shape"]) <= 1_048_576
                and (name != "metadata" or schema["input_shape"] == [2])
                and isinstance(schema["dtype"], str) and 0 < len(schema["dtype"].encode()) <= 128
                and isinstance(schema["identity"], str) and 0 < len(schema["identity"].encode()) <= 1_048_576,
                "native_model_refusal", "Canonical raw shape, representation and identity required")
        try:
            validate_strict_json(load_json_bytes(schema["identity"].encode(), name), "Raw multimodal source identity")
        except ContractError as error:
            raise ArchiveV2ContractError(f"native_model_refusal: {error}") from error
    weights = recipe["source_weights"].values()
    require(all(type(value) in (int, float) and value >= 0 and value < float("inf") for value in weights),
            "native_model_refusal", "Finite nonnegative source weights required")
    require(("nir" not in recipe["encoders"] or tcv1_sha256(recipe["encoders"]["nir"]) == tcv1_sha256({"kind":"standard_scaler", "with_mean":True, "with_std":True}))
            and ("metadata" not in recipe["encoders"] or tcv1_sha256(recipe["encoders"]["metadata"]) == tcv1_sha256({"kind":"column_transformer", "numeric_columns":[0], "categorical_columns":[1],
                "with_mean":True, "with_std":True, "handle_unknown":"ignore", "sparse_output":False, "drop":None})),
            "native_model_refusal", "Closed native population scaler/mixed column recipe required")
    for name in ("image", "series"):
        if name not in recipe["source_order"]:
            continue
        encoder = recipe["encoders"][name]
        require(isinstance(encoder, dict) and set(encoder) == {"kind", "n_components", "whiten", "random_state"}
                and encoder["kind"] == "tensor_pca" and type(encoder["n_components"]) is int
                and 0 < encoder["n_components"] <= min(2**31-1, math.prod(schemas[name]["input_shape"]))
                and encoder["whiten"] is False and type(encoder["random_state"]) is int and 0 <= encoder["random_state"] <= 2**32-1,
                "native_model_refusal", "Closed native unwhitened U07 PCA recipe required")


def validate_multimodal_pipeline_payload(record: dict[str, Any], payload: bytes, plan: dict[str, Any]) -> None:
    """Validate complete declarative U07 state binding, without native hydration."""
    artifact = record["artifact"]
    digest = hashlib.sha256(payload).hexdigest()
    owners = {f"dagml.methods.{host}.multimodal": f"controller:methods.{host}.multimodal"
              for host in ("python", "wasm", "r", "octave")}
    owner = owners.get(artifact.get("plugin"))
    require(0 < len(payload) <= 134_217_728 and owner is not None
            and record.get("controller_id") == artifact.get("controller_id") == owner
            and artifact.get("kind") == "methods_multimodal_pipeline"
            and artifact.get("backend") == "raw" and artifact.get("plugin_version") == "1.0.0"
            and artifact.get("native_predictor_descriptor") is None and artifact.get("native_estimator_descriptor") is None
            and artifact.get("size_bytes") == len(payload) and artifact.get("content_fingerprint") == digest
            and artifact.get("uri") == f"artifacts/{digest}.json", "native_model_refusal",
            "Complete Methods multimodal predictor must bind bounded RAW bytes and its exact owner")
    saved = load_json_bytes(payload, artifact["uri"])
    require(isinstance(saved, dict) and set(saved) == {"schema", "node_id", "params_fingerprint", "target_names", "recipe", "source_schemas", "state"}
            and saved["schema"] == "dagml.methods.multimodal.v1" and saved["node_id"] == record["node_id"]
            and saved["params_fingerprint"] == record["params_fingerprint"], "native_model_refusal", "Closed complete multimodal wrapper required")
    names = saved["target_names"]
    require(isinstance(names, list) and len(names) == 1 and isinstance(names[0], str) and 0 < len(names[0].encode()) <= 4096,
            "native_model_refusal", "Complete U07 predictor requires one named target")
    state = saved["state"]
    require(isinstance(state, list) and 28 <= len(state) <= 67_108_864
            and all(type(byte) is int and 0 <= byte <= 255 for byte in state)
            and bytes(state[:12]) == b"N4MF" + (1).to_bytes(4, "little") + (2).to_bytes(4, "little"),
            "native_model_refusal", "Complete Methods state must use N4MF format1 ABImajor2")
    recipe, schemas = saved["recipe"], saved["source_schemas"]
    validate_raw_encoder_recipe(recipe, schemas)
    model = recipe["model"]
    require(isinstance(model, dict) and set(model) == {"method_id", "params"}
            and model["method_id"] == "models.regularized.ridge" and isinstance(model["params"], dict)
            and set(model["params"]) == {"alpha", "center_x", "center_y", "scale_x"}
            and type(model["params"]["alpha"]) in (int,float) and 0 <= model["params"]["alpha"] < float("inf")
            and model["params"]["center_x"] is True and model["params"]["center_y"] is True and model["params"]["scale_x"] is False,
            "native_model_refusal", "Native Ridge must preserve canonical U07 centering and scaling")
    node = plan.get("node_plans", {}).get(record["node_id"])
    graph_node = next((item for item in plan.get("graph_plan", {}).get("graph", {}).get("nodes", []) if item.get("id") == record["node_id"]), None)
    require(isinstance(node, dict) and isinstance(graph_node, dict) and isinstance(graph_node.get("operator"), dict),
            "native_model_refusal", "Complete multimodal state requires its signed effective graph node")
    operator = graph_node["operator"]
    require(set(operator) == {"type", "recipe", "source_schemas"} and operator["type"] == "N4mMultimodalPipeline",
            "native_model_refusal", "Explicit complete multimodal graph operator required")
    expected = copy.deepcopy(operator["recipe"])
    params = node["params"]
    declarations = {"recipe", "source_schemas"}
    require(set(params) <= {"model__alpha", "source_weights__image", "transformers__image__n_components"} | declarations,
            "native_model_refusal", "Unknown effective multimodal tuning parameter")
    require(not set(params) & declarations or set(params) & declarations == declarations,
            "native_model_refusal", "Both immutable multimodal declarations required")
    if set(params) & declarations:
        require(set(params) <= declarations | {"model__alpha"}, "native_model_refusal", "Structural multimodal tuning permits alpha only")
        require(all(tcv1_sha256(params[key]) == tcv1_sha256(operator[key]) for key in declarations),
                "native_model_refusal", "Immutable declarations differ from the signed graph operator")
    if "model__alpha" in params: expected["model"]["params"]["alpha"] = params["model__alpha"]
    if "source_weights__image" in params:
        require("image" in expected["source_order"], "native_model_refusal", "Image weight is inactive")
        expected["source_weights"]["image"] = params["source_weights__image"]
    if "transformers__image__n_components" in params:
        require("image" in expected["source_order"], "native_model_refusal", "Image PCA is inactive")
        expected["encoders"]["image"]["n_components"] = params["transformers__image__n_components"]
    require(recipe == expected and schemas == operator["source_schemas"] and saved["params_fingerprint"] == node["params_fingerprint"],
            "native_model_refusal", "Saved multimodal recipe/source schema differs from selected effective plan")


def validate_role_pipeline_payload(record: dict[str, Any], payload: bytes) -> None:
    """Independently validate the bounded declarative codec, never hydrate it."""
    artifact = record["artifact"]
    digest = hashlib.sha256(payload).hexdigest()
    require(
        0 < len(payload) <= 134_217_728
        and artifact.get("kind") == "methods_role_pipeline"
        and artifact.get("backend") == "raw"
        and artifact.get("plugin") in {
            "dagml.methods.wasm.regression", "dagml.methods.r.regression",
            "dagml.methods.octave.regression", "dagml.methods.native.regression",
            "dagml.methods.python.regression",
        }
        and artifact.get("plugin_version") == "1.0.0"
        and artifact.get("native_predictor_descriptor") is None
        and artifact.get("native_estimator_descriptor") is None
        and artifact.get("uri") == f"artifacts/{digest}.json",
        "native_model_refusal",
        "RolePipeline must use its trusted RAW codec and content-addressed URI",
    )
    if artifact.get("plugin") == "dagml.methods.native.regression":
        require(artifact.get("controller_id") == record.get("controller_id") == "controller:methods.native.regression",
                "native_model_refusal", "native PLS RAW plugin requires its native controller owner")
    if artifact.get("plugin") == "dagml.methods.octave.regression":
        require(artifact.get("controller_id") == record.get("controller_id") == "controller:methods.octave.regression",
                "native_model_refusal", "Octave Methods RAW plugin requires its exact Octave controller owner")
    if artifact.get("plugin") == "dagml.methods.python.regression":
        require(artifact.get("controller_id") == record.get("controller_id") == "controller:methods.python.regression",
                "native_model_refusal", "Python Methods RAW plugin requires its exact Python controller owner")
    wrapper = load_json_bytes(payload, artifact["uri"])
    def validate_depth(value: Any, depth: int = 0) -> None:
        require(depth <= 128, "native_model_refusal", "RolePipeline JSON exceeds its depth bound")
        if isinstance(value, (dict, list)):
            for child in value.values() if isinstance(value, dict) else value:
                validate_depth(child, depth + 1)

    validate_depth(wrapper)
    try:
        validate_strict_json(wrapper, "RolePipeline wrapper")
    except ContractError as error:
        raise ArchiveV2ContractError(f"native_model_refusal: {error}") from error
    require(
        isinstance(wrapper, dict)
        and set(wrapper) == {
            "schema", "node_id", "params_fingerprint", "target_names",
            "steps", "feature_names", "states",
        }
        and wrapper["schema"] == "dagml.methods.regression.v1"
        and wrapper["node_id"] == record["node_id"]
        and wrapper["params_fingerprint"] == record["params_fingerprint"],
        "native_model_refusal",
        "RolePipeline wrapper must bind its exact node and parameters",
    )
    for field in ("target_names", "feature_names"):
        names = wrapper[field]
        require(
            isinstance(names, list) and 0 < len(names) <= 65_536
            and all(isinstance(name, str) and 0 < len(name.encode("utf-8")) <= 4096
                    for name in names)
            and len(set(names)) == len(names),
            "native_model_refusal",
            f"RolePipeline {field} must be ordered, nonempty and unique",
        )
    steps, states = wrapper["steps"], wrapper["states"]
    require(
        isinstance(steps, list) and 0 < len(steps) <= 256
        and isinstance(states, list) and len(states) == len(steps),
        "native_model_refusal",
        "RolePipeline requires one native state per declared step",
    )
    for step, state in zip(steps, states, strict=True):
        require(
            isinstance(step, dict) and set(step) <= {"class", "methodId", "params"}
            and isinstance(step.get("params", {}), dict)
            and len(step.get("params", {})) <= 256,
            "native_model_refusal", "RolePipeline step is outside the closed native recipe",
        )
        class_name, method_id = step.get("class"), step.get("methodId")
        require(
            (isinstance(class_name, str) and class_name.startswith("n4m:")
             and 0 < len(class_name[4:].encode("utf-8")) <= 256 and method_id is None)
            or (class_name is None and isinstance(method_id, str)
                and 0 < len(method_id.encode("utf-8")) <= 256),
            "native_model_refusal", "RolePipeline step must declare exactly one Methods ID",
        )
        require(
            isinstance(state, list) and len(state) >= 4
            and all(isinstance(byte, int) and not isinstance(byte, bool)
                    and 0 <= byte <= 255 for byte in state)
            and bytes(state[:4]) == b"N4ME",
            "native_model_refusal", "RolePipeline state must be an opaque native N4ME byte array",
        )


def validate_role_pipeline_recipe(record: dict[str, Any], payload: bytes, plan: dict[str, Any]) -> None:
    """Recompute the declarative native recipe from the effective signed plan."""
    node = plan.get("node_plans", {}).get(record["node_id"])
    graph_node = next((node for node in plan.get("graph_plan", {}).get("graph", {}).get("nodes", [])
                       if node.get("id") == record["node_id"]), None)
    require(isinstance(node, dict) and isinstance(graph_node, dict),
            "native_model_refusal", "RolePipeline requires its effective node and graph operator")
    operator = graph_node.get("operator")
    require(isinstance(operator, dict), "native_model_refusal", "RolePipeline requires an explicit operator")
    kind = operator.get("type")
    if record.get("controller_id") == "controller:methods.native.regression":
        require(record["artifact"].get("plugin") == "dagml.methods.native.regression",
                "native_model_refusal", "native PLS must retain its native owner")
        def closed_recipe(params: Any, *, refit: bool) -> list[dict[str, Any]]:
            require(isinstance(params, dict) and set(params) <= {"native_profile", "n_components", "scale", "pipeline", "phase_controls"}
                    and params.get("native_profile") == "n4m.pls_role_pipeline.v1",
                    "native_model_refusal", "closed PLS profile params required")
            n, scale = params.get("n_components"), params.get("scale")
            controls = params.get("phase_controls", {})
            require(isinstance(controls, dict) and set(controls) <= {"train_params", "refit_params"},
                    "native_model_refusal", "closed PLS phase controls required")
            for phase in ("train_params", "refit_params"):
                values = controls.get(phase, {})
                require(isinstance(values, dict) and set(values) <= {"n_components", "scale"}
                        and all((key == "n_components" and type(value) is int and 0 < value <= 2**31-1)
                                or (key == "scale" and type(value) is bool) for key,value in values.items()),
                        "native_model_refusal", "closed PLS executable controls required")
            if refit:
                n = controls.get("refit_params", {}).get("n_components", n)
                scale = controls.get("refit_params", {}).get("scale", scale)
            require(type(n) is int and 0 < n <= 2**31-1 and type(scale) is bool,
                    "native_model_refusal", "closed PLS component/scale types required")
            steps = []
            pipeline = params.get("pipeline")
            if pipeline is not None:
                require(isinstance(pipeline, dict) and set(pipeline) == {"schema_version", "pipeline_type", "savgol_window", "savgol_poly_degree"}
                        and pipeline["schema_version"] == 1 and pipeline["pipeline_type"] == "n4m.snv_savgol_smooth.v1"
                        and type(pipeline["savgol_window"]) is int and 3 <= pipeline["savgol_window"] <= 501 and pipeline["savgol_window"] % 2 == 1
                        and type(pipeline["savgol_poly_degree"]) is int and 0 <= pipeline["savgol_poly_degree"] < pipeline["savgol_window"],
                        "native_model_refusal", "closed PLS SG bounds required")
                steps = [{"methodId":"preprocessing.scatter.snv","params":{"with_mean":True,"with_std":True,"ddof":0}},
                         {"methodId":"preprocessing.derivatives.savitzky_golay","params":{"window_length":pipeline["savgol_window"],"polyorder":pipeline["savgol_poly_degree"],"deriv":0,"delta":1.0,"mode":"interp","cval":0.0}}]
            steps.append({"methodId":"models.pls.pls_regression","params":{"n_components":n,"solver":"nipals","center_x":True,"center_y":True,"scale_x":scale,"scale_y":scale}})
            return steps
        require("phase_controls" not in graph_node.get("params", {}), "native_model_refusal", "graph cannot inject internal phase controls")
        require(tcv1_sha256(operator) == tcv1_sha256({"type":"N4mRolePipeline","steps":closed_recipe(graph_node["params"],refit=False)}),
                "native_model_refusal", "native PLS graph differs from canonical base recipe")
        saved = load_json_bytes(payload,record["artifact"]["uri"])
        require(tcv1_sha256(saved["steps"]) == tcv1_sha256(closed_recipe(node["params"],refit=True))
                and record["params_fingerprint"] == node["params_fingerprint"],
                "native_model_refusal", "native PLS RAW differs from the effective REFIT recipe")
        return
    if kind == "N4mRolePipeline":
        expected = copy.deepcopy(operator.get("steps"))
    else:
        require(isinstance(kind, str) and kind.startswith("n4m:") and len(kind) > 4,
                "native_model_refusal", "RolePipeline graph operator must be native Methods")
        expected = [{"class": kind, "params": {}}]
    require(isinstance(expected, list) and expected and isinstance(expected[-1], dict),
            "native_model_refusal", "RolePipeline planned recipe must be nonempty")
    last_params = expected[-1].get("params", {})
    require(isinstance(last_params, dict), "native_model_refusal", "RolePipeline planned params must be an object")
    expected[-1]["params"] = {**last_params, **node.get("params", {})}

    def normalized(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"class": step.get("class"), "methodId": step.get("methodId"),
                 "params": step.get("params", {})} for step in steps]

    saved = load_json_bytes(payload, record["artifact"]["uri"])
    require(normalized(saved["steps"]) == normalized(expected)
            and record["params_fingerprint"] == node["params_fingerprint"],
            "native_model_refusal", "RolePipeline saved recipe differs from its effective operator and parameters")


def validate_package_portability(package: Any) -> None:
    require(
        isinstance(package, dict),
        "package_refusal",
        "PortablePredictorPackage V2 must be an object",
    )
    require(
        package.get("schema_version") == 2,
        "version_mixing_refusal",
        "Archive V2 package payload must be exact V2",
    )
    require(
        package.get("execution_bundle", {}).get("schema_version") == 2,
        "version_mixing_refusal",
        "PortablePredictorPackage V2 must embed ExecutionBundle V2",
    )
    require(
        package.get("fitted_artifact_mode") == "portable_required",
        "host_artifact_refusal",
        "Archive V2 package must use portable_required",
    )
    artifact_bindings = package.get("artifact_bindings")
    require(
        isinstance(artifact_bindings, list),
        "package_refusal",
        "Archive V2 package artifact_bindings must be an array",
    )
    require(
        all(
            isinstance(binding, dict)
            and binding.get("load_mode") == "native_portable"
            for binding in artifact_bindings
        ),
        "host_artifact_refusal",
        "Archive V2 package artifact bindings must be native_portable",
    )
    binding_ids = {
        binding["artifact_id"]
        for binding in artifact_bindings
        if isinstance(binding, dict) and isinstance(binding.get("artifact_id"), str)
    }
    require(
        len(binding_ids) == len(artifact_bindings),
        "native_model_refusal",
        "Archive V2 package artifact bindings must have unique artifact ids",
    )
    bundle = package["execution_bundle"]
    refit_artifacts = bundle.get("refit_artifacts")
    raw_payloads = bundle.get("raw_artifact_payloads")
    require(
        isinstance(refit_artifacts, list) and isinstance(raw_payloads, dict),
        "native_model_refusal",
        "Archive V2 package requires refit artifacts and raw payloads",
    )
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in refit_artifacts:
        require(
            isinstance(record, dict) and isinstance(record.get("artifact"), dict),
            "native_model_refusal",
            "Archive V2 refit artifact record is malformed",
        )
        artifact = record["artifact"]
        artifact_id = artifact.get("id")
        require(
            isinstance(artifact_id, str) and artifact_id not in records_by_id,
            "native_model_refusal",
            "Archive V2 refit artifact ids must be unique",
        )
        require(
            artifact.get("kind") in {"n4m_model", "methods_role_pipeline", "methods_multimodal_pipeline", CLASSIFIER_RAW_KIND, CLASSIFIER_META_KIND}
            and artifact.get("backend") == "raw",
            "native_model_refusal",
            f"artifact `{artifact_id}` must use an accepted raw Methods codec",
        )
        if artifact.get("kind") in {"methods_role_pipeline", "methods_multimodal_pipeline", CLASSIFIER_RAW_KIND, CLASSIFIER_META_KIND}:
            records_by_id[artifact_id] = record
            continue
        uri = artifact.get("uri")
        require(
            isinstance(uri, str)
            and _safe_member_path(uri)
            and bool(re.fullmatch(r"methods/[A-Za-z0-9][A-Za-z0-9_.-]*\.n4mm", uri)),
            "native_model_refusal",
            f"artifact `{artifact_id}` must use a safe methods/*.n4mm URI",
        )
        require(
            artifact.get("plugin") is None
            and artifact.get("plugin_version") is None,
            "host_artifact_refusal",
            f"raw Methods artifact `{artifact_id}` cannot carry plugin identity",
        )
        n4mm_abi_requirement(artifact)
        records_by_id[artifact_id] = record
    require(
        set(records_by_id) == binding_ids == set(raw_payloads),
        "native_model_refusal",
        "native bindings, refit artifacts and raw payloads must exactly cover the same artifact ids",
    )
    for artifact_id, raw_payload in raw_payloads.items():
        require(
            isinstance(raw_payload, list)
            and all(
                isinstance(byte, int)
                and not isinstance(byte, bool)
                and 0 <= byte <= 255
                for byte in raw_payload
            ),
            "native_model_refusal",
            f"raw artifact payload `{artifact_id}` is not an exact byte array",
        )
        payload = bytes(raw_payload)
        artifact = records_by_id[artifact_id]["artifact"]
        require(
            artifact.get("size_bytes") == len(payload)
            and artifact.get("content_fingerprint")
            == hashlib.sha256(payload).hexdigest(),
            "native_model_refusal",
            f"raw artifact payload `{artifact_id}` does not match its bundle reference",
        )
        if artifact["kind"] in {CLASSIFIER_RAW_KIND, CLASSIFIER_META_KIND}:
            try:
                validate_classifier_payload(records_by_id[artifact_id], payload, package["effective_plan"], load_json_bytes, validate_raw_encoder_recipe)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                raise ArchiveV2ContractError(f"native_model_refusal: {error}") from error
        elif artifact["kind"] == "methods_role_pipeline":
            validate_role_pipeline_payload(records_by_id[artifact_id], payload)
            validate_role_pipeline_recipe(records_by_id[artifact_id], payload, package["effective_plan"])
        elif artifact["kind"] == "methods_multimodal_pipeline":
            validate_multimodal_pipeline_payload(records_by_id[artifact_id], payload, package["effective_plan"])
    require(
        package.get("package_fingerprint")
        == fingerprint_without(package, "package_fingerprint"),
        "package_fingerprint_refusal",
        "PortablePredictorPackage V2 TCV1 fingerprint mismatch",
    )


def materialize_fixture(
    manifest: dict[str, Any], root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Bind the illustrative manifest to deterministic exact payload bytes."""

    document = copy.deepcopy(manifest)
    payloads = {
        member["path"]: f"archive-v2-fixture:{member['path']}".encode("utf-8")
        for member in document["member_inventory"]
    }
    payloads[PACKAGE_MEMBER] = canonical_portable_package_v2_bytes(root)
    package = canonical_portable_package_v2(root)
    raw_artifact_payloads = package["execution_bundle"]["raw_artifact_payloads"]
    methods = document["payloads"]["methods"]
    for n4mm in methods["n4mm"] + methods.get("role_pipelines", []) + methods.get("multimodal_pipelines", []):
        payloads[n4mm["member_path"]] = bytes(
            raw_artifact_payloads[n4mm["artifact_id"]]
        )

    hashes = {
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in payloads.items()
    }
    package_semantic_fingerprint = package["package_fingerprint"]
    for member in document["member_inventory"]:
        path = member["path"]
        member["raw_sha256"] = hashes[path]
        member["uncompressed_size_bytes"] = len(payloads[path])
        if path == PACKAGE_MEMBER:
            member["semantic_fingerprint"] = package_semantic_fingerprint
    for reference in _refs(
        {"replay": document["replay"], "payloads": document["payloads"]}
    ):
        reference["raw_sha256"] = hashes[reference["member_path"]]
        if reference["member_path"] == PACKAGE_MEMBER:
            reference["semantic_fingerprint"] = package_semantic_fingerprint
        elif reference.get("semantic_profile") in {"n4mm_raw_sha256", ROLE_PIPELINE_PROFILE, MULTIMODAL_PIPELINE_PROFILE}:
            reference["semantic_fingerprint"] = hashes[reference["member_path"]]
    for member in document["member_inventory"]:
        if member.get("semantic_profile") in {"n4mm_raw_sha256", ROLE_PIPELINE_PROFILE, MULTIMODAL_PIPELINE_PROFILE}:
            member["semantic_fingerprint"] = hashes[member["path"]]
    return document, payloads


def rebind_package_integrity(
    document: dict[str, Any], payloads: dict[str, bytes]
) -> None:
    """Update only package raw integrity after an intentional policy mutation."""

    raw_sha256 = hashlib.sha256(payloads[PACKAGE_MEMBER]).hexdigest()
    package = load_json_bytes(payloads[PACKAGE_MEMBER], PACKAGE_MEMBER)
    semantic_fingerprint = package.get("package_fingerprint")
    for member in document["member_inventory"]:
        if member["path"] == PACKAGE_MEMBER:
            member["raw_sha256"] = raw_sha256
            member["uncompressed_size_bytes"] = len(payloads[PACKAGE_MEMBER])
            member["semantic_fingerprint"] = semantic_fingerprint
    reference = document["replay"]["portable_predictor_package"]
    reference["raw_sha256"] = raw_sha256
    reference["semantic_fingerprint"] = semantic_fingerprint


def rebind_member_integrity(
    document: dict[str, Any], payloads: dict[str, bytes], member_path: str
) -> None:
    """Rebind one intentional payload change while retaining semantic identity."""

    raw_sha256 = hashlib.sha256(payloads[member_path]).hexdigest()
    for member in document["member_inventory"]:
        if member["path"] == member_path:
            member["raw_sha256"] = raw_sha256
            member["uncompressed_size_bytes"] = len(payloads[member_path])
    for reference in _refs(
        {"replay": document["replay"], "payloads": document["payloads"]}
    ):
        if reference["member_path"] == member_path:
            reference["raw_sha256"] = raw_sha256


def validate_refit_output_scope(outcome: dict[str, Any]) -> None:
    """Independently check the additive training-only OOF REFIT fallback."""
    plan = outcome.get("effective_plan", {})
    edges = plan.get("graph_plan", {}).get("graph", {}).get("edges", [])
    records = outcome.get("execution_bundle", {}).get("refit_artifacts", [])
    lineage = outcome.get("lineage", [])
    reports = outcome.get("score_set", {}).get("reports", [])
    influence = {sample for entry in outcome.get("training_influence", {}).get("entries", [])
                 for key in ("physical_sample_ids", "origin_sample_ids") for sample in entry.get(key, [])}
    for output in outcome.get("outputs", []):
        artifact_only = output.get("artifact_only", False)
        cohort = output.get("refit_test_cohort")
        if not artifact_only and cohort is None:
            continue
        binding = output["binding"]
        node_id = binding["node_id"]
        node = plan.get("node_plans", {}).get(node_id, {})
        owner = node.get("controller_id")
        require(outcome.get("schema_version") == output.get("schema_version") == 2
                and outcome.get("refit", {}).get("requested") is True
                and outcome.get("refit", {}).get("status") == "completed"
                and binding.get("prediction_source") == "final_refit"
                and node.get("kind") == "model"
                and "REFIT" in node.get("supported_phases", [])
                and {"emits_artifacts", "emits_predictions"} <= set(node.get("controller_capabilities", []))
                and any(edge.get("target", {}).get("node_id") == node_id
                        and edge.get("contract", {}).get("requires_oof") is True for edge in edges),
                "refit_output_refusal", "Fallback requires a completed signed OOF model REFIT")
        artifacts = [record for record in records if record.get("node_id") == node_id
                     and record.get("controller_id") == owner and record.get("artifact", {}).get("controller_id") == owner]
        fitted = [record for record in lineage if record.get("node_id") == node_id
                  and record.get("phase") == "REFIT" and record.get("fold_id") is None
                  and record.get("controller_id") == owner and record.get("variant_id") == outcome.get("selected_variant_id")]
        require(bool(artifacts) and len(fitted) == 1
                and all(record["artifact"] in fitted[0].get("artifact_refs", []) for record in artifacts),
                "refit_output_refusal", "Fallback lacks exact selected fitted artifacts and lineage")
        blocks = output.get("predictions", [])
        if artifact_only:
            require(cohort is None and not blocks and not output.get("observation_predictions")
                    and not output.get("aggregated_predictions")
                    and not any(report.get("producer_node") == node_id and report.get("partition") == "test"
                                and report.get("fold_id") is None for report in reports),
                    "refit_output_refusal", "Artifact-only marker cannot hide external-test data")
            continue
        cohort_preimage = {key: cohort.get(key) for key in (
            "role", "physical_sample_ids", "origin_sample_ids", "target_names",
            "relation_fingerprint", "relations", "data_content_fingerprint", "target_content_fingerprint")}
        require(cohort.get("role") == "external_test" and cohort.get("target_names") == binding.get("target_names")
                and cohort.get("cohort_fingerprint") == _serde_sha256(cohort_preimage)
                and cohort.get("relation_fingerprint") == replay_relation_fingerprint(cohort["relations"]),
                "refit_output_refusal", "External-test authority fingerprint or targets differ")
        expected = set(cohort["physical_sample_ids"])
        origins = set(cohort["origin_sample_ids"])
        relation_rows = cohort["relations"]["records"]
        require(expected == {row["sample_id"] for row in relation_rows}
                and origins == {row.get("origin_sample_id") or row["sample_id"] for row in relation_rows}
                and not (expected | origins) & (influence | set(plan.get("fold_set", {}).get("sample_ids", [])))
                and binding.get("prediction_level") == "sample" and bool(blocks)
                and not output.get("observation_predictions") and not output.get("aggregated_predictions")
                and all(block.get("partition") == "test" and block.get("fold_id") is None
                        and block.get("producer_node") == node_id
                        and block.get("producer_port") in (None, binding.get("port_name"))
                        and block.get("target_names") == binding.get("target_names")
                        and len(block.get("sample_ids", [])) == len(expected)
                        and set(block.get("sample_ids", [])) == expected for block in blocks),
                "refit_output_refusal", "REFIT Test blocks differ from their exact disjoint cohort")


def validate_archive_v2_payloads(
    document: dict[str, Any],
    payloads: dict[str, bytes],
    validator: Draft202012Validator,
    *,
    root: Path = ROOT,
) -> None:
    """Run the future writer's fail-closed checks without writing a ZIP."""

    schema_validate(document, validator)
    validate_semantics(document)
    inventory = {member["path"]: member for member in document["member_inventory"]}
    require(
        set(payloads) == set(inventory),
        "member_inventory_refusal",
        "pre-write payload set does not equal the closed inventory",
    )
    for path, payload in payloads.items():
        member = inventory[path]
        require(
            len(payload) == member["uncompressed_size_bytes"],
            "member_integrity_refusal",
            f"uncompressed size mismatch for `{path}`",
        )
        require(
            hashlib.sha256(payload).hexdigest() == member["raw_sha256"],
            "member_integrity_refusal",
            f"raw payload SHA-256 mismatch for `{path}`",
        )

    package = load_json_bytes(payloads[PACKAGE_MEMBER], PACKAGE_MEMBER)
    schemas, registry = contract_schema_registry(root)
    validate_instance(
        package,
        schemas[PACKAGE_V2_SCHEMA_ID],
        registry,
        "PortablePredictorPackage V2 member",
    )
    validate_package_portability(package)
    outcome_reference = document["replay"]["training_artifacts"]["training_outcome"]
    outcome_path = outcome_reference["member_path"]
    outcome_bytes = payloads[outcome_path]
    # The historical illustrative fixture deliberately uses this exact opaque
    # literal for non-package members. Every real outcome must remain strict JSON.
    if outcome_bytes != f"archive-v2-fixture:{outcome_path}".encode("utf-8"):
        outcome = load_json_bytes(outcome_bytes, outcome_path)
        require(isinstance(outcome, dict), "json_refusal", "training outcome must be a JSON object")
        if any(output.get("artifact_only", False) or output.get("refit_test_cohort") is not None
               for output in outcome.get("outputs", [])):
            validate_instance(outcome, schemas[RUNTIME_V2_ARTIFACTS["training_outcome"]], registry,
                              "TrainingOutcome V2 REFIT scope member")
            validate_refit_output_scope(outcome)
    require(
        document["replay"]["portable_predictor_package"]["semantic_fingerprint"]
        == package["package_fingerprint"],
        "member_semantic_refusal",
        "package reference does not bind the package TCV1 fingerprint",
    )
    raw_artifact_payloads = package["execution_bundle"]["raw_artifact_payloads"]
    refit_by_id = {
        record["artifact"]["id"]: record["artifact"]
        for record in package["execution_bundle"]["refit_artifacts"]
    }
    methods = document["payloads"]["methods"]
    n4mm_by_id = {
        reference["artifact_id"]: reference
        for reference in methods["n4mm"] + methods.get("role_pipelines", []) + methods.get("multimodal_pipelines", [])
    }
    require(
        set(n4mm_by_id) == set(raw_artifact_payloads),
        "native_model_refusal",
        "Archive V2 N4MM and RolePipeline members must exactly cover package raw artifacts",
    )
    for artifact_id, raw_payload in raw_artifact_payloads.items():
        reference = n4mm_by_id[artifact_id]
        artifact = refit_by_id[artifact_id]
        is_role = artifact["kind"] in {"methods_role_pipeline", CLASSIFIER_META_KIND}
        is_multimodal = artifact["kind"] in {"methods_multimodal_pipeline", CLASSIFIER_RAW_KIND}
        if is_multimodal:
            require(reference in methods.get("multimodal_pipelines", []) and reference.get("kind") == artifact["kind"]
                    and reference.get("owner") == "dag-ml" and reference.get("format_version") == 1,
                    "native_model_refusal", "Complete multimodal artifact must use its own manifest family")
        elif is_role:
            require(
                reference in methods.get("role_pipelines", [])
                and reference.get("kind") == artifact["kind"]
                and reference.get("owner") == "dag-ml"
                and reference.get("format_version") == 1,
                "native_model_refusal", "RolePipeline artifact must use its own manifest family",
            )
        else:
            require(
                reference in methods["n4mm"], "native_model_refusal",
                "N4MM artifact must use its own manifest family",
            )
            artifact_abi = n4mm_abi_requirement(artifact)
            reference_abi = n4mm_reference_abi_requirement(reference, artifact)
            require(
                reference_abi == artifact_abi,
                "native_model_refusal",
                f"N4MM member `{artifact_id}` ABI minimum differs from package artifact",
            )
        require(
            reference["member_path"] == refit_by_id[artifact_id]["uri"],
            "native_model_refusal",
            f"Methods member path for `{artifact_id}` does not match package URI",
        )
        require(
            payloads[reference["member_path"]] == bytes(raw_payload),
            "native_model_refusal",
            f"Methods member `{artifact_id}` differs from package raw bytes",
        )
        raw_sha256 = hashlib.sha256(payloads[reference["member_path"]]).hexdigest()
        require(
            reference["semantic_profile"] == (CLASSIFIER_RAW_PROFILE if artifact["kind"] == CLASSIFIER_RAW_KIND else CLASSIFIER_META_PROFILE if artifact["kind"] == CLASSIFIER_META_KIND else MULTIMODAL_PIPELINE_PROFILE if is_multimodal else ROLE_PIPELINE_PROFILE if is_role else "n4mm_raw_sha256")
            and reference["semantic_fingerprint"] == raw_sha256,
            "native_model_refusal",
            f"Methods member `{artifact_id}` must use its exact raw SHA-256 semantic profile",
        )


def _regular_zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def write_fixture_zip(
    path: Path, document: dict[str, Any], payloads: dict[str, bytes]
) -> None:
    """Write a deterministic small ZIP used only by contract tests."""

    manifest_bytes = json.dumps(
        document, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_regular_zip_info("manifest.json"), manifest_bytes)
        for member_path in sorted(payloads):
            archive.writestr(_regular_zip_info(member_path), payloads[member_path])


def validate_archive_zip(
    path: Path,
    validator: Draft202012Validator,
    *,
    root: Path = ROOT,
) -> None:
    """Validate one materialized Archive V2 without extracting it."""

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveV2ContractError(f"zip_refusal: invalid ZIP {path}: {exc}") from exc
    with archive:
        infos = archive.infolist()
        require(
            len(infos) <= 256,
            "budget_refusal",
            "central directory exceeds entry budget",
        )
        names: set[str] = set()
        total = 0
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            name = info.filename
            require(
                _safe_member_path(name),
                "member_path_refusal",
                f"unsafe central-directory path `{name}`",
            )
            require(
                name not in names,
                "member_inventory_refusal",
                f"duplicate ZIP member `{name}`",
            )
            names.add(name)
            by_name[name] = info
            mode = (info.external_attr >> 16) & 0xFFFF
            require(
                info.create_system != 3 or stat.S_ISREG(mode),
                "member_type_refusal",
                f"ZIP member `{name}` is not regular",
            )
            require(
                info.file_size <= 134217728,
                "budget_refusal",
                f"ZIP member `{name}` exceeds per-member budget",
            )
            ratio = info.file_size / max(info.compress_size, 1)
            require(
                ratio <= 100,
                "compression_refusal",
                f"ZIP member `{name}` exceeds compression-ratio budget",
            )
            total += info.file_size
        require(
            total <= 536870912,
            "budget_refusal",
            "ZIP exceeds total uncompressed budget",
        )
        require(
            "manifest.json" in by_name,
            "dispatch_refusal",
            "ZIP lacks exact manifest.json dispatch member",
        )
        manifest_info = by_name["manifest.json"]
        require(
            manifest_info.file_size <= 134217728,
            "budget_refusal",
            "manifest exceeds dispatch budget",
        )
        document = load_json_bytes(archive.read(manifest_info), "manifest.json")
        schema_validate(document, validator)
        validate_semantics(document)
        expected_names = {
            "manifest.json",
            *(member["path"] for member in document["member_inventory"]),
        }
        require(
            names == expected_names,
            "member_inventory_refusal",
            "ZIP members do not equal manifest closed inventory",
        )
        payloads: dict[str, bytes] = {}
        for member in document["member_inventory"]:
            info = by_name[member["path"]]
            require(
                info.file_size == member["uncompressed_size_bytes"],
                "member_integrity_refusal",
                f"central-directory size mismatch for `{member['path']}`",
            )
            payloads[member["path"]] = archive.read(info)
        validate_archive_v2_payloads(
            document, payloads, validator, root=root
        )


def _mutation_parent(document: Any, path: list[Any]) -> tuple[Any, Any]:
    require(bool(path), "fixture_refusal", "mutation path cannot be empty")
    parent = document
    for token in path[:-1]:
        parent = parent[token]
    return parent, path[-1]


def apply_mutations(document: Any, mutations: list[dict[str, Any]]) -> Any:
    mutated = copy.deepcopy(document)
    for mutation in mutations:
        operation = mutation["op"]
        parent, token = _mutation_parent(mutated, mutation["path"])
        if operation in {"add", "replace"}:
            parent[token] = copy.deepcopy(mutation["value"])
        elif operation == "delete":
            del parent[token]
        elif operation == "replace_all":
            values = parent[token]
            require(
                isinstance(values, list),
                "fixture_refusal",
                "replace_all target must be an array",
            )
            for value in values:
                value[mutation["field"]] = copy.deepcopy(mutation["value"])
        else:
            raise ArchiveV2ContractError(
                f"fixture_refusal: unknown mutation operation `{operation}`"
            )
    return mutated


def _expect_refusal(expected: str, operation: Any, case_id: str) -> None:
    try:
        operation()
    except (ArchiveV2ContractError, ArchiveV1ContractError) as exc:
        require(
            str(exc).startswith(f"{expected}:"),
            "fixture_refusal",
            f"{case_id} expected {expected}, received {exc}",
        )
    except Exception as exc:
        raise ArchiveV2ContractError(
            f"fixture_refusal: {case_id} raised unstable {type(exc).__name__}: {exc}"
        ) from exc
    else:
        raise ArchiveV2ContractError(
            f"fixture_refusal: {case_id} unexpectedly passed; expected {expected}"
        )


def validate_refusal_cases(
    manifest: dict[str, Any],
    refusals: dict[str, Any],
    validator: Draft202012Validator,
    *,
    root: Path = ROOT,
) -> None:
    cases = refusals.get("cases")
    require(isinstance(cases, list), "fixture_refusal", "refusals cases missing")
    case_ids = {case.get("id") for case in cases}
    require(
        case_ids == REQUIRED_REFUSAL_CASE_IDS,
        "fixture_refusal",
        "Archive V2 refusal case ids drifted",
    )
    require(
        len(cases) == len(case_ids),
        "fixture_refusal",
        "Archive V2 refusal ids must be unique",
    )

    archive_v1 = load_v1_json(
        ARCHIVE_V1_ROOT / "fixtures/positive/portable_split_conformal.json"
    )
    archive_v1_schema = validate_schema_v1(
        load_v1_json(ARCHIVE_V1_ROOT / "archive_workspace_manifest.v1.schema.json")
    )
    for case in cases:
        case_id = case["id"]
        expected = case["expected_error"]
        if case["base"] == "archive_v1":
            mutated_v1 = apply_mutations(archive_v1, case.get("mutations", []))
            _expect_refusal(
                expected,
                lambda value=mutated_v1: schema_validate_v1(
                    value, archive_v1_schema
                ),
                case_id,
            )
            continue

        if "package_mutations" in case:
            document, payloads = materialize_fixture(manifest, root)
            package = load_json_bytes(payloads[PACKAGE_MEMBER], PACKAGE_MEMBER)
            package = apply_mutations(package, case["package_mutations"])
            package["package_fingerprint"] = fingerprint_without(
                package, "package_fingerprint"
            )
            payloads[PACKAGE_MEMBER] = json.dumps(
                package,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if case.get("rebind_package_integrity"):
                rebind_package_integrity(document, payloads)
            _expect_refusal(
                expected,
                lambda document=document, payloads=payloads: validate_archive_v2_payloads(
                    document, payloads, validator, root=root
                ),
                case_id,
            )
            continue

        if "physical_tamper_member" in case:
            document, payloads = materialize_fixture(manifest, root)
            member_path = case["physical_tamper_member"]
            payloads[member_path] = payloads[member_path] + b"tamper"
            if case.get("rebind_tampered_member_integrity"):
                rebind_member_integrity(document, payloads, member_path)
            _expect_refusal(
                expected,
                lambda document=document, payloads=payloads: validate_archive_v2_payloads(
                    document, payloads, validator, root=root
                ),
                case_id,
            )
            continue

        mutated = apply_mutations(manifest, case.get("mutations", []))

        def validate_mutated(value: dict[str, Any] = mutated) -> None:
            schema_validate(value, validator)
            validate_semantics(value)

        _expect_refusal(expected, validate_mutated, case_id)


def validate_archive_v2_contract(root: Path = ROOT) -> None:
    """Run the complete local Archive V2 contract gate."""

    archive_root = root / "docs/contracts/archive-v2"
    schema = load_json(archive_root / SCHEMA_NAME)
    validator = validate_schema(schema)
    manifest = load_json(archive_root / "fixtures/positive" / POSITIVE_NAME)
    refusals = load_json(archive_root / "fixtures/negative" / REFUSALS_NAME)
    schema_validate(manifest, validator)
    validate_semantics(manifest)
    validate_package_schema_boundary(root)
    document, payloads = materialize_fixture(manifest, root)
    validate_archive_v2_payloads(document, payloads, validator, root=root)
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "native-portable-v2.n4a"
        write_fixture_zip(archive_path, document, payloads)
        validate_archive_zip(archive_path, validator, root=root)
    validate_refusal_cases(manifest, refusals, validator, root=root)


def main() -> int:
    try:
        validate_archive_v2_contract(ROOT)
    except ArchiveV2ContractError as exc:
        print(f"Archive V2 contract validation failed: {exc}", file=sys.stderr)
        return 1
    print("Archive V2 contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
