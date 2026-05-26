#!/usr/bin/env python3
"""Validate shared contract artifacts with dag-ml-data.

The script intentionally uses only the Python standard library so CI can run it
before any project dependency is installed. It validates the published envelope
schema shape, validates the local fixture shape, and compares the sibling schema
copy when a dag-ml-data checkout is available.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_REL = Path("docs/contracts/coordinator_data_plan_envelope.schema.json")
FEATURE_FUSION_SCHEMA_REL = Path("docs/contracts/feature_fusion_selector.schema.json")
CONFORMANCE_PACK_REL = Path("docs/contracts/conformance_pack.v1.json")
LOCAL_FIXTURE_REL = Path("examples/fixtures/data/coordinator_data_plan_envelope_nir.json")
LOCAL_FEATURE_FUSION_FIXTURE_REL = Path(
    "examples/fixtures/data/feature_fusion_selector_nir_chem.json"
)
LOCAL_C_HEADER_REL = Path("crates/dag-ml-capi/include/dag_ml.h")
SIBLING_FIXTURE_REL = Path(
    "examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
)
SIBLING_FEATURE_FUSION_FIXTURE_REL = Path(
    "examples/fixtures/oof_campaign/feature_fusion_selector_nir_chem.json"
)
SIBLING_C_HEADER_REL = Path("crates/dag-ml-data-capi/include/dag_ml_data.h")
LOCAL_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "coordinator_data_plan_envelope.v1.schema.json"
)
LOCAL_FEATURE_FUSION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "feature_fusion_selector.v1.schema.json"
)
SIBLING_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml-data/schemas/"
    "coordinator_data_plan_envelope.v1.schema.json"
)
SIBLING_FEATURE_FUSION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml-data/schemas/"
    "feature_fusion_selector.v1.schema.json"
)
SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
CONFORMANCE_PACK_ID = "dag-ml.shared.conformance.v1"


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(f"missing text file: {path}") from exc


def require_non_empty_string(value: Any, label: str) -> None:
    require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")


def require_sha256(value: Any, label: str) -> None:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a 64-character hex digest",
    )


def validate_schema_artifact(schema: Any, expected_id: str, label: str) -> None:
    require(isinstance(schema, dict), f"{label} schema must be a JSON object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == expected_id, f"{label} schema has unexpected $id")
    require(schema.get("type") == "object", f"{label} schema root must be an object")

    required = schema.get("required")
    require(isinstance(required, list), f"{label} schema required list is missing")
    for field in ("schema_version", "schema_fingerprint", "plan_fingerprint", "plan"):
        require(field in required, f"{label} schema does not require `{field}`")

    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} schema properties are missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} schema_version const must be 1",
    )

    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} schema $defs are missing")
    require(
        defs.get("sha256", {}).get("pattern") == "^[0-9A-Fa-f]{64}$",
        f"{label} sha256 definition is not the expected contract",
    )

    relation = defs.get("coordinator_relation")
    require(isinstance(relation, dict), f"{label} relation definition is missing")
    relation_required = relation.get("required")
    require(
        isinstance(relation_required, list)
        and "observation_id" in relation_required
        and "sample_id" in relation_required,
        f"{label} relation must require observation_id and sample_id",
    )
    require(
        relation.get("additionalProperties") is False,
        f"{label} relation must reject unknown identity fields",
    )


def validate_feature_fusion_schema_artifact(schema: Any, expected_id: str, label: str) -> None:
    require(isinstance(schema, dict), f"{label} feature-fusion schema must be a JSON object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} feature-fusion schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == expected_id,
        f"{label} feature-fusion schema has unexpected $id",
    )
    require(schema.get("type") == "object", f"{label} feature-fusion root must be an object")
    required = schema.get("required")
    require(isinstance(required, list), f"{label} feature-fusion required list is missing")
    for field in ("schema_version", "feature_set_id", "sources", "alignment"):
        require(field in required, f"{label} feature-fusion schema does not require `{field}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} feature-fusion properties are missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} feature-fusion schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} feature-fusion $defs are missing")
    for name in ("source", "alignment", "presence_mask"):
        require(name in defs, f"{label} feature-fusion schema misses `{name}` definition")


def validate_envelope(envelope: Any, label: str) -> None:
    require(isinstance(envelope, dict), f"{label} envelope must be a JSON object")
    require(envelope.get("schema_version") == 1, f"{label} envelope schema_version must be 1")
    require_sha256(envelope.get("schema_fingerprint"), f"{label} schema_fingerprint")
    require_sha256(envelope.get("plan_fingerprint"), f"{label} plan_fingerprint")
    relation_fingerprint = envelope.get("relation_fingerprint")
    if relation_fingerprint is not None:
        require_sha256(relation_fingerprint, f"{label} relation_fingerprint")

    plan = envelope.get("plan")
    require(isinstance(plan, dict), f"{label} plan must be an object")
    require_non_empty_string(plan.get("id"), f"{label} plan.id")
    require(isinstance(plan.get("steps"), list), f"{label} plan.steps must be an array")
    require_non_empty_string(
        plan.get("output_representation"), f"{label} plan.output_representation"
    )

    relations = envelope.get("coordinator_relations")
    if relations is None:
        return
    require(isinstance(relations, dict), f"{label} coordinator_relations must be an object")
    records = relations.get("records")
    require(
        isinstance(records, list) and records,
        f"{label} coordinator_relations.records must be a non-empty array",
    )
    for index, record in enumerate(records):
        record_label = f"{label} coordinator relation #{index}"
        require(isinstance(record, dict), f"{record_label} must be an object")
        require_non_empty_string(record.get("observation_id"), f"{record_label}.observation_id")
        require_non_empty_string(record.get("sample_id"), f"{record_label}.sample_id")
        for field in ("target_id", "group_id", "origin_sample_id", "source_id"):
            value = record.get(field)
            if value is not None:
                require_non_empty_string(value, f"{record_label}.{field}")
        if "is_augmented" in record:
            require(
                isinstance(record["is_augmented"], bool),
                f"{record_label}.is_augmented must be boolean",
            )


def validate_feature_fusion_selector(selector: Any, label: str) -> None:
    require(isinstance(selector, dict), f"{label} selector must be a JSON object")
    require(selector.get("schema_version") == 1, f"{label} selector schema_version must be 1")
    require_non_empty_string(selector.get("feature_set_id"), f"{label}.feature_set_id")
    sources = selector.get("sources")
    require(isinstance(sources, list) and sources, f"{label}.sources must be a non-empty array")
    source_ids: list[str] = []
    for index, source in enumerate(sources):
        source_label = f"{label}.sources[{index}]"
        require(isinstance(source, dict), f"{source_label} must be an object")
        require_non_empty_string(source.get("source_id"), f"{source_label}.source_id")
        require_non_empty_string(source.get("feature_set_id"), f"{source_label}.feature_set_id")
        source_ids.append(source["source_id"])
        columns = source.get("columns")
        if columns is not None:
            require(
                isinstance(columns, list) and columns,
                f"{source_label}.columns must be a non-empty array when present",
            )
            for column_index, column in enumerate(columns):
                require_non_empty_string(column, f"{source_label}.columns[{column_index}]")
    require(len(set(source_ids)) == len(source_ids), f"{label}.sources contain duplicate source ids")

    alignment = selector.get("alignment")
    require(isinstance(alignment, dict), f"{label}.alignment must be an object")
    require(
        alignment.get("mode") in {"inner", "left", "outer"},
        f"{label}.alignment.mode must be inner, left or outer",
    )
    sample_ids = alignment.get("sample_ids")
    require(
        isinstance(sample_ids, list) and sample_ids,
        f"{label}.alignment.sample_ids must be a non-empty array",
    )
    for index, sample_id in enumerate(sample_ids):
        require_non_empty_string(sample_id, f"{label}.alignment.sample_ids[{index}]")
    require(
        len(set(sample_ids)) == len(sample_ids),
        f"{label}.alignment.sample_ids contain duplicates",
    )
    masks = alignment.get("masks")
    require(isinstance(masks, list) and masks, f"{label}.alignment.masks must be non-empty")
    mask_source_ids: list[str] = []
    for index, mask in enumerate(masks):
        mask_label = f"{label}.alignment.masks[{index}]"
        require(isinstance(mask, dict), f"{mask_label} must be an object")
        require_non_empty_string(mask.get("source_id"), f"{mask_label}.source_id")
        mask_source_ids.append(mask["source_id"])
        require(mask.get("sample_ids") == sample_ids, f"{mask_label}.sample_ids order mismatch")
        present = mask.get("present")
        require(
            isinstance(present, list) and len(present) == len(sample_ids),
            f"{mask_label}.present length must match sample_ids",
        )
        for present_index, value in enumerate(present):
            require(isinstance(value, bool), f"{mask_label}.present[{present_index}] must be bool")
    require(set(mask_source_ids) == set(source_ids), f"{label}.alignment masks must match sources")

    policy = selector.get("policy")
    if policy is not None:
        require(isinstance(policy, dict), f"{label}.policy must be an object")
        namespace_columns = policy.get("namespace_columns")
        if namespace_columns is not None:
            require(
                isinstance(namespace_columns, bool),
                f"{label}.policy.namespace_columns must be bool",
            )


def validate_data_provider_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION 2u" in header,
        f"{label} header must declare DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION=2",
    )
    require(
        "#define DAG_ML_DATA_VTABLE_DEFINED" in header,
        f"{label} header must guard the shared DagMlDataVTable definition",
    )
    require(
        "typedef struct DagMlDataVTable" in header,
        f"{label} header must expose DagMlDataVTable",
    )
    for field in (
        "materialize",
        "make_view",
        "view_identity",
        "target_arrow",
        "feature_arrow",
        "release",
        "destroy",
    ):
        require(field in header, f"{label} DagMlDataVTable must expose `{field}`")


def validate_dag_ml_data_tensor_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_DATA_TENSOR_F64_ABI_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_DATA_TENSOR_F64_ABI_VERSION=1",
    )
    require("DagMlDataTensorF64" in header, f"{label} header must expose DagMlDataTensorF64")
    require(
        "dagmldata_inmemory_provider_feature_collation_tensor_f64_json" in header,
        f"{label} header must expose provider tensor collation",
    )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_schema(schema: Any) -> Any:
    normalized = copy.deepcopy(schema)
    if isinstance(normalized, dict):
        normalized.pop("$id", None)
    return normalized


def validate_digest_record(
    record: Any,
    expected_sha256: str,
    expected_kind: str | None,
    expected_schema_version: int | None,
    label: str,
) -> None:
    require(isinstance(record, dict), f"{label} must be an object")
    if expected_kind is not None:
        require(record.get("kind") == expected_kind, f"{label}.kind must be {expected_kind}")
    if expected_schema_version is not None:
        require(
            record.get("schema_version") == expected_schema_version,
            f"{label}.schema_version must be {expected_schema_version}",
        )
    digest = record.get("normalized_sha256", record.get("canonical_json_sha256"))
    require_sha256(digest, f"{label} digest")
    require(digest == expected_sha256, f"{label} digest does not match local artifact")


def validate_conformance_pack(
    pack: Any,
    schema: Any,
    feature_fusion_schema: Any,
    fixture: Any,
    feature_fusion_fixture: Any,
    header: str,
    label: str,
) -> None:
    require(isinstance(pack, dict), f"{label} conformance pack must be a JSON object")
    require(pack.get("schema_version") == 1, f"{label} conformance pack schema_version must be 1")
    require(pack.get("pack_id") == CONFORMANCE_PACK_ID, f"{label} conformance pack id mismatch")

    contracts = pack.get("contracts")
    require(isinstance(contracts, dict), f"{label} conformance pack contracts must be an object")
    validate_digest_record(
        contracts.get("coordinator_data_plan_envelope.v1"),
        canonical_json_sha256(normalize_schema(schema)),
        "json_schema",
        1,
        f"{label} coordinator envelope contract",
    )
    validate_digest_record(
        contracts.get("feature_fusion_selector.v1"),
        canonical_json_sha256(normalize_schema(feature_fusion_schema)),
        "json_schema",
        1,
        f"{label} feature fusion selector contract",
    )

    fixtures = pack.get("fixtures")
    require(isinstance(fixtures, dict), f"{label} conformance pack fixtures must be an object")
    coordinator_fixture = fixtures.get("coordinator_data_plan_envelope_nir.v1")
    validate_digest_record(
        coordinator_fixture,
        canonical_json_sha256(fixture),
        None,
        None,
        f"{label} coordinator envelope fixture",
    )
    require(
        coordinator_fixture.get("contract") == "coordinator_data_plan_envelope.v1",
        f"{label} coordinator fixture must reference coordinator contract",
    )
    fusion_fixture = fixtures.get("feature_fusion_selector_nir_chem.v1")
    validate_digest_record(
        fusion_fixture,
        canonical_json_sha256(feature_fusion_fixture),
        None,
        None,
        f"{label} feature fusion fixture",
    )
    require(
        fusion_fixture.get("contract") == "feature_fusion_selector.v1",
        f"{label} feature fusion fixture must reference feature fusion contract",
    )

    c_abi = pack.get("c_abi")
    require(isinstance(c_abi, dict), f"{label} conformance pack c_abi must be an object")
    require(
        c_abi.get("data_provider_vtable_abi_version") == 2,
        f"{label} provider ABI version must be 2",
    )
    callbacks = c_abi.get("required_provider_callbacks")
    require(isinstance(callbacks, list), f"{label} required callbacks must be a list")
    for callback in (
        "materialize",
        "make_view",
        "view_identity",
        "target_arrow",
        "feature_arrow",
        "release",
        "destroy",
    ):
        require(callback in callbacks, f"{label} conformance pack must require `{callback}`")
        require(callback in header, f"{label} header must expose `{callback}`")
    data_symbols = c_abi.get("required_dag_ml_data_symbols")
    require(isinstance(data_symbols, list), f"{label} dag-ml-data symbols must be a list")
    if "DagMlDataTensorF64" in header:
        require(
            c_abi.get("data_tensor_f64_abi_version") == 1,
            f"{label} f64 tensor ABI version must be 1",
        )
        for symbol in data_symbols:
            require_non_empty_string(symbol, f"{label} dag-ml-data symbol")
            require(symbol in header, f"{label} header must expose `{symbol}`")

    cross_repo = pack.get("cross_repo_conformance")
    require(isinstance(cross_repo, dict), f"{label} cross_repo_conformance must be an object")
    required_tests = cross_repo.get("required_when_sibling_checkout_present")
    require(isinstance(required_tests, list), f"{label} cross-repo tests must be a list")
    for test_id in (
        "contracts.schema_and_fixture_equivalence",
        "headers.include_order",
        "provider.f64_predict_replay",
    ):
        require(test_id in required_tests, f"{label} conformance pack must require `{test_id}`")


def candidate_sibling_roots() -> list[Path]:
    candidates = []
    env_path = os.environ.get("DAG_ML_DATA_REPO")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(ROOT.parent / "dag-ml-data")
    candidates.append(ROOT / "external" / "dag-ml-data")
    return candidates


def sibling_root() -> Path | None:
    env_path = os.environ.get("DAG_ML_DATA_REPO")
    for candidate in candidate_sibling_roots():
        if candidate.exists():
            return candidate.resolve()
    if env_path:
        raise ContractError(f"DAG_ML_DATA_REPO points to a missing checkout: {env_path}")
    return None


def main() -> int:
    try:
        local_schema = load_json(ROOT / SCHEMA_REL)
        local_feature_fusion_schema = load_json(ROOT / FEATURE_FUSION_SCHEMA_REL)
        local_pack = load_json(ROOT / CONFORMANCE_PACK_REL)
        local_fixture = load_json(ROOT / LOCAL_FIXTURE_REL)
        local_feature_fusion_fixture = load_json(ROOT / LOCAL_FEATURE_FUSION_FIXTURE_REL)
        local_header = load_text(ROOT / LOCAL_C_HEADER_REL)
        validate_schema_artifact(local_schema, LOCAL_SCHEMA_ID, "dag-ml")
        validate_feature_fusion_schema_artifact(
            local_feature_fusion_schema,
            LOCAL_FEATURE_FUSION_SCHEMA_ID,
            "dag-ml",
        )
        validate_envelope(local_fixture, "dag-ml")
        validate_feature_fusion_selector(local_feature_fusion_fixture, "dag-ml")
        validate_data_provider_header(local_header, "dag-ml")
        validate_conformance_pack(
            local_pack,
            local_schema,
            local_feature_fusion_schema,
            local_fixture,
            local_feature_fusion_fixture,
            local_header,
            "dag-ml",
        )

        sibling = sibling_root()
        if sibling is None:
            print("validated dag-ml contract; sibling dag-ml-data checkout not present")
            return 0

        sibling_schema = load_json(sibling / SCHEMA_REL)
        sibling_feature_fusion_schema = load_json(sibling / FEATURE_FUSION_SCHEMA_REL)
        sibling_pack = load_json(sibling / CONFORMANCE_PACK_REL)
        sibling_fixture = load_json(sibling / SIBLING_FIXTURE_REL)
        sibling_feature_fusion_fixture = load_json(
            sibling / SIBLING_FEATURE_FUSION_FIXTURE_REL
        )
        sibling_header = load_text(sibling / SIBLING_C_HEADER_REL)
        validate_schema_artifact(sibling_schema, SIBLING_SCHEMA_ID, "dag-ml-data")
        validate_feature_fusion_schema_artifact(
            sibling_feature_fusion_schema,
            SIBLING_FEATURE_FUSION_SCHEMA_ID,
            "dag-ml-data",
        )
        validate_envelope(sibling_fixture, "dag-ml-data")
        validate_feature_fusion_selector(sibling_feature_fusion_fixture, "dag-ml-data")
        validate_data_provider_header(sibling_header, "dag-ml-data")
        validate_dag_ml_data_tensor_header(sibling_header, "dag-ml-data")
        validate_conformance_pack(
            sibling_pack,
            sibling_schema,
            sibling_feature_fusion_schema,
            sibling_fixture,
            sibling_feature_fusion_fixture,
            sibling_header,
            "dag-ml-data",
        )
        require(
            normalize_schema(local_schema) == normalize_schema(sibling_schema),
            "coordinator envelope schemas diverge beyond repository-specific $id",
        )
        require(
            normalize_schema(local_feature_fusion_schema)
            == normalize_schema(sibling_feature_fusion_schema),
            "feature fusion selector schemas diverge beyond repository-specific $id",
        )
        require(
            local_fixture == sibling_fixture,
            "coordinator envelope fixtures diverge",
        )
        require(
            local_feature_fusion_fixture == sibling_feature_fusion_fixture,
            "feature fusion selector fixtures diverge",
        )
        require(local_pack == sibling_pack, "shared conformance packs diverge")
        print(f"validated dag-ml contract against dag-ml-data at {sibling}")
        return 0
    except ContractError as exc:
        print(f"contract validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
