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
GRAPH_SPEC_SCHEMA_REL = Path("docs/contracts/graph_spec.schema.json")
MODEL_INPUT_SPEC_SCHEMA_REL = Path("docs/contracts/model_input_spec.schema.json")
DATA_PLAN_SCHEMA_REL = Path("docs/contracts/data_plan.schema.json")
CONFORMANCE_PACK_REL = Path("docs/contracts/conformance_pack.v1.json")
OPENLINEAGE_FACETS_SCHEMA_REL = Path("docs/contracts/openlineage_dagml_facets.schema.json")
PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_REL = Path(
    "docs/contracts/prediction_cache_tensor_metadata.schema.json"
)
DATA_OUTPUT_PROVENANCE_SCHEMA_REL = Path(
    "docs/contracts/data_output_provenance.schema.json"
)
PROCESS_ADAPTER_DESCRIPTION_SCHEMA_REL = Path(
    "docs/contracts/process_adapter_description.schema.json"
)
RESEARCH_PROVENANCE_PROFILE_REL = Path(
    "docs/contracts/research_provenance_package_profile.v1.json"
)
LOCAL_FIXTURE_REL = Path("examples/fixtures/data/coordinator_data_plan_envelope_nir.json")
LOCAL_FEATURE_FUSION_FIXTURE_REL = Path(
    "examples/fixtures/data/feature_fusion_selector_nir_chem.json"
)
LOCAL_GRAPH_SPEC_FIXTURE_REL = Path("examples/branch_merge_oof_graph.json")
LOCAL_MODEL_INPUT_SPEC_FIXTURE_REL = Path(
    "examples/fixtures/data/model_input_spec_tabular_regressor.json"
)
LOCAL_DATA_PLAN_FIXTURE_REL = Path("examples/fixtures/data/data_plan_tabular_fusion.json")
LOCAL_DATA_OUTPUT_PROVENANCE_FIXTURE_REL = Path(
    "examples/fixtures/runtime/data_output_provenance_augmented_view.json"
)
LOCAL_PROCESS_ADAPTER_DESCRIPTION_FIXTURE_REL = Path(
    "examples/fixtures/runtime/process_adapter_description_python.json"
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
GRAPH_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "graph_spec.v1.schema.json"
)
MODEL_INPUT_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "model_input_spec.v1.schema.json"
)
DATA_PLAN_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "data_plan.v1.schema.json"
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
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
CONFORMANCE_PACK_ID = "dag-ml.shared.conformance.v1"
OPENLINEAGE_FACETS_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "openlineage_dagml_facets.v1.schema.json"
)
PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "prediction_cache_tensor_metadata.v1.schema.json"
)
DATA_OUTPUT_PROVENANCE_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "data_output_provenance.v1.schema.json"
)
PROCESS_ADAPTER_DESCRIPTION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "process_adapter_description.v1.schema.json"
)
RESEARCH_PROVENANCE_PROFILE_ID = "dag-ml.research_provenance_package.v1"


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


def require_identifier(value: Any, label: str) -> None:
    require(
        isinstance(value, str) and IDENTIFIER_RE.fullmatch(value) is not None,
        f"{label} must be a DAG-ML identifier",
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


def validate_graph_spec_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} GraphSpec schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} GraphSpec schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == GRAPH_SPEC_SCHEMA_ID, f"{label} GraphSpec schema $id mismatch")
    require(schema.get("type") == "object", f"{label} GraphSpec root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} GraphSpec root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} GraphSpec required list is missing")
    for field in ("id", "nodes"):
        require(field in required, f"{label} GraphSpec schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} GraphSpec schema definitions are missing")
    expected_node_kinds = [
        "transform",
        "y_transform",
        "split",
        "model",
        "fork",
        "map",
        "feature_join",
        "prediction_join",
        "mixed_join",
        "source_join",
        "tag",
        "exclude",
        "augmentation",
        "adapter",
        "aggregator",
        "generator",
        "restructure",
        "tuner",
        "subgraph",
        "chart",
    ]
    require(
        defs.get("node_kind", {}).get("enum") == expected_node_kinds,
        f"{label} GraphSpec node_kind enum is not aligned with Rust",
    )
    require(
        defs.get("port_kind", {}).get("enum")
        == ["data", "target", "prediction", "artifact", "metric", "control"],
        f"{label} GraphSpec port_kind enum is not aligned with Rust",
    )
    require(
        defs.get("port_cardinality", {}).get("enum") == ["one", "many", "optional"],
        f"{label} GraphSpec port_cardinality enum is not aligned with Rust",
    )
    for definition_name in (
        "port_spec",
        "port_schema",
        "port_ref",
        "edge_contract",
        "edge_spec",
        "graph_interface",
        "node_spec",
    ):
        require(
            definition_name in defs,
            f"{label} GraphSpec schema misses `{definition_name}`",
        )


def validate_model_input_spec_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} ModelInputSpec schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} ModelInputSpec schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == MODEL_INPUT_SPEC_SCHEMA_ID,
        f"{label} ModelInputSpec schema $id mismatch",
    )
    require(schema.get("type") == "object", f"{label} ModelInputSpec root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} ModelInputSpec root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} ModelInputSpec required list missing")
    for field in ("schema_version", "ports"):
        require(field in required, f"{label} ModelInputSpec schema must require `{field}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} ModelInputSpec properties missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} ModelInputSpec schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} ModelInputSpec $defs missing")
    require("input_port" in defs, f"{label} ModelInputSpec schema misses input_port")
    require("fusion_policy" in defs, f"{label} ModelInputSpec schema misses fusion_policy")
    require(
        defs.get("fusion_policy", {}).get("properties", {}).get("mode", {}).get("enum")
        == [
            "single_source",
            "concatenate_features",
            "stack_samples",
            "dict_by_source",
            "custom",
        ],
        f"{label} ModelInputSpec fusion modes are not aligned",
    )


def validate_data_plan_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} DataPlan schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} DataPlan schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == DATA_PLAN_SCHEMA_ID, f"{label} DataPlan schema $id mismatch")
    require(schema.get("type") == "object", f"{label} DataPlan root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} DataPlan root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} DataPlan required list missing")
    for field in ("schema_version", "id", "steps", "output_ports"):
        require(field in required, f"{label} DataPlan schema must require `{field}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} DataPlan properties missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} DataPlan schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} DataPlan $defs missing")
    require("data_plan_step" in defs, f"{label} DataPlan schema misses data_plan_step")
    require(
        defs.get("data_plan_step_kind", {}).get("enum")
        == ["materialize", "adapt", "align", "join", "collate"],
        f"{label} DataPlan step kinds are not aligned",
    )


def validate_openlineage_facets_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} OpenLineage facets schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} OpenLineage facets schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == OPENLINEAGE_FACETS_SCHEMA_ID,
        f"{label} OpenLineage facets schema has unexpected $id",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} OpenLineage facets schema $defs are missing")
    for definition_name in (
        "DagmlReproducibilityRunFacet",
        "DagmlOofSafetyRunFacet",
        "DagmlPlanJobFacet",
        "DagmlDatasetContractFacet",
    ):
        definition = defs.get(definition_name)
        require(
            isinstance(definition, dict),
            f"{label} OpenLineage facets schema misses `{definition_name}`",
        )
        required = definition.get("required")
        require(
            isinstance(required, list) and "_schemaURL" in required,
            f"{label} `{definition_name}` must require _schemaURL",
        )
        require(
            definition.get("additionalProperties") in {False, True},
            f"{label} `{definition_name}` must declare additionalProperties explicitly",
        )


def validate_prediction_cache_tensor_metadata_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} prediction-cache tensor metadata schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} prediction-cache tensor metadata schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_ID,
        f"{label} prediction-cache tensor metadata schema has unexpected $id",
    )
    require(
        schema.get("type") == "object",
        f"{label} prediction-cache tensor metadata root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} prediction-cache tensor metadata root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} prediction-cache tensor metadata required list is missing",
    )
    for field in (
        "schema_version",
        "requirement_key",
        "cache_id",
        "prediction_level",
        "rows",
        "cols",
        "blocks",
    ):
        require(
            field in required,
            f"{label} prediction-cache tensor metadata must require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict),
        f"{label} prediction-cache tensor metadata properties are missing",
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} prediction-cache tensor metadata schema_version const must be 1",
    )
    require(
        properties.get("prediction_level", {}).get("enum") == ["sample", "target", "group"],
        f"{label} prediction-cache tensor metadata prediction_level enum mismatch",
    )
    defs = schema.get("$defs")
    require(
        isinstance(defs, dict) and "block_metadata" in defs and "prediction_unit_id" in defs,
        f"{label} prediction-cache tensor metadata schema definitions are incomplete",
    )


def validate_data_output_provenance_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} data-output provenance schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} data-output provenance schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == DATA_OUTPUT_PROVENANCE_SCHEMA_ID,
        f"{label} data-output provenance schema has unexpected $id",
    )
    require(
        schema.get("type") == "object",
        f"{label} data-output provenance root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} data-output provenance root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} data-output provenance required list is missing",
    )
    for field in ("schema_version", "producer_node", "producer_port", "producer_phase"):
        require(
            field in required,
            f"{label} data-output provenance schema must require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict),
        f"{label} data-output provenance properties are missing",
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} data-output provenance schema_version const must be 1",
    )
    require(
        properties.get("producer_phase", {}).get("enum")
        == ["COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"],
        f"{label} data-output provenance phase enum mismatch",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} data-output provenance $defs are missing")
    require(
        defs.get("identifier", {}).get("pattern") == "^[A-Za-z0-9_.:-]+$",
        f"{label} data-output provenance identifier definition mismatch",
    )
    require(
        defs.get("sha256", {}).get("pattern") == "^[0-9A-Fa-f]{64}$",
        f"{label} data-output provenance sha256 definition mismatch",
    )
    shape_delta = defs.get("shape_delta")
    require(isinstance(shape_delta, dict), f"{label} data-output shape_delta definition missing")
    require(
        shape_delta.get("additionalProperties") is False,
        f"{label} data-output shape_delta must reject unknown fields",
    )
    shape_delta_required = shape_delta.get("required")
    require(
        isinstance(shape_delta_required, list)
        and "node_id" in shape_delta_required
        and "kind" in shape_delta_required
        and "before_fingerprint" in shape_delta_required
        and "after_fingerprint" in shape_delta_required,
        f"{label} data-output shape_delta required fields mismatch",
    )


def validate_process_adapter_description_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} process-adapter description schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} process-adapter description schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == PROCESS_ADAPTER_DESCRIPTION_SCHEMA_ID,
        f"{label} process-adapter description schema has unexpected $id",
    )
    require(
        schema.get("type") == "object",
        f"{label} process-adapter description root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} process-adapter description root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} process-adapter description required list is missing",
    )
    for field in (
        "schema_version",
        "protocol",
        "adapter_id",
        "supported_modes",
        "capabilities",
    ):
        require(
            field in required,
            f"{label} process-adapter description must require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict),
        f"{label} process-adapter description properties are missing",
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} process-adapter description schema_version const must be 1",
    )
    require(
        properties.get("protocol", {}).get("const") == "dag-ml-process-adapter",
        f"{label} process-adapter protocol const mismatch",
    )
    supported_modes = properties.get("supported_modes")
    require(
        isinstance(supported_modes, dict)
        and supported_modes.get("uniqueItems") is True
        and supported_modes.get("items", {}).get("enum") == ["one_shot", "jsonl"],
        f"{label} process-adapter supported_modes contract mismatch",
    )
    capabilities = properties.get("capabilities")
    require(
        isinstance(capabilities, dict)
        and capabilities.get("uniqueItems") is True
        and capabilities.get("minItems") == 2,
        f"{label} process-adapter capabilities contract mismatch",
    )
    defs = schema.get("$defs")
    require(
        isinstance(defs, dict)
        and "identifier" in defs
        and "capability" in defs,
        f"{label} process-adapter schema definitions are incomplete",
    )


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


def validate_graph_spec(graph: Any, label: str) -> None:
    require(isinstance(graph, dict), f"{label} GraphSpec must be a JSON object")
    require_non_empty_string(graph.get("id"), f"{label}.id")
    nodes = graph.get("nodes")
    require(isinstance(nodes, list) and nodes, f"{label}.nodes must be non-empty")

    node_ports: dict[str, dict[str, dict[str, str]]] = {}
    for index, node in enumerate(nodes):
        node_label = f"{label}.nodes[{index}]"
        require(isinstance(node, dict), f"{node_label} must be an object")
        node_id = node.get("id")
        require_identifier(node_id, f"{node_label}.id")
        require(node_id not in node_ports, f"{label} has duplicate node id `{node_id}`")
        require(
            node.get("kind")
            in {
                "transform",
                "y_transform",
                "split",
                "model",
                "fork",
                "map",
                "feature_join",
                "prediction_join",
                "mixed_join",
                "source_join",
                "tag",
                "exclude",
                "augmentation",
                "adapter",
                "aggregator",
                "generator",
                "restructure",
                "tuner",
                "subgraph",
                "chart",
            },
            f"{node_label}.kind is invalid",
        )
        ports = node.get("ports", {})
        require(isinstance(ports, dict), f"{node_label}.ports must be an object when present")
        node_ports[node_id] = {
            "inputs": graph_port_kinds(ports.get("inputs", []), f"{node_label}.ports.inputs"),
            "outputs": graph_port_kinds(ports.get("outputs", []), f"{node_label}.ports.outputs"),
        }

    edges = graph.get("edges", [])
    require(isinstance(edges, list), f"{label}.edges must be an array when present")
    for index, edge in enumerate(edges):
        edge_label = f"{label}.edges[{index}]"
        require(isinstance(edge, dict), f"{edge_label} must be an object")
        source = edge.get("source")
        target = edge.get("target")
        contract = edge.get("contract")
        require(isinstance(source, dict), f"{edge_label}.source must be an object")
        require(isinstance(target, dict), f"{edge_label}.target must be an object")
        require(isinstance(contract, dict), f"{edge_label}.contract must be an object")

        source_node = source.get("node_id")
        target_node = target.get("node_id")
        require_identifier(source_node, f"{edge_label}.source.node_id")
        require_identifier(target_node, f"{edge_label}.target.node_id")
        require(source_node in node_ports, f"{edge_label} references missing source `{source_node}`")
        require(target_node in node_ports, f"{edge_label} references missing target `{target_node}`")
        source_port = source.get("port_name")
        target_port = target.get("port_name")
        require_non_empty_string(source_port, f"{edge_label}.source.port_name")
        require_non_empty_string(target_port, f"{edge_label}.target.port_name")
        source_kind = node_ports[source_node]["outputs"].get(source_port)
        target_kind = node_ports[target_node]["inputs"].get(target_port)
        require(source_kind is not None, f"{edge_label} source port `{source_port}` is missing")
        require(target_kind is not None, f"{edge_label} target port `{target_port}` is missing")
        edge_kind = contract.get("kind")
        require(
            edge_kind == source_kind == target_kind,
            f"{edge_label} kind `{edge_kind}` does not match endpoint ports",
        )
        if contract.get("requires_oof") is True:
            require(edge_kind == "prediction", f"{edge_label} requires OOF on non-prediction edge")


def graph_port_kinds(ports: Any, label: str) -> dict[str, str]:
    require(isinstance(ports, list), f"{label} must be an array")
    seen: dict[str, str] = {}
    for index, port in enumerate(ports):
        port_label = f"{label}[{index}]"
        require(isinstance(port, dict), f"{port_label} must be an object")
        name = port.get("name")
        require_non_empty_string(name, f"{port_label}.name")
        require(name not in seen, f"{label} contains duplicate port `{name}`")
        kind = port.get("kind")
        require(
            kind in {"data", "target", "prediction", "artifact", "metric", "control"},
            f"{port_label}.kind is invalid",
        )
        seen[name] = kind
    return seen


def validate_model_input_spec(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} ModelInputSpec must be an object")
    require(value.get("schema_version") == 1, f"{label}.schema_version must be 1")
    ports = value.get("ports")
    require(isinstance(ports, list) and ports, f"{label}.ports must be non-empty")
    port_names: list[str] = []
    for index, port in enumerate(ports):
        port_label = f"{label}.ports[{index}]"
        require(isinstance(port, dict), f"{port_label} must be an object")
        require_non_empty_string(port.get("name"), f"{port_label}.name")
        port_names.append(port["name"])
        for field in ("accepted_representations", "accepted_types"):
            values = port.get(field)
            require(isinstance(values, list) and values, f"{port_label}.{field} must be non-empty")
            require(len(set(values)) == len(values), f"{port_label}.{field} has duplicates")
            for value_index, item in enumerate(values):
                require_non_empty_string(item, f"{port_label}.{field}[{value_index}]")
        rank = port.get("rank")
        if rank is not None:
            require(isinstance(rank, int) and 0 <= rank <= 16, f"{port_label}.rank is invalid")
        for field in ("multi_source", "optional"):
            if field in port:
                require(isinstance(port[field], bool), f"{port_label}.{field} must be boolean")
        metadata = port.get("metadata")
        if metadata is not None:
            require(isinstance(metadata, dict), f"{port_label}.metadata must be an object")
    require(len(set(port_names)) == len(port_names), f"{label}.ports contain duplicate names")

    fusion = value.get("default_fusion")
    if fusion is not None:
        require(isinstance(fusion, dict), f"{label}.default_fusion must be an object")
        mode = fusion.get("mode")
        require(
            mode
            in {
                "single_source",
                "concatenate_features",
                "stack_samples",
                "dict_by_source",
                "custom",
            },
            f"{label}.default_fusion.mode is invalid",
        )
        for field in ("alignment", "adapter_id"):
            field_value = fusion.get(field)
            if field_value is not None:
                require_non_empty_string(field_value, f"{label}.default_fusion.{field}")
        if mode == "custom":
            require_non_empty_string(
                fusion.get("adapter_id"),
                f"{label}.default_fusion.adapter_id",
            )


def validate_data_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} DataPlan must be an object")
    require(value.get("schema_version") == 1, f"{label}.schema_version must be 1")
    require_non_empty_string(value.get("id"), f"{label}.id")
    steps = value.get("steps")
    require(isinstance(steps, list) and steps, f"{label}.steps must be non-empty")
    outputs: set[str] = set()
    for index, step in enumerate(steps):
        step_label = f"{label}.steps[{index}]"
        require(isinstance(step, dict), f"{step_label} must be an object")
        kind = step.get("kind")
        require(
            kind in {"materialize", "adapt", "align", "join", "collate"},
            f"{step_label}.kind is invalid",
        )
        inputs = step.get("inputs", [])
        require(isinstance(inputs, list), f"{step_label}.inputs must be an array")
        if kind != "materialize":
            require(inputs, f"{step_label}.inputs must be non-empty")
        for input_index, input_name in enumerate(inputs):
            require_non_empty_string(input_name, f"{step_label}.inputs[{input_index}]")
            if kind != "materialize":
                require(
                    input_name in outputs,
                    f"{step_label}.inputs[{input_index}] references an unknown prior output",
                )
        output = step.get("output")
        require_non_empty_string(output, f"{step_label}.output")
        require(output not in outputs, f"{step_label}.output duplicates a prior output")
        outputs.add(output)
        adapter_id = step.get("adapter_id")
        if adapter_id is not None:
            require_non_empty_string(adapter_id, f"{step_label}.adapter_id")
        params = step.get("params")
        if params is not None:
            require(isinstance(params, dict), f"{step_label}.params must be an object")

    output_ports = value.get("output_ports")
    require(isinstance(output_ports, dict) and output_ports, f"{label}.output_ports must be non-empty")
    for port_name, output in output_ports.items():
        require_non_empty_string(port_name, f"{label}.output_ports key")
        require_non_empty_string(output, f"{label}.output_ports[{port_name}]")
        require(output in outputs, f"{label}.output_ports[{port_name}] references unknown output")
    for field in ("warnings", "requires_user_choice"):
        values = value.get(field, [])
        require(isinstance(values, list), f"{label}.{field} must be an array")
        for index, item in enumerate(values):
            require_non_empty_string(item, f"{label}.{field}[{index}]")


def validate_data_output_provenance(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} data-output provenance must be an object")
    require(value.get("schema_version") == 1, f"{label} schema_version must be 1")
    require_non_empty_string(value.get("producer_node"), f"{label}.producer_node")
    require_non_empty_string(value.get("producer_port"), f"{label}.producer_port")
    require(
        value.get("producer_phase")
        in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
        f"{label}.producer_phase is invalid",
    )
    for field in ("variant_id", "fold_id", "feature_namespace"):
        field_value = value.get(field)
        if field_value is not None:
            require_non_empty_string(field_value, f"{label}.{field}")
    for field in (
        "shape_plan_fingerprint",
        "aggregation_policy_fingerprint",
        "feature_schema_fingerprint",
    ):
        field_value = value.get(field)
        if field_value is not None:
            require_sha256(field_value, f"{label}.{field}")
    deltas = value.get("shape_deltas", [])
    require(isinstance(deltas, list), f"{label}.shape_deltas must be an array")
    last_feature_after = None
    for index, delta in enumerate(deltas):
        delta_label = f"{label}.shape_deltas[{index}]"
        require(isinstance(delta, dict), f"{delta_label} must be an object")
        require(
            delta.get("node_id") == value.get("producer_node"),
            f"{delta_label}.node_id must match producer_node",
        )
        require(
            delta.get("kind") in {"row", "feature", "target", "prediction"},
            f"{delta_label}.kind is invalid",
        )
        require_sha256(delta.get("before_fingerprint"), f"{delta_label}.before_fingerprint")
        require_sha256(delta.get("after_fingerprint"), f"{delta_label}.after_fingerprint")
        if delta.get("kind") == "feature":
            last_feature_after = delta.get("after_fingerprint")
        metadata = delta.get("metadata")
        if metadata is not None:
            require(isinstance(metadata, dict), f"{delta_label}.metadata must be an object")
    if last_feature_after is not None:
        require(
            value.get("feature_schema_fingerprint") == last_feature_after,
            f"{label}.feature_schema_fingerprint must match the last feature delta",
        )


def validate_process_adapter_description(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} process-adapter description must be an object")
    require(value.get("schema_version") == 1, f"{label}.schema_version must be 1")
    require(
        value.get("protocol") == "dag-ml-process-adapter",
        f"{label}.protocol must be dag-ml-process-adapter",
    )
    require_non_empty_string(value.get("adapter_id"), f"{label}.adapter_id")
    modes = value.get("supported_modes")
    require(isinstance(modes, list) and modes, f"{label}.supported_modes must be non-empty")
    require(len(set(modes)) == len(modes), f"{label}.supported_modes contain duplicates")
    for index, mode in enumerate(modes):
        require(
            mode in {"one_shot", "jsonl"},
            f"{label}.supported_modes[{index}] is invalid",
        )
    capabilities = value.get("capabilities")
    require(
        isinstance(capabilities, list) and capabilities,
        f"{label}.capabilities must be non-empty",
    )
    require(
        len(set(capabilities)) == len(capabilities),
        f"{label}.capabilities contain duplicates",
    )
    for index, capability in enumerate(capabilities):
        require_non_empty_string(capability, f"{label}.capabilities[{index}]")
    for required_capability in ("node_task_json_v1", "node_result_json_v1"):
        require(
            required_capability in capabilities,
            f"{label}.capabilities must include `{required_capability}`",
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


def validate_dag_ml_prediction_cache_tensor_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION=1",
    )
    for symbol in (
        "DagMlF64Tensor",
        "dagml_f64_tensor_free",
        "dagml_prediction_cache_payload_f64_tensor_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_controller_result_header(header: str, label: str) -> None:
    for symbol in (
        "dagml_node_result_validate_for_task_json",
        "dagml_controller_manifest_validate_json",
        "dagml_controller_manifest_list_validate_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_graph_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_GRAPH_SPEC_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_GRAPH_SPEC_SCHEMA_VERSION=1",
    )
    for symbol in ("dagml_graph_spec_contract_json", "dagml_graph_validate_json"):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_data_shape_header(header: str, label: str) -> None:
    for macro in (
        "#define DAG_ML_MODEL_INPUT_SPEC_SCHEMA_VERSION 1u",
        "#define DAG_ML_DATA_PLAN_SCHEMA_VERSION 1u",
    ):
        require(macro in header, f"{label} header must declare `{macro}`")
    for symbol in (
        "dagml_model_input_spec_contract_json",
        "dagml_model_input_spec_validate_json",
        "dagml_data_plan_contract_json",
        "dagml_data_plan_validate_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_data_output_provenance_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION=1",
    )
    require(
        '#define DAG_ML_DATA_OUTPUT_PROVENANCE_EXTRA_KEY "dag_ml_output"' in header,
        f"{label} header must declare the data-output provenance extra key",
    )
    for symbol in (
        "dagml_data_output_provenance_contract_json",
        "dagml_data_output_provenance_validate_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


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


def validate_research_provenance_profile(
    profile: Any,
    openlineage_facets_schema: Any,
    label: str,
) -> None:
    require(isinstance(profile, dict), f"{label} research provenance profile must be an object")
    require(
        profile.get("schema_version") == 1,
        f"{label} research provenance profile schema_version must be 1",
    )
    require(
        profile.get("profile_id") == RESEARCH_PROVENANCE_PROFILE_ID,
        f"{label} research provenance profile id mismatch",
    )

    package = profile.get("package")
    require(isinstance(package, dict), f"{label} profile package must be an object")
    required_files = package.get("required_files")
    require(isinstance(required_files, list), f"{label} profile required_files must be a list")
    required_by_path = {}
    for index, record in enumerate(required_files):
        record_label = f"{label} profile required_files[{index}]"
        require(isinstance(record, dict), f"{record_label} must be an object")
        path = record.get("path")
        require_non_empty_string(path, f"{record_label}.path")
        require(path not in required_by_path, f"{label} profile duplicates required path `{path}`")
        require_non_empty_string(record.get("kind"), f"{record_label}.kind")
        require(
            isinstance(record.get("checksum_in_ro_crate"), bool),
            f"{record_label}.checksum_in_ro_crate must be boolean",
        )
        required_by_path[path] = record
    for path in (
        "execution_plan.json",
        "execution_bundle.json",
        "lineage_records.json",
        "lineage.prov.jsonld",
        "ro-crate-metadata.json",
    ):
        require(path in required_by_path, f"{label} profile must require `{path}`")
    require(
        required_by_path["ro-crate-metadata.json"].get("checksum_in_ro_crate") is False,
        f"{label} profile must not require RO-Crate metadata to checksum itself",
    )
    for path, record in required_by_path.items():
        if path != "ro-crate-metadata.json":
            require(
                record.get("checksum_in_ro_crate") is True,
                f"{label} profile must require checksum for `{path}`",
            )

    optional_files = package.get("optional_files")
    require(isinstance(optional_files, list), f"{label} profile optional_files must be a list")
    optional_kinds = set()
    for index, record in enumerate(optional_files):
        record_label = f"{label} profile optional_files[{index}]"
        require(isinstance(record, dict), f"{record_label} must be an object")
        require_non_empty_string(record.get("kind"), f"{record_label}.kind")
        optional_kinds.add(record["kind"])
        require(
            isinstance(record.get("checksum_in_ro_crate"), bool),
            f"{record_label}.checksum_in_ro_crate must be boolean",
        )
        require(
            "path" in record or "path_pattern" in record,
            f"{record_label} must declare path or path_pattern",
        )
        if "path_pattern" in record:
            require_non_empty_string(record["path_pattern"], f"{record_label}.path_pattern")
            try:
                re.compile(record["path_pattern"])
            except re.error as exc:
                raise ContractError(f"{record_label}.path_pattern is invalid: {exc}") from exc
    for kind in (
        "dagml.prediction_cache_manifest",
        "dagml.artifact_manifest",
        "dagml.external_data_plan_envelope",
    ):
        require(kind in optional_kinds, f"{label} profile must include optional kind `{kind}`")

    ro_crate = profile.get("ro_crate")
    require(isinstance(ro_crate, dict), f"{label} profile ro_crate must be an object")
    require(
        ro_crate.get("metadata_file") == "ro-crate-metadata.json",
        f"{label} profile RO-Crate metadata file mismatch",
    )
    require(
        ro_crate.get("root_dataset_id") == "./",
        f"{label} profile RO-Crate root id must be ./",
    )
    require(
        ro_crate.get("workflow_type") == "ComputationalWorkflow",
        f"{label} profile must require ComputationalWorkflow",
    )
    required_properties = ro_crate.get("required_file_properties")
    require(
        isinstance(required_properties, list),
        f"{label} profile RO-Crate required_file_properties must be a list",
    )
    for field in ("sha256", "dagml:sha256", "contentSize", "encodingFormat"):
        require(field in required_properties, f"{label} profile RO-Crate must require `{field}`")
    require(
        ro_crate.get("required_json_encoding") == "application/json",
        f"{label} profile must require application/json encoding",
    )

    prov_jsonld = profile.get("prov_jsonld")
    require(isinstance(prov_jsonld, dict), f"{label} profile prov_jsonld must be an object")
    require(
        prov_jsonld.get("file") == "lineage.prov.jsonld",
        f"{label} profile PROV JSON-LD file mismatch",
    )
    sections = prov_jsonld.get("required_sections")
    require(isinstance(sections, list), f"{label} profile PROV sections must be a list")
    for section in (
        "entity",
        "activity",
        "agent",
        "used",
        "wasGeneratedBy",
        "wasDerivedFrom",
        "wasAssociatedWith",
    ):
        require(section in sections, f"{label} profile must require PROV section `{section}`")

    openlineage = profile.get("openlineage")
    require(isinstance(openlineage, dict), f"{label} profile openlineage must be an object")
    require(
        openlineage.get("command") == "export-open-lineage",
        f"{label} profile OpenLineage command mismatch",
    )
    require(
        openlineage.get("facet_schema") == OPENLINEAGE_FACETS_SCHEMA_REL.name,
        f"{label} profile OpenLineage facet schema mismatch",
    )
    defs = openlineage_facets_schema.get("$defs")
    require(isinstance(defs, dict), f"{label} OpenLineage facets schema $defs are missing")
    for facet_key, definition_name in (
        ("dagml_reproducibility", "DagmlReproducibilityRunFacet"),
        ("dagml_oof_safety", "DagmlOofSafetyRunFacet"),
    ):
        require(
            facet_key in openlineage.get("required_run_facets", []),
            f"{label} profile must require OpenLineage run facet `{facet_key}`",
        )
        require(definition_name in defs, f"{label} facet schema must define `{definition_name}`")
    require(
        "dagml_plan" in openlineage.get("required_job_facets", []),
        f"{label} profile must require OpenLineage job facet `dagml_plan`",
    )
    require(
        "DagmlPlanJobFacet" in defs,
        f"{label} facet schema must define `DagmlPlanJobFacet`",
    )

    cli_conformance = profile.get("cli_conformance")
    require(isinstance(cli_conformance, dict), f"{label} profile cli_conformance must be an object")
    require(
        cli_conformance.get("export_command") == "export-research-provenance",
        f"{label} profile export command mismatch",
    )
    require(
        cli_conformance.get("validation_command") == "validate-research-provenance",
        f"{label} profile validation command mismatch",
    )
    required_tests = cli_conformance.get("required_tests")
    require(isinstance(required_tests, list), f"{label} profile required_tests must be a list")
    for test_id in (
        "cli_exports_research_provenance_bundle",
        "cli_selects_builds_and_validates_replay_bundle",
    ):
        require(test_id in required_tests, f"{label} profile must require test `{test_id}`")


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
        local_graph_spec_schema = load_json(ROOT / GRAPH_SPEC_SCHEMA_REL)
        local_model_input_spec_schema = load_json(ROOT / MODEL_INPUT_SPEC_SCHEMA_REL)
        local_data_plan_schema = load_json(ROOT / DATA_PLAN_SCHEMA_REL)
        local_pack = load_json(ROOT / CONFORMANCE_PACK_REL)
        local_openlineage_facets_schema = load_json(ROOT / OPENLINEAGE_FACETS_SCHEMA_REL)
        local_prediction_cache_tensor_metadata_schema = load_json(
            ROOT / PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_REL
        )
        local_data_output_provenance_schema = load_json(
            ROOT / DATA_OUTPUT_PROVENANCE_SCHEMA_REL
        )
        local_process_adapter_description_schema = load_json(
            ROOT / PROCESS_ADAPTER_DESCRIPTION_SCHEMA_REL
        )
        local_research_provenance_profile = load_json(ROOT / RESEARCH_PROVENANCE_PROFILE_REL)
        local_fixture = load_json(ROOT / LOCAL_FIXTURE_REL)
        local_feature_fusion_fixture = load_json(ROOT / LOCAL_FEATURE_FUSION_FIXTURE_REL)
        local_graph_spec_fixture = load_json(ROOT / LOCAL_GRAPH_SPEC_FIXTURE_REL)
        local_model_input_spec_fixture = load_json(ROOT / LOCAL_MODEL_INPUT_SPEC_FIXTURE_REL)
        local_data_plan_fixture = load_json(ROOT / LOCAL_DATA_PLAN_FIXTURE_REL)
        local_data_output_provenance_fixture = load_json(
            ROOT / LOCAL_DATA_OUTPUT_PROVENANCE_FIXTURE_REL
        )
        local_process_adapter_description_fixture = load_json(
            ROOT / LOCAL_PROCESS_ADAPTER_DESCRIPTION_FIXTURE_REL
        )
        local_header = load_text(ROOT / LOCAL_C_HEADER_REL)
        validate_schema_artifact(local_schema, LOCAL_SCHEMA_ID, "dag-ml")
        validate_feature_fusion_schema_artifact(
            local_feature_fusion_schema,
            LOCAL_FEATURE_FUSION_SCHEMA_ID,
            "dag-ml",
        )
        validate_graph_spec_schema(local_graph_spec_schema, "dag-ml")
        validate_model_input_spec_schema(local_model_input_spec_schema, "dag-ml")
        validate_data_plan_schema(local_data_plan_schema, "dag-ml")
        validate_openlineage_facets_schema(local_openlineage_facets_schema, "dag-ml")
        validate_prediction_cache_tensor_metadata_schema(
            local_prediction_cache_tensor_metadata_schema,
            "dag-ml",
        )
        validate_data_output_provenance_schema(
            local_data_output_provenance_schema,
            "dag-ml",
        )
        validate_process_adapter_description_schema(
            local_process_adapter_description_schema,
            "dag-ml",
        )
        validate_envelope(local_fixture, "dag-ml")
        validate_feature_fusion_selector(local_feature_fusion_fixture, "dag-ml")
        validate_graph_spec(local_graph_spec_fixture, "dag-ml")
        validate_model_input_spec(local_model_input_spec_fixture, "dag-ml")
        validate_data_plan(local_data_plan_fixture, "dag-ml")
        validate_data_output_provenance(local_data_output_provenance_fixture, "dag-ml")
        validate_process_adapter_description(
            local_process_adapter_description_fixture,
            "dag-ml",
        )
        validate_data_provider_header(local_header, "dag-ml")
        validate_dag_ml_prediction_cache_tensor_header(local_header, "dag-ml")
        validate_dag_ml_controller_result_header(local_header, "dag-ml")
        validate_dag_ml_graph_header(local_header, "dag-ml")
        validate_dag_ml_data_shape_header(local_header, "dag-ml")
        validate_dag_ml_data_output_provenance_header(local_header, "dag-ml")
        validate_conformance_pack(
            local_pack,
            local_schema,
            local_feature_fusion_schema,
            local_fixture,
            local_feature_fusion_fixture,
            local_header,
            "dag-ml",
        )
        validate_research_provenance_profile(
            local_research_provenance_profile,
            local_openlineage_facets_schema,
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
