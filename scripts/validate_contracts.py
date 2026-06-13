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
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_REL = Path("docs/contracts/coordinator_data_plan_envelope.schema.json")
FEATURE_FUSION_SCHEMA_REL = Path("docs/contracts/feature_fusion_selector.schema.json")
BRANCH_VIEW_SCHEMA_REL = Path("docs/contracts/coordinator_branch_view.schema.json")
FITTED_ADAPTER_SCHEMA_REL = Path("docs/contracts/fitted_adapter_ref.schema.json")
GRAPH_SPEC_SCHEMA_REL = Path("docs/contracts/graph_spec.schema.json")
PIPELINE_DSL_SCHEMA_REL = Path("docs/contracts/pipeline_dsl.schema.json")
CAMPAIGN_SPEC_SCHEMA_REL = Path("docs/contracts/campaign_spec.schema.json")
EXECUTION_PLAN_SCHEMA_REL = Path("docs/contracts/execution_plan.schema.json")
MODEL_INPUT_SPEC_SCHEMA_REL = Path("docs/contracts/model_input_spec.schema.json")
DATA_PLAN_SCHEMA_REL = Path("docs/contracts/data_plan.schema.json")
CONTROLLER_MANIFEST_SCHEMA_REL = Path("docs/contracts/controller_manifest.schema.json")
SELECTION_POLICY_SCHEMA_REL = Path("docs/contracts/selection_policy.schema.json")
SELECTION_DECISION_SCHEMA_REL = Path("docs/contracts/selection_decision.schema.json")
CONFORMANCE_PACK_REL = Path("docs/contracts/conformance_pack.v1.json")
PARITY_ORACLE_REL = Path("docs/contracts/parity_oracle.v1.json")
OPENLINEAGE_FACETS_SCHEMA_REL = Path("docs/contracts/openlineage_dagml_facets.schema.json")
PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_REL = Path(
    "docs/contracts/prediction_cache_tensor_metadata.schema.json"
)
PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_REL = Path(
    "docs/contracts/prediction_cache_columnar_tensor_metadata.schema.json"
)
AGGREGATION_CONTROLLER_TASK_SCHEMA_REL = Path(
    "docs/contracts/aggregation_controller_task.schema.json"
)
AGGREGATION_CONTROLLER_RESULT_SCHEMA_REL = Path(
    "docs/contracts/aggregation_controller_result.schema.json"
)
DATA_OUTPUT_PROVENANCE_SCHEMA_REL = Path(
    "docs/contracts/data_output_provenance.schema.json"
)
NODE_TASK_SCHEMA_REL = Path("docs/contracts/node_task.schema.json")
NODE_RESULT_SCHEMA_REL = Path("docs/contracts/node_result.schema.json")
PROCESS_ADAPTER_DESCRIPTION_SCHEMA_REL = Path(
    "docs/contracts/process_adapter_description.schema.json"
)
PROCESS_ADAPTER_FRAME_SCHEMA_REL = Path("docs/contracts/process_adapter_frame.schema.json")
RESEARCH_PROVENANCE_PROFILE_REL = Path(
    "docs/contracts/research_provenance_package_profile.v1.json"
)
LOCAL_FIXTURE_REL = Path("examples/fixtures/data/coordinator_data_plan_envelope_nir.json")
LOCAL_MULTISOURCE_FIXTURE_REL = Path(
    "examples/fixtures/data/coordinator_data_plan_envelope_multisource_repetitions.json"
)
LOCAL_FEATURE_FUSION_FIXTURE_REL = Path(
    "examples/fixtures/data/feature_fusion_selector_nir_chem.json"
)
SHARED_FOLD_SET_FIXTURE_REL = Path("examples/fixtures/shared/fold_set_cv_partition.json")
LOCAL_GRAPH_SPEC_FIXTURE_REL = Path("examples/branch_merge_oof_graph.json")
LOCAL_PIPELINE_DSL_FIXTURE_REL = Path("examples/pipeline_dsl_nirs4all_compat.json")
LOCAL_CAMPAIGN_SPEC_FIXTURE_REL = Path("examples/campaign_oof_generation.json")
LOCAL_EXECUTION_PLAN_FIXTURE_REL = Path(
    "examples/fixtures/runtime/execution_plan_branch_merge_executable.json"
)
LOCAL_MODEL_INPUT_SPEC_FIXTURE_REL = Path(
    "examples/fixtures/data/model_input_spec_tabular_regressor.json"
)
LOCAL_DATA_PLAN_FIXTURE_REL = Path("examples/fixtures/data/data_plan_tabular_fusion.json")
LOCAL_CONTROLLER_MANIFEST_FIXTURE_REL = Path(
    "examples/fixtures/runtime/controller_manifest_data_aware_model.json"
)
LOCAL_CONTROLLER_MANIFEST_LIST_FIXTURE_REL = Path("examples/controller_manifests.json")
LOCAL_SELECTION_POLICY_FIXTURE_REL = Path("examples/fixtures/bundle/selection_policy_rmse.json")
LOCAL_SELECTION_DECISION_FIXTURE_REL = Path(
    "examples/fixtures/bundle/selection_decision_branch_b0.json"
)
LOCAL_DATA_OUTPUT_PROVENANCE_FIXTURE_REL = Path(
    "examples/fixtures/runtime/data_output_provenance_augmented_view.json"
)
LOCAL_NODE_TASK_FIXTURE_REL = Path("examples/fixtures/runtime/node_task_transform_scale.json")
LOCAL_NODE_RESULT_FIXTURE_REL = Path("examples/fixtures/runtime/node_result_transform_scale.json")
LOCAL_PROCESS_ADAPTER_DESCRIPTION_FIXTURE_REL = Path(
    "examples/fixtures/runtime/process_adapter_description_python.json"
)
LOCAL_PROCESS_ADAPTER_FRAME_FIXTURE_RELS = [
    Path("examples/fixtures/runtime/process_adapter_frame_init.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_task_transform_scale.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_result_transform_scale.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_ack_initialized.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_error_retryable_timeout.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_close.json"),
]
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
LOCAL_BRANCH_VIEW_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "coordinator_branch_view.v1.schema.json"
)
LOCAL_FITTED_ADAPTER_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "fitted_adapter_ref.v1.schema.json"
)
GRAPH_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "graph_spec.v1.schema.json"
)
PIPELINE_DSL_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "pipeline_dsl.v1.schema.json"
)
CAMPAIGN_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "campaign_spec.v1.schema.json"
)
EXECUTION_PLAN_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "execution_plan.v1.schema.json"
)
MODEL_INPUT_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "model_input_spec.v1.schema.json"
)
DATA_PLAN_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "data_plan.v1.schema.json"
)
CONTROLLER_MANIFEST_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "controller_manifest.v1.schema.json"
)
SELECTION_POLICY_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "selection_policy.v1.schema.json"
)
SELECTION_DECISION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "selection_decision.v1.schema.json"
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
ENTITY_UNIT_LEVELS = {"physical_sample", "source_sample", "observation", "combo"}
EVALUATION_SCOPES = {"oof", "holdout", "final", "train", "refit"}
PREDICTION_LEVELS = {"observation", "sample", "target", "group"}
SPLIT_UNITS = {"physical_sample", "observation", "sample", "target", "group"}
FIT_INFLUENCE_POLICIES = {
    "auto",
    "uniform_rows",
    "equal_sample_influence",
    "resample_equalized",
    "backend_loss_weight",
    "scorer_only",
    "strict_weight_support",
}
COMBINATION_MODES = {
    "cartesian",
    "zip",
    "match_by",
    "sample_k",
    "reference_broadcast",
}
REPRESENTATION_MISSING_SOURCE_POLICIES = {
    "strict",
    "warn",
    "drop_incomplete",
    "impute_declared",
    "mask",
    "partial_model",
    "pad",
}
REPRESENTATION_CARDINALITIES = {
    "one_to_one",
    "one_to_many",
    "many_to_one",
    "many_to_many",
    "bounded_many",
}
REPRESENTATION_KINDS = {
    "aggregate",
    "cartesian_product",
    "monte_carlo_cartesian",
    "stack_fixed",
    "stack_padded_masked",
}
FIT_INFLUENCE_MECHANISMS = {
    "uniform_rows",
    "sample_weights",
    "row_resampling",
    "backend_loss_weights",
    "scorer_only",
}
CONTROLLER_CAPABILITIES = {
    "deterministic",
    "thread_safe",
    "process_safe",
    "needs_python_gil",
    "emits_predictions",
    "consumes_oof_predictions",
    "emits_artifacts",
    "stateful",
    "emits_relation",
    "uses_core_rng",
    "shape_changing",
    "generates_data",
    "generates_model",
    "expands_variants",
    "aggregates_predictions",
    "supports_sample_weights",
    "supports_row_resampling",
    "supports_backend_loss_weights",
    "supports_missing_masks",
}
MISSINGNESS_POLICIES = {
    "strict",
    "warn",
    "impute_declared",
    "mask",
    "partial_model",
    "pad_representation",
}
CONFORMANCE_PACK_ID = "dag-ml.shared.conformance.v1"
PARITY_ORACLE_ID = "dag-ml.nirs4all.parity_oracle.v1"
REQUIRED_PARITY_CASE_IDS = {
    "nirs4all_lite_browser_compile_plan",
    "repetition_group_leakage_refusal",
    "controller_registry_selector_parity",
    "branch_merge_oof_refit_replay",
    "python_wheel_facade_integration",
}
OPENLINEAGE_FACETS_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "openlineage_dagml_facets.v1.schema.json"
)
PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "prediction_cache_tensor_metadata.v1.schema.json"
)
PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "prediction_cache_columnar_tensor_metadata.v1.schema.json"
)
AGGREGATION_CONTROLLER_TASK_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "aggregation_controller_task.v1.schema.json"
)
AGGREGATION_CONTROLLER_RESULT_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "aggregation_controller_result.v1.schema.json"
)
DATA_OUTPUT_PROVENANCE_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "data_output_provenance.v1.schema.json"
)
NODE_TASK_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "node_task.v1.schema.json"
)
NODE_RESULT_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "node_result.v1.schema.json"
)
PROCESS_ADAPTER_DESCRIPTION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "process_adapter_description.v1.schema.json"
)
PROCESS_ADAPTER_FRAME_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "process_adapter_frame.v1.schema.json"
)
RESEARCH_PROVENANCE_PROFILE_ID = "dag-ml.research_provenance_package.v1"
SHARED_FOLD_SET_FINGERPRINT = (
    "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c"
)


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


def require_optional_non_empty_string(value: Any, label: str) -> None:
    if value is not None:
        require_non_empty_string(value, label)


def require_optional_identifier(value: Any, label: str) -> None:
    if value is not None:
        require_identifier(value, label)


def require_optional_unit_level(value: Any, label: str) -> None:
    if value is not None:
        require(value in ENTITY_UNIT_LEVELS, f"{label} has invalid entity unit level")


def require_no_unknown_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = set(value) - allowed
    require(not extra, f"{label} contains unknown field(s): {sorted(extra)}")


def require_positive_int(value: Any, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )


def require_non_negative_int(value: Any, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )


def validate_optional_string_array(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    require(isinstance(value, list), f"{label} must be an array")
    for index, item in enumerate(value):
        require_non_empty_string(item, f"{label}[{index}]")
    require(len(set(value)) == len(value), f"{label} contains duplicates")
    return value


def validate_metadata_object(value: Any, label: str) -> None:
    if value is None:
        return
    require(isinstance(value, dict), f"{label} must be an object")
    for key in value:
        require_non_empty_string(key, f"{label} key")


def validate_combination_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    mode = value.get("mode")
    require(mode in COMBINATION_MODES, f"{label}.mode is invalid")
    component_source_ids = validate_optional_string_array(
        value.get("component_source_ids"),
        f"{label}.component_source_ids",
    )
    validate_optional_string_array(
        value.get("component_unit_ids"),
        f"{label}.component_unit_ids",
    )
    require_optional_non_empty_string(value.get("match_key"), f"{label}.match_key")
    require_optional_non_empty_string(
        value.get("reference_source_id"),
        f"{label}.reference_source_id",
    )
    seed = value.get("seed")
    if seed is not None:
        require_non_negative_int(seed, f"{label}.seed")
    cap = value.get("cap")
    if cap is not None:
        require_positive_int(cap, f"{label}.cap")
    budget = value.get("budget")
    if budget is not None:
        require_positive_int(budget, f"{label}.budget")
    missing_source_policy = value.get("missing_source_policy")
    if missing_source_policy is not None:
        require(
            missing_source_policy in REPRESENTATION_MISSING_SOURCE_POLICIES,
            f"{label}.missing_source_policy is invalid",
        )
    validate_metadata_object(value.get("metadata"), f"{label}.metadata")

    if mode in {"cartesian", "zip"}:
        require(
            len(component_source_ids) >= 2,
            f"{label}.{mode} requires at least two component_source_ids",
        )
    elif mode == "match_by":
        require(value.get("match_key") is not None, f"{label}.match_key is required")
    elif mode == "sample_k":
        require(seed is not None, f"{label}.seed is required")
        require(cap is not None, f"{label}.cap is required")
    elif mode == "reference_broadcast":
        reference = value.get("reference_source_id")
        require(reference is not None, f"{label}.reference_source_id is required")
        require(
            not component_source_ids or reference in component_source_ids,
            f"{label}.reference_source_id must be listed in component_source_ids",
        )


def require_unit_level(value: Any, label: str) -> None:
    require(value in ENTITY_UNIT_LEVELS, f"{label} has invalid entity unit level")


def require_combo_like_output(value: Any, label: str) -> None:
    require(
        value in {"combo", "observation"},
        f"{label} output_unit_level must be combo or observation",
    )


def representation_output_unit_level(value: dict[str, Any]) -> Any:
    return value.get("output_unit_level")


def validate_representation_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    kind = value.get("kind")
    require(kind in REPRESENTATION_KINDS, f"{label}.kind is invalid")

    if kind == "aggregate":
        require_unit_level(value.get("input_unit_level"), f"{label}.input_unit_level")
        require_unit_level(value.get("output_unit_level"), f"{label}.output_unit_level")
        require_optional_non_empty_string(value.get("reducer_id"), f"{label}.reducer_id")
        require_optional_non_empty_string(value.get("method"), f"{label}.method")
        require(
            value.get("cardinality") == "many_to_one",
            f"{label}.cardinality must be many_to_one",
        )
        return

    if kind in {"cartesian_product", "monte_carlo_cartesian"}:
        combination_plan = value.get("combination_plan")
        validate_combination_plan(combination_plan, f"{label}.combination_plan")
        expected_mode = "cartesian" if kind == "cartesian_product" else "sample_k"
        require(
            combination_plan.get("mode") == expected_mode,
            f"{label}.combination_plan.mode must be {expected_mode}",
        )
        require_unit_level(value.get("output_unit_level"), f"{label}.output_unit_level")
        require_combo_like_output(value.get("output_unit_level"), label)
        expected_cardinality = (
            "many_to_many" if kind == "cartesian_product" else "bounded_many"
        )
        require(
            value.get("cardinality") == expected_cardinality,
            f"{label}.cardinality must be {expected_cardinality}",
        )
        preserve_provenance = value.get("preserve_provenance")
        if preserve_provenance is not None:
            require(isinstance(preserve_provenance, bool), f"{label}.preserve_provenance bool")
        return

    if kind == "stack_fixed":
        require_unit_level(value.get("output_unit_level"), f"{label}.output_unit_level")
        require(
            value.get("cardinality") == "one_to_many",
            f"{label}.cardinality must be one_to_many",
        )
        require_positive_int(
            value.get("expected_cardinality"),
            f"{label}.expected_cardinality",
        )
        validate_optional_string_array(
            value.get("component_source_ids"),
            f"{label}.component_source_ids",
        )
        return

    require(kind == "stack_padded_masked", f"{label}.kind is invalid")
    require_unit_level(value.get("output_unit_level"), f"{label}.output_unit_level")
    require(
        value.get("cardinality") == "bounded_many",
        f"{label}.cardinality must be bounded_many",
    )
    require_positive_int(
        value.get("expected_cardinality"),
        f"{label}.expected_cardinality",
    )
    require(
        value.get("missing_source_policy") in {"mask", "pad"},
        f"{label}.missing_source_policy must be mask or pad",
    )
    requires_missing_masks = value.get("requires_missing_masks", True)
    require(
        requires_missing_masks is True,
        f"{label}.requires_missing_masks must be true",
    )
    validate_optional_string_array(
        value.get("component_source_ids"),
        f"{label}.component_source_ids",
    )


def validate_representation_replay_manifest(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require_identifier(value.get("manifest_id"), f"{label}.manifest_id")
    representation_plan = value.get("representation_plan")
    validate_representation_plan(representation_plan, f"{label}.representation_plan")
    combination_plan = value.get("combination_plan")
    if combination_plan is not None:
        validate_combination_plan(combination_plan, f"{label}.combination_plan")
    output_unit_level = value.get("output_unit_level")
    require_unit_level(output_unit_level, f"{label}.output_unit_level")
    require(
        output_unit_level == representation_output_unit_level(representation_plan),
        f"{label}.output_unit_level must match representation_plan",
    )
    require_optional_non_empty_string(
        value.get("output_representation"),
        f"{label}.output_representation",
    )
    require_optional_non_empty_string(
        value.get("final_reduction_id"),
        f"{label}.final_reduction_id",
    )
    sample_observation_mapping = value.get("sample_observation_mapping", [])
    require(
        isinstance(sample_observation_mapping, list),
        f"{label}.sample_observation_mapping must be an array",
    )
    sample_source_pairs: set[tuple[str, str]] = set()
    for index, mapping in enumerate(sample_observation_mapping):
        mapping_label = f"{label}.sample_observation_mapping[{index}]"
        require(isinstance(mapping, dict), f"{mapping_label} must be an object")
        require_non_empty_string(
            mapping.get("physical_sample_id"),
            f"{mapping_label}.physical_sample_id",
        )
        require_non_empty_string(mapping.get("source_id"), f"{mapping_label}.source_id")
        observation_ids = mapping.get("observation_ids")
        require(
            isinstance(observation_ids, list) and observation_ids,
            f"{mapping_label}.observation_ids must be a non-empty array",
        )
        for obs_index, observation_id in enumerate(observation_ids):
            require_non_empty_string(
                observation_id,
                f"{mapping_label}.observation_ids[{obs_index}]",
            )
        require(
            len(set(observation_ids)) == len(observation_ids),
            f"{mapping_label}.observation_ids contains duplicates",
        )
        pair = (mapping["physical_sample_id"], mapping["source_id"])
        require(
            pair not in sample_source_pairs,
            f"{mapping_label} duplicates physical_sample_id/source_id",
        )
        sample_source_pairs.add(pair)
    combo_selection = value.get("combo_selection", [])
    require(isinstance(combo_selection, list), f"{label}.combo_selection must be an array")
    combo_unit_ids: set[str] = set()
    for index, record in enumerate(combo_selection):
        record_label = f"{label}.combo_selection[{index}]"
        require(isinstance(record, dict), f"{record_label} must be an object")
        require_non_empty_string(record.get("combo_unit_id"), f"{record_label}.combo_unit_id")
        require_non_empty_string(
            record.get("physical_sample_id"),
            f"{record_label}.physical_sample_id",
        )
        component_observation_ids = record.get("component_observation_ids")
        require(
            isinstance(component_observation_ids, list) and component_observation_ids,
            f"{record_label}.component_observation_ids must be a non-empty array",
        )
        for component_index, observation_id in enumerate(component_observation_ids):
            require_non_empty_string(
                observation_id,
                f"{record_label}.component_observation_ids[{component_index}]",
            )
        require(
            len(set(component_observation_ids)) == len(component_observation_ids),
            f"{record_label}.component_observation_ids contains duplicates",
        )
        seed = record.get("seed")
        if seed is not None:
            require_non_negative_int(seed, f"{record_label}.seed")
        require(
            record["combo_unit_id"] not in combo_unit_ids,
            f"{record_label}.combo_unit_id duplicates another combo",
        )
        combo_unit_ids.add(record["combo_unit_id"])
    for field in ("qc_policy_refs", "outlier_policy_refs"):
        validate_optional_string_array(value.get(field), f"{label}.{field}")
    for field in ("missing_source_policy", "missing_repetition_policy"):
        policy = value.get(field)
        if policy is not None:
            require(
                policy in REPRESENTATION_MISSING_SOURCE_POLICIES,
                f"{label}.{field} is invalid",
            )
    require_optional_non_empty_string(
        value.get("prediction_representation"),
        f"{label}.prediction_representation",
    )
    final_output_unit_level = value.get("final_output_unit_level")
    if final_output_unit_level is not None:
        require_unit_level(final_output_unit_level, f"{label}.final_output_unit_level")
    for field in ("relation_fingerprint", "feature_schema_fingerprint"):
        field_value = value.get(field)
        if field_value is not None:
            require_sha256(field_value, f"{label}.{field}")
    validate_metadata_object(value.get("metadata"), f"{label}.metadata")


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
    for field in ("combination_plan", "representation_plan"):
        require(
            field in properties,
            f"{label} feature-fusion schema must declare optional `{field}`",
        )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} feature-fusion $defs are missing")
    for name in (
        "source",
        "alignment",
        "presence_mask",
        "combination_plan",
        "representation_plan",
        "combination_mode",
        "representation_missing_source_policy",
        "representation_cardinality",
    ):
        require(name in defs, f"{label} feature-fusion schema misses `{name}` definition")
    require(
        set(defs.get("combination_mode", {}).get("enum", [])) == COMBINATION_MODES,
        f"{label} feature-fusion combination modes are not aligned",
    )
    require(
        set(defs.get("representation_missing_source_policy", {}).get("enum", []))
        == REPRESENTATION_MISSING_SOURCE_POLICIES,
        f"{label} feature-fusion missing-source policies are not aligned",
    )
    require(
        set(defs.get("representation_cardinality", {}).get("enum", []))
        == REPRESENTATION_CARDINALITIES,
        f"{label} feature-fusion representation cardinalities are not aligned",
    )


def validate_branch_view_schema_artifact(schema: Any, expected_id: str, label: str) -> None:
    require(isinstance(schema, dict), f"{label} branch-view schema must be a JSON object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} branch-view schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == expected_id,
        f"{label} branch-view schema has unexpected $id",
    )
    require(schema.get("type") == "object", f"{label} branch-view root must be an object")
    required = schema.get("required")
    require(isinstance(required, list), f"{label} branch-view required list is missing")
    for field in ("view_id", "branch_id", "mode", "selector"):
        require(field in required, f"{label} branch-view schema does not require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} branch-view $defs are missing")
    for name in ("branch_view_mode", "branch_view_selector"):
        require(name in defs, f"{label} branch-view schema misses `{name}` definition")
    modes = defs.get("branch_view_mode", {}).get("enum")
    require(isinstance(modes, list), f"{label} branch-view mode enum is missing")
    for expected in ("separation", "by_source", "by_metadata", "by_tag", "by_filter"):
        require(
            expected in modes,
            f"{label} branch-view mode enum must include `{expected}`",
        )


def validate_fitted_adapter_ref_schema_artifact(
    schema: Any, expected_id: str, label: str
) -> None:
    require(isinstance(schema, dict), f"{label} fitted-adapter schema must be a JSON object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} fitted-adapter schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == expected_id,
        f"{label} fitted-adapter schema has unexpected $id",
    )
    require(schema.get("type") == "object", f"{label} fitted-adapter root must be an object")
    required = schema.get("required")
    require(isinstance(required, list), f"{label} fitted-adapter required list is missing")
    for field in ("adapter_id", "adapter_version", "params_fingerprint"):
        require(field in required, f"{label} fitted-adapter schema does not require `{field}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} fitted-adapter properties are missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} fitted-adapter schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} fitted-adapter $defs are missing")
    for name in ("non_empty_id", "hex_fingerprint", "backend"):
        require(name in defs, f"{label} fitted-adapter schema misses `{name}` definition")
    backends = defs.get("backend", {}).get("enum")
    require(isinstance(backends, list), f"{label} fitted-adapter backend enum is missing")
    for expected in ("joblib", "pickle", "json", "numpy", "onnx", "raw"):
        require(
            expected in backends,
            f"{label} fitted-adapter backend enum must include `{expected}`",
        )


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
    require(
        defs.get("entity_unit_level", {}).get("enum")
        == ["physical_sample", "source_sample", "observation", "combo"],
        f"{label} GraphSpec entity_unit_level enum is not aligned with Rust",
    )
    require(
        defs.get("missingness_policy", {}).get("enum")
        == [
            "strict",
            "warn",
            "impute_declared",
            "mask",
            "partial_model",
            "pad_representation",
        ],
        f"{label} GraphSpec missingness_policy enum is not aligned with Rust",
    )
    for definition_name in (
        "port_spec",
        "port_schema",
        "port_ref",
        "edge_contract",
        "edge_spec",
        "graph_interface",
        "node_spec",
        "relation_contract",
    ):
        require(
            definition_name in defs,
            f"{label} GraphSpec schema misses `{definition_name}`",
        )
    relation_contract = defs.get("relation_contract")
    require(
        isinstance(relation_contract, dict)
        and relation_contract.get("additionalProperties") is False,
        f"{label} GraphSpec relation_contract must reject unknown fields",
    )
    relation_props = relation_contract.get("properties", {})
    require(
        isinstance(relation_props, dict)
        and "relation_fingerprint" in relation_props
        and "required" in relation_props,
        f"{label} GraphSpec relation_contract properties are incomplete",
    )
    port_props = defs.get("port_spec", {}).get("properties")
    require(isinstance(port_props, dict), f"{label} GraphSpec port_spec properties missing")
    for property_name in ("unit_level", "alignment_key", "target_level"):
        require(
            property_name in port_props,
            f"{label} GraphSpec port_spec misses `{property_name}`",
        )
    edge_props = defs.get("edge_contract", {}).get("properties")
    require(isinstance(edge_props, dict), f"{label} GraphSpec edge_contract properties missing")
    for property_name in (
        "unit_level",
        "alignment_key",
        "target_level",
        "relation_contract",
        "allows_broadcast",
        "missingness_policy",
    ):
        require(
            property_name in edge_props,
            f"{label} GraphSpec edge_contract misses `{property_name}`",
        )


def validate_pipeline_dsl_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} Pipeline DSL schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} Pipeline DSL schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == PIPELINE_DSL_SCHEMA_ID, f"{label} Pipeline DSL $id mismatch")
    require(isinstance(schema.get("oneOf"), list), f"{label} Pipeline DSL root must use oneOf")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} Pipeline DSL $defs missing")
    expected_step_kinds = [
        "transform",
        "y_transform",
        "tag",
        "exclude",
        "filter",
        "sample_filter",
        "augmentation",
        "feature_augmentation",
        "sample_augmentation",
        "data_generation",
        "generation",
        "concat_transform",
        "model",
        "tuner",
        "finetune",
        "branch",
        "generator",
        "sequential",
        "merge",
        "merge_model",
        "chart",
    ]
    require(
        defs.get("canonical_step_kind", {}).get("enum") == expected_step_kinds,
        f"{label} Pipeline DSL canonical step enum is not aligned",
    )
    expected_generator_keys = [
        "_or_",
        "_cartesian_",
        "_chain_",
        "_grid_",
        "_range_",
        "_log_range_",
        "_zip_",
        "_sample_",
    ]
    require(
        defs.get("compat_generator_key", {}).get("enum") == expected_generator_keys,
        f"{label} Pipeline DSL compat generator enum is not aligned",
    )
    for definition_name in (
        "canonical_pipeline_dsl",
        "canonical_step",
        "canonical_branch",
        "canonical_stage",
        "compat_pipeline_object",
        "compat_step",
        "compat_step_array",
        "compat_step_object",
        "pipeline_unit_contract",
    ):
        require(definition_name in defs, f"{label} Pipeline DSL schema misses `{definition_name}`")
    require(
        defs.get("entity_unit_level", {}).get("enum")
        == ["physical_sample", "source_sample", "observation", "combo"],
        f"{label} Pipeline DSL entity_unit_level enum is not aligned",
    )
    unit_contract_props = defs.get("pipeline_unit_contract", {}).get("properties")
    require(
        isinstance(unit_contract_props, dict),
        f"{label} Pipeline DSL unit contract properties missing",
    )
    for property_name in ("name", "representation", "unit_level", "alignment_key", "target_level"):
        require(
            property_name in unit_contract_props,
            f"{label} Pipeline DSL unit contract misses `{property_name}`",
        )
    for pipeline_def in ("canonical_pipeline_dsl", "compat_pipeline_object"):
        pipeline_props = defs.get(pipeline_def, {}).get("properties")
        require(
            isinstance(pipeline_props, dict),
            f"{label} Pipeline DSL {pipeline_def} properties missing",
        )
        for property_name in ("input", "output"):
            require(
                pipeline_props.get(property_name, {}).get("$ref")
                == "#/$defs/pipeline_unit_contract",
                f"{label} Pipeline DSL {pipeline_def}.{property_name} must use pipeline_unit_contract",
            )
    compat_properties = defs.get("compat_step_object", {}).get("properties")
    require(isinstance(compat_properties, dict), f"{label} Pipeline DSL compat properties missing")
    for property_name in ("class", "function", "ref", "type", "name", "step", "tuner", "finetune"):
        require(
            property_name in compat_properties,
            f"{label} Pipeline DSL compat schema misses `{property_name}` alias property",
        )


def validate_campaign_spec_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} CampaignSpec schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} CampaignSpec schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == CAMPAIGN_SPEC_SCHEMA_ID, f"{label} CampaignSpec $id mismatch")
    require(schema.get("type") == "object", f"{label} CampaignSpec root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} CampaignSpec root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} CampaignSpec required list missing")
    require("id" in required, f"{label} CampaignSpec schema must require `id`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} CampaignSpec properties missing")
    require(
        "branch_view_plans" in properties,
        f"{label} CampaignSpec schema must declare branch_view_plans",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} CampaignSpec $defs missing")
    require(
        defs.get("split_unit", {}).get("enum")
        == ["physical_sample", "observation", "sample", "target", "group"],
        f"{label} CampaignSpec split_unit enum is not aligned",
    )
    require(
        defs.get("prediction_level", {}).get("enum")
        == ["observation", "sample", "target", "group"],
        f"{label} CampaignSpec prediction_level enum is not aligned",
    )
    require(
        defs.get("aggregation_method", {}).get("enum")
        == ["none", "mean", "weighted_mean", "median", "vote", "custom_controller"],
        f"{label} CampaignSpec aggregation_method enum is not aligned",
    )
    require(
        defs.get("aggregation_weights", {}).get("enum")
        == ["none", "quality", "repetition_count", "controller_emitted"],
        f"{label} CampaignSpec aggregation_weights enum is not aligned",
    )
    require(
        defs.get("generation_strategy", {}).get("enum") == ["none", "cartesian", "zip"],
        f"{label} CampaignSpec generation_strategy enum is not aligned",
    )
    require(
        defs.get("branch_view_mode", {}).get("enum")
        == ["separation", "by_source", "by_metadata", "by_tag", "by_filter"],
        f"{label} CampaignSpec branch_view_mode enum is not aligned",
    )
    for definition_name in (
        "leakage_policy",
        "aggregation_policy",
        "aggregation_controller_spec",
        "fold_set",
        "split_invocation",
        "generation_spec",
        "data_model_shape_plan",
        "data_view_policy",
        "data_binding",
        "data_view_selector",
        "branch_view_plan",
    ):
        require(definition_name in defs, f"{label} CampaignSpec schema misses `{definition_name}`")
    branch_view_properties = defs.get("branch_view_plan", {}).get("properties")
    require(
        isinstance(branch_view_properties, dict) and "selector" in branch_view_properties,
        f"{label} CampaignSpec branch_view_plan must declare selector",
    )
    # Lock the view_id/branch_id identifier pattern to the strict pattern that
    # dag-ml-data's `coordinator_branch_view.schema.json` uses. Without this
    # check, dag-ml could accept identifiers with spaces or other characters
    # that dag-ml-data would reject on the same payload.
    for field in ("view_id", "branch_id"):
        ref = branch_view_properties.get(field, {}).get("$ref")
        require(
            ref == "#/$defs/identifier",
            f"{label} CampaignSpec branch_view_plan.{field} must $ref identifier pattern, got `{ref}`",
        )


def validate_execution_plan_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} ExecutionPlan schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} ExecutionPlan schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == EXECUTION_PLAN_SCHEMA_ID,
        f"{label} ExecutionPlan schema $id mismatch",
    )
    require(schema.get("type") == "object", f"{label} ExecutionPlan root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} ExecutionPlan root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} ExecutionPlan required list missing")
    for field in (
        "id",
        "graph_plan",
        "campaign",
        "node_plans",
        "controller_manifests",
        "variants",
        "graph_fingerprint",
        "campaign_fingerprint",
        "controller_fingerprint",
    ):
        require(field in required, f"{label} ExecutionPlan schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} ExecutionPlan $defs missing")
    require(
        defs.get("phase", {}).get("enum")
        == ["COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"],
        f"{label} ExecutionPlan phase enum is not aligned",
    )
    require(
        set(defs.get("controller_capability", {}).get("enum", [])) == CONTROLLER_CAPABILITIES,
        f"{label} ExecutionPlan capability enum is not aligned",
    )
    require(
        defs.get("node_kind", {}).get("enum")
        == [
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
        ],
        f"{label} ExecutionPlan node_kind enum is not aligned",
    )
    for definition_name in (
        "graph_plan",
        "node_plan",
        "variant_plan",
        "generation_choice",
        "generation_param_override",
        "fold_set",
    ):
        require(definition_name in defs, f"{label} ExecutionPlan schema misses `{definition_name}`")


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
    for definition_name in (
        "combination_plan",
        "representation_plan",
        "combination_mode",
        "representation_missing_source_policy",
        "representation_cardinality",
    ):
        require(
            definition_name in defs,
            f"{label} ModelInputSpec schema misses {definition_name}",
        )
    require(
        "fit_influence_policy" in properties,
        f"{label} ModelInputSpec schema misses fit_influence_policy property",
    )
    require(
        set(defs.get("fit_influence_policy", {}).get("enum", [])) == FIT_INFLUENCE_POLICIES,
        f"{label} ModelInputSpec fit influence enum is not aligned",
    )
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
    require(
        "representation_plan"
        in defs.get("fusion_policy", {}).get("properties", {}),
        f"{label} ModelInputSpec fusion policy misses representation_plan",
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


def validate_controller_manifest_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} ControllerManifest schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} ControllerManifest schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == CONTROLLER_MANIFEST_SCHEMA_ID,
        f"{label} ControllerManifest schema $id mismatch",
    )
    require(schema.get("type") == "object", f"{label} ControllerManifest root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} ControllerManifest root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} ControllerManifest required list missing")
    for field in (
        "controller_id",
        "controller_version",
        "operator_kind",
        "supported_phases",
        "fit_scope",
        "rng_policy",
        "artifact_policy",
    ):
        require(field in required, f"{label} ControllerManifest schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} ControllerManifest $defs missing")
    require(
        defs.get("node_kind", {}).get("enum")
        == [
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
        ],
        f"{label} ControllerManifest node_kind enum is not aligned",
    )
    require(
        set(defs.get("controller_capability", {}).get("enum", [])) == CONTROLLER_CAPABILITIES,
        f"{label} ControllerManifest capability enum is not aligned",
    )
    require(
        set(defs.get("fit_influence_policy", {}).get("enum", [])) == FIT_INFLUENCE_POLICIES,
        f"{label} ControllerManifest fit influence enum is not aligned",
    )
    require(
        defs.get("controller_fit_scope", {}).get("enum")
        == ["stateless", "fold_train", "full_train", "inference_only"],
        f"{label} ControllerManifest fit_scope enum is not aligned",
    )
    require(
        defs.get("rng_policy", {}).get("enum")
        == [
            "uses_core_seed",
            "ignores_seed",
            "externally_deterministic",
            "nondeterministic",
        ],
        f"{label} ControllerManifest rng_policy enum is not aligned",
    )
    require(
        defs.get("artifact_policy", {}).get("enum")
        == ["serializable", "host_only", "content_addressed", "replay_required"],
        f"{label} ControllerManifest artifact_policy enum is not aligned",
    )
    for definition_name in (
        "port_spec",
        "model_input_spec",
        "model_input_fusion_policy",
        "fit_influence_policy",
        "combination_plan",
        "representation_plan",
        "combination_mode",
        "representation_missing_source_policy",
        "representation_cardinality",
    ):
        require(
            definition_name in defs,
            f"{label} ControllerManifest schema misses `{definition_name}`",
        )
    require(
        "representation_plan"
        in defs.get("model_input_fusion_policy", {}).get("properties", {}),
        f"{label} ControllerManifest model input fusion policy misses representation_plan",
    )


def validate_selection_policy_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} SelectionPolicy schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} SelectionPolicy schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == SELECTION_POLICY_SCHEMA_ID,
        f"{label} SelectionPolicy $id mismatch",
    )
    require(schema.get("type") == "object", f"{label} SelectionPolicy root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} SelectionPolicy root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} SelectionPolicy required list missing")
    for field in ("id", "metric"):
        require(field in required, f"{label} SelectionPolicy schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} SelectionPolicy $defs missing")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} SelectionPolicy properties missing")
    for field in (
        "evaluation_scope",
        "refit_slot_plan",
        "stacking_fit_contract",
        "reduction_id",
    ):
        require(field in properties, f"{label} SelectionPolicy schema must declare `{field}`")
    require(
        defs.get("metric_objective", {}).get("enum") == ["minimize", "maximize"],
        f"{label} SelectionPolicy objective enum is not aligned",
    )
    require(
        defs.get("prediction_level", {}).get("enum")
        == ["observation", "sample", "target", "group"],
        f"{label} SelectionPolicy prediction level enum is not aligned",
    )
    require(
        defs.get("entity_unit_level", {}).get("enum")
        == ["physical_sample", "source_sample", "observation", "combo"],
        f"{label} SelectionPolicy entity unit enum is not aligned",
    )
    require(
        defs.get("evaluation_scope", {}).get("enum")
        == ["oof", "holdout", "final", "train", "refit"],
        f"{label} SelectionPolicy evaluation scope enum is not aligned",
    )
    require(
        defs.get("refit_strategy", {}).get("enum") == ["refit_one", "refit_ensemble"],
        f"{label} SelectionPolicy refit strategy enum is not aligned",
    )
    require(
        defs.get("meta_row_domain", {}).get("enum") == ["sample", "combo"],
        f"{label} SelectionPolicy meta row domain enum is not aligned",
    )
    require("selection_metric" in defs, f"{label} SelectionPolicy misses selection_metric")
    for definition_name in ("evaluation_result", "refit_slot_plan", "stacking_fit_contract"):
        require(
            definition_name in defs,
            f"{label} SelectionPolicy misses {definition_name}",
        )


def validate_selection_decision_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} SelectionDecision schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} SelectionDecision schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == SELECTION_DECISION_SCHEMA_ID,
        f"{label} SelectionDecision $id mismatch",
    )
    require(schema.get("type") == "object", f"{label} SelectionDecision root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} SelectionDecision root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} SelectionDecision required list missing")
    for field in (
        "policy_id",
        "selected_candidate_id",
        "metric_name",
        "objective",
        "selected_score",
        "ranked_candidates",
    ):
        require(field in required, f"{label} SelectionDecision schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} SelectionDecision $defs missing")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} SelectionDecision properties missing")
    for field in ("evaluation_scope", "refit_slot_plan", "reduction_id"):
        require(field in properties, f"{label} SelectionDecision schema must declare `{field}`")
    require(
        defs.get("metric_objective", {}).get("enum") == ["minimize", "maximize"],
        f"{label} SelectionDecision objective enum is not aligned",
    )
    require(
        defs.get("prediction_level", {}).get("enum")
        == ["observation", "sample", "target", "group"],
        f"{label} SelectionDecision prediction level enum is not aligned",
    )
    require(
        defs.get("evaluation_scope", {}).get("enum")
        == ["oof", "holdout", "final", "train", "refit"],
        f"{label} SelectionDecision evaluation scope enum is not aligned",
    )
    require(
        defs.get("refit_strategy", {}).get("enum") == ["refit_one", "refit_ensemble"],
        f"{label} SelectionDecision refit strategy enum is not aligned",
    )
    require("ranked_candidate" in defs, f"{label} SelectionDecision misses ranked_candidate")
    require("refit_slot_plan" in defs, f"{label} SelectionDecision misses refit_slot_plan")


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
    for field in ("relation_fingerprint", "evaluation_scope", "reduction_id"):
        require(
            field in properties,
            f"{label} prediction-cache tensor metadata must declare optional `{field}`",
        )
    defs = schema.get("$defs")
    require(
        isinstance(defs, dict)
        and "block_metadata" in defs
        and "prediction_unit_id" in defs
        and "sha256" in defs
        and "evaluation_scope" in defs,
        f"{label} prediction-cache tensor metadata schema definitions are incomplete",
    )
    require(
        defs.get("sha256", {}).get("pattern") == "^[0-9A-Fa-f]{64}$",
        f"{label} prediction-cache tensor metadata sha256 definition mismatch",
    )
    require(
        defs.get("evaluation_scope", {}).get("enum")
        == ["oof", "holdout", "final", "train", "refit"],
        f"{label} prediction-cache tensor metadata evaluation scope enum mismatch",
    )
    block_properties = defs.get("block_metadata", {}).get("properties")
    require(
        isinstance(block_properties, dict),
        f"{label} prediction-cache tensor metadata block properties missing",
    )
    for field in ("relation_fingerprint", "evaluation_scope", "reduction_id"):
        require(
            field in block_properties,
            f"{label} prediction-cache tensor block metadata must declare optional `{field}`",
        )


def validate_prediction_cache_columnar_tensor_metadata_schema(
    schema: Any, label: str
) -> None:
    require(
        isinstance(schema, dict),
        f"{label} prediction-cache columnar tensor metadata schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} prediction-cache columnar tensor metadata schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_ID,
        f"{label} prediction-cache columnar tensor metadata schema has unexpected $id",
    )
    require(
        schema.get("type") == "object",
        f"{label} prediction-cache columnar tensor metadata root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} prediction-cache columnar tensor metadata root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} prediction-cache columnar tensor metadata required list is missing",
    )
    for field in (
        "schema_version",
        "requirement_key",
        "cache_id",
        "prediction_level",
        "rows",
        "cols",
        "layout",
        "column_offsets",
        "blocks",
    ):
        require(
            field in required,
            f"{label} prediction-cache columnar tensor metadata must require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict),
        f"{label} prediction-cache columnar tensor metadata properties are missing",
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} prediction-cache columnar tensor metadata schema_version const must be 1",
    )
    require(
        properties.get("layout", {}).get("const") == "column_major_f64",
        f"{label} prediction-cache columnar tensor metadata layout const mismatch",
    )
    require(
        properties.get("prediction_level", {}).get("enum") == ["sample", "target", "group"],
        f"{label} prediction-cache columnar tensor metadata prediction_level enum mismatch",
    )
    for field in ("relation_fingerprint", "evaluation_scope", "reduction_id"):
        require(
            field in properties,
            f"{label} prediction-cache columnar tensor metadata must declare optional `{field}`",
        )
    defs = schema.get("$defs")
    require(
        isinstance(defs, dict)
        and "block_metadata" in defs
        and "prediction_unit_id" in defs
        and "sha256" in defs
        and "evaluation_scope" in defs,
        f"{label} prediction-cache columnar tensor metadata schema definitions are incomplete",
    )
    require(
        defs.get("sha256", {}).get("pattern") == "^[0-9A-Fa-f]{64}$",
        f"{label} prediction-cache columnar tensor metadata sha256 definition mismatch",
    )
    require(
        defs.get("evaluation_scope", {}).get("enum")
        == ["oof", "holdout", "final", "train", "refit"],
        f"{label} prediction-cache columnar tensor metadata evaluation scope enum mismatch",
    )
    block_properties = defs.get("block_metadata", {}).get("properties")
    require(
        isinstance(block_properties, dict),
        f"{label} prediction-cache columnar tensor metadata block properties missing",
    )
    for field in ("relation_fingerprint", "evaluation_scope", "reduction_id"):
        require(
            field in block_properties,
            f"{label} prediction-cache columnar tensor block metadata must declare optional `{field}`",
        )


def validate_aggregation_controller_task_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} aggregation-controller task schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} aggregation-controller task schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == AGGREGATION_CONTROLLER_TASK_SCHEMA_ID,
        f"{label} aggregation-controller task schema has unexpected $id",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} aggregation-controller task schema root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} aggregation-controller task schema required list is missing",
    )
    for field in ("schema_version", "task_id", "controller_id", "policy", "input"):
        require(
            field in required,
            f"{label} aggregation-controller task schema must require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict),
        f"{label} aggregation-controller task schema properties are missing",
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} aggregation-controller task schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} aggregation-controller task $defs missing")
    for definition_name in (
        "aggregation_policy",
        "aggregation_controller_spec",
        "reduction_plan",
        "reduction_role",
        "reduction_axis",
        "reduction_method",
        "reduction_task_compatibility",
        "entity_unit_level",
        "observation_to_sample_input",
        "sample_to_unit_input",
        "observation_prediction_block",
        "prediction_block",
        "sample_relation_set",
        "prediction_unit_id",
    ):
        require(
            definition_name in defs,
            f"{label} aggregation-controller task schema misses `{definition_name}`",
        )
    require(
        defs["aggregation_policy"]["properties"]["method"].get("const") == "custom_controller",
        f"{label} aggregation-controller task policy must pin custom_controller method",
    )
    require(
        defs.get("aggregation_method", {}).get("enum")
        == [
            "none",
            "mean",
            "weighted_mean",
            "median",
            "vote",
            "robust_mean",
            "exclude_outliers",
            "custom_controller",
        ],
        f"{label} aggregation-controller task aggregation methods are not aligned",
    )
    require(
        defs.get("reduction_method", {}).get("enum")
        == [
            "mean",
            "weighted_mean",
            "median",
            "vote",
            "robust_mean",
            "exclude_outliers",
            "custom",
        ],
        f"{label} aggregation-controller task reduction methods are not aligned",
    )
    require(
        defs.get("reduction_role", {}).get("enum")
        == ["score", "persist", "fold_ensemble", "meta_feature", "final_output"],
        f"{label} aggregation-controller task reduction roles are not aligned",
    )
    require(
        defs.get("reduction_axis", {}).get("enum") == ["unit", "fold", "model", "metric"],
        f"{label} aggregation-controller task reduction axes are not aligned",
    )
    require(
        defs.get("entity_unit_level", {}).get("enum")
        == ["physical_sample", "source_sample", "observation", "combo"],
        f"{label} aggregation-controller task entity unit levels are not aligned",
    )


def validate_aggregation_controller_result_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} aggregation-controller result schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} aggregation-controller result schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == AGGREGATION_CONTROLLER_RESULT_SCHEMA_ID,
        f"{label} aggregation-controller result schema has unexpected $id",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} aggregation-controller result schema root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} aggregation-controller result schema required list is missing",
    )
    for field in ("schema_version", "task_id", "output"):
        require(
            field in required,
            f"{label} aggregation-controller result schema must require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict),
        f"{label} aggregation-controller result schema properties are missing",
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} aggregation-controller result schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} aggregation-controller result $defs missing")
    for definition_name in (
        "sample_output",
        "unit_output",
        "reduction_plan",
        "reduction_role",
        "reduction_axis",
        "reduction_method",
        "reduction_task_compatibility",
        "entity_unit_level",
        "prediction_block",
        "aggregated_prediction_block",
        "prediction_unit_id",
    ):
        require(
            definition_name in defs,
            f"{label} aggregation-controller result schema misses `{definition_name}`",
        )
    require(
        defs.get("reduction_method", {}).get("enum")
        == [
            "mean",
            "weighted_mean",
            "median",
            "vote",
            "robust_mean",
            "exclude_outliers",
            "custom",
        ],
        f"{label} aggregation-controller result reduction methods are not aligned",
    )
    require(
        defs.get("reduction_role", {}).get("enum")
        == ["score", "persist", "fold_ensemble", "meta_feature", "final_output"],
        f"{label} aggregation-controller result reduction roles are not aligned",
    )
    require(
        defs.get("reduction_axis", {}).get("enum") == ["unit", "fold", "model", "metric"],
        f"{label} aggregation-controller result reduction axes are not aligned",
    )
    require(
        defs.get("entity_unit_level", {}).get("enum")
        == ["physical_sample", "source_sample", "observation", "combo"],
        f"{label} aggregation-controller result entity unit levels are not aligned",
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
    for field in (
        "representation_plan",
        "representation_replay_manifest",
        "relation_delta_fingerprint",
    ):
        require(
            field in properties,
            f"{label} data-output provenance must declare optional `{field}`",
        )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} data-output provenance $defs are missing")
    for definition_name in (
        "combination_plan",
        "representation_plan",
        "representation_replay_manifest",
        "representation_sample_observation_mapping",
        "representation_combo_selection_record",
        "combination_mode",
        "representation_missing_source_policy",
        "representation_cardinality",
    ):
        require(
            definition_name in defs,
            f"{label} data-output provenance misses `{definition_name}` definition",
        )
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


def validate_node_task_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} NodeTask schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} NodeTask schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == NODE_TASK_SCHEMA_ID, f"{label} NodeTask schema $id mismatch")
    require(schema.get("type") == "object", f"{label} NodeTask root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} NodeTask root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} NodeTask required list is missing")
    for field in ("run_id", "node_plan", "phase", "variant_id", "fold_id", "seed"):
        require(field in required, f"{label} NodeTask schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} NodeTask $defs missing")
    for definition_name in (
        "node_plan",
        "handle_ref",
        "variant_execution_spec",
        "data_provider_view_spec",
        "prediction_input_spec",
        "artifact_input_spec",
        "fit_influence_task",
    ):
        require(definition_name in defs, f"{label} NodeTask schema misses `{definition_name}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} NodeTask properties missing")
    require("fit_influence" in properties, f"{label} NodeTask schema misses fit_influence")
    require(
        defs.get("handle_kind", {}).get("enum")
        == ["data", "data_view", "model", "artifact", "prediction", "relation"],
        f"{label} NodeTask handle_kind enum mismatch",
    )
    require(
        defs.get("phase", {}).get("enum")
        == ["COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"],
        f"{label} NodeTask phase enum mismatch",
    )
    require(
        set(defs.get("controller_capability", {}).get("enum", [])) == CONTROLLER_CAPABILITIES,
        f"{label} NodeTask capability enum mismatch",
    )
    require(
        set(defs.get("fit_influence_policy", {}).get("enum", [])) == FIT_INFLUENCE_POLICIES,
        f"{label} NodeTask fit influence policy enum mismatch",
    )
    require(
        set(defs.get("fit_influence_mechanism", {}).get("enum", []))
        == FIT_INFLUENCE_MECHANISMS,
        f"{label} NodeTask fit influence mechanism enum mismatch",
    )


def validate_node_result_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} NodeResult schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} NodeResult schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == NODE_RESULT_SCHEMA_ID, f"{label} NodeResult schema $id mismatch")
    require(schema.get("type") == "object", f"{label} NodeResult root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} NodeResult root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} NodeResult required list is missing")
    for field in ("node_id", "lineage"):
        require(field in required, f"{label} NodeResult schema must require `{field}`")
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} NodeResult $defs missing")
    for definition_name in (
        "handle_ref",
        "prediction_block",
        "observation_prediction_block",
        "aggregated_prediction_block",
        "prediction_unit_id",
        "explanation_block",
        "shape_delta",
        "artifact_ref",
        "lineage_record",
        "fit_influence_diagnostic",
    ):
        require(definition_name in defs, f"{label} NodeResult schema misses `{definition_name}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} NodeResult properties missing")
    require(
        "fit_influence_diagnostics" in properties,
        f"{label} NodeResult schema misses fit_influence_diagnostics",
    )
    require(
        defs.get("prediction_partition", {}).get("enum")
        == ["train", "validation", "test", "final"],
        f"{label} NodeResult prediction_partition enum mismatch",
    )
    require(
        defs.get("shape_delta_kind", {}).get("enum")
        == ["row", "feature", "target", "prediction"],
        f"{label} NodeResult shape_delta_kind enum mismatch",
    )
    require(
        set(defs.get("fit_influence_policy", {}).get("enum", [])) == FIT_INFLUENCE_POLICIES,
        f"{label} NodeResult fit influence policy enum mismatch",
    )
    require(
        set(defs.get("fit_influence_mechanism", {}).get("enum", []))
        == FIT_INFLUENCE_MECHANISMS,
        f"{label} NodeResult fit influence mechanism enum mismatch",
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


def validate_process_adapter_frame_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} process-adapter frame schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} process-adapter frame schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == PROCESS_ADAPTER_FRAME_SCHEMA_ID,
        f"{label} process-adapter frame schema has unexpected $id",
    )
    require(
        schema.get("type") == "object",
        f"{label} process-adapter frame root must be an object",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} process-adapter frame required list is missing")
    for field in ("type", "schema_version"):
        require(field in required, f"{label} process-adapter frame must require `{field}`")

    one_of = schema.get("oneOf")
    require(
        isinstance(one_of, list) and len(one_of) == 6,
        f"{label} process-adapter frame must declare six concrete frame variants",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} process-adapter frame definitions are missing")
    for definition_name in (
        "schema_version",
        "identifier",
        "non_empty_string",
        "worker_index",
        "worker_count",
        "node_task",
        "node_result",
        "init_frame",
        "task_frame",
        "close_frame",
        "ack_frame",
        "result_frame",
        "adapter_error",
        "error_frame",
        "request_frame",
        "response_frame",
    ):
        require(
            definition_name in defs,
            f"{label} process-adapter frame schema misses `{definition_name}`",
        )
    require(
        defs["schema_version"].get("const") == 1,
        f"{label} process-adapter frame schema_version const must be 1",
    )
    require(
        defs["node_task"].get("$ref") == NODE_TASK_SCHEMA_ID,
        f"{label} process-adapter frame NodeTask reference mismatch",
    )
    require(
        defs["node_result"].get("$ref") == NODE_RESULT_SCHEMA_ID,
        f"{label} process-adapter frame NodeResult reference mismatch",
    )
    expected_frame_defs = {
        "init_frame": (
            "init",
            {"type", "schema_version", "controller_id", "worker_index", "worker_count"},
        ),
        "task_frame": ("task", {"type", "schema_version", "task"}),
        "close_frame": ("close", {"type", "schema_version"}),
        "ack_frame": ("ack", {"type", "schema_version", "status"}),
        "result_frame": ("result", {"type", "schema_version", "result"}),
        "error_frame": ("error", {"type", "schema_version", "error"}),
    }
    for definition_name, (frame_type, required_fields) in expected_frame_defs.items():
        definition = defs[definition_name]
        require(
            definition.get("type") == "object",
            f"{label} {definition_name} must be an object",
        )
        require(
            definition.get("additionalProperties") is False,
            f"{label} {definition_name} must reject unknown fields",
        )
        variant_required = definition.get("required")
        require(
            isinstance(variant_required, list) and set(variant_required) == required_fields,
            f"{label} {definition_name} required fields mismatch",
        )
        properties = definition.get("properties")
        require(isinstance(properties, dict), f"{label} {definition_name} properties are missing")
        require(
            properties.get("type", {}).get("const") == frame_type,
            f"{label} {definition_name} type const mismatch",
        )
        require(
            properties.get("schema_version", {}).get("$ref") == "#/$defs/schema_version",
            f"{label} {definition_name} schema_version reference mismatch",
        )
    require(
        defs["ack_frame"]["properties"].get("status", {}).get("enum")
        == ["initialized", "closed"],
        f"{label} process-adapter ack status enum mismatch",
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
    observation_samples: dict[str, str] = {}
    effective_unit_ids: dict[str, str] = {}
    for index, record in enumerate(records):
        record_label = f"{label} coordinator relation #{index}"
        require(isinstance(record, dict), f"{record_label} must be an object")
        unit_level = record.get("unit_level", "observation")
        require(
            unit_level in {"physical_sample", "source_sample", "observation", "combo"},
            f"{record_label}.unit_level invalid",
        )
        observation_id = record.get("observation_id")
        sample_id = record.get("sample_id")
        require_non_empty_string(observation_id, f"{record_label}.observation_id")
        require_non_empty_string(sample_id, f"{record_label}.sample_id")
        require(
            observation_id not in observation_samples,
            f"{record_label}.observation_id duplicates another relation",
        )
        observation_samples[observation_id] = sample_id
        for field in (
            "unit_id",
            "target_id",
            "group_id",
            "origin_sample_id",
            "source_id",
            "derived_unit_id",
            "quality_flag",
        ):
            value = record.get(field)
            if value is not None:
                require_non_empty_string(value, f"{record_label}.{field}")
        rep_id = record.get("rep_id")
        if rep_id is not None:
            require_identifier(rep_id, f"{record_label}.rep_id")
        if unit_level == "source_sample":
            require(
                record.get("source_id") is not None,
                f"{record_label} source_sample requires source_id",
            )
        effective_unit_id = record.get("unit_id")
        if effective_unit_id is None:
            if unit_level == "physical_sample":
                effective_unit_id = sample_id
            elif unit_level == "source_sample":
                effective_unit_id = f"{sample_id}::{record.get('source_id')}"
            elif unit_level == "combo":
                effective_unit_id = record.get("derived_unit_id")
                require(
                    effective_unit_id is not None,
                    f"{record_label} combo requires derived_unit_id",
                )
            else:
                effective_unit_id = observation_id
        require(
            effective_unit_id not in effective_unit_ids,
            f"{record_label}.effective_unit_id duplicates {effective_unit_ids.get(effective_unit_id)}",
        )
        effective_unit_ids[effective_unit_id] = observation_id
        components = record.get("component_observation_ids", [])
        require(isinstance(components, list), f"{record_label}.component_observation_ids must be an array")
        require(len(set(components)) == len(components), f"{record_label}.component_observation_ids duplicate entries")
        for component_index, component_id in enumerate(components):
            require_non_empty_string(component_id, f"{record_label}.component_observation_ids[{component_index}]")
        if unit_level != "combo":
            require(not components, f"{record_label}.component_observation_ids require unit_level combo")
        sample_weight = record.get("sample_influence_weight")
        if sample_weight is not None:
            require(
                isinstance(sample_weight, (int, float))
                and not isinstance(sample_weight, bool)
                and math.isfinite(sample_weight)
                and sample_weight > 0,
                f"{record_label}.sample_influence_weight must be finite and > 0",
            )
        if "is_augmented" in record:
            require(
                isinstance(record["is_augmented"], bool),
                f"{record_label}.is_augmented must be boolean",
            )
    for index, record in enumerate(records):
        if record.get("unit_level", "observation") != "combo":
            continue
        record_label = f"{label} coordinator relation #{index}"
        components = record.get("component_observation_ids", [])
        require(
            record.get("derived_unit_id") is not None,
            f"{record_label} combo requires derived_unit_id",
        )
        require(components, f"{record_label} combo requires component_observation_ids")
        origin_sample_id = record.get("origin_sample_id")
        if origin_sample_id is not None:
            require(
                origin_sample_id == record["sample_id"],
                f"{record_label}.origin_sample_id must equal sample_id for combo",
            )
        for component_id in components:
            require(
                component_id in observation_samples,
                f"{record_label} references missing component observation {component_id}",
            )
            require(
                observation_samples[component_id] == record["sample_id"],
                f"{record_label} component {component_id} belongs to another sample",
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
    combination_plan = selector.get("combination_plan")
    if combination_plan is not None:
        validate_combination_plan(combination_plan, f"{label}.combination_plan")
    representation_plan = selector.get("representation_plan")
    if representation_plan is not None:
        validate_representation_plan(representation_plan, f"{label}.representation_plan")


def validate_fold_set_fixture(fold_set: Any, label: str) -> None:
    require(isinstance(fold_set, dict), f"{label} fold set must be an object")
    require_non_empty_string(fold_set.get("id"), f"{label}.id")
    sample_ids = fold_set.get("sample_ids")
    require(isinstance(sample_ids, list) and sample_ids, f"{label}.sample_ids must be non-empty")
    for index, sample_id in enumerate(sample_ids):
        require_identifier(sample_id, f"{label}.sample_ids[{index}]")
    require(len(set(sample_ids)) == len(sample_ids), f"{label}.sample_ids contain duplicates")
    sample_set = set(sample_ids)

    sample_groups = fold_set.get("sample_groups", {})
    require(isinstance(sample_groups, dict), f"{label}.sample_groups must be an object")
    for sample_id, group_id in sample_groups.items():
        require(sample_id in sample_set, f"{label}.sample_groups references unknown sample")
        require_identifier(group_id, f"{label}.sample_groups[{sample_id}]")
    if sample_groups:
        require(
            set(sample_groups) == sample_set,
            f"{label}.sample_groups must cover every sample when present",
        )

    folds = fold_set.get("folds")
    require(isinstance(folds, list) and folds, f"{label}.folds must be non-empty")
    fold_ids: list[str] = []
    validation_counts = {sample_id: 0 for sample_id in sample_ids}
    for index, fold in enumerate(folds):
        fold_label = f"{label}.folds[{index}]"
        require(isinstance(fold, dict), f"{fold_label} must be an object")
        require_identifier(fold.get("fold_id"), f"{fold_label}.fold_id")
        fold_ids.append(fold["fold_id"])
        train = fold.get("train_sample_ids")
        validation = fold.get("validation_sample_ids")
        require(isinstance(train, list), f"{fold_label}.train_sample_ids must be an array")
        require(
            isinstance(validation, list) and validation,
            f"{fold_label}.validation_sample_ids must be non-empty",
        )
        for sample_id in train + validation:
            require_identifier(sample_id, f"{fold_label} sample id")
            require(sample_id in sample_set, f"{fold_label} references unknown sample `{sample_id}`")
        require(len(set(train)) == len(train), f"{fold_label}.train_sample_ids contain duplicates")
        require(
            len(set(validation)) == len(validation),
            f"{fold_label}.validation_sample_ids contain duplicates",
        )
        require(
            set(train).isdisjoint(validation),
            f"{fold_label} has train/validation overlap",
        )
        for sample_id in validation:
            validation_counts[sample_id] += 1
    require(len(set(fold_ids)) == len(fold_ids), f"{label}.fold_id contains duplicates")
    for sample_id, count in validation_counts.items():
        require(
            count == 1,
            f"{label} sample `{sample_id}` appears in validation {count} time(s)",
        )


def canonical_fold_set_fingerprint(fold_set: Any) -> str:
    canonical = copy.deepcopy(fold_set)
    canonical["sample_ids"] = sorted(canonical["sample_ids"])
    canonical["folds"] = sorted(canonical["folds"], key=lambda fold: fold["fold_id"])
    for fold in canonical["folds"]:
        fold["train_sample_ids"] = sorted(fold["train_sample_ids"])
        fold["validation_sample_ids"] = sorted(fold["validation_sample_ids"])
        if fold.get("metadata") == {}:
            fold.pop("metadata")
    if canonical.get("sample_groups") == {}:
        canonical.pop("sample_groups")
    return canonical_json_sha256(canonical)


def validate_graph_spec(graph: Any, label: str) -> None:
    require(isinstance(graph, dict), f"{label} GraphSpec must be a JSON object")
    require_non_empty_string(graph.get("id"), f"{label}.id")
    nodes = graph.get("nodes")
    require(isinstance(nodes, list) and nodes, f"{label}.nodes must be non-empty")

    interface = graph.get("interface", {})
    require(isinstance(interface, dict), f"{label}.interface must be an object when present")
    graph_port_specs(interface.get("inputs", []), f"{label}.interface.inputs")
    graph_port_specs(interface.get("outputs", []), f"{label}.interface.outputs")

    node_ports: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
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
            "inputs": graph_port_specs(ports.get("inputs", []), f"{node_label}.ports.inputs"),
            "outputs": graph_port_specs(
                ports.get("outputs", []),
                f"{node_label}.ports.outputs",
            ),
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
        source_spec = node_ports[source_node]["outputs"].get(source_port)
        target_spec = node_ports[target_node]["inputs"].get(target_port)
        require(source_spec is not None, f"{edge_label} source port `{source_port}` is missing")
        require(target_spec is not None, f"{edge_label} target port `{target_port}` is missing")
        source_kind = source_spec["kind"]
        target_kind = target_spec["kind"]
        edge_kind = contract.get("kind")
        require(
            edge_kind == source_kind == target_kind,
            f"{edge_label} kind `{edge_kind}` does not match endpoint ports",
        )
        validate_graph_edge_contract(edge_label, contract, source_spec, target_spec)
        if contract.get("requires_oof") is True:
            require(edge_kind == "prediction", f"{edge_label} requires OOF on non-prediction edge")


def graph_port_specs(ports: Any, label: str) -> dict[str, dict[str, Any]]:
    require(isinstance(ports, list), f"{label} must be an array")
    seen: dict[str, dict[str, Any]] = {}
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
        require_optional_non_empty_string(
            port.get("representation"),
            f"{port_label}.representation",
        )
        require_optional_unit_level(port.get("unit_level"), f"{port_label}.unit_level")
        require_optional_identifier(port.get("alignment_key"), f"{port_label}.alignment_key")
        require_optional_unit_level(port.get("target_level"), f"{port_label}.target_level")
        seen[name] = {
            "kind": kind,
            "unit_level": port.get("unit_level"),
            "alignment_key": port.get("alignment_key"),
            "target_level": port.get("target_level"),
        }
    return seen


def validate_graph_edge_contract(
    label: str,
    contract: dict[str, Any],
    source_port: dict[str, Any],
    target_port: dict[str, Any],
) -> None:
    require_optional_non_empty_string(
        contract.get("representation"),
        f"{label}.contract.representation",
    )
    require_optional_unit_level(contract.get("unit_level"), f"{label}.contract.unit_level")
    require_optional_identifier(contract.get("alignment_key"), f"{label}.contract.alignment_key")
    require_optional_unit_level(contract.get("target_level"), f"{label}.contract.target_level")
    if "allows_broadcast" in contract:
        require(
            isinstance(contract.get("allows_broadcast"), bool),
            f"{label}.contract.allows_broadcast must be a boolean",
        )
    missingness_policy = contract.get("missingness_policy")
    if missingness_policy is not None:
        require(
            missingness_policy in MISSINGNESS_POLICIES,
            f"{label}.contract.missingness_policy is invalid",
        )

    relation_contract = contract.get("relation_contract")
    if relation_contract is not None:
        require(
            isinstance(relation_contract, dict),
            f"{label}.contract.relation_contract must be an object when present",
        )
        require_no_unknown_keys(
            relation_contract,
            {"relation_fingerprint", "required"},
            f"{label}.contract.relation_contract",
        )
        relation_fingerprint = relation_contract.get("relation_fingerprint")
        if relation_fingerprint is not None:
            require_sha256(
                relation_fingerprint,
                f"{label}.contract.relation_contract.relation_fingerprint",
            )
        elif relation_contract.get("required") is True:
            raise ContractError(
                f"{label}.contract.relation_contract is required but has no relation_fingerprint"
            )
        if "required" in relation_contract:
            require(
                isinstance(relation_contract.get("required"), bool),
                f"{label}.contract.relation_contract.required must be a boolean",
            )

    allows_broadcast = contract.get("allows_broadcast") is True
    contract_unit = contract.get("unit_level")
    for endpoint, port in (("source", source_port), ("target", target_port)):
        port_unit = port.get("unit_level")
        if contract_unit is not None and port_unit is not None:
            require(
                port_unit == contract_unit or allows_broadcast,
                f"{label} {endpoint} unit `{port_unit}` does not match edge unit `{contract_unit}`",
            )

    source_unit = source_port.get("unit_level")
    target_unit = target_port.get("unit_level")
    require(
        source_unit is None
        or target_unit is None
        or source_unit == target_unit
        or allows_broadcast,
        f"{label} joins incompatible unit levels `{source_unit}` and `{target_unit}`",
    )

    contract_target = contract.get("target_level")
    for endpoint, port in (("source", source_port), ("target", target_port)):
        port_target = port.get("target_level")
        if contract_target is not None and port_target is not None:
            require(
                port_target == contract_target,
                (
                    f"{label} {endpoint} target level `{port_target}` "
                    f"does not match edge target level `{contract_target}`"
                ),
            )
    source_target = source_port.get("target_level")
    target_target = target_port.get("target_level")
    require(
        source_target is None or target_target is None or source_target == target_target,
        f"{label} joins incompatible target levels `{source_target}` and `{target_target}`",
    )

    contract_alignment = contract.get("alignment_key")
    for endpoint, port in (("source", source_port), ("target", target_port)):
        port_alignment = port.get("alignment_key")
        if contract_alignment is not None and port_alignment is not None:
            require(
                port_alignment == contract_alignment or allows_broadcast,
                (
                    f"{label} {endpoint} alignment `{port_alignment}` "
                    f"does not match edge alignment `{contract_alignment}`"
                ),
            )
    source_alignment = source_port.get("alignment_key")
    target_alignment = target_port.get("alignment_key")
    require(
        source_alignment is None
        or target_alignment is None
        or source_alignment == target_alignment
        or allows_broadcast,
        f"{label} joins incompatible alignment keys `{source_alignment}` and `{target_alignment}`",
    )
    if allows_broadcast:
        require(
            contract_alignment is not None
            or source_alignment is not None
            or target_alignment is not None,
            f"{label} allows broadcast but declares no alignment key",
        )

    if graph_edge_is_relation_aware(contract, source_port, target_port):
        relation_fingerprint = None
        if isinstance(relation_contract, dict):
            relation_fingerprint = relation_contract.get("relation_fingerprint")
        require(
            relation_fingerprint is not None,
            f"{label} is relation-aware but has no relation_fingerprint",
        )
        require(
            has_graph_edge_unit_metadata(contract, source_port, target_port),
            f"{label} is relation-aware but has no unit_level metadata",
        )
        require(
            has_graph_edge_alignment_key(contract, source_port, target_port),
            f"{label} is relation-aware but has no alignment_key",
        )


def graph_edge_is_relation_aware(
    contract: dict[str, Any],
    source_port: dict[str, Any],
    target_port: dict[str, Any],
) -> bool:
    if contract.get("relation_contract") is not None or contract.get("allows_broadcast") is True:
        return True
    if contract.get("alignment_key") is not None:
        return True
    if contract.get("unit_level") is not None and contract.get("unit_level") != "physical_sample":
        return True
    if contract.get("target_level") is not None and contract.get("target_level") != "physical_sample":
        return True
    for port in (source_port, target_port):
        if port.get("alignment_key") is not None:
            return True
        if port.get("unit_level") is not None and port.get("unit_level") != "physical_sample":
            return True
        if port.get("target_level") is not None and port.get("target_level") != "physical_sample":
            return True
    return False


def has_graph_edge_unit_metadata(
    contract: dict[str, Any],
    source_port: dict[str, Any],
    target_port: dict[str, Any],
) -> bool:
    return (
        contract.get("unit_level") is not None
        or source_port.get("unit_level") is not None
        or target_port.get("unit_level") is not None
    )


def has_graph_edge_alignment_key(
    contract: dict[str, Any],
    source_port: dict[str, Any],
    target_port: dict[str, Any],
) -> bool:
    return (
        contract.get("alignment_key") is not None
        or source_port.get("alignment_key") is not None
        or target_port.get("alignment_key") is not None
    )


def validate_pipeline_dsl_fixture(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} Pipeline DSL fixture must be an object")
    require_non_empty_string(value.get("id"), f"{label}.id")
    pipeline = value.get("pipeline")
    require(isinstance(pipeline, list) and pipeline, f"{label}.pipeline must be non-empty")
    keys_seen: set[str] = set()
    for index, step in enumerate(pipeline):
        step_label = f"{label}.pipeline[{index}]"
        if isinstance(step, dict):
            keys_seen.update(step)
        elif isinstance(step, str) or step is None or isinstance(step, list):
            continue
        else:
            raise ContractError(f"{step_label} has unsupported JSON type")
    for required_key in ("_comment", "class", "_cartesian_", "split", "_chain_", "merge", "model"):
        require(required_key in keys_seen, f"{label} fixture must exercise `{required_key}`")
    require(
        any(
            isinstance(step, dict)
            and step.get("class") == "sklearn.model_selection.KFold"
            for step in pipeline
        ),
        f"{label} fixture must exercise a plain class splitter alias",
    )

    generator_keys = {
        key
        for step in pipeline
        if isinstance(step, dict)
        for key in step
        if key.startswith("_")
    }
    for expected in ("_cartesian_", "_chain_"):
        require(expected in generator_keys, f"{label} fixture must exercise `{expected}`")
    chain = next(
        step["_chain_"]
        for step in pipeline
        if isinstance(step, dict) and "_chain_" in step
    )
    require(isinstance(chain, list), f"{label}._chain_ must be an array")
    chain_keys = {key for step in chain if isinstance(step, dict) for key in step}
    for expected in ("_grid_", "_sample_"):
        require(expected in chain_keys, f"{label}._chain_ must exercise `{expected}`")


def validate_leakage_policy(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} leakage policy must be an object")
    split_unit = value.get("split_unit", "sample")
    require(
        split_unit in SPLIT_UNITS,
        f"{label}.split_unit is invalid",
    )
    for field in (
        "forbid_origin_cross_fold",
        "allow_observation_split_with_shared_target",
        "require_group_ids",
    ):
        if field in value:
            require(isinstance(value[field], bool), f"{label}.{field} must be boolean")
    flags = value.get("unsafe_flags", [])
    require(isinstance(flags, list), f"{label}.unsafe_flags must be an array")
    require(len(set(flags)) == len(flags), f"{label}.unsafe_flags contain duplicates")
    for index, flag in enumerate(flags):
        require_non_empty_string(flag, f"{label}.unsafe_flags[{index}]")
    if split_unit == "observation" and not value.get("allow_observation_split_with_shared_target", False):
        raise ContractError(f"{label} observation split requires explicit shared-target allowance")
    if value.get("require_group_ids", False) and split_unit != "group":
        raise ContractError(f"{label} require_group_ids=true requires split_unit=group")


def validate_aggregation_policy(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} aggregation policy must be an object")
    level = value.get("aggregation_level", "sample")
    method = value.get("method", "mean")
    weights = value.get("weights", "none")
    for field, field_value in (
        ("aggregation_level", level),
        ("selection_metric_level", value.get("selection_metric_level", "sample")),
    ):
        require(
            field_value in {"observation", "sample", "target", "group"},
            f"{label}.{field} is invalid",
        )
    require(
        method in {"none", "mean", "weighted_mean", "median", "vote", "custom_controller"},
        f"{label}.method is invalid",
    )
    require(
        weights in {"none", "quality", "repetition_count", "controller_emitted"},
        f"{label}.weights is invalid",
    )
    for field in (
        "emit_parallel_metrics",
        "store_raw_predictions",
        "store_aggregated_predictions",
    ):
        if field in value:
            require(isinstance(value[field], bool), f"{label}.{field} must be boolean")
    if method == "none" and level != "observation":
        raise ContractError(f"{label} method=none is only valid at observation level")
    if method == "weighted_mean" and weights == "none":
        raise ContractError(f"{label} weighted_mean requires explicit weights")
    if method != "weighted_mean" and weights != "none":
        raise ContractError(f"{label} weights require weighted_mean")
    if value.get("store_raw_predictions", True) is False and value.get(
        "store_aggregated_predictions", True
    ) is False:
        raise ContractError(f"{label} must store raw and/or aggregated predictions")


def validate_fold_set(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} fold_set must be an object")
    require_non_empty_string(value.get("id"), f"{label}.id")
    sample_ids = value.get("sample_ids")
    require(isinstance(sample_ids, list) and sample_ids, f"{label}.sample_ids must be non-empty")
    require(len(set(sample_ids)) == len(sample_ids), f"{label}.sample_ids contain duplicates")
    for index, sample_id in enumerate(sample_ids):
        require_identifier(sample_id, f"{label}.sample_ids[{index}]")
    sample_set = set(sample_ids)
    folds = value.get("folds")
    require(isinstance(folds, list) and folds, f"{label}.folds must be non-empty")
    validation_counts = {sample_id: 0 for sample_id in sample_ids}
    fold_ids: set[str] = set()
    for index, fold in enumerate(folds):
        fold_label = f"{label}.folds[{index}]"
        require(isinstance(fold, dict), f"{fold_label} must be an object")
        fold_id = fold.get("fold_id")
        require_identifier(fold_id, f"{fold_label}.fold_id")
        require(fold_id not in fold_ids, f"{label} duplicate fold id `{fold_id}`")
        fold_ids.add(fold_id)
        train = fold.get("train_sample_ids")
        validation = fold.get("validation_sample_ids")
        require(isinstance(train, list), f"{fold_label}.train_sample_ids must be an array")
        require(
            isinstance(validation, list) and validation,
            f"{fold_label}.validation_sample_ids must be non-empty",
        )
        require(len(set(train)) == len(train), f"{fold_label}.train_sample_ids duplicate samples")
        require(
            len(set(validation)) == len(validation),
            f"{fold_label}.validation_sample_ids duplicate samples",
        )
        train_set = set(train)
        validation_set = set(validation)
        require(
            train_set.isdisjoint(validation_set),
            f"{fold_label} has train/validation overlap",
        )
        for sample_id in train_set | validation_set:
            require(sample_id in sample_set, f"{fold_label} references unknown sample `{sample_id}`")
        for sample_id in validation:
            validation_counts[sample_id] += 1
        metadata = fold.get("metadata", {})
        require(isinstance(metadata, dict), f"{fold_label}.metadata must be an object")
    for sample_id, count in validation_counts.items():
        require(count == 1, f"{label} sample `{sample_id}` validation count is {count}, expected 1")
    sample_groups = value.get("sample_groups", {})
    require(isinstance(sample_groups, dict), f"{label}.sample_groups must be an object")
    for sample_id, group_id in sample_groups.items():
        require(sample_id in sample_set, f"{label}.sample_groups references unknown sample `{sample_id}`")
        require_identifier(group_id, f"{label}.sample_groups[{sample_id}]")


def validate_generation_spec(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} generation spec must be an object")
    strategy = value.get("strategy", "none")
    require(strategy in {"none", "cartesian", "zip"}, f"{label}.strategy is invalid")
    dimensions = value.get("dimensions", [])
    require(isinstance(dimensions, list), f"{label}.dimensions must be an array")
    max_variants = value.get("max_variants", 1)
    if max_variants is not None:
        require(isinstance(max_variants, int) and max_variants >= 1, f"{label}.max_variants invalid")
    if strategy == "none":
        require(not dimensions, f"{label} strategy=none must not declare dimensions")
        return
    require(dimensions, f"{label} non-none strategy requires dimensions")
    names: list[str] = []
    choice_counts: list[int] = []
    for index, dimension in enumerate(dimensions):
        dimension_label = f"{label}.dimensions[{index}]"
        require(isinstance(dimension, dict), f"{dimension_label} must be an object")
        name = dimension.get("name")
        require_non_empty_string(name, f"{dimension_label}.name")
        names.append(name)
        choices = dimension.get("choices", [])
        require(isinstance(choices, list) and choices, f"{dimension_label}.choices must be non-empty")
        choice_counts.append(len(choices))
        labels: list[str] = []
        for choice_index, choice in enumerate(choices):
            choice_label = f"{dimension_label}.choices[{choice_index}]"
            require(isinstance(choice, dict), f"{choice_label} must be an object")
            label_value = choice.get("label")
            require_non_empty_string(label_value, f"{choice_label}.label")
            labels.append(label_value)
            require("value" in choice, f"{choice_label}.value is required")
            overrides = choice.get("param_overrides", [])
            require(isinstance(overrides, list), f"{choice_label}.param_overrides must be an array")
            for override_index, override in enumerate(overrides):
                override_label = f"{choice_label}.param_overrides[{override_index}]"
                require(isinstance(override, dict), f"{override_label} must be an object")
                require_identifier(override.get("node_id"), f"{override_label}.node_id")
                params = override.get("params")
                require(isinstance(params, dict) and params, f"{override_label}.params non-empty")
        require(len(set(labels)) == len(labels), f"{dimension_label}.choices duplicate labels")
    require(len(set(names)) == len(names), f"{label}.dimensions duplicate names")
    if strategy == "zip":
        require(len(set(choice_counts)) == 1, f"{label} zip dimensions must have equal lengths")


def validate_data_model_shape_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} shape plan must be an object")
    require_identifier(value.get("node_id"), f"{label}.node_id")
    for field in ("input_granularity", "target_granularity"):
        field_value = value.get(field, "sample")
        require(field_value in {"observation", "sample", "target", "group"}, f"{label}.{field} invalid")
    for field in ("fit_rows", "predict_rows"):
        field_value = value.get(field, "fold_train" if field == "fit_rows" else "fold_validation")
        require(
            field_value in {"fold_train", "fold_validation", "full_train", "predict"},
            f"{label}.{field} invalid",
        )
    namespace = value.get("feature_namespace")
    if namespace is not None:
        require_non_empty_string(namespace, f"{label}.feature_namespace")
    fingerprint = value.get("feature_schema_fingerprint")
    if fingerprint is not None:
        require_sha256(fingerprint, f"{label}.feature_schema_fingerprint")
    require_non_empty_string(value.get("target_space", "raw"), f"{label}.target_space")
    validate_aggregation_policy(value.get("aggregation_policy", {}), f"{label}.aggregation_policy")
    augmentation = value.get("augmentation_policy", {})
    require(isinstance(augmentation, dict), f"{label}.augmentation_policy must be an object")
    for field in ("sample_scope", "feature_scope"):
        field_value = augmentation.get(field, "train_only")
        require(field_value in {"none", "train_only", "all_partitions"}, f"{label}.augmentation_policy.{field} invalid")
    for field in ("require_origin_id", "inherit_group", "inherit_target"):
        if field in augmentation:
            require(isinstance(augmentation[field], bool), f"{label}.augmentation_policy.{field} boolean")
    selection = value.get("selection_policy", {})
    require(isinstance(selection, dict), f"{label}.selection_policy must be an object")
    scope = selection.get("scope", "none")
    require(scope in {"none", "unsupervised", "supervised_fold_train"}, f"{label}.selection_policy.scope invalid")
    for field in ("store_masks", "allow_schema_mismatch_on_join"):
        if field in selection:
            require(isinstance(selection[field], bool), f"{label}.selection_policy.{field} boolean")
    if scope == "supervised_fold_train" and value.get("fit_rows", "fold_train") != "fold_train":
        raise ContractError(f"{label} supervised feature selection must fit on fold_train")


def validate_data_binding(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} data binding must be an object")
    require_identifier(value.get("node_id"), f"{label}.node_id")
    require_non_empty_string(value.get("input_name"), f"{label}.input_name")
    require_non_empty_string(value.get("request_id"), f"{label}.request_id")
    require_sha256(value.get("schema_fingerprint"), f"{label}.schema_fingerprint")
    require_sha256(value.get("plan_fingerprint"), f"{label}.plan_fingerprint")
    relation_fingerprint = value.get("relation_fingerprint")
    if relation_fingerprint is not None:
        require_sha256(relation_fingerprint, f"{label}.relation_fingerprint")
    require_non_empty_string(value.get("output_representation"), f"{label}.output_representation")
    feature_set_id = value.get("feature_set_id")
    if feature_set_id is not None:
        require_non_empty_string(feature_set_id, f"{label}.feature_set_id")
    source_ids = value.get("source_ids", [])
    require(isinstance(source_ids, list), f"{label}.source_ids must be an array")
    require(len(set(source_ids)) == len(source_ids), f"{label}.source_ids contain duplicates")
    for index, source_id in enumerate(source_ids):
        require_non_empty_string(source_id, f"{label}.source_ids[{index}]")
    if "require_relations" in value:
        require(isinstance(value["require_relations"], bool), f"{label}.require_relations boolean")
    if value.get("require_relations", False):
        require(relation_fingerprint is not None, f"{label} requires relation_fingerprint")
    view_policy = value.get("view_policy", {})
    require(isinstance(view_policy, dict), f"{label}.view_policy must be an object")
    fit_partition = view_policy.get("fit_partition", "fold_train")
    predict_partition = view_policy.get("predict_partition", "fold_validation")
    require(
        fit_partition in {"fold_train", "fold_validation", "full_train", "predict"},
        f"{label}.view_policy.fit_partition invalid",
    )
    require(
        predict_partition in {"fold_train", "fold_validation", "full_train", "predict"},
        f"{label}.view_policy.predict_partition invalid",
    )
    for field in (
        "include_augmented_train",
        "include_augmented_validation",
        "include_excluded",
        "require_sample_ids",
    ):
        if field in view_policy:
            require(isinstance(view_policy[field], bool), f"{label}.view_policy.{field} boolean")


def validate_data_view_selector(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} selector must be an object")
    source_ids = value.get("source_ids", [])
    require(isinstance(source_ids, list), f"{label}.source_ids must be an array")
    for index, source_id in enumerate(source_ids):
        require_non_empty_string(source_id, f"{label}.source_ids[{index}]")
    require(len(set(source_ids)) == len(source_ids), f"{label}.source_ids contain duplicates")
    metadata = value.get("metadata", {})
    require(isinstance(metadata, dict), f"{label}.metadata must be an object")
    for key in metadata:
        require_non_empty_string(key, f"{label}.metadata key")
    tags = value.get("tags", [])
    require(isinstance(tags, list), f"{label}.tags must be an array")
    for index, tag in enumerate(tags):
        require_non_empty_string(tag, f"{label}.tags[{index}]")
    require(len(set(tags)) == len(tags), f"{label}.tags contain duplicates")
    if "filter" in value:
        require(value["filter"] is not None, f"{label}.filter must not be null")
    require(
        bool(source_ids) or bool(metadata) or bool(tags) or "filter" in value,
        f"{label} selector must constrain source_ids, metadata, tags or filter",
    )


def validate_branch_view_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} branch view plan must be an object")
    require_non_empty_string(value.get("view_id"), f"{label}.view_id")
    require_non_empty_string(value.get("branch_id"), f"{label}.branch_id")
    mode = value.get("mode")
    require(
        mode in {"separation", "by_source", "by_metadata", "by_tag", "by_filter"},
        f"{label}.mode is invalid",
    )
    selector = value.get("selector")
    validate_data_view_selector(selector, f"{label}.selector")
    if mode == "by_source":
        require(bool(selector.get("source_ids")), f"{label}.selector.source_ids required")
    if mode == "by_metadata":
        require(bool(selector.get("metadata")), f"{label}.selector.metadata required")
    if mode == "by_tag":
        require(bool(selector.get("tags")), f"{label}.selector.tags required")
    if mode == "by_filter":
        require("filter" in selector, f"{label}.selector.filter required")
    allow_overlap = value.get("allow_overlap", False)
    require(isinstance(allow_overlap, bool), f"{label}.allow_overlap must be boolean")
    metadata = value.get("metadata", {})
    require(isinstance(metadata, dict), f"{label}.metadata must be an object")


def validate_campaign_spec(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} CampaignSpec must be an object")
    require_non_empty_string(value.get("id"), f"{label}.id")
    root_seed = value.get("root_seed")
    if root_seed is not None:
        require(isinstance(root_seed, int) and root_seed >= 0, f"{label}.root_seed invalid")
    validate_leakage_policy(value.get("leakage_policy", {}), f"{label}.leakage_policy")
    validate_aggregation_policy(value.get("aggregation_policy", {}), f"{label}.aggregation_policy")
    split_invocation = value.get("split_invocation")
    if split_invocation is not None:
        require(isinstance(split_invocation, dict), f"{label}.split_invocation must be object")
        require_non_empty_string(split_invocation.get("id"), f"{label}.split_invocation.id")
        controller_id = split_invocation.get("controller_id")
        if controller_id is not None:
            require_identifier(controller_id, f"{label}.split_invocation.controller_id")
        validate_leakage_policy(
            split_invocation.get("leakage_policy", {}),
            f"{label}.split_invocation.leakage_policy",
        )
        params = split_invocation.get("params", {})
        require(isinstance(params, dict), f"{label}.split_invocation.params must be object")
        fold_set = split_invocation.get("fold_set")
        if fold_set is not None:
            validate_fold_set(fold_set, f"{label}.split_invocation.fold_set")
    validate_generation_spec(value.get("generation", {}), f"{label}.generation")
    shape_plans = value.get("shape_plans", {})
    require(isinstance(shape_plans, dict), f"{label}.shape_plans must be an object")
    for key, shape_plan in shape_plans.items():
        validate_data_model_shape_plan(shape_plan, f"{label}.shape_plans[{key}]")
        require(shape_plan.get("node_id") == key, f"{label}.shape_plans key `{key}` mismatch")
    data_bindings = value.get("data_bindings", {})
    require(isinstance(data_bindings, dict), f"{label}.data_bindings must be an object")
    for key, bindings in data_bindings.items():
        require(isinstance(bindings, list), f"{label}.data_bindings[{key}] must be an array")
        for index, binding in enumerate(bindings):
            validate_data_binding(binding, f"{label}.data_bindings[{key}][{index}]")
            require(binding.get("node_id") == key, f"{label}.data_bindings key `{key}` mismatch")
    branch_view_plans = value.get("branch_view_plans", [])
    require(isinstance(branch_view_plans, list), f"{label}.branch_view_plans must be an array")
    seen_branch_views: set[str] = set()
    for index, view_plan in enumerate(branch_view_plans):
        validate_branch_view_plan(view_plan, f"{label}.branch_view_plans[{index}]")
        view_id = view_plan["view_id"]
        require(view_id not in seen_branch_views, f"{label}.branch_view_plans duplicate `{view_id}`")
        seen_branch_views.add(view_id)
    metadata = value.get("metadata", {})
    require(isinstance(metadata, dict), f"{label}.metadata must be an object")


def validate_execution_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} ExecutionPlan must be an object")
    require_non_empty_string(value.get("id"), f"{label}.id")
    graph_plan = value.get("graph_plan")
    require(isinstance(graph_plan, dict), f"{label}.graph_plan must be an object")
    graph = graph_plan.get("graph")
    validate_graph_spec(graph, f"{label}.graph_plan.graph")
    graph_node_ids = [node["id"] for node in graph["nodes"]]
    graph_node_id_set = set(graph_node_ids)

    topological_order = graph_plan.get("topological_order")
    require(
        isinstance(topological_order, list) and topological_order,
        f"{label}.graph_plan.topological_order must be non-empty",
    )
    for index, node_id in enumerate(topological_order):
        require_identifier(node_id, f"{label}.graph_plan.topological_order[{index}]")
    require(
        set(topological_order) == graph_node_id_set,
        f"{label}.graph_plan.topological_order must cover graph nodes",
    )

    parallel_levels = graph_plan.get("parallel_levels", [])
    require(isinstance(parallel_levels, list), f"{label}.graph_plan.parallel_levels must be an array")
    flattened_levels: list[str] = []
    for level_index, level in enumerate(parallel_levels):
        require(isinstance(level, list), f"{label}.graph_plan.parallel_levels[{level_index}] array")
        for node_index, node_id in enumerate(level):
            require_identifier(
                node_id,
                f"{label}.graph_plan.parallel_levels[{level_index}][{node_index}]",
            )
            flattened_levels.append(node_id)
    if flattened_levels:
        require(
            set(flattened_levels) == graph_node_id_set,
            f"{label}.graph_plan.parallel_levels must cover graph nodes",
        )

    validate_campaign_spec(value.get("campaign"), f"{label}.campaign")
    node_plans = value.get("node_plans")
    require(isinstance(node_plans, dict) and node_plans, f"{label}.node_plans must be non-empty")
    require(set(node_plans.keys()) == graph_node_id_set, f"{label}.node_plans must match graph nodes")
    controllers = value.get("controller_manifests")
    require(
        isinstance(controllers, dict) and controllers,
        f"{label}.controller_manifests must be non-empty",
    )
    for controller_id, manifest in controllers.items():
        require_identifier(controller_id, f"{label}.controller_manifests key")
        validate_controller_manifest(manifest, f"{label}.controller_manifests[{controller_id}]")
        require(
            manifest.get("controller_id") == controller_id,
            f"{label}.controller_manifests key `{controller_id}` mismatch",
        )

    for key, node_plan in node_plans.items():
        node_label = f"{label}.node_plans[{key}]"
        require(isinstance(node_plan, dict), f"{node_label} must be an object")
        require(node_plan.get("node_id") == key, f"{node_label}.node_id must match key")
        require(
            node_plan.get("kind")
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
            f"{node_label}.kind invalid",
        )
        controller_id = node_plan.get("controller_id")
        require_identifier(controller_id, f"{node_label}.controller_id")
        require(controller_id in controllers, f"{node_label}.controller_id has no manifest")
        require_non_empty_string(node_plan.get("controller_version"), f"{node_label}.controller_version")
        phases = node_plan.get("supported_phases")
        require(isinstance(phases, list) and phases, f"{node_label}.supported_phases non-empty")
        for phase_index, phase in enumerate(phases):
            require(
                phase in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
                f"{node_label}.supported_phases[{phase_index}] invalid",
            )
        capabilities = node_plan.get("controller_capabilities", [])
        require(isinstance(capabilities, list), f"{node_label}.controller_capabilities array")
        require(
            node_plan.get("fit_scope")
            in {"stateless", "fold_train", "full_train", "inference_only"},
            f"{node_label}.fit_scope invalid",
        )
        require(
            node_plan.get("rng_policy")
            in {
                "uses_core_seed",
                "ignores_seed",
                "externally_deterministic",
                "nondeterministic",
            },
            f"{node_label}.rng_policy invalid",
        )
        require(
            node_plan.get("artifact_policy")
            in {"serializable", "host_only", "content_addressed", "replay_required"},
            f"{node_label}.artifact_policy invalid",
        )
        for field in ("input_nodes", "output_nodes"):
            node_refs = node_plan.get(field)
            require(isinstance(node_refs, list), f"{node_label}.{field} must be an array")
            for ref_index, node_ref in enumerate(node_refs):
                require_identifier(node_ref, f"{node_label}.{field}[{ref_index}]")
                require(node_ref in graph_node_id_set, f"{node_label}.{field}[{ref_index}] unknown")
        shape_plan = node_plan.get("shape_plan")
        if shape_plan is not None:
            validate_data_model_shape_plan(shape_plan, f"{node_label}.shape_plan")
            require(shape_plan.get("node_id") == key, f"{node_label}.shape_plan node_id mismatch")
        data_bindings = node_plan.get("data_bindings", [])
        require(isinstance(data_bindings, list), f"{node_label}.data_bindings must be an array")
        for binding_index, binding in enumerate(data_bindings):
            validate_data_binding(binding, f"{node_label}.data_bindings[{binding_index}]")
            require(binding.get("node_id") == key, f"{node_label}.data_bindings node_id mismatch")
        params = node_plan.get("params", {})
        require(isinstance(params, dict), f"{node_label}.params must be an object")
        require_sha256(node_plan.get("params_fingerprint"), f"{node_label}.params_fingerprint")

    variants = value.get("variants")
    require(isinstance(variants, list) and variants, f"{label}.variants must be non-empty")
    for index, variant in enumerate(variants):
        variant_label = f"{label}.variants[{index}]"
        require(isinstance(variant, dict), f"{variant_label} must be an object")
        require_identifier(variant.get("variant_id"), f"{variant_label}.variant_id")
        require_sha256(variant.get("fingerprint"), f"{variant_label}.fingerprint")
        seed = variant.get("seed")
        if seed is not None:
            require(isinstance(seed, int) and seed >= 0, f"{variant_label}.seed invalid")
        choices = variant.get("choices", {})
        require(isinstance(choices, dict), f"{variant_label}.choices must be an object")
        for dimension_name, choice in choices.items():
            choice_label = f"{variant_label}.choices[{dimension_name}]"
            require_non_empty_string(dimension_name, f"{choice_label}.dimension")
            require(isinstance(choice, dict), f"{choice_label} must be an object")
            require_non_empty_string(choice.get("label"), f"{choice_label}.label")
            overrides = choice.get("param_overrides", [])
            require(isinstance(overrides, list), f"{choice_label}.param_overrides must be an array")
            for override_index, override in enumerate(overrides):
                override_label = f"{choice_label}.param_overrides[{override_index}]"
                require(isinstance(override, dict), f"{override_label} must be an object")
                override_node = override.get("node_id")
                require_identifier(override_node, f"{override_label}.node_id")
                require(override_node in graph_node_id_set, f"{override_label}.node_id unknown")
                override_params = override.get("params", {})
                require(isinstance(override_params, dict), f"{override_label}.params must be object")

    fold_set = value.get("fold_set")
    if fold_set is not None:
        validate_fold_set(fold_set, f"{label}.fold_set")
    require_sha256(value.get("graph_fingerprint"), f"{label}.graph_fingerprint")
    require_sha256(value.get("campaign_fingerprint"), f"{label}.campaign_fingerprint")
    require_sha256(value.get("controller_fingerprint"), f"{label}.controller_fingerprint")


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
        representation_plan = fusion.get("representation_plan")
        if representation_plan is not None:
            validate_representation_plan(
                representation_plan,
                f"{label}.default_fusion.representation_plan",
            )
    fit_influence_policy = value.get("fit_influence_policy")
    if fit_influence_policy is not None:
        require(
            fit_influence_policy in FIT_INFLUENCE_POLICIES,
            f"{label}.fit_influence_policy is invalid",
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


def validate_controller_manifest(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} ControllerManifest must be an object")
    require_identifier(value.get("controller_id"), f"{label}.controller_id")
    require_non_empty_string(value.get("controller_version"), f"{label}.controller_version")
    require(
        value.get("operator_kind")
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
        f"{label}.operator_kind is invalid",
    )
    priority = value.get("priority", 0)
    require(isinstance(priority, int) and 0 <= priority <= 4294967295, f"{label}.priority invalid")

    phases = value.get("supported_phases")
    require(isinstance(phases, list) and phases, f"{label}.supported_phases must be non-empty")
    require(len(set(phases)) == len(phases), f"{label}.supported_phases contain duplicates")
    for index, phase in enumerate(phases):
        require(
            phase in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
            f"{label}.supported_phases[{index}] is invalid",
        )

    for field in ("input_ports", "output_ports"):
        ports = value.get(field, [])
        require(isinstance(ports, list), f"{label}.{field} must be an array")
        seen: set[str] = set()
        for index, port in enumerate(ports):
            port_label = f"{label}.{field}[{index}]"
            require(isinstance(port, dict), f"{port_label} must be an object")
            name = port.get("name")
            require_non_empty_string(name, f"{port_label}.name")
            require(name not in seen, f"{label}.{field} duplicate port `{name}`")
            seen.add(name)
            require(
                port.get("kind") in {"data", "target", "prediction", "artifact", "metric", "control"},
                f"{port_label}.kind is invalid",
            )
            representation = port.get("representation")
            if representation is not None:
                require_non_empty_string(representation, f"{port_label}.representation")
            require(
                port.get("cardinality") in {"one", "many", "optional"},
                f"{port_label}.cardinality is invalid",
            )

    data_requirements = value.get("data_requirements")
    if data_requirements is not None:
        validate_model_input_spec(data_requirements, f"{label}.data_requirements")

    capabilities = value.get("capabilities", [])
    require(isinstance(capabilities, list), f"{label}.capabilities must be an array")
    require(len(set(capabilities)) == len(capabilities), f"{label}.capabilities contain duplicates")
    for index, capability in enumerate(capabilities):
        require(
            capability in CONTROLLER_CAPABILITIES,
            f"{label}.capabilities[{index}] is invalid",
        )
    require(
        value.get("fit_scope") in {"stateless", "fold_train", "full_train", "inference_only"},
        f"{label}.fit_scope is invalid",
    )
    require(
        value.get("rng_policy")
        in {"uses_core_seed", "ignores_seed", "externally_deterministic", "nondeterministic"},
        f"{label}.rng_policy is invalid",
    )
    require(
        value.get("artifact_policy")
        in {"serializable", "host_only", "content_addressed", "replay_required"},
        f"{label}.artifact_policy is invalid",
    )
    if "deterministic" in capabilities and value.get("rng_policy") == "nondeterministic":
        raise ContractError(f"{label} cannot be deterministic with nondeterministic RNG")
    if any(port.get("kind") == "prediction" for port in value.get("output_ports", [])):
        require(
            "emits_predictions" in capabilities,
            f"{label} prediction outputs require emits_predictions capability",
        )
    if any(port.get("kind") == "artifact" for port in value.get("output_ports", [])):
        require(
            "emits_artifacts" in capabilities,
            f"{label} artifact outputs require emits_artifacts capability",
        )


def validate_controller_manifest_list(value: Any, label: str) -> None:
    require(isinstance(value, list) and value, f"{label} must be a non-empty manifest array")
    seen: set[str] = set()
    for index, manifest in enumerate(value):
        manifest_label = f"{label}[{index}]"
        validate_controller_manifest(manifest, manifest_label)
        controller_id = manifest["controller_id"]
        require(controller_id not in seen, f"{label} duplicate controller id `{controller_id}`")
        seen.add(controller_id)


def validate_selection_policy(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} SelectionPolicy must be an object")
    require_non_empty_string(value.get("id"), f"{label}.id")
    metric = value.get("metric")
    validate_selection_metric(metric, f"{label}.metric")
    level = value.get("required_metric_level")
    if level is not None:
        require(level in PREDICTION_LEVELS, f"{label}.required_metric_level invalid")
    if "require_finite" in value:
        require(isinstance(value["require_finite"], bool), f"{label}.require_finite must be boolean")
    evaluation_scope = value.get("evaluation_scope")
    if evaluation_scope is not None:
        require(evaluation_scope in EVALUATION_SCOPES, f"{label}.evaluation_scope invalid")
    validate_refit_slot_plan(value.get("refit_slot_plan"), f"{label}.refit_slot_plan", optional=True)
    validate_stacking_fit_contract(
        value.get("stacking_fit_contract"),
        f"{label}.stacking_fit_contract",
        optional=True,
    )
    require_optional_non_empty_string(value.get("reduction_id"), f"{label}.reduction_id")


def validate_selection_decision(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} SelectionDecision must be an object")
    require_non_empty_string(value.get("policy_id"), f"{label}.policy_id")
    selected_candidate = value.get("selected_candidate_id")
    require_non_empty_string(selected_candidate, f"{label}.selected_candidate_id")
    require_non_empty_string(value.get("metric_name"), f"{label}.metric_name")
    require(value.get("objective") in {"minimize", "maximize"}, f"{label}.objective invalid")
    metric_level = value.get("metric_level")
    if metric_level is not None:
        require(metric_level in PREDICTION_LEVELS, f"{label}.metric_level invalid")
    evaluation_scope = value.get("evaluation_scope")
    if evaluation_scope is not None:
        require(evaluation_scope in EVALUATION_SCOPES, f"{label}.evaluation_scope invalid")
    validate_refit_slot_plan(value.get("refit_slot_plan"), f"{label}.refit_slot_plan", optional=True)
    require_optional_non_empty_string(value.get("reduction_id"), f"{label}.reduction_id")
    selected_score = value.get("selected_score")
    require(isinstance(selected_score, (int, float)), f"{label}.selected_score must be numeric")
    ranked = value.get("ranked_candidates")
    require(isinstance(ranked, list) and ranked, f"{label}.ranked_candidates must be non-empty")
    require(
        ranked[0].get("candidate_id") == selected_candidate,
        f"{label} first ranked candidate must match selected_candidate_id",
    )
    seen: set[str] = set()
    for index, candidate in enumerate(ranked):
        candidate_label = f"{label}.ranked_candidates[{index}]"
        require(isinstance(candidate, dict), f"{candidate_label} must be an object")
        candidate_id = candidate.get("candidate_id")
        require_non_empty_string(candidate_id, f"{candidate_label}.candidate_id")
        require(candidate_id not in seen, f"{label} duplicate ranked candidate `{candidate_id}`")
        seen.add(candidate_id)
        require(isinstance(candidate.get("score"), (int, float)), f"{candidate_label}.score numeric")
        require(candidate.get("rank") == index + 1, f"{candidate_label}.rank must be {index + 1}")


def validate_selection_metric(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require_non_empty_string(value.get("name"), f"{label}.name")
    require(value.get("objective") in {"minimize", "maximize"}, f"{label}.objective invalid")


def validate_refit_slot_plan(value: Any, label: str, *, optional: bool = False) -> None:
    if value is None:
        require(optional, f"{label} must be an object")
        return
    require(isinstance(value, dict), f"{label} must be an object")
    strategy = value.get("strategy")
    require(strategy in {"refit_one", "refit_ensemble"}, f"{label}.strategy invalid")
    selection_level = value.get("selection_level")
    require(selection_level in PREDICTION_LEVELS, f"{label}.selection_level invalid")
    member_count = value.get("member_count")
    require(isinstance(member_count, int) and member_count >= 1, f"{label}.member_count invalid")
    if strategy == "refit_one":
        require(member_count == 1, f"{label}.refit_one requires member_count=1")
    if strategy == "refit_ensemble":
        require(member_count >= 2, f"{label}.refit_ensemble requires member_count>=2")
    validate_selection_metric(value.get("selection_metric"), f"{label}.selection_metric")
    require_optional_non_empty_string(value.get("reduction_id"), f"{label}.reduction_id")


def validate_stacking_fit_contract(value: Any, label: str, *, optional: bool = False) -> None:
    if value is None:
        require(optional, f"{label} must be an object")
        return
    require(isinstance(value, dict), f"{label} must be an object")
    require(value.get("meta_training_features") == "oof", f"{label}.meta_training_features invalid")
    require(
        value.get("inference_features") == "refit_base_predictions",
        f"{label}.inference_features invalid",
    )
    protocol = value.get("selection_protocol")
    require(protocol in {"nested", "holdout", "reuse_oof"}, f"{label}.selection_protocol invalid")
    domain = value.get("meta_row_domain")
    require(domain in {"sample", "combo"}, f"{label}.meta_row_domain invalid")
    final_reduction_id = value.get("final_reduction_id")
    require_optional_non_empty_string(final_reduction_id, f"{label}.final_reduction_id")
    unsafe_allow_reuse_oof = value.get("unsafe_allow_reuse_oof", False)
    require(isinstance(unsafe_allow_reuse_oof, bool), f"{label}.unsafe_allow_reuse_oof boolean")
    if protocol == "reuse_oof" and not unsafe_allow_reuse_oof:
        raise ContractError(f"{label} reuse_oof requires unsafe_allow_reuse_oof=true")
    if domain == "combo" and final_reduction_id is None:
        raise ContractError(f"{label} combo meta_row_domain requires final_reduction_id")


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
        "relation_delta_fingerprint",
    ):
        field_value = value.get(field)
        if field_value is not None:
            require_sha256(field_value, f"{label}.{field}")
    representation_plan = value.get("representation_plan")
    if representation_plan is not None:
        validate_representation_plan(representation_plan, f"{label}.representation_plan")
    representation_replay_manifest = value.get("representation_replay_manifest")
    if representation_replay_manifest is not None:
        validate_representation_replay_manifest(
            representation_replay_manifest,
            f"{label}.representation_replay_manifest",
        )
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


def validate_handle_ref(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} handle ref must be an object")
    require(isinstance(value.get("handle"), int) and value["handle"] >= 0, f"{label}.handle invalid")
    require(
        value.get("kind") in {"data", "data_view", "model", "artifact", "prediction", "relation"},
        f"{label}.kind invalid",
    )
    require_identifier(value.get("owner_controller"), f"{label}.owner_controller")


def validate_fit_influence_task(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    requested = value.get("requested_policy")
    effective = value.get("effective_policy")
    mechanism = value.get("mechanism")
    require(requested in FIT_INFLUENCE_POLICIES, f"{label}.requested_policy invalid")
    require(effective in FIT_INFLUENCE_POLICIES, f"{label}.effective_policy invalid")
    require(mechanism in FIT_INFLUENCE_MECHANISMS, f"{label}.mechanism invalid")
    weights = value.get("row_weights", [])
    require(isinstance(weights, list), f"{label}.row_weights must be an array")
    for index, weight in enumerate(weights):
        require(
            isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight) and weight > 0,
            f"{label}.row_weights[{index}] must be finite and > 0",
        )
    warnings = value.get("warnings", [])
    require(isinstance(warnings, list), f"{label}.warnings must be an array")
    for index, warning in enumerate(warnings):
        require_non_empty_string(warning, f"{label}.warnings[{index}]")
    if effective in {"equal_sample_influence", "backend_loss_weight"}:
        require(weights, f"{label}.{effective} requires row_weights")
    if requested == "strict_weight_support" and effective == "uniform_rows":
        raise ContractError(f"{label} strict_weight_support cannot fall back to uniform_rows")


def validate_fit_influence_diagnostic(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    for field in ("requested_policy", "effective_policy"):
        require(value.get(field) in FIT_INFLUENCE_POLICIES, f"{label}.{field} invalid")
    require(value.get("mechanism") in FIT_INFLUENCE_MECHANISMS, f"{label}.mechanism invalid")
    require(isinstance(value.get("fallback_used", False), bool), f"{label}.fallback_used boolean")
    row_weight_count = value.get("row_weight_count", 0)
    require(
        isinstance(row_weight_count, int) and row_weight_count >= 0,
        f"{label}.row_weight_count invalid",
    )
    warnings = value.get("warnings", [])
    require(isinstance(warnings, list), f"{label}.warnings must be an array")
    for index, warning in enumerate(warnings):
        require_non_empty_string(warning, f"{label}.warnings[{index}]")


def validate_node_task(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} NodeTask must be an object")
    require_identifier(value.get("run_id"), f"{label}.run_id")
    node_plan = value.get("node_plan")
    require(isinstance(node_plan, dict), f"{label}.node_plan must be an object")
    require_identifier(node_plan.get("node_id"), f"{label}.node_plan.node_id")
    require_identifier(node_plan.get("controller_id"), f"{label}.node_plan.controller_id")
    require_non_empty_string(
        node_plan.get("controller_version"),
        f"{label}.node_plan.controller_version",
    )
    require_non_empty_string(
        node_plan.get("params_fingerprint"),
        f"{label}.node_plan.params_fingerprint",
    )
    require(
        value.get("phase") in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
        f"{label}.phase invalid",
    )
    variant_id = value.get("variant_id")
    if variant_id is not None:
        require_identifier(variant_id, f"{label}.variant_id")
    variant = value.get("variant")
    if variant is not None:
        require(isinstance(variant, dict), f"{label}.variant must be an object")
        require(variant.get("variant_id") == variant_id, f"{label}.variant_id mismatch")
        require_non_empty_string(variant.get("fingerprint"), f"{label}.variant.fingerprint")
        seed = variant.get("seed")
        if seed is not None:
            require(isinstance(seed, int) and seed >= 0, f"{label}.variant.seed invalid")
    fold_id = value.get("fold_id")
    if fold_id is not None:
        require_identifier(fold_id, f"{label}.fold_id")
    seed = value.get("seed")
    if seed is not None:
        require(isinstance(seed, int) and seed >= 0, f"{label}.seed invalid")
    for map_name in ("input_handles", "data_views", "prediction_inputs", "artifact_inputs"):
        mapping = value.get(map_name, {})
        require(isinstance(mapping, dict), f"{label}.{map_name} must be an object")
    for key, handle in value.get("input_handles", {}).items():
        require_non_empty_string(key, f"{label}.input_handles key")
        validate_handle_ref(handle, f"{label}.input_handles[{key}]")
    fit_influence = value.get("fit_influence")
    if fit_influence is not None:
        validate_fit_influence_task(fit_influence, f"{label}.fit_influence")


def validate_node_result(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} NodeResult must be an object")
    require_identifier(value.get("node_id"), f"{label}.node_id")
    outputs = value.get("outputs", {})
    require(isinstance(outputs, dict), f"{label}.outputs must be an object")
    for port_name, handle in outputs.items():
        require_non_empty_string(port_name, f"{label}.outputs key")
        validate_handle_ref(handle, f"{label}.outputs[{port_name}]")
    for list_name in ("predictions", "shape_deltas", "artifacts"):
        require(isinstance(value.get(list_name, []), list), f"{label}.{list_name} must be an array")
    artifact_handles = value.get("artifact_handles", {})
    require(isinstance(artifact_handles, dict), f"{label}.artifact_handles must be an object")
    for artifact_id, handle in artifact_handles.items():
        require_identifier(artifact_id, f"{label}.artifact_handles key")
        validate_handle_ref(handle, f"{label}.artifact_handles[{artifact_id}]")
    diagnostics = value.get("fit_influence_diagnostics", [])
    require(isinstance(diagnostics, list), f"{label}.fit_influence_diagnostics must be an array")
    for index, diagnostic in enumerate(diagnostics):
        validate_fit_influence_diagnostic(
            diagnostic,
            f"{label}.fit_influence_diagnostics[{index}]",
        )
    lineage = value.get("lineage")
    require(isinstance(lineage, dict), f"{label}.lineage must be an object")
    for field in ("record_id", "run_id", "node_id", "controller_id"):
        require_identifier(lineage.get(field), f"{label}.lineage.{field}")
    require(
        lineage.get("phase") in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
        f"{label}.lineage.phase invalid",
    )
    require_non_empty_string(
        lineage.get("controller_version"),
        f"{label}.lineage.controller_version",
    )
    require_non_empty_string(
        lineage.get("params_fingerprint"),
        f"{label}.lineage.params_fingerprint",
    )
    for field in ("variant_id", "fold_id"):
        field_value = lineage.get(field)
        if field_value is not None:
            require_identifier(field_value, f"{label}.lineage.{field}")
    for list_name in ("branch_path", "input_lineage", "artifact_refs", "unsafe_flags"):
        require(
            isinstance(lineage.get(list_name, []), list),
            f"{label}.lineage.{list_name} must be an array",
        )
    metrics = lineage.get("metrics", {})
    require(isinstance(metrics, dict), f"{label}.lineage.metrics must be an object")
    for metric_name, metric_value in metrics.items():
        require_non_empty_string(metric_name, f"{label}.lineage.metrics key")
        require(isinstance(metric_value, (int, float)), f"{label}.lineage.metrics value numeric")


def validate_node_task_result_pair(task: Any, result: Any, label: str) -> None:
    validate_node_task(task, f"{label}.task")
    validate_node_result(result, f"{label}.result")
    node_plan = task["node_plan"]
    lineage = result["lineage"]
    require(result.get("node_id") == node_plan.get("node_id"), f"{label} result node mismatch")
    require(lineage.get("node_id") == node_plan.get("node_id"), f"{label} lineage node mismatch")
    require(lineage.get("run_id") == task.get("run_id"), f"{label} lineage run mismatch")
    require(lineage.get("phase") == task.get("phase"), f"{label} lineage phase mismatch")
    require(
        lineage.get("controller_id") == node_plan.get("controller_id"),
        f"{label} lineage controller mismatch",
    )
    require(
        lineage.get("controller_version") == node_plan.get("controller_version"),
        f"{label} lineage controller_version mismatch",
    )
    require(lineage.get("variant_id") == task.get("variant_id"), f"{label} lineage variant mismatch")
    require(lineage.get("fold_id") == task.get("fold_id"), f"{label} lineage fold mismatch")
    require(lineage.get("seed") == task.get("seed"), f"{label} lineage seed mismatch")
    require(
        lineage.get("params_fingerprint") == node_plan.get("params_fingerprint"),
        f"{label} lineage params fingerprint mismatch",
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


def validate_process_adapter_frame(
    value: Any,
    label: str,
    task_fixture: Any,
    result_fixture: Any,
) -> None:
    require(isinstance(value, dict), f"{label} process-adapter frame must be an object")
    require(value.get("schema_version") == 1, f"{label}.schema_version must be 1")
    frame_type = value.get("type")
    require(
        frame_type in {"init", "task", "close", "ack", "result", "error"},
        f"{label}.type is not a supported process-adapter frame",
    )

    if frame_type == "init":
        require_no_unknown_keys(
            value,
            {"type", "schema_version", "controller_id", "worker_index", "worker_count"},
            label,
        )
        require_identifier(value.get("controller_id"), f"{label}.controller_id")
        worker_index = value.get("worker_index")
        worker_count = value.get("worker_count")
        require(
            isinstance(worker_index, int) and worker_index >= 0,
            f"{label}.worker_index must be a non-negative integer",
        )
        require(
            isinstance(worker_count, int) and worker_count >= 1,
            f"{label}.worker_count must be a positive integer",
        )
        require(
            worker_index < worker_count,
            f"{label}.worker_index must be lower than worker_count",
        )
        return

    if frame_type == "task":
        require_no_unknown_keys(value, {"type", "schema_version", "task"}, label)
        task = value.get("task")
        validate_node_task(task, f"{label}.task")
        require(task == task_fixture, f"{label}.task must match the canonical NodeTask fixture")
        return

    if frame_type == "close":
        require_no_unknown_keys(value, {"type", "schema_version"}, label)
        return

    if frame_type == "ack":
        require_no_unknown_keys(value, {"type", "schema_version", "status"}, label)
        require(
            value.get("status") in {"initialized", "closed"},
            f"{label}.status must be initialized or closed",
        )
        return

    if frame_type == "result":
        require_no_unknown_keys(value, {"type", "schema_version", "result"}, label)
        result = value.get("result")
        validate_node_result(result, f"{label}.result")
        require(
            result == result_fixture,
            f"{label}.result must match the canonical NodeResult fixture",
        )
        return

    require_no_unknown_keys(value, {"type", "schema_version", "error"}, label)
    error = value.get("error")
    require(isinstance(error, dict), f"{label}.error must be an object")
    require_no_unknown_keys(error, {"code", "message", "retryable"}, f"{label}.error")
    require_identifier(error.get("code"), f"{label}.error.code")
    require_non_empty_string(error.get("message"), f"{label}.error.message")
    if "retryable" in error:
        require(isinstance(error["retryable"], bool), f"{label}.error.retryable must be boolean")


def validate_process_adapter_frame_fixtures(
    fixtures: list[tuple[Path, Any]],
    task_fixture: Any,
    result_fixture: Any,
    label: str,
) -> None:
    expected_types = ["init", "task", "result", "ack", "error", "close"]
    observed_types: list[str] = []
    for path, value in fixtures:
        validate_process_adapter_frame(
            value,
            f"{label} {path.name}",
            task_fixture,
            result_fixture,
        )
        observed_types.append(value["type"])
    require(
        observed_types == expected_types,
        f"{label} process-adapter frame fixture order/type set mismatch",
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
    require(
        "#define DAG_ML_DATA_BORROWED_TENSOR_VIEW_ABI_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_DATA_BORROWED_TENSOR_VIEW_ABI_VERSION=1",
    )
    require(
        "#define DAG_ML_DATA_OWNED_TENSOR_ABI_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_DATA_OWNED_TENSOR_ABI_VERSION=1",
    )
    for symbol in (
        "DagMlDataTensorDType",
        "DagMlDataBorrowedTensorView",
        "DagMlDataOwnedTensor",
        "dagmldata_inmemory_provider_new_with_tensor_views",
        "dagmldata_inmemory_provider_nd_tensor_manifest_json",
        "dagmldata_inmemory_provider_data_nd_tensor_manifest_json",
        "dagmldata_inmemory_provider_nd_tensor_export_json",
        "dagmldata_nd_tensor_free",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_prediction_cache_tensor_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION=1",
    )
    require(
        "#define DAG_ML_PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_VERSION=1",
    )
    for symbol in (
        "DagMlF64Tensor",
        "DagMlF64ColumnarTensor",
        "dagml_f64_tensor_free",
        "dagml_f64_columnar_tensor_free",
        "dagml_prediction_cache_payload_f64_tensor_json",
        "dagml_prediction_cache_payload_f64_columnar_tensor_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_controller_result_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_CONTROLLER_MANIFEST_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_CONTROLLER_MANIFEST_SCHEMA_VERSION=1",
    )
    for macro in (
        "#define DAG_ML_NODE_TASK_SCHEMA_VERSION 1u",
        "#define DAG_ML_NODE_RESULT_SCHEMA_VERSION 1u",
    ):
        require(macro in header, f"{label} header must declare `{macro}`")
    for symbol in (
        "dagml_controller_manifest_contract_json",
        "dagml_node_result_validate_for_task_json",
        "dagml_controller_manifest_validate_json",
        "dagml_controller_manifest_list_validate_json",
        "dagml_node_task_contract_json",
        "dagml_node_result_contract_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_process_adapter_header(header: str, label: str) -> None:
    for macro in (
        "#define DAG_ML_PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION 1u",
        "#define DAG_ML_PROCESS_ADAPTER_FRAME_SCHEMA_VERSION 1u",
    ):
        require(macro in header, f"{label} header must declare `{macro}`")
    for symbol in (
        "dagml_process_adapter_description_contract_json",
        "dagml_process_adapter_frame_contract_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_aggregation_controller_header(header: str, label: str) -> None:
    for macro in (
        "#define DAG_ML_AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION 1u",
        "#define DAG_ML_AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION 1u",
    ):
        require(macro in header, f"{label} header must declare `{macro}`")
    for symbol in (
        "dagml_aggregation_controller_task_contract_json",
        "dagml_aggregation_controller_result_contract_json",
        "dagml_aggregation_controller_task_validate_json",
        "dagml_aggregation_controller_result_validate_for_task_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_graph_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_GRAPH_SPEC_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_GRAPH_SPEC_SCHEMA_VERSION=1",
    )
    for symbol in ("dagml_graph_spec_contract_json", "dagml_graph_validate_json"):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_pipeline_dsl_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_PIPELINE_DSL_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_PIPELINE_DSL_SCHEMA_VERSION=1",
    )
    for symbol in (
        "dagml_pipeline_dsl_contract_json",
        "dagml_pipeline_dsl_validate_json",
        "dagml_pipeline_dsl_compile_json",
        "dagml_pipeline_dsl_compile_artifact_json",
        "dagml_pipeline_dsl_execution_plan_build_json",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_campaign_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_CAMPAIGN_SPEC_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_CAMPAIGN_SPEC_SCHEMA_VERSION=1",
    )
    for symbol in ("dagml_campaign_spec_contract_json", "dagml_campaign_validate_json"):
        require(symbol in header, f"{label} header must expose `{symbol}`")


def validate_dag_ml_execution_plan_header(header: str, label: str) -> None:
    require(
        "#define DAG_ML_EXECUTION_PLAN_SCHEMA_VERSION 1u" in header,
        f"{label} header must declare DAG_ML_EXECUTION_PLAN_SCHEMA_VERSION=1",
    )
    for symbol in (
        "dagml_execution_plan_contract_json",
        "dagml_execution_plan_build_json",
        "dagml_execution_plan_schedule_json",
        "dagml_execution_plan_validate_json",
    ):
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


def validate_dag_ml_selection_header(header: str, label: str) -> None:
    for macro in (
        "#define DAG_ML_SELECTION_POLICY_SCHEMA_VERSION 1u",
        "#define DAG_ML_SELECTION_DECISION_SCHEMA_VERSION 1u",
    ):
        require(macro in header, f"{label} header must declare `{macro}`")
    for symbol in (
        "dagml_selection_policy_contract_json",
        "dagml_selection_policy_validate_json",
        "dagml_selection_decision_contract_json",
        "dagml_selection_decision_validate_json",
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
    branch_view_schema: Any,
    fitted_adapter_schema: Any,
    parity_oracle: Any,
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
    validate_digest_record(
        contracts.get("coordinator_branch_view.v1"),
        canonical_json_sha256(normalize_schema(branch_view_schema)),
        "json_schema",
        1,
        f"{label} coordinator branch view contract",
    )
    validate_digest_record(
        contracts.get("fitted_adapter_ref.v1"),
        canonical_json_sha256(normalize_schema(fitted_adapter_schema)),
        "json_schema",
        1,
        f"{label} fitted adapter ref contract",
    )
    validate_digest_record(
        contracts.get("parity_oracle.v1"),
        canonical_json_sha256(parity_oracle),
        "parity_oracle_manifest",
        1,
        f"{label} parity oracle contract",
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
    if "DagMlDataBorrowedTensorView" in header:
        require(
            c_abi.get("data_borrowed_tensor_view_abi_version") == 1,
            f"{label} borrowed tensor view ABI version must be 1",
        )
    if "DagMlDataOwnedTensor" in header:
        require(
            c_abi.get("data_owned_tensor_abi_version") == 1,
            f"{label} owned tensor ABI version must be 1",
        )

    cross_repo = pack.get("cross_repo_conformance")
    require(isinstance(cross_repo, dict), f"{label} cross_repo_conformance must be an object")
    required_tests = cross_repo.get("required_when_sibling_checkout_present")
    require(isinstance(required_tests, list), f"{label} cross-repo tests must be a list")
    for test_id in (
        "contracts.schema_and_fixture_equivalence",
        "headers.include_order",
        "provider.f64_predict_replay",
        "fold_set.fingerprint_parity",
    ):
        require(test_id in required_tests, f"{label} conformance pack must require `{test_id}`")


def validate_parity_oracle_manifest(
    oracle: Any,
    roots_by_repo: dict[str, Path],
    label: str,
) -> None:
    require(isinstance(oracle, dict), f"{label} parity oracle must be a JSON object")
    require(oracle.get("schema_version") == 1, f"{label} parity oracle schema_version must be 1")
    require(oracle.get("oracle_id") == PARITY_ORACLE_ID, f"{label} parity oracle id mismatch")
    require(oracle.get("status") == "producer_handoff", f"{label} parity oracle status mismatch")

    consumer_ledger = oracle.get("consumer_ledger")
    require(isinstance(consumer_ledger, dict), f"{label} parity oracle ledger must be an object")
    require(
        consumer_ledger.get("repo") == "nirs4all",
        f"{label} parity oracle ledger must point to nirs4all",
    )
    require(
        consumer_ledger.get("path") == "docs/compatibility.md",
        f"{label} parity oracle ledger path mismatch",
    )
    require(
        consumer_ledger.get("required_before_bridge") is True,
        f"{label} parity oracle ledger must be required before bridge wiring",
    )

    shared = oracle.get("shared")
    require(isinstance(shared, dict), f"{label} parity oracle shared block must be an object")
    require(
        shared.get("fold_set_fixture_fingerprint") == SHARED_FOLD_SET_FINGERPRINT,
        f"{label} parity oracle shared fold-set fingerprint drifted",
    )

    tolerance_profiles = oracle.get("tolerance_profiles")
    require(
        isinstance(tolerance_profiles, list) and tolerance_profiles,
        f"{label} parity oracle must declare tolerance profiles",
    )
    profile_ids: set[str] = set()
    for index, profile in enumerate(tolerance_profiles):
        profile_label = f"{label} parity oracle tolerance_profiles[{index}]"
        require(isinstance(profile, dict), f"{profile_label} must be an object")
        require_identifier(profile.get("profile_id"), f"{profile_label}.profile_id")
        require_non_empty_string(profile.get("metric"), f"{profile_label}.metric")
        require_non_empty_string(profile.get("owner"), f"{profile_label}.owner")
        require(
            isinstance(profile.get("absolute_tolerance"), (int, float)),
            f"{profile_label}.absolute_tolerance must be numeric",
        )
        require(
            isinstance(profile.get("relative_tolerance"), (int, float)),
            f"{profile_label}.relative_tolerance must be numeric",
        )
        require(
            profile["profile_id"] not in profile_ids,
            f"{profile_label}.profile_id is duplicated",
        )
        profile_ids.add(profile["profile_id"])

    required_case_ids = oracle.get("required_case_ids")
    require(
        isinstance(required_case_ids, list),
        f"{label} parity oracle required_case_ids must be a list",
    )
    require(
        set(required_case_ids) == REQUIRED_PARITY_CASE_IDS,
        f"{label} parity oracle required_case_ids changed",
    )

    cases = oracle.get("cases")
    require(isinstance(cases, list) and cases, f"{label} parity oracle cases must be non-empty")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        case_label = f"{label} parity oracle cases[{index}]"
        require(isinstance(case, dict), f"{case_label} must be an object")
        require_identifier(case.get("case_id"), f"{case_label}.case_id")
        require(case["case_id"] not in case_ids, f"{case_label}.case_id is duplicated")
        case_ids.add(case["case_id"])
        for field in ("ledger_topics", "fixtures", "gates", "invariants"):
            require(
                isinstance(case.get(field), list) and case[field],
                f"{case_label}.{field} must be a non-empty list",
            )
        for topic_index, topic in enumerate(case["ledger_topics"]):
            require_non_empty_string(topic, f"{case_label}.ledger_topics[{topic_index}]")
        for invariant_index, invariant in enumerate(case["invariants"]):
            require_non_empty_string(invariant, f"{case_label}.invariants[{invariant_index}]")
        for fixture_index, fixture in enumerate(case["fixtures"]):
            fixture_label = f"{case_label}.fixtures[{fixture_index}]"
            require(isinstance(fixture, dict), f"{fixture_label} must be an object")
            repo = fixture.get("repo")
            require(repo in {"dag-ml", "dag-ml-data"}, f"{fixture_label}.repo is invalid")
            require_non_empty_string(fixture.get("path"), f"{fixture_label}.path")
            require_non_empty_string(fixture.get("kind"), f"{fixture_label}.kind")
            root = roots_by_repo.get(repo)
            if root is not None:
                require((root / fixture["path"]).is_file(), f"{fixture_label} path is missing")
        for gate_index, gate in enumerate(case["gates"]):
            gate_label = f"{case_label}.gates[{gate_index}]"
            require(isinstance(gate, dict), f"{gate_label} must be an object")
            require(gate.get("repo") in {"dag-ml", "dag-ml-data"}, f"{gate_label}.repo is invalid")
            require_non_empty_string(gate.get("command"), f"{gate_label}.command")
            require_non_empty_string(gate.get("proves"), f"{gate_label}.proves")
    require(case_ids == REQUIRED_PARITY_CASE_IDS, f"{label} parity oracle case set changed")


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
        local_branch_view_schema = load_json(ROOT / BRANCH_VIEW_SCHEMA_REL)
        local_fitted_adapter_schema = load_json(ROOT / FITTED_ADAPTER_SCHEMA_REL)
        local_graph_spec_schema = load_json(ROOT / GRAPH_SPEC_SCHEMA_REL)
        local_pipeline_dsl_schema = load_json(ROOT / PIPELINE_DSL_SCHEMA_REL)
        local_campaign_spec_schema = load_json(ROOT / CAMPAIGN_SPEC_SCHEMA_REL)
        local_execution_plan_schema = load_json(ROOT / EXECUTION_PLAN_SCHEMA_REL)
        local_model_input_spec_schema = load_json(ROOT / MODEL_INPUT_SPEC_SCHEMA_REL)
        local_data_plan_schema = load_json(ROOT / DATA_PLAN_SCHEMA_REL)
        local_controller_manifest_schema = load_json(ROOT / CONTROLLER_MANIFEST_SCHEMA_REL)
        local_selection_policy_schema = load_json(ROOT / SELECTION_POLICY_SCHEMA_REL)
        local_selection_decision_schema = load_json(ROOT / SELECTION_DECISION_SCHEMA_REL)
        local_pack = load_json(ROOT / CONFORMANCE_PACK_REL)
        local_parity_oracle = load_json(ROOT / PARITY_ORACLE_REL)
        local_openlineage_facets_schema = load_json(ROOT / OPENLINEAGE_FACETS_SCHEMA_REL)
        local_prediction_cache_tensor_metadata_schema = load_json(
            ROOT / PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_REL
        )
        local_prediction_cache_columnar_tensor_metadata_schema = load_json(
            ROOT / PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_REL
        )
        local_aggregation_controller_task_schema = load_json(
            ROOT / AGGREGATION_CONTROLLER_TASK_SCHEMA_REL
        )
        local_aggregation_controller_result_schema = load_json(
            ROOT / AGGREGATION_CONTROLLER_RESULT_SCHEMA_REL
        )
        local_data_output_provenance_schema = load_json(
            ROOT / DATA_OUTPUT_PROVENANCE_SCHEMA_REL
        )
        local_node_task_schema = load_json(ROOT / NODE_TASK_SCHEMA_REL)
        local_node_result_schema = load_json(ROOT / NODE_RESULT_SCHEMA_REL)
        local_process_adapter_description_schema = load_json(
            ROOT / PROCESS_ADAPTER_DESCRIPTION_SCHEMA_REL
        )
        local_process_adapter_frame_schema = load_json(ROOT / PROCESS_ADAPTER_FRAME_SCHEMA_REL)
        local_research_provenance_profile = load_json(ROOT / RESEARCH_PROVENANCE_PROFILE_REL)
        local_fixture = load_json(ROOT / LOCAL_FIXTURE_REL)
        local_multisource_fixture = load_json(ROOT / LOCAL_MULTISOURCE_FIXTURE_REL)
        local_feature_fusion_fixture = load_json(ROOT / LOCAL_FEATURE_FUSION_FIXTURE_REL)
        local_fold_set_fixture = load_json(ROOT / SHARED_FOLD_SET_FIXTURE_REL)
        local_graph_spec_fixture = load_json(ROOT / LOCAL_GRAPH_SPEC_FIXTURE_REL)
        local_pipeline_dsl_fixture = load_json(ROOT / LOCAL_PIPELINE_DSL_FIXTURE_REL)
        local_campaign_spec_fixture = load_json(ROOT / LOCAL_CAMPAIGN_SPEC_FIXTURE_REL)
        local_execution_plan_fixture = load_json(ROOT / LOCAL_EXECUTION_PLAN_FIXTURE_REL)
        local_model_input_spec_fixture = load_json(ROOT / LOCAL_MODEL_INPUT_SPEC_FIXTURE_REL)
        local_data_plan_fixture = load_json(ROOT / LOCAL_DATA_PLAN_FIXTURE_REL)
        local_controller_manifest_fixture = load_json(
            ROOT / LOCAL_CONTROLLER_MANIFEST_FIXTURE_REL
        )
        local_controller_manifest_list_fixture = load_json(
            ROOT / LOCAL_CONTROLLER_MANIFEST_LIST_FIXTURE_REL
        )
        local_selection_policy_fixture = load_json(ROOT / LOCAL_SELECTION_POLICY_FIXTURE_REL)
        local_selection_decision_fixture = load_json(ROOT / LOCAL_SELECTION_DECISION_FIXTURE_REL)
        local_data_output_provenance_fixture = load_json(
            ROOT / LOCAL_DATA_OUTPUT_PROVENANCE_FIXTURE_REL
        )
        local_node_task_fixture = load_json(ROOT / LOCAL_NODE_TASK_FIXTURE_REL)
        local_node_result_fixture = load_json(ROOT / LOCAL_NODE_RESULT_FIXTURE_REL)
        local_process_adapter_description_fixture = load_json(
            ROOT / LOCAL_PROCESS_ADAPTER_DESCRIPTION_FIXTURE_REL
        )
        local_process_adapter_frame_fixtures = [
            (fixture_rel, load_json(ROOT / fixture_rel))
            for fixture_rel in LOCAL_PROCESS_ADAPTER_FRAME_FIXTURE_RELS
        ]
        local_header = load_text(ROOT / LOCAL_C_HEADER_REL)
        validate_schema_artifact(local_schema, LOCAL_SCHEMA_ID, "dag-ml")
        validate_feature_fusion_schema_artifact(
            local_feature_fusion_schema,
            LOCAL_FEATURE_FUSION_SCHEMA_ID,
            "dag-ml",
        )
        validate_branch_view_schema_artifact(
            local_branch_view_schema,
            LOCAL_BRANCH_VIEW_SCHEMA_ID,
            "dag-ml",
        )
        validate_fitted_adapter_ref_schema_artifact(
            local_fitted_adapter_schema,
            LOCAL_FITTED_ADAPTER_SCHEMA_ID,
            "dag-ml",
        )
        validate_graph_spec_schema(local_graph_spec_schema, "dag-ml")
        validate_pipeline_dsl_schema(local_pipeline_dsl_schema, "dag-ml")
        validate_campaign_spec_schema(local_campaign_spec_schema, "dag-ml")
        validate_execution_plan_schema(local_execution_plan_schema, "dag-ml")
        validate_model_input_spec_schema(local_model_input_spec_schema, "dag-ml")
        validate_data_plan_schema(local_data_plan_schema, "dag-ml")
        validate_controller_manifest_schema(local_controller_manifest_schema, "dag-ml")
        validate_selection_policy_schema(local_selection_policy_schema, "dag-ml")
        validate_selection_decision_schema(local_selection_decision_schema, "dag-ml")
        validate_openlineage_facets_schema(local_openlineage_facets_schema, "dag-ml")
        validate_prediction_cache_tensor_metadata_schema(
            local_prediction_cache_tensor_metadata_schema,
            "dag-ml",
        )
        validate_prediction_cache_columnar_tensor_metadata_schema(
            local_prediction_cache_columnar_tensor_metadata_schema,
            "dag-ml",
        )
        validate_aggregation_controller_task_schema(
            local_aggregation_controller_task_schema,
            "dag-ml",
        )
        validate_aggregation_controller_result_schema(
            local_aggregation_controller_result_schema,
            "dag-ml",
        )
        validate_data_output_provenance_schema(
            local_data_output_provenance_schema,
            "dag-ml",
        )
        validate_node_task_schema(local_node_task_schema, "dag-ml")
        validate_node_result_schema(local_node_result_schema, "dag-ml")
        validate_process_adapter_description_schema(
            local_process_adapter_description_schema,
            "dag-ml",
        )
        validate_process_adapter_frame_schema(local_process_adapter_frame_schema, "dag-ml")
        validate_envelope(local_fixture, "dag-ml")
        validate_envelope(local_multisource_fixture, "dag-ml multisource")
        validate_feature_fusion_selector(local_feature_fusion_fixture, "dag-ml")
        validate_fold_set_fixture(local_fold_set_fixture, "dag-ml shared")
        require(
            canonical_fold_set_fingerprint(local_fold_set_fixture)
            == SHARED_FOLD_SET_FINGERPRINT,
            "dag-ml shared fold set fingerprint drifted",
        )
        validate_graph_spec(local_graph_spec_fixture, "dag-ml")
        validate_pipeline_dsl_fixture(local_pipeline_dsl_fixture, "dag-ml")
        validate_campaign_spec(local_campaign_spec_fixture, "dag-ml")
        validate_execution_plan(local_execution_plan_fixture, "dag-ml")
        validate_model_input_spec(local_model_input_spec_fixture, "dag-ml")
        validate_data_plan(local_data_plan_fixture, "dag-ml")
        validate_controller_manifest(local_controller_manifest_fixture, "dag-ml")
        validate_controller_manifest_list(
            local_controller_manifest_list_fixture,
            "dag-ml controller manifest list",
        )
        validate_selection_policy(local_selection_policy_fixture, "dag-ml")
        validate_selection_decision(local_selection_decision_fixture, "dag-ml")
        validate_data_output_provenance(local_data_output_provenance_fixture, "dag-ml")
        validate_node_task_result_pair(
            local_node_task_fixture,
            local_node_result_fixture,
            "dag-ml node task/result",
        )
        validate_process_adapter_description(
            local_process_adapter_description_fixture,
            "dag-ml",
        )
        validate_process_adapter_frame_fixtures(
            local_process_adapter_frame_fixtures,
            local_node_task_fixture,
            local_node_result_fixture,
            "dag-ml",
        )
        validate_data_provider_header(local_header, "dag-ml")
        validate_dag_ml_prediction_cache_tensor_header(local_header, "dag-ml")
        validate_dag_ml_controller_result_header(local_header, "dag-ml")
        validate_dag_ml_process_adapter_header(local_header, "dag-ml")
        validate_dag_ml_aggregation_controller_header(local_header, "dag-ml")
        validate_dag_ml_graph_header(local_header, "dag-ml")
        validate_dag_ml_pipeline_dsl_header(local_header, "dag-ml")
        validate_dag_ml_campaign_header(local_header, "dag-ml")
        validate_dag_ml_execution_plan_header(local_header, "dag-ml")
        validate_dag_ml_data_shape_header(local_header, "dag-ml")
        validate_dag_ml_data_output_provenance_header(local_header, "dag-ml")
        validate_dag_ml_selection_header(local_header, "dag-ml")
        validate_parity_oracle_manifest(
            local_parity_oracle,
            {"dag-ml": ROOT},
            "dag-ml",
        )
        validate_conformance_pack(
            local_pack,
            local_schema,
            local_feature_fusion_schema,
            local_branch_view_schema,
            local_fitted_adapter_schema,
            local_parity_oracle,
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
        sibling_parity_oracle = load_json(sibling / PARITY_ORACLE_REL)
        sibling_fixture = load_json(sibling / SIBLING_FIXTURE_REL)
        sibling_feature_fusion_fixture = load_json(
            sibling / SIBLING_FEATURE_FUSION_FIXTURE_REL
        )
        sibling_fold_set_fixture = load_json(sibling / SHARED_FOLD_SET_FIXTURE_REL)
        sibling_header = load_text(sibling / SIBLING_C_HEADER_REL)
        validate_schema_artifact(sibling_schema, SIBLING_SCHEMA_ID, "dag-ml-data")
        validate_feature_fusion_schema_artifact(
            sibling_feature_fusion_schema,
            SIBLING_FEATURE_FUSION_SCHEMA_ID,
            "dag-ml-data",
        )
        validate_envelope(sibling_fixture, "dag-ml-data")
        validate_feature_fusion_selector(sibling_feature_fusion_fixture, "dag-ml-data")
        validate_fold_set_fixture(sibling_fold_set_fixture, "dag-ml-data shared")
        require(
            canonical_fold_set_fingerprint(sibling_fold_set_fixture)
            == SHARED_FOLD_SET_FINGERPRINT,
            "dag-ml-data shared fold set fingerprint drifted",
        )
        validate_data_provider_header(sibling_header, "dag-ml-data")
        validate_dag_ml_data_tensor_header(sibling_header, "dag-ml-data")
        validate_parity_oracle_manifest(
            sibling_parity_oracle,
            {"dag-ml": ROOT, "dag-ml-data": sibling},
            "dag-ml-data",
        )
        validate_conformance_pack(
            sibling_pack,
            sibling_schema,
            sibling_feature_fusion_schema,
            local_branch_view_schema,
            local_fitted_adapter_schema,
            sibling_parity_oracle,
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
        require(
            canonical_fold_set_fingerprint(local_fold_set_fixture)
            == canonical_fold_set_fingerprint(sibling_fold_set_fixture),
            "shared fold set canonical fingerprints diverge",
        )
        require(local_pack == sibling_pack, "shared conformance packs diverge")
        require(local_parity_oracle == sibling_parity_oracle, "parity oracle manifests diverge")
        print(f"validated dag-ml contract against dag-ml-data at {sibling}")
        return 0
    except ContractError as exc:
        print(f"contract validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
