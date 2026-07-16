#!/usr/bin/env python3
"""Validate local and shared contract artifacts with dag-ml-data.

Draft 2020-12 schemas are checked and resolved through an offline registry built
from ``docs/contracts``.  The remaining validators enforce cross-object and
runtime invariants that JSON Schema cannot express, then compare shared artifacts
with a sibling dag-ml-data checkout when one is available.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import struct
import sys
from decimal import Decimal
from json.encoder import encode_basestring
from pathlib import Path
from typing import Any

import unicodedata2 as unicodedata
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TCV1_UNICODE_VERSION = "17.0.0"
if unicodedata.unidata_version != TCV1_UNICODE_VERSION:
    raise RuntimeError(
        "TCV1 requires Unicode "
        f"{TCV1_UNICODE_VERSION}, got {unicodedata.unidata_version}"
    )

from parity.schema_dependencies import (  # noqa: E402
    SchemaDependencyError,
    missing_schema_dependencies,
)

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
REPRESENTATION_REGISTRY_REL = Path("docs/contracts/representation_registry.v1.json")
SELECTION_POLICY_SCHEMA_REL = Path("docs/contracts/selection_policy.schema.json")
SELECTION_DECISION_SCHEMA_REL = Path("docs/contracts/selection_decision.schema.json")
CONFORMANCE_PACK_REL = Path("docs/contracts/conformance_pack.v1.json")
PARITY_ORACLE_REL = Path("docs/contracts/parity_oracle.v1.json")
OPENLINEAGE_FACETS_SCHEMA_REL = Path(
    "docs/contracts/openlineage_dagml_facets.schema.json"
)
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
SCORE_SET_SCHEMA_REL = Path("docs/contracts/score_set.schema.json")
SCORE_SET_FIXTURE_REL = Path("examples/fixtures/score_set.json")
CHAIN_EFFECT_ANALYSIS_SCHEMA_REL = Path(
    "docs/contracts/chain_effect_analysis.schema.json"
)
CHAIN_EFFECT_ANALYSIS_FIXTURE_REL = Path(
    "examples/fixtures/chain_effect_analysis.json"
)
CONFORMAL_CALIBRATION_SCHEMA_REL = Path(
    "docs/contracts/conformal_calibration.schema.json"
)
CONFORMAL_CALIBRATION_FIXTURE_REL = Path(
    "examples/fixtures/conformal/split_absolute_residual_physical_sample.v1.json"
)
PARAMETER_PATCH_SCHEMA_REL = Path("docs/contracts/parameter_patch.schema.json")
OUTPUT_BINDING_SCHEMA_REL = Path("docs/contracts/output_binding.schema.json")
TRAINING_INFLUENCE_SCHEMA_REL = Path(
    "docs/contracts/training_influence_manifest.schema.json"
)
EXECUTION_BUNDLE_SCHEMA_REL = Path("docs/contracts/execution_bundle.schema.json")
PREDICTION_CACHE_PAYLOAD_SET_SCHEMA_REL = Path(
    "docs/contracts/prediction_cache_payload_set.schema.json"
)
TRAINING_OUTCOME_SCHEMA_REL = Path("docs/contracts/training_outcome.schema.json")
REPLAY_OUTCOME_SCHEMA_REL = Path("docs/contracts/replay_outcome.schema.json")
TRAINING_REQUEST_SCHEMA_REL = Path("docs/contracts/training_request.schema.json")
CACHE_NAMESPACE_SCHEMA_REL = Path("docs/contracts/cache_namespace.schema.json")
PARAMETER_PROJECTION_SCHEMA_REL = Path(
    "docs/contracts/parameter_projection.schema.json"
)
PORTABLE_PREDICTOR_PACKAGE_SCHEMA_REL = Path(
    "docs/contracts/portable_predictor_package.schema.json"
)
W10_TRAINING_POSITIVE_FIXTURE_RELS = (
    Path("examples/fixtures/training/training_request_refit.v1.json"),
    Path("examples/fixtures/training/training_request_no_refit.v1.json"),
    Path("examples/fixtures/training/training_request_active_influence.v1.json"),
    Path("examples/fixtures/training/training_request_package_refit.v1.json"),
    Path("examples/fixtures/training/cache_namespace_fit_cv.v1.json"),
    Path("examples/fixtures/training/parameter_projection_empty.v1.json"),
    Path("examples/fixtures/training/portable_predictor_package.v1.json"),
    Path("examples/fixtures/training/training_outcome_refit.v1.json"),
)
W10_TRAINING_NEGATIVE_FIXTURE_REL = Path(
    "examples/fixtures/training/negative_cases.v1.json"
)
W10_TRAINING_PACK_REL = Path(
    "docs/contracts/training_contract_conformance_pack.v1.json"
)
PARAMETER_PATCH_FIXTURE_REL = Path(
    "examples/fixtures/estimator/parameter_patch_operator_alpha.v1.json"
)
OUTPUT_BINDING_FIXTURE_REL = Path(
    "examples/fixtures/estimator/output_binding_regression_final_refit.v1.json"
)
TRAINING_OUTCOME_REFIT_FIXTURE_REL = Path(
    "examples/fixtures/estimator/training_outcome_refit.v1.json"
)
TRAINING_OUTCOME_NO_REFIT_FIXTURE_REL = Path(
    "examples/fixtures/estimator/training_outcome_no_refit.v1.json"
)
REPLAY_OUTCOME_FIXTURE_RELS = [
    Path("examples/fixtures/estimator/replay_outcome_predict.v1.json"),
    Path("examples/fixtures/estimator/replay_outcome_class_probability.v1.json"),
    Path("examples/fixtures/estimator/replay_outcome_explain.v1.json"),
]
CONFORMAL_ROBUSTNESS_CONTRACTS = (
    (
        Path("docs/contracts/conformal_calibration.schema.json"),
        Path("examples/fixtures/conformal/calibration_artifacts.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/"
        "conformal_calibration.v1.schema.json",
    ),
    (
        Path("docs/contracts/cohort_manifest.schema.json"),
        Path("examples/fixtures/conformal/cohort_manifest_roles.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/cohort_manifest.v1.schema.json",
    ),
    (
        Path("docs/contracts/conformal_prediction_block.schema.json"),
        Path("examples/fixtures/conformal/conformal_prediction_blocks.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/"
        "conformal_prediction_block.v1.schema.json",
    ),
    (
        Path("docs/contracts/conformal_metric_set.schema.json"),
        Path("examples/fixtures/conformal/conformal_metric_sets.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/conformal_metric_set.v1.schema.json",
    ),
    (
        Path("docs/contracts/domain_assessment_block.schema.json"),
        Path("examples/fixtures/conformal/domain_assessment_blocks.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/"
        "domain_assessment_block.v1.schema.json",
    ),
    (
        Path("docs/contracts/decision_block.schema.json"),
        Path("examples/fixtures/conformal/decision_blocks.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/decision_block.v1.schema.json",
    ),
    (
        Path("docs/contracts/robustness_scenario_spec.schema.json"),
        Path("examples/fixtures/robustness/robustness_scenarios.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/"
        "robustness_scenario_spec.v1.schema.json",
    ),
    (
        Path("docs/contracts/robustness_report.schema.json"),
        Path("examples/fixtures/robustness/robustness_reports.v1.json"),
        "https://github.com/GBeurier/dag-ml/schemas/robustness_report.v1.schema.json",
    ),
)
CONFORMAL_ROBUSTNESS_PACK_REL = Path(
    "docs/contracts/conformal_robustness_conformance_pack.v1.json"
)
CONFORMAL_ROBUSTNESS_GOLDEN_REL = Path(
    "parity/conformal/golden/split_absolute_residual.v1.json"
)
CONFORMAL_METRICS_GOLDEN_REL = Path(
    "parity/conformal/golden/regression_conformal_metrics.v1.json"
)
CANONICAL_PROFILE_GOLDEN_REL = Path(
    "parity/canonical/golden/tcv1_jcs_cross_language.v1.json"
)
CANONICAL_ORACLE_ARTIFACT_RELS = (
    Path("parity/canonical/README.md"),
    CANONICAL_PROFILE_GOLDEN_REL,
    Path("parity/canonical/rust-oracle/.gitignore"),
    Path("parity/canonical/rust-oracle/Cargo.lock"),
    Path("parity/canonical/rust-oracle/Cargo.toml"),
    Path("parity/canonical/rust-oracle/src/lib.rs"),
    Path("parity/canonical/rust-oracle/src/main.rs"),
    Path("parity/canonical/tests/test_rust_oracle_parity.py"),
)
ROBUSTNESS_RNG_ORACLE_ARTIFACT_RELS = (
    Path("parity/robustness_rng/golden/philox4x32_10_counter.v1.json"),
    Path("parity/robustness_rng/oracle.py"),
    Path("parity/robustness_rng/tests/test_robustness_rng_contract.py"),
)
OPERATOR_VARIANT_LABEL_FIXTURE_REL = Path(
    "docs/contracts/operator_variant_label.v1.json"
)
OPERATOR_VARIANT_LABEL_FIXTURE_ID = "dag-ml.operator_variant_label.v1"
PROCESS_ADAPTER_DESCRIPTION_SCHEMA_REL = Path(
    "docs/contracts/process_adapter_description.schema.json"
)
PROCESS_ADAPTER_FRAME_SCHEMA_REL = Path(
    "docs/contracts/process_adapter_frame.schema.json"
)
RESEARCH_PROVENANCE_PROFILE_REL = Path(
    "docs/contracts/research_provenance_package_profile.v1.json"
)
LOCAL_FIXTURE_REL = Path(
    "examples/fixtures/data/coordinator_data_plan_envelope_nir.json"
)
LOCAL_MULTISOURCE_FIXTURE_REL = Path(
    "examples/fixtures/data/coordinator_data_plan_envelope_multisource_repetitions.json"
)
LOCAL_FEATURE_FUSION_FIXTURE_REL = Path(
    "examples/fixtures/data/feature_fusion_selector_nir_chem.json"
)
SHARED_FOLD_SET_FIXTURE_REL = Path(
    "examples/fixtures/shared/fold_set_cv_partition.json"
)
LOCAL_GRAPH_SPEC_FIXTURE_REL = Path("examples/branch_merge_oof_graph.json")
LOCAL_PIPELINE_DSL_FIXTURE_REL = Path("examples/pipeline_dsl_nirs4all_compat.json")
LOCAL_CAMPAIGN_SPEC_FIXTURE_REL = Path("examples/campaign_oof_generation.json")
LOCAL_EXECUTION_PLAN_FIXTURE_REL = Path(
    "examples/fixtures/runtime/execution_plan_branch_merge_executable.json"
)
LOCAL_MODEL_INPUT_SPEC_FIXTURE_REL = Path(
    "examples/fixtures/data/model_input_spec_tabular_regressor.json"
)
LOCAL_DATA_PLAN_FIXTURE_REL = Path(
    "examples/fixtures/data/data_plan_tabular_fusion.json"
)
LOCAL_CONTROLLER_MANIFEST_FIXTURE_REL = Path(
    "examples/fixtures/runtime/controller_manifest_data_aware_model.json"
)
LOCAL_CONTROLLER_MANIFEST_LIST_FIXTURE_REL = Path("examples/controller_manifests.json")
LOCAL_SELECTION_POLICY_FIXTURE_REL = Path(
    "examples/fixtures/bundle/selection_policy_rmse.json"
)
LOCAL_SELECTION_DECISION_FIXTURE_REL = Path(
    "examples/fixtures/bundle/selection_decision_branch_b0.json"
)
LOCAL_DATA_OUTPUT_PROVENANCE_FIXTURE_REL = Path(
    "examples/fixtures/runtime/data_output_provenance_augmented_view.json"
)
LOCAL_OOF_SUCCESS_FIXTURE_REL = Path(
    "examples/fixtures/oof_campaign/uc6_oof_success_predictions.json"
)
LOCAL_OOF_TRAIN_REFUSAL_FIXTURE_REL = Path(
    "examples/fixtures/oof_campaign/uc11_train_prediction_refusal.json"
)
LOCAL_NODE_TASK_FIXTURE_REL = Path(
    "examples/fixtures/runtime/node_task_transform_scale.json"
)
LOCAL_NODE_RESULT_FIXTURE_REL = Path(
    "examples/fixtures/runtime/node_result_transform_scale.json"
)
LOCAL_PROCESS_ADAPTER_DESCRIPTION_FIXTURE_REL = Path(
    "examples/fixtures/runtime/process_adapter_description_python.json"
)
LOCAL_PROCESS_ADAPTER_FRAME_FIXTURE_RELS = [
    Path("examples/fixtures/runtime/process_adapter_frame_init.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_task_transform_scale.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_result_transform_scale.json"),
    Path("examples/fixtures/runtime/process_adapter_frame_ack_initialized.json"),
    Path(
        "examples/fixtures/runtime/process_adapter_frame_error_retryable_timeout.json"
    ),
    Path("examples/fixtures/runtime/process_adapter_frame_close.json"),
]
LOCAL_C_HEADER_REL = Path("crates/dag-ml-capi/include/dag_ml.h")
SIBLING_FIXTURE_REL = Path(
    "examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
)
SIBLING_FEATURE_FUSION_FIXTURE_REL = Path(
    "examples/fixtures/oof_campaign/feature_fusion_selector_nir_chem.json"
)
SIBLING_MODEL_INPUT_SPEC_FIXTURE_REL = Path(
    "examples/fixtures/data/model_input_spec_tabular_regressor.json"
)
SIBLING_C_HEADER_REL = Path("crates/dag-ml-data-capi/include/dag_ml_data.h")
LOCAL_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "coordinator_data_plan_envelope.v1.schema.json"
)
LOCAL_FEATURE_FUSION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/feature_fusion_selector.v1.schema.json"
)
LOCAL_BRANCH_VIEW_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/coordinator_branch_view.v1.schema.json"
)
LOCAL_FITTED_ADAPTER_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/fitted_adapter_ref.v1.schema.json"
)
GRAPH_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/graph_spec.v1.schema.json"
)
PIPELINE_DSL_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/pipeline_dsl.v1.schema.json"
)
CAMPAIGN_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/campaign_spec.v1.schema.json"
)
EXECUTION_PLAN_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/execution_plan.v1.schema.json"
)
MODEL_INPUT_SPEC_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/model_input_spec.v1.schema.json"
)
DATA_PLAN_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/data_plan.v1.schema.json"
)
CONTROLLER_MANIFEST_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/controller_manifest.v1.schema.json"
)
SELECTION_POLICY_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/selection_policy.v1.schema.json"
)
SELECTION_DECISION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/selection_decision.v1.schema.json"
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
REPRESENTATION_COMPATIBILITY_SEVERITIES = {"info", "warning", "error"}
REPRESENTATION_COMPATIBILITY_OUTCOMES = {
    "compatible",
    "compatible_with_fallback",
    "incompatible",
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
    "uses_training_weights",
    "uses_early_stopping",
    "performs_internal_tuning",
    "trains_aggregation",
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
D8_CONFORMANCE_SCENARIOS = (
    "multisource_a2_b3_c2.v1",
    "sample_level_late_fusion.v1",
    "cartesian_combo_to_sample_reducer.v1",
    "missing_source_with_fallback.v1",
    "stacking_oof_contract.v1",
    "invalid_unit_join.v1",
    "row_vs_sample_selection_mismatch.v1",
    "operator_level_variant_additive_fields.v1",
    "generation_constraints_prune_variants.v1",
)
REQUIRED_PARITY_CASE_IDS = {
    "nirs4all_core_browser_compile_plan",
    "repetition_group_leakage_refusal",
    "controller_registry_selector_parity",
    "branch_merge_oof_refit_replay",
    "python_wheel_facade_integration",
}
EXPECTED_PARITY_TOLERANCE_PROFILES = {
    "regression.cross_impl": {
        "metric": "prediction",
        "absolute_tolerance": 1e-3,
        "relative_tolerance": 1e-3,
        "owner": "nirs4all compatibility ledger",
    },
    "regression.kernel": {
        "metric": "prediction",
        "absolute_tolerance": 1e-9,
        "relative_tolerance": 1e-9,
        "owner": "nirs4all compatibility ledger",
    },
    "regression.native_export": {
        "metric": "prediction",
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
        "owner": "nirs4all compatibility ledger",
    },
    "classification.default": {
        "metric": "class_label",
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "owner": "nirs4all compatibility ledger",
    },
}
OPENLINEAGE_FACETS_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/openlineage_dagml_facets.v1.schema.json"
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
    "https://github.com/GBeurier/dag-ml/schemas/data_output_provenance.v1.schema.json"
)
SCORE_SET_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/score_set.v1.schema.json"
)
CHAIN_EFFECT_ANALYSIS_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/chain_effect_analysis.v1.schema.json"
)
CONFORMAL_CALIBRATION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/conformal_calibration.v1.schema.json"
)
CONFORMAL_CALIBRATION_FIXTURE_ID = (
    "dag-ml.conformal.split_absolute_residual.physical_sample.v1"
)
PARAMETER_PATCH_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/parameter_patch.v1.schema.json"
)
OUTPUT_BINDING_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/output_binding.v1.schema.json"
)
TRAINING_INFLUENCE_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "training_influence_manifest.v1.schema.json"
)
EXECUTION_BUNDLE_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/execution_bundle.v1.schema.json"
)
PREDICTION_CACHE_PAYLOAD_SET_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "prediction_cache_payload_set.v1.schema.json"
)
TRAINING_OUTCOME_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/training_outcome.v1.schema.json"
)
REPLAY_OUTCOME_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/replay_outcome.v1.schema.json"
)
TRAINING_REQUEST_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/training_request.v1.schema.json"
)
CACHE_NAMESPACE_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/cache_namespace.v1.schema.json"
)
PARAMETER_PROJECTION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/parameter_projection.v1.schema.json"
)
PORTABLE_PREDICTOR_PACKAGE_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "portable_predictor_package.v1.schema.json"
)
NODE_TASK_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/node_task.v1.schema.json"
)
NODE_RESULT_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/node_result.v1.schema.json"
)
PROCESS_ADAPTER_DESCRIPTION_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/"
    "process_adapter_description.v1.schema.json"
)
PROCESS_ADAPTER_FRAME_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/process_adapter_frame.v1.schema.json"
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


def reject_nonstandard_json_constant(token: str) -> None:
    raise ContractError(f"non-standard JSON numeric constant `{token}` is forbidden")


def load_json(path: Path) -> Any:
    def reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in document, f"duplicate JSON member `{key}` in {path}")
            document[key] = value
        return document

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicate_members,
                parse_constant=reject_nonstandard_json_constant,
            )
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON file: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def build_local_schema_registry() -> tuple[Registry, dict[str, dict[str, Any]]]:
    """Build an offline Draft 2020-12 registry from published local schemas."""

    registry = Registry()
    schemas: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "docs/contracts").glob("*.schema.json")):
        schema = load_json(path)
        require(isinstance(schema, dict), f"JSON Schema {path} must be an object")
        schema_id = schema.get("$id")
        require_non_empty_string(schema_id, f"JSON Schema {path}.$id")
        require(
            schema_id not in schemas, f"duplicate local JSON Schema $id `{schema_id}`"
        )
        try:
            Draft202012Validator.check_schema(schema)
            resource = Resource.from_contents(schema)
        except SchemaError as exc:
            raise ContractError(
                f"invalid Draft 2020-12 schema {path}: {exc.message}"
            ) from exc
        schemas[schema_id] = schema
        registry = registry.with_resource(schema_id, resource)
    for schema_id, schema in schemas.items():
        references: list[str] = []

        def collect_references(value: Any) -> None:
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str):
                    references.append(reference)
                for member in value.values():
                    collect_references(member)
            elif isinstance(value, list):
                for member in value:
                    collect_references(member)

        collect_references(schema)
        resolver = registry.resolver(schema_id)
        for reference in references:
            try:
                resolver.lookup(reference)
            except Unresolvable as exc:
                raise ContractError(
                    f"schema `{schema_id}` has unresolved local $ref `{reference}`: {exc}"
                ) from exc
    return registry, schemas


def json_schema_error_path(error: ValidationError) -> str:
    path = "$"
    for member in error.absolute_path:
        path += f"[{member}]" if isinstance(member, int) else f".{member}"
    return path


def validate_draft_2020_instance(
    instance: Any,
    schema: dict[str, Any],
    registry: Registry,
    label: str,
) -> None:
    """Validate one instance without permitting remote schema retrieval."""

    try:
        errors = sorted(
            Draft202012Validator(schema, registry=registry).iter_errors(instance),
            key=lambda error: tuple(str(member) for member in error.absolute_path),
        )
    except Unresolvable as exc:
        raise ContractError(
            f"{label} has an unresolved local JSON Schema reference: {exc}"
        ) from exc
    if errors:
        error = errors[0]
        raise ContractError(
            f"{label} fails Draft 2020-12 at {json_schema_error_path(error)}: {error.message}"
        )


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(f"missing text file: {path}") from exc


def require_non_empty_string(value: Any, label: str) -> None:
    require(
        isinstance(value, str) and bool(value), f"{label} must be a non-empty string"
    )


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


def require_no_unknown_keys(
    value: dict[str, Any], allowed: set[str], label: str
) -> None:
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
        require_optional_non_empty_string(
            value.get("reducer_id"), f"{label}.reducer_id"
        )
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
            require(
                isinstance(preserve_provenance, bool),
                f"{label}.preserve_provenance bool",
            )
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


def validate_representation_compatibility_report(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    policy = value.get("policy")
    outcome = value.get("outcome")
    require(
        policy in REPRESENTATION_MISSING_SOURCE_POLICIES,
        f"{label}.policy is invalid",
    )
    require(
        outcome in REPRESENTATION_COMPATIBILITY_OUTCOMES,
        f"{label}.outcome is invalid",
    )
    require_optional_non_empty_string(
        value.get("fallback_used"), f"{label}.fallback_used"
    )
    warning_severity = value.get("warning_severity")
    if warning_severity is not None:
        require(
            warning_severity in REPRESENTATION_COMPATIBILITY_SEVERITIES,
            f"{label}.warning_severity is invalid",
        )
    counts = {}
    for field in (
        "affected_source_count",
        "affected_repetition_count",
        "affected_sample_count",
    ):
        count = value.get(field, 0)
        require_non_negative_int(count, f"{label}.{field}")
        counts[field] = count
    for field in ("train_relation_fingerprint", "predict_relation_fingerprint"):
        fingerprint = value.get(field)
        if fingerprint is not None:
            require_sha256(fingerprint, f"{label}.{field}")
    for field in ("train_unit_count", "predict_unit_count"):
        count = value.get(field)
        if count is not None:
            require_non_negative_int(count, f"{label}.{field}")
    for field in (
        "fixed_width_required",
        "final_reducer_stabilizes_output",
        "cartesian_combo_count_changed",
        "late_fusion_branch_delta",
    ):
        require(
            isinstance(value.get(field, False), bool), f"{label}.{field} must be bool"
        )
    messages = value.get("messages", [])
    require(isinstance(messages, list), f"{label}.messages must be an array")
    for index, message in enumerate(messages):
        require_non_empty_string(message, f"{label}.messages[{index}]")
    validate_metadata_object(value.get("metadata"), f"{label}.metadata")

    affected_total = sum(counts.values())
    fallback_used = value.get("fallback_used")
    if affected_total == 0:
        require(
            outcome != "compatible_with_fallback",
            f"{label}.outcome cannot use fallback when no units are affected",
        )
        require(
            warning_severity is None,
            f"{label}.warning_severity requires affected units",
        )
    elif policy == "strict":
        require(outcome == "incompatible", f"{label}.strict affected outcome invalid")
        require(fallback_used is None, f"{label}.strict cannot use fallback")
    else:
        require(warning_severity is not None, f"{label}.warning_severity required")
        require(
            outcome != "compatible", f"{label}.affected outcome cannot be compatible"
        )
        if outcome == "compatible_with_fallback":
            require(fallback_used is not None, f"{label}.fallback_used required")
    if outcome == "incompatible":
        require(fallback_used is None, f"{label}.incompatible cannot use fallback")

    train_unit_count = value.get("train_unit_count")
    predict_unit_count = value.get("predict_unit_count")
    unit_count_changed = (
        train_unit_count is not None
        and predict_unit_count is not None
        and train_unit_count != predict_unit_count
    )
    if value.get("fixed_width_required", False) and unit_count_changed:
        if outcome != "incompatible":
            require(
                policy in {"mask", "pad"} or fallback_used in {"mask", "pad"},
                f"{label}.fixed_width mismatch requires mask or pad",
            )
    if value.get("cartesian_combo_count_changed", False):
        if outcome != "incompatible":
            require(
                value.get("final_reducer_stabilizes_output", False) is True,
                f"{label}.cartesian_combo_count_changed requires final reducer",
            )
    if value.get("late_fusion_branch_delta", False):
        if outcome != "incompatible":
            require(
                policy
                in {
                    "drop_incomplete",
                    "impute_declared",
                    "mask",
                    "partial_model",
                    "pad",
                }
                or fallback_used
                in {
                    "drop_incomplete",
                    "impute_declared",
                    "mask",
                    "partial_model",
                    "pad",
                },
                f"{label}.late_fusion_branch_delta requires declared fallback policy",
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
    require(
        isinstance(combo_selection, list), f"{label}.combo_selection must be an array"
    )
    combo_unit_ids: set[str] = set()
    for index, record in enumerate(combo_selection):
        record_label = f"{label}.combo_selection[{index}]"
        require(isinstance(record, dict), f"{record_label} must be an object")
        require_non_empty_string(
            record.get("combo_unit_id"), f"{record_label}.combo_unit_id"
        )
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
    for field in ("train_compatibility", "predict_compatibility"):
        report = value.get(field)
        if report is not None:
            validate_representation_compatibility_report(report, f"{label}.{field}")
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


def validate_feature_fusion_schema_artifact(
    schema: Any, expected_id: str, label: str
) -> None:
    require(
        isinstance(schema, dict), f"{label} feature-fusion schema must be a JSON object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} feature-fusion schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == expected_id,
        f"{label} feature-fusion schema has unexpected $id",
    )
    require(
        schema.get("type") == "object", f"{label} feature-fusion root must be an object"
    )
    required = schema.get("required")
    require(
        isinstance(required, list), f"{label} feature-fusion required list is missing"
    )
    for field in ("schema_version", "feature_set_id", "sources", "alignment"):
        require(
            field in required,
            f"{label} feature-fusion schema does not require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict), f"{label} feature-fusion properties are missing"
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} feature-fusion schema_version const must be 1",
    )
    for field in ("combination_plan", "representation_plan", "source_layout"):
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
        "source_layout",
    ):
        require(
            name in defs, f"{label} feature-fusion schema misses `{name}` definition"
        )
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


def validate_branch_view_schema_artifact(
    schema: Any, expected_id: str, label: str
) -> None:
    require(
        isinstance(schema, dict), f"{label} branch-view schema must be a JSON object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} branch-view schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == expected_id,
        f"{label} branch-view schema has unexpected $id",
    )
    require(
        schema.get("type") == "object", f"{label} branch-view root must be an object"
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} branch-view required list is missing")
    for field in ("view_id", "branch_id", "mode", "selector"):
        require(
            field in required, f"{label} branch-view schema does not require `{field}`"
        )
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
    require(
        isinstance(schema, dict), f"{label} fitted-adapter schema must be a JSON object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} fitted-adapter schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == expected_id,
        f"{label} fitted-adapter schema has unexpected $id",
    )
    require(
        schema.get("type") == "object", f"{label} fitted-adapter root must be an object"
    )
    required = schema.get("required")
    require(
        isinstance(required, list), f"{label} fitted-adapter required list is missing"
    )
    for field in ("adapter_id", "adapter_version", "params_fingerprint"):
        require(
            field in required,
            f"{label} fitted-adapter schema does not require `{field}`",
        )
    properties = schema.get("properties")
    require(
        isinstance(properties, dict), f"{label} fitted-adapter properties are missing"
    )
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} fitted-adapter schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} fitted-adapter $defs are missing")
    for name in ("non_empty_id", "hex_fingerprint", "backend"):
        require(
            name in defs, f"{label} fitted-adapter schema misses `{name}` definition"
        )
    backends = defs.get("backend", {}).get("enum")
    require(
        isinstance(backends, list), f"{label} fitted-adapter backend enum is missing"
    )
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
    require(
        schema.get("$id") == GRAPH_SPEC_SCHEMA_ID,
        f"{label} GraphSpec schema $id mismatch",
    )
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
    require(
        isinstance(port_props, dict), f"{label} GraphSpec port_spec properties missing"
    )
    for property_name in ("unit_level", "alignment_key", "target_level"):
        require(
            property_name in port_props,
            f"{label} GraphSpec port_spec misses `{property_name}`",
        )
    edge_props = defs.get("edge_contract", {}).get("properties")
    require(
        isinstance(edge_props, dict),
        f"{label} GraphSpec edge_contract properties missing",
    )
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
    require(
        schema.get("$id") == PIPELINE_DSL_SCHEMA_ID,
        f"{label} Pipeline DSL $id mismatch",
    )
    require(
        isinstance(schema.get("oneOf"), list),
        f"{label} Pipeline DSL root must use oneOf",
    )
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
        require(
            definition_name in defs,
            f"{label} Pipeline DSL schema misses `{definition_name}`",
        )
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
    for property_name in (
        "name",
        "representation",
        "unit_level",
        "alignment_key",
        "target_level",
    ):
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
    require(
        isinstance(compat_properties, dict),
        f"{label} Pipeline DSL compat properties missing",
    )
    for property_name in (
        "class",
        "function",
        "ref",
        "type",
        "name",
        "step",
        "tuner",
        "finetune",
    ):
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
    require(
        schema.get("$id") == CAMPAIGN_SPEC_SCHEMA_ID,
        f"{label} CampaignSpec $id mismatch",
    )
    require(
        schema.get("type") == "object", f"{label} CampaignSpec root must be an object"
    )
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
        "generation_constraints",
        "choice_ref",
        "data_model_shape_plan",
        "data_view_policy",
        "data_binding",
        "data_view_selector",
        "branch_view_plan",
    ):
        require(
            definition_name in defs,
            f"{label} CampaignSpec schema misses `{definition_name}`",
        )
    branch_view_properties = defs.get("branch_view_plan", {}).get("properties")
    require(
        isinstance(branch_view_properties, dict)
        and "selector" in branch_view_properties,
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
    require(
        schema.get("type") == "object", f"{label} ExecutionPlan root must be an object"
    )
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
        require(
            field in required, f"{label} ExecutionPlan schema must require `{field}`"
        )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} ExecutionPlan $defs missing")
    require(
        defs.get("phase", {}).get("enum")
        == ["COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"],
        f"{label} ExecutionPlan phase enum is not aligned",
    )
    require(
        set(defs.get("controller_capability", {}).get("enum", []))
        == CONTROLLER_CAPABILITIES,
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
        require(
            definition_name in defs,
            f"{label} ExecutionPlan schema misses `{definition_name}`",
        )


def validate_model_input_spec_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict), f"{label} ModelInputSpec schema must be an object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} ModelInputSpec schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == MODEL_INPUT_SPEC_SCHEMA_ID,
        f"{label} ModelInputSpec schema $id mismatch",
    )
    require(
        schema.get("type") == "object", f"{label} ModelInputSpec root must be an object"
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} ModelInputSpec root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} ModelInputSpec required list missing")
    for field in ("schema_version", "ports"):
        require(
            field in required, f"{label} ModelInputSpec schema must require `{field}`"
        )
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} ModelInputSpec properties missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} ModelInputSpec schema_version const must be 1",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} ModelInputSpec $defs missing")
    require("input_port" in defs, f"{label} ModelInputSpec schema misses input_port")
    require(
        "fusion_policy" in defs, f"{label} ModelInputSpec schema misses fusion_policy"
    )
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
        set(defs.get("fit_influence_policy", {}).get("enum", []))
        == FIT_INFLUENCE_POLICIES,
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
        "representation_plan" in defs.get("fusion_policy", {}).get("properties", {}),
        f"{label} ModelInputSpec fusion policy misses representation_plan",
    )


def validate_data_plan_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} DataPlan schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} DataPlan schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == DATA_PLAN_SCHEMA_ID,
        f"{label} DataPlan schema $id mismatch",
    )
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
    require(
        isinstance(schema, dict), f"{label} ControllerManifest schema must be an object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} ControllerManifest schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == CONTROLLER_MANIFEST_SCHEMA_ID,
        f"{label} ControllerManifest schema $id mismatch",
    )
    require(
        schema.get("type") == "object",
        f"{label} ControllerManifest root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} ControllerManifest root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list), f"{label} ControllerManifest required list missing"
    )
    for field in (
        "controller_id",
        "controller_version",
        "operator_kind",
        "supported_phases",
        "fit_scope",
        "rng_policy",
        "artifact_policy",
    ):
        require(
            field in required,
            f"{label} ControllerManifest schema must require `{field}`",
        )
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
        set(defs.get("controller_capability", {}).get("enum", []))
        == CONTROLLER_CAPABILITIES,
        f"{label} ControllerManifest capability enum is not aligned",
    )
    require(
        set(defs.get("fit_influence_policy", {}).get("enum", []))
        == FIT_INFLUENCE_POLICIES,
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
    require(
        isinstance(schema, dict), f"{label} SelectionPolicy schema must be an object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} SelectionPolicy schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == SELECTION_POLICY_SCHEMA_ID,
        f"{label} SelectionPolicy $id mismatch",
    )
    require(
        schema.get("type") == "object",
        f"{label} SelectionPolicy root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} SelectionPolicy root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list), f"{label} SelectionPolicy required list missing"
    )
    for field in ("id", "metric"):
        require(
            field in required, f"{label} SelectionPolicy schema must require `{field}`"
        )
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
        require(
            field in properties,
            f"{label} SelectionPolicy schema must declare `{field}`",
        )
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
    require(
        "selection_metric" in defs, f"{label} SelectionPolicy misses selection_metric"
    )
    for definition_name in (
        "evaluation_result",
        "refit_slot_plan",
        "stacking_fit_contract",
    ):
        require(
            definition_name in defs,
            f"{label} SelectionPolicy misses {definition_name}",
        )


def validate_selection_decision_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict), f"{label} SelectionDecision schema must be an object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} SelectionDecision schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == SELECTION_DECISION_SCHEMA_ID,
        f"{label} SelectionDecision $id mismatch",
    )
    require(
        schema.get("type") == "object",
        f"{label} SelectionDecision root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} SelectionDecision root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list), f"{label} SelectionDecision required list missing"
    )
    for field in (
        "policy_id",
        "selected_candidate_id",
        "metric_name",
        "objective",
        "selected_score",
        "ranked_candidates",
    ):
        require(
            field in required,
            f"{label} SelectionDecision schema must require `{field}`",
        )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} SelectionDecision $defs missing")
    properties = schema.get("properties")
    require(
        isinstance(properties, dict), f"{label} SelectionDecision properties missing"
    )
    for field in ("evaluation_scope", "refit_slot_plan", "reduction_id"):
        require(
            field in properties,
            f"{label} SelectionDecision schema must declare `{field}`",
        )
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
    require(
        "ranked_candidate" in defs, f"{label} SelectionDecision misses ranked_candidate"
    )
    require(
        "refit_slot_plan" in defs, f"{label} SelectionDecision misses refit_slot_plan"
    )


def validate_openlineage_facets_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict), f"{label} OpenLineage facets schema must be an object"
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} OpenLineage facets schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == OPENLINEAGE_FACETS_SCHEMA_ID,
        f"{label} OpenLineage facets schema has unexpected $id",
    )
    defs = schema.get("$defs")
    require(
        isinstance(defs, dict), f"{label} OpenLineage facets schema $defs are missing"
    )
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
        properties.get("prediction_level", {}).get("enum")
        == ["sample", "target", "group"],
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
        properties.get("prediction_level", {}).get("enum")
        == ["sample", "target", "group"],
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
    require(
        isinstance(defs, dict), f"{label} aggregation-controller task $defs missing"
    )
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
        defs["aggregation_policy"]["properties"]["method"].get("const")
        == "custom_controller",
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
        defs.get("reduction_axis", {}).get("enum")
        == ["unit", "fold", "model", "metric"],
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
    require(
        isinstance(defs, dict), f"{label} aggregation-controller result $defs missing"
    )
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
        defs.get("reduction_axis", {}).get("enum")
        == ["unit", "fold", "model", "metric"],
        f"{label} aggregation-controller result reduction axes are not aligned",
    )
    require(
        defs.get("entity_unit_level", {}).get("enum")
        == ["physical_sample", "source_sample", "observation", "combo"],
        f"{label} aggregation-controller result entity unit levels are not aligned",
    )


def validate_score_set_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} score-set schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} score-set schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == SCORE_SET_SCHEMA_ID,
        f"{label} score-set schema has unexpected $id",
    )
    require(schema.get("type") == "object", f"{label} score-set root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} score-set root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} score-set required list is missing")
    for field in ("schema_version", "plan_id", "reports"):
        require(field in required, f"{label} score-set schema must require `{field}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} score-set properties missing")
    require("reports" in properties, f"{label} score-set schema must define `reports`")


def validate_score_set_fixture(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} score-set fixture must be an object")
    for field in ("schema_version", "plan_id", "reports"):
        require(field in value, f"{label} score-set fixture missing `{field}`")
    reports = value.get("reports")
    require(
        isinstance(reports, list) and bool(reports),
        f"{label} score-set fixture must have a non-empty reports list",
    )
    for report in reports:
        require(isinstance(report, dict), f"{label} score-set report must be an object")
        for field in (
            "producer_node",
            "partition",
            "level",
            "row_count",
            "target_width",
            "metrics",
        ):
            require(field in report, f"{label} score-set report missing `{field}`")
        require(
            isinstance(report.get("metrics"), dict) and bool(report["metrics"]),
            f"{label} score-set report metrics must be a non-empty object",
        )
        for optional_string in ("variant_id", "variant_label"):
            if optional_string in report and report[optional_string] is not None:
                require_non_empty_string(
                    report[optional_string],
                    f"{label} score-set report.{optional_string}",
                )


CONFORMAL_COHORT_ROLES = {
    "development",
    "calibration",
    "external_test",
    "production",
}
CONFORMAL_INFLUENCE_KINDS = (
    "transform_fit",
    "model_fit",
    "hpo_selection",
    "early_stopping",
    "weighting_resampling",
    "trained_meta_aggregation",
)
CONFORMAL_PARAMETER_NAMESPACES = {"operator", "fit", "control", "structural"}
CONFORMAL_PREDICTION_SOURCES = {"final_refit", "cv_ensemble"}
CONFORMAL_TCV1_PREFIX = b"DAGML-TCV1\0"
CONFORMAL_INT_MIN = -(2**63)
CONFORMAL_INT_MAX = 2**64 - 1
CONFORMAL_PORTABLE_INT_MAX = 2**53 - 1


def validate_chain_effect_analysis_schema(schema: Any, label: str) -> None:
    require(
        isinstance(schema, dict),
        f"{label} chain-effect schema must be an object",
    )
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} chain-effect schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == CHAIN_EFFECT_ANALYSIS_SCHEMA_ID,
        f"{label} chain-effect schema has unexpected $id",
    )
    require(
        schema.get("type") == "object",
        f"{label} chain-effect root must be an object",
    )
    require(
        schema.get("additionalProperties") is False,
        f"{label} chain-effect root must reject unknown fields",
    )
    required = schema.get("required")
    require(
        isinstance(required, list),
        f"{label} chain-effect required list is missing",
    )
    for field in (
        "schema_id",
        "schema_version",
        "metric",
        "lens",
        "baseline",
        "points",
    ):
        require(
            field in required,
            f"{label} chain-effect schema must require `{field}`",
        )
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} chain-effect properties missing")
    require(
        "points" in properties,
        f"{label} chain-effect schema must define `points`",
    )


def validate_chain_effect_analysis_fixture(value: Any, label: str) -> None:
    require(
        isinstance(value, dict),
        f"{label} chain-effect fixture must be an object",
    )
    for field in (
        "schema_id",
        "schema_version",
        "metric",
        "lens",
        "baseline",
        "points",
    ):
        require(field in value, f"{label} chain-effect fixture missing `{field}`")
    require(
        value.get("schema_id") == CHAIN_EFFECT_ANALYSIS_SCHEMA_ID,
        f"{label} chain-effect fixture has unexpected schema_id",
    )
    require(
        value.get("lens") in ("raw", "rank_by_dataset", "z_by_dataset"),
        f"{label} chain-effect fixture has an unknown lens",
    )
    points = value.get("points")
    require(
        isinstance(points, list) and bool(points),
        f"{label} chain-effect fixture must have a non-empty points list",
    )
    dataset_required = value.get("lens") in ("rank_by_dataset", "z_by_dataset")
    for point in points:
        require(
            isinstance(point, dict),
            f"{label} chain-effect point must be an object",
        )
        for field in ("id", "score", "goodness", "ordered_tokens"):
            require(field in point, f"{label} chain-effect point missing `{field}`")
        require(
            not dataset_required or point.get("dataset") not in (None, ""),
            f"{label} chain-effect rank/z point must declare a dataset",
        )
        tokens = point.get("ordered_tokens")
        require(
            isinstance(tokens, list) and bool(tokens),
            f"{label} chain-effect point ordered_tokens must be a non-empty list",
        )
        for token in tokens:
            require(
                isinstance(token, dict) and "token" in token and "role" in token,
                f"{label} chain-effect token must declare `token` and `role`",
            )


def require_version_one(value: Any, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == 1,
        f"{label} schema_version must be integer 1",
    )


def strict_utf8_nfc(value: str, label: str) -> tuple[str, bytes]:
    require(isinstance(value, str), f"{label} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    try:
        encoded = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ContractError(f"{label} contains a Unicode surrogate") from exc
    return normalized, encoded


def validate_strict_json_value(value: Any, label: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        require(
            CONFORMAL_INT_MIN <= value <= CONFORMAL_INT_MAX,
            f"{label} integer is outside the TCV1 serde-compatible range",
        )
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} must not contain a non-finite float")
        return
    if isinstance(value, str):
        strict_utf8_nfc(value, label)
        return
    if isinstance(value, list):
        for index, member in enumerate(value):
            validate_strict_json_value(member, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        normalized_keys: set[bytes] = set()
        for key, member in value.items():
            normalized, encoded = strict_utf8_nfc(key, f"{label} object key")
            require(
                encoded not in normalized_keys,
                f"{label} has colliding NFC object keys including `{normalized}`",
            )
            normalized_keys.add(encoded)
            validate_strict_json_value(member, f"{label}.{normalized}")
        return
    raise ContractError(
        f"{label} contains a non-JSON value of type {type(value).__name__}"
    )


def tcv1_u64(value: int) -> bytes:
    require(0 <= value <= 2**64 - 1, "TCV1 length/count exceeds u64")
    return struct.pack(">Q", value)


def dagml_tcv1_encode_value(value: Any, label: str = "TCV1 value") -> bytes:
    validate_strict_json_value(value, label)
    if value is None:
        return b"N"
    if value is False:
        return b"F"
    if value is True:
        return b"T"
    if isinstance(value, int):
        payload = str(value).encode("ascii")
        return b"I" + tcv1_u64(len(payload)) + payload
    if isinstance(value, float):
        normalized = 0.0 if value == 0.0 else value
        return b"D" + struct.pack(">d", normalized)
    if isinstance(value, str):
        _, payload = strict_utf8_nfc(value, label)
        return b"S" + tcv1_u64(len(payload)) + payload
    if isinstance(value, list):
        return (
            b"A"
            + tcv1_u64(len(value))
            + b"".join(
                dagml_tcv1_encode_value(member, f"{label}[{index}]")
                for index, member in enumerate(value)
            )
        )
    if isinstance(value, dict):
        normalized_items: list[tuple[bytes, str, Any]] = []
        seen: set[bytes] = set()
        for key, member in value.items():
            normalized, encoded = strict_utf8_nfc(key, f"{label} object key")
            require(encoded not in seen, f"{label} has colliding NFC object keys")
            seen.add(encoded)
            normalized_items.append((encoded, normalized, member))
        normalized_items.sort(key=lambda item: item[0])
        payload = bytearray(b"O" + tcv1_u64(len(normalized_items)))
        for _encoded, normalized, member in normalized_items:
            payload.extend(dagml_tcv1_encode_value(normalized, f"{label} key"))
            payload.extend(dagml_tcv1_encode_value(member, f"{label}.{normalized}"))
        return bytes(payload)
    raise AssertionError("strict JSON validation accepted an unsupported TCV1 value")


def dagml_tcv1_preimage(value: Any) -> bytes:
    return CONFORMAL_TCV1_PREFIX + dagml_tcv1_encode_value(value)


def dagml_tcv1_sha256(value: Any) -> str:
    return hashlib.sha256(dagml_tcv1_preimage(value)).hexdigest()


def conformal_finite_sample_rank(sample_count: int, coverage: Any) -> int:
    require_positive_int(sample_count, "calibration sample_count")
    validate_conformal_coverages([coverage])
    canonical_token = repr(float(coverage))
    decimal_coverage = Decimal(canonical_token)
    numerator, denominator = decimal_coverage.as_integer_ratio()
    scaled_numerator = (sample_count + 1) * numerator
    return (scaled_numerator + denominator - 1) // denominator


def require_exact_keys(
    value: Any,
    required: set[str],
    optional: set[str],
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    missing = required - set(value)
    require(not missing, f"{label} is missing required field(s): {sorted(missing)}")
    require_no_unknown_keys(value, required | optional, label)
    return value


def validate_sorted_identifiers(
    value: Any,
    label: str,
    *,
    require_non_empty: bool,
) -> list[str]:
    require(isinstance(value, list), f"{label} must be an array")
    if require_non_empty:
        require(bool(value), f"{label} must be a non-empty array")
    for index, item in enumerate(value):
        require_identifier(item, f"{label}[{index}]")
    require(value == sorted(value), f"{label} must be sorted")
    require(len(set(value)) == len(value), f"{label} must contain unique identifiers")
    return value


def validate_ordered_unique_identifiers(
    value: Any,
    label: str,
    *,
    require_non_empty: bool,
) -> list[str]:
    require(isinstance(value, list), f"{label} must be an array")
    if require_non_empty:
        require(bool(value), f"{label} must be a non-empty array")
    for index, item in enumerate(value):
        require_identifier(item, f"{label}[{index}]")
    require(len(set(value)) == len(value), f"{label} must contain unique identifiers")
    return value


def validate_ordered_unique_strings(
    value: Any,
    label: str,
    *,
    require_non_empty: bool,
) -> list[str]:
    require(isinstance(value, list), f"{label} must be an array")
    if require_non_empty:
        require(bool(value), f"{label} must be a non-empty array")
    for index, item in enumerate(value):
        require_non_empty_string(item, f"{label}[{index}]")
    require(len(set(value)) == len(value), f"{label} must contain unique values")
    return value


def validate_conformal_coverages(value: Any, label: str = "coverages") -> list[float]:
    require(
        isinstance(value, list) and bool(value), f"{label} must be a non-empty array"
    )
    normalized: list[float] = []
    for index, coverage in enumerate(value):
        in_range = isinstance(coverage, float) and 0.0 < coverage < 1.0
        finite = in_range and math.isfinite(coverage)
        require(
            bool(finite),
            f"{label}[{index}] must be a finite binary64 float in (0, 1)",
        )
        normalized.append(coverage)
    require(
        all(left < right for left, right in zip(normalized, normalized[1:])),
        f"{label} must be strictly increasing and unique",
    )
    return normalized


def split_absolute_residual_oracle(
    residuals: Any,
    coverages: Any,
    small_sample_policy: Any,
) -> list[dict[str, Any]]:
    """Independent stdlib finite-sample oracle; never imported by production code."""

    require(
        small_sample_policy in {"error", "unbounded"},
        "small_sample_policy must be error or unbounded",
    )
    require(
        isinstance(residuals, list) and bool(residuals), "residuals must be non-empty"
    )
    finite_residuals: list[float] = []
    for index, residual in enumerate(residuals):
        is_number = isinstance(residual, (int, float)) and not isinstance(
            residual, bool
        )
        finite = is_number and (isinstance(residual, int) or math.isfinite(residual))
        if finite and isinstance(residual, int):
            finite = 0 <= residual <= CONFORMAL_PORTABLE_INT_MAX
        require(
            bool(finite) and residual >= 0.0,
            f"residuals[{index}] must be a finite non-negative number",
        )
        finite_residuals.append(float(residual))
    ordered = sorted(finite_residuals)
    valid_coverages = validate_conformal_coverages(coverages)
    result: list[dict[str, Any]] = []
    for coverage in valid_coverages:
        rank = conformal_finite_sample_rank(len(ordered), coverage)
        if rank > len(ordered):
            if small_sample_policy == "error":
                raise ContractError(
                    f"finite-sample rank {rank} exceeds calibration size {len(ordered)}"
                )
            quantile: dict[str, Any] = {"status": "unbounded"}
        else:
            quantile = {"status": "finite", "value": ordered[rank - 1]}
        result.append({"coverage": coverage, "rank": rank, "quantile": quantile})
    return result


def apply_json_pointer_mutation(
    value: Any, path: Any, replacement: Any, label: str
) -> Any:
    require(isinstance(path, str) and path.startswith("/"), f"{label}.path is invalid")
    mutated = copy.deepcopy(value)
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/")
    ]
    require(all(token for token in tokens), f"{label}.path contains an empty token")
    cursor = mutated
    for token in tokens[:-1]:
        if isinstance(cursor, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise ContractError(
                    f"{label}.path list token `{token}` is not an index"
                ) from exc
            require(
                0 <= index < len(cursor), f"{label}.path index {index} is out of range"
            )
            cursor = cursor[index]
        else:
            require(
                isinstance(cursor, dict),
                f"{label}.path cannot descend through a scalar",
            )
            require(token in cursor, f"{label}.path references missing key `{token}`")
            cursor = cursor[token]
    final = tokens[-1]
    if isinstance(cursor, list):
        try:
            index = int(final)
        except ValueError as exc:
            raise ContractError(
                f"{label}.path final token `{final}` is not an index"
            ) from exc
        require(0 <= index < len(cursor), f"{label}.path index {index} is out of range")
        cursor[index] = replacement
    else:
        require(isinstance(cursor, dict), f"{label}.path cannot update a scalar")
        require(final in cursor, f"{label}.path references missing key `{final}`")
        cursor[final] = replacement
    return mutated


def validate_versioned_object_schema(
    schema: Any,
    expected_id: str,
    required_fields: set[str],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(isinstance(schema, dict), f"{label} schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} schema must declare Draft 2020-12",
    )
    require(schema.get("$id") == expected_id, f"{label} schema $id mismatch")
    require(schema.get("type") == "object", f"{label} root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} required list is missing")
    require(
        required_fields <= set(required),
        f"{label} required fields drifted: {sorted(required_fields - set(required))}",
    )
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} properties are missing")
    version = properties.get("schema_version")
    require(
        isinstance(version, dict)
        and version.get("type") == "integer"
        and version.get("const") == 1,
        f"{label} schema_version must be integer const 1",
    )
    defs = schema.get("$defs", {})
    require(isinstance(defs, dict), f"{label} $defs must be an object")
    return properties, defs


def validate_parameter_patch_schema(schema: Any, label: str) -> None:
    properties, defs = validate_versioned_object_schema(
        schema,
        PARAMETER_PATCH_SCHEMA_ID,
        {"schema_version", "node_id", "namespace", "path", "value"},
        label,
    )
    require(
        properties.get("namespace", {}).get("$ref") == "#/$defs/parameter_namespace",
        f"{label} namespace must use the canonical definition",
    )
    require(
        defs.get("parameter_namespace", {}).get("enum")
        == ["operator", "fit", "control", "structural"],
        f"{label} namespace enum drifted",
    )
    path = properties.get("path", {})
    require(
        path.get("type") == "array" and path.get("minItems") == 1,
        f"{label} path must be a non-empty array",
    )
    target = defs.get("parameter_patch_target", {})
    require(
        set(target.get("required", [])) == {"node_id", "namespace", "path"},
        f"{label} ParameterPatchTarget shape drifted",
    )


def validate_training_influence_schema(schema: Any, label: str) -> None:
    properties, defs = validate_versioned_object_schema(
        schema,
        TRAINING_INFLUENCE_SCHEMA_ID,
        {"schema_version", "relation_fingerprint", "entries", "manifest_fingerprint"},
        label,
    )
    require(
        "training_outcome_fingerprint" not in properties,
        f"{label} must not expose recursive training_outcome_fingerprint",
    )
    require(
        "training_outcome_fingerprint" not in json.dumps(schema, sort_keys=True),
        f"{label} must reject even an optional recursive outcome property",
    )
    entry = defs.get("training_influence_entry", {})
    require(
        {
            "kind",
            "scope_id",
            "node_id",
            "physical_sample_ids",
            "origin_sample_ids",
            "group_ids",
        }
        <= set(entry.get("required", [])),
        f"{label} influence entry required fields drifted",
    )
    entry_properties = entry.get("properties", {})
    require(
        "default" not in entry_properties.get("node_id", {}),
        f"{label} required node_id must not declare a default",
    )
    for field in ("origin_sample_ids", "group_ids"):
        reference = entry_properties.get(field, {}).get("$ref")
        require(reference == "#/$defs/sorted_identifiers", f"{label} {field} drifted")
    require(
        "default" not in defs.get("sorted_identifiers", {}),
        f"{label} required identity sets must not declare defaults",
    )


def validate_output_binding_schema(schema: Any, label: str) -> None:
    required = {
        "schema_version",
        "binding_id",
        "node_id",
        "port_name",
        "prediction_level",
        "unit_level",
        "prediction_kind",
        "prediction_source",
        "refit_strategy",
        "aggregation_fingerprint",
        "target_names",
        "target_units",
        "class_labels",
        "output_order",
        "target_space",
        "binding_fingerprint",
    }
    properties, defs = validate_versioned_object_schema(
        schema, OUTPUT_BINDING_SCHEMA_ID, required, label
    )
    require(
        "default" in properties.get("unit_level", {})
        and properties.get("unit_level", {}).get("default") is None,
        f"{label} unit_level must default explicitly to null",
    )
    require(
        "default" in properties.get("refit_strategy", {})
        and properties.get("refit_strategy", {}).get("default") is None,
        f"{label} refit_strategy must default explicitly to null",
    )
    require(
        defs.get("prediction_kind", {}).get("enum")
        == [
            "regression_point",
            "class_label",
            "class_probability",
            "decision_score",
        ],
        f"{label} prediction kinds drifted",
    )
    bound = defs.get("bound_output", {})
    require(
        {
            "binding",
            "predictions",
            "observation_predictions",
            "aggregated_predictions",
        }
        <= set(bound.get("required", [])),
        f"{label} bound_output fields drifted",
    )


def validate_estimator_support_schema(
    schema: Any,
    expected_id: str,
    required_fields: set[str],
    label: str,
) -> None:
    _properties, defs = validate_versioned_object_schema(
        schema, expected_id, required_fields, label
    )
    if expected_id == EXECUTION_BUNDLE_SCHEMA_ID:
        for definition_name in (
            "prediction_requirement",
            "prediction_cache_block_record",
            "prediction_cache_record",
        ):
            definition = defs.get(definition_name, {})
            properties = definition.get("properties", {})
            require(
                properties.get("prediction_level", {}).get("default") == "sample"
                and properties.get("unit_ids", {}).get("default") == [],
                f"{label} {definition_name} legacy prediction defaults drifted",
            )
            required = set(definition.get("required", []))
            require(
                "prediction_level" not in required and "unit_ids" not in required,
                f"{label} {definition_name} additive prediction fields must stay optional",
            )
    if expected_id == PREDICTION_CACHE_PAYLOAD_SET_SCHEMA_ID:
        payload = defs.get("cache_payload", {})
        properties = payload.get("properties", {})
        require(
            properties.get("prediction_level", {}).get("default") == "sample"
            and properties.get("blocks", {}).get("default") == []
            and properties.get("aggregated_blocks", {}).get("default") == [],
            f"{label} cache payload legacy defaults drifted",
        )
        required = set(payload.get("required", []))
        require(
            not {"prediction_level", "blocks", "aggregated_blocks"} & required,
            f"{label} cache payload additive fields must stay optional",
        )


def validate_training_outcome_schema(schema: Any, label: str) -> None:
    required = {
        "schema_version",
        "outcome_id",
        "run_id",
        "training_request_fingerprint",
        "data_identities",
        "effective_plan",
        "effective_plan_fingerprint",
        "selected_variant_id",
        "selected_variant_fingerprint",
        "selection_output_id",
        "parameter_patches",
        "refit",
        "score_set",
        "outputs",
        "lineage",
        "portable_prediction_caches",
        "training_influence",
        "execution_bundle",
        "replayable_phases",
        "warnings",
        "diagnostics",
        "outcome_fingerprint",
    }
    properties, _defs = validate_versioned_object_schema(
        schema, TRAINING_OUTCOME_SCHEMA_ID, required, label
    )
    require(
        properties.get("parameter_patches", {}).get("default") == [],
        f"{label} parameter_patches must default explicitly to []",
    )
    require(
        "default" in properties.get("portable_prediction_caches", {})
        and properties.get("portable_prediction_caches", {}).get("default") is None,
        f"{label} portable caches must default explicitly to null",
    )
    require(
        properties.get("score_set", {}).get("$ref") == SCORE_SET_SCHEMA_ID,
        f"{label} score_set must be mandatory and non-null",
    )
    require(
        properties.get("warnings", {}).get("default") == []
        and properties.get("diagnostics", {}).get("default") == {},
        f"{label} warning/diagnostic defaults drifted",
    )


def validate_replay_outcome_schema(schema: Any, label: str) -> None:
    required = {
        "schema_version",
        "outcome_id",
        "run_id",
        "bundle_id",
        "plan_id",
        "phase",
        "result_count",
        "lineage_record_count",
        "prediction_block_count",
        "aggregated_prediction_block_count",
        "explanation_block_count",
        "controller_count",
        "prediction_cache_store",
        "outputs",
        "explanations",
        "lineage",
        "warnings",
        "diagnostics",
        "outcome_fingerprint",
    }
    properties, _defs = validate_versioned_object_schema(
        schema, REPLAY_OUTCOME_SCHEMA_ID, required, label
    )
    require(
        properties.get("warnings", {}).get("default") == []
        and properties.get("diagnostics", {}).get("default") == {},
        f"{label} warning/diagnostic defaults drifted",
    )


def validate_parameter_patch(
    value: Any, label: str
) -> tuple[str, str, tuple[str, ...]]:
    patch = require_exact_keys(
        value,
        {"schema_version", "node_id", "namespace", "path", "value"},
        set(),
        label,
    )
    require_version_one(patch["schema_version"], label)
    require_identifier(patch["node_id"], f"{label}.node_id")
    require(
        patch["namespace"] in CONFORMAL_PARAMETER_NAMESPACES,
        f"{label}.namespace is invalid",
    )
    path = patch["path"]
    require(isinstance(path, list) and bool(path), f"{label}.path must be non-empty")
    for index, segment in enumerate(path):
        require_non_empty_string(segment, f"{label}.path[{index}]")
        require(segment != "-", f"{label}.path[{index}] append marker is unsupported")
    validate_strict_json_value(patch["value"], f"{label}.value")
    return patch["node_id"], patch["namespace"], tuple(path)


def output_binding_fingerprint(value: dict[str, Any]) -> str:
    normalized = _norm_output_binding(value)
    return dagml_tcv1_sha256(
        {
            key: member
            for key, member in normalized.items()
            if key != "binding_fingerprint"
        }
    )


def validate_output_binding(
    value: Any,
    label: str,
    *,
    conformal_v1: bool = False,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "binding_id",
        "node_id",
        "port_name",
        "prediction_level",
        "unit_level",
        "prediction_kind",
        "prediction_source",
        "refit_strategy",
        "aggregation_fingerprint",
        "target_names",
        "target_units",
        "class_labels",
        "output_order",
        "target_space",
        "binding_fingerprint",
    }
    output = require_exact_keys(value, fields, set(), label)
    require_version_one(output["schema_version"], label)
    for field in ("binding_id", "node_id"):
        require_identifier(output[field], f"{label}.{field}")
    require_non_empty_string(output["port_name"], f"{label}.port_name")
    require(
        output["prediction_level"] in PREDICTION_LEVELS,
        f"{label}.prediction_level is invalid",
    )
    require_optional_unit_level(output["unit_level"], f"{label}.unit_level")
    if output["prediction_level"] == "sample":
        require(
            output["unit_level"] == "physical_sample",
            f"{label}.unit_level must be physical_sample for sample predictions",
        )
    elif output["prediction_level"] in {"target", "group"}:
        require(
            output["unit_level"] is None,
            f"{label}.unit_level must be null for target/group predictions",
        )
    require(
        output["prediction_kind"]
        in {"regression_point", "class_label", "class_probability", "decision_score"},
        f"{label}.prediction_kind is invalid",
    )
    require(
        output["prediction_source"] in {"final_refit", "cv_ensemble", "fold_member"},
        f"{label}.prediction_source is invalid",
    )
    if output["prediction_source"] == "final_refit":
        require(
            output["refit_strategy"] in {"refit_one", "refit_ensemble"},
            f"{label}.refit_strategy is required for final_refit",
        )
    else:
        require(
            output["refit_strategy"] is None,
            f"{label}.refit_strategy must be null outside final_refit",
        )
    require_sha256(
        output["aggregation_fingerprint"], f"{label}.aggregation_fingerprint"
    )
    targets = validate_ordered_unique_strings(
        output["target_names"], f"{label}.target_names", require_non_empty=True
    )
    units = output["target_units"]
    classes = output["class_labels"]
    require(
        isinstance(units, list) and len(units) == len(targets),
        f"{label}.target_units must match target_names length",
    )
    require(
        isinstance(classes, list) and len(classes) == len(targets),
        f"{label}.class_labels must match target_names length",
    )
    for index, unit in enumerate(units):
        require_optional_non_empty_string(unit, f"{label}.target_units[{index}]")
    for index, labels in enumerate(classes):
        validate_ordered_unique_strings(
            labels, f"{label}.class_labels[{index}]", require_non_empty=False
        )
    if output["prediction_kind"] == "class_probability":
        require(
            all(bool(labels) for labels in classes),
            f"{label}.class_probability requires every class vocabulary",
        )
        require(
            output["output_order"] == "target_major_class_minor",
            f"{label}.class_probability requires target_major_class_minor order",
        )
    else:
        require(
            output["output_order"] == "target_order",
            f"{label}.output_order must be target_order",
        )
        if output["prediction_kind"] == "regression_point":
            require(
                all(not labels for labels in classes),
                f"{label}.regression_point class vocabularies must be empty",
            )
    require_non_empty_string(output["target_space"], f"{label}.target_space")
    require_sha256(output["binding_fingerprint"], f"{label}.binding_fingerprint")
    require(
        output["binding_fingerprint"] == output_binding_fingerprint(output),
        f"{label}.binding_fingerprint does not match TCV1 binding content",
    )
    if conformal_v1:
        require(
            output["prediction_level"] == "sample",
            f"{label}.prediction_level must be sample",
        )
        require(
            output["unit_level"] == "physical_sample",
            f"{label}.unit_level must be physical_sample",
        )
        require(
            output["prediction_kind"] == "regression_point",
            f"{label}.prediction_kind must be regression_point",
        )
        require(
            output["prediction_source"] in CONFORMAL_PREDICTION_SOURCES,
            f"{label}.prediction_source is invalid",
        )
        require(output["target_space"] == "raw", f"{label}.target_space must be raw")
        require(
            all(unit is not None for unit in units),
            f"{label}.target_units must be explicit for conformal V1",
        )
    return output


def expected_output_columns(binding: dict[str, Any]) -> list[str]:
    if binding["prediction_kind"] == "class_probability":
        return [
            f"{target_name}:{class_label}"
            for target_name, class_labels in zip(
                binding["target_names"], binding["class_labels"]
            )
            for class_label in class_labels
        ]
    return list(binding["target_names"])


def validate_prediction_matrix(
    values: Any,
    row_count: int,
    width: int,
    label: str,
) -> None:
    require(
        isinstance(values, list) and len(values) == row_count,
        f"{label} row count does not match identifiers",
    )
    for row_index, row in enumerate(values):
        require(
            isinstance(row, list) and len(row) == width,
            f"{label}[{row_index}] width does not match OutputBinding",
        )
        for column_index, number in enumerate(row):
            require(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number),
                f"{label}[{row_index}][{column_index}] must be finite numeric",
            )


def validate_bound_prediction_block(
    value: Any,
    binding: dict[str, Any],
    label: str,
    *,
    kind: str,
) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require_optional_non_empty_string(
        value.get("prediction_id"), f"{label}.prediction_id"
    )
    require(
        value.get("producer_node") == binding["node_id"],
        f"{label}.producer_node must match OutputBinding.node_id",
    )
    require(
        value.get("partition") in {"train", "validation", "test", "final"},
        f"{label}.partition is invalid",
    )
    require_optional_identifier(value.get("fold_id"), f"{label}.fold_id")
    expected_columns = expected_output_columns(binding)
    require(
        value.get("target_names") == expected_columns,
        f"{label}.target_names must match OutputBinding column order",
    )

    if kind == "prediction":
        identifiers = value.get("sample_ids")
    elif kind == "observation":
        identifiers = value.get("observation_ids")
    else:
        require(kind == "aggregated", f"{label} block kind is invalid")
        level = value.get("level")
        require(
            level == binding["prediction_level"],
            f"{label}.level must match OutputBinding.prediction_level",
        )
        units = value.get("unit_ids")
        require(
            isinstance(units, list) and bool(units),
            f"{label}.unit_ids must be non-empty",
        )
        identifiers = []
        for unit_index, unit in enumerate(units):
            unit_label = f"{label}.unit_ids[{unit_index}]"
            require_exact_keys(unit, {"level", "id"}, set(), unit_label)
            require(
                unit["level"] == level, f"{unit_label}.level must match block level"
            )
            require_identifier(unit["id"], f"{unit_label}.id")
            identifiers.append(unit["id"])

    require(
        isinstance(identifiers, list) and bool(identifiers),
        f"{label} identifiers must be non-empty",
    )
    for identifier_index, identifier in enumerate(identifiers):
        require_identifier(identifier, f"{label}.identifiers[{identifier_index}]")
    require(
        len(set(identifiers)) == len(identifiers),
        f"{label} identifiers contain duplicates",
    )
    validate_prediction_matrix(
        value.get("values"),
        len(identifiers),
        len(expected_columns),
        f"{label}.values",
    )
    if kind == "observation" and "weights" in value:
        weights = value["weights"]
        require(
            isinstance(weights, list) and len(weights) == len(identifiers),
            f"{label}.weights must match observation count",
        )
        for weight_index, weight in enumerate(weights):
            require(
                isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(weight)
                and weight > 0,
                f"{label}.weights[{weight_index}] must be finite and positive",
            )


def validate_bound_output(value: Any, label: str) -> dict[str, Any]:
    output = require_exact_keys(
        value,
        {"binding", "predictions", "observation_predictions", "aggregated_predictions"},
        set(),
        label,
    )
    binding = validate_output_binding(output["binding"], f"{label}.binding")
    block_groups = (
        ("predictions", "prediction"),
        ("observation_predictions", "observation"),
        ("aggregated_predictions", "aggregated"),
    )
    block_count = 0
    for field, kind in block_groups:
        blocks = output[field]
        require(isinstance(blocks, list), f"{label}.{field} must be an array")
        block_count += len(blocks)
        for index, block in enumerate(blocks):
            validate_bound_prediction_block(
                block,
                binding,
                f"{label}.{field}[{index}]",
                kind=kind,
            )
            if binding["prediction_source"] == "final_refit":
                require(
                    block["partition"] == "final" and block["fold_id"] is None,
                    f"{label}.{field}[{index}] final_refit blocks must use final/no-fold",
                )
    require(block_count > 0, f"{label} must contain at least one prediction block")
    if binding["prediction_level"] == "observation":
        require(
            bool(output["observation_predictions"]),
            f"{label} observation binding requires observation predictions",
        )
    elif binding["prediction_level"] in {"target", "group"}:
        require(
            bool(output["aggregated_predictions"]),
            f"{label} target/group binding requires aggregated predictions",
        )
    return output


def execution_plan_graph_nodes(
    plan: dict[str, Any], label: str
) -> dict[str, dict[str, Any]]:
    graph_plan = plan.get("graph_plan")
    require(isinstance(graph_plan, dict), f"{label}.graph_plan must be an object")
    graph = graph_plan.get("graph")
    require(isinstance(graph, dict), f"{label}.graph_plan.graph must be an object")
    nodes = graph.get("nodes")
    require(isinstance(nodes, list), f"{label}.graph_plan.graph.nodes must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        node_label = f"{label}.graph_plan.graph.nodes[{index}]"
        require(isinstance(node, dict), f"{node_label} must be an object")
        node_id = node.get("id")
        require_identifier(node_id, f"{node_label}.id")
        require(
            node_id not in indexed, f"{label} contains duplicate graph node `{node_id}`"
        )
        indexed[node_id] = node
    return indexed


def validate_output_binding_against_plan(
    binding: dict[str, Any],
    plan: dict[str, Any],
    label: str,
) -> None:
    """Bind an OutputBinding to one real prediction output in the frozen plan."""

    nodes = execution_plan_graph_nodes(plan, label)
    node_id = binding["node_id"]
    require(
        node_id in nodes, f"{label}.node_id `{node_id}` is absent from effective plan"
    )
    ports = nodes[node_id].get("ports")
    require(isinstance(ports, dict), f"{label} graph node `{node_id}` has no ports")
    outputs = ports.get("outputs")
    require(
        isinstance(outputs, list), f"{label} graph node `{node_id}` has no output ports"
    )
    matches = [port for port in outputs if port.get("name") == binding["port_name"]]
    require(
        len(matches) == 1,
        f"{label}.port_name `{binding['port_name']}` is not a unique output of `{node_id}`",
    )
    require(
        matches[0].get("kind") == "prediction",
        f"{label}.port_name `{binding['port_name']}` is not a prediction output",
    )
    node_plans = plan.get("node_plans")
    require(
        isinstance(node_plans, dict) and node_id in node_plans,
        f"{label}.node_id `{node_id}` has no executable node plan",
    )


def graph_edge_adjacency(
    graph: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Return sorted node ids, downstream adjacency and in-degrees from edges.

    Adjacency preserves edge multiplicity and in-degree counts every edge, so a
    multiedge is counted once per occurrence — exactly matching the Rust core's
    ``Graph::topological_order``/``parallel_levels`` construction.
    """

    node_ids = sorted(node["id"] for node in graph.get("nodes", []))
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in graph.get("edges", []):
        source = edge["source"]["node_id"]
        target = edge["target"]["node_id"]
        require(
            source in adjacency and target in indegree,
            "graph edge references an unknown node",
        )
        adjacency[source].append(target)
        indegree[target] += 1
    return node_ids, adjacency, indegree


def graph_canonical_topological_order(graph: dict[str, Any]) -> list[str]:
    """Deterministic lexicographic Kahn order (mirrors ``Graph::topological_order``)."""

    node_ids, adjacency, indegree = graph_edge_adjacency(graph)
    ready = sorted(node_id for node_id in node_ids if indegree[node_id] == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        newly: list[str] = []
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                newly.append(target)
        if newly:
            ready = sorted(ready + newly)
    return order


def graph_canonical_parallel_levels(graph: dict[str, Any]) -> list[list[str]]:
    """Canonical dependency levels (mirrors ``Graph::parallel_levels``)."""

    node_ids, adjacency, indegree = graph_edge_adjacency(graph)
    current = sorted(node_id for node_id in node_ids if indegree[node_id] == 0)
    levels: list[list[str]] = []
    while current:
        levels.append(list(current))
        nxt: list[str] = []
        for node_id in current:
            for target in adjacency[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    nxt.append(target)
        current = sorted(nxt)
    return levels


def graph_upstream_node_ids(graph: dict[str, Any], node_id: str) -> list[str]:
    """Sorted, de-duplicated direct predecessors of ``node_id`` from edges."""

    return sorted(
        {
            edge["source"]["node_id"]
            for edge in graph.get("edges", [])
            if edge["target"]["node_id"] == node_id
        }
    )


def graph_downstream_node_ids(graph: dict[str, Any], node_id: str) -> list[str]:
    """Sorted, de-duplicated direct successors of ``node_id`` from edges."""

    return sorted(
        {
            edge["target"]["node_id"]
            for edge in graph.get("edges", [])
            if edge["source"]["node_id"] == node_id
        }
    )


def execution_plan_transitive_node_ids(
    plan: dict[str, Any],
    output_node_ids: set[str],
    label: str,
) -> set[str]:
    """Return the predictor closure for bound outputs from real graph edges.

    The walk uses the graph adjacency (never the serialized ``input_nodes``
    node-plan copies), so a doctored copy cannot silently change the closure.
    ``validate_execution_plan`` separately proves each copy equals the adjacency.
    """

    node_plans = plan.get("node_plans")
    require(isinstance(node_plans, dict), f"{label}.node_plans must be an object")
    graph = plan.get("graph_plan", {}).get("graph", {})
    incoming: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        incoming.setdefault(edge["target"]["node_id"], []).append(
            edge["source"]["node_id"]
        )
    pending = list(output_node_ids)
    closure: set[str] = set()
    while pending:
        node_id = pending.pop()
        require(
            node_id in node_plans,
            f"{label} predictor closure node `{node_id}` has no executable node plan",
        )
        if node_id in closure:
            continue
        closure.add(node_id)
        pending.extend(incoming.get(node_id, []))
    return closure


def derive_replayable_phases(
    plan: dict[str, Any],
    closure: set[str],
    refit_completed: bool,
    bundle: dict[str, Any],
    portable_caches: dict[str, Any] | None,
    label: str,
) -> list[str]:
    """Derive the honest replayable phases from plan/manifests/bundle state.

    Retained inference state is required only for a ``stateful`` or
    ``emits_artifacts`` node (``ArtifactPolicy::ReplayRequired`` alone does NOT
    imply retained state, and ``fit_scope`` is never consulted). A completed
    refit exposes forward inference (PREDICT then EXPLAIN, canonical order) only
    when every closure node supports the phase via its ControllerManifest and
    every state-retaining closure node has a retained refit artifact; it never
    re-advertises REFIT. A skipped refit exposes exactly REFIT when every closure
    node supports REFIT and every in-closure ``requires_oof`` edge is backed by a
    bundle requirement, a cache record and a portable payload. ``[]`` is a valid,
    honest "no replay mode" answer. Support/capabilities are read from the
    manifests, not from the (independently verified) node-plan copies.
    """

    node_plans = plan["node_plans"]
    manifests = plan["controller_manifests"]
    graph = plan["graph_plan"]["graph"]

    def manifest_for(node_id: str) -> dict[str, Any]:
        return manifests[node_plans[node_id]["controller_id"]]

    def all_support(phase: str) -> bool:
        return all(
            phase in manifest_for(node_id)["supported_phases"] for node_id in closure
        )

    artifact_nodes = {record["node_id"] for record in bundle.get("refit_artifacts", [])}
    inference_state_present = all(
        (
            "stateful" not in manifest_for(node_id)["capabilities"]
            and "emits_artifacts" not in manifest_for(node_id)["capabilities"]
        )
        or node_id in artifact_nodes
        for node_id in closure
    )

    requirement_keys = {
        bundle_prediction_requirement_key(
            requirement["producer_node"],
            requirement["source_port"],
            requirement["consumer_node"],
            requirement["target_port"],
        )
        for requirement in bundle.get("prediction_requirements", [])
    }
    cache_keys = {
        record["requirement_key"] for record in bundle.get("prediction_caches", [])
    }
    payload_keys: set[str] = set()
    if portable_caches is not None:
        payload_keys = {
            payload["requirement_key"] for payload in portable_caches.get("caches", [])
        }
    oof_self_contained = True
    for edge in graph.get("edges", []):
        if edge["contract"].get("requires_oof") is not True:
            continue
        source = edge["source"]["node_id"]
        target = edge["target"]["node_id"]
        if source not in closure or target not in closure:
            continue
        key = bundle_prediction_requirement_key(
            source,
            edge["source"]["port_name"],
            target,
            edge["target"]["port_name"],
        )
        if not (key in requirement_keys and key in cache_keys and key in payload_keys):
            oof_self_contained = False

    phases: list[str] = []
    if refit_completed:
        if all_support("PREDICT") and inference_state_present:
            phases.append("PREDICT")
        if all_support("EXPLAIN") and inference_state_present:
            phases.append("EXPLAIN")
    elif all_support("REFIT") and oof_self_contained:
        phases.append("REFIT")
    return phases


def bundle_prediction_requirement_key(
    producer_node: str,
    source_port: str,
    consumer_node: str,
    target_port: str,
) -> str:
    """Mirror ``bundle::bundle_prediction_requirement_key`` exactly."""

    return f"{producer_node}.{source_port}->{consumer_node}.{target_port}"


def selected_variant_parameter_patches(
    variant: dict[str, Any], label: str
) -> list[dict[str, Any]]:
    """Materialize the selected variant's operator overrides as leaf patches."""

    choices = variant.get("choices")
    require(isinstance(choices, dict), f"{label}.choices must be an object")
    patches: list[dict[str, Any]] = []

    def append_leaves(node_id: str, value: Any, path: list[str]) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                require_non_empty_string(key, f"{label} parameter override key")
                append_leaves(node_id, value[key], [*path, key])
            return
        require(bool(path), f"{label} parameter override must contain a leaf value")
        validate_strict_json_value(value, f"{label} parameter override value")
        patches.append(
            {
                "schema_version": 1,
                "node_id": node_id,
                "namespace": "operator",
                "path": path,
                "value": value,
            }
        )

    for choice_name in sorted(choices):
        choice = choices[choice_name]
        require(
            isinstance(choice, dict),
            f"{label}.choices[{choice_name}] must be an object",
        )
        overrides = choice.get("param_overrides")
        require(
            isinstance(overrides, list),
            f"{label}.choices[{choice_name}].param_overrides must be an array",
        )
        for index, override in enumerate(overrides):
            override_label = f"{label}.choices[{choice_name}].param_overrides[{index}]"
            require(isinstance(override, dict), f"{override_label} must be an object")
            node_id = override.get("node_id")
            require_identifier(node_id, f"{override_label}.node_id")
            params = override.get("params")
            require(
                isinstance(params, dict), f"{override_label}.params must be an object"
            )
            append_leaves(node_id, params, [])

    patches.sort(
        key=lambda patch: (patch["node_id"], patch["namespace"], patch["path"])
    )
    keys = [
        (patch["node_id"], patch["namespace"], tuple(patch["path"]))
        for patch in patches
    ]
    require(
        len(set(keys)) == len(keys),
        f"{label} selected parameter overrides contain duplicate leaf paths",
    )
    return patches


def expected_fitting_influence_kinds(
    plan: dict[str, Any], closure: set[str], label: str
) -> dict[str, str]:
    """Map every FIT_CV node in the predictor closure to its influence kind."""

    nodes = execution_plan_graph_nodes(plan, label)
    node_plans = plan["node_plans"]
    oof_consumers = {
        edge["target"]["node_id"]
        for edge in plan["graph_plan"]["graph"]["edges"]
        if edge["contract"].get("requires_oof") is True
    }
    expected: dict[str, str] = {}
    for node_id in closure:
        supported_phases = node_plans[node_id].get("supported_phases")
        require(
            isinstance(supported_phases, list),
            f"{label}.node_plans[{node_id}].supported_phases must be an array",
        )
        if "FIT_CV" not in supported_phases:
            continue
        if node_plans[node_id].get("fit_scope") in {"stateless", "inference_only"}:
            continue
        if node_id in oof_consumers or "trains_aggregation" in node_plans[node_id].get(
            "controller_capabilities", []
        ):
            expected[node_id] = "trained_meta_aggregation"
        elif nodes[node_id].get("kind") == "model":
            expected[node_id] = "model_fit"
        elif nodes[node_id].get("kind") == "tuner":
            expected[node_id] = "hpo_selection"
        else:
            expected[node_id] = "transform_fit"
    return expected


def validate_training_influence_against_plan(
    influence: dict[str, Any],
    plan: dict[str, Any],
    closure: set[str],
    label: str,
) -> None:
    nodes = execution_plan_graph_nodes(plan, label)
    node_plans = plan.get("node_plans")
    require(isinstance(node_plans, dict), f"{label}.node_plans must be an object")
    actual_fit_entries: dict[str, list[str]] = {}
    for index, entry in enumerate(influence["entries"]):
        node_id = entry["node_id"]
        if node_id is None:
            continue
        entry_label = f"{label}.training_influence.entries[{index}]"
        require(
            node_id in nodes, f"{entry_label}.node_id `{node_id}` is absent from graph"
        )
        require(
            node_id in node_plans,
            f"{entry_label}.node_id `{node_id}` has no executable node plan",
        )
        require(
            node_id in closure,
            f"{entry_label}.node_id `{node_id}` is outside the predictor closure",
        )
        if entry["kind"] in {
            "transform_fit",
            "model_fit",
            "hpo_selection",
            "trained_meta_aggregation",
        }:
            actual_fit_entries.setdefault(node_id, []).append(entry["kind"])

    expected = expected_fitting_influence_kinds(plan, closure, label)
    require(
        set(actual_fit_entries) == set(expected),
        f"{label}.training_influence fitting nodes do not exactly match predictor closure",
    )
    for node_id, expected_kind in expected.items():
        kinds = actual_fit_entries[node_id]
        require(
            bool(kinds) and all(kind == expected_kind for kind in kinds),
            f"{label}.training_influence node `{node_id}` entries must all have "
            f"expected kind `{expected_kind}`",
        )


def validate_portable_lineage_record(
    value: Any,
    label: str,
    *,
    run_id: str,
    allowed_phases: set[str],
) -> dict[str, Any]:
    fields = {
        "record_id",
        "run_id",
        "node_id",
        "phase",
        "controller_id",
        "controller_version",
        "variant_id",
        "fold_id",
        "branch_path",
        "input_lineage",
        "artifact_refs",
        "params_fingerprint",
        "data_model_shape_fingerprint",
        "aggregation_policy_fingerprint",
        "seed",
        "unsafe_flags",
        "metrics",
    }
    record = require_exact_keys(value, fields, set(), label)
    for field in ("record_id", "run_id", "node_id", "controller_id"):
        require_identifier(record[field], f"{label}.{field}")
    require(record["run_id"] == run_id, f"{label}.run_id does not match outcome")
    require(record["phase"] in allowed_phases, f"{label}.phase is invalid for outcome")
    require_non_empty_string(
        record["controller_version"], f"{label}.controller_version"
    )
    require_optional_identifier(record["variant_id"], f"{label}.variant_id")
    require_optional_identifier(record["fold_id"], f"{label}.fold_id")
    require_sha256(record["params_fingerprint"], f"{label}.params_fingerprint")
    for field in ("data_model_shape_fingerprint", "aggregation_policy_fingerprint"):
        if record[field] is not None:
            require_sha256(record[field], f"{label}.{field}")
    if record["seed"] is not None:
        require_non_negative_int(record["seed"], f"{label}.seed")
        require(
            record["seed"] <= CONFORMAL_INT_MAX,
            f"{label}.seed exceeds the native u64 maximum",
        )
    for field in ("branch_path", "input_lineage"):
        members = record[field]
        require(isinstance(members, list), f"{label}.{field} must be an array")
        for index, member in enumerate(members):
            require_identifier(member, f"{label}.{field}[{index}]")
    validate_ordered_unique_strings(
        record["unsafe_flags"], f"{label}.unsafe_flags", require_non_empty=False
    )
    validate_metadata_object(record["metrics"], f"{label}.metrics")
    artifacts = record["artifact_refs"]
    require(isinstance(artifacts, list), f"{label}.artifact_refs must be an array")
    artifact_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        artifact_label = f"{label}.artifact_refs[{index}]"
        require(isinstance(artifact, dict), f"{artifact_label} must be an object")
        for field in ("id", "controller_id"):
            require_identifier(artifact.get(field), f"{artifact_label}.{field}")
        require_non_empty_string(artifact.get("kind"), f"{artifact_label}.kind")
        require(
            artifact["id"] not in artifact_ids, f"{artifact_label}.id is duplicated"
        )
        artifact_ids.add(artifact["id"])
        if artifact.get("content_fingerprint") is not None:
            require_sha256(
                artifact["content_fingerprint"],
                f"{artifact_label}.content_fingerprint",
            )
        require_optional_non_empty_string(
            artifact.get("plugin"), f"{artifact_label}.plugin"
        )
        require_optional_non_empty_string(
            artifact.get("plugin_version"), f"{artifact_label}.plugin_version"
        )
        require(
            artifact.get("plugin_version") is None
            or artifact.get("plugin") is not None,
            f"{artifact_label}.plugin_version requires plugin",
        )
    return record


def validate_training_lineage_against_plan(
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    predictor_closure: set[str],
    bundle: dict[str, Any],
    *,
    refit_requested: bool,
    label: str,
) -> None:
    """Require phase-complete, cross-linked lineage for the predictor closure."""

    node_plans = plan["node_plans"]
    record_ids = [record["record_id"] for record in records]
    require(
        record_ids == sorted(record_ids), f"{label}.lineage must be sorted by record_id"
    )
    require(
        len(set(record_ids)) == len(record_ids),
        f"{label}.lineage record_id is duplicated",
    )
    records_by_id = {record["record_id"]: record for record in records}
    records_by_coordinate: dict[tuple[str, str | None, str], dict[str, Any]] = {}

    for index, record in enumerate(records):
        record_label = f"{label}.lineage[{index}]"
        node_id = record["node_id"]
        require(
            node_id in predictor_closure,
            f"{record_label}.node_id is outside the predictor closure",
        )
        node_plan = node_plans[node_id]
        require(
            record["controller_id"] == node_plan["controller_id"],
            f"{record_label}.controller_id does not match node plan",
        )
        require(
            record["controller_version"] == node_plan["controller_version"],
            f"{record_label}.controller_version does not match node plan",
        )
        require(
            record["params_fingerprint"] == node_plan["params_fingerprint"],
            f"{record_label}.params_fingerprint does not match node plan",
        )
        coordinate = (record["phase"], record["fold_id"], node_id)
        require(
            coordinate not in records_by_coordinate,
            f"{record_label} duplicates phase/fold/node lineage",
        )
        records_by_coordinate[coordinate] = record
        for input_record_id in record["input_lineage"]:
            require(
                input_record_id in records_by_id,
                f"{record_label}.input_lineage references unknown record "
                f"`{input_record_id}`",
            )

    fold_set = plan.get("fold_set")
    require(isinstance(fold_set, dict), f"{label} FIT_CV lineage requires a fold_set")
    fold_ids = [fold["fold_id"] for fold in fold_set["folds"]]
    fit_nodes = {
        node_id
        for node_id in predictor_closure
        if "FIT_CV" in node_plans[node_id].get("supported_phases", [])
    }
    expected_fit_coordinates = {
        ("FIT_CV", fold_id, node_id) for node_id in fit_nodes for fold_id in fold_ids
    }
    actual_fit_coordinates = {
        coordinate for coordinate in records_by_coordinate if coordinate[0] == "FIT_CV"
    }
    require(
        actual_fit_coordinates == expected_fit_coordinates,
        f"{label}.lineage FIT_CV records do not exactly cover every "
        "predictor-closure node/fold",
    )

    refit_nodes = {
        node_id
        for node_id in predictor_closure
        if "REFIT" in node_plans[node_id].get("supported_phases", [])
    }
    expected_refit_coordinates = (
        {("REFIT", None, node_id) for node_id in refit_nodes}
        if refit_requested
        else set()
    )
    actual_refit_coordinates = {
        coordinate for coordinate in records_by_coordinate if coordinate[0] == "REFIT"
    }
    require(
        actual_refit_coordinates == expected_refit_coordinates,
        f"{label}.lineage REFIT records do not exactly match the predictor closure",
    )

    bundle_artifacts_by_node: dict[str, list[dict[str, Any]]] = {}
    for artifact_record in bundle["refit_artifacts"]:
        bundle_artifacts_by_node.setdefault(artifact_record["node_id"], []).append(
            artifact_record["artifact"]
        )
    for artifacts in bundle_artifacts_by_node.values():
        artifacts.sort(key=lambda artifact: artifact["id"])

    for coordinate, record in records_by_coordinate.items():
        phase, fold_id, node_id = coordinate
        record_label = f"{label}.lineage[{record['record_id']}]"
        node_plan = node_plans[node_id]
        if phase == "FIT_CV":
            require(fold_id is not None, f"{record_label}.fold_id must be non-null")
            require(
                not record["artifact_refs"],
                f"{record_label} FIT_CV lineage cannot persist refit artifacts",
            )
        elif phase == "REFIT":
            require(fold_id is None, f"{record_label}.fold_id must be null")
            expected_artifacts = bundle_artifacts_by_node.get(node_id, [])
            actual_artifacts = sorted(
                record["artifact_refs"], key=lambda artifact: artifact["id"]
            )
            require(
                actual_artifacts == expected_artifacts,
                f"{record_label}.artifact_refs do not exactly match execution bundle",
            )

        if phase not in {"FIT_CV", "REFIT"}:
            continue
        expected_input_lineage = sorted(
            records_by_coordinate[(phase, fold_id, input_node)]["record_id"]
            for input_node in node_plan["input_nodes"]
            if phase in node_plans[input_node].get("supported_phases", [])
        )
        require(
            record["input_lineage"] == expected_input_lineage,
            f"{record_label}.input_lineage does not exactly match upstream "
            "phase/fold lineage",
        )


def contains_runtime_handle(value: Any) -> bool:
    if isinstance(value, list):
        return any(contains_runtime_handle(member) for member in value)
    if not isinstance(value, dict):
        return False
    for key in value:
        lowered = key.lower()
        if (
            lowered == "handle"
            or lowered.endswith("_handle")
            or lowered.endswith("_handles")
        ):
            return True
    return any(contains_runtime_handle(member) for member in value.values())


def training_outcome_fingerprint(value: dict[str, Any]) -> str:
    return dagml_tcv1_sha256(
        {key: member for key, member in value.items() if key != "outcome_fingerprint"}
    )


def replay_outcome_fingerprint(value: dict[str, Any]) -> str:
    return training_outcome_fingerprint(value)


def legacy_serde_json_sha256(value: Any) -> str:
    """Reproduce stable_json_fingerprint for values normalized to Rust field order."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Independent serde_json-compatible struct fingerprints.
#
# The ExecutionPlan embeds three "historical serde" fingerprints — graph,
# campaign and controller_manifests — each defined as ``SHA-256`` of
# ``serde_json::to_vec`` of the typed Rust value: compact UTF-8 JSON in Rust
# *struct field order* (structs) and *BTreeMap key order* (maps). Rather than
# trusting the fixture's on-disk key order (a forged reordering would then hash
# identically), we re-serialize from a type-aware normalization that rebuilds
# every struct in its declared field order, sorts every BTreeMap / serde_json
# Value object key, injects the Rust serde defaults for skipped fields, and
# formats floats exactly like serde_json 1.0.150. Node-plan ``params`` are a
# ``BTreeMap<String, Value>`` and hash through the same serializer over their
# recursively key-sorted form. This block is duplicated verbatim in
# ``parity/training/oracle.py`` for oracle independence.
# ---------------------------------------------------------------------------

_MISSING = object()


def _serde_float(value: float) -> str:
    """Format a binary64 exactly like serde_json (shortest round-trip digits).

    Uses ``repr(abs(value))`` for the shortest significant digits, derives the
    decimal (scientific) exponent, and emits fixed notation when that exponent
    is within ``[-5, 15]`` inclusive and scientific notation otherwise, with an
    explicit ``+`` on positive exponents, no leading exponent zeros, and a
    preserved ``-0.0``.
    """

    if not math.isfinite(value):
        raise ValueError("serde_json cannot encode a non-finite float")
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0.0 else "0.0"
    sign = "-" if value < 0.0 else ""
    text = repr(abs(value))
    if "e" in text or "E" in text:
        mantissa, _, exponent_text = text.replace("E", "e").partition("e")
        exp10 = int(exponent_text)
    else:
        mantissa, exp10 = text, 0
    integer, _, fraction = mantissa.partition(".")
    combined = integer + fraction
    stripped = combined.lstrip("0")
    leading_zeros = len(combined) - len(stripped)
    digits = stripped.rstrip("0") or "0"
    exponent = exp10 - len(fraction) + len(combined) - 1 - leading_zeros
    if -5 <= exponent <= 15:
        width = len(digits)
        if exponent >= 0:
            if exponent + 1 >= width:
                body = digits + "0" * (exponent + 1 - width) + ".0"
            else:
                body = digits[: exponent + 1] + "." + digits[exponent + 1 :]
        else:
            body = "0." + "0" * (-exponent - 1) + digits
    else:
        mantissa = digits if len(digits) == 1 else digits[0] + "." + digits[1:]
        body = f"{mantissa}e{'+' if exponent > 0 else '-'}{abs(exponent)}"
    return sign + body


def _serde_encode(value: Any) -> bytes:
    """Serialize a pre-normalized value to serde_json's compact byte form."""

    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int):
        return str(value).encode("utf-8")
    if isinstance(value, float):
        return _serde_float(value).encode("utf-8")
    if isinstance(value, str):
        return encode_basestring(value).encode("utf-8")
    if isinstance(value, list):
        return b"[" + b",".join(_serde_encode(item) for item in value) + b"]"
    if isinstance(value, dict):
        members = []
        for key, member in value.items():
            if not isinstance(key, str):
                raise TypeError("serde_json object keys must be strings")
            members.append(
                encode_basestring(key).encode("utf-8") + b":" + _serde_encode(member)
            )
        return b"{" + b",".join(members) + b"}"
    raise TypeError(f"serde_json cannot encode {type(value).__name__}")


def _serde_sha256(value: Any) -> str:
    return hashlib.sha256(_serde_encode(value)).hexdigest()


def _V(value: Any) -> Any:
    """Recursively sort every object key (a serde_json::Value / BTreeMap value)."""

    if isinstance(value, dict):
        return {key: _V(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_V(item) for item in value]
    # serde_json without arbitrary_precision stores integer tokens outside its
    # i64/u64 domains as finite f64 Numbers. Signed TCV1 parents reject those
    # raw integers before this normalizer runs; standalone typed contracts do not.
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and not (-(2**63) <= value <= 2**64 - 1)
    ):
        try:
            converted = float(value)
        except OverflowError as error:
            raise ContractError(
                "integer is outside serde_json's finite number range"
            ) from error
        require(
            math.isfinite(converted),
            "integer is outside serde_json's finite number range",
        )
        return converted
    return value


def _BM(mapping: Any, value_normalizer=lambda item: item) -> dict:
    """Normalize a BTreeMap: keys sorted lexicographically, values transformed."""

    require(isinstance(mapping, dict), "typed serde map must be an object")
    return {key: value_normalizer(mapping[key]) for key in sorted(mapping)}


def _L(values: Any, value_normalizer=lambda item: item) -> list:
    """Normalize a Rust Vec without accepting a wrong JSON container type."""

    require(isinstance(values, list), "typed serde Vec must be an array")
    return [value_normalizer(value) for value in values]


def _T2(values: Any, value_normalizer=lambda item: item) -> list:
    """Normalize a Rust two-tuple encoded as a two-element JSON array."""

    require(
        isinstance(values, list) and len(values) == 2,
        "typed serde pair must be a two-element array",
    )
    return [value_normalizer(value) for value in values]


def _S(source: Any, fields: list) -> dict:
    """Reconstruct a typed struct in declared field order.

    Each field spec is ``{"name", "default"?, "transform"?, "skip"?}``: an
    absent source field falls back to its Rust serde default, an optional
    ``transform`` normalizes the value, and a ``skip`` predicate reproduces
    ``skip_serializing_if`` (drop None / false / empty / default).
    """

    require(isinstance(source, dict), "typed serde struct must be an object")
    result: dict[str, Any] = {}
    for field in fields:
        name = field["name"]
        raw = source.get(name, _MISSING)
        if raw is _MISSING:
            raw = field["default"]() if "default" in field else None
        value = field["transform"](raw) if "transform" in field else raw
        skip = field.get("skip")
        if skip is not None and skip(value):
            continue
        result[name] = value
    return result


def _skip_none(value: Any) -> bool:
    return value is None


def _skip_false(value: Any) -> bool:
    return value is False


def _skip_empty(value: Any) -> bool:
    return not value


def _sorted_set(values: Any) -> list:
    require(isinstance(values, list), "typed serde set must be an array")
    return sorted(set(values))


def _sorted_enum_set(values: Any, order: dict[str, int]) -> list:
    """Normalize a Rust ``BTreeSet<Enum>`` in the enum's derived Ord order."""

    require(isinstance(values, list), "typed serde enum set must be an array")
    return sorted(set(values), key=order.__getitem__)


def _norm_port(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "name"},
            {"name": "kind"},
            {"name": "representation", "default": lambda: None},
            {"name": "cardinality"},
            {"name": "unit_level", "skip": _skip_none},
            {"name": "alignment_key", "skip": _skip_none},
            {"name": "target_level", "skip": _skip_none},
            {"name": "description", "default": lambda: ""},
        ],
    )


def _norm_ports(source: Any) -> list:
    return _L(source, _norm_port)


def _norm_port_schema(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "inputs", "default": lambda: [], "transform": _norm_ports},
            {"name": "outputs", "default": lambda: [], "transform": _norm_ports},
        ],
    )


def _norm_relation_contract(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "relation_fingerprint", "skip": _skip_none},
            {"name": "required", "default": lambda: False, "skip": _skip_false},
        ],
    )


def _norm_edge_contract(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "kind"},
            {"name": "representation", "default": lambda: None},
            {"name": "unit_level", "skip": _skip_none},
            {"name": "alignment_key", "skip": _skip_none},
            {"name": "target_level", "skip": _skip_none},
            {
                "name": "relation_contract",
                "transform": _norm_relation_contract,
                "skip": _skip_none,
            },
            {"name": "allows_broadcast", "default": lambda: False, "skip": _skip_false},
            {"name": "missingness_policy", "skip": _skip_none},
            {"name": "requires_oof", "default": lambda: False},
            {"name": "requires_fold_alignment", "default": lambda: False},
            {"name": "propagates_lineage", "default": lambda: True},
        ],
    )


def _norm_port_ref(source: Any) -> dict:
    return _S(source, [{"name": "node_id"}, {"name": "port_name"}])


def _norm_edge(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "source", "transform": _norm_port_ref},
            {"name": "target", "transform": _norm_port_ref},
            {"name": "contract", "transform": _norm_edge_contract},
        ],
    )


def _norm_node(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "id"},
            {"name": "kind"},
            {"name": "operator", "default": lambda: None, "transform": _V},
            {
                "name": "params",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
            {"name": "ports", "default": lambda: {}, "transform": _norm_port_schema},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
            {"name": "seed_label", "default": lambda: None},
        ],
    )


def _normalize_graph_spec(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "id"},
            {
                "name": "interface",
                "default": lambda: {},
                "transform": _norm_port_schema,
            },
            {
                "name": "nodes",
                "default": lambda: [],
                "transform": lambda nodes: _L(nodes, _norm_node),
            },
            {
                "name": "edges",
                "default": lambda: [],
                "transform": lambda edges: _L(edges, _norm_edge),
            },
            {"name": "search_space_fingerprint", "default": lambda: None},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
        ],
    )


def _norm_leakage_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "split_unit", "default": lambda: "physical_sample"},
            {"name": "forbid_origin_cross_fold", "default": lambda: True},
            {
                "name": "allow_observation_split_with_shared_target",
                "default": lambda: False,
            },
            {"name": "require_group_ids", "default": lambda: False},
            {"name": "unsafe_flags", "default": lambda: [], "transform": _sorted_set},
        ],
    )


def _norm_aggregation_controller(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "controller_id"},
            {"name": "params", "default": lambda: {}, "transform": _V},
        ],
    )


def _norm_aggregation_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "aggregation_level", "default": lambda: "sample"},
            {"name": "method", "default": lambda: "mean"},
            {"name": "weights", "default": lambda: "none"},
            {
                "name": "custom_controller",
                "transform": _norm_aggregation_controller,
                "skip": _skip_none,
            },
            {"name": "emit_parallel_metrics", "default": lambda: True},
            {"name": "selection_metric_level", "default": lambda: "sample"},
            {"name": "store_raw_predictions", "default": lambda: True},
            {"name": "store_aggregated_predictions", "default": lambda: True},
        ],
    )


def _norm_fold_assignment(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "fold_id"},
            {"name": "train_sample_ids", "transform": _L},
            {"name": "validation_sample_ids", "transform": _L},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
        ],
    )


def _norm_fold_set(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "id"},
            {"name": "sample_ids", "transform": _L},
            {
                "name": "folds",
                "default": lambda: [],
                "transform": lambda folds: _L(folds, _norm_fold_assignment),
            },
            {
                "name": "sample_groups",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
            {
                "name": "partition_mode",
                "default": lambda: "partition",
                "skip": lambda value: value == "partition",
            },
        ],
    )


def _norm_nested_cv(source: Any) -> Any:
    if source is None:
        return None
    require(isinstance(source, dict), "typed serde struct must be an object")
    if source.get("kind") == "group_kfold":
        return _S(source, [{"name": "kind"}, {"name": "n_splits"}])
    return _S(
        source,
        [
            {"name": "kind"},
            {"name": "n_splits"},
            {"name": "shuffle", "default": lambda: False},
            {"name": "seed"},
        ],
    )


def _norm_split_invocation(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "id"},
            {"name": "controller_id", "default": lambda: None},
            {
                "name": "leakage_policy",
                "default": lambda: {},
                "transform": _norm_leakage_policy,
            },
            {
                "name": "params",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
            {"name": "fold_set", "default": lambda: None, "transform": _norm_fold_set},
        ],
    )


def _norm_choice_ref(source: Any) -> dict:
    return _S(source, [{"name": "dimension"}, {"name": "label"}])


def _norm_param_override(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {
                "name": "params",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
        ],
    )


def _norm_generation_choice(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "label"},
            {"name": "value", "transform": _V},
            {
                "name": "param_overrides",
                "default": lambda: [],
                "transform": lambda items: _L(items, _norm_param_override),
                "skip": _skip_empty,
            },
            {"name": "active_subsequence", "skip": _skip_none},
        ],
    )


def _norm_generation_dimension(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "name"},
            {
                "name": "choices",
                "default": lambda: [],
                "transform": lambda items: _L(items, _norm_generation_choice),
            },
        ],
    )


def _norm_generation_constraints(source: Any) -> dict:
    return _S(
        source,
        [
            {
                "name": "mutex",
                "default": lambda: [],
                "transform": lambda groups: _L(
                    groups, lambda group: _L(group, _norm_choice_ref)
                ),
                "skip": _skip_empty,
            },
            {
                "name": "requires",
                "default": lambda: [],
                "transform": lambda pairs: _L(
                    pairs, lambda pair: _T2(pair, _norm_choice_ref)
                ),
                "skip": _skip_empty,
            },
            {
                "name": "exclude",
                "default": lambda: [],
                "transform": lambda pairs: _L(
                    pairs, lambda pair: _T2(pair, _norm_choice_ref)
                ),
                "skip": _skip_empty,
            },
        ],
    )


def _norm_generation_spec(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "strategy", "default": lambda: "none"},
            {
                "name": "dimensions",
                "default": lambda: [],
                "transform": lambda items: _L(items, _norm_generation_dimension),
            },
            {"name": "max_variants", "default": lambda: None},
            {
                "name": "constraints",
                "default": lambda: {},
                "transform": _norm_generation_constraints,
                "skip": _skip_empty,
            },
        ],
    )


def _norm_augmentation_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "sample_scope", "default": lambda: "train_only"},
            {"name": "feature_scope", "default": lambda: "train_only"},
            {"name": "require_origin_id", "default": lambda: True},
            {"name": "inherit_group", "default": lambda: True},
            {"name": "inherit_target", "default": lambda: True},
            {
                "name": "unsafe_flags",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
        ],
    )


def _norm_feature_selection_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "scope", "default": lambda: "none"},
            {"name": "store_masks", "default": lambda: True},
            {"name": "allow_schema_mismatch_on_join", "default": lambda: False},
        ],
    )


def _norm_shape_plan(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {"name": "input_granularity", "default": lambda: "sample"},
            {"name": "target_granularity", "default": lambda: "sample"},
            {"name": "fit_rows", "default": lambda: "fold_train"},
            {"name": "predict_rows", "default": lambda: "fold_validation"},
            {"name": "feature_namespace", "default": lambda: None},
            {"name": "feature_schema_fingerprint", "default": lambda: None},
            {"name": "target_space", "default": lambda: "raw"},
            {
                "name": "aggregation_policy",
                "default": lambda: {},
                "transform": _norm_aggregation_policy,
            },
            {
                "name": "augmentation_policy",
                "default": lambda: {},
                "transform": _norm_augmentation_policy,
            },
            {
                "name": "selection_policy",
                "default": lambda: {},
                "transform": _norm_feature_selection_policy,
            },
        ],
    )


def _norm_view_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "fit_partition", "default": lambda: "fold_train"},
            {"name": "predict_partition", "default": lambda: "fold_validation"},
            {"name": "include_augmented_train", "default": lambda: False},
            {"name": "include_augmented_validation", "default": lambda: False},
            {"name": "include_excluded", "default": lambda: False},
            {"name": "require_sample_ids", "default": lambda: True},
            {
                "name": "unsafe_flags",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
        ],
    )


def _norm_data_binding(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {"name": "input_name"},
            {"name": "request_id"},
            {"name": "schema_fingerprint"},
            {"name": "plan_fingerprint"},
            {"name": "relation_fingerprint", "default": lambda: None},
            {"name": "output_representation"},
            {"name": "feature_set_id", "default": lambda: None},
            {"name": "source_ids", "default": lambda: [], "transform": _L},
            {"name": "require_relations", "default": lambda: False},
            {
                "name": "view_policy",
                # DataBinding #[serde(default)] invokes DataViewPolicy::default
                # when the whole block is absent; that custom default enables
                # augmented training. Inside an explicitly present `{}`, the
                # field-level bool default is false.
                "default": lambda: {
                    "fit_partition": "fold_train",
                    "predict_partition": "fold_validation",
                    "include_augmented_train": True,
                    "include_augmented_validation": False,
                    "include_excluded": False,
                    "require_sample_ids": True,
                },
                "transform": _norm_view_policy,
            },
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
        ],
    )


def _norm_data_view_selector(source: Any) -> dict:
    return _S(
        source,
        [
            {
                "name": "source_ids",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
                "skip": _skip_empty,
            },
            {
                "name": "tags",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {"name": "filter", "transform": _V, "skip": _skip_none},
        ],
    )


def _norm_branch_view_plan(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "view_id"},
            {"name": "branch_id"},
            {"name": "mode"},
            {
                "name": "selector",
                "default": lambda: {},
                "transform": _norm_data_view_selector,
            },
            {"name": "allow_overlap", "default": lambda: False},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
                "skip": _skip_empty,
            },
        ],
    )


def _normalize_campaign_spec(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "id"},
            {"name": "root_seed"},
            {
                "name": "leakage_policy",
                "default": lambda: {},
                "transform": _norm_leakage_policy,
            },
            {
                "name": "aggregation_policy",
                "default": lambda: {},
                "transform": _norm_aggregation_policy,
            },
            {
                "name": "split_invocation",
                "default": lambda: None,
                "transform": _norm_split_invocation,
            },
            {
                "name": "generation",
                # CampaignSpec #[serde(default)] calls GenerationSpec::default,
                # whose max_variants is Some(1). An explicitly present `{}`
                # instead uses Option::default (None), so preserve the distinction.
                "default": lambda: {
                    "strategy": "none",
                    "dimensions": [],
                    "max_variants": 1,
                },
                "transform": _norm_generation_spec,
            },
            {
                "name": "shape_plans",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _norm_shape_plan),
            },
            {
                "name": "data_bindings",
                "default": lambda: {},
                "transform": lambda m: _BM(
                    m, lambda bindings: _L(bindings, _norm_data_binding)
                ),
            },
            {
                "name": "branch_view_plans",
                "default": lambda: [],
                "transform": lambda items: _L(items, _norm_branch_view_plan),
                "skip": _skip_empty,
            },
            {"name": "inner_cv", "transform": _norm_nested_cv, "skip": _skip_none},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda m: _BM(m, _V),
            },
        ],
    )


def _norm_operator_selector(source: Any) -> dict:
    return _S(
        source,
        [
            {
                "name": "aliases",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
            {
                "name": "classes",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
            {
                "name": "class_prefixes",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
            {
                "name": "functions",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
            {
                "name": "refs",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
            {
                "name": "types",
                "default": lambda: [],
                "transform": _sorted_set,
                "skip": _skip_empty,
            },
        ],
    )


def _norm_controller_manifest(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "controller_id"},
            {"name": "controller_version"},
            {"name": "operator_kind"},
            {"name": "priority", "default": lambda: 0},
            {
                "name": "supported_phases",
                "default": lambda: [],
                "transform": lambda values: _sorted_enum_set(values, W10_PHASE_ORDER),
            },
            {"name": "input_ports", "default": lambda: [], "transform": _norm_ports},
            {"name": "output_ports", "default": lambda: [], "transform": _norm_ports},
            {"name": "data_requirements", "default": lambda: None, "transform": _V},
            {
                "name": "capabilities",
                "default": lambda: [],
                "transform": lambda values: _sorted_enum_set(
                    values, W10_CAPABILITY_ORDER
                ),
            },
            {
                "name": "operator_selectors",
                "default": lambda: [],
                "transform": lambda items: _L(items, _norm_operator_selector),
                "skip": _skip_empty,
            },
            {"name": "fit_scope"},
            {"name": "rng_policy"},
            {"name": "artifact_policy"},
        ],
    )


def _normalize_controller_manifests(source: Any) -> dict:
    return _BM(source, _norm_controller_manifest)


def _norm_graph_plan(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "graph", "transform": _normalize_graph_spec},
            {"name": "topological_order", "transform": _L},
            {
                "name": "parallel_levels",
                "default": lambda: [],
                "transform": lambda levels: _L(levels, _L),
                "skip": _skip_empty,
            },
        ],
    )


def _norm_optional_shape_plan(source: Any) -> Any:
    return None if source is None else _norm_shape_plan(source)


def _norm_node_plan(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {"name": "kind"},
            {"name": "controller_id"},
            {"name": "controller_version"},
            {
                "name": "supported_phases",
                "transform": lambda values: _sorted_enum_set(values, W10_PHASE_ORDER),
            },
            {
                "name": "controller_capabilities",
                "default": lambda: [],
                "transform": lambda values: _sorted_enum_set(
                    values, W10_CAPABILITY_ORDER
                ),
            },
            {"name": "fit_scope"},
            {"name": "rng_policy"},
            {"name": "artifact_policy"},
            {"name": "input_nodes", "transform": _L},
            {"name": "output_nodes", "transform": _L},
            {"name": "shape_plan", "transform": _norm_optional_shape_plan},
            {
                "name": "data_bindings",
                "default": lambda: [],
                "transform": lambda bindings: _L(bindings, _norm_data_binding),
            },
            {
                "name": "params",
                "default": lambda: {},
                "transform": lambda params: _BM(params, _V),
                "skip": _skip_empty,
            },
            {
                "name": "inner_cv",
                "default": lambda: None,
                "transform": _norm_nested_cv,
                "skip": _skip_none,
            },
            {"name": "params_fingerprint"},
        ],
    )


def _norm_variant_plan(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "variant_id"},
            {
                "name": "choices",
                "default": lambda: {},
                "transform": lambda choices: _BM(choices, _norm_generation_choice),
            },
            {"name": "fingerprint"},
            {"name": "seed"},
        ],
    )


def _normalize_execution_plan(source: Any) -> dict:
    """Rebuild the complete typed ``ExecutionPlan`` serde representation."""

    return _S(
        source,
        [
            {"name": "id"},
            {"name": "graph_plan", "transform": _norm_graph_plan},
            {"name": "campaign", "transform": _normalize_campaign_spec},
            {
                "name": "node_plans",
                "transform": lambda plans: _BM(plans, _norm_node_plan),
            },
            {
                "name": "controller_manifests",
                "transform": _normalize_controller_manifests,
            },
            {
                "name": "variants",
                "transform": lambda variants: _L(variants, _norm_variant_plan),
            },
            {"name": "fold_set", "transform": _norm_fold_set},
            {"name": "graph_fingerprint"},
            {"name": "campaign_fingerprint"},
            {"name": "controller_fingerprint"},
        ],
    )


def _F(value: Any) -> Any:
    """Deserialize one JSON number through a Rust ``f64`` field."""

    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _norm_f64_matrix(values: Any) -> list:
    return _L(values, lambda row: _L(row, _F))


def _norm_training_data_identity(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "requirement_key"},
            {"name": "schema_fingerprint"},
            {"name": "plan_fingerprint"},
            {"name": "relation_fingerprint"},
            {"name": "data_content_fingerprint"},
            {"name": "target_content_fingerprint"},
            {"name": "identity_fingerprint"},
        ],
    )


def _norm_parameter_patch(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "schema_version"},
            {"name": "node_id"},
            {"name": "namespace"},
            {"name": "path", "transform": _L},
            {"name": "value", "transform": _V},
        ],
    )


def _norm_node_patch_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {
                "name": "allowed_namespaces",
                "transform": lambda values: _sorted_enum_set(
                    values, W10_NAMESPACE_ORDER
                ),
            },
        ],
    )


def _norm_influence_requirement(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {"name": "kind"},
            {"name": "scope_id"},
            {"name": "phase"},
            {"name": "fold_id"},
            {"name": "physical_sample_ids", "transform": _L},
        ],
    )


def _norm_selection_metric(source: Any) -> dict:
    return _S(source, [{"name": "name"}, {"name": "objective"}])


def _norm_refit_slot_plan(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "strategy"},
            {"name": "selection_level"},
            {"name": "member_count"},
            {"name": "selection_metric", "transform": _norm_selection_metric},
            {"name": "reduction_id", "skip": _skip_none},
        ],
    )


def _norm_stacking_fit_contract(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "meta_training_features"},
            {"name": "inference_features"},
            {"name": "selection_protocol"},
            {"name": "meta_row_domain"},
            {"name": "final_reduction_id", "skip": _skip_none},
            {"name": "unsafe_allow_reuse_oof", "default": lambda: False},
        ],
    )


def _norm_selection_policy(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "id"},
            {"name": "metric", "transform": _norm_selection_metric},
            {"name": "required_metric_level", "skip": _skip_none},
            {"name": "require_finite", "default": lambda: True},
            {"name": "evaluation_scope", "skip": _skip_none},
            {
                "name": "refit_slot_plan",
                "transform": _norm_refit_slot_plan,
                "skip": _skip_none,
            },
            {
                "name": "stacking_fit_contract",
                "transform": _norm_stacking_fit_contract,
                "skip": _skip_none,
            },
            {"name": "reduction_id", "skip": _skip_none},
        ],
    )


def _norm_training_output_request(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "output_id"},
            {"name": "node_id"},
            {"name": "port_name", "skip": _skip_none},
            {"name": "prediction_level"},
            {"name": "unit_level"},
            {"name": "prediction_kind"},
            {"name": "target_names", "transform": _L},
            {"name": "target_units", "transform": _L},
            {
                "name": "class_labels",
                "transform": lambda values: _L(values, _L),
            },
            {"name": "output_order"},
            {"name": "target_space"},
        ],
    )


def _norm_training_scheduler(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "kind"},
            {"name": "backend", "default": lambda: None},
            {"name": "workers"},
        ],
    )


def _norm_training_resources(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "cpu_threads"},
            {"name": "memory_bytes", "skip": _skip_none},
            {"name": "gpu_devices", "transform": _L},
            {"name": "wall_time_ms", "skip": _skip_none},
        ],
    )


def _norm_training_artifacts(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "cv_artifacts"},
            {"name": "prediction_caches"},
            {"name": "fitted_artifacts"},
        ],
    )


def _norm_training_options(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "refit"},
            {"name": "refit_strategy"},
            {"name": "seed"},
            {"name": "selection", "transform": _norm_selection_policy},
            {"name": "selection_output_id"},
            {
                "name": "outputs",
                "transform": lambda values: _L(values, _norm_training_output_request),
            },
            {"name": "scheduler", "transform": _norm_training_scheduler},
            {"name": "resources", "transform": _norm_training_resources},
            {"name": "artifacts", "transform": _norm_training_artifacts},
        ],
    )


def _normalize_training_request(source: Any) -> dict:
    """Rebuild the complete typed ``TrainingRequest`` serde representation."""

    return _S(
        source,
        [
            {"name": "schema_version"},
            {"name": "request_id"},
            {"name": "plan_id"},
            {"name": "graph", "transform": _normalize_graph_spec},
            {"name": "campaign", "transform": _normalize_campaign_spec},
            {
                "name": "controller_manifests",
                "transform": lambda values: _L(values, _norm_controller_manifest),
            },
            {
                "name": "data_identities",
                "transform": lambda values: _L(values, _norm_training_data_identity),
            },
            {
                "name": "parameter_patches",
                "transform": lambda values: _L(values, _norm_parameter_patch),
            },
            {
                "name": "patch_policies",
                "transform": lambda values: _L(values, _norm_node_patch_policy),
            },
            {
                "name": "influence_requirements",
                "transform": lambda values: _L(values, _norm_influence_requirement),
            },
            {"name": "options", "transform": _norm_training_options},
            {"name": "request_fingerprint"},
        ],
    )


def _norm_output_binding(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "schema_version"},
            {"name": "binding_id"},
            {"name": "node_id"},
            {"name": "port_name"},
            {"name": "prediction_level"},
            {"name": "unit_level", "default": lambda: None},
            {"name": "prediction_kind"},
            {"name": "prediction_source"},
            {"name": "refit_strategy", "default": lambda: None},
            {"name": "aggregation_fingerprint"},
            {"name": "target_names", "transform": _L},
            {"name": "target_units", "transform": _L},
            {
                "name": "class_labels",
                "transform": lambda values: _L(values, _L),
            },
            {"name": "output_order"},
            {"name": "target_space"},
            {"name": "binding_fingerprint"},
        ],
    )


def _norm_prediction_unit(source: Any) -> dict:
    return _S(source, [{"name": "level"}, {"name": "id"}])


def _norm_prediction_block(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "prediction_id", "default": lambda: None},
            {"name": "producer_node"},
            {"name": "partition"},
            {"name": "fold_id"},
            {"name": "sample_ids", "transform": _L},
            {"name": "values", "transform": _norm_f64_matrix},
            {"name": "target_names", "default": lambda: [], "transform": _L},
        ],
    )


def _norm_observation_prediction_block(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "prediction_id", "default": lambda: None},
            {"name": "producer_node"},
            {"name": "partition"},
            {"name": "fold_id"},
            {"name": "observation_ids", "transform": _L},
            {"name": "values", "transform": _norm_f64_matrix},
            {
                "name": "weights",
                "default": lambda: [],
                "transform": lambda values: _L(values, _F),
                "skip": _skip_empty,
            },
            {"name": "target_names", "default": lambda: [], "transform": _L},
        ],
    )


def _norm_aggregated_prediction_block(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "prediction_id", "default": lambda: None},
            {"name": "producer_node"},
            {"name": "partition"},
            {"name": "fold_id"},
            {"name": "level"},
            {
                "name": "unit_ids",
                "transform": lambda values: _L(values, _norm_prediction_unit),
            },
            {"name": "values", "transform": _norm_f64_matrix},
            {"name": "target_names", "default": lambda: [], "transform": _L},
        ],
    )


def _norm_bound_training_output(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "binding", "transform": _norm_output_binding},
            {
                "name": "predictions",
                "transform": lambda values: _L(values, _norm_prediction_block),
            },
            {
                "name": "observation_predictions",
                "transform": lambda values: _L(
                    values, _norm_observation_prediction_block
                ),
            },
            {
                "name": "aggregated_predictions",
                "transform": lambda values: _L(
                    values, _norm_aggregated_prediction_block
                ),
            },
        ],
    )


def _norm_ranked_candidate(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "candidate_id"},
            {"name": "score", "transform": _F},
            {"name": "rank"},
        ],
    )


def _norm_selection_decision(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "policy_id"},
            {"name": "selected_candidate_id"},
            {"name": "metric_name"},
            {"name": "objective"},
            {"name": "metric_level", "skip": _skip_none},
            {"name": "evaluation_scope", "skip": _skip_none},
            {
                "name": "refit_slot_plan",
                "transform": _norm_refit_slot_plan,
                "skip": _skip_none,
            },
            {"name": "reduction_id", "skip": _skip_none},
            {"name": "selected_score", "transform": _F},
            {
                "name": "ranked_candidates",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_ranked_candidate),
            },
        ],
    )


def _norm_regression_metric_report(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "prediction_id", "default": lambda: None},
            {"name": "producer_node"},
            {"name": "variant_id", "skip": _skip_none},
            {"name": "variant_label", "skip": _skip_none},
            {"name": "partition"},
            {"name": "fold_id"},
            {"name": "level"},
            {"name": "row_count"},
            {"name": "target_width"},
            {"name": "target_names", "default": lambda: [], "transform": _L},
            {"name": "metrics", "transform": lambda values: _BM(values, _F)},
        ],
    )


def _norm_score_set(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "schema_version", "default": lambda: 1},
            {"name": "plan_id"},
            {"name": "selection_metric", "skip": _skip_none},
            {
                "name": "reports",
                "transform": lambda values: _L(values, _norm_regression_metric_report),
            },
        ],
    )


def _norm_artifact_ref(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "id"},
            {"name": "kind"},
            {"name": "controller_id"},
            {"name": "backend", "skip": _skip_none},
            {"name": "uri", "skip": _skip_none},
            {"name": "content_fingerprint", "skip": _skip_none},
            {"name": "size_bytes"},
            {"name": "plugin", "skip": _skip_none},
            {"name": "plugin_version", "skip": _skip_none},
        ],
    )


def _norm_lineage_record(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "record_id"},
            {"name": "run_id"},
            {"name": "node_id"},
            {"name": "phase"},
            {"name": "controller_id"},
            {"name": "controller_version"},
            {"name": "variant_id"},
            {"name": "fold_id"},
            {"name": "branch_path", "default": lambda: [], "transform": _L},
            {"name": "input_lineage", "default": lambda: [], "transform": _L},
            {
                "name": "artifact_refs",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_artifact_ref),
            },
            {"name": "params_fingerprint"},
            {"name": "data_model_shape_fingerprint"},
            {"name": "aggregation_policy_fingerprint"},
            {"name": "seed"},
            {"name": "unsafe_flags", "default": lambda: [], "transform": _sorted_set},
            {
                "name": "metrics",
                "default": lambda: {},
                "transform": lambda values: _BM(values, _F),
            },
        ],
    )


def _norm_bundle_prediction_requirement(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "producer_node"},
            {"name": "source_port"},
            {"name": "consumer_node"},
            {"name": "target_port"},
            {"name": "partition"},
            {"name": "prediction_level", "default": lambda: "sample"},
            {"name": "fold_ids", "default": lambda: [], "transform": _L},
            {
                "name": "unit_ids",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_prediction_unit),
                "skip": _skip_empty,
            },
            {"name": "sample_ids", "default": lambda: [], "transform": _L},
            {"name": "prediction_width"},
            {"name": "target_names", "transform": _L},
        ],
    )


def _norm_cache_block_record(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "prediction_id", "default": lambda: None},
            {"name": "fold_id", "default": lambda: None},
            {"name": "prediction_level", "default": lambda: "sample"},
            {"name": "row_count"},
            {
                "name": "unit_ids",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_prediction_unit),
                "skip": _skip_empty,
            },
            {"name": "sample_ids", "default": lambda: [], "transform": _L},
            {"name": "content_fingerprint"},
        ],
    )


def _norm_bundle_prediction_cache_record(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "requirement_key"},
            {"name": "cache_id"},
            {"name": "format"},
            {"name": "partition"},
            {"name": "prediction_level", "default": lambda: "sample"},
            {"name": "fold_ids", "default": lambda: [], "transform": _L},
            {
                "name": "unit_ids",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_prediction_unit),
                "skip": _skip_empty,
            },
            {"name": "sample_ids", "default": lambda: [], "transform": _L},
            {"name": "prediction_width"},
            {"name": "target_names", "transform": _L},
            {"name": "block_count"},
            {"name": "row_count"},
            {"name": "content_fingerprint"},
            {
                "name": "blocks",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_cache_block_record),
            },
        ],
    )


def _norm_prediction_cache_payload(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "requirement_key"},
            {"name": "cache_id"},
            {"name": "format"},
            {"name": "partition"},
            {"name": "prediction_level", "default": lambda: "sample"},
            {"name": "block_count"},
            {"name": "row_count"},
            {"name": "content_fingerprint"},
            {
                "name": "blocks",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_prediction_block),
            },
            {
                "name": "aggregated_blocks",
                "default": lambda: [],
                "transform": lambda values: _L(
                    values, _norm_aggregated_prediction_block
                ),
                "skip": _skip_empty,
            },
        ],
    )


def _norm_prediction_cache_payload_set(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "bundle_id"},
            {"name": "schema_version", "default": lambda: 1},
            {
                "name": "caches",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_prediction_cache_payload),
            },
        ],
    )


def _norm_combination_plan(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "mode", "default": lambda: "cartesian"},
            {
                "name": "component_source_ids",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {
                "name": "component_unit_ids",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {"name": "match_key", "skip": _skip_none},
            {"name": "reference_source_id", "skip": _skip_none},
            {"name": "seed", "skip": _skip_none},
            {"name": "cap", "skip": _skip_none},
            {"name": "budget", "skip": _skip_none},
            {"name": "missing_source_policy", "skip": _skip_none},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda value: _BM(value, _V),
                "skip": _skip_empty,
            },
        ],
    )


def _norm_representation_plan(source: Any) -> dict:
    kind = source.get("kind") if isinstance(source, dict) else None
    fields: list[dict[str, Any]] = [{"name": "kind"}]
    if kind == "aggregate":
        fields += [
            {"name": "input_unit_level"},
            {"name": "output_unit_level"},
            {"name": "reducer_id", "skip": _skip_none},
            {"name": "method", "skip": _skip_none},
            {"name": "cardinality"},
        ]
    elif kind in {"cartesian_product", "monte_carlo_cartesian"}:
        fields += [
            {"name": "combination_plan", "transform": _norm_combination_plan},
            {"name": "output_unit_level"},
            {"name": "cardinality"},
            {"name": "preserve_provenance", "default": lambda: True},
        ]
    elif kind == "stack_fixed":
        fields += [
            {"name": "output_unit_level"},
            {"name": "cardinality"},
            {"name": "expected_cardinality"},
            {
                "name": "component_source_ids",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
        ]
    else:
        fields += [
            {"name": "output_unit_level"},
            {"name": "cardinality"},
            {"name": "expected_cardinality"},
            {"name": "missing_source_policy"},
            {"name": "requires_missing_masks", "default": lambda: True},
            {
                "name": "component_source_ids",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
        ]
    return _S(source, fields)


def _norm_representation_compatibility(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "policy"},
            {"name": "outcome"},
            {"name": "fallback_used", "skip": _skip_none},
            {"name": "warning_severity", "skip": _skip_none},
            {"name": "affected_source_count", "default": lambda: 0},
            {"name": "affected_repetition_count", "default": lambda: 0},
            {"name": "affected_sample_count", "default": lambda: 0},
            {"name": "train_relation_fingerprint", "skip": _skip_none},
            {"name": "predict_relation_fingerprint", "skip": _skip_none},
            {"name": "train_unit_count", "skip": _skip_none},
            {"name": "predict_unit_count", "skip": _skip_none},
            {"name": "fixed_width_required", "default": lambda: False},
            {"name": "final_reducer_stabilizes_output", "default": lambda: False},
            {"name": "cartesian_combo_count_changed", "default": lambda: False},
            {"name": "late_fusion_branch_delta", "default": lambda: False},
            {
                "name": "messages",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda value: _BM(value, _V),
                "skip": _skip_empty,
            },
        ],
    )


def _norm_sample_observation_mapping(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "physical_sample_id"},
            {"name": "source_id"},
            {"name": "observation_ids", "transform": _L},
        ],
    )


def _norm_combo_selection(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "combo_unit_id"},
            {"name": "physical_sample_id"},
            {"name": "component_observation_ids", "transform": _L},
            {"name": "seed", "skip": _skip_none},
        ],
    )


def _norm_representation_replay_manifest(source: Any) -> Any:
    if source is None:
        return None
    return _S(
        source,
        [
            {"name": "manifest_id"},
            {"name": "representation_plan", "transform": _norm_representation_plan},
            {
                "name": "combination_plan",
                "transform": lambda value: (
                    None if value is None else _norm_combination_plan(value)
                ),
                "skip": _skip_none,
            },
            {"name": "output_unit_level"},
            {"name": "output_representation", "skip": _skip_none},
            {"name": "relation_fingerprint", "skip": _skip_none},
            {"name": "feature_schema_fingerprint", "skip": _skip_none},
            {"name": "final_reduction_id", "skip": _skip_none},
            {
                "name": "sample_observation_mapping",
                "default": lambda: [],
                "transform": lambda values: _L(
                    values, _norm_sample_observation_mapping
                ),
                "skip": _skip_empty,
            },
            {
                "name": "combo_selection",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_combo_selection),
                "skip": _skip_empty,
            },
            {
                "name": "qc_policy_refs",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {
                "name": "outlier_policy_refs",
                "default": lambda: [],
                "transform": _L,
                "skip": _skip_empty,
            },
            {"name": "missing_source_policy", "skip": _skip_none},
            {"name": "missing_repetition_policy", "skip": _skip_none},
            {"name": "prediction_representation", "skip": _skip_none},
            {"name": "final_output_unit_level", "skip": _skip_none},
            {
                "name": "train_compatibility",
                "transform": _norm_representation_compatibility,
                "skip": _skip_none,
            },
            {
                "name": "predict_compatibility",
                "transform": _norm_representation_compatibility,
                "skip": _skip_none,
            },
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda value: _BM(value, _V),
                "skip": _skip_empty,
            },
        ],
    )


def _norm_bundle_data_requirement(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {"name": "input_name"},
            {"name": "schema_fingerprint"},
            {"name": "plan_fingerprint"},
            {"name": "relation_fingerprint", "default": lambda: None},
            {"name": "output_representation"},
            {"name": "feature_set_id", "default": lambda: None},
            {
                "name": "representation_replay_manifest",
                "transform": _norm_representation_replay_manifest,
                "skip": _skip_none,
            },
            {
                "name": "representation_compatibility",
                "transform": _norm_representation_compatibility,
                "skip": _skip_none,
            },
        ],
    )


def _norm_refit_artifact_record(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "node_id"},
            {"name": "controller_id"},
            {"name": "artifact", "transform": _norm_artifact_ref},
            {"name": "params_fingerprint"},
            {
                "name": "data_requirement_keys",
                "default": lambda: [],
                "transform": _L,
            },
            {
                "name": "prediction_requirement_keys",
                "default": lambda: [],
                "transform": _L,
            },
        ],
    )


def _norm_execution_bundle(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "bundle_id"},
            {"name": "schema_version", "default": lambda: 1},
            {"name": "plan_id"},
            {"name": "graph_fingerprint"},
            {"name": "campaign_fingerprint"},
            {"name": "controller_fingerprint"},
            {"name": "selected_variant_id", "default": lambda: None},
            {
                "name": "selections",
                "default": lambda: {},
                "transform": lambda value: _BM(value, _norm_selection_decision),
            },
            {
                "name": "refit_artifacts",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_refit_artifact_record),
            },
            {
                "name": "prediction_requirements",
                "default": lambda: [],
                "transform": lambda values: _L(
                    values, _norm_bundle_prediction_requirement
                ),
            },
            {
                "name": "prediction_caches",
                "default": lambda: [],
                "transform": lambda values: _L(
                    values, _norm_bundle_prediction_cache_record
                ),
            },
            {
                "name": "scores",
                "transform": lambda value: (
                    None if value is None else _norm_score_set(value)
                ),
                "skip": _skip_none,
            },
            {
                "name": "data_requirements",
                "default": lambda: [],
                "transform": lambda values: _L(values, _norm_bundle_data_requirement),
            },
            {"name": "unsafe_flags", "default": lambda: [], "transform": _sorted_set},
            {
                "name": "metadata",
                "default": lambda: {},
                "transform": lambda value: _BM(value, _V),
            },
        ],
    )


def _norm_influence_entry(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "kind"},
            {"name": "scope_id"},
            {"name": "node_id"},
            {"name": "physical_sample_ids", "transform": _L},
            {"name": "origin_sample_ids", "transform": _L},
            {"name": "group_ids", "transform": _L},
        ],
    )


def _norm_influence_manifest(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "schema_version"},
            {"name": "relation_fingerprint"},
            {
                "name": "entries",
                "transform": lambda values: _L(values, _norm_influence_entry),
            },
            {"name": "manifest_fingerprint"},
        ],
    )


def _norm_training_refit_outcome(source: Any) -> dict:
    return _S(
        source,
        [{"name": "requested"}, {"name": "status"}, {"name": "strategy"}],
    )


def _normalize_training_outcome(source: Any) -> dict:
    """Rebuild the complete typed ``TrainingOutcome`` serde representation."""

    return _S(
        source,
        [
            {"name": "schema_version"},
            {"name": "outcome_id"},
            {"name": "run_id"},
            {"name": "training_request_fingerprint"},
            {
                "name": "data_identities",
                "transform": lambda values: _L(values, _norm_training_data_identity),
            },
            {"name": "selection_output_id"},
            {"name": "effective_plan", "transform": _normalize_execution_plan},
            {"name": "effective_plan_fingerprint"},
            {"name": "selected_variant_id"},
            {"name": "selected_variant_fingerprint"},
            {
                "name": "parameter_patches",
                "transform": lambda values: _L(values, _norm_parameter_patch),
            },
            {"name": "refit", "transform": _norm_training_refit_outcome},
            {"name": "score_set", "transform": _norm_score_set},
            {
                "name": "outputs",
                "transform": lambda values: _L(values, _norm_bound_training_output),
            },
            {
                "name": "lineage",
                "transform": lambda values: _L(values, _norm_lineage_record),
            },
            {
                "name": "portable_prediction_caches",
                "transform": _norm_prediction_cache_payload_set,
            },
            {"name": "training_influence", "transform": _norm_influence_manifest},
            {"name": "execution_bundle", "transform": _norm_execution_bundle},
            {"name": "replayable_phases", "transform": _L},
            {"name": "warnings", "transform": _L},
            {
                "name": "diagnostics",
                "transform": lambda value: _BM(value, _V),
            },
            {"name": "outcome_fingerprint"},
        ],
    )


def _norm_predictor_template(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "graph", "transform": _normalize_graph_spec},
            {"name": "campaign", "transform": _normalize_campaign_spec},
            {
                "name": "controller_manifests",
                "transform": _normalize_controller_manifests,
            },
            {"name": "template_fingerprint"},
        ],
    )


def _norm_training_outcome_ref(source: Any) -> dict:
    return _S(
        source,
        [
            {"name": "outcome_id"},
            {"name": "outcome_fingerprint"},
            {"name": "training_request_fingerprint"},
            {"name": "effective_plan_fingerprint"},
            {"name": "execution_bundle_id"},
            {"name": "execution_bundle_fingerprint"},
            {"name": "output_binding_fingerprints", "transform": _L},
            {"name": "training_influence_fingerprint"},
            {"name": "data_identities_fingerprint"},
        ],
    )


def _norm_package_artifact_binding(source: Any) -> dict:
    return _S(source, [{"name": "artifact_id"}, {"name": "load_mode"}])


def _normalize_portable_predictor_package(source: Any) -> dict:
    """Rebuild the typed ``PortablePredictorPackage`` serde representation."""

    return _S(
        source,
        [
            {"name": "schema_version"},
            {"name": "package_id"},
            {"name": "template", "transform": _norm_predictor_template},
            {"name": "training_request_fingerprint"},
            {"name": "training_outcome", "transform": _norm_training_outcome_ref},
            {"name": "effective_plan", "transform": _normalize_execution_plan},
            {"name": "execution_bundle", "transform": _norm_execution_bundle},
            {
                "name": "output_bindings",
                "transform": lambda values: _L(values, _norm_output_binding),
            },
            {"name": "predictor_node_ids", "transform": _L},
            {"name": "training_influence", "transform": _norm_influence_manifest},
            {
                "name": "data_identities",
                "transform": lambda values: _L(values, _norm_training_data_identity),
            },
            {"name": "fitted_artifact_mode"},
            {
                "name": "artifact_bindings",
                "transform": lambda values: _L(values, _norm_package_artifact_binding),
            },
            {"name": "package_fingerprint"},
        ],
    )


def _node_params_fingerprint(params: Any) -> str:
    """Fingerprint node-plan params (a BTreeMap<String, Value>) via serde bytes."""

    return _serde_sha256(_BM(params, _V))


def _validate_search_space_fingerprint(
    graph: dict[str, Any], campaign: dict[str, Any], label: str
) -> None:
    expected = graph.get("search_space_fingerprint")
    if expected is None:
        return
    actual = _serde_sha256(_normalize_campaign_spec(campaign)["generation"])
    require(
        expected == actual,
        f"{label}.graph.search_space_fingerprint does not match campaign generation spec",
    )


def _deny_unknown_fields(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        return
    unknown = set(value) - allowed
    require(not unknown, f"{label} has unknown field(s): {sorted(unknown)}")


def _validate_representation_plan_deserialize_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        return
    kind = value.get("kind")
    allowed = {"kind"}
    if kind == "aggregate":
        allowed |= {
            "input_unit_level",
            "output_unit_level",
            "reducer_id",
            "method",
            "cardinality",
        }
    elif kind in {"cartesian_product", "monte_carlo_cartesian"}:
        allowed |= {
            "combination_plan",
            "output_unit_level",
            "cardinality",
            "preserve_provenance",
        }
        combination = value.get("combination_plan")
        _deny_unknown_fields(
            combination,
            {
                "mode",
                "component_source_ids",
                "component_unit_ids",
                "match_key",
                "reference_source_id",
                "seed",
                "cap",
                "budget",
                "missing_source_policy",
                "metadata",
            },
            f"{label}.combination_plan",
        )
    elif kind == "stack_fixed":
        allowed |= {
            "output_unit_level",
            "cardinality",
            "expected_cardinality",
            "component_source_ids",
        }
    elif kind == "stack_padded_masked":
        allowed |= {
            "output_unit_level",
            "cardinality",
            "expected_cardinality",
            "missing_source_policy",
            "requires_missing_masks",
            "component_source_ids",
        }
    _deny_unknown_fields(value, allowed, label)


def _validate_model_input_spec_deserialize_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        return
    _deny_unknown_fields(
        value,
        {
            "schema_version",
            "ports",
            "default_fusion",
            "fit_influence_policy",
            "metadata",
        },
        label,
    )
    ports = value.get("ports")
    if isinstance(ports, list):
        for index, port in enumerate(ports):
            _deny_unknown_fields(
                port,
                {
                    "name",
                    "accepted_representations",
                    "accepted_types",
                    "rank",
                    "multi_source",
                    "optional",
                    "metadata",
                },
                f"{label}.ports[{index}]",
            )
    fusion = value.get("default_fusion")
    if isinstance(fusion, dict):
        _deny_unknown_fields(
            fusion,
            {"mode", "alignment", "adapter_id", "representation_plan", "params"},
            f"{label}.default_fusion",
        )
        _validate_representation_plan_deserialize_shape(
            fusion.get("representation_plan"),
            f"{label}.default_fusion.representation_plan",
        )


def _validate_controller_manifest_deserialize_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        return
    _deny_unknown_fields(
        value,
        {
            "controller_id",
            "controller_version",
            "operator_kind",
            "priority",
            "supported_phases",
            "input_ports",
            "output_ports",
            "data_requirements",
            "capabilities",
            "operator_selectors",
            "fit_scope",
            "rng_policy",
            "artifact_policy",
        },
        label,
    )
    selectors = value.get("operator_selectors")
    if isinstance(selectors, list):
        for index, selector in enumerate(selectors):
            _deny_unknown_fields(
                selector,
                {"aliases", "classes", "class_prefixes", "functions", "refs", "types"},
                f"{label}.operator_selectors[{index}]",
            )
    _validate_model_input_spec_deserialize_shape(
        value.get("data_requirements"), f"{label}.data_requirements"
    )


def validate_prediction_unit_ids(
    value: Any,
    expected_level: str,
    label: str,
) -> list[dict[str, str]]:
    require(isinstance(value, list), f"{label} must be an array")
    keys: list[tuple[str, str]] = []
    normalized: list[dict[str, str]] = []
    for index, unit in enumerate(value):
        unit_label = f"{label}[{index}]"
        unit = require_exact_keys(unit, {"level", "id"}, set(), unit_label)
        require(unit["level"] == expected_level, f"{unit_label}.level is inconsistent")
        require_identifier(unit["id"], f"{unit_label}.id")
        keys.append((unit["level"], unit["id"]))
        normalized.append({"level": unit["level"], "id": unit["id"]})
    require(len(set(keys)) == len(keys), f"{label} contains duplicate units")
    return normalized


def validate_bundle_prediction_requirement(value: Any, label: str) -> dict[str, Any]:
    requirement = require_exact_keys(
        value,
        {
            "producer_node",
            "source_port",
            "consumer_node",
            "target_port",
            "partition",
            "fold_ids",
            "sample_ids",
            "prediction_width",
            "target_names",
        },
        {"prediction_level", "unit_ids"},
        label,
    )
    for field in ("producer_node", "consumer_node"):
        require_identifier(requirement[field], f"{label}.{field}")
    for field in ("source_port", "target_port"):
        require_non_empty_string(requirement[field], f"{label}.{field}")
    require(
        requirement["partition"] == "validation",
        f"{label}.partition must be validation",
    )
    level = requirement.get("prediction_level", "sample")
    require(level in PREDICTION_LEVELS, f"{label}.prediction_level is invalid")
    require(level != "observation", f"{label} cannot cache observation predictions")
    fold_ids = validate_ordered_unique_identifiers(
        requirement["fold_ids"], f"{label}.fold_ids", require_non_empty=False
    )
    samples = validate_ordered_unique_identifiers(
        requirement["sample_ids"],
        f"{label}.sample_ids",
        require_non_empty=level == "sample",
    )
    units = validate_prediction_unit_ids(
        requirement.get("unit_ids", []), level, f"{label}.unit_ids"
    )
    if level == "sample":
        expected_units = [{"level": "sample", "id": sample_id} for sample_id in samples]
        require(
            not units or units == expected_units,
            f"{label}.unit_ids do not match sample_ids",
        )
    else:
        require(
            not samples, f"{label}.sample_ids must be empty for {level} predictions"
        )
        require(
            bool(units), f"{label}.unit_ids must be non-empty for {level} predictions"
        )
    require_positive_int(requirement["prediction_width"], f"{label}.prediction_width")
    targets = validate_ordered_unique_strings(
        requirement["target_names"], f"{label}.target_names", require_non_empty=True
    )
    require(
        len(targets) == requirement["prediction_width"],
        f"{label}.target_names length does not match prediction_width",
    )
    normalized = dict(requirement)
    normalized["prediction_level"] = level
    normalized["unit_ids"] = units
    normalized["fold_ids"] = fold_ids
    normalized["sample_ids"] = samples
    normalized["requirement_key"] = (
        f"{requirement['producer_node']}.{requirement['source_port']}"
        f"->{requirement['consumer_node']}.{requirement['target_port']}"
    )
    return normalized


def validate_prediction_cache_block_record(value: Any, label: str) -> dict[str, Any]:
    block = require_exact_keys(
        value,
        {"prediction_id", "fold_id", "row_count", "sample_ids", "content_fingerprint"},
        {"prediction_level", "unit_ids"},
        label,
    )
    require_optional_non_empty_string(block["prediction_id"], f"{label}.prediction_id")
    require_optional_identifier(block["fold_id"], f"{label}.fold_id")
    level = block.get("prediction_level", "sample")
    require(level in PREDICTION_LEVELS, f"{label}.prediction_level is invalid")
    require(level != "observation", f"{label} cannot cache observation predictions")
    require_positive_int(block["row_count"], f"{label}.row_count")
    samples = validate_ordered_unique_identifiers(
        block["sample_ids"],
        f"{label}.sample_ids",
        require_non_empty=level == "sample",
    )
    units = validate_prediction_unit_ids(
        block.get("unit_ids", []), level, f"{label}.unit_ids"
    )
    if level == "sample":
        expected_units = [{"level": "sample", "id": sample_id} for sample_id in samples]
        require(
            not units or units == expected_units,
            f"{label}.unit_ids do not match sample_ids",
        )
        require(
            block["row_count"] == len(samples),
            f"{label}.row_count does not match samples",
        )
    else:
        require(
            not samples, f"{label}.sample_ids must be empty for {level} predictions"
        )
        require(
            block["row_count"] == len(units), f"{label}.row_count does not match units"
        )
    require_sha256(block["content_fingerprint"], f"{label}.content_fingerprint")
    normalized = dict(block)
    normalized["prediction_level"] = level
    normalized["unit_ids"] = units
    normalized["sample_ids"] = samples
    return normalized


def validate_bundle_prediction_cache_record(value: Any, label: str) -> dict[str, Any]:
    cache = require_exact_keys(
        value,
        {
            "requirement_key",
            "cache_id",
            "format",
            "partition",
            "fold_ids",
            "sample_ids",
            "prediction_width",
            "target_names",
            "block_count",
            "row_count",
            "content_fingerprint",
            "blocks",
        },
        {"prediction_level", "unit_ids"},
        label,
    )
    for field in ("requirement_key", "cache_id"):
        require_non_empty_string(cache[field], f"{label}.{field}")
    require(
        cache["format"] == "dag-ml-json-prediction-blocks-v1",
        f"{label}.format is unsupported",
    )
    require(cache["partition"] == "validation", f"{label}.partition must be validation")
    level = cache.get("prediction_level", "sample")
    require(level in PREDICTION_LEVELS, f"{label}.prediction_level is invalid")
    require(level != "observation", f"{label} cannot cache observation predictions")
    fold_ids = validate_ordered_unique_identifiers(
        cache["fold_ids"], f"{label}.fold_ids", require_non_empty=False
    )
    samples = validate_ordered_unique_identifiers(
        cache["sample_ids"], f"{label}.sample_ids", require_non_empty=level == "sample"
    )
    units = validate_prediction_unit_ids(
        cache.get("unit_ids", []), level, f"{label}.unit_ids"
    )
    if level == "sample":
        expected_units = [{"level": "sample", "id": sample_id} for sample_id in samples]
        require(
            not units or units == expected_units,
            f"{label}.unit_ids do not match sample_ids",
        )
        require(
            cache["row_count"] == len(samples),
            f"{label}.row_count does not match samples",
        )
    else:
        require(
            not samples, f"{label}.sample_ids must be empty for {level} predictions"
        )
        require(
            cache["row_count"] == len(units), f"{label}.row_count does not match units"
        )
    require_positive_int(cache["prediction_width"], f"{label}.prediction_width")
    targets = validate_ordered_unique_strings(
        cache["target_names"], f"{label}.target_names", require_non_empty=True
    )
    require(
        len(targets) == cache["prediction_width"],
        f"{label}.target_names length does not match prediction_width",
    )
    blocks = cache["blocks"]
    require(
        isinstance(blocks, list) and bool(blocks), f"{label}.blocks must be non-empty"
    )
    validated_blocks = [
        validate_prediction_cache_block_record(block, f"{label}.blocks[{index}]")
        for index, block in enumerate(blocks)
    ]
    require_positive_int(cache["block_count"], f"{label}.block_count")
    require(
        cache["block_count"] == len(blocks),
        f"{label}.block_count does not match blocks",
    )
    require_positive_int(cache["row_count"], f"{label}.row_count")
    require(
        cache["row_count"] == sum(block["row_count"] for block in validated_blocks),
        f"{label}.row_count does not match block records",
    )
    if level == "sample":
        block_samples = [
            sample for block in validated_blocks for sample in block["sample_ids"]
        ]
        require(
            len(set(block_samples)) == len(block_samples),
            f"{label}.blocks duplicate samples",
        )
        require(
            set(block_samples) == set(samples),
            f"{label}.blocks do not cover sample_ids",
        )
    else:
        block_units = [
            (unit["level"], unit["id"])
            for block in validated_blocks
            for unit in block["unit_ids"]
        ]
        require(
            len(set(block_units)) == len(block_units), f"{label}.blocks duplicate units"
        )
        require(
            set(block_units) == {(unit["level"], unit["id"]) for unit in units},
            f"{label}.blocks do not cover unit_ids",
        )
    require_sha256(cache["content_fingerprint"], f"{label}.content_fingerprint")
    normalized = dict(cache)
    normalized["prediction_level"] = level
    normalized["unit_ids"] = units
    normalized["sample_ids"] = samples
    normalized["fold_ids"] = fold_ids
    normalized["validated_blocks"] = validated_blocks
    return normalized


def validate_cache_prediction_block(value: Any, label: str) -> dict[str, Any]:
    block = require_exact_keys(
        value,
        {"producer_node", "partition", "fold_id", "sample_ids", "values"},
        {"prediction_id", "target_names"},
        label,
    )
    require_optional_non_empty_string(
        block.get("prediction_id"), f"{label}.prediction_id"
    )
    require_identifier(block["producer_node"], f"{label}.producer_node")
    require(block["partition"] == "validation", f"{label}.partition must be validation")
    require_optional_identifier(block["fold_id"], f"{label}.fold_id")
    samples = validate_ordered_unique_identifiers(
        block["sample_ids"], f"{label}.sample_ids", require_non_empty=True
    )
    values = block["values"]
    require(
        isinstance(values, list) and bool(values), f"{label}.values must be non-empty"
    )
    width = len(values[0]) if isinstance(values[0], list) else 0
    require(width > 0, f"{label}.values must have positive width")
    validate_prediction_matrix(values, len(samples), width, f"{label}.values")
    targets = block.get("target_names", [])
    validate_ordered_unique_strings(
        targets, f"{label}.target_names", require_non_empty=False
    )
    require(
        not targets or len(targets) == width, f"{label}.target_names width mismatch"
    )
    canonical = _norm_prediction_block(
        {
            "prediction_id": block.get("prediction_id"),
            "producer_node": block["producer_node"],
            "partition": block["partition"],
            "fold_id": block["fold_id"],
            "sample_ids": samples,
            "values": values,
            "target_names": targets,
        }
    )
    return {
        "canonical": canonical,
        "fold_id": block["fold_id"],
        "producer_node": block["producer_node"],
        "sample_ids": samples,
        "row_count": len(samples),
        "width": width,
        "target_names": targets or [f"p{index}" for index in range(width)],
    }


def validate_cache_aggregated_block(value: Any, label: str) -> dict[str, Any]:
    block = require_exact_keys(
        value,
        {"producer_node", "partition", "fold_id", "level", "unit_ids", "values"},
        {"prediction_id", "target_names"},
        label,
    )
    require_optional_non_empty_string(
        block.get("prediction_id"), f"{label}.prediction_id"
    )
    require_identifier(block["producer_node"], f"{label}.producer_node")
    require(block["partition"] == "validation", f"{label}.partition must be validation")
    require_optional_identifier(block["fold_id"], f"{label}.fold_id")
    level = block["level"]
    require(level in {"target", "group"}, f"{label}.level must be target or group")
    units = validate_prediction_unit_ids(block["unit_ids"], level, f"{label}.unit_ids")
    require(bool(units), f"{label}.unit_ids must be non-empty")
    values = block["values"]
    require(
        isinstance(values, list) and bool(values), f"{label}.values must be non-empty"
    )
    width = len(values[0]) if isinstance(values[0], list) else 0
    require(width > 0, f"{label}.values must have positive width")
    validate_prediction_matrix(values, len(units), width, f"{label}.values")
    targets = block.get("target_names", [])
    validate_ordered_unique_strings(
        targets, f"{label}.target_names", require_non_empty=False
    )
    require(
        not targets or len(targets) == width, f"{label}.target_names width mismatch"
    )
    canonical = _norm_aggregated_prediction_block(
        {
            "prediction_id": block.get("prediction_id"),
            "producer_node": block["producer_node"],
            "partition": block["partition"],
            "fold_id": block["fold_id"],
            "level": level,
            "unit_ids": units,
            "values": values,
            "target_names": targets,
        }
    )
    return {
        "canonical": canonical,
        "fold_id": block["fold_id"],
        "producer_node": block["producer_node"],
        "unit_ids": units,
        "row_count": len(units),
        "width": width,
        "target_names": targets or [f"p{index}" for index in range(width)],
    }


def validate_prediction_cache_payload(value: Any, label: str) -> dict[str, Any]:
    payload = require_exact_keys(
        value,
        {
            "requirement_key",
            "cache_id",
            "format",
            "partition",
            "block_count",
            "row_count",
            "content_fingerprint",
        },
        {"prediction_level", "blocks", "aggregated_blocks"},
        label,
    )
    for field in ("requirement_key", "cache_id"):
        require_non_empty_string(payload[field], f"{label}.{field}")
    require(
        payload["format"] == "dag-ml-json-prediction-blocks-v1",
        f"{label}.format is unsupported",
    )
    require(
        payload["partition"] == "validation", f"{label}.partition must be validation"
    )
    level = payload.get("prediction_level", "sample")
    require(level in PREDICTION_LEVELS, f"{label}.prediction_level is invalid")
    require(level != "observation", f"{label} cannot cache observation predictions")
    blocks = payload.get("blocks", [])
    aggregated = payload.get("aggregated_blocks", [])
    require(isinstance(blocks, list), f"{label}.blocks must be an array")
    require(isinstance(aggregated, list), f"{label}.aggregated_blocks must be an array")
    if level == "sample":
        require(
            bool(blocks) and not aggregated, f"{label} sample cache block kind mismatch"
        )
        validated = [
            validate_cache_prediction_block(block, f"{label}.blocks[{index}]")
            for index, block in enumerate(blocks)
        ]
    else:
        require(
            not blocks and bool(aggregated),
            f"{label} aggregated cache block kind mismatch",
        )
        validated = [
            validate_cache_aggregated_block(
                block, f"{label}.aggregated_blocks[{index}]"
            )
            for index, block in enumerate(aggregated)
        ]
        require(
            all(block["canonical"]["level"] == level for block in validated),
            f"{label} aggregated block level mismatch",
        )
    require_positive_int(payload["block_count"], f"{label}.block_count")
    require(payload["block_count"] == len(validated), f"{label}.block_count mismatch")
    require_positive_int(payload["row_count"], f"{label}.row_count")
    require(
        payload["row_count"] == sum(block["row_count"] for block in validated),
        f"{label}.row_count mismatch",
    )
    canonical_blocks = [block["canonical"] for block in validated]
    require_sha256(payload["content_fingerprint"], f"{label}.content_fingerprint")
    require(
        payload["content_fingerprint"] == _serde_sha256(canonical_blocks),
        f"{label}.content_fingerprint does not match blocks",
    )
    normalized = dict(payload)
    normalized["prediction_level"] = level
    normalized["blocks"] = blocks
    normalized["aggregated_blocks"] = aggregated
    normalized["validated_blocks"] = validated
    return normalized


def validate_portable_prediction_caches(
    value: Any,
    bundle: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    payload_set = require_exact_keys(
        value, {"bundle_id", "schema_version", "caches"}, set(), label
    )
    require_identifier(payload_set["bundle_id"], f"{label}.bundle_id")
    require_version_one(payload_set["schema_version"], label)
    require(
        payload_set["bundle_id"] == bundle["bundle_id"], f"{label}.bundle_id mismatch"
    )
    caches = payload_set["caches"]
    require(isinstance(caches, list), f"{label}.caches must be an array")
    payloads = [
        validate_prediction_cache_payload(payload, f"{label}.caches[{index}]")
        for index, payload in enumerate(caches)
    ]
    payload_keys = [payload["requirement_key"] for payload in payloads]
    cache_ids = [payload["cache_id"] for payload in payloads]
    require(
        len(set(payload_keys)) == len(payload_keys),
        f"{label} duplicate requirement keys",
    )
    require(len(set(cache_ids)) == len(cache_ids), f"{label} duplicate cache ids")
    records = [
        validate_bundle_prediction_cache_record(
            record, f"{label}.bundle_cache[{index}]"
        )
        for index, record in enumerate(bundle["prediction_caches"])
    ]
    records_by_key = {record["requirement_key"]: record for record in records}
    requirements = [
        validate_bundle_prediction_requirement(
            requirement, f"{label}.bundle_requirement[{index}]"
        )
        for index, requirement in enumerate(bundle["prediction_requirements"])
    ]
    requirements_by_key = {
        requirement["requirement_key"]: requirement for requirement in requirements
    }
    require(
        len(requirements_by_key) == len(requirements),
        f"{label} execution bundle has duplicate prediction requirements",
    )
    require(
        set(records_by_key) == set(requirements_by_key),
        f"{label} cache records do not exactly cover prediction requirements",
    )
    for requirement_key, record in records_by_key.items():
        requirement = requirements_by_key[requirement_key]
        for field in (
            "partition",
            "prediction_level",
            "fold_ids",
            "unit_ids",
            "sample_ids",
            "prediction_width",
            "target_names",
        ):
            require(
                record[field] == requirement[field],
                f"{label} cache record {field} does not match its requirement",
            )
    require(
        set(payload_keys) == set(records_by_key),
        f"{label}.caches do not exactly match execution bundle cache records",
    )
    for index, payload in enumerate(payloads):
        payload_label = f"{label}.caches[{index}]"
        record = records_by_key[payload["requirement_key"]]
        for field in (
            "cache_id",
            "format",
            "partition",
            "prediction_level",
            "block_count",
            "row_count",
            "content_fingerprint",
        ):
            require(
                payload[field] == record[field],
                f"{payload_label}.{field} mismatches cache record",
            )
        derived_records: list[dict[str, Any]] = []
        for block in payload["validated_blocks"]:
            canonical = block["canonical"]
            derived: dict[str, Any] = {
                "prediction_id": canonical["prediction_id"],
                "fold_id": canonical["fold_id"],
                "prediction_level": payload["prediction_level"],
                "row_count": block["row_count"],
                "unit_ids": canonical.get("unit_ids", []),
                "sample_ids": canonical.get("sample_ids", []),
                "content_fingerprint": _serde_sha256(canonical),
            }
            derived_records.append(derived)
        require(
            len(derived_records) == len(record["validated_blocks"]),
            f"{payload_label}.blocks do not match cache record",
        )
        for block_index, (derived, expected) in enumerate(
            zip(derived_records, record["validated_blocks"])
        ):
            for field in (
                "prediction_id",
                "fold_id",
                "prediction_level",
                "row_count",
                "unit_ids",
                "sample_ids",
                "content_fingerprint",
            ):
                require(
                    derived[field] == expected[field],
                    f"{payload_label}.blocks[{block_index}].{field} mismatches cache record",
                )
    return payload_set


def validate_execution_bundle_contract(
    value: Any,
    label: str,
    *,
    plan: dict[str, Any],
    predictor_closure: set[str],
    refit_requested: bool,
    selected_variant_id: str,
    score_set: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "bundle_id",
        "schema_version",
        "plan_id",
        "graph_fingerprint",
        "campaign_fingerprint",
        "controller_fingerprint",
        "selected_variant_id",
        "selections",
        "refit_artifacts",
        "prediction_requirements",
        "prediction_caches",
        "scores",
        "data_requirements",
        "unsafe_flags",
        "metadata",
    }
    bundle = require_exact_keys(value, required, set(), label)
    require_version_one(bundle["schema_version"], label)
    require_identifier(bundle["bundle_id"], f"{label}.bundle_id")
    require(bundle["plan_id"] == plan["id"], f"{label}.plan_id does not match plan")
    for field in (
        "graph_fingerprint",
        "campaign_fingerprint",
        "controller_fingerprint",
    ):
        require_sha256(bundle[field], f"{label}.{field}")
        require(bundle[field] == plan[field], f"{label}.{field} does not match plan")
    require(
        bundle["selected_variant_id"] == selected_variant_id,
        f"{label}.selected_variant_id does not match TrainingOutcome",
    )
    require(bundle["scores"] == score_set, f"{label}.scores must equal score_set")
    require(
        isinstance(bundle["selections"], dict), f"{label}.selections must be an object"
    )
    for selection_key, decision in bundle["selections"].items():
        require_non_empty_string(selection_key, f"{label}.selections key")
        validate_selection_decision(decision, f"{label}.selections[{selection_key}]")
    for field in (
        "refit_artifacts",
        "prediction_requirements",
        "prediction_caches",
        "data_requirements",
    ):
        require(isinstance(bundle[field], list), f"{label}.{field} must be an array")
    validate_ordered_unique_strings(
        bundle["unsafe_flags"], f"{label}.unsafe_flags", require_non_empty=False
    )
    validate_metadata_object(bundle["metadata"], f"{label}.metadata")

    node_plans = plan.get("node_plans")
    require(
        isinstance(node_plans, dict), f"{label} effective plan node_plans are invalid"
    )
    artifact_keys: list[tuple[str, str]] = []
    artifact_ids: list[str] = []
    for index, record in enumerate(bundle["refit_artifacts"]):
        record_label = f"{label}.refit_artifacts[{index}]"
        require(isinstance(record, dict), f"{record_label} must be an object")
        for field in ("node_id", "controller_id"):
            require_identifier(record.get(field), f"{record_label}.{field}")
        artifact = record.get("artifact")
        require(
            isinstance(artifact, dict), f"{record_label}.artifact must be an object"
        )
        require_identifier(artifact.get("id"), f"{record_label}.artifact.id")
        require_sha256(
            artifact.get("content_fingerprint"),
            f"{record_label}.artifact.content_fingerprint",
        )
        require_non_empty_string(
            artifact.get("backend"), f"{record_label}.artifact.backend"
        )
        require_optional_non_empty_string(
            artifact.get("plugin"), f"{record_label}.artifact.plugin"
        )
        require_optional_non_empty_string(
            artifact.get("plugin_version"), f"{record_label}.artifact.plugin_version"
        )
        require(
            artifact.get("plugin_version") is None
            or artifact.get("plugin") is not None,
            f"{record_label}.artifact.plugin_version requires plugin",
        )
        require(
            artifact.get("controller_id") == record["controller_id"],
            f"{record_label}.artifact.controller_id does not match record",
        )
        require_sha256(
            record.get("params_fingerprint"), f"{record_label}.params_fingerprint"
        )
        for field in ("data_requirement_keys", "prediction_requirement_keys"):
            validate_ordered_unique_strings(
                record.get(field), f"{record_label}.{field}", require_non_empty=False
            )
        artifact_keys.append((record["node_id"], artifact["id"]))
        artifact_ids.append(artifact["id"])
    require(
        len(set(artifact_keys)) == len(artifact_keys),
        f"{label}.refit_artifacts duplicate",
    )
    require(
        len(set(artifact_ids)) == len(artifact_ids),
        f"{label}.refit_artifacts contain duplicate artifact ids",
    )

    expected_artifact_nodes = {
        node_id
        for node_id in predictor_closure
        if "REFIT" in node_plans[node_id].get("supported_phases", [])
        and "emits_artifacts" in node_plans[node_id].get("controller_capabilities", [])
    }
    actual_artifact_nodes = {record["node_id"] for record in bundle["refit_artifacts"]}
    if refit_requested:
        require(
            actual_artifact_nodes == expected_artifact_nodes,
            f"{label}.refit_artifacts nodes do not exactly match the predictor closure "
            "with `emits_artifacts` capability",
        )
    else:
        require(
            not bundle["refit_artifacts"],
            f"{label} no-refit outcome cannot contain refit artifacts",
        )

    relation_fingerprints: set[str] = set()
    requirement_keys: list[str] = []
    for index, requirement in enumerate(bundle["data_requirements"]):
        requirement_label = f"{label}.data_requirements[{index}]"
        require(isinstance(requirement, dict), f"{requirement_label} must be an object")
        for field in ("node_id", "input_name"):
            require_non_empty_string(
                requirement.get(field), f"{requirement_label}.{field}"
            )
        for field in ("schema_fingerprint", "plan_fingerprint", "relation_fingerprint"):
            require_sha256(requirement.get(field), f"{requirement_label}.{field}")
        key = f"{requirement['node_id']}.{requirement['input_name']}"
        requirement_keys.append(key)
        relation_fingerprints.add(requirement["relation_fingerprint"])
    require(
        len(set(requirement_keys)) == len(requirement_keys),
        f"{label}.data_requirements contain duplicates",
    )

    expected_data_requirements: dict[str, dict[str, Any]] = {}
    for node_id in predictor_closure:
        node_plan = node_plans[node_id]
        for binding in node_plan.get("data_bindings", []):
            key = f"{node_id}.{binding['input_name']}"
            expected_data_requirements[key] = {
                "node_id": node_id,
                "input_name": binding["input_name"],
                "schema_fingerprint": binding["schema_fingerprint"],
                "plan_fingerprint": binding["plan_fingerprint"],
                "relation_fingerprint": binding.get("relation_fingerprint"),
                "output_representation": binding["output_representation"],
                "feature_set_id": binding.get("feature_set_id"),
            }
    require(
        set(requirement_keys) == set(expected_data_requirements),
        f"{label}.data_requirements do not exactly match effective plan",
    )
    data_by_key = {
        f"{requirement['node_id']}.{requirement['input_name']}": requirement
        for requirement in bundle["data_requirements"]
    }
    for key, expected in expected_data_requirements.items():
        actual = data_by_key[key]
        for field, expected_value in expected.items():
            require(
                actual.get(field) == expected_value,
                f"{label}.data_requirements[{key}].{field} does not match plan",
            )

    validated_requirements = [
        validate_bundle_prediction_requirement(
            requirement, f"{label}.prediction_requirements[{index}]"
        )
        for index, requirement in enumerate(bundle["prediction_requirements"])
    ]
    prediction_keys = [
        requirement["requirement_key"] for requirement in validated_requirements
    ]
    require(
        len(set(prediction_keys)) == len(prediction_keys),
        f"{label}.prediction_requirements contain duplicates",
    )
    graph = plan["graph_plan"]["graph"]
    oof_edges = {
        (
            edge["source"]["node_id"],
            edge["source"]["port_name"],
            edge["target"]["node_id"],
            edge["target"]["port_name"],
        )
        for edge in graph["edges"]
        if edge["contract"].get("requires_oof") is True
        and edge["source"]["node_id"] in predictor_closure
        and edge["target"]["node_id"] in predictor_closure
    }
    expected_prediction_keys = {
        f"{producer}.{source_port}->{consumer}.{target_port}"
        for producer, source_port, consumer, target_port in oof_edges
    }
    require(
        set(prediction_keys) == expected_prediction_keys,
        f"{label}.prediction_requirements do not exactly match predictor-closure OOF edges",
    )
    fold_set = plan.get("fold_set")
    expected_fold_ids: list[str] = []
    expected_samples: list[str] = []
    folds_by_id: dict[str, dict[str, Any]] = {}
    if fold_set is not None:
        expected_fold_ids = [fold["fold_id"] for fold in fold_set["folds"]]
        expected_samples = fold_set["sample_ids"]
        folds_by_id = {fold["fold_id"]: fold for fold in fold_set["folds"]}
    for index, requirement in enumerate(validated_requirements):
        coordinates = (
            requirement["producer_node"],
            requirement["source_port"],
            requirement["consumer_node"],
            requirement["target_port"],
        )
        require(
            coordinates in oof_edges,
            f"{label}.prediction_requirements[{index}] does not match a plan OOF edge",
        )
        if fold_set is not None:
            require(
                set(requirement["fold_ids"]) == set(expected_fold_ids),
                f"{label}.prediction_requirements[{index}].fold_ids do not match plan",
            )
            if requirement["prediction_level"] == "sample":
                require(
                    set(requirement["sample_ids"]) == set(expected_samples),
                    f"{label}.prediction_requirements[{index}].sample_ids do not match plan",
                )

    validated_caches = [
        validate_bundle_prediction_cache_record(
            cache, f"{label}.prediction_caches[{index}]"
        )
        for index, cache in enumerate(bundle["prediction_caches"])
    ]
    cache_keys = [cache["requirement_key"] for cache in validated_caches]
    require(
        len(set(cache_keys)) == len(cache_keys), f"{label}.prediction_caches duplicate"
    )
    require(
        set(cache_keys) == set(prediction_keys),
        f"{label}.prediction_caches do not exactly cover prediction requirements",
    )
    requirements_by_key = {
        requirement["requirement_key"]: requirement
        for requirement in validated_requirements
    }
    caches_by_key = {cache["requirement_key"]: cache for cache in validated_caches}
    for index, cache in enumerate(validated_caches):
        cache_label = f"{label}.prediction_caches[{index}]"
        require(
            cache["requirement_key"] in requirements_by_key,
            f"{cache_label} references unknown prediction requirement",
        )
        requirement = requirements_by_key[cache["requirement_key"]]
        for field in (
            "partition",
            "prediction_level",
            "unit_ids",
            "prediction_width",
            "target_names",
        ):
            require(
                cache[field] == requirement[field],
                f"{cache_label}.{field} mismatches requirement",
            )
        for field in ("fold_ids", "sample_ids"):
            require(
                cache[field] == requirement[field],
                f"{cache_label}.{field} mismatches requirement",
            )
        if fold_set is not None and cache["prediction_level"] == "sample":
            covered_folds: set[str] = set()
            for block_index, block in enumerate(cache["validated_blocks"]):
                block_label = f"{cache_label}.blocks[{block_index}]"
                fold_id = block["fold_id"]
                require(
                    fold_id in folds_by_id, f"{block_label}.fold_id is absent from plan"
                )
                require(
                    fold_id not in covered_folds,
                    f"{cache_label} duplicates fold `{fold_id}`",
                )
                covered_folds.add(fold_id)
                require(
                    set(block["sample_ids"])
                    == set(folds_by_id[fold_id]["validation_sample_ids"]),
                    f"{block_label}.sample_ids do not match fold validation samples",
                )
            require(
                covered_folds == set(expected_fold_ids),
                f"{cache_label}.blocks do not cover every plan fold",
            )

    for index, record in enumerate(bundle["refit_artifacts"]):
        record_label = f"{label}.refit_artifacts[{index}]"
        node_id = record["node_id"]
        require(node_id in node_plans, f"{record_label}.node_id is absent from plan")
        node_plan = node_plans[node_id]
        require(
            record["controller_id"] == node_plan["controller_id"],
            f"{record_label}.controller_id does not match plan",
        )
        require(
            record["params_fingerprint"] == node_plan["params_fingerprint"],
            f"{record_label}.params_fingerprint does not match plan",
        )
        for key in record["data_requirement_keys"]:
            require(
                key in data_by_key,
                f"{record_label} references unknown data requirement `{key}`",
            )
            require(
                data_by_key[key]["node_id"] == node_id,
                f"{record_label} references a foreign data requirement `{key}`",
            )
        for key in record["prediction_requirement_keys"]:
            require(
                key in requirements_by_key,
                f"{record_label} references unknown prediction requirement `{key}`",
            )
            require(
                requirements_by_key[key]["consumer_node"] == node_id,
                f"{record_label} references a foreign prediction requirement `{key}`",
            )
            require(
                key in caches_by_key, f"{record_label} requirement `{key}` has no cache"
            )
        expected_data_keys = sorted(
            f"{node_id}.{binding['input_name']}"
            for binding in node_plan.get("data_bindings", [])
        )
        require(
            len(record["data_requirement_keys"]) == len(expected_data_keys)
            and set(record["data_requirement_keys"]) == set(expected_data_keys),
            f"{record_label}.data_requirement_keys do not exactly match node plan",
        )
        expected_prediction_requirement_keys = sorted(
            key
            for key, requirement in requirements_by_key.items()
            if requirement["consumer_node"] == node_id
        )
        require(
            len(record["prediction_requirement_keys"])
            == len(expected_prediction_requirement_keys)
            and set(record["prediction_requirement_keys"])
            == set(expected_prediction_requirement_keys),
            f"{record_label}.prediction_requirement_keys do not exactly match "
            "incoming OOF requirements",
        )
    return bundle


def patch_value_in_effective_plan(
    plan: dict[str, Any], patch: dict[str, Any], label: str
) -> Any:
    node_plans = plan.get("node_plans")
    require(isinstance(node_plans, dict), f"{label} plan.node_plans is invalid")
    require(
        patch["node_id"] in node_plans, f"{label}.node_id is absent from effective plan"
    )
    node_plan = node_plans[patch["node_id"]]
    namespace_roots = {
        "operator": "params",
        "fit": "fit_params",
        "control": "control_params",
        "structural": "structural_params",
    }
    root = namespace_roots[patch["namespace"]]
    require(
        root in node_plan,
        f"{label} namespace `{patch['namespace']}` is not materialized",
    )
    cursor: Any = node_plan[root]
    for index, segment in enumerate(patch["path"]):
        require(isinstance(cursor, dict), f"{label}.path[{index}] crosses a scalar")
        require(
            segment in cursor, f"{label}.path[{index}] is absent from effective plan"
        )
        cursor = cursor[segment]
    return cursor


def validate_training_outcome(value: Any, label: str) -> dict[str, Any]:
    required = {
        "schema_version",
        "outcome_id",
        "run_id",
        "training_request_fingerprint",
        "data_identities",
        "effective_plan",
        "effective_plan_fingerprint",
        "selected_variant_id",
        "selected_variant_fingerprint",
        "selection_output_id",
        "parameter_patches",
        "refit",
        "score_set",
        "outputs",
        "lineage",
        "portable_prediction_caches",
        "training_influence",
        "execution_bundle",
        "replayable_phases",
        "warnings",
        "diagnostics",
        "outcome_fingerprint",
    }
    outcome = require_exact_keys(value, required, set(), label)
    require(
        isinstance(outcome["score_set"], dict), f"{label}.score_set must be non-null"
    )
    _normalize_training_outcome(outcome)  # fail closed on serde container types
    validate_strict_json_value(outcome, label)
    require(
        not contains_runtime_handle(outcome),
        f"{label} must not contain runtime handles",
    )
    require_version_one(outcome["schema_version"], label)
    for field in ("outcome_id", "run_id", "selected_variant_id"):
        require_identifier(outcome[field], f"{label}.{field}")
    require_sha256(
        outcome["training_request_fingerprint"],
        f"{label}.training_request_fingerprint",
    )
    plan = outcome["effective_plan"]
    validate_execution_plan(plan, f"{label}.effective_plan")
    identities = [
        validate_w10_data_identity(identity, f"{label}.data_identities[{index}]")
        for index, identity in enumerate(outcome["data_identities"])
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    bindings = {
        f"{binding['node_id']}.{binding['input_name']}": binding
        for values in plan["campaign"].get("data_bindings", {}).values()
        for binding in values
    }
    require(
        identity_keys == sorted(bindings),
        f"{label}.data identities must exactly cover campaign bindings in order",
    )
    for identity in identities:
        binding = bindings[identity["requirement_key"]]
        require(
            identity["schema_fingerprint"] == binding["schema_fingerprint"]
            and identity["plan_fingerprint"] == binding["plan_fingerprint"]
            and identity["relation_fingerprint"] == binding.get("relation_fingerprint"),
            f"{label}.data identity does not match campaign binding fingerprints",
        )
    require_sha256(
        outcome["effective_plan_fingerprint"], f"{label}.effective_plan_fingerprint"
    )
    require(
        outcome["effective_plan_fingerprint"]
        == dagml_tcv1_sha256(_normalize_execution_plan(plan)),
        f"{label}.effective_plan_fingerprint does not match TCV1 plan content",
    )
    variants = plan.get("variants")
    require(
        isinstance(variants, list), f"{label}.effective_plan.variants must be an array"
    )
    selected = [
        variant
        for variant in variants
        if variant.get("variant_id") == outcome["selected_variant_id"]
    ]
    require(len(selected) == 1, f"{label}.selected_variant_id is absent or duplicated")
    require_sha256(
        outcome["selected_variant_fingerprint"],
        f"{label}.selected_variant_fingerprint",
    )
    require(
        selected[0].get("fingerprint") == outcome["selected_variant_fingerprint"],
        f"{label}.selected_variant_fingerprint does not match effective plan",
    )

    patches = outcome["parameter_patches"]
    require(isinstance(patches, list), f"{label}.parameter_patches must be an array")
    patch_keys = [
        validate_parameter_patch(patch, f"{label}.parameter_patches[{index}]")
        for index, patch in enumerate(patches)
    ]
    require(
        patch_keys == sorted(patch_keys), f"{label}.parameter_patches must be sorted"
    )
    require(
        len(set(patch_keys)) == len(patch_keys), f"{label}.parameter_patches duplicate"
    )
    for index, patch in enumerate(patches):
        require(
            patch_value_in_effective_plan(
                plan, patch, f"{label}.parameter_patches[{index}]"
            )
            == patch["value"],
            f"{label}.parameter_patches[{index}] is not materialized in effective plan",
        )
    expected_patches = selected_variant_parameter_patches(
        selected[0], f"{label}.effective_plan.selected_variant"
    )
    require(
        patches == expected_patches,
        f"{label}.parameter_patches do not exactly match selected variant overrides",
    )

    score_set = outcome["score_set"]
    validate_score_set_fixture(score_set, f"{label}.score_set")
    require(score_set["plan_id"] == plan["id"], f"{label}.score_set.plan_id mismatch")
    reports = score_set["reports"]
    require(
        any(
            report.get("variant_id") == outcome["selected_variant_id"]
            for report in reports
        ),
        f"{label}.score_set has no report for selected variant",
    )

    outputs = outcome["outputs"]
    require(
        isinstance(outputs, list) and bool(outputs),
        f"{label}.outputs must be non-empty",
    )
    validated_outputs = [
        validate_bound_output(output, f"{label}.outputs[{index}]")
        for index, output in enumerate(outputs)
    ]
    for index, output in enumerate(validated_outputs):
        validate_output_binding_against_plan(
            output["binding"],
            plan,
            f"{label}.outputs[{index}].binding",
        )
    predictor_closure = execution_plan_transitive_node_ids(
        plan,
        {output["binding"]["node_id"] for output in validated_outputs},
        label,
    )
    # V1 standalone invariant: the effective predictor closure must cover every
    # effective plan node, so replay derivation trusts `closure` as the full
    # predictor without reconstructing a partial schedule from outcome data.
    require(
        predictor_closure == set(plan["node_plans"].keys()),
        f"{label} predictor closure must equal all effective plan nodes in V1",
    )
    binding_ids = [output["binding"]["binding_id"] for output in validated_outputs]
    require(
        binding_ids == sorted(binding_ids),
        f"{label}.outputs must be sorted by binding_id",
    )
    require(
        len(set(binding_ids)) == len(binding_ids),
        f"{label}.outputs duplicate binding_id",
    )

    refit = require_exact_keys(
        outcome["refit"], {"requested", "status", "strategy"}, set(), f"{label}.refit"
    )
    require(
        isinstance(refit["requested"], bool), f"{label}.refit.requested must be boolean"
    )
    if refit["requested"]:
        require(
            refit["status"] == "completed"
            and refit["strategy"] in {"refit_one", "refit_ensemble"},
            f"{label}.refit requested outcome is inconsistent",
        )
        require(
            all(
                output["binding"]["prediction_source"] == "final_refit"
                for output in outputs
            ),
            f"{label} completed refit outputs must use final_refit",
        )
    else:
        require(
            refit["status"] == "skipped" and refit["strategy"] is None,
            f"{label}.refit skipped outcome is inconsistent",
        )
        require(
            all(
                output["binding"]["prediction_source"] != "final_refit"
                for output in outputs
            ),
            f"{label} no-refit outputs cannot use final_refit",
        )

    influence = validate_training_influence_manifest(
        outcome["training_influence"], f"{label}.training_influence"
    )
    validate_training_influence_against_plan(influence, plan, predictor_closure, label)
    if patches:
        require(
            any(entry["kind"] == "hpo_selection" for entry in influence["entries"]),
            f"{label} parameter patches require hpo_selection influence",
        )
    fit_nodes = {
        entry["node_id"]
        for entry in influence["entries"]
        if entry["kind"] in {"transform_fit", "model_fit", "trained_meta_aggregation"}
        and entry["node_id"] is not None
    }
    for output in outputs:
        require(
            output["binding"]["node_id"] in fit_nodes,
            f"{label} output node has no fitting influence",
        )

    bundle = validate_execution_bundle_contract(
        outcome["execution_bundle"],
        f"{label}.execution_bundle",
        plan=plan,
        predictor_closure=predictor_closure,
        refit_requested=refit["requested"],
        selected_variant_id=outcome["selected_variant_id"],
        score_set=score_set,
    )
    require_identifier(outcome["selection_output_id"], f"{label}.selection_output_id")
    selection_outputs = [
        output
        for output in validated_outputs
        if output["binding"]["binding_id"] == outcome["selection_output_id"]
    ]
    require(
        len(selection_outputs) == 1,
        f"{label}.selection_output_id does not resolve exactly one output",
    )
    selection_binding = selection_outputs[0]["binding"]
    require(
        selection_binding["prediction_level"]
        == plan["campaign"]["aggregation_policy"]["selection_metric_level"],
        f"{label} selection output does not match campaign selection_metric_level",
    )
    selections = bundle["selections"]
    require(
        len(selections) == 1,
        f"{label} execution bundle must contain exactly one SELECT decision",
    )
    selection_key, decision = next(iter(selections.items()))
    require(
        selection_key == decision["policy_id"]
        and decision["selected_candidate_id"] == outcome["selected_variant_id"]
        and decision.get("metric_level") == selection_binding["prediction_level"]
        and decision.get("evaluation_scope") == "oof"
        and score_set.get("selection_metric") == decision["metric_name"],
        f"{label} SELECT decision metadata is inconsistent with selected output",
    )
    supported_metric = {
        "regression_point": {
            ("mse", "minimize"),
            ("rmse", "minimize"),
            ("mae", "minimize"),
            ("r2", "maximize"),
        },
        "class_label": {
            ("accuracy", "maximize"),
            ("balanced_accuracy", "maximize"),
        },
    }
    require(
        (decision["metric_name"], decision["objective"])
        in supported_metric.get(selection_binding["prediction_kind"], set()),
        f"{label} selection metric is incompatible with prediction kind",
    )
    average_reports: dict[str, dict[str, Any]] = {}
    for report in reports:
        if (
            report["producer_node"] == selection_binding["node_id"]
            and report["partition"] == "validation"
            and report["level"] == selection_binding["prediction_level"]
            and report.get("fold_id") == "avg"
        ):
            variant_id = report.get("variant_id")
            require_identifier(variant_id, f"{label} selection report variant_id")
            require(
                variant_id not in average_reports,
                f"{label} has multiple selection average reports for one variant",
            )
            average_reports[variant_id] = report
    expected_variants = {variant["variant_id"] for variant in plan["variants"]}
    require(
        set(average_reports) == expected_variants,
        f"{label} selection reports do not exactly cover plan variants",
    )
    candidates = [
        (variant_id, report["metrics"][decision["metric_name"]])
        for variant_id, report in average_reports.items()
    ]
    require(
        all(
            isinstance(score, (int, float)) and math.isfinite(float(score))
            for _, score in candidates
        ),
        f"{label} selection reports contain a missing or non-finite metric",
    )
    candidates.sort(
        key=(
            (lambda candidate: (candidate[1], candidate[0]))
            if decision["objective"] == "minimize"
            else (lambda candidate: (-candidate[1], candidate[0]))
        )
    )
    expected_ranking = [
        {"candidate_id": variant_id, "score": score, "rank": index + 1}
        for index, (variant_id, score) in enumerate(candidates)
    ]
    require(
        decision["ranked_candidates"] == expected_ranking
        and decision["selected_candidate_id"] == expected_ranking[0]["candidate_id"]
        and decision["selected_score"] == expected_ranking[0]["score"],
        f"{label} SELECT decision does not equal ranking reconstructed from scores",
    )
    require(
        influence["relation_fingerprint"]
        in {
            requirement["relation_fingerprint"]
            for requirement in bundle["data_requirements"]
        },
        f"{label} training influence relation is not bound by execution bundle",
    )
    if refit["requested"]:
        require(
            bundle["refit_artifacts"], f"{label} completed refit requires artifacts"
        )
        artifact_nodes = {record["node_id"] for record in bundle["refit_artifacts"]}
        require(
            all(output["binding"]["node_id"] in artifact_nodes for output in outputs),
            f"{label} final output has no refit artifact",
        )
    else:
        require(
            not bundle["refit_artifacts"],
            f"{label} no-refit outcome cannot contain refit artifacts",
        )

    lineage = outcome["lineage"]
    require(
        isinstance(lineage, list) and bool(lineage),
        f"{label}.lineage must be non-empty",
    )
    lineage_records = [
        validate_portable_lineage_record(
            record,
            f"{label}.lineage[{index}]",
            run_id=outcome["run_id"],
            allowed_phases={"FIT_CV", "SELECT", "REFIT"},
        )
        for index, record in enumerate(lineage)
    ]
    require(
        all(
            record["variant_id"] == outcome["selected_variant_id"]
            for record in lineage_records
        ),
        f"{label}.lineage variant does not match selected variant",
    )
    if refit["requested"]:
        require(
            any(record["phase"] == "REFIT" for record in lineage_records),
            f"{label} refit lineage missing",
        )
    validate_training_lineage_against_plan(
        lineage_records,
        plan,
        predictor_closure,
        bundle,
        refit_requested=refit["requested"],
        label=label,
    )

    caches = outcome["portable_prediction_caches"]
    if caches is not None:
        validate_portable_prediction_caches(
            caches,
            bundle,
            f"{label}.portable_prediction_caches",
        )
    else:
        require(
            not bundle["prediction_caches"],
            f"{label}.portable_prediction_caches cannot be null when bundle caches are present",
        )

    replayable = outcome["replayable_phases"]
    # `[]` is a valid, honest "no replay mode" answer, so the vector may be empty.
    validate_ordered_unique_strings(
        replayable, f"{label}.replayable_phases", require_non_empty=False
    )
    require(
        set(replayable) <= {"REFIT", "PREDICT", "EXPLAIN"},
        f"{label}.replayable_phases contains unsupported phase",
    )
    expected_replayable = derive_replayable_phases(
        plan,
        predictor_closure,
        refit["status"] == "completed",
        bundle,
        caches,
        label,
    )
    require(
        replayable == expected_replayable,
        f"{label}.replayable_phases do not match the phases derivable from the "
        "full predictor closure and retained state",
    )
    validate_ordered_unique_strings(
        outcome["warnings"], f"{label}.warnings", require_non_empty=False
    )
    require(
        outcome["warnings"] == sorted(outcome["warnings"]),
        f"{label}.warnings must be sorted",
    )
    validate_metadata_object(outcome["diagnostics"], f"{label}.diagnostics")
    require(
        dagml_tcv1_sha256(outcome)
        == dagml_tcv1_sha256(_normalize_training_outcome(outcome)),
        f"{label} wire content does not match its typed Rust serde representation",
    )
    require_sha256(outcome["outcome_fingerprint"], f"{label}.outcome_fingerprint")
    require(
        outcome["outcome_fingerprint"] == training_outcome_fingerprint(outcome),
        f"{label}.outcome_fingerprint does not match TCV1 outcome content",
    )
    return outcome


def validate_replay_outcome(value: Any, label: str) -> dict[str, Any]:
    fields = {
        "schema_version",
        "outcome_id",
        "run_id",
        "bundle_id",
        "plan_id",
        "phase",
        "result_count",
        "lineage_record_count",
        "prediction_block_count",
        "aggregated_prediction_block_count",
        "explanation_block_count",
        "controller_count",
        "prediction_cache_store",
        "outputs",
        "explanations",
        "lineage",
        "warnings",
        "diagnostics",
        "outcome_fingerprint",
    }
    outcome = require_exact_keys(value, fields, set(), label)
    validate_strict_json_value(outcome, label)
    require(
        not contains_runtime_handle(outcome),
        f"{label} must not contain runtime handles",
    )
    require_version_one(outcome["schema_version"], label)
    for field in ("outcome_id", "run_id", "bundle_id"):
        require_identifier(outcome[field], f"{label}.{field}")
    require_non_empty_string(outcome["plan_id"], f"{label}.plan_id")
    require(
        outcome["phase"] in {"REFIT", "PREDICT", "EXPLAIN"}, f"{label}.phase invalid"
    )
    for field in (
        "result_count",
        "lineage_record_count",
        "prediction_block_count",
        "aggregated_prediction_block_count",
        "explanation_block_count",
        "controller_count",
    ):
        require_non_negative_int(outcome[field], f"{label}.{field}")
    require(
        isinstance(outcome["prediction_cache_store"], bool),
        f"{label}.prediction_cache_store must be boolean",
    )
    outputs = outcome["outputs"]
    require(isinstance(outputs, list), f"{label}.outputs must be an array")
    validated_outputs = [
        validate_bound_output(output, f"{label}.outputs[{index}]")
        for index, output in enumerate(outputs)
    ]
    prediction_count = sum(
        len(output["predictions"]) + len(output["observation_predictions"])
        for output in validated_outputs
    )
    aggregated_count = sum(
        len(output["aggregated_predictions"]) for output in validated_outputs
    )
    require(
        outcome["prediction_block_count"] == prediction_count,
        f"{label}.prediction_block_count does not match payload",
    )
    require(
        outcome["aggregated_prediction_block_count"] == aggregated_count,
        f"{label}.aggregated_prediction_block_count does not match payload",
    )

    explanations = outcome["explanations"]
    require(isinstance(explanations, list), f"{label}.explanations must be an array")
    for index, explanation in enumerate(explanations):
        explanation_label = f"{label}.explanations[{index}]"
        explanation = require_exact_keys(
            explanation,
            {"producer_node", "method", "payload"},
            {"target_name"},
            explanation_label,
        )
        require_identifier(
            explanation["producer_node"], f"{explanation_label}.producer_node"
        )
        require_non_empty_string(explanation["method"], f"{explanation_label}.method")
        if "target_name" in explanation:
            require_non_empty_string(
                explanation["target_name"], f"{explanation_label}.target_name"
            )
        validate_strict_json_value(
            explanation["payload"], f"{explanation_label}.payload"
        )
    require(
        outcome["explanation_block_count"] == len(explanations),
        f"{label}.explanation_block_count does not match payload",
    )
    if outcome["phase"] == "PREDICT":
        require(not explanations, f"{label} PREDICT cannot emit explanations")
    if outcome["phase"] == "EXPLAIN":
        require(bool(explanations), f"{label} EXPLAIN must emit explanations")

    lineage = outcome["lineage"]
    require(isinstance(lineage, list), f"{label}.lineage must be an array")
    lineage_records = [
        validate_portable_lineage_record(
            record,
            f"{label}.lineage[{index}]",
            run_id=outcome["run_id"],
            allowed_phases={outcome["phase"]},
        )
        for index, record in enumerate(lineage)
    ]
    require(
        outcome["lineage_record_count"] == len(lineage_records),
        f"{label}.lineage_record_count does not match payload",
    )
    require(
        outcome["result_count"] == len(lineage_records),
        f"{label}.result_count does not match replay results",
    )
    producer_nodes = {output["binding"]["node_id"] for output in validated_outputs} | {
        explanation["producer_node"] for explanation in explanations
    }
    lineage_nodes = {record["node_id"] for record in lineage_records}
    require(producer_nodes <= lineage_nodes, f"{label} emitted payload lacks lineage")
    validate_ordered_unique_strings(
        outcome["warnings"], f"{label}.warnings", require_non_empty=False
    )
    validate_metadata_object(outcome["diagnostics"], f"{label}.diagnostics")
    require_sha256(outcome["outcome_fingerprint"], f"{label}.outcome_fingerprint")
    require(
        outcome["outcome_fingerprint"] == replay_outcome_fingerprint(outcome),
        f"{label}.outcome_fingerprint does not match TCV1 outcome content",
    )
    return outcome


def validate_estimator_contract_fixtures(
    parameter_patch: Any,
    output_binding: Any,
    training_refit: Any,
    training_no_refit: Any,
    replay_outcomes: list[tuple[Path, Any]],
    conformal_fixture: Any,
    label: str,
) -> None:
    validate_parameter_patch(parameter_patch, f"{label}.parameter_patch")
    validate_output_binding(output_binding, f"{label}.output_binding")
    refit = validate_training_outcome(training_refit, f"{label}.training_outcome_refit")
    no_refit = validate_training_outcome(
        training_no_refit, f"{label}.training_outcome_no_refit"
    )
    require(
        parameter_patch in refit["parameter_patches"]
        and parameter_patch in no_refit["parameter_patches"],
        f"{label} standalone ParameterPatch drifted from TrainingOutcome",
    )
    require(
        output_binding == refit["outputs"][0]["binding"],
        f"{label} standalone OutputBinding drifted from refit TrainingOutcome",
    )
    require(
        refit["effective_plan"] == no_refit["effective_plan"]
        and refit["effective_plan_fingerprint"]
        == no_refit["effective_plan_fingerprint"],
        f"{label} refit/no-refit fixtures must share one effective plan",
    )
    require(
        refit["parameter_patches"] == no_refit["parameter_patches"],
        f"{label} refit/no-refit selected patches drifted",
    )
    require(
        refit["training_influence"] == no_refit["training_influence"],
        f"{label} refit/no-refit influence drifted",
    )
    require(
        refit["portable_prediction_caches"] is not None
        and bool(refit["portable_prediction_caches"]["caches"])
        and bool(refit["execution_bundle"]["prediction_requirements"])
        and bool(refit["execution_bundle"]["prediction_caches"]),
        f"{label} refit fixture must exercise a positive portable OOF cache",
    )
    require(
        no_refit["portable_prediction_caches"] == refit["portable_prediction_caches"]
        and no_refit["execution_bundle"]["prediction_requirements"]
        == refit["execution_bundle"]["prediction_requirements"]
        and no_refit["execution_bundle"]["prediction_caches"]
        == refit["execution_bundle"]["prediction_caches"],
        f"{label} no-refit fixture must preserve the complete portable OOF cache",
    )

    replay_by_name: dict[str, dict[str, Any]] = {}
    for fixture_path, replay_value in replay_outcomes:
        replay_by_name[fixture_path.name] = validate_replay_outcome(
            replay_value, f"{label}.{fixture_path.name}"
        )
        for output_index, output in enumerate(
            replay_by_name[fixture_path.name]["outputs"]
        ):
            validate_output_binding_against_plan(
                output["binding"],
                refit["effective_plan"],
                f"{label}.{fixture_path.name}.outputs[{output_index}].binding",
            )
    require(
        replay_by_name["replay_outcome_predict.v1.json"]["outputs"][0]["binding"]
        == output_binding,
        f"{label} prediction replay OutputBinding drifted",
    )
    require(
        replay_by_name["replay_outcome_explain.v1.json"]["outputs"][0]["binding"]
        == output_binding,
        f"{label} explanation replay OutputBinding drifted",
    )

    conformal = conformal_fixture["calibration_artifact"]
    predictor = conformal["predictor_binding"]
    require(
        predictor["training_outcome_fingerprint"] == refit["outcome_fingerprint"],
        f"{label} conformal predictor is not bound to TrainingOutcome",
    )
    require(
        predictor["training_influence_fingerprint"]
        == refit["training_influence"]["manifest_fingerprint"]
        and conformal["training_influence"] == refit["training_influence"],
        f"{label} conformal training influence drifted from TrainingOutcome",
    )
    require(
        predictor["selected_patches"] == refit["parameter_patches"]
        and predictor["output_binding"] == output_binding,
        f"{label} conformal patch/output binding drifted from TrainingOutcome",
    )
    require(
        predictor["plan_id"] == refit["effective_plan"]["id"]
        and predictor["graph_fingerprint"]
        == refit["effective_plan"]["graph_fingerprint"]
        and predictor["campaign_fingerprint"]
        == refit["effective_plan"]["campaign_fingerprint"]
        and predictor["controller_fingerprint"]
        == refit["effective_plan"]["controller_fingerprint"]
        and predictor["selected_variant_id"] == refit["selected_variant_id"]
        and predictor["selected_variant_fingerprint"]
        == refit["selected_variant_fingerprint"],
        f"{label} conformal predictor plan/selection drifted",
    )
    plan_data_bindings = {
        f"{node_id}.{binding['input_name']}": binding
        for node_id, node_plan in refit["effective_plan"]["node_plans"].items()
        for binding in node_plan["data_bindings"]
    }
    expected_predictor_data_bindings = [
        {
            "requirement_key": key,
            "schema_fingerprint": requirement["schema_fingerprint"],
            "plan_fingerprint": requirement["plan_fingerprint"],
            "relation_fingerprint": requirement["relation_fingerprint"],
            "source_ids": plan_data_bindings[key]["source_ids"],
        }
        for key, requirement in sorted(
            (
                (f"{requirement['node_id']}.{requirement['input_name']}", requirement)
                for requirement in refit["execution_bundle"]["data_requirements"]
            )
        )
    ]
    require(
        predictor["data_bindings"] == expected_predictor_data_bindings,
        f"{label} conformal predictor data bindings do not exactly match "
        "TrainingOutcome predictor closure",
    )
    expected_predictor_artifacts = [
        {
            "node_id": record["node_id"],
            "controller_id": record["controller_id"],
            "artifact_id": record["artifact"]["id"],
            "backend": record["artifact"]["backend"],
            "content_fingerprint": record["artifact"]["content_fingerprint"],
            "params_fingerprint": record["params_fingerprint"],
            "plugin": record["artifact"].get("plugin"),
            "plugin_version": record["artifact"].get("plugin_version"),
        }
        for record in refit["execution_bundle"]["refit_artifacts"]
    ]
    require(
        predictor["artifacts"] == expected_predictor_artifacts,
        f"{label} conformal predictor artifacts do not exactly match TrainingOutcome",
    )

    bad_score = copy.deepcopy(refit)
    bad_score["score_set"] = None
    expect_contract_error(
        lambda: validate_training_outcome(bad_score, f"{label}.null_score_set"),
        "score_set must be non-null",
        f"{label}.null_score_set",
    )
    bad_patch = copy.deepcopy(refit)
    bad_patch["parameter_patches"][0]["value"] = 0.9
    expect_contract_error(
        lambda: validate_training_outcome(bad_patch, f"{label}.unmaterialized_patch"),
        "is not materialized in effective plan",
        f"{label}.unmaterialized_patch",
    )
    missing_artifact = copy.deepcopy(refit)
    missing_artifact["execution_bundle"]["refit_artifacts"] = []
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_artifact, f"{label}.missing_refit_artifact"
        ),
        "refit_artifacts nodes do not exactly match",
        f"{label}.missing_refit_artifact",
    )
    missing_base_artifact = copy.deepcopy(refit)
    missing_base_artifact["execution_bundle"]["refit_artifacts"].pop(0)
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_base_artifact, f"{label}.missing_base_refit_artifact"
        ),
        "refit_artifacts nodes do not exactly match",
        f"{label}.missing_base_refit_artifact",
    )
    unexpected_transform_artifact = copy.deepcopy(refit)
    transform_artifact = copy.deepcopy(
        unexpected_transform_artifact["execution_bundle"]["refit_artifacts"][0]
    )
    transform_artifact.update(
        {
            "node_id": "branch:b1.augment:noise",
            "controller_id": "controller:augmentation.mock",
            "params_fingerprint": "33d2aa022dffa4cf09394b0edc178bf379bab06c569e00b7260fb5c546f433c6",
            "data_requirement_keys": ["branch:b1.augment:noise.x"],
        }
    )
    transform_artifact["artifact"].update(
        {
            "id": "artifact:branch:b1.augment:noise:refit",
            "controller_id": "controller:augmentation.mock",
        }
    )
    unexpected_transform_artifact["execution_bundle"]["refit_artifacts"].insert(
        1, transform_artifact
    )
    expect_contract_error(
        lambda: validate_training_outcome(
            unexpected_transform_artifact,
            f"{label}.unexpected_transform_refit_artifact",
        ),
        "refit_artifacts nodes do not exactly match",
        f"{label}.unexpected_transform_refit_artifact",
    )

    missing_fit_fold = copy.deepcopy(refit)
    missing_fit_fold["lineage"] = [
        record
        for record in missing_fit_fold["lineage"]
        if not (
            record["node_id"] == "merge:stack.pred_plus_original.meta:ridge"
            and record["phase"] == "FIT_CV"
            and record["fold_id"] == "fold:1"
        )
    ]
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_fit_fold, f"{label}.missing_fit_fold_lineage"
        ),
        "FIT_CV records do not exactly cover",
        f"{label}.missing_fit_fold_lineage",
    )
    missing_transform_refit = copy.deepcopy(refit)
    missing_transform_refit["lineage"] = [
        record
        for record in missing_transform_refit["lineage"]
        if not (
            record["node_id"] == "branch:b1.augment:noise"
            and record["phase"] == "REFIT"
        )
    ]
    next(
        record
        for record in missing_transform_refit["lineage"]
        if record["node_id"] == "branch:b1.model:rf" and record["phase"] == "REFIT"
    )["input_lineage"] = []
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_transform_refit, f"{label}.missing_transform_refit_lineage"
        ),
        "REFIT records do not exactly match",
        f"{label}.missing_transform_refit_lineage",
    )
    wrong_upstream_lineage = copy.deepcopy(refit)
    b1_fit_record = next(
        record
        for record in wrong_upstream_lineage["lineage"]
        if record["node_id"] == "branch:b1.model:rf"
        and record["phase"] == "FIT_CV"
        and record["fold_id"] == "fold:0"
    )
    b1_fit_record["input_lineage"] = [
        "lineage:branch:b0.model:ridge:FIT_CV:variant:a964828b1417c6e7:fold:0"
    ]
    expect_contract_error(
        lambda: validate_training_outcome(
            wrong_upstream_lineage, f"{label}.wrong_upstream_lineage"
        ),
        "input_lineage does not exactly match upstream",
        f"{label}.wrong_upstream_lineage",
    )
    oversized_lineage_seed = copy.deepcopy(refit)
    oversized_lineage_seed["lineage"][0]["seed"] = CONFORMAL_INT_MAX + 1
    expect_contract_error(
        lambda: validate_training_outcome(
            oversized_lineage_seed, f"{label}.oversized_lineage_seed"
        ),
        "integer is outside the TCV1 serde-compatible range",
        f"{label}.oversized_lineage_seed",
    )

    missing_transform_influence = copy.deepcopy(refit)
    missing_transform_influence["training_influence"]["entries"] = [
        entry
        for entry in missing_transform_influence["training_influence"]["entries"]
        if entry["node_id"] != "branch:b1.augment:noise"
    ]
    missing_transform_influence["training_influence"]["manifest_fingerprint"] = (
        conformal_manifest_fingerprint(
            missing_transform_influence["training_influence"],
            "manifest_fingerprint",
        )
    )
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_transform_influence, f"{label}.missing_transform_influence"
        ),
        "fitting nodes do not exactly match predictor closure",
        f"{label}.missing_transform_influence",
    )

    incomplete_selected_patches = copy.deepcopy(refit)
    incomplete_selected_patches["parameter_patches"].pop(0)
    expect_contract_error(
        lambda: validate_training_outcome(
            incomplete_selected_patches, f"{label}.incomplete_selected_patches"
        ),
        "do not exactly match selected variant overrides",
        f"{label}.incomplete_selected_patches",
    )
    bad_width = copy.deepcopy(replay_by_name["replay_outcome_predict.v1.json"])
    bad_width["outputs"][0]["predictions"][0]["values"][0].append(9.0)
    expect_contract_error(
        lambda: validate_replay_outcome(bad_width, f"{label}.bad_prediction_width"),
        "width does not match OutputBinding",
        f"{label}.bad_prediction_width",
    )
    bad_count = copy.deepcopy(replay_by_name["replay_outcome_predict.v1.json"])
    bad_count["prediction_block_count"] = 2
    expect_contract_error(
        lambda: validate_replay_outcome(bad_count, f"{label}.bad_prediction_count"),
        "prediction_block_count does not match payload",
        f"{label}.bad_prediction_count",
    )

    ghost_port = copy.deepcopy(refit)
    ghost_binding = ghost_port["outputs"][0]["binding"]
    ghost_binding["port_name"] = "ghost_port"
    ghost_binding["binding_fingerprint"] = output_binding_fingerprint(ghost_binding)
    ghost_port["outcome_fingerprint"] = training_outcome_fingerprint(ghost_port)
    expect_contract_error(
        lambda: validate_training_outcome(ghost_port, f"{label}.ghost_output_port"),
        "is not a unique output",
        f"{label}.ghost_output_port",
    )

    ghost_influence = copy.deepcopy(refit)
    ghost_influence["training_influence"]["entries"][1]["node_id"] = "ghost:model"
    ghost_influence["training_influence"]["manifest_fingerprint"] = (
        conformal_manifest_fingerprint(
            ghost_influence["training_influence"], "manifest_fingerprint"
        )
    )
    ghost_influence["outcome_fingerprint"] = training_outcome_fingerprint(
        ghost_influence
    )
    expect_contract_error(
        lambda: validate_training_outcome(
            ghost_influence, f"{label}.ghost_influence_node"
        ),
        "is absent from graph",
        f"{label}.ghost_influence_node",
    )

    mismatched_artifact_controller = copy.deepcopy(refit)
    mismatched_artifact_controller["execution_bundle"]["refit_artifacts"][0][
        "artifact"
    ]["controller_id"] = "controller:augmentation.mock"
    mismatched_artifact_controller["outcome_fingerprint"] = (
        training_outcome_fingerprint(mismatched_artifact_controller)
    )
    expect_contract_error(
        lambda: validate_training_outcome(
            mismatched_artifact_controller, f"{label}.artifact_controller_mismatch"
        ),
        "artifact.controller_id does not match record",
        f"{label}.artifact_controller_mismatch",
    )
    mismatched_artifact_params = copy.deepcopy(refit)
    mismatched_artifact_params["execution_bundle"]["refit_artifacts"][0][
        "params_fingerprint"
    ] = "2424242424242424242424242424242424242424242424242424242424242424"
    expect_contract_error(
        lambda: validate_training_outcome(
            mismatched_artifact_params, f"{label}.artifact_params_mismatch"
        ),
        "params_fingerprint does not match plan",
        f"{label}.artifact_params_mismatch",
    )
    missing_artifact_data_link = copy.deepcopy(refit)
    missing_artifact_data_link["execution_bundle"]["refit_artifacts"][0][
        "data_requirement_keys"
    ] = []
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_artifact_data_link, f"{label}.missing_artifact_data_link"
        ),
        "data_requirement_keys do not exactly match node plan",
        f"{label}.missing_artifact_data_link",
    )
    missing_artifact_prediction_link = copy.deepcopy(refit)
    missing_artifact_prediction_link["execution_bundle"]["refit_artifacts"][2][
        "prediction_requirement_keys"
    ].pop()
    expect_contract_error(
        lambda: validate_training_outcome(
            missing_artifact_prediction_link,
            f"{label}.missing_artifact_prediction_link",
        ),
        "prediction_requirement_keys do not exactly match incoming OOF requirements",
        f"{label}.missing_artifact_prediction_link",
    )

    bad_requirement = copy.deepcopy(refit)
    bad_requirement["execution_bundle"]["prediction_requirements"][0][
        "prediction_width"
    ] = 2
    bad_requirement["outcome_fingerprint"] = training_outcome_fingerprint(
        bad_requirement
    )
    expect_contract_error(
        lambda: validate_training_outcome(
            bad_requirement, f"{label}.bad_cache_requirement"
        ),
        "target_names length does not match prediction_width",
        f"{label}.bad_cache_requirement",
    )

    orphan_payload = copy.deepcopy(refit)
    orphan_payload["portable_prediction_caches"]["caches"][1]["requirement_key"] = (
        "ghost:model.oof->ghost:consumer.input"
    )
    orphan_payload["outcome_fingerprint"] = training_outcome_fingerprint(orphan_payload)
    expect_contract_error(
        lambda: validate_training_outcome(
            orphan_payload, f"{label}.orphan_cache_payload"
        ),
        "do not exactly match execution bundle cache records",
        f"{label}.orphan_cache_payload",
    )
    incomplete_no_refit_bundle_cache = copy.deepcopy(no_refit)
    incomplete_no_refit_bundle_cache["execution_bundle"]["prediction_caches"].pop()
    expect_contract_error(
        lambda: validate_training_outcome(
            incomplete_no_refit_bundle_cache,
            f"{label}.incomplete_no_refit_bundle_cache",
        ),
        "prediction_caches do not exactly cover prediction requirements",
        f"{label}.incomplete_no_refit_bundle_cache",
    )
    incomplete_no_refit_payload_cache = copy.deepcopy(no_refit)
    incomplete_no_refit_payload_cache["portable_prediction_caches"]["caches"].pop()
    expect_contract_error(
        lambda: validate_training_outcome(
            incomplete_no_refit_payload_cache,
            f"{label}.incomplete_no_refit_payload_cache",
        ),
        "do not exactly match execution bundle cache records",
        f"{label}.incomplete_no_refit_payload_cache",
    )


def validate_estimator_draft_2020_contracts(
    registry: Registry,
    schemas: dict[str, dict[str, Any]],
    parameter_patch: Any,
    output_binding: Any,
    training_refit: Any,
    training_no_refit: Any,
    replay_outcomes: list[tuple[Path, Any]],
    conformal_fixture: Any,
    label: str,
) -> None:
    instances: list[tuple[str, Any, str]] = [
        (PARAMETER_PATCH_SCHEMA_ID, parameter_patch, "parameter_patch"),
        (OUTPUT_BINDING_SCHEMA_ID, output_binding, "output_binding"),
        (TRAINING_OUTCOME_SCHEMA_ID, training_refit, "training_outcome_refit"),
        (TRAINING_OUTCOME_SCHEMA_ID, training_no_refit, "training_outcome_no_refit"),
        (
            CONFORMAL_CALIBRATION_SCHEMA_ID,
            conformal_fixture["calibration_artifact"],
            "conformal_calibration_artifact",
        ),
    ]
    instances.extend(
        (REPLAY_OUTCOME_SCHEMA_ID, value, path.name) for path, value in replay_outcomes
    )
    for schema_id, instance, instance_label in instances:
        require(schema_id in schemas, f"{label} registry is missing `{schema_id}`")
        validate_draft_2020_instance(
            instance,
            schemas[schema_id],
            registry,
            f"{label}.{instance_label}",
        )

    unknown_root = copy.deepcopy(training_refit)
    unknown_root["unexpected_runtime_field"] = True
    expect_contract_error(
        lambda: validate_draft_2020_instance(
            unknown_root,
            schemas[TRAINING_OUTCOME_SCHEMA_ID],
            registry,
            f"{label}.unknown_training_field",
        ),
        "Additional properties are not allowed",
        f"{label}.unknown_training_field",
    )
    unknown_binding = copy.deepcopy(training_refit)
    unknown_binding["outputs"][0]["binding"]["opaque_handle"] = "host:model:1"
    expect_contract_error(
        lambda: validate_draft_2020_instance(
            unknown_binding,
            schemas[TRAINING_OUTCOME_SCHEMA_ID],
            registry,
            f"{label}.unknown_output_binding_field",
        ),
        "Additional properties are not allowed",
        f"{label}.unknown_output_binding_field",
    )


def validate_conformal_calibration_schema(schema: Any, label: str) -> None:
    require(isinstance(schema, dict), f"{label} conformal schema must be an object")
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        f"{label} conformal schema must declare Draft 2020-12",
    )
    require(
        schema.get("$id") == CONFORMAL_CALIBRATION_SCHEMA_ID,
        f"{label} conformal schema has unexpected $id",
    )
    require(schema.get("type") == "object", f"{label} conformal root must be an object")
    require(
        schema.get("additionalProperties") is False,
        f"{label} conformal root must reject unknown fields",
    )
    required = schema.get("required")
    require(isinstance(required, list), f"{label} conformal required list is missing")
    for field in (
        "schema_version",
        "predictor_binding",
        "calibration_spec",
        "calibration_cohort",
        "training_influence",
        "quantiles",
        "diagnostics",
        "warnings",
        "checksum",
    ):
        require(field in required, f"{label} conformal schema must require `{field}`")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} conformal properties are missing")
    require(
        properties.get("schema_version", {}).get("const") == 1,
        f"{label} conformal schema_version must be pinned to 1",
    )
    require(
        properties.get("schema_version", {}).get("type") == "integer",
        f"{label} conformal schema_version must reject booleans",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{label} conformal schema $defs are missing")
    for definition in (
        "cohort_manifest",
        "conformal_output_binding",
        "predictor_binding",
        "coverage_array",
        "calibration_spec",
        "tagged_quantile",
        "calibration_quantile",
    ):
        require(
            definition in defs,
            f"{label} conformal schema is missing $defs.{definition}",
        )
    for extracted_definition in (
        "training_influence_manifest",
        "training_influence_entry",
        "parameter_namespace",
        "parameter_patch",
        "output_binding",
    ):
        require(
            extracted_definition not in defs,
            f"{label} must reference canonical `{extracted_definition}` ownership",
        )
    require(
        properties.get("training_influence", {}).get("$ref")
        == TRAINING_INFLUENCE_SCHEMA_ID,
        f"{label} training_influence must use the canonical external schema",
    )
    require(
        properties.get("diagnostics", {}).get("default") == {}
        and properties.get("warnings", {}).get("default") == [],
        f"{label} warning/diagnostic defaults drifted",
    )
    require(
        defs.get("exchangeability_unit", {}).get("const") == "physical_sample",
        f"{label} conformal V1 must pin exchangeability to physical_sample",
    )
    coverage = defs.get("coverage_array")
    require(isinstance(coverage, dict), f"{label} conformal coverage_array is missing")
    require(
        coverage.get("type") == "array", f"{label} conformal coverage must be an array"
    )
    require(
        coverage.get("uniqueItems") is True,
        f"{label} conformal coverages must be unique",
    )
    require(
        defs.get("calibration_spec", {})
        .get("properties", {})
        .get("method", {})
        .get("const")
        == "split_absolute_residual",
        f"{label} conformal V1 method must be split_absolute_residual",
    )
    spec_properties = defs.get("calibration_spec", {}).get("properties", {})
    require(
        spec_properties.get("numeric_version", {}).get("const")
        == "split_absolute_residual.v1",
        f"{label} conformal numeric_version is not pinned",
    )
    require(
        spec_properties.get("seed", {}).get("anyOf", [{}])[0].get("maximum")
        == CONFORMAL_PORTABLE_INT_MAX,
        f"{label} conformal seed is not portably bounded",
    )
    cohort = defs.get("cohort_manifest", {})
    require(
        cohort.get("$ref")
        == "https://github.com/GBeurier/dag-ml/schemas/cohort_manifest.v1.schema.json",
        f"{label} conformal cohort must use the canonical standalone schema",
    )
    predictor_required = set(defs.get("predictor_binding", {}).get("required", []))
    require(
        {"training_outcome_fingerprint", "training_influence_fingerprint"}
        <= predictor_required,
        f"{label} conformal PredictorBinding must bind outcome and influence",
    )
    predictor_properties = defs.get("predictor_binding", {}).get("properties", {})
    require(
        predictor_properties.get("selected_patches", {}).get("items", {}).get("$ref")
        == PARAMETER_PATCH_SCHEMA_ID,
        f"{label} selected_patches must use the canonical ParameterPatch schema",
    )
    require(
        predictor_properties.get("selected_patches", {}).get("default") == [],
        f"{label} selected_patches default drifted",
    )
    require(
        predictor_properties.get("output_binding", {}).get("$ref")
        == "#/$defs/conformal_output_binding",
        f"{label} output_binding must use the conformal OutputBinding profile",
    )
    artifact_binding_required = set(
        defs.get("predictor_artifact_binding", {}).get("required", [])
    )
    require(
        {"plugin", "plugin_version"} <= artifact_binding_required,
        f"{label} conformal artifact plugin fields must be explicit",
    )
    artifact_binding_properties = defs.get("predictor_artifact_binding", {}).get(
        "properties", {}
    )
    require(
        "default" in artifact_binding_properties.get("plugin", {})
        and artifact_binding_properties.get("plugin", {}).get("default") is None
        and "default" in artifact_binding_properties.get("plugin_version", {})
        and artifact_binding_properties.get("plugin_version", {}).get("default")
        is None,
        f"{label} conformal artifact plugin defaults drifted",
    )
    calibration_spec_required = set(
        defs.get("calibration_spec", {}).get("required", [])
    )
    require(
        "seed" in calibration_spec_required,
        f"{label} conformal calibration seed must be explicit",
    )
    require(
        "default" in spec_properties.get("seed", {})
        and spec_properties.get("seed", {}).get("default") is None,
        f"{label} conformal calibration seed default drifted",
    )
    output_profile = defs.get("conformal_output_binding", {}).get("allOf", [])
    require(
        isinstance(output_profile, list)
        and len(output_profile) == 2
        and output_profile[0].get("$ref") == OUTPUT_BINDING_SCHEMA_ID,
        f"{label} conformal output must refine canonical OutputBinding",
    )
    output_properties = output_profile[1].get("properties", {})
    require(
        output_properties.get("unit_level", {}).get("const") == "physical_sample"
        and output_properties.get("prediction_kind", {}).get("const")
        == "regression_point",
        f"{label} conformal output unit/kind are not pinned",
    )
    require(
        output_properties.get("output_order", {}).get("const") == "target_order"
        and output_properties.get("target_space", {}).get("const") == "raw"
        and output_properties.get("class_labels", {}).get("items", {}).get("maxItems")
        == 0,
        f"{label} conformal output order/target-space/class vocabulary drifted",
    )
    tagged = defs.get("tagged_quantile", {}).get("oneOf")
    require(
        isinstance(tagged, list) and len(tagged) == 2,
        f"{label} tagged quantile is invalid",
    )
    statuses = {
        branch.get("properties", {}).get("status", {}).get("const")
        for branch in tagged
        if isinstance(branch, dict)
    }
    require(
        statuses == {"finite", "unbounded"},
        f"{label} tagged quantile must define finite and unbounded",
    )
    finite_branch = next(
        branch
        for branch in tagged
        if branch.get("properties", {}).get("status", {}).get("const") == "finite"
    )
    require(
        finite_branch.get("properties", {})
        .get("value", {})
        .get("x-dagml-json-token-type")
        == "binary64-float",
        f"{label} finite quantile float-token profile drifted",
    )


def validate_conformal_cohort_manifest(value: Any, label: str) -> dict[str, Any]:
    required = {
        "schema_version",
        "role",
        "exchangeability_unit",
        "physical_sample_ids",
        "origin_sample_ids",
        "group_ids",
        "source_ids",
        "unit_relations",
        "target_names",
        "relation_fingerprint",
        "content_fingerprint",
        "manifest_fingerprint",
    }
    required.add("axes_fingerprint")
    cohort = require_exact_keys(value, required, set(), label)
    require_version_one(cohort["schema_version"], label)
    require(cohort["role"] in CONFORMAL_COHORT_ROLES, f"{label} role is invalid")
    require(
        cohort["exchangeability_unit"] == "physical_sample",
        f"{label} exchangeability_unit must be physical_sample in V1",
    )
    validate_sorted_identifiers(
        cohort["physical_sample_ids"],
        f"{label}.physical_sample_ids",
        require_non_empty=True,
    )
    for field in ("origin_sample_ids", "group_ids", "source_ids"):
        validate_sorted_identifiers(
            cohort[field],
            f"{label}.{field}",
            require_non_empty=False,
        )
    unit_relations = cohort["unit_relations"]
    require(
        isinstance(unit_relations, list) and bool(unit_relations),
        f"{label}.unit_relations must be non-empty",
    )
    relation_sample_ids: list[str] = []
    relation_origins: set[str] = set()
    relation_groups: set[str] = set()
    relation_sources: set[str] = set()
    for index, relation_value in enumerate(unit_relations):
        relation_label = f"{label}.unit_relations[{index}]"
        relation = require_exact_keys(
            relation_value,
            {"physical_sample_id", "origin_sample_id", "group_ids", "source_ids"},
            set(),
            relation_label,
        )
        require_identifier(
            relation["physical_sample_id"], f"{relation_label}.physical_sample_id"
        )
        relation_sample_ids.append(relation["physical_sample_id"])
        require_optional_identifier(
            relation["origin_sample_id"], f"{relation_label}.origin_sample_id"
        )
        if relation["origin_sample_id"] is not None:
            relation_origins.add(relation["origin_sample_id"])
        validate_sorted_identifiers(
            relation["group_ids"],
            f"{relation_label}.group_ids",
            require_non_empty=False,
        )
        validate_sorted_identifiers(
            relation["source_ids"],
            f"{relation_label}.source_ids",
            require_non_empty=False,
        )
        relation_groups.update(relation["group_ids"])
        relation_sources.update(relation["source_ids"])
    require(
        relation_sample_ids == cohort["physical_sample_ids"],
        f"{label}.unit_relations do not align with physical_sample_ids",
    )
    require(
        relation_origins == set(cohort["origin_sample_ids"]),
        f"{label}.unit_relations origin closure drifted",
    )
    require(
        relation_groups == set(cohort["group_ids"]),
        f"{label}.unit_relations group closure drifted",
    )
    require(
        relation_sources == set(cohort["source_ids"]),
        f"{label}.unit_relations source closure drifted",
    )
    validate_ordered_unique_strings(
        cohort["target_names"],
        f"{label}.target_names",
        require_non_empty=True,
    )
    require_sha256(cohort["relation_fingerprint"], f"{label}.relation_fingerprint")
    require_sha256(cohort["content_fingerprint"], f"{label}.content_fingerprint")
    require_sha256(cohort["manifest_fingerprint"], f"{label}.manifest_fingerprint")
    require(
        cohort["manifest_fingerprint"]
        == conformal_manifest_fingerprint(cohort, "manifest_fingerprint"),
        f"{label}.manifest_fingerprint does not match TCV1 cohort content",
    )
    if cohort["axes_fingerprint"] is not None:
        require_sha256(cohort["axes_fingerprint"], f"{label}.axes_fingerprint")
    return cohort


def conformal_manifest_fingerprint(value: dict[str, Any], field: str) -> str:
    return dagml_tcv1_sha256(
        {key: member for key, member in value.items() if key != field}
    )


def validate_training_influence_manifest(value: Any, label: str) -> dict[str, Any]:
    influence = require_exact_keys(
        value,
        {
            "schema_version",
            "relation_fingerprint",
            "entries",
            "manifest_fingerprint",
        },
        set(),
        label,
    )
    require_version_one(influence["schema_version"], label)
    require_sha256(influence["relation_fingerprint"], f"{label}.relation_fingerprint")
    entries = influence["entries"]
    require(
        isinstance(entries, list) and bool(entries),
        f"{label}.entries must be non-empty",
    )
    kind_order = {kind: index for index, kind in enumerate(CONFORMAL_INFLUENCE_KINDS)}
    entry_keys: list[tuple[int, str, str]] = []
    for index, entry_value in enumerate(entries):
        entry_label = f"{label}.entries[{index}]"
        entry = require_exact_keys(
            entry_value,
            {
                "kind",
                "scope_id",
                "node_id",
                "physical_sample_ids",
                "origin_sample_ids",
                "group_ids",
            },
            set(),
            entry_label,
        )
        require(entry["kind"] in kind_order, f"{entry_label}.kind is invalid")
        require_identifier(entry["scope_id"], f"{entry_label}.scope_id")
        require_optional_identifier(entry["node_id"], f"{entry_label}.node_id")
        validate_sorted_identifiers(
            entry["physical_sample_ids"],
            f"{entry_label}.physical_sample_ids",
            require_non_empty=True,
        )
        validate_sorted_identifiers(
            entry["origin_sample_ids"],
            f"{entry_label}.origin_sample_ids",
            require_non_empty=False,
        )
        validate_sorted_identifiers(
            entry["group_ids"],
            f"{entry_label}.group_ids",
            require_non_empty=False,
        )
        entry_keys.append(
            (kind_order[entry["kind"]], entry["scope_id"], entry["node_id"] or "")
        )
    require(
        entry_keys == sorted(entry_keys), f"{label}.entries must be canonically sorted"
    )
    require(
        len(set(entry_keys)) == len(entry_keys), f"{label}.entries contain duplicates"
    )
    require_sha256(influence["manifest_fingerprint"], f"{label}.manifest_fingerprint")
    require(
        influence["manifest_fingerprint"]
        == conformal_manifest_fingerprint(influence, "manifest_fingerprint"),
        f"{label}.manifest_fingerprint does not match TCV1 manifest content",
    )
    return influence


def validate_conformal_parameter_patch(
    value: Any, label: str
) -> tuple[str, str, tuple[str, ...]]:
    return validate_parameter_patch(value, label)


def validate_predictor_data_binding(value: Any, label: str) -> str:
    binding = require_exact_keys(
        value,
        {
            "requirement_key",
            "schema_fingerprint",
            "plan_fingerprint",
            "relation_fingerprint",
            "source_ids",
        },
        set(),
        label,
    )
    require_non_empty_string(binding["requirement_key"], f"{label}.requirement_key")
    for field in ("schema_fingerprint", "plan_fingerprint", "relation_fingerprint"):
        require_sha256(binding[field], f"{label}.{field}")
    validate_sorted_identifiers(
        binding["source_ids"],
        f"{label}.source_ids",
        require_non_empty=False,
    )
    return binding["requirement_key"]


def validate_predictor_artifact_binding(value: Any, label: str) -> tuple[str, str]:
    artifact = require_exact_keys(
        value,
        {
            "node_id",
            "controller_id",
            "artifact_id",
            "backend",
            "content_fingerprint",
            "params_fingerprint",
            "plugin",
            "plugin_version",
        },
        set(),
        label,
    )
    for field in ("node_id", "controller_id", "artifact_id"):
        require_identifier(artifact[field], f"{label}.{field}")
    require_non_empty_string(artifact["backend"], f"{label}.backend")
    require_sha256(artifact["content_fingerprint"], f"{label}.content_fingerprint")
    require_sha256(artifact["params_fingerprint"], f"{label}.params_fingerprint")
    require_optional_non_empty_string(artifact["plugin"], f"{label}.plugin")
    require_optional_non_empty_string(
        artifact["plugin_version"], f"{label}.plugin_version"
    )
    require(
        artifact["plugin_version"] is None or artifact["plugin"] is not None,
        f"{label}.plugin_version requires plugin",
    )
    return artifact["node_id"], artifact["artifact_id"]


def validate_conformal_output_binding(value: Any, label: str) -> dict[str, Any]:
    return validate_output_binding(value, label, conformal_v1=True)


def validate_predictor_binding(value: Any, label: str) -> dict[str, Any]:
    predictor = require_exact_keys(
        value,
        {
            "schema_version",
            "plan_id",
            "graph_fingerprint",
            "campaign_fingerprint",
            "controller_fingerprint",
            "predictor_node_ids",
            "data_bindings",
            "selected_variant_id",
            "selected_variant_fingerprint",
            "selected_patches",
            "artifacts",
            "output_binding",
            "target_processing_fingerprint",
            "training_outcome_fingerprint",
            "training_influence_fingerprint",
        },
        set(),
        label,
    )
    require_version_one(predictor["schema_version"], label)
    require_non_empty_string(predictor["plan_id"], f"{label}.plan_id")
    for field in (
        "graph_fingerprint",
        "campaign_fingerprint",
        "controller_fingerprint",
        "selected_variant_fingerprint",
        "target_processing_fingerprint",
        "training_outcome_fingerprint",
        "training_influence_fingerprint",
    ):
        require_sha256(predictor[field], f"{label}.{field}")
    require_identifier(predictor["selected_variant_id"], f"{label}.selected_variant_id")
    predictor_node_ids = predictor["predictor_node_ids"]
    validate_sorted_identifiers(
        predictor_node_ids,
        f"{label}.predictor_node_ids",
        require_non_empty=True,
    )
    predictor_node_set = set(predictor_node_ids)

    data_bindings = predictor["data_bindings"]
    require(
        isinstance(data_bindings, list) and bool(data_bindings),
        f"{label}.data_bindings must be non-empty",
    )
    data_keys = [
        validate_predictor_data_binding(binding, f"{label}.data_bindings[{index}]")
        for index, binding in enumerate(data_bindings)
    ]
    require(data_keys == sorted(data_keys), f"{label}.data_bindings must be sorted")
    require(
        len(set(data_keys)) == len(data_keys),
        f"{label}.data_bindings contain duplicates",
    )
    data_binding_nodes = {
        requirement_key.rsplit(".", maxsplit=1)[0] for requirement_key in data_keys
    }
    require(
        all("." in requirement_key for requirement_key in data_keys),
        f"{label}.data_bindings requirement_key must end in a port name",
    )

    patches = predictor["selected_patches"]
    require(isinstance(patches, list), f"{label}.selected_patches must be an array")
    patch_keys = [
        validate_conformal_parameter_patch(patch, f"{label}.selected_patches[{index}]")
        for index, patch in enumerate(patches)
    ]
    require(
        patch_keys == sorted(patch_keys), f"{label}.selected_patches must be sorted"
    )
    require(
        len(set(patch_keys)) == len(patch_keys),
        f"{label}.selected_patches contain duplicates",
    )

    artifacts = predictor["artifacts"]
    require(
        isinstance(artifacts, list) and bool(artifacts),
        f"{label}.artifacts must be non-empty",
    )
    artifact_keys = [
        validate_predictor_artifact_binding(artifact, f"{label}.artifacts[{index}]")
        for index, artifact in enumerate(artifacts)
    ]
    require(artifact_keys == sorted(artifact_keys), f"{label}.artifacts must be sorted")
    require(
        len(set(artifact_keys)) == len(artifact_keys),
        f"{label}.artifacts contain duplicates",
    )
    validate_conformal_output_binding(
        predictor["output_binding"], f"{label}.output_binding"
    )
    referenced_nodes = {
        predictor["output_binding"]["node_id"],
        *(node_id for node_id, _artifact_id in artifact_keys),
        *(node_id for node_id, _namespace, _path in patch_keys),
        *data_binding_nodes,
    }
    require(
        referenced_nodes <= predictor_node_set,
        f"{label}.predictor_node_ids omit nodes referenced by the predictor closure",
    )
    return predictor


def predictor_binding_compatibility_oracle(
    expected_fingerprint: str,
    candidate: Any,
    label: str,
) -> dict[str, Any]:
    require_sha256(expected_fingerprint, f"{label}.expected_fingerprint")
    validated = validate_predictor_binding(candidate, f"{label}.candidate")
    actual_fingerprint = dagml_tcv1_sha256(validated)
    if actual_fingerprint != expected_fingerprint:
        return {
            "status": "refused",
            "code": "dagml.conformal.stale_predictor_binding",
            "expected_fingerprint": expected_fingerprint,
            "actual_fingerprint": actual_fingerprint,
        }
    return {
        "status": "compatible",
        "code": "dagml.conformal.predictor_binding_compatible",
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual_fingerprint,
    }


def validate_calibration_spec(value: Any, label: str) -> dict[str, Any]:
    spec = require_exact_keys(
        value,
        {
            "schema_version",
            "method",
            "coverages",
            "exchangeability_unit",
            "multi_target_policy",
            "small_sample_policy",
            "target_space",
            "numeric_version",
            "seed",
        },
        set(),
        label,
    )
    require_version_one(spec["schema_version"], "calibration spec")
    require(
        spec["method"] == "split_absolute_residual",
        f"{label}.method must be split_absolute_residual",
    )
    validate_conformal_coverages(spec["coverages"])
    require(
        spec["exchangeability_unit"] == "physical_sample",
        "exchangeability_unit must be physical_sample in V1",
    )
    require(
        spec["multi_target_policy"] in {"marginal", "joint_max"},
        f"{label}.multi_target_policy is invalid",
    )
    require(
        spec["small_sample_policy"] in {"error", "unbounded"},
        f"{label}.small_sample_policy is invalid",
    )
    require(spec["target_space"] == "raw", f"{label}.target_space must be raw")
    require(
        spec["numeric_version"] == "split_absolute_residual.v1",
        "numeric_version must be split_absolute_residual.v1",
    )
    seed = spec["seed"]
    if seed is not None:
        require_non_negative_int(seed, f"{label}.seed")
        require(
            seed <= CONFORMAL_PORTABLE_INT_MAX,
            f"{label}.seed exceeds the portable integer maximum",
        )
    return spec


def validate_tagged_quantile(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    status = value.get("status")
    if status == "finite":
        quantile = require_exact_keys(value, {"status", "value"}, set(), label)
        number = quantile["value"]
        finite = isinstance(number, float) and math.isfinite(number)
        require(
            bool(finite) and number >= 0.0,
            f"{label}.value must be a finite non-negative binary64 float token",
        )
        return quantile
    require(status == "unbounded", f"{label}.status must be finite or unbounded")
    return require_exact_keys(value, {"status"}, set(), label)


def validate_calibration_quantiles(
    quantiles: Any,
    spec: dict[str, Any],
    sample_count: int,
    target_count: int,
    label: str,
) -> None:
    coverages = validate_conformal_coverages(spec["coverages"])
    require(isinstance(quantiles, list), f"{label} must be an array")
    require(len(quantiles) == len(coverages), f"{label} must match coverages length")
    expected_width = target_count if spec["multi_target_policy"] == "marginal" else 1
    previous_values: list[float | None] = [None] * expected_width
    was_unbounded = [False] * expected_width
    for index, (record_value, coverage) in enumerate(zip(quantiles, coverages)):
        record_label = f"{label}[{index}]"
        record = require_exact_keys(
            record_value,
            {"coverage", "rank", "values"},
            set(),
            record_label,
        )
        record_coverage = validate_conformal_coverages(
            [record["coverage"]], f"{record_label}.coverage"
        )[0]
        require(
            Decimal(repr(record_coverage)) == Decimal(repr(coverage)),
            f"{record_label}.coverage is out of order",
        )
        expected_rank = conformal_finite_sample_rank(sample_count, coverage)
        require_positive_int(record["rank"], f"{record_label}.rank")
        require(
            record["rank"] == expected_rank,
            f"{record_label}.rank is not finite-sample rank",
        )
        values = record["values"]
        require(
            isinstance(values, list) and len(values) == expected_width,
            f"{record_label}.values has wrong target width",
        )
        if expected_rank > sample_count:
            require(
                spec["small_sample_policy"] == "unbounded",
                f"{record_label} rank exceeds sample count under error policy",
            )
        for target_index, tagged_value in enumerate(values):
            tagged = validate_tagged_quantile(
                tagged_value,
                f"{record_label}.values[{target_index}]",
            )
            if expected_rank > sample_count:
                require(
                    tagged["status"] == "unbounded",
                    f"{record_label}.values[{target_index}] must be tagged unbounded",
                )
            else:
                require(
                    tagged["status"] == "finite",
                    f"{record_label}.values[{target_index}] must be finite",
                )
            if tagged["status"] == "unbounded":
                was_unbounded[target_index] = True
                continue
            require(
                not was_unbounded[target_index],
                f"{record_label}.values[{target_index}] cannot become finite after unbounded",
            )
            current = float(tagged["value"])
            previous = previous_values[target_index]
            require(
                previous is None or current >= previous,
                f"{record_label}.values[{target_index}] violates nested coverage monotonicity",
            )
            previous_values[target_index] = current


def validate_calibration_influence_disjoint(
    cohort: dict[str, Any],
    influence: dict[str, Any],
    label: str,
) -> None:
    calibration_closure = set(cohort["physical_sample_ids"]) | set(
        cohort["origin_sample_ids"]
    )
    training_closure: set[str] = set()
    for entry in influence["entries"]:
        training_closure.update(entry["physical_sample_ids"])
        training_closure.update(entry["origin_sample_ids"])
    overlap = sorted(calibration_closure & training_closure)
    require(
        not overlap,
        f"{label}: calibration identity closure overlaps training influence: {overlap}",
    )


def conformal_artifact_checksum(value: dict[str, Any]) -> str:
    return dagml_tcv1_sha256(
        {key: member for key, member in value.items() if key != "checksum"}
    )


def refresh_conformal_derived_hashes(
    value: dict[str, Any],
    *,
    bind_influence_to_predictor: bool,
) -> None:
    cohort = value["calibration_cohort"]
    cohort["manifest_fingerprint"] = conformal_manifest_fingerprint(
        cohort, "manifest_fingerprint"
    )
    influence = value["training_influence"]
    influence["manifest_fingerprint"] = conformal_manifest_fingerprint(
        influence, "manifest_fingerprint"
    )
    if bind_influence_to_predictor:
        value["predictor_binding"]["training_influence_fingerprint"] = influence[
            "manifest_fingerprint"
        ]
    output_binding = value["predictor_binding"]["output_binding"]
    output_binding["binding_fingerprint"] = output_binding_fingerprint(output_binding)
    value["predictor_binding_fingerprint"] = dagml_tcv1_sha256(
        value["predictor_binding"]
    )
    value["calibration_spec_fingerprint"] = dagml_tcv1_sha256(value["calibration_spec"])
    value["checksum"] = conformal_artifact_checksum(value)


def validate_conformal_calibration_artifact(value: Any, label: str) -> dict[str, Any]:
    artifact = require_exact_keys(
        value,
        {
            "schema_version",
            "artifact_id",
            "predictor_binding",
            "predictor_binding_fingerprint",
            "calibration_spec",
            "calibration_spec_fingerprint",
            "calibration_cohort",
            "training_influence",
            "effective_sample_count",
            "quantiles",
            "diagnostics",
            "warnings",
            "checksum",
        },
        set(),
        label,
    )
    validate_strict_json_value(artifact, label)
    require_version_one(artifact["schema_version"], label)
    require_identifier(artifact["artifact_id"], f"{label}.artifact_id")
    predictor = validate_predictor_binding(
        artifact["predictor_binding"], f"{label}.predictor_binding"
    )
    require_sha256(
        artifact["predictor_binding_fingerprint"],
        f"{label}.predictor_binding_fingerprint",
    )
    require(
        artifact["predictor_binding_fingerprint"] == dagml_tcv1_sha256(predictor),
        f"{label}.predictor_binding_fingerprint does not match predictor_binding",
    )
    spec = validate_calibration_spec(
        artifact["calibration_spec"], f"{label}.calibration_spec"
    )
    require_sha256(
        artifact["calibration_spec_fingerprint"],
        f"{label}.calibration_spec_fingerprint",
    )
    require(
        artifact["calibration_spec_fingerprint"] == dagml_tcv1_sha256(spec),
        f"{label}.calibration_spec_fingerprint does not match calibration_spec",
    )
    cohort = validate_conformal_cohort_manifest(
        artifact["calibration_cohort"], f"{label}.calibration_cohort"
    )
    require(cohort["role"] == "calibration", f"{label} cohort role must be calibration")
    influence = validate_training_influence_manifest(
        artifact["training_influence"], f"{label}.training_influence"
    )
    require(
        influence["manifest_fingerprint"]
        == predictor["training_influence_fingerprint"],
        f"{label} training influence fingerprint does not match predictor binding",
    )
    require(
        all(
            binding["relation_fingerprint"] == influence["relation_fingerprint"]
            for binding in predictor["data_bindings"]
        ),
        f"{label} predictor data requirements are not all bound to the training influence relation fingerprint",
    )
    fitted_influence_kinds = {
        "transform_fit",
        "model_fit",
        "trained_meta_aggregation",
    }
    influenced_fit_nodes = {
        entry.get("node_id")
        for entry in influence["entries"]
        if entry["kind"] in fitted_influence_kinds and entry.get("node_id") is not None
    }
    for fitted_artifact in predictor["artifacts"]:
        require(
            fitted_artifact["node_id"] in influenced_fit_nodes,
            f"{label} fitted artifact node {fitted_artifact['node_id']} has no fitting influence",
        )
    if predictor["selected_patches"]:
        require(
            any(entry["kind"] == "hpo_selection" for entry in influence["entries"]),
            f"{label} selected patches require hpo_selection influence",
        )
    validate_calibration_influence_disjoint(cohort, influence, label)
    require_positive_int(
        artifact["effective_sample_count"], f"{label}.effective_sample_count"
    )
    require(
        artifact["effective_sample_count"] == len(cohort["physical_sample_ids"]),
        f"{label}.effective_sample_count must match physical_sample_ids",
    )
    output = predictor["output_binding"]
    require(
        cohort["target_names"] == output["target_names"],
        f"{label} cohort targets must match output binding order",
    )
    validate_calibration_quantiles(
        artifact["quantiles"],
        spec,
        artifact["effective_sample_count"],
        len(output["target_names"]),
        f"{label}.quantiles",
    )
    diagnostics = artifact["diagnostics"]
    validate_metadata_object(diagnostics, f"{label}.diagnostics")
    warnings = artifact["warnings"]
    validate_ordered_unique_strings(
        warnings,
        f"{label}.warnings",
        require_non_empty=False,
    )
    require_sha256(artifact["checksum"], f"{label}.checksum")
    require(
        artifact["checksum"] == conformal_artifact_checksum(artifact),
        f"{label}.checksum does not match TCV1 artifact content",
    )
    return artifact


def expect_contract_error(action: Any, expected: Any, label: str) -> None:
    require_non_empty_string(expected, f"{label}.expected_error")
    try:
        action()
    except ContractError as exc:
        require(
            expected in str(exc),
            f"{label} raised unexpected error `{exc}`; expected substring `{expected}`",
        )
    else:
        raise ContractError(f"{label} unexpectedly passed; expected `{expected}`")


def validate_conformal_calibration_fixture(value: Any, label: str) -> None:
    fixture = require_exact_keys(
        value,
        {
            "fixture_id",
            "schema_version",
            "calibration_artifact",
            "oracle_cases",
            "invalid_spec_cases",
            "invalid_influence_cases",
            "invalid_output_cases",
            "leakage_cases",
            "staleness_mutations",
            "coverage_rank_vectors",
            "typed_canonical_value_vectors",
        },
        set(),
        label,
    )
    require(
        fixture["fixture_id"] == CONFORMAL_CALIBRATION_FIXTURE_ID,
        f"{label}.fixture_id is invalid",
    )
    validate_strict_json_value(fixture, label)
    require_version_one(fixture["schema_version"], label)
    artifact = validate_conformal_calibration_artifact(
        fixture["calibration_artifact"], f"{label}.calibration_artifact"
    )

    oracle_cases = fixture["oracle_cases"]
    require(
        isinstance(oracle_cases, list) and bool(oracle_cases),
        f"{label}.oracle_cases empty",
    )
    oracle_ids: set[str] = set()
    for index, case_value in enumerate(oracle_cases):
        case_label = f"{label}.oracle_cases[{index}]"
        case = require_exact_keys(
            case_value,
            {"id", "residuals", "coverages", "small_sample_policy"},
            {"expected", "expected_error"},
            case_label,
        )
        require_identifier(case["id"], f"{case_label}.id")
        require(case["id"] not in oracle_ids, f"{case_label}.id is duplicated")
        oracle_ids.add(case["id"])
        require(
            ("expected" in case) != ("expected_error" in case),
            f"{case_label} must declare exactly one expected outcome",
        )
        if "expected_error" in case:
            expect_contract_error(
                lambda case=case: split_absolute_residual_oracle(
                    case["residuals"], case["coverages"], case["small_sample_policy"]
                ),
                case["expected_error"],
                case_label,
            )
        else:
            actual = split_absolute_residual_oracle(
                case["residuals"], case["coverages"], case["small_sample_policy"]
            )
            require(
                actual == case["expected"],
                f"{case_label} exact quantile oracle drifted",
            )

    for bad_residual in (math.nan, math.inf, -math.inf, -0.1):
        expect_contract_error(
            lambda bad_residual=bad_residual: split_absolute_residual_oracle(
                [0.1, bad_residual], [0.5], "error"
            ),
            "must be a finite non-negative number",
            f"{label}.synthetic_nonfinite_or_negative_residual",
        )
    expect_contract_error(
        lambda: split_absolute_residual_oracle(
            [0.1, CONFORMAL_PORTABLE_INT_MAX + 1], [0.5], "error"
        ),
        "must be a finite non-negative number",
        f"{label}.synthetic_nonportable_integer_residual",
    )
    require(
        conformal_finite_sample_rank(CONFORMAL_INT_MAX, 0.9999999999999999)
        == 18446744073709549772,
        f"{label} exact finite-sample rank boundary drifted",
    )
    rank_vectors = fixture["coverage_rank_vectors"]
    require(
        isinstance(rank_vectors, list) and bool(rank_vectors),
        f"{label}.coverage_rank_vectors empty",
    )
    rank_vector_ids: set[str] = set()
    for index, vector_value in enumerate(rank_vectors):
        vector_label = f"{label}.coverage_rank_vectors[{index}]"
        vector = require_exact_keys(
            vector_value,
            {
                "id",
                "sample_count",
                "coverage",
                "shortest_token",
                "expected_rank",
            },
            set(),
            vector_label,
        )
        require_identifier(vector["id"], f"{vector_label}.id")
        require(
            vector["id"] not in rank_vector_ids,
            f"{vector_label}.id is duplicated",
        )
        rank_vector_ids.add(vector["id"])
        require_positive_int(vector["sample_count"], f"{vector_label}.sample_count")
        coverage = validate_conformal_coverages(
            [vector["coverage"]], f"{vector_label}.coverage"
        )[0]
        require_non_empty_string(
            vector["shortest_token"], f"{vector_label}.shortest_token"
        )
        require(
            repr(coverage) == vector["shortest_token"]
            and float(vector["shortest_token"]) == coverage,
            f"{vector_label}.shortest_token is not the binary64 shortest-roundtrip token",
        )
        require_positive_int(vector["expected_rank"], f"{vector_label}.expected_rank")
        require(
            conformal_finite_sample_rank(vector["sample_count"], coverage)
            == vector["expected_rank"],
            f"{vector_label}.expected_rank drifted",
        )

    invalid_spec_cases = fixture["invalid_spec_cases"]
    require(
        isinstance(invalid_spec_cases, list) and bool(invalid_spec_cases),
        f"{label}.invalid_spec_cases empty",
    )
    invalid_ids: set[str] = set()
    for index, case_value in enumerate(invalid_spec_cases):
        case_label = f"{label}.invalid_spec_cases[{index}]"
        case = require_exact_keys(
            case_value,
            {"id", "path", "value", "expected_error"},
            set(),
            case_label,
        )
        require_identifier(case["id"], f"{case_label}.id")
        require(case["id"] not in invalid_ids, f"{case_label}.id is duplicated")
        invalid_ids.add(case["id"])
        mutated_spec = apply_json_pointer_mutation(
            artifact["calibration_spec"], case["path"], case["value"], case_label
        )
        expect_contract_error(
            lambda mutated_spec=mutated_spec: validate_calibration_spec(
                mutated_spec, f"{case_label}.mutated_spec"
            ),
            case["expected_error"],
            case_label,
        )

    invalid_influence_cases = fixture["invalid_influence_cases"]
    require(
        isinstance(invalid_influence_cases, list) and bool(invalid_influence_cases),
        f"{label}.invalid_influence_cases empty",
    )
    influence_case_ids: set[str] = set()
    for index, case_value in enumerate(invalid_influence_cases):
        case_label = f"{label}.invalid_influence_cases[{index}]"
        case = require_exact_keys(
            case_value,
            {"id", "operation", "expected_error"},
            {"kind", "path", "value"},
            case_label,
        )
        require_identifier(case["id"], f"{case_label}.id")
        require(case["id"] not in influence_case_ids, f"{case_label}.id is duplicated")
        influence_case_ids.add(case["id"])
        mutated_artifact = copy.deepcopy(artifact)
        operation = case["operation"]
        bind_influence = True
        if operation == "remove_kind":
            require(
                case.get("kind") in CONFORMAL_INFLUENCE_KINDS,
                f"{case_label}.kind is invalid",
            )
            mutated_artifact["training_influence"]["entries"] = [
                entry
                for entry in mutated_artifact["training_influence"]["entries"]
                if entry["kind"] != case["kind"]
            ]
        elif operation == "replace":
            mutated_artifact["training_influence"] = apply_json_pointer_mutation(
                mutated_artifact["training_influence"],
                case.get("path"),
                case.get("value"),
                case_label,
            )
        elif operation == "override_predictor_influence_fingerprint":
            require_sha256(case.get("value"), f"{case_label}.value")
            mutated_artifact["predictor_binding"]["training_influence_fingerprint"] = (
                case["value"]
            )
            bind_influence = False
        else:
            raise ContractError(f"{case_label}.operation is invalid")
        refresh_conformal_derived_hashes(
            mutated_artifact,
            bind_influence_to_predictor=bind_influence,
        )
        expect_contract_error(
            lambda mutated_artifact=mutated_artifact: (
                validate_conformal_calibration_artifact(
                    mutated_artifact, f"{case_label}.artifact"
                )
            ),
            case["expected_error"],
            case_label,
        )

    invalid_output_cases = fixture["invalid_output_cases"]
    require(
        isinstance(invalid_output_cases, list) and bool(invalid_output_cases),
        f"{label}.invalid_output_cases empty",
    )
    output_case_ids: set[str] = set()
    for index, case_value in enumerate(invalid_output_cases):
        case_label = f"{label}.invalid_output_cases[{index}]"
        case = require_exact_keys(
            case_value,
            {"id", "path", "value", "expected_error"},
            set(),
            case_label,
        )
        require_identifier(case["id"], f"{case_label}.id")
        require(case["id"] not in output_case_ids, f"{case_label}.id is duplicated")
        output_case_ids.add(case["id"])
        mutated_artifact = copy.deepcopy(artifact)
        mutated_artifact["predictor_binding"]["output_binding"] = (
            apply_json_pointer_mutation(
                mutated_artifact["predictor_binding"]["output_binding"],
                case["path"],
                case["value"],
                case_label,
            )
        )
        refresh_conformal_derived_hashes(
            mutated_artifact,
            bind_influence_to_predictor=True,
        )
        expect_contract_error(
            lambda mutated_artifact=mutated_artifact: (
                validate_conformal_calibration_artifact(
                    mutated_artifact, f"{case_label}.artifact"
                )
            ),
            case["expected_error"],
            case_label,
        )

    leakage_cases = fixture["leakage_cases"]
    require(
        isinstance(leakage_cases, list) and bool(leakage_cases),
        f"{label}.leakage_cases empty",
    )
    leakage_ids: set[str] = set()
    for index, case_value in enumerate(leakage_cases):
        case_label = f"{label}.leakage_cases[{index}]"
        case = require_exact_keys(
            case_value,
            {"id", "physical_sample_ids", "origin_sample_ids", "expected_error"},
            {"training_origin_sample_id"},
            case_label,
        )
        require_identifier(case["id"], f"{case_label}.id")
        require(case["id"] not in leakage_ids, f"{case_label}.id is duplicated")
        leakage_ids.add(case["id"])
        mutated_cohort = copy.deepcopy(artifact["calibration_cohort"])
        mutated_cohort["physical_sample_ids"] = case["physical_sample_ids"]
        mutated_cohort["origin_sample_ids"] = case["origin_sample_ids"]
        require(
            len(case["origin_sample_ids"]) <= len(case["physical_sample_ids"]),
            f"{case_label}.origin_sample_ids cannot outnumber physical samples",
        )
        mutated_cohort["group_ids"] = []
        mutated_cohort["source_ids"] = []
        mutated_cohort["unit_relations"] = [
            {
                "physical_sample_id": sample_id,
                "origin_sample_id": (
                    case["origin_sample_ids"][sample_index]
                    if sample_index < len(case["origin_sample_ids"])
                    else None
                ),
                "group_ids": [],
                "source_ids": [],
            }
            for sample_index, sample_id in enumerate(case["physical_sample_ids"])
        ]
        mutated_cohort["manifest_fingerprint"] = conformal_manifest_fingerprint(
            mutated_cohort, "manifest_fingerprint"
        )
        validated_cohort = validate_conformal_cohort_manifest(
            mutated_cohort, f"{case_label}.cohort"
        )
        influence = copy.deepcopy(artifact["training_influence"])
        training_origin_sample_id = case.get("training_origin_sample_id")
        if training_origin_sample_id is not None:
            require_identifier(
                training_origin_sample_id,
                f"{case_label}.training_origin_sample_id",
            )
            influence["entries"][0]["origin_sample_ids"] = sorted(
                {
                    *influence["entries"][0]["origin_sample_ids"],
                    training_origin_sample_id,
                }
            )
            influence["manifest_fingerprint"] = conformal_manifest_fingerprint(
                influence, "manifest_fingerprint"
            )
            influence = validate_training_influence_manifest(
                influence, f"{case_label}.training_influence"
            )
        expect_contract_error(
            lambda validated_cohort=validated_cohort: (
                validate_calibration_influence_disjoint(
                    validated_cohort, influence, case_label
                )
            ),
            case["expected_error"],
            case_label,
        )

    baseline_binding = artifact["predictor_binding"]
    baseline_fingerprint = dagml_tcv1_sha256(baseline_binding)
    require(
        baseline_fingerprint == artifact["predictor_binding_fingerprint"],
        f"{label} baseline predictor fingerprint drifted",
    )
    baseline_compatibility = predictor_binding_compatibility_oracle(
        baseline_fingerprint,
        baseline_binding,
        f"{label}.baseline_predictor_compatibility",
    )
    require(
        baseline_compatibility["status"] == "compatible",
        f"{label} baseline PredictorBinding was unexpectedly refused",
    )
    staleness_cases = fixture["staleness_mutations"]
    require(
        isinstance(staleness_cases, list) and bool(staleness_cases),
        f"{label}.staleness_mutations empty",
    )
    stale_ids: set[str] = set()
    for index, case_value in enumerate(staleness_cases):
        case_label = f"{label}.staleness_mutations[{index}]"
        case = require_exact_keys(
            case_value, {"id", "path", "value"}, set(), case_label
        )
        require_identifier(case["id"], f"{case_label}.id")
        require(case["id"] not in stale_ids, f"{case_label}.id is duplicated")
        stale_ids.add(case["id"])
        mutated = apply_json_pointer_mutation(
            baseline_binding, case["path"], case["value"], case_label
        )
        if (
            case["path"] == "/output_binding/prediction_source"
            and case["value"] != "final_refit"
        ):
            mutated["output_binding"]["refit_strategy"] = None
        if case["path"].startswith("/output_binding/"):
            mutated["output_binding"]["binding_fingerprint"] = (
                output_binding_fingerprint(mutated["output_binding"])
            )
        compatibility = predictor_binding_compatibility_oracle(
            baseline_fingerprint,
            mutated,
            case_label,
        )
        require(
            compatibility["status"] == "refused"
            and compatibility["code"] == "dagml.conformal.stale_predictor_binding",
            f"{case_label} did not produce a stale-predictor refusal",
        )

    vectors = fixture["typed_canonical_value_vectors"]
    require(
        isinstance(vectors, list) and bool(vectors),
        f"{label}.typed_canonical_value_vectors empty",
    )
    vector_ids: set[str] = set()
    for index, vector_value in enumerate(vectors):
        vector_label = f"{label}.typed_canonical_value_vectors[{index}]"
        vector = require_exact_keys(
            vector_value,
            {
                "id",
                "value",
                "equivalent_value",
                "expected_preimage_hex",
                "expected_sha256",
            },
            set(),
            vector_label,
        )
        require_identifier(vector["id"], f"{vector_label}.id")
        require(vector["id"] not in vector_ids, f"{vector_label}.id is duplicated")
        vector_ids.add(vector["id"])
        preimage = dagml_tcv1_preimage(vector["value"])
        equivalent_preimage = dagml_tcv1_preimage(vector["equivalent_value"])
        require(
            preimage == equivalent_preimage,
            f"{vector_label} equivalent value has a different TCV1 preimage",
        )
        require(
            preimage.hex() == vector["expected_preimage_hex"],
            f"{vector_label} TCV1 preimage drifted",
        )
        require_sha256(vector["expected_sha256"], f"{vector_label}.expected_sha256")
        require(
            hashlib.sha256(preimage).hexdigest() == vector["expected_sha256"],
            f"{vector_label} TCV1 digest drifted",
        )
    require(
        dagml_tcv1_preimage(1) != dagml_tcv1_preimage(1.0),
        f"{label} TCV1 must distinguish integer and float types",
    )
    expect_contract_error(
        lambda: dagml_tcv1_preimage("\ud800"),
        "Unicode surrogate",
        f"{label}.tcv1_surrogate_refusal",
    )
    expect_contract_error(
        lambda: dagml_tcv1_preimage({"é": 1, "é": 2}),
        "colliding NFC object keys",
        f"{label}.tcv1_normalization_collision",
    )

    for owner_path in (
        ("predictor_binding",),
        ("calibration_cohort",),
        ("training_influence",),
    ):
        mutated_artifact = copy.deepcopy(artifact)
        owner = mutated_artifact
        for segment in owner_path:
            owner = owner[segment]
        owner["schema_version"] = True
        refresh_conformal_derived_hashes(
            mutated_artifact,
            bind_influence_to_predictor=True,
        )
        expect_contract_error(
            lambda mutated_artifact=mutated_artifact: (
                validate_conformal_calibration_artifact(
                    mutated_artifact, f"{label}.boolean_nested_version"
                )
            ),
            "schema_version must be integer 1",
            f"{label}.boolean_{owner_path[0]}_schema_version",
        )

    boolean_count_artifact = copy.deepcopy(artifact)
    boolean_count_artifact["effective_sample_count"] = True
    expect_contract_error(
        lambda: validate_conformal_calibration_artifact(
            boolean_count_artifact, f"{label}.boolean_effective_sample_count"
        ),
        "must be a positive integer",
        f"{label}.boolean_effective_sample_count",
    )
    boolean_rank_artifact = copy.deepcopy(artifact)
    boolean_rank_artifact["quantiles"][0]["rank"] = True
    expect_contract_error(
        lambda: validate_conformal_calibration_artifact(
            boolean_rank_artifact, f"{label}.boolean_quantile_rank"
        ),
        "must be a positive integer",
        f"{label}.boolean_quantile_rank",
    )
    nonfinite_patch_artifact = copy.deepcopy(artifact)
    nonfinite_patch_artifact["predictor_binding"]["selected_patches"][0]["value"] = (
        math.nan
    )
    expect_contract_error(
        lambda: validate_conformal_calibration_artifact(
            nonfinite_patch_artifact, f"{label}.nonfinite_patch"
        ),
        "must not contain a non-finite float",
        f"{label}.nonfinite_patch",
    )
    nonfinite_diagnostics_artifact = copy.deepcopy(artifact)
    nonfinite_diagnostics_artifact["diagnostics"]["bad"] = math.inf
    expect_contract_error(
        lambda: validate_conformal_calibration_artifact(
            nonfinite_diagnostics_artifact, f"{label}.nonfinite_diagnostics"
        ),
        "must not contain a non-finite float",
        f"{label}.nonfinite_diagnostics",
    )
    integer_quantile_artifact = copy.deepcopy(artifact)
    integer_quantile_artifact["quantiles"][0]["values"][0]["value"] = 2
    expect_contract_error(
        lambda: validate_conformal_calibration_artifact(
            integer_quantile_artifact, f"{label}.integer_token_quantile"
        ),
        "must be a finite non-negative binary64 float token",
        f"{label}.integer_token_quantile",
    )

    cohort_without_axes = copy.deepcopy(artifact["calibration_cohort"])
    del cohort_without_axes["axes_fingerprint"]
    cohort_without_origins = copy.deepcopy(artifact["calibration_cohort"])
    del cohort_without_origins["origin_sample_ids"]
    influence_without_node = copy.deepcopy(artifact["training_influence"])
    del influence_without_node["entries"][0]["node_id"]
    influence_without_origins = copy.deepcopy(artifact["training_influence"])
    del influence_without_origins["entries"][0]["origin_sample_ids"]
    predictor_without_patches = copy.deepcopy(artifact["predictor_binding"])
    del predictor_without_patches["selected_patches"]
    binding_without_plugin = copy.deepcopy(
        artifact["predictor_binding"]["artifacts"][0]
    )
    del binding_without_plugin["plugin"]
    binding_without_plugin_version = copy.deepcopy(
        artifact["predictor_binding"]["artifacts"][0]
    )
    del binding_without_plugin_version["plugin_version"]
    spec_without_seed = copy.deepcopy(artifact["calibration_spec"])
    del spec_without_seed["seed"]
    for omission_label, action in (
        (
            "artifact_diagnostics",
            lambda: validate_conformal_calibration_artifact(
                {
                    key: member
                    for key, member in artifact.items()
                    if key != "diagnostics"
                },
                f"{label}.missing_diagnostics",
            ),
        ),
        (
            "artifact_warnings",
            lambda: validate_conformal_calibration_artifact(
                {key: member for key, member in artifact.items() if key != "warnings"},
                f"{label}.missing_warnings",
            ),
        ),
        (
            "cohort_axes_fingerprint",
            lambda: validate_conformal_cohort_manifest(
                cohort_without_axes, f"{label}.missing_axes_fingerprint"
            ),
        ),
        (
            "cohort_origin_sample_ids",
            lambda: validate_conformal_cohort_manifest(
                cohort_without_origins, f"{label}.missing_cohort_origin_sample_ids"
            ),
        ),
        (
            "influence_node_id",
            lambda: validate_training_influence_manifest(
                influence_without_node, f"{label}.missing_influence_node_id"
            ),
        ),
        (
            "influence_origin_sample_ids",
            lambda: validate_training_influence_manifest(
                influence_without_origins,
                f"{label}.missing_influence_origin_sample_ids",
            ),
        ),
        (
            "predictor_selected_patches",
            lambda: validate_predictor_binding(
                predictor_without_patches, f"{label}.missing_selected_patches"
            ),
        ),
        (
            "artifact_plugin",
            lambda: validate_predictor_artifact_binding(
                binding_without_plugin, f"{label}.missing_artifact_plugin"
            ),
        ),
        (
            "artifact_plugin_version",
            lambda: validate_predictor_artifact_binding(
                binding_without_plugin_version,
                f"{label}.missing_artifact_plugin_version",
            ),
        ),
        (
            "calibration_seed",
            lambda: validate_calibration_spec(
                spec_without_seed, f"{label}.missing_calibration_seed"
            ),
        ),
    ):
        expect_contract_error(
            action,
            "is missing required field(s)",
            f"{label}.canonical_omission_{omission_label}",
        )

    for unsupported_version in (0, 2, True):
        mutated_artifact = copy.deepcopy(artifact)
        mutated_artifact["schema_version"] = unsupported_version
        expect_contract_error(
            lambda mutated_artifact=mutated_artifact: (
                validate_conformal_calibration_artifact(
                    mutated_artifact, f"{label}.unsupported_artifact"
                )
            ),
            "schema_version must be integer 1",
            f"{label}.artifact_schema_version_{unsupported_version}",
        )


def validate_operator_variant_label_fixture(value: Any, label: str) -> None:
    """Validate the cross-language operator-variant content-fingerprint contract.

    Re-derives the sha256 of the fixture's ``canonical_value`` with the SAME primitive dag-ml uses
    (``sha256`` of the compact, sorted-key JSON bytes) and asserts it equals the pinned
    ``variant_label``. This is the byte-identity check the nirs4all host runs to prove it can
    recompute the SAME fingerprint dag-ml stamps on per-variant reports.
    """
    require(
        isinstance(value, dict),
        f"{label} operator-variant-label fixture must be an object",
    )
    require(
        value.get("schema_version") == 1,
        f"{label} operator-variant-label fixture schema_version must be 1",
    )
    require(
        value.get("fixture_id") == OPERATOR_VARIANT_LABEL_FIXTURE_ID,
        f"{label} operator-variant-label fixture_id mismatch",
    )
    canonical_form = value.get("canonical_form")
    require(
        isinstance(canonical_form, dict),
        f"{label} operator-variant-label fixture must declare canonical_form",
    )
    for field in (
        "definition",
        "kind",
        "class",
        "params",
        "key_ordering",
        "numeric_policy",
        "digest",
    ):
        require_non_empty_string(
            canonical_form.get(field),
            f"{label} operator-variant-label canonical_form.{field}",
        )
    case = value.get("case")
    require(
        isinstance(case, dict),
        f"{label} operator-variant-label fixture must declare a case",
    )
    require_non_empty_string(
        case.get("case_id"), f"{label} operator-variant-label case.case_id"
    )
    canonical_value = case.get("canonical_value")
    require(
        isinstance(canonical_value, list) and bool(canonical_value),
        f"{label} operator-variant-label case.canonical_value must be a non-empty array",
    )
    for index, step in enumerate(canonical_value):
        step_label = f"{label} operator-variant-label canonical_value[{index}]"
        require(isinstance(step, dict), f"{step_label} must be an object")
        require(
            set(step.keys()) == {"kind", "class", "params"},
            f"{step_label} keys must be exactly kind/class/params",
        )
        require_non_empty_string(step.get("kind"), f"{step_label}.kind")
        require(
            isinstance(step.get("class"), str), f"{step_label}.class must be a string"
        )
        require(
            isinstance(step.get("params"), dict),
            f"{step_label}.params must be an object",
        )
    require_sha256(
        case.get("variant_label"), f"{label} operator-variant-label case.variant_label"
    )
    # Byte-identity: the host recomputes the SAME digest from canonical_value.
    recomputed = canonical_json_sha256(canonical_value)
    require(
        recomputed == case["variant_label"],
        f"{label} operator-variant-label digest drifted: recomputed {recomputed} != pinned {case['variant_label']}",
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
        "representation_compatibility",
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
        "representation_compatibility_report",
        "representation_sample_observation_mapping",
        "representation_combo_selection_record",
        "representation_compatibility_severity",
        "representation_compatibility_outcome",
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
    require(
        set(defs.get("representation_compatibility_severity", {}).get("enum", []))
        == REPRESENTATION_COMPATIBILITY_SEVERITIES,
        f"{label} data-output provenance compatibility severities mismatch",
    )
    require(
        set(defs.get("representation_compatibility_outcome", {}).get("enum", []))
        == REPRESENTATION_COMPATIBILITY_OUTCOMES,
        f"{label} data-output provenance compatibility outcomes mismatch",
    )
    shape_delta = defs.get("shape_delta")
    require(
        isinstance(shape_delta, dict),
        f"{label} data-output shape_delta definition missing",
    )
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
    require(
        schema.get("$id") == NODE_TASK_SCHEMA_ID,
        f"{label} NodeTask schema $id mismatch",
    )
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
        require(
            definition_name in defs,
            f"{label} NodeTask schema misses `{definition_name}`",
        )
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{label} NodeTask properties missing")
    require(
        "fit_influence" in properties, f"{label} NodeTask schema misses fit_influence"
    )
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
        set(defs.get("controller_capability", {}).get("enum", []))
        == CONTROLLER_CAPABILITIES,
        f"{label} NodeTask capability enum mismatch",
    )
    require(
        set(defs.get("fit_influence_policy", {}).get("enum", []))
        == FIT_INFLUENCE_POLICIES,
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
    require(
        schema.get("$id") == NODE_RESULT_SCHEMA_ID,
        f"{label} NodeResult schema $id mismatch",
    )
    require(
        schema.get("type") == "object", f"{label} NodeResult root must be an object"
    )
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
        require(
            definition_name in defs,
            f"{label} NodeResult schema misses `{definition_name}`",
        )
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
        set(defs.get("fit_influence_policy", {}).get("enum", []))
        == FIT_INFLUENCE_POLICIES,
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
        isinstance(defs, dict) and "identifier" in defs and "capability" in defs,
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
    require(
        isinstance(required, list),
        f"{label} process-adapter frame required list is missing",
    )
    for field in ("type", "schema_version"):
        require(
            field in required, f"{label} process-adapter frame must require `{field}`"
        )

    one_of = schema.get("oneOf")
    require(
        isinstance(one_of, list) and len(one_of) == 6,
        f"{label} process-adapter frame must declare six concrete frame variants",
    )
    defs = schema.get("$defs")
    require(
        isinstance(defs, dict), f"{label} process-adapter frame definitions are missing"
    )
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
            isinstance(variant_required, list)
            and set(variant_required) == required_fields,
            f"{label} {definition_name} required fields mismatch",
        )
        properties = definition.get("properties")
        require(
            isinstance(properties, dict),
            f"{label} {definition_name} properties are missing",
        )
        require(
            properties.get("type", {}).get("const") == frame_type,
            f"{label} {definition_name} type const mismatch",
        )
        require(
            properties.get("schema_version", {}).get("$ref")
            == "#/$defs/schema_version",
            f"{label} {definition_name} schema_version reference mismatch",
        )
    require(
        defs["ack_frame"]["properties"].get("status", {}).get("enum")
        == ["initialized", "closed"],
        f"{label} process-adapter ack status enum mismatch",
    )


def validate_envelope(envelope: Any, label: str) -> None:
    require(isinstance(envelope, dict), f"{label} envelope must be a JSON object")
    require(
        envelope.get("schema_version") == 1,
        f"{label} envelope schema_version must be 1",
    )
    require_sha256(envelope.get("schema_fingerprint"), f"{label} schema_fingerprint")
    require_sha256(envelope.get("plan_fingerprint"), f"{label} plan_fingerprint")
    relation_fingerprint = envelope.get("relation_fingerprint")
    if relation_fingerprint is not None:
        require_sha256(relation_fingerprint, f"{label} relation_fingerprint")
    for field in ("data_content_fingerprint", "target_content_fingerprint"):
        fingerprint = envelope.get(field)
        if fingerprint is not None:
            require_sha256(fingerprint, f"{label} {field}")

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
    require(
        isinstance(relations, dict), f"{label} coordinator_relations must be an object"
    )
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
        require(
            isinstance(components, list),
            f"{record_label}.component_observation_ids must be an array",
        )
        require(
            len(set(components)) == len(components),
            f"{record_label}.component_observation_ids duplicate entries",
        )
        for component_index, component_id in enumerate(components):
            require_non_empty_string(
                component_id,
                f"{record_label}.component_observation_ids[{component_index}]",
            )
        if unit_level != "combo":
            require(
                not components,
                f"{record_label}.component_observation_ids require unit_level combo",
            )
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
    require(
        selector.get("schema_version") == 1,
        f"{label} selector schema_version must be 1",
    )
    require_non_empty_string(selector.get("feature_set_id"), f"{label}.feature_set_id")
    sources = selector.get("sources")
    require(
        isinstance(sources, list) and sources,
        f"{label}.sources must be a non-empty array",
    )
    source_ids: list[str] = []
    for index, source in enumerate(sources):
        source_label = f"{label}.sources[{index}]"
        require(isinstance(source, dict), f"{source_label} must be an object")
        require_non_empty_string(source.get("source_id"), f"{source_label}.source_id")
        require_non_empty_string(
            source.get("feature_set_id"), f"{source_label}.feature_set_id"
        )
        source_ids.append(source["source_id"])
        columns = source.get("columns")
        if columns is not None:
            require(
                isinstance(columns, list) and columns,
                f"{source_label}.columns must be a non-empty array when present",
            )
            for column_index, column in enumerate(columns):
                require_non_empty_string(
                    column, f"{source_label}.columns[{column_index}]"
                )
    require(
        len(set(source_ids)) == len(source_ids),
        f"{label}.sources contain duplicate source ids",
    )

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
    require(
        isinstance(masks, list) and masks, f"{label}.alignment.masks must be non-empty"
    )
    mask_source_ids: list[str] = []
    for index, mask in enumerate(masks):
        mask_label = f"{label}.alignment.masks[{index}]"
        require(isinstance(mask, dict), f"{mask_label} must be an object")
        require_non_empty_string(mask.get("source_id"), f"{mask_label}.source_id")
        mask_source_ids.append(mask["source_id"])
        require(
            mask.get("sample_ids") == sample_ids,
            f"{mask_label}.sample_ids order mismatch",
        )
        present = mask.get("present")
        require(
            isinstance(present, list) and len(present) == len(sample_ids),
            f"{mask_label}.present length must match sample_ids",
        )
        for present_index, value in enumerate(present):
            require(
                isinstance(value, bool),
                f"{mask_label}.present[{present_index}] must be bool",
            )
    require(
        set(mask_source_ids) == set(source_ids),
        f"{label}.alignment masks must match sources",
    )

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
        validate_representation_plan(
            representation_plan, f"{label}.representation_plan"
        )

    source_layout = selector.get("source_layout")
    if source_layout is not None:
        validate_feature_fusion_source_layout(
            source_layout,
            selector["feature_set_id"],
            sources,
            policy,
            label,
        )


def validate_feature_fusion_source_layout(
    source_layout: Any,
    feature_set_id: str,
    sources: list[Any],
    policy: Any,
    label: str,
) -> None:
    require(isinstance(source_layout, dict), f"{label}.source_layout must be an object")
    require(
        source_layout.get("kind") == "by_source_concat",
        f"{label}.source_layout.kind must be by_source_concat",
    )
    source_order = source_layout.get("source_order")
    require(
        isinstance(source_order, list) and source_order,
        f"{label}.source_layout.source_order must be a non-empty array",
    )
    for index, source_id in enumerate(source_order):
        require_non_empty_string(
            source_id, f"{label}.source_layout.source_order[{index}]"
        )
    require(
        len(set(source_order)) == len(source_order),
        f"{label}.source_layout.source_order contains duplicates",
    )
    selector_order = [source["source_id"] for source in sources]
    require(
        source_order == selector_order,
        f"{label}.source_layout.source_order must match sources order",
    )

    blocks = source_layout.get("blocks")
    require(
        isinstance(blocks, list) and len(blocks) == len(source_order),
        f"{label}.source_layout.blocks must match source_order length",
    )
    expected_column_start = 0
    for index, block in enumerate(blocks):
        block_label = f"{label}.source_layout.blocks[{index}]"
        require(isinstance(block, dict), f"{block_label} must be an object")
        require(
            block.get("source_id") == source_order[index],
            f"{block_label}.source_id must match source_order",
        )
        preprocessing_output = block.get("preprocessing_output")
        require(
            isinstance(preprocessing_output, dict),
            f"{block_label}.preprocessing_output must be an object",
        )
        require_non_empty_string(
            preprocessing_output.get("feature_set_id"),
            f"{block_label}.preprocessing_output.feature_set_id",
        )
        require_non_empty_string(
            preprocessing_output.get("representation_id"),
            f"{block_label}.preprocessing_output.representation_id",
        )
        source = sources[index]
        require(
            preprocessing_output["feature_set_id"] == source["feature_set_id"],
            f"{block_label}.preprocessing_output.feature_set_id must match source feature_set_id",
        )
        fit_scope = preprocessing_output.get("fit_scope")
        if fit_scope is not None:
            require(
                fit_scope
                in {"stateless", "fold_train", "full_train", "inference_only"},
                f"{block_label}.preprocessing_output.fit_scope is invalid",
            )
        adapter_id = preprocessing_output.get("adapter_id")
        if adapter_id is not None:
            require_non_empty_string(
                adapter_id, f"{block_label}.preprocessing_output.adapter_id"
            )
        column_start = block.get("column_start")
        require(
            isinstance(column_start, int) and column_start == expected_column_start,
            f"{block_label}.column_start must be contiguous",
        )
        column_count = block.get("column_count")
        require(
            isinstance(column_count, int) and column_count > 0,
            f"{block_label}.column_count must be a positive integer",
        )
        feature_names = block.get("feature_names")
        if feature_names is not None:
            require(
                isinstance(feature_names, list) and len(feature_names) == column_count,
                f"{block_label}.feature_names length must match column_count",
            )
            require(
                len(set(feature_names)) == len(feature_names),
                f"{block_label}.feature_names contains duplicates",
            )
            for feature_index, feature_name in enumerate(feature_names):
                require_non_empty_string(
                    feature_name,
                    f"{block_label}.feature_names[{feature_index}]",
                )
            columns = source.get("columns")
            if columns is not None:
                require(
                    feature_names == columns,
                    f"{block_label}.feature_names must match source columns",
                )
        expected_column_start += column_count

    concat = source_layout.get("concat")
    require(isinstance(concat, dict), f"{label}.source_layout.concat must be an object")
    require(
        concat.get("feature_set_id") == feature_set_id,
        f"{label}.source_layout.concat.feature_set_id must match selector feature_set_id",
    )
    require_non_empty_string(
        concat.get("representation_id"),
        f"{label}.source_layout.concat.representation_id",
    )
    require(
        concat.get("axis") == "feature",
        f"{label}.source_layout.concat.axis must be feature",
    )
    require(
        concat.get("total_column_count") == expected_column_start,
        f"{label}.source_layout.concat.total_column_count must match block spans",
    )
    require(
        concat.get("preserve_source_order") is True,
        f"{label}.source_layout.concat.preserve_source_order must be true",
    )
    namespace_columns = concat.get("namespace_columns")
    require(
        isinstance(namespace_columns, bool),
        f"{label}.source_layout.concat.namespace_columns must be bool",
    )
    if isinstance(policy, dict) and "namespace_columns" in policy:
        require(
            namespace_columns == policy["namespace_columns"],
            f"{label}.source_layout.concat.namespace_columns must match policy",
        )


def validate_fold_set_fixture(fold_set: Any, label: str) -> None:
    require(isinstance(fold_set, dict), f"{label} fold set must be an object")
    require_non_empty_string(fold_set.get("id"), f"{label}.id")
    sample_ids = fold_set.get("sample_ids")
    require(
        isinstance(sample_ids, list) and sample_ids,
        f"{label}.sample_ids must be non-empty",
    )
    for index, sample_id in enumerate(sample_ids):
        require_identifier(sample_id, f"{label}.sample_ids[{index}]")
    require(
        len(set(sample_ids)) == len(sample_ids),
        f"{label}.sample_ids contain duplicates",
    )
    sample_set = set(sample_ids)

    sample_groups = fold_set.get("sample_groups", {})
    require(isinstance(sample_groups, dict), f"{label}.sample_groups must be an object")
    for sample_id, group_id in sample_groups.items():
        require(
            sample_id in sample_set, f"{label}.sample_groups references unknown sample"
        )
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
        require(
            isinstance(train, list), f"{fold_label}.train_sample_ids must be an array"
        )
        require(
            isinstance(validation, list) and validation,
            f"{fold_label}.validation_sample_ids must be non-empty",
        )
        for sample_id in train + validation:
            require_identifier(sample_id, f"{fold_label} sample id")
            require(
                sample_id in sample_set,
                f"{fold_label} references unknown sample `{sample_id}`",
            )
        require(
            len(set(train)) == len(train),
            f"{fold_label}.train_sample_ids contain duplicates",
        )
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
    require(
        isinstance(interface, dict), f"{label}.interface must be an object when present"
    )
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
        require(
            isinstance(ports, dict),
            f"{node_label}.ports must be an object when present",
        )
        node_ports[node_id] = {
            "inputs": graph_port_specs(
                ports.get("inputs", []), f"{node_label}.ports.inputs"
            ),
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
        require(
            source_node in node_ports,
            f"{edge_label} references missing source `{source_node}`",
        )
        require(
            target_node in node_ports,
            f"{edge_label} references missing target `{target_node}`",
        )
        source_port = source.get("port_name")
        target_port = target.get("port_name")
        require_non_empty_string(source_port, f"{edge_label}.source.port_name")
        require_non_empty_string(target_port, f"{edge_label}.target.port_name")
        source_spec = node_ports[source_node]["outputs"].get(source_port)
        target_spec = node_ports[target_node]["inputs"].get(target_port)
        require(
            source_spec is not None,
            f"{edge_label} source port `{source_port}` is missing",
        )
        require(
            target_spec is not None,
            f"{edge_label} target port `{target_port}` is missing",
        )
        source_kind = source_spec["kind"]
        target_kind = target_spec["kind"]
        edge_kind = contract.get("kind")
        require(
            edge_kind == source_kind == target_kind,
            f"{edge_label} kind `{edge_kind}` does not match endpoint ports",
        )
        validate_graph_edge_contract(edge_label, contract, source_spec, target_spec)
        if contract.get("requires_oof") is True:
            require(
                edge_kind == "prediction",
                f"{edge_label} requires OOF on non-prediction edge",
            )


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
        require_optional_identifier(
            port.get("alignment_key"), f"{port_label}.alignment_key"
        )
        require_optional_unit_level(
            port.get("target_level"), f"{port_label}.target_level"
        )
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
    require_optional_unit_level(
        contract.get("unit_level"), f"{label}.contract.unit_level"
    )
    require_optional_identifier(
        contract.get("alignment_key"), f"{label}.contract.alignment_key"
    )
    require_optional_unit_level(
        contract.get("target_level"), f"{label}.contract.target_level"
    )
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
        source_target is None
        or target_target is None
        or source_target == target_target,
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
    if (
        contract.get("relation_contract") is not None
        or contract.get("allows_broadcast") is True
    ):
        return True
    if contract.get("alignment_key") is not None:
        return True
    if (
        contract.get("unit_level") is not None
        and contract.get("unit_level") != "physical_sample"
    ):
        return True
    if (
        contract.get("target_level") is not None
        and contract.get("target_level") != "physical_sample"
    ):
        return True
    for port in (source_port, target_port):
        if port.get("alignment_key") is not None:
            return True
        if (
            port.get("unit_level") is not None
            and port.get("unit_level") != "physical_sample"
        ):
            return True
        if (
            port.get("target_level") is not None
            and port.get("target_level") != "physical_sample"
        ):
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
    require(
        isinstance(pipeline, list) and pipeline, f"{label}.pipeline must be non-empty"
    )
    keys_seen: set[str] = set()
    for index, step in enumerate(pipeline):
        step_label = f"{label}.pipeline[{index}]"
        if isinstance(step, dict):
            keys_seen.update(step)
        elif isinstance(step, str) or step is None or isinstance(step, list):
            continue
        else:
            raise ContractError(f"{step_label} has unsupported JSON type")
    for required_key in (
        "_comment",
        "class",
        "_cartesian_",
        "split",
        "_chain_",
        "merge",
        "model",
    ):
        require(
            required_key in keys_seen, f"{label} fixture must exercise `{required_key}`"
        )
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
        require(
            expected in generator_keys, f"{label} fixture must exercise `{expected}`"
        )
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
    if split_unit == "observation" and not value.get(
        "allow_observation_split_with_shared_target", False
    ):
        raise ContractError(
            f"{label} observation split requires explicit shared-target allowance"
        )
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
        method
        in {"none", "mean", "weighted_mean", "median", "vote", "custom_controller"},
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
    if (
        value.get("store_raw_predictions", True) is False
        and value.get("store_aggregated_predictions", True) is False
    ):
        raise ContractError(f"{label} must store raw and/or aggregated predictions")


def validate_fold_set(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} fold_set must be an object")
    require_non_empty_string(value.get("id"), f"{label}.id")
    sample_ids = value.get("sample_ids")
    require(
        isinstance(sample_ids, list) and sample_ids,
        f"{label}.sample_ids must be non-empty",
    )
    require(
        len(set(sample_ids)) == len(sample_ids),
        f"{label}.sample_ids contain duplicates",
    )
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
        require(
            isinstance(train, list), f"{fold_label}.train_sample_ids must be an array"
        )
        require(
            isinstance(validation, list) and validation,
            f"{fold_label}.validation_sample_ids must be non-empty",
        )
        require(
            len(set(train)) == len(train),
            f"{fold_label}.train_sample_ids duplicate samples",
        )
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
            require(
                sample_id in sample_set,
                f"{fold_label} references unknown sample `{sample_id}`",
            )
        for sample_id in validation:
            validation_counts[sample_id] += 1
        metadata = fold.get("metadata", {})
        require(isinstance(metadata, dict), f"{fold_label}.metadata must be an object")
    for sample_id, count in validation_counts.items():
        require(
            count == 1,
            f"{label} sample `{sample_id}` validation count is {count}, expected 1",
        )
    sample_groups = value.get("sample_groups", {})
    require(isinstance(sample_groups, dict), f"{label}.sample_groups must be an object")
    for sample_id, group_id in sample_groups.items():
        require(
            sample_id in sample_set,
            f"{label}.sample_groups references unknown sample `{sample_id}`",
        )
        require_identifier(group_id, f"{label}.sample_groups[{sample_id}]")


def validate_generation_spec(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} generation spec must be an object")
    strategy = value.get("strategy", "none")
    require(strategy in {"none", "cartesian", "zip"}, f"{label}.strategy is invalid")
    dimensions = value.get("dimensions", [])
    require(isinstance(dimensions, list), f"{label}.dimensions must be an array")
    max_variants = value.get("max_variants", 1)
    if max_variants is not None:
        require(
            isinstance(max_variants, int) and max_variants >= 1,
            f"{label}.max_variants invalid",
        )
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
        require(
            isinstance(choices, list) and choices,
            f"{dimension_label}.choices must be non-empty",
        )
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
            require(
                isinstance(overrides, list),
                f"{choice_label}.param_overrides must be an array",
            )
            for override_index, override in enumerate(overrides):
                override_label = f"{choice_label}.param_overrides[{override_index}]"
                require(
                    isinstance(override, dict), f"{override_label} must be an object"
                )
                require_identifier(override.get("node_id"), f"{override_label}.node_id")
                params = override.get("params")
                require(
                    isinstance(params, dict) and params,
                    f"{override_label}.params non-empty",
                )
            if "active_subsequence" in choice:
                active_subsequence = choice.get("active_subsequence")
                require_non_empty_string(
                    active_subsequence, f"{choice_label}.active_subsequence"
                )
                # Mirror Rust GenerationChoice::validate (trim().is_empty()): reject
                # whitespace-only, matching the schema `\\S` pattern.
                require(
                    active_subsequence.strip() != "",
                    f"{choice_label}.active_subsequence must not be whitespace-only",
                )
                require(
                    not overrides,
                    f"{choice_label} cannot set both param_overrides and active_subsequence",
                )
        require(
            len(set(labels)) == len(labels),
            f"{dimension_label}.choices duplicate labels",
        )
    require(len(set(names)) == len(names), f"{label}.dimensions duplicate names")
    if strategy == "zip":
        require(
            len(set(choice_counts)) == 1,
            f"{label} zip dimensions must have equal lengths",
        )


def validate_data_model_shape_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} shape plan must be an object")
    require_identifier(value.get("node_id"), f"{label}.node_id")
    for field in ("input_granularity", "target_granularity"):
        field_value = value.get(field, "sample")
        require(
            field_value in {"observation", "sample", "target", "group"},
            f"{label}.{field} invalid",
        )
    for field in ("fit_rows", "predict_rows"):
        field_value = value.get(
            field, "fold_train" if field == "fit_rows" else "fold_validation"
        )
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
    validate_aggregation_policy(
        value.get("aggregation_policy", {}), f"{label}.aggregation_policy"
    )
    augmentation = value.get("augmentation_policy", {})
    require(
        isinstance(augmentation, dict), f"{label}.augmentation_policy must be an object"
    )
    for field in ("sample_scope", "feature_scope"):
        field_value = augmentation.get(field, "train_only")
        require(
            field_value in {"none", "train_only", "all_partitions"},
            f"{label}.augmentation_policy.{field} invalid",
        )
    for field in ("require_origin_id", "inherit_group", "inherit_target"):
        if field in augmentation:
            require(
                isinstance(augmentation[field], bool),
                f"{label}.augmentation_policy.{field} boolean",
            )
    selection = value.get("selection_policy", {})
    require(isinstance(selection, dict), f"{label}.selection_policy must be an object")
    scope = selection.get("scope", "none")
    require(
        scope in {"none", "unsupervised", "supervised_fold_train"},
        f"{label}.selection_policy.scope invalid",
    )
    for field in ("store_masks", "allow_schema_mismatch_on_join"):
        if field in selection:
            require(
                isinstance(selection[field], bool),
                f"{label}.selection_policy.{field} boolean",
            )
    if (
        scope == "supervised_fold_train"
        and value.get("fit_rows", "fold_train") != "fold_train"
    ):
        raise ContractError(
            f"{label} supervised feature selection must fit on fold_train"
        )


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
    require_non_empty_string(
        value.get("output_representation"), f"{label}.output_representation"
    )
    feature_set_id = value.get("feature_set_id")
    if feature_set_id is not None:
        require_non_empty_string(feature_set_id, f"{label}.feature_set_id")
    source_ids = value.get("source_ids", [])
    require(isinstance(source_ids, list), f"{label}.source_ids must be an array")
    require(
        len(set(source_ids)) == len(source_ids),
        f"{label}.source_ids contain duplicates",
    )
    for index, source_id in enumerate(source_ids):
        require_non_empty_string(source_id, f"{label}.source_ids[{index}]")
    if "require_relations" in value:
        require(
            isinstance(value["require_relations"], bool),
            f"{label}.require_relations boolean",
        )
    if value.get("require_relations", False):
        require(
            relation_fingerprint is not None, f"{label} requires relation_fingerprint"
        )
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
            require(
                isinstance(view_policy[field], bool),
                f"{label}.view_policy.{field} boolean",
            )


def validate_data_view_selector(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} selector must be an object")
    source_ids = value.get("source_ids", [])
    require(isinstance(source_ids, list), f"{label}.source_ids must be an array")
    for index, source_id in enumerate(source_ids):
        require_non_empty_string(source_id, f"{label}.source_ids[{index}]")
    require(
        len(set(source_ids)) == len(source_ids),
        f"{label}.source_ids contain duplicates",
    )
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
        require(
            bool(selector.get("source_ids")), f"{label}.selector.source_ids required"
        )
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
        require(
            isinstance(root_seed, int) and root_seed >= 0, f"{label}.root_seed invalid"
        )
    validate_leakage_policy(value.get("leakage_policy", {}), f"{label}.leakage_policy")
    validate_aggregation_policy(
        value.get("aggregation_policy", {}), f"{label}.aggregation_policy"
    )
    split_invocation = value.get("split_invocation")
    if split_invocation is not None:
        require(
            isinstance(split_invocation, dict),
            f"{label}.split_invocation must be object",
        )
        require_non_empty_string(
            split_invocation.get("id"), f"{label}.split_invocation.id"
        )
        controller_id = split_invocation.get("controller_id")
        if controller_id is not None:
            require_identifier(controller_id, f"{label}.split_invocation.controller_id")
        validate_leakage_policy(
            split_invocation.get("leakage_policy", {}),
            f"{label}.split_invocation.leakage_policy",
        )
        params = split_invocation.get("params", {})
        require(
            isinstance(params, dict), f"{label}.split_invocation.params must be object"
        )
        fold_set = split_invocation.get("fold_set")
        if fold_set is not None:
            validate_fold_set(fold_set, f"{label}.split_invocation.fold_set")
    validate_generation_spec(value.get("generation", {}), f"{label}.generation")
    shape_plans = value.get("shape_plans", {})
    require(isinstance(shape_plans, dict), f"{label}.shape_plans must be an object")
    for key, shape_plan in shape_plans.items():
        validate_data_model_shape_plan(shape_plan, f"{label}.shape_plans[{key}]")
        require(
            shape_plan.get("node_id") == key,
            f"{label}.shape_plans key `{key}` mismatch",
        )
    data_bindings = value.get("data_bindings", {})
    require(isinstance(data_bindings, dict), f"{label}.data_bindings must be an object")
    for key, bindings in data_bindings.items():
        require(
            isinstance(bindings, list), f"{label}.data_bindings[{key}] must be an array"
        )
        for index, binding in enumerate(bindings):
            validate_data_binding(binding, f"{label}.data_bindings[{key}][{index}]")
            require(
                binding.get("node_id") == key,
                f"{label}.data_bindings key `{key}` mismatch",
            )
    branch_view_plans = value.get("branch_view_plans", [])
    require(
        isinstance(branch_view_plans, list),
        f"{label}.branch_view_plans must be an array",
    )
    seen_branch_views: set[str] = set()
    for index, view_plan in enumerate(branch_view_plans):
        validate_branch_view_plan(view_plan, f"{label}.branch_view_plans[{index}]")
        view_id = view_plan["view_id"]
        require(
            view_id not in seen_branch_views,
            f"{label}.branch_view_plans duplicate `{view_id}`",
        )
        seen_branch_views.add(view_id)
    metadata = value.get("metadata", {})
    require(isinstance(metadata, dict), f"{label}.metadata must be an object")


def validate_execution_plan(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} ExecutionPlan must be an object")
    raw_manifests = value.get("controller_manifests")
    if isinstance(raw_manifests, dict):
        for controller_id, manifest in raw_manifests.items():
            _validate_controller_manifest_deserialize_shape(
                manifest, f"{label}.controller_manifests[{controller_id}]"
            )
    # ExecutionPlan has no self-fingerprint parse boundary. Rust validates its
    # deserialized typed value, while the signed request/outcome/package parents
    # separately refuse any raw-vs-typed wire drift.
    value = _normalize_execution_plan(value)
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
    # Recompute the canonical order from real edges (deterministic lexicographic
    # Kahn, counting multiedges); a serialized order is never trusted.
    require(
        topological_order == graph_canonical_topological_order(graph),
        f"{label}.graph_plan topological_order does not match the graph canonical order",
    )

    parallel_levels = graph_plan.get("parallel_levels", [])
    require(
        isinstance(parallel_levels, list),
        f"{label}.graph_plan.parallel_levels must be an array",
    )
    flattened_levels: list[str] = []
    for level_index, level in enumerate(parallel_levels):
        require(
            isinstance(level, list),
            f"{label}.graph_plan.parallel_levels[{level_index}] array",
        )
        for node_index, node_id in enumerate(level):
            require_identifier(
                node_id,
                f"{label}.graph_plan.parallel_levels[{level_index}][{node_index}]",
            )
            flattened_levels.append(node_id)
    # Gate on the raw serialized outer list, never on the flattened node ids:
    # Rust rejects any non-empty outer list (e.g. `[[]]`) unless it is exactly the
    # canonical levels, so `[[]]` must not slip through by flattening to `[]`.
    if parallel_levels:
        require(
            set(flattened_levels) == graph_node_id_set,
            f"{label}.graph_plan.parallel_levels must cover graph nodes",
        )
        # A serialized non-empty level list must be exactly the canonical levels.
        require(
            parallel_levels == graph_canonical_parallel_levels(graph),
            f"{label}.graph_plan.parallel_levels are not the canonical dependency levels",
        )

    validate_campaign_spec(value.get("campaign"), f"{label}.campaign")
    node_plans = value.get("node_plans")
    require(
        isinstance(node_plans, dict) and node_plans,
        f"{label}.node_plans must be non-empty",
    )
    require(
        set(node_plans.keys()) == graph_node_id_set,
        f"{label}.node_plans must match graph nodes",
    )
    controllers = value.get("controller_manifests")
    require(
        isinstance(controllers, dict) and controllers,
        f"{label}.controller_manifests must be non-empty",
    )
    for controller_id, manifest in controllers.items():
        require_identifier(controller_id, f"{label}.controller_manifests key")
        validate_controller_manifest(
            manifest, f"{label}.controller_manifests[{controller_id}]"
        )
        require(
            manifest.get("controller_id") == controller_id,
            f"{label}.controller_manifests key `{controller_id}` mismatch",
        )
        # Plan manifests are BTreeSet-backed on the wire: phases/capabilities
        # must already be in canonical enum order (and unique).
        manifest_phases = manifest.get("supported_phases", [])
        require(
            manifest_phases
            == sorted(set(manifest_phases), key=W10_PHASE_ORDER.__getitem__),
            f"{label}.controller_manifests[{controller_id}].supported_phases must be in canonical order",
        )
        manifest_capabilities = manifest.get("capabilities", [])
        require(
            manifest_capabilities
            == sorted(set(manifest_capabilities), key=W10_CAPABILITY_ORDER.__getitem__),
            f"{label}.controller_manifests[{controller_id}].capabilities must be in canonical order",
        )

    graph_nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    for key, node_plan in node_plans.items():
        node_label = f"{label}.node_plans[{key}]"
        require(isinstance(node_plan, dict), f"{node_label} must be an object")
        require(node_plan.get("node_id") == key, f"{node_label}.node_id must match key")
        require(
            node_plan.get("kind") == graph_nodes_by_id[key].get("kind"),
            f"{node_label} node plan kind does not match graph node kind",
        )
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
        require(
            controller_id in controllers, f"{node_label}.controller_id has no manifest"
        )
        require_non_empty_string(
            node_plan.get("controller_version"), f"{node_label}.controller_version"
        )
        phases = node_plan.get("supported_phases")
        require(
            isinstance(phases, list) and phases,
            f"{node_label}.supported_phases non-empty",
        )
        for phase_index, phase in enumerate(phases):
            require(
                phase
                in {
                    "COMPILE",
                    "PLAN",
                    "FIT_CV",
                    "SELECT",
                    "REFIT",
                    "PREDICT",
                    "EXPLAIN",
                },
                f"{node_label}.supported_phases[{phase_index}] invalid",
            )
        capabilities = node_plan.get("controller_capabilities", [])
        require(
            isinstance(capabilities, list),
            f"{node_label}.controller_capabilities array",
        )
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
        # The node plan copies each capability-bearing ControllerManifest field
        # (version, phases, capabilities, the policy triple and kind); every one
        # must match exactly so replay derivation can trust either source.
        manifest = controllers[controller_id]
        require(
            node_plan.get("controller_version") == manifest.get("controller_version")
            and phases == manifest.get("supported_phases")
            and capabilities == manifest.get("capabilities")
            and node_plan.get("fit_scope") == manifest.get("fit_scope")
            and node_plan.get("rng_policy") == manifest.get("rng_policy")
            and node_plan.get("artifact_policy") == manifest.get("artifact_policy")
            and node_plan.get("kind") == manifest.get("operator_kind"),
            f"{node_label} node plan does not match controller manifest {controller_id}",
        )
        # input/output adjacency copies must equal the graph edges exactly
        # (sorted, de-duplicated) — never merely "each ref is some node id".
        require(
            node_plan.get("input_nodes") == graph_upstream_node_ids(graph, key)
            and node_plan.get("output_nodes") == graph_downstream_node_ids(graph, key),
            f"{node_label} input/output adjacency does not match the graph",
        )
        shape_plan = node_plan.get("shape_plan")
        if shape_plan is not None:
            validate_data_model_shape_plan(shape_plan, f"{node_label}.shape_plan")
            require(
                shape_plan.get("node_id") == key,
                f"{node_label}.shape_plan node_id mismatch",
            )
        data_bindings = node_plan.get("data_bindings", [])
        require(
            isinstance(data_bindings, list),
            f"{node_label}.data_bindings must be an array",
        )
        for binding_index, binding in enumerate(data_bindings):
            validate_data_binding(
                binding, f"{node_label}.data_bindings[{binding_index}]"
            )
            require(
                binding.get("node_id") == key,
                f"{node_label}.data_bindings node_id mismatch",
            )
        params = node_plan.get("params", {})
        require(isinstance(params, dict), f"{node_label}.params must be an object")
        require_sha256(
            node_plan.get("params_fingerprint"), f"{node_label}.params_fingerprint"
        )
        require(
            node_plan.get("params_fingerprint") == _node_params_fingerprint(params),
            f"{node_label} node plan params fingerprint does not match params",
        )

    variants = value.get("variants")
    require(
        isinstance(variants, list) and variants, f"{label}.variants must be non-empty"
    )
    for index, variant in enumerate(variants):
        variant_label = f"{label}.variants[{index}]"
        require(isinstance(variant, dict), f"{variant_label} must be an object")
        require_identifier(variant.get("variant_id"), f"{variant_label}.variant_id")
        require_sha256(variant.get("fingerprint"), f"{variant_label}.fingerprint")
        seed = variant.get("seed")
        if seed is not None:
            require(
                isinstance(seed, int) and seed >= 0, f"{variant_label}.seed invalid"
            )
        choices = variant.get("choices", {})
        require(isinstance(choices, dict), f"{variant_label}.choices must be an object")
        for dimension_name, choice in choices.items():
            choice_label = f"{variant_label}.choices[{dimension_name}]"
            require_non_empty_string(dimension_name, f"{choice_label}.dimension")
            require(isinstance(choice, dict), f"{choice_label} must be an object")
            require_non_empty_string(choice.get("label"), f"{choice_label}.label")
            overrides = choice.get("param_overrides", [])
            require(
                isinstance(overrides, list),
                f"{choice_label}.param_overrides must be an array",
            )
            for override_index, override in enumerate(overrides):
                override_label = f"{choice_label}.param_overrides[{override_index}]"
                require(
                    isinstance(override, dict), f"{override_label} must be an object"
                )
                override_node = override.get("node_id")
                require_identifier(override_node, f"{override_label}.node_id")
                require(
                    override_node in graph_node_id_set,
                    f"{override_label}.node_id unknown",
                )
                override_params = override.get("params", {})
                require(
                    isinstance(override_params, dict),
                    f"{override_label}.params must be object",
                )
            if "active_subsequence" in choice:
                active_subsequence = choice.get("active_subsequence")
                require_non_empty_string(
                    active_subsequence, f"{choice_label}.active_subsequence"
                )
                # Mirror Rust GenerationChoice::validate (trim().is_empty()): reject
                # whitespace-only, matching the schema `\\S` pattern.
                require(
                    active_subsequence.strip() != "",
                    f"{choice_label}.active_subsequence must not be whitespace-only",
                )
                require(
                    not overrides,
                    f"{choice_label} cannot set both param_overrides and active_subsequence",
                )

    fold_set = value.get("fold_set")
    if fold_set is not None:
        validate_fold_set(fold_set, f"{label}.fold_set")
    require_sha256(value.get("graph_fingerprint"), f"{label}.graph_fingerprint")
    require_sha256(value.get("campaign_fingerprint"), f"{label}.campaign_fingerprint")
    require_sha256(
        value.get("controller_fingerprint"), f"{label}.controller_fingerprint"
    )
    # Recompute the three embedded top-level plan fingerprints exactly as Rust
    # ExecutionPlan::validate does: SHA-256 of serde_json::to_vec of the typed
    # value in Rust struct field order (graph, campaign) and BTreeMap key order
    # (controller_manifests). This is the historical serde profile, NOT TCV1 and
    # NOT global sort_keys; a serialized value is never trusted, so a stale or
    # forged graph/campaign/controller fingerprint is rejected here.
    require(
        value.get("graph_fingerprint") == _serde_sha256(_normalize_graph_spec(graph)),
        f"{label}.graph_fingerprint does not match the embedded graph",
    )
    require(
        value.get("campaign_fingerprint")
        == _serde_sha256(_normalize_campaign_spec(value.get("campaign"))),
        f"{label}.campaign_fingerprint does not match the embedded campaign",
    )
    require(
        value.get("controller_fingerprint")
        == _serde_sha256(
            _normalize_controller_manifests(value.get("controller_manifests"))
        ),
        f"{label}.controller_fingerprint does not match the embedded controller manifests",
    )


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
            require(
                isinstance(values, list) and values,
                f"{port_label}.{field} must be non-empty",
            )
            require(
                len(set(values)) == len(values), f"{port_label}.{field} has duplicates"
            )
            for value_index, item in enumerate(values):
                require_non_empty_string(item, f"{port_label}.{field}[{value_index}]")
        rank = port.get("rank")
        if rank is not None:
            require(
                isinstance(rank, int) and 0 <= rank <= 16,
                f"{port_label}.rank is invalid",
            )
        for field in ("multi_source", "optional"):
            if field in port:
                require(
                    isinstance(port[field], bool),
                    f"{port_label}.{field} must be boolean",
                )
        metadata = port.get("metadata")
        if metadata is not None:
            require(
                isinstance(metadata, dict), f"{port_label}.metadata must be an object"
            )
    require(
        len(set(port_names)) == len(port_names),
        f"{label}.ports contain duplicate names",
    )

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
    require(
        isinstance(output_ports, dict) and output_ports,
        f"{label}.output_ports must be non-empty",
    )
    for port_name, output in output_ports.items():
        require_non_empty_string(port_name, f"{label}.output_ports key")
        require_non_empty_string(output, f"{label}.output_ports[{port_name}]")
        require(
            output in outputs,
            f"{label}.output_ports[{port_name}] references unknown output",
        )
    for field in ("warnings", "requires_user_choice"):
        values = value.get(field, [])
        require(isinstance(values, list), f"{label}.{field} must be an array")
        for index, item in enumerate(values):
            require_non_empty_string(item, f"{label}.{field}[{index}]")


def validate_controller_manifest(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} ControllerManifest must be an object")
    require_identifier(value.get("controller_id"), f"{label}.controller_id")
    require_non_empty_string(
        value.get("controller_version"), f"{label}.controller_version"
    )
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
    require(
        isinstance(priority, int) and 0 <= priority <= 4294967295,
        f"{label}.priority invalid",
    )

    phases = value.get("supported_phases")
    require(
        isinstance(phases, list) and phases,
        f"{label}.supported_phases must be non-empty",
    )
    require(
        len(set(phases)) == len(phases), f"{label}.supported_phases contain duplicates"
    )
    for index, phase in enumerate(phases):
        require(
            phase
            in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
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
                port.get("kind")
                in {"data", "target", "prediction", "artifact", "metric", "control"},
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
    require(
        len(set(capabilities)) == len(capabilities),
        f"{label}.capabilities contain duplicates",
    )
    for index, capability in enumerate(capabilities):
        require(
            capability in CONTROLLER_CAPABILITIES,
            f"{label}.capabilities[{index}] is invalid",
        )
    require(
        value.get("fit_scope")
        in {"stateless", "fold_train", "full_train", "inference_only"},
        f"{label}.fit_scope is invalid",
    )
    require(
        value.get("rng_policy")
        in {
            "uses_core_seed",
            "ignores_seed",
            "externally_deterministic",
            "nondeterministic",
        },
        f"{label}.rng_policy is invalid",
    )
    require(
        value.get("artifact_policy")
        in {"serializable", "host_only", "content_addressed", "replay_required"},
        f"{label}.artifact_policy is invalid",
    )
    if (
        "deterministic" in capabilities
        and value.get("rng_policy") == "nondeterministic"
    ):
        raise ContractError(
            f"{label} cannot be deterministic with nondeterministic RNG"
        )
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
    active_training_capabilities = {
        "uses_training_weights",
        "uses_early_stopping",
        "performs_internal_tuning",
        "trains_aggregation",
    }
    if value.get("fit_scope") in {"stateless", "inference_only"}:
        require(
            not (set(capabilities) & active_training_capabilities),
            f"{label} inactive fit scope has active training-influence capabilities",
        )
    if "uses_training_weights" in capabilities:
        require(
            bool(
                set(capabilities)
                & {
                    "supports_sample_weights",
                    "supports_row_resampling",
                    "supports_backend_loss_weights",
                }
            ),
            f"{label} uses training weights without a supported weighting mechanism",
        )
    if "trains_aggregation" in capabilities:
        require(
            "aggregates_predictions" in capabilities,
            f"{label} trains aggregation without aggregates_predictions",
        )


def validate_controller_manifest_list(value: Any, label: str) -> None:
    require(
        isinstance(value, list) and value, f"{label} must be a non-empty manifest array"
    )
    seen: set[str] = set()
    for index, manifest in enumerate(value):
        manifest_label = f"{label}[{index}]"
        validate_controller_manifest(manifest, manifest_label)
        controller_id = manifest["controller_id"]
        require(
            controller_id not in seen,
            f"{label} duplicate controller id `{controller_id}`",
        )
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
        require(
            isinstance(value["require_finite"], bool),
            f"{label}.require_finite must be boolean",
        )
    evaluation_scope = value.get("evaluation_scope")
    if evaluation_scope is not None:
        require(
            evaluation_scope in EVALUATION_SCOPES, f"{label}.evaluation_scope invalid"
        )
    validate_refit_slot_plan(
        value.get("refit_slot_plan"), f"{label}.refit_slot_plan", optional=True
    )
    validate_stacking_fit_contract(
        value.get("stacking_fit_contract"),
        f"{label}.stacking_fit_contract",
        optional=True,
    )
    require_optional_non_empty_string(
        value.get("reduction_id"), f"{label}.reduction_id"
    )


def validate_selection_decision(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} SelectionDecision must be an object")
    require_non_empty_string(value.get("policy_id"), f"{label}.policy_id")
    selected_candidate = value.get("selected_candidate_id")
    require_non_empty_string(selected_candidate, f"{label}.selected_candidate_id")
    require_non_empty_string(value.get("metric_name"), f"{label}.metric_name")
    require(
        value.get("objective") in {"minimize", "maximize"}, f"{label}.objective invalid"
    )
    metric_level = value.get("metric_level")
    if metric_level is not None:
        require(metric_level in PREDICTION_LEVELS, f"{label}.metric_level invalid")
    evaluation_scope = value.get("evaluation_scope")
    if evaluation_scope is not None:
        require(
            evaluation_scope in EVALUATION_SCOPES, f"{label}.evaluation_scope invalid"
        )
    validate_refit_slot_plan(
        value.get("refit_slot_plan"), f"{label}.refit_slot_plan", optional=True
    )
    require_optional_non_empty_string(
        value.get("reduction_id"), f"{label}.reduction_id"
    )
    selected_score = value.get("selected_score")
    require(
        isinstance(selected_score, (int, float)),
        f"{label}.selected_score must be numeric",
    )
    ranked = value.get("ranked_candidates")
    require(
        isinstance(ranked, list) and ranked,
        f"{label}.ranked_candidates must be non-empty",
    )
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
        require(
            candidate_id not in seen,
            f"{label} duplicate ranked candidate `{candidate_id}`",
        )
        seen.add(candidate_id)
        require(
            isinstance(candidate.get("score"), (int, float)),
            f"{candidate_label}.score numeric",
        )
        require(
            candidate.get("rank") == index + 1,
            f"{candidate_label}.rank must be {index + 1}",
        )


def validate_selection_metric(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require_non_empty_string(value.get("name"), f"{label}.name")
    require(
        value.get("objective") in {"minimize", "maximize"}, f"{label}.objective invalid"
    )


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
    require(
        isinstance(member_count, int) and member_count >= 1,
        f"{label}.member_count invalid",
    )
    if strategy == "refit_one":
        require(member_count == 1, f"{label}.refit_one requires member_count=1")
    if strategy == "refit_ensemble":
        require(member_count >= 2, f"{label}.refit_ensemble requires member_count>=2")
    validate_selection_metric(
        value.get("selection_metric"), f"{label}.selection_metric"
    )
    require_optional_non_empty_string(
        value.get("reduction_id"), f"{label}.reduction_id"
    )


def validate_stacking_fit_contract(
    value: Any, label: str, *, optional: bool = False
) -> None:
    if value is None:
        require(optional, f"{label} must be an object")
        return
    require(isinstance(value, dict), f"{label} must be an object")
    require(
        value.get("meta_training_features") == "oof",
        f"{label}.meta_training_features invalid",
    )
    require(
        value.get("inference_features") == "refit_base_predictions",
        f"{label}.inference_features invalid",
    )
    protocol = value.get("selection_protocol")
    require(
        protocol in {"nested", "holdout", "reuse_oof"},
        f"{label}.selection_protocol invalid",
    )
    domain = value.get("meta_row_domain")
    require(domain in {"sample", "combo"}, f"{label}.meta_row_domain invalid")
    final_reduction_id = value.get("final_reduction_id")
    require_optional_non_empty_string(final_reduction_id, f"{label}.final_reduction_id")
    unsafe_allow_reuse_oof = value.get("unsafe_allow_reuse_oof", False)
    require(
        isinstance(unsafe_allow_reuse_oof, bool),
        f"{label}.unsafe_allow_reuse_oof boolean",
    )
    if protocol == "reuse_oof" and not unsafe_allow_reuse_oof:
        raise ContractError(f"{label} reuse_oof requires unsafe_allow_reuse_oof=true")
    if domain == "combo" and final_reduction_id is None:
        raise ContractError(
            f"{label} combo meta_row_domain requires final_reduction_id"
        )


def validate_data_output_provenance(value: Any, label: str) -> None:
    require(
        isinstance(value, dict), f"{label} data-output provenance must be an object"
    )
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
        validate_representation_plan(
            representation_plan, f"{label}.representation_plan"
        )
    representation_replay_manifest = value.get("representation_replay_manifest")
    if representation_replay_manifest is not None:
        validate_representation_replay_manifest(
            representation_replay_manifest,
            f"{label}.representation_replay_manifest",
        )
    representation_compatibility = value.get("representation_compatibility")
    if representation_compatibility is not None:
        validate_representation_compatibility_report(
            representation_compatibility,
            f"{label}.representation_compatibility",
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
        require_sha256(
            delta.get("before_fingerprint"), f"{delta_label}.before_fingerprint"
        )
        require_sha256(
            delta.get("after_fingerprint"), f"{delta_label}.after_fingerprint"
        )
        if delta.get("kind") == "feature":
            last_feature_after = delta.get("after_fingerprint")
        metadata = delta.get("metadata")
        if metadata is not None:
            require(
                isinstance(metadata, dict), f"{delta_label}.metadata must be an object"
            )
    if last_feature_after is not None:
        require(
            value.get("feature_schema_fingerprint") == last_feature_after,
            f"{label}.feature_schema_fingerprint must match the last feature delta",
        )


def validate_handle_ref(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} handle ref must be an object")
    require(
        isinstance(value.get("handle"), int) and value["handle"] >= 0,
        f"{label}.handle invalid",
    )
    require(
        value.get("kind")
        in {"data", "data_view", "model", "artifact", "prediction", "relation"},
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
            isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and math.isfinite(weight)
            and weight > 0,
            f"{label}.row_weights[{index}] must be finite and > 0",
        )
    warnings = value.get("warnings", [])
    require(isinstance(warnings, list), f"{label}.warnings must be an array")
    for index, warning in enumerate(warnings):
        require_non_empty_string(warning, f"{label}.warnings[{index}]")
    if effective in {"equal_sample_influence", "backend_loss_weight"}:
        require(weights, f"{label}.{effective} requires row_weights")
    if requested == "strict_weight_support" and effective == "uniform_rows":
        raise ContractError(
            f"{label} strict_weight_support cannot fall back to uniform_rows"
        )


def validate_fit_influence_diagnostic(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    for field in ("requested_policy", "effective_policy"):
        require(value.get(field) in FIT_INFLUENCE_POLICIES, f"{label}.{field} invalid")
    require(
        value.get("mechanism") in FIT_INFLUENCE_MECHANISMS, f"{label}.mechanism invalid"
    )
    require(
        isinstance(value.get("fallback_used", False), bool),
        f"{label}.fallback_used boolean",
    )
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
    require_identifier(
        node_plan.get("controller_id"), f"{label}.node_plan.controller_id"
    )
    require_non_empty_string(
        node_plan.get("controller_version"),
        f"{label}.node_plan.controller_version",
    )
    require_non_empty_string(
        node_plan.get("params_fingerprint"),
        f"{label}.node_plan.params_fingerprint",
    )
    require(
        value.get("phase")
        in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
        f"{label}.phase invalid",
    )
    variant_id = value.get("variant_id")
    if variant_id is not None:
        require_identifier(variant_id, f"{label}.variant_id")
    variant = value.get("variant")
    if variant is not None:
        require(isinstance(variant, dict), f"{label}.variant must be an object")
        require(variant.get("variant_id") == variant_id, f"{label}.variant_id mismatch")
        require_non_empty_string(
            variant.get("fingerprint"), f"{label}.variant.fingerprint"
        )
        seed = variant.get("seed")
        if seed is not None:
            require(
                isinstance(seed, int) and seed >= 0, f"{label}.variant.seed invalid"
            )
    fold_id = value.get("fold_id")
    if fold_id is not None:
        require_identifier(fold_id, f"{label}.fold_id")
    seed = value.get("seed")
    if seed is not None:
        require(isinstance(seed, int) and seed >= 0, f"{label}.seed invalid")
    for map_name in (
        "input_handles",
        "data_views",
        "prediction_inputs",
        "artifact_inputs",
    ):
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
        require(
            isinstance(value.get(list_name, []), list),
            f"{label}.{list_name} must be an array",
        )
    artifact_handles = value.get("artifact_handles", {})
    require(
        isinstance(artifact_handles, dict),
        f"{label}.artifact_handles must be an object",
    )
    for artifact_id, handle in artifact_handles.items():
        require_identifier(artifact_id, f"{label}.artifact_handles key")
        validate_handle_ref(handle, f"{label}.artifact_handles[{artifact_id}]")
    diagnostics = value.get("fit_influence_diagnostics", [])
    require(
        isinstance(diagnostics, list),
        f"{label}.fit_influence_diagnostics must be an array",
    )
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
        lineage.get("phase")
        in {"COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"},
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
        require(
            isinstance(metric_value, (int, float)),
            f"{label}.lineage.metrics value numeric",
        )


def validate_node_task_result_pair(task: Any, result: Any, label: str) -> None:
    validate_node_task(task, f"{label}.task")
    validate_node_result(result, f"{label}.result")
    node_plan = task["node_plan"]
    lineage = result["lineage"]
    require(
        result.get("node_id") == node_plan.get("node_id"),
        f"{label} result node mismatch",
    )
    require(
        lineage.get("node_id") == node_plan.get("node_id"),
        f"{label} lineage node mismatch",
    )
    require(
        lineage.get("run_id") == task.get("run_id"), f"{label} lineage run mismatch"
    )
    require(
        lineage.get("phase") == task.get("phase"), f"{label} lineage phase mismatch"
    )
    require(
        lineage.get("controller_id") == node_plan.get("controller_id"),
        f"{label} lineage controller mismatch",
    )
    require(
        lineage.get("controller_version") == node_plan.get("controller_version"),
        f"{label} lineage controller_version mismatch",
    )
    require(
        lineage.get("variant_id") == task.get("variant_id"),
        f"{label} lineage variant mismatch",
    )
    require(
        lineage.get("fold_id") == task.get("fold_id"), f"{label} lineage fold mismatch"
    )
    require(lineage.get("seed") == task.get("seed"), f"{label} lineage seed mismatch")
    require(
        lineage.get("params_fingerprint") == node_plan.get("params_fingerprint"),
        f"{label} lineage params fingerprint mismatch",
    )


def validate_process_adapter_description(value: Any, label: str) -> None:
    require(
        isinstance(value, dict),
        f"{label} process-adapter description must be an object",
    )
    require(value.get("schema_version") == 1, f"{label}.schema_version must be 1")
    require(
        value.get("protocol") == "dag-ml-process-adapter",
        f"{label}.protocol must be dag-ml-process-adapter",
    )
    require_non_empty_string(value.get("adapter_id"), f"{label}.adapter_id")
    modes = value.get("supported_modes")
    require(
        isinstance(modes, list) and modes, f"{label}.supported_modes must be non-empty"
    )
    require(
        len(set(modes)) == len(modes), f"{label}.supported_modes contain duplicates"
    )
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
        require(
            task == task_fixture,
            f"{label}.task must match the canonical NodeTask fixture",
        )
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
        require(
            isinstance(error["retryable"], bool),
            f"{label}.error.retryable must be boolean",
        )


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
    require(
        "DagMlDataTensorF64" in header, f"{label} header must expose DagMlDataTensorF64"
    )
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
        "#define DAG_ML_PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_VERSION 1u"
        in header,
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


def validate_dag_ml_training_header(header: str, label: str) -> None:
    for symbol in (
        "typedef struct DagMlTrainingResult",
        "typedef struct DagMlTrainingExecuteRequest",
        "dagml_training_execute",
        "dagml_training_result_outcome_json",
        "dagml_training_result_free",
    ):
        require(symbol in header, f"{label} header must expose `{symbol}`")
    for field in (
        "request_json",
        "outcome_id",
        "run_id",
        "bundle_id",
        "relations_json",
        "influence_json",
        "envelopes_json",
        "warnings_json",
        "diagnostics_json",
        "data_provider",
        "controller_bindings",
        "controller_binding_count",
    ):
        require(
            field in header,
            f"{label} DagMlTrainingExecuteRequest must expose `{field}`",
        )


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
        require(
            record.get("kind") == expected_kind, f"{label}.kind must be {expected_kind}"
        )
    if expected_schema_version is not None:
        require(
            record.get("schema_version") == expected_schema_version,
            f"{label}.schema_version must be {expected_schema_version}",
        )
    digest = record.get("normalized_sha256", record.get("canonical_json_sha256"))
    require_sha256(digest, f"{label} digest")
    require(digest == expected_sha256, f"{label} digest does not match local artifact")


def validate_d8_conformance_scenarios(
    scenarios: Any,
    fixtures: dict[str, Any],
    label: str,
) -> None:
    require(
        isinstance(scenarios, dict), f"{label} conformance scenarios must be an object"
    )
    fixture_ids = set(fixtures.keys())
    for scenario_id in D8_CONFORMANCE_SCENARIOS:
        scenario = scenarios.get(scenario_id)
        require(isinstance(scenario, dict), f"{label} scenario `{scenario_id}` missing")
        polarity = scenario.get("polarity")
        require(
            polarity in {"positive", "negative"},
            f"{label} scenario `{scenario_id}` polarity invalid",
        )
        for field in ("surfaces", "assertions", "test_refs"):
            values = scenario.get(field)
            require(
                isinstance(values, list) and values,
                f"{label} scenario `{scenario_id}` {field} must be a non-empty list",
            )
            for index, value in enumerate(values):
                require_non_empty_string(
                    value, f"{label} scenario `{scenario_id}` {field}[{index}]"
                )
        fixture_refs = scenario.get("fixtures", [])
        require(
            isinstance(fixture_refs, list),
            f"{label} scenario `{scenario_id}` fixtures must be a list",
        )
        for index, fixture_ref in enumerate(fixture_refs):
            require_non_empty_string(
                fixture_ref,
                f"{label} scenario `{scenario_id}` fixtures[{index}]",
            )
            require(
                fixture_ref in fixture_ids,
                f"{label} scenario `{scenario_id}` references unknown fixture `{fixture_ref}`",
            )


def validate_conformance_pack(
    pack: Any,
    schema: Any,
    feature_fusion_schema: Any,
    branch_view_schema: Any,
    fitted_adapter_schema: Any,
    data_output_provenance_schema: Any,
    parity_oracle: Any,
    representation_registry: Any,
    fixture: Any,
    multisource_fixture: Any,
    feature_fusion_fixture: Any,
    model_input_spec_fixture: Any,
    data_output_provenance_fixture: Any,
    oof_success_fixture: Any,
    oof_train_refusal_fixture: Any,
    header: str,
    label: str,
) -> None:
    require(isinstance(pack, dict), f"{label} conformance pack must be a JSON object")
    require(
        pack.get("schema_version") == 1,
        f"{label} conformance pack schema_version must be 1",
    )
    require(
        pack.get("pack_id") == CONFORMANCE_PACK_ID,
        f"{label} conformance pack id mismatch",
    )

    contracts = pack.get("contracts")
    require(
        isinstance(contracts, dict),
        f"{label} conformance pack contracts must be an object",
    )
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
        contracts.get("data_output_provenance.v1"),
        canonical_json_sha256(normalize_schema(data_output_provenance_schema)),
        "json_schema",
        1,
        f"{label} data output provenance contract",
    )
    validate_digest_record(
        contracts.get("parity_oracle.v1"),
        canonical_json_sha256(parity_oracle),
        "parity_oracle_manifest",
        1,
        f"{label} parity oracle contract",
    )
    validate_digest_record(
        contracts.get("representation_registry.v1"),
        canonical_json_sha256(representation_registry),
        "representation_registry_manifest",
        1,
        f"{label} representation registry contract",
    )

    fixtures = pack.get("fixtures")
    require(
        isinstance(fixtures, dict),
        f"{label} conformance pack fixtures must be an object",
    )
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
    multisource_record = fixtures.get(
        "coordinator_data_plan_envelope_multisource_repetitions.v1"
    )
    validate_digest_record(
        multisource_record,
        canonical_json_sha256(multisource_fixture),
        None,
        None,
        f"{label} multisource coordinator envelope fixture",
    )
    require(
        multisource_record.get("contract") == "coordinator_data_plan_envelope.v1",
        f"{label} multisource fixture must reference coordinator contract",
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
    model_input_record = fixtures.get("model_input_spec_tabular_regressor.v1")
    validate_digest_record(
        model_input_record,
        canonical_json_sha256(model_input_spec_fixture),
        None,
        None,
        f"{label} model input spec fixture",
    )
    require(
        model_input_record.get("contract") == "model_input_spec.v1",
        f"{label} model input fixture must reference model input contract",
    )
    provenance_fixture = fixtures.get("data_output_provenance_augmented_view.v1")
    validate_digest_record(
        provenance_fixture,
        canonical_json_sha256(data_output_provenance_fixture),
        None,
        None,
        f"{label} data output provenance fixture",
    )
    require(
        provenance_fixture.get("contract") == "data_output_provenance.v1",
        f"{label} data output provenance fixture must reference provenance contract",
    )
    oof_success_record = fixtures.get("oof_uc6_success_predictions.v1")
    validate_digest_record(
        oof_success_record,
        canonical_json_sha256(oof_success_fixture),
        None,
        None,
        f"{label} OOF success fixture",
    )
    require(
        oof_success_record.get("contract") == "oof_campaign_fixture.v1",
        f"{label} OOF success fixture must reference OOF campaign fixture contract",
    )
    oof_refusal_record = fixtures.get("oof_uc11_train_prediction_refusal.v1")
    validate_digest_record(
        oof_refusal_record,
        canonical_json_sha256(oof_train_refusal_fixture),
        None,
        None,
        f"{label} OOF train-refusal fixture",
    )
    require(
        oof_refusal_record.get("contract") == "oof_campaign_fixture.v1",
        f"{label} OOF train-refusal fixture must reference OOF campaign fixture contract",
    )

    validate_d8_conformance_scenarios(
        pack.get("scenarios"),
        fixtures,
        label,
    )

    c_abi = pack.get("c_abi")
    require(
        isinstance(c_abi, dict), f"{label} conformance pack c_abi must be an object"
    )
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
        require(
            callback in callbacks, f"{label} conformance pack must require `{callback}`"
        )
        require(callback in header, f"{label} header must expose `{callback}`")
    data_symbols = c_abi.get("required_dag_ml_data_symbols")
    require(
        isinstance(data_symbols, list), f"{label} dag-ml-data symbols must be a list"
    )
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
    require(
        isinstance(cross_repo, dict),
        f"{label} cross_repo_conformance must be an object",
    )
    required_tests = cross_repo.get("required_when_sibling_checkout_present")
    require(
        isinstance(required_tests, list), f"{label} cross-repo tests must be a list"
    )
    for test_id in (
        "contracts.schema_and_fixture_equivalence",
        "headers.include_order",
        "provider.f64_predict_replay",
        "fold_set.fingerprint_parity",
        "representation_registry.parity",
        "model_input_spec.fixture_equivalence",
    ):
        require(
            test_id in required_tests,
            f"{label} conformance pack must require `{test_id}`",
        )


def validate_parity_oracle_manifest(
    oracle: Any,
    roots_by_repo: dict[str, Path],
    label: str,
) -> None:
    require(isinstance(oracle, dict), f"{label} parity oracle must be a JSON object")
    require(
        oracle.get("schema_version") == 1,
        f"{label} parity oracle schema_version must be 1",
    )
    require(
        oracle.get("oracle_id") == PARITY_ORACLE_ID,
        f"{label} parity oracle id mismatch",
    )
    require(
        oracle.get("status") == "producer_handoff",
        f"{label} parity oracle status mismatch",
    )

    consumer_ledger = oracle.get("consumer_ledger")
    require(
        isinstance(consumer_ledger, dict),
        f"{label} parity oracle ledger must be an object",
    )
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
    require(
        isinstance(shared, dict),
        f"{label} parity oracle shared block must be an object",
    )
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
    profiles_by_id: dict[str, dict[str, Any]] = {}
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
        profiles_by_id[profile["profile_id"]] = profile
    require(
        profile_ids == set(EXPECTED_PARITY_TOLERANCE_PROFILES),
        f"{label} parity oracle tolerance profile set changed",
    )
    for profile_id, expected_profile in EXPECTED_PARITY_TOLERANCE_PROFILES.items():
        profile = profiles_by_id[profile_id]
        for field, expected_value in expected_profile.items():
            require(
                profile.get(field) == expected_value,
                f"{label} parity oracle tolerance profile `{profile_id}` {field} drifted",
            )

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
    require(
        isinstance(cases, list) and cases,
        f"{label} parity oracle cases must be non-empty",
    )
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
            require_non_empty_string(
                topic, f"{case_label}.ledger_topics[{topic_index}]"
            )
        for invariant_index, invariant in enumerate(case["invariants"]):
            require_non_empty_string(
                invariant, f"{case_label}.invariants[{invariant_index}]"
            )
        for fixture_index, fixture in enumerate(case["fixtures"]):
            fixture_label = f"{case_label}.fixtures[{fixture_index}]"
            require(isinstance(fixture, dict), f"{fixture_label} must be an object")
            repo = fixture.get("repo")
            require(
                repo in {"dag-ml", "dag-ml-data"}, f"{fixture_label}.repo is invalid"
            )
            require_non_empty_string(fixture.get("path"), f"{fixture_label}.path")
            require_non_empty_string(fixture.get("kind"), f"{fixture_label}.kind")
            root = roots_by_repo.get(repo)
            if root is not None:
                require(
                    (root / fixture["path"]).is_file(),
                    f"{fixture_label} path is missing",
                )
        for gate_index, gate in enumerate(case["gates"]):
            gate_label = f"{case_label}.gates[{gate_index}]"
            require(isinstance(gate, dict), f"{gate_label} must be an object")
            require(
                gate.get("repo") in {"dag-ml", "dag-ml-data"},
                f"{gate_label}.repo is invalid",
            )
            require_non_empty_string(gate.get("command"), f"{gate_label}.command")
            require_non_empty_string(gate.get("proves"), f"{gate_label}.proves")
    require(
        case_ids == REQUIRED_PARITY_CASE_IDS, f"{label} parity oracle case set changed"
    )


def validate_research_provenance_profile(
    profile: Any,
    openlineage_facets_schema: Any,
    label: str,
) -> None:
    require(
        isinstance(profile, dict),
        f"{label} research provenance profile must be an object",
    )
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
    require(
        isinstance(required_files, list),
        f"{label} profile required_files must be a list",
    )
    required_by_path = {}
    for index, record in enumerate(required_files):
        record_label = f"{label} profile required_files[{index}]"
        require(isinstance(record, dict), f"{record_label} must be an object")
        path = record.get("path")
        require_non_empty_string(path, f"{record_label}.path")
        require(
            path not in required_by_path,
            f"{label} profile duplicates required path `{path}`",
        )
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
    require(
        isinstance(optional_files, list),
        f"{label} profile optional_files must be a list",
    )
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
            require_non_empty_string(
                record["path_pattern"], f"{record_label}.path_pattern"
            )
            try:
                re.compile(record["path_pattern"])
            except re.error as exc:
                raise ContractError(
                    f"{record_label}.path_pattern is invalid: {exc}"
                ) from exc
    for kind in (
        "dagml.prediction_cache_manifest",
        "dagml.artifact_manifest",
        "dagml.external_data_plan_envelope",
    ):
        require(
            kind in optional_kinds,
            f"{label} profile must include optional kind `{kind}`",
        )

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
        require(
            field in required_properties,
            f"{label} profile RO-Crate must require `{field}`",
        )
    require(
        ro_crate.get("required_json_encoding") == "application/json",
        f"{label} profile must require application/json encoding",
    )

    prov_jsonld = profile.get("prov_jsonld")
    require(
        isinstance(prov_jsonld, dict), f"{label} profile prov_jsonld must be an object"
    )
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
        require(
            section in sections,
            f"{label} profile must require PROV section `{section}`",
        )

    openlineage = profile.get("openlineage")
    require(
        isinstance(openlineage, dict), f"{label} profile openlineage must be an object"
    )
    require(
        openlineage.get("command") == "export-open-lineage",
        f"{label} profile OpenLineage command mismatch",
    )
    require(
        openlineage.get("facet_schema") == OPENLINEAGE_FACETS_SCHEMA_REL.name,
        f"{label} profile OpenLineage facet schema mismatch",
    )
    defs = openlineage_facets_schema.get("$defs")
    require(
        isinstance(defs, dict), f"{label} OpenLineage facets schema $defs are missing"
    )
    for facet_key, definition_name in (
        ("dagml_reproducibility", "DagmlReproducibilityRunFacet"),
        ("dagml_oof_safety", "DagmlOofSafetyRunFacet"),
    ):
        require(
            facet_key in openlineage.get("required_run_facets", []),
            f"{label} profile must require OpenLineage run facet `{facet_key}`",
        )
        require(
            definition_name in defs,
            f"{label} facet schema must define `{definition_name}`",
        )
    require(
        "dagml_plan" in openlineage.get("required_job_facets", []),
        f"{label} profile must require OpenLineage job facet `dagml_plan`",
    )
    require(
        "DagmlPlanJobFacet" in defs,
        f"{label} facet schema must define `DagmlPlanJobFacet`",
    )

    cli_conformance = profile.get("cli_conformance")
    require(
        isinstance(cli_conformance, dict),
        f"{label} profile cli_conformance must be an object",
    )
    require(
        cli_conformance.get("export_command") == "export-research-provenance",
        f"{label} profile export command mismatch",
    )
    require(
        cli_conformance.get("validation_command") == "validate-research-provenance",
        f"{label} profile validation command mismatch",
    )
    required_tests = cli_conformance.get("required_tests")
    require(
        isinstance(required_tests, list),
        f"{label} profile required_tests must be a list",
    )
    for test_id in (
        "cli_exports_research_provenance_bundle",
        "cli_selects_builds_and_validates_replay_bundle",
    ):
        require(
            test_id in required_tests, f"{label} profile must require test `{test_id}`"
        )


def conformal_robustness_is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def conformal_robustness_is_binary64(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def reject_conformal_robustness_runtime_handles(
    value: Any, label: str = "document"
) -> None:
    """Reject opaque handles anywhere in a portable W0.6 document."""

    if isinstance(value, list):
        for index, member in enumerate(value):
            reject_conformal_robustness_runtime_handles(member, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, member in value.items():
            lowered = key.lower()
            require(
                lowered != "handle"
                and not lowered.endswith("_handle")
                and not lowered.endswith("_handles"),
                f"{label}.{key} contains a forbidden runtime handle",
            )
            reject_conformal_robustness_runtime_handles(member, f"{label}.{key}")


def conformal_robustness_sorted_strings(
    value: Any, label: str, *, non_empty: bool
) -> list[str]:
    require(isinstance(value, list), f"{label} must be an array")
    if non_empty:
        require(bool(value), f"{label} must be non-empty")
    require(
        all(isinstance(item, str) and item for item in value),
        f"{label} has invalid text",
    )
    require(value == sorted(value), f"{label} must be sorted")
    require(len(set(value)) == len(value), f"{label} must be unique")
    return value


def validate_conformal_robustness_fingerprint(
    document: dict[str, Any], field: str, label: str
) -> None:
    require(field in document, f"{label} is missing `{field}`")
    expected = dagml_tcv1_sha256(
        {key: value for key, value in document.items() if key != field}
    )
    require(
        document[field] == expected,
        f"{label}.{field} does not match TCV1 content",
    )


def validate_w06_cohort_manifest(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "cohort manifest must be an object")
    reject_conformal_robustness_runtime_handles(document, "cohort")
    require(document.get("schema_version") == 1, "cohort schema_version must be 1")
    require(
        document.get("role") in CONFORMAL_COHORT_ROLES,
        "cohort role is invalid",
    )
    require(
        document.get("exchangeability_unit") == "physical_sample",
        "cohort unit is invalid",
    )
    physical_sample_ids = conformal_robustness_sorted_strings(
        document.get("physical_sample_ids"),
        "physical_sample_ids",
        non_empty=True,
    )
    for field in ("origin_sample_ids", "group_ids", "source_ids"):
        conformal_robustness_sorted_strings(document.get(field), field, non_empty=False)
    unit_relations = document.get("unit_relations")
    require(
        isinstance(unit_relations, list) and unit_relations,
        "unit_relations must be non-empty",
    )
    relation_sample_ids = [
        relation.get("physical_sample_id") for relation in unit_relations
    ]
    require(
        relation_sample_ids == physical_sample_ids,
        "unit_relations must align exactly with physical_sample_ids",
    )
    relation_origins: set[str] = set()
    relation_groups: set[str] = set()
    relation_sources: set[str] = set()
    for index, relation in enumerate(unit_relations):
        require(
            isinstance(relation, dict), f"unit_relations[{index}] must be an object"
        )
        origin = relation.get("origin_sample_id")
        if origin is not None:
            require_non_empty_string(
                origin, f"unit_relations[{index}].origin_sample_id"
            )
            relation_origins.add(origin)
        relation_groups.update(
            conformal_robustness_sorted_strings(
                relation.get("group_ids"),
                f"unit_relations[{index}].group_ids",
                non_empty=False,
            )
        )
        relation_sources.update(
            conformal_robustness_sorted_strings(
                relation.get("source_ids"),
                f"unit_relations[{index}].source_ids",
                non_empty=False,
            )
        )
    require(
        relation_origins == set(document["origin_sample_ids"]),
        "unit_relations origin closure differs from origin_sample_ids",
    )
    require(
        relation_groups == set(document["group_ids"]),
        "unit_relations group closure differs from group_ids",
    )
    require(
        relation_sources == set(document["source_ids"]),
        "unit_relations source closure differs from source_ids",
    )
    targets = document.get("target_names")
    require(isinstance(targets, list) and targets, "target_names must be non-empty")
    require(
        all(isinstance(target, str) and target for target in targets),
        "target_names has invalid text",
    )
    require(len(set(targets)) == len(targets), "target_names must be unique")
    validate_conformal_robustness_fingerprint(
        document, "manifest_fingerprint", "cohort"
    )
    return document


def assert_w06_calibration_disjoint(
    training_sample_ids: Any,
    training_origin_sample_ids: Any,
    calibration_cohort: Any,
) -> None:
    cohort = validate_w06_cohort_manifest(calibration_cohort)
    require(cohort["role"] == "calibration", "disjointness requires calibration role")
    training_samples = set(
        conformal_robustness_sorted_strings(
            training_sample_ids, "training samples", non_empty=True
        )
    )
    training_origins = set(
        conformal_robustness_sorted_strings(
            training_origin_sample_ids, "training origins", non_empty=False
        )
    )
    calibration_identity = set(cohort["physical_sample_ids"]) | set(
        cohort["origin_sample_ids"]
    )
    overlap = (training_samples | training_origins) & calibration_identity
    require(not overlap, f"calibration influence overlap: {sorted(overlap)}")


def validate_w06_prediction_block(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "prediction block must be an object")
    reject_conformal_robustness_runtime_handles(document, "prediction block")
    units = conformal_robustness_sorted_strings(
        document.get("unit_ids"), "unit_ids", non_empty=True
    )
    targets = document.get("target_names")
    require(isinstance(targets, list) and targets, "target_names must be non-empty")
    binding = document.get("point_output_binding")
    require(isinstance(binding, dict), "point_output_binding must be an object")
    require(
        targets == binding.get("target_names"),
        "target order differs from OutputBinding",
    )
    validate_conformal_robustness_fingerprint(
        binding, "binding_fingerprint", "point OutputBinding"
    )
    intervals = document.get("intervals")
    require(isinstance(intervals, list) and intervals, "intervals must be non-empty")
    require(
        all(isinstance(interval.get("coverage"), float) for interval in intervals),
        "prediction interval coverages must be represented as binary64",
    )
    coverages = validate_conformal_coverages(
        [interval.get("coverage") for interval in intervals]
    )
    previous: tuple[list[list[Any]], list[list[Any]]] | None = None
    has_unbounded = False
    for interval_index, interval in enumerate(intervals):
        lower = interval.get("lower")
        upper = interval.get("upper")
        require(
            isinstance(lower, list) and isinstance(upper, list),
            "interval bounds must be matrices",
        )
        require(
            len(lower) == len(units) == len(upper),
            "interval row count differs from unit_ids",
        )
        for row_index, (lower_row, upper_row) in enumerate(zip(lower, upper)):
            require(
                isinstance(lower_row, list) and isinstance(upper_row, list),
                f"interval row {row_index} must be an array",
            )
            require(
                len(lower_row) == len(targets) == len(upper_row),
                f"interval row {row_index} width differs from target_names",
            )
            for column_index, (lo, hi) in enumerate(zip(lower_row, upper_row)):
                if lo is None or hi is None:
                    require(
                        lo is None and hi is None,
                        "split absolute-residual unbounded endpoints must be paired",
                    )
                    has_unbounded = True
                    continue
                require(
                    isinstance(lo, float)
                    and isinstance(hi, float)
                    and math.isfinite(lo)
                    and math.isfinite(hi),
                    f"interval {interval_index} cell {row_index},{column_index} "
                    "must use finite binary64 endpoints or null",
                )
                require(
                    lo <= hi,
                    f"interval {interval_index} has lower bound above upper bound",
                )
        if previous is not None:
            previous_lower, previous_upper = previous
            for row_index, (lower_row, upper_row) in enumerate(zip(lower, upper)):
                for lo, previous_lo in zip(lower_row, previous_lower[row_index]):
                    require(
                        lo is None or (previous_lo is not None and lo <= previous_lo),
                        "higher coverage interval is not nested",
                    )
                for hi, previous_hi in zip(upper_row, previous_upper[row_index]):
                    require(
                        hi is None or (previous_hi is not None and hi >= previous_hi),
                        "higher coverage interval is not nested",
                    )
        previous = (lower, upper)
    assumption_status = document.get("assumption_status")
    require(
        assumption_status
        in {"declared_exchangeable", "distribution_shift", "not_assessed"},
        "prediction block assumption_status is invalid",
    )
    guarantee_status = document.get("guarantee_status")
    if has_unbounded:
        require(
            guarantee_status == "unavailable",
            "unbounded interval guarantee status must be unavailable",
        )
    else:
        expected_formal = (
            "joint_coverage"
            if document.get("multi_target_policy") == "joint_max"
            else "marginal_coverage"
        )
        if assumption_status == "declared_exchangeable":
            require(
                guarantee_status == expected_formal,
                "declared exchangeability has a mismatched coverage guarantee",
            )
        else:
            require(
                guarantee_status == "diagnostic_only",
                "shifted or unassessed finite interval overclaims coverage",
            )
    require(
        coverages == [interval["coverage"] for interval in intervals],
        "coverage normalization drifted",
    )
    validate_conformal_robustness_fingerprint(
        document, "block_fingerprint", "prediction block"
    )
    return document


def conformal_metric_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    slice_value = record["slice"]["value"] or ""
    return (
        record["scenario_id"] or "",
        record["severity"] if record["severity"] is not None else -1.0,
        record["slice"]["kind"],
        slice_value,
        record["target_name"] or "",
        record["coverage"],
        record["fold_id"] or "",
        record["repeat_id"] or "",
        record["seed"] if record["seed"] is not None else -1,
    )


def validate_w06_metric_set(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "metric set must be an object")
    reject_conformal_robustness_runtime_handles(document, "metric set")
    for field in (
        "point_prediction_fingerprint",
        "conformal_prediction_block_fingerprint",
        "truth_fingerprint",
        "unit_ids_fingerprint",
    ):
        require_sha256(document.get(field), f"metric set.{field}")
    records = document.get("records")
    require(isinstance(records, list) and records, "metric records must be non-empty")
    keys = [conformal_metric_record_key(record) for record in records]
    require(keys == sorted(keys), "metric records must be canonically sorted")
    require(len(set(keys)) == len(keys), "metric records contain duplicate coordinates")
    for record in records:
        require(
            isinstance(record["coverage"], float),
            "metric coverage must be represented as binary64",
        )
        require(
            record["severity"] is None or isinstance(record["severity"], float),
            "metric severity must be represented as binary64",
        )
        status = record.get("measurement_status")
        if status == "finite":
            numeric_fields = (
                "empirical_coverage",
                "coverage_gap",
                "mean_width",
                "median_width",
                "interval_score",
            )
            require(
                all(
                    isinstance(record[field], float) and math.isfinite(record[field])
                    for field in numeric_fields
                ),
                "finite metric record must contain binary64 values",
            )
            expected_gap = record["empirical_coverage"] - record["coverage"]
            require(
                math.isclose(
                    record["coverage_gap"],
                    expected_gap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "coverage_gap does not equal empirical_coverage - coverage",
            )
            if record["guarantee_status"] != "diagnostic_only":
                expected_guarantee = (
                    "joint_coverage"
                    if document.get("multi_target_policy") == "joint_max"
                    else "marginal_coverage"
                )
                require(
                    record["slice"]["kind"] == "all"
                    and record["guarantee_status"] == expected_guarantee,
                    "sliced or mismatched metric record overclaims a coverage guarantee",
                )
        elif status == "unbounded":
            require(
                isinstance(record["empirical_coverage"], float)
                and isinstance(record["coverage_gap"], float)
                and record["empirical_coverage"] == 1.0
                and math.isclose(
                    record["coverage_gap"],
                    1.0 - record["coverage"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "unbounded metric coverage arithmetic is invalid",
            )
            require(
                all(
                    record[field] is None
                    for field in ("mean_width", "median_width", "interval_score")
                ),
                "unbounded metric widths and interval score must be null",
            )
            require(
                record["guarantee_status"] == "unavailable",
                "unbounded metric guarantee status must be unavailable",
            )
        else:
            require(status == "unavailable", "metric measurement_status is invalid")
            require(
                all(
                    record[field] is None
                    for field in (
                        "empirical_coverage",
                        "coverage_gap",
                        "mean_width",
                        "median_width",
                        "interval_score",
                        "set_size",
                    )
                ),
                "unavailable metric values must be null",
            )
            require(
                record["guarantee_status"] == "unavailable",
                "unavailable metric guarantee status must be unavailable",
            )
        if document.get("multi_target_policy") == "joint_max":
            require(
                record["target_name"] is None,
                "joint_max metric target_name must be null",
            )
        else:
            require(
                isinstance(record["target_name"], str),
                "marginal metric needs target_name",
            )
        require(record["set_size"] is None, "regression metric set_size must be null")
        require(
            isinstance(record.get("sample_count"), int)
            and not isinstance(record["sample_count"], bool)
            and record["sample_count"] > 0,
            "metric sample_count must be positive",
        )
        require_sha256(
            record.get("unit_ids_fingerprint"),
            "metric record.unit_ids_fingerprint",
        )
    validate_conformal_robustness_fingerprint(
        document, "metric_set_fingerprint", "metric set"
    )
    return document


def validate_w06_domain_assessment(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "domain block must be an object")
    reject_conformal_robustness_runtime_handles(document, "domain block")
    units = conformal_robustness_sorted_strings(
        document.get("unit_ids"), "unit_ids", non_empty=True
    )
    assessments = document.get("assessments")
    require(isinstance(assessments, list), "assessments must be an array")
    ids = [record["unit_id"] for record in assessments]
    require(ids == units, "domain assessments must align exactly with unit_ids")
    for assessment in assessments:
        methods = assessment.get("methods")
        require(
            isinstance(methods, list) and methods, "domain methods must be non-empty"
        )
        method_ids = [method["method_id"] for method in methods]
        require(
            method_ids == sorted(method_ids),
            "domain methods must be sorted by method_id",
        )
        require(
            len(set(method_ids)) == len(method_ids), "domain method ids must be unique"
        )
        for method in methods:
            score = method["score"]
            threshold = method["threshold"]
            if method["supported"] is None:
                require(
                    score is None and threshold is None,
                    "unknown domain method must not carry a partial support decision",
                )
            else:
                require(
                    all(
                        conformal_robustness_is_binary64(value)
                        for value in (score, threshold)
                    ),
                    "decided domain method needs finite binary64 score and threshold",
                )
        supported = [method["supported"] for method in methods]
        expected_status = (
            "out_of_support"
            if any(value is False for value in supported)
            else "in_support"
            if all(value is True for value in supported)
            else "unknown"
        )
        require(
            assessment.get("status") == expected_status,
            "domain assessment status contradicts method support",
        )
        conformal_robustness_sorted_strings(
            assessment.get("reasons"), "domain reasons", non_empty=False
        )
    validate_conformal_robustness_fingerprint(
        document, "block_fingerprint", "domain block"
    )
    return document


def validate_w06_decision_block(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "decision block must be an object")
    reject_conformal_robustness_runtime_handles(document, "decision block")
    units = conformal_robustness_sorted_strings(
        document.get("unit_ids"), "unit_ids", non_empty=True
    )
    decisions = document.get("decisions")
    require(isinstance(decisions, list), "decisions must be an array")
    ids = [record["unit_id"] for record in decisions]
    require(ids == units, "decisions must align exactly with unit_ids")
    require(
        document.get("conformal_block_fingerprint") is not None
        or document.get("domain_assessment_fingerprint") is not None,
        "decision block needs conformal or domain evidence",
    )
    thresholds = document.get("thresholds")
    require(isinstance(thresholds, list) and thresholds, "thresholds must be non-empty")
    threshold_names = [threshold["name"] for threshold in thresholds]
    require(threshold_names == sorted(threshold_names), "thresholds must be sorted")
    require(
        len(set(threshold_names)) == len(threshold_names), "thresholds must be unique"
    )
    for threshold in thresholds:
        operator = threshold.get("operator")
        threshold_value = threshold.get("value")
        if operator in {"lt", "lte", "gt", "gte"}:
            require(
                conformal_robustness_is_binary64(threshold_value),
                f"threshold `{threshold['name']}` needs a finite numeric binary64 value",
            )
        elif operator in {"in", "not_in"}:
            require(
                isinstance(threshold_value, list),
                f"threshold `{threshold['name']}` needs an array value",
            )
        if isinstance(threshold_value, list):
            require(
                all(
                    not conformal_robustness_is_number(member)
                    or conformal_robustness_is_binary64(member)
                    for member in threshold_value
                ),
                f"threshold `{threshold['name']}` numeric members must be binary64",
            )
        elif conformal_robustness_is_number(threshold_value):
            require(
                conformal_robustness_is_binary64(threshold_value),
                f"threshold `{threshold['name']}` numeric value must be binary64",
            )
    for decision in decisions:
        conformal_robustness_sorted_strings(
            decision.get("reasons"), "decision reasons", non_empty=True
        )
    validate_conformal_robustness_fingerprint(
        document, "block_fingerprint", "decision block"
    )
    return document


def validate_w06_scenario(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "scenario must be an object")
    reject_conformal_robustness_runtime_handles(document, "scenario")
    require(
        document.get("cohort_role") in {"external_test", "production"},
        "scenario cohort role must be evaluation-only",
    )
    for field, non_empty in (
        ("source_ids", False),
        ("node_ids", False),
        ("slice_by", False),
        ("metrics", True),
    ):
        conformal_robustness_sorted_strings(
            document.get(field), field, non_empty=non_empty
        )
    severities = document.get("severities")
    require(isinstance(severities, list) and severities, "severities must be non-empty")
    require(severities[0] == 0.0, "severities must begin with identity severity 0.0")
    require(
        document.get("zero_severity_semantics") == "identity",
        "zero severity must have identity semantics",
    )
    require(
        all(
            isinstance(value, float) and math.isfinite(value) and value >= 0
            for value in severities
        ),
        "severities must be finite, non-negative binary64 values",
    )
    require(
        all(left < right for left, right in zip(severities, severities[1:])),
        "severities must be strictly increasing",
    )
    rng = document.get("rng")
    require(isinstance(rng, dict), "scenario rng must be an object")
    require(rng.get("algorithm") == "philox4x32-10", "scenario RNG algorithm drifted")
    require(rng.get("algorithm_version") == 1, "scenario RNG version drifted")
    require(
        rng.get("counter_profile") == "dagml-robustness-counter.v1",
        "scenario RNG counter profile drifted",
    )
    require(
        rng.get("counter_derivation") == "sha256-tcv1-first128",
        "scenario RNG counter derivation drifted",
    )
    require(
        rng.get("counter_fields")
        == [
            "scenario_fingerprint",
            "severity_binary64",
            "unit_id",
            "target_kind",
            "target_id",
            "draw_index",
        ],
        "scenario RNG counter fields drifted",
    )
    require(
        rng.get("key_derivation") == "uint64-seed-as-two-little-endian-u32",
        "scenario RNG key derivation drifted",
    )
    expected = {
        "clean_frozen": (False, False),
        "matched_recalibration": (False, True),
        "structural_refit": (True, True),
    }
    mode = document.get("mode")
    require(mode in expected, "scenario mode is invalid")
    require(
        (document.get("requires_refit"), document.get("requires_recalibration"))
        == expected[mode],
        "scenario mode/refit/recalibration policy is inconsistent",
    )
    perturbation = document.get("perturbation")
    require(isinstance(perturbation, dict), "scenario perturbation must be an object")
    perturbation_kind = perturbation.get("kind")
    require(
        perturbation_kind
        in {
            "identity",
            "gaussian_noise",
            "ordered_axis_shift",
            "source_dropout",
            "node_replacement",
            "custom_host",
        },
        "scenario perturbation kind is invalid",
    )
    require(
        (perturbation_kind == "node_replacement") == (mode == "structural_refit"),
        "node_replacement is valid if and only if mode is structural_refit",
    )
    if perturbation_kind == "identity":
        require(
            severities == [0.0],
            "identity perturbation only permits the [0.0] severity grid",
        )
    if mode == "structural_refit":
        require(
            perturbation_kind == "node_replacement",
            "structural_refit needs node_replacement",
        )
        require(bool(document["node_ids"]), "structural_refit needs a target node")
        require(
            rng.get("target_kind") == "node", "structural RNG target_kind must be node"
        )
    elif perturbation_kind in {
        "gaussian_noise",
        "ordered_axis_shift",
        "source_dropout",
    }:
        require(bool(document["source_ids"]), "source perturbation needs a source")
        require(
            rng.get("target_kind") == "source", "source RNG target_kind must be source"
        )
    else:
        require(rng.get("target_kind") == "global", "global RNG target_kind drifted")
    validate_conformal_robustness_fingerprint(
        document, "scenario_fingerprint", "scenario"
    )
    return document


def validate_w06_point_metrics(
    value: Any,
    scenario: dict[str, Any],
    *,
    severity_zero: bool,
    baseline: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    require(isinstance(value, list) and value, "point_metrics must be non-empty")
    names = [record["metric"] for record in value]
    require(names == sorted(names), "point metrics must be sorted")
    require(len(set(names)) == len(names), "point metrics contain duplicates")
    require(set(names) <= set(scenario["metrics"]), "point metric was not requested")
    if baseline is not None:
        require(set(names) == set(baseline), "point metrics differ from exact baseline")
    records = {record["metric"]: record for record in value}
    for record in value:
        metric = record["metric"]
        if record["status"] == "unavailable":
            require(
                all(
                    record[field] is None
                    for field in ("value", "baseline_value", "degradation")
                ),
                "unavailable point metric values must be null",
            )
            require(not severity_zero, "severity-zero point metric is unavailable")
            if baseline is not None:
                require(
                    baseline[metric]["status"] == "unavailable",
                    "unavailable point metric differs from baseline status",
                )
            continue
        require(record["status"] == "finite", "point metric status is invalid")
        require(
            all(
                isinstance(record[field], float) and math.isfinite(record[field])
                for field in ("value", "baseline_value", "degradation")
            ),
            "finite point metric values must be represented as binary64",
        )
        if severity_zero:
            require(
                record["value"] == record["baseline_value"]
                and record["degradation"] == 0.0,
                "severity zero point metric is not identity",
            )
        else:
            require(baseline is not None, "nonzero point metric has no exact baseline")
            baseline_record = baseline[metric]
            require(
                baseline_record["status"] == "finite"
                and baseline_record["direction"] == record["direction"],
                "point metric baseline status or direction drifted",
            )
            require(
                record["baseline_value"] == baseline_record["value"],
                "point metric baseline_value differs from severity-zero value",
            )
        expected = (
            record["value"] - record["baseline_value"]
            if record["direction"] == "minimize"
            else record["baseline_value"] - record["value"]
        )
        require(
            math.isclose(record["degradation"], expected, rel_tol=0.0, abs_tol=1e-12),
            "point metric degradation is inconsistent with direction and baseline",
        )
    return records


def w06_result_coordinate(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result["scenario_id"],
        result["severity"],
        result["split_id"],
        result["environment_id"],
        result["fold_id"] or "",
        result["repeat_id"] or "",
        result["seed"],
        result["slice"]["kind"],
        result["slice"]["value"] or "",
        result["unit_level"],
    )


def w06_result_baseline_coordinate(result: dict[str, Any]) -> tuple[Any, ...]:
    coordinate = list(w06_result_coordinate(result))
    coordinate[1] = 0.0
    return tuple(coordinate)


def w06_influence_identity(influence: dict[str, Any]) -> set[str]:
    identity: set[str] = set()
    for entry in influence["entries"]:
        identity.update(entry["physical_sample_ids"])
        identity.update(entry["origin_sample_ids"])
    return identity


def validate_w06_block_calibration_closure(
    block: dict[str, Any], artifact: dict[str, Any]
) -> None:
    """Close split-absolute-residual interval widths over calibrated quantiles."""

    spec = artifact["calibration_spec"]
    quantiles = {record["coverage"]: record for record in artifact["quantiles"]}
    target_count = len(block["target_names"])
    require(
        block["method"] == spec["method"]
        and block["numeric_version"] == spec["numeric_version"],
        "prediction block method or numeric version differs from calibrator",
    )
    for interval in block["intervals"]:
        coverage = interval["coverage"]
        require(
            coverage in quantiles,
            "prediction block coverage is not calibrated by its artifact",
        )
        values = quantiles[coverage]["values"]
        expected_width = (
            target_count if spec["multi_target_policy"] == "marginal" else 1
        )
        require(
            len(values) == expected_width,
            "calibration quantile width differs from prediction target policy",
        )
        for row_index, (lower_row, upper_row) in enumerate(
            zip(interval["lower"], interval["upper"])
        ):
            for target_index, (lower, upper) in enumerate(zip(lower_row, upper_row)):
                quantile = values[
                    target_index if spec["multi_target_policy"] == "marginal" else 0
                ]
                if quantile["status"] == "unbounded":
                    require(
                        lower is None and upper is None,
                        "unbounded calibration quantile produced finite endpoints",
                    )
                    continue
                require(
                    isinstance(lower, float) and isinstance(upper, float),
                    "finite calibration quantile requires binary64 endpoints",
                )
                radius = quantile["value"]
                require(
                    Decimal(repr(upper)) - Decimal(repr(lower))
                    == Decimal(2) * Decimal(repr(radius)),
                    f"prediction interval radius drifted at row {row_index}, target {target_index}",
                )


def w06_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def validate_w06_metric_width_closure(
    metric_set: dict[str, Any], block: dict[str, Any]
) -> None:
    """Reconstruct mean/median widths from the referenced prediction bounds."""

    intervals = {interval["coverage"]: interval for interval in block["intervals"]}
    target_indices = {
        target_name: index for index, target_name in enumerate(block["target_names"])
    }
    for record in metric_set["records"]:
        coverage = record["coverage"]
        require(
            coverage in intervals,
            "metric record coverage does not resolve to its prediction block",
        )
        interval = intervals[coverage]
        if metric_set["multi_target_policy"] == "marginal":
            require(
                record["target_name"] in target_indices,
                "marginal metric target does not resolve to prediction block",
            )
            selected_targets = [target_indices[record["target_name"]]]
        else:
            selected_targets = list(range(len(block["target_names"])))
        widths: list[float | None] = []
        for lower_row, upper_row in zip(interval["lower"], interval["upper"]):
            for target_index in selected_targets:
                lower = lower_row[target_index]
                upper = upper_row[target_index]
                widths.append(
                    None
                    if lower is None or upper is None
                    else float(upper) - float(lower)
                )
        if any(width is None for width in widths):
            require(
                record["measurement_status"] == "unbounded"
                and record["mean_width"] is None
                and record["median_width"] is None,
                "unbounded prediction bounds have finite width metrics",
            )
            continue
        finite_widths = [float(width) for width in widths if width is not None]
        require(
            record["measurement_status"] == "finite",
            "finite prediction bounds have a non-finite metric status",
        )
        require(
            math.isclose(
                record["mean_width"],
                sum(finite_widths) / len(finite_widths),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                record["median_width"],
                w06_median(finite_widths),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "metric mean/median width does not reconstruct from prediction bounds",
        )


def validate_w06_report(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "report must be an object")
    reject_conformal_robustness_runtime_handles(document, "report")
    cohort_value = document.get("cohort_manifest")
    require(
        isinstance(cohort_value, dict)
        and cohort_value.get("role") in {"external_test", "production"},
        "robustness report cohort must be external_test or production",
    )
    cohort = validate_w06_cohort_manifest(cohort_value)
    cohort_identity = set(cohort["physical_sample_ids"]) | set(
        cohort["origin_sample_ids"]
    )
    relations = {
        relation["physical_sample_id"]: relation
        for relation in cohort["unit_relations"]
    }

    scenarios = [
        validate_w06_scenario(scenario) for scenario in document.get("scenarios", [])
    ]
    scenario_ids = [scenario["scenario_id"] for scenario in scenarios]
    require(
        scenario_ids == sorted(scenario_ids), "report scenarios must be sorted by id"
    )
    require(
        len(set(scenario_ids)) == len(scenario_ids),
        "report scenarios contain duplicates",
    )
    scenario_map = {scenario["scenario_id"]: scenario for scenario in scenarios}
    require(
        all(scenario["cohort_role"] == cohort["role"] for scenario in scenarios),
        "report scenario cohort role differs from report cohort",
    )
    require(
        all(
            set(scenario["source_ids"]) <= set(cohort["source_ids"])
            for scenario in scenarios
        ),
        "report scenario references a source outside the cohort",
    )

    artifacts = document.get("calibration_artifacts")
    require(
        isinstance(artifacts, list), "report calibration_artifacts must be an array"
    )
    validated_artifacts = [
        validate_conformal_calibration_artifact(
            artifact, f"report.calibration_artifacts[{index}]"
        )
        for index, artifact in enumerate(artifacts)
    ]
    artifact_ids = [artifact["artifact_id"] for artifact in validated_artifacts]
    require(
        artifact_ids == sorted(artifact_ids), "calibration artifacts must be sorted"
    )
    require(
        len(set(artifact_ids)) == len(artifact_ids),
        "calibration artifact ids duplicate",
    )
    artifact_checksums = [artifact["checksum"] for artifact in validated_artifacts]
    require(
        len(set(artifact_checksums)) == len(artifact_checksums),
        "calibration artifact checksums duplicate",
    )
    artifact_map = {artifact["checksum"]: artifact for artifact in validated_artifacts}
    for artifact in validated_artifacts:
        calibration_cohort = validate_w06_cohort_manifest(
            artifact["calibration_cohort"]
        )
        require(
            calibration_cohort["role"] == "calibration",
            "report calibration artifact cohort role is not calibration",
        )
        calibration_identity = set(calibration_cohort["physical_sample_ids"]) | set(
            calibration_cohort["origin_sample_ids"]
        )
        require(
            not (calibration_identity & cohort_identity),
            "calibration cohort overlaps external_test identity closure",
        )
        influence = validate_training_influence_manifest(
            artifact["training_influence"],
            f"report calibration artifact `{artifact['artifact_id']}` influence",
        )
        require(
            not (w06_influence_identity(influence) & cohort_identity),
            "predictor training influence overlaps external_test identity closure",
        )
    base_checksum = document.get("calibration_artifact_checksum")
    require(
        base_checksum is None or base_checksum in artifact_map,
        "report baseline calibration checksum does not resolve",
    )
    if base_checksum is not None:
        require(
            artifact_map[base_checksum]["predictor_binding_fingerprint"]
            == document.get("predictor_binding_fingerprint"),
            "report baseline calibrator is bound to another predictor",
        )
    structural_scenarios = [
        scenario for scenario in scenarios if scenario["mode"] == "structural_refit"
    ]
    if structural_scenarios:
        require(
            base_checksum is not None,
            "structural scenario cannot resolve the baseline predictor closure",
        )
        predictor = artifact_map[base_checksum]["predictor_binding"]
        predictor_nodes = set(predictor["predictor_node_ids"])
        for scenario in structural_scenarios:
            require(
                set(scenario["node_ids"]) <= predictor_nodes,
                "structural scenario targets a node outside the predictor closure",
            )

    prediction_blocks = [
        validate_w06_prediction_block(block)
        for block in document.get("conformal_prediction_blocks", [])
    ]
    block_ids = [block["block_id"] for block in prediction_blocks]
    require(
        block_ids == sorted(block_ids), "conformal prediction blocks must be sorted"
    )
    require(
        len(set(block_ids)) == len(block_ids),
        "conformal prediction block ids duplicate",
    )
    block_fingerprints = [block["block_fingerprint"] for block in prediction_blocks]
    require(
        len(set(block_fingerprints)) == len(block_fingerprints),
        "conformal prediction block fingerprints duplicate",
    )
    block_map = {block["block_fingerprint"]: block for block in prediction_blocks}
    for block in prediction_blocks:
        require(
            block["cohort_manifest_fingerprint"] == cohort["manifest_fingerprint"],
            "prediction block is bound to another cohort",
        )
        require(
            set(block["unit_ids"]) <= set(cohort["physical_sample_ids"]),
            "prediction block units are outside cohort",
        )
        require(
            block["target_names"] == cohort["target_names"],
            "prediction block targets differ from cohort",
        )
        checksum = block["calibration_artifact_checksum"]
        require(
            checksum in artifact_map, "prediction block calibrator does not resolve"
        )
        artifact = artifact_map[checksum]
        require(
            block["calibration_artifact_id"] == artifact["artifact_id"],
            "prediction block calibration id drifted",
        )
        require(
            block["predictor_binding_fingerprint"]
            == artifact["predictor_binding_fingerprint"],
            "prediction block predictor differs from calibrator",
        )
        require(
            block["point_output_binding"]
            == artifact["predictor_binding"]["output_binding"],
            "prediction block OutputBinding differs from calibrator",
        )
        require(
            block["multi_target_policy"]
            == artifact["calibration_spec"]["multi_target_policy"],
            "prediction block target policy differs from calibrator",
        )
        validate_w06_block_calibration_closure(block, artifact)

    metric_sets = [
        validate_w06_metric_set(metric_set)
        for metric_set in document.get("conformal_metric_sets", [])
    ]
    metric_ids = [metric_set["metric_set_id"] for metric_set in metric_sets]
    require(metric_ids == sorted(metric_ids), "report metric sets must be sorted")
    require(len(set(metric_ids)) == len(metric_ids), "report metric sets duplicate")
    require(
        all(
            metric_set["cohort_manifest_fingerprint"] == cohort["manifest_fingerprint"]
            for metric_set in metric_sets
        ),
        "report metric set is bound to another cohort",
    )
    require(
        all(
            metric_set["conformal_prediction_block_fingerprint"] in block_map
            for metric_set in metric_sets
        ),
        "report metric set references an unknown conformal prediction block",
    )
    metric_map = {metric_set["metric_set_id"]: metric_set for metric_set in metric_sets}

    results = document.get("results")
    require(isinstance(results, list) and results, "report results must be non-empty")
    result_coordinates = [w06_result_coordinate(result) for result in results]
    require(
        result_coordinates == sorted(result_coordinates),
        "report results must be sorted",
    )
    require(
        len(set(result_coordinates)) == len(result_coordinates),
        "report results duplicate",
    )
    results_by_coordinate = {
        w06_result_coordinate(result): result for result in results
    }
    used_calibrators: set[str] = set()
    used_prediction_blocks: set[str] = set()
    used_metric_sets: set[str] = set()
    for result in results:
        scenario_id = result["scenario_id"]
        require(scenario_id in scenario_map, "result references unknown scenario")
        scenario = scenario_map[scenario_id]
        require(
            isinstance(result["severity"], float),
            "result severity must be represented as binary64",
        )
        require(
            result["severity"] in scenario["severities"],
            "result severity is not declared",
        )
        require(
            result["environment_id"] == scenario["environment_id"],
            "result environment differs from scenario",
        )
        require(
            result["split_id"] == scenario["split_id"],
            "result split differs from scenario",
        )
        require(
            result["seed"] == scenario["rng"]["seed"],
            "result seed differs from scenario",
        )
        unit_ids = conformal_robustness_sorted_strings(
            result.get("unit_ids"), "result unit_ids", non_empty=True
        )
        require(
            result["unit_count"] == len(unit_ids),
            "result unit_count differs from unit_ids",
        )
        require(
            set(unit_ids) <= set(cohort["physical_sample_ids"]),
            "result units outside cohort",
        )
        slice_kind = result["slice"]["kind"]
        slice_value = result["slice"]["value"]
        if slice_kind != "all":
            require(
                slice_kind in scenario["slice_by"],
                "result uses an undeclared slice dimension",
            )
        if slice_kind == "all":
            expected_units = cohort["physical_sample_ids"]
        elif slice_kind == "group":
            require(
                slice_value in cohort["group_ids"],
                "result references unknown group slice",
            )
            expected_units = sorted(
                sample_id
                for sample_id, relation in relations.items()
                if slice_value in relation["group_ids"]
            )
        elif slice_kind == "source":
            require(
                slice_value in cohort["source_ids"],
                "result references unknown source slice",
            )
            expected_units = sorted(
                sample_id
                for sample_id, relation in relations.items()
                if slice_value in relation["source_ids"]
            )
        elif slice_kind == "environment":
            require(
                slice_value == scenario["environment_id"],
                "result environment slice drifted",
            )
            expected_units = cohort["physical_sample_ids"]
        else:
            require(slice_kind == "target", "result slice kind is invalid")
            require(
                slice_value in cohort["target_names"],
                "result references unknown target slice",
            )
            expected_units = cohort["physical_sample_ids"]
        require(
            unit_ids == expected_units,
            "result unit_ids differ from cohort slice relation",
        )

        for field in (
            "before_predictor_fingerprint",
            "after_predictor_fingerprint",
            "before_input_fingerprint",
            "after_input_fingerprint",
            "before_relation_fingerprint",
            "after_relation_fingerprint",
            "before_point_prediction_fingerprint",
            "after_point_prediction_fingerprint",
        ):
            require_sha256(result.get(field), f"result.{field}")
        before_predictor = result["before_predictor_fingerprint"]
        after_predictor = result["after_predictor_fingerprint"]
        before_calibration = result["before_calibration_checksum"]
        after_calibration = result["after_calibration_checksum"]
        require(
            before_predictor == document["predictor_binding_fingerprint"],
            "result baseline predictor differs from report binding",
        )
        require(
            before_calibration == base_checksum,
            "result baseline calibrator differs from report",
        )
        require(
            result["before_relation_fingerprint"] == cohort["relation_fingerprint"],
            "result baseline relation differs from cohort",
        )
        for checksum in (before_calibration, after_calibration):
            if checksum is not None:
                require(
                    checksum in artifact_map,
                    "result calibration checksum does not resolve",
                )
                used_calibrators.add(checksum)
        block_fingerprint = result["conformal_prediction_block_fingerprint"]
        if block_fingerprint is not None:
            require(
                block_fingerprint in block_map,
                "result references unknown conformal prediction block",
            )
            used_prediction_blocks.add(block_fingerprint)
            block = block_map[block_fingerprint]
            require(
                block["unit_ids"] == unit_ids,
                "prediction block unit_ids differ from result",
            )
            require(
                block["point_prediction_fingerprint"]
                == result["after_point_prediction_fingerprint"],
                "prediction block point prediction differs from result",
            )
            require(
                block["predictor_binding_fingerprint"] == after_predictor,
                "prediction block is bound to another predictor",
            )
            require(
                block["calibration_artifact_checksum"] == after_calibration,
                "prediction block is bound to another calibrator",
            )
            require(
                block["guarantee_status"] == result["coverage_guarantee_status"],
                "prediction block guarantee differs from result",
            )
        else:
            require(
                result["coverage_guarantee_status"] == "unavailable",
                "result without conformal prediction block overclaims coverage",
            )

        if result["severity"] != 0:
            baseline_state = results_by_coordinate.get(
                w06_result_baseline_coordinate(result)
            )
            require(
                baseline_state is not None,
                "result has no exact severity-zero slice baseline",
            )
            require(
                baseline_state["unit_ids"] == unit_ids,
                "result baseline unit_ids differ",
            )
            for field in (
                "before_predictor_fingerprint",
                "before_input_fingerprint",
                "before_relation_fingerprint",
                "before_point_prediction_fingerprint",
                "before_calibration_checksum",
            ):
                after_field = field.replace("before_", "after_")
                require(
                    result[field] == baseline_state[after_field],
                    "result before-state differs from exact baseline",
                )
            require(
                result["after_input_fingerprint"] != result["before_input_fingerprint"]
                or after_predictor != before_predictor,
                "positive severity has no observable perturbation or refit",
            )

        if result["severity"] == 0:
            for before_field, after_field in (
                ("before_predictor_fingerprint", "after_predictor_fingerprint"),
                ("before_input_fingerprint", "after_input_fingerprint"),
                ("before_relation_fingerprint", "after_relation_fingerprint"),
                (
                    "before_point_prediction_fingerprint",
                    "after_point_prediction_fingerprint",
                ),
                ("before_calibration_checksum", "after_calibration_checksum"),
            ):
                require(
                    result[before_field] == result[after_field],
                    "severity zero is not identity",
                )
            require(
                result["predictor_status"] == "reused",
                "severity-zero predictor status drifted",
            )
            expected_calibration_status = (
                "reused" if base_checksum is not None else "absent"
            )
            require(
                result["calibration_status"] == expected_calibration_status,
                "severity-zero calibration status drifted",
            )
        elif scenario["mode"] == "clean_frozen":
            require(
                after_predictor == before_predictor, "clean_frozen changed predictor"
            )
            require(
                after_calibration == before_calibration,
                "clean_frozen changed calibration",
            )
            require(
                result["predictor_status"] == "reused",
                "clean_frozen predictor status drifted",
            )
            expected_calibration_status = (
                "reused" if base_checksum is not None else "absent"
            )
            require(
                result["calibration_status"] == expected_calibration_status,
                "clean_frozen calibration status drifted",
            )
            require(
                result["coverage_guarantee_status"]
                in {"diagnostic_only", "unavailable"},
                "clean_frozen shift overclaims coverage",
            )
        elif scenario["mode"] == "matched_recalibration":
            require(
                after_predictor == before_predictor,
                "matched_recalibration changed predictor",
            )
            require(
                result["predictor_status"] == "reused",
                "matched predictor status drifted",
            )
            require(
                result["calibration_status"] == "recalibrated"
                and after_calibration is not None
                and after_calibration != before_calibration,
                "matched_recalibration did not create a new calibrator",
            )
            require(
                artifact_map[after_calibration]["predictor_binding_fingerprint"]
                == after_predictor,
                "matched calibrator is bound to another predictor",
            )
        else:
            require(
                after_predictor != before_predictor,
                "structural_refit reused stale predictor",
            )
            require(
                result["predictor_status"] == "refit",
                "structural predictor status drifted",
            )
            if result["calibration_status"] == "recalibrated":
                require(
                    after_calibration is not None
                    and after_calibration != before_calibration
                    and artifact_map[after_calibration]["predictor_binding_fingerprint"]
                    == after_predictor,
                    "structural refit has no compatible new calibrator",
                )
                base_artifact = artifact_map[base_checksum]
                refit_artifact = artifact_map[after_calibration]
                base_binding = base_artifact["predictor_binding"]
                refit_binding = refit_artifact["predictor_binding"]
                for field in (
                    "campaign_fingerprint",
                    "controller_fingerprint",
                    "data_bindings",
                    "predictor_node_ids",
                    "target_processing_fingerprint",
                    "training_influence_fingerprint",
                ):
                    require(
                        refit_binding[field] == base_binding[field],
                        f"structural node replacement changed invariant predictor field {field}",
                    )
                require(
                    refit_artifact["training_influence"]
                    == base_artifact["training_influence"],
                    "structural node replacement changed training influence closure",
                )
                for field in (
                    "plan_id",
                    "graph_fingerprint",
                    "selected_variant_id",
                    "selected_variant_fingerprint",
                    "training_outcome_fingerprint",
                ):
                    require(
                        refit_binding[field] != base_binding[field],
                        f"structural node replacement did not change predictor field {field}",
                    )
            else:
                require(
                    result["calibration_status"] == "invalidated"
                    and after_calibration is None,
                    "structural refit calibration status is inconsistent",
                )
                require(
                    result["coverage_guarantee_status"] == "unavailable",
                    "invalidated structural calibration overclaims coverage",
                )
        if result["severity"] != 0 and result["calibration_status"] == "recalibrated":
            require(
                after_calibration is not None,
                "recalibrated result has no calibration artifact",
            )
            recalibration_artifact = artifact_map[after_calibration]
            diagnostics = recalibration_artifact["diagnostics"]
            require(
                diagnostics.get("scenario_id") == scenario_id
                and diagnostics.get("severity") == result["severity"],
                "recalibration diagnostics do not identify the exact scenario and severity",
            )
            require(
                diagnostics.get("calibration_input_fingerprint")
                == recalibration_artifact["calibration_cohort"]["content_fingerprint"],
                "recalibration diagnostics omit the exact calibration input fingerprint",
            )
        if after_calibration is not None:
            require(
                artifact_map[after_calibration]["predictor_binding_fingerprint"]
                == after_predictor,
                "result calibrator is bound to another predictor",
            )
        if result["coverage_guarantee_status"] in {
            "marginal_coverage",
            "joint_coverage",
        }:
            require(
                result["calibration_status"] in {"reused", "recalibrated"},
                "formal coverage guarantee has no valid calibrator",
            )
            require(
                result["slice"]["kind"] == "all",
                "sliced result overclaims formal coverage",
            )

    require(
        used_calibrators == set(artifact_checksums),
        "report calibration_artifacts are incomplete or contain unused artifacts",
    )
    require(
        used_prediction_blocks == set(block_fingerprints),
        "report conformal prediction blocks are incomplete or unused",
    )
    for scenario in scenarios:
        for severity in scenario["severities"]:
            require(
                any(
                    result["scenario_id"] == scenario["scenario_id"]
                    and result["severity"] == severity
                    and result["slice"] == {"kind": "all", "value": None}
                    for result in results
                ),
                f"scenario {scenario['scenario_id']} severity {severity} has no all slice",
            )
            expected_slices: list[dict[str, Any]] = []
            for dimension in scenario["slice_by"]:
                if dimension == "group":
                    expected_slices.extend(
                        {"kind": "group", "value": value}
                        for value in cohort["group_ids"]
                    )
                elif dimension == "source":
                    expected_slices.extend(
                        {"kind": "source", "value": value}
                        for value in cohort["source_ids"]
                    )
                elif dimension == "environment":
                    expected_slices.append(
                        {"kind": "environment", "value": scenario["environment_id"]}
                    )
                else:
                    require(
                        dimension == "target",
                        "scenario slice_by dimension is invalid",
                    )
                    expected_slices.extend(
                        {"kind": "target", "value": value}
                        for value in cohort["target_names"]
                    )
            for expected_slice in expected_slices:
                require(
                    any(
                        result["scenario_id"] == scenario["scenario_id"]
                        and result["severity"] == severity
                        and result["slice"] == expected_slice
                        for result in results
                    ),
                    f"scenario {scenario['scenario_id']} severity {severity} "
                    f"omits declared slice {expected_slice}",
                )

    for result in results:
        baseline = None
        if result["severity"] != 0:
            baseline_result = results_by_coordinate.get(
                w06_result_baseline_coordinate(result)
            )
            require(
                baseline_result is not None,
                "result has no exact severity-zero slice baseline",
            )
            require(
                baseline_result["unit_ids"] == result["unit_ids"],
                "result baseline unit_ids differ",
            )
            baseline = {
                record["metric"]: record for record in baseline_result["point_metrics"]
            }
        point_metrics = validate_w06_point_metrics(
            result["point_metrics"],
            scenario_map[result["scenario_id"]],
            severity_zero=result["severity"] == 0,
            baseline=baseline,
        )
        metric_id = result["conformal_metric_set_id"]
        metric_set = None
        result_metric_records: list[dict[str, Any]] = []
        if metric_id is None:
            require(
                result["conformal_prediction_block_fingerprint"] is None,
                "result without metric set still references a conformal prediction block",
            )
        else:
            require(metric_id in metric_map, "result references unknown metric set")
            used_metric_sets.add(metric_id)
            metric_set = metric_map[metric_id]
            result_metric_records = [
                record
                for record in metric_set["records"]
                if record["scenario_id"] == result["scenario_id"]
                and record["severity"] == result["severity"]
                and record["slice"] == result["slice"]
                and record["fold_id"] == result["fold_id"]
                and record["repeat_id"] == result["repeat_id"]
                and record["seed"] == result["seed"]
            ]
            require(result_metric_records, "result metric set has no matching record")
        produced_metrics = set(point_metrics)
        if any(
            record["empirical_coverage"] is not None for record in result_metric_records
        ):
            produced_metrics.add("conformal_coverage")
        if any(record["mean_width"] is not None for record in result_metric_records):
            produced_metrics.add("mean_width")
        unavailable_metrics = {
            error["code"].removeprefix("metric_unavailable.")
            for error in result["errors"]
            if error["phase"] == "score"
            and error["code"].startswith("metric_unavailable.")
        }
        require(
            produced_metrics.isdisjoint(unavailable_metrics),
            "result both produces and marks the same requested metric unavailable",
        )
        require(
            set(scenario_map[result["scenario_id"]]["metrics"])
            <= produced_metrics | unavailable_metrics,
            "result silently omits a requested metric",
        )
        for interval in result["confidence_intervals"]:
            require(
                all(
                    isinstance(interval[field], float)
                    and math.isfinite(interval[field])
                    for field in ("level", "lower", "upper")
                ),
                "confidence interval level and endpoints must be finite binary64 values",
            )
            require(
                interval["lower"] <= interval["upper"],
                "confidence interval is inverted",
            )
            if interval["metric_family"] == "point":
                require(
                    interval["metric"] in point_metrics,
                    "point CI references unknown metric",
                )
                require(
                    interval["coverage"] is None and interval["target_name"] is None,
                    "point CI must not carry conformal coordinates",
                )
            else:
                require(
                    metric_set is not None,
                    "conformal CI has no metric set",
                )
                require(
                    interval["metric"]
                    in {
                        "empirical_coverage",
                        "coverage_gap",
                        "mean_width",
                        "median_width",
                        "interval_score",
                    },
                    "conformal CI metric is not a conformal measurement",
                )
                require(
                    isinstance(interval["coverage"], float),
                    "conformal CI coverage must be represented as binary64",
                )
                matching_records = [
                    record
                    for record in metric_set["records"]
                    if record["scenario_id"] == result["scenario_id"]
                    and record["severity"] == result["severity"]
                    and record["slice"] == result["slice"]
                    and record["fold_id"] == result["fold_id"]
                    and record["repeat_id"] == result["repeat_id"]
                    and record["seed"] == result["seed"]
                    and record["coverage"] == interval["coverage"]
                    and record["target_name"] == interval["target_name"]
                ]
                require(
                    len(matching_records) == 1,
                    "conformal CI has no unique metric record coordinate",
                )
                require(
                    matching_records[0][interval["metric"]] is not None,
                    "conformal CI targets an unavailable measurement",
                )

    require(
        used_metric_sets == set(metric_ids),
        "report metric sets are incomplete or unused",
    )

    matched_metric_records: set[tuple[str, int]] = set()
    for metric_set in metric_sets:
        metric_id = metric_set["metric_set_id"]
        metric_block = block_map[metric_set["conformal_prediction_block_fingerprint"]]
        require(
            metric_set["point_prediction_fingerprint"]
            == metric_block["point_prediction_fingerprint"],
            "metric set point prediction differs from prediction block",
        )
        require(
            metric_set["predictor_binding_fingerprint"]
            == metric_block["predictor_binding_fingerprint"],
            "metric set predictor differs from prediction block",
        )
        require(
            metric_set["calibration_artifact_checksum"]
            == metric_block["calibration_artifact_checksum"],
            "metric set calibrator differs from prediction block",
        )
        require(
            metric_set["calibration_artifact_id"]
            == metric_block["calibration_artifact_id"],
            "metric set calibration id differs from prediction block",
        )
        require(
            metric_set["multi_target_policy"] == metric_block["multi_target_policy"],
            "metric set target policy differs from prediction block",
        )
        require(
            metric_set["unit_ids_fingerprint"]
            == dagml_tcv1_sha256(metric_block["unit_ids"]),
            "metric set unit identity differs from prediction block",
        )
        validate_w06_metric_width_closure(metric_set, metric_block)
        for record_index, record in enumerate(metric_set["records"]):
            scenario_id = record["scenario_id"]
            require(
                scenario_id in scenario_map, "metric record references unknown scenario"
            )
            scenario = scenario_map[scenario_id]
            require(
                record["severity"] in scenario["severities"],
                "metric severity is undeclared",
            )
            candidates = [
                result
                for result in results
                if result["conformal_metric_set_id"] == metric_id
                and result["scenario_id"] == scenario_id
                and result["severity"] == record["severity"]
                and result["slice"] == record["slice"]
                and result["fold_id"] == record["fold_id"]
                and result["repeat_id"] == record["repeat_id"]
                and result["seed"] == record["seed"]
            ]
            require(len(candidates) == 1, "metric record has no unique matching result")
            result = candidates[0]
            require(
                record["sample_count"] == result["unit_count"],
                "metric sample_count drifted",
            )
            require(
                record["unit_ids_fingerprint"] == dagml_tcv1_sha256(result["unit_ids"]),
                "metric unit_ids_fingerprint differs from result units",
            )
            require(
                record["guarantee_status"] == result["coverage_guarantee_status"],
                "metric guarantee differs from result",
            )
            if metric_set["multi_target_policy"] == "joint_max":
                require(
                    record["target_name"] is None, "joint metric target must be null"
                )
            else:
                require(
                    record["target_name"] in cohort["target_names"],
                    "marginal metric target is outside cohort targets",
                )
            require(
                metric_set["predictor_binding_fingerprint"]
                == result["after_predictor_fingerprint"],
                "metric set is bound to another predictor",
            )
            require(
                metric_set["calibration_artifact_checksum"]
                == result["after_calibration_checksum"],
                "metric set is bound to another calibrator",
            )
            require(
                metric_set["conformal_prediction_block_fingerprint"]
                == result["conformal_prediction_block_fingerprint"],
                "metric set is bound to another prediction block",
            )
            require(
                metric_set["point_prediction_fingerprint"]
                == result["after_point_prediction_fingerprint"],
                "metric set point prediction differs from result",
            )
            if scenario["mode"] == "clean_frozen" and record["severity"] > 0:
                require(
                    record["guarantee_status"] in {"diagnostic_only", "unavailable"},
                    "clean_frozen metric overclaims coverage",
                )
            matched_metric_records.add((metric_id, record_index))
    for result in results:
        metric_id = result["conformal_metric_set_id"]
        if metric_id is not None:
            require(metric_id in metric_map, "result references unknown metric set")
            require(
                any(
                    matched_id == metric_id
                    and metric_map[metric_id]["records"][index]["scenario_id"]
                    == result["scenario_id"]
                    and metric_map[metric_id]["records"][index]["severity"]
                    == result["severity"]
                    and metric_map[metric_id]["records"][index]["slice"]
                    == result["slice"]
                    and metric_map[metric_id]["records"][index]["fold_id"]
                    == result["fold_id"]
                    and metric_map[metric_id]["records"][index]["repeat_id"]
                    == result["repeat_id"]
                    and metric_map[metric_id]["records"][index]["seed"]
                    == result["seed"]
                    for matched_id, index in matched_metric_records
                ),
                "result metric set has no matching record",
            )

    require(
        matched_metric_records
        == {
            (metric_set["metric_set_id"], index)
            for metric_set in metric_sets
            for index, _record in enumerate(metric_set["records"])
        },
        "metric set contains an orphan record",
    )

    provenance = document.get("provenance")
    require(isinstance(provenance, dict), "report provenance must be an object")
    run_ids = provenance.get("run_ids")
    require(
        isinstance(run_ids, list) and run_ids, "provenance.run_ids must be non-empty"
    )
    require(run_ids == sorted(run_ids), "provenance.run_ids must be sorted")
    require(len(set(run_ids)) == len(run_ids), "provenance.run_ids must be unique")
    provenance_checksums = provenance.get("artifact_checksums")
    require(
        isinstance(provenance_checksums, list),
        "provenance.artifact_checksums must be an array",
    )
    require(
        provenance_checksums == sorted(provenance_checksums),
        "provenance.artifact_checksums must be sorted",
    )
    require(
        len(set(provenance_checksums)) == len(provenance_checksums),
        "provenance.artifact_checksums must be unique",
    )
    require(
        set(artifact_checksums) <= set(provenance_checksums),
        "provenance omits a calibration artifact checksum",
    )
    require(
        provenance.get("relation_fingerprint") == cohort["relation_fingerprint"],
        "report provenance relation differs from cohort",
    )
    validate_conformal_robustness_fingerprint(document, "report_fingerprint", "report")
    return document


def split_absolute_residual_multi_target_oracle(
    residuals: Any,
    coverages: Any,
    *,
    multi_target_policy: Any,
    small_sample_policy: Any,
) -> list[dict[str, Any]]:
    """Compute the W0.6 golden without importing the test-only Python oracle."""

    require(
        multi_target_policy in {"marginal", "joint_max"},
        "invalid multi-target policy",
    )
    require(
        small_sample_policy in {"error", "unbounded"},
        "invalid small-sample policy",
    )
    require(isinstance(residuals, list) and residuals, "residuals must be non-empty")
    rows: list[list[float]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(residuals):
        row = raw_row if isinstance(raw_row, list) else [raw_row]
        require(row, f"residuals[{row_index}] must be non-empty")
        if width is None:
            width = len(row)
        require(len(row) == width, "residual rows must have equal target width")
        normalized_row: list[float] = []
        for column_index, residual in enumerate(row):
            require(
                conformal_robustness_is_number(residual),
                f"residuals[{row_index}][{column_index}] must be numeric",
            )
            if isinstance(residual, int):
                require(
                    0 <= residual <= CONFORMAL_PORTABLE_INT_MAX,
                    f"residuals[{row_index}][{column_index}] integer is not portable",
                )
            number = float(residual)
            require(
                math.isfinite(number) and number >= 0.0,
                f"residuals[{row_index}][{column_index}] must be finite and non-negative",
            )
            normalized_row.append(number)
        rows.append(normalized_row)
    valid_coverages = validate_conformal_coverages(coverages)
    score_columns = (
        [tuple(max(row) for row in rows)]
        if multi_target_policy == "joint_max"
        else list(zip(*rows))
    )
    ordered_columns = [sorted(column) for column in score_columns]
    records: list[dict[str, Any]] = []
    for coverage in valid_coverages:
        rank = conformal_finite_sample_rank(len(rows), coverage)
        if rank > len(rows):
            if small_sample_policy == "error":
                raise ContractError(
                    f"finite-sample rank {rank} exceeds calibration size {len(rows)}"
                )
            values = [{"status": "unbounded"} for _column in ordered_columns]
        else:
            values = [
                {"status": "finite", "value": column[rank - 1]}
                for column in ordered_columns
            ]
        records.append({"coverage": coverage, "rank": rank, "values": values})
    return records


def regression_conformal_metrics_oracle(
    truth: Any,
    interval: Any,
    *,
    multi_target_policy: Any,
) -> list[dict[str, Any]]:
    """Reconstruct regression conformal metrics without the test-only oracle."""

    require(
        multi_target_policy in {"marginal", "joint_max"},
        "metric multi-target policy is invalid",
    )
    require(isinstance(interval, dict), "metric interval must be an object")
    coverage = validate_conformal_coverages([interval.get("coverage")])[0]
    lower = interval.get("lower")
    upper = interval.get("upper")
    require(
        isinstance(truth, list)
        and truth
        and isinstance(lower, list)
        and isinstance(upper, list)
        and len(truth) == len(lower) == len(upper),
        "metric truth and interval row counts differ",
    )
    target_count = len(truth[0]) if isinstance(truth[0], list) else 0
    require(target_count > 0, "metric truth rows must be non-empty")
    covered: list[list[bool]] = []
    widths: list[list[float | None]] = []
    scores: list[list[float | None]] = []
    alpha = 1.0 - coverage
    for row_index, (truth_row, lower_row, upper_row) in enumerate(
        zip(truth, lower, upper)
    ):
        require(
            isinstance(truth_row, list)
            and isinstance(lower_row, list)
            and isinstance(upper_row, list)
            and len(truth_row) == len(lower_row) == len(upper_row) == target_count,
            f"metric row {row_index} width differs",
        )
        covered_row: list[bool] = []
        width_row: list[float | None] = []
        score_row: list[float | None] = []
        for column_index, (raw_truth, lo, hi) in enumerate(
            zip(truth_row, lower_row, upper_row)
        ):
            require(
                conformal_robustness_is_number(raw_truth)
                and math.isfinite(float(raw_truth)),
                f"metric truth {row_index},{column_index} must be finite",
            )
            value = float(raw_truth)
            if lo is None or hi is None:
                require(
                    lo is None and hi is None,
                    "metric unbounded endpoints must be paired",
                )
                covered_row.append(True)
                width_row.append(None)
                score_row.append(None)
                continue
            require(
                conformal_robustness_is_number(lo)
                and conformal_robustness_is_number(hi)
                and math.isfinite(float(lo))
                and math.isfinite(float(hi))
                and lo <= hi,
                f"metric bounds {row_index},{column_index} are invalid",
            )
            cell_width = float(hi) - float(lo)
            cell_score = cell_width
            if value < float(lo):
                cell_score += (2.0 / alpha) * (float(lo) - value)
            elif value > float(hi):
                cell_score += (2.0 / alpha) * (value - float(hi))
            covered_row.append(float(lo) <= value <= float(hi))
            width_row.append(cell_width)
            score_row.append(cell_score)
        covered.append(covered_row)
        widths.append(width_row)
        scores.append(score_row)

    def summarize(
        coverage_values: list[bool],
        width_values: list[float | None],
        score_values: list[float | None],
        target_index: int | None,
    ) -> dict[str, Any]:
        empirical = sum(coverage_values) / len(coverage_values)
        if any(value is None for value in width_values + score_values):
            return {
                "target_index": target_index,
                "measurement_status": "unbounded",
                "empirical_coverage": empirical,
                "coverage_gap": empirical - coverage,
                "mean_width": None,
                "median_width": None,
                "interval_score": None,
            }
        finite_widths = sorted(
            float(value) for value in width_values if value is not None
        )
        finite_scores = [float(value) for value in score_values if value is not None]
        middle = len(finite_widths) // 2
        median_width = (
            finite_widths[middle]
            if len(finite_widths) % 2
            else (finite_widths[middle - 1] + finite_widths[middle]) / 2.0
        )
        return {
            "target_index": target_index,
            "measurement_status": "finite",
            "empirical_coverage": empirical,
            "coverage_gap": empirical - coverage,
            "mean_width": sum(finite_widths) / len(finite_widths),
            "median_width": median_width,
            "interval_score": sum(finite_scores) / len(finite_scores),
        }

    if multi_target_policy == "marginal":
        return [
            summarize(
                [row[target_index] for row in covered],
                [row[target_index] for row in widths],
                [row[target_index] for row in scores],
                target_index,
            )
            for target_index in range(target_count)
        ]
    return [
        summarize(
            [all(row) for row in covered],
            [value for row in widths for value in row],
            [value for row in scores for value in row],
            None,
        )
    ]


def validate_w06_numeric_evidence(
    evidence_records: Any,
    blocks: Any,
    metric_sets: Any,
    label: str,
) -> None:
    """Reconstruct test-only point, truth, interval and metric evidence exactly."""

    require(isinstance(evidence_records, list), f"{label} must be an array")
    require(isinstance(blocks, list), f"{label} blocks must be an array")
    require(isinstance(metric_sets, list), f"{label} metric sets must be an array")
    block_map = {block["block_fingerprint"]: block for block in blocks}
    metric_map = {metric_set["metric_set_id"]: metric_set for metric_set in metric_sets}
    require(
        len(block_map) == len(blocks),
        f"{label} cannot resolve duplicate prediction blocks",
    )
    require(
        len(metric_map) == len(metric_sets),
        f"{label} cannot resolve duplicate metric sets",
    )
    evidence_ids: list[str] = []
    evidence_blocks: list[str] = []
    evidence_metrics: list[str] = []
    expected_members = {
        "evidence_id",
        "block_fingerprint",
        "metric_set_id",
        "point_predictions",
        "point_prediction_fingerprint",
        "truth",
        "truth_fingerprint",
        "evidence_fingerprint",
    }
    reconstructed_fields = (
        "measurement_status",
        "empirical_coverage",
        "coverage_gap",
        "mean_width",
        "median_width",
        "interval_score",
    )
    for evidence_index, evidence in enumerate(evidence_records):
        evidence_label = f"{label}[{evidence_index}]"
        require(isinstance(evidence, dict), f"{evidence_label} must be an object")
        require(
            set(evidence) == expected_members,
            f"{evidence_label} must have the exact numeric evidence shape",
        )
        evidence_id = evidence.get("evidence_id")
        require_non_empty_string(evidence_id, f"{evidence_label}.evidence_id")
        block_fingerprint = evidence.get("block_fingerprint")
        metric_set_id = evidence.get("metric_set_id")
        require_non_empty_string(metric_set_id, f"{evidence_label}.metric_set_id")
        require(
            block_fingerprint in block_map,
            f"{evidence_label} references an unknown prediction block",
        )
        require(
            metric_set_id in metric_map,
            f"{evidence_label} references an unknown metric set",
        )
        block = block_map[block_fingerprint]
        metric_set = metric_map[metric_set_id]
        require(
            metric_set["conformal_prediction_block_fingerprint"] == block_fingerprint,
            f"{evidence_label} block/metric crosslink differs",
        )
        require(
            metric_set["predictor_binding_fingerprint"]
            == block["predictor_binding_fingerprint"],
            f"{evidence_label} block/metric predictor binding differs",
        )
        require(
            metric_set["calibration_artifact_id"] == block["calibration_artifact_id"]
            and metric_set["calibration_artifact_checksum"]
            == block["calibration_artifact_checksum"],
            f"{evidence_label} block/metric calibration binding differs",
        )
        require(
            metric_set["multi_target_policy"] == block["multi_target_policy"]
            and metric_set["method"] == block["method"],
            f"{evidence_label} block/metric conformal policy differs",
        )
        require(
            metric_set["unit_ids_fingerprint"] == dagml_tcv1_sha256(block["unit_ids"]),
            f"{evidence_label} block/metric unit binding differs",
        )
        point_predictions = evidence.get("point_predictions")
        truth = evidence.get("truth")
        require(
            isinstance(point_predictions, list)
            and isinstance(truth, list)
            and len(point_predictions) == len(truth) == len(block["unit_ids"]),
            f"{evidence_label} row count differs from its prediction block",
        )
        target_count = len(block["target_names"])
        for matrix_name, matrix in (
            ("point_predictions", point_predictions),
            ("truth", truth),
        ):
            for row_index, row in enumerate(matrix):
                require(
                    isinstance(row, list) and len(row) == target_count,
                    f"{evidence_label}.{matrix_name}[{row_index}] target width differs",
                )
                require(
                    all(
                        isinstance(value, float) and math.isfinite(value)
                        for value in row
                    ),
                    f"{evidence_label}.{matrix_name}[{row_index}] must contain finite binary64 values",
                )
        point_fingerprint = dagml_tcv1_sha256(point_predictions)
        truth_fingerprint = dagml_tcv1_sha256(truth)
        require(
            evidence.get("point_prediction_fingerprint")
            == point_fingerprint
            == block["point_prediction_fingerprint"]
            == metric_set["point_prediction_fingerprint"],
            f"{evidence_label} point prediction fingerprint differs",
        )
        require(
            evidence.get("truth_fingerprint")
            == truth_fingerprint
            == metric_set["truth_fingerprint"],
            f"{evidence_label} truth fingerprint differs",
        )
        require(
            evidence.get("evidence_fingerprint")
            == dagml_tcv1_sha256(
                {
                    key: value
                    for key, value in evidence.items()
                    if key != "evidence_fingerprint"
                }
            ),
            f"{evidence_label} evidence fingerprint differs",
        )

        expected_metrics: dict[tuple[float, str | None], dict[str, Any]] = {}
        for interval in block["intervals"]:
            for row_index, (point_row, lower_row, upper_row) in enumerate(
                zip(point_predictions, interval["lower"], interval["upper"])
            ):
                for column_index, (point, lower, upper) in enumerate(
                    zip(point_row, lower_row, upper_row)
                ):
                    if lower is None or upper is None:
                        require(
                            lower is None and upper is None,
                            f"{evidence_label} unbounded endpoints must be paired",
                        )
                    else:
                        require(
                            Decimal(repr(lower)) + Decimal(repr(upper))
                            == Decimal(2) * Decimal(repr(point)),
                            f"{evidence_label} interval midpoint does not reconstruct its point prediction at {row_index},{column_index}",
                        )
            summaries = regression_conformal_metrics_oracle(
                truth,
                interval,
                multi_target_policy=block["multi_target_policy"],
            )
            for summary in summaries:
                target_index = summary["target_index"]
                target_name = (
                    None
                    if target_index is None
                    else block["target_names"][target_index]
                )
                coordinate = (interval["coverage"], target_name)
                require(
                    coordinate not in expected_metrics,
                    f"{evidence_label} reconstructs a duplicate metric coordinate",
                )
                expected_metrics[coordinate] = summary
        actual_metrics = {
            (record["coverage"], record["target_name"]): record
            for record in metric_set["records"]
        }
        require(
            len(actual_metrics) == len(metric_set["records"])
            and set(actual_metrics) == set(expected_metrics),
            f"{evidence_label} metric coordinates do not reconstruct from the block",
        )
        for coordinate, expected in expected_metrics.items():
            actual = actual_metrics[coordinate]
            require(
                all(actual[field] == expected[field] for field in reconstructed_fields),
                f"{evidence_label} metrics do not reconstruct from truth and bounds at {coordinate}",
            )
        evidence_ids.append(evidence_id)
        evidence_blocks.append(block_fingerprint)
        evidence_metrics.append(metric_set_id)

    require(
        evidence_ids == sorted(evidence_ids)
        and len(set(evidence_ids)) == len(evidence_ids),
        f"{label} ids must be sorted and unique",
    )
    require(
        len(evidence_blocks) == len(set(evidence_blocks))
        and set(evidence_blocks) == set(block_map),
        f"{label} must provide exactly one record for every prediction block",
    )
    require(
        len(evidence_metrics) == len(set(evidence_metrics))
        and set(evidence_metrics) == set(metric_map),
        f"{label} must provide exactly one record for every metric set",
    )


def validate_conformal_metrics_golden(golden: Any) -> None:
    require(isinstance(golden, dict), "conformal metrics golden must be an object")
    require(
        golden.get("fixture_id") == "dag-ml.conformal.regression-metrics.v1",
        "conformal metrics fixture_id drifted",
    )
    require(golden.get("schema_version") == 1, "conformal metrics version drifted")
    require(
        golden.get("numeric_version") == "split_absolute_residual.metrics.v1",
        "conformal metrics numeric version drifted",
    )
    cases = golden.get("cases")
    require(isinstance(cases, list) and cases, "conformal metrics cases are missing")
    case_ids = [case.get("id") for case in cases]
    require(
        all(isinstance(case_id, str) and case_id for case_id in case_ids),
        "conformal metrics case id is invalid",
    )
    require(
        len(set(case_ids)) == len(case_ids),
        "conformal metrics case ids duplicate",
    )
    require(
        {case.get("multi_target_policy") for case in cases}
        == {"marginal", "joint_max"},
        "conformal metrics golden misses a multi-target policy",
    )
    for case in cases:
        actual = regression_conformal_metrics_oracle(
            case.get("truth"),
            case.get("interval"),
            multi_target_policy=case.get("multi_target_policy"),
        )
        require(
            actual == case.get("expected"),
            f"conformal metrics golden `{case['id']}` output drifted",
        )


def validate_conformal_robustness_golden(golden: Any) -> None:
    require(isinstance(golden, dict), "conformal golden must be an object")
    require(
        golden.get("fixture_id")
        == "dag-ml.conformal.oracle.split-absolute-residual.v1",
        "conformal golden fixture_id drifted",
    )
    require(golden.get("schema_version") == 1, "conformal golden version drifted")
    require(
        golden.get("numeric_version") == "split_absolute_residual.v1",
        "conformal numeric version drifted",
    )
    cases = golden.get("cases")
    require(isinstance(cases, list) and cases, "conformal golden cases are missing")
    case_ids = [case.get("id") for case in cases]
    require(
        all(isinstance(case_id, str) and case_id for case_id in case_ids),
        "conformal golden case id is invalid",
    )
    require(len(set(case_ids)) == len(case_ids), "conformal golden case ids duplicate")
    for case in cases:
        try:
            actual = split_absolute_residual_multi_target_oracle(
                case.get("residuals"),
                case.get("coverages"),
                multi_target_policy=case.get("multi_target_policy"),
                small_sample_policy=case.get("small_sample_policy"),
            )
        except ContractError as exc:
            expected_error = case.get("expected_error")
            require(
                isinstance(expected_error, str) and expected_error in str(exc),
                f"conformal golden `{case['id']}` failed unexpectedly: {exc}",
            )
        else:
            require(
                "expected_error" not in case and actual == case.get("expected"),
                f"conformal golden `{case['id']}` output drifted",
            )
    vectors = golden.get("tcv1_vectors")
    require(isinstance(vectors, list) and vectors, "conformal TCV1 vectors are missing")
    vector_ids = [vector.get("id") for vector in vectors]
    require(
        len(set(vector_ids)) == len(vector_ids), "conformal TCV1 vector ids duplicate"
    )
    require(
        "utf8_key_order_differs_from_utf16" in vector_ids,
        "conformal TCV1 vectors miss the UTF-8/UTF-16 discriminator",
    )
    for vector in vectors:
        preimage = dagml_tcv1_preimage(vector.get("value"))
        require(
            preimage.hex() == vector.get("expected_preimage_hex"),
            f"conformal TCV1 vector `{vector['id']}` preimage drifted",
        )
        require(
            hashlib.sha256(preimage).hexdigest() == vector.get("expected_sha256"),
            f"conformal TCV1 vector `{vector['id']}` digest drifted",
        )
        require(
            dagml_tcv1_preimage(vector.get("equivalent_value")) == preimage,
            f"conformal TCV1 vector `{vector['id']}` equivalence drifted",
        )


def parse_strict_json_text(document: Any, label: str) -> Any:
    require(isinstance(document, str), f"{label} document_json must be text")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, member in pairs:
            require(key not in value, f"{label} has a duplicate JSON object key")
            value[key] = member
        return value

    try:
        return json.loads(
            document,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is not strict JSON: {exc}") from exc


def restricted_jcs_bytes(value: Any, label: str = "restricted JCS value") -> bytes:
    """Render the narrow OrderedSearchSpaceSpec JCS domain independently."""

    if value is None:
        return b"null"
    if value is False:
        return b"false"
    if value is True:
        return b"true"
    if isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        except UnicodeEncodeError as exc:
            raise ContractError(f"{label} contains an unpaired surrogate") from exc
    if isinstance(value, int):
        require(
            0 <= value <= CONFORMAL_PORTABLE_INT_MAX,
            f"{label} structural integer is outside 0..2^53-1",
        )
        return str(value).encode("ascii")
    if isinstance(value, float):
        raise ContractError(
            f"{label} requires binary64-derived values to be tagged strings"
        )
    if isinstance(value, list):
        return (
            b"["
            + b",".join(
                restricted_jcs_bytes(member, f"{label}[{index}]")
                for index, member in enumerate(value)
            )
            + b"]"
        )
    if isinstance(value, dict):
        require(
            all(isinstance(key, str) for key in value),
            f"{label} object keys must be strings",
        )
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return (
            b"{"
            + b",".join(
                restricted_jcs_bytes(key, f"{label} key")
                + b":"
                + restricted_jcs_bytes(value[key], f"{label}.{key}")
                for key in keys
            )
            + b"}"
        )
    raise ContractError(f"{label} contains unsupported type {type(value).__name__}")


def validate_canonical_profile_golden(golden: Any) -> None:
    require(isinstance(golden, dict), "canonical profile golden must be an object")
    require(
        golden.get("fixture_id") == "dag-ml.tcv1-jcs-cross-language.v1",
        "canonical profile fixture_id drifted",
    )
    require(golden.get("schema_version") == 1, "canonical profile version drifted")
    profiles = golden.get("profiles")
    require(
        isinstance(profiles, dict) and set(profiles) == {"tcv1", "jcs"},
        "canonical profiles must distinguish exactly TCV1 and restricted JCS",
    )
    require(
        "UTF-8" in profiles["tcv1"]
        and "NFC" in profiles["tcv1"]
        and "UTF-16" in profiles["jcs"]
        and (
            "no NFC" in profiles["jcs"] or "no Unicode normalization" in profiles["jcs"]
        ),
        "canonical profile descriptions conflate TCV1 and restricted JCS",
    )

    tcv1_vectors = golden.get("tcv1_vectors")
    require(isinstance(tcv1_vectors, list) and tcv1_vectors, "TCV1 vectors missing")
    tcv1_ids = [vector.get("id") for vector in tcv1_vectors]
    require(len(set(tcv1_ids)) == len(tcv1_ids), "TCV1 vector ids duplicate")
    required_tcv1 = {
        "utf8_key_order_differs_from_utf16",
        "unicode_nfc",
        "negative_zero",
        "integer_two",
        "binary64_integral_two",
        "binary64_min_subnormal",
        "binary64_largest_subnormal",
        "binary64_min_normal",
        "binary64_two_pow_53",
        "binary64_max_finite",
    }
    require(required_tcv1 <= set(tcv1_ids), "TCV1 discriminator matrix is incomplete")
    for vector in tcv1_vectors:
        value = parse_strict_json_text(
            vector.get("document_json"), f"TCV1 vector `{vector['id']}`"
        )
        preimage = dagml_tcv1_preimage(value)
        require(
            preimage.hex() == vector.get("expected_preimage_hex"),
            f"TCV1 vector `{vector['id']}` canonical bytes drifted",
        )
        require(
            hashlib.sha256(preimage).hexdigest() == vector.get("expected_sha256"),
            f"TCV1 vector `{vector['id']}` digest drifted",
        )
        if "equivalent_json" in vector:
            equivalent = parse_strict_json_text(
                vector["equivalent_json"], f"TCV1 vector `{vector['id']}` equivalent"
            )
            require(
                dagml_tcv1_preimage(equivalent) == preimage,
                f"TCV1 vector `{vector['id']}` equivalence drifted",
            )
        if "binary64_be_hex" in vector:
            require(
                isinstance(value, float),
                f"TCV1 vector `{vector['id']}` lost float type",
            )
            require(
                struct.pack(">d", value).hex() == vector["binary64_be_hex"],
                f"TCV1 vector `{vector['id']}` binary64 bits drifted",
            )

    jcs_vectors = golden.get("restricted_jcs_vectors")
    require(
        isinstance(jcs_vectors, list) and jcs_vectors, "restricted JCS vectors missing"
    )
    jcs_ids = [vector.get("id") for vector in jcs_vectors]
    require(len(set(jcs_ids)) == len(jcs_ids), "restricted JCS vector ids duplicate")
    require(
        {
            "utf16_key_order_differs_from_utf8",
            "unicode_nfc_is_not_applied",
            "binary64_labels_are_strings",
        }
        <= set(jcs_ids),
        "restricted JCS discriminator matrix is incomplete",
    )
    for vector in jcs_vectors:
        value = parse_strict_json_text(
            vector.get("document_json"), f"restricted JCS vector `{vector['id']}`"
        )
        canonical = restricted_jcs_bytes(value)
        digest = hashlib.sha256(canonical).hexdigest()
        require(
            canonical.hex() == vector.get("expected_canonical_hex"),
            f"restricted JCS vector `{vector['id']}` canonical bytes drifted",
        )
        require(
            digest == vector.get("expected_sha256"),
            f"restricted JCS vector `{vector['id']}` digest drifted",
        )
        require(
            vector.get("expected_fingerprint") == f"sha256:{digest}",
            f"restricted JCS vector `{vector['id']}` fingerprint drifted",
        )
        if "equivalent_json" in vector:
            equivalent = parse_strict_json_text(
                vector["equivalent_json"],
                f"restricted JCS vector `{vector['id']}` equivalent",
            )
            require(
                restricted_jcs_bytes(equivalent) == canonical,
                f"restricted JCS vector `{vector['id']}` equivalence drifted",
            )
        if "non_equivalent_json" in vector:
            other = parse_strict_json_text(
                vector["non_equivalent_json"],
                f"restricted JCS vector `{vector['id']}` non-equivalent",
            )
            other_digest = hashlib.sha256(restricted_jcs_bytes(other)).hexdigest()
            require(
                other_digest == vector.get("non_equivalent_sha256")
                and other_digest != digest,
                f"restricted JCS vector `{vector['id']}` non-equivalence drifted",
            )

    invalid_vectors = golden.get("invalid_vectors")
    require(
        isinstance(invalid_vectors, list) and invalid_vectors, "invalid vectors missing"
    )
    invalid_ids = [vector.get("id") for vector in invalid_vectors]
    require(len(set(invalid_ids)) == len(invalid_ids), "invalid vector ids duplicate")
    for vector in invalid_vectors:
        try:
            value = parse_strict_json_text(
                vector.get("document_json"), f"invalid vector `{vector['id']}`"
            )
            if vector.get("profile") == "tcv1":
                dagml_tcv1_preimage(value)
            else:
                require(
                    vector.get("profile") == "jcs", "invalid vector profile drifted"
                )
                restricted_jcs_bytes(value)
        except ContractError:
            pass
        else:
            raise ContractError(
                f"invalid canonical vector `{vector['id']}` was accepted"
            )

    tcv1_order = next(
        vector
        for vector in tcv1_vectors
        if vector["id"] == "utf8_key_order_differs_from_utf16"
    )
    jcs_order = next(
        vector
        for vector in jcs_vectors
        if vector["id"] == "utf16_key_order_differs_from_utf8"
    )
    require(
        tcv1_order["expected_sha256"] != jcs_order["expected_sha256"],
        "TCV1 and restricted JCS profile discriminators collapsed",
    )


def apply_w06_json_pointer_mutation(
    value: Any, path: Any, replacement: Any, label: str
) -> Any:
    """Apply fixture mutations, including addition of an unknown final member."""

    require(isinstance(path, str) and path.startswith("/"), f"{label}.path is invalid")
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/")
    ]
    require(all(tokens), f"{label}.path contains an empty token")
    mutated = copy.deepcopy(value)
    cursor = mutated
    for token in tokens[:-1]:
        if isinstance(cursor, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise ContractError(
                    f"{label}.path token `{token}` is not an index"
                ) from exc
            require(0 <= index < len(cursor), f"{label}.path index is out of range")
            cursor = cursor[index]
        else:
            require(isinstance(cursor, dict), f"{label}.path crosses a scalar")
            require(token in cursor, f"{label}.path references missing `{token}`")
            cursor = cursor[token]
    final = tokens[-1]
    if isinstance(cursor, list):
        try:
            index = int(final)
        except ValueError as exc:
            raise ContractError(
                f"{label}.path token `{final}` is not an index"
            ) from exc
        require(0 <= index < len(cursor), f"{label}.path index is out of range")
        cursor[index] = copy.deepcopy(replacement)
    else:
        require(isinstance(cursor, dict), f"{label}.path cannot update a scalar")
        cursor[final] = copy.deepcopy(replacement)
    return mutated


def conformal_robustness_file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise ContractError(
            f"conformance pack references missing file: {path}"
        ) from exc


def validate_artifact_pack_schema_dependency_closure(
    paths: list[str], label: str
) -> None:
    """Require every local schema reached by an external $ref to be pinned."""

    try:
        missing = missing_schema_dependencies(ROOT, paths)
    except SchemaDependencyError as error:
        raise ContractError(
            f"{label} schema dependency resolution failed: {error}"
        ) from error
    require(
        not missing,
        f"{label} omits transitive schema dependencies: {sorted(missing)}",
    )


def validate_conformal_robustness_pack(pack: Any) -> None:
    require(isinstance(pack, dict), "conformal/robustness pack must be an object")
    require(
        pack.get("pack_id") == "dag-ml.conformal-robustness-conformance.v1",
        "conformal/robustness pack_id drifted",
    )
    require(
        pack.get("schema_version") == 1, "conformal/robustness pack version drifted"
    )
    require(
        pack.get("hash_algorithm") == "sha256-file-bytes",
        "conformal/robustness pack file hash profile drifted",
    )
    require(
        pack.get("fingerprint_profile") == "DAGML-TCV1",
        "conformal/robustness pack fingerprint profile drifted",
    )
    profiles = pack.get("canonical_profiles")
    require(
        isinstance(profiles, list)
        and [profile.get("id") for profile in profiles]
        == ["DAGML-TCV1", "RFC8785-JCS-restricted"],
        "conformance pack must keep TCV1 and restricted JCS profiles distinct",
    )
    require(
        profiles[0].get("object_key_order") == "utf8"
        and profiles[0].get("unicode_normalization") == "NFC"
        and profiles[1].get("object_key_order") == "utf16"
        and profiles[1].get("unicode_normalization") == "none",
        "conformance pack canonical profile semantics drifted",
    )

    artifacts = pack.get("artifacts")
    require(
        isinstance(artifacts, list) and artifacts, "pack artifacts must be non-empty"
    )
    paths = [entry.get("path") for entry in artifacts]
    require(
        all(isinstance(path, str) and path for path in paths),
        "pack artifact path is invalid",
    )
    require(paths == sorted(paths), "pack artifact paths must be sorted")
    require(len(set(paths)) == len(paths), "pack artifact paths must be unique")
    root = ROOT.resolve()
    for index, entry in enumerate(artifacts):
        label = f"pack.artifacts[{index}]"
        require(isinstance(entry, dict), f"{label} must be an object")
        require(set(entry) == {"path", "sha256", "kind"}, f"{label} fields drifted")
        raw_path = entry["path"]
        require("\\" not in raw_path, f"{label}.path must use portable separators")
        relative = Path(raw_path)
        require(not relative.is_absolute(), f"{label}.path must be relative")
        require(
            relative.as_posix() == raw_path
            and all(part not in {"", ".", ".."} for part in relative.parts),
            f"{label}.path is not canonical or contains traversal",
        )
        path_cursor = ROOT
        for part in relative.parts:
            path_cursor /= part
            require(
                not path_cursor.is_symlink(),
                f"{label}.path must not traverse or name a symbolic link",
            )
        resolved = (ROOT / relative).resolve()
        require(
            resolved.is_relative_to(root),
            f"{label}.path escapes the repository root",
        )
        require(resolved.is_file(), f"{label}.path does not name a regular file")
        require_non_empty_string(entry["kind"], f"{label}.kind")
        require_sha256(entry["sha256"], f"{label}.sha256")
        require(
            entry["sha256"] == entry["sha256"].lower(),
            f"{label}.sha256 must be lowercase",
        )
        require(
            entry["sha256"] == conformal_robustness_file_sha256(resolved),
            f"{label}.sha256 does not match file bytes",
        )

    validate_artifact_pack_schema_dependency_closure(paths, "conformal/robustness pack")

    required_paths = {
        *(
            schema_rel.as_posix()
            for schema_rel, _fixture_rel, _schema_id in CONFORMAL_ROBUSTNESS_CONTRACTS
        ),
        *(
            fixture_rel.as_posix()
            for _schema_rel, fixture_rel, _schema_id in CONFORMAL_ROBUSTNESS_CONTRACTS
        ),
        CONFORMAL_ROBUSTNESS_GOLDEN_REL.as_posix(),
        CONFORMAL_METRICS_GOLDEN_REL.as_posix(),
        "parity/conformal/generate_fixtures.py",
        "parity/conformal/oracle.py",
        "parity/conformal/tests/test_conformal_robustness_contracts.py",
        "parity/schema_dependencies.py",
        *(path.as_posix() for path in CANONICAL_ORACLE_ARTIFACT_RELS),
        *(path.as_posix() for path in ROBUSTNESS_RNG_ORACLE_ARTIFACT_RELS),
        "docs/contracts/conformal_calibration.schema.json",
        "docs/contracts/output_binding.schema.json",
        "docs/contracts/parameter_patch.schema.json",
        "docs/contracts/training_influence_manifest.schema.json",
        "examples/fixtures/conformal/split_absolute_residual_physical_sample.v1.json",
        "examples/fixtures/estimator/output_binding_regression_final_refit.v1.json",
        "examples/fixtures/estimator/parameter_patch_operator_alpha.v1.json",
    }
    missing = required_paths - set(paths)
    require(
        not missing, f"conformance pack misses transitive artifacts: {sorted(missing)}"
    )

    cases = pack.get("conformance_cases")
    require(
        isinstance(cases, list) and cases, "pack conformance cases must be non-empty"
    )
    case_ids = [case.get("id") for case in cases]
    require(
        all(isinstance(case_id, str) and case_id for case_id in case_ids),
        "pack conformance case id is invalid",
    )
    require(case_ids == sorted(case_ids), "pack conformance case ids must be sorted")
    require(
        len(set(case_ids)) == len(case_ids), "pack conformance case ids must be unique"
    )
    require(
        "cross_language_tcv1_restricted_jcs" in case_ids,
        "pack misses the cross-language canonical-profile case",
    )
    require(
        "regression_conformal_metric_reconstruction" in case_ids,
        "pack misses exact regression conformal metric reconstruction",
    )
    require(
        "robustness_philox_counter_profile" in case_ids,
        "pack misses the frozen robustness RNG profile",
    )
    for index, case in enumerate(cases):
        label = f"pack.conformance_cases[{index}]"
        require(
            isinstance(case.get("fixture"), str) and case["fixture"] in set(paths),
            f"{label}.fixture is not a pinned artifact",
        )
        invariants = case.get("invariants")
        require(
            isinstance(invariants, list) and invariants, f"{label}.invariants missing"
        )
        require(
            invariants == sorted(invariants)
            and len(set(invariants)) == len(invariants),
            f"{label}.invariants must be sorted and unique",
        )
    require_sha256(pack.get("pack_checksum"), "pack.pack_checksum")
    expected_checksum = dagml_tcv1_sha256(
        {key: value for key, value in pack.items() if key != "pack_checksum"}
    )
    require(
        pack["pack_checksum"] == expected_checksum,
        "conformal/robustness pack checksum does not match TCV1 content",
    )


def conformal_robustness_semantic_validator(
    schema_name: str,
) -> Any:
    validators = {
        "conformal_calibration.schema.json": lambda document: (
            validate_conformal_calibration_artifact(
                document, "W0.6 calibration artifact"
            )
        ),
        "cohort_manifest.schema.json": validate_w06_cohort_manifest,
        "conformal_prediction_block.schema.json": validate_w06_prediction_block,
        "conformal_metric_set.schema.json": validate_w06_metric_set,
        "domain_assessment_block.schema.json": validate_w06_domain_assessment,
        "decision_block.schema.json": validate_w06_decision_block,
        "robustness_scenario_spec.schema.json": validate_w06_scenario,
        "robustness_report.schema.json": validate_w06_report,
    }
    require(
        schema_name in validators, f"no W0.6 semantic validator for `{schema_name}`"
    )
    return validators[schema_name]


def validate_conformal_robustness_snapshot(
    registry: Registry,
    schemas_by_id: dict[str, dict[str, Any]],
    fixtures: dict[str, Any],
    conformal_golden: Any,
    conformal_metrics_golden: Any,
    canonical_golden: Any,
    pack: Any,
) -> None:
    """Validate W0.6 schemas, fixtures, goldens and both independent oracles."""

    for schema_rel, fixture_rel, expected_id in CONFORMAL_ROBUSTNESS_CONTRACTS:
        schema_name = schema_rel.name
        require(
            expected_id in schemas_by_id,
            f"W0.6 schema `{schema_name}` is not registered",
        )
        schema = schemas_by_id[expected_id]
        require(
            schema.get("$id") == expected_id, f"W0.6 schema `{schema_name}` $id drifted"
        )
        require(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            f"W0.6 schema `{schema_name}` must declare Draft 2020-12",
        )
        fixture = fixtures.get(fixture_rel.as_posix())
        require(isinstance(fixture, dict), f"W0.6 fixture `{fixture_rel}` is missing")
        require(fixture.get("schema_version") == 1, f"{fixture_rel} version drifted")
        require(
            fixture.get("schema") == schema_name, f"{fixture_rel} schema link drifted"
        )
        require_non_empty_string(fixture.get("fixture_id"), f"{fixture_rel}.fixture_id")
        positive_cases = fixture.get("valid_cases")
        negative_cases = fixture.get("invalid_cases")
        require(
            isinstance(positive_cases, list) and positive_cases,
            f"{fixture_rel}.valid_cases must be non-empty",
        )
        require(
            isinstance(negative_cases, list) and negative_cases,
            f"{fixture_rel}.invalid_cases must be non-empty",
        )
        positive_ids = [case.get("id") for case in positive_cases]
        negative_ids = [case.get("id") for case in negative_cases]
        require(
            len(set(positive_ids)) == len(positive_ids),
            f"{fixture_rel} positive ids duplicate",
        )
        require(
            len(set(negative_ids)) == len(negative_ids),
            f"{fixture_rel} negative ids duplicate",
        )
        positive_by_id = {case["id"]: case["document"] for case in positive_cases}
        semantic = conformal_robustness_semantic_validator(schema_name)
        validator = Draft202012Validator(schema, registry=registry)
        for case in positive_cases:
            label = f"{fixture_rel} valid case `{case['id']}`"
            validate_draft_2020_instance(case["document"], schema, registry, label)
            semantic(case["document"])
        for case in negative_cases:
            label = f"{fixture_rel} invalid case `{case.get('id')}`"
            require(
                case.get("base_case") in positive_by_id, f"{label} base_case is unknown"
            )
            mutated = positive_by_id[case["base_case"]]
            mutations = case.get("mutations")
            require(
                isinstance(mutations, list) and mutations, f"{label} mutations missing"
            )
            for mutation_index, mutation in enumerate(mutations):
                mutated = apply_w06_json_pointer_mutation(
                    mutated,
                    mutation.get("path"),
                    mutation.get("value"),
                    f"{label}.mutations[{mutation_index}]",
                )
            fingerprint_field = {
                "conformal_calibration.schema.json": "checksum",
                "cohort_manifest.schema.json": "manifest_fingerprint",
                "conformal_prediction_block.schema.json": "block_fingerprint",
                "conformal_metric_set.schema.json": "metric_set_fingerprint",
                "domain_assessment_block.schema.json": "block_fingerprint",
                "decision_block.schema.json": "block_fingerprint",
                "robustness_scenario_spec.schema.json": "scenario_fingerprint",
                "robustness_report.schema.json": "report_fingerprint",
            }[schema_name]
            recompute_fingerprints = case.get("recompute_fingerprints") is True
            targets_fingerprint = case.get("targets_fingerprint") is True
            require(
                recompute_fingerprints != targets_fingerprint,
                f"{label} must either recompute or explicitly target its fingerprint",
            )
            expected_fingerprint = dagml_tcv1_sha256(
                {
                    key: value
                    for key, value in mutated.items()
                    if key != fingerprint_field
                }
            )
            if recompute_fingerprints:
                mutated[fingerprint_field] = dagml_tcv1_sha256(
                    {
                        key: value
                        for key, value in mutated.items()
                        if key != fingerprint_field
                    }
                )
                require(
                    mutated[fingerprint_field] == expected_fingerprint,
                    f"{label} did not preserve its self fingerprint",
                )
            else:
                require(
                    mutated[fingerprint_field] != expected_fingerprint,
                    f"{label} declares a fingerprint target without breaking identity",
                )
            schema_errors = list(validator.iter_errors(mutated))
            semantic_error = ""
            try:
                semantic(mutated)
            except (ContractError, KeyError, IndexError, TypeError) as exc:
                semantic_error = str(exc)
            require(schema_errors or semantic_error, f"{label} unexpectedly passed")
            combined = "\n".join(error.message for error in schema_errors)
            combined = f"{combined}\n{semantic_error}".lower()
            expected_error = case.get("expected_error")
            require_non_empty_string(expected_error, f"{label}.expected_error")
            require(
                expected_error.lower() in combined,
                f"{label} failed for the wrong cause; expected `{expected_error}`, got `{combined}`",
            )

    cohort_fixture = fixtures[
        "examples/fixtures/conformal/cohort_manifest_roles.v1.json"
    ]
    cohorts = {case["id"]: case["document"] for case in cohort_fixture["valid_cases"]}
    require(
        set(cohorts) == {"development", "calibration", "external_test", "production"},
        "cohort role fixture does not cover the four canonical roles",
    )
    for case in cohort_fixture.get("disjointness_cases", []):
        try:
            assert_w06_calibration_disjoint(
                case.get("training_sample_ids"),
                case.get("training_origin_sample_ids"),
                cohorts[case["calibration_case"]],
            )
        except ContractError as exc:
            require(
                case.get("expected") != "valid"
                and case.get("expected_error") in str(exc),
                f"cohort disjointness case `{case.get('id')}` failed unexpectedly: {exc}",
            )
        else:
            require(
                case.get("expected") == "valid",
                f"cohort disjointness case `{case.get('id')}` unexpectedly passed",
            )

    report_fixture = fixtures["examples/fixtures/robustness/robustness_reports.v1.json"]
    reports = {case["id"]: case["document"] for case in report_fixture["valid_cases"]}
    require(
        set(reports)
        == {
            "three_modes_resolved_conformal",
            "structural_calibration_invalidated",
            "production_point_only",
        },
        "robustness report fixture case matrix drifted",
    )
    report = reports["three_modes_resolved_conformal"]
    scenario_documents = [
        case["document"]
        for case in fixtures[
            "examples/fixtures/robustness/robustness_scenarios.v1.json"
        ]["valid_cases"]
    ]
    require(
        report["cohort_manifest"] == cohorts["external_test"],
        "report cohort copy drifted",
    )
    require(report["scenarios"] == scenario_documents, "report scenario copies drifted")
    require(
        len(report["calibration_artifacts"]) == 3
        and len(report["conformal_prediction_blocks"]) == 18
        and len(report["conformal_metric_sets"]) == 18
        and len(report["results"]) == 18,
        "resolved robustness report must keep the exact 3/18/18/18 evidence matrix",
    )
    production_report = reports["production_point_only"]
    require(
        production_report["calibration_artifact_checksum"] is None
        and production_report["calibration_artifacts"] == []
        and production_report["conformal_prediction_blocks"] == []
        and production_report["conformal_metric_sets"] == [],
        "production point-only report acquired conformal evidence",
    )

    prediction_fixture = fixtures[
        "examples/fixtures/conformal/conformal_prediction_blocks.v1.json"
    ]
    metric_fixture = fixtures[
        "examples/fixtures/conformal/conformal_metric_sets.v1.json"
    ]
    validate_w06_numeric_evidence(
        metric_fixture.get("evidence_cases"),
        [case["document"] for case in prediction_fixture["valid_cases"]],
        [case["document"] for case in metric_fixture["valid_cases"]],
        "standalone conformal numeric evidence",
    )

    evidence_sets = report_fixture.get("evidence_sets")
    require(
        isinstance(evidence_sets, list),
        "robustness report evidence_sets must be an array",
    )
    evidence_by_report = {
        evidence_set.get("report_case"): evidence_set.get("records")
        for evidence_set in evidence_sets
        if isinstance(evidence_set, dict)
    }
    expected_evidence_reports = {
        case_id
        for case_id, candidate in reports.items()
        if candidate["conformal_prediction_blocks"]
    }
    require(
        len(evidence_by_report) == len(evidence_sets)
        and set(evidence_by_report) == expected_evidence_reports,
        "report evidence_sets must cover exactly the reports with conformal blocks",
    )
    for report_case in sorted(expected_evidence_reports):
        candidate = reports[report_case]
        validate_w06_numeric_evidence(
            evidence_by_report[report_case],
            candidate["conformal_prediction_blocks"],
            candidate["conformal_metric_sets"],
            f"report numeric evidence `{report_case}`",
        )

    invalid_evidence_cases = report_fixture.get("invalid_evidence_cases")
    require(
        isinstance(invalid_evidence_cases, list) and invalid_evidence_cases,
        "report invalid_evidence_cases must be non-empty",
    )
    for invalid_case in invalid_evidence_cases:
        invalid_label = f"invalid numeric evidence `{invalid_case.get('id')}`"
        report_case = invalid_case.get("report_case")
        require(report_case in evidence_by_report, f"{invalid_label} report is unknown")
        mutated_report = copy.deepcopy(reports[report_case])
        mutated_evidence = copy.deepcopy(evidence_by_report[report_case])
        report_mutations = invalid_case.get("report_mutations", [])
        require(
            isinstance(report_mutations, list),
            f"{invalid_label} report_mutations must be an array",
        )
        for mutation_index, mutation in enumerate(report_mutations):
            mutated_report = apply_w06_json_pointer_mutation(
                mutated_report,
                mutation.get("path"),
                mutation.get("value"),
                f"{invalid_label}.report_mutations[{mutation_index}]",
            )
        base_evidence_id = invalid_case.get("base_evidence_id")
        evidence_index = next(
            (
                index
                for index, evidence in enumerate(mutated_evidence)
                if evidence.get("evidence_id") == base_evidence_id
            ),
            None,
        )
        require(evidence_index is not None, f"{invalid_label} base evidence is unknown")
        evidence = mutated_evidence[evidence_index]
        mutations = invalid_case.get("mutations")
        require(
            isinstance(mutations, list) and mutations,
            f"{invalid_label} mutations are missing",
        )
        for mutation_index, mutation in enumerate(mutations):
            evidence = apply_w06_json_pointer_mutation(
                evidence,
                mutation.get("path"),
                mutation.get("value"),
                f"{invalid_label}.mutations[{mutation_index}]",
            )
        mutated_evidence[evidence_index] = evidence
        if invalid_case.get("rebind_metric_truth", False):
            metric_set = next(
                metric_set
                for metric_set in mutated_report["conformal_metric_sets"]
                if metric_set["metric_set_id"] == evidence["metric_set_id"]
            )
            metric_set["truth_fingerprint"] = evidence["truth_fingerprint"]
            metric_set["metric_set_fingerprint"] = dagml_tcv1_sha256(
                {
                    key: value
                    for key, value in metric_set.items()
                    if key != "metric_set_fingerprint"
                }
            )
            mutated_report["report_fingerprint"] = dagml_tcv1_sha256(
                {
                    key: value
                    for key, value in mutated_report.items()
                    if key != "report_fingerprint"
                }
            )
        validate_w06_report(mutated_report)
        try:
            validate_w06_numeric_evidence(
                mutated_evidence,
                mutated_report["conformal_prediction_blocks"],
                mutated_report["conformal_metric_sets"],
                invalid_label,
            )
        except ContractError as exc:
            expected_error = invalid_case.get("expected_error")
            require_non_empty_string(expected_error, f"{invalid_label}.expected_error")
            require(
                expected_error.lower() in str(exc).lower(),
                f"{invalid_label} failed for the wrong cause: {exc}",
            )
        else:
            raise ContractError(f"{invalid_label} unexpectedly passed")

    validate_conformal_robustness_golden(conformal_golden)
    validate_conformal_metrics_golden(conformal_metrics_golden)
    validate_canonical_profile_golden(canonical_golden)
    validate_conformal_robustness_pack(pack)


W10_INFLUENCE_ORDER = {
    "transform_fit": 0,
    "model_fit": 1,
    "hpo_selection": 2,
    "early_stopping": 3,
    "weighting_resampling": 4,
    "trained_meta_aggregation": 5,
}
W10_NAMESPACE_ORDER = {
    "operator": 0,
    "fit": 1,
    "control": 2,
    "structural": 3,
}
W10_CAPABILITY_ORDER = {
    capability: index
    for index, capability in enumerate(
        (
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
            "uses_training_weights",
            "uses_early_stopping",
            "performs_internal_tuning",
            "trains_aggregation",
        )
    )
}
W10_PHASE_ORDER = {
    phase: index
    for index, phase in enumerate(
        ("COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN")
    )
}


def w10_fingerprint_without(document: dict[str, Any], field: str) -> str:
    return dagml_tcv1_sha256(
        {key: value for key, value in document.items() if key != field}
    )


def validate_w10_data_identity(identity: Any, label: str) -> dict[str, Any]:
    require(isinstance(identity, dict), f"{label} must be an object")
    require_no_unknown_keys(
        identity,
        {
            "requirement_key",
            "schema_fingerprint",
            "plan_fingerprint",
            "relation_fingerprint",
            "data_content_fingerprint",
            "target_content_fingerprint",
            "identity_fingerprint",
        },
        label,
    )
    require_non_empty_string(
        identity.get("requirement_key"), f"{label}.requirement_key"
    )
    for field in (
        "schema_fingerprint",
        "plan_fingerprint",
        "relation_fingerprint",
        "data_content_fingerprint",
        "target_content_fingerprint",
        "identity_fingerprint",
    ):
        require_sha256(identity.get(field), f"{label}.{field}")
    require(
        identity["identity_fingerprint"]
        == w10_fingerprint_without(identity, "identity_fingerprint"),
        f"{label}.identity_fingerprint mismatch",
    )
    return identity


def w10_predictor_closure(graph: dict[str, Any], roots: list[str]) -> list[str]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for edge in graph.get("edges", []):
        incoming[edge["target"]["node_id"]].append(edge["source"]["node_id"])
    pending = list(roots)
    closure: set[str] = set()
    while pending:
        node_id = pending.pop()
        require(node_id in nodes, f"W1 training closure references `{node_id}`")
        if node_id in closure:
            continue
        closure.add(node_id)
        pending.extend(incoming[node_id])
    return sorted(closure)


def w10_selector_matches(selector: dict[str, Any], operator: Any) -> bool:
    if not isinstance(operator, (str, dict)):
        return False
    if isinstance(operator, str):
        descriptors: dict[str, list[str]] = {"aliases": [operator]}
    else:
        descriptors = {
            "aliases": [
                value
                for field in ("type", "ref", "class", "function")
                if isinstance((value := operator.get(field)), str)
            ],
            "types": [operator.get("type")],
            "refs": [operator.get("ref")],
            "classes": [operator.get("class")],
            "functions": [operator.get("function")],
        }
    for field in ("aliases", "types", "refs", "classes", "functions"):
        expected = {str(value).strip().lower() for value in selector.get(field, [])}
        actual = {
            str(value).strip().lower()
            for value in descriptors.get(field, [])
            if isinstance(value, str)
        }
        if expected & actual:
            return True
    operator_class = operator.get("class") if isinstance(operator, dict) else None
    return isinstance(operator_class, str) and any(
        operator_class.strip().lower().startswith(str(prefix).strip().lower())
        for prefix in selector.get("class_prefixes", [])
    )


def w10_controller_for_node(
    request: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any]:
    explicit = node.get("metadata", {}).get("controller_id")
    if explicit is not None:
        candidates = [
            (0, manifest)
            for manifest in request["controller_manifests"]
            if manifest["controller_id"] == explicit
            and manifest["operator_kind"] == node["kind"]
        ]
    else:
        candidates = []
        for manifest in request["controller_manifests"]:
            if manifest["operator_kind"] != node["kind"]:
                continue
            selectors = manifest.get("operator_selectors", [])
            if selectors:
                if not any(
                    w10_selector_matches(selector, node.get("operator"))
                    for selector in selectors
                ):
                    continue
                rank = 0
            else:
                rank = 1
            candidates.append((rank, manifest))
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].get("priority", 0),
            item[1]["controller_id"],
        )
    )
    require(candidates, f"W1 training node `{node['id']}` has no generic controller")
    if len(candidates) > 1:
        require(
            (candidates[0][0], candidates[0][1].get("priority", 0))
            != (candidates[1][0], candidates[1][1].get("priority", 0)),
            f"W1 training node `{node['id']}` has ambiguous controllers",
        )
    return candidates[0][1]


def validate_w10_parameter_patches(request: dict[str, Any], label: str) -> None:
    patches = request["parameter_patches"]
    policies = request["patch_policies"]
    graph_nodes = {node["id"]: node for node in request["graph"]["nodes"]}
    patch_keys: list[tuple[str, int, tuple[str, ...]]] = []
    for index, patch in enumerate(patches):
        path = patch["path"]
        require(
            isinstance(path, list)
            and path
            and all(
                isinstance(segment, str) and bool(segment.strip()) and segment != "-"
                for segment in path
            ),
            f"{label}.parameter_patches[{index}].path is invalid",
        )
        require(
            patch["node_id"] in graph_nodes,
            f"{label}.parameter patch references unknown node",
        )
        patch_keys.append(
            (
                patch["node_id"],
                W10_NAMESPACE_ORDER[patch["namespace"]],
                tuple(path),
            )
        )
    require(
        patch_keys == sorted(patch_keys) and len(patch_keys) == len(set(patch_keys)),
        f"{label}.parameter patches must be sorted and unique",
    )
    for left, right in zip(patch_keys, patch_keys[1:]):
        if left[:2] != right[:2]:
            continue
        shorter, longer = sorted((left[2], right[2]), key=len)
        require(
            longer[: len(shorter)] != shorter,
            f"{label}.parameter patches contain a parent/child conflict",
        )
    policy_ids = [policy["node_id"] for policy in policies]
    require(
        policy_ids == sorted(set(policy_ids)),
        f"{label}.patch policies must be sorted and unique",
    )
    require(
        policy_ids == sorted({patch["node_id"] for patch in patches}),
        f"{label}.patch policies must exactly cover patched nodes",
    )
    allowed = {
        policy["node_id"]: set(policy["allowed_namespaces"]) for policy in policies
    }
    for policy in policies:
        namespaces = policy["allowed_namespaces"]
        require(
            namespaces == sorted(set(namespaces), key=W10_NAMESPACE_ORDER.__getitem__),
            f"{label}.allowed_namespaces are not in canonical BTreeSet wire order",
        )
        require(
            policy["node_id"] in graph_nodes,
            f"{label}.patch policy references unknown node",
        )
    roots = {
        node_id: {
            "operator": copy.deepcopy(node.get("params", {})),
            "fit": {},
            "control": {},
            "structural": {},
        }
        for node_id, node in graph_nodes.items()
    }
    for patch in patches:
        require(
            patch["namespace"] in allowed[patch["node_id"]],
            f"{label}.parameter namespace is forbidden by patch policy",
        )
        cursor = roots[patch["node_id"]][patch["namespace"]]
        for segment in patch["path"][:-1]:
            require(
                segment in cursor and isinstance(cursor[segment], dict),
                f"{label}.parameter patch has missing/non-object intermediate path",
            )
            cursor = cursor[segment]
        cursor[patch["path"][-1]] = copy.deepcopy(patch["value"])


def w10_base_influence_kind(
    request: dict[str, Any], node_id: str, manifest: dict[str, Any]
) -> str | None:
    if manifest["fit_scope"] in {"stateless", "inference_only"}:
        return None
    oof_consumers = {
        edge["target"]["node_id"]
        for edge in request["graph"].get("edges", [])
        if edge["contract"].get("requires_oof") is True
    }
    if node_id in oof_consumers or "trains_aggregation" in manifest["capabilities"]:
        return "trained_meta_aggregation"
    node = next(node for node in request["graph"]["nodes"] if node["id"] == node_id)
    if node["kind"] == "model":
        return "model_fit"
    if node["kind"] == "tuner":
        return "hpo_selection"
    return "transform_fit"


def validate_w10_influence_requirements(
    request: dict[str, Any], closure: list[str], label: str
) -> None:
    fold_set = request["campaign"]["split_invocation"]["fold_set"]
    all_samples = set(fold_set["sample_ids"])
    folds = {fold["fold_id"]: fold for fold in fold_set["folds"]}
    nodes = {node["id"]: node for node in request["graph"]["nodes"]}
    expected: dict[tuple[str, str, str, str | None], set[str]] = {}
    capability_kinds = {
        "performs_internal_tuning": "hpo_selection",
        "uses_early_stopping": "early_stopping",
        "uses_training_weights": "weighting_resampling",
    }
    for node_id in closure:
        manifest = w10_controller_for_node(request, nodes[node_id])
        base_kind = w10_base_influence_kind(request, node_id, manifest)
        if base_kind is None:
            continue
        kinds = {
            kind
            for capability, kind in capability_kinds.items()
            if capability in manifest["capabilities"]
        }
        if base_kind == "hpo_selection":
            kinds.discard("hpo_selection")
        if "FIT_CV" in manifest["supported_phases"]:
            for fold in folds.values():
                for kind in kinds:
                    expected[(node_id, kind, "FIT_CV", fold["fold_id"])] = set(
                        fold["train_sample_ids"]
                    )
        if request["options"]["refit"] and "REFIT" in manifest["supported_phases"]:
            for kind in kinds:
                expected[(node_id, kind, "REFIT", None)] = set(all_samples)
    previous: tuple[int, str, str] | None = None
    actual: set[tuple[str, str, str, str | None]] = set()
    for index, requirement in enumerate(request["influence_requirements"]):
        item_label = f"{label}.influence_requirements[{index}]"
        order_key = (
            W10_INFLUENCE_ORDER[requirement["kind"]],
            requirement["scope_id"],
            requirement["node_id"],
        )
        require(
            previous is None or previous < order_key,
            f"{label}.influence requirements are not canonically sorted",
        )
        previous = order_key
        physical_sample_ids = requirement["physical_sample_ids"]
        require(
            physical_sample_ids == sorted(set(physical_sample_ids))
            and bool(physical_sample_ids),
            f"{item_label}.physical_sample_ids must be sorted and unique",
        )
        slot = (
            requirement["node_id"],
            requirement["kind"],
            requirement["phase"],
            requirement["fold_id"],
        )
        require(
            slot in expected,
            f"{item_label} is not required by active controller capabilities",
        )
        samples = set(physical_sample_ids)
        eligible = expected[slot]
        if not samples <= eligible:
            fold = folds.get(requirement["fold_id"])
            if fold and samples & set(fold["validation_sample_ids"]):
                raise ContractError(f"{item_label} leaks outer validation samples")
            raise ContractError(
                f"{item_label} uses samples outside its training cohort"
            )
        if requirement["kind"] == "weighting_resampling":
            require(
                samples == eligible,
                f"{item_label} must cover its complete fit cohort",
            )
        if requirement["kind"] == "early_stopping":
            require(
                len(samples) < len(eligible),
                f"{item_label} must be a strict training-cohort subset",
            )
        require(slot not in actual, f"{item_label} duplicates a capability slot")
        actual.add(slot)
    require(
        actual == set(expected),
        f"{label}.influence requirements do not exactly cover active capability scopes",
    )


def validate_w10_manifest_wire(manifest: dict[str, Any], label: str) -> None:
    phases = manifest["supported_phases"]
    capabilities = manifest.get("capabilities", [])
    require(
        phases == sorted(set(phases), key=W10_PHASE_ORDER.__getitem__),
        f"{label}.supported_phases are not in canonical BTreeSet wire order",
    )
    require(
        capabilities == sorted(set(capabilities), key=W10_CAPABILITY_ORDER.__getitem__),
        f"{label}.capabilities are not in canonical BTreeSet wire order",
    )
    for selector_index, selector in enumerate(manifest.get("operator_selectors", [])):
        for field, values in selector.items():
            require(
                values == sorted(set(values)),
                f"{label}.operator_selectors[{selector_index}].{field} is not canonical",
            )


def w10_resolve_output_port(
    output: dict[str, Any], graph: dict[str, Any], label: str
) -> str:
    node = next(
        (node for node in graph["nodes"] if node["id"] == output["node_id"]),
        None,
    )
    require(node is not None, f"{label} references an unknown output node")
    prediction_ports = [
        port["name"]
        for port in node["ports"]["outputs"]
        if port["kind"] == "prediction"
    ]
    port_name = output.get("port_name")
    if port_name is None:
        require(
            len(prediction_ports) == 1,
            f"{label} output port is ambiguous or absent",
        )
        return prediction_ports[0]
    require(
        port_name in prediction_ports,
        f"{label}.port_name is not a prediction output",
    )
    return port_name


def validate_w10_training_request(request: Any, label: str) -> None:
    require(isinstance(request, dict), f"{label} must be an object")
    _normalize_training_request(request)  # fail closed on serde container types
    require_no_unknown_keys(
        request["options"],
        {
            "refit",
            "refit_strategy",
            "seed",
            "selection",
            "selection_output_id",
            "outputs",
            "scheduler",
            "resources",
            "artifacts",
        },
        f"{label}.options",
    )
    require(
        request["request_fingerprint"]
        == w10_fingerprint_without(request, "request_fingerprint"),
        f"{label}.request_fingerprint mismatch",
    )
    _validate_search_space_fingerprint(request["graph"], request["campaign"], label)
    controller_ids = [
        manifest["controller_id"] for manifest in request["controller_manifests"]
    ]
    require(
        controller_ids == sorted(set(controller_ids)),
        f"{label}.controller manifests must be sorted and unique",
    )
    for index, manifest in enumerate(request["controller_manifests"]):
        _validate_controller_manifest_deserialize_shape(
            manifest, f"{label}.controller_manifests[{index}]"
        )
        validate_controller_manifest(manifest, f"{label}.controller_manifests[{index}]")
        validate_w10_manifest_wire(manifest, f"{label}.controller_manifests[{index}]")
    bindings = {
        f"{binding['node_id']}.{binding['input_name']}": binding
        for values in request["campaign"].get("data_bindings", {}).values()
        for binding in values
    }
    identities = [
        validate_w10_data_identity(identity, f"{label}.data_identities[{index}]")
        for index, identity in enumerate(request["data_identities"])
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    require(
        identity_keys == sorted(bindings),
        f"{label}.data identities must exactly cover campaign bindings in order",
    )
    for identity in identities:
        binding = bindings[identity["requirement_key"]]
        require(
            identity["schema_fingerprint"] == binding["schema_fingerprint"]
            and identity["plan_fingerprint"] == binding["plan_fingerprint"]
            and identity["relation_fingerprint"] == binding.get("relation_fingerprint"),
            f"{label}.data identity does not match data binding fingerprints",
        )
    validate_w10_parameter_patches(request, label)
    options = request["options"]
    require(
        options["seed"] == request["campaign"]["root_seed"],
        f"{label}.options.seed differs from campaign.root_seed",
    )
    require(
        options["scheduler"]["workers"] <= options["resources"]["cpu_threads"],
        f"{label}.scheduler workers exceeds cpu_threads",
    )
    gpu_devices = options["resources"]["gpu_devices"]
    require(
        gpu_devices == sorted(set(gpu_devices)),
        f"{label}.resources.gpu_devices must be sorted and unique",
    )
    output_ids = [output["output_id"] for output in options["outputs"]]
    require(
        output_ids == sorted(set(output_ids)),
        f"{label}.outputs must be strictly sorted by output_id and unique",
    )
    coordinates: set[tuple[str, str]] = set()
    output_nodes: list[str] = []
    graph_nodes = {node["id"]: node for node in request["graph"]["nodes"]}
    for index, output in enumerate(options["outputs"]):
        require(
            "unit_level" in output,
            f"{label}.outputs[{index}].unit_level must be explicit",
        )
        if output["prediction_level"] == "sample":
            require(
                output.get("unit_level") == "physical_sample",
                f"{label}.outputs[{index}].unit_level must be physical_sample for sample predictions",
            )
        elif output["prediction_level"] in {"target", "group"}:
            require(
                output.get("unit_level") is None,
                f"{label}.outputs[{index}].unit_level must be null for target/group predictions",
            )
        class_labels = output["class_labels"]
        if output["prediction_kind"] == "class_probability":
            require(
                all(bool(labels) for labels in class_labels),
                f"{label}.outputs[{index}].class labels must be non-empty for class_probability",
            )
        elif output["prediction_kind"] == "regression_point":
            require(
                all(not labels for labels in class_labels),
                f"{label}.outputs[{index}].regression class vocabularies must be empty",
            )
        port_name = w10_resolve_output_port(
            output, request["graph"], f"{label}.outputs[{index}]"
        )
        coordinate = (output["node_id"], port_name)
        require(
            coordinate not in coordinates,
            f"{label}.outputs duplicate producer/port coordinates",
        )
        coordinates.add(coordinate)
        output_nodes.append(output["node_id"])
    require_identifier(
        options["selection_output_id"], f"{label}.options.selection_output_id"
    )
    selection_outputs = [
        output
        for output in options["outputs"]
        if output["output_id"] == options["selection_output_id"]
    ]
    require(
        len(selection_outputs) == 1,
        f"{label}.selection_output_id does not identify a declared output",
    )
    selection_output = selection_outputs[0]
    selection_node = graph_nodes[selection_output["node_id"]]
    selection_manifest = w10_controller_for_node(request, selection_node)
    require(
        "FIT_CV" in selection_manifest["supported_phases"],
        f"{label}.selection output is not scorable in FIT_CV",
    )
    prediction_ports = [
        port
        for port in selection_node["ports"]["outputs"]
        if port["kind"] == "prediction"
    ]
    require(
        len(prediction_ports) == 1,
        f"{label}.selection output producer must expose exactly one prediction port",
    )
    campaign_level = request["campaign"]["aggregation_policy"]["selection_metric_level"]
    require(
        selection_output["prediction_level"] == campaign_level,
        f"{label}.selection output does not match campaign selection_metric_level",
    )
    required_level = options["selection"].get("required_metric_level")
    require(
        required_level is None or required_level == campaign_level,
        f"{label}.selection output does not match required_metric_level",
    )
    metric = options["selection"]["metric"]
    supported_metric = {
        "regression_point": {
            ("mse", "minimize"),
            ("rmse", "minimize"),
            ("mae", "minimize"),
            ("r2", "maximize"),
        },
        "class_label": {
            ("accuracy", "maximize"),
            ("balanced_accuracy", "maximize"),
        },
    }
    prediction_kind_label = {
        "regression_point": "RegressionPoint",
        "class_label": "ClassLabel",
        "class_probability": "ClassProbability",
        "decision_score": "DecisionScore",
    }[selection_output["prediction_kind"]]
    require(
        (metric["name"], metric["objective"])
        in supported_metric.get(selection_output["prediction_kind"], set()),
        f"{label}.selection metric is not supported for {prediction_kind_label}",
    )
    closure = w10_predictor_closure(request["graph"], output_nodes)
    for output_node in output_nodes:
        manifest = w10_controller_for_node(request, graph_nodes[output_node])
        require(
            "emits_predictions" in manifest["capabilities"],
            f"{label}.output controller does not emit predictions",
        )
    validate_w10_influence_requirements(request, closure, label)
    require(
        dagml_tcv1_sha256(request)
        == dagml_tcv1_sha256(_normalize_training_request(request)),
        f"{label} wire content does not match its typed Rust serde representation",
    )


def validate_w10_cache_namespace(namespace: Any, label: str) -> None:
    require(isinstance(namespace, dict), f"{label} must be an object")
    require(namespace["phase"] == "FIT_CV", f"{label} is FIT_CV-only")
    expected_key = (
        f"{namespace['producer_node_id']}.{namespace['source_port_name']}->"
        f"{namespace['consumer_node_id']}.{namespace['target_port_name']}"
    )
    require(
        namespace["prediction_requirement_key"] == expected_key,
        f"{label}.prediction_requirement_key mismatch",
    )
    require(
        namespace["namespace_fingerprint"]
        == w10_fingerprint_without(namespace, "namespace_fingerprint"),
        f"{label}.namespace_fingerprint mismatch",
    )


def validate_w10_parameter_projection(projection: Any, label: str) -> None:
    require(isinstance(projection, dict), f"{label} must be an object")
    require(
        projection["requires_recompile"] == (projection["structural_patch_count"] > 0),
        f"{label}.requires_recompile mismatch",
    )
    require(
        projection["projection_fingerprint"]
        == w10_fingerprint_without(projection, "projection_fingerprint"),
        f"{label}.projection_fingerprint mismatch",
    )


def w10_contains_runtime_handle(value: Any) -> bool:
    if isinstance(value, list):
        return any(w10_contains_runtime_handle(member) for member in value)
    if not isinstance(value, dict):
        return False
    if any(
        key.lower() == "handle"
        or key.lower().endswith("_handle")
        or key.lower().endswith("_handles")
        for key in value
    ):
        return True
    return any(w10_contains_runtime_handle(member) for member in value.values())


def w10_validate_portable_uri(uri: Any, label: str) -> None:
    require_non_empty_string(uri, label)
    require(not uri.startswith(("/", "\\")), f"{label} must be relative")
    require(
        not (len(uri) >= 2 and uri[0].isalpha() and uri[1] == ":"),
        f"{label} must be relative",
    )
    segments = re.split(r"[/\\]", uri)
    require(":" not in segments[0], f"{label} must not contain a scheme")
    require(".." not in segments, f"{label} must not traverse parents")


def validate_w10_bundle_data_requirements_against_plan(
    bundle: dict[str, Any], plan: dict[str, Any], label: str
) -> None:
    expected: dict[str, dict[str, Any]] = {}
    for node_id, node_plan in plan["node_plans"].items():
        for binding in node_plan.get("data_bindings", []):
            key = f"{node_id}.{binding['input_name']}"
            require(key not in expected, f"{label} duplicate plan data requirement")
            expected[key] = binding
    actual: dict[str, dict[str, Any]] = {}
    for index, requirement in enumerate(bundle.get("data_requirements", [])):
        key = f"{requirement['node_id']}.{requirement['input_name']}"
        require(
            key not in actual,
            f"{label}.data_requirements[{index}] duplicates `{key}`",
        )
        actual[key] = requirement
    require(
        set(actual) == set(expected),
        f"{label}.data_requirements do not exactly cover plan data bindings",
    )
    for key, requirement in actual.items():
        binding = expected[key]
        for field in (
            "node_id",
            "input_name",
            "schema_fingerprint",
            "plan_fingerprint",
            "relation_fingerprint",
            "output_representation",
            "feature_set_id",
        ):
            require(
                requirement.get(field) == binding.get(field),
                f"{label}.data_requirements[{key}].{field} does not match plan",
            )


def validate_w10_bundle_selection_and_prediction_links(
    bundle: dict[str, Any], label: str
) -> None:
    selections = bundle.get("selections")
    require(isinstance(selections, dict), f"{label}.selections must be an object")
    for key, decision in selections.items():
        ranked = decision.get("ranked_candidates")
        require(
            isinstance(ranked, list) and bool(ranked),
            f"{label}.selections[{key}].ranked_candidates must be non-empty",
        )
        require(
            ranked[0].get("candidate_id") == decision.get("selected_candidate_id"),
            f"{label}.selections[{key}] first ranked candidate does not match selected candidate",
        )

    requirements = [
        validate_bundle_prediction_requirement(
            requirement, f"{label}.prediction_requirements[{index}]"
        )
        for index, requirement in enumerate(bundle.get("prediction_requirements", []))
    ]
    records = [
        validate_bundle_prediction_cache_record(
            record, f"{label}.prediction_caches[{index}]"
        )
        for index, record in enumerate(bundle.get("prediction_caches", []))
    ]
    requirements_by_key = {
        requirement["requirement_key"]: requirement for requirement in requirements
    }
    records_by_key = {record["requirement_key"]: record for record in records}
    require(
        len(requirements_by_key) == len(requirements),
        f"{label}.prediction_requirements contain duplicates",
    )
    require(
        len(records_by_key) == len(records),
        f"{label}.prediction_caches contain duplicates",
    )
    require(
        set(records_by_key) == set(requirements_by_key),
        f"{label}.prediction_caches do not exactly cover requirements",
    )
    for key, record in records_by_key.items():
        requirement = requirements_by_key[key]
        for field in (
            "partition",
            "prediction_level",
            "fold_ids",
            "unit_ids",
            "sample_ids",
            "prediction_width",
            "target_names",
        ):
            require(
                record[field] == requirement[field],
                f"{label}.prediction_caches[{key}].{field} does not match requirement",
            )


def validate_w10_portable_package(package: Any, label: str) -> None:
    require(isinstance(package, dict), f"{label} must be an object")
    _normalize_portable_predictor_package(
        package
    )  # fail closed on serde container types
    require(
        package["package_fingerprint"]
        == w10_fingerprint_without(package, "package_fingerprint"),
        f"{label}.package_fingerprint mismatch",
    )
    require(
        not w10_contains_runtime_handle(package), f"{label} contains runtime handles"
    )
    template = package["template"]
    require(
        template["template_fingerprint"]
        == w10_fingerprint_without(
            _norm_predictor_template(template), "template_fingerprint"
        ),
        f"{label}.template fingerprint mismatch",
    )
    for controller_id, manifest in template["controller_manifests"].items():
        _validate_controller_manifest_deserialize_shape(
            manifest, f"{label}.template.controller_manifests[{controller_id}]"
        )
        require(
            controller_id == manifest["controller_id"],
            f"{label}.template controller key mismatch",
        )
    plan = package["effective_plan"]
    # A portable package is a deployable predictor: independently re-validate the
    # embedded plan (graph topology, manifests, node-plan/manifest copies,
    # adjacency, params fingerprints) instead of trusting the outcome crosslink.
    validate_execution_plan(plan, f"{label}.effective_plan")
    require(
        template["graph"] == plan["graph_plan"]["graph"]
        and template["campaign"] == plan["campaign"]
        and template["controller_manifests"] == plan["controller_manifests"],
        f"{label}.template does not exactly match effective plan",
    )
    outcome = package["training_outcome"]
    bundle = package["execution_bundle"]
    validate_w10_bundle_data_requirements_against_plan(
        bundle, plan, f"{label}.execution_bundle"
    )
    validate_w10_bundle_selection_and_prediction_links(
        bundle, f"{label}.execution_bundle"
    )
    require(
        package["training_request_fingerprint"]
        == outcome["training_request_fingerprint"],
        f"{label}.request fingerprint is not cross-linked",
    )
    require(
        dagml_tcv1_sha256(_normalize_execution_plan(plan))
        == outcome["effective_plan_fingerprint"],
        f"{label}.effective plan crosslink mismatch",
    )
    require(
        bundle["bundle_id"] == outcome["execution_bundle_id"],
        f"{label}.bundle crosslink mismatch",
    )
    require(
        dagml_tcv1_sha256(_norm_execution_bundle(bundle))
        == outcome["execution_bundle_fingerprint"],
        f"{label}.execution bundle content is not cross-linked",
    )
    require(
        bundle["plan_id"] == plan["id"]
        and bundle["graph_fingerprint"] == plan["graph_fingerprint"]
        and bundle["campaign_fingerprint"] == plan["campaign_fingerprint"]
        and bundle["controller_fingerprint"] == plan["controller_fingerprint"],
        f"{label}.bundle does not match effective plan fingerprints",
    )
    # Mirror ExecutionBundle::validate_against_plan (and RefitArtifactRecord::
    # validate): a re-signed package must not forge refit-artifact provenance.
    # Each record's node must exist in the plan, and its controller_id, nested
    # artifact.controller_id and params_fingerprint must match the owning NodePlan
    # (the effective plan is materialized, so the expected params fingerprint is
    # the node plan's own params_fingerprint).
    package_node_plans = plan["node_plans"]
    for artifact_index, artifact_record in enumerate(bundle.get("refit_artifacts", [])):
        artifact_record_label = (
            f"{label}.execution_bundle.refit_artifacts[{artifact_index}]"
        )
        artifact_node_id = artifact_record["node_id"]
        require(
            artifact_node_id in package_node_plans,
            f"{artifact_record_label} references unknown node {artifact_node_id}",
        )
        artifact_node_plan = package_node_plans[artifact_node_id]
        require(
            artifact_record["artifact"]["controller_id"]
            == artifact_record["controller_id"],
            f"{artifact_record_label} nested artifact controller does not match record controller",
        )
        require(
            artifact_record["controller_id"] == artifact_node_plan["controller_id"],
            f"{artifact_record_label} artifact controller does not match plan",
        )
        require(
            artifact_record["params_fingerprint"]
            == artifact_node_plan["params_fingerprint"],
            f"{artifact_record_label} artifact params do not match plan",
        )
    bindings = package["output_bindings"]
    binding_ids = [binding["binding_id"] for binding in bindings]
    require(binding_ids == sorted(set(binding_ids)), f"{label}.bindings are not sorted")
    coordinates: set[tuple[str, str]] = set()
    for binding in bindings:
        validate_output_binding(
            binding, f"{label}.output_binding[{binding['binding_id']}]"
        )
        require(
            binding["binding_fingerprint"]
            == w10_fingerprint_without(
                _norm_output_binding(binding), "binding_fingerprint"
            ),
            f"{label}.output binding fingerprint mismatch",
        )
        coordinate = (binding["node_id"], binding["port_name"])
        require(
            coordinate not in coordinates, f"{label} binds an output more than once"
        )
        coordinates.add(coordinate)
        w10_resolve_output_port(
            binding,
            plan["graph_plan"]["graph"],
            f"{label}.output binding `{binding['binding_id']}`",
        )
        if binding["prediction_source"] == "final_refit":
            require(
                bool(bundle["refit_artifacts"]),
                f"{label}.final_refit binding requires refit artifacts",
            )
    require(
        [binding["binding_fingerprint"] for binding in bindings]
        == outcome["output_binding_fingerprints"],
        f"{label}.output binding crosslink mismatch",
    )
    influence = package["training_influence"]
    require(
        influence["manifest_fingerprint"]
        == w10_fingerprint_without(influence, "manifest_fingerprint"),
        f"{label}.influence fingerprint mismatch",
    )
    require(
        influence["manifest_fingerprint"] == outcome["training_influence_fingerprint"],
        f"{label}.influence outcome crosslink mismatch",
    )
    previous: tuple[int, str, tuple[int, str]] | None = None
    for entry in influence["entries"]:
        node_key = (0, "") if entry["node_id"] is None else (1, entry["node_id"])
        key = (W10_INFLUENCE_ORDER[entry["kind"]], entry["scope_id"], node_key)
        require(
            previous is None or previous < key,
            f"{label}.influence entries are not canonically sorted",
        )
        previous = key
        for field in ("physical_sample_ids", "origin_sample_ids", "group_ids"):
            values = entry[field]
            require(
                values == sorted(set(values))
                and (field != "physical_sample_ids" or bool(values)),
                f"{label}.influence {field} must be sorted and unique",
            )
    closure = sorted(
        execution_plan_transitive_node_ids(
            plan,
            {binding["node_id"] for binding in bindings},
            label,
        )
    )
    require(
        package["predictor_node_ids"] == closure,
        f"{label}.predictor closure mismatch",
    )
    # A portable package must independently prove PREDICT-replayability from its
    # own plan/closure/retained artifacts — never infer it from an outcome claim.
    # PREDICT replay never consumes OOF payloads, so no portable caches are read.
    require(
        "PREDICT"
        in derive_replayable_phases(plan, set(closure), True, bundle, None, label),
        f"{label} is not PREDICT-replayable: its predictor closure does not "
        "support PREDICT with self-contained retained artifacts",
    )
    require(
        all(
            entry["node_id"] is None or entry["node_id"] in closure
            for entry in influence["entries"]
        ),
        f"{label}.influence references node outside predictor closure",
    )
    validate_training_influence_against_plan(influence, plan, set(closure), label)
    requirements = {
        f"{requirement['node_id']}.{requirement['input_name']}": requirement
        for requirement in package["execution_bundle"]["data_requirements"]
    }
    identities = [
        validate_w10_data_identity(identity, f"{label}.data_identities[{index}]")
        for index, identity in enumerate(package["data_identities"])
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    require(
        identity_keys == sorted(set(identity_keys)),
        f"{label}.data identities must be sorted by requirement_key",
    )
    require(identity_keys == sorted(requirements), f"{label}.data identity coverage")
    for identity in identities:
        requirement = requirements[identity["requirement_key"]]
        require(
            identity["schema_fingerprint"] == requirement["schema_fingerprint"]
            and identity["plan_fingerprint"] == requirement["plan_fingerprint"]
            and identity["relation_fingerprint"]
            == requirement.get("relation_fingerprint")
            == influence["relation_fingerprint"],
            f"{label}.data identity does not match bundle fingerprints",
        )
    require(
        dagml_tcv1_sha256(identities) == outcome["data_identities_fingerprint"],
        f"{label}.data identity content is not cross-linked",
    )
    records = {
        record["artifact"]["id"]: record
        for record in package["execution_bundle"]["refit_artifacts"]
    }
    artifact_ids = [binding["artifact_id"] for binding in package["artifact_bindings"]]
    require(
        artifact_ids == sorted(set(artifact_ids)) == sorted(records),
        f"{label}.artifact coverage/order",
    )
    for binding in package["artifact_bindings"]:
        record = records[binding["artifact_id"]]
        artifact = record["artifact"]
        if binding["load_mode"] == "native_portable":
            require(
                artifact.get("backend") is not None
                and artifact.get("uri") is not None
                and artifact.get("content_fingerprint") is not None,
                f"{label}.native artifact is not portable",
            )
            require_sha256(
                artifact["content_fingerprint"],
                f"{label}.native artifact content_fingerprint",
            )
            w10_validate_portable_uri(artifact["uri"], f"{label}.native artifact uri")
            require(
                plan["node_plans"][record["node_id"]]["artifact_policy"] != "host_only",
                f"{label}.host_only artifact is not native portable",
            )
        else:
            require(
                package["fitted_artifact_mode"] == "allow_host_sidecar",
                f"{label}.portable_required package forbids host sidecar",
            )
    require(
        dagml_tcv1_sha256(package)
        == dagml_tcv1_sha256(_normalize_portable_predictor_package(package)),
        f"{label} wire content does not match its typed Rust serde representation",
    )


def w10_semantic_validator(contract: str) -> Any:
    validators = {
        "training_request": validate_w10_training_request,
        "cache_namespace": validate_w10_cache_namespace,
        "parameter_projection": validate_w10_parameter_projection,
        "portable_predictor_package": validate_w10_portable_package,
        "training_outcome": validate_training_outcome,
    }
    require(contract in validators, f"unknown W1 training contract `{contract}`")
    validator = validators[contract]
    return lambda document: validator(document, f"W1 {contract}")


def validate_w10_training_snapshot(
    registry: Registry,
    schemas_by_id: dict[str, dict[str, Any]],
    positive_fixtures: dict[str, Any],
    negatives: Any,
) -> None:
    """Validate the W1-0 schemas, deterministic fixtures and semantic refusals."""

    fixture_contracts = {
        "training_request_refit.v1.json": (
            TRAINING_REQUEST_SCHEMA_ID,
            "training_request",
        ),
        "training_request_no_refit.v1.json": (
            TRAINING_REQUEST_SCHEMA_ID,
            "training_request",
        ),
        "training_request_active_influence.v1.json": (
            TRAINING_REQUEST_SCHEMA_ID,
            "training_request",
        ),
        "training_request_package_refit.v1.json": (
            TRAINING_REQUEST_SCHEMA_ID,
            "training_request",
        ),
        "cache_namespace_fit_cv.v1.json": (
            CACHE_NAMESPACE_SCHEMA_ID,
            "cache_namespace",
        ),
        "parameter_projection_empty.v1.json": (
            PARAMETER_PROJECTION_SCHEMA_ID,
            "parameter_projection",
        ),
        "portable_predictor_package.v1.json": (
            PORTABLE_PREDICTOR_PACKAGE_SCHEMA_ID,
            "portable_predictor_package",
        ),
        "training_outcome_refit.v1.json": (
            TRAINING_OUTCOME_SCHEMA_ID,
            "training_outcome",
        ),
    }
    for filename, (schema_id, contract) in fixture_contracts.items():
        require(
            schema_id in schemas_by_id, f"W1 schema `{schema_id}` is not registered"
        )
        document = positive_fixtures[filename]
        validate_draft_2020_instance(
            document,
            schemas_by_id[schema_id],
            registry,
            f"W1 positive fixture {filename}",
        )
        w10_semantic_validator(contract)(document)
    namespace = positive_fixtures["cache_namespace_fit_cv.v1.json"]
    package = positive_fixtures["portable_predictor_package.v1.json"]
    matching_identities = [
        identity
        for identity in package["data_identities"]
        if identity["requirement_key"] == namespace["data_requirement_key"]
    ]
    require(
        len(matching_identities) == 1
        and matching_identities[0]["identity_fingerprint"]
        == namespace["data_identity_fingerprint"],
        "W1 cache namespace does not bind one complete package data identity",
    )
    require(
        package["training_request_fingerprint"]
        == positive_fixtures["training_request_package_refit.v1.json"][
            "request_fingerprint"
        ],
        "W1 portable package does not bind the committed refit request",
    )
    outcome_fixture = positive_fixtures["training_outcome_refit.v1.json"]
    require(
        package["training_outcome"]["outcome_fingerprint"]
        == outcome_fixture["outcome_fingerprint"]
        and package["training_outcome"]["training_request_fingerprint"]
        == outcome_fixture["training_request_fingerprint"]
        and package["data_identities"] == outcome_fixture["data_identities"],
        "W1 portable package does not bind the committed training outcome",
    )
    require(
        isinstance(negatives, dict)
        and negatives.get("schema_version") == 1
        and isinstance(negatives.get("cases"), list)
        and negatives["cases"],
        "W1 negative fixture is malformed",
    )
    schema_by_contract = {
        "training_request": TRAINING_REQUEST_SCHEMA_ID,
        "cache_namespace": CACHE_NAMESPACE_SCHEMA_ID,
        "portable_predictor_package": PORTABLE_PREDICTOR_PACKAGE_SCHEMA_ID,
        "training_outcome": TRAINING_OUTCOME_SCHEMA_ID,
    }
    seen: set[str] = set()
    for case in negatives["cases"]:
        case_id = case.get("id")
        require_non_empty_string(case_id, "W1 negative case id")
        require(case_id not in seen, f"duplicate W1 negative id `{case_id}`")
        seen.add(case_id)
        contract = case["contract"]
        document = case["document"]
        fingerprint_field = {
            "training_request": "request_fingerprint",
            "cache_namespace": "namespace_fingerprint",
            "portable_predictor_package": "package_fingerprint",
            "training_outcome": "outcome_fingerprint",
        }[contract]
        require(
            document[fingerprint_field]
            == w10_fingerprint_without(document, fingerprint_field),
            f"W1 negative `{case_id}` is not re-fingerprinted",
        )
        schema = schemas_by_id[schema_by_contract[contract]]
        validator = Draft202012Validator(schema, registry=registry)
        schema_errors = list(validator.iter_errors(document))
        semantic_error = ""
        try:
            w10_semantic_validator(contract)(document)
        except (ContractError, KeyError, IndexError, TypeError) as exc:
            semantic_error = str(exc)
        require(schema_errors or semantic_error, f"W1 negative `{case_id}` passed")
        combined = "\n".join(error.message for error in schema_errors)
        combined = f"{combined}\n{semantic_error}".lower()
        expected_error = case.get("expected_error")
        require_non_empty_string(
            expected_error, f"W1 negative `{case_id}` expected_error"
        )
        require(
            expected_error.lower() in combined,
            f"W1 negative `{case_id}` failed for the wrong cause: {combined}",
        )


def validate_w10_training_pack(pack: Any, negatives: dict[str, Any]) -> None:
    require(isinstance(pack, dict), "W1 training conformance pack must be an object")
    require(
        set(pack)
        == {
            "schema_version",
            "pack_id",
            "hash_algorithm",
            "canonical_profile",
            "artifacts",
            "positive_fixture_ids",
            "negative_case_ids",
            "pack_checksum",
        },
        "W1 training conformance pack fields drifted",
    )
    require(pack["schema_version"] == 1, "W1 training pack version must be 1")
    require(
        pack["pack_id"] == "dag-ml.training-contracts.v1",
        "W1 training pack id drifted",
    )
    require(
        pack["hash_algorithm"] == "sha256-file-bytes",
        "W1 training pack hash algorithm drifted",
    )
    require(
        pack["canonical_profile"] == "DAG-ML TCV1",
        "W1 training pack canonical profile drifted",
    )
    artifacts = pack["artifacts"]
    require(isinstance(artifacts, list) and artifacts, "W1 training pack is empty")
    paths = [entry.get("path") for entry in artifacts]
    require(paths == sorted(set(paths)), "W1 training pack paths are not canonical")
    root = ROOT.resolve()
    for index, entry in enumerate(artifacts):
        entry_label = f"W1 training pack artifact[{index}]"
        require(
            set(entry) == {"path", "sha256", "kind"},
            f"{entry_label} fields drifted",
        )
        require_non_empty_string(entry["kind"], f"{entry_label}.kind")
        require_sha256(entry["sha256"], f"{entry_label}.sha256")
        raw_path = entry["path"]
        require(
            "\\" not in raw_path,
            f"{entry_label}.path must use portable separators",
        )
        relative = Path(entry["path"])
        require(
            not relative.is_absolute()
            and relative.as_posix() == raw_path
            and all(part not in {"", ".", ".."} for part in relative.parts),
            f"{entry_label}.path is not canonical or contains traversal",
        )
        path_cursor = ROOT
        for part in relative.parts:
            path_cursor /= part
            require(
                not path_cursor.is_symlink(),
                f"{entry_label}.path must not traverse or name a symbolic link",
            )
        resolved = (ROOT / relative).resolve()
        require(
            resolved.is_relative_to(root),
            f"{entry_label}.path escapes the repository root",
        )
        require(resolved.is_file(), f"{entry_label}.path is missing")
        require(
            hashlib.sha256(resolved.read_bytes()).hexdigest() == entry["sha256"],
            f"{entry_label}.sha256 does not match file bytes",
        )
    validate_artifact_pack_schema_dependency_closure(paths, "W1 training pack")
    required_paths = {
        "docs/TRAINING_CONTRACTS.md",
        "docs/contracts/training_request.schema.json",
        "docs/contracts/cache_namespace.schema.json",
        "docs/contracts/parameter_projection.schema.json",
        "docs/contracts/portable_predictor_package.schema.json",
        "docs/contracts/training_outcome.schema.json",
        "crates/dag-ml-core/src/training_runtime.rs",
        "examples/fixtures/training/negative_cases.v1.json",
        "parity/training/generate_fixtures.py",
        "parity/training/oracle.py",
        "parity/training/tests/test_training_contracts.py",
        "parity/schema_dependencies.py",
    }
    missing = required_paths - set(paths)
    require(
        not missing,
        f"W1 training pack misses required artifacts: {sorted(missing)}",
    )
    require(
        pack["positive_fixture_ids"]
        == [
            "cache_namespace_fit_cv.v1",
            "parameter_projection_empty.v1",
            "portable_predictor_package.v1",
            "python_training_multiport_smoke.v1",
            "python_training_smoke.v1",
            "training_outcome_refit.v1",
            "training_request_active_influence.v1",
            "training_request_no_refit.v1",
            "training_request_package_refit.v1",
            "training_request_refit.v1",
        ],
        "W1 training pack positive fixture ids drifted",
    )
    require(
        pack["negative_case_ids"] == [case["id"] for case in negatives["cases"]],
        "W1 training pack negative case ids drifted",
    )
    require(
        pack["pack_checksum"] == w10_fingerprint_without(pack, "pack_checksum"),
        "W1 training pack checksum mismatch",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-sibling",
        action="store_true",
        help="Fail when the sibling dag-ml-data checkout is not available.",
    )
    parser.add_argument(
        "--sibling-root",
        type=Path,
        default=None,
        help="Explicit dag-ml-data checkout path; overrides env/default candidates.",
    )
    return parser.parse_args(argv)


def candidate_sibling_roots(explicit_root: Path | None = None) -> list[Path]:
    if explicit_root is not None:
        return [explicit_root.expanduser()]
    candidates = []
    env_path = os.environ.get("DAG_ML_DATA_REPO")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(ROOT.parent / "dag-ml-data")
    candidates.append(ROOT / "external" / "dag-ml-data")
    return candidates


def sibling_root(explicit_root: Path | None = None) -> Path | None:
    env_path = os.environ.get("DAG_ML_DATA_REPO")
    for candidate in candidate_sibling_roots(explicit_root):
        if candidate.exists():
            return candidate.resolve()
    if explicit_root is not None:
        raise ContractError(
            f"--sibling-root points to a missing checkout: {explicit_root}"
        )
    if env_path:
        raise ContractError(
            f"DAG_ML_DATA_REPO points to a missing checkout: {env_path}"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        local_controller_manifest_schema = load_json(
            ROOT / CONTROLLER_MANIFEST_SCHEMA_REL
        )
        local_representation_registry = load_json(ROOT / REPRESENTATION_REGISTRY_REL)
        local_selection_policy_schema = load_json(ROOT / SELECTION_POLICY_SCHEMA_REL)
        local_selection_decision_schema = load_json(
            ROOT / SELECTION_DECISION_SCHEMA_REL
        )
        local_pack = load_json(ROOT / CONFORMANCE_PACK_REL)
        local_parity_oracle = load_json(ROOT / PARITY_ORACLE_REL)
        local_openlineage_facets_schema = load_json(
            ROOT / OPENLINEAGE_FACETS_SCHEMA_REL
        )
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
        local_score_set_schema = load_json(ROOT / SCORE_SET_SCHEMA_REL)
        local_score_set_fixture = load_json(ROOT / SCORE_SET_FIXTURE_REL)
        local_chain_effect_analysis_schema = load_json(
            ROOT / CHAIN_EFFECT_ANALYSIS_SCHEMA_REL
        )
        local_chain_effect_analysis_fixture = load_json(
            ROOT / CHAIN_EFFECT_ANALYSIS_FIXTURE_REL
        )
        local_conformal_calibration_schema = load_json(
            ROOT / CONFORMAL_CALIBRATION_SCHEMA_REL
        )
        local_conformal_calibration_fixture = load_json(
            ROOT / CONFORMAL_CALIBRATION_FIXTURE_REL
        )
        local_conformal_robustness_fixtures = {
            fixture_rel.as_posix(): load_json(ROOT / fixture_rel)
            for _schema_rel, fixture_rel, _schema_id in CONFORMAL_ROBUSTNESS_CONTRACTS
        }
        local_conformal_robustness_golden = load_json(
            ROOT / CONFORMAL_ROBUSTNESS_GOLDEN_REL
        )
        local_conformal_metrics_golden = load_json(ROOT / CONFORMAL_METRICS_GOLDEN_REL)
        local_canonical_profile_golden = load_json(ROOT / CANONICAL_PROFILE_GOLDEN_REL)
        local_conformal_robustness_pack = load_json(
            ROOT / CONFORMAL_ROBUSTNESS_PACK_REL
        )
        local_w10_training_fixtures = {
            fixture_rel.name: load_json(ROOT / fixture_rel)
            for fixture_rel in W10_TRAINING_POSITIVE_FIXTURE_RELS
        }
        local_w10_training_negatives = load_json(
            ROOT / W10_TRAINING_NEGATIVE_FIXTURE_REL
        )
        local_w10_training_pack = load_json(ROOT / W10_TRAINING_PACK_REL)
        local_parameter_patch_schema = load_json(ROOT / PARAMETER_PATCH_SCHEMA_REL)
        local_output_binding_schema = load_json(ROOT / OUTPUT_BINDING_SCHEMA_REL)
        local_training_influence_schema = load_json(
            ROOT / TRAINING_INFLUENCE_SCHEMA_REL
        )
        local_execution_bundle_schema = load_json(ROOT / EXECUTION_BUNDLE_SCHEMA_REL)
        local_prediction_cache_payload_set_schema = load_json(
            ROOT / PREDICTION_CACHE_PAYLOAD_SET_SCHEMA_REL
        )
        local_training_outcome_schema = load_json(ROOT / TRAINING_OUTCOME_SCHEMA_REL)
        local_replay_outcome_schema = load_json(ROOT / REPLAY_OUTCOME_SCHEMA_REL)
        local_parameter_patch_fixture = load_json(ROOT / PARAMETER_PATCH_FIXTURE_REL)
        local_output_binding_fixture = load_json(ROOT / OUTPUT_BINDING_FIXTURE_REL)
        local_training_outcome_refit_fixture = load_json(
            ROOT / TRAINING_OUTCOME_REFIT_FIXTURE_REL
        )
        local_training_outcome_no_refit_fixture = load_json(
            ROOT / TRAINING_OUTCOME_NO_REFIT_FIXTURE_REL
        )
        local_replay_outcome_fixtures = [
            (fixture_rel, load_json(ROOT / fixture_rel))
            for fixture_rel in REPLAY_OUTCOME_FIXTURE_RELS
        ]
        local_operator_variant_label_fixture = load_json(
            ROOT / OPERATOR_VARIANT_LABEL_FIXTURE_REL
        )
        local_process_adapter_description_schema = load_json(
            ROOT / PROCESS_ADAPTER_DESCRIPTION_SCHEMA_REL
        )
        local_process_adapter_frame_schema = load_json(
            ROOT / PROCESS_ADAPTER_FRAME_SCHEMA_REL
        )
        local_research_provenance_profile = load_json(
            ROOT / RESEARCH_PROVENANCE_PROFILE_REL
        )
        local_fixture = load_json(ROOT / LOCAL_FIXTURE_REL)
        local_multisource_fixture = load_json(ROOT / LOCAL_MULTISOURCE_FIXTURE_REL)
        local_feature_fusion_fixture = load_json(
            ROOT / LOCAL_FEATURE_FUSION_FIXTURE_REL
        )
        local_fold_set_fixture = load_json(ROOT / SHARED_FOLD_SET_FIXTURE_REL)
        local_graph_spec_fixture = load_json(ROOT / LOCAL_GRAPH_SPEC_FIXTURE_REL)
        local_pipeline_dsl_fixture = load_json(ROOT / LOCAL_PIPELINE_DSL_FIXTURE_REL)
        local_campaign_spec_fixture = load_json(ROOT / LOCAL_CAMPAIGN_SPEC_FIXTURE_REL)
        local_execution_plan_fixture = load_json(
            ROOT / LOCAL_EXECUTION_PLAN_FIXTURE_REL
        )
        local_model_input_spec_fixture = load_json(
            ROOT / LOCAL_MODEL_INPUT_SPEC_FIXTURE_REL
        )
        local_data_plan_fixture = load_json(ROOT / LOCAL_DATA_PLAN_FIXTURE_REL)
        local_controller_manifest_fixture = load_json(
            ROOT / LOCAL_CONTROLLER_MANIFEST_FIXTURE_REL
        )
        local_controller_manifest_list_fixture = load_json(
            ROOT / LOCAL_CONTROLLER_MANIFEST_LIST_FIXTURE_REL
        )
        local_selection_policy_fixture = load_json(
            ROOT / LOCAL_SELECTION_POLICY_FIXTURE_REL
        )
        local_selection_decision_fixture = load_json(
            ROOT / LOCAL_SELECTION_DECISION_FIXTURE_REL
        )
        local_data_output_provenance_fixture = load_json(
            ROOT / LOCAL_DATA_OUTPUT_PROVENANCE_FIXTURE_REL
        )
        local_oof_success_fixture = load_json(ROOT / LOCAL_OOF_SUCCESS_FIXTURE_REL)
        local_oof_train_refusal_fixture = load_json(
            ROOT / LOCAL_OOF_TRAIN_REFUSAL_FIXTURE_REL
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
        local_schema_registry, local_schemas_by_id = build_local_schema_registry()
        validate_estimator_draft_2020_contracts(
            local_schema_registry,
            local_schemas_by_id,
            local_parameter_patch_fixture,
            local_output_binding_fixture,
            local_training_outcome_refit_fixture,
            local_training_outcome_no_refit_fixture,
            local_replay_outcome_fixtures,
            local_conformal_calibration_fixture,
            "dag-ml estimator Draft 2020-12",
        )
        validate_conformal_robustness_snapshot(
            local_schema_registry,
            local_schemas_by_id,
            local_conformal_robustness_fixtures,
            local_conformal_robustness_golden,
            local_conformal_metrics_golden,
            local_canonical_profile_golden,
            local_conformal_robustness_pack,
        )
        validate_w10_training_snapshot(
            local_schema_registry,
            local_schemas_by_id,
            local_w10_training_fixtures,
            local_w10_training_negatives,
        )
        validate_w10_training_pack(
            local_w10_training_pack,
            local_w10_training_negatives,
        )
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
        validate_score_set_schema(local_score_set_schema, "dag-ml")
        validate_score_set_fixture(local_score_set_fixture, "dag-ml")
        validate_chain_effect_analysis_schema(
            local_chain_effect_analysis_schema, "dag-ml"
        )
        validate_chain_effect_analysis_fixture(
            local_chain_effect_analysis_fixture, "dag-ml"
        )
        validate_conformal_calibration_schema(
            local_conformal_calibration_schema,
            "dag-ml",
        )
        validate_conformal_calibration_fixture(
            local_conformal_calibration_fixture,
            "dag-ml",
        )
        validate_parameter_patch_schema(local_parameter_patch_schema, "dag-ml")
        validate_output_binding_schema(local_output_binding_schema, "dag-ml")
        validate_training_influence_schema(local_training_influence_schema, "dag-ml")
        validate_estimator_support_schema(
            local_execution_bundle_schema,
            EXECUTION_BUNDLE_SCHEMA_ID,
            {
                "bundle_id",
                "schema_version",
                "plan_id",
                "graph_fingerprint",
                "campaign_fingerprint",
                "controller_fingerprint",
                "selected_variant_id",
                "selections",
                "refit_artifacts",
                "prediction_requirements",
                "prediction_caches",
                "data_requirements",
                "unsafe_flags",
                "metadata",
            },
            "dag-ml",
        )
        validate_estimator_support_schema(
            local_prediction_cache_payload_set_schema,
            PREDICTION_CACHE_PAYLOAD_SET_SCHEMA_ID,
            {"bundle_id", "schema_version", "caches"},
            "dag-ml",
        )
        validate_training_outcome_schema(local_training_outcome_schema, "dag-ml")
        validate_replay_outcome_schema(local_replay_outcome_schema, "dag-ml")
        validate_estimator_contract_fixtures(
            local_parameter_patch_fixture,
            local_output_binding_fixture,
            local_training_outcome_refit_fixture,
            local_training_outcome_no_refit_fixture,
            local_replay_outcome_fixtures,
            local_conformal_calibration_fixture,
            "dag-ml estimator contracts",
        )
        validate_operator_variant_label_fixture(
            local_operator_variant_label_fixture, "dag-ml"
        )
        validate_process_adapter_description_schema(
            local_process_adapter_description_schema,
            "dag-ml",
        )
        validate_process_adapter_frame_schema(
            local_process_adapter_frame_schema, "dag-ml"
        )
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
        validate_dag_ml_training_header(local_header, "dag-ml")
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
            local_data_output_provenance_schema,
            local_parity_oracle,
            local_representation_registry,
            local_fixture,
            local_multisource_fixture,
            local_feature_fusion_fixture,
            local_model_input_spec_fixture,
            local_data_output_provenance_fixture,
            local_oof_success_fixture,
            local_oof_train_refusal_fixture,
            local_header,
            "dag-ml",
        )
        validate_research_provenance_profile(
            local_research_provenance_profile,
            local_openlineage_facets_schema,
            "dag-ml",
        )

        sibling = sibling_root(args.sibling_root)
        if sibling is None:
            if args.require_sibling:
                raise ContractError(
                    "sibling dag-ml-data checkout is required but was not found"
                )
            print("validated dag-ml contract; sibling dag-ml-data checkout not present")
            return 0

        sibling_schema = load_json(sibling / SCHEMA_REL)
        sibling_feature_fusion_schema = load_json(sibling / FEATURE_FUSION_SCHEMA_REL)
        sibling_pack = load_json(sibling / CONFORMANCE_PACK_REL)
        sibling_parity_oracle = load_json(sibling / PARITY_ORACLE_REL)
        sibling_representation_registry = load_json(
            sibling / REPRESENTATION_REGISTRY_REL
        )
        sibling_fixture = load_json(sibling / SIBLING_FIXTURE_REL)
        sibling_feature_fusion_fixture = load_json(
            sibling / SIBLING_FEATURE_FUSION_FIXTURE_REL
        )
        sibling_model_input_spec_fixture = load_json(
            sibling / SIBLING_MODEL_INPUT_SPEC_FIXTURE_REL
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
        validate_model_input_spec(sibling_model_input_spec_fixture, "dag-ml-data")
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
            local_data_output_provenance_schema,
            sibling_parity_oracle,
            sibling_representation_registry,
            sibling_fixture,
            local_multisource_fixture,
            sibling_feature_fusion_fixture,
            sibling_model_input_spec_fixture,
            local_data_output_provenance_fixture,
            local_oof_success_fixture,
            local_oof_train_refusal_fixture,
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
            local_representation_registry == sibling_representation_registry,
            "representation registries diverge",
        )
        require(
            local_model_input_spec_fixture == sibling_model_input_spec_fixture,
            "model input spec fixtures diverge",
        )
        require(
            canonical_fold_set_fingerprint(local_fold_set_fixture)
            == canonical_fold_set_fingerprint(sibling_fold_set_fixture),
            "shared fold set canonical fingerprints diverge",
        )
        require(local_pack == sibling_pack, "shared conformance packs diverge")
        require(
            local_parity_oracle == sibling_parity_oracle,
            "parity oracle manifests diverge",
        )
        print(f"validated dag-ml contract against dag-ml-data at {sibling}")
        return 0
    except ContractError as exc:
        print(f"contract validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
