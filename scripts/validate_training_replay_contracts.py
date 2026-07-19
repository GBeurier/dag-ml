#!/usr/bin/env python3
"""Validate the isolated D4 public training-replay contracts and pack."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import validate_contracts as base  # noqa: E402
from parity.training import training_replay_oracle as replay_oracle  # noqa: E402
from parity.schema_dependencies import with_transitive_schema_dependencies  # noqa: E402

ContractError = base.ContractError
contains_runtime_handle = base.contains_runtime_handle
dagml_tcv1_sha256 = base.dagml_tcv1_sha256
execution_plan_graph_nodes = base.execution_plan_graph_nodes
execution_plan_transitive_node_ids = base.execution_plan_transitive_node_ids
expected_output_columns = base.expected_output_columns
legacy_serde_json_sha256 = base.legacy_serde_json_sha256
_norm_aggregation_policy = base._norm_aggregation_policy
replay_outcome_fingerprint = base.replay_outcome_fingerprint
require = base.require
require_exact_keys = base.require_exact_keys
require_identifier = base.require_identifier
require_non_empty_string = base.require_non_empty_string
require_non_negative_int = base.require_non_negative_int
require_optional_identifier = base.require_optional_identifier
require_optional_non_empty_string = base.require_optional_non_empty_string
require_sha256 = base.require_sha256
require_version_one = base.require_version_one
validate_draft_2020_instance = base.validate_draft_2020_instance
validate_envelope = base.validate_envelope
validate_metadata_object = base.validate_metadata_object
validate_ordered_unique_strings = base.validate_ordered_unique_strings
validate_output_binding = base.validate_output_binding
validate_portable_lineage_record = base.validate_portable_lineage_record
validate_prediction_matrix = base.validate_prediction_matrix
validate_strict_json_value = base.validate_strict_json_value
validate_training_outcome = base.validate_training_outcome
validate_w10_data_identity = base.validate_w10_data_identity
w10_fingerprint_without = base.w10_fingerprint_without

_RELATION_REQUIRED_FIELDS = {"observation_id", "sample_id"}
_RELATION_OPTIONAL_FIELDS = {
    "unit_level",
    "unit_id",
    "source_id",
    "rep_id",
    "target_id",
    "group_id",
    "origin_sample_id",
    "derived_unit_id",
    "component_observation_ids",
    "sample_influence_weight",
    "quality_flag",
    "is_augmented",
    "excluded",
    "metadata",
    "tags",
}


def _validated_identifier(value: Any, label: str) -> str:
    require_identifier(value, label)
    return value


def _validated_non_blank(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and bool(value.strip()),
        f"{label} must be non-blank",
    )
    return value


def _sorted_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json_value(member) for member in value]
    return value


def _contains_raw_feature_payload(value: Any) -> bool:
    forbidden = {"raw_features", "feature_matrix", "raw_spectra", "raw_wavelengths"}
    if isinstance(value, dict):
        return any(key.lower() in forbidden for key in value) or any(
            _contains_raw_feature_payload(member) for member in value.values()
        )
    if isinstance(value, list):
        return any(_contains_raw_feature_payload(member) for member in value)
    return False


def _optional_relation_validated_identifier(value: Any, label: str) -> Any:
    if value is not None:
        _validated_identifier(value, label)
    return value


def _optional_relation_text(value: Any, label: str) -> Any:
    if value is not None:
        _validated_non_blank(value, label)
    return value


def validate_output_binding_against_plan(
    binding: dict[str, Any], plan: dict[str, Any], label: str
) -> None:
    """Validate the graph coordinate and effective aggregation policy."""

    base.validate_output_binding_against_plan(binding, plan, label)
    campaign = plan.get("campaign")
    require(isinstance(campaign, dict), f"{label} source campaign is absent")
    policy = campaign.get("aggregation_policy")
    require(isinstance(policy, dict), f"{label} source aggregation policy is absent")
    expected = legacy_serde_json_sha256(_norm_aggregation_policy(policy))
    require(
        binding["aggregation_fingerprint"] == expected,
        f"{label}.aggregation_fingerprint does not match effective policy",
    )


REQUEST_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/training_replay_request.v1.schema.json"
)
OUTCOME_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/training_replay_outcome.v1.schema.json"
)
LEGACY_OUTCOME_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/replay_outcome.v1.schema.json"
)
NODE_RESULT_V2_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/node_result.v2.schema.json"
)
BOUND_OUTPUT_V2_SCHEMA_ID = (
    "https://github.com/GBeurier/dag-ml/schemas/bound_training_output.v2.schema.json"
)
PACK_PATH = ROOT / "docs/contracts/training_replay_contract_conformance_pack.v1.json"
BASE_PACK_PATH = ROOT / "docs/contracts/training_contract_conformance_pack.v1.json"
TRAINING_FIXTURE_ROOT = ROOT / "examples/fixtures/training"
FIXTURE_ROOT = TRAINING_FIXTURE_ROOT / "replay"
BASE_PACK_SHA256 = "2038a1d13ad1fa29808b76582324b34b98541b0357757b9e2cb9ec3cbd9f288c"
BASE_PACK_CHECKSUM = "8e684dcac19df0cf22604f5391a79f25ea7e629e8ab3f2d54fdc2a8e60101290"
LEGACY_AUTHORITY_SHA256 = {
    "docs/contracts/replay_outcome.schema.json": "c57279e8c76e4e2467af0eca5eb59804a2f7bb97bec6cce9d8b23975f223c36a",
    "examples/fixtures/estimator/replay_outcome_predict.v1.json": "037fad7f3cb907f3474cce4f51526538f2c4d6fcad3af93a320c6d282ce470c5",
    "examples/fixtures/estimator/replay_outcome_class_probability.v1.json": "2bcb925b79f1766515c924697f7b5ff62ede396e10095e183a14baefdd622329",
    "examples/fixtures/estimator/replay_outcome_explain.v1.json": "fe593f9bdd89ecfcffdb224435b0ce842f5a492a7b8045657ba22bfc63185db7",
    "docs/contracts/aggregation_controller_task.schema.json": "2b12131727f5e3a355b0c6b5e402f6075c37cf5ed3e7a186c9e0890da5583ccd",
    "docs/contracts/aggregation_controller_result.schema.json": "e782d57c2bff01031ab4cf453b362afab5bf25e1e83eac5cf65ef463347045ff",
    "docs/contracts/process_adapter_frame.schema.json": "024ee268eca668479acc1e0ddf979247fb1214f5022373ce36f85e55bf9499f3",
}
D4_EXPECTED_ARTIFACTS = {
    ".github/workflows/training-replay-contracts.yml": "ci_gate",
    "docs/TRAINING_REPLAY_CONTRACTS.md": "documentation",
    "docs/adr/ADR-21-forward-replay-ownership.md": "architecture_decision",
    "docs/adr/README.md": "documentation_index",
    "docs/index.md": "documentation_index",
    "crates/dag-ml-core/src/lib.rs": "training_replay_public_export",
    "crates/dag-ml-core/src/replay.rs": "training_replay_core_contract",
    "docs/contracts/aggregation_controller_result.v2.schema.json": "schema",
    "docs/contracts/aggregation_controller_result.schema.json": "legacy_schema_authority",
    "docs/contracts/aggregation_controller_task.v2.schema.json": "schema",
    "docs/contracts/aggregation_controller_task.schema.json": "legacy_schema_authority",
    "docs/contracts/bound_training_output.v2.schema.json": "schema",
    "docs/contracts/execution_bundle.v2.schema.json": "schema",
    "docs/contracts/node_result.v2.schema.json": "schema",
    "docs/contracts/prediction_cache_payload_set.v2.schema.json": "schema",
    "docs/contracts/process_adapter_frame.v2.schema.json": "schema",
    "docs/contracts/process_adapter_frame.schema.json": "legacy_schema_authority",
    "docs/contracts/replay_outcome.schema.json": "legacy_schema_context",
    "docs/contracts/score_set.v2.schema.json": "schema",
    "docs/contracts/training_outcome.v2.schema.json": "schema",
    "docs/contracts/training_replay_outcome.schema.json": "schema",
    "docs/contracts/training_replay_request.schema.json": "schema",
    "examples/fixtures/estimator/replay_outcome_class_probability.v1.json": "legacy_fixture",
    "examples/fixtures/estimator/replay_outcome_explain.v1.json": "legacy_fixture",
    "examples/fixtures/estimator/replay_outcome_predict.v1.json": "legacy_fixture",
    "examples/fixtures/training/replay/training_outcome_port_explicit.v2.json": "fixture",
    "examples/fixtures/training/replay/training_port_explicit_protocols.v2.json": "fixture",
    "examples/fixtures/training/replay/training_replay_input_envelopes.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_multi_port_outputs.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_negative_cases.v1.json": "negative_fixture",
    "examples/fixtures/training/replay/training_replay_outcome_explain.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_outcome_explain_only.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_outcome_predict.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_output_class_label.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_output_class_probability.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_output_observation.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_request_explain.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_request_predict.v1.json": "fixture",
    "examples/fixtures/training/replay/training_replay_source_outcome_explain.v1.json": "fixture",
    "parity/training/generate_training_replay_fixtures.py": "generator",
    "parity/training/tests/test_training_replay_contracts.py": "test",
    "parity/training/training_replay_oracle.py": "test_oracle",
    "scripts/requirements-contracts.txt": "ci_dependency",
    "scripts/validate_training_replay_contracts.py": "production_validator",
}
EXPECTED_POSITIVE_FIXTURE_IDS = sorted(
    [
        "training_outcome_port_explicit.v2",
        "training_port_explicit_protocols.v2",
        "training_replay_input_envelopes.v1",
        "training_replay_multi_port_outputs.v1",
        "training_replay_outcome_explain.v1",
        "training_replay_outcome_explain_only.v1",
        "training_replay_outcome_predict.v1",
        "training_replay_output_class_label.v1",
        "training_replay_output_class_probability.v1",
        "training_replay_output_observation.v1",
        "training_replay_request_explain.v1",
        "training_replay_request_predict.v1",
        "training_replay_source_outcome_explain.v1",
    ]
)
EXPECTED_NEGATIVE_CASE_IDS = [
    "d4_replay_request_refit_is_not_public",
    "d4_replay_request_unknown_phase",
    "d4_replay_request_unsorted_envelope_keys",
    "d4_replay_request_source_transplant",
    "d4_replay_request_empty_output_binding_ids",
    "d4_replay_request_duplicate_output_binding_ids",
    "d4_replay_request_unsorted_output_binding_ids",
    "d4_replay_request_unknown_output_binding_id",
    "d4_replay_outcome_source_ref_transplant",
    "d4_replay_outcome_request_transplant",
    "d4_replay_outcome_schema_rebinding_forbidden",
    "d4_replay_outcome_cache_store_forbidden",
    "d4_replay_outcome_unknown_producer_port",
    "d4_replay_outcome_missing_v2_producer_port",
    "d4_replay_outcome_rejects_legacy_bound_output",
    "d4_replay_outcome_aggregation_transplant",
    "d4_replay_outcome_non_final_partition",
    "d4_replay_outcome_non_null_fold",
    "d4_replay_outcome_duplicate_aggregated_unit",
    "d4_replay_outcome_duplicate_sample_across_blocks",
    "d4_replay_outcome_alien_sample",
    "d4_replay_outcome_wrong_lineage_controller",
    "d4_replay_outcome_invalid_data_shape_fingerprint",
    "d4_replay_outcome_lineage_plugin_version_requires_plugin",
    "d4_replay_outcome_empty_lineage_metric_key",
    "d4_replay_outcome_observation_count_drift",
    "d4_replay_outcome_result_count_drift",
    "d4_replay_outcome_lineage_count_drift",
    "d4_replay_outcome_prediction_count_drift",
    "d4_replay_outcome_controller_count_drift",
    "d4_replay_outcome_requested_binding_absent",
    "d4_replay_outcome_duplicate_binding",
    "d4_replay_outcome_unsorted_bindings",
    "d4_replay_outcome_unsorted_warnings",
    "d4_replay_outcome_blank_warning",
    "d4_replay_outcome_nested_diagnostics",
    "d4_replay_explanation_unknown_target",
    "d4_replay_explanation_blank_method",
    "d4_replay_explanation_raw_feature_payload",
    "d4_replay_envelopes_missing_key",
    "d4_replay_envelopes_extra_key",
    "d4_replay_envelopes_stale_relation",
    "d4_replay_envelopes_plan_rebinding",
    "d4_replay_envelopes_feature_set_rebinding",
    "d4_replay_envelopes_missing_feature_set_id",
    "d4_replay_relations_sample_multiple_targets",
    "d4_replay_relations_combo_self_component",
    "d4_replay_relations_invalid_identifier",
    "d4_replay_probability_outside_unit_interval",
    "d4_replay_probability_not_on_simplex",
    "d4_replay_class_label_negative_index",
    "d4_replay_class_label_out_of_vocabulary",
    "d4_replay_class_label_fractional_index",
    "d4_replay_observation_binding_requires_observation_predictions",
    "d4_replay_legacy_port_omission_ambiguous",
    "d4_replay_legacy_port_omission_zero_prediction_ports",
    "d4_replay_legacy_wrapper_rejects_explicit_port",
    "d4_score_set_v2_duplicate_port_aware_key",
]


def resolve_replay_producer_port(
    plan: dict[str, Any], producer_node: str, producer_port: Any, label: str
) -> str:
    """Resolve the additive D5a port with the frozen D4 legacy fallback."""

    nodes = execution_plan_graph_nodes(plan, label)
    require(
        producer_node in nodes,
        f"{label}.producer_node `{producer_node}` is absent from source plan",
    )
    outputs = nodes[producer_node].get("ports", {}).get("outputs")
    require(isinstance(outputs, list), f"{label} producer output ports are absent")
    prediction_ports = [
        port.get("name") for port in outputs if port.get("kind") == "prediction"
    ]
    if producer_port is None:
        require(
            len(prediction_ports) == 1,
            f"{label}.producer_port legacy omission requires exactly one prediction port",
        )
        return prediction_ports[0]
    _validated_non_blank(producer_port, f"{label}.producer_port")
    matches = [port for port in outputs if port.get("name") == producer_port]
    require(
        len(matches) == 1 and matches[0].get("kind") == "prediction",
        f"{label}.producer_port `{producer_port}` is not a unique prediction output",
    )
    return producer_port


def validate_replay_classification_values(
    values: list[list[Any]], binding: dict[str, Any], label: str
) -> None:
    if binding["prediction_kind"] == "class_probability":
        for row_index, row in enumerate(values):
            cursor = 0
            for target_index, labels in enumerate(binding["class_labels"]):
                probabilities = row[cursor : cursor + len(labels)]
                require(
                    all(0.0 <= float(value) <= 1.0 for value in probabilities),
                    f"{label}[{row_index}] target {target_index} probability is outside [0,1]",
                )
                total = 0.0
                for probability in probabilities:
                    total += float(probability)
                require(
                    abs(total - 1.0) <= 1e-12,
                    f"{label}[{row_index}] target {target_index} does not lie on the probability simplex",
                )
                cursor += len(labels)
    elif binding["prediction_kind"] == "class_label":
        require(
            all(bool(labels) for labels in binding["class_labels"]),
            f"{label} class_label replay requires an explicit vocabulary per target",
        )
        for row_index, row in enumerate(values):
            for target_index, (value, labels) in enumerate(
                zip(row, binding["class_labels"])
            ):
                require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and float(value).is_integer()
                    and 0 <= int(value) < len(labels),
                    f"{label}[{row_index}][{target_index}] class_label must be a zero-based vocabulary index",
                )


def validate_bound_prediction_block(
    value: Any,
    binding: dict[str, Any],
    label: str,
    *,
    kind: str,
    plan: dict[str, Any] | None = None,
) -> None:
    required_by_kind = {
        "prediction": {"producer_node", "partition", "fold_id", "sample_ids", "values"},
        "observation": {
            "producer_node",
            "partition",
            "fold_id",
            "observation_ids",
            "values",
        },
        "aggregated": {
            "producer_node",
            "partition",
            "fold_id",
            "level",
            "unit_ids",
            "values",
        },
    }
    optional_by_kind = {
        "prediction": {"prediction_id", "producer_port", "target_names"},
        "observation": {
            "prediction_id",
            "producer_port",
            "weights",
            "target_names",
        },
        "aggregated": {"prediction_id", "producer_port", "target_names"},
    }
    require(kind in required_by_kind, f"{label} block kind is invalid")
    block = require_exact_keys(
        value, required_by_kind[kind], optional_by_kind[kind], label
    )
    require_optional_non_empty_string(
        block.get("prediction_id"), f"{label}.prediction_id"
    )
    require(
        block.get("producer_node") == binding["node_id"],
        f"{label}.producer_node must match OutputBinding.node_id",
    )
    if plan is not None:
        producer_port = resolve_replay_producer_port(
            plan,
            block["producer_node"],
            block.get("producer_port"),
            label,
        )
        require(
            producer_port == binding["port_name"],
            f"{label}.producer_port must match OutputBinding.port_name",
        )
    elif "producer_port" in block:
        require(
            block["producer_port"] == binding["port_name"],
            f"{label}.producer_port must match OutputBinding.port_name",
        )
    require(
        block.get("partition") in {"train", "validation", "test", "final"},
        f"{label}.partition is invalid",
    )
    require_optional_identifier(block.get("fold_id"), f"{label}.fold_id")
    expected_columns = expected_output_columns(binding)
    require(
        block.get("target_names") == expected_columns,
        f"{label}.target_names must match OutputBinding column order",
    )

    if kind == "prediction":
        identifiers = block.get("sample_ids")
    elif kind == "observation":
        identifiers = block.get("observation_ids")
    else:
        level = block.get("level")
        require(
            level == binding["prediction_level"],
            f"{label}.level must match OutputBinding.prediction_level",
        )
        units = block.get("unit_ids")
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
        block.get("values"),
        len(identifiers),
        len(expected_columns),
        f"{label}.values",
    )
    validate_replay_classification_values(block["values"], binding, f"{label}.values")
    if kind == "observation" and "weights" in block:
        weights = block["weights"]
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


def validate_bound_output(
    value: Any, label: str, *, plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    output = require_exact_keys(
        value,
        {"binding", "predictions", "observation_predictions", "aggregated_predictions"},
        {"schema_version"},
        label,
    )
    if "schema_version" in output:
        require(
            output["schema_version"] == 2,
            f"{label}.schema_version must be 2 for a port-explicit bound output",
        )
    binding = validate_output_binding(output["binding"], f"{label}.binding")
    if plan is not None:
        validate_output_binding_against_plan(binding, plan, f"{label}.binding")
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
        family_units: set[Any] = set()
        for index, block in enumerate(blocks):
            if output.get("schema_version") == 2:
                require(
                    "producer_port" in block,
                    f"{label}.{field}[{index}].producer_port is required in v2",
                )
            else:
                require(
                    "producer_port" not in block,
                    f"{label}.{field}[{index}] legacy bound output must not contain producer_port",
                )
            validate_bound_prediction_block(
                block,
                binding,
                f"{label}.{field}[{index}]",
                kind=kind,
                plan=plan,
            )
            if kind == "prediction":
                coordinates = set(block["sample_ids"])
            elif kind == "observation":
                coordinates = set(block["observation_ids"])
            else:
                coordinates = {
                    (unit["level"], unit["id"]) for unit in block["unit_ids"]
                }
            require(
                family_units.isdisjoint(coordinates),
                f"{label}.{field} has duplicate final unit across blocks",
            )
            family_units.update(coordinates)
            require(
                block["partition"] == "final" and block["fold_id"] is None,
                f"{label}.{field}[{index}] forward replay blocks must use final partition and null fold",
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


def validate_replay_request(
    value: Any,
    label: str,
    *,
    source_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "request_id",
        "source_outcome_fingerprint",
        "phase",
        "data_envelope_keys",
        "output_binding_ids",
        "request_fingerprint",
    }
    request = require_exact_keys(value, fields, set(), label)
    validate_strict_json_value(request, label)
    require_version_one(request["schema_version"], label)
    require_identifier(request["request_id"], f"{label}.request_id")
    require_sha256(
        request["source_outcome_fingerprint"],
        f"{label}.source_outcome_fingerprint",
    )
    require(
        request["phase"] in {"PREDICT", "EXPLAIN"},
        f"{label}.phase must be PREDICT or EXPLAIN",
    )
    validate_ordered_unique_strings(
        request["data_envelope_keys"],
        f"{label}.data_envelope_keys",
        require_non_empty=True,
    )
    for index, key in enumerate(request["data_envelope_keys"]):
        _validated_non_blank(key, f"{label}.data_envelope_keys[{index}]")
    require(
        request["data_envelope_keys"] == sorted(request["data_envelope_keys"]),
        f"{label}.data_envelope_keys must be sorted",
    )
    output_ids = validate_ordered_unique_strings(
        request["output_binding_ids"],
        f"{label}.output_binding_ids",
        require_non_empty=True,
    )
    for index, binding_id in enumerate(output_ids):
        require_identifier(binding_id, f"{label}.output_binding_ids[{index}]")
    require(
        output_ids == sorted(output_ids),
        f"{label}.output_binding_ids must be sorted",
    )
    require_sha256(request["request_fingerprint"], f"{label}.request_fingerprint")
    require(
        request["request_fingerprint"]
        == w10_fingerprint_without(request, "request_fingerprint"),
        f"{label}.request_fingerprint does not match TCV1 request content",
    )
    if source_outcome is not None:
        require(
            request["source_outcome_fingerprint"]
            == source_outcome["outcome_fingerprint"],
            f"{label} does not bind the supplied source outcome",
        )
        require(
            request["phase"] in source_outcome["replayable_phases"],
            f"{label} source outcome does not advertise {request['phase']}",
        )
        source_keys = sorted(
            f"{binding['node_id']}.{binding['input_name']}"
            for bindings in source_outcome["effective_plan"]["campaign"]
            .get("data_bindings", {})
            .values()
            for binding in bindings
        )
        require(
            request["data_envelope_keys"] == source_keys,
            f"{label}.data_envelope_keys do not exactly cover source plan bindings",
        )
        source_output_ids = {
            output["binding"]["binding_id"] for output in source_outcome["outputs"]
        }
        require(
            set(output_ids) <= source_output_ids,
            f"{label}.output_binding_ids reference an unknown source output",
        )
    return request


def validate_replay_training_outcome_ref(
    value: Any, source_outcome: dict[str, Any], label: str
) -> dict[str, Any]:
    fields = {
        "outcome_id",
        "outcome_fingerprint",
        "training_request_fingerprint",
        "effective_plan_fingerprint",
        "execution_bundle_id",
        "execution_bundle_fingerprint",
        "output_binding_fingerprints",
        "training_influence_fingerprint",
        "data_identities_fingerprint",
    }
    reference = require_exact_keys(value, fields, set(), label)
    require_identifier(reference["outcome_id"], f"{label}.outcome_id")
    for field in fields - {
        "outcome_id",
        "execution_bundle_id",
        "output_binding_fingerprints",
    }:
        require_sha256(reference[field], f"{label}.{field}")
    require_identifier(reference["execution_bundle_id"], f"{label}.execution_bundle_id")
    validate_ordered_unique_strings(
        reference["output_binding_fingerprints"],
        f"{label}.output_binding_fingerprints",
        require_non_empty=True,
    )
    expected = {
        "outcome_id": source_outcome["outcome_id"],
        "outcome_fingerprint": source_outcome["outcome_fingerprint"],
        "training_request_fingerprint": source_outcome["training_request_fingerprint"],
        "effective_plan_fingerprint": source_outcome["effective_plan_fingerprint"],
        "execution_bundle_id": source_outcome["execution_bundle"]["bundle_id"],
        "execution_bundle_fingerprint": dagml_tcv1_sha256(
            source_outcome["execution_bundle"]
        ),
        "output_binding_fingerprints": [
            output["binding"]["binding_fingerprint"]
            for output in source_outcome["outputs"]
        ],
        "training_influence_fingerprint": source_outcome["training_influence"][
            "manifest_fingerprint"
        ],
        "data_identities_fingerprint": dagml_tcv1_sha256(
            source_outcome["data_identities"]
        ),
    }
    require(
        reference == expected,
        f"{label} does not equal the complete source TrainingOutcomeRef",
    )
    return reference


def replay_relation_fingerprint(
    relations: Any, label: str = "coordinator_relations"
) -> str:
    """Validate and fingerprint a coordinator relation set like DAG-ML Rust."""

    relation_set = require_exact_keys(relations, {"records"}, set(), label)
    records = relation_set["records"]
    require(
        isinstance(records, list) and bool(records),
        f"{label}.records must be a non-empty array",
    )
    canonical: list[dict[str, Any]] = []
    observation_samples: dict[str, str] = {}
    effective_units: dict[str, str] = {}
    sample_targets: dict[str, str] = {}
    sample_groups: dict[str, str] = {}

    for index, value in enumerate(records):
        record_label = f"{label}.records[{index}]"
        source = require_exact_keys(
            value, _RELATION_REQUIRED_FIELDS, _RELATION_OPTIONAL_FIELDS, record_label
        )
        unit_level = source.get("unit_level", "observation")
        require(
            unit_level in {"physical_sample", "source_sample", "observation", "combo"},
            f"{record_label}.unit_level is invalid",
        )
        observation_id = _validated_identifier(
            source["observation_id"], f"{record_label}.observation_id"
        )
        sample_id = _validated_identifier(
            source["sample_id"], f"{record_label}.sample_id"
        )
        require(
            observation_id not in observation_samples,
            f"{label} contains duplicate observation `{observation_id}`",
        )
        observation_samples[observation_id] = sample_id

        for field in ("rep_id", "target_id", "group_id", "origin_sample_id"):
            _optional_relation_validated_identifier(
                source.get(field), f"{record_label}.{field}"
            )
        for field in ("unit_id", "source_id", "derived_unit_id", "quality_flag"):
            _optional_relation_text(source.get(field), f"{record_label}.{field}")

        component_ids = source.get("component_observation_ids", [])
        require(
            isinstance(component_ids, list)
            and len(component_ids) == len(set(component_ids)),
            f"{record_label}.component_observation_ids must be unique",
        )
        for component_id in component_ids:
            _validated_identifier(
                component_id, f"{record_label}.component_observation_ids"
            )
        if unit_level != "combo":
            require(
                not component_ids,
                f"{record_label} has components but is not a combo relation",
            )

        weight = source.get("sample_influence_weight")
        if weight is not None:
            require(
                isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(weight)
                and weight > 0.0,
                f"{record_label}.sample_influence_weight must be finite and positive",
            )
        is_augmented = source.get("is_augmented", False)
        excluded = source.get("excluded", False)
        require(isinstance(is_augmented, bool), f"{record_label}.is_augmented")
        require(isinstance(excluded, bool), f"{record_label}.excluded")
        metadata = source.get("metadata", {})
        tags = source.get("tags", [])
        require(isinstance(metadata, dict), f"{record_label}.metadata must be object")
        require(
            isinstance(tags, list) and len(tags) == len(set(tags)),
            f"{record_label}.tags must be a unique array",
        )
        for tag in tags:
            _validated_non_blank(tag, f"{record_label}.tags")

        unit_id = source.get("unit_id")
        if unit_id is not None:
            effective_unit_id = unit_id
        elif unit_level == "physical_sample":
            effective_unit_id = sample_id
        elif unit_level == "source_sample":
            source_id = source.get("source_id")
            require(
                source_id is not None,
                f"{record_label} source_sample relation requires source_id",
            )
            effective_unit_id = f"{sample_id}::{source_id}"
        elif unit_level == "combo":
            derived_unit_id = source.get("derived_unit_id")
            require(
                derived_unit_id is not None,
                f"{record_label} combo relation requires derived_unit_id",
            )
            effective_unit_id = derived_unit_id
        else:
            effective_unit_id = observation_id
        require(
            effective_unit_id not in effective_units,
            f"{label} relations `{effective_units.get(effective_unit_id)}` and "
            f"`{observation_id}` share effective unit `{effective_unit_id}`",
        )
        effective_units[effective_unit_id] = observation_id

        target_id = source.get("target_id")
        if target_id is not None:
            require(
                sample_targets.get(sample_id, target_id) == target_id,
                f"{label} sample `{sample_id}` maps to multiple targets",
            )
            sample_targets[sample_id] = target_id
        group_id = source.get("group_id")
        if group_id is not None:
            require(
                sample_groups.get(sample_id, group_id) == group_id,
                f"{label} sample `{sample_id}` maps to multiple groups",
            )
            sample_groups[sample_id] = group_id

        canonical_record: dict[str, Any] = {
            "effective_unit_id": effective_unit_id,
            "unit_level": unit_level,
            "unit_id": unit_id,
            "observation_id": observation_id,
            "sample_id": sample_id,
            "source_id": source.get("source_id"),
            "rep_id": source.get("rep_id"),
            "target_id": target_id,
            "group_id": group_id,
            "origin_sample_id": source.get("origin_sample_id"),
            "derived_unit_id": source.get("derived_unit_id"),
            "component_observation_ids": component_ids,
            "sample_influence_weight": weight,
            "quality_flag": source.get("quality_flag"),
            "is_augmented": is_augmented,
        }
        if excluded:
            canonical_record["excluded"] = True
        if metadata:
            canonical_record["metadata"] = _sorted_json_value(metadata)
        if tags:
            canonical_record["tags"] = tags
        canonical.append(canonical_record)

    for index, source in enumerate(records):
        if source.get("unit_level", "observation") != "combo":
            continue
        record_label = f"{label}.records[{index}]"
        observation_id = source["observation_id"]
        sample_id = source["sample_id"]
        component_ids = source.get("component_observation_ids", [])
        require(bool(component_ids), f"{record_label} combo has no components")
        origin_sample_id = source.get("origin_sample_id")
        require(
            origin_sample_id is None or origin_sample_id == sample_id,
            f"{record_label} combo origin differs from its sample",
        )
        for component_id in component_ids:
            require(
                component_id != observation_id,
                f"{record_label} combo cannot list itself as a component",
            )
            require(
                component_id in observation_samples,
                f"{record_label} references missing component `{component_id}`",
            )
            require(
                observation_samples[component_id] == sample_id,
                f"{record_label} component `{component_id}` belongs to another sample",
            )

    canonical.sort(
        key=lambda record: (
            record["effective_unit_id"],
            record["observation_id"],
            record["sample_id"],
        )
    )
    return legacy_serde_json_sha256(canonical)


def validate_replay_envelopes(
    value: Any,
    request: dict[str, Any],
    source_outcome: dict[str, Any],
    identities: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    fixture = require_exact_keys(value, {"schema_version", "envelopes"}, set(), label)
    validate_strict_json_value(fixture, label)
    require_version_one(fixture["schema_version"], label)
    envelopes = fixture["envelopes"]
    require(isinstance(envelopes, dict), f"{label}.envelopes must be an object")
    keys = list(envelopes)
    require(keys == sorted(keys), f"{label}.envelope keys must be sorted")
    require(
        keys == request["data_envelope_keys"],
        f"{label}.envelope keys do not exactly cover ReplayRequest",
    )
    identity_by_key = {identity["requirement_key"]: identity for identity in identities}
    require(
        list(identity_by_key) == keys,
        f"{label} identities do not exactly cover envelope keys in order",
    )
    bindings = {
        f"{binding['node_id']}.{binding['input_name']}": binding
        for values in source_outcome["effective_plan"]["campaign"]
        .get("data_bindings", {})
        .values()
        for binding in values
    }
    require(keys == sorted(bindings), f"{label} envelopes do not cover source plan")
    for key, envelope in envelopes.items():
        envelope_label = f"{label}.envelopes[{key}]"
        validate_envelope(envelope, envelope_label)
        for field in (
            "relation_fingerprint",
            "data_content_fingerprint",
            "target_content_fingerprint",
        ):
            require_sha256(envelope.get(field), f"{envelope_label}.{field}")
        require(
            envelope["plan_fingerprint"] == legacy_serde_json_sha256(envelope["plan"]),
            f"{envelope_label}.plan_fingerprint does not match plan content",
        )
        relations = envelope.get("coordinator_relations")
        require(relations is not None, f"{envelope_label} requires relations")
        require(
            envelope["relation_fingerprint"]
            == replay_relation_fingerprint(relations, f"{envelope_label}.relations"),
            f"{envelope_label}.relation_fingerprint does not match coordinator_relations",
        )
        binding = bindings[key]
        require(
            envelope["schema_fingerprint"] == binding["schema_fingerprint"]
            and envelope["plan_fingerprint"] == binding["plan_fingerprint"],
            f"{envelope_label} schema/plan rebinding is forbidden",
        )
        require(
            envelope["plan"].get("output_representation")
            == binding["output_representation"],
            f"{envelope_label} output representation changed",
        )
        envelope_sources = sorted(
            {
                step.get("source_id")
                for step in envelope["plan"].get("steps", [])
                if step.get("source_id") is not None
            }
        )
        require(
            envelope_sources == binding["source_ids"],
            f"{envelope_label} source ids changed",
        )
        metadata = envelope.get("metadata", {})
        require(isinstance(metadata, dict), f"{envelope_label}.metadata must be object")
        require(
            "feature_set_id" in metadata
            and metadata["feature_set_id"] == binding["feature_set_id"],
            f"{envelope_label} feature_set_id changed or is missing",
        )
        identity = identity_by_key[key]
        for field in (
            "schema_fingerprint",
            "plan_fingerprint",
            "relation_fingerprint",
            "data_content_fingerprint",
            "target_content_fingerprint",
        ):
            require(
                identity[field] == envelope[field],
                f"{envelope_label} does not attest current identity {field}",
            )
    return fixture


def validate_replay_output_cohort(
    outputs: list[dict[str, Any]], envelope_fixture: dict[str, Any], label: str
) -> None:
    """Reject alien or duplicate units while preserving current-cohort membership.

    D4 freezes the safe B0 claim only: every emitted identity belongs to the
    union of the supplied current coordinator relations.  Exact transitive row
    coverage across multi-source/missingness cases is deliberately deferred to
    D4.1.  Relation ``excluded`` is training-only and therefore does not remove
    a unit from this replay membership set.
    """

    envelopes = envelope_fixture["envelopes"]
    relation_sets = [
        envelope["coordinator_relations"] for envelope in envelopes.values()
    ]
    require(bool(relation_sets), f"{label} has no coordinator relations")
    for index, relations in enumerate(relation_sets):
        replay_relation_fingerprint(relations, f"{label}.relations[{index}]")
    # Relation exclusion is training-only in DAG-ML; final replay still predicts
    # every current-cohort unit, including records marked excluded.
    records = [record for relations in relation_sets for record in relations["records"]]
    expected = {
        "observation": {record["observation_id"] for record in records},
        "target": {
            record["target_id"] for record in records if record.get("target_id")
        },
        "group": {record["group_id"] for record in records if record.get("group_id")},
    }
    for output_index, output in enumerate(outputs):
        output_label = f"{label}.outputs[{output_index}]"
        unit_level = output["binding"].get("unit_level")
        if unit_level == "source_sample":
            sample_units = {
                f"{record['sample_id']}::{record['source_id']}"
                for record in records
                if record.get("source_id") is not None
            }
        elif unit_level == "observation":
            sample_units = expected["observation"]
        elif unit_level == "combo":
            sample_units = {
                record["derived_unit_id"]
                for record in records
                if record.get("derived_unit_id") is not None
            }
        else:
            sample_units = {record["sample_id"] for record in records}
        if output["predictions"]:
            actual = {
                sample_id
                for block in output["predictions"]
                for sample_id in block["sample_ids"]
            }
            require(
                actual <= sample_units,
                f"{output_label}.predictions contains an id outside current cohort",
            )
        if output["observation_predictions"]:
            actual = {
                observation_id
                for block in output["observation_predictions"]
                for observation_id in block["observation_ids"]
            }
            require(
                actual <= expected["observation"],
                f"{output_label}.observation_predictions contains an id outside current cohort",
            )
        if output["aggregated_predictions"]:
            level = output["binding"]["prediction_level"]
            actual = {
                unit["id"]
                for block in output["aggregated_predictions"]
                for unit in block["unit_ids"]
            }
            require(
                actual <= (sample_units if level == "sample" else expected[level]),
                f"{output_label}.aggregated_predictions contains an id outside current cohort",
            )


def validate_replay_outcome(
    value: Any,
    label: str,
    *,
    request: dict[str, Any] | None = None,
    source_outcome: dict[str, Any] | None = None,
    envelope_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "outcome_id",
        "run_id",
        "source_training_outcome",
        "replay_request_id",
        "replay_request_fingerprint",
        "input_data_identities",
        "bundle_id",
        "plan_id",
        "phase",
        "result_count",
        "lineage_record_count",
        "prediction_block_count",
        "observation_prediction_block_count",
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
    for field in ("outcome_id", "run_id", "replay_request_id", "bundle_id"):
        require_identifier(outcome[field], f"{label}.{field}")
    require_sha256(
        outcome["replay_request_fingerprint"],
        f"{label}.replay_request_fingerprint",
    )
    require_non_empty_string(outcome["plan_id"], f"{label}.plan_id")
    require(outcome["phase"] in {"PREDICT", "EXPLAIN"}, f"{label}.phase invalid")
    for field in (
        "result_count",
        "lineage_record_count",
        "prediction_block_count",
        "observation_prediction_block_count",
        "aggregated_prediction_block_count",
        "explanation_block_count",
        "controller_count",
    ):
        require_non_negative_int(outcome[field], f"{label}.{field}")
    require(
        outcome["prediction_cache_store"] is False,
        f"{label}.prediction_cache_store must be false",
    )
    identities = [
        validate_w10_data_identity(identity, f"{label}.input_data_identities[{index}]")
        for index, identity in enumerate(outcome["input_data_identities"])
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    require(
        identity_keys == sorted(set(identity_keys)) and bool(identity_keys),
        f"{label}.input_data_identities must be non-empty, sorted and unique",
    )
    plan = source_outcome["effective_plan"] if source_outcome is not None else None
    outputs = outcome["outputs"]
    require(isinstance(outputs, list), f"{label}.outputs must be an array")
    validated_outputs = [
        validate_bound_output(output, f"{label}.outputs[{index}]", plan=plan)
        for index, output in enumerate(outputs)
    ]
    for index, output in enumerate(validated_outputs):
        require(
            output.get("schema_version") == 2,
            f"{label}.outputs[{index}].schema_version must be 2 for public training replay",
        )
    prediction_count = sum(len(output["predictions"]) for output in validated_outputs)
    observation_count = sum(
        len(output["observation_predictions"]) for output in validated_outputs
    )
    aggregated_count = sum(
        len(output["aggregated_predictions"]) for output in validated_outputs
    )
    require(
        outcome["prediction_block_count"] == prediction_count,
        f"{label}.prediction_block_count does not match payload",
    )
    require(
        outcome["observation_prediction_block_count"] == observation_count,
        f"{label}.observation_prediction_block_count does not match payload",
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
            {"producer_port", "target_name"},
            explanation_label,
        )
        require(
            "producer_port" in explanation,
            f"{explanation_label}.producer_port is required for public training replay",
        )
        require_identifier(
            explanation["producer_node"], f"{explanation_label}.producer_node"
        )
        _validated_non_blank(explanation["method"], f"{explanation_label}.method")
        if "target_name" in explanation:
            _validated_non_blank(
                explanation["target_name"], f"{explanation_label}.target_name"
            )
        validate_strict_json_value(
            explanation["payload"], f"{explanation_label}.payload"
        )
        require(
            not contains_runtime_handle(explanation["payload"]),
            f"{explanation_label}.payload must not contain runtime handles",
        )
        require(
            not _contains_raw_feature_payload(explanation["payload"]),
            f"{explanation_label}.payload must not embed raw feature data",
        )
        if plan is not None:
            resolve_replay_producer_port(
                plan,
                explanation["producer_node"],
                explanation.get("producer_port"),
                explanation_label,
            )
    require(
        outcome["explanation_block_count"] == len(explanations),
        f"{label}.explanation_block_count does not match payload",
    )
    if outcome["phase"] == "PREDICT":
        require(bool(outputs), f"{label} PREDICT must emit at least one output")
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
    record_ids = [record["record_id"] for record in lineage_records]
    require(
        record_ids == sorted(set(record_ids)),
        f"{label}.lineage must be sorted by unique record_id",
    )
    producer_nodes = {output["binding"]["node_id"] for output in validated_outputs} | {
        explanation["producer_node"] for explanation in explanations
    }
    lineage_nodes = {record["node_id"] for record in lineage_records}
    require(producer_nodes <= lineage_nodes, f"{label} emitted payload lacks lineage")
    require(
        outcome["controller_count"]
        == len({record["controller_id"] for record in lineage_records}),
        f"{label}.controller_count does not match lineage controllers",
    )
    binding_ids = [output["binding"]["binding_id"] for output in validated_outputs]
    require(
        binding_ids == sorted(set(binding_ids)),
        f"{label}.outputs must be sorted by unique binding_id",
    )

    if source_outcome is not None:
        validate_replay_training_outcome_ref(
            outcome["source_training_outcome"],
            source_outcome,
            f"{label}.source_training_outcome",
        )
        require(
            outcome["bundle_id"] == source_outcome["execution_bundle"]["bundle_id"]
            and outcome["plan_id"] == source_outcome["effective_plan"]["id"],
            f"{label} plan/bundle ids do not match source outcome",
        )
        source_bindings = {
            output["binding"]["binding_id"]: output["binding"]
            for output in source_outcome["outputs"]
        }
        for index, output in enumerate(validated_outputs):
            binding = output["binding"]
            validate_output_binding_against_plan(
                binding, source_outcome["effective_plan"], f"{label}.outputs[{index}]"
            )
            require(
                source_bindings.get(binding["binding_id"]) == binding,
                f"{label}.outputs[{index}] binding does not match source outcome",
            )
        node_plans = source_outcome["effective_plan"]["node_plans"]
        requested_bindings = {
            binding_id: source_bindings[binding_id]
            for binding_id in (request["output_binding_ids"] if request else [])
        }
        if request is not None:
            validate_replay_request(
                request, f"{label}.request", source_outcome=source_outcome
            )
            require(
                outcome["replay_request_id"] == request["request_id"]
                and outcome["replay_request_fingerprint"]
                == request["request_fingerprint"]
                and outcome["phase"] == request["phase"],
                f"{label} does not match ReplayRequest",
            )
            if outcome["phase"] == "PREDICT":
                require(
                    binding_ids == request["output_binding_ids"],
                    f"{label}.outputs do not exactly cover ReplayRequest bindings",
                )
            else:
                require(
                    set(binding_ids) <= set(request["output_binding_ids"]),
                    f"{label}.outputs contain an unrequested binding",
                )
            requested_coordinates = {
                (binding["node_id"], binding["port_name"]): binding
                for binding in requested_bindings.values()
            }
            for index, explanation in enumerate(explanations):
                port = resolve_replay_producer_port(
                    plan,
                    explanation["producer_node"],
                    explanation.get("producer_port"),
                    f"{label}.explanations[{index}]",
                )
                coordinate = (explanation["producer_node"], port)
                require(
                    coordinate in requested_coordinates,
                    f"{label}.explanations[{index}] does not explain a requested prediction port",
                )
                if "target_name" in explanation:
                    require(
                        explanation["target_name"]
                        in requested_coordinates[coordinate]["target_names"],
                        f"{label}.explanations[{index}].target_name is absent from OutputBinding",
                    )
        closure = execution_plan_transitive_node_ids(
            plan,
            {binding["node_id"] for binding in requested_bindings.values()},
            label,
        )
        require(
            lineage_nodes == closure,
            f"{label}.lineage does not exactly cover requested predictor closure",
        )
        records_by_node = {record["node_id"]: record for record in lineage_records}
        require(
            len(records_by_node) == len(lineage_records),
            f"{label}.lineage has duplicate node results",
        )
        for node_id in sorted(closure):
            record = records_by_node[node_id]
            node_plan = node_plans[node_id]
            require(
                outcome["phase"] in node_plan["supported_phases"],
                f"{label}.lineage node `{node_id}` does not support replay phase",
            )
            require(
                record["controller_id"] == node_plan["controller_id"]
                and record["controller_version"] == node_plan["controller_version"]
                and record["params_fingerprint"] == node_plan["params_fingerprint"],
                f"{label}.lineage node `{node_id}` controller/version/params mismatch",
            )
            require(
                record["variant_id"] == source_outcome["selected_variant_id"]
                and record["fold_id"] is None,
                f"{label}.lineage node `{node_id}` variant/fold mismatch",
            )
            expected_inputs = sorted(
                records_by_node[input_node]["record_id"]
                for input_node in node_plan["input_nodes"]
            )
            require(
                record["input_lineage"] == expected_inputs,
                f"{label}.lineage node `{node_id}` input lineage mismatch",
            )
        if envelope_fixture is not None and request is not None:
            validate_replay_envelopes(
                envelope_fixture,
                request,
                source_outcome,
                identities,
                f"{label}.input_envelopes",
            )
            validate_replay_output_cohort(
                validated_outputs, envelope_fixture, f"{label}.current_cohort"
            )
    validate_ordered_unique_strings(
        outcome["warnings"], f"{label}.warnings", require_non_empty=False
    )
    require(
        outcome["warnings"] == sorted(outcome["warnings"]),
        f"{label}.warnings must be sorted",
    )
    for index, warning in enumerate(outcome["warnings"]):
        _validated_non_blank(warning, f"{label}.warnings[{index}]")
    validate_metadata_object(outcome["diagnostics"], f"{label}.diagnostics")
    for key, member in outcome["diagnostics"].items():
        require_non_empty_string(key, f"{label}.diagnostics key")
        require(
            member is None
            or isinstance(member, (bool, str))
            or (
                isinstance(member, (int, float))
                and not isinstance(member, bool)
                and math.isfinite(member)
            ),
            f"{label}.diagnostics[{key}] must be a finite JSON scalar",
        )
    require(
        not contains_runtime_handle(outcome["diagnostics"]),
        f"{label}.diagnostics must not contain runtime handles",
    )
    require_sha256(outcome["outcome_fingerprint"], f"{label}.outcome_fingerprint")
    require(
        outcome["outcome_fingerprint"] == replay_outcome_fingerprint(outcome),
        f"{label}.outcome_fingerprint does not match TCV1 outcome content",
    )
    return outcome


def _load(name: str) -> Any:
    replay_path = FIXTURE_ROOT / name
    return base.load_json(
        replay_path if replay_path.is_file() else TRAINING_FIXTURE_ROOT / name
    )


def _normalize_cache_block_v2(
    block: dict[str, Any], plan: dict[str, Any], *, aggregated: bool
) -> dict[str, Any]:
    """Freeze the V2 stable-json preimage in the future Rust field order."""

    source = copy.deepcopy(block)
    source["producer_port"] = resolve_replay_producer_port(
        plan, source["producer_node"], source.get("producer_port"), "cache.v2"
    )
    fields = (
        (
            "prediction_id",
            "producer_node",
            "producer_port",
            "partition",
            "fold_id",
            "level",
            "unit_ids",
            "values",
            "target_names",
        )
        if aggregated
        else (
            "prediction_id",
            "producer_node",
            "producer_port",
            "partition",
            "fold_id",
            "sample_ids",
            "values",
            "target_names",
        )
    )
    return {field: source[field] for field in fields if field in source}


def validate_score_set_v2(
    value: Any, plan: dict[str, Any], label: str = "ScoreSetV2"
) -> dict[str, Any]:
    """Validate port-aware report identity and reject only full-key collisions."""

    score_set = require_exact_keys(
        value,
        {"schema_version", "plan_id", "reports"},
        {"selection_metric"},
        label,
    )
    require(score_set["schema_version"] == 2, f"{label}.schema_version must be 2")
    require(score_set["plan_id"] == plan["id"], f"{label}.plan_id mismatch")
    reports = score_set["reports"]
    require(
        isinstance(reports, list),
        f"{label}.reports must be an array",
    )
    keys: list[tuple[Any, ...]] = []
    required = {
        "producer_node",
        "producer_port",
        "partition",
        "level",
        "row_count",
        "target_width",
        "metrics",
    }
    optional = {
        "prediction_id",
        "variant_id",
        "variant_label",
        "fold_id",
        "target_names",
    }
    for index, report in enumerate(reports):
        report_label = f"{label}.reports[{index}]"
        report = require_exact_keys(report, required, optional, report_label)
        require_identifier(report["producer_node"], f"{report_label}.producer_node")
        port = resolve_replay_producer_port(
            plan,
            report["producer_node"],
            report["producer_port"],
            report_label,
        )
        require_optional_identifier(
            report.get("variant_id"), f"{report_label}.variant_id"
        )
        require_optional_identifier(report.get("fold_id"), f"{report_label}.fold_id")
        require(
            report["partition"] in {"train", "validation", "test", "final"},
            f"{report_label}.partition is invalid",
        )
        require(
            report["level"] in {"observation", "sample", "target", "group"},
            f"{report_label}.level is invalid",
        )
        for field in ("row_count", "target_width"):
            require(
                isinstance(report[field], int)
                and not isinstance(report[field], bool)
                and report[field] > 0,
                f"{report_label}.{field} must be a positive integer",
            )
        target_names = report.get("target_names", [])
        require(
            isinstance(target_names, list)
            and (not target_names or len(target_names) == report["target_width"]),
            f"{report_label}.target_names must be empty or match target_width",
        )
        metrics = report["metrics"]
        require(
            isinstance(metrics, dict) and bool(metrics),
            f"{report_label}.metrics must be a non-empty object",
        )
        for name, metric in metrics.items():
            require(
                isinstance(name, str) and bool(name.strip()),
                f"{report_label}.metrics key must be non-blank",
            )
            require(
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric),
                f"{report_label}.metrics[{name}] must be finite",
            )
        key = (
            report["producer_node"],
            port,
            report.get("variant_id"),
            report["partition"],
            report.get("fold_id"),
            report["level"],
        )
        keys.append(key)
    require(len(keys) == len(set(keys)), f"{label} has duplicate score report key")
    return score_set


def _expected_score_set_v2(
    source: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    score_set = copy.deepcopy(source)
    score_set["schema_version"] = 2
    reports = []
    order = (
        "prediction_id",
        "producer_node",
        "producer_port",
        "variant_id",
        "variant_label",
        "partition",
        "fold_id",
        "level",
        "row_count",
        "target_width",
        "target_names",
        "metrics",
    )
    for report in score_set["reports"]:
        migrated = copy.deepcopy(report)
        migrated["producer_port"] = resolve_replay_producer_port(
            plan, report["producer_node"], None, "score_set.v2"
        )
        reports.append({field: migrated[field] for field in order if field in migrated})
    score_set["reports"] = reports
    return score_set


def _expected_training_outcome_v2(source: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(source)
    expected["schema_version"] = 2
    expected["outcome_id"] = "training:estimator.refit.port-explicit-v2"
    plan = expected["effective_plan"]
    for output in expected["outputs"]:
        output["schema_version"] = 2
        for field in (
            "predictions",
            "observation_predictions",
            "aggregated_predictions",
        ):
            for block in output[field]:
                block["producer_port"] = output["binding"]["port_name"]
    bundle = expected["execution_bundle"]
    bundle["schema_version"] = 2
    expected["score_set"] = _expected_score_set_v2(expected["score_set"], plan)
    bundle["scores"] = copy.deepcopy(expected["score_set"])
    payload_set = expected["portable_prediction_caches"]
    require(payload_set is not None and bool(payload_set["caches"]), "V2 cache fixture")
    payload_set["schema_version"] = 2
    records = {
        record["requirement_key"]: record for record in bundle["prediction_caches"]
    }
    for payload in payload_set["caches"]:
        payload["format"] = "dag-ml-json-prediction-blocks-v2"
        predictions = [
            _normalize_cache_block_v2(block, plan, aggregated=False)
            for block in payload.get("blocks", [])
        ]
        aggregated = [
            _normalize_cache_block_v2(block, plan, aggregated=True)
            for block in payload.get("aggregated_blocks", [])
        ]
        if "blocks" in payload:
            payload["blocks"] = predictions
        if "aggregated_blocks" in payload:
            payload["aggregated_blocks"] = aggregated
        blocks = [*predictions, *aggregated]
        payload["content_fingerprint"] = legacy_serde_json_sha256(blocks)
        record = records[payload["requirement_key"]]
        record["format"] = "dag-ml-json-prediction-blocks-v2"
        record["content_fingerprint"] = payload["content_fingerprint"]
        by_fold = {block["fold_id"]: block for block in blocks}
        namespace_fingerprints = payload.get("cache_namespace_fingerprints")
        if not namespace_fingerprints:
            namespace_fingerprints = [
                dagml_tcv1_sha256(
                    {
                        "schema_version": 1,
                        "kind": "training-replay-v2-fixture-cache-namespace",
                        "requirement_key": payload["requirement_key"],
                        "cache_id": payload["cache_id"],
                        "fold_id": block["fold_id"],
                        "block_index": index,
                    }
                )
                for index, block in enumerate(blocks)
            ]
        payload["cache_namespace_fingerprints"] = namespace_fingerprints
        record["cache_namespace_fingerprints"] = copy.deepcopy(namespace_fingerprints)
        for block_record in record["blocks"]:
            block_record["content_fingerprint"] = legacy_serde_json_sha256(
                by_fold[block_record["fold_id"]]
            )
    expected["outcome_fingerprint"] = w10_fingerprint_without(
        expected, "outcome_fingerprint"
    )
    return expected


def validate_training_outcome_v2_migration(
    value: Any, source: dict[str, Any], label: str
) -> dict[str, Any]:
    """Prove the non-null cache V2 closure without claiming a runtime writer."""

    validate_strict_json_value(value, label)
    require(not contains_runtime_handle(value), f"{label} contains runtime handles")
    expected = _expected_training_outcome_v2(source)
    require(value == expected, f"{label} is not the exact port-explicit V2 migration")
    for output_index, output in enumerate(value["outputs"]):
        validate_bound_output(
            output,
            f"{label}.outputs[{output_index}]",
            plan=value["effective_plan"],
        )
    validate_score_set_v2(
        value["score_set"], value["effective_plan"], f"{label}.score_set"
    )
    validate_score_set_v2(
        value["execution_bundle"]["scores"],
        value["effective_plan"],
        f"{label}.execution_bundle.scores",
    )
    require(
        value["execution_bundle"]["scores"] == value["score_set"],
        f"{label}.execution_bundle.scores must equal score_set",
    )
    selection = next(
        output["binding"]
        for output in value["outputs"]
        if output["binding"]["binding_id"] == value["selection_output_id"]
    )
    selection_reports = [
        report
        for report in value["score_set"]["reports"]
        if report["producer_node"] == selection["node_id"]
        and report["producer_port"] == selection["port_name"]
        and report["partition"] == "validation"
        and report.get("fold_id") == "avg"
        and report["level"] == selection["prediction_level"]
    ]
    require(
        bool(selection_reports)
        and value["selected_variant_id"]
        in {report.get("variant_id") for report in selection_reports},
        f"{label}.score_set lacks selected validation/avg report coordinate",
    )
    require(
        value["outcome_fingerprint"]
        == w10_fingerprint_without(value, "outcome_fingerprint"),
        f"{label}.outcome_fingerprint mismatch",
    )
    return value


def _schema(schemas: dict[str, dict[str, Any]], schema_id: str) -> dict[str, Any]:
    require(schema_id in schemas, f"missing local JSON Schema {schema_id}")
    return schemas[schema_id]


def validate_schema_contracts(
    schemas: dict[str, dict[str, Any]], registry: Any
) -> None:
    request_schema = _schema(schemas, REQUEST_SCHEMA_ID)
    outcome_schema = _schema(schemas, OUTCOME_SCHEMA_ID)
    require(
        request_schema.get("additionalProperties") is False
        and request_schema.get("properties", {}).get("phase", {}).get("enum")
        == ["PREDICT", "EXPLAIN"],
        "training-replay ReplayRequest schema drifted",
    )
    require(
        outcome_schema.get("additionalProperties") is False
        and outcome_schema.get("properties", {})
        .get("outputs", {})
        .get("items", {})
        .get("$ref")
        == BOUND_OUTPUT_V2_SCHEMA_ID
        and outcome_schema.get("properties", {})
        .get("explanations", {})
        .get("items", {})
        .get("$ref")
        == f"{NODE_RESULT_V2_SCHEMA_ID}#/$defs/explanation_block",
        "training-replay ReplayOutcome V2 payload references drifted",
    )
    node_v1 = _schema(
        schemas,
        "https://github.com/GBeurier/dag-ml/schemas/node_result.v1.schema.json",
    )
    node_v2 = _schema(schemas, NODE_RESULT_V2_SCHEMA_ID)
    for name in (
        "prediction_block",
        "observation_prediction_block",
        "aggregated_prediction_block",
        "explanation_block",
    ):
        require(
            "producer_port"
            not in node_v1.get("$defs", {}).get(name, {}).get("properties", {}),
            f"NodeResult v1 {name} must remain port-absent",
        )
        definition = node_v2.get("$defs", {}).get(name, {})
        require(
            "producer_port" in definition.get("properties", {})
            and "producer_port" in definition.get("required", []),
            f"NodeResult v2 {name} must require producer_port",
        )
    for schema_id in (
        NODE_RESULT_V2_SCHEMA_ID,
        BOUND_OUTPUT_V2_SCHEMA_ID,
        "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_task.v2.schema.json",
        "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_result.v2.schema.json",
        "https://github.com/GBeurier/dag-ml/schemas/execution_bundle.v2.schema.json",
        "https://github.com/GBeurier/dag-ml/schemas/process_adapter_frame.v2.schema.json",
        "https://github.com/GBeurier/dag-ml/schemas/prediction_cache_payload_set.v2.schema.json",
        "https://github.com/GBeurier/dag-ml/schemas/score_set.v2.schema.json",
        "https://github.com/GBeurier/dag-ml/schemas/training_outcome.v2.schema.json",
    ):
        schema = _schema(schemas, schema_id)
        version_schema = (
            schema.get("$defs", {}).get("schema_version", {})
            if schema_id.endswith("process_adapter_frame.v2.schema.json")
            else schema.get("properties", {}).get("schema_version", {})
        )
        require(version_schema.get("const") == 2, f"{schema_id} version drifted")

    fixtures = (
        (
            "request.predict",
            _load("training_replay_request_predict.v1.json"),
            request_schema,
        ),
        (
            "request.explain",
            _load("training_replay_request_explain.v1.json"),
            request_schema,
        ),
        (
            "outcome.predict",
            _load("training_replay_outcome_predict.v1.json"),
            outcome_schema,
        ),
        (
            "outcome.explain",
            _load("training_replay_outcome_explain.v1.json"),
            outcome_schema,
        ),
        (
            "outcome.explain-only",
            _load("training_replay_outcome_explain_only.v1.json"),
            outcome_schema,
        ),
        (
            "training-outcome.v2",
            _load("training_outcome_port_explicit.v2.json"),
            _schema(
                schemas,
                "https://github.com/GBeurier/dag-ml/schemas/training_outcome.v2.schema.json",
            ),
        ),
    )
    for label, document, schema in fixtures:
        validate_draft_2020_instance(document, schema, registry, label)
    bound_output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": BOUND_OUTPUT_V2_SCHEMA_ID,
    }
    for name in (
        "training_replay_output_class_probability.v1.json",
        "training_replay_output_class_label.v1.json",
        "training_replay_output_observation.v1.json",
    ):
        validate_draft_2020_instance(
            _load(name), bound_output_schema, registry, f"bound-output.{name}"
        )
    for index, output in enumerate(
        _load("training_replay_multi_port_outputs.v1.json")["outputs"]
    ):
        validate_draft_2020_instance(
            output, bound_output_schema, registry, f"multi-port.outputs[{index}]"
        )
    envelope_schema = _schema(
        schemas,
        "https://github.com/GBeurier/dag-ml/schemas/coordinator_data_plan_envelope.v1.schema.json",
    )
    for key, envelope in _load("training_replay_input_envelopes.v1.json")[
        "envelopes"
    ].items():
        validate_draft_2020_instance(
            envelope, envelope_schema, registry, f"training-replay envelope.{key}"
        )
    outcome_v2 = _load("training_outcome_port_explicit.v2.json")
    for label, instance, schema_id in (
        (
            "root.execution-bundle-v2",
            outcome_v2["execution_bundle"],
            "https://github.com/GBeurier/dag-ml/schemas/execution_bundle.v2.schema.json",
        ),
        (
            "root.prediction-cache-v2",
            outcome_v2["portable_prediction_caches"],
            "https://github.com/GBeurier/dag-ml/schemas/prediction_cache_payload_set.v2.schema.json",
        ),
        (
            "root.score-set-v2",
            outcome_v2["score_set"],
            "https://github.com/GBeurier/dag-ml/schemas/score_set.v2.schema.json",
        ),
    ):
        validate_draft_2020_instance(
            instance, _schema(schemas, schema_id), registry, label
        )
    protocols = _load("training_port_explicit_protocols.v2.json")
    protocol_schemas = {
        "node_result": "https://github.com/GBeurier/dag-ml/schemas/node_result.v2.schema.json",
        "process_adapter_result": "https://github.com/GBeurier/dag-ml/schemas/process_adapter_frame.v2.schema.json",
        "aggregation_task_observation": "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_task.v2.schema.json",
        "aggregation_task_unit": "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_task.v2.schema.json",
        "aggregation_result_sample": "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_result.v2.schema.json",
        "aggregation_result_unit": "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_result.v2.schema.json",
    }
    for name, schema_id in protocol_schemas.items():
        validate_draft_2020_instance(
            protocols[name], _schema(schemas, schema_id), registry, f"root.{name}"
        )

    legacy_schema = _schema(schemas, LEGACY_OUTCOME_SCHEMA_ID)
    for name in (
        "replay_outcome_predict.v1.json",
        "replay_outcome_class_probability.v1.json",
        "replay_outcome_explain.v1.json",
    ):
        document = base.load_json(ROOT / "examples/fixtures/estimator" / name)
        validate_draft_2020_instance(
            document, legacy_schema, registry, f"legacy.{name}"
        )
        try:
            validate_draft_2020_instance(
                document, outcome_schema, registry, f"legacy-cross.{name}"
            )
        except ContractError:
            pass
        else:
            raise ContractError(f"legacy fixture {name} passed training-replay schema")
    for name in ("predict", "explain"):
        document = _load(f"training_replay_outcome_{name}.v1.json")
        try:
            validate_draft_2020_instance(
                document, legacy_schema, registry, f"public-cross.{name}"
            )
        except ContractError:
            pass
        else:
            raise ContractError(f"training-replay {name} passed legacy schema")


def validate_positive_fixtures() -> None:
    predict_source = validate_training_outcome(
        _load("training_outcome_refit.v1.json"), "predict_source"
    )
    explain_source = validate_training_outcome(
        _load("training_replay_source_outcome_explain.v1.json"), "explain_source"
    )
    validate_training_outcome_v2_migration(
        _load("training_outcome_port_explicit.v2.json"),
        predict_source,
        "training_outcome_port_explicit.v2",
    )
    envelopes = _load("training_replay_input_envelopes.v1.json")
    sources = {"PREDICT": predict_source, "EXPLAIN": explain_source}
    for phase in ("PREDICT", "EXPLAIN"):
        request_document = _load(f"training_replay_request_{phase.lower()}.v1.json")
        outcome_document = _load(f"training_replay_outcome_{phase.lower()}.v1.json")
        request = validate_replay_request(
            request_document,
            f"training-replay ReplayRequest.{phase}",
            source_outcome=sources[phase],
        )
        validate_replay_outcome(
            outcome_document,
            f"training-replay ReplayOutcome.{phase}",
            request=request,
            source_outcome=sources[phase],
            envelope_fixture=envelopes,
        )
        replay_oracle.validate_replay_request(
            request_document,
            f"oracle.training-replay ReplayRequest.{phase}",
            source_outcome=sources[phase],
        )
        replay_oracle.validate_replay_outcome(
            outcome_document,
            f"oracle.training-replay ReplayOutcome.{phase}",
            request=request_document,
            source_outcome=sources[phase],
            envelope_fixture=envelopes,
        )
    for name in (
        "training_replay_output_class_probability.v1.json",
        "training_replay_output_class_label.v1.json",
        "training_replay_output_observation.v1.json",
    ):
        document = _load(name)
        validate_bound_output(document, name, plan=predict_source["effective_plan"])
        replay_oracle.validate_replay_bound_output(
            document, predict_source["effective_plan"], f"oracle.{name}"
        )
    legacy = base.load_json(
        ROOT / "examples/fixtures/estimator/replay_outcome_predict.v1.json"
    )["outputs"][0]
    validate_bound_output(
        legacy, "legacy.single-port", plan=predict_source["effective_plan"]
    )
    replay_oracle.validate_replay_bound_output(
        legacy, predict_source["effective_plan"], "oracle.legacy.single-port"
    )
    multi_port = _load("training_replay_multi_port_outputs.v1.json")
    for index, output in enumerate(multi_port["outputs"]):
        validate_bound_output(
            output,
            f"multi-port.outputs[{index}]",
            plan=multi_port["effective_plan"],
        )
        replay_oracle.validate_replay_bound_output(
            output,
            multi_port["effective_plan"],
            f"oracle.multi-port.outputs[{index}]",
        )
    explain_request = _load("training_replay_request_explain.v1.json")
    explain_only = _load("training_replay_outcome_explain_only.v1.json")
    validate_replay_outcome(
        explain_only,
        "training-replay ReplayOutcome.EXPLAIN-only",
        request=explain_request,
        source_outcome=explain_source,
        envelope_fixture=envelopes,
    )
    replay_oracle.validate_replay_outcome(
        explain_only,
        "oracle.training-replay ReplayOutcome.EXPLAIN-only",
        request=explain_request,
        source_outcome=explain_source,
        envelope_fixture=envelopes,
    )


def validate_negative_fixtures() -> None:
    source = validate_training_outcome(
        _load("training_outcome_refit.v1.json"), "negative.source"
    )
    request = validate_replay_request(
        _load("training_replay_request_predict.v1.json"),
        "negative.request",
        source_outcome=source,
    )
    explain_source = validate_training_outcome(
        _load("training_replay_source_outcome_explain.v1.json"),
        "negative.explain_source",
    )
    explain_request = validate_replay_request(
        _load("training_replay_request_explain.v1.json"),
        "negative.explain_request",
        source_outcome=explain_source,
    )
    envelopes = _load("training_replay_input_envelopes.v1.json")
    identities = _load("training_replay_outcome_predict.v1.json")[
        "input_data_identities"
    ]
    fixture = _load("training_replay_negative_cases.v1.json")
    registry, schemas = base.build_local_schema_registry()
    envelope_schema = _schema(
        schemas,
        "https://github.com/GBeurier/dag-ml/schemas/coordinator_data_plan_envelope.v1.schema.json",
    )
    require_exact_keys(fixture, {"schema_version", "cases"}, set(), "negative_cases")
    require_version_one(fixture["schema_version"], "negative_cases")
    ids: list[str] = []
    for index, case in enumerate(fixture["cases"]):
        label = f"negative_cases[{index}]"
        require_exact_keys(
            case, {"id", "contract", "document", "expected_error"}, set(), label
        )
        case_id = case["id"]
        require_identifier(case_id, f"{label}.id")
        ids.append(case_id)
        if case["contract"] == "training_replay_envelopes":
            for key, envelope in case["document"]["envelopes"].items():
                validate_draft_2020_instance(
                    envelope, envelope_schema, registry, f"{case_id}.{key}"
                )
        for engine in ("production", "oracle"):
            try:
                if case["contract"] == "training_replay_request":
                    if engine == "production":
                        validate_replay_request(
                            case["document"], case_id, source_outcome=source
                        )
                    else:
                        replay_oracle.validate_replay_request(
                            case["document"], case_id, source_outcome=source
                        )
                elif case["contract"] == "training_replay_outcome":
                    if engine == "production":
                        validate_replay_outcome(
                            case["document"],
                            case_id,
                            request=request,
                            source_outcome=source,
                            envelope_fixture=envelopes,
                        )
                    else:
                        replay_oracle.validate_replay_outcome(
                            case["document"],
                            case_id,
                            request=request,
                            source_outcome=source,
                            envelope_fixture=envelopes,
                        )
                elif case["contract"] == "training_replay_explain_outcome":
                    if engine == "production":
                        validate_replay_outcome(
                            case["document"],
                            case_id,
                            request=explain_request,
                            source_outcome=explain_source,
                            envelope_fixture=envelopes,
                        )
                    else:
                        replay_oracle.validate_replay_outcome(
                            case["document"],
                            case_id,
                            request=explain_request,
                            source_outcome=explain_source,
                            envelope_fixture=envelopes,
                        )
                elif case["contract"] == "training_replay_envelopes":
                    if engine == "production":
                        validate_replay_envelopes(
                            case["document"], request, source, identities, case_id
                        )
                    else:
                        replay_oracle.validate_replay_envelopes(
                            case["document"], request, source, identities, case_id
                        )
                elif case["contract"] == "training_replay_bound_output":
                    if engine == "production":
                        validate_bound_output(
                            case["document"],
                            case_id,
                            plan=source["effective_plan"],
                        )
                    else:
                        replay_oracle.validate_replay_bound_output(
                            case["document"], source["effective_plan"], case_id
                        )
                elif case["contract"] == "training_replay_legacy_bound_output":
                    composite = case["document"]
                    if engine == "production":
                        validate_bound_output(
                            composite["output"],
                            case_id,
                            plan=composite["effective_plan"],
                        )
                    else:
                        replay_oracle.validate_replay_bound_output(
                            composite["output"], composite["effective_plan"], case_id
                        )
                elif case["contract"] == "training_score_set_v2":
                    composite = case["document"]
                    if engine == "production":
                        validate_score_set_v2(
                            composite["score_set"], composite["effective_plan"], case_id
                        )
                    else:
                        replay_oracle.validate_score_set_v2(
                            composite["score_set"], composite["effective_plan"], case_id
                        )
                elif case["contract"] == "training_replay_port_resolution":
                    composite = case["document"]
                    resolver = (
                        resolve_replay_producer_port
                        if engine == "production"
                        else replay_oracle.resolve_replay_producer_port
                    )
                    resolver(
                        composite["effective_plan"],
                        composite["producer_node"],
                        composite["producer_port"],
                        case_id,
                    )
                elif case["contract"] == "training_replay_relations":
                    relation_validator = (
                        replay_relation_fingerprint
                        if engine == "production"
                        else replay_oracle.replay_relation_fingerprint
                    )
                    relation_validator(case["document"], case_id)
                else:
                    raise ContractError(f"{case_id} has unknown negative contract")
            except (ContractError, replay_oracle.ContractError) as error:
                require(
                    case["expected_error"].lower() in str(error).lower(),
                    f"{case_id} {engine} failed for unexpected reason: {error}",
                )
            else:
                raise ContractError(f"{case_id} unexpectedly passed {engine}")
    require(len(ids) == len(set(ids)), "negative case ids must be unique")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confined_artifact_path(relative_path: str) -> Path:
    path = Path(relative_path)
    require(
        not path.is_absolute() and "\\" not in relative_path and ".." not in path.parts,
        f"unsafe replay-pack artifact path: {relative_path}",
    )
    candidate = ROOT / path
    cursor = ROOT
    for part in path.parts:
        cursor /= part
        require(
            not cursor.is_symlink(),
            f"replay-pack artifact traverses symlink: {relative_path}",
        )
    require(
        candidate.is_file() and candidate.resolve().is_relative_to(ROOT.resolve()),
        f"replay-pack artifact is not a confined regular file: {relative_path}",
    )
    return candidate


def _expected_pack_artifacts(base_pack: dict[str, Any]) -> dict[str, str]:
    expected = {
        artifact["path"]: f"base_{artifact['kind']}"
        for artifact in base_pack["artifacts"]
    }
    expected["docs/contracts/training_contract_conformance_pack.v1.json"] = (
        "base_conformance_pack"
    )
    expected.update(D4_EXPECTED_ARTIFACTS)
    return with_transitive_schema_dependencies(ROOT, expected)


def validate_pack() -> None:
    pack = base.load_json(PACK_PATH)
    base_pack = base.load_json(BASE_PACK_PATH)
    for relative_path, expected_sha256 in LEGACY_AUTHORITY_SHA256.items():
        require(
            _sha256(_confined_artifact_path(relative_path)) == expected_sha256,
            f"legacy replay authority changed: {relative_path}",
        )
    require(
        _sha256(BASE_PACK_PATH) == BASE_PACK_SHA256
        and base_pack.get("pack_id") == "dag-ml.training-contracts.v1"
        and base_pack.get("pack_checksum") == BASE_PACK_CHECKSUM
        and len(base_pack.get("artifacts", [])) == 91,
        "pinned D1-D3 base pack authority drifted",
    )
    base_entries: dict[str, dict[str, Any]] = {}
    for artifact in base_pack["artifacts"]:
        require_exact_keys(
            artifact, {"path", "sha256", "kind"}, set(), "base_pack.artifact"
        )
        path = _confined_artifact_path(artifact["path"])
        require(
            _sha256(path) == artifact["sha256"],
            f"base artifact stale: {artifact['path']}",
        )
        require(
            artifact["path"] not in base_entries,
            f"duplicate base artifact: {artifact['path']}",
        )
        base_entries[artifact["path"]] = artifact

    require(
        pack.get("pack_id") == "dag-ml.training-replay-contracts.v1"
        and pack.get("mode") == "current",
        "training-replay pack identity/mode drifted",
    )
    require(
        pack.get("base_pack_id") == "dag-ml.training-contracts.v1"
        and pack.get("base_pack_sha256") == BASE_PACK_SHA256
        and pack.get("base_pack_checksum") == BASE_PACK_CHECKSUM
        and pack.get("base_pack_mode") == "current",
        "training-replay base-pack authority drifted",
    )
    require(
        pack.get("hash_algorithm") == "sha256-file-bytes"
        and pack.get("canonical_profile") == "DAG-ML TCV1",
        "training-replay pack hash/canonical profile drifted",
    )
    require(
        pack.get("pack_checksum") == w10_fingerprint_without(pack, "pack_checksum"),
        "training-replay pack checksum drifted",
    )

    expected = _expected_pack_artifacts(base_pack)
    artifacts = pack.get("artifacts")
    require(isinstance(artifacts, list) and bool(artifacts), "pack artifacts missing")
    paths = [artifact.get("path") for artifact in artifacts]
    require(
        paths == sorted(expected),
        "training-replay pack artifact closure is not exact/current/transitive",
    )
    entries: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        require_exact_keys(artifact, {"path", "sha256", "kind"}, set(), "pack.artifact")
        path = _confined_artifact_path(artifact["path"])
        require(
            artifact["kind"] == expected[artifact["path"]],
            f"pack artifact kind drifted: {artifact['path']}",
        )
        require(
            _sha256(path) == artifact["sha256"],
            f"pack artifact stale: {artifact['path']}",
        )
        entries[artifact["path"]] = artifact

    for relative_path, base_entry in base_entries.items():
        replay_entry = entries[relative_path]
        require(
            replay_entry["sha256"] == base_entry["sha256"]
            and replay_entry["kind"] == f"base_{base_entry['kind']}",
            f"replay pack did not preserve base entry: {relative_path}",
        )
    require(
        entries["docs/contracts/training_contract_conformance_pack.v1.json"]["sha256"]
        == BASE_PACK_SHA256,
        "replay pack did not preserve base pack bytes",
    )
    require(
        pack.get("positive_fixture_ids") == EXPECTED_POSITIVE_FIXTURE_IDS,
        "training-replay positive fixture ids drifted",
    )
    negative_ids = [
        case["id"] for case in _load("training_replay_negative_cases.v1.json")["cases"]
    ]
    require(
        pack.get("negative_case_ids") == EXPECTED_NEGATIVE_CASE_IDS
        and negative_ids == EXPECTED_NEGATIVE_CASE_IDS
        and len(negative_ids) == len(set(negative_ids)),
        "training-replay negative fixture ids drifted",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-sibling", action="store_true")
    args = parser.parse_args()
    try:
        if args.require_sibling:
            previous = sys.argv
            try:
                sys.argv = [
                    str(ROOT / "scripts/validate_contracts.py"),
                    "--require-sibling",
                ]
                require(base.main() == 0, "base D1-D3 validator failed")
            finally:
                sys.argv = previous
        registry, schemas = base.build_local_schema_registry()
        validate_schema_contracts(schemas, registry)
        validate_positive_fixtures()
        validate_negative_fixtures()
        validate_pack()
    except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"training-replay validation failed: {error}", file=sys.stderr)
        return 1
    print("validated isolated D4 public training-replay contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
