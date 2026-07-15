"""Independent semantic oracle for public training replay contracts.

This module is deliberately separate from
``scripts/validate_training_replay_contracts.py``
and imports no DAG-ML production package.  It reuses the frozen D1--D3 typed
normalizers from :mod:`parity.training.oracle`, while implementing every D4
ReplayRequest/ReplayOutcome decision independently.

The public replay surface is forward-only in v1: PREDICT and EXPLAIN are
accepted, while REFIT remains a low-level operation until its public contract
is frozen.  Canonical replay prediction payloads are port-explicit v2 values.
Multi-output producers are valid only when payloads carry explicit
producer-port provenance; the semantic oracle also models the documented legacy
read fallback when the source node exposes exactly one prediction port.
"""

from __future__ import annotations

import math
from typing import Any

from parity.conformal.oracle import (
    ContractError,
    fingerprint_without,
    require,
    tcv1_sha256,
    validate_strict_json,
)
from parity.training.oracle import (
    _V,
    _campaign_bindings,
    _contains_runtime_handle,
    _exact_keys,
    _graph_closure,
    _graph_nodes,
    _identifier,
    _non_blank,
    _norm_aggregation_policy,
    _norm_output_binding,
    _serde_sha256,
    _sha256,
    _sorted_unique,
    _validate_output_binding,
    validate_data_identity,
    validate_training_outcome,
)

REPLAY_PHASES = frozenset({"PREDICT", "EXPLAIN"})

_OUTPUT_BINDING_FIELDS = {
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

_LINEAGE_FIELDS = {
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


def _contains_raw_feature_payload(value: Any) -> bool:
    forbidden = {"raw_features", "feature_matrix", "raw_spectra", "raw_wavelengths"}
    if isinstance(value, dict):
        return any(key.lower() in forbidden for key in value) or any(
            _contains_raw_feature_payload(member) for member in value.values()
        )
    if isinstance(value, list):
        return any(_contains_raw_feature_payload(member) for member in value)
    return False


def _exact_optional_keys(
    value: Any, required: set[str], optional: set[str], label: str
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    unknown = set(value) - required - optional
    missing = required - set(value)
    require(not unknown, f"{label} has unknown field(s): {sorted(unknown)}")
    require(not missing, f"{label} is missing field(s): {sorted(missing)}")
    return value


def _non_negative_integer(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def _optional_identifier(value: Any, label: str) -> None:
    if value is not None:
        _identifier(value, label)


def _non_empty_string(value: Any, label: str) -> None:
    require(isinstance(value, str) and bool(value), f"{label} must be non-empty")


def _optional_non_empty_string(value: Any, label: str) -> None:
    if value is not None:
        _non_empty_string(value, label)


def _source_bindings(source_outcome: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = source_outcome["outputs"]
    result = {output["binding"]["binding_id"]: output["binding"] for output in outputs}
    require(
        len(result) == len(outputs),
        "source TrainingOutcome output binding ids must be unique",
    )
    return result


def validate_replay_request(
    value: Any,
    label: str = "ReplayRequest",
    *,
    source_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one public ReplayRequest and its optional source authority."""

    request = _exact_keys(
        value,
        {
            "schema_version",
            "request_id",
            "source_outcome_fingerprint",
            "phase",
            "data_envelope_keys",
            "output_binding_ids",
            "request_fingerprint",
        },
        label,
    )
    validate_strict_json(request, label)
    require(request["schema_version"] == 1, f"{label}.schema_version must be 1")
    _identifier(request["request_id"], f"{label}.request_id")
    _sha256(
        request["source_outcome_fingerprint"],
        f"{label}.source_outcome_fingerprint",
    )
    require(
        request["phase"] in REPLAY_PHASES,
        f"{label}.phase must be PREDICT or EXPLAIN",
    )
    _sorted_unique(
        request["data_envelope_keys"],
        f"{label}.data_envelope_keys",
        non_empty=True,
        identifiers=False,
    )
    _sorted_unique(
        request["output_binding_ids"],
        f"{label}.output_binding_ids",
        non_empty=True,
    )
    _sha256(request["request_fingerprint"], f"{label}.request_fingerprint")
    require(
        request["request_fingerprint"]
        == fingerprint_without(request, "request_fingerprint"),
        f"{label}.request_fingerprint does not match TCV1 request content",
    )

    if source_outcome is not None:
        source = validate_training_outcome(
            source_outcome, f"{label}.source_training_outcome"
        )
        require(
            request["source_outcome_fingerprint"] == source["outcome_fingerprint"],
            f"{label} does not bind the supplied source TrainingOutcome",
        )
        require(
            request["phase"] in source["replayable_phases"],
            f"{label} source outcome does not advertise {request['phase']}",
        )
        data_bindings = _campaign_bindings(source["effective_plan"]["campaign"])
        require(
            request["data_envelope_keys"] == sorted(data_bindings),
            f"{label}.data_envelope_keys do not exactly cover source plan bindings",
        )
        source_outputs = _source_bindings(source)
        require(
            set(request["output_binding_ids"]) <= set(source_outputs),
            f"{label}.output_binding_ids reference an unknown source output",
        )
    return request


def validate_score_set_v2(
    value: Any, plan: dict[str, Any], label: str = "ScoreSetV2"
) -> dict[str, Any]:
    """Independently validate the port-aware score report identity key."""

    score_set = _exact_optional_keys(
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
    for index, report in enumerate(reports):
        report_label = f"{label}.reports[{index}]"
        report = _exact_optional_keys(
            report,
            {
                "producer_node",
                "producer_port",
                "partition",
                "level",
                "row_count",
                "target_width",
                "metrics",
            },
            {
                "prediction_id",
                "variant_id",
                "variant_label",
                "fold_id",
                "target_names",
            },
            report_label,
        )
        _identifier(report["producer_node"], f"{report_label}.producer_node")
        port = resolve_replay_producer_port(
            plan, report["producer_node"], report["producer_port"], report_label
        )
        _optional_identifier(report.get("variant_id"), f"{report_label}.variant_id")
        _optional_identifier(report.get("fold_id"), f"{report_label}.fold_id")
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
            _non_blank(name, f"{report_label}.metrics key")
            require(
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric),
                f"{report_label}.metrics[{name}] must be finite",
            )
        keys.append(
            (
                report["producer_node"],
                port,
                report.get("variant_id"),
                report["partition"],
                report.get("fold_id"),
                report["level"],
            )
        )
    require(len(keys) == len(set(keys)), f"{label} has duplicate score report key")
    return score_set


def replay_training_outcome_ref(
    source_outcome: dict[str, Any], label: str = "TrainingOutcome"
) -> dict[str, Any]:
    """Build the complete transplant-resistant reference for a valid source."""

    source = validate_training_outcome(source_outcome, label)
    return {
        "outcome_id": source["outcome_id"],
        "outcome_fingerprint": source["outcome_fingerprint"],
        "training_request_fingerprint": source["training_request_fingerprint"],
        "effective_plan_fingerprint": source["effective_plan_fingerprint"],
        "execution_bundle_id": source["execution_bundle"]["bundle_id"],
        "execution_bundle_fingerprint": tcv1_sha256(source["execution_bundle"]),
        "output_binding_fingerprints": [
            output["binding"]["binding_fingerprint"] for output in source["outputs"]
        ],
        "training_influence_fingerprint": source["training_influence"][
            "manifest_fingerprint"
        ],
        "data_identities_fingerprint": tcv1_sha256(source["data_identities"]),
    }


def resolve_replay_producer_port(
    plan: dict[str, Any], node_id: str, producer_port: Any, label: str
) -> str:
    """Resolve a canonical port or the unambiguous legacy omission fallback."""

    nodes = _graph_nodes(plan["graph_plan"]["graph"])
    require(node_id in nodes, f"{label}.producer_node is absent from source plan")
    outputs = nodes[node_id].get("ports", {}).get("outputs")
    require(isinstance(outputs, list), f"{label} producer output ports are absent")
    names = [port.get("name") for port in outputs]
    require(
        all(isinstance(name, str) and bool(name.strip()) for name in names)
        and len(names) == len(set(names)),
        f"{label} producer output ports are not uniquely named",
    )
    prediction_ports = [
        port["name"] for port in outputs if port.get("kind") == "prediction"
    ]
    if producer_port is None:
        require(
            len(prediction_ports) == 1,
            f"{label}.producer_port legacy omission requires exactly one "
            "prediction port",
        )
        return prediction_ports[0]

    _non_blank(producer_port, f"{label}.producer_port")
    matches = [port for port in outputs if port.get("name") == producer_port]
    require(
        len(matches) == 1 and matches[0].get("kind") == "prediction",
        f"{label}.producer_port `{producer_port}` is not a unique prediction output",
    )
    return producer_port


def _expected_columns(binding: dict[str, Any]) -> list[str]:
    if binding["prediction_kind"] == "class_probability":
        return [
            f"{target}:{class_label}"
            for target, labels in zip(binding["target_names"], binding["class_labels"])
            for class_label in labels
        ]
    return binding["target_names"]


def _validate_prediction_values(
    values: Any,
    row_count: int,
    binding: dict[str, Any],
    label: str,
) -> None:
    columns = _expected_columns(binding)
    require(
        isinstance(values, list) and len(values) == row_count,
        f"{label} row count does not match identifiers",
    )
    for row_index, row in enumerate(values):
        row_label = f"{label}[{row_index}]"
        require(
            isinstance(row, list) and len(row) == len(columns),
            f"{row_label} width does not match OutputBinding",
        )
        require(
            all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number)
                for number in row
            ),
            f"{row_label} must contain finite JSON numbers",
        )

        if binding["prediction_kind"] == "class_probability":
            cursor = 0
            for target_index, labels in enumerate(binding["class_labels"]):
                probabilities = row[cursor : cursor + len(labels)]
                require(
                    all(0.0 <= float(number) <= 1.0 for number in probabilities),
                    f"{row_label} target {target_index} probability is outside [0,1]",
                )
                # Validation is deliberately observational: values are never
                # clipped or renormalized by the oracle.
                total = 0.0
                for probability in probabilities:
                    total += float(probability)
                require(
                    abs(total - 1.0) <= 1e-12,
                    f"{row_label} target {target_index} is not on the simplex",
                )
                cursor += len(labels)
        elif binding["prediction_kind"] == "class_label":
            require(
                all(bool(labels) for labels in binding["class_labels"]),
                f"{label} class_label replay requires an explicit vocabulary "
                "per target",
            )
            for target_index, (number, labels) in enumerate(
                zip(row, binding["class_labels"])
            ):
                require(
                    isinstance(number, (int, float))
                    and not isinstance(number, bool)
                    and math.isfinite(number)
                    and float(number).is_integer()
                    and 0 <= int(number) < len(labels),
                    f"{row_label}[{target_index}] class_label must be a zero-based "
                    "vocabulary index",
                )


def _validate_output_binding_for_replay(
    value: Any, plan: dict[str, Any], label: str
) -> dict[str, Any]:
    binding = _exact_keys(value, _OUTPUT_BINDING_FIELDS, label)
    require(binding["schema_version"] == 1, f"{label}.schema_version must be 1")
    _validate_output_binding(binding, plan["graph_plan"]["graph"], label)
    expected_aggregation = _serde_sha256(
        _norm_aggregation_policy(plan["campaign"]["aggregation_policy"])
    )
    require(
        binding["aggregation_fingerprint"] == expected_aggregation,
        f"{label}.aggregation_fingerprint does not match the effective policy",
    )
    # Exercise the typed normalizer independently of equality checks against a
    # source binding; this rejects wrong serde container shapes.
    _norm_output_binding(binding)
    return binding


def validate_replay_bound_output(
    value: Any, plan: dict[str, Any], label: str = "BoundTrainingOutput"
) -> dict[str, Any]:
    """Validate canonical v2 or unambiguous legacy bound prediction payloads."""

    output = _exact_optional_keys(
        value,
        {
            "binding",
            "predictions",
            "observation_predictions",
            "aggregated_predictions",
        },
        {"schema_version"},
        label,
    )
    if "schema_version" in output:
        require(
            output["schema_version"] == 2,
            f"{label}.schema_version must be 2 when present",
        )
    binding = _validate_output_binding_for_replay(
        output["binding"], plan, f"{label}.binding"
    )

    required = {
        "prediction": {
            "producer_node",
            "partition",
            "fold_id",
            "sample_ids",
            "values",
        },
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
    optional = {
        "prediction": {"prediction_id", "producer_port", "target_names"},
        "observation": {
            "prediction_id",
            "producer_port",
            "weights",
            "target_names",
        },
        "aggregated": {"prediction_id", "producer_port", "target_names"},
    }
    groups = (
        ("predictions", "prediction"),
        ("observation_predictions", "observation"),
        ("aggregated_predictions", "aggregated"),
    )
    block_count = 0
    seen_by_family: dict[str, set[Any]] = {
        "prediction": set(),
        "observation": set(),
        "aggregated": set(),
    }
    for field, kind in groups:
        blocks = output[field]
        require(isinstance(blocks, list), f"{label}.{field} must be an array")
        block_count += len(blocks)
        for index, source_block in enumerate(blocks):
            block_label = f"{label}.{field}[{index}]"
            block = _exact_optional_keys(
                source_block, required[kind], optional[kind], block_label
            )
            if output.get("schema_version") == 2:
                require(
                    "producer_port" in block,
                    f"{block_label}.producer_port is required in v2",
                )
            else:
                require(
                    "producer_port" not in block,
                    f"{block_label} legacy bound output must not contain producer_port",
                )
            if "prediction_id" in block:
                _optional_non_empty_string(
                    block["prediction_id"], f"{block_label}.prediction_id"
                )
            _identifier(block["producer_node"], f"{block_label}.producer_node")
            require(
                block["producer_node"] == binding["node_id"],
                f"{block_label}.producer_node must match OutputBinding.node_id",
            )
            port = resolve_replay_producer_port(
                plan,
                block["producer_node"],
                block.get("producer_port"),
                block_label,
            )
            require(
                port == binding["port_name"],
                f"{block_label}.producer_port must match OutputBinding.port_name",
            )
            require(
                block["partition"] == "final" and block["fold_id"] is None,
                f"{block_label} forward replay must use final partition and null fold",
            )
            require(
                block.get("target_names") == _expected_columns(binding),
                f"{block_label}.target_names do not match OutputBinding order",
            )

            if kind == "prediction":
                identifiers = block["sample_ids"]
            elif kind == "observation":
                identifiers = block["observation_ids"]
            else:
                require(
                    block["level"] == binding["prediction_level"],
                    f"{block_label}.level does not match OutputBinding",
                )
                units = block["unit_ids"]
                require(
                    isinstance(units, list), f"{block_label}.unit_ids must be array"
                )
                identifiers = []
                for unit_index, unit in enumerate(units):
                    unit_label = f"{block_label}.unit_ids[{unit_index}]"
                    item = _exact_keys(unit, {"level", "id"}, unit_label)
                    require(
                        item["level"] == block["level"],
                        f"{unit_label}.level does not match block level",
                    )
                    identifiers.append(item["id"])

            require(
                isinstance(identifiers, list)
                and bool(identifiers)
                and len(identifiers) == len(set(identifiers)),
                f"{block_label} identifiers must be non-empty and unique",
            )
            for identifier in identifiers:
                _identifier(identifier, f"{block_label} identifier")
            _validate_prediction_values(
                block["values"], len(identifiers), binding, f"{block_label}.values"
            )
            coordinates: set[Any] = (
                {(block["level"], identifier) for identifier in identifiers}
                if kind == "aggregated"
                else set(identifiers)
            )
            require(
                seen_by_family[kind].isdisjoint(coordinates),
                f"{label}.{field} has duplicate final unit across blocks",
            )
            seen_by_family[kind].update(coordinates)

            if kind == "observation" and "weights" in block:
                weights = block["weights"]
                require(
                    isinstance(weights, list) and len(weights) == len(identifiers),
                    f"{block_label}.weights must align with observation ids",
                )
                require(
                    all(
                        isinstance(weight, (int, float))
                        and not isinstance(weight, bool)
                        and math.isfinite(weight)
                        and weight > 0.0
                        for weight in weights
                    ),
                    f"{block_label}.weights must be finite and positive",
                )

    require(block_count > 0, f"{label} must emit at least one prediction block")
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


def _optional_relation_identifier(value: Any, label: str) -> Any:
    if value is not None:
        _identifier(value, label)
    return value


def _optional_relation_text(value: Any, label: str) -> Any:
    if value is not None:
        _non_blank(value, label)
    return value


def replay_relation_fingerprint(
    relations: Any, label: str = "coordinator_relations"
) -> str:
    """Validate and fingerprint a coordinator relation set like DAG-ML Rust."""

    relation_set = _exact_keys(relations, {"records"}, label)
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
        source = _exact_optional_keys(
            value, _RELATION_REQUIRED_FIELDS, _RELATION_OPTIONAL_FIELDS, record_label
        )
        unit_level = source.get("unit_level", "observation")
        require(
            unit_level in {"physical_sample", "source_sample", "observation", "combo"},
            f"{record_label}.unit_level is invalid",
        )
        observation_id = _identifier(
            source["observation_id"], f"{record_label}.observation_id"
        )
        sample_id = _identifier(source["sample_id"], f"{record_label}.sample_id")
        require(
            observation_id not in observation_samples,
            f"{label} contains duplicate observation `{observation_id}`",
        )
        observation_samples[observation_id] = sample_id

        for field in ("rep_id", "target_id", "group_id", "origin_sample_id"):
            _optional_relation_identifier(source.get(field), f"{record_label}.{field}")
        for field in ("unit_id", "source_id", "derived_unit_id", "quality_flag"):
            _optional_relation_text(source.get(field), f"{record_label}.{field}")

        component_ids = source.get("component_observation_ids", [])
        require(
            isinstance(component_ids, list)
            and len(component_ids) == len(set(component_ids)),
            f"{record_label}.component_observation_ids must be unique",
        )
        for component_id in component_ids:
            _identifier(component_id, f"{record_label}.component_observation_ids")
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
            _non_blank(tag, f"{record_label}.tags")

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
            canonical_record["metadata"] = _V(metadata)
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
    return _serde_sha256(canonical)


def validate_replay_envelopes(
    value: Any,
    request: dict[str, Any],
    source_outcome: dict[str, Any],
    identities: list[dict[str, Any]],
    label: str = "ReplayInputEnvelopes",
) -> dict[str, Any]:
    """Validate exact new-cohort rebinding and current data identities."""

    fixture = _exact_keys(value, {"schema_version", "envelopes"}, label)
    validate_strict_json(fixture, label)
    require(fixture["schema_version"] == 1, f"{label}.schema_version must be 1")
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
        len(identity_by_key) == len(identities) and list(identity_by_key) == keys,
        f"{label} identities do not exactly cover envelope keys in order",
    )
    bindings = _campaign_bindings(source_outcome["effective_plan"]["campaign"])
    require(keys == sorted(bindings), f"{label} does not cover source plan bindings")

    for key, envelope in envelopes.items():
        envelope_label = f"{label}.envelopes[{key}]"
        require(isinstance(envelope, dict), f"{envelope_label} must be an object")
        require(envelope.get("schema_version") == 1, f"{envelope_label}.schema_version")
        for field in (
            "schema_fingerprint",
            "plan_fingerprint",
            "relation_fingerprint",
            "data_content_fingerprint",
            "target_content_fingerprint",
        ):
            _sha256(envelope.get(field), f"{envelope_label}.{field}")
        plan = envelope.get("plan")
        require(isinstance(plan, dict), f"{envelope_label}.plan must be an object")
        require(
            envelope["plan_fingerprint"] == _serde_sha256(plan),
            f"{envelope_label}.plan_fingerprint does not match plan content",
        )
        relations = envelope.get("coordinator_relations")
        require(relations is not None, f"{envelope_label} requires relations")
        require(
            envelope["relation_fingerprint"]
            == replay_relation_fingerprint(relations, f"{envelope_label}.relations"),
            f"{envelope_label}.relation_fingerprint does not match relations",
        )

        binding = bindings[key]
        require(
            envelope["schema_fingerprint"] == binding["schema_fingerprint"]
            and envelope["plan_fingerprint"] == binding["plan_fingerprint"],
            f"{envelope_label} schema/plan rebinding is forbidden",
        )
        require(
            plan.get("output_representation") == binding["output_representation"],
            f"{envelope_label} output representation changed",
        )
        sources = sorted(
            {
                step.get("source_id")
                for step in plan.get("steps", [])
                if isinstance(step, dict) and step.get("source_id") is not None
            }
        )
        require(
            sources == binding["source_ids"],
            f"{envelope_label} source ids changed",
        )
        # feature_set_id is frozen by the source binding. Replay envelopes
        # repeat it explicitly so omission cannot erase that attestation.
        feature_set_id = binding["feature_set_id"]
        if feature_set_id is not None:
            _non_blank(feature_set_id, f"{envelope_label}.feature_set_id")
        metadata = envelope.get("metadata", {})
        require(isinstance(metadata, dict), f"{envelope_label}.metadata must be object")
        require(
            "feature_set_id" in metadata
            and metadata["feature_set_id"] == feature_set_id,
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


def _validate_output_cohort(
    outputs: list[dict[str, Any]], envelope_fixture: dict[str, Any], label: str
) -> None:
    relation_sets = [
        envelope["coordinator_relations"]
        for envelope in envelope_fixture["envelopes"].values()
    ]
    require(bool(relation_sets), f"{label} has no coordinator relations")
    for index, relations in enumerate(relation_sets):
        replay_relation_fingerprint(relations, f"{label}.relations[{index}]")
    # `excluded` is a training-only relation flag; replay covers these units.
    records = [record for relations in relation_sets for record in relations["records"]]
    expected = {
        "observation": {record["observation_id"] for record in records},
        "target": {
            record["target_id"] for record in records if record.get("target_id")
        },
        "group": {record["group_id"] for record in records if record.get("group_id")},
    }
    for index, output in enumerate(outputs):
        output_label = f"{label}.outputs[{index}]"
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
        families = (
            (
                "predictions",
                "sample",
                {
                    identifier
                    for block in output["predictions"]
                    for identifier in block["sample_ids"]
                },
            ),
            (
                "observation_predictions",
                "observation",
                {
                    identifier
                    for block in output["observation_predictions"]
                    for identifier in block["observation_ids"]
                },
            ),
            (
                "aggregated_predictions",
                output["binding"]["prediction_level"],
                {
                    unit["id"]
                    for block in output["aggregated_predictions"]
                    for unit in block["unit_ids"]
                },
            ),
        )
        for family, level, actual in families:
            if output[family]:
                eligible = sample_units if level == "sample" else expected[level]
                require(
                    actual <= eligible,
                    f"{output_label}.{family} contains an id outside current cohort",
                )


def _validate_lineage(
    lineage: Any,
    outcome: dict[str, Any],
    source_outcome: dict[str, Any],
    request: dict[str, Any],
    source_bindings: dict[str, dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    require(isinstance(lineage, list), f"{label} must be an array")
    records = [
        _exact_keys(record, _LINEAGE_FIELDS, f"{label}[{index}]")
        for index, record in enumerate(lineage)
    ]
    record_ids = [record["record_id"] for record in records]
    require(
        record_ids == sorted(set(record_ids)),
        f"{label} must be sorted by unique record_id",
    )
    for index, record in enumerate(records):
        record_label = f"{label}[{index}]"
        for field in ("record_id", "run_id", "node_id", "controller_id"):
            _identifier(record[field], f"{record_label}.{field}")
        _non_empty_string(
            record["controller_version"], f"{record_label}.controller_version"
        )
        _optional_identifier(record["variant_id"], f"{record_label}.variant_id")
        _optional_identifier(record["fold_id"], f"{record_label}.fold_id")
        for field in ("params_fingerprint",):
            _sha256(record[field], f"{record_label}.{field}")
        for field in (
            "data_model_shape_fingerprint",
            "aggregation_policy_fingerprint",
        ):
            if record[field] is not None:
                _sha256(record[field], f"{record_label}.{field}")
        if record["seed"] is not None:
            _non_negative_integer(record["seed"], f"{record_label}.seed")
            require(
                record["seed"] <= (1 << 64) - 1,
                f"{record_label}.seed exceeds the native u64 maximum",
            )
        require(
            isinstance(record["branch_path"], list),
            f"{record_label}.branch_path must be an array",
        )
        for branch in record["branch_path"]:
            _identifier(branch, f"{record_label}.branch_path")
        _sorted_unique(record["input_lineage"], f"{record_label}.input_lineage")
        artifacts = record["artifact_refs"]
        require(
            isinstance(artifacts, list),
            f"{record_label}.artifact_refs must be an array",
        )
        artifact_ids: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            artifact_label = f"{record_label}.artifact_refs[{artifact_index}]"
            require(isinstance(artifact, dict), f"{artifact_label} must be an object")
            for field in ("id", "controller_id"):
                _identifier(artifact.get(field), f"{artifact_label}.{field}")
            _non_empty_string(artifact.get("kind"), f"{artifact_label}.kind")
            require(
                artifact["id"] not in artifact_ids,
                f"{artifact_label}.id is duplicated",
            )
            artifact_ids.add(artifact["id"])
            if artifact.get("content_fingerprint") is not None:
                _sha256(
                    artifact["content_fingerprint"],
                    f"{artifact_label}.content_fingerprint",
                )
            _optional_non_empty_string(
                artifact.get("plugin"), f"{artifact_label}.plugin"
            )
            _optional_non_empty_string(
                artifact.get("plugin_version"),
                f"{artifact_label}.plugin_version",
            )
            require(
                artifact.get("plugin_version") is None
                or artifact.get("plugin") is not None,
                f"{artifact_label}.plugin_version requires plugin",
            )
        unsafe_flags = record["unsafe_flags"]
        require(
            isinstance(unsafe_flags, list)
            and len(unsafe_flags) == len(set(unsafe_flags)),
            f"{record_label}.unsafe_flags must contain unique values",
        )
        for flag in unsafe_flags:
            _non_empty_string(flag, f"{record_label}.unsafe_flags entry")
        require(isinstance(record["metrics"], dict), f"{record_label}.metrics")
        for key in record["metrics"]:
            _non_empty_string(key, f"{record_label}.metrics key")
        require(
            all(
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric)
                for metric in record["metrics"].values()
            ),
            f"{record_label}.metrics must contain finite numbers",
        )

    requested_nodes = [
        source_bindings[binding_id]["node_id"]
        for binding_id in request["output_binding_ids"]
    ]
    plan = source_outcome["effective_plan"]
    closure = set(_graph_closure(plan["graph_plan"]["graph"], requested_nodes))
    records_by_node = {record["node_id"]: record for record in records}
    require(
        len(records_by_node) == len(records) and set(records_by_node) == closure,
        f"{label} does not exactly cover the requested predictor closure",
    )
    node_plans = plan["node_plans"]
    for node_id in sorted(closure):
        record = records_by_node[node_id]
        node_plan = node_plans[node_id]
        require(
            record["run_id"] == outcome["run_id"]
            and record["phase"] == outcome["phase"],
            f"{label} node `{node_id}` run/phase mismatch",
        )
        require(
            outcome["phase"] in node_plan["supported_phases"],
            f"{label} node `{node_id}` does not support replay phase",
        )
        require(
            record["controller_id"] == node_plan["controller_id"]
            and record["controller_version"] == node_plan["controller_version"]
            and record["params_fingerprint"] == node_plan["params_fingerprint"],
            f"{label} node `{node_id}` controller/version/params mismatch",
        )
        require(
            record["variant_id"] == source_outcome["selected_variant_id"]
            and record["fold_id"] is None,
            f"{label} node `{node_id}` variant/fold mismatch",
        )
        expected_inputs = sorted(
            records_by_node[parent]["record_id"] for parent in node_plan["input_nodes"]
        )
        require(
            record["input_lineage"] == expected_inputs,
            f"{label} node `{node_id}` input lineage mismatch",
        )
    return records


def validate_replay_outcome(
    value: Any,
    label: str = "ReplayOutcome",
    *,
    request: dict[str, Any],
    source_outcome: dict[str, Any],
    envelope_fixture: dict[str, Any],
) -> dict[str, Any]:
    """Validate a complete public ReplayOutcome against its three authorities."""

    outcome = _exact_keys(
        value,
        {
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
        },
        label,
    )
    validate_strict_json(outcome, label)
    require(not _contains_runtime_handle(outcome), f"{label} contains runtime handles")
    require(outcome["schema_version"] == 1, f"{label}.schema_version must be 1")
    for field in ("outcome_id", "run_id", "replay_request_id", "bundle_id"):
        _identifier(outcome[field], f"{label}.{field}")
    _non_blank(outcome["plan_id"], f"{label}.plan_id")
    _sha256(
        outcome["replay_request_fingerprint"],
        f"{label}.replay_request_fingerprint",
    )
    _sha256(outcome["outcome_fingerprint"], f"{label}.outcome_fingerprint")
    for field in (
        "result_count",
        "lineage_record_count",
        "prediction_block_count",
        "observation_prediction_block_count",
        "aggregated_prediction_block_count",
        "explanation_block_count",
        "controller_count",
    ):
        _non_negative_integer(outcome[field], f"{label}.{field}")
    require(
        outcome["prediction_cache_store"] is False,
        f"{label}.prediction_cache_store must be false",
    )

    source = validate_training_outcome(
        source_outcome, f"{label}.source_training_outcome_document"
    )
    replay_request = validate_replay_request(
        request, f"{label}.request", source_outcome=source
    )
    require(
        outcome["source_training_outcome"]
        == replay_training_outcome_ref(
            source, f"{label}.source_training_outcome_document"
        ),
        f"{label}.source_training_outcome is not the complete source reference",
    )
    require(
        outcome["replay_request_id"] == replay_request["request_id"]
        and outcome["replay_request_fingerprint"]
        == replay_request["request_fingerprint"]
        and outcome["phase"] == replay_request["phase"],
        f"{label} does not match ReplayRequest",
    )
    require(
        outcome["bundle_id"] == source["execution_bundle"]["bundle_id"]
        and outcome["plan_id"] == source["effective_plan"]["id"],
        f"{label} plan/bundle ids do not match source outcome",
    )

    identities_value = outcome["input_data_identities"]
    require(
        isinstance(identities_value, list) and bool(identities_value),
        f"{label}.input_data_identities must be a non-empty array",
    )
    identities = [
        validate_data_identity(identity, f"{label}.input_data_identities[{index}]")
        for index, identity in enumerate(identities_value)
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    require(
        identity_keys == sorted(set(identity_keys)),
        f"{label}.input_data_identities must be sorted and unique",
    )
    validate_replay_envelopes(
        envelope_fixture,
        replay_request,
        source,
        identities,
        f"{label}.input_envelopes",
    )

    plan = source["effective_plan"]
    outputs_value = outcome["outputs"]
    require(isinstance(outputs_value, list), f"{label}.outputs must be an array")
    outputs = [
        validate_replay_bound_output(output, plan, f"{label}.outputs[{index}]")
        for index, output in enumerate(outputs_value)
    ]
    _validate_output_cohort(outputs, envelope_fixture, f"{label}.current_cohort")
    for index, output in enumerate(outputs):
        require(
            output.get("schema_version") == 2,
            f"{label}.outputs[{index}].schema_version must be 2 for public training replay",
        )
    binding_ids = [output["binding"]["binding_id"] for output in outputs]
    require(
        binding_ids == sorted(set(binding_ids)),
        f"{label}.outputs must be sorted by unique binding_id",
    )
    source_bindings = _source_bindings(source)
    for index, output in enumerate(outputs):
        binding = output["binding"]
        require(
            source_bindings.get(binding["binding_id"]) == binding,
            f"{label}.outputs[{index}] binding does not match source outcome",
        )

    explanations = outcome["explanations"]
    require(isinstance(explanations, list), f"{label}.explanations must be an array")
    if outcome["phase"] == "PREDICT":
        require(bool(outputs), f"{label} PREDICT must emit at least one output")
        require(
            binding_ids == replay_request["output_binding_ids"],
            f"{label}.outputs do not exactly cover requested bindings",
        )
        require(not explanations, f"{label} PREDICT cannot emit explanations")
    else:
        require(
            set(binding_ids) <= set(replay_request["output_binding_ids"]),
            f"{label}.outputs contain an unrequested binding",
        )
        require(bool(explanations), f"{label} EXPLAIN requires an explanation")

    requested_coordinates = {
        (
            source_bindings[binding_id]["node_id"],
            source_bindings[binding_id]["port_name"],
        ): source_bindings[binding_id]
        for binding_id in replay_request["output_binding_ids"]
    }
    for index, explanation in enumerate(explanations):
        explanation_label = f"{label}.explanations[{index}]"
        block = _exact_optional_keys(
            explanation,
            {"producer_node", "method", "payload"},
            {"producer_port", "target_name"},
            explanation_label,
        )
        require(
            "producer_port" in block,
            f"{explanation_label}.producer_port is required for public training replay",
        )
        _identifier(block["producer_node"], f"{explanation_label}.producer_node")
        port = resolve_replay_producer_port(
            plan,
            block["producer_node"],
            block.get("producer_port"),
            explanation_label,
        )
        coordinate = (block["producer_node"], port)
        require(
            coordinate in requested_coordinates,
            f"{explanation_label} does not explain a requested prediction port",
        )
        _non_blank(block["method"], f"{explanation_label}.method")
        if "target_name" in block:
            _non_blank(block["target_name"], f"{explanation_label}.target_name")
            require(
                block["target_name"]
                in requested_coordinates[coordinate]["target_names"],
                f"{explanation_label}.target_name is absent from OutputBinding",
            )
        validate_strict_json(block["payload"], f"{explanation_label}.payload")
        require(
            not _contains_runtime_handle(block["payload"]),
            f"{explanation_label}.payload contains runtime handles",
        )
        require(
            not _contains_raw_feature_payload(block["payload"]),
            f"{explanation_label}.payload must not embed raw feature data",
        )

    prediction_count = sum(len(output["predictions"]) for output in outputs)
    observation_count = sum(
        len(output["observation_predictions"]) for output in outputs
    )
    aggregated_count = sum(len(output["aggregated_predictions"]) for output in outputs)
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
    require(
        outcome["explanation_block_count"] == len(explanations),
        f"{label}.explanation_block_count does not match payload",
    )

    lineage = _validate_lineage(
        outcome["lineage"],
        outcome,
        source,
        replay_request,
        source_bindings,
        f"{label}.lineage",
    )
    require(
        outcome["lineage_record_count"] == outcome["result_count"] == len(lineage),
        f"{label}.lineage/result counts do not match payload",
    )
    require(
        outcome["controller_count"]
        == len({record["controller_id"] for record in lineage}),
        f"{label}.controller_count does not match lineage controllers",
    )

    _sorted_unique(outcome["warnings"], f"{label}.warnings", identifiers=False)
    require(isinstance(outcome["diagnostics"], dict), f"{label}.diagnostics")
    for key, member in outcome["diagnostics"].items():
        _non_empty_string(key, f"{label}.diagnostics key")
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
        not _contains_runtime_handle(outcome["diagnostics"]),
        f"{label}.diagnostics must not contain runtime handles",
    )
    require(
        outcome["outcome_fingerprint"]
        == fingerprint_without(outcome, "outcome_fingerprint"),
        f"{label}.outcome_fingerprint does not match TCV1 outcome content",
    )
    return outcome


__all__ = [
    "ContractError",
    "replay_relation_fingerprint",
    "replay_training_outcome_ref",
    "resolve_replay_producer_port",
    "validate_replay_bound_output",
    "validate_replay_envelopes",
    "validate_replay_outcome",
    "validate_replay_request",
    "validate_score_set_v2",
]
