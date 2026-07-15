"""Test-only oracle for DAG-ML conformal/robustness W0 contracts.

This module intentionally imports no DAG-ML production module. It freezes the
exact finite-sample rank, the TCV1 fingerprint profile, leakage closure checks,
and cross-field semantics that JSON Schema cannot express. Native Rust/C/WASM
implementations must be compared to these committed results; production code
must never import this oracle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import unicodedata2 as unicodedata

TCV1_PREFIX = b"DAGML-TCV1\0"
TCV1_UNICODE_VERSION = "17.0.0"
TCV1_INT_MIN = -(2**63)
TCV1_INT_MAX = 2**64 - 1
PORTABLE_EXACT_INT_MAX = 2**53 - 1
COHORT_ROLES = {"development", "calibration", "external_test", "production"}
INFLUENCE_KINDS = (
    "transform_fit",
    "model_fit",
    "hpo_selection",
    "early_stopping",
    "weighting_resampling",
    "trained_meta_aggregation",
)
CONFORMAL_METRIC_REQUESTS = {
    "conformal_coverage": "empirical_coverage",
    "empirical_coverage": "empirical_coverage",
    "coverage_gap": "coverage_gap",
    "mean_width": "mean_width",
    "median_width": "median_width",
    "interval_score": "interval_score",
    "set_size": "set_size",
}

if unicodedata.unidata_version != TCV1_UNICODE_VERSION:
    raise RuntimeError(
        "TCV1 requires Unicode "
        f"{TCV1_UNICODE_VERSION}, got {unicodedata.unidata_version}"
    )


class ContractError(ValueError):
    """A fixture violates a semantic contract not expressible in JSON Schema."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`ContractError` when ``condition`` is false."""

    if not condition:
        raise ContractError(message)


def load_json(path: Path) -> Any:
    """Load strict JSON while refusing duplicate object members."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_nonfinite(token: str) -> None:
        raise ContractError(f"{path} contains non-finite JSON number {token}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path} is not strict UTF-8 JSON: {exc}") from exc


def _strict_text(value: Any, label: str) -> tuple[str, bytes]:
    require(isinstance(value, str), f"{label} must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError(f"{label} contains a surrogate code point") from exc
    normalized = unicodedata.normalize("NFC", value)
    return normalized, normalized.encode("utf-8")


def validate_strict_json(value: Any, label: str = "value") -> None:
    """Validate the TCV1-supported strict JSON domain recursively."""

    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str):
            _strict_text(value, label)
        return
    if isinstance(value, int):
        require(
            TCV1_INT_MIN <= value <= TCV1_INT_MAX,
            f"{label} integer is outside the TCV1 range",
        )
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, member in enumerate(value):
            validate_strict_json(member, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        normalized_keys: set[bytes] = set()
        for key, member in value.items():
            normalized, encoded = _strict_text(key, f"{label} key")
            require(encoded not in normalized_keys, f"{label} has NFC-colliding keys")
            normalized_keys.add(encoded)
            validate_strict_json(member, f"{label}.{normalized}")
        return
    raise ContractError(f"{label} contains non-JSON type {type(value).__name__}")


def _u64(value: int) -> bytes:
    require(0 <= value <= 2**64 - 1, "TCV1 length exceeds u64")
    return struct.pack(">Q", value)


def tcv1_encode(value: Any, label: str = "value") -> bytes:
    """Encode one strict JSON value using DAG-ML Typed Canonical Value v1."""

    validate_strict_json(value, label)
    if value is None:
        return b"N"
    if value is False:
        return b"F"
    if value is True:
        return b"T"
    if isinstance(value, int):
        payload = str(value).encode("ascii")
        return b"I" + _u64(len(payload)) + payload
    if isinstance(value, float):
        return b"D" + struct.pack(">d", 0.0 if value == 0.0 else value)
    if isinstance(value, str):
        _normalized, payload = _strict_text(value, label)
        return b"S" + _u64(len(payload)) + payload
    if isinstance(value, list):
        return (
            b"A"
            + _u64(len(value))
            + b"".join(
                tcv1_encode(member, f"{label}[{index}]")
                for index, member in enumerate(value)
            )
        )
    if isinstance(value, dict):
        items: list[tuple[bytes, str, Any]] = []
        for key, member in value.items():
            normalized, encoded = _strict_text(key, f"{label} key")
            items.append((encoded, normalized, member))
        items.sort(key=lambda item: item[0])
        return (
            b"O"
            + _u64(len(items))
            + b"".join(
                tcv1_encode(key, f"{label} key") + tcv1_encode(member, f"{label}.{key}")
                for _encoded, key, member in items
            )
        )
    raise AssertionError("strict JSON validation accepted an unsupported value")


def tcv1_preimage(value: Any) -> bytes:
    """Return the domain-separated TCV1 SHA-256 preimage."""

    return TCV1_PREFIX + tcv1_encode(value)


def tcv1_sha256(value: Any) -> str:
    """Return lowercase SHA-256 of the domain-separated TCV1 preimage."""

    return hashlib.sha256(tcv1_preimage(value)).hexdigest()


def fingerprint_without(document: dict[str, Any], field: str) -> str:
    """Fingerprint a contract object while omitting its self-fingerprint."""

    require(field in document, f"missing self-fingerprint field {field}")
    return tcv1_sha256({key: value for key, value in document.items() if key != field})


def file_sha256(path: Path) -> str:
    """Return the byte-level SHA-256 used by the conformance-pack manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_binary64(value: Any) -> bool:
    """Return whether ``value`` came from a finite JSON binary64 token."""

    return isinstance(value, float) and math.isfinite(value)


def _require_binary64(value: Any, label: str) -> float:
    require(_is_binary64(value), f"{label} must be represented as finite binary64")
    return value


def validate_coverages(coverages: Any) -> list[float]:
    """Return canonical binary64 coverages after strict ordered validation."""

    require(isinstance(coverages, list) and coverages, "coverages must be non-empty")
    result: list[float] = []
    for index, coverage in enumerate(coverages):
        normalized = _require_binary64(coverage, f"coverages[{index}]")
        require(
            0.0 < normalized < 1.0,
            f"coverages[{index}] must be finite and in (0, 1)",
        )
        result.append(normalized)
    require(
        all(left < right for left, right in zip(result, result[1:])),
        "coverages must be strictly increasing and unique",
    )
    return result


def finite_sample_rank(sample_count: int, coverage: Any) -> int:
    """Compute ``ceil((n + 1) * coverage)`` from the binary64 shortest token."""

    require(
        isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0,
        "sample_count must be a positive integer",
    )
    normalized = validate_coverages([coverage])[0]
    numerator, denominator = Decimal(repr(normalized)).as_integer_ratio()
    scaled = (sample_count + 1) * numerator
    return (scaled + denominator - 1) // denominator


def _validated_residual_rows(residuals: Any) -> list[list[float]]:
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
                _is_number(residual),
                f"residuals[{row_index}][{column_index}] must be numeric",
            )
            if isinstance(residual, int):
                require(
                    0 <= residual <= PORTABLE_EXACT_INT_MAX,
                    f"residuals[{row_index}][{column_index}] integer is not portable",
                )
            number = float(residual)
            require(
                math.isfinite(number) and number >= 0.0,
                f"residuals[{row_index}][{column_index}] must be finite and non-negative",
            )
            normalized_row.append(number)
        rows.append(normalized_row)
    return rows


def split_absolute_residual(
    residuals: Any,
    coverages: Any,
    *,
    multi_target_policy: str,
    small_sample_policy: str,
) -> list[dict[str, Any]]:
    """Independent split absolute-residual quantile oracle.

    ``marginal`` returns one tagged quantile per target. ``joint_max`` first
    reduces each sample to the maximum target residual and returns one tagged
    quantile that applies simultaneously to every target.
    """

    require(
        multi_target_policy in {"marginal", "joint_max"}, "invalid multi-target policy"
    )
    require(
        small_sample_policy in {"error", "unbounded"}, "invalid small-sample policy"
    )
    rows = _validated_residual_rows(residuals)
    valid_coverages = validate_coverages(coverages)
    columns = list(zip(*rows))
    score_columns = (
        [tuple(max(row) for row in rows)]
        if multi_target_policy == "joint_max"
        else columns
    )
    ordered_columns = [sorted(column) for column in score_columns]
    records: list[dict[str, Any]] = []
    for coverage in valid_coverages:
        rank = finite_sample_rank(len(rows), coverage)
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


def regression_conformal_metrics(
    truth: Any,
    interval: Any,
    *,
    multi_target_policy: str,
) -> list[dict[str, Any]]:
    """Reconstruct coverage, width and interval score from exact bounds/truth.

    Marginal records are emitted per target. For ``joint_max``, empirical
    coverage requires every target of a unit to be covered, while width and
    Winkler interval score are averaged over all unit/target cells. Null/null
    endpoints are treated as unbounded and make width/score unavailable.
    """

    require(
        multi_target_policy in {"marginal", "joint_max"},
        "metric multi_target_policy is invalid",
    )
    require(isinstance(interval, dict), "metric interval must be an object")
    coverage = validate_coverages([interval.get("coverage")])[0]
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
    width = len(truth[0]) if isinstance(truth[0], list) else 0
    require(width > 0, "metric truth rows must be non-empty")
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
            and len(truth_row) == len(lower_row) == len(upper_row) == width,
            f"metric row {row_index} width differs",
        )
        covered_row: list[bool] = []
        width_row: list[float | None] = []
        score_row: list[float | None] = []
        for column_index, (raw_truth, lo, hi) in enumerate(
            zip(truth_row, lower_row, upper_row)
        ):
            require(
                _is_binary64(raw_truth),
                f"metric truth {row_index},{column_index} must be finite binary64",
            )
            value = raw_truth
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
                _is_binary64(lo) and _is_binary64(hi) and lo <= hi,
                f"metric bounds {row_index},{column_index} must be ordered finite binary64",
            )
            cell_width = hi - lo
            cell_score = cell_width
            if value < lo:
                cell_score += (2.0 / alpha) * (lo - value)
            elif value > hi:
                cell_score += (2.0 / alpha) * (value - hi)
            covered_row.append(lo <= value <= hi)
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
            for target_index in range(width)
        ]
    return [
        summarize(
            [all(row) for row in covered],
            [value for row in widths for value in row],
            [value for row in scores for value in row],
            None,
        )
    ]


def validate_numeric_evidence(
    records: Any,
    blocks: Any,
    metric_sets: Any,
    *,
    label: str = "numeric evidence",
) -> None:
    """Validate the test-only numeric bridge from points to conformal metrics."""

    require(isinstance(records, list), f"{label} must be an array")
    require(isinstance(blocks, list), f"{label} blocks must be an array")
    require(isinstance(metric_sets, list), f"{label} metric sets must be an array")
    by_block = {block["block_fingerprint"]: block for block in blocks}
    by_metric = {metric_set["metric_set_id"]: metric_set for metric_set in metric_sets}
    require(len(by_block) == len(blocks), f"{label} has duplicate blocks")
    require(len(by_metric) == len(metric_sets), f"{label} has duplicate metric sets")
    member_names = {
        "evidence_id",
        "block_fingerprint",
        "metric_set_id",
        "point_predictions",
        "point_prediction_fingerprint",
        "truth",
        "truth_fingerprint",
        "evidence_fingerprint",
    }
    ids: list[str] = []
    covered_blocks: list[str] = []
    covered_metrics: list[str] = []
    statistic_names = (
        "measurement_status",
        "empirical_coverage",
        "coverage_gap",
        "mean_width",
        "median_width",
        "interval_score",
    )
    for index, evidence in enumerate(records):
        item_label = f"{label}[{index}]"
        require(isinstance(evidence, dict), f"{item_label} must be an object")
        require(
            set(evidence) == member_names,
            f"{item_label} must have the exact evidence members",
        )
        evidence_id = evidence.get("evidence_id")
        block_key = evidence.get("block_fingerprint")
        metric_key = evidence.get("metric_set_id")
        require(
            isinstance(evidence_id, str) and evidence_id,
            f"{item_label}.evidence_id is invalid",
        )
        require(block_key in by_block, f"{item_label} block does not resolve")
        require(metric_key in by_metric, f"{item_label} metric set does not resolve")
        block = by_block[block_key]
        metric_set = by_metric[metric_key]
        require(
            metric_set["conformal_prediction_block_fingerprint"] == block_key,
            f"{item_label} metric set references another block",
        )
        require(
            metric_set["predictor_binding_fingerprint"]
            == block["predictor_binding_fingerprint"],
            f"{item_label} predictor binding differs",
        )
        require(
            metric_set["calibration_artifact_id"] == block["calibration_artifact_id"]
            and metric_set["calibration_artifact_checksum"]
            == block["calibration_artifact_checksum"],
            f"{item_label} calibrator binding differs",
        )
        require(
            metric_set["method"] == block["method"]
            and metric_set["multi_target_policy"] == block["multi_target_policy"],
            f"{item_label} conformal policy differs",
        )
        require(
            metric_set["unit_ids_fingerprint"] == tcv1_sha256(block["unit_ids"]),
            f"{item_label} unit binding differs",
        )

        points = evidence.get("point_predictions")
        truth = evidence.get("truth")
        row_count = len(block["unit_ids"])
        column_count = len(block["target_names"])
        require(
            isinstance(points, list)
            and isinstance(truth, list)
            and len(points) == len(truth) == row_count,
            f"{item_label} dimensions differ from block rows",
        )
        for matrix_name, matrix in (("point_predictions", points), ("truth", truth)):
            for row_number, row in enumerate(matrix):
                require(
                    isinstance(row, list)
                    and len(row) == column_count
                    and all(_is_binary64(value) for value in row),
                    f"{item_label}.{matrix_name}[{row_number}] must match the binary64 target shape",
                )

        point_hash = tcv1_sha256(points)
        truth_hash = tcv1_sha256(truth)
        require(
            evidence["point_prediction_fingerprint"]
            == point_hash
            == block["point_prediction_fingerprint"]
            == metric_set["point_prediction_fingerprint"],
            f"{item_label} point prediction fingerprint differs",
        )
        require(
            evidence["truth_fingerprint"]
            == truth_hash
            == metric_set["truth_fingerprint"],
            f"{item_label} truth fingerprint differs",
        )
        _validate_fingerprint(evidence, "evidence_fingerprint", item_label)

        reconstructed: dict[tuple[float, str | None], dict[str, Any]] = {}
        for interval in block["intervals"]:
            for row_number, (point_row, lower_row, upper_row) in enumerate(
                zip(points, interval["lower"], interval["upper"])
            ):
                for target_number, (point, lower, upper) in enumerate(
                    zip(point_row, lower_row, upper_row)
                ):
                    if lower is None or upper is None:
                        require(
                            lower is None and upper is None,
                            f"{item_label} unbounded endpoints differ",
                        )
                    else:
                        require(
                            Decimal(repr(lower)) + Decimal(repr(upper))
                            == Decimal(2) * Decimal(repr(point)),
                            f"{item_label} midpoint does not reconstruct at {row_number},{target_number}",
                        )
            summaries = regression_conformal_metrics(
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
                    coordinate not in reconstructed,
                    f"{item_label} reconstructs duplicate metric coordinates",
                )
                reconstructed[coordinate] = summary
        actual = {
            (record["coverage"], record["target_name"]): record
            for record in metric_set["records"]
        }
        require(
            len(actual) == len(metric_set["records"])
            and set(actual) == set(reconstructed),
            f"{item_label} metric coordinates do not reconstruct",
        )
        for coordinate, expected in reconstructed.items():
            require(
                all(
                    actual[coordinate][name] == expected[name]
                    for name in statistic_names
                ),
                f"{item_label} metrics do not reconstruct at {coordinate}",
            )
        ids.append(evidence_id)
        covered_blocks.append(block_key)
        covered_metrics.append(metric_key)

    require(
        ids == sorted(ids) and len(ids) == len(set(ids)),
        f"{label} ids must be sorted and unique",
    )
    require(
        len(covered_blocks) == len(set(covered_blocks))
        and set(covered_blocks) == set(by_block),
        f"{label} must cover every block exactly once",
    )
    require(
        len(covered_metrics) == len(set(covered_metrics))
        and set(covered_metrics) == set(by_metric),
        f"{label} must cover every metric set exactly once",
    )


def reject_runtime_handles(value: Any, label: str = "document") -> None:
    """Refuse opaque/runtime handle fields anywhere in a portable contract."""

    if isinstance(value, list):
        for index, member in enumerate(value):
            reject_runtime_handles(member, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, member in value.items():
            lowered = key.lower()
            require(
                lowered != "handle"
                and not lowered.endswith("_handle")
                and not lowered.endswith("_handles"),
                f"{label}.{key} contains a forbidden runtime handle",
            )
            reject_runtime_handles(member, f"{label}.{key}")


def _sorted_unique_strings(value: Any, label: str, *, non_empty: bool) -> list[str]:
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


def _validate_fingerprint(document: dict[str, Any], field: str, label: str) -> None:
    expected = fingerprint_without(document, field)
    require(document[field] == expected, f"{label}.{field} does not match TCV1 content")


def validate_cohort_manifest(document: Any) -> dict[str, Any]:
    """Validate canonical order, role and TCV1 identity for one cohort."""

    require(isinstance(document, dict), "cohort manifest must be an object")
    reject_runtime_handles(document, "cohort")
    require(document.get("schema_version") == 1, "cohort schema_version must be 1")
    require(document.get("role") in COHORT_ROLES, "cohort role is invalid")
    require(
        document.get("exchangeability_unit") == "physical_sample",
        "cohort unit is invalid",
    )
    physical_sample_ids = _sorted_unique_strings(
        document.get("physical_sample_ids"), "physical_sample_ids", non_empty=True
    )
    for field in ("origin_sample_ids", "group_ids", "source_ids"):
        _sorted_unique_strings(document.get(field), field, non_empty=False)
    relations = document.get("unit_relations")
    require(
        isinstance(relations, list) and relations, "unit_relations must be non-empty"
    )
    require(
        [relation.get("physical_sample_id") for relation in relations]
        == physical_sample_ids,
        "unit_relations must align exactly with physical_sample_ids",
    )
    relation_origins: set[str] = set()
    relation_groups: set[str] = set()
    relation_sources: set[str] = set()
    for index, relation in enumerate(relations):
        require(
            isinstance(relation, dict), f"unit_relations[{index}] must be an object"
        )
        origin = relation.get("origin_sample_id")
        if origin is not None:
            require(
                isinstance(origin, str) and origin,
                f"unit_relations[{index}] origin is invalid",
            )
            relation_origins.add(origin)
        relation_groups.update(
            _sorted_unique_strings(
                relation.get("group_ids"),
                f"unit_relations[{index}].group_ids",
                non_empty=False,
            )
        )
        relation_sources.update(
            _sorted_unique_strings(
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
    require(len(set(targets)) == len(targets), "target_names must be unique")
    _validate_fingerprint(document, "manifest_fingerprint", "cohort")
    return document


def assert_calibration_disjoint(
    training_sample_ids: Any,
    training_origin_sample_ids: Any,
    calibration_cohort: Any,
) -> None:
    """Reject any sample/origin closure overlap with a calibration cohort."""

    cohort = validate_cohort_manifest(calibration_cohort)
    require(cohort["role"] == "calibration", "disjointness requires calibration role")
    training_samples = set(
        _sorted_unique_strings(training_sample_ids, "training samples", non_empty=True)
    )
    training_origins = set(
        _sorted_unique_strings(
            training_origin_sample_ids, "training origins", non_empty=False
        )
    )
    calibration_samples = set(cohort["physical_sample_ids"])
    calibration_origins = set(cohort["origin_sample_ids"])
    overlap = (training_samples | training_origins) & (
        calibration_samples | calibration_origins
    )
    require(not overlap, f"calibration influence overlap: {sorted(overlap)}")


def validate_training_influence(document: Any) -> dict[str, Any]:
    """Validate a portable, canonically ordered training-influence closure."""

    require(isinstance(document, dict), "training influence must be an object")
    reject_runtime_handles(document, "training influence")
    require(
        document.get("schema_version") == 1,
        "training influence schema_version must be 1",
    )
    entries = document.get("entries")
    require(
        isinstance(entries, list) and entries, "training influence entries are empty"
    )
    order = {kind: index for index, kind in enumerate(INFLUENCE_KINDS)}
    coordinates: list[tuple[int, str, str]] = []
    for index, entry in enumerate(entries):
        kind = entry.get("kind")
        require(kind in order, f"training influence entry {index} kind is invalid")
        node_id = entry.get("node_id") or ""
        coordinates.append((order[kind], entry.get("scope_id", ""), node_id))
        _sorted_unique_strings(
            entry.get("physical_sample_ids"),
            f"training influence entry {index} physical_sample_ids",
            non_empty=True,
        )
        for field in ("origin_sample_ids", "group_ids"):
            _sorted_unique_strings(
                entry.get(field),
                f"training influence entry {index} {field}",
                non_empty=False,
            )
    require(
        coordinates == sorted(coordinates), "training influence entries must be sorted"
    )
    require(
        len(set(coordinates)) == len(coordinates),
        "training influence entries contain duplicate coordinates",
    )
    _validate_fingerprint(document, "manifest_fingerprint", "training influence")
    return document


def _influence_identity_closure(document: dict[str, Any]) -> set[str]:
    return {
        identity
        for entry in document["entries"]
        for field in ("physical_sample_ids", "origin_sample_ids")
        for identity in entry[field]
    }


def validate_predictor_binding(document: Any) -> dict[str, Any]:
    """Validate the canonical, explicit transitive predictor closure."""

    require(isinstance(document, dict), "predictor binding must be an object")
    reject_runtime_handles(document, "predictor binding")
    require(
        document.get("schema_version") == 1,
        "predictor binding schema_version must be 1",
    )

    data_bindings = document.get("data_bindings")
    require(
        isinstance(data_bindings, list) and data_bindings,
        "predictor data_bindings must be non-empty",
    )
    requirement_keys: list[str] = []
    requirement_nodes: set[str] = set()
    for index, binding in enumerate(data_bindings):
        require(
            isinstance(binding, dict),
            f"predictor data_bindings[{index}] must be an object",
        )
        requirement_key = binding.get("requirement_key")
        require(
            isinstance(requirement_key, str) and requirement_key,
            f"predictor data_bindings[{index}] requirement_key is invalid",
        )
        requirement_keys.append(requirement_key)
        node_id, separator, _port = requirement_key.rpartition(".")
        require(
            bool(separator and node_id),
            f"predictor data binding {requirement_key!r} has no node prefix",
        )
        requirement_nodes.add(node_id)
        _sorted_unique_strings(
            binding.get("source_ids"),
            f"predictor data_bindings[{index}].source_ids",
            non_empty=False,
        )
    require(
        requirement_keys == sorted(requirement_keys),
        "predictor data_bindings must be sorted by requirement_key",
    )
    require(
        len(set(requirement_keys)) == len(requirement_keys),
        "predictor data_bindings contain duplicate requirement_key values",
    )

    selected_patches = document.get("selected_patches")
    require(isinstance(selected_patches, list), "selected_patches must be an array")
    patch_coordinates: list[tuple[str, str, tuple[str, ...]]] = []
    for index, patch in enumerate(selected_patches):
        require(isinstance(patch, dict), f"selected_patches[{index}] must be an object")
        path = patch.get("path")
        require(
            isinstance(path, list)
            and path
            and all(isinstance(part, str) and part for part in path),
            f"selected_patches[{index}].path is invalid",
        )
        patch_coordinates.append(
            (patch.get("node_id", ""), patch.get("namespace", ""), tuple(path))
        )
    require(
        patch_coordinates == sorted(patch_coordinates),
        "selected_patches must be canonically sorted",
    )
    require(
        len(set(patch_coordinates)) == len(patch_coordinates),
        "selected_patches contain duplicate targets",
    )

    artifacts = document.get("artifacts")
    require(
        isinstance(artifacts, list) and artifacts,
        "predictor artifacts must be non-empty",
    )
    artifact_coordinates: list[tuple[str, str]] = []
    for index, artifact in enumerate(artifacts):
        require(
            isinstance(artifact, dict),
            f"predictor artifacts[{index}] must be an object",
        )
        artifact_coordinates.append(
            (artifact.get("node_id", ""), artifact.get("artifact_id", ""))
        )
    require(
        artifact_coordinates == sorted(artifact_coordinates),
        "predictor artifacts must be sorted by node_id and artifact_id",
    )
    require(
        len(set(artifact_coordinates)) == len(artifact_coordinates),
        "predictor artifacts contain duplicate coordinates",
    )

    output_binding = document.get("output_binding")
    require(isinstance(output_binding, dict), "predictor output_binding is missing")
    _validate_fingerprint(
        output_binding, "binding_fingerprint", "predictor OutputBinding"
    )
    output_node = output_binding.get("node_id")
    require(
        isinstance(output_node, str) and output_node,
        "predictor output_binding node_id is invalid",
    )

    predictor_node_ids = set(
        _sorted_unique_strings(
            document.get("predictor_node_ids"),
            "predictor_node_ids",
            non_empty=True,
        )
    )
    required_nodes = {
        output_node,
        *requirement_nodes,
        *(artifact["node_id"] for artifact in artifacts),
        *(patch["node_id"] for patch in selected_patches),
    }
    require(
        required_nodes <= predictor_node_ids,
        "predictor_node_ids omits a node from the authoritative predictor closure",
    )
    return document


def validate_calibration_artifact(document: Any) -> dict[str, Any]:
    """Validate a complete split-conformal calibration artifact and leakage closure."""

    require(isinstance(document, dict), "calibration artifact must be an object")
    reject_runtime_handles(document, "calibration artifact")
    require(
        document.get("schema_version") == 1,
        "calibration artifact schema_version must be 1",
    )
    predictor = validate_predictor_binding(document.get("predictor_binding"))
    require(
        document.get("predictor_binding_fingerprint") == tcv1_sha256(predictor),
        "calibration predictor binding fingerprint does not match TCV1 content",
    )
    output_binding = predictor["output_binding"]

    spec = document.get("calibration_spec")
    require(isinstance(spec, dict), "calibration_spec must be an object")
    require(
        document.get("calibration_spec_fingerprint") == tcv1_sha256(spec),
        "calibration spec fingerprint does not match TCV1 content",
    )
    require(
        spec.get("method") == "split_absolute_residual", "calibration method is invalid"
    )
    require(
        spec.get("numeric_version") == "split_absolute_residual.v1",
        "calibration numeric version is invalid",
    )
    require(
        spec.get("exchangeability_unit") == "physical_sample",
        "calibration unit is invalid",
    )
    require(
        spec.get("multi_target_policy") in {"marginal", "joint_max"},
        "calibration target policy is invalid",
    )
    require(
        spec.get("small_sample_policy") in {"error", "unbounded"},
        "calibration small-sample policy is invalid",
    )
    coverages = validate_coverages(spec.get("coverages"))

    cohort = validate_cohort_manifest(document.get("calibration_cohort"))
    require(
        cohort["role"] == "calibration",
        "calibration artifact cohort role is not calibration",
    )
    require(
        cohort["target_names"] == output_binding.get("target_names"),
        "calibration targets differ from predictor OutputBinding",
    )
    influence = validate_training_influence(document.get("training_influence"))
    require(
        predictor.get("training_influence_fingerprint")
        == influence["manifest_fingerprint"],
        "predictor binding references another training influence",
    )
    require(
        all(
            binding.get("relation_fingerprint") == influence["relation_fingerprint"]
            for binding in predictor["data_bindings"]
        ),
        "predictor data binding relation_fingerprint differs from training influence",
    )
    overlap = _influence_identity_closure(influence) & (
        set(cohort["physical_sample_ids"]) | set(cohort["origin_sample_ids"])
    )
    require(not overlap, f"calibration influence overlap: {sorted(overlap)}")
    sample_count = len(cohort["physical_sample_ids"])
    require(
        document.get("effective_sample_count") == sample_count,
        "calibration effective_sample_count differs from cohort",
    )
    quantiles = document.get("quantiles")
    require(isinstance(quantiles, list), "calibration quantiles must be an array")
    require(
        [record.get("coverage") for record in quantiles] == coverages,
        "calibration quantiles differ from requested coverages",
    )
    value_count = (
        1 if spec["multi_target_policy"] == "joint_max" else len(cohort["target_names"])
    )
    previous_values: list[float | None] = [None] * value_count
    for record in quantiles:
        rank = finite_sample_rank(sample_count, record["coverage"])
        require(record.get("rank") == rank, "calibration quantile rank drifted")
        values = record.get("values")
        require(
            isinstance(values, list) and len(values) == value_count,
            "calibration quantile target width drifted",
        )
        if rank > sample_count:
            require(
                spec["small_sample_policy"] == "unbounded",
                "small calibration cohort requires unbounded policy",
            )
            require(
                all(value == {"status": "unbounded"} for value in values),
                "unbounded calibration quantile drifted",
            )
        else:
            for target_index, value in enumerate(values):
                require(
                    value.get("status") == "finite",
                    "finite calibration rank has unavailable quantile",
                )
                quantile = _require_binary64(
                    value.get("value"), "calibration quantile value"
                )
                require(quantile >= 0.0, "calibration quantile must be non-negative")
                previous = previous_values[target_index]
                require(
                    previous is None or quantile >= previous,
                    "calibration quantiles must be monotone across coverage",
                )
                previous_values[target_index] = quantile
    _validate_fingerprint(document, "checksum", "calibration artifact")
    return document


def validate_prediction_block(document: Any) -> dict[str, Any]:
    """Validate interval dimensions, nesting and TCV1 identity."""

    require(isinstance(document, dict), "prediction block must be an object")
    reject_runtime_handles(document, "prediction block")
    units = _sorted_unique_strings(document.get("unit_ids"), "unit_ids", non_empty=True)
    targets = document.get("target_names")
    require(isinstance(targets, list) and targets, "target_names must be non-empty")
    binding = document["point_output_binding"]
    require(
        targets == binding["target_names"], "target order differs from OutputBinding"
    )
    _validate_fingerprint(binding, "binding_fingerprint", "point OutputBinding")
    intervals = document.get("intervals")
    require(isinstance(intervals, list) and intervals, "intervals must be non-empty")
    coverages = validate_coverages([interval["coverage"] for interval in intervals])
    previous: tuple[list[list[float | None]], list[list[float | None]]] | None = None
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
                    _is_binary64(lo) and _is_binary64(hi),
                    f"interval {interval_index} cell {row_index},{column_index} must be finite binary64 or null",
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
    _validate_fingerprint(document, "block_fingerprint", "prediction block")
    return document


def _metric_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
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


def validate_metric_set(document: Any) -> dict[str, Any]:
    """Validate conformal metric indexing and arithmetic."""

    require(isinstance(document, dict), "metric set must be an object")
    reject_runtime_handles(document, "metric set")
    records = document.get("records")
    require(isinstance(records, list) and records, "metric records must be non-empty")
    keys = [_metric_record_key(record) for record in records]
    require(keys == sorted(keys), "metric records must be canonically sorted")
    require(len(set(keys)) == len(keys), "metric records contain duplicate coordinates")
    for record in records:
        validate_coverages([record.get("coverage")])
        require(
            record["severity"] is None or _is_binary64(record["severity"]),
            "metric severity must be represented as binary64",
        )
        for field in (
            "empirical_coverage",
            "coverage_gap",
            "mean_width",
            "median_width",
            "interval_score",
            "set_size",
        ):
            require(
                record[field] is None or _is_binary64(record[field]),
                f"metric {field} must be null or finite binary64",
            )
        require(
            isinstance(record.get("unit_ids_fingerprint"), str)
            and len(record["unit_ids_fingerprint"]) == 64,
            "metric record unit_ids_fingerprint is invalid",
        )
        status = record.get("measurement_status")
        if status == "finite":
            require(
                all(
                    _is_binary64(record[field])
                    for field in (
                        "empirical_coverage",
                        "coverage_gap",
                        "mean_width",
                        "median_width",
                        "interval_score",
                    )
                ),
                "finite metric record contains unavailable values",
            )
            expected_gap = record["empirical_coverage"] - record["coverage"]
            require(
                math.isclose(
                    record["coverage_gap"], expected_gap, rel_tol=0.0, abs_tol=1e-12
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
                record["empirical_coverage"] == 1.0
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
    _validate_fingerprint(document, "metric_set_fingerprint", "metric set")
    return document


def validate_domain_assessment(document: Any) -> dict[str, Any]:
    """Validate a domain-only block and identity alignment."""

    require(isinstance(document, dict), "domain block must be an object")
    reject_runtime_handles(document, "domain block")
    units = _sorted_unique_strings(document.get("unit_ids"), "unit_ids", non_empty=True)
    assessments = document.get("assessments")
    require(isinstance(assessments, list), "assessments must be an array")
    ids = [record["unit_id"] for record in assessments]
    require(ids == units, "domain assessments must align exactly with unit_ids")
    for record in assessments:
        methods = record.get("methods")
        require(
            isinstance(methods, list) and methods, "domain methods must be non-empty"
        )
        method_ids = [method.get("method_id") for method in methods]
        require(
            method_ids == sorted(method_ids),
            "domain methods must be sorted by method_id",
        )
        require(
            len(set(method_ids)) == len(method_ids), "domain method ids must be unique"
        )
        for method in methods:
            score = method.get("score")
            threshold = method.get("threshold")
            supported_value = method.get("supported")
            if supported_value is None:
                require(
                    score is None and threshold is None,
                    "unknown domain method must not carry a partial support decision",
                )
            else:
                require(
                    all(_is_binary64(value) for value in (score, threshold)),
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
            record.get("status") == expected_status,
            "domain assessment status contradicts method support",
        )
        _sorted_unique_strings(record.get("reasons"), "domain reasons", non_empty=False)
    _validate_fingerprint(document, "block_fingerprint", "domain block")
    return document


def validate_decision_block(document: Any) -> dict[str, Any]:
    """Validate a decision-only block and identity alignment."""

    require(isinstance(document, dict), "decision block must be an object")
    reject_runtime_handles(document, "decision block")
    units = _sorted_unique_strings(document.get("unit_ids"), "unit_ids", non_empty=True)
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
        value = threshold.get("value")
        if operator in {"lt", "lte", "gt", "gte"}:
            require(
                _is_binary64(value),
                f"threshold {threshold['name']} needs a finite numeric binary64 value",
            )
        elif operator in {"in", "not_in"}:
            require(
                isinstance(value, list),
                f"threshold {threshold['name']} needs an array value",
            )
        if isinstance(value, list):
            require(
                all(not _is_number(member) or _is_binary64(member) for member in value),
                f"threshold {threshold['name']} numeric members must be binary64",
            )
        elif _is_number(value):
            require(
                _is_binary64(value),
                f"threshold {threshold['name']} numeric value must be binary64",
            )
    for decision in decisions:
        _sorted_unique_strings(
            decision.get("reasons"), "decision reasons", non_empty=True
        )
    _validate_fingerprint(document, "block_fingerprint", "decision block")
    return document


def validate_scenario(document: Any) -> dict[str, Any]:
    """Validate mode policy, paired RNG and canonical scenario identity."""

    require(isinstance(document, dict), "scenario must be an object")
    reject_runtime_handles(document, "scenario")
    require(
        document.get("cohort_role") in {"external_test", "production"},
        "scenario cohort role must be evaluation-only",
    )
    _sorted_unique_strings(document.get("source_ids"), "source_ids", non_empty=False)
    _sorted_unique_strings(document.get("node_ids"), "node_ids", non_empty=False)
    _sorted_unique_strings(document.get("slice_by"), "slice_by", non_empty=False)
    _sorted_unique_strings(document.get("metrics"), "metrics", non_empty=True)
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
    mode = document.get("mode")
    perturbation = document.get("perturbation")
    require(isinstance(perturbation, dict), "scenario perturbation must be an object")
    perturbation_kind = perturbation.get("kind")
    expected = {
        "clean_frozen": (False, False),
        "matched_recalibration": (False, True),
        "structural_refit": (True, True),
    }
    require(mode in expected, "scenario mode is invalid")
    require(
        (document.get("requires_refit"), document.get("requires_recalibration"))
        == expected[mode],
        "scenario mode/refit/recalibration policy is inconsistent",
    )
    require(
        (perturbation_kind == "node_replacement") == (mode == "structural_refit"),
        "node_replacement is valid if and only if mode is structural_refit",
    )
    if perturbation_kind == "identity":
        require(
            severities == [0.0],
            "identity perturbation must use exactly severities [0.0]",
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
        require(
            rng.get("target_kind") == "global",
            "global perturbation RNG target_kind drifted",
        )
    _validate_fingerprint(document, "scenario_fingerprint", "scenario")
    return document


def _validate_point_metrics(
    value: Any,
    scenario: dict[str, Any],
    *,
    severity_zero: bool,
    baseline: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    require(isinstance(value, list) and value, "point_metrics must be non-empty")
    names = [record["metric"] for record in value]
    require(names == sorted(names), "point metrics must be sorted")
    require(len(set(names)) == len(names), "point metrics contain duplicates")
    requested_point_metrics = set(scenario["metrics"]) - set(CONFORMAL_METRIC_REQUESTS)
    require(
        set(names) == requested_point_metrics,
        "point metrics do not exactly cover the metrics requested by the scenario",
    )
    if baseline is not None:
        require(set(names) == set(baseline), "point metrics differ from exact baseline")
    for record in value:
        metric = record["metric"]
        if record["status"] == "unavailable":
            require(
                record["value"] is None and record["degradation"] is None,
                "unavailable point metric value and degradation must be null",
            )
            require(not severity_zero, "severity-zero point metric is unavailable")
            require(
                baseline is not None, "unavailable point metric has no exact baseline"
            )
            baseline_record = baseline[metric]
            require(
                baseline_record["status"] == "finite"
                and record["baseline_value"] == baseline_record["value"],
                "unavailable point metric baseline differs from severity-zero value",
            )
            continue
        require(record["status"] == "finite", "point metric status is invalid")
        require(
            all(
                _is_binary64(record[field])
                for field in ("value", "baseline_value", "degradation")
            ),
            "finite point metric must contain binary64 values",
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
    return {record["metric"]: record for record in value}


def _result_coordinate(result: dict[str, Any]) -> tuple[Any, ...]:
    """Return the complete canonical coordinate of a robustness result."""

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


def _baseline_coordinate(result: dict[str, Any]) -> tuple[Any, ...]:
    coordinate = list(_result_coordinate(result))
    coordinate[1] = 0.0
    return tuple(coordinate)


def _slice_units(
    cohort: dict[str, Any], scenario: dict[str, Any], slice_key: dict[str, Any]
) -> list[str]:
    kind = slice_key["kind"]
    value = slice_key["value"]
    relations = {
        relation["physical_sample_id"]: relation
        for relation in cohort["unit_relations"]
    }
    if kind == "all":
        return cohort["physical_sample_ids"]
    require(kind in scenario["slice_by"], "result uses an undeclared slice dimension")
    if kind == "group":
        require(
            value in cohort["group_ids"], "result references an unknown group slice"
        )
        return sorted(
            unit_id
            for unit_id, relation in relations.items()
            if value in relation["group_ids"]
        )
    if kind == "source":
        require(
            value in cohort["source_ids"], "result references an unknown source slice"
        )
        return sorted(
            unit_id
            for unit_id, relation in relations.items()
            if value in relation["source_ids"]
        )
    if kind == "environment":
        require(value == scenario["environment_id"], "result environment slice drifted")
        return cohort["physical_sample_ids"]
    require(kind == "target", "result slice kind is invalid")
    require(
        value in cohort["target_names"], "result references an unknown target slice"
    )
    return cohort["physical_sample_ids"]


def _metric_records_for_result(
    metric_set: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        record
        for record in metric_set["records"]
        if record["scenario_id"] == result["scenario_id"]
        and record["severity"] == result["severity"]
        and record["slice"] == result["slice"]
        and record["fold_id"] == result["fold_id"]
        and record["repeat_id"] == result["repeat_id"]
        and record["seed"] == result["seed"]
    ]


def _validate_block_against_calibrator(
    block: dict[str, Any], artifact: dict[str, Any]
) -> None:
    spec = artifact["calibration_spec"]
    quantiles = {record["coverage"]: record for record in artifact["quantiles"]}
    require(
        block["method"] == spec["method"]
        and block["numeric_version"] == spec["numeric_version"],
        "prediction block method differs from calibrator",
    )
    for interval in block["intervals"]:
        coverage = interval["coverage"]
        require(
            coverage in quantiles,
            "prediction block coverage is not calibrated by its artifact",
        )
        tagged_values = quantiles[coverage]["values"]
        for row_index, (lower_row, upper_row) in enumerate(
            zip(interval["lower"], interval["upper"])
        ):
            for target_index, (lower, upper) in enumerate(zip(lower_row, upper_row)):
                tagged = tagged_values[
                    target_index if spec["multi_target_policy"] == "marginal" else 0
                ]
                if tagged["status"] == "unbounded":
                    require(
                        lower is None and upper is None,
                        "unbounded calibration quantile produced finite endpoints",
                    )
                    continue
                require(
                    isinstance(lower, float) and isinstance(upper, float),
                    "finite calibration quantile requires binary64 endpoints",
                )
                require(
                    Decimal(repr(upper)) - Decimal(repr(lower))
                    == Decimal(2) * Decimal(repr(tagged["value"])),
                    f"prediction interval radius drifted at row {row_index}, target {target_index}",
                )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def _validate_metric_widths_from_block(
    metric_set: dict[str, Any], block: dict[str, Any]
) -> None:
    intervals = {interval["coverage"]: interval for interval in block["intervals"]}
    target_indices = {
        target_name: index for index, target_name in enumerate(block["target_names"])
    }
    for record in metric_set["records"]:
        require(
            record["coverage"] in intervals,
            "metric coverage does not resolve to prediction bounds",
        )
        interval = intervals[record["coverage"]]
        if metric_set["multi_target_policy"] == "joint_max":
            indices = list(range(len(block["target_names"])))
        else:
            require(
                record["target_name"] in target_indices,
                "marginal metric target does not resolve to prediction bounds",
            )
            indices = [target_indices[record["target_name"]]]
        widths: list[float | None] = []
        for lower_row, upper_row in zip(interval["lower"], interval["upper"]):
            for target_index in indices:
                lower = lower_row[target_index]
                upper = upper_row[target_index]
                widths.append(None if lower is None or upper is None else upper - lower)
        if any(width is None for width in widths):
            require(
                record["measurement_status"] == "unbounded"
                and record["mean_width"] is None
                and record["median_width"] is None,
                "unbounded bounds have finite width metrics",
            )
            continue
        finite = [float(width) for width in widths if width is not None]
        require(
            record["measurement_status"] == "finite"
            and math.isclose(
                record["mean_width"],
                sum(finite) / len(finite),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                record["median_width"],
                _median(finite),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "metric mean/median width does not reconstruct from prediction bounds",
        )


def validate_report(document: Any) -> dict[str, Any]:
    """Validate a fully resolved robustness report and all statistical evidence."""

    require(isinstance(document, dict), "report must be an object")
    reject_runtime_handles(document, "report")
    cohort_value = document.get("cohort_manifest")
    require(
        isinstance(cohort_value, dict)
        and cohort_value.get("role") in {"external_test", "production"},
        "robustness report cohort must be external_test or production",
    )
    cohort = validate_cohort_manifest(cohort_value)
    evaluation_identity = set(cohort["physical_sample_ids"]) | set(
        cohort["origin_sample_ids"]
    )

    scenarios = [
        validate_scenario(scenario) for scenario in document.get("scenarios", [])
    ]
    require(scenarios, "report scenarios must be non-empty")
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

    artifacts = [
        validate_calibration_artifact(artifact)
        for artifact in document.get("calibration_artifacts", [])
    ]
    artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
    require(
        artifact_ids == sorted(artifact_ids), "calibration artifacts must be sorted"
    )
    require(
        len(set(artifact_ids)) == len(artifact_ids),
        "calibration artifact ids duplicate",
    )
    checksums = [artifact["checksum"] for artifact in artifacts]
    require(
        len(set(checksums)) == len(checksums),
        "calibration artifact checksums duplicate",
    )
    artifact_map = {artifact["checksum"]: artifact for artifact in artifacts}
    for artifact in artifacts:
        calibration = artifact["calibration_cohort"]
        calibration_identity = set(calibration["physical_sample_ids"]) | set(
            calibration["origin_sample_ids"]
        )
        require(
            not (calibration_identity & evaluation_identity),
            "calibration cohort overlaps evaluation identity closure",
        )
        require(
            not (
                _influence_identity_closure(artifact["training_influence"])
                & evaluation_identity
            ),
            "predictor training influence overlaps evaluation identity closure",
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

    blocks = [
        validate_prediction_block(block)
        for block in document.get("conformal_prediction_blocks", [])
    ]
    block_ids = [block["block_id"] for block in blocks]
    require(
        block_ids == sorted(block_ids), "conformal prediction blocks must be sorted"
    )
    require(
        len(set(block_ids)) == len(block_ids),
        "conformal prediction block ids duplicate",
    )
    block_fingerprints = [block["block_fingerprint"] for block in blocks]
    require(
        len(set(block_fingerprints)) == len(block_fingerprints),
        "prediction block fingerprints duplicate",
    )
    block_map = {block["block_fingerprint"]: block for block in blocks}
    for block in blocks:
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
        _validate_block_against_calibrator(block, artifact)

    metric_sets = [
        validate_metric_set(metric_set)
        for metric_set in document.get("conformal_metric_sets", [])
    ]
    metric_ids = [metric_set["metric_set_id"] for metric_set in metric_sets]
    require(metric_ids == sorted(metric_ids), "report metric sets must be sorted")
    require(
        len(set(metric_ids)) == len(metric_ids), "report metric sets contain duplicates"
    )
    metric_map = {metric_set["metric_set_id"]: metric_set for metric_set in metric_sets}
    for metric_set in metric_sets:
        require(
            metric_set["cohort_manifest_fingerprint"] == cohort["manifest_fingerprint"],
            "metric set is bound to another cohort",
        )
        block_fingerprint = metric_set["conformal_prediction_block_fingerprint"]
        require(
            block_fingerprint in block_map,
            "metric set references an unknown prediction block",
        )
        block = block_map[block_fingerprint]
        require(
            metric_set["calibration_artifact_id"] == block["calibration_artifact_id"],
            "metric set calibration id differs from block",
        )
        require(
            metric_set["calibration_artifact_checksum"]
            == block["calibration_artifact_checksum"],
            "metric set calibrator differs from block",
        )
        require(
            metric_set["predictor_binding_fingerprint"]
            == block["predictor_binding_fingerprint"],
            "metric set predictor differs from block",
        )
        require(
            metric_set["point_prediction_fingerprint"]
            == block["point_prediction_fingerprint"],
            "metric set point prediction differs from block",
        )
        require(
            metric_set["unit_ids_fingerprint"] == tcv1_sha256(block["unit_ids"]),
            "metric set units differ from prediction block",
        )
        require(
            metric_set["multi_target_policy"] == block["multi_target_policy"],
            "metric set target policy differs from block",
        )
        _validate_metric_widths_from_block(metric_set, block)

    results = document.get("results")
    require(isinstance(results, list) and results, "report results must be non-empty")
    coordinates = [_result_coordinate(result) for result in results]
    require(
        coordinates == sorted(coordinates), "report results must be canonically sorted"
    )
    require(
        len(set(coordinates)) == len(coordinates), "report results contain duplicates"
    )
    result_map = {_result_coordinate(result): result for result in results}
    used_artifacts: set[str] = set()
    used_blocks: set[str] = set()
    used_metrics: set[str] = set()

    for result in results:
        scenario_id = result["scenario_id"]
        require(scenario_id in scenario_map, "result references unknown scenario")
        scenario = scenario_map[scenario_id]
        require(
            _is_binary64(result["severity"]),
            "result severity must be represented as finite binary64",
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
        unit_ids = _sorted_unique_strings(
            result.get("unit_ids"), "result unit_ids", non_empty=True
        )
        require(
            result["unit_count"] == len(unit_ids),
            "result unit_count differs from unit_ids",
        )
        require(
            unit_ids == _slice_units(cohort, scenario, result["slice"]),
            "result unit_ids differ from cohort slice relation",
        )
        errors = result.get("errors")
        require(isinstance(errors, list), "result errors must be an array")
        error_keys = [
            (error["phase"], error["code"], error["message"], error["retriable"])
            for error in errors
        ]
        require(
            error_keys == sorted(error_keys), "result errors must be canonically sorted"
        )
        require(
            len(set(error_keys)) == len(error_keys), "result errors contain duplicates"
        )

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
            "result baseline calibration differs from report binding",
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
                used_artifacts.add(checksum)

        block_fingerprint = result["conformal_prediction_block_fingerprint"]
        if block_fingerprint is None:
            require(
                result["coverage_guarantee_status"] == "unavailable",
                "result without prediction block overclaims coverage",
            )
        else:
            require(
                block_fingerprint in block_map,
                "result references unknown prediction block",
            )
            used_blocks.add(block_fingerprint)
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
        else:
            baseline = result_map.get(_baseline_coordinate(result))
            require(
                baseline is not None, "result has no exact severity-zero slice baseline"
            )
            require(baseline["unit_ids"] == unit_ids, "result baseline unit_ids differ")
            for field in (
                "before_predictor_fingerprint",
                "before_input_fingerprint",
                "before_relation_fingerprint",
                "before_point_prediction_fingerprint",
                "before_calibration_checksum",
            ):
                after_field = field.replace("before_", "after_")
                require(
                    result[field] == baseline[after_field],
                    "result before-state differs from exact baseline",
                )
            require(
                result["after_input_fingerprint"] != result["before_input_fingerprint"]
                or after_predictor != before_predictor,
                "positive severity has no observable perturbation or refit",
            )
            if scenario["mode"] == "clean_frozen":
                require(
                    after_predictor == before_predictor,
                    "clean_frozen changed predictor",
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
                    and after_calibration != before_calibration,
                    "matched_recalibration did not create a new calibrator",
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
                require(
                    result["calibration_status"] in {"recalibrated", "invalidated"},
                    "structural calibration status drifted",
                )
                if result["calibration_status"] == "recalibrated":
                    require(
                        after_calibration is not None
                        and after_calibration != before_calibration,
                        "structural refit has no new calibrator",
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
                        after_calibration is None
                        and result["coverage_guarantee_status"] == "unavailable",
                        "invalidated structural calibration overclaims coverage",
                    )
            if result["calibration_status"] == "recalibrated":
                require(
                    after_calibration is not None,
                    "recalibrated result has no calibration artifact",
                )
                recalibration_artifact = artifact_map[after_calibration]
                diagnostics = recalibration_artifact["diagnostics"]
                require(
                    _is_binary64(diagnostics.get("severity")),
                    "recalibration diagnostics severity must be finite binary64",
                )
                require(
                    diagnostics.get("scenario_id") == scenario_id
                    and diagnostics.get("severity") == result["severity"],
                    "recalibration diagnostics do not identify the exact scenario and severity",
                )
                require(
                    diagnostics.get("calibration_input_fingerprint")
                    == recalibration_artifact["calibration_cohort"][
                        "content_fingerprint"
                    ],
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
                result["slice"]["kind"] == "all",
                "sliced result overclaims formal coverage",
            )
            require(
                result["calibration_status"] in {"reused", "recalibrated"},
                "formal coverage has no valid calibrator",
            )

    require(
        used_artifacts == set(checksums),
        "calibration artifacts are incomplete or unused",
    )
    require(
        used_blocks == set(block_fingerprints),
        "prediction blocks are incomplete or unused",
    )
    for scenario in scenarios:
        for severity in scenario["severities"]:
            expected_slices = [{"kind": "all", "value": None}]
            for slice_kind in scenario["slice_by"]:
                if slice_kind == "group":
                    expected_slices.extend(
                        {"kind": "group", "value": value}
                        for value in cohort["group_ids"]
                    )
                elif slice_kind == "source":
                    expected_slices.extend(
                        {"kind": "source", "value": value}
                        for value in cohort["source_ids"]
                    )
                elif slice_kind == "environment":
                    expected_slices.append(
                        {"kind": "environment", "value": scenario["environment_id"]}
                    )
                else:
                    require(slice_kind == "target", "scenario slice kind is invalid")
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
                    f"has no exact {expected_slice} slice",
                )

    conformal_metric_names = {
        "empirical_coverage",
        "coverage_gap",
        "mean_width",
        "median_width",
        "interval_score",
        "set_size",
    }
    matched_records: set[tuple[str, int]] = set()
    for result in results:
        baseline_metrics = None
        if result["severity"] != 0:
            baseline = result_map[_baseline_coordinate(result)]
            baseline_metrics = {
                record["metric"]: record for record in baseline["point_metrics"]
            }
        point_metrics = _validate_point_metrics(
            result["point_metrics"],
            scenario_map[result["scenario_id"]],
            severity_zero=result["severity"] == 0,
            baseline=baseline_metrics,
        )
        requested_conformal_metrics = {
            CONFORMAL_METRIC_REQUESTS[name]
            for name in scenario_map[result["scenario_id"]]["metrics"]
            if name in CONFORMAL_METRIC_REQUESTS
        }
        explicitly_unavailable = {
            CONFORMAL_METRIC_REQUESTS[metric_name]
            for error in result["errors"]
            if error["phase"] == "score"
            and error["code"].startswith("metric_unavailable.")
            and (metric_name := error["code"].removeprefix("metric_unavailable."))
            in CONFORMAL_METRIC_REQUESTS
        }
        metric_id = result["conformal_metric_set_id"]
        records: list[dict[str, Any]] = []
        if metric_id is None:
            require(
                requested_conformal_metrics <= explicitly_unavailable,
                "result omits conformal metrics requested by the scenario",
            )
            require(
                result["conformal_prediction_block_fingerprint"] is None,
                "result with a prediction block omits its conformal metric set",
            )
        else:
            require(metric_id in metric_map, "result references unknown metric set")
            used_metrics.add(metric_id)
            metric_set = metric_map[metric_id]
            records = _metric_records_for_result(metric_set, result)
            require(records, "result metric set has no matching record")
            require(
                metric_set["predictor_binding_fingerprint"]
                == result["after_predictor_fingerprint"],
                "result metric set is bound to another predictor",
            )
            require(
                metric_set["calibration_artifact_checksum"]
                == result["after_calibration_checksum"],
                "result metric set is bound to another calibrator",
            )
            require(
                metric_set["conformal_prediction_block_fingerprint"]
                == result["conformal_prediction_block_fingerprint"],
                "result metric set is bound to another prediction block",
            )
            require(
                metric_set["point_prediction_fingerprint"]
                == result["after_point_prediction_fingerprint"],
                "result metric set point prediction drifted",
            )
            block = block_map[result["conformal_prediction_block_fingerprint"]]
            expected_metric_coordinates = {
                (interval["coverage"], target_name)
                for interval in block["intervals"]
                for target_name in (
                    [None]
                    if metric_set["multi_target_policy"] == "joint_max"
                    else block["target_names"]
                )
            }
            require(
                {(record["coverage"], record["target_name"]) for record in records}
                == expected_metric_coordinates,
                "result metric records do not cover every requested coverage and target",
            )
            for index, record in enumerate(metric_set["records"]):
                if record in records:
                    matched_records.add((metric_id, index))
                    require(
                        record["sample_count"] == result["unit_count"],
                        "metric sample_count differs from result",
                    )
                    require(
                        record["unit_ids_fingerprint"]
                        == tcv1_sha256(result["unit_ids"]),
                        "metric units differ from result",
                    )
                    require(
                        record["guarantee_status"]
                        == result["coverage_guarantee_status"],
                        "metric guarantee differs from result",
                    )
                    if metric_set["multi_target_policy"] == "marginal":
                        require(
                            record["target_name"] in cohort["target_names"],
                            "metric target is outside cohort",
                        )
        confidence_intervals = result["confidence_intervals"]
        require(
            isinstance(confidence_intervals, list),
            "confidence_intervals must be an array",
        )
        for interval in confidence_intervals:
            level = _require_binary64(
                interval.get("level"), "confidence interval level"
            )
            require(
                0.0 < level < 1.0,
                "confidence interval level must be in (0, 1)",
            )
            _require_binary64(interval.get("lower"), "confidence interval lower")
            _require_binary64(interval.get("upper"), "confidence interval upper")
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
                    point_metrics[interval["metric"]]["status"] == "finite",
                    "point CI references an unavailable metric",
                )
                require(
                    interval["coverage"] is None and interval["target_name"] is None,
                    "point CI carries conformal coordinates",
                )
            else:
                require(metric_id is not None, "conformal CI has no metric set")
                require(
                    interval["metric"] in conformal_metric_names,
                    "conformal CI metric is invalid",
                )
                require(
                    interval["metric"] in requested_conformal_metrics,
                    "conformal CI metric was not requested by the scenario",
                )
                _require_binary64(interval.get("coverage"), "conformal CI coverage")
                matching_records = [
                    record
                    for record in records
                    if record["coverage"] == interval["coverage"]
                    and record["target_name"] == interval["target_name"]
                ]
                require(
                    bool(matching_records),
                    "conformal CI has no matching metric record",
                )
                require(
                    all(
                        _is_binary64(record[interval["metric"]])
                        for record in matching_records
                    ),
                    "conformal CI references an unavailable metric value",
                )

    require(used_metrics == set(metric_ids), "metric sets are incomplete or unused")
    require(
        matched_records
        == {
            (metric_set["metric_set_id"], index)
            for metric_set in metric_sets
            for index, _record in enumerate(metric_set["records"])
        },
        "metric set contains an orphan record",
    )
    provenance = document.get("provenance")
    require(isinstance(provenance, dict), "report provenance must be an object")
    _sorted_unique_strings(
        provenance.get("run_ids"), "provenance.run_ids", non_empty=True
    )
    _sorted_unique_strings(
        provenance.get("artifact_checksums"),
        "provenance.artifact_checksums",
        non_empty=False,
    )
    require(
        set(checksums) <= set(provenance["artifact_checksums"]),
        "provenance omits a calibration artifact checksum",
    )
    require(
        provenance["relation_fingerprint"] == cohort["relation_fingerprint"],
        "report provenance relation differs from cohort",
    )
    _sorted_unique_strings(document.get("warnings"), "report warnings", non_empty=False)
    _validate_fingerprint(document, "report_fingerprint", "report")
    return document


def apply_json_pointer_mutation(document: Any, path: str, value: Any) -> Any:
    """Return a deep copy with one RFC 6901 replacement applied."""

    require(isinstance(path, str) and path.startswith("/"), "mutation path is invalid")
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/")
    ]
    require(all(tokens), "mutation path contains an empty token")
    mutated = copy.deepcopy(document)
    cursor = mutated
    for token in tokens[:-1]:
        cursor = cursor[int(token)] if isinstance(cursor, list) else cursor[token]
    final = tokens[-1]
    if isinstance(cursor, list):
        cursor[int(final)] = copy.deepcopy(value)
    else:
        cursor[final] = copy.deepcopy(value)
    return mutated


def semantic_validator(schema_name: str) -> Callable[[Any], dict[str, Any]]:
    """Return the semantic validator associated with one published schema."""

    validators: dict[str, Callable[[Any], dict[str, Any]]] = {
        "conformal_calibration.schema.json": validate_calibration_artifact,
        "cohort_manifest.schema.json": validate_cohort_manifest,
        "conformal_prediction_block.schema.json": validate_prediction_block,
        "conformal_metric_set.schema.json": validate_metric_set,
        "domain_assessment_block.schema.json": validate_domain_assessment,
        "decision_block.schema.json": validate_decision_block,
        "robustness_scenario_spec.schema.json": validate_scenario,
        "robustness_report.schema.json": validate_report,
    }
    try:
        return validators[schema_name]
    except KeyError as exc:
        raise ContractError(f"no semantic validator for {schema_name}") from exc


__all__ = [
    "ContractError",
    "apply_json_pointer_mutation",
    "assert_calibration_disjoint",
    "file_sha256",
    "finite_sample_rank",
    "fingerprint_without",
    "load_json",
    "regression_conformal_metrics",
    "semantic_validator",
    "split_absolute_residual",
    "tcv1_preimage",
    "tcv1_sha256",
    "validate_cohort_manifest",
    "validate_calibration_artifact",
    "validate_decision_block",
    "validate_domain_assessment",
    "validate_metric_set",
    "validate_numeric_evidence",
    "validate_predictor_binding",
    "validate_prediction_block",
    "validate_report",
    "validate_scenario",
    "validate_training_influence",
]
