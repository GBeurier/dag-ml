"""Independent semantic oracle for W1-0 training contracts.

This module deliberately imports no DAG-ML production package.  It implements
the cross-document and capability semantics that JSON Schema cannot express,
using the independent Python TCV1 implementation shared by the W0 parity kit.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from json.encoder import encode_basestring
from typing import Any

from parity.conformal.oracle import (  # test-only independent implementation
    ContractError,
    fingerprint_without,
    require,
    tcv1_sha256,
)

IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
NAMESPACE_ORDER = {
    "operator": 0,
    "fit": 1,
    "control": 2,
    "structural": 3,
}
INFLUENCE_ORDER = {
    "transform_fit": 0,
    "model_fit": 1,
    "hpo_selection": 2,
    "early_stopping": 3,
    "weighting_resampling": 4,
    "trained_meta_aggregation": 5,
}
ACTIVE_CAPABILITIES = {
    "performs_internal_tuning": "hpo_selection",
    "uses_early_stopping": "early_stopping",
    "uses_training_weights": "weighting_resampling",
}
CAPABILITY_ORDER = {
    name: index
    for index, name in enumerate(
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
PHASE_ORDER = {
    name: index
    for index, name in enumerate(
        ("COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN")
    )
}


def _sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase SHA-256",
    )
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    unknown = set(value) - expected
    missing = expected - set(value)
    require(not unknown, f"{label} has unknown field(s): {sorted(unknown)}")
    require(not missing, f"{label} is missing field(s): {sorted(missing)}")
    return value


def _identifier(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
        f"{label} must be a DAG-ML identifier",
    )
    return value


def _non_blank(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and bool(value.strip()), f"{label} must be non-blank"
    )
    return value


def _sorted_unique(
    values: Any,
    label: str,
    *,
    non_empty: bool = False,
    identifiers: bool = True,
) -> list[str]:
    require(isinstance(values, list), f"{label} must be an array")
    if non_empty:
        require(bool(values), f"{label} must be non-empty")
    for value in values:
        if identifiers:
            _identifier(value, f"{label} entry")
        else:
            _non_blank(value, f"{label} entry")
    require(values == sorted(set(values)), f"{label} must be sorted and unique")
    return values


def validate_data_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _exact_keys(
        value,
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
    _non_blank(identity["requirement_key"], f"{label}.requirement_key")
    for field in (
        "schema_fingerprint",
        "plan_fingerprint",
        "relation_fingerprint",
        "data_content_fingerprint",
        "target_content_fingerprint",
        "identity_fingerprint",
    ):
        _sha256(identity[field], f"{label}.{field}")
    require(
        identity["identity_fingerprint"]
        == fingerprint_without(identity, "identity_fingerprint"),
        f"{label}.identity_fingerprint mismatch",
    )
    return identity


def _graph_nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = graph.get("nodes")
    require(isinstance(nodes, list), "graph.nodes must be an array")
    result = {node["id"]: node for node in nodes}
    require(len(result) == len(nodes), "graph node ids must be unique")
    return result


def _graph_prediction_ports(graph: dict[str, Any], node_id: str) -> list[str]:
    node = _graph_nodes(graph).get(node_id)
    require(node is not None, f"output node {node_id} is absent")
    return [
        port["name"]
        for port in node.get("ports", {}).get("outputs", [])
        if port["kind"] == "prediction"
    ]


def _graph_closure(graph: dict[str, Any], roots: list[str]) -> list[str]:
    nodes = _graph_nodes(graph)
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for edge in graph.get("edges", []):
        incoming[edge["target"]["node_id"]].append(edge["source"]["node_id"])
    pending = list(roots)
    closure: set[str] = set()
    while pending:
        node_id = pending.pop()
        require(node_id in nodes, f"unknown predictor closure node {node_id}")
        if node_id in closure:
            continue
        closure.add(node_id)
        pending.extend(incoming[node_id])
    return sorted(closure)


def _graph_edge_adjacency(
    graph: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Sorted node ids, downstream adjacency (edge-multiplicity) and in-degrees."""

    node_ids = sorted(_graph_nodes(graph))
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


def _canonical_topological_order(graph: dict[str, Any]) -> list[str]:
    """Deterministic lexicographic Kahn order (mirrors Graph::topological_order)."""

    node_ids, adjacency, indegree = _graph_edge_adjacency(graph)
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


def _canonical_parallel_levels(graph: dict[str, Any]) -> list[list[str]]:
    """Canonical dependency levels (mirrors Graph::parallel_levels)."""

    node_ids, adjacency, indegree = _graph_edge_adjacency(graph)
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


def _graph_upstream(graph: dict[str, Any], node_id: str) -> list[str]:
    return sorted(
        {
            edge["source"]["node_id"]
            for edge in graph.get("edges", [])
            if edge["target"]["node_id"] == node_id
        }
    )


def _graph_downstream(graph: dict[str, Any], node_id: str) -> list[str]:
    return sorted(
        {
            edge["target"]["node_id"]
            for edge in graph.get("edges", [])
            if edge["source"]["node_id"] == node_id
        }
    )


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
# ``scripts/validate_contracts.py`` for oracle independence.
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
                "transform": lambda values: _sorted_enum_set(values, PHASE_ORDER),
            },
            {"name": "input_ports", "default": lambda: [], "transform": _norm_ports},
            {"name": "output_ports", "default": lambda: [], "transform": _norm_ports},
            {"name": "data_requirements", "default": lambda: None, "transform": _V},
            {
                "name": "capabilities",
                "default": lambda: [],
                "transform": lambda values: _sorted_enum_set(values, CAPABILITY_ORDER),
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
                "transform": lambda values: _sorted_enum_set(values, PHASE_ORDER),
            },
            {
                "name": "controller_capabilities",
                "default": lambda: [],
                "transform": lambda values: _sorted_enum_set(values, CAPABILITY_ORDER),
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
                "transform": lambda values: _sorted_enum_set(values, NAMESPACE_ORDER),
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


def _oof_requirement_key(
    producer_node: str, source_port: str, consumer_node: str, target_port: str
) -> str:
    """Mirror ``bundle::bundle_prediction_requirement_key`` exactly."""

    return f"{producer_node}.{source_port}->{consumer_node}.{target_port}"


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


def _validate_execution_plan(plan: dict[str, Any], label: str) -> None:
    """Independently validate an ExecutionPlan without trusting node-plan copies.

    Recomputes graph topology (lexicographic Kahn + canonical levels), requires
    exact graph-node-id/node-plan-key agreement, GraphNode.kind == NodePlan.kind,
    every ControllerManifest's map key/id and canonical unique phases/capabilities,
    every NodePlan field copy against its manifest, sorted/de-duplicated
    input/output adjacency rebuilt from real edges, and each params_fingerprint.
    """

    require(isinstance(plan, dict), f"{label} must be an object")
    raw_manifests = plan.get("controller_manifests")
    if isinstance(raw_manifests, dict):
        for controller_id, manifest in raw_manifests.items():
            _validate_controller_manifest_deserialize_shape(
                manifest, f"{label}.controller_manifests[{controller_id}]"
            )
    # ExecutionPlan itself is not self-fingerprinted at its parse boundary.
    # Rust validates the deserialized value (defaults injected, skipped fields
    # removed, BTreeSets sorted/de-duplicated), so semantic validation must use
    # that same typed view. Self-fingerprinted parents gate raw-vs-typed drift.
    plan = _normalize_execution_plan(plan)
    graph = plan["graph_plan"]["graph"]
    graph_nodes = _graph_nodes(graph)
    node_ids = set(graph_nodes)

    topological_order = plan["graph_plan"]["topological_order"]
    # The serialized order must cover every graph node. A cyclic graph makes the
    # canonical lexicographic Kahn order partial (the cyclic nodes never reach
    # in-degree 0), so full-coverage + canonical-equality together reject a cycle
    # even when the fixture supplies exactly that partial order.
    require(
        set(topological_order) == node_ids and len(topological_order) == len(node_ids),
        f"{label} topological_order must cover every graph node (graph is cyclic)",
    )
    require(
        topological_order == _canonical_topological_order(graph),
        f"{label} topological_order does not match the graph canonical order",
    )
    parallel_levels = plan["graph_plan"].get("parallel_levels", [])
    # Gate on the raw outer list, never on flattened ids: Rust rejects any
    # non-empty outer list (e.g. `[[]]`) unless it equals the canonical levels.
    if parallel_levels:
        require(
            {node_id for level in parallel_levels for node_id in level} == node_ids,
            f"{label}.parallel_levels must cover every graph node",
        )
        require(
            parallel_levels == _canonical_parallel_levels(graph),
            f"{label}.parallel_levels are not the canonical dependency levels",
        )

    # Recompute the three embedded top-level plan fingerprints exactly as Rust
    # ExecutionPlan::validate does (SHA-256 of serde_json::to_vec of the typed
    # value in struct/BTreeMap field order); a serialized value is never trusted,
    # so a stale or forged graph/campaign/controller fingerprint is rejected.
    require(
        plan.get("graph_fingerprint") == _serde_sha256(_normalize_graph_spec(graph)),
        f"{label}.graph_fingerprint does not match the embedded graph",
    )
    require(
        plan.get("campaign_fingerprint")
        == _serde_sha256(_normalize_campaign_spec(plan.get("campaign"))),
        f"{label}.campaign_fingerprint does not match the embedded campaign",
    )
    require(
        plan.get("controller_fingerprint")
        == _serde_sha256(
            _normalize_controller_manifests(plan.get("controller_manifests"))
        ),
        f"{label}.controller_fingerprint does not match the embedded controller manifests",
    )

    manifests = plan["controller_manifests"]
    require(isinstance(manifests, dict) and manifests, f"{label}.controller_manifests")
    for controller_id, manifest in manifests.items():
        require(
            controller_id == manifest["controller_id"],
            f"{label}.controller_manifests key {controller_id} mismatch",
        )
        _validate_controller_manifest_semantics(
            manifest, f"{label}.controller_manifests[{controller_id}]"
        )

    node_plans = plan["node_plans"]
    require(
        set(node_plans) == node_ids,
        f"{label}.node_plans keys must equal graph node ids",
    )
    for node_id, node_plan in node_plans.items():
        node_label = f"{label}.node_plans[{node_id}]"
        require(node_plan["node_id"] == node_id, f"{node_label}.node_id must match key")
        require(
            node_plan["kind"] == graph_nodes[node_id]["kind"],
            f"{node_label} node plan kind does not match graph node kind",
        )
        controller_id = node_plan["controller_id"]
        require(
            controller_id in manifests, f"{node_label}.controller_id has no manifest"
        )
        manifest = manifests[controller_id]
        require(
            node_plan["controller_version"] == manifest["controller_version"]
            and node_plan["supported_phases"] == manifest["supported_phases"]
            and node_plan["controller_capabilities"] == manifest["capabilities"]
            and node_plan["fit_scope"] == manifest["fit_scope"]
            and node_plan["rng_policy"] == manifest["rng_policy"]
            and node_plan["artifact_policy"] == manifest["artifact_policy"]
            and node_plan["kind"] == manifest["operator_kind"],
            f"{node_label} node plan does not match controller manifest {controller_id}",
        )
        require(
            node_plan["input_nodes"] == _graph_upstream(graph, node_id)
            and node_plan["output_nodes"] == _graph_downstream(graph, node_id),
            f"{node_label} input/output adjacency does not match the graph",
        )
        require(
            node_plan["params_fingerprint"]
            == _node_params_fingerprint(node_plan["params"]),
            f"{node_label} node plan params fingerprint does not match params",
        )


def _validate_refit_artifacts_against_plan(
    bundle: dict[str, Any], plan: dict[str, Any], label: str
) -> None:
    """Mirror ExecutionBundle::validate_against_plan refit-artifact identity checks.

    Each refit-artifact record must reference a real plan node, and its
    controller_id, nested artifact.controller_id and params_fingerprint must match
    the owning NodePlan (the effective plan is materialized, so the expected params
    fingerprint is the node plan's own params_fingerprint). This refuses a
    re-signed bundle/package whose refit-artifact provenance was forged, mirroring
    ``RefitArtifactRecord::validate`` and the artifact loop in
    ``ExecutionBundle::validate_against_plan``.
    """

    node_plans = plan["node_plans"]
    for index, record in enumerate(bundle.get("refit_artifacts", [])):
        record_label = f"{label}.refit_artifacts[{index}]"
        node_id = record["node_id"]
        require(
            node_id in node_plans, f"{record_label} references unknown node {node_id}"
        )
        node_plan = node_plans[node_id]
        require(
            record["artifact"]["controller_id"] == record["controller_id"],
            f"{record_label} nested artifact controller does not match record controller",
        )
        require(
            record["controller_id"] == node_plan["controller_id"],
            f"{record_label} artifact controller does not match plan",
        )
        require(
            record["params_fingerprint"] == node_plan["params_fingerprint"],
            f"{record_label} artifact params do not match plan",
        )


def _validate_bundle_data_requirements_against_plan(
    bundle: dict[str, Any], plan: dict[str, Any], label: str
) -> None:
    """Mirror ExecutionBundle's exact DataBinding requirement cross-link."""

    expected: dict[str, dict[str, Any]] = {}
    for node_id, node_plan in plan["node_plans"].items():
        for binding in node_plan.get("data_bindings", []):
            key = f"{node_id}.{binding['input_name']}"
            require(
                key not in expected, f"{label} duplicate plan data requirement {key}"
            )
            expected[key] = binding

    actual: dict[str, dict[str, Any]] = {}
    for index, requirement in enumerate(bundle.get("data_requirements", [])):
        requirement_label = f"{label}.data_requirements[{index}]"
        key = f"{requirement['node_id']}.{requirement['input_name']}"
        require(key not in actual, f"{requirement_label} duplicates {key}")
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


def _validate_bundle_selection_and_prediction_links(
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
        _validate_bundle_cache_requirement(
            requirement, f"{label}.prediction_requirements[{index}]"
        )
        for index, requirement in enumerate(bundle.get("prediction_requirements", []))
    ]
    records = [
        _validate_bundle_cache_record(record, f"{label}.prediction_caches[{index}]")
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


def _validate_portable_caches_against_bundle(
    portable_caches: dict[str, Any] | None,
    bundle: dict[str, Any],
    label: str,
) -> None:
    """Independently validate portable payloads and their bundle cross-links.

    This mirrors ``BundlePredictionCachePayloadSet::validate_against_bundle``:
    payload order is immaterial, requirement/cache ids are unique, every payload
    is internally self-fingerprinted with Rust serde field order, and its complete
    metadata plus ordered block records must match the execution bundle.
    """

    bundle_caches = bundle.get("prediction_caches", [])
    require(isinstance(bundle_caches, list), f"{label} bundle caches must be an array")
    if portable_caches is None:
        require(
            not bundle_caches,
            f"{label} cannot be null while the execution bundle announces caches",
        )
        return

    payload_set = _cache_exact_keys(
        portable_caches,
        {"bundle_id", "schema_version", "caches"},
        set(),
        label,
    )
    _identifier(payload_set["bundle_id"], f"{label}.bundle_id")
    require(payload_set["schema_version"] == 1, f"{label}.schema_version must be 1")
    require(
        payload_set["bundle_id"] == bundle.get("bundle_id"),
        f"{label}.bundle_id does not match the execution bundle",
    )
    caches = payload_set["caches"]
    require(isinstance(caches, list), f"{label}.caches must be an array")
    payloads = [
        _validate_portable_cache_payload(payload, f"{label}.caches[{index}]")
        for index, payload in enumerate(caches)
    ]
    payload_keys = [payload["requirement_key"] for payload in payloads]
    cache_ids = [payload["cache_id"] for payload in payloads]
    require(
        len(set(payload_keys)) == len(payload_keys),
        f"{label} has duplicate requirement keys",
    )
    require(len(set(cache_ids)) == len(cache_ids), f"{label} has duplicate cache ids")

    records = [
        _validate_bundle_cache_record(record, f"{label}.bundle_cache[{index}]")
        for index, record in enumerate(bundle_caches)
    ]
    records_by_key = {record["requirement_key"]: record for record in records}
    require(
        len(records_by_key) == len(records),
        f"{label} execution bundle has duplicate cache requirements",
    )
    bundle_requirements = bundle.get("prediction_requirements", [])
    require(
        isinstance(bundle_requirements, list),
        f"{label} bundle prediction requirements must be an array",
    )
    requirements = [
        _validate_bundle_cache_requirement(
            requirement, f"{label}.bundle_requirement[{index}]"
        )
        for index, requirement in enumerate(bundle_requirements)
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
    require(
        len(payloads) == len(records) and set(payload_keys) == set(records_by_key),
        f"{label} payloads do not exactly cover execution bundle cache records",
    )

    for index, payload in enumerate(payloads):
        payload_label = f"{label}.caches[{index}]"
        record = records_by_key[payload["requirement_key"]]
        requirement = requirements_by_key[payload["requirement_key"]]
        for field in (
            "partition",
            "prediction_level",
            "unit_ids",
            "prediction_width",
            "target_names",
        ):
            require(
                record[field] == requirement[field],
                f"{payload_label} cache record {field} does not match its requirement",
            )
        for field in ("fold_ids", "sample_ids"):
            require(
                record[field] == requirement[field],
                f"{payload_label} cache record {field} does not match its requirement",
            )
        for field in (
            "cache_id",
            "cache_namespace_fingerprints",
            "format",
            "partition",
            "prediction_level",
            "block_count",
            "row_count",
            "content_fingerprint",
        ):
            require(
                payload[field] == record[field],
                f"{payload_label}.{field} does not match its cache record",
            )
        require(
            payload["derived_records"] == record["validated_blocks"],
            f"{payload_label} block records do not match its cache record",
        )


def _cache_exact_keys(
    value: Any,
    required: set[str],
    optional: set[str],
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(
        not (set(value) - required - optional),
        f"{label} has unknown field(s): {sorted(set(value) - required - optional)}",
    )
    require(
        not (required - set(value)),
        f"{label} is missing field(s): {sorted(required - set(value))}",
    )
    return value


def _cache_positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def _cache_unit_ids(
    value: Any, expected_level: str, label: str
) -> list[dict[str, str]]:
    require(isinstance(value, list), f"{label} must be an array")
    keys: list[tuple[str, str]] = []
    normalized: list[dict[str, str]] = []
    for index, unit in enumerate(value):
        unit_label = f"{label}[{index}]"
        parsed = _cache_exact_keys(unit, {"level", "id"}, set(), unit_label)
        require(
            parsed["level"] == expected_level,
            f"{unit_label}.level is inconsistent",
        )
        _identifier(parsed["id"], f"{unit_label}.id")
        keys.append((parsed["level"], parsed["id"]))
        normalized.append({"level": parsed["level"], "id": parsed["id"]})
    require(len(keys) == len(set(keys)), f"{label} contains duplicate units")
    return normalized


def _cache_unique_identifiers(
    value: Any, label: str, *, non_empty: bool = False
) -> list[str]:
    require(isinstance(value, list), f"{label} must be an array")
    if non_empty:
        require(bool(value), f"{label} must be non-empty")
    for index, item in enumerate(value):
        _identifier(item, f"{label}[{index}]")
    require(len(value) == len(set(value)), f"{label} must contain unique identifiers")
    return value


def _cache_target_names(value: Any, width: int, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be an array")
    for index, name in enumerate(value):
        _non_blank(name, f"{label}[{index}]")
    require(len(value) == len(set(value)), f"{label} must be unique")
    require(not value or len(value) == width, f"{label} width mismatch")
    return value


def _cache_prediction_matrix(
    value: Any, row_count: int, label: str
) -> tuple[list[list[int | float]], int]:
    require(
        isinstance(value, list) and len(value) == row_count and bool(value),
        f"{label} row count does not match identifiers",
    )
    width = len(value[0]) if isinstance(value[0], list) else 0
    require(width > 0, f"{label} must have positive width")
    for row_index, row in enumerate(value):
        require(
            isinstance(row, list) and len(row) == width,
            f"{label}[{row_index}] has inconsistent width",
        )
        for column_index, number in enumerate(row):
            require(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number),
                f"{label}[{row_index}][{column_index}] must be finite numeric",
            )
    return value, width


def _validate_portable_sample_block(value: Any, label: str) -> dict[str, Any]:
    block = _cache_exact_keys(
        value,
        {"producer_node", "partition", "fold_id", "sample_ids", "values"},
        {"prediction_id", "target_names"},
        label,
    )
    prediction_id = block.get("prediction_id")
    if prediction_id is not None:
        _non_blank(prediction_id, f"{label}.prediction_id")
    _identifier(block["producer_node"], f"{label}.producer_node")
    require(block["partition"] == "validation", f"{label}.partition must be validation")
    if block["fold_id"] is not None:
        _identifier(block["fold_id"], f"{label}.fold_id")
    samples = _cache_unique_identifiers(
        block["sample_ids"], f"{label}.sample_ids", non_empty=True
    )
    values, width = _cache_prediction_matrix(
        block["values"], len(samples), f"{label}.values"
    )
    targets = _cache_target_names(
        block.get("target_names", []), width, f"{label}.target_names"
    )
    canonical = _norm_prediction_block(
        {
            "prediction_id": prediction_id,
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
        "record": {
            "prediction_id": prediction_id,
            "fold_id": block["fold_id"],
            "prediction_level": "sample",
            "row_count": len(samples),
            "unit_ids": [],
            "sample_ids": samples,
            "content_fingerprint": _serde_sha256(canonical),
        },
    }


def _validate_portable_aggregated_block(value: Any, label: str) -> dict[str, Any]:
    block = _cache_exact_keys(
        value,
        {
            "producer_node",
            "partition",
            "fold_id",
            "level",
            "unit_ids",
            "values",
        },
        {"prediction_id", "target_names"},
        label,
    )
    prediction_id = block.get("prediction_id")
    if prediction_id is not None:
        _non_blank(prediction_id, f"{label}.prediction_id")
    _identifier(block["producer_node"], f"{label}.producer_node")
    require(block["partition"] == "validation", f"{label}.partition must be validation")
    if block["fold_id"] is not None:
        _identifier(block["fold_id"], f"{label}.fold_id")
    level = block["level"]
    require(level in {"target", "group"}, f"{label}.level must be target or group")
    units = _cache_unit_ids(block["unit_ids"], level, f"{label}.unit_ids")
    require(bool(units), f"{label}.unit_ids must be non-empty")
    values, width = _cache_prediction_matrix(
        block["values"], len(units), f"{label}.values"
    )
    targets = _cache_target_names(
        block.get("target_names", []), width, f"{label}.target_names"
    )
    canonical = _norm_aggregated_prediction_block(
        {
            "prediction_id": prediction_id,
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
        "record": {
            "prediction_id": prediction_id,
            "fold_id": block["fold_id"],
            "prediction_level": level,
            "row_count": len(units),
            "unit_ids": units,
            "sample_ids": [],
            "content_fingerprint": _serde_sha256(canonical),
        },
    }


def _validate_portable_cache_payload(value: Any, label: str) -> dict[str, Any]:
    payload = _cache_exact_keys(
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
        {
            "prediction_level",
            "blocks",
            "aggregated_blocks",
            "cache_namespace_fingerprints",
        },
        label,
    )
    _non_blank(payload["requirement_key"], f"{label}.requirement_key")
    _non_blank(payload["cache_id"], f"{label}.cache_id")
    namespace_fingerprints = payload.get("cache_namespace_fingerprints", [])
    if namespace_fingerprints:
        require(
            isinstance(namespace_fingerprints, list),
            f"{label}.cache_namespace_fingerprints must be an array",
        )
        require(
            len(set(namespace_fingerprints)) == len(namespace_fingerprints),
            f"{label}.cache_namespace_fingerprints must be unique",
        )
        for index, fingerprint in enumerate(namespace_fingerprints):
            _sha256(fingerprint, f"{label}.cache_namespace_fingerprints[{index}]")
    else:
        require(
            namespace_fingerprints == [],
            f"{label}.cache_namespace_fingerprints must be an array",
        )
    require(
        payload["format"] == "dag-ml-json-prediction-blocks-v1",
        f"{label}.format is unsupported",
    )
    require(
        payload["partition"] == "validation", f"{label}.partition must be validation"
    )
    level = payload.get("prediction_level", "sample")
    require(
        level in {"sample", "target", "group"},
        f"{label}.prediction_level is invalid",
    )
    blocks = payload.get("blocks", [])
    aggregated = payload.get("aggregated_blocks", [])
    require(isinstance(blocks, list), f"{label}.blocks must be an array")
    require(isinstance(aggregated, list), f"{label}.aggregated_blocks must be an array")
    if level == "sample":
        require(bool(blocks) and not aggregated, f"{label} sample block kind mismatch")
        validated = [
            _validate_portable_sample_block(block, f"{label}.blocks[{index}]")
            for index, block in enumerate(blocks)
        ]
    else:
        require(
            not blocks and bool(aggregated), f"{label} aggregated block kind mismatch"
        )
        validated = [
            _validate_portable_aggregated_block(
                block, f"{label}.aggregated_blocks[{index}]"
            )
            for index, block in enumerate(aggregated)
        ]
        require(
            all(block["canonical"]["level"] == level for block in validated),
            f"{label} aggregated block level mismatch",
        )
    _cache_positive_int(payload["block_count"], f"{label}.block_count")
    require(
        payload["block_count"] == len(validated),
        f"{label}.block_count does not match blocks",
    )
    if namespace_fingerprints:
        require(
            len(namespace_fingerprints) == len(validated),
            f"{label}.cache_namespace_fingerprints does not match blocks",
        )
    _cache_positive_int(payload["row_count"], f"{label}.row_count")
    require(
        payload["row_count"]
        == sum(block["record"]["row_count"] for block in validated),
        f"{label}.row_count does not match blocks",
    )
    _sha256(payload["content_fingerprint"], f"{label}.content_fingerprint")
    canonical_blocks = [block["canonical"] for block in validated]
    require(
        payload["content_fingerprint"] == _serde_sha256(canonical_blocks),
        f"{label}.content_fingerprint does not match blocks",
    )
    normalized = dict(payload)
    normalized["prediction_level"] = level
    normalized["cache_namespace_fingerprints"] = namespace_fingerprints
    normalized["derived_records"] = [block["record"] for block in validated]
    return normalized


def _validate_bundle_cache_block_record(
    value: Any, parent_level: str, label: str
) -> dict[str, Any]:
    block = _cache_exact_keys(
        value,
        {"prediction_id", "fold_id", "row_count", "sample_ids", "content_fingerprint"},
        {"prediction_level", "unit_ids"},
        label,
    )
    if block["prediction_id"] is not None:
        _non_blank(block["prediction_id"], f"{label}.prediction_id")
    if block["fold_id"] is not None:
        _identifier(block["fold_id"], f"{label}.fold_id")
    level = block.get("prediction_level", "sample")
    require(level == parent_level, f"{label}.prediction_level does not match cache")
    row_count = _cache_positive_int(block["row_count"], f"{label}.row_count")
    samples = _cache_unique_identifiers(block["sample_ids"], f"{label}.sample_ids")
    units = _cache_unit_ids(block.get("unit_ids", []), level, f"{label}.unit_ids")
    if level == "sample":
        require(not units, f"{label}.unit_ids must be empty for sample predictions")
        require(bool(samples), f"{label}.sample_ids must be non-empty")
        require(row_count == len(samples), f"{label}.row_count does not match samples")
    else:
        require(not samples, f"{label}.sample_ids must be empty for {level}")
        require(bool(units), f"{label}.unit_ids must be non-empty for {level}")
        require(row_count == len(units), f"{label}.row_count does not match units")
    _sha256(block["content_fingerprint"], f"{label}.content_fingerprint")
    return {
        "prediction_id": block["prediction_id"],
        "fold_id": block["fold_id"],
        "prediction_level": level,
        "row_count": row_count,
        "unit_ids": units,
        "sample_ids": samples,
        "content_fingerprint": block["content_fingerprint"],
    }


def _validate_bundle_cache_requirement(value: Any, label: str) -> dict[str, Any]:
    requirement = _cache_exact_keys(
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
    _identifier(requirement["producer_node"], f"{label}.producer_node")
    _identifier(requirement["consumer_node"], f"{label}.consumer_node")
    _non_blank(requirement["source_port"], f"{label}.source_port")
    _non_blank(requirement["target_port"], f"{label}.target_port")
    require(
        requirement["partition"] == "validation",
        f"{label}.partition must be validation",
    )
    level = requirement.get("prediction_level", "sample")
    require(
        level in {"sample", "target", "group"}, f"{label}.prediction_level is invalid"
    )
    fold_ids = _cache_unique_identifiers(requirement["fold_ids"], f"{label}.fold_ids")
    samples = _cache_unique_identifiers(
        requirement["sample_ids"],
        f"{label}.sample_ids",
        non_empty=level == "sample",
    )
    units = _cache_unit_ids(requirement.get("unit_ids", []), level, f"{label}.unit_ids")
    if level == "sample":
        expected_units = [{"level": "sample", "id": sample_id} for sample_id in samples]
        require(
            not units or units == expected_units,
            f"{label}.unit_ids do not match sample_ids",
        )
    else:
        require(not samples, f"{label}.sample_ids must be empty for {level}")
        require(bool(units), f"{label}.unit_ids must be non-empty for {level}")
    width = _cache_positive_int(
        requirement["prediction_width"], f"{label}.prediction_width"
    )
    targets = _cache_target_names(
        requirement["target_names"], width, f"{label}.target_names"
    )
    require(bool(targets), f"{label}.target_names must be non-empty")
    normalized = dict(requirement)
    normalized["prediction_level"] = level
    normalized["fold_ids"] = fold_ids
    normalized["sample_ids"] = samples
    normalized["unit_ids"] = units
    normalized["target_names"] = targets
    normalized["requirement_key"] = (
        f"{requirement['producer_node']}.{requirement['source_port']}->"
        f"{requirement['consumer_node']}.{requirement['target_port']}"
    )
    return normalized


def _validate_bundle_cache_record(value: Any, label: str) -> dict[str, Any]:
    record = _cache_exact_keys(
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
        {"prediction_level", "unit_ids", "cache_namespace_fingerprints"},
        label,
    )
    _non_blank(record["requirement_key"], f"{label}.requirement_key")
    _non_blank(record["cache_id"], f"{label}.cache_id")
    namespace_fingerprints = record.get("cache_namespace_fingerprints", [])
    if namespace_fingerprints:
        require(
            isinstance(namespace_fingerprints, list),
            f"{label}.cache_namespace_fingerprints must be an array",
        )
        require(
            len(set(namespace_fingerprints)) == len(namespace_fingerprints),
            f"{label}.cache_namespace_fingerprints must be unique",
        )
        for index, fingerprint in enumerate(namespace_fingerprints):
            _sha256(fingerprint, f"{label}.cache_namespace_fingerprints[{index}]")
    else:
        require(
            namespace_fingerprints == [],
            f"{label}.cache_namespace_fingerprints must be an array",
        )
    require(
        record["format"] == "dag-ml-json-prediction-blocks-v1",
        f"{label}.format is unsupported",
    )
    require(
        record["partition"] == "validation", f"{label}.partition must be validation"
    )
    level = record.get("prediction_level", "sample")
    require(
        level in {"sample", "target", "group"}, f"{label}.prediction_level is invalid"
    )
    fold_ids = _cache_unique_identifiers(record["fold_ids"], f"{label}.fold_ids")
    samples = _cache_unique_identifiers(record["sample_ids"], f"{label}.sample_ids")
    units = _cache_unit_ids(record.get("unit_ids", []), level, f"{label}.unit_ids")
    row_count = _cache_positive_int(record["row_count"], f"{label}.row_count")
    if level == "sample":
        expected_units = [{"level": "sample", "id": sample_id} for sample_id in samples]
        require(
            not units or units == expected_units,
            f"{label}.unit_ids do not match sample_ids",
        )
        require(bool(samples), f"{label}.sample_ids must be non-empty")
        require(row_count == len(samples), f"{label}.row_count does not match samples")
    else:
        require(not samples, f"{label}.sample_ids must be empty for {level}")
        require(bool(units), f"{label}.unit_ids must be non-empty for {level}")
        require(row_count == len(units), f"{label}.row_count does not match units")
    width = _cache_positive_int(record["prediction_width"], f"{label}.prediction_width")
    targets = _cache_target_names(
        record["target_names"], width, f"{label}.target_names"
    )
    require(bool(targets), f"{label}.target_names must be non-empty")
    blocks = record["blocks"]
    require(
        isinstance(blocks, list) and bool(blocks), f"{label}.blocks must be non-empty"
    )
    validated_blocks = [
        _validate_bundle_cache_block_record(block, level, f"{label}.blocks[{index}]")
        for index, block in enumerate(blocks)
    ]
    block_count = _cache_positive_int(record["block_count"], f"{label}.block_count")
    require(
        block_count == len(validated_blocks),
        f"{label}.block_count does not match blocks",
    )
    if namespace_fingerprints:
        require(
            len(namespace_fingerprints) == len(validated_blocks),
            f"{label}.cache_namespace_fingerprints does not match blocks",
        )
    require(
        row_count == sum(block["row_count"] for block in validated_blocks),
        f"{label}.row_count does not match block records",
    )
    if level == "sample":
        covered = [
            sample for block in validated_blocks for sample in block["sample_ids"]
        ]
        require(len(covered) == len(set(covered)), f"{label}.blocks duplicate samples")
        require(set(covered) == set(samples), f"{label}.blocks do not cover sample_ids")
    else:
        covered_units = [
            (unit["level"], unit["id"])
            for block in validated_blocks
            for unit in block["unit_ids"]
        ]
        require(
            len(covered_units) == len(set(covered_units)),
            f"{label}.blocks duplicate units",
        )
        require(
            set(covered_units) == {(unit["level"], unit["id"]) for unit in units},
            f"{label}.blocks do not cover unit_ids",
        )
    _sha256(record["content_fingerprint"], f"{label}.content_fingerprint")
    normalized = dict(record)
    normalized["prediction_level"] = level
    normalized["fold_ids"] = fold_ids
    normalized["unit_ids"] = units
    normalized["sample_ids"] = samples
    normalized["target_names"] = targets
    normalized["cache_namespace_fingerprints"] = namespace_fingerprints
    normalized["validated_blocks"] = validated_blocks
    return normalized


def _derive_replayable_phases(
    plan: dict[str, Any],
    closure: list[str],
    refit_completed: bool,
    bundle: dict[str, Any],
    portable_caches: dict[str, Any] | None,
) -> list[str]:
    """Derive the honest replayable phases (mirror of the Rust derivation).

    Retained inference state is required only for a ``stateful`` or
    ``emits_artifacts`` node (``artifact_policy``/``ReplayRequired`` and
    ``fit_scope`` are never consulted). Support/capabilities are read from the
    manifests, not the node-plan copies. Canonical order is [REFIT, PREDICT,
    EXPLAIN]; a completed refit exposes forward inference only, a skipped refit
    exposes exactly REFIT when the closure OOF edges are self-contained. ``[]`` is
    a valid answer.
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
        _oof_requirement_key(
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
    payload_keys = (
        {payload["requirement_key"] for payload in portable_caches.get("caches", [])}
        if portable_caches is not None
        else set()
    )
    closure_set = set(closure)
    oof_self_contained = True
    for edge in graph.get("edges", []):
        if edge["contract"].get("requires_oof") is not True:
            continue
        source = edge["source"]["node_id"]
        target = edge["target"]["node_id"]
        if source not in closure_set or target not in closure_set:
            continue
        key = _oof_requirement_key(
            source, edge["source"]["port_name"], target, edge["target"]["port_name"]
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


def _selector_matches(selector: dict[str, Any], operator: Any) -> bool:
    if not isinstance(operator, (str, dict)):
        return False
    if isinstance(operator, str):
        descriptor = {"aliases": [operator]}
    else:
        descriptor = {
            "aliases": [
                value
                for key in ("type", "ref", "class", "function")
                if isinstance((value := operator.get(key)), str)
            ],
            "types": [operator.get("type")],
            "refs": [operator.get("ref")],
            "classes": [operator.get("class")],
            "functions": [operator.get("function")],
        }
    for field in ("aliases", "types", "refs", "classes", "functions"):
        expected = {str(item).strip().lower() for item in selector.get(field, [])}
        actual = {
            str(item).strip().lower()
            for item in descriptor.get(field, [])
            if isinstance(item, str)
        }
        if expected & actual:
            return True
    operator_class = operator.get("class") if isinstance(operator, dict) else None
    if isinstance(operator_class, str):
        lowered = operator_class.strip().lower()
        if any(
            lowered.startswith(str(prefix).strip().lower())
            for prefix in selector.get("class_prefixes", [])
        ):
            return True
    return False


def _controller_for_node(
    request: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any]:
    manifests = request["controller_manifests"]
    explicit_id = node.get("metadata", {}).get("controller_id")
    if explicit_id is not None:
        matches = [m for m in manifests if m["controller_id"] == explicit_id]
        require(len(matches) == 1, f"node {node['id']} explicit controller is absent")
        require(
            matches[0]["operator_kind"] == node["kind"],
            f"node {node['id']} explicit controller has wrong kind",
        )
        return matches[0]
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for manifest in manifests:
        if manifest["operator_kind"] != node["kind"]:
            continue
        selectors = manifest.get("operator_selectors", [])
        if selectors:
            if not any(
                _selector_matches(selector, node.get("operator"))
                for selector in selectors
            ):
                continue
            rank = 0
        else:
            rank = 1
        candidates.append((rank, manifest["priority"], manifest))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]["controller_id"]))
    require(bool(candidates), f"no controller for node {node['id']}")
    if len(candidates) > 1:
        require(
            candidates[0][:2] != candidates[1][:2],
            f"ambiguous controllers for node {node['id']}",
        )
    return candidates[0][2]


def _validate_controller_manifest_semantics(
    manifest: dict[str, Any], label: str
) -> None:
    """Mirror the active-capability rules and BTreeSet wire normalization.

    TrainingRequest's strict parser fingerprints both the original JSON and the
    Rust re-serialization.  Consequently BTreeSet-backed arrays must already be
    in enum/string order (and defaulted non-skipped fields must be present), or
    the two fingerprints intentionally differ.
    """

    _identifier(manifest["controller_id"], f"{label}.controller_id")
    _non_blank(manifest["controller_version"], f"{label}.controller_version")
    for emitted_default in (
        "priority",
        "input_ports",
        "output_ports",
        "data_requirements",
        "capabilities",
    ):
        require(
            emitted_default in manifest,
            f"{label}.{emitted_default} must be explicit in canonical TrainingRequest wire JSON",
        )
    phases = manifest["supported_phases"]
    require(
        phases == sorted(set(phases), key=PHASE_ORDER.__getitem__) and bool(phases),
        f"{label}.supported_phases must be in canonical BTreeSet wire order",
    )
    capabilities = manifest["capabilities"]
    require(
        capabilities == sorted(set(capabilities), key=CAPABILITY_ORDER.__getitem__),
        f"{label}.capabilities must be in canonical BTreeSet wire order",
    )
    for selector_index, selector in enumerate(manifest.get("operator_selectors", [])):
        require(
            bool(selector), f"{label}.operator_selectors[{selector_index}] is empty"
        )
        for field, values in selector.items():
            require(
                values == sorted(set(values)) and bool(values),
                f"{label}.operator_selectors[{selector_index}].{field} must be canonical",
            )
            for value in values:
                _non_blank(
                    value, f"{label}.operator_selectors[{selector_index}].{field}"
                )
    capability_set = set(capabilities)
    training_capabilities = {
        "uses_training_weights",
        "uses_early_stopping",
        "performs_internal_tuning",
        "trains_aggregation",
    }
    require(
        not (
            manifest["fit_scope"] in {"stateless", "inference_only"}
            and capability_set & training_capabilities
        ),
        f"{label} has active training-influence capabilities on an inactive fit scope",
    )
    if "uses_training_weights" in capability_set:
        require(
            bool(
                capability_set
                & {
                    "supports_sample_weights",
                    "supports_row_resampling",
                    "supports_backend_loss_weights",
                }
            ),
            f"{label} uses training weights without a supported weighting mechanism",
        )
    if "trains_aggregation" in capability_set:
        require(
            "aggregates_predictions" in capability_set,
            f"{label} trains aggregation without aggregates_predictions",
        )
    require(
        not (
            manifest["rng_policy"] == "nondeterministic"
            and "deterministic" in capability_set
        ),
        f"{label} cannot be deterministic with nondeterministic RNG",
    )
    require(
        not (
            manifest["fit_scope"] == "inference_only"
            and ({"FIT_CV", "REFIT"} & set(phases))
        ),
        f"{label} inference_only controller supports training phases",
    )
    require(
        not (
            "FIT_CV" in phases
            and manifest["fit_scope"] in {"full_train", "inference_only"}
        ),
        f"{label} FIT_CV controller has an incompatible fit_scope",
    )
    if any(port["kind"] == "prediction" for port in manifest["output_ports"]):
        require(
            "emits_predictions" in capability_set,
            f"{label} has prediction ports without emits_predictions",
        )
    if any(port["kind"] == "artifact" for port in manifest["output_ports"]):
        require(
            "emits_artifacts" in capability_set,
            f"{label} has artifact ports without emits_artifacts",
        )


def _validate_output_shape(output: dict[str, Any], label: str) -> str:
    require("unit_level" in output, f"{label}.unit_level must be explicit")
    if output["prediction_level"] == "sample":
        require(
            output.get("unit_level") == "physical_sample",
            f"{label}.unit_level must be physical_sample for sample predictions",
        )
    elif output["prediction_level"] in {"target", "group"}:
        require(
            output.get("unit_level") is None,
            f"{label}.unit_level must be null for target/group predictions",
        )
    targets = output["target_names"]
    require(
        isinstance(targets, list)
        and bool(targets)
        and len(targets) == len(set(targets)),
        f"{label}.target_names must preserve a unique semantic order",
    )
    for target in targets:
        _non_blank(target, f"{label}.target_names entry")
    require(
        len(output["target_units"]) == len(targets)
        and len(output["class_labels"]) == len(targets),
        f"{label} target metadata alignment",
    )
    for unit in output["target_units"]:
        if unit is not None:
            _non_blank(unit, f"{label}.target_units entry")
    for labels in output["class_labels"]:
        require(
            isinstance(labels, list) and len(labels) == len(set(labels)),
            f"{label}.class_labels do not match prediction kind",
        )
        for class_label in labels:
            _non_blank(class_label, f"{label}.class_labels entry")
    if output["prediction_kind"] == "class_probability":
        require(
            all(bool(labels) for labels in output["class_labels"]),
            f"{label}.class labels must be non-empty for class_probability",
        )
    elif output["prediction_kind"] == "regression_point":
        require(
            all(not labels for labels in output["class_labels"]),
            f"{label}.regression_point class vocabularies must be empty",
        )
    require(
        (
            output["prediction_kind"] == "class_probability"
            and output["output_order"] == "target_major_class_minor"
        )
        or (
            output["prediction_kind"] != "class_probability"
            and output["output_order"] == "target_order"
        ),
        f"{label}.output_order is incompatible with prediction_kind",
    )
    _non_blank(output["target_space"], f"{label}.target_space")
    ports = _graph_prediction_ports(output["_graph"], output["node_id"])
    port_name = output.get("port_name")
    if port_name is None:
        require(len(ports) == 1, f"{label} output port is ambiguous or absent")
        return ports[0]
    _non_blank(port_name, f"{label}.port_name")
    require(port_name in ports, f"{label}.port_name is not a prediction output")
    return port_name


def _campaign_bindings(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = [
        binding
        for values in campaign.get("data_bindings", {}).values()
        for binding in values
    ]
    result = {
        f"{binding['node_id']}.{binding['input_name']}": binding for binding in bindings
    }
    require(len(result) == len(bindings), "campaign data binding keys must be unique")
    return result


def _validate_data_identities_for_campaign(
    identities: list[dict[str, Any]], campaign: dict[str, Any], label: str
) -> None:
    keys = [identity["requirement_key"] for identity in identities]
    require(keys == sorted(set(keys)), f"{label} order/uniqueness")
    bindings = _campaign_bindings(campaign)
    require(keys == sorted(bindings), f"{label} must exactly cover campaign bindings")
    for identity in identities:
        binding = bindings[identity["requirement_key"]]
        require(
            identity["schema_fingerprint"] == binding["schema_fingerprint"]
            and identity["plan_fingerprint"] == binding["plan_fingerprint"]
            and identity["relation_fingerprint"] == binding.get("relation_fingerprint"),
            f"{label} {identity['requirement_key']} does not match data binding fingerprints",
        )


def _validate_patches(request: dict[str, Any], label: str) -> None:
    patches = request["parameter_patches"]
    policies = request["patch_policies"]
    nodes = _graph_nodes(request["graph"])
    patch_keys: list[tuple[str, int, tuple[str, ...]]] = []
    for index, patch in enumerate(patches):
        path = patch["path"]
        require(
            bool(path)
            and all(
                isinstance(part, str) and bool(part.strip()) and part != "-"
                for part in path
            ),
            f"{label}.parameter_patches[{index}].path",
        )
        require(patch["node_id"] in nodes, f"{label} patch references unknown node")
        patch_keys.append(
            (patch["node_id"], NAMESPACE_ORDER[patch["namespace"]], tuple(path))
        )
    require(patch_keys == sorted(patch_keys), f"{label}.parameter patch order")
    require(
        len(patch_keys) == len(set(patch_keys)), f"{label}.duplicate parameter patch"
    )
    for left, right in zip(patch_keys, patch_keys[1:]):
        if left[:2] == right[:2]:
            shorter, longer = sorted((left[2], right[2]), key=len)
            require(
                longer[: len(shorter)] != shorter,
                f"{label}.parameter patches contain a parent/child conflict",
            )

    policy_ids = [policy["node_id"] for policy in policies]
    require(policy_ids == sorted(set(policy_ids)), f"{label}.patch policy order")
    target_ids = sorted({patch["node_id"] for patch in patches})
    require(
        policy_ids == target_ids,
        f"{label}.patch policies must exactly cover patched nodes",
    )
    policy_by_node: dict[str, set[str]] = {}
    for policy in policies:
        namespaces = policy["allowed_namespaces"]
        require(bool(namespaces), f"{label}.patch policy allows no namespace")
        require(
            namespaces == sorted(set(namespaces), key=NAMESPACE_ORDER.__getitem__),
            f"{label}.patch policy namespaces must be in canonical BTreeSet wire order and unique",
        )
        policy_by_node[policy["node_id"]] = set(namespaces)

    roots = {
        node_id: {
            "operator": copy.deepcopy(node.get("params", {})),
            "fit": {},
            "control": {},
            "structural": {},
        }
        for node_id, node in nodes.items()
    }
    for patch in patches:
        require(
            patch["namespace"] in policy_by_node[patch["node_id"]],
            f"{label}.parameter namespace is forbidden by patch policy",
        )
        cursor = roots[patch["node_id"]][patch["namespace"]]
        path = patch["path"]
        for segment in path[:-1]:
            require(
                segment in cursor and isinstance(cursor[segment], dict),
                f"{label}.parameter patch has missing/non-object intermediate path",
            )
            cursor = cursor[segment]
        cursor[path[-1]] = copy.deepcopy(patch["value"])


def _folds(request: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fold_set = request["campaign"].get("split_invocation", {}).get("fold_set")
    require(
        isinstance(fold_set, dict), "training influence requires an explicit fold_set"
    )
    samples = fold_set["sample_ids"]
    require(
        isinstance(samples, list) and bool(samples),
        "fold_set.sample_ids must be a non-empty array",
    )
    for sample in samples:
        _identifier(sample, "fold_set.sample_ids entry")
    require(
        len(samples) == len(set(samples)),
        "fold_set.sample_ids must be unique (wire order is semantic-neutral)",
    )
    folds = fold_set["folds"]
    require(isinstance(folds, list) and bool(folds), "fold_set.folds must be non-empty")
    fold_ids: set[str] = set()
    universe = set(samples)
    for fold in folds:
        _identifier(fold["fold_id"], "fold_set.fold_id")
        require(fold["fold_id"] not in fold_ids, "fold_set fold ids must be unique")
        fold_ids.add(fold["fold_id"])
        train = fold["train_sample_ids"]
        validation = fold["validation_sample_ids"]
        require(
            isinstance(train, list)
            and len(train) == len(set(train))
            and isinstance(validation, list)
            and bool(validation)
            and len(validation) == len(set(validation)),
            f"fold {fold['fold_id']} samples must be unique and validation non-empty",
        )
        require(
            set(train) | set(validation) <= universe,
            f"fold {fold['fold_id']} references samples outside the universe",
        )
        require(
            not (set(train) & set(validation)),
            f"fold {fold['fold_id']} leaks train/validation samples",
        )
    return samples, folds


def _base_influence_kind(
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
    node = _graph_nodes(request["graph"])[node_id]
    if node["kind"] == "model":
        return "model_fit"
    if node["kind"] == "tuner":
        return "hpo_selection"
    return "transform_fit"


def _active_influence_slots(
    request: dict[str, Any], closure: list[str]
) -> dict[tuple[str, str, str, str | None], set[str]]:
    nodes = _graph_nodes(request["graph"])
    all_samples, folds = _folds(request)
    slots: dict[tuple[str, str, str, str | None], set[str]] = {}
    for node_id in closure:
        manifest = _controller_for_node(request, nodes[node_id])
        base_kind = _base_influence_kind(request, node_id, manifest)
        if base_kind is None:
            continue
        kinds = {
            kind
            for capability, kind in ACTIVE_CAPABILITIES.items()
            if capability in manifest["capabilities"]
        }
        # A tuner already declares its base HPO-selection influence; the explicit
        # capability slot is only additional for non-tuner controllers.
        if base_kind == "hpo_selection":
            kinds.discard("hpo_selection")
        phases = set(manifest["supported_phases"])
        if "FIT_CV" in phases:
            if manifest["fit_scope"] == "fold_train":
                for fold in folds:
                    for kind in kinds:
                        slots[(node_id, kind, "FIT_CV", fold["fold_id"])] = set(
                            fold["train_sample_ids"]
                        )
            elif manifest["fit_scope"] == "full_train":
                for kind in kinds:
                    slots[(node_id, kind, "FIT_CV", None)] = set(all_samples)
        if request["options"]["refit"] and "REFIT" in phases:
            for kind in kinds:
                slots[(node_id, kind, "REFIT", None)] = set(all_samples)
    return slots


def _validate_influence_requirements(
    request: dict[str, Any], closure: list[str], label: str
) -> None:
    _, folds = _folds(request)
    fold_by_id = {fold["fold_id"]: fold for fold in folds}
    expected = _active_influence_slots(request, closure)
    actual: set[tuple[str, str, str, str | None]] = set()
    previous: tuple[int, str, str] | None = None
    for index, requirement in enumerate(request["influence_requirements"]):
        item_label = f"{label}.influence_requirements[{index}]"
        _identifier(requirement["scope_id"], f"{item_label}.scope_id")
        samples = _sorted_unique(
            requirement["physical_sample_ids"],
            f"{item_label}.physical_sample_ids",
            non_empty=True,
        )
        order_key = (
            INFLUENCE_ORDER[requirement["kind"]],
            requirement["scope_id"],
            requirement["node_id"],
        )
        require(
            previous is None or previous < order_key,
            f"{label}.influence requirements must be strictly canonically sorted",
        )
        previous = order_key
        require(
            requirement["node_id"] in closure,
            f"{item_label}.node_id is outside predictor closure",
        )
        require(
            (requirement["phase"] == "FIT_CV" and requirement["fold_id"] is not None)
            or (requirement["phase"] == "REFIT" and requirement["fold_id"] is None),
            f"{item_label}.fold_id does not match phase",
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
        sample_set = set(samples)
        eligible = expected[slot]
        if not sample_set <= eligible:
            fold = fold_by_id.get(requirement["fold_id"])
            if fold is not None and sample_set & set(fold["validation_sample_ids"]):
                raise ContractError(f"{item_label} leaks outer validation samples")
            raise ContractError(
                f"{item_label} uses samples outside its training cohort"
            )
        if requirement["kind"] == "weighting_resampling":
            require(
                sample_set == eligible,
                f"{item_label} must cover its complete fit cohort",
            )
        if requirement["kind"] == "early_stopping":
            require(
                len(sample_set) < len(eligible),
                f"{item_label} must be a strict training-cohort subset",
            )
        require(slot not in actual, f"{item_label} duplicates a capability slot")
        actual.add(slot)
    require(
        actual == set(expected),
        f"{label}.influence requirements do not exactly cover active capability scopes",
    )


def validate_training_request(
    value: Any, label: str = "TrainingRequest"
) -> dict[str, Any]:
    request = _exact_keys(
        value,
        {
            "schema_version",
            "request_id",
            "plan_id",
            "graph",
            "campaign",
            "controller_manifests",
            "data_identities",
            "parameter_patches",
            "patch_policies",
            "influence_requirements",
            "options",
            "request_fingerprint",
        },
        label,
    )
    require(request["schema_version"] == 1, f"{label}.schema_version")
    _identifier(request["request_id"], f"{label}.request_id")
    _non_blank(request["plan_id"], f"{label}.plan_id")
    _sha256(request["request_fingerprint"], f"{label}.request_fingerprint")
    require(
        request["request_fingerprint"]
        == fingerprint_without(request, "request_fingerprint"),
        f"{label}.request_fingerprint mismatch",
    )
    _normalize_training_request(request)  # fail closed on serde container types
    _validate_search_space_fingerprint(request["graph"], request["campaign"], label)
    controller_ids = [
        manifest["controller_id"] for manifest in request["controller_manifests"]
    ]
    require(
        controller_ids == sorted(set(controller_ids)),
        f"{label}.controller_manifests order/uniqueness",
    )
    for index, manifest in enumerate(request["controller_manifests"]):
        _validate_controller_manifest_deserialize_shape(
            manifest, f"{label}.controller_manifests[{index}]"
        )
        _validate_controller_manifest_semantics(
            manifest, f"{label}.controller_manifests[{index}]"
        )
    identities = [
        validate_data_identity(identity, f"{label}.data_identities[{index}]")
        for index, identity in enumerate(request["data_identities"])
    ]
    _validate_data_identities_for_campaign(
        identities, request["campaign"], f"{label}.data_identities"
    )
    options = _exact_keys(
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
        options["seed"] == request["campaign"]["root_seed"],
        f"{label}.options.seed mismatch",
    )
    require(
        (
            options["refit"]
            and options["refit_strategy"] in {"refit_one", "refit_ensemble"}
        )
        or (not options["refit"] and options["refit_strategy"] is None),
        f"{label}.refit/refit_strategy mismatch",
    )
    if not options["refit"]:
        require(
            options["artifacts"]["prediction_caches"] == "retain",
            f"{label}.no-refit must retain caches",
        )
    scheduler = options["scheduler"]
    require(
        (
            scheduler["kind"] == "sequential"
            and scheduler.get("backend") is None
            and scheduler["workers"] == 1
        )
        or (
            scheduler["kind"] == "parallel"
            and scheduler.get("backend") in {"threads", "processes"}
            and scheduler["workers"] >= 2
        ),
        f"{label}.scheduler kind/backend/workers mismatch",
    )
    require(
        scheduler["workers"] <= options["resources"]["cpu_threads"],
        f"{label}.scheduler workers exceeds resources.cpu_threads",
    )
    _sorted_unique(
        options["resources"]["gpu_devices"],
        f"{label}.resources.gpu_devices",
        identifiers=False,
    )
    output_ids = [output["output_id"] for output in options["outputs"]]
    require(
        output_ids == sorted(set(output_ids)),
        f"{label}.outputs must be strictly sorted by output_id and unique",
    )
    coordinates: set[tuple[str, str]] = set()
    output_nodes: list[str] = []
    for index, source_output in enumerate(options["outputs"]):
        output = dict(source_output)
        output["_graph"] = request["graph"]
        output_label = f"{label}.options.outputs[{index}]"
        port = _validate_output_shape(output, output_label)
        coordinate = (output["node_id"], port)
        require(
            coordinate not in coordinates,
            f"{output_label} duplicates output coordinates",
        )
        coordinates.add(coordinate)
        output_nodes.append(output["node_id"])
    _identifier(options["selection_output_id"], f"{label}.options.selection_output_id")
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
    closure = _graph_closure(request["graph"], output_nodes)
    nodes = _graph_nodes(request["graph"])
    controllers = {
        node_id: _controller_for_node(request, nodes[node_id]) for node_id in closure
    }
    require(
        "FIT_CV" in controllers[selection_output["node_id"]]["supported_phases"],
        f"{label}.selection output is not scorable in FIT_CV",
    )
    prediction_ports = [
        port
        for port in nodes[selection_output["node_id"]]["ports"]["outputs"]
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
    for node_id in output_nodes:
        require(
            "emits_predictions" in controllers[node_id]["capabilities"],
            f"{label}.output controller {node_id} does not emit predictions",
        )
    if scheduler["kind"] == "parallel":
        for node_id, manifest in controllers.items():
            capabilities = set(manifest["capabilities"])
            if scheduler.get("backend") == "threads":
                require(
                    "thread_safe" in capabilities,
                    f"{label}.{node_id} is not thread_safe",
                )
                require(
                    "needs_python_gil" not in capabilities,
                    f"{label}.{node_id} needs Python GIL",
                )
            else:
                require(
                    "process_safe" in capabilities,
                    f"{label}.{node_id} is not process_safe",
                )
    if options["artifacts"]["fitted_artifacts"] == "portable_required":
        for node_id, manifest in controllers.items():
            require(
                not (
                    "emits_artifacts" in manifest["capabilities"]
                    and manifest["artifact_policy"] == "host_only"
                ),
                f"{label}.{node_id} host_only artifact is not portable",
            )
    _validate_patches(request, label)
    _validate_influence_requirements(request, closure, label)
    require(
        tcv1_sha256(request) == tcv1_sha256(_normalize_training_request(request)),
        f"{label} wire content does not match its typed Rust serde representation",
    )
    return request


def validate_cache_namespace(
    value: Any,
    identity: dict[str, Any] | None = None,
    label: str = "CacheNamespace",
) -> dict[str, Any]:
    namespace = _exact_keys(
        value,
        {
            "schema_version",
            "prediction_requirement_key",
            "data_requirement_key",
            "producer_node_id",
            "source_port_name",
            "consumer_node_id",
            "target_port_name",
            "phase",
            "params_fingerprint",
            "data_identity_fingerprint",
            "fold_id",
            "trial_id",
            "seed",
            "namespace_fingerprint",
        },
        label,
    )
    require(namespace["schema_version"] == 1, f"{label}.schema_version")
    require(namespace["phase"] == "FIT_CV", f"{label} is FIT_CV-only")
    for field in (
        "producer_node_id",
        "consumer_node_id",
        "fold_id",
        "trial_id",
    ):
        _identifier(namespace[field], f"{label}.{field}")
    for field in ("source_port_name", "target_port_name", "data_requirement_key"):
        _non_blank(namespace[field], f"{label}.{field}")
    expected_key = (
        f"{namespace['producer_node_id']}.{namespace['source_port_name']}->"
        f"{namespace['consumer_node_id']}.{namespace['target_port_name']}"
    )
    require(
        namespace["prediction_requirement_key"] == expected_key,
        f"{label}.prediction_requirement_key mismatch",
    )
    for field in (
        "params_fingerprint",
        "data_identity_fingerprint",
        "namespace_fingerprint",
    ):
        _sha256(namespace[field], f"{label}.{field}")
    require(
        namespace["namespace_fingerprint"]
        == fingerprint_without(namespace, "namespace_fingerprint"),
        f"{label}.namespace_fingerprint mismatch",
    )
    if identity is not None:
        validate_data_identity(identity, f"{label}.identity")
        require(
            namespace["data_requirement_key"] == identity["requirement_key"]
            and namespace["data_identity_fingerprint"]
            == identity["identity_fingerprint"],
            f"{label} does not bind complete data identity",
        )
    return namespace


def validate_parameter_projection(
    value: Any, label: str = "ParameterProjection"
) -> dict[str, Any]:
    projection = _exact_keys(
        value,
        {
            "schema_version",
            "nodes",
            "requires_recompile",
            "structural_patch_count",
            "patches_fingerprint",
            "projection_fingerprint",
        },
        label,
    )
    require(projection["schema_version"] == 1, f"{label}.schema_version")
    require(
        projection["requires_recompile"] == (projection["structural_patch_count"] > 0),
        f"{label}.requires_recompile mismatch",
    )
    for node_id, roots in projection["nodes"].items():
        _identifier(node_id, f"{label}.nodes key")
        _exact_keys(
            roots,
            {"params", "fit_params", "control_params", "structural_params"},
            f"{label}.nodes[{node_id}]",
        )
    _sha256(projection["patches_fingerprint"], f"{label}.patches_fingerprint")
    require(
        projection["projection_fingerprint"]
        == fingerprint_without(projection, "projection_fingerprint"),
        f"{label}.projection_fingerprint mismatch",
    )
    return projection


def _validate_outcome_patch_order(patches: Any, label: str) -> None:
    require(isinstance(patches, list), f"{label}.parameter_patches must be an array")
    keys: list[tuple[str, int, tuple[str, ...]]] = []
    for index, patch in enumerate(patches):
        patch_label = f"{label}.parameter_patches[{index}]"
        _exact_keys(
            patch,
            {"schema_version", "node_id", "namespace", "path", "value"},
            patch_label,
        )
        _identifier(patch["node_id"], f"{patch_label}.node_id")
        require(patch["namespace"] in NAMESPACE_ORDER, f"{patch_label}.namespace")
        require(isinstance(patch["path"], list), f"{patch_label}.path must be an array")
        keys.append(
            (
                patch["node_id"],
                NAMESPACE_ORDER[patch["namespace"]],
                tuple(patch["path"]),
            )
        )
    require(keys == sorted(keys), f"{label}.parameter_patches must be sorted")
    require(len(keys) == len(set(keys)), f"{label}.parameter_patches duplicate")


def _validate_outcome_lineage_order(lineage: Any, label: str) -> None:
    require(
        isinstance(lineage, list) and bool(lineage),
        f"{label}.lineage must be a non-empty array",
    )
    record_ids = [record["record_id"] for record in lineage]
    require(
        record_ids == sorted(record_ids),
        f"{label}.lineage must be sorted by record_id",
    )
    require(
        len(record_ids) == len(set(record_ids)),
        f"{label}.lineage contains duplicate record ids",
    )
    known = set(record_ids)
    for index, record in enumerate(lineage):
        inputs = record["input_lineage"]
        require(
            isinstance(inputs, list),
            f"{label}.lineage[{index}].input_lineage must be an array",
        )
        require(
            inputs == sorted(set(inputs)),
            f"{label}.lineage[{index}].input_lineage must be sorted and unique",
        )
        require(
            set(inputs) <= known,
            f"{label}.lineage[{index}].input_lineage references unknown records",
        )


def validate_training_outcome(
    value: Any, label: str = "TrainingOutcome"
) -> dict[str, Any]:
    outcome = _exact_keys(
        value,
        {
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
        },
        label,
    )
    _normalize_training_outcome(outcome)  # fail closed on serde container types
    require(outcome["schema_version"] == 1, f"{label}.schema_version")
    _identifier(outcome["outcome_id"], f"{label}.outcome_id")
    _identifier(outcome["run_id"], f"{label}.run_id")
    _sha256(
        outcome["training_request_fingerprint"],
        f"{label}.training_request_fingerprint",
    )
    identities = [
        validate_data_identity(identity, f"{label}.data_identities[{index}]")
        for index, identity in enumerate(outcome["data_identities"])
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    require(
        identity_keys == sorted(set(identity_keys)),
        f"{label}.data identities must be sorted by requirement_key and unique",
    )
    _validate_data_identities_for_campaign(
        identities,
        outcome["effective_plan"]["campaign"],
        f"{label}.data_identities",
    )
    require(
        outcome["effective_plan_fingerprint"]
        == tcv1_sha256(_normalize_execution_plan(outcome["effective_plan"])),
        f"{label}.effective_plan_fingerprint mismatch",
    )
    plan = outcome["effective_plan"]
    _validate_outcome_patch_order(outcome["parameter_patches"], label)
    _validate_outcome_lineage_order(outcome["lineage"], label)
    require(
        outcome["execution_bundle"].get("scores") == outcome["score_set"],
        f"{label}.score_set does not exactly match execution_bundle.scores",
    )
    _validate_execution_plan(plan, f"{label}.effective_plan")
    _validate_refit_artifacts_against_plan(
        outcome["execution_bundle"], plan, f"{label}.execution_bundle"
    )
    _validate_bundle_data_requirements_against_plan(
        outcome["execution_bundle"], plan, f"{label}.execution_bundle"
    )
    _validate_bundle_selection_and_prediction_links(
        outcome["execution_bundle"], f"{label}.execution_bundle"
    )
    _validate_portable_caches_against_bundle(
        outcome["portable_prediction_caches"],
        outcome["execution_bundle"],
        f"{label}.portable_prediction_caches",
    )
    # Closure is derived from real graph edges rooted at the output-binding
    # node_ids, never from serialized node-plan copies.
    output_nodes = [output["binding"]["node_id"] for output in outcome["outputs"]]
    closure = _graph_closure(plan["graph_plan"]["graph"], output_nodes)
    require(
        closure == sorted(plan["node_plans"]),
        f"{label}.predictor closure must equal all effective plan nodes in V1",
    )
    influence = _validate_influence_manifest(
        outcome["training_influence"], f"{label}.training_influence"
    )
    _validate_base_influence_against_plan(
        influence, plan, closure, f"{label}.training_influence"
    )
    refit = outcome["refit"]
    expected_replayable = _derive_replayable_phases(
        plan,
        closure,
        refit["status"] == "completed",
        outcome["execution_bundle"],
        outcome["portable_prediction_caches"],
    )
    require(
        outcome["replayable_phases"] == expected_replayable,
        f"{label}.replayable_phases do not match the phases derivable from the "
        "full predictor closure and retained state",
    )
    _sorted_unique(outcome["warnings"], f"{label}.warnings", identifiers=False)
    _identifier(outcome["selection_output_id"], f"{label}.selection_output_id")
    selection_outputs = [
        output
        for output in outcome["outputs"]
        if output["binding"]["binding_id"] == outcome["selection_output_id"]
    ]
    require(
        len(selection_outputs) == 1,
        f"{label}.selection_output_id does not resolve exactly one output",
    )
    binding = selection_outputs[0]["binding"]
    require(
        binding["prediction_level"]
        == outcome["effective_plan"]["campaign"]["aggregation_policy"][
            "selection_metric_level"
        ],
        f"{label}.selection output does not match campaign selection_metric_level",
    )
    selections = outcome["execution_bundle"]["selections"]
    require(
        len(selections) == 1,
        f"{label}.execution bundle must contain exactly one SELECT decision",
    )
    selection_key, decision = next(iter(selections.items()))
    require(
        selection_key == decision["policy_id"]
        and decision["selected_candidate_id"] == outcome["selected_variant_id"]
        and decision.get("metric_level") == binding["prediction_level"]
        and decision.get("evaluation_scope") == "oof"
        and outcome["score_set"].get("selection_metric") == decision["metric_name"],
        f"{label}.SELECT decision metadata mismatch",
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
        in supported_metric.get(binding["prediction_kind"], set()),
        f"{label}.selection metric is incompatible with prediction kind",
    )
    reports: dict[str, dict[str, Any]] = {}
    for report in outcome["score_set"]["reports"]:
        if (
            report["producer_node"] == binding["node_id"]
            and report["partition"] == "validation"
            and report["level"] == binding["prediction_level"]
            and report.get("fold_id") == "avg"
        ):
            variant_id = report.get("variant_id")
            _identifier(variant_id, f"{label}.selection report variant_id")
            require(
                variant_id not in reports,
                f"{label}.multiple average reports for one variant",
            )
            reports[variant_id] = report
    variants = {
        variant["variant_id"] for variant in outcome["effective_plan"]["variants"]
    }
    require(
        set(reports) == variants,
        f"{label}.selection reports do not exactly cover plan variants",
    )
    candidates = [
        (variant_id, report["metrics"][decision["metric_name"]])
        for variant_id, report in reports.items()
    ]
    candidates.sort(
        key=(
            (lambda candidate: (candidate[1], candidate[0]))
            if decision["objective"] == "minimize"
            else (lambda candidate: (-candidate[1], candidate[0]))
        )
    )
    ranking = [
        {"candidate_id": variant_id, "score": score, "rank": index + 1}
        for index, (variant_id, score) in enumerate(candidates)
    ]
    require(
        decision["ranked_candidates"] == ranking
        and decision["selected_score"] == ranking[0]["score"],
        f"{label}.SELECT decision does not equal ranking reconstructed from scores",
    )
    require(
        tcv1_sha256(outcome) == tcv1_sha256(_normalize_training_outcome(outcome)),
        f"{label} wire content does not match its typed Rust serde representation",
    )
    require(
        outcome["outcome_fingerprint"]
        == fingerprint_without(outcome, "outcome_fingerprint"),
        f"{label}.outcome_fingerprint mismatch",
    )
    return outcome


def _contains_runtime_handle(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_runtime_handle(member) for member in value)
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
    return any(_contains_runtime_handle(member) for member in value.values())


def _validate_base_influence_against_plan(
    influence: dict[str, Any], plan: dict[str, Any], closure: list[str], label: str
) -> None:
    nodes = {node["id"]: node for node in plan["graph_plan"]["graph"]["nodes"]}
    oof_consumers = {
        edge["target"]["node_id"]
        for edge in plan["graph_plan"]["graph"].get("edges", [])
        if edge["contract"].get("requires_oof") is True
    }
    expected: dict[str, str] = {}
    for node_id in closure:
        node_plan = plan["node_plans"][node_id]
        if "FIT_CV" not in node_plan["supported_phases"] or node_plan["fit_scope"] in {
            "stateless",
            "inference_only",
        }:
            continue
        if (
            node_id in oof_consumers
            or "trains_aggregation" in node_plan["controller_capabilities"]
        ):
            kind = "trained_meta_aggregation"
        elif nodes[node_id]["kind"] == "model":
            kind = "model_fit"
        elif nodes[node_id]["kind"] == "tuner":
            kind = "hpo_selection"
        else:
            kind = "transform_fit"
        expected[node_id] = kind
    base_kinds = {
        "transform_fit",
        "model_fit",
        "hpo_selection",
        "trained_meta_aggregation",
    }
    actual: dict[str, list[str]] = {}
    for entry in influence["entries"]:
        if entry["node_id"] is not None and entry["kind"] in base_kinds:
            actual.setdefault(entry["node_id"], []).append(entry["kind"])
    require(
        set(actual) == set(expected),
        f"{label} fitting nodes do not exactly match predictor closure",
    )
    for node_id, expected_kind in expected.items():
        require(
            bool(actual[node_id])
            and all(kind == expected_kind for kind in actual[node_id]),
            f"{label} node {node_id} entries do not all have expected kind {expected_kind}",
        )


def _validate_influence_manifest(
    influence: Any,
    label: str,
    relations: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = _exact_keys(
        influence,
        {"schema_version", "relation_fingerprint", "entries", "manifest_fingerprint"},
        label,
    )
    require(manifest["schema_version"] == 1, f"{label}.schema_version")
    _sha256(manifest["relation_fingerprint"], f"{label}.relation_fingerprint")
    require(bool(manifest["entries"]), f"{label}.entries must be non-empty")
    relation_records = None
    if relations is not None:
        relation_records = (
            relations.get("records") if isinstance(relations, dict) else relations
        )
        require(
            isinstance(relation_records, list),
            f"{label}.relations must contain records",
        )
    previous: tuple[int, str, tuple[int, str]] | None = None
    for index, entry in enumerate(manifest["entries"]):
        entry_label = f"{label}.entries[{index}]"
        _exact_keys(
            entry,
            {
                "kind",
                "scope_id",
                "node_id",
                "physical_sample_ids",
                "origin_sample_ids",
                "group_ids",
            },
            entry_label,
        )
        _identifier(entry["scope_id"], f"{entry_label}.scope_id")
        if entry["node_id"] is not None:
            _identifier(entry["node_id"], f"{entry_label}.node_id")
        _sorted_unique(
            entry["physical_sample_ids"],
            f"{entry_label}.physical_sample_ids",
            non_empty=True,
        )
        _sorted_unique(entry["origin_sample_ids"], f"{entry_label}.origin_sample_ids")
        _sorted_unique(entry["group_ids"], f"{entry_label}.group_ids")
        node_key = (0, "") if entry["node_id"] is None else (1, entry["node_id"])
        order_key = (INFLUENCE_ORDER[entry["kind"]], entry["scope_id"], node_key)
        require(
            previous is None or previous < order_key,
            f"{label}.entries are not canonically sorted",
        )
        previous = order_key
        if relation_records is not None:
            by_sample = {record["sample_id"]: record for record in relation_records}
            require(
                set(entry["physical_sample_ids"]) <= set(by_sample),
                f"{entry_label} has samples absent from relations",
            )
            origins = sorted(
                {
                    by_sample[sample].get("origin_sample_id")
                    for sample in entry["physical_sample_ids"]
                    if by_sample[sample].get("origin_sample_id") is not None
                }
            )
            groups = sorted(
                {
                    by_sample[sample].get("group_id")
                    for sample in entry["physical_sample_ids"]
                    if by_sample[sample].get("group_id") is not None
                }
            )
            require(
                entry["origin_sample_ids"] == origins,
                f"{entry_label} origin closure mismatch",
            )
            require(
                entry["group_ids"] == groups, f"{entry_label} group closure mismatch"
            )
    require(
        manifest["manifest_fingerprint"]
        == fingerprint_without(manifest, "manifest_fingerprint"),
        f"{label}.manifest_fingerprint mismatch",
    )
    return manifest


def _safe_relative_uri(uri: Any, label: str) -> None:
    _non_blank(uri, label)
    require(
        not any(ord(character) < 32 or ord(character) == 127 for character in uri),
        f"{label} has control characters",
    )
    require(not uri.startswith(("/", "\\")), f"{label} must be relative")
    require(
        not (len(uri) >= 2 and uri[0].isalpha() and uri[1] == ":"),
        f"{label} must be relative",
    )
    first = re.split(r"[/\\]", uri, maxsplit=1)[0]
    require(":" not in first, f"{label} must not contain a scheme")
    require(".." not in re.split(r"[/\\]", uri), f"{label} must not traverse parents")


def _validate_output_binding(
    binding: dict[str, Any], graph: dict[str, Any], label: str
) -> None:
    require(
        binding["binding_fingerprint"]
        == fingerprint_without(_norm_output_binding(binding), "binding_fingerprint"),
        f"{label}.binding_fingerprint mismatch",
    )
    output = {
        "node_id": binding["node_id"],
        "port_name": binding["port_name"],
        "prediction_level": binding["prediction_level"],
        "unit_level": binding["unit_level"],
        "prediction_kind": binding["prediction_kind"],
        "target_names": binding["target_names"],
        "target_units": binding["target_units"],
        "class_labels": binding["class_labels"],
        "output_order": binding["output_order"],
        "target_space": binding["target_space"],
        "_graph": graph,
    }
    _validate_output_shape(output, label)
    require(
        (
            binding["prediction_source"] == "final_refit"
            and binding["refit_strategy"] is not None
        )
        or (
            binding["prediction_source"] != "final_refit"
            and binding["refit_strategy"] is None
        ),
        f"{label}.refit_strategy mismatch",
    )


def validate_portable_package(
    value: Any,
    label: str = "PortablePredictorPackage",
    *,
    relations: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    package = _exact_keys(
        value,
        {
            "schema_version",
            "package_id",
            "template",
            "training_request_fingerprint",
            "training_outcome",
            "effective_plan",
            "execution_bundle",
            "output_bindings",
            "predictor_node_ids",
            "training_influence",
            "data_identities",
            "fitted_artifact_mode",
            "artifact_bindings",
            "package_fingerprint",
        },
        label,
    )
    require(package["schema_version"] == 1, f"{label}.schema_version")
    _identifier(package["package_id"], f"{label}.package_id")
    require(not _contains_runtime_handle(package), f"{label} contains runtime handle")
    require(
        package["package_fingerprint"]
        == fingerprint_without(package, "package_fingerprint"),
        f"{label}.package_fingerprint mismatch",
    )
    _normalize_portable_predictor_package(
        package
    )  # fail closed on serde container types
    template = package["template"]
    require(
        template["template_fingerprint"]
        == fingerprint_without(
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
    # embedded plan instead of trusting the outcome-reference crosslinks.
    _validate_execution_plan(plan, f"{label}.effective_plan")
    require(
        template["graph"] == plan["graph_plan"]["graph"]
        and template["campaign"] == plan["campaign"]
        and template["controller_manifests"] == plan["controller_manifests"],
        f"{label}.template/plan mismatch",
    )
    bundle = package["execution_bundle"]
    require(
        bundle["plan_id"] == plan["id"]
        and bundle["graph_fingerprint"] == plan["graph_fingerprint"]
        and bundle["campaign_fingerprint"] == plan["campaign_fingerprint"]
        and bundle["controller_fingerprint"] == plan["controller_fingerprint"],
        f"{label}.bundle/plan fingerprint crosslink mismatch",
    )
    _validate_refit_artifacts_against_plan(bundle, plan, f"{label}.execution_bundle")
    _validate_bundle_data_requirements_against_plan(
        bundle, plan, f"{label}.execution_bundle"
    )
    _validate_bundle_selection_and_prediction_links(bundle, f"{label}.execution_bundle")
    outcome = _exact_keys(
        package["training_outcome"],
        {
            "outcome_id",
            "outcome_fingerprint",
            "training_request_fingerprint",
            "effective_plan_fingerprint",
            "execution_bundle_id",
            "execution_bundle_fingerprint",
            "output_binding_fingerprints",
            "training_influence_fingerprint",
            "data_identities_fingerprint",
        },
        f"{label}.training_outcome",
    )
    for field in (
        "outcome_fingerprint",
        "training_request_fingerprint",
        "effective_plan_fingerprint",
        "execution_bundle_fingerprint",
        "training_influence_fingerprint",
        "data_identities_fingerprint",
    ):
        _sha256(outcome[field], f"{label}.training_outcome.{field}")
    require(
        package["training_request_fingerprint"]
        == outcome["training_request_fingerprint"],
        f"{label}.request fingerprint is not cross-linked",
    )
    require(
        tcv1_sha256(_normalize_execution_plan(plan))
        == outcome["effective_plan_fingerprint"],
        f"{label}.effective plan crosslink",
    )
    require(
        bundle["bundle_id"] == outcome["execution_bundle_id"],
        f"{label}.bundle crosslink",
    )
    require(
        tcv1_sha256(_norm_execution_bundle(bundle))
        == outcome["execution_bundle_fingerprint"],
        f"{label}.execution bundle content is not cross-linked",
    )
    bindings = package["output_bindings"]
    require(bool(bindings), f"{label}.output_bindings must be non-empty")
    binding_ids = [binding["binding_id"] for binding in bindings]
    require(
        binding_ids == sorted(set(binding_ids)),
        f"{label}.output binding order/uniqueness",
    )
    coordinates: set[tuple[str, str]] = set()
    binding_fingerprints: list[str] = []
    for index, binding in enumerate(bindings):
        _validate_output_binding(
            binding, plan["graph_plan"]["graph"], f"{label}.output_bindings[{index}]"
        )
        coordinate = (binding["node_id"], binding["port_name"])
        require(
            coordinate not in coordinates,
            f"{label} binds an output coordinate more than once",
        )
        coordinates.add(coordinate)
        if binding["prediction_source"] == "final_refit":
            require(
                bool(bundle["refit_artifacts"]),
                f"{label}.final_refit requires artifacts",
            )
        binding_fingerprints.append(binding["binding_fingerprint"])
    require(
        binding_fingerprints == outcome["output_binding_fingerprints"],
        f"{label}.output binding crosslink",
    )
    influence = _validate_influence_manifest(
        package["training_influence"], f"{label}.training_influence", relations
    )
    require(
        influence["manifest_fingerprint"] == outcome["training_influence_fingerprint"],
        f"{label}.influence crosslink",
    )
    closure = _graph_closure(
        plan["graph_plan"]["graph"], [binding["node_id"] for binding in bindings]
    )
    require(
        package["predictor_node_ids"] == sorted(set(package["predictor_node_ids"])),
        f"{label}.predictor_node_ids order/uniqueness",
    )
    require(
        package["predictor_node_ids"] == closure, f"{label}.predictor closure mismatch"
    )
    # A portable package must independently prove PREDICT-replayability from its
    # own plan/closure/retained artifacts. PREDICT never consumes OOF payloads.
    require(
        "PREDICT" in _derive_replayable_phases(plan, closure, True, bundle, None),
        f"{label} is not PREDICT-replayable: its predictor closure does not "
        "support PREDICT with self-contained retained artifacts",
    )
    require(
        all(
            entry["node_id"] is None or entry["node_id"] in closure
            for entry in influence["entries"]
        ),
        f"{label}.influence node is outside predictor closure",
    )
    _validate_base_influence_against_plan(
        influence, plan, closure, f"{label}.training_influence"
    )
    identities = [
        validate_data_identity(identity, f"{label}.data_identities[{index}]")
        for index, identity in enumerate(package["data_identities"])
    ]
    identity_keys = [identity["requirement_key"] for identity in identities]
    require(
        identity_keys == sorted(set(identity_keys)),
        f"{label}.data identities must be sorted by requirement_key and unique",
    )
    requirements = {
        f"{requirement['node_id']}.{requirement['input_name']}": requirement
        for requirement in bundle["data_requirements"]
    }
    require(
        identity_keys == sorted(requirements), f"{label}.data identity key coverage"
    )
    for identity in identities:
        requirement = requirements[identity["requirement_key"]]
        require(
            identity["schema_fingerprint"] == requirement["schema_fingerprint"]
            and identity["plan_fingerprint"] == requirement["plan_fingerprint"]
            and identity["relation_fingerprint"]
            == requirement.get("relation_fingerprint")
            == influence["relation_fingerprint"],
            f"{label}.data identity does not match bundle fingerprints or influence relation",
        )
    require(
        tcv1_sha256(identities) == outcome["data_identities_fingerprint"],
        f"{label}.data identity content is not cross-linked",
    )
    artifact_records = {
        record["artifact"]["id"]: record for record in bundle["refit_artifacts"]
    }
    artifact_bindings = package["artifact_bindings"]
    artifact_ids = [binding["artifact_id"] for binding in artifact_bindings]
    require(
        artifact_ids == sorted(set(artifact_ids)),
        f"{label}.artifact binding order/uniqueness",
    )
    require(artifact_ids == sorted(artifact_records), f"{label}.artifact coverage")
    for binding in artifact_bindings:
        record = artifact_records[binding["artifact_id"]]
        artifact = record["artifact"]
        node_plan = plan["node_plans"][record["node_id"]]
        if binding["load_mode"] == "native_portable":
            require(
                artifact.get("backend") is not None
                and artifact.get("content_fingerprint") is not None
                and artifact.get("uri") is not None,
                f"{label}.native artifact is not portable",
            )
            _sha256(
                artifact.get("content_fingerprint"), f"{label}.native artifact content"
            )
            _safe_relative_uri(artifact.get("uri"), f"{label}.native artifact uri")
            require(
                node_plan["artifact_policy"] != "host_only",
                f"{label}.host_only artifact is not native portable",
            )
        else:
            require(
                package["fitted_artifact_mode"] == "allow_host_sidecar",
                f"{label}.portable_required package forbids host sidecar",
            )
    require(
        tcv1_sha256(package)
        == tcv1_sha256(_normalize_portable_predictor_package(package)),
        f"{label} wire content does not match its typed Rust serde representation",
    )
    return package


__all__ = [
    "ContractError",
    "validate_cache_namespace",
    "validate_data_identity",
    "validate_parameter_projection",
    "validate_portable_package",
    "validate_training_outcome",
    "validate_training_request",
]
