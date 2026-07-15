"""D4 public training-replay schema, oracle and pack conformance tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import load_json  # noqa: E402
from parity.training.generate_training_replay_fixtures import (  # noqa: E402
    OUT,
    build_conformance_pack,
    confined_artifact_path,
    generate,
    replay_pack_artifacts,
)
from parity.training import training_replay_oracle as oracle  # noqa: E402
from scripts import validate_contracts as base  # noqa: E402
from scripts import validate_training_replay_contracts as production  # noqa: E402

TRAINING_FIXTURES = ROOT / "examples" / "fixtures" / "training"
FIXTURES = TRAINING_FIXTURES / "replay"
BASE_PACK = ROOT / "docs" / "contracts" / "training_contract_conformance_pack.v1.json"
REPLAY_PACK = (
    ROOT / "docs" / "contracts" / "training_replay_contract_conformance_pack.v1.json"
)
LEGACY_AUTHORITY_SHA256 = {
    "docs/contracts/replay_outcome.schema.json": "c57279e8c76e4e2467af0eca5eb59804a2f7bb97bec6cce9d8b23975f223c36a",
    "examples/fixtures/estimator/replay_outcome_predict.v1.json": "037fad7f3cb907f3474cce4f51526538f2c4d6fcad3af93a320c6d282ce470c5",
    "examples/fixtures/estimator/replay_outcome_class_probability.v1.json": "2bcb925b79f1766515c924697f7b5ff62ede396e10095e183a14baefdd622329",
    "examples/fixtures/estimator/replay_outcome_explain.v1.json": "fe593f9bdd89ecfcffdb224435b0ce842f5a492a7b8045657ba22bfc63185db7",
    "docs/contracts/aggregation_controller_task.schema.json": "2b12131727f5e3a355b0c6b5e402f6075c37cf5ed3e7a186c9e0890da5583ccd",
    "docs/contracts/aggregation_controller_result.schema.json": "e782d57c2bff01031ab4cf453b362afab5bf25e1e83eac5cf65ef463347045ff",
    "docs/contracts/process_adapter_frame.schema.json": "024ee268eca668479acc1e0ddf979247fb1214f5022373ce36f85e55bf9499f3",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_base_pack_remains_byte_current() -> None:
    pack = load_json(BASE_PACK)
    assert _sha256(BASE_PACK) == (
        "e6b044da102a208353f3952d8887694dfc43ab9327ac62f4f9797f962d50e7d3"
    )
    assert pack["pack_checksum"] == (
        "e190bdfd6fb9485d1e905d1a4c78c1a155cab364a9004dcf78c54a2e147e85d4"
    )
    assert len(pack["artifacts"]) == 88
    assert all(
        _sha256(ROOT / artifact["path"]) == artifact["sha256"]
        for artifact in pack["artifacts"]
    )


def test_legacy_replay_and_protocol_authorities_remain_byte_current() -> None:
    assert {
        relative_path: _sha256(ROOT / relative_path)
        for relative_path in LEGACY_AUTHORITY_SHA256
    } == LEGACY_AUTHORITY_SHA256


def test_all_v2_roots_and_legacy_boundary_validate_offline() -> None:
    registry, schemas = base.build_local_schema_registry()
    production.validate_schema_contracts(schemas, registry)


def test_production_and_independent_oracle_accept_all_positives() -> None:
    production.validate_positive_fixtures()


def test_production_and_independent_oracle_reject_all_negatives() -> None:
    production.validate_negative_fixtures()


def test_fixture_generation_is_byte_deterministic(tmp_path: Path) -> None:
    generate(tmp_path)
    names = sorted(
        path.name
        for path in OUT.glob("*.json")
        if path.name.startswith("training_replay_")
        or path.name
        in {
            "training_outcome_port_explicit.v2.json",
            "training_port_explicit_protocols.v2.json",
        }
    )
    assert names
    assert all(
        (tmp_path / name).read_bytes() == (OUT / name).read_bytes() for name in names
    )


def test_multi_port_explicit_blocks_and_port_aware_score_keys() -> None:
    fixture = load_json(FIXTURES / "training_replay_multi_port_outputs.v1.json")
    for output in fixture["outputs"]:
        production.validate_bound_output(
            output, "multi-port", plan=fixture["effective_plan"]
        )
        oracle.validate_replay_bound_output(
            output, fixture["effective_plan"], "oracle.multi-port"
        )

    score = load_json(FIXTURES / "training_outcome_port_explicit.v2.json")["score_set"]
    report = copy.deepcopy(score["reports"][0])
    alternate = copy.deepcopy(report)
    alternate["prediction_id"] = "prediction:alternate-port"
    alternate["producer_port"] = "alternate_prediction"
    two_ports = {**score, "reports": [report, alternate]}
    production.validate_score_set_v2(two_ports, fixture["effective_plan"])
    oracle.validate_score_set_v2(two_ports, fixture["effective_plan"])

    duplicate = copy.deepcopy(alternate)
    duplicate["producer_port"] = report["producer_port"]
    same_port = {**score, "reports": [report, duplicate]}
    with pytest.raises(base.ContractError, match="duplicate score report"):
        production.validate_score_set_v2(same_port, fixture["effective_plan"])
    with pytest.raises(oracle.ContractError, match="duplicate score report"):
        oracle.validate_score_set_v2(same_port, fixture["effective_plan"])


def test_probability_simplex_absolute_tolerance_and_no_renormalization() -> None:
    source = load_json(TRAINING_FIXTURES / "training_outcome_refit.v1.json")
    original = load_json(FIXTURES / "training_replay_output_class_probability.v1.json")
    accepted = copy.deepcopy(original)
    accepted["predictions"][0]["values"][0][:2] = [0.5, 0.5000000000005]
    snapshot = copy.deepcopy(accepted)
    production.validate_bound_output(
        accepted, "simplex.accept", plan=source["effective_plan"]
    )
    oracle.validate_replay_bound_output(
        accepted, source["effective_plan"], "simplex.accept"
    )
    assert accepted == snapshot

    rejected = copy.deepcopy(original)
    rejected["predictions"][0]["values"][0][:2] = [0.5, 0.500000000002]
    with pytest.raises(base.ContractError, match="simplex"):
        production.validate_bound_output(
            rejected, "simplex.reject", plan=source["effective_plan"]
        )
    with pytest.raises(oracle.ContractError, match="simplex"):
        oracle.validate_replay_bound_output(
            rejected, source["effective_plan"], "simplex.reject"
        )


def test_v1_non_empty_text_boundaries_match_both_replay_engines() -> None:
    source = load_json(TRAINING_FIXTURES / "training_outcome_refit.v1.json")
    output = load_json(FIXTURES / "training_replay_outcome_predict.v1.json")["outputs"][
        0
    ]
    output["predictions"][0]["prediction_id"] = "   "
    production.validate_bound_output(
        output, "whitespace.production", plan=source["effective_plan"]
    )
    oracle.validate_replay_bound_output(
        output, source["effective_plan"], "whitespace.oracle"
    )

    request = load_json(FIXTURES / "training_replay_request_predict.v1.json")
    envelopes = load_json(FIXTURES / "training_replay_input_envelopes.v1.json")
    outcome = load_json(FIXTURES / "training_replay_outcome_predict.v1.json")
    outcome["diagnostics"] = {"   ": True}
    outcome["lineage"][0]["unsafe_flags"] = ["z", "a"]
    outcome["lineage"][0]["metrics"] = {"   ": 1.0}
    outcome["outcome_fingerprint"] = production.replay_outcome_fingerprint(outcome)
    production.validate_replay_outcome(
        outcome,
        "whitespace.production.outcome",
        request=request,
        source_outcome=source,
        envelope_fixture=envelopes,
    )
    oracle.validate_replay_outcome(
        outcome,
        "whitespace.oracle.outcome",
        request=request,
        source_outcome=source,
        envelope_fixture=envelopes,
    )


def test_score_set_v2_preserves_rust_v1_validation_boundaries() -> None:
    outcome = load_json(FIXTURES / "training_outcome_port_explicit.v2.json")
    plan = outcome["effective_plan"]
    empty = {
        "schema_version": 2,
        "plan_id": plan["id"],
        "selection_metric": "",
        "reports": [],
    }
    production.validate_score_set_v2(empty, plan, "empty.production")
    oracle.validate_score_set_v2(empty, plan, "empty.oracle")
    registry, schemas = base.build_local_schema_registry()
    # The historical V1 schema used minimum=1 rather than const=1. Runtime
    # readers still reject V2 through their exact version constant.
    base.validate_draft_2020_instance(
        empty,
        _v1_schema(schemas, "score_set.schema.json"),
        registry,
        "empty-v2-score-against-permissive-v1-schema",
    )

    permissive = copy.deepcopy(outcome["score_set"])
    report = permissive["reports"][0]
    permissive["reports"] = [report]
    permissive["selection_metric"] = ""
    report["prediction_id"] = ""
    report["variant_label"] = ""
    report["target_width"] = 2
    report["target_names"] = ["same", "same"]
    production.validate_score_set_v2(permissive, plan, "permissive.production")
    oracle.validate_score_set_v2(permissive, plan, "permissive.oracle")

    blank_metric = copy.deepcopy(permissive)
    blank_metric["reports"][0]["metrics"] = {"   ": 1.0}
    with pytest.raises(base.ContractError, match="non-blank"):
        production.validate_score_set_v2(blank_metric, plan, "blank.production")
    with pytest.raises(oracle.ContractError):
        oracle.validate_score_set_v2(blank_metric, plan, "blank.oracle")


def test_relation_metadata_is_btreemap_canonical() -> None:
    envelopes = load_json(FIXTURES / "training_replay_input_envelopes.v1.json")
    relations = copy.deepcopy(
        next(iter(envelopes["envelopes"].values()))["coordinator_relations"]
    )
    left = copy.deepcopy(relations)
    right = copy.deepcopy(relations)
    left["records"][0]["metadata"] = {"z": {"b": 2, "a": 1}, "a": True}
    right["records"][0]["metadata"] = {"a": True, "z": {"a": 1, "b": 2}}
    production_left = production.replay_relation_fingerprint(left, "left")
    production_right = production.replay_relation_fingerprint(right, "right")
    oracle_left = oracle.replay_relation_fingerprint(left, "left")
    oracle_right = oracle.replay_relation_fingerprint(right, "right")
    assert production_left == production_right == oracle_left == oracle_right


def test_v1_schemas_remain_port_absent_and_v2_is_port_explicit() -> None:
    v1 = load_json(ROOT / "docs/contracts/node_result.schema.json")
    v2 = load_json(ROOT / "docs/contracts/node_result.v2.schema.json")
    for name in (
        "prediction_block",
        "observation_prediction_block",
        "aggregated_prediction_block",
        "explanation_block",
    ):
        assert "producer_port" not in v1["$defs"][name]["properties"]
        assert "producer_port" in v2["$defs"][name]["properties"]
        assert "producer_port" in v2["$defs"][name]["required"]

    score_v1 = load_json(ROOT / "docs/contracts/score_set.schema.json")
    score_v2 = load_json(ROOT / "docs/contracts/score_set.v2.schema.json")
    report_v1 = copy.deepcopy(score_v1["$defs"]["regression_metric_report"])
    report_v2 = copy.deepcopy(score_v2["$defs"]["regression_metric_report"])
    report_v2["required"].remove("producer_port")
    report_v2["properties"].pop("producer_port")
    assert report_v2 == report_v1

    legacy_node_result = load_json(
        ROOT / "examples/fixtures/runtime/node_result_transform_scale.json"
    )
    legacy_prediction = copy.deepcopy(
        load_json(TRAINING_FIXTURES / "training_outcome_refit.v1.json")["outputs"][0][
            "predictions"
        ][0]
    )
    legacy_prediction["producer_port"] = "prediction"
    legacy_node_result["predictions"] = [legacy_prediction]
    registry, schemas = base.build_local_schema_registry()
    with pytest.raises(base.ContractError):
        base.validate_draft_2020_instance(
            legacy_node_result,
            _v1_schema(schemas, "node_result.schema.json"),
            registry,
            "v1-node-result-with-port",
        )


def test_all_v2_schema_families_preserve_v1_constraints_exactly() -> None:
    schemas = {
        schema["$id"]: schema
        for path in sorted((ROOT / "docs/contracts").glob("*.schema.json"))
        if "$id" in (schema := load_json(path))
    }
    pairs = [
        ("node_result.schema.json", "node_result.v2.schema.json", None),
        (
            "aggregation_controller_task.schema.json",
            "aggregation_controller_task.v2.schema.json",
            None,
        ),
        (
            "aggregation_controller_result.schema.json",
            "aggregation_controller_result.v2.schema.json",
            None,
        ),
        (
            "process_adapter_frame.schema.json",
            "process_adapter_frame.v2.schema.json",
            None,
        ),
        (
            "prediction_cache_payload_set.schema.json",
            "prediction_cache_payload_set.v2.schema.json",
            None,
        ),
        ("score_set.schema.json", "score_set.v2.schema.json", None),
        ("execution_bundle.schema.json", "execution_bundle.v2.schema.json", None),
        ("training_outcome.schema.json", "training_outcome.v2.schema.json", None),
        (
            "output_binding.schema.json",
            "bound_training_output.v2.schema.json",
            "/$defs/bound_output",
        ),
    ]
    for v1_name, v2_name, v1_pointer in pairs:
        v1_document = load_json(ROOT / "docs/contracts" / v1_name)
        v2_document = load_json(ROOT / "docs/contracts" / v2_name)
        v1_root = _json_pointer(v1_document, v1_pointer) if v1_pointer else v1_document
        v1 = _normalize_v2_schema_delta(
            _dereference_schema(v1_root, v1_document, schemas, frozenset())
        )
        v2 = _normalize_v2_schema_delta(
            _dereference_schema(v2_document, v2_document, schemas, frozenset())
        )
        assert v2 == v1, f"non-additive V2 schema drift in {v2_name}"


def test_every_v2_positive_root_is_rejected_by_its_v1_schema() -> None:
    registry, schemas = base.build_local_schema_registry()
    protocols = load_json(FIXTURES / "training_port_explicit_protocols.v2.json")
    outcome = load_json(FIXTURES / "training_outcome_port_explicit.v2.json")
    multi_port = load_json(FIXTURES / "training_replay_multi_port_outputs.v1.json")
    cases = [
        ("node_result.schema.json", protocols["node_result"]),
        ("process_adapter_frame.schema.json", protocols["process_adapter_result"]),
        (
            "aggregation_controller_task.schema.json",
            protocols["aggregation_task_observation"],
        ),
        (
            "aggregation_controller_task.schema.json",
            protocols["aggregation_task_unit"],
        ),
        (
            "aggregation_controller_result.schema.json",
            protocols["aggregation_result_sample"],
        ),
        (
            "aggregation_controller_result.schema.json",
            protocols["aggregation_result_unit"],
        ),
        (
            "prediction_cache_payload_set.schema.json",
            outcome["portable_prediction_caches"],
        ),
        ("execution_bundle.schema.json", outcome["execution_bundle"]),
        ("score_set.schema.json", outcome["score_set"]),
        ("training_outcome.schema.json", outcome),
        ("bound_training_output.schema.json", multi_port["outputs"][0]),
        ("bound_training_output.schema.json", multi_port["outputs"][1]),
    ]
    for index, (schema_name, instance) in enumerate(cases):
        with pytest.raises(base.ContractError):
            base.validate_draft_2020_instance(
                instance,
                _v1_schema(schemas, schema_name),
                registry,
                f"v2-against-v1[{index}]",
            )


def test_v2_prediction_cache_namespace_fingerprints_are_required_and_non_empty() -> None:
    registry, schemas = base.build_local_schema_registry()
    outcome = load_json(FIXTURES / "training_outcome_port_explicit.v2.json")
    bundle_schema = schemas[
        load_json(ROOT / "docs/contracts/execution_bundle.v2.schema.json")["$id"]
    ]
    payload_schema = schemas[
        load_json(ROOT / "docs/contracts/prediction_cache_payload_set.v2.schema.json")[
            "$id"
        ]
    ]
    bundle = outcome["execution_bundle"]
    payload_set = outcome["portable_prediction_caches"]

    for label, document, path, schema in (
        (
            "v2-bundle",
            bundle,
            ("prediction_caches", 0, "cache_namespace_fingerprints"),
            bundle_schema,
        ),
        (
            "v2-payload",
            payload_set,
            ("caches", 0, "cache_namespace_fingerprints"),
            payload_schema,
        ),
    ):
        missing = copy.deepcopy(document)
        target = missing
        for member in path[:-1]:
            target = target[member]
        target.pop(path[-1])
        with pytest.raises(base.ContractError, match="cache_namespace_fingerprints"):
            base.validate_draft_2020_instance(
                missing,
                schema,
                registry,
                f"{label}-missing-cache-namespace-fingerprints",
            )

        empty = copy.deepcopy(document)
        target = empty
        for member in path[:-1]:
            target = target[member]
        target[path[-1]] = []
        with pytest.raises(base.ContractError, match="cache_namespace_fingerprints"):
            base.validate_draft_2020_instance(
                empty,
                schema,
                registry,
                f"{label}-empty-cache-namespace-fingerprints",
            )


def test_v1_positive_roots_remain_accepted() -> None:
    registry, schemas = base.build_local_schema_registry()
    source = load_json(TRAINING_FIXTURES / "training_outcome_refit.v1.json")
    protocols = load_json(FIXTURES / "training_port_explicit_protocols.v2.json")
    cases = [
        (
            "node_result.schema.json",
            load_json(
                ROOT / "examples/fixtures/runtime/node_result_transform_scale.json"
            ),
        ),
        (
            "process_adapter_frame.schema.json",
            load_json(
                ROOT
                / "examples/fixtures/runtime/process_adapter_frame_result_transform_scale.json"
            ),
        ),
        (
            "prediction_cache_payload_set.schema.json",
            source["portable_prediction_caches"],
        ),
        ("execution_bundle.schema.json", source["execution_bundle"]),
        ("score_set.schema.json", source["score_set"]),
        ("training_outcome.schema.json", source),
        ("bound_training_output.schema.json", source["outputs"][0]),
        (
            "aggregation_controller_task.schema.json",
            _legacy_port_projection(protocols["aggregation_task_observation"]),
        ),
        (
            "aggregation_controller_task.schema.json",
            _legacy_port_projection(protocols["aggregation_task_unit"]),
        ),
        (
            "aggregation_controller_result.schema.json",
            _legacy_port_projection(protocols["aggregation_result_sample"]),
        ),
        (
            "aggregation_controller_result.schema.json",
            _legacy_port_projection(protocols["aggregation_result_unit"]),
        ),
    ]
    for index, (schema_name, instance) in enumerate(cases):
        base.validate_draft_2020_instance(
            instance,
            _v1_schema(schemas, schema_name),
            registry,
            f"v1-current[{index}]",
        )


def test_training_exclusion_does_not_remove_replay_membership() -> None:
    envelopes = load_json(FIXTURES / "training_replay_input_envelopes.v1.json")
    for envelope in envelopes["envelopes"].values():
        envelope["coordinator_relations"]["records"][0]["excluded"] = True
    outputs = load_json(FIXTURES / "training_replay_outcome_predict.v1.json")["outputs"]
    production.validate_replay_output_cohort(outputs, envelopes, "excluded.production")
    oracle._validate_output_cohort(outputs, envelopes, "excluded.oracle")


def test_pack_paths_reject_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe replay-pack artifact path"):
        confined_artifact_path("../outside.json")
    with pytest.raises(base.ContractError, match="unsafe replay-pack artifact path"):
        production._confined_artifact_path("../outside.json")


def test_replay_pack_is_exact_current_transitive_closure() -> None:
    pack = load_json(REPLAY_PACK)
    expected_artifacts = replay_pack_artifacts()
    assert pack == build_conformance_pack()
    assert [entry["path"] for entry in pack["artifacts"]] == sorted(expected_artifacts)
    assert all(
        entry["kind"] == expected_artifacts[entry["path"]]
        and entry["sha256"] == _sha256(ROOT / entry["path"])
        for entry in pack["artifacts"]
    )
    production.validate_pack()


def test_json_fixtures_have_no_duplicate_members() -> None:
    for path in sorted(FIXTURES.glob("training_replay_*.json")):
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _reject_duplicate_pairs(path, pairs),
        )


def _reject_duplicate_pairs(path: Path, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate key {key!r} in {path}"
        result[key] = value
    return result


def _legacy_port_projection(value: Any) -> Any:
    if isinstance(value, dict):
        projected = {
            key: _legacy_port_projection(member)
            for key, member in value.items()
            if key != "producer_port"
        }
        if projected.get("schema_version") == 2:
            projected["schema_version"] = 1
        return projected
    if isinstance(value, list):
        return [_legacy_port_projection(member) for member in value]
    return value


def _v1_schema(schemas: dict[str, dict[str, Any]], schema_name: str) -> dict[str, Any]:
    if schema_name == "bound_training_output.schema.json":
        output_binding = load_json(ROOT / "docs/contracts/output_binding.schema.json")
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": f"{output_binding['$id']}#/$defs/bound_output",
        }
    schema = load_json(ROOT / "docs/contracts" / schema_name)
    return schemas[schema["$id"]]


def _json_pointer(document: Any, pointer: str | None) -> Any:
    if pointer in (None, "", "/"):
        return document
    value = document
    for token in pointer.removeprefix("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _dereference_schema(
    value: Any,
    document: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    active_refs: frozenset[str],
) -> Any:
    if isinstance(value, list):
        return [
            _dereference_schema(member, document, schemas, active_refs)
            for member in value
        ]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        reference = value["$ref"]
        base_id, separator, fragment = reference.partition("#")
        target_document = schemas[base_id] if base_id else document
        canonical_ref = f"{target_document.get('$id', '<local>')}#{fragment}"
        if canonical_ref in active_refs:
            resolved: Any = {"$recursive_schema": True}
        else:
            target = _json_pointer(target_document, fragment or None)
            resolved = _dereference_schema(
                target,
                target_document,
                schemas,
                active_refs | {canonical_ref},
            )
        siblings = {
            key: _dereference_schema(member, document, schemas, active_refs)
            for key, member in value.items()
            if key != "$ref"
        }
        if isinstance(resolved, dict):
            return {**resolved, **siblings}
        assert not siblings
        return resolved
    return {
        key: _dereference_schema(member, document, schemas, active_refs)
        for key, member in value.items()
    }


def _normalize_v2_schema_delta(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_v2_schema_delta(member) for member in value]
    if not isinstance(value, dict):
        if value in {
            "dag-ml-json-prediction-blocks-v1",
            "dag-ml-json-prediction-blocks-v2",
        }:
            return "dag-ml-json-prediction-blocks-vN"
        return value
    normalized: dict[str, Any] = {}
    v2_only_fields = {"schema_version", "producer_port", "cache_namespace_fingerprints"}
    for key, member in value.items():
        if key in {"$schema", "$id", "$defs", "title", "description"}:
            continue
        if key == "properties":
            normalized[key] = {
                property_name: _normalize_v2_schema_delta(property_schema)
                for property_name, property_schema in member.items()
                if property_name not in v2_only_fields
            }
            continue
        if key == "required":
            normalized[key] = [name for name in member if name not in v2_only_fields]
            continue
        normalized[key] = _normalize_v2_schema_delta(member)
    return normalized
