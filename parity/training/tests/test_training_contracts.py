"""Independent W1-0 schema, golden and semantic conformance tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import (  # noqa: E402
    ContractError,
    fingerprint_without,
    load_json,
    tcv1_sha256,
)
from parity.schema_dependencies import (  # noqa: E402
    missing_schema_dependencies,
    schema_dependency_closure,
)
from parity.training import oracle as training_oracle  # noqa: E402
from scripts import validate_contracts as production_contracts  # noqa: E402
from parity.training.generate_fixtures import (  # noqa: E402
    BASE_PACK_ARTIFACTS,
    PACK_ARTIFACTS,
    PACK_PATH,
    build_conformance_pack,
    generate,
    resign_package_plan,
    resign_training_outcome,
)
from parity.training.oracle import (  # noqa: E402
    _derive_replayable_phases,
    _graph_closure,
    validate_cache_namespace,
    validate_parameter_projection,
    validate_portable_package,
    validate_training_outcome,
    validate_training_request,
)

SCHEMA_DIR = ROOT / "docs" / "contracts"
FIXTURE_DIR = ROOT / "examples" / "fixtures" / "training"

POSITIVE_FIXTURES: dict[str, list[str]] = {
    "training_request.schema.json": [
        "training_request_refit.v1.json",
        "training_request_no_refit.v1.json",
        "training_request_active_influence.v1.json",
        "training_request_package_refit.v1.json",
    ],
    "portable_predictor_package.schema.json": ["portable_predictor_package.v1.json"],
    "cache_namespace.schema.json": ["cache_namespace_fit_cv.v1.json"],
    "parameter_projection.schema.json": ["parameter_projection_empty.v1.json"],
    "training_outcome.schema.json": ["training_outcome_refit.v1.json"],
}

SCHEMA_BY_CONTRACT = {
    "training_request": "training_request.schema.json",
    "portable_predictor_package": "portable_predictor_package.schema.json",
    "cache_namespace": "cache_namespace.schema.json",
    "training_outcome": "training_outcome.schema.json",
}


def _schemas_and_registry() -> tuple[dict[str, dict[str, Any]], Registry]:
    schemas: dict[str, dict[str, Any]] = {}
    resources: list[tuple[str, Resource[Any]]] = []
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = load_json(path)
        schema_id = schema.get("$id")
        assert isinstance(schema_id, str), f"{path} has no $id"
        Draft202012Validator.check_schema(schema)
        schemas[path.name] = schema
        resources.append((schema_id, Resource.from_contents(schema)))
    return schemas, Registry().with_resources(resources)


@pytest.fixture(scope="module")
def schemas_and_registry() -> tuple[dict[str, dict[str, Any]], Registry]:
    """Load all references locally; parity tests never fetch schemas."""

    return _schemas_and_registry()


def _semantic_validator(contract: str) -> Callable[[Any], Any]:
    return {
        "training_request": validate_training_request,
        "portable_predictor_package": validate_portable_package,
        "cache_namespace": validate_cache_namespace,
        "parameter_projection": validate_parameter_projection,
        "training_outcome": validate_training_outcome,
    }[contract]


def _contract_for_schema(schema_name: str) -> str:
    return {
        "training_request.schema.json": "training_request",
        "portable_predictor_package.schema.json": "portable_predictor_package",
        "cache_namespace.schema.json": "cache_namespace",
        "parameter_projection.schema.json": "parameter_projection",
        "training_outcome.schema.json": "training_outcome",
    }[schema_name]


@pytest.mark.parametrize("schema_name", sorted(POSITIVE_FIXTURES))
def test_w10_schemas_are_draft_2020_12_with_offline_refs(
    schema_name: str,
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    assert (
        schemas[schema_name]["$schema"]
        == "https://json-schema.org/draft/2020-12/schema"
    )
    Draft202012Validator(schemas[schema_name], registry=registry)


@pytest.mark.parametrize(
    ("schema_name", "fixture_name"),
    [
        (schema_name, fixture_name)
        for schema_name, fixture_names in sorted(POSITIVE_FIXTURES.items())
        for fixture_name in fixture_names
    ],
)
def test_positive_fixtures_pass_schema_and_independent_semantics(
    schema_name: str,
    fixture_name: str,
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    document = load_json(FIXTURE_DIR / fixture_name)
    Draft202012Validator(schemas[schema_name], registry=registry).validate(document)
    _semantic_validator(_contract_for_schema(schema_name))(document)


def _schema_errors(errors: list[Any]) -> str:
    return "\n".join(
        f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in errors
    )


def test_refingerprinted_negative_corpus_fails_for_declared_reason(
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    fixture = load_json(FIXTURE_DIR / "negative_cases.v1.json")
    ids: set[str] = set()
    for case in fixture["cases"]:
        assert case["id"] not in ids
        ids.add(case["id"])
        document = case["document"]
        contract = case["contract"]
        fingerprint_field = {
            "training_request": "request_fingerprint",
            "portable_predictor_package": "package_fingerprint",
            "cache_namespace": "namespace_fingerprint",
            "training_outcome": "outcome_fingerprint",
        }[contract]
        assert document[fingerprint_field] == fingerprint_without(
            document, fingerprint_field
        ), f"{case['id']} is not independently re-fingerprinted"
        validator = Draft202012Validator(
            schemas[SCHEMA_BY_CONTRACT[contract]], registry=registry
        )
        schema_errors = list(validator.iter_errors(document))
        semantic_error = ""
        try:
            _semantic_validator(contract)(document)
        except ContractError as exc:
            semantic_error = str(exc)
        combined = f"{_schema_errors(schema_errors)}\n{semantic_error}".lower()
        assert schema_errors or semantic_error, f"{case['id']} unexpectedly passed"
        assert case["expected_error"].lower() in combined, (
            f"{case['id']} failed for the wrong cause; expected "
            f"{case['expected_error']!r}, got {combined!r}"
        )


def test_active_capability_fixture_covers_two_folds_and_refit() -> None:
    request = load_json(FIXTURE_DIR / "training_request_active_influence.v1.json")
    validate_training_request(request)
    requirements = request["influence_requirements"]
    assert len(requirements) == 6
    assert {
        (requirement["kind"], requirement["phase"], requirement["fold_id"])
        for requirement in requirements
    } == {
        ("early_stopping", "FIT_CV", "fold:0"),
        ("early_stopping", "FIT_CV", "fold:1"),
        ("early_stopping", "REFIT", None),
        ("weighting_resampling", "FIT_CV", "fold:0"),
        ("weighting_resampling", "FIT_CV", "fold:1"),
        ("weighting_resampling", "REFIT", None),
    }


def test_fold_sample_wire_order_is_semantically_neutral() -> None:
    request = load_json(FIXTURE_DIR / "training_request_active_influence.v1.json")
    fold_set = request["campaign"]["split_invocation"]["fold_set"]
    fold_set["sample_ids"].reverse()
    for fold in fold_set["folds"]:
        fold["train_sample_ids"].reverse()
        fold["validation_sample_ids"].reverse()
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    validate_training_request(request)


@pytest.mark.parametrize("mutation", ("choices", "param_overrides"))
def test_request_search_space_fingerprint_binds_campaign_generation(
    mutation: str,
) -> None:
    request = load_json(FIXTURE_DIR / "training_request_package_refit.v1.json")
    generation = request["campaign"]["generation"]
    if mutation == "choices":
        generation["dimensions"][0]["choices"].reverse()
    else:
        overrides = generation["dimensions"][0]["choices"][0]["param_overrides"]
        assert len(overrides) > 1
        overrides.reverse()
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")

    with pytest.raises(ContractError, match="search_space_fingerprint"):
        validate_training_request(request, f"search-space {mutation} mutation")
    with pytest.raises(
        production_contracts.ContractError, match="search_space_fingerprint"
    ):
        production_contracts.validate_w10_training_request(
            request, f"search-space {mutation} mutation"
        )


def test_training_request_gpu_devices_must_be_sorted_and_unique() -> None:
    request = load_json(FIXTURE_DIR / "training_request_package_refit.v1.json")
    request["options"]["resources"]["gpu_devices"] = ["z", "a"]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")

    with pytest.raises(ContractError, match="gpu_devices"):
        validate_training_request(request, "non-canonical gpu devices")
    with pytest.raises(production_contracts.ContractError, match="gpu_devices"):
        production_contracts.validate_w10_training_request(
            request, "non-canonical gpu devices"
        )


def test_cache_namespace_binds_every_prediction_affecting_coordinate() -> None:
    namespace = load_json(FIXTURE_DIR / "cache_namespace_fit_cv.v1.json")
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    identity = next(
        item
        for item in package["data_identities"]
        if item["requirement_key"] == namespace["data_requirement_key"]
    )
    validate_cache_namespace(namespace, identity)
    fields = [
        "source_port_name",
        "target_port_name",
        "fold_id",
        "trial_id",
        "seed",
        "params_fingerprint",
        "data_identity_fingerprint",
    ]
    fingerprints: set[str] = set()
    for field in fields:
        changed = copy.deepcopy(namespace)
        if field in {"params_fingerprint", "data_identity_fingerprint"}:
            changed[field] = "f" * 64
        elif field == "seed":
            changed[field] += 1
        else:
            changed[field] += ":other"
        if field in {"source_port_name", "target_port_name"}:
            changed["prediction_requirement_key"] = (
                f"{changed['producer_node_id']}.{changed['source_port_name']}->"
                f"{changed['consumer_node_id']}.{changed['target_port_name']}"
            )
        changed["namespace_fingerprint"] = fingerprint_without(
            changed, "namespace_fingerprint"
        )
        fingerprints.add(changed["namespace_fingerprint"])
    assert len(fingerprints) == len(fields)
    assert namespace["namespace_fingerprint"] not in fingerprints


def test_parameter_patch_schema_refuses_blank_and_append_segments(
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    validator = Draft202012Validator(
        schemas["parameter_patch.schema.json"], registry=registry
    )
    base = {
        "schema_version": 1,
        "node_id": "model:base",
        "namespace": "operator",
        "path": ["alpha"],
        "value": 2,
    }
    validator.validate(base)
    for segment in (" ", "-"):
        invalid = copy.deepcopy(base)
        invalid["path"] = [segment]
        assert list(validator.iter_errors(invalid)), segment


def test_non_blank_schema_fields_match_rust_trim_semantics(
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    request = load_json(FIXTURE_DIR / "training_request_refit.v1.json")
    request["options"]["outputs"][0]["port_name"] = "   "
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    assert list(
        Draft202012Validator(
            schemas["training_request.schema.json"], registry=registry
        ).iter_errors(request)
    )
    namespace = load_json(FIXTURE_DIR / "cache_namespace_fit_cv.v1.json")
    namespace["trial_id"] = " "
    namespace["namespace_fingerprint"] = fingerprint_without(
        namespace, "namespace_fingerprint"
    )
    assert list(
        Draft202012Validator(
            schemas["cache_namespace.schema.json"], registry=registry
        ).iter_errors(namespace)
    )
    binding = load_json(
        ROOT
        / "examples"
        / "fixtures"
        / "estimator"
        / "output_binding_regression_final_refit.v1.json"
    )
    binding["target_space"] = "   "
    binding["binding_fingerprint"] = fingerprint_without(binding, "binding_fingerprint")
    assert list(
        Draft202012Validator(
            schemas["output_binding.schema.json"], registry=registry
        ).iter_errors(binding)
    )


def test_training_output_requires_explicit_null_unit_for_target_level(
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    validator = Draft202012Validator(
        schemas["training_request.schema.json"], registry=registry
    )
    request = load_json(FIXTURE_DIR / "training_request_refit.v1.json")
    output = request["options"]["outputs"][0]
    output["prediction_level"] = "target"
    output["unit_level"] = None
    request["campaign"]["aggregation_policy"]["selection_metric_level"] = "target"
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    validator.validate(request)
    validate_training_request(request)

    del output["unit_level"]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    assert list(validator.iter_errors(request))
    with pytest.raises(ContractError, match="unit_level must be explicit"):
        validate_training_request(request)


def _outcome_predictor_closure(outcome: dict[str, Any]) -> list[str]:
    roots = [output["binding"]["node_id"] for output in outcome["outputs"]]
    return production_contracts.w10_predictor_closure(
        outcome["effective_plan"]["graph_plan"]["graph"], roots
    )


def _assert_replay_derivation(outcome: dict[str, Any], expected: list[str]) -> None:
    plan = outcome["effective_plan"]
    production_closure = _outcome_predictor_closure(outcome)
    roots = [output["binding"]["node_id"] for output in outcome["outputs"]]
    oracle_closure = _graph_closure(plan["graph_plan"]["graph"], roots)
    assert production_closure == oracle_closure
    bundle = outcome["execution_bundle"]
    portable_caches = outcome.get("portable_prediction_caches")
    assert (
        production_contracts.derive_replayable_phases(
            plan,
            set(production_closure),
            outcome["refit"]["status"] == "completed",
            bundle,
            portable_caches,
            "replay parity probe",
        )
        == expected
    )
    assert (
        _derive_replayable_phases(
            plan,
            oracle_closure,
            outcome["refit"]["status"] == "completed",
            bundle,
            portable_caches,
        )
        == expected
    )


def test_no_refit_replay_requires_each_oof_portability_leg() -> None:
    source = load_json(
        ROOT
        / "examples"
        / "fixtures"
        / "estimator"
        / "training_outcome_no_refit.v1.json"
    )
    _assert_replay_derivation(source, ["REFIT"])

    graph = source["effective_plan"]["graph_plan"]["graph"]
    oof_edges = [
        edge for edge in graph["edges"] if edge["contract"].get("requires_oof") is True
    ]
    assert len(oof_edges) >= 2
    for edge in oof_edges:
        requirement_key = (
            f"{edge['source']['node_id']}.{edge['source']['port_name']}->"
            f"{edge['target']['node_id']}.{edge['target']['port_name']}"
        )
        for leg in ("requirement", "cache_record", "portable_payload"):
            mutated = copy.deepcopy(source)
            if leg == "requirement":
                records = mutated["execution_bundle"]["prediction_requirements"]
                before = len(records)
                mutated["execution_bundle"]["prediction_requirements"] = [
                    record
                    for record in records
                    if (
                        f"{record['producer_node']}.{record['source_port']}->"
                        f"{record['consumer_node']}.{record['target_port']}"
                    )
                    != requirement_key
                ]
                after = mutated["execution_bundle"]["prediction_requirements"]
            elif leg == "cache_record":
                records = mutated["execution_bundle"]["prediction_caches"]
                before = len(records)
                mutated["execution_bundle"]["prediction_caches"] = [
                    record
                    for record in records
                    if record["requirement_key"] != requirement_key
                ]
                after = mutated["execution_bundle"]["prediction_caches"]
            else:
                records = mutated["portable_prediction_caches"]["caches"]
                before = len(records)
                mutated["portable_prediction_caches"]["caches"] = [
                    record
                    for record in records
                    if record["requirement_key"] != requirement_key
                ]
                after = mutated["portable_prediction_caches"]["caches"]
            assert len(after) == before - 1, (requirement_key, leg)
            _assert_replay_derivation(mutated, [])


def test_stateless_replay_required_node_remains_predict_replayable() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    node_id = "branch:b1.augment:noise"
    controller_id = "controller:augmentation.mock"
    manifest = plan["controller_manifests"][controller_id]
    node_plan = plan["node_plans"][node_id]
    assert not ({"stateful", "emits_artifacts"} & set(manifest["capabilities"]))
    assert all(
        record["node_id"] != node_id
        for record in outcome["execution_bundle"]["refit_artifacts"]
    )

    manifest["artifact_policy"] = "replay_required"
    node_plan["artifact_policy"] = "replay_required"
    resign_training_outcome(outcome)

    production_contracts.validate_training_outcome(
        outcome, "stateless ReplayRequired outcome"
    )
    validate_training_outcome(outcome)
    _assert_replay_derivation(outcome, ["PREDICT"])


def _sample_cache_block_canonical(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_id": block.get("prediction_id"),
        "producer_node": block["producer_node"],
        "partition": block["partition"],
        "fold_id": block["fold_id"],
        "sample_ids": block["sample_ids"],
        "values": block["values"],
        "target_names": block.get("target_names", []),
    }


def _resign_sample_prediction_caches(outcome: dict[str, Any]) -> None:
    records_by_key = {
        record["requirement_key"]: record
        for record in outcome["execution_bundle"]["prediction_caches"]
    }
    for payload in outcome["portable_prediction_caches"]["caches"]:
        record = records_by_key[payload["requirement_key"]]
        assert len(payload["blocks"]) == len(record["blocks"])
        canonical_blocks = []
        for payload_block, record_block in zip(payload["blocks"], record["blocks"]):
            canonical = _sample_cache_block_canonical(payload_block)
            canonical_blocks.append(canonical)
            block_fingerprint = production_contracts._serde_sha256(canonical)
            assert block_fingerprint == training_oracle._serde_sha256(canonical)
            record_block["content_fingerprint"] = block_fingerprint
        content_fingerprint = production_contracts._serde_sha256(canonical_blocks)
        assert content_fingerprint == training_oracle._serde_sha256(canonical_blocks)
        payload["content_fingerprint"] = content_fingerprint
        record["content_fingerprint"] = content_fingerprint
    resign_training_outcome(outcome)


@pytest.mark.parametrize(
    "mutation",
    (
        "null",
        "cache_id",
        "content_fingerprint",
        "row_count",
        "block",
        "value",
        "format",
    ),
)
def test_portable_cache_mutations_are_rejected_after_outer_resign(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    payload = outcome["portable_prediction_caches"]["caches"][0]
    if mutation == "null":
        outcome["portable_prediction_caches"] = None
    elif mutation == "cache_id":
        payload["cache_id"] += ":forged"
    elif mutation == "content_fingerprint":
        payload["content_fingerprint"] = "0" * 64
    elif mutation == "row_count":
        payload["row_count"] += 1
    elif mutation == "block":
        payload["blocks"][0]["prediction_id"] += ":forged"
    elif mutation == "value":
        payload["blocks"][0]["values"][0][0] += 1.0
    else:
        payload["format"] = "forged-cache-format"
    resign_training_outcome(outcome)

    with pytest.raises(production_contracts.ContractError):
        production_contracts.validate_training_outcome(
            outcome, f"portable cache {mutation} mutation"
        )
    with pytest.raises(ContractError):
        validate_training_outcome(outcome, f"portable cache {mutation} mutation")


def test_portable_cache_binary64_value_passes_after_typed_serde_resign() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    block = outcome["portable_prediction_caches"]["caches"][0]["blocks"][0]
    block["values"][0][0] = 1e-7
    canonical = _sample_cache_block_canonical(block)
    assert production_contracts._serde_sha256(
        canonical
    ) != production_contracts.legacy_serde_json_sha256(canonical)
    _resign_sample_prediction_caches(outcome)

    production_contracts.validate_training_outcome(
        outcome, "typed serde binary64 cache outcome"
    )
    validate_training_outcome(outcome, "typed serde binary64 cache outcome")


def test_portable_cache_vec_orders_are_semantic_when_coupled() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    bundle = outcome["execution_bundle"]
    outcome["portable_prediction_caches"]["caches"].reverse()
    for requirement in bundle["prediction_requirements"]:
        requirement["fold_ids"].reverse()
        requirement["sample_ids"].reverse()
    payloads_by_key = {
        payload["requirement_key"]: payload
        for payload in outcome["portable_prediction_caches"]["caches"]
    }
    for record in bundle["prediction_caches"]:
        record["fold_ids"].reverse()
        record["sample_ids"].reverse()
        payload = payloads_by_key[record["requirement_key"]]
        payload["blocks"].reverse()
        record["blocks"].reverse()
        for payload_block, record_block in zip(payload["blocks"], record["blocks"]):
            payload_block["sample_ids"].reverse()
            payload_block["values"].reverse()
            record_block["sample_ids"].reverse()
    _resign_sample_prediction_caches(outcome)

    production_contracts.validate_training_outcome(
        outcome, "semantic cache Vec order outcome"
    )
    validate_training_outcome(outcome, "semantic cache Vec order outcome")


@pytest.mark.parametrize(
    "field",
    (
        "data_requirements",
        "prediction_requirements",
        "prediction_caches",
        "refit_artifacts",
    ),
)
def test_execution_bundle_record_vecs_allow_arbitrary_order(field: str) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    records = outcome["execution_bundle"][field]
    assert len(records) > 1, field
    records.reverse()
    resign_training_outcome(outcome)

    production_contracts.validate_training_outcome(
        outcome, f"permuted execution bundle {field}"
    )
    validate_training_outcome(outcome, f"permuted execution bundle {field}")


def test_refit_artifact_prediction_requirement_keys_allow_arbitrary_order() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    record = max(
        outcome["execution_bundle"]["refit_artifacts"],
        key=lambda value: len(value["prediction_requirement_keys"]),
    )
    assert len(record["prediction_requirement_keys"]) > 1
    record["prediction_requirement_keys"].reverse()
    resign_training_outcome(outcome)

    production_contracts.validate_training_outcome(
        outcome, "permuted refit prediction requirement keys"
    )
    validate_training_outcome(outcome, "permuted refit prediction requirement keys")


@pytest.mark.parametrize("field", ("fold_ids", "sample_ids"))
def test_portable_cache_vec_orders_must_match_requirement_and_record(
    field: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    outcome["execution_bundle"]["prediction_requirements"][0][field].reverse()
    resign_training_outcome(outcome)

    with pytest.raises(ContractError, match=field):
        validate_training_outcome(outcome, f"unilateral cache {field} order")
    with pytest.raises(production_contracts.ContractError, match=field):
        production_contracts.validate_training_outcome(
            outcome, f"unilateral cache {field} order"
        )


def test_aggregated_cache_unit_key_order_is_serde_normalized() -> None:
    # PredictionUnitId is a Vec, not a BTreeSet: vector order is semantic and
    # may be non-lexical, while each tagged unit object's key order is typed.
    wire_units = [
        {"id": "target:z", "level": "target"},
        {"id": "target:a", "level": "target"},
    ]
    canonical_units = [
        {"level": "target", "id": "target:z"},
        {"level": "target", "id": "target:a"},
    ]
    assert all(list(unit) == ["id", "level"] for unit in wire_units)
    canonical_block = {
        "prediction_id": "prediction:aggregate",
        "producer_node": "model:aggregate",
        "partition": "validation",
        "fold_id": "fold:0",
        "level": "target",
        "unit_ids": canonical_units,
        "values": [[1e-7], [2.0]],
        "target_names": ["y"],
    }
    block_fingerprint = production_contracts._serde_sha256(canonical_block)
    content_fingerprint = production_contracts._serde_sha256([canonical_block])
    assert block_fingerprint == training_oracle._serde_sha256(canonical_block)
    assert content_fingerprint == training_oracle._serde_sha256([canonical_block])
    requirement_key = "model:aggregate.oof->model:consumer.x"
    payload_set = {
        "bundle_id": "bundle:aggregate",
        "schema_version": 1,
        "caches": [
            {
                "requirement_key": requirement_key,
                "cache_id": "prediction-cache:aggregate",
                "format": "dag-ml-json-prediction-blocks-v1",
                "partition": "validation",
                "prediction_level": "target",
                "block_count": 1,
                "row_count": 2,
                "content_fingerprint": content_fingerprint,
                "aggregated_blocks": [
                    {
                        "prediction_id": "prediction:aggregate",
                        "producer_node": "model:aggregate",
                        "partition": "validation",
                        "fold_id": "fold:0",
                        "level": "target",
                        "unit_ids": wire_units,
                        "values": [[1e-7], [2.0]],
                        "target_names": ["y"],
                    }
                ],
            }
        ],
    }
    bundle = {
        "bundle_id": "bundle:aggregate",
        "prediction_requirements": [
            {
                "producer_node": "model:aggregate",
                "source_port": "oof",
                "consumer_node": "model:consumer",
                "target_port": "x",
                "partition": "validation",
                "prediction_level": "target",
                "fold_ids": ["fold:0"],
                "unit_ids": wire_units,
                "sample_ids": [],
                "prediction_width": 1,
                "target_names": ["y"],
            }
        ],
        "prediction_caches": [
            {
                "requirement_key": requirement_key,
                "cache_id": "prediction-cache:aggregate",
                "format": "dag-ml-json-prediction-blocks-v1",
                "partition": "validation",
                "prediction_level": "target",
                "fold_ids": ["fold:0"],
                "unit_ids": wire_units,
                "sample_ids": [],
                "prediction_width": 1,
                "target_names": ["y"],
                "block_count": 1,
                "row_count": 2,
                "content_fingerprint": content_fingerprint,
                "blocks": [
                    {
                        "prediction_id": "prediction:aggregate",
                        "fold_id": "fold:0",
                        "prediction_level": "target",
                        "unit_ids": wire_units,
                        "sample_ids": [],
                        "row_count": 2,
                        "content_fingerprint": block_fingerprint,
                    }
                ],
            }
        ],
    }

    production_contracts.validate_portable_prediction_caches(
        payload_set, bundle, "aggregated cache key-order probe"
    )
    training_oracle._validate_portable_caches_against_bundle(
        payload_set, bundle, "aggregated cache key-order probe"
    )

    unilateral = copy.deepcopy(bundle)
    unilateral["prediction_requirements"][0]["unit_ids"] = list(
        reversed(unilateral["prediction_requirements"][0]["unit_ids"])
    )
    with pytest.raises(production_contracts.ContractError, match="unit_ids"):
        production_contracts.validate_portable_prediction_caches(
            payload_set, unilateral, "unilateral aggregated unit order"
        )
    with pytest.raises(ContractError, match="unit_ids"):
        training_oracle._validate_portable_caches_against_bundle(
            payload_set, unilateral, "unilateral aggregated unit order"
        )


def _reversed_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in reversed(value)}


def _reverse_child(parent: dict[str, Any], key: str) -> dict[str, Any]:
    parent[key] = _reversed_mapping(parent[key])
    return parent[key]


def test_typed_serde_fingerprints_ignore_wire_key_order_and_match_float_spelling() -> (
    None
):
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    plan = package["effective_plan"]

    # Rust serde_json/zmij emits 1e-7 (not CPython json.dumps' 1e-07) for this
    # BTreeMap<String, Value>. Pin the exact Rust SHA before permuting its keys.
    params = plan["node_plans"]["branch:b1.augment:noise"]["params"]
    params["std"] = 1e-7
    rust_params_fingerprint = (
        "3f417903752f65005bc9b69bcd23dfcf3ede2cda010e4f1ead6090a1a407b851"
    )
    python_json_fingerprint = hashlib.sha256(
        json.dumps(
            params,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert python_json_fingerprint != rust_params_fingerprint
    plan["node_plans"]["branch:b1.augment:noise"]["params_fingerprint"] = (
        rust_params_fingerprint
    )

    graph = plan["graph_plan"]["graph"]
    plan["graph_plan"]["graph"] = _reversed_mapping(graph)
    graph = plan["graph_plan"]["graph"]
    _reverse_child(graph, "interface")
    node = graph["nodes"][0]
    graph["nodes"][0] = _reversed_mapping(node)
    node = graph["nodes"][0]
    _reverse_child(node, "ports")
    if node["ports"]["inputs"]:
        node["ports"]["inputs"][0] = _reversed_mapping(node["ports"]["inputs"][0])
    if isinstance(node["operator"], dict):
        _reverse_child(node, "operator")
    _reverse_child(node, "params")
    _reverse_child(node, "metadata")
    edge = graph["edges"][0]
    graph["edges"][0] = _reversed_mapping(edge)
    edge = graph["edges"][0]
    _reverse_child(edge, "source")
    _reverse_child(edge, "target")
    _reverse_child(edge, "contract")

    campaign = plan["campaign"]
    plan["campaign"] = _reversed_mapping(campaign)
    campaign = plan["campaign"]
    _reverse_child(campaign, "leakage_policy")
    _reverse_child(campaign, "aggregation_policy")
    split = _reverse_child(campaign, "split_invocation")
    _reverse_child(split, "leakage_policy")
    _reverse_child(split, "params")
    fold_set = _reverse_child(split, "fold_set")
    fold_set["folds"][0] = _reversed_mapping(fold_set["folds"][0])
    _reverse_child(fold_set["folds"][0], "metadata")
    _reverse_child(fold_set, "sample_groups")
    generation = _reverse_child(campaign, "generation")
    generation["dimensions"][0] = _reversed_mapping(generation["dimensions"][0])
    choice = generation["dimensions"][0]["choices"][0]
    generation["dimensions"][0]["choices"][0] = _reversed_mapping(choice)
    choice = generation["dimensions"][0]["choices"][0]
    _reverse_child(choice, "value")
    choice["param_overrides"][0] = _reversed_mapping(choice["param_overrides"][0])
    _reverse_child(choice["param_overrides"][0], "params")
    _reverse_child(campaign, "shape_plans")
    shape_plan = next(iter(campaign["shape_plans"].values()))
    normalized_shape = _reversed_mapping(shape_plan)
    campaign["shape_plans"][normalized_shape["node_id"]] = normalized_shape
    _reverse_child(normalized_shape, "aggregation_policy")
    _reverse_child(normalized_shape, "augmentation_policy")
    _reverse_child(normalized_shape, "selection_policy")
    _reverse_child(campaign, "data_bindings")
    binding = next(iter(campaign["data_bindings"].values()))[0]
    reversed_binding = _reversed_mapping(binding)
    next(iter(campaign["data_bindings"].values()))[0] = reversed_binding
    _reverse_child(reversed_binding, "view_policy")
    _reverse_child(reversed_binding, "metadata")
    _reverse_child(campaign, "metadata")

    manifests = plan["controller_manifests"]
    plan["controller_manifests"] = _reversed_mapping(manifests)
    controller_id = next(iter(plan["controller_manifests"]))
    manifest = plan["controller_manifests"][controller_id]
    plan["controller_manifests"][controller_id] = _reversed_mapping(manifest)
    input_port = plan["controller_manifests"][controller_id]["input_ports"][0]
    plan["controller_manifests"][controller_id]["input_ports"][0] = _reversed_mapping(
        input_port
    )
    metadata_node = next(
        node
        for node in plan["graph_plan"]["graph"]["nodes"]
        if len(node["metadata"]) >= 2
    )
    metadata_node["metadata"] = _reversed_mapping(metadata_node["metadata"])
    plan["node_plans"]["branch:b1.augment:noise"]["params"] = _reversed_mapping(params)

    package["training_outcome"]["effective_plan_fingerprint"] = tcv1_sha256(plan)
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")

    production_contracts.validate_w10_portable_package(
        package, "typed serde key-order package"
    )
    validate_portable_package(package)


def test_typed_serde_defaults_and_binary64_tokens_match_rust_goldens() -> None:
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    campaign = package["effective_plan"]["campaign"]

    cases: list[tuple[str, dict[str, Any], str]] = []

    missing_generation = copy.deepcopy(campaign)
    missing_generation.pop("generation")
    cases.append(
        (
            "missing whole GenerationSpec invokes GenerationSpec::default",
            missing_generation,
            "67adb4596b69e3b9e7d0c6c96c4f745092e00899478d27cbe9a9f7d902165efc",
        )
    )

    missing_view_policy = copy.deepcopy(campaign)
    for bindings in missing_view_policy["data_bindings"].values():
        for binding in bindings:
            binding.pop("view_policy")
    cases.append(
        (
            "missing whole DataViewPolicy invokes its custom default",
            missing_view_policy,
            "0778413bc8e3069ef4c2148cb17a5e15ec8ffbc38ca588d2146c6d2aee713c05",
        )
    )

    nested_cv = copy.deepcopy(campaign)
    nested_cv["inner_cv"] = {
        "seed": 5,
        "shuffle": False,
        "n_splits": 3,
        "kind": "kfold",
    }
    cases.append(
        (
            "NestedCvSpec uses enum struct order, not input key order",
            nested_cv,
            "3d7f5c05f9bcf373c8b949aaf0d8c4a9e869f55608954790e0e23f823c8782b3",
        )
    )

    for label, value, expected in cases:
        oracle_normalized = training_oracle._normalize_campaign_spec(value)
        production_normalized = production_contracts._normalize_campaign_spec(value)
        assert training_oracle._serde_sha256(oracle_normalized) == expected, label
        assert production_contracts._serde_sha256(production_normalized) == expected, (
            label
        )

    scalar_goldens = {
        1e-7: (
            "1e-7",
            "5b33e02f2c5103a05d32f6ba9cb058294452bfbf393967f68bb30c1bdcbbab22",
        ),
        1e-6: (
            "1e-6",
            "f465f55ffed8578e62d598c801779430834a6a908a7b2d25b4b6e9cb1e65b68d",
        ),
        1e-5: (
            "0.00001",
            "661710915adfa7c40b5dbd6b2122dfa65b1accb57a8b7cdee05423ecfe14b0c7",
        ),
        1e20: (
            "1e+20",
            "7c18c9fbdcc8281573e9db9e04f04c3790b10696f3706f0f03fa87427d33e28b",
        ),
        1e21: (
            "1e+21",
            "241c4643fa70b1dcde1205b71be4e3bebb17e9f880c8e1a33d0ead6c27271d3c",
        ),
        -0.0: (
            "-0.0",
            "c26617c7ccbcaa6631b45d851b8cf56e21d2ca624bdb1193afdbd4b560702cec",
        ),
        0.1: (
            "0.1",
            "14be4b45f18e0d8c67b4f719b5144eee88497e413709d11d85b096d8e2346310",
        ),
        2.0: (
            "2.0",
            "d84bdb34d4eeef4034d77e5403f850e35bc4a51b1143e3a83510e1aaad839748",
        ),
        5e-324: (
            "5e-324",
            "c46e7ca1be4c8734f373a56530787288fa2058d73d07855e9247e949f811a42a",
        ),
    }
    for value, (token, expected_sha256) in scalar_goldens.items():
        oracle_bytes = training_oracle._serde_encode(value)
        production_bytes = production_contracts._serde_encode(value)
        assert oracle_bytes == production_bytes == token.encode("ascii")
        assert hashlib.sha256(oracle_bytes).hexdigest() == expected_sha256


@pytest.mark.parametrize(
    "mutation",
    (
        "port_description_omitted",
        "edge_unit_level_null",
        "campaign_generation_omitted",
        "unknown_graph_field",
        "manifest_priority_omitted",
        "output_port_name_null",
        "resource_memory_null",
        "resource_wall_time_null",
        "scheduler_backend_omitted",
        "graph_interface_omitted",
        "campaign_metadata_omitted",
        "empty_operator_selectors_injected",
        "unknown_manifest_field",
    ),
)
def test_training_request_rejects_raw_wire_drift_from_typed_rust_serde(
    mutation: str,
) -> None:
    request = load_json(FIXTURE_DIR / "training_request_refit.v1.json")
    if mutation == "port_description_omitted":
        port = next(
            port
            for node in request["graph"]["nodes"]
            for direction in ("inputs", "outputs")
            for port in node["ports"][direction]
            if port.get("description") == ""
        )
        port.pop("description")
    elif mutation == "edge_unit_level_null":
        contract = request["graph"]["edges"][0]["contract"]
        assert "unit_level" not in contract
        contract["unit_level"] = None
    elif mutation == "campaign_generation_omitted":
        request["campaign"].pop("generation")
    elif mutation == "unknown_graph_field":
        request["graph"]["unknown_forward_field"] = True
    elif mutation == "manifest_priority_omitted":
        request["controller_manifests"][0].pop("priority")
    elif mutation == "output_port_name_null":
        request["options"]["outputs"][0]["port_name"] = None
    elif mutation == "resource_memory_null":
        request["options"]["resources"]["memory_bytes"] = None
    elif mutation == "resource_wall_time_null":
        request["options"]["resources"]["wall_time_ms"] = None
    elif mutation == "scheduler_backend_omitted":
        request["options"]["scheduler"].pop("backend")
    elif mutation == "graph_interface_omitted":
        request["graph"].pop("interface")
    elif mutation == "campaign_metadata_omitted":
        request["campaign"].pop("metadata")
    elif mutation == "empty_operator_selectors_injected":
        manifest = request["controller_manifests"][0]
        assert "operator_selectors" not in manifest
        manifest["operator_selectors"] = []
    else:
        request["controller_manifests"][0]["unknown_forward_field"] = True
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")

    with pytest.raises(ContractError):
        validate_training_request(request, f"typed request wire {mutation}")
    with pytest.raises(production_contracts.ContractError):
        production_contracts.validate_w10_training_request(
            request, f"typed request wire {mutation}"
        )


@pytest.mark.parametrize(
    "mutation", ("missing_prediction_id", "unknown_field", "integer_f64_token")
)
def test_training_outcome_output_blocks_reject_raw_typed_serde_drift(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    block = outcome["outputs"][0]["predictions"][0]
    if mutation == "missing_prediction_id":
        block.pop("prediction_id")
    elif mutation == "unknown_field":
        block["unknown_forward_field"] = True
    else:
        block["values"][0][0] = 1
    resign_training_outcome(outcome)

    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_training_outcome(outcome, f"typed output block {mutation}")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_training_outcome(
            outcome, f"typed output block {mutation}"
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "requirement_prediction_level_omitted",
        "requirement_empty_unit_ids_injected",
        "record_prediction_level_omitted",
        "record_empty_unit_ids_injected",
        "block_prediction_level_omitted",
        "block_empty_unit_ids_injected",
        "payload_prediction_level_omitted",
        "payload_empty_aggregated_blocks_injected",
    ),
)
def test_training_outcome_cache_defaults_and_skips_reject_raw_typed_serde_drift(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    bundle = outcome["execution_bundle"]
    requirement = bundle["prediction_requirements"][0]
    record = bundle["prediction_caches"][0]
    block = record["blocks"][0]
    payload = outcome["portable_prediction_caches"]["caches"][0]
    if mutation == "requirement_prediction_level_omitted":
        requirement.pop("prediction_level")
    elif mutation == "requirement_empty_unit_ids_injected":
        assert "unit_ids" not in requirement
        requirement["unit_ids"] = []
    elif mutation == "record_prediction_level_omitted":
        record.pop("prediction_level")
    elif mutation == "record_empty_unit_ids_injected":
        assert "unit_ids" not in record
        record["unit_ids"] = []
    elif mutation == "block_prediction_level_omitted":
        block.pop("prediction_level")
    elif mutation == "block_empty_unit_ids_injected":
        assert "unit_ids" not in block
        block["unit_ids"] = []
    elif mutation == "payload_prediction_level_omitted":
        payload.pop("prediction_level")
    else:
        assert "aggregated_blocks" not in payload
        payload["aggregated_blocks"] = []
    resign_training_outcome(outcome)

    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_training_outcome(outcome, f"typed cache wire {mutation}")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_training_outcome(
            outcome, f"typed cache wire {mutation}"
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "variant_label_null",
        "prediction_id_omitted",
        "target_names_omitted",
        "unknown_report_field",
    ),
)
def test_training_outcome_score_reports_reject_raw_typed_serde_drift(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    report = outcome["score_set"]["reports"][0]
    if mutation == "variant_label_null":
        assert "variant_label" not in report
        report["variant_label"] = None
    elif mutation == "prediction_id_omitted":
        report.pop("prediction_id")
    elif mutation == "target_names_omitted":
        report.pop("target_names")
    else:
        report["unknown_forward_field"] = True
    outcome["execution_bundle"]["scores"] = copy.deepcopy(outcome["score_set"])
    resign_training_outcome(outcome)

    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_training_outcome(outcome, f"typed score report {mutation}")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_training_outcome(
            outcome, f"typed score report {mutation}"
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "parameter_patch_order",
        "lineage_order",
        "input_lineage_order",
        "influence_entry_order",
        "influence_physical_sample_order",
        "bundle_score_drift",
    ),
)
def test_training_outcome_rejects_re_signed_canonical_order_and_score_drift(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    if mutation == "parameter_patch_order":
        outcome["parameter_patches"].reverse()
    elif mutation == "lineage_order":
        outcome["lineage"].reverse()
    elif mutation == "input_lineage_order":
        record = next(
            record for record in outcome["lineage"] if len(record["input_lineage"]) > 1
        )
        record["input_lineage"].reverse()
    elif mutation == "influence_entry_order":
        outcome["training_influence"]["entries"].reverse()
    elif mutation == "influence_physical_sample_order":
        outcome["training_influence"]["entries"][0]["physical_sample_ids"].reverse()
    else:
        reports = outcome["execution_bundle"]["scores"]["reports"]
        reports[0]["metrics"]["rmse"] += 0.01
    if mutation.startswith("influence_"):
        influence = outcome["training_influence"]
        influence["manifest_fingerprint"] = fingerprint_without(
            influence, "manifest_fingerprint"
        )
    resign_training_outcome(outcome)

    with pytest.raises(ContractError):
        validate_training_outcome(outcome, f"outcome {mutation}")
    with pytest.raises(production_contracts.ContractError):
        production_contracts.validate_training_outcome(outcome, f"outcome {mutation}")


def test_training_outcome_warnings_must_be_sorted_and_unique() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    outcome["warnings"] = ["z", "a"]
    resign_training_outcome(outcome)

    with pytest.raises(ContractError, match="warnings"):
        validate_training_outcome(outcome, "non-canonical warnings")
    with pytest.raises(production_contracts.ContractError, match="warnings"):
        production_contracts.validate_training_outcome(
            outcome, "non-canonical warnings"
        )


@pytest.mark.parametrize(
    "mutation",
    ("port_default", "edge_skip_none", "node_plan_default", "unknown_graph_field"),
)
def test_effective_plan_crosslink_hashes_typed_rust_serde_value(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    graph = plan["graph_plan"]["graph"]
    if mutation == "port_default":
        port = next(
            port
            for node in graph["nodes"]
            for direction in ("inputs", "outputs")
            for port in node["ports"][direction]
            if port.get("description") == ""
        )
        port.pop("description")
    elif mutation == "edge_skip_none":
        contract = graph["edges"][0]["contract"]
        assert "unit_level" not in contract
        contract["unit_level"] = None
    elif mutation == "node_plan_default":
        node_plan = next(
            node_plan
            for node_plan in plan["node_plans"].values()
            if node_plan.get("data_bindings") == []
        )
        node_plan.pop("data_bindings")
    else:
        graph["unknown_forward_field"] = {"must": "not affect typed content"}

    # Rust rejects the alternative wire shape even when every embedded hash and
    # the plan crosslink are recomputed from the typed serde value: its outer
    # typed TrainingOutcome fingerprint no longer equals the raw JSON one.
    resign_training_outcome(outcome)
    typed_fingerprint = outcome["effective_plan_fingerprint"]
    assert tcv1_sha256(plan) != typed_fingerprint
    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_training_outcome(outcome, f"typed plan wire {mutation}")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_training_outcome(
            outcome, f"typed plan wire {mutation}"
        )


def test_standalone_execution_plan_validates_deserialized_btreesets() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    manifest = max(
        plan["controller_manifests"].values(),
        key=lambda value: len(value["supported_phases"]) + len(value["capabilities"]),
    )
    assert len(manifest["supported_phases"]) > 1
    manifest["supported_phases"].reverse()
    manifest["capabilities"].reverse()

    # ExecutionPlan is parsed directly into BTreeSets and has no raw-wire
    # self-fingerprint, so both standalone validators consume the typed view.
    production_contracts.validate_execution_plan(plan, "typed standalone plan")
    training_oracle._validate_execution_plan(plan, "typed standalone plan")

    # The same non-canonical wire shape inside a signed parent is rejected by
    # TrainingOutcome's raw-vs-typed fingerprint parity gate.
    resign_training_outcome(outcome)
    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_training_outcome(outcome, "typed parent plan BTreeSet order")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_training_outcome(
            outcome, "typed parent plan BTreeSet order"
        )


def test_standalone_execution_plan_coerces_wide_value_integer_to_f64() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    wide_integer = 2**64
    plan["graph_plan"]["graph"]["metadata"]["wide_integer"] = wide_integer
    oracle_graph = training_oracle._normalize_graph_spec(plan["graph_plan"]["graph"])
    production_graph = production_contracts._normalize_graph_spec(
        plan["graph_plan"]["graph"]
    )
    assert oracle_graph == production_graph
    assert oracle_graph["metadata"]["wide_integer"] == float(wide_integer)
    plan["graph_fingerprint"] = training_oracle._serde_sha256(oracle_graph)

    production_contracts.validate_execution_plan(plan, "wide Value integer plan")
    training_oracle._validate_execution_plan(plan, "wide Value integer plan")

    # Strict TCV1 parents reject the out-of-range raw integer before typed serde.
    with pytest.raises(ContractError):
        fingerprint_without(outcome, "outcome_fingerprint")
    with pytest.raises(production_contracts.ContractError):
        production_contracts.training_outcome_fingerprint(outcome)


def test_standalone_execution_plan_rejects_integer_outside_finite_f64() -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    plan["graph_plan"]["graph"]["metadata"]["overflow"] = 10**1000

    with pytest.raises(ContractError, match="finite number range"):
        training_oracle._validate_execution_plan(plan, "overflow Value integer plan")
    with pytest.raises(production_contracts.ContractError, match="finite number range"):
        production_contracts.validate_execution_plan(
            plan, "overflow Value integer plan"
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "parallel_levels_map",
        "parallel_levels_null",
        "parallel_level_map",
        "choice_param_overrides_map",
        "campaign_data_binding_value_map",
        "node_data_bindings_map",
        "variants_map",
        "folds_map",
        "fold_sample_ids_map",
        "fold_train_ids_map",
        "fold_validation_ids_map",
        "binding_source_ids_map",
        "shape_plans_array",
        "campaign_data_bindings_array",
        "campaign_metadata_array",
        "campaign_leakage_policy_array",
        "campaign_aggregation_policy_array",
        "graph_interface_array",
        "node_inner_cv_array",
    ),
)
def test_standalone_execution_plan_rejects_wrong_serde_container_shapes(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    campaign = plan["campaign"]
    graph_plan = plan["graph_plan"]
    fold_set = plan["fold_set"]
    node_plan = next(iter(plan["node_plans"].values()))
    if mutation == "parallel_levels_map":
        graph_plan["parallel_levels"] = {}
    elif mutation == "parallel_levels_null":
        graph_plan["parallel_levels"] = None
    elif mutation == "parallel_level_map":
        graph_plan["parallel_levels"][0] = {}
    elif mutation == "choice_param_overrides_map":
        campaign["generation"]["dimensions"][0]["choices"][0]["param_overrides"] = {}
    elif mutation == "campaign_data_binding_value_map":
        first_key = next(iter(campaign["data_bindings"]))
        campaign["data_bindings"][first_key] = {}
    elif mutation == "node_data_bindings_map":
        node_plan["data_bindings"] = {}
    elif mutation == "variants_map":
        plan["variants"] = {}
    elif mutation == "folds_map":
        fold_set["folds"] = {}
    elif mutation == "fold_sample_ids_map":
        fold_set["sample_ids"] = {}
    elif mutation == "fold_train_ids_map":
        fold_set["folds"][0]["train_sample_ids"] = {}
    elif mutation == "fold_validation_ids_map":
        fold_set["folds"][0]["validation_sample_ids"] = {}
    elif mutation == "binding_source_ids_map":
        binding = next(
            binding
            for bindings in campaign["data_bindings"].values()
            for binding in bindings
        )
        binding["source_ids"] = {}
    elif mutation == "shape_plans_array":
        campaign["shape_plans"] = []
    elif mutation == "campaign_data_bindings_array":
        campaign["data_bindings"] = []
    elif mutation == "campaign_metadata_array":
        campaign["metadata"] = []
    elif mutation == "campaign_leakage_policy_array":
        campaign["leakage_policy"] = []
    elif mutation == "campaign_aggregation_policy_array":
        campaign["aggregation_policy"] = []
    elif mutation == "graph_interface_array":
        graph_plan["graph"]["interface"] = []
    else:
        node_plan["inner_cv"] = []

    with pytest.raises(ContractError):
        training_oracle._validate_execution_plan(plan, f"wrong-shape {mutation}")
    with pytest.raises(production_contracts.ContractError):
        production_contracts.validate_execution_plan(plan, f"wrong-shape {mutation}")


@pytest.mark.parametrize(
    "mutation",
    (
        "target_names",
        "target_units",
        "class_labels",
        "class_labels_inner",
        "gpu_devices",
        "fold_sample_ids",
        "binding_source_ids",
    ),
)
def test_signed_training_request_rejects_wrong_vec_shapes(mutation: str) -> None:
    request = load_json(FIXTURE_DIR / "training_request_package_refit.v1.json")
    output = request["options"]["outputs"][0]
    if mutation in {"target_names", "target_units", "class_labels"}:
        output[mutation] = {}
    elif mutation == "class_labels_inner":
        output["class_labels"] = [{}]
    elif mutation == "gpu_devices":
        request["options"]["resources"]["gpu_devices"] = {}
    elif mutation == "fold_sample_ids":
        request["campaign"]["split_invocation"]["fold_set"]["sample_ids"] = {}
    else:
        binding = next(
            binding
            for bindings in request["campaign"]["data_bindings"].values()
            for binding in bindings
        )
        binding["source_ids"] = {}
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")

    with pytest.raises(ContractError):
        validate_training_request(request, f"wrong request Vec {mutation}")
    with pytest.raises(production_contracts.ContractError):
        production_contracts.validate_w10_training_request(
            request, f"wrong request Vec {mutation}"
        )


@pytest.mark.parametrize("parent", ("outcome", "package"))
def test_signed_training_parents_reject_wrong_nested_vec_shape(parent: str) -> None:
    if parent == "outcome":
        document = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
        document["effective_plan"]["graph_plan"]["parallel_levels"] = {}
        document["outcome_fingerprint"] = fingerprint_without(
            document, "outcome_fingerprint"
        )
        validators = (
            lambda: validate_training_outcome(document, "wrong-shape outcome"),
            lambda: production_contracts.validate_training_outcome(
                document, "wrong-shape outcome"
            ),
        )
    else:
        document = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
        document["effective_plan"]["graph_plan"]["parallel_levels"] = {}
        document["package_fingerprint"] = fingerprint_without(
            document, "package_fingerprint"
        )
        validators = (
            lambda: validate_portable_package(document, "wrong-shape package"),
            lambda: production_contracts.validate_w10_portable_package(
                document, "wrong-shape package"
            ),
        )
    for validate in validators:
        with pytest.raises((ContractError, production_contracts.ContractError)):
            validate()


@pytest.mark.parametrize("field", ("supported_phases", "controller_capabilities"))
def test_node_plan_manifest_phase_and_capability_copies_must_match(field: str) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    node_plan = next(value for value in plan["node_plans"].values() if value[field])
    node_plan[field] = node_plan[field][:-1]

    with pytest.raises(ContractError, match=field):
        training_oracle._validate_execution_plan(plan, f"mismatched {field}")
    with pytest.raises(production_contracts.ContractError):
        production_contracts.validate_execution_plan(plan, f"mismatched {field}")


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest",
        "operator_selector",
        "model_input",
        "model_input_port",
        "representation_plan",
        "combination_plan",
    ),
)
def test_standalone_execution_plan_rejects_deny_unknown_manifest_closure(
    mutation: str,
) -> None:
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    manifest = next(iter(plan["controller_manifests"].values()))
    if mutation == "manifest":
        manifest["unknown_forward_field"] = True
    elif mutation == "operator_selector":
        manifest["operator_selectors"] = [
            {"aliases": ["mock"], "unknown_forward_field": True}
        ]
    else:
        port = {
            "name": "x",
            "accepted_representations": ["tabular"],
            "accepted_types": ["f64"],
        }
        data_requirements: dict[str, Any] = {
            "schema_version": 1,
            "ports": [port],
            "metadata": {},
        }
        manifest["data_requirements"] = data_requirements
        if mutation == "model_input":
            data_requirements["unknown_forward_field"] = True
        elif mutation == "model_input_port":
            port["unknown_forward_field"] = True
        else:
            representation: dict[str, Any] = {
                "kind": "cartesian_product",
                "combination_plan": {"mode": "cartesian"},
                "output_unit_level": "combo",
                "cardinality": "many_to_many",
            }
            data_requirements["default_fusion"] = {
                "mode": "concat",
                "representation_plan": representation,
            }
            if mutation == "representation_plan":
                representation["unknown_forward_field"] = True
            else:
                representation["combination_plan"]["unknown_forward_field"] = True

    with pytest.raises(ContractError, match="unknown field"):
        training_oracle._validate_execution_plan(plan, f"unknown {mutation} plan")
    with pytest.raises(production_contracts.ContractError, match="unknown field"):
        production_contracts.validate_execution_plan(plan, f"unknown {mutation} plan")


def test_standalone_execution_plan_ignores_unknown_fields_without_deny_unknown() -> (
    None
):
    outcome = load_json(FIXTURE_DIR / "training_outcome_refit.v1.json")
    plan = outcome["effective_plan"]
    graph = plan["graph_plan"]["graph"]
    graph["unknown_forward_field"] = True
    manifest = next(
        value
        for value in plan["controller_manifests"].values()
        if value["input_ports"] or value["output_ports"]
    )
    ports = manifest["input_ports"] or manifest["output_ports"]
    ports[0]["unknown_forward_field"] = True
    plan["graph_fingerprint"] = training_oracle._serde_sha256(
        training_oracle._normalize_graph_spec(graph)
    )
    plan["controller_fingerprint"] = training_oracle._serde_sha256(
        training_oracle._normalize_controller_manifests(plan["controller_manifests"])
    )

    production_contracts.validate_execution_plan(plan, "allowed unknown typed plan")
    training_oracle._validate_execution_plan(plan, "allowed unknown typed plan")


def test_portable_package_rejects_noncanonical_typed_plan_wire() -> None:
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    plan = package["effective_plan"]
    node_plan = next(
        node_plan
        for node_plan in plan["node_plans"].values()
        if node_plan.get("data_bindings") == []
    )
    node_plan.pop("data_bindings")
    resign_package_plan(package)

    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_portable_package(package, "typed package plan wire")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_w10_portable_package(
            package, "typed package plan wire"
        )


def _resign_package_bundle(package: dict[str, Any]) -> None:
    bundle = package["execution_bundle"]
    package["training_outcome"]["execution_bundle_fingerprint"] = tcv1_sha256(
        production_contracts._norm_execution_bundle(bundle)
    )
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")


def test_portable_package_selection_must_start_with_selected_candidate() -> None:
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    decision = next(iter(package["execution_bundle"]["selections"].values()))
    assert len(decision["ranked_candidates"]) > 1
    decision["ranked_candidates"].reverse()
    _resign_package_bundle(package)

    with pytest.raises(ContractError, match="first ranked candidate"):
        validate_portable_package(package, "misranked package selection")
    with pytest.raises(
        production_contracts.ContractError, match="first ranked candidate"
    ):
        production_contracts.validate_w10_portable_package(
            package, "misranked package selection"
        )


@pytest.mark.parametrize("field", ("fold_ids", "sample_ids"))
@pytest.mark.parametrize("coupled", (False, True))
def test_portable_package_prediction_requirement_vecs_are_exactly_crosslinked(
    field: str,
    coupled: bool,
) -> None:
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    bundle = package["execution_bundle"]
    requirement = bundle["prediction_requirements"][0]
    requirement_key = (
        f"{requirement['producer_node']}.{requirement['source_port']}->"
        f"{requirement['consumer_node']}.{requirement['target_port']}"
    )
    record = next(
        record
        for record in bundle["prediction_caches"]
        if record["requirement_key"] == requirement_key
    )
    assert len(requirement[field]) > 1
    requirement[field].reverse()
    if coupled:
        record[field].reverse()
    _resign_package_bundle(package)

    if coupled:
        production_contracts.validate_w10_portable_package(
            package, f"coupled package {field} order"
        )
        validate_portable_package(package, f"coupled package {field} order")
    else:
        with pytest.raises(ContractError, match=field):
            validate_portable_package(package, f"unilateral package {field} order")
        with pytest.raises(production_contracts.ContractError, match=field):
            production_contracts.validate_w10_portable_package(
                package, f"unilateral package {field} order"
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "null_representation_replay_manifest",
        "unknown_data_requirement_field",
        "unknown_refit_artifact_record_field",
    ),
)
def test_portable_package_rejects_nested_bundle_raw_typed_serde_drift(
    mutation: str,
) -> None:
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    bundle = package["execution_bundle"]
    if mutation == "null_representation_replay_manifest":
        requirement = bundle["data_requirements"][0]
        assert "representation_replay_manifest" not in requirement
        requirement["representation_replay_manifest"] = None
    elif mutation == "unknown_data_requirement_field":
        bundle["data_requirements"][0]["unknown_forward_field"] = True
    else:
        bundle["refit_artifacts"][0]["unknown_forward_field"] = True
    package["training_outcome"]["execution_bundle_fingerprint"] = tcv1_sha256(
        production_contracts._norm_execution_bundle(bundle)
    )
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")

    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_portable_package(package, f"typed package bundle {mutation}")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_w10_portable_package(
            package, f"typed package bundle {mutation}"
        )


def test_portable_package_rejects_integer_tokens_deserialized_as_f64_scores() -> None:
    package = load_json(FIXTURE_DIR / "portable_predictor_package.v1.json")
    bundle = package["execution_bundle"]
    selected_variant = package["training_outcome"]["output_binding_fingerprints"]
    assert selected_variant
    decision = next(iter(bundle["selections"].values()))
    winning_variant = decision["selected_candidate_id"]
    scores_by_variant = {
        candidate["candidate_id"]: rank
        for rank, candidate in enumerate(decision["ranked_candidates"], start=1)
    }
    for report in bundle["scores"]["reports"]:
        report["metrics"][decision["metric_name"]] = scores_by_variant[
            report["variant_id"]
        ]
    for candidate in decision["ranked_candidates"]:
        candidate["score"] = scores_by_variant[candidate["candidate_id"]]
    decision["selected_score"] = scores_by_variant[winning_variant]
    package["training_outcome"]["execution_bundle_fingerprint"] = tcv1_sha256(
        production_contracts._norm_execution_bundle(bundle)
    )
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")
    assert tcv1_sha256(bundle) != tcv1_sha256(
        production_contracts._norm_execution_bundle(bundle)
    )

    with pytest.raises(ContractError, match="typed Rust serde"):
        validate_portable_package(package, "typed package integer f64 scores")
    with pytest.raises(production_contracts.ContractError, match="typed Rust serde"):
        production_contracts.validate_w10_portable_package(
            package, "typed package integer f64 scores"
        )


def test_generator_is_byte_identical(tmp_path: Path) -> None:
    generate(tmp_path)
    committed = sorted(path.name for path in FIXTURE_DIR.glob("*.json"))
    generated = sorted(path.name for path in tmp_path.glob("*.json"))
    assert generated == committed
    for name in committed:
        assert (tmp_path / name).read_bytes() == (FIXTURE_DIR / name).read_bytes(), name


def test_training_conformance_pack_is_current() -> None:
    committed = load_json(PACK_PATH)
    assert committed == build_conformance_pack()
    assert committed["artifacts"]
    assert committed["negative_case_ids"] == [
        case["id"]
        for case in load_json(FIXTURE_DIR / "negative_cases.v1.json")["cases"]
    ]


def test_training_pack_hashes_all_transitive_schema_references() -> None:
    assert not missing_schema_dependencies(ROOT, PACK_ARTIFACTS)


def test_training_pack_refuses_missing_true_two_hop_schema_dependency() -> None:
    seed = "docs/contracts/training_outcome.schema.json"
    intermediate = "docs/contracts/execution_bundle.schema.json"
    omitted = "docs/contracts/selection_decision.schema.json"
    assert seed in BASE_PACK_ARTIFACTS
    assert intermediate not in BASE_PACK_ARTIFACTS
    assert omitted not in BASE_PACK_ARTIFACTS
    intermediate_id = load_json(ROOT / intermediate)["$id"]
    omitted_id = load_json(ROOT / omitted)["$id"]
    seed_wire = json.dumps(load_json(ROOT / seed), sort_keys=True)
    intermediate_wire = json.dumps(load_json(ROOT / intermediate), sort_keys=True)
    assert intermediate_id in seed_wire
    assert omitted_id not in seed_wire
    assert omitted_id in intermediate_wire
    assert {seed, intermediate, omitted} <= set(
        schema_dependency_closure(ROOT, [seed]).paths
    )

    pack = build_conformance_pack()
    assert omitted in {entry["path"] for entry in pack["artifacts"]}
    pack["artifacts"] = [
        entry for entry in pack["artifacts"] if entry["path"] != omitted
    ]
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    with pytest.raises(
        production_contracts.ContractError,
        match="selection_decision.schema.json",
    ):
        production_contracts.validate_w10_training_pack(
            pack,
            load_json(FIXTURE_DIR / "negative_cases.v1.json"),
        )


def test_training_pack_refuses_missing_schema_dependency_resolver() -> None:
    pack = build_conformance_pack()
    omitted = "parity/schema_dependencies.py"
    assert omitted in {entry["path"] for entry in pack["artifacts"]}
    pack["artifacts"] = [
        entry for entry in pack["artifacts"] if entry["path"] != omitted
    ]
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    with pytest.raises(
        production_contracts.ContractError,
        match="schema_dependencies.py",
    ):
        production_contracts.validate_w10_training_pack(
            pack,
            load_json(FIXTURE_DIR / "negative_cases.v1.json"),
        )


def test_training_pack_refuses_a_symlinked_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".github").symlink_to(ROOT / ".github", target_is_directory=True)
    (tmp_path / "crates").symlink_to(ROOT / "crates", target_is_directory=True)
    monkeypatch.setattr(production_contracts, "ROOT", tmp_path)
    with pytest.raises(production_contracts.ContractError, match="symbolic link"):
        production_contracts.validate_w10_training_pack(
            load_json(PACK_PATH),
            load_json(FIXTURE_DIR / "negative_cases.v1.json"),
        )


def test_tcv1_distinguishes_integer_and_binary64_patch_values() -> None:
    integer = {
        "schema_version": 1,
        "node_id": "model:base",
        "namespace": "operator",
        "path": ["alpha"],
        "value": 2,
    }
    binary64 = copy.deepcopy(integer)
    binary64["value"] = 2.0
    assert tcv1_sha256(integer) != tcv1_sha256(binary64)


def test_strict_loader_refuses_duplicate_keys_and_tcv1_refuses_nfc_collisions(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate key"):
        load_json(duplicate)
    with pytest.raises(ContractError, match="NFC-colliding"):
        tcv1_sha256({"é": 1, "e\u0301": 2})


def test_semantic_target_order_is_not_lexicographically_sorted() -> None:
    request = load_json(FIXTURE_DIR / "training_request_refit.v1.json")
    output = request["options"]["outputs"][0]
    output["target_names"] = ["zinc", "ash"]
    output["target_units"] = ["percent", "percent"]
    output["class_labels"] = [[], []]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    validate_training_request(request)
    original = request["request_fingerprint"]
    output["target_names"].reverse()
    output["target_units"].reverse()
    output["class_labels"].reverse()
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    validate_training_request(request)
    assert request["request_fingerprint"] != original


def test_class_label_and_decision_score_allow_empty_w0_vocabulary(
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    validator = Draft202012Validator(
        schemas["training_request.schema.json"], registry=registry
    )
    request = load_json(FIXTURE_DIR / "training_request_refit.v1.json")
    output = request["options"]["outputs"][0]
    for prediction_kind in ("class_label", "decision_score"):
        for vocabulary in ([], ["low", "high"]):
            output["prediction_kind"] = prediction_kind
            output["output_order"] = "target_order"
            output["class_labels"] = [vocabulary]
            request["request_fingerprint"] = fingerprint_without(
                request, "request_fingerprint"
            )
            validator.validate(request)
            if prediction_kind == "class_label":
                request["options"]["selection"]["metric"] = {
                    "name": "accuracy",
                    "objective": "maximize",
                }
                request["request_fingerprint"] = fingerprint_without(
                    request, "request_fingerprint"
                )
                validate_training_request(request)
            else:
                with pytest.raises(ContractError, match="DecisionScore"):
                    validate_training_request(request)

    request["options"]["selection"]["metric"] = {
        "name": "rmse",
        "objective": "minimize",
    }
    output["prediction_kind"] = "regression_point"
    output["class_labels"] = [["invalid"]]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    with pytest.raises(ContractError, match="must be empty"):
        validate_training_request(request)

    output["prediction_kind"] = "class_probability"
    output["output_order"] = "target_major_class_minor"
    output["class_labels"] = [[]]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    with pytest.raises(ContractError, match="must be non-empty"):
        validate_training_request(request)
