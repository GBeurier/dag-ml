#!/usr/bin/env python3
"""Deterministically generate the isolated D4 training-replay contract pack."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parity.conformal.generate_fixtures import (  # noqa: E402
    derive_source_outcome_replayable_phases,
)
from parity.conformal.oracle import (  # noqa: E402
    fingerprint_without,
    load_json,
    tcv1_sha256,
)
from parity.schema_dependencies import (  # noqa: E402
    with_transitive_schema_dependencies,
)
from parity.training.generate_fixtures import (  # noqa: E402
    _mutate_manifest_and_node_plans,
    file_sha256,
    opaque,
    resign_training_outcome,
    write_json,
)
from parity.training.oracle import (  # noqa: E402
    _normalize_campaign_spec,
    _serde_sha256,
)

TRAINING_FIXTURES = ROOT / "examples" / "fixtures" / "training"
OUT = TRAINING_FIXTURES / "replay"
BASE_PACK_PATH = (
    ROOT / "docs" / "contracts" / "training_contract_conformance_pack.v1.json"
)
PACK_PATH = (
    ROOT / "docs" / "contracts" / "training_replay_contract_conformance_pack.v1.json"
)
BASE_PACK_ID = "dag-ml.training-contracts.v1"
BASE_PACK_SHA256 = "8e340296758cd9d75b1348901e40d91e7e34ebe3e665b5486f8ae87ad7e49061"
BASE_PACK_CHECKSUM = "c329f9839ba6c029e63c36a40efa050984ed9a8dd7a85fce92927fc05e4f6d5a"
serde_json_sha256 = _serde_sha256
LEGACY_AUTHORITY_SHA256 = {
    "docs/contracts/replay_outcome.schema.json": "c57279e8c76e4e2467af0eca5eb59804a2f7bb97bec6cce9d8b23975f223c36a",
    "examples/fixtures/estimator/replay_outcome_predict.v1.json": "037fad7f3cb907f3474cce4f51526538f2c4d6fcad3af93a320c6d282ce470c5",
    "examples/fixtures/estimator/replay_outcome_class_probability.v1.json": "2bcb925b79f1766515c924697f7b5ff62ede396e10095e183a14baefdd622329",
    "examples/fixtures/estimator/replay_outcome_explain.v1.json": "fe593f9bdd89ecfcffdb224435b0ce842f5a492a7b8045657ba22bfc63185db7",
    "docs/contracts/aggregation_controller_task.schema.json": "2b12131727f5e3a355b0c6b5e402f6075c37cf5ed3e7a186c9e0890da5583ccd",
    "docs/contracts/aggregation_controller_result.schema.json": "e782d57c2bff01031ab4cf453b362afab5bf25e1e83eac5cf65ef463347045ff",
    "docs/contracts/process_adapter_frame.schema.json": "024ee268eca668479acc1e0ddf979247fb1214f5022373ce36f85e55bf9499f3",
}

D4_ARTIFACTS = {
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
    "docs/contracts/score_set.v2.schema.json": "schema",
    "docs/contracts/replay_outcome.schema.json": "legacy_schema_context",
    "docs/contracts/training_outcome.v2.schema.json": "schema",
    "docs/contracts/training_replay_outcome.schema.json": "schema",
    "docs/contracts/training_replay_request.schema.json": "schema",
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
    "examples/fixtures/training/replay/training_outcome_port_explicit.v2.json": "fixture",
    "examples/fixtures/training/replay/training_port_explicit_protocols.v2.json": "fixture",
    "examples/fixtures/estimator/replay_outcome_class_probability.v1.json": "legacy_fixture",
    "examples/fixtures/estimator/replay_outcome_explain.v1.json": "legacy_fixture",
    "examples/fixtures/estimator/replay_outcome_predict.v1.json": "legacy_fixture",
    "parity/training/generate_training_replay_fixtures.py": "generator",
    "parity/training/training_replay_oracle.py": "test_oracle",
    "parity/training/tests/test_training_replay_contracts.py": "test",
    "scripts/requirements-contracts.txt": "ci_dependency",
    "scripts/validate_training_replay_contracts.py": "production_validator",
}


def confined_artifact_path(relative_path: str) -> Path:
    """Resolve a regular repository file without traversal or symlinks."""

    path = Path(relative_path)
    if path.is_absolute() or "\\" in relative_path or ".." in path.parts:
        raise ValueError(f"unsafe replay-pack artifact path: {relative_path}")
    candidate = ROOT / path
    cursor = ROOT
    for part in path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"replay-pack artifact traverses symlink: {relative_path}")
    if not candidate.is_file() or not candidate.resolve().is_relative_to(
        ROOT.resolve()
    ):
        raise ValueError(
            f"replay-pack artifact is not a confined file: {relative_path}"
        )
    return candidate


def artifact_sha256(relative_path: str) -> str:
    return file_sha256(confined_artifact_path(relative_path))


PHASE_ORDER = {
    phase: index
    for index, phase in enumerate(
        ("COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN")
    )
}


def training_outcome_ref(outcome: dict[str, Any]) -> dict[str, Any]:
    """Build the complete transplant-resistant reference frozen by D2."""

    bindings = [output["binding"] for output in outcome["outputs"]]
    return {
        "outcome_id": outcome["outcome_id"],
        "outcome_fingerprint": outcome["outcome_fingerprint"],
        "training_request_fingerprint": outcome["training_request_fingerprint"],
        "effective_plan_fingerprint": outcome["effective_plan_fingerprint"],
        "execution_bundle_id": outcome["execution_bundle"]["bundle_id"],
        "execution_bundle_fingerprint": tcv1_sha256(outcome["execution_bundle"]),
        "output_binding_fingerprints": [
            binding["binding_fingerprint"] for binding in bindings
        ],
        "training_influence_fingerprint": outcome["training_influence"][
            "manifest_fingerprint"
        ],
        "data_identities_fingerprint": tcv1_sha256(outcome["data_identities"]),
    }


def replay_source_outcome_explain(
    source_outcome: dict[str, Any],
) -> dict[str, Any]:
    """Derive an honest source whose complete closure supports EXPLAIN.

    The ordinary source fixture deliberately advertises PREDICT only.  D4 needs
    one positive EXPLAIN authority without weakening that D2 golden, so this
    dedicated fixture adds EXPLAIN consistently to each manifest/node plan and
    re-signs every affected historical and TCV1 fingerprint.
    """

    outcome = copy.deepcopy(source_outcome)
    outcome["outcome_id"] = "training:estimator.refit.explain"
    for controller_id in sorted(outcome["effective_plan"]["controller_manifests"]):
        _mutate_manifest_and_node_plans(
            outcome["effective_plan"],
            controller_id,
            supported_phases=lambda phases: sorted(
                {*phases, "EXPLAIN"}, key=PHASE_ORDER.__getitem__
            ),
        )
    outcome["replayable_phases"] = derive_source_outcome_replayable_phases(outcome)
    assert outcome["replayable_phases"] == ["PREDICT", "EXPLAIN"]
    return resign_training_outcome(outcome)


def canonical_relation_fingerprint(relations: dict[str, Any]) -> str:
    """Mirror ``SampleRelationSet::fingerprint`` for replay fixtures."""

    canonical: list[dict[str, Any]] = []
    for source in relations["records"]:
        unit_level = source.get("unit_level", "observation")
        unit_id = source.get("unit_id")
        if unit_id is not None:
            effective_unit_id = unit_id
        elif unit_level == "physical_sample":
            effective_unit_id = source["sample_id"]
        elif unit_level == "source_sample":
            effective_unit_id = f"{source['sample_id']}::{source['source_id']}"
        elif unit_level == "combo":
            effective_unit_id = source["derived_unit_id"]
        else:
            effective_unit_id = source["observation_id"]
        record = {
            "effective_unit_id": effective_unit_id,
            "unit_level": unit_level,
            "unit_id": unit_id,
            "observation_id": source["observation_id"],
            "sample_id": source["sample_id"],
            "source_id": source.get("source_id"),
            "rep_id": source.get("rep_id"),
            "target_id": source.get("target_id"),
            "group_id": source.get("group_id"),
            "origin_sample_id": source.get("origin_sample_id"),
            "derived_unit_id": source.get("derived_unit_id"),
            "component_observation_ids": source.get("component_observation_ids", []),
            "sample_influence_weight": source.get("sample_influence_weight"),
            "quality_flag": source.get("quality_flag"),
            "is_augmented": source.get("is_augmented", False),
        }
        if source.get("excluded", False):
            record["excluded"] = True
        if source.get("metadata"):
            record["metadata"] = source["metadata"]
        if source.get("tags"):
            record["tags"] = source["tags"]
        canonical.append(record)
    canonical.sort(
        key=lambda record: (
            record["effective_unit_id"],
            record["observation_id"],
            record["sample_id"],
        )
    )
    return serde_json_sha256(canonical)


def replay_input_envelopes(source_outcome: dict[str, Any]) -> dict[str, Any]:
    """Build exact current-cohort envelopes for every source-plan binding."""

    template = load_json(
        ROOT
        / "examples"
        / "fixtures"
        / "data"
        / "coordinator_data_plan_envelope_sample12.json"
    )
    relations = {
        "records": [
            {
                "observation_id": "obs:prod.1",
                "sample_id": "sample:prod.1",
                "target_id": "target:prod.1",
                "group_id": "group:production",
                "origin_sample_id": None,
                "source_id": "nir",
                "is_augmented": False,
            },
            {
                "observation_id": "obs:prod.2",
                "sample_id": "sample:prod.2",
                "target_id": "target:prod.2",
                "group_id": "group:production",
                "origin_sample_id": None,
                "source_id": "nir",
                "is_augmented": False,
            },
        ]
    }
    relation_fingerprint = canonical_relation_fingerprint(relations)
    plan = source_outcome["effective_plan"]
    bindings = {
        f"{binding['node_id']}.{binding['input_name']}": binding
        for bindings in plan["campaign"]["data_bindings"].values()
        for binding in bindings
    }
    requirement_keys = sorted(bindings)
    envelopes: dict[str, Any] = {}
    for key in requirement_keys:
        envelope = copy.deepcopy(template)
        envelope["relation_fingerprint"] = relation_fingerprint
        envelope["data_content_fingerprint"] = opaque(f"replay:{key}:data")
        envelope["target_content_fingerprint"] = opaque(f"replay:{key}:target")
        envelope["coordinator_relations"] = relations
        envelope["metadata"] = {
            "contract_fixture": "replay_input_envelopes.v1",
            "feature_set_id": bindings[key]["feature_set_id"],
            "requirement_key": key,
        }
        envelopes[key] = envelope
    return {"schema_version": 1, "envelopes": envelopes}


def replay_data_identities(envelope_fixture: dict[str, Any]) -> list[dict[str, Any]]:
    identities = []
    for key, envelope in envelope_fixture["envelopes"].items():
        identity = {
            "requirement_key": key,
            "schema_fingerprint": envelope["schema_fingerprint"],
            "plan_fingerprint": envelope["plan_fingerprint"],
            "relation_fingerprint": envelope["relation_fingerprint"],
            "data_content_fingerprint": envelope["data_content_fingerprint"],
            "target_content_fingerprint": envelope["target_content_fingerprint"],
            "identity_fingerprint": "0" * 64,
        }
        identity["identity_fingerprint"] = fingerprint_without(
            identity, "identity_fingerprint"
        )
        identities.append(identity)
    return identities


def replay_request(
    source_outcome: dict[str, Any],
    envelope_fixture: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "request_id": f"replay:request.{phase.lower()}",
        "source_outcome_fingerprint": source_outcome["outcome_fingerprint"],
        "phase": phase,
        "data_envelope_keys": list(envelope_fixture["envelopes"]),
        "output_binding_ids": [
            output["binding"]["binding_id"] for output in source_outcome["outputs"]
        ],
        "request_fingerprint": "0" * 64,
    }
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    return request


def replay_lineage(
    source_outcome: dict[str, Any], run_id: str, phase: str
) -> list[Any]:
    plan = source_outcome["effective_plan"]
    record_ids = {
        node_id: f"lineage:replay.{phase.lower()}.{index}"
        for index, node_id in enumerate(sorted(plan["node_plans"]))
    }
    records = []
    for node_id, node_plan in plan["node_plans"].items():
        records.append(
            {
                "record_id": record_ids[node_id],
                "run_id": run_id,
                "node_id": node_id,
                "phase": phase,
                "controller_id": node_plan["controller_id"],
                "controller_version": node_plan["controller_version"],
                "variant_id": source_outcome["selected_variant_id"],
                "fold_id": None,
                "branch_path": [],
                "input_lineage": sorted(
                    record_ids[input_node] for input_node in node_plan["input_nodes"]
                ),
                "artifact_refs": [],
                "params_fingerprint": node_plan["params_fingerprint"],
                "data_model_shape_fingerprint": None,
                "aggregation_policy_fingerprint": None,
                "seed": None,
                "unsafe_flags": [],
                "metrics": {},
            }
        )
    records.sort(key=lambda record: record["record_id"])
    return records


def port_explicit_prediction_block_v2(
    block: dict[str, Any], producer_port: str, *, kind: str
) -> dict[str, Any]:
    """Normalize V2 block field order with producer_port after producer_node.

    This is the stable-json cache fingerprint preimage that D5a-R must mirror
    in Rust field order; it is deliberately independent of the V1 normalizer.
    """

    orders = {
        "prediction": (
            "prediction_id",
            "producer_node",
            "producer_port",
            "partition",
            "fold_id",
            "sample_ids",
            "values",
            "target_names",
        ),
        "observation": (
            "prediction_id",
            "producer_node",
            "producer_port",
            "partition",
            "fold_id",
            "observation_ids",
            "values",
            "weights",
            "target_names",
        ),
        "aggregated": (
            "prediction_id",
            "producer_node",
            "producer_port",
            "partition",
            "fold_id",
            "level",
            "unit_ids",
            "values",
            "target_names",
        ),
    }
    if kind not in orders:
        raise ValueError(f"unknown V2 prediction block kind: {kind}")
    source = copy.deepcopy(block)
    source["producer_port"] = producer_port
    normalized: dict[str, Any] = {}
    for field in orders[kind]:
        if field in source:
            normalized[field] = source[field]
    return normalized


def replay_bound_output(*, phase: str) -> dict[str, Any]:
    source = load_json(
        ROOT
        / "examples"
        / "fixtures"
        / "estimator"
        / f"replay_outcome_{phase.lower()}.v1.json"
    )["outputs"][0]
    output = copy.deepcopy(source)
    output["schema_version"] = 2
    for field, kind in (
        ("predictions", "prediction"),
        ("observation_predictions", "observation"),
        ("aggregated_predictions", "aggregated"),
    ):
        output[field] = [
            port_explicit_prediction_block_v2(
                block, output["binding"]["port_name"], kind=kind
            )
            for block in output[field]
        ]
    return output


def replay_explanations() -> list[dict[str, Any]]:
    explanations = copy.deepcopy(
        load_json(
            ROOT
            / "examples"
            / "fixtures"
            / "estimator"
            / "replay_outcome_explain.v1.json"
        )["explanations"]
    )
    for explanation in explanations:
        explanation["producer_port"] = "oof"
    explanations = [
        {
            "producer_node": explanation["producer_node"],
            "producer_port": explanation["producer_port"],
            **{
                key: value
                for key, value in explanation.items()
                if key not in {"producer_node", "producer_port"}
            },
        }
        for explanation in explanations
    ]
    return explanations


def replay_outcome(
    source_outcome: dict[str, Any],
    request: dict[str, Any],
    envelope_fixture: dict[str, Any],
    *,
    outputs: list[dict[str, Any]],
    explanations: list[dict[str, Any]],
) -> dict[str, Any]:
    phase = request["phase"]
    run_id = f"run:replay.{phase.lower()}"
    lineage = replay_lineage(source_outcome, run_id, phase)
    outcome = {
        "schema_version": 1,
        "outcome_id": f"replay:outcome.{phase.lower()}",
        "run_id": run_id,
        "source_training_outcome": training_outcome_ref(source_outcome),
        "replay_request_id": request["request_id"],
        "replay_request_fingerprint": request["request_fingerprint"],
        "input_data_identities": replay_data_identities(envelope_fixture),
        "bundle_id": source_outcome["execution_bundle"]["bundle_id"],
        "plan_id": source_outcome["effective_plan"]["id"],
        "phase": phase,
        "result_count": len(lineage),
        "lineage_record_count": len(lineage),
        "prediction_block_count": sum(len(output["predictions"]) for output in outputs),
        "observation_prediction_block_count": sum(
            len(output["observation_predictions"]) for output in outputs
        ),
        "aggregated_prediction_block_count": sum(
            len(output["aggregated_predictions"]) for output in outputs
        ),
        "explanation_block_count": len(explanations),
        "controller_count": len({record["controller_id"] for record in lineage}),
        "prediction_cache_store": False,
        "outputs": sorted(outputs, key=lambda output: output["binding"]["binding_id"]),
        "explanations": explanations,
        "lineage": lineage,
        "warnings": [],
        "diagnostics": {"contract_fixture": True},
        "outcome_fingerprint": "0" * 64,
    }
    outcome["outcome_fingerprint"] = fingerprint_without(outcome, "outcome_fingerprint")
    return outcome


def replay_explanation_only_outcome(source: dict[str, Any]) -> dict[str, Any]:
    """Prove EXPLAIN may omit point-prediction outputs."""

    outcome = copy.deepcopy(source)
    outcome["outputs"] = []
    outcome["prediction_block_count"] = 0
    outcome["observation_prediction_block_count"] = 0
    outcome["aggregated_prediction_block_count"] = 0
    outcome["outcome_id"] = "replay:outcome.explain-only"
    outcome["outcome_fingerprint"] = fingerprint_without(outcome, "outcome_fingerprint")
    return outcome


def replay_class_probability_output(source_outcome: dict[str, Any]) -> dict[str, Any]:
    """Positive two-target classification block for segmented simplex semantics."""

    output = copy.deepcopy(
        load_json(
            ROOT
            / "examples"
            / "fixtures"
            / "estimator"
            / "replay_outcome_class_probability.v1.json"
        )["outputs"][0]
    )
    output["schema_version"] = 2
    binding = output["binding"]
    binding["target_names"] = ["class_primary", "class_secondary"]
    binding["target_units"] = [None, None]
    binding["class_labels"] = [["A", "B"], ["low", "mid", "high"]]
    binding["aggregation_fingerprint"] = serde_json_sha256(
        _normalize_campaign_spec(source_outcome["effective_plan"]["campaign"])[
            "aggregation_policy"
        ]
    )
    columns = [
        "class_primary:A",
        "class_primary:B",
        "class_secondary:low",
        "class_secondary:mid",
        "class_secondary:high",
    ]
    for field, kind in (
        ("predictions", "prediction"),
        ("observation_predictions", "observation"),
        ("aggregated_predictions", "aggregated"),
    ):
        output[field] = [
            port_explicit_prediction_block_v2(block, binding["port_name"], kind=kind)
            for block in output[field]
        ]
        for block in output[field]:
            block["target_names"] = columns
            block["values"] = [
                [0.8, 0.2, 0.1, 0.2, 0.7],
                [0.3, 0.7, 0.6, 0.1, 0.3],
            ]
    binding["binding_fingerprint"] = fingerprint_without(binding, "binding_fingerprint")
    return output


def build_d4_replay_negative_cases(
    source_outcome: dict[str, Any],
    predict_request: dict[str, Any],
    predict_outcome: dict[str, Any],
    explain_outcome: dict[str, Any],
    envelope_fixture: dict[str, Any],
    class_probability_output: dict[str, Any],
    class_label_output: dict[str, Any],
    observation_output: dict[str, Any],
    score_set_v2: dict[str, Any],
) -> list[dict[str, Any]]:
    """Re-fingerprinted D4 request/outcome adversarial contracts."""

    cases: list[dict[str, Any]] = []

    def request_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(predict_request)
        mutate(document)
        document["request_fingerprint"] = fingerprint_without(
            document, "request_fingerprint"
        )
        cases.append(
            {
                "id": case_id,
                "contract": "training_replay_request",
                "document": document,
                "expected_error": expected,
            }
        )

    def outcome_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(predict_outcome)
        mutate(document)
        document["outcome_fingerprint"] = fingerprint_without(
            document, "outcome_fingerprint"
        )
        cases.append(
            {
                "id": case_id,
                "contract": "training_replay_outcome",
                "document": document,
                "expected_error": expected,
            }
        )

    def envelope_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(envelope_fixture)
        mutate(document)
        cases.append(
            {
                "id": case_id,
                "contract": "training_replay_envelopes",
                "document": document,
                "expected_error": expected,
            }
        )

    def bound_output_case(
        case_id: str,
        expected: str,
        source: dict[str, Any],
        mutate,
    ) -> None:
        document = copy.deepcopy(source)
        mutate(document)
        cases.append(
            {
                "id": case_id,
                "contract": "training_replay_bound_output",
                "document": document,
                "expected_error": expected,
            }
        )

    def relation_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(
            next(iter(envelope_fixture["envelopes"].values()))["coordinator_relations"]
        )
        mutate(document)
        cases.append(
            {
                "id": case_id,
                "contract": "training_replay_relations",
                "document": document,
                "expected_error": expected,
            }
        )

    request_case(
        "d4_replay_request_refit_is_not_public",
        "PREDICT",
        lambda document: document.__setitem__("phase", "REFIT"),
    )
    request_case(
        "d4_replay_request_unknown_phase",
        "PREDICT or EXPLAIN",
        lambda document: document.__setitem__("phase", "SCORE"),
    )
    request_case(
        "d4_replay_request_unsorted_envelope_keys",
        "sorted",
        lambda document: document["data_envelope_keys"].reverse(),
    )
    request_case(
        "d4_replay_request_source_transplant",
        "does not bind",
        lambda document: document.__setitem__(
            "source_outcome_fingerprint", opaque("foreign-training-outcome")
        ),
    )
    request_case(
        "d4_replay_request_empty_output_binding_ids",
        "non-empty",
        lambda document: document.__setitem__("output_binding_ids", []),
    )
    request_case(
        "d4_replay_request_duplicate_output_binding_ids",
        "unique",
        lambda document: document.__setitem__(
            "output_binding_ids",
            [
                document["output_binding_ids"][0],
                document["output_binding_ids"][0],
            ],
        ),
    )
    request_case(
        "d4_replay_request_unsorted_output_binding_ids",
        "sorted",
        lambda document: document.__setitem__(
            "output_binding_ids",
            [document["output_binding_ids"][0], "output:aaa"],
        ),
    )
    request_case(
        "d4_replay_request_unknown_output_binding_id",
        "unknown source output",
        lambda document: document.__setitem__("output_binding_ids", ["output:unknown"]),
    )

    outcome_case(
        "d4_replay_outcome_source_ref_transplant",
        "complete source",
        lambda document: document["source_training_outcome"].__setitem__(
            "outcome_fingerprint", opaque("foreign-training-outcome")
        ),
    )
    outcome_case(
        "d4_replay_outcome_request_transplant",
        "ReplayRequest",
        lambda document: document.__setitem__(
            "replay_request_fingerprint", opaque("foreign-replay-request")
        ),
    )

    def drift_schema(document: dict[str, Any]) -> None:
        identity = document["input_data_identities"][0]
        identity["schema_fingerprint"] = opaque("foreign-schema")
        identity["identity_fingerprint"] = fingerprint_without(
            identity, "identity_fingerprint"
        )

    outcome_case(
        "d4_replay_outcome_schema_rebinding_forbidden",
        "does not attest current identity",
        drift_schema,
    )
    outcome_case(
        "d4_replay_outcome_cache_store_forbidden",
        "prediction_cache_store must be false",
        lambda document: document.__setitem__("prediction_cache_store", True),
    )

    def unknown_port(document: dict[str, Any]) -> None:
        document["outputs"][0]["predictions"][0]["producer_port"] = "unknown"

    outcome_case(
        "d4_replay_outcome_unknown_producer_port",
        "producer_port",
        unknown_port,
    )
    outcome_case(
        "d4_replay_outcome_missing_v2_producer_port",
        "required in v2",
        lambda document: document["outputs"][0]["predictions"][0].pop("producer_port"),
    )

    def legacy_public_output(document: dict[str, Any]) -> None:
        document["outputs"] = [
            copy.deepcopy(
                load_json(
                    ROOT / "examples/fixtures/estimator/replay_outcome_predict.v1.json"
                )["outputs"][0]
            )
        ]

    outcome_case(
        "d4_replay_outcome_rejects_legacy_bound_output",
        "schema_version must be 2",
        legacy_public_output,
    )

    def drift_aggregation(document: dict[str, Any]) -> None:
        binding = document["outputs"][0]["binding"]
        binding["aggregation_fingerprint"] = opaque("foreign-aggregation")
        binding["binding_fingerprint"] = fingerprint_without(
            binding, "binding_fingerprint"
        )

    outcome_case(
        "d4_replay_outcome_aggregation_transplant",
        "aggregation_fingerprint",
        drift_aggregation,
    )

    outcome_case(
        "d4_replay_outcome_non_final_partition",
        "final partition",
        lambda document: document["outputs"][0]["predictions"][0].__setitem__(
            "partition", "test"
        ),
    )
    outcome_case(
        "d4_replay_outcome_non_null_fold",
        "null fold",
        lambda document: document["outputs"][0]["predictions"][0].__setitem__(
            "fold_id", "fold:0"
        ),
    )

    def duplicate_unit(document: dict[str, Any]) -> None:
        block = document["outputs"][0]["aggregated_predictions"][0]
        block["unit_ids"].append(copy.deepcopy(block["unit_ids"][0]))
        block["values"].append(copy.deepcopy(block["values"][0]))

    outcome_case(
        "d4_replay_outcome_duplicate_aggregated_unit",
        "identifiers",
        duplicate_unit,
    )

    def duplicate_sample_across_blocks(document: dict[str, Any]) -> None:
        block = copy.deepcopy(document["outputs"][0]["predictions"][0])
        block["prediction_id"] = "prediction:duplicate-final-units"
        document["outputs"][0]["predictions"].append(block)
        document["prediction_block_count"] += 1

    outcome_case(
        "d4_replay_outcome_duplicate_sample_across_blocks",
        "duplicate final unit",
        duplicate_sample_across_blocks,
    )

    def alien_sample(document: dict[str, Any]) -> None:
        document["outputs"][0]["predictions"][0]["sample_ids"][0] = "sample:alien"

    outcome_case(
        "d4_replay_outcome_alien_sample",
        "outside current cohort",
        alien_sample,
    )
    outcome_case(
        "d4_replay_outcome_wrong_lineage_controller",
        "controller",
        lambda document: document["lineage"][0].__setitem__(
            "controller_id", "controller:foreign"
        ),
    )
    outcome_case(
        "d4_replay_outcome_invalid_data_shape_fingerprint",
        "data_model_shape_fingerprint",
        lambda document: document["lineage"][0].__setitem__(
            "data_model_shape_fingerprint", "not-a-sha256"
        ),
    )

    def lineage_plugin_version_without_plugin(document: dict[str, Any]) -> None:
        record = document["lineage"][0]
        record["artifact_refs"] = [
            {
                "id": "artifact:replay.invalid",
                "kind": "model",
                "controller_id": record["controller_id"],
                "backend": None,
                "uri": None,
                "content_fingerprint": None,
                "size_bytes": 0,
                "plugin": None,
                "plugin_version": "1.0.0",
            }
        ]

    outcome_case(
        "d4_replay_outcome_lineage_plugin_version_requires_plugin",
        "plugin_version requires plugin",
        lineage_plugin_version_without_plugin,
    )
    outcome_case(
        "d4_replay_outcome_empty_lineage_metric_key",
        "metrics key",
        lambda document: document["lineage"][0].__setitem__("metrics", {"": 1.0}),
    )
    outcome_case(
        "d4_replay_outcome_observation_count_drift",
        "observation_prediction_block_count",
        lambda document: document.__setitem__("observation_prediction_block_count", 1),
    )
    outcome_case(
        "d4_replay_outcome_result_count_drift",
        "result_count",
        lambda document: document.__setitem__(
            "result_count", document["result_count"] + 1
        ),
    )
    outcome_case(
        "d4_replay_outcome_lineage_count_drift",
        "match payload",
        lambda document: document.__setitem__(
            "lineage_record_count", document["lineage_record_count"] + 1
        ),
    )
    outcome_case(
        "d4_replay_outcome_prediction_count_drift",
        "prediction_block_count",
        lambda document: document.__setitem__(
            "prediction_block_count", document["prediction_block_count"] + 1
        ),
    )
    outcome_case(
        "d4_replay_outcome_controller_count_drift",
        "controller_count",
        lambda document: document.__setitem__(
            "controller_count", document["controller_count"] + 1
        ),
    )

    def no_output(document: dict[str, Any]) -> None:
        document["outputs"] = []
        document["prediction_block_count"] = 0
        document["observation_prediction_block_count"] = 0
        document["aggregated_prediction_block_count"] = 0

    outcome_case(
        "d4_replay_outcome_requested_binding_absent",
        "at least one output",
        no_output,
    )

    def duplicate_output(document: dict[str, Any]) -> None:
        document["outputs"].append(copy.deepcopy(document["outputs"][0]))
        document["prediction_block_count"] *= 2
        document["observation_prediction_block_count"] *= 2
        document["aggregated_prediction_block_count"] *= 2

    outcome_case(
        "d4_replay_outcome_duplicate_binding",
        "sorted",
        duplicate_output,
    )

    def unsorted_outputs(document: dict[str, Any]) -> None:
        second = copy.deepcopy(document["outputs"][0])
        second["binding"]["binding_id"] = "output:aaa"
        second["binding"]["binding_fingerprint"] = fingerprint_without(
            second["binding"], "binding_fingerprint"
        )
        document["outputs"].append(second)
        document["prediction_block_count"] *= 2
        document["observation_prediction_block_count"] *= 2
        document["aggregated_prediction_block_count"] *= 2

    outcome_case(
        "d4_replay_outcome_unsorted_bindings",
        "sorted",
        unsorted_outputs,
    )
    outcome_case(
        "d4_replay_outcome_unsorted_warnings",
        "warnings",
        lambda document: document.__setitem__("warnings", ["z-warning", "a-warning"]),
    )
    outcome_case(
        "d4_replay_outcome_blank_warning",
        "non-blank",
        lambda document: document.__setitem__("warnings", ["   "]),
    )
    outcome_case(
        "d4_replay_outcome_nested_diagnostics",
        "finite JSON scalar",
        lambda document: document.__setitem__(
            "diagnostics", {"nested": {"forbidden": True}}
        ),
    )

    invalid_explanation = copy.deepcopy(explain_outcome)
    invalid_explanation["explanations"][0]["target_name"] = "foreign-target"
    invalid_explanation["outcome_fingerprint"] = fingerprint_without(
        invalid_explanation, "outcome_fingerprint"
    )
    cases.append(
        {
            "id": "d4_replay_explanation_unknown_target",
            "contract": "training_replay_explain_outcome",
            "document": invalid_explanation,
            "expected_error": "absent from OutputBinding",
        }
    )
    blank_method = copy.deepcopy(explain_outcome)
    blank_method["explanations"][0]["method"] = "   "
    blank_method["outcome_fingerprint"] = fingerprint_without(
        blank_method, "outcome_fingerprint"
    )
    cases.append(
        {
            "id": "d4_replay_explanation_blank_method",
            "contract": "training_replay_explain_outcome",
            "document": blank_method,
            "expected_error": "non-blank",
        }
    )
    raw_feature_payload = copy.deepcopy(explain_outcome)
    raw_feature_payload["explanations"][0]["payload"] = {"raw_features": [[1.0, 2.0]]}
    raw_feature_payload["outcome_fingerprint"] = fingerprint_without(
        raw_feature_payload, "outcome_fingerprint"
    )
    cases.append(
        {
            "id": "d4_replay_explanation_raw_feature_payload",
            "contract": "training_replay_explain_outcome",
            "document": raw_feature_payload,
            "expected_error": "raw feature data",
        }
    )

    envelope_case(
        "d4_replay_envelopes_missing_key",
        "exactly cover",
        lambda document: document["envelopes"].pop(next(iter(document["envelopes"]))),
    )

    def add_extra_envelope(document: dict[str, Any]) -> None:
        document["envelopes"]["zz.extra"] = copy.deepcopy(
            next(iter(document["envelopes"].values()))
        )

    envelope_case(
        "d4_replay_envelopes_extra_key",
        "exactly cover",
        add_extra_envelope,
    )

    def stale_relation(document: dict[str, Any]) -> None:
        envelope = next(iter(document["envelopes"].values()))
        envelope["relation_fingerprint"] = opaque("stale-relation")

    envelope_case(
        "d4_replay_envelopes_stale_relation",
        "does not match",
        stale_relation,
    )

    def rebind_plan(document: dict[str, Any]) -> None:
        envelope = next(iter(document["envelopes"].values()))
        envelope["plan"]["output_representation"] = "tensor_2d"
        envelope["plan_fingerprint"] = serde_json_sha256(envelope["plan"])

    envelope_case(
        "d4_replay_envelopes_plan_rebinding",
        "schema/plan rebinding",
        rebind_plan,
    )

    envelope_case(
        "d4_replay_envelopes_feature_set_rebinding",
        "feature_set_id changed",
        lambda document: next(iter(document["envelopes"].values()))[
            "metadata"
        ].__setitem__("feature_set_id", "foreign-feature-set"),
    )
    envelope_case(
        "d4_replay_envelopes_missing_feature_set_id",
        "feature_set_id changed or is missing",
        lambda document: next(iter(document["envelopes"].values()))["metadata"].pop(
            "feature_set_id"
        ),
    )

    def conflicting_target(document: dict[str, Any]) -> None:
        document["records"][1]["sample_id"] = document["records"][0]["sample_id"]
        document["records"][1]["target_id"] = "target:conflict"

    relation_case(
        "d4_replay_relations_sample_multiple_targets",
        "multiple targets",
        conflicting_target,
    )

    def self_combo(document: dict[str, Any]) -> None:
        record = document["records"][0]
        record["unit_level"] = "combo"
        record["derived_unit_id"] = "combo:self"
        record["component_observation_ids"] = [record["observation_id"]]

    relation_case(
        "d4_replay_relations_combo_self_component",
        "cannot list itself",
        self_combo,
    )
    relation_case(
        "d4_replay_relations_invalid_identifier",
        "identifier",
        lambda document: document["records"][0].__setitem__(
            "observation_id", "invalid identifier"
        ),
    )

    def probability_range(document: dict[str, Any]) -> None:
        document["predictions"][0]["values"][0][:2] = [-0.1, 1.1]

    bound_output_case(
        "d4_replay_probability_outside_unit_interval",
        "outside [0,1]",
        class_probability_output,
        probability_range,
    )

    def probability_simplex(document: dict[str, Any]) -> None:
        document["predictions"][0]["values"][0][:2] = [0.7, 0.2]

    bound_output_case(
        "d4_replay_probability_not_on_simplex",
        "simplex",
        class_probability_output,
        probability_simplex,
    )
    bound_output_case(
        "d4_replay_class_label_negative_index",
        "zero-based vocabulary index",
        class_label_output,
        lambda document: document["predictions"][0]["values"][0].__setitem__(0, -1),
    )
    bound_output_case(
        "d4_replay_class_label_out_of_vocabulary",
        "zero-based vocabulary index",
        class_label_output,
        lambda document: document["predictions"][0]["values"][0].__setitem__(1, 3),
    )
    bound_output_case(
        "d4_replay_class_label_fractional_index",
        "zero-based vocabulary index",
        class_label_output,
        lambda document: document["predictions"][0]["values"][0].__setitem__(0, 1.5),
    )

    def replace_observations_with_samples(document: dict[str, Any]) -> None:
        observation = document["observation_predictions"].pop()
        document["predictions"] = [
            {
                "prediction_id": observation["prediction_id"],
                "producer_node": observation["producer_node"],
                "producer_port": observation["producer_port"],
                "partition": observation["partition"],
                "fold_id": observation["fold_id"],
                "sample_ids": observation["observation_ids"],
                "values": observation["values"],
                "target_names": observation["target_names"],
            }
        ]

    bound_output_case(
        "d4_replay_observation_binding_requires_observation_predictions",
        "observation binding requires observation predictions",
        observation_output,
        replace_observations_with_samples,
    )

    legacy_output = copy.deepcopy(
        load_json(ROOT / "examples/fixtures/estimator/replay_outcome_predict.v1.json")[
            "outputs"
        ][0]
    )
    ambiguous_plan = copy.deepcopy(source_outcome["effective_plan"])
    producer = legacy_output["binding"]["node_id"]
    producer_node = next(
        node
        for node in ambiguous_plan["graph_plan"]["graph"]["nodes"]
        if node["id"] == producer
    )
    producer_node["ports"]["outputs"].append(
        {
            "name": "alternate_prediction",
            "kind": "prediction",
            "representation": None,
            "cardinality": "one",
            "description": "D4 ambiguity probe",
        }
    )
    cases.append(
        {
            "id": "d4_replay_legacy_port_omission_ambiguous",
            "contract": "training_replay_legacy_bound_output",
            "document": {"output": legacy_output, "effective_plan": ambiguous_plan},
            "expected_error": "exactly one prediction port",
        }
    )

    zero_port_plan = copy.deepcopy(source_outcome["effective_plan"])
    zero_port_node = next(
        node
        for node in zero_port_plan["graph_plan"]["graph"]["nodes"]
        if node["id"] == producer
    )
    zero_port_node["ports"]["outputs"] = []
    cases.append(
        {
            "id": "d4_replay_legacy_port_omission_zero_prediction_ports",
            "contract": "training_replay_port_resolution",
            "document": {
                "effective_plan": zero_port_plan,
                "producer_node": producer,
                "producer_port": None,
            },
            "expected_error": "exactly one prediction port",
        }
    )

    legacy_with_port = copy.deepcopy(legacy_output)
    for field in ("predictions", "observation_predictions", "aggregated_predictions"):
        for block in legacy_with_port[field]:
            block["producer_port"] = legacy_with_port["binding"]["port_name"]
    cases.append(
        {
            "id": "d4_replay_legacy_wrapper_rejects_explicit_port",
            "contract": "training_replay_legacy_bound_output",
            "document": {
                "output": legacy_with_port,
                "effective_plan": copy.deepcopy(source_outcome["effective_plan"]),
            },
            "expected_error": "legacy bound output",
        }
    )

    duplicate_score_set = copy.deepcopy(score_set_v2)
    duplicate_report = copy.deepcopy(duplicate_score_set["reports"][0])
    duplicate_report["prediction_id"] = "prediction:duplicate-score-key"
    duplicate_score_set["reports"].append(duplicate_report)
    cases.append(
        {
            "id": "d4_score_set_v2_duplicate_port_aware_key",
            "contract": "training_score_set_v2",
            "document": {
                "score_set": duplicate_score_set,
                "effective_plan": copy.deepcopy(source_outcome["effective_plan"]),
            },
            "expected_error": "duplicate score report",
        }
    )

    # Keep the source value live in this generator dependency: if the D2 source
    # fingerprint changes, every D4 authority and negative transplant is rebuilt.
    assert (
        predict_request["source_outcome_fingerprint"]
        == source_outcome["outcome_fingerprint"]
    )
    return cases


def replay_class_label_output(source_outcome: dict[str, Any]) -> dict[str, Any]:
    """Build a positive zero-based class-vocabulary index payload."""

    output = replay_class_probability_output(source_outcome)
    binding = output["binding"]
    binding["binding_id"] = "output:meta.class_label"
    binding["prediction_kind"] = "class_label"
    binding["output_order"] = "target_order"
    binding["binding_fingerprint"] = fingerprint_without(binding, "binding_fingerprint")
    for field in ("predictions", "aggregated_predictions"):
        for block in output[field]:
            block["prediction_id"] = block["prediction_id"].replace(
                "class_probability", "class_label"
            )
            block["values"] = [[0.0, 2.0], [1.0, 0.0]]
            block["target_names"] = ["class_primary", "class_secondary"]
    return output


def replay_observation_output() -> dict[str, Any]:
    """Build a positive port-explicit ObservationPredictionBlock payload."""

    output = replay_bound_output(phase="PREDICT")
    binding = output["binding"]
    binding["binding_id"] = "output:meta.observation"
    binding["prediction_level"] = "observation"
    binding["unit_level"] = "observation"
    binding["binding_fingerprint"] = fingerprint_without(binding, "binding_fingerprint")
    output["predictions"] = []
    output["aggregated_predictions"] = []
    output["observation_predictions"] = [
        port_explicit_prediction_block_v2(
            {
                "prediction_id": "prediction:replay.observation",
                "producer_node": binding["node_id"],
                "partition": "final",
                "fold_id": None,
                "observation_ids": ["obs:prod.1", "obs:prod.2"],
                "values": [[2.4], [3.1]],
                "weights": [1.0, 0.5],
                "target_names": ["protein"],
            },
            binding["port_name"],
            kind="observation",
        )
    ]
    return output


def replay_multi_port_outputs(source_outcome: dict[str, Any]) -> dict[str, Any]:
    """Build two explicit sibling prediction ports with distinct values."""

    plan = copy.deepcopy(source_outcome["effective_plan"])
    primary = replay_bound_output(phase="PREDICT")
    producer = primary["binding"]["node_id"]
    node = next(
        node for node in plan["graph_plan"]["graph"]["nodes"] if node["id"] == producer
    )
    node["ports"]["outputs"].append(
        {
            "name": "alternate_prediction",
            "kind": "prediction",
            "representation": None,
            "cardinality": "one",
            "description": "D5a-C explicit multi-port fixture",
        }
    )
    alternate = copy.deepcopy(primary)
    binding = alternate["binding"]
    binding["binding_id"] = "output:meta.alternate"
    binding["port_name"] = "alternate_prediction"
    binding["binding_fingerprint"] = fingerprint_without(binding, "binding_fingerprint")
    for field in ("predictions", "observation_predictions", "aggregated_predictions"):
        for block in alternate[field]:
            block["producer_port"] = "alternate_prediction"
            block["values"] = [
                [float(value) + 10.0 for value in row] for row in block["values"]
            ]
    return {
        "schema_version": 1,
        "effective_plan": plan,
        "outputs": sorted(
            [primary, alternate], key=lambda output: output["binding"]["binding_id"]
        ),
    }


def _single_prediction_port(plan: dict[str, Any], node_id: str) -> str:
    node = next(
        node for node in plan["graph_plan"]["graph"]["nodes"] if node["id"] == node_id
    )
    ports = [
        port["name"]
        for port in node["ports"]["outputs"]
        if port["kind"] == "prediction"
    ]
    if len(ports) != 1:
        raise ValueError(f"fixture node {node_id} has ambiguous prediction ports")
    return ports[0]


def score_set_port_explicit_v2(
    source: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Migrate ScoreSet reports with producer_port after producer_node."""

    score_set = copy.deepcopy(source)
    score_set["schema_version"] = 2
    reports: list[dict[str, Any]] = []
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
        migrated["producer_port"] = _single_prediction_port(
            plan, report["producer_node"]
        )
        reports.append({field: migrated[field] for field in order if field in migrated})
    score_set["reports"] = reports
    return score_set


def training_outcome_port_explicit_v2(
    source_outcome: dict[str, Any],
) -> dict[str, Any]:
    """Migrate one complete non-null-cache TrainingOutcome to the V2 closure."""

    outcome = copy.deepcopy(source_outcome)
    outcome["schema_version"] = 2
    outcome["outcome_id"] = "training:estimator.refit.port-explicit-v2"
    plan = outcome["effective_plan"]
    for output in outcome["outputs"]:
        output["schema_version"] = 2
        port = output["binding"]["port_name"]
        for field, kind in (
            ("predictions", "prediction"),
            ("observation_predictions", "observation"),
            ("aggregated_predictions", "aggregated"),
        ):
            output[field] = [
                port_explicit_prediction_block_v2(block, port, kind=kind)
                for block in output[field]
            ]

    bundle = outcome["execution_bundle"]
    bundle["schema_version"] = 2
    outcome["score_set"] = score_set_port_explicit_v2(outcome["score_set"], plan)
    bundle["scores"] = copy.deepcopy(outcome["score_set"])
    payload_set = outcome["portable_prediction_caches"]
    if payload_set is None or not payload_set["caches"]:
        raise ValueError("V2 migration fixture requires non-null portable caches")
    payload_set["schema_version"] = 2
    records = {
        record["requirement_key"]: record for record in bundle["prediction_caches"]
    }
    for payload in payload_set["caches"]:
        payload["format"] = "dag-ml-json-prediction-blocks-v2"
        blocks = payload.get("blocks", [])
        aggregated = payload.get("aggregated_blocks", [])
        payload["blocks"] = [
            port_explicit_prediction_block_v2(
                block,
                _single_prediction_port(plan, block["producer_node"]),
                kind="prediction",
            )
            for block in blocks
        ]
        migrated_aggregated = [
            port_explicit_prediction_block_v2(
                block,
                _single_prediction_port(plan, block["producer_node"]),
                kind="aggregated",
            )
            for block in aggregated
        ]
        blocks = payload["blocks"]
        if "aggregated_blocks" in payload:
            payload["aggregated_blocks"] = migrated_aggregated
        aggregated = migrated_aggregated
        payload["content_fingerprint"] = serde_json_sha256([*blocks, *aggregated])
        record = records[payload["requirement_key"]]
        record["format"] = "dag-ml-json-prediction-blocks-v2"
        record["content_fingerprint"] = payload["content_fingerprint"]
        portable_by_fold = {block["fold_id"]: block for block in [*blocks, *aggregated]}
        namespace_fingerprints = payload.get("cache_namespace_fingerprints")
        if not namespace_fingerprints:
            namespace_fingerprints = [
                tcv1_sha256(
                    {
                        "schema_version": 1,
                        "kind": "training-replay-v2-fixture-cache-namespace",
                        "requirement_key": payload["requirement_key"],
                        "cache_id": payload["cache_id"],
                        "fold_id": block["fold_id"],
                        "block_index": index,
                    }
                )
                for index, block in enumerate([*blocks, *aggregated])
            ]
        payload["cache_namespace_fingerprints"] = namespace_fingerprints
        record["cache_namespace_fingerprints"] = copy.deepcopy(namespace_fingerprints)
        for block_record in record["blocks"]:
            block_record["content_fingerprint"] = serde_json_sha256(
                portable_by_fold[block_record["fold_id"]]
            )
    outcome["outcome_fingerprint"] = fingerprint_without(outcome, "outcome_fingerprint")
    return outcome


def port_explicit_protocols_v2() -> dict[str, Any]:
    """Positive root instances for the remaining V2 protocol family."""

    node_result = load_json(
        ROOT / "examples/fixtures/runtime/node_result_transform_scale.json"
    )
    node_result = {"schema_version": 2, **node_result}
    process_result = {
        "type": "result",
        "schema_version": 2,
        "result": copy.deepcopy(node_result),
    }
    relations = {
        "records": [
            {
                "observation_id": "obs:1",
                "sample_id": "sample:1",
                "target_id": "target:1",
                "group_id": "group:1",
                "origin_sample_id": None,
                "source_id": "nir",
                "is_augmented": False,
            },
            {
                "observation_id": "obs:2",
                "sample_id": "sample:2",
                "target_id": "target:2",
                "group_id": "group:1",
                "origin_sample_id": None,
                "source_id": "nir",
                "is_augmented": False,
            },
        ]
    }
    policy = {
        "method": "custom_controller",
        "custom_controller": {"controller_id": "controller:aggregation.fixture"},
    }
    observation_block = port_explicit_prediction_block_v2(
        {
            "prediction_id": "prediction:observation.fixture",
            "producer_node": "model:fixture",
            "partition": "final",
            "fold_id": None,
            "observation_ids": ["obs:1", "obs:2"],
            "values": [[1.0], [2.0]],
            "weights": [1.0, 1.0],
            "target_names": ["y"],
        },
        "prediction",
        kind="observation",
    )
    sample_block = port_explicit_prediction_block_v2(
        {
            "prediction_id": "prediction:sample.fixture",
            "producer_node": "model:fixture",
            "partition": "final",
            "fold_id": None,
            "sample_ids": ["sample:1", "sample:2"],
            "values": [[1.0], [2.0]],
            "target_names": ["y"],
        },
        "prediction",
        kind="prediction",
    )
    unit_block = port_explicit_prediction_block_v2(
        {
            "prediction_id": "prediction:group.fixture",
            "producer_node": "model:fixture",
            "partition": "final",
            "fold_id": None,
            "level": "group",
            "unit_ids": [{"level": "group", "id": "group:1"}],
            "values": [[1.5]],
            "target_names": ["y"],
        },
        "prediction",
        kind="aggregated",
    )
    return {
        "schema_version": 1,
        "node_result": node_result,
        "process_adapter_result": process_result,
        "aggregation_task_observation": {
            "schema_version": 2,
            "task_id": "task:aggregate.observation",
            "controller_id": "controller:aggregation.fixture",
            "policy": policy,
            "input": {
                "input_kind": "observation_to_sample",
                "block": observation_block,
                "relations": relations,
                "requested_sample_order": ["sample:1", "sample:2"],
            },
        },
        "aggregation_task_unit": {
            "schema_version": 2,
            "task_id": "task:aggregate.unit",
            "controller_id": "controller:aggregation.fixture",
            "policy": policy,
            "input": {
                "input_kind": "sample_to_unit",
                "block": sample_block,
                "relations": relations,
                "requested_unit_order": [{"level": "group", "id": "group:1"}],
            },
        },
        "aggregation_result_sample": {
            "schema_version": 2,
            "task_id": "task:aggregate.observation",
            "output": {"output_kind": "sample", "block": sample_block},
        },
        "aggregation_result_unit": {
            "schema_version": 2,
            "task_id": "task:aggregate.unit",
            "output": {"output_kind": "unit", "block": unit_block},
        },
    }


def generate(output_dir: Path = OUT) -> None:
    source = load_json(TRAINING_FIXTURES / "training_outcome_refit.v1.json")
    explain_source = replay_source_outcome_explain(source)
    envelopes = replay_input_envelopes(source)
    predict_request = replay_request(source, envelopes, phase="PREDICT")
    explain_request = replay_request(explain_source, envelopes, phase="EXPLAIN")
    predict_outcome = replay_outcome(
        source,
        predict_request,
        envelopes,
        outputs=[replay_bound_output(phase="PREDICT")],
        explanations=[],
    )
    explain_outcome = replay_outcome(
        explain_source,
        explain_request,
        envelopes,
        outputs=[replay_bound_output(phase="EXPLAIN")],
        explanations=replay_explanations(),
    )
    explain_only_outcome = replay_explanation_only_outcome(explain_outcome)
    class_probability = replay_class_probability_output(source)
    class_label = replay_class_label_output(source)
    observation_output = replay_observation_output()
    multi_port_outputs = replay_multi_port_outputs(source)
    outcome_v2 = training_outcome_port_explicit_v2(source)
    protocols_v2 = port_explicit_protocols_v2()
    negatives = build_d4_replay_negative_cases(
        source,
        predict_request,
        predict_outcome,
        explain_outcome,
        envelopes,
        class_probability,
        class_label,
        observation_output,
        outcome_v2["score_set"],
    )

    documents = {
        "training_replay_source_outcome_explain.v1.json": explain_source,
        "training_replay_input_envelopes.v1.json": envelopes,
        "training_replay_multi_port_outputs.v1.json": multi_port_outputs,
        "training_replay_request_predict.v1.json": predict_request,
        "training_replay_request_explain.v1.json": explain_request,
        "training_replay_outcome_predict.v1.json": predict_outcome,
        "training_replay_outcome_explain.v1.json": explain_outcome,
        "training_replay_outcome_explain_only.v1.json": explain_only_outcome,
        "training_replay_output_class_probability.v1.json": class_probability,
        "training_replay_output_class_label.v1.json": class_label,
        "training_replay_output_observation.v1.json": observation_output,
        "training_outcome_port_explicit.v2.json": outcome_v2,
        "training_port_explicit_protocols.v2.json": protocols_v2,
        "training_replay_negative_cases.v1.json": {
            "schema_version": 1,
            "cases": negatives,
        },
    }
    for name, document in documents.items():
        write_json(output_dir / name, document)


def _base_pack() -> dict[str, Any]:
    pack = load_json(BASE_PACK_PATH)
    if pack.get("pack_id") != BASE_PACK_ID:
        raise ValueError("unexpected base training pack id")
    if pack.get("pack_checksum") != BASE_PACK_CHECKSUM:
        raise ValueError("unexpected base training pack checksum")
    if file_sha256(BASE_PACK_PATH) != BASE_PACK_SHA256:
        raise ValueError("base training pack bytes changed")
    for artifact in pack.get("artifacts", []):
        if artifact_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"base training artifact changed: {artifact['path']}")
    for relative_path, expected_sha256 in LEGACY_AUTHORITY_SHA256.items():
        if artifact_sha256(relative_path) != expected_sha256:
            raise ValueError(f"legacy replay authority changed: {relative_path}")
    return pack


def replay_pack_artifacts() -> dict[str, str]:
    base = _base_pack()
    artifacts = {
        artifact["path"]: f"base_{artifact['kind']}" for artifact in base["artifacts"]
    }
    artifacts["docs/contracts/training_contract_conformance_pack.v1.json"] = (
        "base_conformance_pack"
    )
    artifacts.update(D4_ARTIFACTS)
    return with_transitive_schema_dependencies(ROOT, artifacts)


def build_conformance_pack() -> dict[str, Any]:
    base = _base_pack()
    artifacts = replay_pack_artifacts()
    negatives = load_json(OUT / "training_replay_negative_cases.v1.json")
    pack = {
        "schema_version": 1,
        "pack_id": "dag-ml.training-replay-contracts.v1",
        "mode": "current",
        "base_pack_id": base["pack_id"],
        "base_pack_sha256": file_sha256(BASE_PACK_PATH),
        "base_pack_checksum": base["pack_checksum"],
        "base_pack_mode": "current",
        "hash_algorithm": "sha256-file-bytes",
        "canonical_profile": "DAG-ML TCV1",
        "artifacts": [
            {
                "path": relative_path,
                "sha256": artifact_sha256(relative_path),
                "kind": artifacts[relative_path],
            }
            for relative_path in sorted(artifacts)
        ],
        "positive_fixture_ids": sorted(
            [
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
                "training_outcome_port_explicit.v2",
                "training_port_explicit_protocols.v2",
            ]
        ),
        "negative_case_ids": [case["id"] for case in negatives["cases"]],
        "pack_checksum": "0" * 64,
    }
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    return pack


def generate_pack(pack_path: Path = PACK_PATH) -> None:
    write_json(pack_path, build_conformance_pack())


if __name__ == "__main__":
    generate()
    generate_pack()
