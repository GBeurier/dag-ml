#!/usr/bin/env python3
"""Deterministically regenerate the W0 conformal/robustness fixture graph.

This is a test-data generator, not production code.  It deliberately uses the
independent parity oracle for TCV1 so every nested identity is rebuilt from the
leaves upward and no hand-edited checksum can survive unnoticed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import (  # noqa: E402
    fingerprint_without,
    load_json,
    regression_conformal_metrics,
    tcv1_sha256,
)
from parity.schema_dependencies import (  # noqa: E402
    with_transitive_schema_dependencies,
)

CONFORMAL_DIR = ROOT / "examples" / "fixtures" / "conformal"
ROBUSTNESS_DIR = ROOT / "examples" / "fixtures" / "robustness"
CALIBRATION_FIXTURE = CONFORMAL_DIR / "split_absolute_residual_physical_sample.v1.json"
CALIBRATION_ARTIFACT_FIXTURE = CONFORMAL_DIR / "calibration_artifacts.v1.json"
PACK_PATH = (
    ROOT / "docs" / "contracts" / "conformal_robustness_conformance_pack.v1.json"
)

BASE_PACK_ARTIFACTS = {
    "docs/contracts/cohort_manifest.schema.json": "schema",
    "docs/contracts/conformal_calibration.schema.json": "schema",
    "docs/contracts/conformal_metric_set.schema.json": "schema",
    "docs/contracts/conformal_prediction_block.schema.json": "schema",
    "docs/contracts/decision_block.schema.json": "schema",
    "docs/contracts/domain_assessment_block.schema.json": "schema",
    "docs/contracts/output_binding.schema.json": "schema",
    "docs/contracts/parameter_patch.schema.json": "schema",
    "docs/contracts/robustness_report.schema.json": "schema",
    "docs/contracts/robustness_scenario_spec.schema.json": "schema",
    "docs/contracts/training_influence_manifest.schema.json": "schema",
    "docs/contracts/training_outcome.schema.json": "schema",
    "docs/contracts/training_request.schema.json": "schema_dependency",
    "examples/fixtures/conformal/cohort_manifest_roles.v1.json": "fixture",
    "examples/fixtures/conformal/calibration_artifacts.v1.json": "fixture",
    "examples/fixtures/conformal/conformal_metric_sets.v1.json": "fixture",
    "examples/fixtures/conformal/conformal_prediction_blocks.v1.json": "fixture",
    "examples/fixtures/conformal/decision_blocks.v1.json": "fixture",
    "examples/fixtures/conformal/domain_assessment_blocks.v1.json": "fixture",
    "examples/fixtures/conformal/split_absolute_residual_physical_sample.v1.json": "fixture",
    "examples/fixtures/estimator/output_binding_regression_final_refit.v1.json": "fixture",
    "examples/fixtures/estimator/parameter_patch_operator_alpha.v1.json": "fixture",
    "examples/fixtures/estimator/training_outcome_no_refit.v1.json": "fixture",
    "examples/fixtures/estimator/training_outcome_refit.v1.json": "fixture",
    "examples/fixtures/robustness/robustness_reports.v1.json": "fixture",
    "examples/fixtures/robustness/robustness_scenarios.v1.json": "fixture",
    "parity/canonical/README.md": "documentation",
    "parity/canonical/golden/tcv1_jcs_cross_language.v1.json": "golden",
    "parity/canonical/rust-oracle/.gitignore": "configuration",
    "parity/canonical/rust-oracle/Cargo.lock": "oracle_source",
    "parity/canonical/rust-oracle/Cargo.toml": "oracle_source",
    "parity/canonical/rust-oracle/src/lib.rs": "oracle_source",
    "parity/canonical/rust-oracle/src/main.rs": "oracle_source",
    "parity/canonical/tests/test_rust_oracle_parity.py": "test",
    "parity/conformal/generate_fixtures.py": "generator",
    "parity/conformal/golden/regression_conformal_metrics.v1.json": "golden",
    "parity/conformal/golden/split_absolute_residual.v1.json": "golden",
    "parity/conformal/oracle.py": "test_oracle",
    "parity/conformal/tests/test_conformal_robustness_contracts.py": "test",
    "parity/robustness_rng/golden/philox4x32_10_counter.v1.json": "golden",
    "parity/robustness_rng/oracle.py": "test_oracle",
    "parity/robustness_rng/tests/test_robustness_rng_contract.py": "test",
    "parity/schema_dependencies.py": "schema_dependency_resolver",
}

PACK_ARTIFACTS = with_transitive_schema_dependencies(ROOT, BASE_PACK_ARTIFACTS)


def opaque(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def set_fingerprint(document: dict[str, Any], field: str) -> None:
    document[field] = "0" * 64
    document[field] = fingerprint_without(document, field)


def write_json(path: Path, document: Any) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def enforce_invalid_fingerprint_policy(fixture: dict[str, Any]) -> dict[str, Any]:
    """Require every schema negative to preserve or intentionally target identity."""

    for case in fixture.get("invalid_cases", []):
        if case.get("targets_fingerprint") is True:
            if case.get("recompute_fingerprints") is not None:
                raise ValueError(
                    f"{case['id']} cannot target and recompute its fingerprint"
                )
        else:
            case["recompute_fingerprints"] = True
    return fixture


def derive_source_outcome_replayable_phases(outcome: dict[str, Any]) -> list[str]:
    """Derive a source training outcome's honest replayable phases from the graph.

    Independent mirror of the Rust ``derive_replayable_phases`` (and the training
    oracle): the predictor closure comes from real graph edges rooted at the
    output bindings, retained inference state is required only for a
    ``stateful``/``emits_artifacts`` node (never ``artifact_policy`` or
    ``fit_scope``), a completed refit exposes forward inference (PREDICT then
    EXPLAIN) while a skipped refit exposes exactly REFIT when every in-closure
    ``requires_oof`` edge is backed by a bundle requirement, a cache record and a
    portable payload. ``[]`` is a valid answer.
    """

    plan = outcome["effective_plan"]
    graph = plan["graph_plan"]["graph"]
    node_plans = plan["node_plans"]
    manifests = plan["controller_manifests"]
    incoming: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        incoming.setdefault(edge["target"]["node_id"], []).append(
            edge["source"]["node_id"]
        )
    pending = [output["binding"]["node_id"] for output in outcome["outputs"]]
    closure: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in closure:
            continue
        closure.add(node_id)
        pending.extend(incoming.get(node_id, []))

    def manifest_for(node_id: str) -> dict[str, Any]:
        return manifests[node_plans[node_id]["controller_id"]]

    def all_support(phase: str) -> bool:
        return all(phase in manifest_for(node_id)["supported_phases"] for node_id in closure)

    bundle = outcome["execution_bundle"]
    artifact_nodes = {record["node_id"] for record in bundle["refit_artifacts"]}
    inference_state_present = all(
        (
            "stateful" not in manifest_for(node_id)["capabilities"]
            and "emits_artifacts" not in manifest_for(node_id)["capabilities"]
        )
        or node_id in artifact_nodes
        for node_id in closure
    )

    def key(producer: str, source_port: str, consumer: str, target_port: str) -> str:
        return f"{producer}.{source_port}->{consumer}.{target_port}"

    requirement_keys = {
        key(
            requirement["producer_node"],
            requirement["source_port"],
            requirement["consumer_node"],
            requirement["target_port"],
        )
        for requirement in bundle["prediction_requirements"]
    }
    cache_keys = {record["requirement_key"] for record in bundle["prediction_caches"]}
    payloads = outcome.get("portable_prediction_caches")
    payload_keys = (
        {payload["requirement_key"] for payload in payloads["caches"]}
        if payloads is not None
        else set()
    )
    oof_self_contained = True
    for edge in graph.get("edges", []):
        if edge["contract"].get("requires_oof") is not True:
            continue
        source = edge["source"]["node_id"]
        target = edge["target"]["node_id"]
        if source not in closure or target not in closure:
            continue
        edge_key = key(
            source, edge["source"]["port_name"], target, edge["target"]["port_name"]
        )
        if not (
            edge_key in requirement_keys
            and edge_key in cache_keys
            and edge_key in payload_keys
        ):
            oof_self_contained = False

    phases: list[str] = []
    if outcome["refit"]["status"] == "completed":
        if all_support("PREDICT") and inference_state_present:
            phases.append("PREDICT")
        if all_support("EXPLAIN") and inference_state_present:
            phases.append("EXPLAIN")
    elif all_support("REFIT") and oof_self_contained:
        phases.append("REFIT")
    return phases


def restore_explicit_sample_bundle_wire(outcome: dict[str, Any]) -> None:
    """Restore the published v0.2.7 write profile for sample cache records."""

    bundle = outcome["execution_bundle"]
    for requirement in bundle["prediction_requirements"]:
        requirement.setdefault("prediction_level", "sample")
    for cache in bundle["prediction_caches"]:
        cache.setdefault("prediction_level", "sample")
        for block in cache["blocks"]:
            block.setdefault("prediction_level", "sample")
    payloads = outcome.get("portable_prediction_caches")
    if payloads is not None:
        for payload in payloads["caches"]:
            payload.setdefault("prediction_level", "sample")
    set_fingerprint(outcome, "outcome_fingerprint")


def cohort(
    role: str,
    unit_relations: list[dict[str, Any]],
    *,
    target_names: list[str],
    relation_label: str,
    axes: bool = True,
) -> dict[str, Any]:
    physical_sample_ids = [
        relation["physical_sample_id"] for relation in unit_relations
    ]
    origin_sample_ids = sorted(
        {
            relation["origin_sample_id"]
            for relation in unit_relations
            if relation["origin_sample_id"] is not None
        }
    )
    group_ids = sorted(
        {group for relation in unit_relations for group in relation["group_ids"]}
    )
    source_ids = sorted(
        {source for relation in unit_relations for source in relation["source_ids"]}
    )
    document = {
        "schema_version": 1,
        "role": role,
        "exchangeability_unit": "physical_sample",
        "physical_sample_ids": physical_sample_ids,
        "origin_sample_ids": origin_sample_ids,
        "group_ids": group_ids,
        "source_ids": source_ids,
        "unit_relations": unit_relations,
        "target_names": target_names,
        "relation_fingerprint": opaque(f"relation:{relation_label}"),
        "content_fingerprint": opaque(f"content:{relation_label}"),
        "manifest_fingerprint": "0" * 64,
        "axes_fingerprint": opaque(f"axes:{relation_label}") if axes else None,
    }
    set_fingerprint(document, "manifest_fingerprint")
    return document


def relation(
    sample_id: str,
    *,
    origin: str | None = None,
    groups: tuple[str, ...] = (),
    sources: tuple[str, ...] = ("nir",),
) -> dict[str, Any]:
    return {
        "physical_sample_id": sample_id,
        "origin_sample_id": origin,
        "group_ids": sorted(groups),
        "source_ids": sorted(sources),
    }


def role_cohorts() -> dict[str, dict[str, Any]]:
    return {
        "development": cohort(
            "development",
            [
                relation(
                    "sample:dev.01",
                    origin="sample:dev.origin.01",
                    groups=("group:batch.A",),
                ),
                relation("sample:dev.02", groups=("group:batch.A",)),
                relation("sample:dev.03", groups=("group:batch.A",)),
            ],
            target_names=["protein"],
            relation_label="development",
        ),
        "calibration": cohort(
            "calibration",
            [
                relation(
                    "sample:cal.01",
                    origin="sample:cal.origin.01",
                    groups=("group:batch.B",),
                ),
                relation("sample:cal.02", groups=("group:batch.B",)),
                relation("sample:cal.03", groups=("group:batch.B",)),
            ],
            target_names=["protein"],
            relation_label="calibration.roles",
        ),
        "external_test": cohort(
            "external_test",
            [
                relation("sample:ext.01", groups=("group:batch.A",)),
                relation("sample:ext.02", groups=("group:batch.B",)),
                relation("sample:ext.03", groups=("group:batch.A",)),
                relation(
                    "sample:ext.04",
                    groups=("group:batch.B",),
                    sources=("nir.secondary",),
                ),
            ],
            target_names=["protein"],
            relation_label="external",
        ),
        "production": cohort(
            "production",
            [relation("sample:prod.01"), relation("sample:prod.02")],
            target_names=["protein"],
            relation_label="production",
            axes=False,
        ),
    }


def calibration_cohort(prefix: str, *, relation_label: str) -> dict[str, Any]:
    records = []
    for index in range(1, 21):
        records.append(
            relation(
                f"sample:{prefix}.{index:02d}",
                origin=f"sample:{prefix}.origin.01" if index == 1 else None,
                groups=(f"group:{prefix}.{'A' if index <= 10 else 'B'}",),
            )
        )
    return cohort(
        "calibration",
        records,
        target_names=["protein"],
        relation_label=relation_label,
    )


def finalize_artifact(artifact: dict[str, Any]) -> None:
    set_fingerprint(artifact["calibration_cohort"], "manifest_fingerprint")
    set_fingerprint(artifact["training_influence"], "manifest_fingerprint")
    predictor = artifact["predictor_binding"]
    referenced_nodes = {
        predictor["output_binding"]["node_id"],
        *(binding["node_id"] for binding in predictor["artifacts"]),
        *(patch["node_id"] for patch in predictor["selected_patches"]),
        *(
            binding["requirement_key"].rsplit(".", maxsplit=1)[0]
            for binding in predictor["data_bindings"]
        ),
    }
    predictor.setdefault("predictor_node_ids", sorted(referenced_nodes))
    assert referenced_nodes <= set(predictor["predictor_node_ids"])
    predictor["training_influence_fingerprint"] = artifact["training_influence"][
        "manifest_fingerprint"
    ]
    set_fingerprint(predictor["output_binding"], "binding_fingerprint")
    artifact["predictor_binding_fingerprint"] = tcv1_sha256(predictor)
    artifact["calibration_spec_fingerprint"] = tcv1_sha256(artifact["calibration_spec"])
    set_fingerprint(artifact, "checksum")


def calibration_artifacts(
    calibration_fixture: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = copy.deepcopy(calibration_fixture["calibration_artifact"])
    base["calibration_cohort"] = calibration_cohort(
        "base.cal", relation_label="calibration.base"
    )
    base["effective_sample_count"] = 20
    finalize_artifact(base)

    matched = copy.deepcopy(base)
    matched["artifact_id"] = "calibration:matched.v1"
    matched["calibration_cohort"] = calibration_cohort(
        "matched.cal", relation_label="calibration.matched"
    )
    matched["diagnostics"] = copy.deepcopy(base["diagnostics"])
    matched["diagnostics"].update(
        {
            "context": "matched_recalibration",
            "scenario_id": "scenario:matched.noise",
            "severity": 0.5,
            "calibration_input_fingerprint": matched["calibration_cohort"][
                "content_fingerprint"
            ],
        }
    )
    finalize_artifact(matched)

    structural = copy.deepcopy(base)
    structural["artifact_id"] = "calibration:structural.v1"
    structural["calibration_cohort"] = calibration_cohort(
        "struct.cal", relation_label="calibration.structural"
    )
    predictor = structural["predictor_binding"]
    predictor["plan_id"] = "plan:fixture.execution.structural.node-replacement"
    predictor["graph_fingerprint"] = opaque("graph:structural.node-replacement")
    selected_variant_fingerprint = opaque("variant:structural.refit")
    predictor["selected_variant_id"] = f"variant:{selected_variant_fingerprint[:16]}"
    predictor["selected_variant_fingerprint"] = selected_variant_fingerprint
    predictor["training_outcome_fingerprint"] = opaque(
        "training-outcome:structural.refit"
    )
    affected_nodes = {
        "branch:b1.model:rf",
        "merge:stack.pred_plus_original.meta:ridge",
    }
    for artifact_binding in predictor["artifacts"]:
        if artifact_binding["node_id"] in affected_nodes:
            artifact_binding["content_fingerprint"] = opaque(
                f"structural:{artifact_binding['artifact_id']}"
            )
            artifact_binding["params_fingerprint"] = opaque(
                f"structural-params:{artifact_binding['artifact_id']}"
            )
    structural["diagnostics"] = copy.deepcopy(base["diagnostics"])
    structural["diagnostics"].update(
        {
            "context": "structural_refit_recalibration",
            "scenario_id": "scenario:structural.node",
            "severity": 1.0,
            "calibration_input_fingerprint": structural["calibration_cohort"][
                "content_fingerprint"
            ],
        }
    )
    finalize_artifact(structural)
    return base, matched, structural


def unbounded_calibration_artifact(base: dict[str, Any]) -> dict[str, Any]:
    artifact = copy.deepcopy(base)
    artifact["artifact_id"] = "calibration:small-n.unbounded.v1"
    artifact["calibration_cohort"]["target_names"] = ["moisture", "protein"]
    output_binding = artifact["predictor_binding"]["output_binding"]
    output_binding["target_names"] = ["moisture", "protein"]
    output_binding["target_units"] = ["percent", "percent"]
    output_binding["class_labels"] = [[], []]
    artifact["predictor_binding"]["target_processing_fingerprint"] = opaque(
        "target-processing:small-n.unbounded"
    )
    artifact["calibration_spec"]["coverages"] = [0.99, 0.999]
    artifact["calibration_spec"]["small_sample_policy"] = "unbounded"
    artifact["quantiles"] = [
        {
            "coverage": coverage,
            "rank": 21,
            "values": [
                {"status": "unbounded"},
                {"status": "unbounded"},
            ],
        }
        for coverage in (0.99, 0.999)
    ]
    artifact["diagnostics"] = {
        "context": "small_n_unbounded",
        "quantile_convention": "rank=ceil((n+1)*coverage), one-indexed",
        "guarantee": "unbounded interval when the finite-sample rank exceeds n",
    }
    finalize_artifact(artifact)
    return artifact


def calibration_artifact_fixture(
    base: dict[str, Any], unbounded: dict[str, Any]
) -> dict[str, Any]:
    null_value = copy.deepcopy(unbounded)
    null_value["quantiles"][0]["values"][0] = {
        "status": "unbounded",
        "value": None,
    }
    set_fingerprint(null_value, "checksum")

    mixed = copy.deepcopy(unbounded)
    mixed["quantiles"][0]["values"][1] = {
        "status": "finite",
        "value": 2.0,
    }
    set_fingerprint(mixed, "checksum")

    returns_finite = copy.deepcopy(unbounded)
    returns_finite["quantiles"][1]["values"] = [
        {"status": "finite", "value": 2.0},
        {"status": "finite", "value": 2.0},
    ]
    set_fingerprint(returns_finite, "checksum")

    integer_quantile = copy.deepcopy(base)
    integer_quantile["quantiles"][1]["values"][0]["value"] = 2
    set_fingerprint(integer_quantile, "checksum")

    nonmonotone = copy.deepcopy(base)
    nonmonotone["quantiles"][1]["values"][0]["value"] = 1.8
    set_fingerprint(nonmonotone, "checksum")

    unsorted_artifacts = copy.deepcopy(base)
    unsorted_artifacts["predictor_binding"]["artifacts"].reverse()
    unsorted_artifacts["predictor_binding_fingerprint"] = tcv1_sha256(
        unsorted_artifacts["predictor_binding"]
    )
    set_fingerprint(unsorted_artifacts, "checksum")

    mismatched_relation = copy.deepcopy(base)
    for binding in mismatched_relation["predictor_binding"]["data_bindings"]:
        binding["relation_fingerprint"] = "f" * 64
    mismatched_relation["predictor_binding_fingerprint"] = tcv1_sha256(
        mismatched_relation["predictor_binding"]
    )
    set_fingerprint(mismatched_relation, "checksum")

    incomplete_closure = copy.deepcopy(base)
    incomplete_closure["predictor_binding"]["predictor_node_ids"].remove(
        "branch:b1.augment:noise"
    )
    incomplete_closure["predictor_binding_fingerprint"] = tcv1_sha256(
        incomplete_closure["predictor_binding"]
    )
    set_fingerprint(incomplete_closure, "checksum")

    def mutations(
        document: dict[str, Any], path: str, value: Any
    ) -> list[dict[str, Any]]:
        return [
            {"path": path, "value": value},
            {"path": "/checksum", "value": document["checksum"]},
        ]

    def predictor_mutations(document: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"path": "/predictor_binding", "value": document["predictor_binding"]},
            {
                "path": "/predictor_binding_fingerprint",
                "value": document["predictor_binding_fingerprint"],
            },
            {"path": "/checksum", "value": document["checksum"]},
        ]

    return {
        "fixture_id": "dag-ml.conformal.calibration-artifacts.v1",
        "schema_version": 1,
        "schema": "conformal_calibration.schema.json",
        "valid_cases": [
            {"id": "finite_calibration_artifact", "document": base},
            {"id": "small_n_unbounded_artifact", "document": unbounded},
        ],
        "invalid_cases": [
            {
                "id": "unbounded_value_null_refuses",
                "base_case": "small_n_unbounded_artifact",
                "mutations": mutations(
                    null_value,
                    "/quantiles",
                    null_value["quantiles"],
                ),
                "expected_error": "value",
            },
            {
                "id": "mixed_unbounded_finite_refuses",
                "base_case": "small_n_unbounded_artifact",
                "mutations": mutations(mixed, "/quantiles", mixed["quantiles"]),
                "expected_error": "unbounded",
            },
            {
                "id": "unbounded_cannot_return_finite",
                "base_case": "small_n_unbounded_artifact",
                "mutations": mutations(
                    returns_finite,
                    "/quantiles",
                    returns_finite["quantiles"],
                ),
                "expected_error": "unbounded",
            },
            {
                "id": "finite_quantile_integer_token_refuses",
                "base_case": "finite_calibration_artifact",
                "mutations": mutations(
                    integer_quantile,
                    "/quantiles",
                    integer_quantile["quantiles"],
                ),
                "expected_error": "binary64",
            },
            {
                "id": "finite_quantiles_must_be_monotone",
                "base_case": "finite_calibration_artifact",
                "mutations": mutations(
                    nonmonotone,
                    "/quantiles",
                    nonmonotone["quantiles"],
                ),
                "expected_error": "monot",
            },
            {
                "id": "predictor_artifacts_must_be_sorted",
                "base_case": "finite_calibration_artifact",
                "mutations": predictor_mutations(unsorted_artifacts),
                "expected_error": "artifacts must be sorted",
            },
            {
                "id": "all_data_bindings_must_match_influence_relation",
                "base_case": "finite_calibration_artifact",
                "mutations": predictor_mutations(mismatched_relation),
                "expected_error": "training influence",
            },
            {
                "id": "predictor_node_closure_cannot_omit_transform",
                "base_case": "finite_calibration_artifact",
                "mutations": predictor_mutations(incomplete_closure),
                "expected_error": "predictor closure",
            },
        ],
    }


def scenario_rng(target_kind: str) -> dict[str, Any]:
    return {
        "algorithm": "philox4x32-10",
        "algorithm_version": 1,
        "counter_profile": "dagml-robustness-counter.v1",
        "counter_derivation": "sha256-tcv1-first128",
        "counter_fields": [
            "scenario_fingerprint",
            "severity_binary64",
            "unit_id",
            "target_kind",
            "target_id",
            "draw_index",
        ],
        "key_derivation": "uint64-seed-as-two-little-endian-u32",
        "target_kind": target_kind,
        "seed": 17,
    }


def make_scenario(
    *,
    scenario_id: str,
    mode: str,
    role: str,
    source_ids: list[str],
    node_ids: list[str],
    perturbation_kind: str,
    severities: list[float],
    environment_id: str,
    slice_by: list[str],
    target_kind: str,
) -> dict[str, Any]:
    policy = {
        "clean_frozen": (False, False),
        "matched_recalibration": (False, True),
        "structural_refit": (True, True),
    }[mode]
    scenario = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "mode": mode,
        "cohort_role": role,
        "source_ids": sorted(source_ids),
        "node_ids": sorted(node_ids),
        "perturbation": {
            "kind": perturbation_kind,
            "parameters": {
                "profile": f"{perturbation_kind}.v1",
            },
        },
        "severities": severities,
        "zero_severity_semantics": "identity",
        "rng": scenario_rng(target_kind),
        "split_id": "split:external.holdout"
        if role == "external_test"
        else "split:production.window",
        "environment_id": environment_id,
        "slice_by": sorted(slice_by),
        "metrics": ["conformal_coverage", "mae", "mean_width", "r2", "rmse"],
        "requires_refit": policy[0],
        "requires_recalibration": policy[1],
        "scenario_fingerprint": "0" * 64,
    }
    set_fingerprint(scenario, "scenario_fingerprint")
    return scenario


def scenarios() -> list[dict[str, Any]]:
    return [
        make_scenario(
            scenario_id="scenario:clean.noise",
            mode="clean_frozen",
            role="external_test",
            source_ids=["nir", "nir.secondary"],
            node_ids=[],
            perturbation_kind="gaussian_noise",
            severities=[0.0, 0.01],
            environment_id="environment:instrument.A",
            slice_by=["group", "source"],
            target_kind="source",
        ),
        make_scenario(
            scenario_id="scenario:matched.noise",
            mode="matched_recalibration",
            role="external_test",
            source_ids=["nir"],
            node_ids=[],
            perturbation_kind="ordered_axis_shift",
            severities=[0.0, 0.5],
            environment_id="environment:instrument.B",
            slice_by=["group"],
            target_kind="source",
        ),
        make_scenario(
            scenario_id="scenario:structural.node",
            mode="structural_refit",
            role="external_test",
            source_ids=[],
            node_ids=["branch:b1.augment:noise"],
            perturbation_kind="node_replacement",
            severities=[0.0, 1.0],
            environment_id="environment:instrument.C",
            slice_by=[],
            target_kind="node",
        ),
    ]


def standalone_output_binding() -> dict[str, Any]:
    binding = {
        "schema_version": 1,
        "binding_id": "output:external.final",
        "node_id": "model:pls",
        "port_name": "prediction",
        "prediction_level": "sample",
        "unit_level": "physical_sample",
        "prediction_kind": "regression_point",
        "prediction_source": "final_refit",
        "refit_strategy": "refit_one",
        "aggregation_fingerprint": opaque("aggregation:standalone"),
        "target_names": ["moisture", "protein"],
        "target_units": ["percent", "percent"],
        "class_labels": [[], []],
        "output_order": "target_order",
        "target_space": "raw",
        "binding_fingerprint": "0" * 64,
    }
    set_fingerprint(binding, "binding_fingerprint")
    return binding


def make_prediction_block(
    *,
    block_id: str,
    artifact_id: str,
    artifact_checksum: str,
    predictor_fingerprint: str,
    cohort_fingerprint: str,
    point_fingerprint: str,
    output_binding: dict[str, Any],
    unit_ids: list[str],
    policy: str,
    intervals: list[dict[str, Any]],
    assumption_status: str,
    guarantee_status: str,
) -> dict[str, Any]:
    block = {
        "schema_version": 1,
        "block_id": block_id,
        "calibration_artifact_id": artifact_id,
        "calibration_artifact_checksum": artifact_checksum,
        "predictor_binding_fingerprint": predictor_fingerprint,
        "cohort_manifest_fingerprint": cohort_fingerprint,
        "point_prediction_fingerprint": point_fingerprint,
        "point_output_binding": copy.deepcopy(output_binding),
        "method": "split_absolute_residual",
        "numeric_version": "split_absolute_residual.v1",
        "unit_level": "physical_sample",
        "prediction_level": "sample",
        "unit_ids": unit_ids,
        "target_names": output_binding["target_names"],
        "multi_target_policy": policy,
        "intervals": intervals,
        "assumption_status": assumption_status,
        "guarantee_status": guarantee_status,
        "block_fingerprint": "0" * 64,
    }
    set_fingerprint(block, "block_fingerprint")
    return block


def metric_record(
    *,
    coverage: float,
    target_name: str | None,
    scenario_id: str | None,
    severity: float | None,
    slice_key: dict[str, Any],
    seed: int | None,
    unit_ids: list[str],
    guarantee_status: str,
    width: float | None,
    measurement_status: str = "finite",
) -> dict[str, Any]:
    empirical = 1.0 if measurement_status != "unavailable" else None
    return {
        "coverage": coverage,
        "target_name": target_name,
        "scenario_id": scenario_id,
        "severity": severity,
        "slice": slice_key,
        "fold_id": None,
        "repeat_id": None,
        "seed": seed,
        "unit_ids_fingerprint": tcv1_sha256(unit_ids),
        "sample_count": len(unit_ids),
        "measurement_status": measurement_status,
        "guarantee_status": guarantee_status,
        "empirical_coverage": empirical,
        "coverage_gap": None if empirical is None else empirical - coverage,
        "mean_width": width,
        "median_width": width,
        "interval_score": width,
        "set_size": None,
    }


def metric_records_from_truth(
    *,
    block: dict[str, Any],
    truth: list[list[float]],
    scenario_id: str | None,
    severity: float | None,
    slice_key: dict[str, Any],
    seed: int | None,
    guarantee_status: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for interval in block["intervals"]:
        summaries = regression_conformal_metrics(
            truth,
            interval,
            multi_target_policy=block["multi_target_policy"],
        )
        for summary in summaries:
            target_index = summary["target_index"]
            target_name = (
                None if target_index is None else block["target_names"][target_index]
            )
            record = metric_record(
                coverage=interval["coverage"],
                target_name=target_name,
                scenario_id=scenario_id,
                severity=severity,
                slice_key=copy.deepcopy(slice_key),
                seed=seed,
                unit_ids=block["unit_ids"],
                guarantee_status=guarantee_status,
                width=summary["mean_width"],
                measurement_status=summary["measurement_status"],
            )
            for field in (
                "measurement_status",
                "empirical_coverage",
                "coverage_gap",
                "mean_width",
                "median_width",
                "interval_score",
            ):
                record[field] = summary[field]
            records.append(record)
    return records


def metric_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["scenario_id"] or "",
        record["severity"] if record["severity"] is not None else -1.0,
        record["slice"]["kind"],
        record["slice"]["value"] or "",
        record["target_name"] or "",
        record["coverage"],
        record["fold_id"] or "",
        record["repeat_id"] or "",
        record["seed"] if record["seed"] is not None else -1,
    )


def make_metric_set(
    *,
    metric_set_id: str,
    block: dict[str, Any],
    cohort_fingerprint: str,
    truth_fingerprint: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_set = {
        "schema_version": 1,
        "metric_set_id": metric_set_id,
        "calibration_artifact_id": block["calibration_artifact_id"],
        "calibration_artifact_checksum": block["calibration_artifact_checksum"],
        "predictor_binding_fingerprint": block["predictor_binding_fingerprint"],
        "point_prediction_fingerprint": block["point_prediction_fingerprint"],
        "conformal_prediction_block_fingerprint": block["block_fingerprint"],
        "truth_fingerprint": truth_fingerprint,
        "unit_ids_fingerprint": tcv1_sha256(block["unit_ids"]),
        "cohort_manifest_fingerprint": cohort_fingerprint,
        "method": "split_absolute_residual",
        "unit_level": "physical_sample",
        "multi_target_policy": block["multi_target_policy"],
        "records": sorted(records, key=metric_key),
        "metric_set_fingerprint": "0" * 64,
    }
    set_fingerprint(metric_set, "metric_set_fingerprint")
    return metric_set


def make_numeric_evidence(
    *,
    evidence_id: str,
    block: dict[str, Any],
    metric_set: dict[str, Any],
    point_predictions: list[list[float]],
    truth: list[list[float]],
) -> dict[str, Any]:
    evidence = {
        "evidence_id": evidence_id,
        "block_fingerprint": block["block_fingerprint"],
        "metric_set_id": metric_set["metric_set_id"],
        "point_predictions": point_predictions,
        "point_prediction_fingerprint": tcv1_sha256(point_predictions),
        "truth": truth,
        "truth_fingerprint": tcv1_sha256(truth),
        "evidence_fingerprint": "0" * 64,
    }
    set_fingerprint(evidence, "evidence_fingerprint")
    return evidence


def standalone_prediction_and_metrics(
    external: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = standalone_output_binding()
    units = ["sample:ext.01", "sample:ext.02"]
    point_predictions = [[10.0, 20.0], [11.0, 21.0]]
    truth = [[10.0, 21.0], [14.0, 21.0]]
    marginal = make_prediction_block(
        block_id="conformal:block.external.marginal",
        artifact_id="calibration:standalone.marginal",
        artifact_checksum=opaque("calibration:standalone.marginal"),
        predictor_fingerprint=opaque("predictor:standalone"),
        cohort_fingerprint=external["manifest_fingerprint"],
        point_fingerprint=tcv1_sha256(point_predictions),
        output_binding=binding,
        unit_ids=units,
        policy="marginal",
        intervals=[
            {
                "coverage": 0.8,
                "lower": [[8.0, 18.0], [9.0, 19.0]],
                "upper": [[12.0, 22.0], [13.0, 23.0]],
            },
            {
                "coverage": 0.9,
                "lower": [[7.0, 17.0], [8.0, 18.0]],
                "upper": [[13.0, 23.0], [14.0, 24.0]],
            },
        ],
        assumption_status="declared_exchangeable",
        guarantee_status="marginal_coverage",
    )
    joint = make_prediction_block(
        block_id="conformal:block.external.joint",
        artifact_id="calibration:standalone.joint",
        artifact_checksum=opaque("calibration:standalone.joint"),
        predictor_fingerprint=opaque("predictor:standalone"),
        cohort_fingerprint=external["manifest_fingerprint"],
        point_fingerprint=tcv1_sha256(point_predictions),
        output_binding=binding,
        unit_ids=units,
        policy="joint_max",
        intervals=[
            {
                "coverage": 0.8,
                "lower": [[7.0, 17.0], [8.0, 18.0]],
                "upper": [[13.0, 23.0], [14.0, 24.0]],
            },
            {
                "coverage": 0.9,
                "lower": [[6.0, 16.0], [7.0, 17.0]],
                "upper": [[14.0, 24.0], [15.0, 25.0]],
            },
        ],
        assumption_status="declared_exchangeable",
        guarantee_status="joint_coverage",
    )
    unbounded = make_prediction_block(
        block_id="conformal:block.external.unbounded",
        artifact_id="calibration:standalone.unbounded",
        artifact_checksum=opaque("calibration:standalone.unbounded"),
        predictor_fingerprint=opaque("predictor:standalone"),
        cohort_fingerprint=external["manifest_fingerprint"],
        point_fingerprint=tcv1_sha256([point_predictions[0]]),
        output_binding=binding,
        unit_ids=["sample:ext.01"],
        policy="marginal",
        intervals=[
            {
                "coverage": 0.99,
                "lower": [[None, None]],
                "upper": [[None, None]],
            }
        ],
        assumption_status="not_assessed",
        guarantee_status="unavailable",
    )
    target_order_mismatch = copy.deepcopy(marginal)
    target_order_mismatch["point_output_binding"]["target_names"] = [
        "protein",
        "moisture",
    ]
    set_fingerprint(
        target_order_mismatch["point_output_binding"], "binding_fingerprint"
    )
    set_fingerprint(target_order_mismatch, "block_fingerprint")
    prediction_fixture = {
        "fixture_id": "dag-ml.conformal.prediction-blocks.v1",
        "schema_version": 1,
        "schema": "conformal_prediction_block.schema.json",
        "valid_cases": [
            {"id": "joint_max_two_target_nested", "document": joint},
            {"id": "marginal_two_target_nested", "document": marginal},
            {"id": "small_n_unbounded_interval", "document": unbounded},
        ],
        "invalid_cases": [
            {
                "id": "coverages_not_increasing",
                "base_case": "marginal_two_target_nested",
                "mutations": [{"path": "/intervals/1/coverage", "value": 0.7}],
                "expected_error": "coverage",
            },
            {
                "id": "higher_coverage_not_nested",
                "base_case": "marginal_two_target_nested",
                "mutations": [{"path": "/intervals/1/lower/0/0", "value": 9.0}],
                "expected_error": "nested",
            },
            {
                "id": "lower_above_upper",
                "base_case": "marginal_two_target_nested",
                "mutations": [{"path": "/intervals/0/lower/1/1", "value": 24.0}],
                "expected_error": "lower bound",
            },
            {
                "id": "target_order_mismatch",
                "base_case": "marginal_two_target_nested",
                "mutations": [
                    {
                        "path": "/point_output_binding",
                        "value": target_order_mismatch["point_output_binding"],
                    },
                    {
                        "path": "/block_fingerprint",
                        "value": target_order_mismatch["block_fingerprint"],
                    },
                ],
                "expected_error": "target order",
            },
            {
                "id": "shift_cannot_claim_formal_coverage",
                "base_case": "marginal_two_target_nested",
                "mutations": [
                    {"path": "/assumption_status", "value": "distribution_shift"}
                ],
                "expected_error": "overclaims",
            },
            {
                "id": "domain_status_is_not_conformal",
                "base_case": "marginal_two_target_nested",
                "mutations": [{"path": "/domain_status", "value": "out_of_support"}],
                "expected_error": "domain_status",
            },
            {
                "id": "runtime_handle_refuses",
                "base_case": "marginal_two_target_nested",
                "mutations": [{"path": "/point_prediction_handle", "value": 99}],
                "expected_error": "handle",
            },
            {
                "id": "one_sided_unbounded_refuses",
                "base_case": "small_n_unbounded_interval",
                "mutations": [{"path": "/intervals/0/upper/0/0", "value": 12.0}],
                "expected_error": "paired",
            },
        ],
    }

    marginal_records = metric_records_from_truth(
        block=marginal,
        truth=truth,
        scenario_id=None,
        severity=None,
        slice_key={"kind": "all", "value": None},
        seed=17,
        guarantee_status="marginal_coverage",
    )
    marginal_metrics = make_metric_set(
        metric_set_id="metrics:standalone.marginal",
        block=marginal,
        cohort_fingerprint=external["manifest_fingerprint"],
        truth_fingerprint=tcv1_sha256(truth),
        records=marginal_records,
    )
    joint_metrics = make_metric_set(
        metric_set_id="metrics:standalone.joint",
        block=joint,
        cohort_fingerprint=external["manifest_fingerprint"],
        truth_fingerprint=tcv1_sha256(truth),
        records=metric_records_from_truth(
            block=joint,
            truth=truth,
            scenario_id=None,
            severity=None,
            slice_key={"kind": "all", "value": None},
            seed=17,
            guarantee_status="joint_coverage",
        ),
    )
    unbounded_truth = [[100.0, -100.0]]
    unbounded_metrics = make_metric_set(
        metric_set_id="metrics:standalone.unbounded",
        block=unbounded,
        cohort_fingerprint=external["manifest_fingerprint"],
        truth_fingerprint=tcv1_sha256(unbounded_truth),
        records=metric_records_from_truth(
            block=unbounded,
            truth=unbounded_truth,
            scenario_id=None,
            severity=None,
            slice_key={"kind": "all", "value": None},
            seed=17,
            guarantee_status="unavailable",
        ),
    )
    evidence_cases = sorted(
        [
            make_numeric_evidence(
                evidence_id="evidence:standalone.joint",
                block=joint,
                metric_set=joint_metrics,
                point_predictions=point_predictions,
                truth=truth,
            ),
            make_numeric_evidence(
                evidence_id="evidence:standalone.marginal",
                block=marginal,
                metric_set=marginal_metrics,
                point_predictions=point_predictions,
                truth=truth,
            ),
            make_numeric_evidence(
                evidence_id="evidence:standalone.unbounded",
                block=unbounded,
                metric_set=unbounded_metrics,
                point_predictions=[point_predictions[0]],
                truth=unbounded_truth,
            ),
        ],
        key=lambda evidence: evidence["evidence_id"],
    )
    metric_fixture = {
        "fixture_id": "dag-ml.conformal.metric-sets.v1",
        "schema_version": 1,
        "schema": "conformal_metric_set.schema.json",
        "valid_cases": [
            {"id": "joint_max_exact_metrics", "document": joint_metrics},
            {"id": "marginal_two_target_metrics", "document": marginal_metrics},
            {"id": "small_n_unbounded_metrics", "document": unbounded_metrics},
        ],
        "evidence_cases": evidence_cases,
        "invalid_cases": [
            {
                "id": "coverage_gap_mismatch",
                "base_case": "marginal_two_target_metrics",
                "mutations": [{"path": "/records/0/coverage_gap", "value": 0.25}],
                "expected_error": "coverage_gap",
            },
            {
                "id": "marginal_target_missing",
                "base_case": "marginal_two_target_metrics",
                "mutations": [{"path": "/records/0/target_name", "value": None}],
                "expected_error": "target_name",
            },
            {
                "id": "joint_target_must_be_null",
                "base_case": "joint_max_exact_metrics",
                "mutations": [
                    {"path": "/records/0/target_name", "value": "protein"},
                    {"path": "/records/1/target_name", "value": "protein"},
                ],
                "expected_error": "target_name",
            },
            {
                "id": "duplicate_metric_coordinate",
                "base_case": "marginal_two_target_metrics",
                "mutations": [
                    {"path": "/records/1", "value": marginal_metrics["records"][0]}
                ],
                "expected_error": "duplicate",
            },
            {
                "id": "runtime_handle_refuses",
                "base_case": "marginal_two_target_metrics",
                "mutations": [{"path": "/metric_handle", "value": 7}],
                "expected_error": "handle",
            },
            {
                "id": "sliced_metric_cannot_claim_formal_guarantee",
                "base_case": "marginal_two_target_metrics",
                "mutations": [
                    {
                        "path": "/records/3/slice",
                        "value": {"kind": "group", "value": "g"},
                    }
                ],
                "expected_error": "overclaims",
            },
            {
                "id": "unbounded_width_must_be_null",
                "base_case": "small_n_unbounded_metrics",
                "mutations": [{"path": "/records/0/mean_width", "value": 99.0}],
                "expected_error": "must be null",
            },
        ],
    }
    return prediction_fixture, metric_fixture


def domain_and_decision_fixtures(
    external: dict[str, Any], predictor_fingerprint: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = {
        "schema_version": 1,
        "block_id": "domain:external.support.v1",
        "predictor_binding_fingerprint": predictor_fingerprint,
        "cohort_manifest_fingerprint": external["manifest_fingerprint"],
        "assessment_policy_id": "policy:domain.support.v1",
        "assessment_policy_fingerprint": opaque("policy:domain.support.v1"),
        "unit_level": "physical_sample",
        "unit_ids": ["sample:ext.01", "sample:ext.02"],
        "assessments": [
            {
                "unit_id": "sample:ext.01",
                "status": "in_support",
                "methods": [
                    {
                        "method_id": "method:leverage",
                        "kind": "leverage",
                        "score": 0.25,
                        "threshold": 0.8,
                        "supported": True,
                    }
                ],
                "reasons": [],
            },
            {
                "unit_id": "sample:ext.02",
                "status": "out_of_support",
                "methods": [
                    {
                        "method_id": "method:feature.support",
                        "kind": "feature_support",
                        "score": 1.2,
                        "threshold": 1.0,
                        "supported": False,
                    }
                ],
                "reasons": ["feature_support_exceeded"],
            },
        ],
        "block_fingerprint": "0" * 64,
    }
    set_fingerprint(domain, "block_fingerprint")
    domain_fixture = {
        "fixture_id": "dag-ml.conformal.domain-assessment-blocks.v1",
        "schema_version": 1,
        "schema": "domain_assessment_block.schema.json",
        "valid_cases": [{"id": "two_unit_support_assessment", "document": domain}],
        "invalid_cases": [
            {
                "id": "assessment_identity_order_mismatch",
                "base_case": "two_unit_support_assessment",
                "mutations": [
                    {"path": "/assessments/0/unit_id", "value": "sample:ext.02"}
                ],
                "expected_error": "align",
            },
            {
                "id": "status_contradicts_method_support",
                "base_case": "two_unit_support_assessment",
                "mutations": [
                    {"path": "/assessments/0/status", "value": "out_of_support"}
                ],
                "expected_error": "contradicts",
            },
            {
                "id": "coverage_claim_is_not_domain_assessment",
                "base_case": "two_unit_support_assessment",
                "mutations": [{"path": "/coverage", "value": 0.9}],
                "expected_error": "coverage",
            },
            {
                "id": "runtime_handle_refuses",
                "base_case": "two_unit_support_assessment",
                "mutations": [{"path": "/support_handle", "value": 12}],
                "expected_error": "handle",
            },
            {
                "id": "nirs_specific_method_is_not_core_contract",
                "base_case": "two_unit_support_assessment",
                "mutations": [
                    {
                        "path": "/assessments/1/methods/0/kind",
                        "value": "spectral_support",
                    }
                ],
                "expected_error": "not one of",
            },
        ],
    }

    decision = {
        "schema_version": 1,
        "block_id": "decision:external.application.v1",
        "policy_id": "policy:application.review.v1",
        "policy_fingerprint": opaque("policy:application.review.v1"),
        "predictor_binding_fingerprint": predictor_fingerprint,
        "cohort_manifest_fingerprint": external["manifest_fingerprint"],
        "conformal_block_fingerprint": opaque("block:decision.conformal"),
        "domain_assessment_fingerprint": domain["block_fingerprint"],
        "thresholds": [
            {
                "name": "allowed_actions",
                "operator": "in",
                "value": ["accept", "refer"],
                "unit": None,
            },
            {
                "name": "max_interval_width",
                "operator": "lte",
                "value": 6.0,
                "unit": "percent",
            },
        ],
        "unit_level": "physical_sample",
        "unit_ids": ["sample:ext.01", "sample:ext.02"],
        "decisions": [
            {
                "unit_id": "sample:ext.01",
                "action": "accept",
                "reasons": ["within_policy"],
            },
            {
                "unit_id": "sample:ext.02",
                "action": "refer",
                "reasons": ["out_of_support"],
            },
        ],
        "block_fingerprint": "0" * 64,
    }
    set_fingerprint(decision, "block_fingerprint")
    decision_fixture = {
        "fixture_id": "dag-ml.conformal.decision-blocks.v1",
        "schema_version": 1,
        "schema": "decision_block.schema.json",
        "valid_cases": [{"id": "accept_and_refer", "document": decision}],
        "invalid_cases": [
            {
                "id": "decision_identity_mismatch",
                "base_case": "accept_and_refer",
                "mutations": [
                    {"path": "/decisions/0/unit_id", "value": "sample:ext.02"}
                ],
                "expected_error": "align",
            },
            {
                "id": "numeric_threshold_requires_number",
                "base_case": "accept_and_refer",
                "mutations": [{"path": "/thresholds/1/value", "value": "6"}],
                "expected_error": "finite numeric",
            },
            {
                "id": "membership_threshold_requires_array",
                "base_case": "accept_and_refer",
                "mutations": [{"path": "/thresholds/0/value", "value": "accept"}],
                "expected_error": "array",
            },
            {
                "id": "domain_score_is_not_decision",
                "base_case": "accept_and_refer",
                "mutations": [{"path": "/domain_score", "value": 0.7}],
                "expected_error": "domain_score",
            },
            {
                "id": "runtime_handle_refuses",
                "base_case": "accept_and_refer",
                "mutations": [{"path": "/policy_handle", "value": "opaque"}],
                "expected_error": "handle",
            },
            {
                "id": "decision_without_evidence_refuses",
                "base_case": "accept_and_refer",
                "mutations": [
                    {"path": "/conformal_block_fingerprint", "value": None},
                    {"path": "/domain_assessment_fingerprint", "value": None},
                ],
                "expected_error": "evidence",
            },
        ],
    }
    return domain_fixture, decision_fixture


def result_coordinate(result: dict[str, Any]) -> tuple[Any, ...]:
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


def slice_units(
    cohort_document: dict[str, Any], slice_key: dict[str, Any]
) -> list[str]:
    if slice_key["kind"] in {"all", "environment", "target"}:
        return cohort_document["physical_sample_ids"]
    member = "group_ids" if slice_key["kind"] == "group" else "source_ids"
    return [
        relation["physical_sample_id"]
        for relation in cohort_document["unit_relations"]
        if slice_key["value"] in relation[member]
    ]


def point_metrics(*, severity: float, delta: float) -> list[dict[str, Any]]:
    baselines = {"mae": 0.4, "r2": 0.85, "rmse": 0.6}
    directions = {"mae": "minimize", "r2": "maximize", "rmse": "minimize"}
    records = []
    for metric in sorted(baselines):
        baseline = baselines[metric]
        value = (
            baseline
            if severity == 0.0
            else baseline + delta
            if directions[metric] == "minimize"
            else baseline - delta
        )
        degradation = (
            value - baseline if directions[metric] == "minimize" else baseline - value
        )
        records.append(
            {
                "metric": metric,
                "direction": directions[metric],
                "status": "finite",
                "value": value,
                "baseline_value": baseline,
                "degradation": degradation,
            }
        )
    return records


def report_block_and_metrics(
    *,
    coordinate_label: str,
    artifact: dict[str, Any],
    cohort_document: dict[str, Any],
    unit_ids: list[str],
    point_predictions: list[list[float]],
    truth: list[list[float]],
    scenario_id: str,
    severity: float,
    slice_key: dict[str, Any],
    guarantee_status: str,
    assumption_status: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intervals = []
    for quantile in artifact["quantiles"]:
        coverage = quantile["coverage"]
        tagged_radius = quantile["values"][0]
        if tagged_radius["status"] == "unbounded":
            lower = [[None] for _point in point_predictions]
            upper = [[None] for _point in point_predictions]
        else:
            radius = tagged_radius["value"]
            lower = [
                [float(Decimal(repr(row[0])) - Decimal(repr(radius)))]
                for row in point_predictions
            ]
            upper = [
                [float(Decimal(repr(row[0])) + Decimal(repr(radius)))]
                for row in point_predictions
            ]
        intervals.append(
            {
                "coverage": coverage,
                "lower": lower,
                "upper": upper,
            }
        )
    point_fingerprint = tcv1_sha256(point_predictions)
    block = make_prediction_block(
        block_id=f"conformal:block.{coordinate_label}",
        artifact_id=artifact["artifact_id"],
        artifact_checksum=artifact["checksum"],
        predictor_fingerprint=artifact["predictor_binding_fingerprint"],
        cohort_fingerprint=cohort_document["manifest_fingerprint"],
        point_fingerprint=point_fingerprint,
        output_binding=artifact["predictor_binding"]["output_binding"],
        unit_ids=unit_ids,
        policy="marginal",
        intervals=intervals,
        assumption_status=assumption_status,
        guarantee_status=guarantee_status,
    )
    records = metric_records_from_truth(
        block=block,
        truth=truth,
        scenario_id=scenario_id,
        severity=severity,
        slice_key=slice_key,
        seed=17,
        guarantee_status=guarantee_status,
    )
    metric_set = make_metric_set(
        metric_set_id=f"metrics:{coordinate_label}",
        block=block,
        cohort_fingerprint=cohort_document["manifest_fingerprint"],
        truth_fingerprint=tcv1_sha256(truth),
        records=records,
    )
    evidence = make_numeric_evidence(
        evidence_id=f"evidence:{coordinate_label}",
        block=block,
        metric_set=metric_set,
        point_predictions=point_predictions,
        truth=truth,
    )
    return block, metric_set, evidence


def conformal_report(
    external: dict[str, Any],
    scenario_documents: list[dict[str, Any]],
    artifacts: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base, matched, structural = artifacts
    scenario_map = {
        scenario["scenario_id"]: scenario for scenario in scenario_documents
    }
    specs = [
        ("scenario:clean.noise", 0.0, {"kind": "all", "value": None}),
        ("scenario:clean.noise", 0.0, {"kind": "group", "value": "group:batch.A"}),
        ("scenario:clean.noise", 0.0, {"kind": "group", "value": "group:batch.B"}),
        ("scenario:clean.noise", 0.0, {"kind": "source", "value": "nir"}),
        ("scenario:clean.noise", 0.0, {"kind": "source", "value": "nir.secondary"}),
        ("scenario:clean.noise", 0.01, {"kind": "all", "value": None}),
        ("scenario:clean.noise", 0.01, {"kind": "group", "value": "group:batch.A"}),
        ("scenario:clean.noise", 0.01, {"kind": "group", "value": "group:batch.B"}),
        ("scenario:clean.noise", 0.01, {"kind": "source", "value": "nir"}),
        ("scenario:clean.noise", 0.01, {"kind": "source", "value": "nir.secondary"}),
        ("scenario:matched.noise", 0.0, {"kind": "all", "value": None}),
        ("scenario:matched.noise", 0.0, {"kind": "group", "value": "group:batch.A"}),
        ("scenario:matched.noise", 0.0, {"kind": "group", "value": "group:batch.B"}),
        ("scenario:matched.noise", 0.5, {"kind": "all", "value": None}),
        ("scenario:matched.noise", 0.5, {"kind": "group", "value": "group:batch.A"}),
        ("scenario:matched.noise", 0.5, {"kind": "group", "value": "group:batch.B"}),
        ("scenario:structural.node", 0.0, {"kind": "all", "value": None}),
        ("scenario:structural.node", 1.0, {"kind": "all", "value": None}),
    ]
    results: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    metric_sets: list[dict[str, Any]] = []
    evidence_cases: list[dict[str, Any]] = []
    for index, (scenario_id, severity, slice_key) in enumerate(specs):
        scenario = scenario_map[scenario_id]
        unit_ids = slice_units(external, slice_key)
        baseline_label = (
            f"{scenario_id}:{slice_key['kind']}:{slice_key['value'] or 'all'}"
        )
        before_input = opaque(f"input:baseline:{baseline_label}")
        baseline_points = [[20.0 + index] for index in range(len(unit_ids))]
        before_point = tcv1_sha256(baseline_points)
        if severity == 0.0:
            after_input = before_input
            after_points = baseline_points
            after_predictor = base["predictor_binding_fingerprint"]
            after_artifact = base
            predictor_status = "reused"
            calibration_status = "reused"
            delta = 0.0
        else:
            delta = {
                "clean_frozen": 0.1,
                "matched_recalibration": 0.2,
                "structural_refit": 0.3,
            }[scenario["mode"]]
            after_input = (
                before_input
                if scenario["mode"] == "structural_refit"
                else opaque(f"input:shifted:{baseline_label}:{severity}")
            )
            after_points = [[row[0] + delta] for row in baseline_points]
            after_predictor = (
                structural["predictor_binding_fingerprint"]
                if scenario["mode"] == "structural_refit"
                else base["predictor_binding_fingerprint"]
            )
            after_artifact = {
                "clean_frozen": base,
                "matched_recalibration": matched,
                "structural_refit": structural,
            }[scenario["mode"]]
            predictor_status = (
                "refit" if scenario["mode"] == "structural_refit" else "reused"
            )
            calibration_status = (
                "reused" if scenario["mode"] == "clean_frozen" else "recalibrated"
            )
        after_point = tcv1_sha256(after_points)
        formal = (
            slice_key["kind"] == "all"
            and scenario["mode"] != "clean_frozen"
            or (severity == 0.0 and slice_key["kind"] == "all")
        )
        guarantee = "marginal_coverage" if formal else "diagnostic_only"
        assumption = (
            "declared_exchangeable"
            if formal
            else "distribution_shift"
            if severity > 0.0
            else "not_assessed"
        )
        coordinate_label = (
            f"r{index:02d}.{scenario_id.split(':')[-1]}.{slice_key['kind']}"
        )
        truth = [[row[0] + 0.5] for row in after_points]
        block, metric_set, evidence = report_block_and_metrics(
            coordinate_label=coordinate_label,
            artifact=after_artifact,
            cohort_document=external,
            unit_ids=unit_ids,
            point_predictions=after_points,
            truth=truth,
            scenario_id=scenario_id,
            severity=severity,
            slice_key=slice_key,
            guarantee_status=guarantee,
            assumption_status=assumption,
        )
        blocks.append(block)
        metric_sets.append(metric_set)
        evidence_cases.append(evidence)
        metrics = point_metrics(severity=severity, delta=delta)
        rmse = next(record["value"] for record in metrics if record["metric"] == "rmse")
        result = {
            "scenario_id": scenario_id,
            "severity": severity,
            "split_id": scenario["split_id"],
            "environment_id": scenario["environment_id"],
            "fold_id": None,
            "repeat_id": None,
            "seed": 17,
            "slice": copy.deepcopy(slice_key),
            "unit_level": "physical_sample",
            "unit_ids": unit_ids,
            "unit_count": len(unit_ids),
            "point_metrics": metrics,
            "conformal_metric_set_id": metric_set["metric_set_id"],
            "confidence_intervals": [
                {
                    "metric_family": "conformal",
                    "metric": "empirical_coverage",
                    "coverage": 0.9,
                    "target_name": "protein",
                    "level": 0.95,
                    "lower": 0.8,
                    "upper": 1.0,
                    "method": "paired_bootstrap",
                },
                {
                    "metric_family": "point",
                    "metric": "rmse",
                    "coverage": None,
                    "target_name": None,
                    "level": 0.95,
                    "lower": max(0.0, rmse - 0.1),
                    "upper": rmse + 0.1,
                    "method": "paired_bootstrap",
                },
            ],
            "errors": [],
            "predictor_status": predictor_status,
            "calibration_status": calibration_status,
            "coverage_guarantee_status": guarantee,
            "before_predictor_fingerprint": base["predictor_binding_fingerprint"],
            "after_predictor_fingerprint": after_predictor,
            "before_input_fingerprint": before_input,
            "after_input_fingerprint": after_input,
            "before_relation_fingerprint": external["relation_fingerprint"],
            "after_relation_fingerprint": external["relation_fingerprint"],
            "before_point_prediction_fingerprint": before_point,
            "after_point_prediction_fingerprint": after_point,
            "conformal_prediction_block_fingerprint": block["block_fingerprint"],
            "before_calibration_checksum": base["checksum"],
            "after_calibration_checksum": after_artifact["checksum"],
        }
        results.append(result)
    report = {
        "schema_version": 1,
        "report_id": "robustness:external.three_modes",
        "plan_fingerprint": opaque("plan:robustness.external"),
        "predictor_binding_fingerprint": base["predictor_binding_fingerprint"],
        "calibration_artifact_checksum": base["checksum"],
        "calibration_artifacts": sorted(
            [copy.deepcopy(artifact) for artifact in artifacts],
            key=lambda artifact: artifact["artifact_id"],
        ),
        "cohort_manifest": copy.deepcopy(external),
        "scenarios": copy.deepcopy(scenario_documents),
        "conformal_prediction_blocks": sorted(
            blocks, key=lambda block: block["block_id"]
        ),
        "conformal_metric_sets": sorted(
            metric_sets, key=lambda metric_set: metric_set["metric_set_id"]
        ),
        "results": sorted(results, key=result_coordinate),
        "provenance": {
            "run_ids": ["run:robustness.external"],
            "artifact_checksums": sorted(
                artifact["checksum"] for artifact in artifacts
            ),
            "relation_fingerprint": external["relation_fingerprint"],
        },
        "warnings": [],
        "report_fingerprint": "0" * 64,
    }
    set_fingerprint(report, "report_fingerprint")
    return report, sorted(evidence_cases, key=lambda item: item["evidence_id"])


def production_report(production: dict[str, Any]) -> dict[str, Any]:
    scenario = make_scenario(
        scenario_id="scenario:production.noise",
        mode="clean_frozen",
        role="production",
        source_ids=["nir"],
        node_ids=[],
        perturbation_kind="gaussian_noise",
        severities=[0.0, 0.1],
        environment_id="environment:production.A",
        slice_by=[],
        target_kind="source",
    )
    scenario["metrics"] = ["prediction_mean_shift"]
    set_fingerprint(scenario, "scenario_fingerprint")
    predictor = opaque("predictor:production.point-only")
    before_input = opaque("input:production.baseline")
    before_point = opaque("point:production.baseline")
    results = []
    for severity in scenario["severities"]:
        shifted = severity > 0.0
        shift = 0.1 if shifted else 0.0
        metrics = [
            {
                "metric": "prediction_mean_shift",
                "direction": "minimize",
                "status": "finite",
                "value": shift,
                "baseline_value": 0.0,
                "degradation": shift,
            }
        ]
        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "severity": severity,
                "split_id": scenario["split_id"],
                "environment_id": scenario["environment_id"],
                "fold_id": None,
                "repeat_id": None,
                "seed": 17,
                "slice": {"kind": "all", "value": None},
                "unit_level": "physical_sample",
                "unit_ids": production["physical_sample_ids"],
                "unit_count": len(production["physical_sample_ids"]),
                "point_metrics": metrics,
                "conformal_metric_set_id": None,
                "confidence_intervals": [
                    {
                        "metric_family": "point",
                        "metric": "prediction_mean_shift",
                        "coverage": None,
                        "target_name": None,
                        "level": 0.95,
                        "lower": max(0.0, shift - 0.01),
                        "upper": shift + 0.01,
                        "method": "rolling_block_bootstrap",
                    }
                ],
                "errors": [],
                "predictor_status": "reused",
                "calibration_status": "absent",
                "coverage_guarantee_status": "unavailable",
                "before_predictor_fingerprint": predictor,
                "after_predictor_fingerprint": predictor,
                "before_input_fingerprint": before_input,
                "after_input_fingerprint": opaque("input:production.shifted")
                if shifted
                else before_input,
                "before_relation_fingerprint": production["relation_fingerprint"],
                "after_relation_fingerprint": production["relation_fingerprint"],
                "before_point_prediction_fingerprint": before_point,
                "after_point_prediction_fingerprint": opaque("point:production.shifted")
                if shifted
                else before_point,
                "conformal_prediction_block_fingerprint": None,
                "before_calibration_checksum": None,
                "after_calibration_checksum": None,
            }
        )
    report = {
        "schema_version": 1,
        "report_id": "robustness:production.point-only",
        "plan_fingerprint": opaque("plan:production.point-only"),
        "predictor_binding_fingerprint": predictor,
        "calibration_artifact_checksum": None,
        "calibration_artifacts": [],
        "cohort_manifest": copy.deepcopy(production),
        "scenarios": [scenario],
        "conformal_prediction_blocks": [],
        "conformal_metric_sets": [],
        "results": sorted(results, key=result_coordinate),
        "provenance": {
            "run_ids": ["run:production.point-only"],
            "artifact_checksums": [],
            "relation_fingerprint": production["relation_fingerprint"],
        },
        "warnings": ["coverage_unavailable_without_calibration"],
        "report_fingerprint": "0" * 64,
    }
    set_fingerprint(report, "report_fingerprint")
    return report


def cohort_fixture(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "fixture_id": "dag-ml.conformal.cohort-manifest-roles.v1",
        "schema_version": 1,
        "schema": "cohort_manifest.schema.json",
        "valid_cases": [
            {"id": role, "document": documents[role]}
            for role in ("development", "calibration", "external_test", "production")
        ],
        "invalid_cases": [
            {
                "id": "unknown_test_role_refuses",
                "base_case": "external_test",
                "mutations": [{"path": "/role", "value": "test"}],
                "expected_error": "role",
            },
            {
                "id": "unsorted_samples_refuse",
                "base_case": "external_test",
                "mutations": [
                    {
                        "path": "/physical_sample_ids",
                        "value": [
                            "sample:ext.02",
                            "sample:ext.01",
                            "sample:ext.03",
                            "sample:ext.04",
                        ],
                    }
                ],
                "expected_error": "sorted",
            },
            {
                "id": "unit_relation_order_refuses",
                "base_case": "external_test",
                "mutations": [
                    {
                        "path": "/unit_relations/0/physical_sample_id",
                        "value": "sample:ext.02",
                    }
                ],
                "expected_error": "align",
            },
            {
                "id": "unit_relation_source_closure_refuses",
                "base_case": "external_test",
                "mutations": [{"path": "/unit_relations/3/source_ids", "value": []}],
                "expected_error": "source closure",
            },
            {
                "id": "runtime_handle_refuses",
                "base_case": "external_test",
                "mutations": [{"path": "/cohort_handle", "value": 1}],
                "expected_error": "handle",
            },
            {
                "id": "manifest_fingerprint_tamper_refuses",
                "base_case": "external_test",
                "mutations": [{"path": "/manifest_fingerprint", "value": "f" * 64}],
                "expected_error": "TCV1",
                "targets_fingerprint": True,
            },
        ],
        "disjointness_cases": [
            {
                "id": "disjoint_training_and_calibration",
                "training_sample_ids": ["sample:dev.01", "sample:dev.02"],
                "training_origin_sample_ids": ["sample:dev.origin.01"],
                "calibration_case": "calibration",
                "expected": "valid",
            },
            {
                "id": "direct_sample_overlap",
                "training_sample_ids": ["sample:cal.01", "sample:dev.02"],
                "training_origin_sample_ids": [],
                "calibration_case": "calibration",
                "expected": "invalid",
                "expected_error": "overlap",
            },
            {
                "id": "origin_overlap",
                "training_sample_ids": ["sample:dev.01"],
                "training_origin_sample_ids": ["sample:cal.origin.01"],
                "calibration_case": "calibration",
                "expected": "invalid",
                "expected_error": "overlap",
            },
        ],
    }


def scenario_fixture(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fixture_id": "dag-ml.robustness.scenarios.v1",
        "schema_version": 1,
        "schema": "robustness_scenario_spec.schema.json",
        "valid_cases": [
            {"id": "clean_frozen", "document": documents[0]},
            {"id": "matched_recalibration", "document": documents[1]},
            {"id": "structural_refit", "document": documents[2]},
        ],
        "invalid_cases": [
            {
                "id": "legacy_recalibrated_name_refuses",
                "base_case": "matched_recalibration",
                "mutations": [{"path": "/mode", "value": "recalibrated"}],
                "expected_error": "mode",
            },
            {
                "id": "unsorted_severities_refuse",
                "base_case": "clean_frozen",
                "mutations": [{"path": "/severities", "value": [0.01, 0.0]}],
                "expected_error": "severities",
            },
            {
                "id": "integer_severity_refuses",
                "base_case": "clean_frozen",
                "mutations": [{"path": "/severities", "value": [0, 0.01]}],
                "expected_error": "binary64",
            },
            {
                "id": "clean_frozen_refit_refuses",
                "base_case": "clean_frozen",
                "mutations": [{"path": "/requires_refit", "value": True}],
                "expected_error": "policy",
            },
            {
                "id": "matched_without_recalibration_refuses",
                "base_case": "matched_recalibration",
                "mutations": [{"path": "/requires_recalibration", "value": False}],
                "expected_error": "policy",
            },
            {
                "id": "structural_without_node_replacement_refuses",
                "base_case": "structural_refit",
                "mutations": [
                    {"path": "/perturbation/kind", "value": "gaussian_noise"}
                ],
                "expected_error": "node_replacement",
                "recompute_fingerprints": True,
            },
            {
                "id": "node_replacement_requires_structural_mode",
                "base_case": "clean_frozen",
                "mutations": [
                    {"path": "/perturbation/kind", "value": "node_replacement"}
                ],
                "expected_error": "if and only if",
                "recompute_fingerprints": True,
            },
            {
                "id": "identity_requires_zero_only_grid",
                "base_case": "clean_frozen",
                "mutations": [
                    {"path": "/perturbation/kind", "value": "identity"},
                    {"path": "/rng/target_kind", "value": "global"},
                ],
                "expected_error": "[0.0]",
                "recompute_fingerprints": True,
            },
            {
                "id": "rng_algorithm_drift_refuses",
                "base_case": "clean_frozen",
                "mutations": [{"path": "/rng/algorithm", "value": "counter_based"}],
                "expected_error": "algorithm",
            },
            {
                "id": "rng_target_kind_refuses",
                "base_case": "structural_refit",
                "mutations": [{"path": "/rng/target_kind", "value": "source"}],
                "expected_error": "target_kind",
            },
            {
                "id": "parameter_runtime_handle_refuses",
                "base_case": "clean_frozen",
                "mutations": [
                    {"path": "/perturbation/parameters/model_handle", "value": 9}
                ],
                "expected_error": "handle",
            },
            {
                "id": "zero_severity_must_be_identity",
                "base_case": "clean_frozen",
                "mutations": [
                    {"path": "/zero_severity_semantics", "value": "baseline_only"}
                ],
                "expected_error": "identity",
            },
            {
                "id": "nirs_specific_shift_is_host_owned",
                "base_case": "matched_recalibration",
                "mutations": [
                    {"path": "/perturbation/kind", "value": "spectrometer_transfer"}
                ],
                "expected_error": "not one of",
            },
        ],
    }


def structural_invalidated_report(
    external_report: dict[str, Any], evidence_cases: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = copy.deepcopy(external_report)
    report["report_id"] = "robustness:external.structural-invalidated"
    report["scenarios"] = [
        scenario
        for scenario in report["scenarios"]
        if scenario["mode"] == "structural_refit"
    ]
    report["results"] = [
        result
        for result in report["results"]
        if result["scenario_id"] == "scenario:structural.node"
    ]
    baseline_result = next(
        result for result in report["results"] if result["severity"] == 0.0
    )
    shifted_result = next(
        result for result in report["results"] if result["severity"] == 1.0
    )
    shifted_result["calibration_status"] = "invalidated"
    shifted_result["coverage_guarantee_status"] = "unavailable"
    shifted_result["after_calibration_checksum"] = None
    shifted_result["conformal_prediction_block_fingerprint"] = None
    shifted_result["conformal_metric_set_id"] = None
    shifted_result["confidence_intervals"] = [
        interval
        for interval in shifted_result["confidence_intervals"]
        if interval["metric_family"] == "point"
    ]
    shifted_result["errors"] = [
        {
            "code": "metric_unavailable.conformal_coverage",
            "message": "conformal coverage is unavailable after calibration invalidation",
            "phase": "score",
            "retriable": False,
        },
        {
            "code": "metric_unavailable.mean_width",
            "message": "interval width is unavailable after calibration invalidation",
            "phase": "score",
            "retriable": False,
        },
    ]
    baseline_block_fingerprint = baseline_result[
        "conformal_prediction_block_fingerprint"
    ]
    baseline_metric_id = baseline_result["conformal_metric_set_id"]
    report["conformal_prediction_blocks"] = [
        block
        for block in report["conformal_prediction_blocks"]
        if block["block_fingerprint"] == baseline_block_fingerprint
    ]
    report["conformal_metric_sets"] = [
        metric_set
        for metric_set in report["conformal_metric_sets"]
        if metric_set["metric_set_id"] == baseline_metric_id
    ]
    base_checksum = report["calibration_artifact_checksum"]
    report["calibration_artifacts"] = [
        artifact
        for artifact in report["calibration_artifacts"]
        if artifact["checksum"] == base_checksum
    ]
    report["provenance"]["artifact_checksums"] = [base_checksum]
    report["warnings"] = ["structural_calibration_invalidated"]
    set_fingerprint(report, "report_fingerprint")
    filtered_evidence = [
        evidence
        for evidence in evidence_cases
        if evidence["block_fingerprint"] == baseline_block_fingerprint
    ]
    return report, filtered_evidence


def report_fixture(
    external_report: dict[str, Any],
    external_evidence: list[dict[str, Any]],
    point_only_report: dict[str, Any],
    invalidated_report: dict[str, Any],
    invalidated_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    results = external_report["results"]
    matched_positive = next(
        index
        for index, result in enumerate(results)
        if result["scenario_id"] == "scenario:matched.noise"
        and result["severity"] == 0.5
        and result["slice"]["kind"] == "all"
    )
    structural_positive = next(
        index
        for index, result in enumerate(results)
        if result["scenario_id"] == "scenario:structural.node"
        and result["severity"] == 1.0
    )
    clean_positive = next(
        index
        for index, result in enumerate(results)
        if result["scenario_id"] == "scenario:clean.noise"
        and result["severity"] == 0.01
        and result["slice"]["kind"] == "all"
    )
    invalidated_positive = next(
        index
        for index, result in enumerate(invalidated_report["results"])
        if result["severity"] == 1.0
    )
    structural_scenario = next(
        (index, scenario)
        for index, scenario in enumerate(external_report["scenarios"])
        if scenario["mode"] == "structural_refit"
    )
    structural_scenario_index, scenario_with_unknown_node = structural_scenario
    scenario_with_unknown_node = copy.deepcopy(scenario_with_unknown_node)
    scenario_with_unknown_node["node_ids"] = ["node:outside.predictor"]
    set_fingerprint(scenario_with_unknown_node, "scenario_fingerprint")
    wrong_metric_units = "f" * 64
    metric_with_wrong_units = copy.deepcopy(external_report["conformal_metric_sets"][0])
    metric_with_wrong_units["records"][0]["unit_ids_fingerprint"] = wrong_metric_units
    set_fingerprint(metric_with_wrong_units, "metric_set_fingerprint")

    clean_block_fingerprint = results[clean_positive][
        "conformal_prediction_block_fingerprint"
    ]
    clean_block_index = next(
        index
        for index, block in enumerate(external_report["conformal_prediction_blocks"])
        if block["block_fingerprint"] == clean_block_fingerprint
    )
    clean_shift_report = copy.deepcopy(external_report)
    clean_shift_block = clean_shift_report["conformal_prediction_blocks"][
        clean_block_index
    ]
    clean_shift_block["guarantee_status"] = "marginal_coverage"
    set_fingerprint(clean_shift_block, "block_fingerprint")
    clean_shift_metric = next(
        metric_set
        for metric_set in clean_shift_report["conformal_metric_sets"]
        if metric_set["conformal_prediction_block_fingerprint"]
        == clean_block_fingerprint
    )
    clean_shift_metric["conformal_prediction_block_fingerprint"] = clean_shift_block[
        "block_fingerprint"
    ]
    for record in clean_shift_metric["records"]:
        record["guarantee_status"] = "marginal_coverage"
    set_fingerprint(clean_shift_metric, "metric_set_fingerprint")
    clean_shift_result = next(
        result
        for result in clean_shift_report["results"]
        if result["conformal_prediction_block_fingerprint"] == clean_block_fingerprint
    )
    clean_shift_result["conformal_prediction_block_fingerprint"] = clean_shift_block[
        "block_fingerprint"
    ]
    clean_shift_result["coverage_guarantee_status"] = "marginal_coverage"
    set_fingerprint(clean_shift_report, "report_fingerprint")
    unknown_coverage_block = copy.deepcopy(
        external_report["conformal_prediction_blocks"][0]
    )
    unknown_coverage_block["intervals"][0]["coverage"] = 0.85
    set_fingerprint(unknown_coverage_block, "block_fingerprint")
    wrong_radius_block = copy.deepcopy(
        external_report["conformal_prediction_blocks"][0]
    )
    wrong_radius_block["intervals"][0]["lower"][0][0] += 0.1
    set_fingerprint(wrong_radius_block, "block_fingerprint")
    integer_endpoint_block = copy.deepcopy(
        external_report["conformal_prediction_blocks"][0]
    )
    integer_endpoint_block["intervals"][0]["lower"][0][0] = 18
    set_fingerprint(integer_endpoint_block, "block_fingerprint")
    wrong_width_metric = copy.deepcopy(external_report["conformal_metric_sets"][0])
    wrong_width_metric["records"][0]["mean_width"] += 0.1
    set_fingerprint(wrong_width_metric, "metric_set_fingerprint")
    integer_metric = copy.deepcopy(external_report["conformal_metric_sets"][0])
    integer_metric["records"][0]["empirical_coverage"] = 1
    set_fingerprint(integer_metric, "metric_set_fingerprint")

    def cascade_block_change(mutated_block: dict[str, Any]) -> dict[str, Any]:
        mutated_report = copy.deepcopy(external_report)
        original_block = external_report["conformal_prediction_blocks"][0]
        original_fingerprint = original_block["block_fingerprint"]
        mutated_report["conformal_prediction_blocks"][0] = mutated_block
        metric_set = next(
            metric_set
            for metric_set in mutated_report["conformal_metric_sets"]
            if metric_set["conformal_prediction_block_fingerprint"]
            == original_fingerprint
        )
        evidence = next(
            evidence
            for evidence in external_evidence
            if evidence["block_fingerprint"] == original_fingerprint
        )
        metric_set["conformal_prediction_block_fingerprint"] = mutated_block[
            "block_fingerprint"
        ]
        for record, interval in zip(metric_set["records"], mutated_block["intervals"]):
            record["coverage"] = interval["coverage"]
            metric_interval = copy.deepcopy(interval)
            for endpoint in ("lower", "upper"):
                metric_interval[endpoint] = [
                    [float(value) if isinstance(value, int) else value for value in row]
                    for row in metric_interval[endpoint]
                ]
            summary = regression_conformal_metrics(
                evidence["truth"],
                metric_interval,
                multi_target_policy=mutated_block["multi_target_policy"],
            )[0]
            for field in (
                "measurement_status",
                "empirical_coverage",
                "coverage_gap",
                "mean_width",
                "median_width",
                "interval_score",
            ):
                record[field] = summary[field]
        set_fingerprint(metric_set, "metric_set_fingerprint")
        result = next(
            result
            for result in mutated_report["results"]
            if result["conformal_prediction_block_fingerprint"] == original_fingerprint
        )
        result["conformal_prediction_block_fingerprint"] = mutated_block[
            "block_fingerprint"
        ]
        for interval in result["confidence_intervals"]:
            if interval["metric_family"] == "conformal":
                interval["coverage"] = mutated_block["intervals"][0]["coverage"]
        set_fingerprint(mutated_report, "report_fingerprint")
        return mutated_report

    unknown_coverage_report = cascade_block_change(unknown_coverage_block)
    wrong_radius_report = cascade_block_change(wrong_radius_block)
    integer_endpoint_report = cascade_block_change(integer_endpoint_block)

    missing_slice_report = copy.deepcopy(external_report)
    missing_result = next(
        result
        for result in missing_slice_report["results"]
        if result["scenario_id"] == "scenario:clean.noise"
        and result["severity"] == 0.01
        and result["slice"] == {"kind": "group", "value": "group:batch.B"}
    )
    missing_block_fingerprint = missing_result["conformal_prediction_block_fingerprint"]
    missing_metric_id = missing_result["conformal_metric_set_id"]
    missing_slice_report["results"].remove(missing_result)
    missing_slice_report["conformal_prediction_blocks"] = [
        block
        for block in missing_slice_report["conformal_prediction_blocks"]
        if block["block_fingerprint"] != missing_block_fingerprint
    ]
    missing_slice_report["conformal_metric_sets"] = [
        metric_set
        for metric_set in missing_slice_report["conformal_metric_sets"]
        if metric_set["metric_set_id"] != missing_metric_id
    ]
    set_fingerprint(missing_slice_report, "report_fingerprint")

    missing_requested_metric_scenario = copy.deepcopy(point_only_report["scenarios"][0])
    missing_requested_metric_scenario["metrics"] = sorted(
        [*missing_requested_metric_scenario["metrics"], "rmse"]
    )
    set_fingerprint(missing_requested_metric_scenario, "scenario_fingerprint")

    mutated_truth_evidence = copy.deepcopy(external_evidence[0])
    mutated_truth_evidence["truth"][0][0] += 10.0
    mutated_truth_evidence["truth_fingerprint"] = tcv1_sha256(
        mutated_truth_evidence["truth"]
    )
    set_fingerprint(mutated_truth_evidence, "evidence_fingerprint")

    midpoint_report = copy.deepcopy(external_report)
    midpoint_evidence = copy.deepcopy(external_evidence[0])
    midpoint_block_index = next(
        index
        for index, block in enumerate(midpoint_report["conformal_prediction_blocks"])
        if block["block_fingerprint"] == midpoint_evidence["block_fingerprint"]
    )
    midpoint_block = midpoint_report["conformal_prediction_blocks"][
        midpoint_block_index
    ]
    for endpoint in ("lower", "upper"):
        original_endpoint = midpoint_block["intervals"][0][endpoint][0][0]
        midpoint_block["intervals"][0][endpoint][0][0] = float(
            Decimal(repr(original_endpoint)) + Decimal("0.1")
        )
    set_fingerprint(midpoint_block, "block_fingerprint")
    midpoint_metric_index = next(
        index
        for index, metric_set in enumerate(midpoint_report["conformal_metric_sets"])
        if metric_set["metric_set_id"] == midpoint_evidence["metric_set_id"]
    )
    midpoint_metric = midpoint_report["conformal_metric_sets"][midpoint_metric_index]
    midpoint_metric["conformal_prediction_block_fingerprint"] = midpoint_block[
        "block_fingerprint"
    ]
    set_fingerprint(midpoint_metric, "metric_set_fingerprint")
    midpoint_result_index = next(
        index
        for index, result in enumerate(midpoint_report["results"])
        if result["conformal_metric_set_id"] == midpoint_evidence["metric_set_id"]
    )
    midpoint_report["results"][midpoint_result_index][
        "conformal_prediction_block_fingerprint"
    ] = midpoint_block["block_fingerprint"]
    set_fingerprint(midpoint_report, "report_fingerprint")
    midpoint_evidence["block_fingerprint"] = midpoint_block["block_fingerprint"]
    set_fingerprint(midpoint_evidence, "evidence_fingerprint")
    return {
        "fixture_id": "dag-ml.robustness.reports.v1",
        "schema_version": 1,
        "schema": "robustness_report.schema.json",
        "valid_cases": [
            {"id": "three_modes_resolved_conformal", "document": external_report},
            {
                "id": "structural_calibration_invalidated",
                "document": invalidated_report,
            },
            {"id": "production_point_only", "document": point_only_report},
        ],
        "evidence_sets": [
            {
                "report_case": "structural_calibration_invalidated",
                "records": invalidated_evidence,
            },
            {
                "report_case": "three_modes_resolved_conformal",
                "records": external_evidence,
            },
        ],
        "invalid_evidence_cases": [
            {
                "id": "truth_change_reconstructs_different_metrics",
                "report_case": "three_modes_resolved_conformal",
                "base_evidence_id": external_evidence[0]["evidence_id"],
                "mutations": [
                    {"path": "/truth", "value": mutated_truth_evidence["truth"]},
                    {
                        "path": "/truth_fingerprint",
                        "value": mutated_truth_evidence["truth_fingerprint"],
                    },
                    {
                        "path": "/evidence_fingerprint",
                        "value": mutated_truth_evidence["evidence_fingerprint"],
                    },
                ],
                "rebind_metric_truth": True,
                "expected_error": "reconstruct",
            },
            {
                "id": "translated_interval_breaks_point_midpoint",
                "report_case": "three_modes_resolved_conformal",
                "base_evidence_id": external_evidence[0]["evidence_id"],
                "report_mutations": [
                    {
                        "path": f"/conformal_prediction_blocks/{midpoint_block_index}/intervals/0/lower/0/0",
                        "value": midpoint_block["intervals"][0]["lower"][0][0],
                    },
                    {
                        "path": f"/conformal_prediction_blocks/{midpoint_block_index}/intervals/0/upper/0/0",
                        "value": midpoint_block["intervals"][0]["upper"][0][0],
                    },
                    {
                        "path": f"/conformal_prediction_blocks/{midpoint_block_index}/block_fingerprint",
                        "value": midpoint_block["block_fingerprint"],
                    },
                    {
                        "path": f"/conformal_metric_sets/{midpoint_metric_index}/conformal_prediction_block_fingerprint",
                        "value": midpoint_block["block_fingerprint"],
                    },
                    {
                        "path": f"/conformal_metric_sets/{midpoint_metric_index}/metric_set_fingerprint",
                        "value": midpoint_metric["metric_set_fingerprint"],
                    },
                    {
                        "path": f"/results/{midpoint_result_index}/conformal_prediction_block_fingerprint",
                        "value": midpoint_block["block_fingerprint"],
                    },
                    {
                        "path": "/report_fingerprint",
                        "value": midpoint_report["report_fingerprint"],
                    },
                ],
                "mutations": [
                    {
                        "path": "/block_fingerprint",
                        "value": midpoint_evidence["block_fingerprint"],
                    },
                    {
                        "path": "/evidence_fingerprint",
                        "value": midpoint_evidence["evidence_fingerprint"],
                    },
                ],
                "expected_error": "midpoint",
            },
        ],
        "invalid_cases": [
            {
                "id": "unknown_scenario_result_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": f"/results/{len(results) - 1}/scenario_id",
                        "value": "scenario:zzzz",
                    }
                ],
                "expected_error": "unknown scenario",
            },
            {
                "id": "structural_node_must_resolve_in_predictor_closure",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": f"/scenarios/{structural_scenario_index}/node_ids",
                        "value": scenario_with_unknown_node["node_ids"],
                    },
                    {
                        "path": f"/scenarios/{structural_scenario_index}/scenario_fingerprint",
                        "value": scenario_with_unknown_node["scenario_fingerprint"],
                    },
                ],
                "expected_error": "predictor closure",
                "recompute_fingerprints": True,
            },
            {
                "id": "matched_reuses_stale_calibrator",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": f"/results/{matched_positive}/calibration_status",
                        "value": "reused",
                    }
                ],
                "expected_error": "matched",
                "recompute_fingerprints": True,
            },
            {
                "id": "structural_reuses_stale_predictor",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": f"/results/{structural_positive}/predictor_status",
                        "value": "reused",
                    }
                ],
                "expected_error": "structural predictor",
                "recompute_fingerprints": True,
            },
            {
                "id": "invalidated_structural_cannot_reuse_calibrator",
                "base_case": "structural_calibration_invalidated",
                "mutations": [
                    {
                        "path": f"/results/{invalidated_positive}/calibration_status",
                        "value": "reused",
                    }
                ],
                "expected_error": "calibration status",
                "recompute_fingerprints": True,
            },
            {
                "id": "clean_with_calibrator_cannot_be_absent",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": f"/results/{clean_positive}/calibration_status",
                        "value": "absent",
                    }
                ],
                "expected_error": "clean_frozen calibration",
                "recompute_fingerprints": True,
            },
            {
                "id": "confidence_interval_inverted",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {"path": "/results/0/confidence_intervals/0/lower", "value": 1.1}
                ],
                "expected_error": "inverted",
            },
            {
                "id": "runtime_handle_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [{"path": "/report_handle", "value": "opaque"}],
                "expected_error": "handle",
            },
            {
                "id": "severity_zero_input_change_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {"path": "/results/0/after_input_fingerprint", "value": "f" * 64}
                ],
                "expected_error": "severity zero",
            },
            {
                "id": "clean_shift_cannot_claim_coverage",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_prediction_blocks",
                        "value": clean_shift_report["conformal_prediction_blocks"],
                    },
                    {
                        "path": "/conformal_metric_sets",
                        "value": clean_shift_report["conformal_metric_sets"],
                    },
                    {"path": "/results", "value": clean_shift_report["results"]},
                ],
                "expected_error": "overclaims",
            },
            {
                "id": "unknown_calibration_coverage_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_prediction_blocks",
                        "value": unknown_coverage_report["conformal_prediction_blocks"],
                    },
                    {
                        "path": "/conformal_metric_sets",
                        "value": unknown_coverage_report["conformal_metric_sets"],
                    },
                    {
                        "path": "/results",
                        "value": unknown_coverage_report["results"],
                    },
                ],
                "expected_error": "not calibrated",
                "recompute_fingerprints": True,
            },
            {
                "id": "prediction_radius_must_match_quantile",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_prediction_blocks",
                        "value": wrong_radius_report["conformal_prediction_blocks"],
                    },
                    {
                        "path": "/conformal_metric_sets",
                        "value": wrong_radius_report["conformal_metric_sets"],
                    },
                    {
                        "path": "/results",
                        "value": wrong_radius_report["results"],
                    },
                ],
                "expected_error": "radius",
                "recompute_fingerprints": True,
            },
            {
                "id": "metric_width_must_reconstruct_from_bounds",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_metric_sets/0/records",
                        "value": wrong_width_metric["records"],
                    },
                    {
                        "path": "/conformal_metric_sets/0/metric_set_fingerprint",
                        "value": wrong_width_metric["metric_set_fingerprint"],
                    },
                ],
                "expected_error": "reconstruct",
                "recompute_fingerprints": True,
            },
            {
                "id": "integer_prediction_endpoint_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_prediction_blocks",
                        "value": integer_endpoint_report["conformal_prediction_blocks"],
                    },
                    {
                        "path": "/conformal_metric_sets",
                        "value": integer_endpoint_report["conformal_metric_sets"],
                    },
                    {
                        "path": "/results",
                        "value": integer_endpoint_report["results"],
                    },
                ],
                "expected_error": "binary64",
                "recompute_fingerprints": True,
            },
            {
                "id": "integer_metric_value_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_metric_sets/0/records",
                        "value": integer_metric["records"],
                    },
                    {
                        "path": "/conformal_metric_sets/0/metric_set_fingerprint",
                        "value": integer_metric["metric_set_fingerprint"],
                    },
                ],
                "expected_error": "binary64",
                "recompute_fingerprints": True,
            },
            {
                "id": "integer_confidence_interval_refuses",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {"path": "/results/0/confidence_intervals/0/lower", "value": 0}
                ],
                "expected_error": "binary64",
                "recompute_fingerprints": True,
            },
            {
                "id": "regression_confidence_interval_cannot_target_set_size",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/results/0/confidence_intervals/0/metric",
                        "value": "set_size",
                    }
                ],
                "expected_error": "conformal CI",
                "recompute_fingerprints": True,
            },
            {
                "id": "requested_metric_cannot_disappear",
                "base_case": "production_point_only",
                "mutations": [
                    {
                        "path": "/scenarios/0/metrics",
                        "value": missing_requested_metric_scenario["metrics"],
                    },
                    {
                        "path": "/scenarios/0/scenario_fingerprint",
                        "value": missing_requested_metric_scenario[
                            "scenario_fingerprint"
                        ],
                    },
                ],
                "expected_error": "requested",
                "recompute_fingerprints": True,
            },
            {
                "id": "declared_slice_value_cannot_disappear",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {"path": "/results", "value": missing_slice_report["results"]},
                    {
                        "path": "/conformal_prediction_blocks",
                        "value": missing_slice_report["conformal_prediction_blocks"],
                    },
                    {
                        "path": "/conformal_metric_sets",
                        "value": missing_slice_report["conformal_metric_sets"],
                    },
                ],
                "expected_error": "slice",
                "recompute_fingerprints": True,
            },
            {
                "id": "point_metric_direction_controls_degradation",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": f"/results/{clean_positive}/point_metrics/1/degradation",
                        "value": -0.1,
                    }
                ],
                "expected_error": "degradation",
            },
            {
                "id": "metric_units_must_match_result",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/conformal_metric_sets/0/records/0/unit_ids_fingerprint",
                        "value": wrong_metric_units,
                    },
                    {
                        "path": "/conformal_metric_sets/0/metric_set_fingerprint",
                        "value": metric_with_wrong_units["metric_set_fingerprint"],
                    },
                ],
                "expected_error": "units",
            },
            {
                "id": "provenance_must_cover_calibrators",
                "base_case": "three_modes_resolved_conformal",
                "mutations": [
                    {
                        "path": "/provenance/artifact_checksums",
                        "value": [external_report["calibration_artifact_checksum"]],
                    }
                ],
                "expected_error": "provenance",
            },
            {
                "id": "point_only_cannot_claim_conformal_coverage",
                "base_case": "production_point_only",
                "mutations": [
                    {
                        "path": "/results/1/coverage_guarantee_status",
                        "value": "marginal_coverage",
                    }
                ],
                "expected_error": "overclaims",
            },
        ],
    }


def conformance_pack() -> dict[str, Any]:
    artifacts = []
    for relative_path in sorted(PACK_ARTIFACTS):
        path = ROOT / relative_path
        artifacts.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": PACK_ARTIFACTS[relative_path],
            }
        )
    pack = {
        "pack_id": "dag-ml.conformal-robustness-conformance.v1",
        "schema_version": 1,
        "hash_algorithm": "sha256-file-bytes",
        "fingerprint_profile": "DAGML-TCV1",
        "canonical_profiles": [
            {
                "id": "DAGML-TCV1",
                "object_key_order": "utf8",
                "unicode_normalization": "NFC",
                "number_profile": "typed integer-or-binary64",
            },
            {
                "id": "RFC8785-JCS-restricted",
                "object_key_order": "utf16",
                "unicode_normalization": "none",
                "number_profile": "integer-and-tagged-binary64-labels",
            },
        ],
        "artifacts": artifacts,
        "conformance_cases": sorted(
            [
                {
                    "id": "cohort_roles_and_leakage",
                    "fixture": "examples/fixtures/conformal/cohort_manifest_roles.v1.json",
                    "invariants": sorted(
                        [
                            "four_canonical_roles",
                            "no_runtime_handle",
                            "physical_sample_unit",
                            "sample_origin_disjointness",
                        ]
                    ),
                },
                {
                    "id": "conformal_metrics_by_slice",
                    "fixture": "examples/fixtures/conformal/conformal_metric_sets.v1.json",
                    "invariants": sorted(
                        [
                            "coverage_gap",
                            "midpoint_reconstruction",
                            "numeric_evidence_reconstruction",
                            "separate_from_score_set",
                            "slice_coordinates",
                            "target_policy",
                        ]
                    ),
                },
                {
                    "id": "cross_language_tcv1_restricted_jcs",
                    "fixture": "parity/canonical/golden/tcv1_jcs_cross_language.v1.json",
                    "invariants": sorted(
                        [
                            "cross_language_digest_identity",
                            "jcs_utf16_key_order",
                            "profile_separation",
                            "tcv1_utf8_key_order",
                        ]
                    ),
                },
                {
                    "id": "domain_and_decision_separation",
                    "fixture": "examples/fixtures/conformal/domain_assessment_blocks.v1.json",
                    "invariants": sorted(
                        [
                            "identity_alignment",
                            "no_coverage_claim",
                            "no_decision_action",
                        ]
                    ),
                },
                {
                    "id": "exact_rank_and_multi_target_oracle",
                    "fixture": "parity/conformal/golden/split_absolute_residual.v1.json",
                    "invariants": sorted(
                        [
                            "joint_max",
                            "marginal",
                            "shortest_binary64_token",
                            "small_n_policy",
                            "tcv1_utf8_key_order",
                            "ties",
                        ]
                    ),
                },
                {
                    "id": "explicit_application_decisions",
                    "fixture": "examples/fixtures/conformal/decision_blocks.v1.json",
                    "invariants": sorted(
                        ["identity_alignment", "no_domain_score", "policy_binding"]
                    ),
                },
                {
                    "id": "multi_target_nested_intervals",
                    "fixture": "examples/fixtures/conformal/conformal_prediction_blocks.v1.json",
                    "invariants": sorted(
                        [
                            "bound_nesting",
                            "coverage_order",
                            "no_domain_conflation",
                            "output_binding_ref",
                            "target_order",
                        ]
                    ),
                },
                {
                    "id": "regression_conformal_metric_reconstruction",
                    "fixture": "parity/conformal/golden/regression_conformal_metrics.v1.json",
                    "invariants": sorted(
                        [
                            "coverage_reconstruction",
                            "interval_score_reconstruction",
                            "joint_and_marginal",
                            "unbounded_metrics",
                            "width_reconstruction",
                        ]
                    ),
                },
                {
                    "id": "robustness_group_source_report",
                    "fixture": "examples/fixtures/robustness/robustness_reports.v1.json",
                    "invariants": sorted(
                        [
                            "calibrator_invalidation",
                            "confidence_interval_order",
                            "exact_18_coordinate_matrix",
                            "exact_severity_zero_baseline",
                            "group_slice",
                            "metric_unavailable_accounting",
                            "point_only_production",
                            "predictor_reuse",
                            "slice_value_completeness",
                            "source_slice",
                        ]
                    ),
                },
                {
                    "id": "robustness_philox_counter_profile",
                    "fixture": "parity/robustness_rng/golden/philox4x32_10_counter.v1.json",
                    "invariants": sorted(
                        [
                            "binary64_severity",
                            "counter_tcv1_sha256",
                            "philox4x32_10_known_answers",
                            "portable_seed_key_words",
                            "target_identity",
                        ]
                    ),
                },
                {
                    "id": "three_robustness_modes",
                    "fixture": "examples/fixtures/robustness/robustness_scenarios.v1.json",
                    "invariants": sorted(
                        [
                            "clean_frozen",
                            "matched_recalibration",
                            "philox_counter_rng",
                            "severity_zero",
                            "structural_refit",
                        ]
                    ),
                },
            ],
            key=lambda case: case["id"],
        ),
        "pack_checksum": "0" * 64,
    }
    set_fingerprint(pack, "pack_checksum")
    return pack


def generate(*, output_root: Path = ROOT, include_pack: bool = True) -> None:
    estimator_dir = ROOT / "examples" / "fixtures" / "estimator"
    refit_outcome_path = estimator_dir / "training_outcome_refit.v1.json"
    no_refit_outcome_path = estimator_dir / "training_outcome_no_refit.v1.json"
    refit_outcome = load_json(refit_outcome_path)
    no_refit_outcome = load_json(no_refit_outcome_path)
    # Derive the honest replayable phases for both source outcomes from the graph
    # (never hardcoded) before the fingerprint cascade recomputes their
    # self-fingerprints. A completed refit that supports PREDICT with retained
    # inference state is PREDICT-replayable; a self-contained skipped refit is
    # REFIT-replayable.
    refit_outcome["replayable_phases"] = derive_source_outcome_replayable_phases(
        refit_outcome
    )
    no_refit_outcome["replayable_phases"] = derive_source_outcome_replayable_phases(
        no_refit_outcome
    )
    assert refit_outcome["replayable_phases"] == ["PREDICT"], (
        "refit source outcome must derive PREDICT-only replay, got "
        f"{refit_outcome['replayable_phases']}"
    )
    assert no_refit_outcome["replayable_phases"] == ["REFIT"], (
        "no-refit source outcome must derive REFIT-only replay, got "
        f"{no_refit_outcome['replayable_phases']}"
    )
    restore_explicit_sample_bundle_wire(refit_outcome)
    restore_explicit_sample_bundle_wire(no_refit_outcome)
    if output_root == ROOT:
        write_json(refit_outcome_path, refit_outcome)
        write_json(no_refit_outcome_path, no_refit_outcome)

    conformal_dir = output_root / "examples" / "fixtures" / "conformal"
    robustness_dir = output_root / "examples" / "fixtures" / "robustness"
    calibration_fixture_path = (
        conformal_dir / "split_absolute_residual_physical_sample.v1.json"
    )
    calibration_artifact_fixture_path = conformal_dir / "calibration_artifacts.v1.json"
    calibration_fixture = load_json(calibration_fixture_path)
    calibration_fixture["calibration_artifact"]["predictor_binding"][
        "training_outcome_fingerprint"
    ] = refit_outcome["outcome_fingerprint"]
    cohorts = role_cohorts()
    artifacts = calibration_artifacts(calibration_fixture)
    unbounded_artifact = unbounded_calibration_artifact(artifacts[0])
    calibration_fixture["calibration_artifact"] = artifacts[0]

    scenario_documents = scenarios()
    prediction_fixture, metric_fixture = standalone_prediction_and_metrics(
        cohorts["external_test"]
    )
    domain_fixture, decision_fixture = domain_and_decision_fixtures(
        cohorts["external_test"], artifacts[0]["predictor_binding_fingerprint"]
    )
    full_report, full_evidence = conformal_report(
        cohorts["external_test"], scenario_documents, artifacts
    )
    invalidated_report, invalidated_evidence = structural_invalidated_report(
        full_report, full_evidence
    )
    point_only = production_report(cohorts["production"])
    calibration_artifacts_fixture = calibration_artifact_fixture(
        artifacts[0], unbounded_artifact
    )
    cohorts_fixture = cohort_fixture(cohorts)
    scenarios_fixture = scenario_fixture(scenario_documents)
    reports_fixture = report_fixture(
        full_report,
        full_evidence,
        point_only,
        invalidated_report,
        invalidated_evidence,
    )
    for fixture in (
        calibration_artifacts_fixture,
        cohorts_fixture,
        prediction_fixture,
        metric_fixture,
        domain_fixture,
        decision_fixture,
        scenarios_fixture,
        reports_fixture,
    ):
        enforce_invalid_fingerprint_policy(fixture)

    write_json(calibration_fixture_path, calibration_fixture)
    write_json(
        calibration_artifact_fixture_path,
        calibration_artifacts_fixture,
    )
    write_json(conformal_dir / "cohort_manifest_roles.v1.json", cohorts_fixture)
    write_json(
        conformal_dir / "conformal_prediction_blocks.v1.json", prediction_fixture
    )
    write_json(conformal_dir / "conformal_metric_sets.v1.json", metric_fixture)
    write_json(conformal_dir / "domain_assessment_blocks.v1.json", domain_fixture)
    write_json(conformal_dir / "decision_blocks.v1.json", decision_fixture)
    write_json(
        robustness_dir / "robustness_scenarios.v1.json",
        scenarios_fixture,
    )
    write_json(
        robustness_dir / "robustness_reports.v1.json",
        reports_fixture,
    )
    if include_pack:
        if output_root != ROOT:
            raise ValueError("pack generation requires the repository output root")
        write_json(PACK_PATH, conformance_pack())


if __name__ == "__main__":
    generate()
