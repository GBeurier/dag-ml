#!/usr/bin/env python3
"""Deterministically generate the W1-0 training-contract fixtures.

This generator is test-only. It imports the independent Python TCV1 oracle,
never DAG-ML production code, and rebuilds every new fingerprint from leaves.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parity.conformal.generate_fixtures import (  # noqa: E402
    derive_source_outcome_replayable_phases,
)
from parity.conformal.oracle import fingerprint_without, load_json, tcv1_sha256  # noqa: E402
from parity.schema_dependencies import (  # noqa: E402
    with_transitive_schema_dependencies,
)
from parity.training.oracle import (  # noqa: E402
    _normalize_campaign_spec,
    _normalize_controller_manifests,
    _normalize_execution_plan,
    _normalize_graph_spec,
    _serde_sha256,
)

OUT = ROOT / "examples" / "fixtures" / "training"
PACK_PATH = ROOT / "docs" / "contracts" / "training_contract_conformance_pack.v1.json"

BASE_PACK_ARTIFACTS = {
    ".github/workflows/ci.yml": "ci_gate",
    "crates/dag-ml-cli/tests/training_contracts.rs": "binding_test",
    "crates/dag-ml-capi/include/dag_ml.h": "c_abi_header",
    "crates/dag-ml-capi/src/lib.rs": "c_training_binding",
    "crates/dag-ml-capi/tests/c_conformance.rs": "c_binding_test",
    "crates/dag-ml-capi/tests/training_execute.rs": "c_training_test",
    "crates/dag-ml-core/src/bundle.rs": "native_bundle_contract",
    "crates/dag-ml-core/src/aggregation.rs": "typed_cache_dependency",
    "crates/dag-ml-core/src/campaign.rs": "historical_fingerprint_source",
    "crates/dag-ml-core/src/canonical.rs": "native_canonical_fingerprint",
    "crates/dag-ml-core/src/controller.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/data.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/fold.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/generation.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/graph.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/ids.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/metrics.rs": "typed_cache_dependency",
    "crates/dag-ml-core/src/oof.rs": "typed_cache_dependency",
    "crates/dag-ml-core/src/phase.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/plan.rs": "native_plan_validator",
    "crates/dag-ml-core/src/policy.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/relation.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/runtime/artifact.rs": "typed_runtime_dependency",
    "crates/dag-ml-core/src/runtime/task.rs": "typed_runtime_dependency",
    "crates/dag-ml-core/src/selection.rs": "typed_contract_dependency",
    "crates/dag-ml-core/src/training.rs": "native_contract",
    "crates/dag-ml-core/src/training_runtime.rs": "native_runtime_contract",
    "crates/dag-ml-core/tests/training_runtime_operation.rs": "native_integration_test",
    "crates/dag-ml-py/README.md": "python_binding_documentation",
    "crates/dag-ml-py/python/dag_ml/__init__.py": "python_facade",
    "crates/dag-ml-py/python/dag_ml/__init__.pyi": "python_typing",
    "crates/dag-ml-py/src/in_process.rs": "python_controller_bridge",
    "crates/dag-ml-py/src/lib.rs": "python_binding",
    "crates/dag-ml-py/src/training.rs": "python_training_binding",
    "crates/dag-ml-py/tests/test_training_result.py": "python_binding_test",
    "crates/dag-ml-wasm/src/lib.rs": "wasm_binding",
    "docs/TRAINING_CONTRACTS.md": "documentation",
    "docs/contracts/abi_snapshot.v1.json": "c_abi_snapshot",
    "docs/contracts/README.md": "contract_index_documentation",
    "docs/contracts/cache_namespace.schema.json": "schema",
    "docs/contracts/campaign_spec.schema.json": "schema_dependency",
    "docs/contracts/controller_manifest.schema.json": "schema_dependency",
    "docs/contracts/coordinator_data_plan_envelope.schema.json": "schema_dependency",
    "docs/contracts/execution_plan.schema.json": "schema_dependency",
    "docs/contracts/graph_spec.schema.json": "schema_dependency",
    "docs/contracts/node_result.schema.json": "schema_dependency",
    "docs/contracts/node_task.schema.json": "schema_dependency",
    "docs/contracts/output_binding.schema.json": "schema_dependency",
    "docs/contracts/parameter_patch.schema.json": "schema_dependency",
    "docs/contracts/parameter_projection.schema.json": "schema",
    "docs/contracts/portable_predictor_package.schema.json": "schema",
    "docs/contracts/prediction_cache_payload_set.schema.json": "schema_dependency",
    "docs/contracts/score_set.schema.json": "schema_dependency",
    "docs/contracts/selection_policy.schema.json": "schema_dependency",
    "docs/contracts/training_influence_manifest.schema.json": "schema_dependency",
    "docs/contracts/training_outcome.schema.json": "schema_dependency",
    "docs/contracts/training_request.schema.json": "schema",
    "examples/fixtures/training/cache_namespace_fit_cv.v1.json": "fixture",
    "examples/fixtures/training/negative_cases.v1.json": "negative_fixture",
    "examples/fixtures/training/parameter_projection_empty.v1.json": "fixture",
    "examples/fixtures/training/portable_predictor_package.v1.json": "fixture",
    "examples/fixtures/training/python_training_multiport_smoke.v1.json": "binding_smoke_fixture",
    "examples/fixtures/training/python_training_smoke.v1.json": "binding_smoke_fixture",
    "examples/fixtures/training/training_request_active_influence.v1.json": "fixture",
    "examples/fixtures/training/training_request_no_refit.v1.json": "fixture",
    "examples/fixtures/training/training_request_package_refit.v1.json": "fixture",
    "examples/fixtures/training/training_request_refit.v1.json": "fixture",
    "examples/fixtures/training/training_outcome_refit.v1.json": "fixture",
    "examples/fixtures/estimator/training_outcome_no_refit.v1.json": "source_fixture",
    "examples/fixtures/estimator/training_outcome_refit.v1.json": "source_fixture",
    # This generator imports `derive_source_outcome_replayable_phases` from the
    # conformal generator, so its bytes are a true generator dependency: hashing it
    # keeps the training pack's dependency closure honest (editing that helper must
    # invalidate this pack, not only the conformal one).
    "parity/conformal/generate_fixtures.py": "generator_dependency",
    "parity/conformal/oracle.py": "tcv1_oracle_dependency",
    "parity/schema_dependencies.py": "schema_dependency_resolver",
    "parity/training/generate_fixtures.py": "generator",
    "parity/training/oracle.py": "test_oracle",
    "parity/training/tests/test_training_contracts.py": "test",
    "scripts/check_so_freshness.py": "python_extension_freshness_gate",
    "scripts/smoke_python_bindings.py": "binding_smoke",
    "scripts/validate_abi_snapshot.py": "c_abi_gate",
    "scripts/validate_contracts.py": "production_validator",
}

PACK_ARTIFACTS = with_transitive_schema_dependencies(ROOT, BASE_PACK_ARTIFACTS)

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
            "supports_configurable_loss",
            "supports_custom_loss",
            "supports_differentiable_loss",
            "uses_training_weights",
            "uses_early_stopping",
            "performs_internal_tuning",
            "trains_aggregation",
        )
    )
}
INFLUENCE_ORDER = {
    kind: index
    for index, kind in enumerate(
        (
            "transform_fit",
            "model_fit",
            "hpo_selection",
            "early_stopping",
            "weighting_resampling",
            "trained_meta_aggregation",
        )
    )
}


def sort_influence_entries(entries: list[dict[str, Any]]) -> None:
    entries.sort(
        key=lambda entry: (
            INFLUENCE_ORDER[entry["kind"]],
            entry["scope_id"],
            (0, "") if entry["node_id"] is None else (1, entry["node_id"]),
        )
    )


def opaque(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_identity(requirement: dict[str, Any], label: str) -> dict[str, Any]:
    identity = {
        "requirement_key": f"{requirement['node_id']}.{requirement['input_name']}",
        "schema_fingerprint": requirement["schema_fingerprint"],
        "plan_fingerprint": requirement["plan_fingerprint"],
        "relation_fingerprint": requirement["relation_fingerprint"],
        "data_content_fingerprint": opaque(f"{label}:data"),
        "target_content_fingerprint": opaque(f"{label}:targets"),
        "identity_fingerprint": "0" * 64,
    }
    identity["identity_fingerprint"] = fingerprint_without(
        identity, "identity_fingerprint"
    )
    return identity


def training_request(*, refit: bool) -> dict[str, Any]:
    graph = load_json(ROOT / "examples" / "minimal_graph.json")
    campaign = load_json(ROOT / "examples" / "campaign_oof_generation.json")
    manifests = [
        manifest
        for manifest in load_json(ROOT / "examples" / "controller_manifests.json")
        if manifest["operator_kind"] in {"model", "transform"}
    ]
    manifests.sort(key=lambda manifest: manifest["controller_id"])
    for manifest in manifests:
        manifest["capabilities"].sort(key=CAPABILITY_ORDER.__getitem__)
    binding = campaign["data_bindings"]["model:base"][0]
    identity = data_identity(binding, "minimal-training")
    request = {
        "schema_version": 1,
        "request_id": f"training:fixture.{'refit' if refit else 'no_refit'}",
        "plan_id": "plan:training.fixture",
        "graph": graph,
        "campaign": campaign,
        "controller_manifests": manifests,
        "data_identities": [identity],
        "parameter_patches": [],
        "patch_policies": [],
        "influence_requirements": [],
        "options": {
            "refit": refit,
            "refit_strategy": "refit_one" if refit else None,
            "seed": 12345,
            "selection": {
                "id": "selection:rmse",
                "metric": {"name": "rmse", "objective": "minimize"},
                "require_finite": True,
            },
            "selection_output_id": "output:prediction",
            "outputs": [
                {
                    "output_id": "output:prediction",
                    "node_id": "model:base",
                    "prediction_level": "sample",
                    "unit_level": "physical_sample",
                    "prediction_kind": "regression_point",
                    "target_names": ["protein"],
                    "target_units": ["percent"],
                    "class_labels": [[]],
                    "output_order": "target_order",
                    "target_space": "raw",
                }
            ],
            "scheduler": {"kind": "sequential", "backend": None, "workers": 1},
            "resources": {
                "cpu_threads": 1,
                "memory_bytes": 1024,
                "gpu_devices": [],
                "wall_time_ms": 10000,
            },
            "artifacts": {
                "cv_artifacts": "metadata_only",
                "prediction_caches": "retain",
                "fitted_artifacts": "allow_host_sidecar",
            },
        },
        "request_fingerprint": "0" * 64,
    }
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    return request


def active_influence_training_request() -> dict[str, Any]:
    """Build a refit request exercising every fold/refit influence slot.

    The model actively consumes early-stopping samples and training weights.
    Early-stopping samples are strict subsets of the corresponding fit cohort;
    weighting samples exactly cover it.
    """

    request = training_request(refit=True)
    request["request_id"] = "training:fixture.active_influence"
    model = next(
        manifest
        for manifest in request["controller_manifests"]
        if manifest["controller_id"] == "controller:model.mock"
    )
    model["capabilities"].extend(
        [
            "supports_sample_weights",
            "uses_training_weights",
            "uses_early_stopping",
        ]
    )
    model["capabilities"].sort(key=CAPABILITY_ORDER.__getitem__)
    request["influence_requirements"] = [
        {
            "node_id": "model:base",
            "kind": "early_stopping",
            "scope_id": "early:fit_cv:fold:0",
            "phase": "FIT_CV",
            "fold_id": "fold:0",
            "physical_sample_ids": ["sample:3"],
        },
        {
            "node_id": "model:base",
            "kind": "early_stopping",
            "scope_id": "early:fit_cv:fold:1",
            "phase": "FIT_CV",
            "fold_id": "fold:1",
            "physical_sample_ids": ["sample:1"],
        },
        {
            "node_id": "model:base",
            "kind": "early_stopping",
            "scope_id": "early:refit",
            "phase": "REFIT",
            "fold_id": None,
            "physical_sample_ids": ["sample:1", "sample:2"],
        },
        {
            "node_id": "model:base",
            "kind": "weighting_resampling",
            "scope_id": "weights:fit_cv:fold:0",
            "phase": "FIT_CV",
            "fold_id": "fold:0",
            "physical_sample_ids": ["sample:3", "sample:4"],
        },
        {
            "node_id": "model:base",
            "kind": "weighting_resampling",
            "scope_id": "weights:fit_cv:fold:1",
            "phase": "FIT_CV",
            "fold_id": "fold:1",
            "physical_sample_ids": ["sample:1", "sample:2"],
        },
        {
            "node_id": "model:base",
            "kind": "weighting_resampling",
            "scope_id": "weights:refit",
            "phase": "REFIT",
            "fold_id": None,
            "physical_sample_ids": [
                "sample:1",
                "sample:2",
                "sample:3",
                "sample:4",
            ],
        },
    ]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    return request


def python_training_smoke_contract(
    *, explicit_multi_output: bool = False
) -> dict[str, Any]:
    """Build a complete signed contract for the public Python success smoke."""

    relations = {
        "records": [
            {
                "unit_level": "observation",
                "unit_id": None,
                "observation_id": f"observation:{index}",
                "sample_id": f"sample:{index}",
                "source_id": None,
                "rep_id": None,
                "target_id": None,
                "group_id": "group:0" if index <= 2 else "group:1",
                "origin_sample_id": None,
                "derived_unit_id": None,
                "component_observation_ids": [],
                "sample_influence_weight": None,
                "quality_flag": None,
                "is_augmented": False,
            }
            for index in range(1, 5)
        ]
    }
    canonical_relations = [
        {
            "effective_unit_id": relation["observation_id"],
            "unit_level": relation["unit_level"],
            "unit_id": relation["unit_id"],
            "observation_id": relation["observation_id"],
            "sample_id": relation["sample_id"],
            "source_id": relation["source_id"],
            "rep_id": relation["rep_id"],
            "target_id": relation["target_id"],
            "group_id": relation["group_id"],
            "origin_sample_id": relation["origin_sample_id"],
            "derived_unit_id": relation["derived_unit_id"],
            "component_observation_ids": relation["component_observation_ids"],
            "sample_influence_weight": relation["sample_influence_weight"],
            "quality_flag": relation["quality_flag"],
            "is_augmented": relation["is_augmented"],
        }
        for relation in relations["records"]
    ]
    canonical_relations.sort(
        key=lambda relation: (
            relation["effective_unit_id"],
            relation["observation_id"],
            relation["sample_id"],
        )
    )
    relation_fingerprint = _serde_sha256(canonical_relations)

    request = training_request(refit=True)
    request["request_id"] = (
        "training:fixture.python_multiport_smoke"
        if explicit_multi_output
        else "training:fixture.python_smoke"
    )
    request["campaign"]["generation"] = {
        "strategy": "none",
        "dimensions": [],
        "max_variants": 1,
    }
    selection = request["options"]["selection"]
    selection["required_metric_level"] = "sample"
    selection["evaluation_scope"] = "oof"
    request["options"]["resources"].pop("memory_bytes", None)
    request["options"]["resources"].pop("wall_time_ms", None)
    request["options"]["artifacts"]["cv_artifacts"] = "discard"
    if explicit_multi_output:
        model_node = next(
            node for node in request["graph"]["nodes"] if node["id"] == "model:base"
        )
        probability_port = copy.deepcopy(
            next(port for port in model_node["ports"]["outputs"] if port["name"] == "oof")
        )
        probability_port["name"] = "probability"
        model_node["ports"]["outputs"].append(probability_port)
        model_manifest = next(
            manifest
            for manifest in request["controller_manifests"]
            if manifest["controller_id"] == "controller:model.mock"
        )
        model_manifest["output_ports"].append(copy.deepcopy(probability_port))
        request["options"]["outputs"][0]["port_name"] = "oof"
    binding = request["campaign"]["data_bindings"]["model:base"][0]
    binding["relation_fingerprint"] = relation_fingerprint
    identity = data_identity(
        binding,
        "python-training-multiport-smoke"
        if explicit_multi_output
        else "python-training-smoke",
    )
    request["data_identities"] = [identity]
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")

    fold_set = request["campaign"]["split_invocation"]["fold_set"]
    all_samples = sorted(fold_set["sample_ids"])
    groups_by_sample = {
        relation["sample_id"]: relation["group_id"] for relation in relations["records"]
    }

    def influence_entry(
        kind: str, scope_id: str, node_id: str | None, sample_ids: list[str]
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "scope_id": scope_id,
            "node_id": node_id,
            "physical_sample_ids": sorted(sample_ids),
            "origin_sample_ids": [],
            "group_ids": sorted({groups_by_sample[sample] for sample in sample_ids}),
        }

    entries: list[dict[str, Any]] = []
    for kind, node_id in [
        ("transform_fit", "transform:snv"),
        ("model_fit", "model:base"),
    ]:
        for fold in fold_set["folds"]:
            entries.append(
                influence_entry(
                    kind,
                    f"fit_cv:{fold['fold_id']}",
                    node_id,
                    fold["train_sample_ids"],
                )
            )
        entries.append(influence_entry(kind, "refit:full", node_id, all_samples))
    entries.append(
        influence_entry(
            "hpo_selection",
            f"select:{selection['id']}",
            None,
            all_samples,
        )
    )
    kind_order = {
        "transform_fit": 0,
        "model_fit": 1,
        "hpo_selection": 2,
        "early_stopping": 3,
        "weighting_resampling": 4,
        "trained_meta_aggregation": 5,
    }
    entries.sort(
        key=lambda entry: (
            kind_order[entry["kind"]],
            entry["scope_id"],
            entry["node_id"] or "",
        )
    )
    influence = {
        "schema_version": 1,
        "relation_fingerprint": relation_fingerprint,
        "entries": entries,
        "manifest_fingerprint": "0" * 64,
    }
    influence["manifest_fingerprint"] = fingerprint_without(
        influence, "manifest_fingerprint"
    )

    requirement_key = identity["requirement_key"]
    envelope = {
        "schema_version": 1,
        "schema_fingerprint": binding["schema_fingerprint"],
        "plan_fingerprint": binding["plan_fingerprint"],
        "relation_fingerprint": relation_fingerprint,
        "data_content_fingerprint": identity["data_content_fingerprint"],
        "target_content_fingerprint": identity["target_content_fingerprint"],
        "coordinator_relations": copy.deepcopy(relations),
    }
    return {
        "request": request,
        "data_envelopes": {requirement_key: envelope},
        "relations": relations,
        "training_influence": influence,
    }


def package_training_request() -> dict[str, Any]:
    """Build the signed request whose selected outcome feeds the package fixture."""

    outcome = load_json(
        ROOT / "examples" / "fixtures" / "estimator" / "training_outcome_refit.v1.json"
    )
    plan = outcome["effective_plan"]
    bindings = sorted(
        (
            binding
            for node_bindings in plan["campaign"].get("data_bindings", {}).values()
            for binding in node_bindings
        ),
        key=lambda binding: f"{binding['node_id']}.{binding['input_name']}",
    )
    policies: dict[str, set[str]] = {}
    for patch in outcome["parameter_patches"]:
        policies.setdefault(patch["node_id"], set()).add(patch["namespace"])
    namespace_order = {
        name: index
        for index, name in enumerate(("operator", "fit", "control", "structural"))
    }
    request = {
        "schema_version": 1,
        "request_id": "training:fixture.package_refit",
        "plan_id": plan["id"],
        "graph": copy.deepcopy(plan["graph_plan"]["graph"]),
        "campaign": copy.deepcopy(plan["campaign"]),
        "controller_manifests": sorted(
            (
                copy.deepcopy(manifest)
                for manifest in plan["controller_manifests"].values()
            ),
            key=lambda manifest: manifest["controller_id"],
        ),
        "data_identities": [
            data_identity(
                binding,
                f"package-training:{binding['node_id']}.{binding['input_name']}",
            )
            for binding in bindings
        ],
        "parameter_patches": copy.deepcopy(outcome["parameter_patches"]),
        "patch_policies": [
            {
                "node_id": node_id,
                "allowed_namespaces": sorted(
                    namespaces, key=namespace_order.__getitem__
                ),
            }
            for node_id, namespaces in sorted(policies.items())
        ],
        "influence_requirements": [],
        "options": {
            "refit": True,
            "refit_strategy": outcome["refit"]["strategy"],
            "seed": plan["campaign"]["root_seed"],
            "selection": {
                "id": "selection:outcome.rmse",
                "metric": {"name": "rmse", "objective": "minimize"},
                "require_finite": True,
            },
            "selection_output_id": outcome["outputs"][0]["binding"]["binding_id"],
            "outputs": [
                {
                    "output_id": bound["binding"]["binding_id"],
                    "node_id": bound["binding"]["node_id"],
                    "port_name": bound["binding"]["port_name"],
                    "prediction_level": bound["binding"]["prediction_level"],
                    "unit_level": bound["binding"]["unit_level"],
                    "prediction_kind": bound["binding"]["prediction_kind"],
                    "target_names": copy.deepcopy(bound["binding"]["target_names"]),
                    "target_units": copy.deepcopy(bound["binding"]["target_units"]),
                    "class_labels": copy.deepcopy(bound["binding"]["class_labels"]),
                    "output_order": bound["binding"]["output_order"],
                    "target_space": bound["binding"]["target_space"],
                }
                for bound in outcome["outputs"]
            ],
            "scheduler": {"kind": "sequential", "backend": None, "workers": 1},
            "resources": {
                "cpu_threads": 1,
                "memory_bytes": 1024,
                "gpu_devices": [],
                "wall_time_ms": 10000,
            },
            "artifacts": {
                "cv_artifacts": "metadata_only",
                "prediction_caches": "retain",
                "fitted_artifacts": "allow_host_sidecar",
            },
        },
        "request_fingerprint": "0" * 64,
    }
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    return request


def refingerprint_request(request: dict[str, Any]) -> dict[str, Any]:
    request["request_fingerprint"] = fingerprint_without(request, "request_fingerprint")
    return request


def refingerprint_package(package: dict[str, Any]) -> dict[str, Any]:
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")
    return package


def serde_json_sha256(value: Any) -> str:
    """SHA-256 of a pre-normalized value's exact ``serde_json::to_vec`` bytes."""

    return _serde_sha256(value)


def resign_training_outcome(
    outcome: dict[str, Any],
    *,
    graph: bool = True,
    campaign: bool = True,
    controllers: bool = True,
) -> dict[str, Any]:
    """Re-sign a mutated training outcome leaf-up.

    Recompute the plan's three embedded serde fingerprints from content (unless a
    flag is cleared to intentionally leave a stale/forged inner fingerprint for a
    negative), synchronize the bundle's plan-fingerprint crosslinks, then repair
    the TCV1 ``effective_plan_fingerprint`` and ``outcome_fingerprint``.
    """

    plan = outcome["effective_plan"]
    bundle = outcome["execution_bundle"]
    if graph:
        plan["graph_fingerprint"] = serde_json_sha256(
            _normalize_graph_spec(plan["graph_plan"]["graph"])
        )
    if campaign:
        plan["campaign_fingerprint"] = serde_json_sha256(
            _normalize_campaign_spec(plan["campaign"])
        )
    if controllers:
        plan["controller_fingerprint"] = serde_json_sha256(
            _normalize_controller_manifests(plan["controller_manifests"])
        )
    bundle["graph_fingerprint"] = plan["graph_fingerprint"]
    bundle["campaign_fingerprint"] = plan["campaign_fingerprint"]
    bundle["controller_fingerprint"] = plan["controller_fingerprint"]
    outcome["effective_plan_fingerprint"] = tcv1_sha256(_normalize_execution_plan(plan))
    outcome["outcome_fingerprint"] = fingerprint_without(outcome, "outcome_fingerprint")
    return outcome


def resign_package_plan(
    package: dict[str, Any],
    *,
    graph: bool = True,
    campaign: bool = True,
    controllers: bool = True,
) -> dict[str, Any]:
    """Re-sign a package after mutating its embedded plan/bundle, leaf-up.

    Recompute the plan serde fingerprints (unless a flag is cleared), mirror the
    template content + fingerprint, synchronize the bundle plan-fingerprint
    crosslinks and the training-outcome reference TCV1 crosslinks, then repair the
    outer ``package_fingerprint``.
    """

    plan = package["effective_plan"]
    bundle = package["execution_bundle"]
    template = package["template"]
    if graph:
        plan["graph_fingerprint"] = serde_json_sha256(
            _normalize_graph_spec(plan["graph_plan"]["graph"])
        )
    if campaign:
        plan["campaign_fingerprint"] = serde_json_sha256(
            _normalize_campaign_spec(plan["campaign"])
        )
    if controllers:
        plan["controller_fingerprint"] = serde_json_sha256(
            _normalize_controller_manifests(plan["controller_manifests"])
        )
    bundle["graph_fingerprint"] = plan["graph_fingerprint"]
    bundle["campaign_fingerprint"] = plan["campaign_fingerprint"]
    bundle["controller_fingerprint"] = plan["controller_fingerprint"]
    template["graph"] = plan["graph_plan"]["graph"]
    template["campaign"] = plan["campaign"]
    template["controller_manifests"] = plan["controller_manifests"]
    template["template_fingerprint"] = fingerprint_without(
        template, "template_fingerprint"
    )
    reference = package["training_outcome"]
    reference["effective_plan_fingerprint"] = tcv1_sha256(
        _normalize_execution_plan(plan)
    )
    reference["execution_bundle_fingerprint"] = tcv1_sha256(bundle)
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")
    return package


def sort_capabilities(capabilities: list[str]) -> list[str]:
    """Sort a capability list into canonical BTreeSet wire order."""

    return sorted(set(capabilities), key=CAPABILITY_ORDER.__getitem__)


def predictor_closure(plan: dict[str, Any], output_nodes: list[str]) -> list[str]:
    """Predictor closure rooted at the output nodes, from real graph edges.

    The walk uses the graph adjacency, never the serialized ``input_nodes``
    node-plan copies, so the package predictor closure is graph-derived.
    """

    incoming: dict[str, list[str]] = {}
    for edge in plan["graph_plan"]["graph"].get("edges", []):
        incoming.setdefault(edge["target"]["node_id"], []).append(
            edge["source"]["node_id"]
        )
    pending = list(output_nodes)
    closure: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in closure:
            continue
        closure.add(node_id)
        pending.extend(incoming.get(node_id, []))
    return sorted(closure)


def w1_training_outcome(request: dict[str, Any]) -> dict[str, Any]:
    outcome = load_json(
        ROOT / "examples" / "fixtures" / "estimator" / "training_outcome_refit.v1.json"
    )
    outcome["training_request_fingerprint"] = request["request_fingerprint"]
    outcome["data_identities"] = copy.deepcopy(request["data_identities"])
    outcome["selection_output_id"] = request["options"]["selection_output_id"]
    # Replayable phases are graph-derived from the effective plan closure, never
    # inherited blindly; a completed-refit PREDICT-only predictor is the honest
    # answer for this branch/merge fixture.
    outcome["replayable_phases"] = derive_source_outcome_replayable_phases(outcome)
    assert outcome["replayable_phases"] == ["PREDICT"], outcome["replayable_phases"]
    outcome["outcome_fingerprint"] = fingerprint_without(outcome, "outcome_fingerprint")
    return outcome


def portable_package(
    request: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    request_fingerprint = request["request_fingerprint"]
    plan = outcome["effective_plan"]
    bundle = copy.deepcopy(outcome["execution_bundle"])
    # The source W0 outcome carries the published explicit sample-level wire
    # fields for bundle prediction/cache records and portable cache payloads.
    template = {
        "graph": plan["graph_plan"]["graph"],
        "campaign": plan["campaign"],
        "controller_manifests": plan["controller_manifests"],
        "template_fingerprint": "0" * 64,
    }
    template["template_fingerprint"] = fingerprint_without(
        template, "template_fingerprint"
    )
    bindings = [bound["binding"] for bound in outcome["outputs"]]
    identities = copy.deepcopy(request["data_identities"])
    artifact_bindings = sorted(
        (
            {
                "artifact_id": record["artifact"]["id"],
                "load_mode": "host_sidecar",
            }
            for record in bundle["refit_artifacts"]
        ),
        key=lambda binding: binding["artifact_id"],
    )
    package = {
        "schema_version": 1,
        "package_id": "predictor:package.fixture",
        "template": template,
        "training_request_fingerprint": request_fingerprint,
        "training_outcome": {
            "outcome_id": outcome["outcome_id"],
            "outcome_fingerprint": outcome["outcome_fingerprint"],
            "training_request_fingerprint": request_fingerprint,
            "effective_plan_fingerprint": outcome["effective_plan_fingerprint"],
            "execution_bundle_id": bundle["bundle_id"],
            "execution_bundle_fingerprint": tcv1_sha256(bundle),
            "output_binding_fingerprints": [
                binding["binding_fingerprint"] for binding in bindings
            ],
            "training_influence_fingerprint": outcome["training_influence"][
                "manifest_fingerprint"
            ],
            "data_identities_fingerprint": tcv1_sha256(identities),
        },
        "effective_plan": plan,
        "execution_bundle": bundle,
        "output_bindings": bindings,
        "predictor_node_ids": predictor_closure(
            plan, [binding["node_id"] for binding in bindings]
        ),
        "training_influence": outcome["training_influence"],
        "data_identities": identities,
        "fitted_artifact_mode": "allow_host_sidecar",
        "artifact_bindings": artifact_bindings,
        "package_fingerprint": "0" * 64,
    }
    package["package_fingerprint"] = fingerprint_without(package, "package_fingerprint")
    return package


def cache_namespace(package: dict[str, Any]) -> dict[str, Any]:
    requirement = package["execution_bundle"]["prediction_requirements"][0]
    data_identity_value = package["data_identities"][0]
    prediction_key = (
        f"{requirement['producer_node']}.{requirement['source_port']}->"
        f"{requirement['consumer_node']}.{requirement['target_port']}"
    )
    namespace = {
        "schema_version": 1,
        "prediction_requirement_key": prediction_key,
        "data_requirement_key": data_identity_value["requirement_key"],
        "producer_node_id": requirement["producer_node"],
        "source_port_name": requirement["source_port"],
        "consumer_node_id": requirement["consumer_node"],
        "target_port_name": requirement["target_port"],
        "phase": "FIT_CV",
        "params_fingerprint": package["effective_plan"]["node_plans"][
            requirement["producer_node"]
        ]["params_fingerprint"],
        "data_identity_fingerprint": data_identity_value["identity_fingerprint"],
        "fold_id": requirement["fold_ids"][0],
        "trial_id": package["effective_plan"]["variants"][0]["variant_id"],
        "seed": package["effective_plan"]["variants"][0]["seed"],
        "namespace_fingerprint": "0" * 64,
    }
    namespace["namespace_fingerprint"] = fingerprint_without(
        namespace, "namespace_fingerprint"
    )
    return namespace


def empty_projection() -> dict[str, Any]:
    projection = {
        "schema_version": 1,
        "nodes": {
            "model:base": {
                "params": {"n_estimators": 100},
                "fit_params": {},
                "control_params": {},
                "structural_params": {},
            }
        },
        "requires_recompile": False,
        "structural_patch_count": 0,
        "patches_fingerprint": tcv1_sha256([]),
        "projection_fingerprint": "0" * 64,
    }
    projection["projection_fingerprint"] = fingerprint_without(
        projection, "projection_fingerprint"
    )
    return projection


def _first_stateless_closure_node(plan: dict[str, Any]) -> str:
    """A closure node that is neither stateful nor an artifact emitter."""

    for node_id, node_plan in plan["node_plans"].items():
        caps = node_plan["controller_capabilities"]
        if "stateful" not in caps and "emits_artifacts" not in caps:
            return node_id
    raise AssertionError("fixture has no stateless closure node")


def _mutate_manifest_and_node_plans(
    plan: dict[str, Any],
    controller_id: str,
    *,
    capabilities=None,
    supported_phases=None,
) -> None:
    """Apply the same capability/phase transform to a manifest and every node plan
    bound to it, so `node_plan == manifest` is preserved (the drift under test is
    the intended one, not an incidental plan/manifest mismatch)."""

    manifest = plan["controller_manifests"][controller_id]
    if capabilities is not None:
        manifest["capabilities"] = capabilities(manifest["capabilities"])
    if supported_phases is not None:
        manifest["supported_phases"] = supported_phases(manifest["supported_phases"])
    for node_plan in plan["node_plans"].values():
        if node_plan["controller_id"] != controller_id:
            continue
        if capabilities is not None:
            node_plan["controller_capabilities"] = capabilities(
                node_plan["controller_capabilities"]
            )
        if supported_phases is not None:
            node_plan["supported_phases"] = supported_phases(
                node_plan["supported_phases"]
            )


def build_d2_replay_negative_cases(
    outcome: dict[str, Any],
    no_refit_outcome: dict[str, Any],
    package: dict[str, Any],
) -> list[dict[str, Any]]:
    """Adversarial D2 replayable-phase cases, each re-signed to reach its target
    check. Every mutation re-signs inner and outer fingerprints in the right order
    (except a deliberately un-repaired inner fingerprint for the stale/forged
    cases), so the intended validator — not a coarse outer fingerprint — is what
    rejects the document. Each ``expected_error`` is a common substring across the
    Rust `from_json` path and the Python oracle/production validators."""

    cases: list[dict[str, Any]] = []

    def outcome_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(outcome)
        mutate(document)
        cases.append(
            {
                "id": case_id,
                "contract": "training_outcome",
                "document": document,
                "expected_error": expected,
            }
        )

    def no_refit_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(no_refit_outcome)
        mutate(document)
        cases.append(
            {
                "id": case_id,
                "contract": "training_outcome",
                "document": document,
                "expected_error": expected,
            }
        )

    def package_case(case_id: str, expected: str, mutate) -> None:
        document = copy.deepcopy(package)
        mutate(document)
        cases.append(
            {
                "id": case_id,
                "contract": "portable_predictor_package",
                "document": document,
                "expected_error": expected,
            }
        )

    # --- Forged replay vectors (completed refit is PREDICT-only here) ----------
    def forge_predict_explain(document):
        document["replayable_phases"] = ["PREDICT", "EXPLAIN"]
        document["outcome_fingerprint"] = fingerprint_without(
            document, "outcome_fingerprint"
        )

    outcome_case(
        "d2_outcome_forged_replay_vector",
        "replayable_phases do not match",
        forge_predict_explain,
    )

    def forge_refit_readvertise(document):
        document["replayable_phases"] = ["REFIT"]
        document["outcome_fingerprint"] = fingerprint_without(
            document, "outcome_fingerprint"
        )

    outcome_case(
        "d2_outcome_completed_refit_readvertises_refit",
        "replayable_phases do not match",
        forge_refit_readvertise,
    )

    # --- Canonical topology / parallel-level drift ----------------------------
    def drift_topology(document):
        document["effective_plan"]["graph_plan"]["topological_order"].reverse()
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_topological_order_drift",
        "topological_order does not match the graph",
        drift_topology,
    )

    def singleton_empty_parallel(document):
        document["effective_plan"]["graph_plan"]["parallel_levels"] = [[]]
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_parallel_levels_singleton_empty",
        "parallel",
        singleton_empty_parallel,
    )

    # --- Edge-derived adjacency drift -----------------------------------------
    def drift_adjacency(document):
        plan = document["effective_plan"]
        target = document["outputs"][0]["binding"]["node_id"]
        plan["node_plans"][target]["input_nodes"] = []
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_input_adjacency_drift",
        "input/output adjacency does not match the graph",
        drift_adjacency,
    )

    def drift_output_adjacency(document):
        plan = document["effective_plan"]
        node_id, node_plan = next(
            (node_id, node_plan)
            for node_id, node_plan in plan["node_plans"].items()
            if node_plan["output_nodes"]
        )
        plan["node_plans"][node_id]["output_nodes"] = []
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_output_adjacency_drift",
        "input/output adjacency does not match the graph",
        drift_output_adjacency,
    )

    # --- NodePlan vs ControllerManifest phase / version drift -----------------
    def drift_node_plan_phase(document):
        plan = document["effective_plan"]
        node_id = _first_stateless_closure_node(plan)
        phases = plan["node_plans"][node_id]["supported_phases"]
        plan["node_plans"][node_id]["supported_phases"] = [
            phase for phase in phases if phase != "PREDICT"
        ]
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_node_plan_phase_drift",
        "does not match controller manifest",
        drift_node_plan_phase,
    )

    def drift_node_plan_version(document):
        plan = document["effective_plan"]
        node_id = _first_stateless_closure_node(plan)
        plan["node_plans"][node_id]["controller_version"] = "0.0.0-d2-drift"
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_node_plan_version_drift",
        "does not match controller manifest",
        drift_node_plan_version,
    )

    # --- Stateful (non-emitter) closure node without a retained artifact -------
    # Adding `stateful` to a stateless, non-emitting closure node makes it require
    # retained inference state it does not have (only `emits_artifacts` nodes
    # produce refit artifacts), so the honest completed-refit answer is []. The
    # document still claims [PREDICT], so re-derivation refuses it.
    def add_stateful_without_artifact(document):
        plan = document["effective_plan"]
        node_id = _first_stateless_closure_node(plan)
        controller_id = plan["node_plans"][node_id]["controller_id"]
        _mutate_manifest_and_node_plans(
            plan,
            controller_id,
            capabilities=lambda caps: sort_capabilities(caps + ["stateful"]),
        )
        resign_training_outcome(document)

    outcome_case(
        "d2_outcome_stateful_without_artifact_forges_predict",
        "replayable_phases do not match",
        add_stateful_without_artifact,
    )

    # --- Stale / forged embedded plan fingerprints ----------------------------
    def forge_graph_fingerprint(document):
        document["effective_plan"]["graph_fingerprint"] = "a" * 64
        resign_training_outcome(document, graph=False)

    outcome_case(
        "d2_outcome_forged_graph_fingerprint",
        "graph_fingerprint does not match",
        forge_graph_fingerprint,
    )

    def forge_campaign_fingerprint(document):
        document["effective_plan"]["campaign_fingerprint"] = "b" * 64
        resign_training_outcome(document, campaign=False)

    outcome_case(
        "d2_outcome_forged_campaign_fingerprint",
        "campaign_fingerprint does not match",
        forge_campaign_fingerprint,
    )

    def forge_controller_fingerprint(document):
        document["effective_plan"]["controller_fingerprint"] = "c" * 64
        resign_training_outcome(document, controllers=False)

    outcome_case(
        "d2_outcome_forged_controller_fingerprint",
        "controller_fingerprint does not match",
        forge_controller_fingerprint,
    )

    def stale_controller_fingerprint(document):
        # Mutate a manifest field (`priority`) that is NOT copied into any
        # NodePlan, then leave the controller_fingerprint stale: only the
        # recomputed serde fingerprint can detect the drift.
        plan = document["effective_plan"]
        manifest = next(iter(plan["controller_manifests"].values()))
        manifest["priority"] = manifest.get("priority", 0) + 100
        resign_training_outcome(document, controllers=False)

    outcome_case(
        "d2_outcome_stale_controller_fingerprint",
        "controller_fingerprint does not match",
        stale_controller_fingerprint,
    )

    def stale_graph_content(document):
        document["effective_plan"]["graph_plan"]["graph"]["metadata"][
            "d2_stale_probe"
        ] = True
        resign_training_outcome(document, graph=False)

    outcome_case(
        "d2_outcome_stale_graph_content",
        "graph_fingerprint does not match",
        stale_graph_content,
    )

    def stale_campaign_content(document):
        document["effective_plan"]["campaign"]["metadata"]["d2_stale_probe"] = True
        resign_training_outcome(document, campaign=False)

    outcome_case(
        "d2_outcome_stale_campaign_content",
        "campaign_fingerprint does not match",
        stale_campaign_content,
    )

    # --- Forged refit-artifact provenance (package validators) -----------------
    def forge_artifact_record_controller(document):
        record = document["execution_bundle"]["refit_artifacts"][0]
        record["controller_id"] = "controller:forged"
        record["artifact"]["controller_id"] = "controller:forged"
        resign_package_plan(document)

    package_case(
        "d2_package_forged_artifact_record_controller",
        "does not match plan",
        forge_artifact_record_controller,
    )

    def forge_artifact_nested_controller(document):
        record = document["execution_bundle"]["refit_artifacts"][0]
        record["artifact"]["controller_id"] = "controller:forged"
        resign_package_plan(document)

    package_case(
        "d2_package_forged_artifact_nested_controller",
        "does not match record controller",
        forge_artifact_nested_controller,
    )

    def forge_artifact_params(document):
        record = document["execution_bundle"]["refit_artifacts"][0]
        record["params_fingerprint"] = "2" * 64
        resign_package_plan(document)

    package_case(
        "d2_package_forged_artifact_params",
        "do not match plan",
        forge_artifact_params,
    )

    # --- Graph-derived predictor closure drift (package) ----------------------
    def drift_predictor_closure(document):
        document["predictor_node_ids"] = document["predictor_node_ids"][:-1]
        resign_package_plan(document)

    package_case(
        "d2_package_predictor_closure_drift",
        "closure",
        drift_predictor_closure,
    )

    # --- Package whose closure cannot replay PREDICT --------------------------
    def remove_predict_support(document):
        plan = document["effective_plan"]
        node_id = _first_stateless_closure_node(plan)
        controller_id = plan["node_plans"][node_id]["controller_id"]
        _mutate_manifest_and_node_plans(
            plan,
            controller_id,
            supported_phases=lambda phases: [
                phase for phase in phases if phase != "PREDICT"
            ],
        )
        resign_package_plan(document)

    package_case(
        "d2_package_closure_missing_predict",
        "PREDICT-replayable",
        remove_predict_support,
    )

    # --- No-refit: REFIT refused without full support / OOF triple ------------
    def remove_refit_support(document):
        plan = document["effective_plan"]
        node_id = _first_stateless_closure_node(plan)
        controller_id = plan["node_plans"][node_id]["controller_id"]
        _mutate_manifest_and_node_plans(
            plan,
            controller_id,
            supported_phases=lambda phases: [
                phase for phase in phases if phase != "REFIT"
            ],
        )
        resign_training_outcome(document)

    no_refit_case(
        "d2_no_refit_without_refit_support",
        "replayable_phases do not match",
        remove_refit_support,
    )

    def remove_portable_oof_payload(document):
        payloads = document["portable_prediction_caches"]["caches"]
        if not payloads:
            raise AssertionError("no-refit fixture has no portable OOF payload")
        payloads.pop()
        document["outcome_fingerprint"] = fingerprint_without(
            document, "outcome_fingerprint"
        )

    no_refit_case(
        "d2_no_refit_missing_portable_oof_payload",
        "cache record",
        remove_portable_oof_payload,
    )

    return cases


def generate(output_dir: Path = OUT) -> None:
    refit = training_request(refit=True)
    no_refit = training_request(refit=False)
    active_influence = active_influence_training_request()
    package_request = package_training_request()
    outcome = w1_training_outcome(package_request)
    no_refit_outcome = load_json(
        ROOT
        / "examples"
        / "fixtures"
        / "estimator"
        / "training_outcome_no_refit.v1.json"
    )
    package = portable_package(package_request, outcome)
    namespace = cache_namespace(package)
    projection = empty_projection()
    python_smoke = python_training_smoke_contract()
    python_multiport_smoke = python_training_smoke_contract(explicit_multi_output=True)
    write_json(output_dir / "training_request_refit.v1.json", refit)
    write_json(output_dir / "training_request_no_refit.v1.json", no_refit)
    write_json(
        output_dir / "training_request_active_influence.v1.json",
        active_influence,
    )
    write_json(
        output_dir / "training_request_package_refit.v1.json",
        package_request,
    )
    write_json(output_dir / "portable_predictor_package.v1.json", package)
    write_json(output_dir / "training_outcome_refit.v1.json", outcome)
    write_json(output_dir / "cache_namespace_fit_cv.v1.json", namespace)
    write_json(output_dir / "parameter_projection_empty.v1.json", projection)
    write_json(output_dir / "python_training_smoke.v1.json", python_smoke)
    write_json(
        output_dir / "python_training_multiport_smoke.v1.json",
        python_multiport_smoke,
    )

    bad_phase = copy.deepcopy(namespace)
    bad_phase["phase"] = "REFIT"
    bad_phase["namespace_fingerprint"] = fingerprint_without(
        bad_phase, "namespace_fingerprint"
    )
    relation_drift = copy.deepcopy(package)
    relation_drift["data_identities"][0]["relation_fingerprint"] = opaque(
        "different-relation"
    )
    relation_drift["data_identities"][0]["identity_fingerprint"] = fingerprint_without(
        relation_drift["data_identities"][0], "identity_fingerprint"
    )
    refingerprint_package(relation_drift)
    unknown_option = copy.deepcopy(refit)
    unknown_option["options"]["unknown_option"] = True
    refingerprint_request(unknown_option)

    missing_selection_output = copy.deepcopy(refit)
    missing_selection_output["options"].pop("selection_output_id")
    refingerprint_request(missing_selection_output)

    unknown_selection_output = copy.deepcopy(refit)
    unknown_selection_output["options"]["selection_output_id"] = "output:unknown"
    refingerprint_request(unknown_selection_output)

    duplicate_selection_output = copy.deepcopy(refit)
    duplicate_selection_output["options"]["outputs"].append(
        copy.deepcopy(duplicate_selection_output["options"]["outputs"][0])
    )
    refingerprint_request(duplicate_selection_output)

    nonscorable_selection_output = copy.deepcopy(refit)
    nonscorable_manifest = next(
        manifest
        for manifest in nonscorable_selection_output["controller_manifests"]
        if manifest["controller_id"] == "controller:model.mock"
    )
    nonscorable_manifest["supported_phases"].remove("FIT_CV")
    refingerprint_request(nonscorable_selection_output)

    selection_wrong_objective = copy.deepcopy(refit)
    selection_wrong_objective["options"]["selection"]["metric"]["objective"] = (
        "maximize"
    )
    refingerprint_request(selection_wrong_objective)

    selection_level_mismatch = copy.deepcopy(refit)
    selection_level_mismatch["options"]["outputs"][0]["prediction_level"] = "target"
    selection_level_mismatch["options"]["outputs"][0]["unit_level"] = None
    refingerprint_request(selection_level_mismatch)

    selection_probability_implicit = copy.deepcopy(refit)
    selection_probability_implicit["options"]["outputs"][0]["prediction_kind"] = (
        "class_probability"
    )
    selection_probability_implicit["options"]["outputs"][0]["class_labels"] = [
        ["low", "high"]
    ]
    selection_probability_implicit["options"]["outputs"][0]["output_order"] = (
        "target_major_class_minor"
    )
    refingerprint_request(selection_probability_implicit)

    selection_class_label_wrong_metric = copy.deepcopy(refit)
    selection_class_label_wrong_metric["options"]["outputs"][0]["prediction_kind"] = (
        "class_label"
    )
    selection_class_label_wrong_metric["options"]["outputs"][0]["class_labels"] = [
        ["low", "high"]
    ]
    refingerprint_request(selection_class_label_wrong_metric)

    selection_decision_score_implicit = copy.deepcopy(refit)
    selection_decision_score_implicit["options"]["outputs"][0]["prediction_kind"] = (
        "decision_score"
    )
    refingerprint_request(selection_decision_score_implicit)

    data_binding_drift = copy.deepcopy(refit)
    data_binding_drift["data_identities"][0]["schema_fingerprint"] = opaque(
        "different-schema"
    )
    data_binding_drift["data_identities"][0]["identity_fingerprint"] = (
        fingerprint_without(
            data_binding_drift["data_identities"][0], "identity_fingerprint"
        )
    )
    refingerprint_request(data_binding_drift)

    missing_patch_policy = copy.deepcopy(refit)
    missing_patch_policy["parameter_patches"] = [
        {
            "schema_version": 1,
            "node_id": "model:base",
            "namespace": "operator",
            "path": ["max_depth"],
            "value": 8,
        }
    ]
    refingerprint_request(missing_patch_policy)

    forbidden_patch_namespace = copy.deepcopy(missing_patch_policy)
    forbidden_patch_namespace["patch_policies"] = [
        {"node_id": "model:base", "allowed_namespaces": ["fit"]}
    ]
    refingerprint_request(forbidden_patch_namespace)

    parent_child_patch = copy.deepcopy(refit)
    parent_child_patch["parameter_patches"] = [
        {
            "schema_version": 1,
            "node_id": "model:base",
            "namespace": "operator",
            "path": ["nested"],
            "value": {"leaf": 1},
        },
        {
            "schema_version": 1,
            "node_id": "model:base",
            "namespace": "operator",
            "path": ["nested", "leaf"],
            "value": 2,
        },
    ]
    parent_child_patch["patch_policies"] = [
        {"node_id": "model:base", "allowed_namespaces": ["operator"]}
    ]
    refingerprint_request(parent_child_patch)

    missing_influence_slot = copy.deepcopy(active_influence)
    missing_influence_slot["influence_requirements"].pop()
    refingerprint_request(missing_influence_slot)

    outer_validation_leak = copy.deepcopy(active_influence)
    outer_validation_leak["influence_requirements"][0]["physical_sample_ids"] = [
        "sample:1",
        "sample:3",
    ]
    refingerprint_request(outer_validation_leak)

    weighting_subset = copy.deepcopy(active_influence)
    next(
        requirement
        for requirement in weighting_subset["influence_requirements"]
        if requirement["scope_id"] == "weights:fit_cv:fold:0"
    )["physical_sample_ids"] = ["sample:3"]
    refingerprint_request(weighting_subset)

    capability_inactive = copy.deepcopy(active_influence)
    inactive_model = next(
        manifest
        for manifest in capability_inactive["controller_manifests"]
        if manifest["controller_id"] == "controller:model.mock"
    )
    inactive_model["capabilities"].remove("uses_early_stopping")
    refingerprint_request(capability_inactive)

    unit_level_mismatch = copy.deepcopy(refit)
    unit_level_mismatch["options"]["outputs"][0]["unit_level"] = None
    refingerprint_request(unit_level_mismatch)

    empty_probability_vocab = copy.deepcopy(refit)
    empty_probability_vocab["options"]["outputs"][0]["prediction_kind"] = (
        "class_probability"
    )
    empty_probability_vocab["options"]["outputs"][0]["output_order"] = (
        "target_major_class_minor"
    )
    refingerprint_request(empty_probability_vocab)

    duplicate_output_coordinate = copy.deepcopy(package)
    duplicate_binding = copy.deepcopy(duplicate_output_coordinate["output_bindings"][0])
    duplicate_binding["binding_id"] = "zz:duplicate"
    duplicate_binding["binding_fingerprint"] = fingerprint_without(
        duplicate_binding, "binding_fingerprint"
    )
    duplicate_output_coordinate["output_bindings"].append(duplicate_binding)
    duplicate_output_coordinate["training_outcome"][
        "output_binding_fingerprints"
    ].append(duplicate_binding["binding_fingerprint"])
    refingerprint_package(duplicate_output_coordinate)

    influence_order = copy.deepcopy(package)
    influence_order["training_influence"]["entries"][0:2] = reversed(
        influence_order["training_influence"]["entries"][0:2]
    )
    influence_order["training_influence"]["manifest_fingerprint"] = fingerprint_without(
        influence_order["training_influence"], "manifest_fingerprint"
    )
    influence_order["training_outcome"]["training_influence_fingerprint"] = (
        influence_order["training_influence"]["manifest_fingerprint"]
    )
    refingerprint_package(influence_order)

    data_identity_order = copy.deepcopy(package)
    data_identity_order["data_identities"][0:2] = reversed(
        data_identity_order["data_identities"][0:2]
    )
    refingerprint_package(data_identity_order)

    native_nonportable = copy.deepcopy(package)
    native_nonportable["artifact_bindings"][0]["load_mode"] = "native_portable"
    native_artifact_id = native_nonportable["artifact_bindings"][0]["artifact_id"]
    native_artifact = next(
        record["artifact"]
        for record in native_nonportable["execution_bundle"]["refit_artifacts"]
        if record["artifact"]["id"] == native_artifact_id
    )
    native_artifact.pop("uri")
    native_artifact.pop("content_fingerprint")
    native_nonportable["training_outcome"]["execution_bundle_fingerprint"] = (
        tcv1_sha256(native_nonportable["execution_bundle"])
    )
    refingerprint_package(native_nonportable)

    portable_required_sidecar = copy.deepcopy(package)
    portable_required_sidecar["fitted_artifact_mode"] = "portable_required"
    refingerprint_package(portable_required_sidecar)

    influence_wrong_kind = copy.deepcopy(package)
    influence_wrong_kind["training_influence"]["entries"][0]["kind"] = "model_fit"
    sort_influence_entries(influence_wrong_kind["training_influence"]["entries"])
    influence_wrong_kind["training_influence"]["manifest_fingerprint"] = (
        fingerprint_without(
            influence_wrong_kind["training_influence"], "manifest_fingerprint"
        )
    )
    influence_wrong_kind["training_outcome"]["training_influence_fingerprint"] = (
        influence_wrong_kind["training_influence"]["manifest_fingerprint"]
    )
    refingerprint_package(influence_wrong_kind)

    influence_extra_node = copy.deepcopy(package)
    extra_entry = copy.deepcopy(
        influence_extra_node["training_influence"]["entries"][0]
    )
    extra_entry["scope_id"] = "fit:ghost"
    extra_entry["node_id"] = "ghost:model"
    influence_extra_node["training_influence"]["entries"].append(extra_entry)
    sort_influence_entries(influence_extra_node["training_influence"]["entries"])
    influence_extra_node["training_influence"]["manifest_fingerprint"] = (
        fingerprint_without(
            influence_extra_node["training_influence"], "manifest_fingerprint"
        )
    )
    influence_extra_node["training_outcome"]["training_influence_fingerprint"] = (
        influence_extra_node["training_influence"]["manifest_fingerprint"]
    )
    refingerprint_package(influence_extra_node)

    request_crosslink = copy.deepcopy(package)
    request_crosslink["training_request_fingerprint"] = "e" * 64
    refingerprint_package(request_crosslink)

    content_identity_drift = copy.deepcopy(package)
    content_identity_drift["data_identities"][0]["data_content_fingerprint"] = "d" * 64
    content_identity_drift["data_identities"][0]["target_content_fingerprint"] = (
        "e" * 64
    )
    content_identity_drift["data_identities"][0]["identity_fingerprint"] = (
        fingerprint_without(
            content_identity_drift["data_identities"][0],
            "identity_fingerprint",
        )
    )
    refingerprint_package(content_identity_drift)

    same_id_bundle_drift = copy.deepcopy(package)
    same_id_bundle_drift["execution_bundle"].setdefault("metadata", {})[
        "same_id_drift"
    ] = True
    refingerprint_package(same_id_bundle_drift)

    outcome_selection_score_drift = copy.deepcopy(outcome)
    outcome_decision = next(
        iter(outcome_selection_score_drift["execution_bundle"]["selections"].values())
    )
    outcome_decision["selected_score"] = 0.2
    outcome_selection_score_drift["outcome_fingerprint"] = fingerprint_without(
        outcome_selection_score_drift, "outcome_fingerprint"
    )
    d2_cases = build_d2_replay_negative_cases(outcome, no_refit_outcome, package)
    write_json(
        output_dir / "negative_cases.v1.json",
        {
            "schema_version": 1,
            "cases": [
                {
                    "id": "cache_phase_refit",
                    "contract": "cache_namespace",
                    "document": bad_phase,
                    "expected_error": "FIT_CV",
                },
                {
                    "id": "package_relation_drift",
                    "contract": "portable_predictor_package",
                    "document": relation_drift,
                    "expected_error": "bundle fingerprints",
                },
                {
                    "id": "package_request_crosslink_drift",
                    "contract": "portable_predictor_package",
                    "document": request_crosslink,
                    "expected_error": "request fingerprint is not cross-linked",
                },
                {
                    "id": "package_content_identity_crosslink_drift",
                    "contract": "portable_predictor_package",
                    "document": content_identity_drift,
                    "expected_error": "data identity content is not cross-linked",
                },
                {
                    "id": "package_same_id_bundle_crosslink_drift",
                    "contract": "portable_predictor_package",
                    "document": same_id_bundle_drift,
                    "expected_error": "execution bundle content is not cross-linked",
                },
                {
                    "id": "training_outcome_selection_score_drift",
                    "contract": "training_outcome",
                    "document": outcome_selection_score_drift,
                    "expected_error": "SELECT decision does not equal ranking reconstructed from scores",
                },
                {
                    "id": "training_unknown_option",
                    "contract": "training_request",
                    "document": unknown_option,
                    "expected_error": "unknown field",
                },
                {
                    "id": "training_selection_output_missing",
                    "contract": "training_request",
                    "document": missing_selection_output,
                    "expected_error": "selection_output_id",
                },
                {
                    "id": "training_selection_output_unknown",
                    "contract": "training_request",
                    "document": unknown_selection_output,
                    "expected_error": "does not identify a declared output",
                },
                {
                    "id": "training_selection_output_duplicate_id",
                    "contract": "training_request",
                    "document": duplicate_selection_output,
                    "expected_error": "strictly sorted by output_id",
                },
                {
                    "id": "training_selection_output_not_scorable",
                    "contract": "training_request",
                    "document": nonscorable_selection_output,
                    "expected_error": "not scorable in FIT_CV",
                },
                {
                    "id": "training_selection_metric_wrong_objective",
                    "contract": "training_request",
                    "document": selection_wrong_objective,
                    "expected_error": "not supported for RegressionPoint",
                },
                {
                    "id": "training_selection_output_campaign_level_mismatch",
                    "contract": "training_request",
                    "document": selection_level_mismatch,
                    "expected_error": "campaign selection_metric_level",
                },
                {
                    "id": "training_selection_probability_requires_explicit_metric",
                    "contract": "training_request",
                    "document": selection_probability_implicit,
                    "expected_error": "not supported for ClassProbability",
                },
                {
                    "id": "training_selection_class_label_wrong_metric",
                    "contract": "training_request",
                    "document": selection_class_label_wrong_metric,
                    "expected_error": "not supported for ClassLabel",
                },
                {
                    "id": "training_selection_decision_score_requires_explicit_metric",
                    "contract": "training_request",
                    "document": selection_decision_score_implicit,
                    "expected_error": "not supported for DecisionScore",
                },
                {
                    "id": "training_data_binding_fingerprint_drift",
                    "contract": "training_request",
                    "document": data_binding_drift,
                    "expected_error": "data binding fingerprints",
                },
                {
                    "id": "training_patch_missing_policy",
                    "contract": "training_request",
                    "document": missing_patch_policy,
                    "expected_error": "exactly cover",
                },
                {
                    "id": "training_patch_forbidden_namespace",
                    "contract": "training_request",
                    "document": forbidden_patch_namespace,
                    "expected_error": "forbidden",
                },
                {
                    "id": "training_patch_parent_child",
                    "contract": "training_request",
                    "document": parent_child_patch,
                    "expected_error": "parent/child",
                },
                {
                    "id": "training_influence_missing_slot",
                    "contract": "training_request",
                    "document": missing_influence_slot,
                    "expected_error": "exactly cover active capability scopes",
                },
                {
                    "id": "training_influence_outer_validation_leak",
                    "contract": "training_request",
                    "document": outer_validation_leak,
                    "expected_error": "leaks outer validation samples",
                },
                {
                    "id": "training_influence_weighting_subset",
                    "contract": "training_request",
                    "document": weighting_subset,
                    "expected_error": "complete fit cohort",
                },
                {
                    "id": "training_influence_capability_inactive",
                    "contract": "training_request",
                    "document": capability_inactive,
                    "expected_error": "not required by active controller capabilities",
                },
                {
                    "id": "training_output_unit_level_mismatch",
                    "contract": "training_request",
                    "document": unit_level_mismatch,
                    "expected_error": "physical_sample",
                },
                {
                    "id": "training_output_empty_probability_vocabulary",
                    "contract": "training_request",
                    "document": empty_probability_vocab,
                    "expected_error": "class labels must be non-empty",
                },
                {
                    "id": "package_duplicate_output_coordinate",
                    "contract": "portable_predictor_package",
                    "document": duplicate_output_coordinate,
                    "expected_error": "more than once",
                },
                {
                    "id": "package_influence_order",
                    "contract": "portable_predictor_package",
                    "document": influence_order,
                    "expected_error": "canonically sorted",
                },
                {
                    "id": "package_data_identity_order",
                    "contract": "portable_predictor_package",
                    "document": data_identity_order,
                    "expected_error": "sorted by requirement_key",
                },
                {
                    "id": "package_native_artifact_not_portable",
                    "contract": "portable_predictor_package",
                    "document": native_nonportable,
                    "expected_error": "not portable",
                },
                {
                    "id": "package_portable_required_host_sidecar",
                    "contract": "portable_predictor_package",
                    "document": portable_required_sidecar,
                    "expected_error": "forbids host sidecar",
                },
                {
                    "id": "package_influence_wrong_base_kind",
                    "contract": "portable_predictor_package",
                    "document": influence_wrong_kind,
                    "expected_error": "expected kind",
                },
                {
                    "id": "package_influence_extra_node",
                    "contract": "portable_predictor_package",
                    "document": influence_extra_node,
                    "expected_error": "outside predictor closure",
                },
                *d2_cases,
            ],
        },
    )


def build_conformance_pack() -> dict[str, Any]:
    negative_fixture = load_json(OUT / "negative_cases.v1.json")
    pack = {
        "schema_version": 1,
        "pack_id": "dag-ml.training-contracts.v1",
        "hash_algorithm": "sha256-file-bytes",
        "canonical_profile": "DAG-ML TCV1",
        "artifacts": [
            {
                "path": relative_path,
                "sha256": file_sha256(ROOT / relative_path),
                "kind": PACK_ARTIFACTS[relative_path],
            }
            for relative_path in sorted(PACK_ARTIFACTS)
        ],
        "positive_fixture_ids": [
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
        "negative_case_ids": [case["id"] for case in negative_fixture["cases"]],
        "pack_checksum": "0" * 64,
    }
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    return pack


def generate_pack(pack_path: Path = PACK_PATH) -> None:
    write_json(pack_path, build_conformance_pack())


if __name__ == "__main__":
    generate()
    generate_pack()
