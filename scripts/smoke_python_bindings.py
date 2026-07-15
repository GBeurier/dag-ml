#!/usr/bin/env python3
"""Smoke-test the installed dag_ml Python wheel."""

from __future__ import annotations

import json
from importlib import metadata
from importlib import resources
from pathlib import Path

import dag_ml
import dag_ml._dag_ml as native

EXPECTED_NATIVE_VERSION = "0.2.7"
SHARED_FOLD_SET_FINGERPRINT = (
    "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c"
)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    dsl_json = (repo / "examples" / "pipeline_dsl_generation.json").read_text(
        encoding="utf-8"
    )
    controller_manifests_json = (
        repo / "examples" / "controller_manifests.json"
    ).read_text(encoding="utf-8")
    shared_fold_set_json = (
        repo / "examples" / "fixtures" / "shared" / "fold_set_cv_partition.json"
    ).read_text(encoding="utf-8")
    training_request_json = (
        repo
        / "examples"
        / "fixtures"
        / "training"
        / "training_request_active_influence.v1.json"
    ).read_text(encoding="utf-8")
    portable_package_json = (
        repo
        / "examples"
        / "fixtures"
        / "training"
        / "portable_predictor_package.v1.json"
    ).read_text(encoding="utf-8")
    cache_namespace_json = (
        repo / "examples" / "fixtures" / "training" / "cache_namespace_fit_cv.v1.json"
    ).read_text(encoding="utf-8")
    parameter_projection_json = (
        repo
        / "examples"
        / "fixtures"
        / "training"
        / "parameter_projection_empty.v1.json"
    ).read_text(encoding="utf-8")
    training_outcome_json = (
        repo / "examples" / "fixtures" / "training" / "training_outcome_refit.v1.json"
    ).read_text(encoding="utf-8")

    native_version = native.version()
    if native_version != EXPECTED_NATIVE_VERSION:
        raise SystemExit(
            f"native version drifted: {native_version} != {EXPECTED_NATIVE_VERSION}"
        )
    if dag_ml.__version__ != native_version or dag_ml.version() != native_version:
        raise SystemExit(
            "Python facade version does not match the loaded native extension: "
            f"__version__={dag_ml.__version__}, version()={dag_ml.version()}, "
            f"native={native_version}"
        )

    dag_ml.validate_pipeline_dsl_json(dsl_json)
    manifest = json.loads(dag_ml.contract_manifest_json())
    if manifest["crate"] != "dag-ml":
        raise SystemExit("contract manifest has wrong crate name")
    if manifest["python_package_version"] != dag_ml.version():
        raise SystemExit(
            "contract manifest Python package version does not match package version"
        )
    if "compile_pipeline_dsl_artifact_json" not in manifest["python_exports"]:
        raise SystemExit("contract manifest is missing Python DSL export")
    if "derive_controller_manifest_json" not in manifest["python_exports"]:
        raise SystemExit(
            "contract manifest is missing Python controller-derivation export"
        )
    if "project_training_request_json" not in manifest["python_exports"]:
        raise SystemExit(
            "contract manifest is missing Python training projection export"
        )
    if "execute_training_json" not in manifest["python_exports"]:
        raise SystemExit("contract manifest is missing owning Python training export")
    if "compile_pipeline_dsl_artifact_json" not in manifest["wasm_exports"]:
        raise SystemExit("contract manifest is missing WASM DSL export")
    if "structured_error_descriptors" not in manifest["capabilities"]:
        raise SystemExit("contract manifest is missing structured error capability")
    if "CompiledPipelineArtifact" not in manifest["python_facade_exports"]:
        raise SystemExit("contract manifest is missing Python facade artifact export")
    if "derive_controller_manifests" not in manifest["python_facade_exports"]:
        raise SystemExit(
            "contract manifest is missing Python facade controller derivation"
        )
    if "TrainingRequest" not in manifest["python_facade_exports"]:
        raise SystemExit("contract manifest is missing Python TrainingRequest facade")
    if "TrainingResult" not in manifest["python_facade_exports"]:
        raise SystemExit("contract manifest is missing Python TrainingResult facade")
    if "owning_training_result" not in manifest["capabilities"]:
        raise SystemExit("contract manifest is missing owning training capability")
    if (
        manifest["shared"]["fold_set_fixture_fingerprint"]
        != SHARED_FOLD_SET_FINGERPRINT
    ):
        raise SystemExit("contract manifest shared fold fingerprint drifted")
    artifact = json.loads(dag_ml.compile_pipeline_dsl_artifact_json(dsl_json))
    if "campaign_template" not in artifact:
        raise SystemExit("compiled artifact is missing campaign_template")
    dsl = dag_ml.PipelineDslSpec(dsl_json)
    typed_artifact = dag_ml.compile_pipeline_dsl_artifact(dsl)
    dag_ml.GraphSpec(typed_artifact.graph)
    dag_ml.CampaignSpec(typed_artifact.campaign_template)
    controllers = dag_ml.ControllerManifests(controller_manifests_json)
    host_controller_specs = [
        {
            "controller_id": "controller:python.smoke.transform",
            "controller_version": "0.10.0",
            "operator_kind": "transform",
        },
        {
            "controller_id": "controller:python.smoke.model",
            "controller_version": "0.10.0",
            "operator_kind": "model",
            "priority": 20,
        },
    ]
    derived_controllers = dag_ml.derive_controller_manifests(host_controller_specs)
    if len(derived_controllers.to_dict()) != 2:
        raise SystemExit("derived controller manifest list has wrong length")
    dag_ml.ControllerManifest(
        dag_ml.derive_controller_manifest(host_controller_specs[0]).json()
    )
    typed_plan = dag_ml.build_execution_plan(
        "plan:python.facade",
        typed_artifact.graph,
        typed_artifact.campaign_template,
        controllers,
    )
    if not typed_plan.to_dict()["node_plans"]:
        raise SystemExit("typed Python facade built an empty execution plan")
    request = dag_ml.TrainingRequest(training_request_json)
    projection = request.project()
    projected = projection.to_dict()
    if projected["request_id"] != "training:fixture.active_influence":
        raise SystemExit("typed training projection request id drifted")
    if projected["request_fingerprint"] != (
        "df52b77b52dfb4e6436da726b22698079f2441cdd15b55d5de3c5e204bb73f2b"
    ):
        raise SystemExit("typed training projection request fingerprint drifted")
    if projected["predictor_node_ids"] != ["model:base", "transform:snv"]:
        raise SystemExit("typed training projection predictor closure drifted")
    if projected["outputs"] != [
        {
            "output_id": "output:prediction",
            "node_id": "model:base",
            "port_name": "oof",
            "prediction_level": "sample",
            "unit_level": "physical_sample",
            "prediction_kind": "regression_point",
            "target_names": ["protein"],
            "target_units": ["percent"],
            "class_labels": [[]],
            "output_order": "target_order",
            "target_space": "raw",
        }
    ]:
        raise SystemExit("typed training projection output binding drifted")
    if projected["parameters"]["nodes"]["model:base"]["params"] != {
        "n_estimators": 100
    }:
        raise SystemExit("typed training parameter projection drifted")
    if dag_ml.project_training_request(request).to_dict() != projected:
        raise SystemExit("functional and typed training projections diverged")
    outcome = dag_ml.TrainingOutcome(training_outcome_json).to_dict()
    if outcome["outcome_id"] != "training:estimator.refit":
        raise SystemExit("typed training outcome id drifted")
    package = dag_ml.PortablePredictorPackage(portable_package_json).to_dict()
    if package["package_id"] != "predictor:package.fixture":
        raise SystemExit("portable package id drifted")
    if package["package_fingerprint"] != (
        "7d5b7a33d90211de2676a43f45dd102d60910eec419a7daef427cc0fff228dd0"
    ):
        raise SystemExit("portable package fingerprint drifted")
    if package["predictor_node_ids"] != [
        "branch:b0.model:ridge",
        "branch:b1.augment:noise",
        "branch:b1.model:rf",
        "merge:stack.pred_plus_original.meta:ridge",
    ]:
        raise SystemExit("portable package predictor closure drifted")
    if [
        (
            binding["binding_id"],
            binding["node_id"],
            binding["port_name"],
            binding["prediction_source"],
        )
        for binding in package["output_bindings"]
    ] != [
        (
            "output:meta.final",
            "merge:stack.pred_plus_original.meta:ridge",
            "oof",
            "final_refit",
        )
    ]:
        raise SystemExit("portable package output binding drifted")
    dag_ml.CacheNamespace(cache_namespace_json)
    parameter_projection = dag_ml.ParameterProjection(
        parameter_projection_json
    ).to_dict()
    if parameter_projection != {
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
        "patches_fingerprint": (
            "cea5f239e81001721b763cebf40cd71bca04972c51313fba335e0a96d7e81979"
        ),
        "projection_fingerprint": (
            "9eff58f693db68df88c00637e38e860472099a90e9c8a271ed350fdcf67ca837"
        ),
    }:
        raise SystemExit("standalone parameter projection content drifted")
    missing_patch_policies = json.loads(training_request_json)
    del missing_patch_policies["patch_policies"]
    try:
        dag_ml.validate_training_request_json(
            json.dumps(missing_patch_policies, sort_keys=True, separators=(",", ":"))
        )
    except dag_ml.DagMlError as error:
        if "patch_policies" not in str(error):
            raise SystemExit(
                "missing required patch_policies raised the wrong native error"
            ) from error
    else:
        raise SystemExit("missing required patch_policies was accepted")
    fold_set = {
        "id": "cv.partition",
        "sample_ids": ["s1", "s2", "s3"],
        "folds": [
            {
                "fold_id": "fold1",
                "train_sample_ids": ["s1", "s2"],
                "validation_sample_ids": ["s3"],
            },
            {
                "fold_id": "fold0",
                "train_sample_ids": ["s3"],
                "validation_sample_ids": ["s2", "s1"],
            },
        ],
    }
    fold_set_json = json.dumps(fold_set, sort_keys=True, separators=(",", ":"))
    dag_ml.validate_fold_set_json(fold_set_json)
    dag_ml.validate_fold_set_json(shared_fold_set_json)
    if (
        dag_ml.fold_set_fingerprint_json(shared_fold_set_json)
        != SHARED_FOLD_SET_FINGERPRINT
    ):
        raise SystemExit("shared fold set fingerprint drifted")
    typed_shared_fold_set = dag_ml.FoldSet(shared_fold_set_json)
    if typed_shared_fold_set.fingerprint() != SHARED_FOLD_SET_FINGERPRINT:
        raise SystemExit("typed fold set fingerprint drifted")
    fold_fingerprint = dag_ml.fold_set_fingerprint_json(fold_set_json)
    if len(fold_fingerprint) != 64:
        raise SystemExit("fold set fingerprint is not a sha256 hex digest")
    reordered_fold_set = {
        **fold_set,
        "sample_ids": list(reversed(fold_set["sample_ids"])),
        "folds": list(reversed(fold_set["folds"])),
    }
    if (
        dag_ml.fold_set_fingerprint_json(
            json.dumps(reordered_fold_set, sort_keys=True, separators=(",", ":"))
        )
        != fold_fingerprint
    ):
        raise SystemExit("fold set fingerprint changed after irrelevant reordering")
    if not dag_ml.version():
        raise SystemExit("dag_ml.version() returned an empty version")
    try:
        dag_ml.validate_graph_json('{"id":"","interface":{},"nodes":[],"edges":[]}')
    except dag_ml.DagMlError as error:
        if error.category != "validation" or error.code != "graph_validation":
            raise SystemExit("DagMlError taxonomy attributes drifted")
        descriptor = json.loads(error.descriptor_json)
        if descriptor["context"] != error.context:
            raise SystemExit("DagMlError descriptor context does not match attribute")
    else:
        raise SystemExit("invalid graph JSON was accepted")
    distribution_version = metadata.version("dag-ml")
    source_tree = (repo / "crates" / "dag-ml-py" / "python").resolve()
    imported_package = Path(dag_ml.__file__).resolve()
    if distribution_version != native_version and not imported_package.is_relative_to(
        source_tree
    ):
        raise SystemExit(
            "installed dag_ml package metadata does not match its native extension: "
            f"metadata={distribution_version}, native={native_version}"
        )
    package_root = resources.files("dag_ml")
    if not package_root.joinpath("py.typed").is_file():
        raise SystemExit("dag_ml wheel is missing py.typed")
    if not package_root.joinpath("__init__.pyi").is_file():
        raise SystemExit("dag_ml wheel is missing __init__.pyi")


if __name__ == "__main__":
    main()
