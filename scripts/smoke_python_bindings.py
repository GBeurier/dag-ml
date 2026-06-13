#!/usr/bin/env python3
"""Smoke-test the installed dag_ml Python wheel."""

from __future__ import annotations

import json
from importlib import metadata
from importlib import resources
from pathlib import Path

import dag_ml

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
    if "compile_pipeline_dsl_artifact_json" not in manifest["wasm_exports"]:
        raise SystemExit("contract manifest is missing WASM DSL export")
    if "structured_error_descriptors" not in manifest["capabilities"]:
        raise SystemExit("contract manifest is missing structured error capability")
    if "CompiledPipelineArtifact" not in manifest["python_facade_exports"]:
        raise SystemExit("contract manifest is missing Python facade artifact export")
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
    typed_plan = dag_ml.build_execution_plan(
        "plan:python.facade",
        typed_artifact.graph,
        typed_artifact.campaign_template,
        controllers,
    )
    if not typed_plan.to_dict()["node_plans"]:
        raise SystemExit("typed Python facade built an empty execution plan")
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
    if dag_ml.__version__ != metadata.version("dag-ml"):
        raise SystemExit("dag_ml.__version__ does not match package metadata")
    if dag_ml.version() != dag_ml.__version__:
        raise SystemExit("dag_ml.version() does not match dag_ml.__version__")
    package_root = resources.files("dag_ml")
    if not package_root.joinpath("py.typed").is_file():
        raise SystemExit("dag_ml wheel is missing py.typed")
    if not package_root.joinpath("__init__.pyi").is_file():
        raise SystemExit("dag_ml wheel is missing __init__.pyi")


if __name__ == "__main__":
    main()
