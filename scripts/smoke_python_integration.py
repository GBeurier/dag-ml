#!/usr/bin/env python3
"""Smoke-test installed dag_ml + dag_ml_data wheels together."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import dag_ml
import dag_ml_data

SHARED_FOLD_SET_FINGERPRINT = (
    "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c"
)


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    dag_ml_data_repo = (
        Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else repo.parent / "dag-ml-data"
    )

    dag_manifest = json.loads(dag_ml.contract_manifest_json())
    data_manifest = json.loads(dag_ml_data.contract_manifest_json())
    _require(
        dag_manifest["crate"] == "dag-ml",
        "dag-ml manifest has wrong crate name",
    )
    _require(
        data_manifest["crate"] == "dag-ml-data",
        "dag-ml-data manifest has wrong crate name",
    )
    _require(
        dag_manifest["python_package_version"] == dag_ml.version(),
        "dag-ml manifest Python package version does not match installed package",
    )
    _require(
        data_manifest["python_package_version"] == dag_ml_data.version(),
        "dag-ml-data manifest Python package version does not match installed package",
    )
    _require(
        dag_manifest["shared"]["fold_set_fixture_fingerprint"]
        == data_manifest["shared"]["fold_set_fixture_fingerprint"],
        "shared fold set fingerprint differs between Python packages",
    )
    _require(
        dag_manifest["shared"]["fold_set_fixture_fingerprint"]
        == SHARED_FOLD_SET_FINGERPRINT,
        "shared fold set fingerprint drifted",
    )
    for name in ["compile_pipeline_dsl_artifact", "build_execution_plan"]:
        _require(
            name in dag_manifest["python_facade_exports"],
            f"dag-ml manifest misses Python facade export {name}",
        )
    for name in ["plan_model_input", "build_coordinator_data_plan_envelope"]:
        _require(
            name in data_manifest["python_facade_exports"],
            f"dag-ml-data manifest misses Python facade export {name}",
        )

    data_fixture_root = dag_ml_data_repo / "examples" / "fixtures" / "oof_campaign"
    schema = dag_ml_data.DatasetSchema(
        _read(data_fixture_root, "schema_nirs4all_core_contract.json")
    )
    model_input = dag_ml_data.ModelInputSpec(
        _read(data_fixture_root, "model_input_tabular_numeric.json")
    )
    registry = dag_ml_data.AdapterRegistry(
        _read(data_fixture_root, "adapter_registry_signal_to_tabular.json")
    )
    relations = dag_ml_data.SampleRelationTable(
        _read(data_fixture_root, "sample_relations_grouped_augmented.json")
    )
    data_plan = dag_ml_data.plan_model_input(
        schema,
        model_input,
        registry,
        {"id": "nir-to-tabular", "source_ids": ["nir"]},
    )
    envelope = dag_ml_data.build_coordinator_data_plan_envelope(
        schema,
        data_plan,
        relations,
    )
    envelope_payload = envelope.to_dict()
    _require(
        envelope_payload["plan"]["id"] == "nir-to-tabular",
        "data envelope has wrong plan id",
    )
    _require(
        isinstance(envelope_payload.get("relation_fingerprint"), str)
        and len(envelope_payload["relation_fingerprint"]) == 64,
        "data envelope is missing relation fingerprint",
    )

    dsl = dag_ml.PipelineDslSpec(
        _read(repo, "examples/pipeline_dsl_nirs4all_compat.json")
    )
    controllers = dag_ml.ControllerManifests(
        _read(repo, "examples/controller_manifests.json")
    )
    artifact = dag_ml.compile_pipeline_dsl_artifact(dsl)
    artifact_payload = artifact.to_dict()
    _require(artifact_payload["graph"]["nodes"], "compiled graph contains no nodes")
    _require(
        artifact_payload["campaign_template"].get("split_invocation"),
        "compiled campaign template is missing split invocation",
    )
    execution_plan = dag_ml.build_execution_plan(
        "plan:python.integration",
        artifact.graph,
        artifact.campaign_template,
        controllers,
    )
    plan_payload = execution_plan.to_dict()
    _require(plan_payload["node_plans"], "execution plan has no nodes")
    _require(plan_payload["variants"], "execution plan has no variants")


if __name__ == "__main__":
    main()
