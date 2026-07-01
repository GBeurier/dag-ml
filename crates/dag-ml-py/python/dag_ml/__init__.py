"""Typed Python facade for DAG-ML JSON contracts."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from os import PathLike
from pathlib import Path
from typing import Any

from ._dag_ml import (
    DagMlBundleError,
    DagMlCompatibilityError,
    DagMlControllerError,
    DagMlDataError,
    DagMlError,
    DagMlInternalError,
    DagMlLineageError,
    DagMlReplayError,
    DagMlRuntimeError,
    DagMlSecurityError,
    DagMlValidationError,
    build_execution_plan_json,
    canonical_operator_variant_label,
    contract_manifest_json as _native_contract_manifest_json,
    compile_pipeline_dsl_artifact_json,
    compile_pipeline_dsl_artifact_with_controllers_json,
    compile_pipeline_dsl_graph_json,
    derive_controller_manifest_json,
    derive_controller_manifest_list_json,
    fan_out_data_aware_branches_json,
    fold_set_fingerprint_json,
    validate_campaign_json,
    validate_controller_manifest_json,
    validate_controller_manifest_list_json,
    validate_execution_bundle_json,
    validate_execution_plan_json,
    validate_fold_set_json,
    validate_graph_json,
    validate_pipeline_dsl_json,
    version as _native_version,
)

try:
    __version__ = _distribution_version("dag-ml")
except PackageNotFoundError:
    __version__ = _native_version()


_FACADE_EXPORTS = [
    "JsonContract",
    "GraphSpec",
    "CampaignSpec",
    "ControllerManifest",
    "ControllerManifests",
    "HostControllerSpec",
    "HostControllerSpecs",
    "PipelineDslSpec",
    "ExecutionPlan",
    "ExecutionBundle",
    "FoldSet",
    "CompiledPipelineArtifact",
    "compile_pipeline_dsl_graph",
    "compile_pipeline_dsl_artifact",
    "compile_pipeline_dsl_artifact_with_controllers",
    "derive_controller_manifest",
    "derive_controller_manifests",
    "fan_out_data_aware_branches",
    "build_execution_plan",
]


def _coerce_json(value: Any) -> str:
    if isinstance(value, JsonContract):
        return value.json()
    if isinstance(value, PathLike):
        return Path(value).read_text(encoding="utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _facade_contract_error(message: str) -> DagMlError:
    error = DagMlValidationError(message)
    context = {"detail": message}
    descriptor = {
        "category": "validation",
        "code": "python_facade_contract",
        "severity": "error",
        "message": message,
        "remediation_hint": "Pass a compiled dag-ml pipeline artifact containing graph and campaign_template JSON objects.",
        "context": context,
    }
    error.category = descriptor["category"]
    error.code = descriptor["code"]
    error.severity = descriptor["severity"]
    error.remediation_hint = descriptor["remediation_hint"]
    error.context = context
    error.context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
    error.descriptor_json = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return error


class JsonContract:
    """Immutable validated JSON contract wrapper."""

    __slots__ = ("_json",)

    def __init__(self, value: Any) -> None:
        json_text = _coerce_json(value)
        self._validate_json(json_text)
        self._json = json_text

    @classmethod
    def from_path(cls, path: str | PathLike[str]) -> "JsonContract":
        return cls(Path(path))

    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        json.loads(json_text)

    def json(self) -> str:
        return self._json

    def to_dict(self) -> Any:
        return json.loads(self._json)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._json!r})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._json == other._json


class GraphSpec(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_graph_json(json_text)


class CampaignSpec(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_campaign_json(json_text)


class ControllerManifest(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_controller_manifest_json(json_text)


class ControllerManifests(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_controller_manifest_list_json(json_text)


class HostControllerSpec(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        derive_controller_manifest_json(json_text)


class HostControllerSpecs(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        derive_controller_manifest_list_json(json_text)


class PipelineDslSpec(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_pipeline_dsl_json(json_text)


class ExecutionPlan(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_execution_plan_json(json_text)


class ExecutionBundle(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_execution_bundle_json(json_text)


class FoldSet(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_fold_set_json(json_text)

    def fingerprint(self) -> str:
        return fold_set_fingerprint_json(self._json)


class CompiledPipelineArtifact(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        artifact = json.loads(json_text)
        if not isinstance(artifact, dict):
            raise _facade_contract_error(
                "compiled pipeline artifact must be a JSON object"
            )
        if "graph" not in artifact:
            raise _facade_contract_error("compiled pipeline artifact is missing graph")
        if "campaign_template" not in artifact:
            raise _facade_contract_error(
                "compiled pipeline artifact is missing campaign_template"
            )
        GraphSpec(artifact["graph"])
        CampaignSpec(artifact["campaign_template"])

    @property
    def graph(self) -> GraphSpec:
        return GraphSpec(self.to_dict()["graph"])

    @property
    def campaign_template(self) -> CampaignSpec:
        return CampaignSpec(self.to_dict()["campaign_template"])


def version() -> str:
    """Return the installed Python package version."""
    return __version__


def contract_manifest_json() -> str:
    """Return the native contract manifest plus Python packaging metadata."""
    manifest = json.loads(_native_contract_manifest_json())
    manifest["python_package_version"] = __version__
    manifest["python_facade_exports"] = _FACADE_EXPORTS
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def compile_pipeline_dsl_graph(dsl: Any) -> GraphSpec:
    """Compile a pipeline DSL contract into a validated graph wrapper."""
    return GraphSpec(compile_pipeline_dsl_graph_json(_coerce_json(dsl)))


def compile_pipeline_dsl_artifact(dsl: Any) -> CompiledPipelineArtifact:
    """Compile a pipeline DSL contract into a validated graph/campaign artifact."""
    return CompiledPipelineArtifact(
        compile_pipeline_dsl_artifact_json(_coerce_json(dsl))
    )


def compile_pipeline_dsl_artifact_with_controllers(
    dsl: Any,
    controller_manifests: Any,
) -> CompiledPipelineArtifact:
    """Compile a pipeline DSL contract using controller selector metadata."""
    return CompiledPipelineArtifact(
        compile_pipeline_dsl_artifact_with_controllers_json(
            _coerce_json(dsl),
            _coerce_json(controller_manifests),
        )
    )


def derive_controller_manifest(host_controller_spec: Any) -> ControllerManifest:
    """Derive a validated ControllerManifest from a HostControllerSpec JSON object."""
    return ControllerManifest(
        derive_controller_manifest_json(_coerce_json(host_controller_spec))
    )


def derive_controller_manifests(host_controller_specs: Any) -> ControllerManifests:
    """Derive a validated ControllerManifest list from HostControllerSpec JSON objects."""
    return ControllerManifests(
        derive_controller_manifest_list_json(_coerce_json(host_controller_specs))
    )


def fan_out_data_aware_branches(dsl: Any, envelope: Any) -> PipelineDslSpec:
    """Expand an ``auto_separate`` separation-branch template into one branch per partition.

    Calls dag-ml-core's native data-aware fan-out: a branch step marked
    ``metadata.auto_separate=true`` with one template branch is expanded into N explicit
    branches — one per sorted distinct value of the criterion column discovered from the
    envelope's coordinator relations. The native fan-out owns the per-partition node-id
    suffixing and selector assignment, so the host never replicates that logic.
    """
    return PipelineDslSpec(
        fan_out_data_aware_branches_json(_coerce_json(dsl), _coerce_json(envelope))
    )


def build_execution_plan(
    plan_id: str,
    graph: Any,
    campaign: Any,
    controller_manifests: Any,
) -> ExecutionPlan:
    """Build a validated execution plan from typed or raw JSON contracts."""
    return ExecutionPlan(
        build_execution_plan_json(
            plan_id,
            _coerce_json(graph),
            _coerce_json(campaign),
            _coerce_json(controller_manifests),
        )
    )


__all__ = [
    "__version__",
    "DagMlBundleError",
    "DagMlCompatibilityError",
    "DagMlControllerError",
    "DagMlDataError",
    "DagMlError",
    "DagMlInternalError",
    "DagMlLineageError",
    "DagMlReplayError",
    "DagMlRuntimeError",
    "DagMlSecurityError",
    "DagMlValidationError",
    "CampaignSpec",
    "CompiledPipelineArtifact",
    "ControllerManifest",
    "ControllerManifests",
    "ExecutionBundle",
    "ExecutionPlan",
    "FoldSet",
    "GraphSpec",
    "HostControllerSpec",
    "HostControllerSpecs",
    "JsonContract",
    "PipelineDslSpec",
    "build_execution_plan",
    "build_execution_plan_json",
    "canonical_operator_variant_label",
    "contract_manifest_json",
    "compile_pipeline_dsl_artifact",
    "compile_pipeline_dsl_artifact_json",
    "compile_pipeline_dsl_artifact_with_controllers",
    "compile_pipeline_dsl_artifact_with_controllers_json",
    "compile_pipeline_dsl_graph",
    "compile_pipeline_dsl_graph_json",
    "derive_controller_manifest",
    "derive_controller_manifest_json",
    "derive_controller_manifest_list_json",
    "derive_controller_manifests",
    "fan_out_data_aware_branches",
    "fan_out_data_aware_branches_json",
    "fold_set_fingerprint_json",
    "validate_campaign_json",
    "validate_controller_manifest_json",
    "validate_controller_manifest_list_json",
    "validate_execution_bundle_json",
    "validate_execution_plan_json",
    "validate_fold_set_json",
    "validate_graph_json",
    "validate_pipeline_dsl_json",
    "version",
]
