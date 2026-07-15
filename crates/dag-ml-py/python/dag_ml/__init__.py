"""Typed Python facade for DAG-ML JSON contracts."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterable
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
    LocalImplementationRegistry as _NativeLocalImplementationRegistry,
    TrainingResult as _NativeTrainingResult,
    build_execution_plan_json,
    canonical_operator_variant_label,
    contract_manifest_json as _native_contract_manifest_json,
    compile_pipeline_dsl_artifact_json,
    compile_pipeline_dsl_artifact_with_controllers_json,
    compile_pipeline_dsl_graph_json,
    derive_controller_manifest_json,
    derive_controller_manifest_list_json,
    execute_loaded_predictor_replay_json as _native_execute_loaded_predictor_replay_json,
    execute_training_json as _native_execute_training_json,
    fan_out_data_aware_branches_json,
    fold_set_fingerprint_json,
    loss_execution_attestation_json as _native_loss_execution_attestation_json,
    project_training_request_json,
    sample_relation_set_fingerprint_json,
    sign_training_request_json as _native_sign_training_request_json,
    validate_campaign_json,
    validate_cache_namespace_json,
    validate_controller_manifest_json,
    validate_controller_manifest_list_json,
    validate_execution_bundle_json,
    validate_execution_plan_json,
    validate_fold_set_json,
    validate_graph_json,
    validate_parameter_projection_json,
    validate_pipeline_dsl_json,
    validate_portable_predictor_package_json,
    validate_training_contract_projection_json,
    validate_training_outcome_json,
    validate_training_replay_outcome_json,
    validate_training_replay_request_json,
    validate_training_request_json,
    version as _native_version,
)

# The loaded extension is the authoritative code version. Distribution metadata
# can describe another installation when this package is imported directly from
# a source tree via PYTHONPATH, so it must never override the native module.
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
    "TrainingRequest",
    "TrainingOutcome",
    "TrainingReplayRequest",
    "TrainingReplayOutcome",
    "TrainingResult",
    "TrainingContractProjection",
    "ParameterProjection",
    "CacheNamespace",
    "PortablePredictorPackage",
    "LocalImplementationRegistry",
    "loss_execution_attestation",
    "CompiledPipelineArtifact",
    "compile_pipeline_dsl_graph",
    "compile_pipeline_dsl_artifact",
    "compile_pipeline_dsl_artifact_with_controllers",
    "derive_controller_manifest",
    "derive_controller_manifests",
    "fan_out_data_aware_branches",
    "build_execution_plan",
    "project_training_request",
    "sign_training_request",
    "execute_training",
    "execute_training_json",
    "replay_loaded_predictor_package",
    "replay_loaded_predictor_package_json",
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
    error.descriptor_json = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    )
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


class LocalImplementationRegistry:
    """Process-local Python callables resolved by exact DAG-ML descriptors."""

    __slots__ = ("_native",)

    def __init__(self) -> None:
        self._native = _NativeLocalImplementationRegistry()

    def register_loss(self, loss_reference: Any, implementation: Any) -> None:
        self._native.register_loss(_coerce_json(loss_reference), implementation)

    def register_metric(self, metric_reference: Any, implementation: Any) -> None:
        self._native.register_metric(_coerce_json(metric_reference), implementation)

    def register_local_loss(
        self,
        loss_spec: Any,
        implementation: Any,
        *,
        registry_key: str | None = None,
        provider_id: str = "provider:python-local",
        implementation_version: str = "0+local",
        implementation_fingerprint: str | None = None,
        supported_controller_families: Iterable[str] = (),
        runtime_requirements: Iterable[str] = (),
        capabilities: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Register a callable and return its native host-local loss reference."""

        options = _host_local_registration_options(
            "loss",
            registry_key=registry_key,
            provider_id=provider_id,
            implementation_version=implementation_version,
            implementation_fingerprint=implementation_fingerprint,
            supported_controller_families=supported_controller_families,
            runtime_requirements=runtime_requirements,
            capabilities=capabilities,
        )
        return json.loads(
            self._native.register_host_local_loss(
                _unsigned_spec_json(loss_spec, "loss"),
                _coerce_json(options),
                implementation,
            )
        )

    def register_local_metric(
        self,
        metric_spec: Any,
        implementation: Any,
        *,
        registry_key: str | None = None,
        provider_id: str = "provider:python-local",
        implementation_version: str = "0+local",
        implementation_fingerprint: str | None = None,
        supported_controller_families: Iterable[str] = (),
        runtime_requirements: Iterable[str] = (),
        capabilities: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Register a callable and return its native host-local metric reference."""

        options = _host_local_registration_options(
            "metric",
            registry_key=registry_key,
            provider_id=provider_id,
            implementation_version=implementation_version,
            implementation_fingerprint=implementation_fingerprint,
            supported_controller_families=supported_controller_families,
            runtime_requirements=runtime_requirements,
            capabilities=capabilities,
        )
        return json.loads(
            self._native.register_host_local_metric(
                _unsigned_spec_json(metric_spec, "metric"),
                _coerce_json(options),
                implementation,
            )
        )

    def resolve_loss(self, loss_reference: Any) -> Any:
        return self._native.resolve_loss(_coerce_json(loss_reference))

    def resolve_training_loss(self, training_loss_role: Any, phase: str) -> Any:
        return self._native.resolve_training_loss(
            _coerce_json(training_loss_role), phase
        )

    def resolve_metric(self, metric_reference: Any) -> Any:
        return self._native.resolve_metric(_coerce_json(metric_reference))

    def invoke_training_loss(
        self,
        node_task: Any,
        *args: Any,
        role_index: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute one native-required local loss and return its attestation."""

        implementation, attestation_json = self._native.resolve_task_training_loss(
            _coerce_json(node_task), role_index
        )
        try:
            value = implementation(*args, **kwargs)
        except Exception as error:
            raise DagMlRuntimeError(
                f"Python training loss callback raised an exception: {error}"
            ) from error
        return {"value": value, "attestation": json.loads(attestation_json)}

    def evaluate_metric(self, metric_task: Any) -> dict[str, Any]:
        """Execute a local metric for a native typed evaluation task."""

        return json.loads(self._native.evaluate_metric(_coerce_json(metric_task)))

    def unregister_loss(self, loss_reference: Any) -> Any:
        return self._native.unregister_loss(_coerce_json(loss_reference))

    def unregister_metric(self, metric_reference: Any) -> Any:
        return self._native.unregister_metric(_coerce_json(metric_reference))

    def descriptors(self) -> list[dict[str, Any]]:
        return json.loads(self._native.descriptors_json())

    def clear(self) -> None:
        self._native.clear()

    def __len__(self) -> int:
        return len(self._native)

    def __reduce__(self) -> Any:
        raise TypeError("DAG-ML local implementation registries cannot be serialized")


def _unsigned_spec_json(value: Any, semantic_kind: str) -> str:
    document = json.loads(_coerce_json(value))
    if not isinstance(document, dict):
        raise TypeError(f"local {semantic_kind} spec must be a JSON object")
    document.setdefault("spec_fingerprint", "")
    return _coerce_json(document)


def _host_local_registration_options(
    semantic_kind: str,
    *,
    registry_key: str | None,
    provider_id: str,
    implementation_version: str,
    implementation_fingerprint: str | None,
    supported_controller_families: Iterable[str],
    runtime_requirements: Iterable[str],
    capabilities: Iterable[str],
) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "implementation_version": implementation_version,
        "implementation_fingerprint": (
            secrets.token_hex(32)
            if implementation_fingerprint is None
            else implementation_fingerprint
        ),
        "registry_key": (
            f"{semantic_kind}:python-local:{secrets.token_hex(16)}"
            if registry_key is None
            else registry_key
        ),
        "supported_controller_families": sorted(set(supported_controller_families)),
        "runtime_requirements": sorted(set(runtime_requirements)),
        "capabilities": sorted(set(capabilities)),
    }


def loss_execution_attestation(training_loss_role: Any, phase: str) -> dict[str, Any]:
    """Build the exact lineage attestation for an executed loss role."""

    return json.loads(
        _native_loss_execution_attestation_json(_coerce_json(training_loss_role), phase)
    )


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


class TrainingRequest(JsonContract):
    """Strict W1 training request with native projection semantics."""

    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_training_request_json(json_text)

    def project(self) -> "TrainingContractProjection":
        return TrainingContractProjection(project_training_request_json(self._json))


class TrainingContractProjection(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_training_contract_projection_json(json_text)


class TrainingOutcome(JsonContract):
    """Self-fingerprinted portable result of one native training run."""

    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_training_outcome_json(json_text)


class TrainingReplayRequest(JsonContract):
    """Self-fingerprinted attached replay request for a TrainingResult."""

    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_training_replay_request_json(json_text)


class TrainingReplayOutcome(JsonContract):
    """Self-fingerprinted portable result of one attached training replay."""

    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_training_replay_outcome_json(json_text)


class ParameterProjection(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_parameter_projection_json(json_text)


class CacheNamespace(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_cache_namespace_json(json_text)


class PortablePredictorPackage(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_portable_predictor_package_json(json_text)


class TrainingResult:
    """Owning native training result with explicit process-local detach."""

    __slots__ = ("_native",)

    def __init__(self, native: _NativeTrainingResult) -> None:
        self._native = native

    @property
    def is_attached(self) -> bool:
        """Whether callbacks, data views and artifact handles are retained."""

        return self._native.is_attached

    @property
    def process_local_artifact_count(self) -> int | None:
        return self._native.process_local_artifact_count

    @property
    def process_local_data_handle_count(self) -> int | None:
        return self._native.process_local_data_handle_count

    @property
    def process_local_data_view_count(self) -> int | None:
        return self._native.process_local_data_view_count

    @property
    def outcome_fingerprint(self) -> str:
        return self._native.outcome_fingerprint

    @property
    def outcome(self) -> TrainingOutcome:
        return TrainingOutcome(self.outcome_json())

    @property
    def execution_bundle(self) -> ExecutionBundle:
        return ExecutionBundle(self.execution_bundle_json())

    @property
    def score_set(self) -> dict[str, Any]:
        return json.loads(self.score_set_json())

    @property
    def outputs(self) -> list[dict[str, Any]]:
        return json.loads(self.outputs_json())

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        """Portable artifact records; process-local handles are never returned."""

        return json.loads(self.artifacts_json())

    @property
    def portable_prediction_caches(self) -> dict[str, Any] | None:
        payload = self.portable_prediction_caches_json()
        return None if payload is None else json.loads(payload)

    def detach(self) -> bool:
        """Release process-local resources, preserving every portable property."""

        return self._native.detach()

    def replay(
        self,
        request: Any,
        data_envelopes: Any,
        *,
        outcome_id: str,
        run_id: str,
        warnings: list[str] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> TrainingReplayOutcome:
        """Execute attached PREDICT/EXPLAIN replay while native resources live."""

        return TrainingReplayOutcome(
            self.replay_json(
                request,
                data_envelopes,
                outcome_id=outcome_id,
                run_id=run_id,
                warnings=warnings,
                diagnostics=diagnostics,
            )
        )

    def export_portable_predictor_package(
        self,
        package_id: str,
        *,
        fitted_artifact_mode: str = "allow_host_sidecar",
        artifact_load_mode: str = "host_sidecar",
    ) -> PortablePredictorPackage:
        """Export a signed portable predictor package JSON contract."""

        return PortablePredictorPackage(
            self.portable_predictor_package_json(
                package_id,
                fitted_artifact_mode=fitted_artifact_mode,
                artifact_load_mode=artifact_load_mode,
            )
        )

    def outcome_json(self) -> str:
        return self._native.outcome_json()

    def execution_bundle_json(self) -> str:
        return self._native.execution_bundle_json()

    def score_set_json(self) -> str:
        return self._native.score_set_json()

    def outputs_json(self) -> str:
        return self._native.outputs_json()

    def artifacts_json(self) -> str:
        return self._native.artifacts_json()

    def portable_prediction_caches_json(self) -> str | None:
        return self._native.portable_prediction_caches_json()

    def portable_predictor_package_json(
        self,
        package_id: str,
        *,
        fitted_artifact_mode: str = "allow_host_sidecar",
        artifact_load_mode: str = "host_sidecar",
    ) -> str:
        return self._native.portable_predictor_package_json(
            package_id,
            fitted_artifact_mode,
            artifact_load_mode,
        )

    def replay_json(
        self,
        request: Any,
        data_envelopes: Any,
        *,
        outcome_id: str,
        run_id: str,
        warnings: list[str] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> str:
        replay_request = TrainingReplayRequest(request)
        envelopes_json = _coerce_json(data_envelopes)
        warnings_json = _coerce_json([] if warnings is None else warnings)
        diagnostics_json = _coerce_json({} if diagnostics is None else diagnostics)
        return self._native.replay_json(
            replay_request.json(),
            envelopes_json,
            outcome_id,
            run_id,
            warnings_json,
            diagnostics_json,
        )


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


def project_training_request(request: Any) -> TrainingContractProjection:
    """Validate and project a W1 training request through dag-ml-core."""

    return TrainingContractProjection(
        project_training_request_json(_coerce_json(request))
    )


def sign_training_request_json(request: Any) -> str:
    """Sign and validate a W1 training request through dag-ml-core."""

    return _native_sign_training_request_json(_coerce_json(request))


def sign_training_request(request: Any) -> TrainingRequest:
    """Return a typed, signed W1 training request."""

    return TrainingRequest(sign_training_request_json(request))


def execute_training_json(
    request_json: str,
    data_envelopes_json: str,
    relations_json: str,
    training_influence_json: str,
    op_callback: Any,
    outcome_id: str,
    run_id: str,
    bundle_id: str,
    warnings_json: str = "[]",
    diagnostics_json: str = "{}",
) -> TrainingResult:
    """Run native training from already serialized strict JSON contracts."""

    return TrainingResult(
        _native_execute_training_json(
            request_json,
            data_envelopes_json,
            relations_json,
            training_influence_json,
            op_callback,
            outcome_id,
            run_id,
            bundle_id,
            warnings_json,
            diagnostics_json,
        )
    )


def execute_training(
    request: Any,
    data_envelopes: Any,
    relations: Any,
    training_influence: Any,
    op_callback: Any,
    *,
    outcome_id: str,
    run_id: str,
    bundle_id: str,
    warnings: Any = (),
    diagnostics: Any = None,
) -> TrainingResult:
    """Execute native DAG-ML training with an in-process Python controller.

    ``data_envelopes`` maps each exact ``node_id.input_name`` requirement key
    to its signed coordinator envelope. ``op_callback`` receives one native
    ``NodeTask`` dictionary and returns its ``NodeResult`` dictionary. DAG-ML
    owns orchestration, OOF scoring, SELECT, optional REFIT and outcome binding.
    """

    return execute_training_json(
        _coerce_json(request),
        _coerce_json(data_envelopes),
        _coerce_json(relations),
        _coerce_json(training_influence),
        op_callback,
        outcome_id,
        run_id,
        bundle_id,
        _coerce_json(warnings),
        _coerce_json({} if diagnostics is None else diagnostics),
    )


def replay_loaded_predictor_package_json(
    package: Any,
    request: Any,
    data_envelopes: Any,
    artifact_handles: Any,
    op_callback: Any,
    *,
    outcome_id: str,
    run_id: str,
    warnings: Any = (),
    diagnostics: Any = None,
) -> str:
    """Run stateless PREDICT/EXPLAIN replay from a portable package plus host sidecars.

    ``package`` is a signed ``PortablePredictorPackage`` contract and must not
    contain process-local handles. ``artifact_handles`` is the host-side sidecar
    map keyed by package artifact id, with ``HandleRef`` values supplied by the
    current runtime. ``request.phase`` selects ``PREDICT`` or ``EXPLAIN``; an
    ``EXPLAIN`` request returns explanation blocks and may also return the final
    bound predictions emitted by the requested package binding.
    """

    portable_package = PortablePredictorPackage(package)
    replay_request = TrainingReplayRequest(request)
    return _native_execute_loaded_predictor_replay_json(
        portable_package.json(),
        replay_request.json(),
        _coerce_json(data_envelopes),
        _coerce_json(artifact_handles),
        op_callback,
        outcome_id,
        run_id,
        _coerce_json(warnings),
        _coerce_json({} if diagnostics is None else diagnostics),
    )


def replay_loaded_predictor_package(
    package: Any,
    request: Any,
    data_envelopes: Any,
    artifact_handles: Any,
    op_callback: Any,
    *,
    outcome_id: str,
    run_id: str,
    warnings: Any = (),
    diagnostics: Any = None,
) -> TrainingReplayOutcome:
    """Run stateless PREDICT/EXPLAIN replay and return a validated replay outcome."""

    return TrainingReplayOutcome(
        replay_loaded_predictor_package_json(
            package,
            request,
            data_envelopes,
            artifact_handles,
            op_callback,
            outcome_id=outcome_id,
            run_id=run_id,
            warnings=warnings,
            diagnostics=diagnostics,
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
    "CacheNamespace",
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
    "LocalImplementationRegistry",
    "ParameterProjection",
    "PipelineDslSpec",
    "PortablePredictorPackage",
    "TrainingContractProjection",
    "TrainingOutcome",
    "TrainingReplayOutcome",
    "TrainingReplayRequest",
    "TrainingResult",
    "TrainingRequest",
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
    "execute_training",
    "execute_training_json",
    "replay_loaded_predictor_package",
    "replay_loaded_predictor_package_json",
    "fan_out_data_aware_branches",
    "fan_out_data_aware_branches_json",
    "fold_set_fingerprint_json",
    "loss_execution_attestation",
    "sample_relation_set_fingerprint_json",
    "project_training_request",
    "project_training_request_json",
    "sign_training_request",
    "sign_training_request_json",
    "validate_campaign_json",
    "validate_cache_namespace_json",
    "validate_controller_manifest_json",
    "validate_controller_manifest_list_json",
    "validate_execution_bundle_json",
    "validate_execution_plan_json",
    "validate_fold_set_json",
    "validate_graph_json",
    "validate_parameter_projection_json",
    "validate_pipeline_dsl_json",
    "validate_portable_predictor_package_json",
    "validate_training_contract_projection_json",
    "validate_training_outcome_json",
    "validate_training_request_json",
    "version",
]
