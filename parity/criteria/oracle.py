"""Independent semantic validator for DAG-ML criterion contracts."""

from __future__ import annotations

import re
from typing import Any, Callable

from parity.conformal.oracle import fingerprint_without


class CriteriaContractError(ValueError):
    """Raised when a criterion contract violates semantic invariants."""


FORBIDDEN_EXECUTABLE_KEYS = {
    "bytecode",
    "callable",
    "code",
    "function_source",
    "import_path",
    "module_path",
    "pickle",
    "serialized_callable",
    "source_code",
}
VERSIONED_ID = re.compile(r"^[^@\s\x00-\x1f\x7f-\x9f]+@[1-9][0-9]*$")
TOKEN = re.compile(r"^[^\s\x00-\x1f\x7f-\x9f]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CriteriaContractError(message)


def _require_fingerprint(document: dict[str, Any], field: str) -> None:
    declared = document.get(field)
    require(isinstance(declared, str) and SHA256.fullmatch(declared) is not None, field)
    require(declared == fingerprint_without(document, field), f"{field} mismatch")


def _reject_executable_payload(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, member in value.items():
            require(key.lower() not in FORBIDDEN_EXECUTABLE_KEYS, f"{path}.{key}")
            _reject_executable_payload(member, f"{path}.{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            _reject_executable_payload(member, f"{path}[{index}]")


def _validate_common_spec(document: dict[str, Any], id_field: str) -> None:
    require(document.get("schema_version") == 1, "schema_version")
    identifier = document.get(id_field)
    require(isinstance(identifier, str) and VERSIONED_ID.fullmatch(identifier) is not None, id_field)
    require(bool(document.get("task_kinds")), "task_kinds")
    require(bool(document.get("prediction_kinds")), "prediction_kinds")
    inputs = set(document.get("required_inputs", []))
    require({"target", "prediction"} <= inputs, "required_inputs")
    parameters = document.get("parameters")
    require(isinstance(parameters, dict), "parameters")
    _reject_executable_payload(parameters, "parameters")


def validate_loss_spec(document: dict[str, Any]) -> None:
    _validate_common_spec(document, "loss_id")
    require(document.get("objective") == "minimize", "loss objective")
    inputs = set(document["required_inputs"])
    capabilities = set(document.get("capabilities", []))
    if document.get("reduction") == "weighted_mean":
        require("sample_weight" in inputs, "weighted loss input")
        require("supports_sample_weights" in capabilities, "weighted loss capability")
    if "sample_weight" in inputs:
        require("supports_sample_weights" in capabilities, "sample weight capability")
    if "missing_mask" in inputs:
        require("supports_missing_mask" in capabilities, "missing mask capability")
    _require_fingerprint(document, "spec_fingerprint")


def validate_metric_spec(document: dict[str, Any]) -> None:
    _validate_common_spec(document, "metric_id")
    require(document.get("objective") in {"minimize", "maximize"}, "metric objective")
    require(bool(document.get("supported_levels")), "supported_levels")
    decomposition = document.get("decomposition")
    reduction = document.get("reduction")
    capabilities = set(document.get("capabilities", []))
    inputs = set(document["required_inputs"])
    if decomposition == "global":
        require(reduction == "global", "global metric reduction")
    else:
        require(reduction != "global", "decomposed metric reduction")
        require("decomposable" in capabilities, "decomposable capability")
    if reduction == "weighted_mean":
        require("sample_weight" in inputs, "weighted metric input")
        require("supports_sample_weights" in capabilities, "weighted metric capability")
    if "sample_weight" in inputs:
        require("supports_sample_weights" in capabilities, "sample weight capability")
    if "missing_mask" in inputs:
        require("supports_missing_mask" in capabilities, "missing mask capability")
    _require_fingerprint(document, "spec_fingerprint")


def validate_implementation_descriptor(document: dict[str, Any]) -> None:
    require(document.get("schema_version") == 1, "schema_version")
    semantic_id = document.get("semantic_id")
    require(
        isinstance(semantic_id, str) and VERSIONED_ID.fullmatch(semantic_id) is not None,
        "semantic_id",
    )
    for field in ("semantic_fingerprint", "implementation_fingerprint"):
        value = document.get(field)
        require(isinstance(value, str) and SHA256.fullmatch(value) is not None, field)
    for field in ("provider_id", "binding_id", "implementation_version"):
        value = document.get(field)
        require(isinstance(value, str) and TOKEN.fullmatch(value) is not None, field)
    portability = document.get("portability")
    replayability = document.get("replayability")
    registry_key = document.get("registry_key")
    if registry_key is not None:
        require(
            isinstance(registry_key, str) and TOKEN.fullmatch(registry_key) is not None,
            "registry_key",
        )
    if portability == "host_local":
        require(bool(registry_key), "host_local registry_key")
        require(replayability != "detached", "host_local replayability")
    elif portability == "portable_registered":
        require(bool(registry_key), "portable_registered registry_key")
        require(replayability == "registry_required", "portable_registered replayability")
    elif portability == "portable_built_in":
        require(registry_key is None, "portable_built_in registry_key")
        require(replayability == "detached", "portable_built_in replayability")
    else:
        raise CriteriaContractError("portability")
    _require_fingerprint(document, "descriptor_fingerprint")


def _validate_reference(
    reference: dict[str, Any],
    semantic_kind: str,
    id_field: str,
    spec_validator: Callable[[dict[str, Any]], None],
) -> None:
    spec = reference["spec"]
    implementation = reference["implementation"]
    spec_validator(spec)
    validate_implementation_descriptor(implementation)
    require(implementation["semantic_kind"] == semantic_kind, "semantic_kind")
    require(implementation["semantic_id"] == spec[id_field], "semantic_id")
    require(
        implementation["semantic_fingerprint"] == spec["spec_fingerprint"],
        "semantic_fingerprint",
    )


def validate_training_loss_role(document: dict[str, Any]) -> None:
    require(document.get("schema_version") == 1, "schema_version")
    phases = document.get("phases")
    require(bool(phases) and set(phases) <= {"FIT_CV", "REFIT"}, "loss phases")
    _validate_reference(document["loss"], "loss", "loss_id", validate_loss_spec)


def validate_metric_role(document: dict[str, Any]) -> None:
    require(document.get("schema_version") == 1, "schema_version")
    if document.get("missing_value_policy") == "skip":
        require(document.get("role") == "reporting", "missing value policy")
    _validate_reference(document["metric"], "metric", "metric_id", validate_metric_spec)
    require(document["level"] in document["metric"]["spec"]["supported_levels"], "level")


VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "loss_spec": validate_loss_spec,
    "metric_spec": validate_metric_spec,
    "implementation_descriptor": validate_implementation_descriptor,
    "training_loss_role": validate_training_loss_role,
    "metric_role": validate_metric_role,
}
