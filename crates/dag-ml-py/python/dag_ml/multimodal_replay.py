"""DAG-owned U07 replay over an IO-owned independent raw cohort.

No dependency on the Python nirs4all SDK. Core owns reading the archive ZIP.

Install ``dag-ml[multimodal]`` for the optional Python Methods controller.
Metadata rows use lossless object storage. Published Methods 1.3.2 accepts
archives whose metadata schema declares ``object``; an archive with fixed-width
Unicode metadata needs a Methods binding that supports logical text storage.
Its signed descriptor is preserved, and unsupported bindings refuse replay.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise ImportError(
        "Python Methods multimodal execution requires dag-ml[multimodal] "
        "(nirs4all-methods, NumPy and scikit-learn)"
    ) from error


def replay_multimodal_predictor_package(package: Any, current: Mapping[str, Any], *,
                                        request_id: str = "replay:public.multimodal",
                                        outcome_id: str = "outcome:public.multimodal",
                                        run_id: str = "run:public.multimodal") -> dict[str, Any]:
    """Validate, sign and schedule PREDICT; never synthesize package trust.

    ``current`` is an independently built IO runtime input. Controller trust
    comes from DAG's installed fixed profile, independently of archive metadata.
    Native DAG validates the package and signs the current replay request.
    """
    import dag_ml as native
    from dag_ml.multimodal_methods import MethodsMultimodalController, manifest_for_host

    validated = native.PortablePredictorPackage(package.to_dict() if hasattr(package, "to_dict") else package)
    document = validated.to_dict()
    plan = document["effective_plan"]
    graph = plan["graph_plan"]["graph"]
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if len(models) != 1 or len(graph["nodes"]) != 1 or models[0]["operator"].get("type") != "N4mMultimodalPipeline":
        raise ValueError("Public raw replay requires one complete native multimodal model")
    node = models[0]; selected = plan["node_plans"][node["id"]]
    owner = selected["controller_id"]
    hosts = {f"controller:methods.{host}.multimodal": host for host in ("python", "wasm", "r", "octave")}
    if owner not in hosts or node.get("metadata", {}).get("controller_id") != owner:
        raise ValueError("Signed native producer owner differs from the fixed replay profile")
    if len(selected["data_bindings"]) != 1 or selected["data_bindings"][0]["source_ids"] != current["source_ids"]:
        raise ValueError("Current raw source binding differs from the signed package")
    saved_schemas = node["operator"]["source_schemas"]
    current_schemas = current["source_schemas"]
    if set(saved_schemas) != set(current_schemas):
        raise ValueError("Current raw source names differ")
    for name in saved_schemas:
        a, b = copy.deepcopy(current_schemas[name]), copy.deepcopy(saved_schemas[name])
        a["identity"], b["identity"] = json.loads(a["identity"]), json.loads(b["identity"])
        if a != b:
            raise ValueError(f"Current raw source schema differs: {name}")
    bindings = document["output_bindings"]
    if len(bindings) != 1 or bindings[0]["target_names"] != ["y"]:
        raise ValueError("Public U07 replay requires one output named y")
    relations = current["data_envelope"]["coordinator_relations"]
    relation_fingerprint = native.sample_relation_set_fingerprint_json(json.dumps(relations))
    envelopes = {}
    for requirement in document["execution_bundle"]["data_requirements"]:
        key = f"{requirement['node_id']}.{requirement['input_name']}"
        if key in envelopes:
            raise ValueError("Repeated native data requirement")
        envelope = {"schema_version": 1, "schema_fingerprint": requirement["schema_fingerprint"],
                    "plan_fingerprint": requirement["plan_fingerprint"], "relation_fingerprint": relation_fingerprint,
                    "data_content_fingerprint": current["data_content_fingerprint"], "target_content_fingerprint": None,
                    "coordinator_relations": relations}
        envelopes[key] = native.attach_predict_cohort_to_envelope(envelope, {
            "role": "inference", "relations": relations, "target_names": ["y"],
            "data_content_fingerprint": current["data_content_fingerprint"], "target_content_fingerprint": None,
        }).to_dict()
    if not envelopes:
        raise ValueError("Native predictor has no replay data requirements")
    request = native.sign_training_replay_request({"schema_version": 1, "request_id": request_id,
        "source_outcome_fingerprint": document["training_outcome"]["outcome_fingerprint"], "phase": "PREDICT",
        "data_envelope_keys": sorted(envelopes), "output_binding_ids": [bindings[0]["binding_id"]], "request_fingerprint": "0" * 64})
    sources = {}
    for name, source in current["sources"].items():
        values = np.asarray(source["rows"], dtype=object) if name == "metadata" else np.asarray(source["data"], dtype=np.dtype(saved_schemas[name]["dtype"])).reshape(source["shape"])
        sources[name] = {"sample_ids": source["sample_ids"], "descriptor": saved_schemas[name], "values": values}
    controller = MethodsMultimodalController(operators={node["id"]: node["operator"]}, sources=sources, targets=None,
        target_names=("y",), allow_fit=False, controller_id=owner, source_ids=tuple(current["source_ids"]),
        node_params={key: value.get("params", {}) for key, value in plan["node_plans"].items()})
    trusted = manifest_for_host(hosts[owner])
    try:
        outcome = native.replay_loaded_predictor_package(validated, request, envelopes, {}, controller.operator,
            outcome_id=outcome_id, run_id=run_id, artifact_callback=controller.artifact, trusted_controller_manifests=[trusted])
        audit = copy.deepcopy(controller.audit)
        if any(event.get("operation") in {"fit", "FIT_CV", "REFIT"} for event in audit):
            raise RuntimeError("Cold replay unexpectedly performed training")
        return {"outcome": outcome.to_dict(), "sample_ids": list(current["sample_ids"]), "audit": audit,
                "training_performed": False}
    finally:
        controller.close()
