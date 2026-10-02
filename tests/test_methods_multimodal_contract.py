"""Declarative transport tests; the byte witness is deliberately not fitted."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_archive_v2_contract import (
    MULTIMODAL_PIPELINE_PROFILE, ArchiveV2ContractError, load_json,
    validate_multimodal_pipeline_payload, validate_schema,
)

ROOT = Path(__file__).resolve().parents[1]


def transport_fixture(host: str = "python") -> tuple[dict, dict, dict]:
    """Independent generic-shape witness, never a native-state fit surrogate."""
    recipe = {"schema_version": 1, "fusion": "early", "source_order": ["nir", "image", "series", "metadata"],
              "encoders": {"nir": {"kind": "standard_scaler", "with_mean": True, "with_std": True},
                           "image": {"kind": "tensor_pca", "n_components": 3, "whiten": False, "random_state": 7},
                           "series": {"kind": "tensor_pca", "n_components": 1, "whiten": False, "random_state": 31},
                           "metadata": {"kind": "column_transformer", "numeric_columns": [0], "categorical_columns": [1], "with_mean": True, "with_std": True, "handle_unknown": "ignore", "sparse_output": False, "drop": None}},
              "source_weights": {"nir": 1.0, "image": .5, "series": 0.0, "metadata": 1.0},
              "model": {"method_id": "models.regularized.ridge", "params": {"alpha": .1, "center_x": True, "center_y": True, "scale_x": False}}}
    schemas = {name: {"representation_id": repr_id, "input_shape": shape, "dtype": dtype,
                     "identity": json.dumps({"source_id": f"src{index}", "axis": {"unit": "µm", "coordinates": [1.25, 2.5]}}, ensure_ascii=False)}
               for index, (name, repr_id, shape, dtype) in enumerate((
                   ("nir", "signal_1d", [7], "float64"), ("image", "rgb_image", [3, 4, 3], "float32"),
                   ("series", "series_mv", [5, 2], "float64"), ("metadata", "tabular_mixed", [2], "object")))}
    saved = {"schema": "dagml.methods.multimodal.v1", "node_id": "model:/u07.v1", "params_fingerprint": "1" * 64,
             "target_names": ["y"], "recipe": recipe, "source_schemas": schemas,
             "state": list(b"N4MF" + (1).to_bytes(4, "little") + (2).to_bytes(4, "little") + bytes(16))}
    record = {"node_id": saved["node_id"], "params_fingerprint": saved["params_fingerprint"], "controller_id": f"controller:methods.{host}.multimodal",
              "artifact": {"id": "artifact:/complete.v1", "kind": "methods_multimodal_pipeline", "backend": "raw", "controller_id": f"controller:methods.{host}.multimodal", "plugin": f"dagml.methods.{host}.multimodal", "plugin_version": "1.0.0"}}
    plan = {"node_plans": {saved["node_id"]: {"params": {}, "params_fingerprint": saved["params_fingerprint"]}},
            "graph_plan": {"graph": {"nodes": [{"id": saved["node_id"], "operator": {"type": "N4mMultimodalPipeline", "recipe": copy.deepcopy(recipe), "source_schemas": copy.deepcopy(schemas)}}]}}}
    return record, saved, plan


def seal(record: dict, saved: dict) -> bytes:
    raw = json.dumps(saved, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    record["artifact"].update(uri=f"artifacts/{digest}.json", content_fingerprint=digest, size_bytes=len(raw))
    return raw


@pytest.mark.parametrize("host", ["python", "wasm", "r", "octave"])
def test_exact_closed_owners_and_generic_declared_shapes(host: str) -> None:
    record, saved, plan = transport_fixture(host)
    validate_multimodal_pipeline_payload(record, seal(record, saved), plan)
    for location in (record, record["artifact"]):
        location["controller_id"] = "controller:foreign"
        with pytest.raises(ArchiveV2ContractError, match="native_model_refusal"):
            validate_multimodal_pipeline_payload(record, seal(record, saved), plan)
        location["controller_id"] = f"controller:methods.{host}.multimodal"


@pytest.mark.parametrize("field", ["shape", "identity", "order", "recipe", "column", "scale", "flag_type", "unknown", "format", "magic", "boolean_byte", "state_size", "negative_weight", "seed", "components"])
def test_resigned_payload_cannot_change_signed_source_or_recipe(field: str) -> None:
    record, saved, plan = transport_fixture()
    if field == "shape": saved["source_schemas"]["image"]["input_shape"] = [3, 4, 1]
    elif field == "identity": saved["source_schemas"]["nir"]["identity"] = '{"unit":"changed"}'
    elif field == "order": saved["recipe"]["source_order"].reverse()
    elif field == "recipe": saved["recipe"]["model"]["params"]["alpha"] = .2
    elif field == "column": saved["recipe"]["encoders"]["metadata"]["numeric_columns"] = [1]
    elif field == "scale": saved["recipe"]["model"]["params"]["scale_x"] = True
    elif field == "flag_type": saved["recipe"]["encoders"]["nir"]["with_mean"] = 1
    elif field == "unknown": saved["pickle"] = "not-portable"
    elif field == "format": saved["state"][4] = 2
    elif field == "magic": saved["state"][3] = 69
    elif field == "boolean_byte": saved["state"][20] = True
    elif field == "state_size": saved["state"] = saved["state"][:12]
    elif field == "negative_weight": saved["recipe"]["source_weights"]["image"] = -.1
    elif field == "seed": saved["recipe"]["encoders"]["image"]["random_state"] = 2**32
    else: saved["recipe"]["encoders"]["image"]["n_components"] = 37
    with pytest.raises(ArchiveV2ContractError, match="native_model_refusal"):
        validate_multimodal_pipeline_payload(record, seal(record, saved), plan)


def test_selected_three_parameter_paths_are_bound_before_native_import() -> None:
    record, saved, plan = transport_fixture()
    plan["node_plans"][saved["node_id"]]["params"] = {"model__alpha": 2.0, "source_weights__image": .25, "transformers__image__n_components": 2}
    with pytest.raises(ArchiveV2ContractError, match="selected effective plan"):
        validate_multimodal_pipeline_payload(record, seal(record, saved), plan)
    saved["recipe"]["model"]["params"]["alpha"] = 2.0
    saved["recipe"]["source_weights"]["image"] = .25
    saved["recipe"]["encoders"]["image"]["n_components"] = 2
    validate_multimodal_pipeline_payload(record, seal(record, saved), plan)
    plan["node_plans"][saved["node_id"]]["params"]["unplanned"] = 1
    with pytest.raises(ArchiveV2ContractError, match="Unknown effective"):
        validate_multimodal_pipeline_payload(record, seal(record, saved), plan)


def test_manifest_additive_composite_family_is_closed() -> None:
    document = load_json(ROOT / "docs/contracts/archive-v2/fixtures/positive/native_portable_replay.json")
    schema = validate_schema(load_json(ROOT / "docs/contracts/archive-v2/archive_workspace_manifest.v2.schema.json"))
    reference = {"artifact_id": "artifact:composite.v1", "kind": "methods_multimodal_pipeline", "owner": "dag-ml", "format_version": 1, "member_path": "artifacts/" + "0" * 64 + ".json", "raw_sha256": "0" * 64, "semantic_fingerprint": "0" * 64, "semantic_profile": MULTIMODAL_PIPELINE_PROFILE}
    document["payloads"]["methods"]["multimodal_pipelines"] = [reference]
    assert not list(schema.iter_errors(document))
    for field, value in (("owner", "methods"), ("kind", "methods_role_pipeline"), ("semantic_profile", "dagml_methods_role_pipeline_raw_sha256"), ("format_version", 2), ("member_path", "../escape.json"), ("unknown", True)):
        changed = copy.deepcopy(document)
        changed["payloads"]["methods"]["multimodal_pipelines"][0][field] = value
        assert list(schema.iter_errors(changed)), field


@pytest.mark.parametrize("host", ["python", "wasm", "r", "octave"])
def test_native_plan_preserves_four_raw_sources_and_refuses_implicit_fusion(host: str) -> None:
    """Exercise the real compiler/planner without fitting a transport witness."""
    import dag_ml
    from dag_ml.multimodal_methods import manifest_for_host

    _, saved, _ = transport_fixture(host)
    node_id = "model:raw-u07.v1"
    operator = {"type": "N4mMultimodalPipeline", "recipe": saved["recipe"], "source_schemas": saved["source_schemas"]}
    binding = {"node_id": node_id, "input_name": "x", "request_id": "request:/raw-u07.v1",
               "schema_fingerprint": "1" * 64, "plan_fingerprint": "2" * 64,
               "output_representation": "feature_block_set", "source_ids": ["src0", "src1", "src2", "src3"]}
    manifest = manifest_for_host(host)
    dsl = {"id": "dsl:/raw-u07.v1", "input": {"representation": "feature_block_set"},
           "steps": [{"kind": "model", "id": node_id, "operator": operator,
                      "metadata": {"controller_id": manifest["controller_id"]}}], "data_bindings": [binding]}
    compiled = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, [manifest])
    plan = dag_ml.build_execution_plan("plan:/raw-u07.v1", compiled.graph, compiled.campaign_template, [manifest]).to_dict()
    assert "params" not in plan["node_plans"][node_id]
    assert plan["node_plans"][node_id]["data_bindings"][0]["source_ids"] == binding["source_ids"]
    assert plan["graph_plan"]["graph"]["nodes"][0]["operator"] == operator
    requirements = plan["controller_manifests"][manifest["controller_id"]]["data_requirements"]
    assert requirements["default_fusion"] == {"mode": "dict_by_source", "alignment": "sample_id", "adapter_id": None, "params": {}}
    assert requirements["ports"][0]["rank"] is None
    implicit = copy.deepcopy(manifest)
    implicit["data_requirements"]["default_fusion"] = None
    with pytest.raises(Exception, match="dagml.data_requirement.missing_multisource_fusion"):
        dag_ml.build_execution_plan("plan:/implicit-u07.v1", compiled.graph, compiled.campaign_template, [implicit])


def test_native_replay_transport_aligns_exact_ids_to_requested_order() -> None:
    """Decode native wire rows by identity without claiming a fitted witness."""
    from scripts.qualify_u07_methods_multimodal import replay_prediction_values

    outcome = {"outputs": [{"binding": {"node_id": "model:/transport.v1"},
                           "predictions": [{"sample_ids": ["sample:a", "sample:z"],
                                            "target_names": ["y"], "values": [[1.25], [2.5]]}]}]}
    before = copy.deepcopy(outcome)
    assert replay_prediction_values(outcome, ["sample:z", "sample:a"], ["y"]) == [2.5, 1.25]
    assert outcome == before


def test_native_replay_transport_refuses_ambiguous_ids_or_output_shape() -> None:
    from scripts.qualify_u07_methods_multimodal import replay_prediction_values

    outcome = {"outputs": [{"binding": {"node_id": "model:/transport.v1"},
                           "predictions": [{"sample_ids": ["sample:a", "sample:z"],
                                            "target_names": ["y"], "values": [[1.25], [2.5]]}]}]}
    for mutation in ("duplicate_id", "missing_id", "extra_output", "extra_prediction", "target", "nonfinite", "row_width"):
        changed = copy.deepcopy(outcome)
        block = changed["outputs"][0]["predictions"][0]
        if mutation == "duplicate_id": block["sample_ids"][1] = "sample:a"
        elif mutation == "missing_id": block["sample_ids"][1] = "sample:foreign"
        elif mutation == "extra_output": changed["outputs"].append(copy.deepcopy(changed["outputs"][0]))
        elif mutation == "extra_prediction": changed["outputs"][0]["predictions"].append(copy.deepcopy(block))
        elif mutation == "target": block["target_names"] = ["foreign"]
        elif mutation == "nonfinite": block["values"][0][0] = float("nan")
        else: block["values"][0].append(0.0)
        with pytest.raises(Exception, match="(?i)sample|output|prediction|target|finite|row|shape"):
            replay_prediction_values(changed, ["sample:z", "sample:a"], ["y"])
