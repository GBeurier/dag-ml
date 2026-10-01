"""Independent recipe checks; opaque N4ME bytes are not numerical evidence."""

import copy
import hashlib
import json

import pytest

from scripts.validate_archive_v2_contract import (
    ArchiveV2ContractError,
    validate_role_pipeline_recipe,
)


def _fixture():
    params = {"native_profile": "n4m.pls_role_pipeline.v1", "n_components": 1, "scale": False}
    base = [{"methodId": "models.pls.pls_regression", "params": {
        "n_components": 1, "solver": "nipals", "center_x": True, "center_y": True,
        "scale_x": False, "scale_y": False,
    }}]
    refit = copy.deepcopy(base)
    refit[0]["params"].update(n_components=3, scale_x=True, scale_y=True)
    node_params = {**params, "phase_controls": {
        "train_params": {}, "refit_params": {"n_components": 3, "scale": True},
    }}
    plan = {"node_plans": {"model:pls": {"params": node_params, "params_fingerprint": "a" * 64}},
            "graph_plan": {"graph": {"nodes": [{"id": "model:pls", "params": params,
                "operator": {"type": "N4mRolePipeline", "steps": base}}]}}}
    saved = {"steps": refit}
    record = {"node_id": "model:pls", "params_fingerprint": "a" * 64,
              "controller_id": "controller:methods.native.regression",
              "artifact": {"plugin": "dagml.methods.native.regression"}}
    return record, saved, plan


def _reseal(record, saved):
    raw = json.dumps(saved, separators=(",", ":")).encode()
    sha = hashlib.sha256(raw).hexdigest()
    record["artifact"].update(uri=f"artifacts/{sha}.json", content_fingerprint=sha, size_bytes=len(raw))
    return raw


def test_effective_refit_recipe_is_distinct_from_base_and_valid():
    record, saved, plan = _fixture()
    validate_role_pipeline_recipe(record, _reseal(record, saved), plan)


@pytest.mark.parametrize("mutation", [
    "component", "scale", "split_scale", "solver", "bool_as_int", "int_as_bool", "extra_param",
])
def test_resealed_saved_recipe_cannot_override_effective_plan(mutation):
    record, saved, plan = _fixture()
    params = saved["steps"][0]["params"]
    if mutation == "component":
        params["n_components"] = 2
    elif mutation == "scale":
        params.update(scale_x=False, scale_y=False)
    elif mutation == "split_scale":
        params["scale_y"] = False
    elif mutation == "solver":
        params["solver"] = "svd"
    elif mutation == "bool_as_int":
        params["scale_x"] = 1
    elif mutation == "int_as_bool":
        plan["node_plans"]["model:pls"]["params"]["phase_controls"]["refit_params"]["n_components"] = 1
        params["n_components"] = True
    else:
        params["epochs"] = 10
    with pytest.raises(ArchiveV2ContractError, match="RAW differs from the effective REFIT recipe"):
        validate_role_pipeline_recipe(record, _reseal(record, saved), plan)


@pytest.mark.parametrize("mutation", ["base_recipe", "owner", "internal_controls", "scale_alias", "float_components"])
def test_resealed_plan_still_requires_closed_profile(mutation):
    record, saved, plan = _fixture()
    graph = plan["graph_plan"]["graph"]["nodes"][0]
    if mutation == "base_recipe":
        graph["operator"]["steps"][0]["params"]["scale_x"] = 0
    elif mutation == "owner":
        record["artifact"]["plugin"] = "dagml.methods.wasm.regression"
    elif mutation == "internal_controls":
        graph["params"]["phase_controls"] = {}
    elif mutation == "scale_alias":
        graph["params"]["scale_x"] = False
    else:
        graph["params"]["n_components"] = 1.0
    with pytest.raises(ArchiveV2ContractError, match="native_model_refusal"):
        validate_role_pipeline_recipe(record, _reseal(record, saved), plan)
