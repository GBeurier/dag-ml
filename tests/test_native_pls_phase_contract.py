"""Pure public binding proof for the additive closed native PLS contract."""
import pytest
import dag_ml

PROFILE = "n4m.pls_role_pipeline.v1"


def params(**overrides):
    return {"native_profile": PROFILE, "n_components": 2, "scale": True, **overrides}


def test_public_closed_profile_compiles_exact_native_owner_and_atomic_scale():
    contract = dag_ml.methods_pls_role_pipeline_contract(params(scale=False))
    assert contract["manifest"]["controller_id"] == "controller:methods.native.regression"
    assert contract["manifest"]["controller_version"] == "1.0.0"
    last = contract["operator"]["steps"][-1]
    assert last["methodId"] == "models.pls.pls_regression"
    assert last["params"]["scale_x"] is last["params"]["scale_y"] is False
    assert last["params"]["n_components"] == 2


@pytest.mark.parametrize("key,value", [("tol", 1e-6), ("epochs", 3), ("scale_x", False), ("warm_start", True), ("phase_controls", None)])
def test_public_closed_profile_refuses_unexecuted_or_internal_controls(key, value):
    with pytest.raises(Exception, match="closed parameters|internal"):
        dag_ml.methods_pls_role_pipeline_contract(params(**{key: value}))


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, 2**31])
def test_public_closed_profile_requires_genuine_positive_integer_components(value):
    with pytest.raises(Exception, match="closed parameters|positive integer"):
        dag_ml.methods_pls_role_pipeline_contract(params(n_components=value))


def test_public_recipe_keeps_native_sg_interp_without_data_or_library():
    contract = dag_ml.methods_pls_role_pipeline_contract(params(pipeline={
        "schema_version": 1, "pipeline_type": "n4m.snv_savgol_smooth.v1",
        "savgol_window": 7, "savgol_poly_degree": 2,
    }))
    steps = contract["operator"]["steps"]
    assert [s["methodId"] for s in steps] == [
        "preprocessing.scatter.snv", "preprocessing.derivatives.savitzky_golay",
        "models.pls.pls_regression",
    ]
    assert steps[1]["params"]["mode"] == "interp"
    assert steps[1]["params"]["deriv"] == 0


def test_public_inspection_does_not_accept_mutable_or_text_payloads():
    for payload in ("N4ME", bytearray(b"N4ME"), [78, 52]):
        with pytest.raises(TypeError, match="requires bytes"):
            dag_ml.inspect_methods_role_pipeline_params(payload, "/missing/libn4m.so")
