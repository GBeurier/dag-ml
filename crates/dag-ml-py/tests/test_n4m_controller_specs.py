"""n4m method manifests derive role controller specs through the Python facade."""

import dag_ml


def test_n4m_manifest_derives_one_resolvable_controller_per_role() -> None:
    manifest = {
        "abi": "2.13.0",
        "methods": [
            {
                "method_id": "models.pls.pls_regression",
                "roles": ["transformer", "regressor"],
                "node_kinds": ["model", "transform"],
            },
            {"method_id": "filters.y_outlier", "roles": ["sample_filter"], "node_kinds": ["exclude"]},
        ],
    }
    specs = dag_ml.n4m_host_controller_specs(manifest).to_dict()
    assert [spec["controller_id"] for spec in specs] == [
        "controller:n4m.transformer",
        "controller:n4m.regressor",
        "controller:n4m.sample_filter",
    ]
    assert specs[1]["operator_selectors"] == [{"refs": ["n4m:models.pls.pls_regression"]}]
    manifests = dag_ml.derive_controller_manifests(specs).to_dict()
    assert [manifest["operator_kind"] for manifest in manifests] == ["transform", "model", "exclude"]
