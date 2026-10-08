import json
from pathlib import Path

import dag_ml
import dag_ml._dag_ml as native

manifest = json.loads(dag_ml.contract_manifest_json())
assert native.version() == dag_ml.version()
assert manifest["python_package_version"] == native.version()
assert "TrainingRequest" in manifest["python_facade_exports"]
assert callable(dag_ml.validate_training_request_json)

fixtures = Path("examples/fixtures/training")
request_json = (fixtures / "training_request_active_influence.v1.json").read_text()
package_json = (fixtures / "portable_predictor_package.v1.json").read_text()
projection_json = (fixtures / "parameter_projection_empty.v1.json").read_text()

request = dag_ml.TrainingRequest(request_json)
projection = request.project()
projected = projection.to_dict()
assert projected["request_id"] == "training:fixture.active_influence"
assert projected["request_fingerprint"] == (
    "df52b77b52dfb4e6436da726b22698079f2441cdd15b55d5de3c5e204bb73f2b"
)
assert projected["predictor_node_ids"] == ["model:base", "transform:snv"]
assert projected["outputs"] == [{
    "output_id": "output:prediction",
    "node_id": "model:base",
    "port_name": "oof",
    "prediction_level": "sample",
    "unit_level": "physical_sample",
    "prediction_kind": "regression_point",
    "target_names": ["protein"],
    "target_units": ["percent"],
    "class_labels": [[]],
    "output_order": "target_order",
    "target_space": "raw",
}]
assert projected["parameters"]["nodes"]["model:base"]["params"] == {
    "n_estimators": 100
}
assert dag_ml.project_training_request(request).to_dict() == projected

package = dag_ml.PortablePredictorPackage(package_json).to_dict()
assert package["package_id"] == "predictor:package.fixture"
assert package["package_fingerprint"] == (
    "7d5b7a33d90211de2676a43f45dd102d60910eec419a7daef427cc0fff228dd0"
)
assert package["predictor_node_ids"] == [
    "branch:b0.model:ridge",
    "branch:b1.augment:noise",
    "branch:b1.model:rf",
    "merge:stack.pred_plus_original.meta:ridge",
]
assert [(binding["binding_id"], binding["port_name"], binding["prediction_source"])
        for binding in package["output_bindings"]] == [
    ("output:meta.final", "oof", "final_refit")
]

parameter_projection = dag_ml.ParameterProjection(projection_json).to_dict()
assert parameter_projection["nodes"] == {
    "model:base": {
        "params": {"n_estimators": 100},
        "fit_params": {},
        "control_params": {},
        "structural_params": {},
    }
}
assert parameter_projection["requires_recompile"] is False
assert parameter_projection["structural_patch_count"] == 0
assert parameter_projection["patches_fingerprint"] == (
    "cea5f239e81001721b763cebf40cd71bca04972c51313fba335e0a96d7e81979"
)
assert parameter_projection["projection_fingerprint"] == (
    "9eff58f693db68df88c00637e38e860472099a90e9c8a271ed350fdcf67ca837"
)

missing_parameter_patches = json.loads(request_json)
del missing_parameter_patches["parameter_patches"]
try:
    dag_ml.validate_training_request_json(json.dumps(missing_parameter_patches))
except dag_ml.DagMlError as error:
    assert "parameter_patches" in str(error)
else:
    raise AssertionError("missing required parameter_patches was accepted")
assert native.version() == dag_ml.version() == manifest["python_package_version"] == "0.3.41"
assert callable(native.configure_methods_runtime)
print("Source041 facade/native/manifest and unchanged CI training API assertions PASS")
