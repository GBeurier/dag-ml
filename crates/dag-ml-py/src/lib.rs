//! Python bindings for DAG-ML JSON contracts.
//!
//! The first Python surface mirrors the browser/WASM surface: it validates,
//! compiles and plans serialized JSON contracts, but does not execute host
//! controllers or own data buffers.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};
use serde::de::DeserializeOwned;

use dag_ml_core::{
    build_execution_plan, compile_pipeline_dsl, compile_pipeline_dsl_with_generation,
    compile_pipeline_dsl_with_generation_and_controller_registry, fold_set_fingerprint,
    parse_pipeline_dsl_json, CampaignSpec, ControllerManifest, ControllerRegistry,
    DagMlError as CoreDagMlError, ExecutionBundle, ExecutionPlan, FoldSet, GraphSpec,
};

create_exception!(_dag_ml, DagMlError, PyException);
// ADR-11 per-category subclasses spanning the full taxonomy surface. Every
// refusal raises the subclass matching its category while staying catchable as
// the base `DagMlError`.
create_exception!(_dag_ml, DagMlValidationError, DagMlError);
create_exception!(_dag_ml, DagMlRuntimeError, DagMlError);
create_exception!(_dag_ml, DagMlDataError, DagMlError);
create_exception!(_dag_ml, DagMlControllerError, DagMlError);
create_exception!(_dag_ml, DagMlBundleError, DagMlError);
create_exception!(_dag_ml, DagMlLineageError, DagMlError);
create_exception!(_dag_ml, DagMlReplayError, DagMlError);
create_exception!(_dag_ml, DagMlSecurityError, DagMlError);
create_exception!(_dag_ml, DagMlCompatibilityError, DagMlError);
create_exception!(_dag_ml, DagMlInternalError, DagMlError);

/// Resolve the ADR-11 category string to its Python exception subclass. Unknown
/// categories fall back to the base `DagMlError`.
fn dag_ml_error_type_for_category<'py>(py: Python<'py>, category: &str) -> Bound<'py, PyType> {
    match category {
        "validation" => py.get_type::<DagMlValidationError>(),
        "runtime" => py.get_type::<DagMlRuntimeError>(),
        "data" => py.get_type::<DagMlDataError>(),
        "controller" => py.get_type::<DagMlControllerError>(),
        "bundle" => py.get_type::<DagMlBundleError>(),
        "lineage" => py.get_type::<DagMlLineageError>(),
        "replay" => py.get_type::<DagMlReplayError>(),
        "security" => py.get_type::<DagMlSecurityError>(),
        "compatibility" => py.get_type::<DagMlCompatibilityError>(),
        "internal" => py.get_type::<DagMlInternalError>(),
        _ => py.get_type::<DagMlError>(),
    }
}

const SHARED_FOLD_SET_FINGERPRINT: &str =
    "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c";

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn contract_manifest_json() -> PyResult<String> {
    serde_json::to_string(&contract_manifest()).map_err(py_serde_error)
}

#[pyfunction]
fn validate_graph_json(json: &str) -> PyResult<()> {
    validate_json::<GraphSpec>(json, GraphSpec::validate)
}

#[pyfunction]
fn validate_campaign_json(json: &str) -> PyResult<()> {
    validate_json::<CampaignSpec>(json, CampaignSpec::validate)
}

#[pyfunction]
fn validate_controller_manifest_json(json: &str) -> PyResult<()> {
    validate_json::<ControllerManifest>(json, ControllerManifest::validate)
}

#[pyfunction]
fn validate_controller_manifest_list_json(json: &str) -> PyResult<()> {
    controller_registry_from_json(json).map(|_| ())
}

#[pyfunction]
fn validate_pipeline_dsl_json(json: &str) -> PyResult<()> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(py_core_error)?;
    compile_pipeline_dsl_with_generation(&spec)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_execution_plan_json(json: &str) -> PyResult<()> {
    validate_json::<ExecutionPlan>(json, ExecutionPlan::validate)
}

#[pyfunction]
fn validate_execution_bundle_json(json: &str) -> PyResult<()> {
    validate_json::<ExecutionBundle>(json, ExecutionBundle::validate)
}

#[pyfunction]
fn validate_fold_set_json(json: &str) -> PyResult<()> {
    validate_json::<FoldSet>(json, FoldSet::validate)
}

#[pyfunction]
fn fold_set_fingerprint_json(json: &str) -> PyResult<String> {
    let fold_set = parse_and_validate::<FoldSet>(json, FoldSet::validate)?;
    fold_set_fingerprint(&fold_set).map_err(py_core_error)
}

#[pyfunction]
fn compile_pipeline_dsl_graph_json(json: &str) -> PyResult<String> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(py_core_error)?;
    let graph = compile_pipeline_dsl(&spec).map_err(py_core_error)?;
    serde_json::to_string(&graph).map_err(py_serde_error)
}

#[pyfunction]
fn compile_pipeline_dsl_artifact_json(json: &str) -> PyResult<String> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(py_core_error)?;
    let artifact = compile_pipeline_dsl_with_generation(&spec).map_err(py_core_error)?;
    serde_json::to_string(&artifact).map_err(py_serde_error)
}

#[pyfunction]
fn compile_pipeline_dsl_artifact_with_controllers_json(
    dsl_json: &str,
    controller_manifests_json: &str,
) -> PyResult<String> {
    let spec = parse_pipeline_dsl_json(dsl_json.as_bytes()).map_err(py_core_error)?;
    let registry = controller_registry_from_json(controller_manifests_json)?;
    let artifact = compile_pipeline_dsl_with_generation_and_controller_registry(&spec, &registry)
        .map_err(py_core_error)?;
    serde_json::to_string(&artifact).map_err(py_serde_error)
}

#[pyfunction]
fn build_execution_plan_json(
    plan_id: &str,
    graph_json: &str,
    campaign_json: &str,
    controller_manifests_json: &str,
) -> PyResult<String> {
    let graph = parse_and_validate::<GraphSpec>(graph_json, GraphSpec::validate)?;
    let campaign = parse_and_validate::<CampaignSpec>(campaign_json, CampaignSpec::validate)?;
    let registry = controller_registry_from_json(controller_manifests_json)?;
    let plan = build_execution_plan(plan_id.to_string(), graph, campaign, &registry)
        .map_err(py_core_error)?;
    serde_json::to_string(&plan).map_err(py_serde_error)
}

#[pymodule]
fn _dag_ml(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("DagMlError", py.get_type::<DagMlError>())?;
    module.add(
        "DagMlValidationError",
        py.get_type::<DagMlValidationError>(),
    )?;
    module.add("DagMlRuntimeError", py.get_type::<DagMlRuntimeError>())?;
    module.add("DagMlDataError", py.get_type::<DagMlDataError>())?;
    module.add(
        "DagMlControllerError",
        py.get_type::<DagMlControllerError>(),
    )?;
    module.add("DagMlBundleError", py.get_type::<DagMlBundleError>())?;
    module.add("DagMlLineageError", py.get_type::<DagMlLineageError>())?;
    module.add("DagMlReplayError", py.get_type::<DagMlReplayError>())?;
    module.add("DagMlSecurityError", py.get_type::<DagMlSecurityError>())?;
    module.add(
        "DagMlCompatibilityError",
        py.get_type::<DagMlCompatibilityError>(),
    )?;
    module.add("DagMlInternalError", py.get_type::<DagMlInternalError>())?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(contract_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_graph_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_campaign_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_controller_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_controller_manifest_list_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_pipeline_dsl_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_execution_plan_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_execution_bundle_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_fold_set_json, module)?)?;
    module.add_function(wrap_pyfunction!(fold_set_fingerprint_json, module)?)?;
    module.add_function(wrap_pyfunction!(compile_pipeline_dsl_graph_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        compile_pipeline_dsl_artifact_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        compile_pipeline_dsl_artifact_with_controllers_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_execution_plan_json, module)?)?;
    Ok(())
}

fn contract_manifest() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "crate": "dag-ml",
        "package": "dag-ml",
        "version": env!("CARGO_PKG_VERSION"),
        "surface": "json-contract-bindings",
        "contracts": [
            {"id": "graph_spec", "version": 1},
            {"id": "campaign_spec", "version": 1},
            {"id": "controller_manifest", "version": 1},
            {"id": "pipeline_dsl", "version": 1},
            {"id": "execution_plan", "version": 1},
            {"id": "execution_bundle", "version": 1},
            {"id": "fold_set", "version": 1}
        ],
        "capabilities": [
            "validate_json_contracts",
            "compile_pipeline_dsl",
            "compile_pipeline_dsl_with_generation",
            "compile_pipeline_dsl_with_controller_registry",
            "build_execution_plan",
            "fold_set_fingerprint",
            "structured_error_descriptors"
        ],
        "shared": {
            "fold_set_fixture_fingerprint": SHARED_FOLD_SET_FINGERPRINT
        },
        "python_exports": [
            "version",
            "contract_manifest_json",
            "validate_graph_json",
            "validate_campaign_json",
            "validate_controller_manifest_json",
            "validate_controller_manifest_list_json",
            "validate_pipeline_dsl_json",
            "validate_execution_plan_json",
            "validate_execution_bundle_json",
            "validate_fold_set_json",
            "fold_set_fingerprint_json",
            "compile_pipeline_dsl_graph_json",
            "compile_pipeline_dsl_artifact_json",
            "compile_pipeline_dsl_artifact_with_controllers_json",
            "build_execution_plan_json"
        ],
        "wasm_exports": [
            "dag_ml_version",
            "contract_manifest_json",
            "validate_graph_json",
            "validate_campaign_json",
            "validate_controller_manifest_json",
            "validate_controller_manifest_list_json",
            "validate_pipeline_dsl_json",
            "validate_execution_plan_json",
            "validate_execution_bundle_json",
            "validate_fold_set_json",
            "fold_set_fingerprint_json",
            "compile_pipeline_dsl_graph_json",
            "compile_pipeline_dsl_artifact_json",
            "compile_pipeline_dsl_artifact_with_controllers_json",
            "build_execution_plan_json"
        ]
    })
}

fn validate_json<T>(
    json: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> PyResult<()>
where
    T: DeserializeOwned,
{
    parse_and_validate::<T>(json, validate).map(|_| ())
}

fn parse_and_validate<T>(
    json: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> PyResult<T>
where
    T: DeserializeOwned,
{
    let value = serde_json::from_str::<T>(json).map_err(py_serde_error)?;
    validate(&value).map_err(py_core_error)?;
    Ok(value)
}

fn controller_registry_from_json(json: &str) -> PyResult<ControllerRegistry> {
    let manifests =
        serde_json::from_str::<Vec<ControllerManifest>>(json).map_err(py_serde_error)?;
    let mut registry = ControllerRegistry::new();
    for manifest in manifests {
        registry.register(manifest).map_err(py_core_error)?;
    }
    Ok(registry)
}

fn py_serde_error(error: serde_json::Error) -> PyErr {
    py_core_error(CoreDagMlError::Serialization(error))
}

fn py_core_error(error: CoreDagMlError) -> PyErr {
    Python::attach(|py| {
        let descriptor = error.descriptor();
        let instance: PyResult<Bound<'_, PyAny>> = (|| {
            let exc_type = dag_ml_error_type_for_category(py, descriptor.category.as_str());
            let instance = exc_type.call1((descriptor.message.clone(),))?;
            instance.setattr("category", descriptor.category.as_str())?;
            instance.setattr("code", descriptor.code.as_str())?;
            instance.setattr("severity", descriptor.severity.as_str())?;
            instance.setattr("remediation_hint", descriptor.remediation_hint.as_str())?;

            let context_json =
                serde_json::to_string(&descriptor.context).unwrap_or_else(|_| "{}".to_string());
            let descriptor_json =
                serde_json::to_string(&descriptor).unwrap_or_else(|_| descriptor.message.clone());
            let context = py.import("json")?.call_method1("loads", (&context_json,))?;
            instance.setattr("context", context)?;
            instance.setattr("context_json", context_json)?;
            instance.setattr("descriptor_json", descriptor_json)?;
            Ok(instance)
        })();

        match instance {
            Ok(instance) => PyErr::from_value(instance),
            Err(_) => DagMlError::new_err(error.to_string()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_subclasses_map_by_category() {
        Python::initialize();
        // Valid JSON but invalid graph -> "validation" -> DagMlValidationError.
        let validation_err =
            validate_graph_json(r#"{"id":"","interface":{},"nodes":[],"edges":[]}"#)
                .expect_err("invalid graph should fail");
        // Malformed JSON -> "compatibility" -> DagMlCompatibilityError.
        let compat_err = validate_graph_json("{").expect_err("malformed JSON should fail");
        Python::attach(|py| {
            let validation_value = validation_err.value(py);
            assert!(validation_value.is_instance_of::<DagMlValidationError>());
            assert!(validation_value.is_instance_of::<DagMlError>());
            assert!(!validation_value.is_instance_of::<DagMlCompatibilityError>());

            let compat_value = compat_err.value(py);
            assert!(compat_value.is_instance_of::<DagMlCompatibilityError>());
            assert!(compat_value.is_instance_of::<DagMlError>());
            assert_eq!(
                compat_value
                    .getattr("category")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "compatibility"
            );
            assert_eq!(
                compat_value
                    .getattr("code")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "serialization_error"
            );
        });
    }

    #[test]
    fn invalid_graph_raises_binding_error() {
        Python::initialize();
        let error = validate_graph_json(r#"{"id":"","interface":{},"nodes":[],"edges":[]}"#)
            .expect_err("invalid graph should fail");
        assert!(error.to_string().contains("graph"));
        Python::attach(|py| {
            let value = error.value(py);
            let category = value
                .getattr("category")
                .unwrap()
                .extract::<String>()
                .unwrap();
            let code = value.getattr("code").unwrap().extract::<String>().unwrap();
            let context_json = value
                .getattr("context_json")
                .unwrap()
                .extract::<String>()
                .unwrap();
            assert_eq!(category, "validation");
            assert_eq!(code, "graph_validation");
            assert!(context_json.contains("detail"));
        });
    }

    #[test]
    fn python_binding_rejects_d9_invalid_unit_join_plan() {
        Python::initialize();
        let error = validate_graph_json(include_str!(
            "../../../examples/fixtures/runtime/d9_invalid_unit_join_graph.json"
        ))
        .expect_err("D9 invalid unit-join graph should fail");

        assert!(
            error.to_string().contains("incompatible unit levels"),
            "unexpected D9 Python invalid unit join error: {error}"
        );
        Python::attach(|py| {
            let value = error.value(py);
            assert!(value.is_instance_of::<DagMlValidationError>());
            assert_eq!(
                value
                    .getattr("category")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "validation"
            );
        });
    }

    #[test]
    fn compiles_minimal_fixture_dsl() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root");
        let dsl = std::fs::read_to_string(root.join("examples/pipeline_dsl_generation.json"))
            .expect("read DSL fixture");
        let graph = compile_pipeline_dsl_graph_json(&dsl).expect("compile graph JSON");
        assert!(graph.contains("\"nodes\""));
    }

    #[test]
    fn contract_manifest_declares_binding_surface() {
        let manifest =
            serde_json::from_str::<serde_json::Value>(&contract_manifest_json().unwrap()).unwrap();
        assert_eq!(manifest["crate"], "dag-ml");
        assert_eq!(manifest["version"], env!("CARGO_PKG_VERSION"));
        assert!(manifest["python_exports"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("contract_manifest_json")));
        assert_eq!(
            manifest["shared"]["fold_set_fixture_fingerprint"],
            SHARED_FOLD_SET_FINGERPRINT
        );
    }
}
