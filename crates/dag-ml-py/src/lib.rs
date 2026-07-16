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

mod in_process;
mod training;

use dag_ml_core::{
    build_execution_plan, compile_pipeline_dsl, compile_pipeline_dsl_with_generation,
    compile_pipeline_dsl_with_generation_and_controller_registry, fan_out_data_aware_branches,
    fold_set_fingerprint, operator_variant_canonical_value, operator_variant_label_from_steps_json,
    parse_pipeline_dsl_json, CacheNamespace, CampaignSpec, ControllerManifest, ControllerRegistry,
    DagMlError as CoreDagMlError, ExecutionBundle, ExecutionPlan, ExternalDataPlanEnvelope,
    FoldSet, GraphSpec, HostControllerSpec, ParameterProjection, PortablePredictorPackage,
    SampleRelationSet,
    TrainingContractProjection, TrainingOutcome, TrainingReplayOutcome, TrainingReplayRequest,
    TrainingRequest,
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
    GraphSpec::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_campaign_json(json: &str) -> PyResult<()> {
    CampaignSpec::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
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
fn derive_controller_manifest_json(host_controller_spec_json: &str) -> PyResult<String> {
    let spec: HostControllerSpec =
        serde_json::from_str(host_controller_spec_json).map_err(py_serde_error)?;
    let manifest = spec.derive().map_err(py_core_error)?;
    serde_json::to_string(&manifest).map_err(py_serde_error)
}

#[pyfunction]
fn derive_controller_manifest_list_json(host_controller_specs_json: &str) -> PyResult<String> {
    let specs = serde_json::from_str::<Vec<HostControllerSpec>>(host_controller_specs_json)
        .map_err(py_serde_error)?;
    let manifests = derive_controller_manifests(specs)?;
    serde_json::to_string(&manifests).map_err(py_serde_error)
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
    ExecutionPlan::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_execution_bundle_json(json: &str) -> PyResult<()> {
    ExecutionBundle::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_training_request_json(json: &str) -> PyResult<()> {
    TrainingRequest::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn sample_relation_set_fingerprint_json(json: &str) -> PyResult<String> {
    let relations: SampleRelationSet = dag_ml_core::deserialize_external_contract(
        json,
        "sample relation set",
        dag_ml_core::DagMlError::CampaignValidation,
    )
    .map_err(py_core_error)?;
    relations.fingerprint().map_err(py_core_error)
}

#[pyfunction]
fn sign_training_request_json(json: &str) -> PyResult<String> {
    let mut request: TrainingRequest = dag_ml_core::deserialize_external_contract(
        json,
        "training request",
        dag_ml_core::DagMlError::CampaignValidation,
    )
    .map_err(py_core_error)?;
    request.request_fingerprint = request.compute_fingerprint().map_err(py_core_error)?;
    request.validate().map_err(py_core_error)?;
    serde_json::to_string(&request).map_err(py_serde_error)
}

#[pyfunction]
fn project_training_request_json(json: &str) -> PyResult<String> {
    let request = TrainingRequest::from_json(json).map_err(py_core_error)?;
    let projection = request.project().map_err(py_core_error)?;
    serde_json::to_string(&projection).map_err(py_serde_error)
}

#[pyfunction]
fn validate_training_contract_projection_json(json: &str) -> PyResult<()> {
    TrainingContractProjection::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_parameter_projection_json(json: &str) -> PyResult<()> {
    ParameterProjection::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_cache_namespace_json(json: &str) -> PyResult<()> {
    CacheNamespace::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_portable_predictor_package_json(json: &str) -> PyResult<()> {
    PortablePredictorPackage::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_training_outcome_json(json: &str) -> PyResult<()> {
    TrainingOutcome::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_training_replay_request_json(json: &str) -> PyResult<()> {
    TrainingReplayRequest::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
}

#[pyfunction]
fn validate_training_replay_outcome_json(json: &str) -> PyResult<()> {
    TrainingReplayOutcome::from_json(json)
        .map(|_| ())
        .map_err(py_core_error)
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
fn fan_out_data_aware_branches_json(dsl_json: &str, envelope_json: &str) -> PyResult<String> {
    let spec = parse_pipeline_dsl_json(dsl_json.as_bytes()).map_err(py_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        serde_json::from_str(envelope_json).map_err(py_serde_error)?;
    let expanded = fan_out_data_aware_branches(&spec, &envelope).map_err(py_core_error)?;
    serde_json::to_string(&expanded).map_err(py_serde_error)
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

/// Compute the cross-language `variant_label` (hex sha256) of a lowered operator sub-sequence
/// (Phase 5). `steps_json` is a JSON array of `PipelineDslStep`s — the SAME shape a generator branch
/// carries. The nirs4all host calls THIS for each of its operator-choice configs so the fingerprint
/// is computed over the EXACT SAME canonicalization + `serde_json::to_vec` (ryu) codepath dag-ml
/// uses to stamp per-variant reports — making the host label byte-identical to the report label by
/// construction (rather than relying on Python `json.dumps`, whose float formatting diverges from
/// Rust's for common params like `1e-05` / `1e-7` / 1-ULP shortest decimals).
///
/// The canonical form (the contract): a JSON array of steps in step order, each
/// `{"kind": <step-kind str>, "class": <operator FQN str>, "params": {<sorted params>}}` — a
/// bare-string operator renders `class` to itself, an object operator to its compact canonical JSON
/// text, structural steps to `class:""` / `params:{}`; keys sorted everywhere; numbers finite-only;
/// value forms preserved (`1` is not `1.0`).
#[pyfunction]
fn canonical_operator_variant_label(steps_json: &str) -> PyResult<String> {
    operator_variant_label_from_steps_json(steps_json).map_err(py_core_error)
}

/// The canonical `{"kind", "class", "params"}` array (as JSON text) that
/// [`canonical_operator_variant_label`] hashes — exposed for host-side debugging / fixture
/// inspection so a mismatch can be diffed against the exact bytes dag-ml fingerprints.
#[pyfunction]
fn canonical_operator_variant_value_json(steps_json: &str) -> PyResult<String> {
    let steps: Vec<dag_ml_core::PipelineDslStep> =
        serde_json::from_str(steps_json).map_err(py_serde_error)?;
    let canonical = operator_variant_canonical_value(&steps).map_err(py_core_error)?;
    serde_json::to_string(&canonical).map_err(py_serde_error)
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
    module.add_function(wrap_pyfunction!(derive_controller_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        derive_controller_manifest_list_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_pipeline_dsl_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_execution_plan_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_execution_bundle_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_training_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_relation_set_fingerprint_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sign_training_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(project_training_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_training_contract_projection_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_parameter_projection_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_cache_namespace_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_portable_predictor_package_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_training_outcome_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_training_replay_request_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_training_replay_outcome_json,
        module
    )?)?;
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
    module.add_function(wrap_pyfunction!(fan_out_data_aware_branches_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_execution_plan_json, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_operator_variant_label, module)?)?;
    module.add_function(wrap_pyfunction!(
        canonical_operator_variant_value_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        in_process::run_cv_refit_in_process,
        module
    )?)?;
    module.add_class::<training::TrainingResult>()?;
    module.add_function(wrap_pyfunction!(training::execute_training_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        training::execute_loaded_predictor_replay_json,
        module
    )?)?;
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
            {"id": "fold_set", "version": 1},
            {"id": "training_request", "version": 1},
            {"id": "training_contract_projection", "version": 1},
            {"id": "parameter_projection", "version": 1},
            {"id": "cache_namespace", "version": 1},
            {"id": "portable_predictor_package", "version": 1},
            {"id": "training_outcome", "version": 1},
            {"id": "training_replay_request", "version": 1},
            {"id": "training_replay_outcome", "version": 1}
        ],
        "capabilities": [
            "validate_json_contracts",
            "compile_pipeline_dsl",
            "compile_pipeline_dsl_with_generation",
            "compile_pipeline_dsl_with_controller_registry",
            "derive_controller_manifest_from_host_spec",
            "derive_controller_manifest_registry_from_host_specs",
            "build_execution_plan",
            "fold_set_fingerprint",
            "project_training_request",
            "validate_portable_predictor_package",
            "execute_training",
            "execute_training_replay",
            "execute_loaded_predictor_replay",
            "owning_training_result",
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
            "derive_controller_manifest_json",
            "derive_controller_manifest_list_json",
            "validate_pipeline_dsl_json",
            "validate_execution_plan_json",
            "validate_execution_bundle_json",
            "validate_training_request_json",
            "sample_relation_set_fingerprint_json",
            "sign_training_request_json",
            "project_training_request_json",
            "validate_training_contract_projection_json",
            "validate_parameter_projection_json",
            "validate_cache_namespace_json",
            "validate_portable_predictor_package_json",
            "validate_training_outcome_json",
            "validate_training_replay_request_json",
            "validate_training_replay_outcome_json",
            "validate_fold_set_json",
            "fold_set_fingerprint_json",
            "compile_pipeline_dsl_graph_json",
            "compile_pipeline_dsl_artifact_json",
            "compile_pipeline_dsl_artifact_with_controllers_json",
            "build_execution_plan_json",
            "canonical_operator_variant_label",
            "canonical_operator_variant_value_json",
            "run_cv_refit_in_process",
            "TrainingResult",
            "execute_training_json",
            "execute_loaded_predictor_replay_json"
        ],
        "wasm_exports": [
            "dag_ml_version",
            "contract_manifest_json",
            "validate_graph_json",
            "validate_campaign_json",
            "validate_controller_manifest_json",
            "validate_controller_manifest_list_json",
            "derive_controller_manifest_json",
            "derive_controller_manifest_list_json",
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

fn derive_controller_manifests(
    specs: Vec<HostControllerSpec>,
) -> PyResult<Vec<ControllerManifest>> {
    let mut registry = ControllerRegistry::new();
    let mut manifests = Vec::with_capacity(specs.len());
    for spec in specs {
        let manifest = spec.derive().map_err(py_core_error)?;
        registry.register(manifest.clone()).map_err(py_core_error)?;
        manifests.push(manifest);
    }
    Ok(manifests)
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
    fn fingerprint_and_signing_helpers_reject_duplicate_json_keys() {
        Python::initialize();
        let relation_error = sample_relation_set_fingerprint_json(
            r#"{"records":[],"records":[]}"#,
        )
        .expect_err("duplicate relation-set keys must be rejected");
        assert!(relation_error
            .to_string()
            .contains("duplicate JSON object key"));

        let request_error = sign_training_request_json(
            r#"{"schema_version":1,"schema_version":1}"#,
        )
        .expect_err("duplicate training-request keys must be rejected");
        assert!(request_error
            .to_string()
            .contains("duplicate JSON object key"));
    }

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
    fn canonical_operator_variant_label_matches_cross_repo_fixture() {
        // Phase 5 host contract: the PyO3 helper reproduces the pinned `variant_label` from the
        // cross-repo fixture's `steps_json`, over the SAME canonicalization + serde_json::to_vec
        // (ryu) path dag-ml uses to stamp reports. The host calls THIS (not Python json.dumps), so
        // its label is byte-identical to the report label by construction.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/contracts/operator_variant_label.v1.json"
        ))
        .unwrap();
        let case = &fixture["case"];
        let steps_json = case["steps_json"].as_str().unwrap();
        let expected = case["variant_label"].as_str().unwrap();

        // Direct Rust call into the #[pyfunction].
        let label = canonical_operator_variant_label(steps_json).expect("helper computes label");
        assert_eq!(
            label, expected,
            "PyO3 helper must match the pinned fixture label"
        );

        // And through the Python module surface, proving the binding is registered and callable.
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_dag_ml_test").unwrap();
            _dag_ml(py, &module).unwrap();
            let helper = module.getattr("canonical_operator_variant_label").unwrap();
            let via_python: String = helper.call1((steps_json,)).unwrap().extract().unwrap();
            assert_eq!(
                via_python, expected,
                "the registered Python helper must match the pinned fixture label"
            );
        });
    }

    #[test]
    fn derives_controller_manifest_from_host_spec_json() {
        let spec_json = r#"{
            "controller_id": "controller:nirs4all.model",
            "controller_version": "0.10.0",
            "operator_kind": "model",
            "priority": 20,
            "added_capabilities": ["needs_python_gil"],
            "operator_selectors": [{"aliases": ["Ridge"]}]
        }"#;
        let manifest_json =
            derive_controller_manifest_json(spec_json).expect("host spec derives manifest JSON");
        validate_controller_manifest_json(&manifest_json).expect("derived manifest validates");
        let manifest: ControllerManifest =
            serde_json::from_str(&manifest_json).expect("manifest JSON decodes");
        assert_eq!(manifest.controller_id.as_str(), "controller:nirs4all.model");
        assert_eq!(manifest.priority, 20);
        assert!(manifest
            .capabilities
            .contains(&dag_ml_core::ControllerCapability::NeedsPythonGil));

        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_dag_ml_test").unwrap();
            _dag_ml(py, &module).unwrap();
            let helper = module.getattr("derive_controller_manifest_json").unwrap();
            let via_python: String = helper.call1((spec_json,)).unwrap().extract().unwrap();
            assert_eq!(via_python, manifest_json);
        });
    }

    #[test]
    fn derives_controller_manifest_list_and_validates_registry() {
        let specs_json = r#"[
            {
                "controller_id": "controller:nirs4all.transform",
                "controller_version": "0.10.0",
                "operator_kind": "transform"
            },
            {
                "controller_id": "controller:nirs4all.model",
                "controller_version": "0.10.0",
                "operator_kind": "model"
            }
        ]"#;
        let manifests_json = derive_controller_manifest_list_json(specs_json)
            .expect("host specs derive manifest list JSON");
        validate_controller_manifest_list_json(&manifests_json)
            .expect("derived manifest list validates");
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(&manifests_json).expect("manifest list decodes");
        assert_eq!(manifests.len(), 2);
        assert_eq!(
            manifests[0].controller_id.as_str(),
            "controller:nirs4all.transform"
        );

        let duplicate_specs_json = r#"[
            {
                "controller_id": "controller:duplicate",
                "controller_version": "0.10.0",
                "operator_kind": "transform"
            },
            {
                "controller_id": "controller:duplicate",
                "controller_version": "0.10.0",
                "operator_kind": "model"
            }
        ]"#;
        let error = derive_controller_manifest_list_json(duplicate_specs_json)
            .expect_err("duplicate controller ids are rejected");
        assert!(error.to_string().contains("duplicate controller id"));
    }

    #[test]
    fn w10_training_contract_bindings_validate_and_project_fixtures() {
        Python::initialize();
        let request = include_str!(
            "../../../examples/fixtures/training/training_request_active_influence.v1.json"
        );
        validate_training_request_json(request).expect("training request validates");
        let projection = project_training_request_json(request).expect("training request projects");
        validate_training_contract_projection_json(&projection)
            .expect("native training projection validates");
        let projection_value: serde_json::Value = serde_json::from_str(&projection).unwrap();
        assert_eq!(
            projection_value["request_id"],
            "training:fixture.active_influence"
        );
        let mut unknown_graph_field = projection_value.clone();
        unknown_graph_field["plan"]["graph_plan"]["graph"]["unknown_projection_field"] =
            serde_json::json!(true);
        let error = validate_training_contract_projection_json(
            &serde_json::to_string(&unknown_graph_field).unwrap(),
        )
        .expect_err("unknown nested graph fields are rejected");
        assert!(
            error
                .to_string()
                .contains("plan.graph_plan.graph.unknown_projection_field"),
            "{error}"
        );
        validate_parameter_projection_json(include_str!(
            "../../../examples/fixtures/training/parameter_projection_empty.v1.json"
        ))
        .expect("parameter projection validates");
        let duplicate_parameter_projection =
            include_str!("../../../examples/fixtures/training/parameter_projection_empty.v1.json")
                .replacen(
                    "\"schema_version\": 1",
                    "\"schema_version\": 1, \"schema_version\": 1",
                    1,
                );
        assert!(validate_parameter_projection_json(&duplicate_parameter_projection).is_err());
        let nfc_projection = projection.replacen('{', "{\"é\":1,\"e\\u0301\":2,", 1);
        assert!(validate_training_contract_projection_json(&nfc_projection).is_err());
        validate_cache_namespace_json(include_str!(
            "../../../examples/fixtures/training/cache_namespace_fit_cv.v1.json"
        ))
        .expect("cache namespace validates");
        validate_portable_predictor_package_json(include_str!(
            "../../../examples/fixtures/training/portable_predictor_package.v1.json"
        ))
        .expect("portable predictor package validates");
        validate_training_outcome_json(include_str!(
            "../../../examples/fixtures/training/training_outcome_refit.v1.json"
        ))
        .expect("training outcome validates");

        Python::attach(|py| {
            let module = PyModule::new(py, "_dag_ml_test").unwrap();
            _dag_ml(py, &module).unwrap();
            for export in [
                "validate_training_request_json",
                "project_training_request_json",
                "validate_training_contract_projection_json",
                "validate_parameter_projection_json",
                "validate_cache_namespace_json",
                "validate_portable_predictor_package_json",
                "validate_training_outcome_json",
                "execute_training_json",
                "TrainingResult",
            ] {
                assert!(
                    module.getattr(export).is_ok(),
                    "missing Python export {export}"
                );
            }
        });
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
        assert!(manifest["python_exports"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("derive_controller_manifest_json")));
        assert!(manifest["python_exports"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("project_training_request_json")));
        assert!(manifest["python_exports"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("execute_training_json")));
        assert!(manifest["capabilities"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("owning_training_result")));
        assert!(manifest["wasm_exports"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("derive_controller_manifest_list_json")));
        assert_eq!(
            manifest["shared"]["fold_set_fixture_fingerprint"],
            SHARED_FOLD_SET_FINGERPRINT
        );
    }
}
