//! Browser-friendly bindings for DAG-ML JSON contracts.
//!
//! The WASM surface intentionally exposes validation, DSL compilation and plan
//! construction over UTF-8 JSON strings only. Controller execution, host
//! artifacts and data buffers stay outside this crate.

use serde::de::DeserializeOwned;
use wasm_bindgen::prelude::*;

use dag_ml_core::{
    build_execution_plan, compile_pipeline_dsl, compile_pipeline_dsl_with_generation,
    compile_pipeline_dsl_with_generation_and_controller_registry, fold_set_fingerprint,
    parse_pipeline_dsl_json, CampaignSpec, ControllerManifest, ControllerRegistry,
    DagMlError as CoreDagMlError, ExecutionBundle, ExecutionPlan, FoldSet, GraphSpec,
};

const SHARED_FOLD_SET_FINGERPRINT: &str =
    "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c";

#[wasm_bindgen]
pub fn dag_ml_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn contract_manifest_json() -> Result<String, JsValue> {
    serde_json::to_string(&contract_manifest()).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn validate_graph_json(json: &str) -> Result<(), JsValue> {
    validate_json::<GraphSpec>(json, GraphSpec::validate)
}

#[wasm_bindgen]
pub fn validate_campaign_json(json: &str) -> Result<(), JsValue> {
    validate_json::<CampaignSpec>(json, CampaignSpec::validate)
}

#[wasm_bindgen]
pub fn validate_controller_manifest_json(json: &str) -> Result<(), JsValue> {
    validate_json::<ControllerManifest>(json, ControllerManifest::validate)
}

#[wasm_bindgen]
pub fn validate_controller_manifest_list_json(json: &str) -> Result<(), JsValue> {
    controller_registry_from_json(json).map(|_| ())
}

#[wasm_bindgen]
pub fn validate_pipeline_dsl_json(json: &str) -> Result<(), JsValue> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(js_core_error)?;
    compile_pipeline_dsl_with_generation(&spec)
        .map(|_| ())
        .map_err(js_core_error)
}

#[wasm_bindgen]
pub fn validate_execution_plan_json(json: &str) -> Result<(), JsValue> {
    validate_json::<ExecutionPlan>(json, ExecutionPlan::validate)
}

#[wasm_bindgen]
pub fn validate_execution_bundle_json(json: &str) -> Result<(), JsValue> {
    validate_json::<ExecutionBundle>(json, ExecutionBundle::validate)
}

#[wasm_bindgen]
pub fn validate_fold_set_json(json: &str) -> Result<(), JsValue> {
    validate_json::<FoldSet>(json, FoldSet::validate)
}

#[wasm_bindgen]
pub fn fold_set_fingerprint_json(json: &str) -> Result<String, JsValue> {
    let fold_set = parse_and_validate::<FoldSet>(json, FoldSet::validate)?;
    fold_set_fingerprint(&fold_set).map_err(js_core_error)
}

#[wasm_bindgen]
pub fn compile_pipeline_dsl_graph_json(json: &str) -> Result<String, JsValue> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(js_core_error)?;
    let graph = compile_pipeline_dsl(&spec).map_err(js_core_error)?;
    serde_json::to_string(&graph).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn compile_pipeline_dsl_artifact_json(json: &str) -> Result<String, JsValue> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(js_core_error)?;
    let artifact = compile_pipeline_dsl_with_generation(&spec).map_err(js_core_error)?;
    serde_json::to_string(&artifact).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn compile_pipeline_dsl_artifact_with_controllers_json(
    dsl_json: &str,
    controller_manifests_json: &str,
) -> Result<String, JsValue> {
    let spec = parse_pipeline_dsl_json(dsl_json.as_bytes()).map_err(js_core_error)?;
    let registry = controller_registry_from_json(controller_manifests_json)?;
    let artifact = compile_pipeline_dsl_with_generation_and_controller_registry(&spec, &registry)
        .map_err(js_core_error)?;
    serde_json::to_string(&artifact).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn build_execution_plan_json(
    plan_id: &str,
    graph_json: &str,
    campaign_json: &str,
    controller_manifests_json: &str,
) -> Result<String, JsValue> {
    let graph = parse_and_validate::<GraphSpec>(graph_json, GraphSpec::validate)?;
    let campaign = parse_and_validate::<CampaignSpec>(campaign_json, CampaignSpec::validate)?;
    let registry = controller_registry_from_json(controller_manifests_json)?;
    let plan = build_execution_plan(plan_id.to_string(), graph, campaign, &registry)
        .map_err(js_core_error)?;
    serde_json::to_string(&plan).map_err(js_serde_error)
}

fn validate_json<T>(
    json: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> Result<(), JsValue>
where
    T: DeserializeOwned,
{
    parse_and_validate::<T>(json, validate).map(|_| ())
}

fn parse_and_validate<T>(
    json: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> Result<T, JsValue>
where
    T: DeserializeOwned,
{
    let value = serde_json::from_str::<T>(json).map_err(js_serde_error)?;
    validate(&value).map_err(js_core_error)?;
    Ok(value)
}

fn controller_registry_from_json(json: &str) -> Result<ControllerRegistry, JsValue> {
    let manifests =
        serde_json::from_str::<Vec<ControllerManifest>>(json).map_err(js_serde_error)?;
    let mut registry = ControllerRegistry::new();
    for manifest in manifests {
        registry.register(manifest).map_err(js_core_error)?;
    }
    Ok(registry)
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

fn js_serde_error(error: serde_json::Error) -> JsValue {
    js_core_error(CoreDagMlError::Serialization(error))
}

fn js_core_error(error: CoreDagMlError) -> JsValue {
    let payload = error
        .descriptor_json()
        .unwrap_or_else(|_| error.to_string());
    JsValue::from_str(&payload)
}
