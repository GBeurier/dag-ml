//! Browser nested grouped selection. Every loop, selection and score stays in
//! dag-ml-core; JavaScript owns operators through the synchronous controller
//! callback (for example `N4mWasmRegressionController`), and numeric rows
//! never cross this boundary.

use super::*;
use dag_ml_core::{execute_nested_selection, ExternalDataPlanEnvelope, InMemoryDataProvider};

/// Run `execute_nested_selection` with JS controllers in the same worker.
/// `trusted_controller_manifests_json` resolves the request's graph nodes and
/// registers one bridge controller per manifest; `js_invoke` is
/// `(controllerId, taskJson, exactSeed) => nodeResultJson`. The envelope must
/// attest the request campaign's data bindings. Returns the outcome JSON.
#[wasm_bindgen]
pub fn execute_nested_selection_json(
    request_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let request: dag_ml_core::NestedSelectionRequest = deserialize_external_contract(
        request_json,
        "nested selection request",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let registry = controller_registry_from_json(trusted_controller_manifests_json)?;
    let envelope: ExternalDataPlanEnvelope = deserialize_external_contract(
        data_envelope_json,
        "data envelope",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:wasm.nested_selection.provider").map_err(js_core_error)?,
        envelope,
    )
    .map_err(js_core_error)?;
    let manifests =
        serde_json::from_str::<Vec<ControllerManifest>>(trusted_controller_manifests_json)
            .map_err(js_serde_error)?;
    let mut controllers = RuntimeControllerRegistry::new();
    for manifest in manifests {
        controllers
            .register(Box::new(JsRuntimeController {
                id: manifest.controller_id,
                js_invoke: js_invoke.clone(),
            }))
            .map_err(js_core_error)?;
    }
    let outcome = execute_nested_selection(&request, &registry, &controllers, &provider)
        .map_err(js_core_error)?;
    serde_json::to_string(&outcome).map_err(js_serde_error)
}
