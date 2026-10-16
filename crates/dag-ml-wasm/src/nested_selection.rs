//! Browser nested grouped selection. Every loop, selection and score stays in
//! dag-ml-core; JavaScript owns operators through the synchronous controller
//! callback (for example `N4mWasmRegressionController`), and numeric rows
//! never cross this boundary.

use super::*;
use dag_ml_core::{
    execute_nested_selection_with_metrics, BuiltinNestedSelectionMetrics, ExternalDataPlanEnvelope,
    InMemoryDataProvider, NestedSelectionMetrics, RegressionMetricKind,
};

/// Host metrics through a synchronous callback
/// `(metricName, payloadJson) => number`, payload `{"y_true": [...], "y_pred": [...]}`.
struct JsMetrics(js_sys::Function);

impl NestedSelectionMetrics for JsMetrics {
    fn regression_metric(
        &self,
        metric: RegressionMetricKind,
        y_true: &[f64],
        y_pred: &[f64],
    ) -> CoreResult<f64> {
        let payload =
            serde_json::to_string(&serde_json::json!({"y_true": y_true, "y_pred": y_pred}))
                .map_err(CoreDagMlError::Serialization)?;
        let value = self
            .0
            .call2(
                &JsValue::NULL,
                &JsValue::from_str(metric.name()),
                &JsValue::from_str(&payload),
            )
            .map_err(|error| {
                CoreDagMlError::RuntimeValidation(format!(
                    "JS metrics `{}` threw: {error:?}",
                    metric.name()
                ))
            })?;
        value.as_f64().ok_or_else(|| {
            CoreDagMlError::RuntimeValidation(format!(
                "JS metrics `{}` must return a number",
                metric.name()
            ))
        })
    }
}

/// Run `execute_nested_selection` with JS controllers in the same worker.
/// `trusted_controller_manifests_json` resolves the request's graph nodes and
/// registers one bridge controller per manifest; `js_invoke` is
/// `(controllerId, taskJson, exactSeed) => nodeResultJson`. The envelope must
/// attest the request campaign's data bindings. The optional `js_metrics`
/// computes the selection and pooled metrics (`mse`, `rmse`, `mae`); without
/// it the DAG-ML metric registry is used. Returns the outcome JSON.
#[wasm_bindgen]
pub fn execute_nested_selection_json(
    request_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    js_invoke: &js_sys::Function,
    js_metrics: Option<js_sys::Function>,
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
    let outcome = match js_metrics {
        Some(callback) => execute_nested_selection_with_metrics(
            &request,
            &registry,
            &controllers,
            &provider,
            &JsMetrics(callback),
        ),
        None => execute_nested_selection_with_metrics(
            &request,
            &registry,
            &controllers,
            &provider,
            &BuiltinNestedSelectionMetrics,
        ),
    }
    .map_err(js_core_error)?;
    serde_json::to_string(&outcome).map_err(js_serde_error)
}
