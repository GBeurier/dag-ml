//! JavaScript-owned process-local loss and metric implementations.

use dag_ml_core::{
    deserialize_external_contract, DagMlError as CoreDagMlError, ImplementationDescriptor,
    LocalImplementationRegistry as CoreLocalImplementationRegistry, LossExecutionAttestation,
    LossReference, MetricReference, NodeTask, Phase, PortabilityClass, TrainingLossRoleReference,
};
use wasm_bindgen::prelude::*;

use crate::{js_core_error, js_serde_error};

const JAVASCRIPT_BINDING_ID: &str = "binding:javascript";

#[wasm_bindgen]
pub struct LocalImplementationRegistry {
    registry: CoreLocalImplementationRegistry<js_sys::Function>,
}

#[wasm_bindgen]
pub struct TrainingLossBinding {
    implementation: js_sys::Function,
    required_attestation_json: String,
}

#[wasm_bindgen]
impl TrainingLossBinding {
    #[wasm_bindgen(getter)]
    pub fn invoke(&self) -> js_sys::Function {
        self.implementation.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn required_attestation_json(&self) -> String {
        self.required_attestation_json.clone()
    }
}

#[wasm_bindgen]
impl LocalImplementationRegistry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            registry: CoreLocalImplementationRegistry::new(),
        }
    }

    pub fn register_loss(
        &mut self,
        loss_reference_json: &str,
        implementation: js_sys::Function,
    ) -> Result<(), JsValue> {
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_javascript_descriptor(&loss.implementation)?;
        self.registry
            .register_loss(&loss, implementation)
            .map_err(js_core_error)
    }

    pub fn register_metric(
        &mut self,
        metric_reference_json: &str,
        implementation: js_sys::Function,
    ) -> Result<(), JsValue> {
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_javascript_descriptor(&metric.implementation)?;
        self.registry
            .register_metric(&metric, implementation)
            .map_err(js_core_error)
    }

    pub fn resolve_loss(&self, loss_reference_json: &str) -> Result<js_sys::Function, JsValue> {
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_javascript_descriptor(&loss.implementation)?;
        self.registry
            .resolve_loss(&loss)
            .cloned()
            .map_err(js_core_error)
    }

    pub fn resolve_training_loss(
        &self,
        training_loss_role_json: &str,
        phase: &str,
    ) -> Result<js_sys::Function, JsValue> {
        let role = parse_training_loss_role(training_loss_role_json)?;
        let phase = parse_phase(phase)?;
        validate_javascript_descriptor(&role.loss.implementation)?;
        LossExecutionAttestation::for_role(&role, phase).map_err(js_core_error)?;
        self.registry
            .resolve_loss(&role.loss)
            .cloned()
            .map_err(js_core_error)
    }

    pub fn bind_training_loss(
        &self,
        node_task_json: &str,
        #[wasm_bindgen(unchecked_param_type = "number")] role_index: JsValue,
    ) -> Result<TrainingLossBinding, JsValue> {
        let role_index = parse_role_index(role_index).map_err(js_core_error)?;
        let task = parse_node_task(node_task_json)?;
        let (role, attestation) = task
            .training_loss_binding(role_index)
            .map_err(js_core_error)?;
        validate_javascript_descriptor(&role.loss.implementation)?;
        let implementation = self
            .registry
            .resolve_loss(&role.loss)
            .cloned()
            .map_err(js_core_error)?;
        let attestation_json = serde_json::to_string(attestation).map_err(js_serde_error)?;
        Ok(TrainingLossBinding {
            implementation,
            required_attestation_json: attestation_json,
        })
    }

    pub fn resolve_metric(&self, metric_reference_json: &str) -> Result<js_sys::Function, JsValue> {
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_javascript_descriptor(&metric.implementation)?;
        self.registry
            .resolve_metric(&metric)
            .cloned()
            .map_err(js_core_error)
    }

    pub fn unregister_loss(
        &mut self,
        loss_reference_json: &str,
    ) -> Result<js_sys::Function, JsValue> {
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_javascript_descriptor(&loss.implementation)?;
        self.registry
            .unregister(&loss.implementation)
            .map_err(js_core_error)
    }

    pub fn unregister_metric(
        &mut self,
        metric_reference_json: &str,
    ) -> Result<js_sys::Function, JsValue> {
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_javascript_descriptor(&metric.implementation)?;
        self.registry
            .unregister(&metric.implementation)
            .map_err(js_core_error)
    }

    pub fn descriptors_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.registry.descriptors().collect::<Vec<_>>())
            .map_err(js_serde_error)
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.registry.len()
    }

    pub fn clear(&mut self) {
        self.registry.clear();
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        Err(JsValue::from_str(
            "DAG-ML local implementation registries cannot be serialized",
        ))
    }
}

impl Default for LocalImplementationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub fn loss_execution_attestation_json(
    training_loss_role_json: &str,
    phase: &str,
) -> Result<String, JsValue> {
    let role = parse_training_loss_role(training_loss_role_json)?;
    let phase = parse_phase(phase)?;
    let attestation = LossExecutionAttestation::for_role(&role, phase).map_err(js_core_error)?;
    serde_json::to_string(&attestation).map_err(js_serde_error)
}

fn parse_loss_reference(json: &str) -> Result<LossReference, JsValue> {
    let loss: LossReference =
        deserialize_external_contract(json, "loss reference", CoreDagMlError::CampaignValidation)
            .map_err(js_core_error)?;
    loss.validate().map_err(js_core_error)?;
    Ok(loss)
}

fn parse_metric_reference(json: &str) -> Result<MetricReference, JsValue> {
    let metric: MetricReference =
        deserialize_external_contract(json, "metric reference", CoreDagMlError::CampaignValidation)
            .map_err(js_core_error)?;
    metric.validate().map_err(js_core_error)?;
    Ok(metric)
}

fn parse_training_loss_role(json: &str) -> Result<TrainingLossRoleReference, JsValue> {
    let role: TrainingLossRoleReference = deserialize_external_contract(
        json,
        "training loss role",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    role.validate().map_err(js_core_error)?;
    Ok(role)
}

fn parse_node_task(json: &str) -> Result<NodeTask, JsValue> {
    deserialize_external_contract(json, "node task", CoreDagMlError::RuntimeValidation)
        .map_err(js_core_error)
}

fn parse_role_index(role_index: JsValue) -> Result<usize, CoreDagMlError> {
    role_index.as_f64().map_or_else(
        || {
            Err(CoreDagMlError::RuntimeValidation(
                "role_index must be a non-negative safe integer".to_string(),
            ))
        },
        validate_role_index,
    )
}

fn validate_role_index(role_index: f64) -> Result<usize, CoreDagMlError> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if !role_index.is_finite()
        || role_index < 0.0
        || role_index.fract() != 0.0
        || role_index > MAX_SAFE_INTEGER
        || role_index > usize::MAX as f64
    {
        return Err(CoreDagMlError::RuntimeValidation(
            "role_index must be a non-negative safe integer".to_string(),
        ));
    }
    Ok(role_index as usize)
}

fn parse_phase(phase: &str) -> Result<Phase, JsValue> {
    serde_json::from_value(serde_json::Value::String(phase.to_string())).map_err(|_| {
        js_core_error(CoreDagMlError::CampaignValidation(format!(
            "unsupported training loss phase `{phase}`"
        )))
    })
}

fn validate_javascript_descriptor(descriptor: &ImplementationDescriptor) -> Result<(), JsValue> {
    if descriptor.binding_id != JAVASCRIPT_BINDING_ID {
        return Err(js_core_error(CoreDagMlError::CampaignValidation(format!(
            "JavaScript local implementation requires binding_id `{JAVASCRIPT_BINDING_ID}`, got `{}`",
            descriptor.binding_id
        ))));
    }
    if descriptor.portability == PortabilityClass::PortableBuiltIn {
        return Err(js_core_error(CoreDagMlError::CampaignValidation(
            "JavaScript local implementation registry rejects portable_builtin descriptors"
                .to_string(),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    fn javascript_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/criteria/javascript_local_implementations.v1.json"
        ))
        .unwrap()
    }

    #[test]
    fn javascript_descriptors_are_binding_local_and_phase_checked() {
        let fixture = javascript_fixture();
        let loss: LossReference =
            serde_json::from_value(fixture["loss_reference"].clone()).unwrap();
        validate_javascript_descriptor(&loss.implementation).unwrap();

        let role: TrainingLossRoleReference =
            serde_json::from_value(fixture["training_loss_role"].clone()).unwrap();
        LossExecutionAttestation::for_role(&role, Phase::FitCv).unwrap();
        LossExecutionAttestation::for_role(&role, Phase::Refit).unwrap();
        assert!(LossExecutionAttestation::for_role(&role, Phase::Predict).is_err());
    }

    #[test]
    fn javascript_task_binding_requires_exact_native_attestation() {
        let fixture = javascript_fixture();
        let role: TrainingLossRoleReference =
            serde_json::from_value(fixture["training_loss_role"].clone()).unwrap();
        let attestation = LossExecutionAttestation::for_role(&role, Phase::FitCv).unwrap();
        let task_json = json!({
            "run_id": "run:javascript-local-fit-cv",
            "node_plan": {
                "node_id": "model:custom",
                "kind": "model",
                "controller_id": "controller:javascript-local",
                "controller_version": "1.0.0",
                "supported_phases": ["FIT_CV", "REFIT"],
                "controller_capabilities": [
                    "deterministic",
                    "supports_configurable_loss",
                    "supports_custom_loss",
                    "supports_differentiable_loss"
                ],
                "training_losses": [role],
                "fit_scope": "fold_train",
                "rng_policy": "uses_core_seed",
                "artifact_policy": "serializable",
                "input_nodes": [],
                "output_nodes": [],
                "shape_plan": null,
                "data_bindings": [],
                "params": {},
                "params_fingerprint": "0".repeat(64)
            },
            "phase": "FIT_CV",
            "variant_id": null,
            "variant": null,
            "fold_id": "fold:0",
            "branch_path": [],
            "input_handles": {},
            "data_views": {},
            "prediction_inputs": {},
            "artifact_inputs": {},
            "required_loss_attestations": [attestation],
            "seed": 42
        });
        let task = parse_node_task(&task_json.to_string()).unwrap();
        task.validate_required_loss_attestations().unwrap();

        let mut tampered = task_json;
        tampered["required_loss_attestations"][0]["loss_id"] =
            Value::String("example.loss.tampered@1".to_string());
        let task = parse_node_task(&tampered.to_string()).unwrap();
        assert!(task.validate_required_loss_attestations().is_err());
    }

    #[test]
    fn javascript_role_index_rejects_lossy_numeric_values() {
        assert_eq!(validate_role_index(0.0).unwrap(), 0);
        for invalid in [-1.0, 0.5, f64::NAN, f64::INFINITY, 9_007_199_254_740_992.0] {
            assert!(validate_role_index(invalid).is_err());
        }
    }
}
