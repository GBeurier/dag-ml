//! Python-owned process-local loss and metric implementations.

use dag_ml_core::{
    deserialize_external_contract, DagMlError as CoreDagMlError, ImplementationCapability,
    ImplementationDescriptor, LocalImplementationRegistry, LossExecutionAttestation, LossReference,
    MetricReference, Phase, PortabilityClass, TrainingLossRoleReference,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::py_core_error;

const PYTHON_BINDING_ID: &str = "binding:python";

#[pyclass(name = "LocalImplementationRegistry")]
pub(crate) struct PyLocalImplementationRegistry {
    registry: LocalImplementationRegistry<Py<PyAny>>,
}

#[pymethods]
impl PyLocalImplementationRegistry {
    #[new]
    fn new() -> Self {
        Self {
            registry: LocalImplementationRegistry::new(),
        }
    }

    fn register_loss(
        &mut self,
        py: Python<'_>,
        loss_reference_json: &str,
        implementation: Py<PyAny>,
    ) -> PyResult<()> {
        ensure_callable(py, &implementation)?;
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_python_descriptor(&loss.implementation)?;
        self.registry
            .register_loss(&loss, implementation)
            .map_err(py_core_error)
    }

    fn register_metric(
        &mut self,
        py: Python<'_>,
        metric_reference_json: &str,
        implementation: Py<PyAny>,
    ) -> PyResult<()> {
        ensure_callable(py, &implementation)?;
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_python_descriptor(&metric.implementation)?;
        self.registry
            .register_metric(&metric, implementation)
            .map_err(py_core_error)
    }

    fn resolve_loss(&self, py: Python<'_>, loss_reference_json: &str) -> PyResult<Py<PyAny>> {
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_python_descriptor(&loss.implementation)?;
        self.registry
            .resolve_loss(&loss)
            .map(|implementation| implementation.clone_ref(py))
            .map_err(py_core_error)
    }

    fn resolve_training_loss(
        &self,
        py: Python<'_>,
        training_loss_role_json: &str,
        phase: &str,
    ) -> PyResult<Py<PyAny>> {
        let role = parse_training_loss_role(training_loss_role_json)?;
        let phase = parse_phase(phase)?;
        validate_python_descriptor(&role.loss.implementation)?;
        LossExecutionAttestation::for_role(&role, phase).map_err(py_core_error)?;
        self.registry
            .resolve_loss(&role.loss)
            .map(|implementation| implementation.clone_ref(py))
            .map_err(py_core_error)
    }

    fn resolve_metric(&self, py: Python<'_>, metric_reference_json: &str) -> PyResult<Py<PyAny>> {
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_python_descriptor(&metric.implementation)?;
        self.registry
            .resolve_metric(&metric)
            .map(|implementation| implementation.clone_ref(py))
            .map_err(py_core_error)
    }

    fn unregister_loss(&mut self, loss_reference_json: &str) -> PyResult<Py<PyAny>> {
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_python_descriptor(&loss.implementation)?;
        self.registry
            .unregister(&loss.implementation)
            .map_err(py_core_error)
    }

    fn unregister_metric(&mut self, metric_reference_json: &str) -> PyResult<Py<PyAny>> {
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_python_descriptor(&metric.implementation)?;
        self.registry
            .unregister(&metric.implementation)
            .map_err(py_core_error)
    }

    fn descriptors_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.registry.descriptors().collect::<Vec<_>>())
            .map_err(crate::py_serde_error)
    }

    fn clear(&mut self) {
        self.registry.clear();
    }

    fn __len__(&self) -> usize {
        self.registry.len()
    }

    fn __reduce__(&self) -> PyResult<Py<PyAny>> {
        Err(PyTypeError::new_err(
            "DAG-ML local implementation registries cannot be serialized",
        ))
    }
}

#[pyfunction]
pub(crate) fn loss_execution_attestation_json(
    training_loss_role_json: &str,
    phase: &str,
) -> PyResult<String> {
    let role = parse_training_loss_role(training_loss_role_json)?;
    let phase = parse_phase(phase)?;
    let attestation = LossExecutionAttestation::for_role(&role, phase).map_err(py_core_error)?;
    serde_json::to_string(&attestation).map_err(crate::py_serde_error)
}

fn parse_loss_reference(json: &str) -> PyResult<LossReference> {
    let loss: LossReference =
        deserialize_external_contract(json, "loss reference", CoreDagMlError::CampaignValidation)
            .map_err(py_core_error)?;
    loss.validate().map_err(py_core_error)?;
    Ok(loss)
}

fn parse_metric_reference(json: &str) -> PyResult<MetricReference> {
    let metric: MetricReference =
        deserialize_external_contract(json, "metric reference", CoreDagMlError::CampaignValidation)
            .map_err(py_core_error)?;
    metric.validate().map_err(py_core_error)?;
    Ok(metric)
}

fn parse_training_loss_role(json: &str) -> PyResult<TrainingLossRoleReference> {
    let role: TrainingLossRoleReference = deserialize_external_contract(
        json,
        "training loss role",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(py_core_error)?;
    role.validate().map_err(py_core_error)?;
    Ok(role)
}

fn parse_phase(phase: &str) -> PyResult<Phase> {
    serde_json::from_value(serde_json::Value::String(phase.to_string())).map_err(|_| {
        py_core_error(CoreDagMlError::CampaignValidation(format!(
            "unsupported training loss phase `{phase}`"
        )))
    })
}

fn ensure_callable(py: Python<'_>, implementation: &Py<PyAny>) -> PyResult<()> {
    if implementation.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyTypeError::new_err(
            "local loss and metric implementations must be callable",
        ))
    }
}

fn validate_python_descriptor(descriptor: &ImplementationDescriptor) -> PyResult<()> {
    if descriptor.binding_id != PYTHON_BINDING_ID {
        return Err(py_core_error(CoreDagMlError::CampaignValidation(format!(
            "Python local implementation requires binding_id `{PYTHON_BINDING_ID}`, got `{}`",
            descriptor.binding_id
        ))));
    }
    if descriptor.portability == PortabilityClass::PortableBuiltIn {
        return Err(py_core_error(CoreDagMlError::CampaignValidation(
            "Python local implementation registry rejects portable_builtin descriptors".to_string(),
        )));
    }
    if !descriptor
        .capabilities
        .contains(&ImplementationCapability::NeedsGil)
    {
        return Err(py_core_error(CoreDagMlError::CampaignValidation(
            "Python local implementation descriptor must declare needs_gil".to_string(),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    #[pyclass]
    struct AbsoluteError;

    #[pymethods]
    impl AbsoluteError {
        fn __call__(&self, target: f64, prediction: f64) -> f64 {
            (prediction - target).abs()
        }
    }

    fn loss_contracts() -> (String, String) {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../examples/fixtures/criteria/criteria_contracts.v1.json"
        ))
        .unwrap();
        let role = fixture["valid"]["training_loss_role"].clone();
        (
            role["loss"].to_string(),
            serde_json::to_string(&role).unwrap(),
        )
    }

    fn python_metric_json() -> String {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../examples/fixtures/criteria/criteria_contracts.v1.json"
        ))
        .unwrap();
        let mut metric: MetricReference =
            serde_json::from_value(fixture["valid"]["metric_role"]["metric"].clone()).unwrap();
        metric.implementation.binding_id = PYTHON_BINDING_ID.to_string();
        metric
            .implementation
            .capabilities
            .insert(ImplementationCapability::NeedsGil);
        metric.implementation.descriptor_fingerprint =
            metric.implementation.compute_fingerprint().unwrap();
        metric.validate().unwrap();
        serde_json::to_string(&metric).unwrap()
    }

    #[test]
    fn python_callable_is_retained_resolved_and_executed() {
        Python::initialize();
        Python::attach(|py| {
            let (loss_json, role_json) = loss_contracts();
            let implementation = Py::new(py, AbsoluteError).unwrap().into_any();
            let mut registry = PyLocalImplementationRegistry::new();
            registry
                .register_loss(py, &loss_json, implementation.clone_ref(py))
                .unwrap();

            let resolved = registry
                .resolve_training_loss(py, &role_json, "FIT_CV")
                .unwrap();
            let value = resolved
                .bind(py)
                .call1((2.0, 5.5))
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert_eq!(value, 3.5);
            assert_eq!(registry.__len__(), 1);

            let descriptor_json = registry.descriptors_json().unwrap();
            let descriptors: Vec<ImplementationDescriptor> =
                serde_json::from_str(&descriptor_json).unwrap();
            assert_eq!(descriptors.len(), 1);
        });
    }

    #[test]
    fn python_registry_rejects_non_callables_and_serialization() {
        Python::initialize();
        Python::attach(|py| {
            let (loss_json, _) = loss_contracts();
            let not_callable = 42_i64.into_pyobject(py).unwrap().unbind().into_any();
            let mut registry = PyLocalImplementationRegistry::new();
            assert!(registry
                .register_loss(py, &loss_json, not_callable)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            assert!(registry
                .__reduce__()
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn python_registry_rejects_another_bindings_descriptor() {
        Python::initialize();
        Python::attach(|py| {
            let fixture: Value = serde_json::from_str(include_str!(
                "../../../examples/fixtures/criteria/criteria_contracts.v1.json"
            ))
            .unwrap();
            let metric_json = fixture["valid"]["metric_role"]["metric"].to_string();
            let implementation = Py::new(py, AbsoluteError).unwrap().into_any();
            let mut registry = PyLocalImplementationRegistry::new();
            let error = registry
                .register_metric(py, &metric_json, implementation)
                .unwrap_err();
            assert!(error.to_string().contains("binding_id `binding:python`"));
        });
    }

    #[test]
    fn python_metric_callable_uses_the_separate_metric_resolution_path() {
        Python::initialize();
        Python::attach(|py| {
            let metric_json = python_metric_json();
            let implementation = Py::new(py, AbsoluteError).unwrap().into_any();
            let mut registry = PyLocalImplementationRegistry::new();
            registry
                .register_metric(py, &metric_json, implementation.clone_ref(py))
                .unwrap();

            let resolved = registry.resolve_metric(py, &metric_json).unwrap();
            let value = resolved
                .bind(py)
                .call1((2.0, 5.5))
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert_eq!(value, 3.5);
        });
    }

    #[test]
    fn python_attestation_helper_uses_the_resolved_role_and_phase() {
        Python::initialize();
        let (_, role_json) = loss_contracts();
        let json = loss_execution_attestation_json(&role_json, "REFIT").unwrap();
        let attestation: LossExecutionAttestation = serde_json::from_str(&json).unwrap();
        assert_eq!(attestation.phase, Phase::Refit);
        assert!(loss_execution_attestation_json(&role_json, "PREDICT").is_err());
    }
}
