//! Python-owned process-local loss and metric implementations.

use std::collections::BTreeSet;

use dag_ml_core::{
    deserialize_external_contract, DagMlError as CoreDagMlError, ImplementationCapability,
    ImplementationDescriptor, ImplementationSemanticKind, LocalImplementationRegistry,
    LossCapability, LossExecutionAttestation, LossReference, LossSpec, MetricCapability,
    MetricEvaluationResult, MetricEvaluationTask, MetricEvaluationValue, MetricReference,
    MetricSpec, NodeTask, Phase, PortabilityClass, ReplayabilityClass,
    TrainingLossRoleReference,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};

use crate::py_core_error;

const PYTHON_BINDING_ID: &str = "binding:python";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HostLocalRegistrationOptions {
    provider_id: String,
    implementation_version: String,
    implementation_fingerprint: String,
    registry_key: String,
    #[serde(default)]
    supported_controller_families: BTreeSet<String>,
    #[serde(default)]
    runtime_requirements: BTreeSet<String>,
    #[serde(default)]
    capabilities: BTreeSet<ImplementationCapability>,
}

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

    fn register_host_local_loss(
        &mut self,
        py: Python<'_>,
        loss_spec_json: &str,
        options_json: &str,
        implementation: Py<PyAny>,
    ) -> PyResult<String> {
        ensure_callable(py, &implementation)?;
        let mut spec: LossSpec = deserialize_external_contract(
            loss_spec_json,
            "Python host-local loss spec",
            CoreDagMlError::CampaignValidation,
        )
        .map_err(py_core_error)?;
        if spec.spec_fingerprint.is_empty() {
            spec.spec_fingerprint = spec.compute_fingerprint().map_err(py_core_error)?;
        }
        spec.validate().map_err(py_core_error)?;
        let mut options = parse_host_local_options(options_json)?;
        add_loss_required_capabilities(&spec, &mut options.capabilities);
        let reference = LossReference {
            implementation: host_local_descriptor(
                ImplementationSemanticKind::Loss,
                &spec.loss_id,
                &spec.spec_fingerprint,
                options,
            )?,
            spec,
        };
        reference.validate().map_err(py_core_error)?;
        self.registry
            .register_loss(&reference, implementation)
            .map_err(py_core_error)?;
        serde_json::to_string(&reference).map_err(crate::py_serde_error)
    }

    fn register_host_local_metric(
        &mut self,
        py: Python<'_>,
        metric_spec_json: &str,
        options_json: &str,
        implementation: Py<PyAny>,
    ) -> PyResult<String> {
        ensure_callable(py, &implementation)?;
        let mut spec: MetricSpec = deserialize_external_contract(
            metric_spec_json,
            "Python host-local metric spec",
            CoreDagMlError::CampaignValidation,
        )
        .map_err(py_core_error)?;
        if spec.spec_fingerprint.is_empty() {
            spec.spec_fingerprint = spec.compute_fingerprint().map_err(py_core_error)?;
        }
        spec.validate().map_err(py_core_error)?;
        let mut options = parse_host_local_options(options_json)?;
        add_metric_required_capabilities(&spec, &mut options.capabilities);
        let reference = MetricReference {
            implementation: host_local_descriptor(
                ImplementationSemanticKind::Metric,
                &spec.metric_id,
                &spec.spec_fingerprint,
                options,
            )?,
            spec,
        };
        reference.validate().map_err(py_core_error)?;
        self.registry
            .register_metric(&reference, implementation)
            .map_err(py_core_error)?;
        serde_json::to_string(&reference).map_err(crate::py_serde_error)
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

    fn resolve_task_training_loss(
        &self,
        py: Python<'_>,
        node_task_json: &str,
        role_index: usize,
    ) -> PyResult<(Py<PyAny>, String)> {
        let task = parse_node_task(node_task_json)?;
        if !matches!(task.phase, Phase::FitCv | Phase::Refit) {
            return Err(py_core_error(CoreDagMlError::RuntimeValidation(
                "training loss phase must be FIT_CV or REFIT".to_string(),
            )));
        }
        task.validate_required_loss_attestations()
            .map_err(py_core_error)?;
        let roles = task
            .node_plan
            .training_losses_for_phase(task.phase)
            .collect::<Vec<_>>();
        let role = roles.get(role_index).ok_or_else(|| {
            py_core_error(CoreDagMlError::RuntimeValidation(format!(
                "role_index {role_index} is outside the active training loss range"
            )))
        })?;
        validate_python_descriptor(&role.loss.implementation)?;
        let implementation = self
            .registry
            .resolve_loss(&role.loss)
            .map(|implementation| implementation.clone_ref(py))
            .map_err(py_core_error)?;
        let attestation = task
            .required_loss_attestations
            .get(role_index)
            .expect("validated loss requirements match active roles");
        let attestation_json = serde_json::to_string(attestation).map_err(crate::py_serde_error)?;
        Ok((implementation, attestation_json))
    }

    fn evaluate_metric(&self, py: Python<'_>, metric_task_json: &str) -> PyResult<String> {
        let task = MetricEvaluationTask::from_json(metric_task_json).map_err(py_core_error)?;
        validate_python_descriptor(&task.metric.implementation)?;
        let implementation = self
            .registry
            .resolve_metric(&task.metric)
            .map_err(py_core_error)?;
        let task_object = pythonize(py, &task).map_err(|error| {
            py_core_error(CoreDagMlError::RuntimeValidation(format!(
                "Python metric task conversion failed: {error}"
            )))
        })?;
        let callback_result = implementation
            .bind(py)
            .call1((task_object,))
            .map_err(|error| {
                py_core_error(CoreDagMlError::RuntimeValidation(format!(
                    "Python metric callback raised an exception: {error}"
                )))
            })?;
        let values: Vec<MetricEvaluationValue> = depythonize(&callback_result).map_err(|error| {
            py_core_error(CoreDagMlError::RuntimeValidation(format!(
                "Python metric callback result conversion failed: {error}"
            )))
        })?;
        let result = MetricEvaluationResult::for_task(&task, values).map_err(py_core_error)?;
        let aggregate = result.aggregate_for_task(&task).map_err(py_core_error)?;
        serde_json::to_string(&serde_json::json!({
            "result": result,
            "aggregate": aggregate,
        }))
        .map_err(crate::py_serde_error)
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

fn parse_node_task(json: &str) -> PyResult<NodeTask> {
    deserialize_external_contract(json, "node task", CoreDagMlError::RuntimeValidation)
        .map_err(py_core_error)
}

fn parse_phase(phase: &str) -> PyResult<Phase> {
    serde_json::from_value(serde_json::Value::String(phase.to_string())).map_err(|_| {
        py_core_error(CoreDagMlError::CampaignValidation(format!(
            "unsupported training loss phase `{phase}`"
        )))
    })
}

fn parse_host_local_options(json: &str) -> PyResult<HostLocalRegistrationOptions> {
    deserialize_external_contract(
        json,
        "Python host-local registration options",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(py_core_error)
}

fn host_local_descriptor(
    semantic_kind: ImplementationSemanticKind,
    semantic_id: &str,
    semantic_fingerprint: &str,
    mut options: HostLocalRegistrationOptions,
) -> PyResult<ImplementationDescriptor> {
    options
        .capabilities
        .insert(ImplementationCapability::NeedsGil);
    ImplementationDescriptor::new(
        semantic_kind,
        semantic_id,
        semantic_fingerprint,
        options.provider_id,
        PYTHON_BINDING_ID,
        options.implementation_version,
        options.implementation_fingerprint,
        options.supported_controller_families,
        options.runtime_requirements,
        options.capabilities,
        PortabilityClass::HostLocal,
        ReplayabilityClass::RegistryRequired,
        Some(options.registry_key),
    )
    .map_err(py_core_error)
}

fn add_loss_required_capabilities(
    spec: &LossSpec,
    capabilities: &mut BTreeSet<ImplementationCapability>,
) {
    for (required, capability) in [
        (
            spec.capabilities.contains(&LossCapability::Differentiable),
            ImplementationCapability::Differentiable,
        ),
        (
            spec.capabilities
                .contains(&LossCapability::DistributedReduction),
            ImplementationCapability::DistributedReduction,
        ),
        (
            spec.capabilities
                .contains(&LossCapability::SupportsMissingMask),
            ImplementationCapability::SupportsMissingMask,
        ),
        (
            spec.capabilities
                .contains(&LossCapability::SupportsSampleWeights),
            ImplementationCapability::SupportsSampleWeights,
        ),
    ] {
        if required {
            capabilities.insert(capability);
        }
    }
}

fn add_metric_required_capabilities(
    spec: &MetricSpec,
    capabilities: &mut BTreeSet<ImplementationCapability>,
) {
    for (required, capability) in [
        (
            spec.capabilities
                .contains(&MetricCapability::DistributedReduction),
            ImplementationCapability::DistributedReduction,
        ),
        (
            spec.capabilities
                .contains(&MetricCapability::SupportsMissingMask),
            ImplementationCapability::SupportsMissingMask,
        ),
        (
            spec.capabilities
                .contains(&MetricCapability::SupportsSampleWeights),
            ImplementationCapability::SupportsSampleWeights,
        ),
    ] {
        if required {
            capabilities.insert(capability);
        }
    }
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
