//! Owning PyO3 surface for the native W1 training operation.
//!
//! The binding only translates strict JSON contracts and Python controller
//! callbacks. Compile/plan/FIT_CV/SELECT/REFIT, scoring, output binding,
//! lineage and artifact capture remain implemented once in `dag-ml-core`.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use dag_ml_core::{
    execute_attached_training_replay, execute_loaded_predictor_replay, execute_training,
    parse_typed_json, ArtifactId, ArtifactLoadMode, AttachedTrainingReplayInput, BundleId,
    DataBinding, EnvelopeAttestedRuntimeDataProvider, ExternalDataPlanEnvelope, FittedArtifactMode,
    HandleKind, HandleRef, InMemoryArtifactStore, InMemoryDataProvider, LoadedPredictor,
    LoadedPredictorReplayInput, PortablePredictorPackage, RunId, RuntimeControllerRegistry,
    SampleRelationSet, TrainingExecutionInput, TrainingInfluenceManifest, TrainingOutcome,
    TrainingReplayRequest, TrainingRequest,
};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::in_process::build_runtime_controllers;
use crate::{py_core_error, py_serde_error};

const PY_DATA_PROVIDER_CONTROLLER_ID: &str = "controller:python.data.provider";

/// Process-local resources which make the retained artifact handles and data
/// views meaningful. Field order intentionally drops the handle stores before
/// the Python-backed controller registry.
struct TrainingResources {
    artifact_store: InMemoryArtifactStore,
    data_provider: EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>,
    controllers: RuntimeControllerRegistry,
}

/// Owning result of one native DAG-ML training run.
///
/// Portable JSON remains available after [`TrainingResult::detach`]. The
/// process-local controller callbacks, data-provider records and artifact
/// handles are retained until that explicit detach (or normal object drop).
#[pyclass(module = "dag_ml._dag_ml")]
pub struct TrainingResult {
    outcome: TrainingOutcome,
    // InMemoryDataProvider uses RefCell internally and is Send but not Sync.
    // The mutex makes the owning pyclass safely shareable across Python
    // threads without weakening the provider's single-operation semantics.
    resources: Mutex<Option<TrainingResources>>,
}

#[pymethods]
impl TrainingResult {
    /// Whether process-local callbacks, handles and provider state are retained.
    #[getter]
    fn is_attached(&self) -> PyResult<bool> {
        Ok(self.lock_resources()?.is_some())
    }

    /// Number of process-local refit artifact handles, or `None` after detach.
    #[getter]
    fn process_local_artifact_count(&self) -> PyResult<Option<usize>> {
        Ok(self
            .lock_resources()?
            .as_ref()
            .map(|resources| resources.artifact_store.len()))
    }

    /// Number of materialized data handles retained for replay/audit.
    #[getter]
    fn process_local_data_handle_count(&self) -> PyResult<Option<usize>> {
        Ok(self
            .lock_resources()?
            .as_ref()
            .map(|resources| resources.data_provider.inner().handle_records().len()))
    }

    /// Number of materialized data-view handles retained for replay/audit.
    #[getter]
    fn process_local_data_view_count(&self) -> PyResult<Option<usize>> {
        Ok(self
            .lock_resources()?
            .as_ref()
            .map(|resources| resources.data_provider.inner().view_records().len()))
    }

    /// Release every process-local resource while preserving portable output.
    ///
    /// Returns `True` only for the transition from attached to detached. Calling
    /// it again is safe and returns `False`.
    fn detach(&self) -> PyResult<bool> {
        // Take under the lock, then drop Python-backed controllers only after
        // releasing it. A callback finalizer may re-enter Python and must never
        // observe a mutex held by this method.
        let resources = {
            let mut guard = self.lock_resources()?;
            guard.take()
        };
        let detached = resources.is_some();
        drop(resources);
        Ok(detached)
    }

    /// Complete self-fingerprinted [`TrainingOutcome`] JSON.
    fn outcome_json(&self) -> PyResult<String> {
        serialize_json(&self.outcome)
    }

    /// Validated execution bundle JSON from the outcome.
    fn execution_bundle_json(&self) -> PyResult<String> {
        serialize_json(&self.outcome.execution_bundle)
    }

    /// Native score-set JSON from the outcome.
    fn score_set_json(&self) -> PyResult<String> {
        serialize_json(&self.outcome.score_set)
    }

    /// Resolved portable output blocks JSON from the outcome.
    fn outputs_json(&self) -> PyResult<String> {
        serialize_json(&self.outcome.outputs)
    }

    /// Portable refit artifact records JSON (never process-local handles).
    fn artifacts_json(&self) -> PyResult<String> {
        serialize_json(&self.outcome.execution_bundle.refit_artifacts)
    }

    /// Retained portable OOF cache payloads, if requested by the contract.
    fn portable_prediction_caches_json(&self) -> PyResult<Option<String>> {
        self.outcome
            .portable_prediction_caches
            .as_ref()
            .map(serialize_json)
            .transpose()
    }

    /// Export a signed portable predictor package JSON contract from the outcome.
    #[pyo3(signature = (
        package_id,
        fitted_artifact_mode = "allow_host_sidecar",
        artifact_load_mode = "host_sidecar"
    ))]
    fn portable_predictor_package_json(
        &self,
        package_id: &str,
        fitted_artifact_mode: &str,
        artifact_load_mode: &str,
    ) -> PyResult<String> {
        let fitted_artifact_mode = parse_fitted_artifact_mode(fitted_artifact_mode)?;
        let artifact_load_mode = parse_artifact_load_mode(artifact_load_mode)?;
        let package = self
            .outcome
            .to_portable_predictor_package(package_id, fitted_artifact_mode, artifact_load_mode)
            .map_err(py_core_error)?;
        serialize_json(&package)
    }

    /// Execute an attached PREDICT/EXPLAIN replay against the live training result.
    #[pyo3(signature = (
        request_json,
        data_envelopes_json,
        outcome_id,
        run_id,
        warnings_json = "[]",
        diagnostics_json = "{}"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn replay_json(
        &self,
        _py: Python<'_>,
        request_json: &str,
        data_envelopes_json: &str,
        outcome_id: &str,
        run_id: &str,
        warnings_json: &str,
        diagnostics_json: &str,
    ) -> PyResult<String> {
        let request = TrainingReplayRequest::from_json(request_json).map_err(py_core_error)?;
        let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
            data_envelopes_json,
            "training replay data envelope map",
        )?;
        for envelope in envelopes.values() {
            envelope.validate().map_err(py_core_error)?;
        }
        let warnings = parse_strict_json::<Vec<String>>(warnings_json, "training replay warnings")?;
        let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
            diagnostics_json,
            "training replay diagnostics",
        )?;
        let run_id = RunId::new(run_id).map_err(py_core_error)?;
        let outcome_id = outcome_id.to_string();

        let mut inner_provider =
            InMemoryDataProvider::new(provider_controller_id().map_err(py_core_error)?);
        for envelope in envelopes.values().cloned() {
            inner_provider
                .register_envelope(envelope)
                .map_err(py_core_error)?;
        }

        let guard = self.lock_resources()?;
        let Some(resources) = guard.as_ref() else {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "training result is detached; attached replay requires live process-local resources"
                    .to_string(),
            )));
        };
        let outcome = execute_attached_training_replay(AttachedTrainingReplayInput {
            source: &self.outcome,
            request: &request,
            outcome_id,
            run_id,
            controllers: &resources.controllers,
            data_provider: &inner_provider,
            artifact_store: &resources.artifact_store,
            data_envelopes: &envelopes,
            warnings,
            diagnostics,
        })
        .map_err(py_core_error)?;
        serialize_json(&outcome)
    }

    /// Stable fingerprint of the complete outcome.
    #[getter]
    fn outcome_fingerprint(&self) -> &str {
        &self.outcome.outcome_fingerprint
    }
}

impl TrainingResult {
    fn lock_resources(&self) -> PyResult<MutexGuard<'_, Option<TrainingResources>>> {
        self.resources.lock().map_err(|_| {
            py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "training result resource lock is poisoned".to_string(),
            ))
        })
    }
}

/// Execute native COMPILE/PLAN -> FIT_CV -> SELECT -> optional REFIT.
///
/// `data_envelopes_json` is an object keyed by the exact V1
/// `node_id.input_name` requirement key. The binding constructs an
/// `EnvelopeAttestedRuntimeDataProvider`, so missing, extra, colliding or
/// field-mismatched bindings fail before any controller callback is invoked.
///
/// The GIL is explicitly detached around the core operation. Controllers call
/// back into Python through the existing in-process bridge and reattach only
/// for each callback; this also prevents a parallel scheduler from deadlocking
/// while worker threads wait to enter Python.
#[pyfunction]
#[pyo3(signature = (
    request_json,
    data_envelopes_json,
    relations_json,
    training_influence_json,
    op_callback,
    outcome_id,
    run_id,
    bundle_id,
    warnings_json = "[]",
    diagnostics_json = "{}"
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_training_json(
    py: Python<'_>,
    request_json: &str,
    data_envelopes_json: &str,
    relations_json: &str,
    training_influence_json: &str,
    op_callback: Py<PyAny>,
    outcome_id: &str,
    run_id: &str,
    bundle_id: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<TrainingResult> {
    if !op_callback.bind(py).is_callable() {
        return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "training op_callback must be callable".to_string(),
        )));
    }
    // TrainingRequest::from_json performs raw-token TCV1 verification before
    // serde can erase Integer/Binary64 distinctions.
    let request = TrainingRequest::from_json(request_json).map_err(py_core_error)?;
    let projection = request.project().map_err(py_core_error)?;
    let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        data_envelopes_json,
        "training data envelope map",
    )?;
    for envelope in envelopes.values() {
        envelope.validate().map_err(py_core_error)?;
    }
    let relations =
        parse_strict_json::<SampleRelationSet>(relations_json, "training sample relations")?;
    relations.validate().map_err(py_core_error)?;
    let training_influence = parse_strict_json::<TrainingInfluenceManifest>(
        training_influence_json,
        "training influence manifest",
    )?;
    training_influence.validate().map_err(py_core_error)?;
    let warnings = parse_strict_json::<Vec<String>>(warnings_json, "training warnings")?;
    let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
        diagnostics_json,
        "training diagnostics",
    )?;

    let bindings = projection
        .plan
        .node_plans
        .values()
        .flat_map(|node_plan| node_plan.data_bindings.iter().cloned())
        .collect::<Vec<DataBinding>>();
    let provider_controller = provider_controller_id().map_err(py_core_error)?;
    let mut inner_provider = InMemoryDataProvider::new(provider_controller);
    for envelope in envelopes.values().cloned() {
        inner_provider
            .register_envelope(envelope)
            .map_err(py_core_error)?;
    }
    let data_provider =
        EnvelopeAttestedRuntimeDataProvider::new(inner_provider, bindings, envelopes)
            .map_err(py_core_error)?;
    let controllers =
        build_runtime_controllers(py, &projection.plan, &op_callback).map_err(py_core_error)?;
    let run_id = RunId::new(run_id).map_err(py_core_error)?;
    let bundle_id = BundleId::new(bundle_id).map_err(py_core_error)?;
    let outcome_id = outcome_id.to_string();

    let resources = TrainingResources {
        artifact_store: InMemoryArtifactStore::new(),
        data_provider,
        controllers,
    };

    let (outcome, resources) = py
        .detach(move || {
            let mut resources = resources;
            let outcome = execute_training(TrainingExecutionInput {
                request: &request,
                outcome_id,
                run_id,
                bundle_id,
                controllers: &resources.controllers,
                data_provider: &resources.data_provider,
                relations: &relations,
                training_influence: &training_influence,
                artifact_store: &mut resources.artifact_store,
                warnings,
                diagnostics,
            })?;
            Ok::<_, dag_ml_core::DagMlError>((outcome, resources))
        })
        .map_err(py_core_error)?;

    Ok(TrainingResult {
        outcome,
        resources: Mutex::new(Some(resources)),
    })
}

/// Execute stateless PREDICT/EXPLAIN replay from a loaded portable predictor package.
///
/// `artifact_handles_json` is a host-side sidecar map keyed by artifact id. The
/// portable package remains handle-free; this binding only joins the package to
/// the explicit handles supplied by the host for this process.
#[pyfunction]
#[pyo3(signature = (
    package_json,
    request_json,
    data_envelopes_json,
    artifact_handles_json,
    op_callback,
    outcome_id,
    run_id,
    warnings_json = "[]",
    diagnostics_json = "{}"
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_loaded_predictor_replay_json(
    py: Python<'_>,
    package_json: &str,
    request_json: &str,
    data_envelopes_json: &str,
    artifact_handles_json: &str,
    op_callback: Py<PyAny>,
    outcome_id: &str,
    run_id: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<String> {
    if !op_callback.bind(py).is_callable() {
        return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "loaded predictor replay op_callback must be callable".to_string(),
        )));
    }

    let package = PortablePredictorPackage::from_json(package_json).map_err(py_core_error)?;
    let request = TrainingReplayRequest::from_json(request_json).map_err(py_core_error)?;
    let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        data_envelopes_json,
        "loaded predictor replay data envelope map",
    )?;
    for envelope in envelopes.values() {
        envelope.validate().map_err(py_core_error)?;
    }
    let artifact_handles = parse_strict_json::<BTreeMap<ArtifactId, HandleRef>>(
        artifact_handles_json,
        "loaded predictor artifact handle map",
    )?;
    validate_loaded_predictor_handles(&package, &artifact_handles)?;
    let warnings =
        parse_strict_json::<Vec<String>>(warnings_json, "loaded predictor replay warnings")?;
    let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
        diagnostics_json,
        "loaded predictor replay diagnostics",
    )?;
    let run_id = RunId::new(run_id).map_err(py_core_error)?;
    let outcome_id = outcome_id.to_string();

    let predictor = LoadedPredictor::new(package, artifact_handles).map_err(py_core_error)?;
    let mut inner_provider =
        InMemoryDataProvider::new(provider_controller_id().map_err(py_core_error)?);
    for envelope in envelopes.values().cloned() {
        inner_provider
            .register_envelope(envelope)
            .map_err(py_core_error)?;
    }
    let controllers =
        build_runtime_controllers(py, &predictor.package().effective_plan, &op_callback)
            .map_err(py_core_error)?;

    let outcome = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
        predictor: &predictor,
        request: &request,
        outcome_id,
        run_id,
        controllers: &controllers,
        data_provider: &inner_provider,
        data_envelopes: &envelopes,
        warnings,
        diagnostics,
    })
    .map_err(py_core_error)?;
    serialize_json(&outcome)
}

fn parse_strict_json<T>(json: &str, label: &str) -> PyResult<T>
where
    T: DeserializeOwned + Serialize,
{
    parse_typed_json(json).map_err(|error| {
        py_core_error(dag_ml_core::DagMlError::CampaignValidation(format!(
            "{label} is not strict TCV1 JSON: {error}"
        )))
    })?;
    dag_ml_core::deserialize_external_contract(
        json,
        label,
        dag_ml_core::DagMlError::CampaignValidation,
    )
    .map_err(py_core_error)
}

fn serialize_json<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(py_serde_error)
}

fn parse_fitted_artifact_mode(value: &str) -> PyResult<FittedArtifactMode> {
    match value {
        "allow_host_sidecar" => Ok(FittedArtifactMode::AllowHostSidecar),
        "portable_required" => Ok(FittedArtifactMode::PortableRequired),
        other => Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            format!("unsupported fitted_artifact_mode `{other}`"),
        ))),
    }
}

fn parse_artifact_load_mode(value: &str) -> PyResult<ArtifactLoadMode> {
    match value {
        "host_sidecar" => Ok(ArtifactLoadMode::HostSidecar),
        "native_portable" => Ok(ArtifactLoadMode::NativePortable),
        other => Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            format!("unsupported artifact_load_mode `{other}`"),
        ))),
    }
}

fn validate_loaded_predictor_handles(
    package: &PortablePredictorPackage,
    artifact_handles: &BTreeMap<ArtifactId, HandleRef>,
) -> PyResult<()> {
    let records = package
        .execution_bundle
        .refit_artifacts
        .iter()
        .map(|record| (record.artifact.id.clone(), record))
        .collect::<BTreeMap<_, _>>();
    for (artifact_id, handle) in artifact_handles {
        let Some(record) = records.get(artifact_id) else {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                format!(
                    "loaded predictor sidecar handle references unknown artifact `{artifact_id}`"
                ),
            )));
        };
        if !matches!(handle.kind, HandleKind::Model | HandleKind::Artifact) {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                format!(
                    "loaded predictor sidecar handle for `{artifact_id}` must be model or artifact"
                ),
            )));
        }
        if handle.owner_controller != record.controller_id {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                format!(
                    "loaded predictor sidecar handle for `{artifact_id}` is owned by `{}` instead of `{}`",
                    handle.owner_controller, record.controller_id
                ),
            )));
        }
    }
    Ok(())
}

fn provider_controller_id() -> dag_ml_core::Result<dag_ml_core::ControllerId> {
    dag_ml_core::ControllerId::new(PY_DATA_PROVIDER_CONTROLLER_ID)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicU64, Ordering};

    use dag_ml_core::{
        ArtifactId, ArtifactRef, ControllerCapability, ControllerFitScope, CvArtifactRetention,
        DataBinding, EntityUnitLevel, EvaluationScope, ExternalDataPlanEnvelope,
        FittedArtifactMode, FoldId, GenerationSpec, GroupId, HandleKind, HandleRef, LineageId,
        LineageRecord, NodeKind, NodeResult, NodeTask, ObservationId, Phase, PredictionBlock,
        PredictionLevel, PredictionPartition, PredictionUnitId, RegressionTargetBlock, SampleId,
        SampleRelation, SampleRelationSet, TrainingContractProjection, TrainingDataIdentity,
        TrainingInfluenceEntry, TrainingInfluenceKind, TrainingInfluenceManifest, TrainingRequest,
        TrainingSchedulerBackend, TRAINING_INFLUENCE_MANIFEST_SCHEMA_VERSION,
    };
    use pyo3::exceptions::PyValueError;

    use super::*;

    const REQUEST_FIXTURE: &str =
        include_str!("../../../examples/fixtures/training/training_request_refit.v1.json");

    #[derive(Default)]
    #[pyclass]
    struct TestOperatorCallback {
        call_count: AtomicU64,
        next_handle: AtomicU64,
        explicit_model_ports: bool,
    }

    impl TestOperatorCallback {
        fn handle(&self) -> u64 {
            self.next_handle.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    #[pymethods]
    impl TestOperatorCallback {
        fn __call__(&self, py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let task: NodeTask = pythonize::depythonize(payload)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let is_model = task.node_plan.kind == NodeKind::Model;
            let sample_ids = match task.fold_id.as_ref().map(FoldId::as_str) {
                Some("fold:0") => vec![sample("sample:1"), sample("sample:2")],
                Some("fold:1") => vec![sample("sample:3"), sample("sample:4")],
                _ => (1..=4)
                    .map(|index| sample(&format!("sample:{index}")))
                    .collect(),
            };
            let partition = if task.phase == Phase::Refit {
                PredictionPartition::Final
            } else {
                PredictionPartition::Validation
            };
            let explicit_model_ports = is_model && self.explicit_model_ports;
            let mut predictions = if is_model && matches!(task.phase, Phase::FitCv | Phase::Refit) {
                vec![PredictionBlock {
                    prediction_id: Some(format!(
                        "prediction:{}:{}:{}",
                        task.node_plan.node_id,
                        task.phase.as_str(),
                        task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                    )),
                    producer_node: task.node_plan.node_id.clone(),
                    producer_port: explicit_model_ports.then(|| "oof".to_string()),
                    partition,
                    fold_id: (task.phase == Phase::FitCv)
                        .then(|| task.fold_id.clone())
                        .flatten(),
                    sample_ids: sample_ids.clone(),
                    values: sample_ids.iter().map(|_| vec![0.0]).collect(),
                    target_names: vec!["protein".to_string()],
                }]
            } else {
                Vec::new()
            };
            if explicit_model_ports {
                let mut sibling = predictions
                    .first()
                    .expect("explicit model port callback emits primary predictions")
                    .clone();
                sibling.prediction_id = sibling
                    .prediction_id
                    .as_ref()
                    .map(|id| format!("{id}:probability"));
                sibling.producer_port = Some("probability".to_string());
                sibling.values = sibling
                    .values
                    .iter()
                    .map(|row| row.iter().map(|value| value + 100.0).collect())
                    .collect();
                predictions.push(sibling);
            }
            let regression_targets = if is_model && task.phase == Phase::FitCv {
                vec![RegressionTargetBlock {
                    level: PredictionLevel::Sample,
                    unit_ids: sample_ids
                        .iter()
                        .cloned()
                        .map(PredictionUnitId::Sample)
                        .collect(),
                    values: sample_ids.iter().map(|_| vec![0.0]).collect(),
                    target_names: vec!["protein".to_string()],
                }]
            } else {
                Vec::new()
            };
            let artifacts = if is_model && task.phase == Phase::Refit {
                vec![ArtifactRef {
                    id: ArtifactId::new("artifact:model.base:refit").unwrap(),
                    kind: "test_model".to_string(),
                    controller_id: task.node_plan.controller_id.clone(),
                    backend: None,
                    uri: None,
                    content_fingerprint: None,
                    size_bytes: Some(8),
                    plugin: None,
                    plugin_version: None,
                }]
            } else {
                Vec::new()
            };
            let artifact_handles = artifacts
                .iter()
                .map(|artifact| {
                    (
                        artifact.id.clone(),
                        HandleRef {
                            handle: self.handle(),
                            kind: HandleKind::Artifact,
                            owner_controller: task.node_plan.controller_id.clone(),
                        },
                    )
                })
                .collect();
            let output_name = if is_model { "oof" } else { "x_out" };
            let output_kind = if is_model {
                HandleKind::Prediction
            } else {
                HandleKind::Data
            };
            let mut outputs = BTreeMap::from([(
                output_name.to_string(),
                HandleRef {
                    handle: self.handle(),
                    kind: output_kind,
                    owner_controller: task.node_plan.controller_id.clone(),
                },
            )]);
            if explicit_model_ports {
                outputs.insert(
                    "probability".to_string(),
                    HandleRef {
                        handle: self.handle(),
                        kind: HandleKind::Prediction,
                        owner_controller: task.node_plan.controller_id.clone(),
                    },
                );
            }
            let result = NodeResult {
                schema_version: None,
                node_id: task.node_plan.node_id.clone(),
                outputs,
                predictions,
                observation_predictions: Vec::new(),
                aggregated_predictions: Vec::new(),
                explanations: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: artifacts.clone(),
                artifact_handles,
                fit_influence_diagnostics: Vec::new(),
                regression_targets,
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:{}:{}:{}:{}",
                        task.node_plan.node_id,
                        task.phase.as_str(),
                        task.variant_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "base".to_string()),
                        task.fold_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "full".to_string())
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: task.node_plan.controller_id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: artifacts,
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                    loss_attestations: Vec::new(),
                    early_stopping_records: Vec::new(),
                },
            };
            pythonize::pythonize(py, &result)
                .map(|value| value.unbind())
                .map_err(|error| PyValueError::new_err(error.to_string()))
        }
    }

    #[test]
    fn owning_training_result_retains_resources_and_detaches_portably() {
        Python::initialize();
        let (request, envelopes, relations, influence) = executable_contracts();
        Python::attach(|py| {
            let callback = Py::new(py, TestOperatorCallback::default())
                .unwrap()
                .into_any();
            let result = execute_training_json(
                py,
                &request,
                &envelopes,
                &relations,
                &influence,
                callback,
                "outcome:python.native",
                "run:python.native",
                "bundle:python.native",
                "[]",
                r#"{"binding":"pyo3"}"#,
            )
            .expect("native PyO3 training succeeds");

            assert!(result.is_attached().unwrap());
            assert_eq!(result.process_local_artifact_count().unwrap(), Some(1));
            assert!(result.process_local_data_handle_count().unwrap().unwrap() > 0);
            assert!(result.process_local_data_view_count().unwrap().unwrap() > 0);
            assert_eq!(
                serde_json::from_str::<Vec<serde_json::Value>>(&result.artifacts_json().unwrap())
                    .unwrap()
                    .len(),
                1
            );
            let outcome_json = result.outcome_json().unwrap();
            TrainingOutcome::from_json(&outcome_json).expect("binding emits a valid outcome");
            assert!(result.detach().unwrap());
            assert!(!result.is_attached().unwrap());
            assert_eq!(result.process_local_artifact_count().unwrap(), None);
            assert!(!result.detach().unwrap());
            TrainingOutcome::from_json(&result.outcome_json().unwrap())
                .expect("portable outcome survives detach");
        });
    }

    #[test]
    fn pyo3_training_result_filters_explicit_multi_prediction_port_outputs() {
        Python::initialize();
        let (request, envelopes, relations, influence) = executable_contracts_with(|request| {
            let extra = request
                .graph
                .nodes
                .iter()
                .find(|node| node.id.as_str() == "model:base")
                .unwrap()
                .ports
                .outputs
                .iter()
                .find(|port| port.name == "oof")
                .unwrap()
                .clone();
            let mut probability = extra;
            probability.name = "probability".to_string();
            request
                .graph
                .nodes
                .iter_mut()
                .find(|node| node.id.as_str() == "model:base")
                .unwrap()
                .ports
                .outputs
                .push(probability.clone());
            request
                .controller_manifests
                .iter_mut()
                .find(|manifest| manifest.controller_id.as_str() == "controller:model.mock")
                .unwrap()
                .output_ports
                .push(probability);
            request.options.outputs[0].port_name = Some("oof".to_string());
        });
        Python::attach(|py| {
            let callback = Py::new(
                py,
                TestOperatorCallback {
                    explicit_model_ports: true,
                    ..TestOperatorCallback::default()
                },
            )
            .unwrap()
            .into_any();
            let result = execute_training_json(
                py,
                &request,
                &envelopes,
                &relations,
                &influence,
                callback,
                "outcome:python.native.multiport",
                "run:python.native.multiport",
                "bundle:python.native.multiport",
                "[]",
                r#"{"binding":"pyo3_multiport"}"#,
            )
            .expect("native PyO3 multi-port training succeeds");
            let outputs: Vec<serde_json::Value> =
                serde_json::from_str(&result.outputs_json().unwrap()).unwrap();
            assert_eq!(outputs.len(), 1);
            let output = &outputs[0];
            assert_eq!(output["binding"]["node_id"], "model:base");
            assert_eq!(output["binding"]["port_name"], "oof");
            let predictions = output["predictions"].as_array().unwrap();
            assert!(!predictions.is_empty());
            assert!(predictions.iter().all(|block| {
                block["producer_node"] == "model:base"
                    && block["producer_port"] == "oof"
                    && block["partition"] == "final"
                    && block["fold_id"].is_null()
            }));
            let outcome = TrainingOutcome::from_json(&result.outcome_json().unwrap())
                .expect("binding emits a valid multi-port outcome");
            assert!(outcome.score_set.reports.iter().any(|report| {
                report.producer_node.as_str() == "model:base"
                    && report.producer_port.as_deref() == Some("probability")
            }));
        });
    }

    #[test]
    fn strict_envelope_map_rejects_duplicate_keys_before_callback() {
        Python::initialize();
        Python::attach(|py| {
            let callback = Py::new(py, TestOperatorCallback::default())
                .unwrap()
                .into_any();
            let error = match execute_training_json(
                py,
                REQUEST_FIXTURE,
                r#"{"model:base.x":{},"model:base.x":{}}"#,
                r#"{"records":[]}"#,
                r#"{}"#,
                callback,
                "outcome:duplicate",
                "run:duplicate",
                "bundle:duplicate",
                "[]",
                "{}",
            ) {
                Ok(_) => panic!("duplicate requirement keys are rejected"),
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("duplicate JSON object key"),
                "{error}"
            );
        });
    }

    #[test]
    fn strict_relation_and_envelope_contracts_reject_unknown_or_positional_fields_before_callback()
    {
        Python::initialize();
        let (request, envelopes, relations, influence) = executable_contracts();

        let mut unknown_relation_set: serde_json::Value = serde_json::from_str(&relations).unwrap();
        unknown_relation_set.as_object_mut().unwrap().insert(
            "unexpected_contract_field".to_string(),
            serde_json::json!(true),
        );

        let mut unknown_relation_record: serde_json::Value =
            serde_json::from_str(&relations).unwrap();
        unknown_relation_record["records"][0]
            .as_object_mut()
            .unwrap()
            .insert(
                "unexpected_contract_field".to_string(),
                serde_json::json!(true),
            );

        let mut unknown_envelope_relation_set: serde_json::Value =
            serde_json::from_str(&envelopes).unwrap();
        unknown_envelope_relation_set
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()["coordinator_relations"]
            .as_object_mut()
            .unwrap()
            .insert(
                "unexpected_contract_field".to_string(),
                serde_json::json!(true),
            );

        let mut unknown_envelope_relation_record: serde_json::Value =
            serde_json::from_str(&envelopes).unwrap();
        unknown_envelope_relation_record
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()["coordinator_relations"]["records"][0]
            .as_object_mut()
            .unwrap()
            .insert(
                "unexpected_contract_field".to_string(),
                serde_json::json!(true),
            );

        let mut positional_envelope_relation_set: serde_json::Value =
            serde_json::from_str(&envelopes).unwrap();
        positional_envelope_relation_set
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()["coordinator_relations"] = serde_json::json!([[]]);

        let cases = [
            (
                "relation-set root unknown field",
                envelopes.clone(),
                serde_json::to_string(&unknown_relation_set).unwrap(),
                "unexpected_contract_field",
            ),
            (
                "relation record unknown field",
                envelopes.clone(),
                serde_json::to_string(&unknown_relation_record).unwrap(),
                "unexpected_contract_field",
            ),
            (
                "envelope relation-set unknown field",
                serde_json::to_string(&unknown_envelope_relation_set).unwrap(),
                relations.clone(),
                "unexpected_contract_field",
            ),
            (
                "envelope relation record unknown field",
                serde_json::to_string(&unknown_envelope_relation_record).unwrap(),
                relations.clone(),
                "unexpected_contract_field",
            ),
            (
                "positional relation-set contract",
                envelopes.clone(),
                "[[]]".to_string(),
                "must use a JSON object at the external contract boundary",
            ),
            (
                "positional envelope relation-set contract",
                serde_json::to_string(&positional_envelope_relation_set).unwrap(),
                relations.clone(),
                "must use a JSON object at the external contract boundary",
            ),
        ];

        Python::attach(|py| {
            for (label, envelope_json, relations_json, expected_error) in cases {
                let callback = Py::new(py, TestOperatorCallback::default()).unwrap();
                let error = match execute_training_json(
                    py,
                    &request,
                    &envelope_json,
                    &relations_json,
                    &influence,
                    callback.clone_ref(py).into_any(),
                    "outcome:strict.contract",
                    "run:strict.contract",
                    "bundle:strict.contract",
                    "[]",
                    "{}",
                ) {
                    Ok(_) => panic!("{label} must be rejected"),
                    Err(error) => error,
                };
                assert!(
                    error.to_string().contains(expected_error),
                    "{label} returned an unexpected error: {error}"
                );
                assert_eq!(
                    callback.bind(py).borrow().call_count.load(Ordering::SeqCst),
                    0,
                    "{label} reached the operator callback: {error}"
                );
            }
        });
    }

    fn executable_contracts() -> (String, String, String, String) {
        executable_contracts_with(|_| {})
    }

    fn executable_contracts_with(
        mutate: impl FnOnce(&mut TrainingRequest),
    ) -> (String, String, String, String) {
        let mut request: TrainingRequest = serde_json::from_str(REQUEST_FIXTURE).unwrap();
        request.campaign.generation = GenerationSpec::default();
        request.options.selection.required_metric_level = Some(PredictionLevel::Sample);
        request.options.selection.evaluation_scope = Some(EvaluationScope::Oof);
        request.options.scheduler.kind = dag_ml_core::TrainingSchedulerKind::Parallel;
        request.options.scheduler.backend = Some(TrainingSchedulerBackend::Threads);
        request.options.scheduler.workers = 2;
        request.options.resources.cpu_threads = 2;
        request.options.resources.memory_bytes = None;
        request.options.resources.wall_time_ms = None;
        request.options.artifacts.cv_artifacts = CvArtifactRetention::Discard;
        request.options.artifacts.fitted_artifacts = FittedArtifactMode::AllowHostSidecar;
        mutate(&mut request);

        let relations = relations();
        let relation_fingerprint = relations.fingerprint().unwrap();
        let binding = request
            .campaign
            .data_bindings
            .values_mut()
            .flat_map(|bindings| bindings.iter_mut())
            .next()
            .unwrap();
        binding.relation_fingerprint = Some(relation_fingerprint.clone());
        let envelope = envelope(binding, &request.data_identities[0], relations.clone());
        request.data_identities =
            vec![TrainingDataIdentity::from_binding_envelope(binding, &envelope).unwrap()];
        request.request_fingerprint = "0".repeat(64);
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        let projection = request.project().unwrap();
        let influence = influence_manifest(&request, &projection, &relations);
        let requirement_key = request.data_identities[0].requirement_key.clone();

        (
            serde_json::to_string(&request).unwrap(),
            serde_json::to_string(&BTreeMap::from([(requirement_key, envelope)])).unwrap(),
            serde_json::to_string(&relations).unwrap(),
            serde_json::to_string(&influence).unwrap(),
        )
    }

    fn envelope(
        binding: &DataBinding,
        identity: &TrainingDataIdentity,
        relations: SampleRelationSet,
    ) -> ExternalDataPlanEnvelope {
        ExternalDataPlanEnvelope {
            schema_version: 1,
            schema_fingerprint: binding.schema_fingerprint.clone(),
            plan_fingerprint: binding.plan_fingerprint.clone(),
            relation_fingerprint: binding.relation_fingerprint.clone(),
            data_content_fingerprint: Some(identity.data_content_fingerprint.clone()),
            target_content_fingerprint: Some(identity.target_content_fingerprint.clone()),
            coordinator_relations: Some(relations),
        }
    }

    fn relations() -> SampleRelationSet {
        SampleRelationSet {
            records: (1..=4)
                .map(|index| {
                    let mut relation = SampleRelation::new(
                        ObservationId::new(format!("observation:{index}")).unwrap(),
                        sample(&format!("sample:{index}")),
                    );
                    relation.unit_level = EntityUnitLevel::Observation;
                    relation.group_id =
                        Some(GroupId::new(if index <= 2 { "group:0" } else { "group:1" }).unwrap());
                    relation
                })
                .collect(),
        }
    }

    fn influence_manifest(
        request: &TrainingRequest,
        projection: &TrainingContractProjection,
        relations: &SampleRelationSet,
    ) -> TrainingInfluenceManifest {
        let all = projection
            .plan
            .fold_set
            .as_ref()
            .unwrap()
            .sample_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut coordinates = Vec::new();
        for node_id in &projection.predictor_node_ids {
            let node_plan = &projection.plan.node_plans[node_id];
            if matches!(
                node_plan.fit_scope,
                ControllerFitScope::Stateless | ControllerFitScope::InferenceOnly
            ) {
                continue;
            }
            let kind = if node_plan
                .controller_capabilities
                .contains(&ControllerCapability::TrainsAggregation)
            {
                TrainingInfluenceKind::TrainedMetaAggregation
            } else if node_plan.kind == NodeKind::Model {
                TrainingInfluenceKind::ModelFit
            } else if node_plan.kind == NodeKind::Tuner {
                TrainingInfluenceKind::HpoSelection
            } else {
                TrainingInfluenceKind::TransformFit
            };
            if node_plan.supported_phases.contains(&Phase::FitCv) {
                for fold in &projection.plan.fold_set.as_ref().unwrap().folds {
                    coordinates.push((
                        kind,
                        format!("fit_cv:{}", fold.fold_id),
                        Some(node_id.clone()),
                        fold.train_sample_ids
                            .iter()
                            .cloned()
                            .collect::<BTreeSet<_>>(),
                    ));
                }
            }
            if request.options.refit && node_plan.supported_phases.contains(&Phase::Refit) {
                coordinates.push((
                    kind,
                    "refit:full".to_string(),
                    Some(node_id.clone()),
                    all.clone(),
                ));
            }
        }
        coordinates.push((
            TrainingInfluenceKind::HpoSelection,
            format!("select:{}", request.options.selection.id),
            None,
            all,
        ));
        let mut entries = coordinates
            .into_iter()
            .map(
                |(kind, scope_id, node_id, samples)| TrainingInfluenceEntry {
                    kind,
                    scope_id,
                    node_id,
                    physical_sample_ids: samples.iter().cloned().collect(),
                    origin_sample_ids: Vec::new(),
                    group_ids: relations
                        .records
                        .iter()
                        .filter(|relation| samples.contains(&relation.sample_id))
                        .filter_map(|relation| relation.group_id.clone())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                },
            )
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            (&left.kind, &left.scope_id, &left.node_id).cmp(&(
                &right.kind,
                &right.scope_id,
                &right.node_id,
            ))
        });
        let mut manifest = TrainingInfluenceManifest {
            schema_version: TRAINING_INFLUENCE_MANIFEST_SCHEMA_VERSION,
            relation_fingerprint: relations.fingerprint().unwrap(),
            entries,
            manifest_fingerprint: "0".repeat(64),
        };
        manifest.manifest_fingerprint = manifest.compute_fingerprint().unwrap();
        manifest
    }

    fn sample(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }
}
