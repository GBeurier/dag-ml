//! Owning PyO3 surface for the native W1 training operation.
//!
//! The binding only translates strict JSON contracts and Python controller
//! callbacks. Compile/plan/FIT_CV/SELECT/REFIT, scoring, output binding,
//! lineage and artifact capture remain implemented once in `dag-ml-core`.

use std::collections::BTreeMap;
#[cfg(feature = "methods-optimizer")]
use std::collections::BTreeSet;
use std::sync::{Mutex, MutexGuard};

#[cfg(feature = "methods-optimizer")]
use dag_ml_core::{
    build_portable_refit_package_v3, derive_portable_full_refit_target_plan,
    execute_loaded_methods_portable_refit_replay_v3, execute_portable_full_refit,
    ArtifactMaterializationRequest, ExecutionBundle, MethodsPlsDataset, MethodsPlsMatrix,
    MethodsPortablePredictorReplayInput, MethodsPortableRefitReplayInputV3,
    PortableFullRefitExecutionInput, PortableRefitPackageV3, PortableRefitPackageV3BuildInput,
    PortableRefitRecipe, RuntimeArtifactStore, SampleId,
};
use dag_ml_core::{
    calibrate_attached_training_replay_with_derived_context, execute_attached_training_replay,
    execute_loaded_predictor_replay, execute_training, parse_typed_json, ArtifactId,
    ArtifactLoadMode, AttachedTrainingReplayInput, BundleId, ConformalCalibrationTruth,
    ConformalMultiTargetPolicy, ConformalSmallSamplePolicy, DataBinding,
    DataMaterializationRequest, DataViewRequest, EnvelopeAttestedRuntimeDataProvider,
    ExternalDataPlanEnvelope, FittedArtifactMode, HandleKind, HandleRef, InMemoryArtifactStore,
    InMemoryDataProvider, LoadedPredictor, LoadedPredictorReplayInput, MethodsPlsData,
    MethodsPlsDataRequest, PortablePredictorPackage, RunId, RuntimeControllerRegistry,
    RuntimeDataProvider, SampleRelationSet, TrainingExecutionInput, TrainingInfluenceManifest,
    TrainingOutcome, TrainingReplayOutcome, TrainingReplayRequest, TrainingRequest,
};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
#[cfg(feature = "methods-optimizer")]
use serde::Deserialize;
use serde::Serialize;
#[cfg(feature = "methods-optimizer")]
use sha2::{Digest, Sha256};

use crate::in_process::{
    build_runtime_controllers, build_runtime_controllers_with_artifact_callback,
};
use crate::{py_core_error, py_serde_error};

const PY_DATA_PROVIDER_CONTROLLER_ID: &str = "controller:python.data.provider";

/// Strict JSON representation of one raw, host-materialized Methods input.
///
/// Python remains the data owner: it supplies a finite row-major feature and
/// target matrix only after constructing the signed request/envelope.  Core
/// remains the sole owner of scheduler-selected row views; this payload is
/// reindexed only by those views below and never by Python positions.
#[cfg(feature = "methods-optimizer")]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MethodsTrainingInputJson {
    sample_ids: Vec<String>,
    x: Vec<Vec<f64>>,
    #[serde(default)]
    y: Option<Vec<Vec<f64>>>,
    target_names: Vec<String>,
}

/// Strict X-only payload for the closed Methods terminal-prediction facade.
///
/// Deliberately unlike [`MethodsTrainingInputJson`], this has no optional
/// target field. `deny_unknown_fields` therefore makes an accidental `y`
/// payload a boundary refusal instead of silently retaining labels in an
/// inference operation.
#[cfg(feature = "methods-optimizer")]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MethodsPredictInputJson {
    sample_ids: Vec<String>,
    x: Vec<Vec<f64>>,
    target_names: Vec<String>,
}

/// Native Methods training input provider for the Python binding.
///
/// This is deliberately separate from the generic Python callback provider.
/// A native controller can consume only explicit numeric PLS views, while all
/// ordinary nodes keep using opaque host handles and callbacks.
#[cfg(feature = "methods-optimizer")]
struct PyMethodsPlsTrainingProvider {
    inner: EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>,
    inputs: BTreeMap<String, MethodsPlsDataset>,
}

enum TrainingDataProvider {
    Host(EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>),
    #[cfg(feature = "methods-optimizer")]
    Methods(PyMethodsPlsTrainingProvider),
}

impl TrainingDataProvider {
    fn data_handle_count(&self) -> usize {
        match self {
            Self::Host(provider) => provider.inner().handle_records().len(),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.inner.inner().handle_records().len(),
        }
    }

    fn data_view_count(&self) -> usize {
        match self {
            Self::Host(provider) => provider.inner().view_records().len(),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.inner.inner().view_records().len(),
        }
    }
}

impl RuntimeDataProvider for TrainingDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> dag_ml_core::Result<HandleRef> {
        match self {
            Self::Host(provider) => provider.materialize(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.materialize(request),
        }
    }

    fn make_view(&self, request: &DataViewRequest) -> dag_ml_core::Result<HandleRef> {
        match self {
            Self::Host(provider) => provider.make_view(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.make_view(request),
        }
    }

    fn training_data_identity(
        &self,
        binding: &DataBinding,
    ) -> dag_ml_core::Result<Option<dag_ml_core::TrainingDataIdentity>> {
        match self {
            Self::Host(provider) => provider.training_data_identity(binding),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.training_data_identity(binding),
        }
    }

    fn coordinator_relations(
        &self,
        binding: &DataBinding,
    ) -> dag_ml_core::Result<Option<SampleRelationSet>> {
        match self {
            Self::Host(provider) => provider.coordinator_relations(binding),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.coordinator_relations(binding),
        }
    }

    fn methods_pls_capability(&self) -> dag_ml_core::Result<()> {
        match self {
            Self::Host(provider) => provider.methods_pls_capability(),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.methods_pls_capability(),
        }
    }

    fn preflight_methods_pls(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<()> {
        match self {
            Self::Host(provider) => provider.preflight_methods_pls(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.preflight_methods_pls(request),
        }
    }

    fn methods_pls_data(
        &self,
        request: &MethodsPlsDataRequest,
    ) -> dag_ml_core::Result<MethodsPlsData> {
        match self {
            Self::Host(provider) => provider.methods_pls_data(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.methods_pls_data(request),
        }
    }
}

#[cfg(feature = "methods-optimizer")]
impl PyMethodsPlsTrainingProvider {
    fn new(
        bindings: Vec<DataBinding>,
        envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
        inputs: BTreeMap<String, MethodsPlsDataset>,
    ) -> dag_ml_core::Result<Self> {
        let expected_keys = bindings
            .iter()
            .map(|binding| {
                dag_ml_core::data_binding_requirement_key(&binding.node_id, &binding.input_name)
            })
            .collect::<BTreeSet<_>>();
        let actual_keys = inputs.keys().cloned().collect::<BTreeSet<_>>();
        if expected_keys != actual_keys {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "native Methods training inputs must exactly cover runtime bindings (missing: [{}]; unexpected: [{}])",
                expected_keys
                    .difference(&actual_keys)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                actual_keys
                    .difference(&expected_keys)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
            )));
        }

        let mut raw = InMemoryDataProvider::new(dag_ml_core::ControllerId::new(
            PY_DATA_PROVIDER_CONTROLLER_ID,
        )?);
        for envelope in envelopes.values().cloned() {
            raw.register_envelope(envelope)?;
        }
        let inner = EnvelopeAttestedRuntimeDataProvider::new(raw, bindings, envelopes)?;
        for (key, dataset) in &inputs {
            // PREDICT inputs legitimately carry no target matrix. FIT_CV and
            // REFIT demand it again at the exact scheduler request below.
            dataset.validate(&format!("native Methods input `{key}`"), false)?;
        }
        Ok(Self { inner, inputs })
    }

    fn dataset_for_view(
        dataset: &MethodsPlsDataset,
        sample_ids: &[SampleId],
        require_targets: bool,
        label: &str,
    ) -> dag_ml_core::Result<MethodsPlsDataset> {
        let index_by_id = dataset
            .sample_ids
            .iter()
            .enumerate()
            .map(|(index, sample_id)| (sample_id, index))
            .collect::<BTreeMap<_, _>>();
        let indices = sample_ids
            .iter()
            .map(|sample_id| {
                index_by_id.get(sample_id).copied().ok_or_else(|| {
                    dag_ml_core::DagMlError::RuntimeValidation(format!(
                        "native Methods training {label} view requests sample `{sample_id}` absent from its attested input"
                    ))
                })
            })
            .collect::<dag_ml_core::Result<Vec<_>>>()?;
        let select = |matrix: &MethodsPlsMatrix| MethodsPlsMatrix {
            values: indices
                .iter()
                .flat_map(|index| {
                    let start = index * matrix.cols;
                    matrix.values[start..start + matrix.cols].iter().copied()
                })
                .collect(),
            rows: indices.len(),
            cols: matrix.cols,
        };
        let y = match &dataset.y {
            Some(matrix) => Some(select(matrix)),
            None if require_targets => {
                return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                    "native Methods training {label} view requires targets"
                )))
            }
            None => None,
        };
        Ok(MethodsPlsDataset {
            sample_ids: sample_ids.to_vec(),
            x: select(&dataset.x),
            y,
            target_names: dataset.target_names.clone(),
        })
    }

    fn data_for(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<MethodsPlsData> {
        request.validate()?;
        if !matches!(
            request.phase,
            dag_ml_core::Phase::FitCv | dag_ml_core::Phase::Refit | dag_ml_core::Phase::Predict
        ) {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods provider supports FIT_CV, REFIT, and PREDICT only".to_string(),
            ));
        }
        if request.identity.is_none() {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods training requires a target-bound data identity".to_string(),
            ));
        }
        let key = dag_ml_core::data_binding_requirement_key(
            &request.binding.node_id,
            &request.binding.input_name,
        );
        let dataset = self.inputs.get(&key).ok_or_else(|| {
            dag_ml_core::DagMlError::RuntimeValidation(format!(
                "native Methods training has no input for `{key}`"
            ))
        })?;
        let fit_ids = request.fit_view.sample_ids.as_deref().ok_or_else(|| {
            dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods training fit view lacks scheduler-selected sample identities"
                    .to_string(),
            )
        })?;
        let requires_targets = matches!(
            request.phase,
            dag_ml_core::Phase::FitCv | dag_ml_core::Phase::Refit
        );
        let prediction = match &request.prediction_view {
            Some(view) => Some(Self::dataset_for_view(
                dataset,
                view.sample_ids.as_deref().ok_or_else(|| {
                    dag_ml_core::DagMlError::RuntimeValidation(
                        "native Methods training prediction view lacks scheduler-selected sample identities".to_string(),
                    )
                })?,
                requires_targets,
                "prediction",
            )?),
            None => None,
        };
        Ok(MethodsPlsData {
            fit: Self::dataset_for_view(dataset, fit_ids, requires_targets, "fit")?,
            prediction,
        })
    }
}

#[cfg(feature = "methods-optimizer")]
impl RuntimeDataProvider for PyMethodsPlsTrainingProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> dag_ml_core::Result<HandleRef> {
        self.inner.materialize(request)
    }

    fn make_view(&self, request: &DataViewRequest) -> dag_ml_core::Result<HandleRef> {
        self.inner.make_view(request)
    }

    fn training_data_identity(
        &self,
        binding: &DataBinding,
    ) -> dag_ml_core::Result<Option<dag_ml_core::TrainingDataIdentity>> {
        self.inner.training_data_identity(binding)
    }

    fn coordinator_relations(
        &self,
        binding: &DataBinding,
    ) -> dag_ml_core::Result<Option<SampleRelationSet>> {
        self.inner.coordinator_relations(binding)
    }

    fn methods_pls_capability(&self) -> dag_ml_core::Result<()> {
        Ok(())
    }

    fn preflight_methods_pls(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<()> {
        self.data_for(request).map(|_| ())
    }

    fn methods_pls_data(
        &self,
        request: &MethodsPlsDataRequest,
    ) -> dag_ml_core::Result<MethodsPlsData> {
        self.data_for(request)
    }
}

/// Process-local resources which make the retained artifact handles and data
/// views meaningful. Field order intentionally drops the handle stores before
/// the Python-backed controller registry.
struct TrainingResources {
    artifact_store: InMemoryArtifactStore,
    data_provider: TrainingDataProvider,
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
            .map(|resources| resources.data_provider.data_handle_count()))
    }

    /// Number of materialized data-view handles retained for replay/audit.
    #[getter]
    fn process_local_data_view_count(&self) -> PyResult<Option<usize>> {
        Ok(self
            .lock_resources()?
            .as_ref()
            .map(|resources| resources.data_provider.data_view_count()))
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

    /// Attach a validated native split-conformal calibration to this outcome.
    ///
    /// Every argument is a strict external JSON contract.  The core owns the
    /// calibration algorithm and validates the exact replay, relation, truth
    /// and provenance closure before it updates the signed outcome.
    #[pyo3(signature = (
        replay_json,
        binding_id,
        calibration_relations_json,
        truth_json,
        coverages_json,
        multi_target_policy_json,
        small_sample_policy_json
    ))]
    #[allow(clippy::too_many_arguments)]
    fn attach_conformal_calibration_json(
        &mut self,
        replay_json: &str,
        binding_id: &str,
        calibration_relations_json: &str,
        truth_json: &str,
        coverages_json: &str,
        multi_target_policy_json: &str,
        small_sample_policy_json: &str,
    ) -> PyResult<String> {
        let replay = parse_strict_json::<TrainingReplayOutcome>(
            replay_json,
            "conformal calibration replay outcome",
        )?;
        let relations = parse_strict_json::<SampleRelationSet>(
            calibration_relations_json,
            "conformal calibration relations",
        )?;
        let truth = parse_strict_json::<ConformalCalibrationTruth>(
            truth_json,
            "conformal calibration truth",
        )?;
        let coverages =
            parse_strict_json::<Vec<f64>>(coverages_json, "conformal calibration coverages")?;
        let multi_target_policy = parse_strict_json::<ConformalMultiTargetPolicy>(
            multi_target_policy_json,
            "conformal multi-target policy",
        )?;
        let small_sample_policy = parse_strict_json::<ConformalSmallSamplePolicy>(
            small_sample_policy_json,
            "conformal small-sample policy",
        )?;

        let calibration = calibrate_attached_training_replay_with_derived_context(
            &mut self.outcome,
            &replay,
            binding_id,
            &relations,
            truth,
            coverages,
            multi_target_policy,
            small_sample_policy,
        )
        .map_err(py_core_error)?;
        serialize_json(&calibration)
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

/// Opaque, native-owned attestation for the strict Methods terminal PREDICT
/// facade.
///
/// The class intentionally has no Python constructor and no writable fields.
/// It is created only after native terminal execution has produced a closed,
/// fingerprint-validated receipt.  Bindings may expose a read-only projection
/// of its JSON, but must not turn it into an authority-bearing mutable dict.
#[pyclass(module = "dag_ml._dag_ml", frozen)]
pub struct MethodsTerminalPredictionReceipt {
    json: String,
    terminal_run_id: String,
    receipt_fingerprint: String,
}

#[pymethods]
impl MethodsTerminalPredictionReceipt {
    /// Canonical closed receipt JSON. Consumers needing a mutable value must
    /// explicitly create a non-attesting snapshot from this string.
    fn json(&self) -> &str {
        &self.json
    }

    /// Derived RunContext identity used exclusively for terminal PREDICT.
    #[getter]
    fn terminal_run_id(&self) -> &str {
        &self.terminal_run_id
    }

    /// SHA-256 of the closed receipt content, excluding this field itself.
    #[getter]
    fn receipt_fingerprint(&self) -> &str {
        &self.receipt_fingerprint
    }

    /// Return a mutable JSON snapshot for display or serialization only.
    ///
    /// This is intentionally not an attesting mapping: changing the returned
    /// object cannot modify or rebind this native receipt.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        py.import("json")?
            .getattr("loads")?
            .call1((&self.json,))
            .map(|value| value.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "MethodsTerminalPredictionReceipt(terminal_run_id={:?}, receipt_fingerprint={:?})",
            self.terminal_run_id, self.receipt_fingerprint
        )
    }
}

/// Native-owned outcome of the strict Methods terminal facade.
///
/// This public type is deliberately a frozen PyO3 class rather than a Python
/// convenience wrapper.  Its terminal receipt therefore remains tied to the
/// native execution result; package and prediction getters create ordinary,
/// non-attesting convenience views from their retained JSON snapshots.
#[pyclass(module = "dag_ml._dag_ml", frozen)]
pub struct MethodsTerminalPredictionResult {
    training_result: Py<TrainingResult>,
    portable_predictor_package_json: String,
    terminal_prediction_json: String,
    terminal_receipt: Py<MethodsTerminalPredictionReceipt>,
}

#[pymethods]
impl MethodsTerminalPredictionResult {
    /// Attached native training result retained by this terminal outcome.
    #[getter]
    fn training_result(&self, py: Python<'_>) -> Py<TrainingResult> {
        self.training_result.clone_ref(py)
    }

    /// Ordinary Python Package V2 view. It is reconstructed from JSON on each
    /// access and is not the terminal receipt's authority boundary.
    #[getter]
    fn portable_predictor_package(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        py.import("dag_ml")?
            .getattr("PortablePredictorPackage")?
            .call1((&self.portable_predictor_package_json,))
            .map(|value| value.unbind())
    }

    /// Native Package V2 JSON retained by the terminal result.
    #[getter]
    fn portable_predictor_package_json(&self) -> &str {
        &self.portable_predictor_package_json
    }

    /// Ordinary mutable prediction snapshot.  It is output data, not a
    /// sealed receipt, so callers may transform it without affecting native
    /// terminal attestation.
    #[getter]
    fn terminal_prediction(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        py.import("json")?
            .getattr("loads")?
            .call1((&self.terminal_prediction_json,))
            .map(|value| value.unbind())
    }

    /// Native serialized terminal prediction for explicit snapshot handling.
    #[getter]
    fn terminal_prediction_json(&self) -> &str {
        &self.terminal_prediction_json
    }

    /// Authoritative frozen native receipt for this exact terminal execution.
    #[getter]
    fn terminal_receipt(&self, py: Python<'_>) -> Py<MethodsTerminalPredictionReceipt> {
        self.terminal_receipt.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let receipt = self.terminal_receipt.bind(py).borrow();
        format!(
            "MethodsTerminalPredictionResult(terminal_run_id={:?}, receipt_fingerprint={:?})",
            receipt.terminal_run_id, receipt.receipt_fingerprint
        )
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
    let data_provider = TrainingDataProvider::Host(
        EnvelopeAttestedRuntimeDataProvider::new(inner_provider, bindings, envelopes)
            .map_err(py_core_error)?,
    );
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

/// Execute the narrow portable Methods model training lane.
///
/// Unlike [`execute_training_json`], this entry point does not accept a Python
/// operator callback. Every executable node must be the registered native
/// Methods PLS or Ridge controller, and numeric rows are supplied through the
/// typed `methods_inputs_json` provider. This is the public bridge for hosts
/// that want a durable N4MM Package V2 rather than a process-local sidecar.
#[pyfunction]
#[pyo3(signature = (
    request_json,
    data_envelopes_json,
    relations_json,
    training_influence_json,
    methods_inputs_json,
    methods_library_path,
    outcome_id,
    run_id,
    bundle_id,
    warnings_json = "[]",
    diagnostics_json = "{}"
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_methods_training_json(
    py: Python<'_>,
    request_json: &str,
    data_envelopes_json: &str,
    relations_json: &str,
    training_influence_json: &str,
    methods_inputs_json: &str,
    methods_library_path: &str,
    outcome_id: &str,
    run_id: &str,
    bundle_id: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<TrainingResult> {
    #[cfg(not(feature = "methods-optimizer"))]
    {
        let _ = (
            py,
            request_json,
            data_envelopes_json,
            relations_json,
            training_influence_json,
            methods_inputs_json,
            methods_library_path,
            outcome_id,
            run_id,
            bundle_id,
            warnings_json,
            diagnostics_json,
        );
        Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "Methods training support is absent from this dag-ml binding; install a wheel rebuilt with the `methods-optimizer` feature".to_string(),
        )))
    }
    #[cfg(feature = "methods-optimizer")]
    {
        let request = TrainingRequest::from_json(request_json).map_err(py_core_error)?;
        let projection = request.project().map_err(py_core_error)?;
        let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
            data_envelopes_json,
            "native Methods training data envelope map",
        )?;
        for envelope in envelopes.values() {
            envelope.validate().map_err(py_core_error)?;
        }
        let relations = parse_strict_json::<SampleRelationSet>(
            relations_json,
            "native Methods training sample relations",
        )?;
        relations.validate().map_err(py_core_error)?;
        let training_influence = parse_strict_json::<TrainingInfluenceManifest>(
            training_influence_json,
            "native Methods training influence manifest",
        )?;
        training_influence.validate().map_err(py_core_error)?;
        let warnings =
            parse_strict_json::<Vec<String>>(warnings_json, "native Methods training warnings")?;
        let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
            diagnostics_json,
            "native Methods training diagnostics",
        )?;
        let bindings = projection
            .plan
            .node_plans
            .values()
            .flat_map(|node_plan| node_plan.data_bindings.iter().cloned())
            .collect::<Vec<DataBinding>>();
        if projection
            .plan
            .node_plans
            .values()
            .any(|node_plan| !is_native_methods_model_controller(&node_plan.controller_id))
        {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods training requires every executable node to use controller:methods.pls or controller:methods.ridge; host controller fallback is forbidden".to_string(),
            )));
        }
        let raw_inputs = parse_strict_json::<BTreeMap<String, MethodsTrainingInputJson>>(
            methods_inputs_json,
            "native Methods training input map",
        )?;
        let inputs = raw_inputs
            .into_iter()
            .map(|(key, input)| Ok((key, methods_dataset_from_json(input, true)?)))
            .collect::<dag_ml_core::Result<BTreeMap<_, _>>>()
            .map_err(py_core_error)?;
        let data_provider = TrainingDataProvider::Methods(
            PyMethodsPlsTrainingProvider::new(bindings, envelopes, inputs)
                .map_err(py_core_error)?,
        );
        let runtime =
            dag_ml_core::MethodsRuntime::configure(methods_library_path).map_err(|error| {
                py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                    error.to_string(),
                ))
            })?;
        let mut controllers = RuntimeControllerRegistry::new();
        register_methods_training_controllers(&projection, runtime, &mut controllers)
            .map_err(py_core_error)?;
        let run_id = RunId::new(run_id).map_err(py_core_error)?;
        let bundle_id = BundleId::new(bundle_id).map_err(py_core_error)?;
        let resources = TrainingResources {
            artifact_store: InMemoryArtifactStore::new(),
            data_provider,
            controllers,
        };
        let outcome_id = outcome_id.to_string();
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
}

/// Execute the bounded callback-free Methods CV -> REFIT -> terminal PREDICT
/// facade.
///
/// This is intentionally *not* a convenience wrapper around
/// `run_cv_refit_predict_in_process`: that generic API requires a host Python
/// node callback.  Here the native Methods PLS controller owns every model
/// invocation, and the final X-only cohort is consumed through the
/// target-free, envelope-attested Methods provider.
///
/// Native CV necessarily computes an *ephemeral internal* OOF score to choose
/// the one candidate before REFIT.  This facade never accepts, retains, or
/// exposes an OOF lane: OOF-consuming graph edges, caches, reductions, and
/// stacking contracts are rejected before libn4m is configured.
///
/// The return value is an object rather than JSON because it deliberately
/// retains the owning [`TrainingResult`] alongside portable JSON values:
/// `{training_result, portable_predictor_package_json,
/// terminal_prediction_json, terminal_receipt}`.  The receipt is an opaque
/// native object, not a mutable decoded JSON mapping.
#[pyfunction]
#[pyo3(signature = (
    request_json,
    data_envelopes_json,
    relations_json,
    training_influence_json,
    methods_inputs_json,
    predict_envelope_json,
    predict_input_json,
    methods_library_path,
    outcome_id,
    run_id,
    bundle_id,
    package_id,
    terminal_selector_json,
    warnings_json = "[]",
    diagnostics_json = "{}"
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_methods_cv_refit_terminal_predict_json(
    py: Python<'_>,
    request_json: &str,
    data_envelopes_json: &str,
    relations_json: &str,
    training_influence_json: &str,
    methods_inputs_json: &str,
    predict_envelope_json: &str,
    predict_input_json: &str,
    methods_library_path: &str,
    outcome_id: &str,
    run_id: &str,
    bundle_id: &str,
    package_id: &str,
    terminal_selector_json: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<Py<MethodsTerminalPredictionResult>> {
    #[cfg(not(feature = "methods-optimizer"))]
    {
        let _ = (
            py,
            request_json,
            data_envelopes_json,
            relations_json,
            training_influence_json,
            methods_inputs_json,
            predict_envelope_json,
            predict_input_json,
            methods_library_path,
            outcome_id,
            run_id,
            bundle_id,
            package_id,
            terminal_selector_json,
            warnings_json,
            diagnostics_json,
        );
        Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "strict Methods terminal prediction support is absent from this dag-ml binding; install a wheel rebuilt with the `methods-optimizer` feature".to_string(),
        )))
    }
    #[cfg(feature = "methods-optimizer")]
    {
        // Validate every caller-controlled execution identity before parsing
        // data or configuring libn4m.  The terminal RunId is deliberately
        // derived and validated here as well: constructing it later would let
        // an overlong but otherwise valid CV run id reach native execution.
        let identity = prevalidate_strict_methods_terminal_identity(
            outcome_id, run_id, bundle_id, package_id,
        )?;
        // All narrow-contract checks happen before the existing training entry
        // can configure libn4m or execute FIT_CV.  In particular, a bad V2
        // cohort or terminal port is never allowed to turn into partial CV.
        let strict = parse_strict_methods_terminal_facade_inputs(
            request_json,
            data_envelopes_json,
            relations_json,
            training_influence_json,
            methods_inputs_json,
            predict_envelope_json,
            predict_input_json,
            terminal_selector_json,
            warnings_json,
            diagnostics_json,
        )?;

        let training_result = execute_methods_training_json(
            py,
            request_json,
            data_envelopes_json,
            relations_json,
            training_influence_json,
            methods_inputs_json,
            methods_library_path,
            &identity.outcome_id,
            identity.run_id.as_str(),
            identity.bundle_id.as_str(),
            warnings_json,
            diagnostics_json,
        )?;
        let package = training_result
            .outcome
            .to_portable_predictor_package(
                &identity.package_id,
                FittedArtifactMode::PortableRequired,
                ArtifactLoadMode::NativePortable,
            )
            .map_err(py_core_error)?;
        let terminal = execute_attached_methods_terminal_prediction(
            &training_result,
            &package,
            &strict.predict_envelope,
            strict.predict_dataset,
            &strict.selector,
            &identity.terminal_run_id,
        )?;
        let terminal_receipt = Py::new(
            py,
            MethodsTerminalPredictionReceipt::from_terminal(&terminal, &identity.terminal_run_id)?,
        )?;
        Py::new(
            py,
            MethodsTerminalPredictionResult {
                training_result: Py::new(py, training_result)?,
                portable_predictor_package_json: serialize_json(&package)?,
                terminal_prediction_json: serialize_json(terminal.prediction())?,
                terminal_receipt,
            },
        )
    }
}

#[cfg(feature = "methods-optimizer")]
#[derive(Debug)]
struct StrictMethodsTerminalFacadeInputs {
    predict_envelope: ExternalDataPlanEnvelope,
    predict_dataset: MethodsPlsDataset,
    selector: dag_ml_core::TerminalPredictionSelector,
}

/// Caller-supplied identity material validated before any Methods runtime or
/// native data path is constructed.
#[cfg(feature = "methods-optimizer")]
#[derive(Debug)]
struct StrictMethodsTerminalIdentity {
    outcome_id: String,
    run_id: RunId,
    bundle_id: BundleId,
    package_id: String,
    terminal_run_id: RunId,
}

#[cfg(feature = "methods-optimizer")]
fn prevalidate_strict_methods_terminal_identity(
    outcome_id: &str,
    run_id: &str,
    bundle_id: &str,
    package_id: &str,
) -> PyResult<StrictMethodsTerminalIdentity> {
    let run_id = RunId::new(run_id).map_err(py_core_error)?;
    let bundle_id = BundleId::new(bundle_id).map_err(py_core_error)?;
    validate_strict_methods_terminal_identifier("outcome_id", outcome_id).map_err(py_core_error)?;
    validate_strict_methods_terminal_identifier("package_id", package_id).map_err(py_core_error)?;
    let terminal_run_id =
        RunId::new(format!("{run_id}:methods-terminal-predict")).map_err(py_core_error)?;
    Ok(StrictMethodsTerminalIdentity {
        outcome_id: outcome_id.to_string(),
        run_id,
        bundle_id,
        package_id: package_id.to_string(),
        terminal_run_id,
    })
}

#[cfg(feature = "methods-optimizer")]
fn validate_strict_methods_terminal_identifier(
    label: &str,
    value: &str,
) -> dag_ml_core::Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
    {
        return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
            "strict Methods terminal {label} must be a valid DAG-ML identifier"
        )));
    }
    Ok(())
}

/// Closed wire contract for a strict Methods terminal receipt.
///
/// `dag-ml-core` deliberately keeps its terminal receipt as a one-way
/// serialization type.  The Python boundary adds the terminal RunContext
/// identity and a self-fingerprint, while retaining the core attestation
/// fields verbatim.  The explicit shape plus `deny_unknown_fields` means a
/// decoded receipt cannot silently acquire an un-attested field.
#[cfg(feature = "methods-optimizer")]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StrictMethodsTerminalPredictionReceipt {
    schema_version: u32,
    terminal_run_id: RunId,
    bundle_id: BundleId,
    plan_id: String,
    graph_fingerprint: String,
    campaign_fingerprint: String,
    controller_fingerprint: String,
    selected_variant_id: dag_ml_core::VariantId,
    terminal_node_id: dag_ml_core::NodeId,
    terminal_port: String,
    cohort_fingerprint: String,
    refit_artifacts: Vec<dag_ml_core::RefitArtifactRecord>,
    output_fingerprint: String,
    receipt_fingerprint: String,
}

#[cfg(feature = "methods-optimizer")]
#[derive(Serialize)]
struct StrictMethodsTerminalReceiptFingerprintPayload<'a> {
    schema_version: u32,
    terminal_run_id: &'a RunId,
    bundle_id: &'a BundleId,
    plan_id: &'a str,
    graph_fingerprint: &'a str,
    campaign_fingerprint: &'a str,
    controller_fingerprint: &'a str,
    selected_variant_id: &'a dag_ml_core::VariantId,
    terminal_node_id: &'a dag_ml_core::NodeId,
    terminal_port: &'a str,
    cohort_fingerprint: &'a str,
    refit_artifacts: &'a [dag_ml_core::RefitArtifactRecord],
    output_fingerprint: &'a str,
}

#[cfg(feature = "methods-optimizer")]
impl StrictMethodsTerminalPredictionReceipt {
    fn from_terminal_execution(
        terminal: &dag_ml_core::TerminalPredictionExecution,
        terminal_run_id: &RunId,
    ) -> dag_ml_core::Result<Self> {
        let core_receipt = terminal.receipt();
        let mut receipt = Self {
            schema_version: core_receipt.schema_version(),
            terminal_run_id: terminal_run_id.clone(),
            bundle_id: core_receipt.bundle_id().clone(),
            plan_id: core_receipt.plan_id().to_string(),
            graph_fingerprint: core_receipt.graph_fingerprint().to_string(),
            campaign_fingerprint: core_receipt.campaign_fingerprint().to_string(),
            controller_fingerprint: core_receipt.controller_fingerprint().to_string(),
            selected_variant_id: core_receipt.selected_variant_id().clone(),
            terminal_node_id: core_receipt.terminal_node_id().clone(),
            terminal_port: core_receipt.terminal_port().to_string(),
            cohort_fingerprint: core_receipt.cohort_fingerprint().to_string(),
            refit_artifacts: core_receipt.refit_artifacts().to_vec(),
            output_fingerprint: core_receipt.output_fingerprint().to_string(),
            receipt_fingerprint: String::new(),
        };
        receipt.receipt_fingerprint = receipt.compute_fingerprint()?;
        receipt.validate()?;
        Ok(receipt)
    }

    fn fingerprint_payload(&self) -> StrictMethodsTerminalReceiptFingerprintPayload<'_> {
        StrictMethodsTerminalReceiptFingerprintPayload {
            schema_version: self.schema_version,
            terminal_run_id: &self.terminal_run_id,
            bundle_id: &self.bundle_id,
            plan_id: &self.plan_id,
            graph_fingerprint: &self.graph_fingerprint,
            campaign_fingerprint: &self.campaign_fingerprint,
            controller_fingerprint: &self.controller_fingerprint,
            selected_variant_id: &self.selected_variant_id,
            terminal_node_id: &self.terminal_node_id,
            terminal_port: &self.terminal_port,
            cohort_fingerprint: &self.cohort_fingerprint,
            refit_artifacts: &self.refit_artifacts,
            output_fingerprint: &self.output_fingerprint,
        }
    }

    fn compute_fingerprint(&self) -> dag_ml_core::Result<String> {
        let canonical = serde_json::to_vec(&self.fingerprint_payload()).map_err(|error| {
            dag_ml_core::DagMlError::RuntimeValidation(format!(
                "strict Methods terminal receipt cannot be canonically serialized: {error}"
            ))
        })?;
        Ok(format!("{:x}", Sha256::digest(canonical)))
    }

    fn validate(&self) -> dag_ml_core::Result<()> {
        if self.schema_version != dag_ml_core::TERMINAL_PREDICTION_RECEIPT_SCHEMA_VERSION {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "strict Methods terminal receipt schema V{} is unsupported",
                self.schema_version
            )));
        }
        validate_strict_methods_terminal_identifier("receipt plan_id", &self.plan_id)?;
        if !self
            .terminal_run_id
            .as_str()
            .ends_with(":methods-terminal-predict")
        {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "strict Methods terminal receipt is not bound to a terminal RunId".to_string(),
            ));
        }
        if self.terminal_port != "oof" {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "strict Methods terminal receipt selects a port other than oof".to_string(),
            ));
        }
        for (label, value) in [
            ("graph", self.graph_fingerprint.as_str()),
            ("campaign", self.campaign_fingerprint.as_str()),
            ("controller", self.controller_fingerprint.as_str()),
            ("cohort", self.cohort_fingerprint.as_str()),
            ("output", self.output_fingerprint.as_str()),
            ("receipt", self.receipt_fingerprint.as_str()),
        ] {
            validate_strict_methods_terminal_fingerprint(label, value)?;
        }
        if self.refit_artifacts.len() != 1 {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "strict Methods terminal receipt must attest exactly one REFIT artifact"
                    .to_string(),
            ));
        }
        let refit = &self.refit_artifacts[0];
        refit.validate()?;
        if refit.node_id != self.terminal_node_id {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "strict Methods terminal receipt REFIT artifact does not belong to its terminal node"
                    .to_string(),
            ));
        }
        let expected = self.compute_fingerprint()?;
        if self.receipt_fingerprint != expected {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "strict Methods terminal receipt fingerprint does not match its closed canonical content"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "methods-optimizer")]
fn validate_strict_methods_terminal_fingerprint(
    label: &str,
    value: &str,
) -> dag_ml_core::Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
            "strict Methods terminal receipt {label} fingerprint must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(feature = "methods-optimizer")]
fn parse_strict_methods_terminal_receipt_json(
    json: &str,
) -> dag_ml_core::Result<StrictMethodsTerminalPredictionReceipt> {
    let original = serde_json::from_str::<serde_json::Value>(json).map_err(|error| {
        dag_ml_core::DagMlError::RuntimeValidation(format!(
            "strict Methods terminal receipt is not valid JSON: {error}"
        ))
    })?;
    let receipt =
        serde_json::from_value::<StrictMethodsTerminalPredictionReceipt>(original.clone())
            .map_err(|error| {
                dag_ml_core::DagMlError::RuntimeValidation(format!(
                    "strict Methods terminal receipt violates its closed schema: {error}"
                ))
            })?;
    receipt.validate()?;
    let canonical = serde_json::to_value(&receipt).map_err(|error| {
        dag_ml_core::DagMlError::RuntimeValidation(format!(
            "strict Methods terminal receipt cannot be reserialized: {error}"
        ))
    })?;
    if canonical != original {
        return Err(dag_ml_core::DagMlError::RuntimeValidation(
            "strict Methods terminal receipt contains non-canonical or unknown content".to_string(),
        ));
    }
    Ok(receipt)
}

#[cfg(feature = "methods-optimizer")]
impl MethodsTerminalPredictionReceipt {
    fn from_terminal(
        terminal: &dag_ml_core::TerminalPredictionExecution,
        terminal_run_id: &RunId,
    ) -> PyResult<Self> {
        let receipt = StrictMethodsTerminalPredictionReceipt::from_terminal_execution(
            terminal,
            terminal_run_id,
        )
        .map_err(py_core_error)?;
        let json = serialize_json(&receipt)?;
        let receipt = parse_strict_methods_terminal_receipt_json(&json).map_err(py_core_error)?;
        Ok(Self {
            terminal_run_id: receipt.terminal_run_id.to_string(),
            receipt_fingerprint: receipt.receipt_fingerprint,
            json,
        })
    }
}

/// Parse and refuse every unsupported strict-facade feature before native
/// model work.  This is deliberately structural rather than a best-effort
/// "PLS-like" recognition: accepting a transform, variant generator, HPO
/// descriptor, group relation, opaque metadata, or OOF dependency would
/// weaken the bounded callback-free integration contract.
#[cfg(feature = "methods-optimizer")]
#[allow(clippy::too_many_arguments)]
fn parse_strict_methods_terminal_facade_inputs(
    request_json: &str,
    data_envelopes_json: &str,
    relations_json: &str,
    training_influence_json: &str,
    methods_inputs_json: &str,
    predict_envelope_json: &str,
    predict_input_json: &str,
    terminal_selector_json: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<StrictMethodsTerminalFacadeInputs> {
    let request = TrainingRequest::from_json(request_json).map_err(py_core_error)?;
    let projection = request.project().map_err(py_core_error)?;
    let training_envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        data_envelopes_json,
        "strict Methods training data envelope map",
    )?;
    for envelope in training_envelopes.values() {
        envelope.validate().map_err(py_core_error)?;
    }
    let relations = parse_strict_json::<SampleRelationSet>(
        relations_json,
        "strict Methods training sample relations",
    )?;
    relations.validate().map_err(py_core_error)?;
    let training_influence = parse_strict_json::<TrainingInfluenceManifest>(
        training_influence_json,
        "strict Methods training influence manifest",
    )?;
    training_influence.validate().map_err(py_core_error)?;
    let raw_training_inputs = parse_strict_json::<BTreeMap<String, MethodsTrainingInputJson>>(
        methods_inputs_json,
        "strict Methods training input map",
    )?;
    let training_inputs = raw_training_inputs
        .into_iter()
        .map(|(key, input)| Ok((key, methods_dataset_from_json(input, true)?)))
        .collect::<dag_ml_core::Result<BTreeMap<_, _>>>()
        .map_err(py_core_error)?;
    let predict_envelope = parse_strict_json::<ExternalDataPlanEnvelope>(
        predict_envelope_json,
        "strict Methods terminal PREDICT envelope",
    )?;
    predict_envelope.validate().map_err(py_core_error)?;
    let predict_input = parse_strict_json::<MethodsPredictInputJson>(
        predict_input_json,
        "strict Methods terminal PREDICT input",
    )?;
    let predict_dataset =
        methods_predict_dataset_from_json(predict_input).map_err(py_core_error)?;
    let selector = parse_strict_json::<dag_ml_core::TerminalPredictionSelector>(
        terminal_selector_json,
        "strict Methods terminal prediction selector",
    )?;
    let warnings =
        parse_strict_json::<Vec<String>>(warnings_json, "strict Methods terminal warnings")?;
    let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
        diagnostics_json,
        "strict Methods terminal diagnostics",
    )?;

    validate_strict_methods_terminal_facade_contract(
        &request,
        &projection,
        &training_envelopes,
        &relations,
        &training_influence,
        &training_inputs,
        &predict_envelope,
        &predict_dataset,
        &selector,
        &warnings,
        &diagnostics,
    )
    .map_err(py_core_error)?;
    Ok(StrictMethodsTerminalFacadeInputs {
        predict_envelope,
        predict_dataset,
        selector,
    })
}

#[cfg(feature = "methods-optimizer")]
#[allow(clippy::too_many_arguments)]
fn validate_strict_methods_terminal_facade_contract(
    request: &TrainingRequest,
    projection: &dag_ml_core::TrainingContractProjection,
    training_envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    relations: &SampleRelationSet,
    training_influence: &TrainingInfluenceManifest,
    training_inputs: &BTreeMap<String, MethodsPlsDataset>,
    predict_envelope: &ExternalDataPlanEnvelope,
    predict_dataset: &MethodsPlsDataset,
    selector: &dag_ml_core::TerminalPredictionSelector,
    warnings: &[String],
    diagnostics: &BTreeMap<String, serde_json::Value>,
) -> dag_ml_core::Result<()> {
    use dag_ml_core::{
        AggregationMethod, AggregationWeights, ArtifactPolicy, AugmentationScope,
        ControllerCapability, ControllerFitScope, CvArtifactRetention, DataRequestPartition,
        EntityUnitLevel, EvaluationScope, FeatureSelectionScope, FittedArtifactMode,
        GenerationStrategy, Granularity, Phase, PredictionCacheRetention, PredictionKind,
        PredictionLevel, RefitStrategy, RngPolicy, SplitUnit, TrainingSchedulerKind,
    };

    let refuse = |detail: &str| {
        dag_ml_core::DagMlError::RuntimeValidation(format!(
            "strict Methods CV/REFIT/terminal-PREDICT facade refuses {detail}"
        ))
    };

    if !warnings.is_empty() || !diagnostics.is_empty() {
        return Err(refuse("warnings or diagnostics metadata"));
    }

    if !request.parameter_patches.is_empty()
        || !request.patch_policies.is_empty()
        || !request.influence_requirements.is_empty()
        || !request.training_losses.is_empty()
    {
        return Err(refuse(
            "parameter patches, training-loss roles, or controller influence extensions",
        ));
    }
    if !request.graph.metadata.is_empty() || !request.campaign.metadata.is_empty() {
        return Err(refuse("graph or campaign metadata"));
    }
    if request.graph.nodes.len() != 1 || !request.graph.edges.is_empty() {
        return Err(refuse("transforms, joins, generators, or graph edges"));
    }
    let graph_node = &request.graph.nodes[0];
    if graph_node.kind != dag_ml_core::NodeKind::Model
        || !graph_node.metadata.is_empty()
        || graph_node.seed_label.is_some()
        || !operator_is_strict_pls(graph_node.operator.as_ref())
    {
        return Err(refuse(
            "a non-PLS model node, node metadata, or a seeded model variant",
        ));
    }
    if graph_node.ports.inputs.len() != 1
        || graph_node.ports.inputs[0].name != "x"
        || graph_node.ports.inputs[0].kind != dag_ml_core::PortKind::Data
        || graph_node.ports.outputs.len() != 1
        || graph_node.ports.outputs[0].name != "oof"
        || graph_node.ports.outputs[0].kind != dag_ml_core::PortKind::Prediction
    {
        return Err(refuse("a model port layout other than x -> oof prediction"));
    }
    if graph_node.params.len() != 1
        || graph_node
            .params
            .get("n_components")
            .and_then(serde_json::Value::as_i64)
            .filter(|value| *value > 0)
            .is_none()
    {
        return Err(refuse(
            "PLS parameters other than one positive integer n_components",
        ));
    }

    if projection.plan.node_plans.len() != 1
        || projection.plan.controller_manifests.len() != 1
        || projection.plan.variants.len() != 1
        || projection.plan.graph_plan.topological_order != vec![graph_node.id.clone()]
    {
        return Err(refuse(
            "multiple executable nodes, controller alternatives, or generated variants",
        ));
    }
    let node_plan = projection
        .plan
        .node_plans
        .get(&graph_node.id)
        .ok_or_else(|| refuse("a graph node absent from its effective plan"))?;
    if node_plan.controller_id.as_str() != dag_ml_core::METHODS_PLS_CONTROLLER_ID
        || node_plan.kind != dag_ml_core::NodeKind::Model
        || node_plan.fit_scope != ControllerFitScope::FoldTrain
        || node_plan.rng_policy != RngPolicy::UsesCoreSeed
        || node_plan.artifact_policy != ArtifactPolicy::Serializable
        || node_plan.inner_cv.is_some()
        || !node_plan.input_nodes.is_empty()
        || !node_plan.output_nodes.is_empty()
        || !node_plan.training_losses.is_empty()
        || node_plan.params != graph_node.params
    {
        return Err(refuse(
            "a non-native-PLS execution plan, nested CV, or graph dependency",
        ));
    }
    let expected_phases = BTreeSet::from([Phase::FitCv, Phase::Refit, Phase::Predict]);
    let expected_capabilities = BTreeSet::from([
        ControllerCapability::Deterministic,
        ControllerCapability::ThreadSafe,
        ControllerCapability::ProcessSafe,
        ControllerCapability::UsesCoreRng,
        ControllerCapability::EmitsPredictions,
        ControllerCapability::EmitsArtifacts,
        ControllerCapability::Stateful,
    ]);
    if node_plan.supported_phases != expected_phases
        || node_plan.controller_capabilities != expected_capabilities
    {
        return Err(refuse(
            "a callback/GIL, OOF-consumption, HPO, generator, aggregation, or other non-PLS controller capability",
        ));
    }
    if node_plan.data_bindings.len() != 1 || node_plan.shape_plan.is_none() {
        return Err(refuse(
            "multiple data bindings or a missing native PLS shape plan",
        ));
    }
    let binding = &node_plan.data_bindings[0];
    if binding.node_id != graph_node.id
        || binding.input_name != "x"
        || !binding.metadata.is_empty()
        || binding.view_policy.fit_partition != DataRequestPartition::FoldTrain
        || binding.view_policy.predict_partition != DataRequestPartition::FoldValidation
        || binding.view_policy.include_augmented_train
        || binding.view_policy.include_augmented_validation
        || binding.view_policy.include_excluded
        || !binding.view_policy.require_sample_ids
        || !binding.view_policy.unsafe_flags.is_empty()
    {
        return Err(refuse(
            "binding metadata, augmentation, or a non-explicit-ID data-view policy",
        ));
    }
    let shape = node_plan.shape_plan.as_ref().expect("checked above");
    if shape.input_granularity != Granularity::Sample
        || shape.target_granularity != Granularity::Sample
        || shape.fit_rows != dag_ml_core::FitBoundary::FoldTrain
        || shape.predict_rows != dag_ml_core::FitBoundary::FoldValidation
        || shape.augmentation_policy.sample_scope != AugmentationScope::None
        || shape.augmentation_policy.feature_scope != AugmentationScope::None
        || !shape.augmentation_policy.unsafe_flags.is_empty()
        || shape.selection_policy.scope != FeatureSelectionScope::None
        || shape.selection_policy.allow_schema_mismatch_on_join
    {
        return Err(refuse(
            "shape transforms, augmentation, or feature selection",
        ));
    }

    let campaign = &projection.plan.campaign;
    if campaign.generation.strategy != GenerationStrategy::None
        || !campaign.generation.dimensions.is_empty()
        || !campaign.generation.constraints.is_empty()
        || !matches!(campaign.generation.max_variants, None | Some(1))
        || !campaign.branch_view_plans.is_empty()
        || campaign.inner_cv.is_some()
        || !campaign.metadata.is_empty()
        || campaign.data_bindings.len() != 1
        || campaign.shape_plans.len() != 1
        || campaign.leakage_policy.split_unit != SplitUnit::Sample
        || !campaign.leakage_policy.forbid_origin_cross_fold
        || campaign
            .leakage_policy
            .allow_observation_split_with_shared_target
        || campaign.leakage_policy.require_group_ids
        || !campaign.leakage_policy.unsafe_flags.is_empty()
        || campaign.aggregation_policy.aggregation_level != PredictionLevel::Sample
        || campaign.aggregation_policy.selection_metric_level != PredictionLevel::Sample
        || campaign.aggregation_policy.method != AggregationMethod::Mean
        || campaign.aggregation_policy.weights != AggregationWeights::None
        || campaign.aggregation_policy.custom_controller.is_some()
    {
        return Err(refuse(
            "generators, HPO, branch views, groups, metadata, or non-sample aggregation",
        ));
    }
    let split = campaign
        .split_invocation
        .as_ref()
        .ok_or_else(|| refuse("a missing explicit KFold invocation"))?;
    let fold_set = split
        .fold_set
        .as_ref()
        .ok_or_else(|| refuse("a missing explicit KFold fold set"))?;
    if split.controller_id.is_some()
        || split.leakage_policy.split_unit != SplitUnit::Sample
        || !split.leakage_policy.forbid_origin_cross_fold
        || split
            .leakage_policy
            .allow_observation_split_with_shared_target
        || split.leakage_policy.require_group_ids
        || !split.leakage_policy.unsafe_flags.is_empty()
        || split.params.get("kind").and_then(serde_json::Value::as_str) != Some("kfold")
        || split
            .params
            .get("shuffle")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || split
            .params
            .get("n_splits")
            .and_then(serde_json::Value::as_u64)
            != Some(fold_set.folds.len() as u64)
        || split.params.len() != 3
        || !fold_set.sample_groups.is_empty()
        || fold_set.partition_mode != dag_ml_core::FoldPartitionMode::Partition
        || fold_set.folds.iter().any(|fold| !fold.metadata.is_empty())
    {
        return Err(refuse(
            "a non-KFold/no-shuffle split, groups, or fold metadata",
        ));
    }

    if !request.options.refit
        || request.options.refit_strategy != Some(RefitStrategy::RefitOne)
        || request.options.outputs.len() != 1
        // CV still computes its own ephemeral OOF scores.  This excludes any
        // caller-supplied, retained, reduced, or stack-consumed OOF lane while
        // requiring the native runtime's one supported selection scope.
        || request.options.selection.required_metric_level != Some(PredictionLevel::Sample)
        || request.options.selection.evaluation_scope != Some(EvaluationScope::Oof)
        || !request.options.selection.require_finite
        || request.options.selection.stacking_fit_contract.is_some()
        || request.options.selection.refit_slot_plan.is_some()
        || request.options.selection.reduction_id.is_some()
        || request.options.scheduler.kind != TrainingSchedulerKind::Sequential
        || request.options.scheduler.backend.is_some()
        || request.options.scheduler.workers != 1
        || !request.options.resources.gpu_devices.is_empty()
        || request.options.artifacts.cv_artifacts != CvArtifactRetention::Discard
        || request.options.artifacts.prediction_caches != PredictionCacheRetention::Discard
        || request.options.artifacts.fitted_artifacts != FittedArtifactMode::PortableRequired
    {
        return Err(refuse(
            "non-refit-one options, non-internal-OOF selection, scheduler extensions, GPU resources, or retained OOF caches",
        ));
    }
    let output = projection
        .outputs
        .first()
        .ok_or_else(|| refuse("a missing output"))?;
    if selector.node_id != graph_node.id
        || selector.port != "oof"
        || output.node_id != graph_node.id
        || output.port_name != "oof"
        || output.prediction_level != PredictionLevel::Sample
        || output.unit_level != Some(EntityUnitLevel::PhysicalSample)
        || output.prediction_kind != PredictionKind::RegressionPoint
        || output.target_names.len() != 1
        || output.target_units.len() != 1
        || output.class_labels != vec![Vec::<String>::new()]
        || output.target_space != "raw"
    {
        return Err(refuse(
            "a terminal port other than model.oof or a non-sample numeric single-target output",
        ));
    }
    if request.options.selection_output_id != output.output_id {
        return Err(refuse(
            "a selection output different from the terminal PLS output",
        ));
    }

    let binding_key =
        dag_ml_core::data_binding_requirement_key(&binding.node_id, &binding.input_name);
    if training_envelopes.len() != 1 || training_inputs.len() != 1 {
        return Err(refuse("more than one raw training array binding"));
    }
    let training_envelope = training_envelopes
        .get(&binding_key)
        .ok_or_else(|| refuse("a training envelope that does not exactly cover model.x"))?;
    let training_dataset = training_inputs
        .get(&binding_key)
        .ok_or_else(|| refuse("a training raw-array map that does not exactly cover model.x"))?;
    if training_envelope.schema_version
        != dag_ml_core::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V1
        || training_envelope.predict_cohort.is_some()
        || training_envelope.coordinator_relations.as_ref() != Some(relations)
        || training_dataset.sample_ids != fold_set.sample_ids
        || training_dataset.target_names != output.target_names
        || training_dataset
            .y
            .as_ref()
            .is_none_or(|target| target.cols != 1)
    {
        return Err(refuse("a non-V1 target-bound raw training cohort, non-explicit training IDs, or multiple targets"));
    }
    if training_dataset.x.cols != predict_dataset.x.cols {
        return Err(refuse(
            "train/PREDICT X feature-width compatibility before native execution",
        ));
    }
    dag_ml_core::validate_data_binding_envelope(binding, training_envelope)?;
    training_influence.validate_for_projection(projection, request, relations)?;
    let strict_raw_relation = |record: &dag_ml_core::SampleRelation| {
        record.unit_level == EntityUnitLevel::Observation
            && record.unit_id.is_none()
            && record.source_id.is_none()
            && record.rep_id.is_none()
            && record.target_id.is_none()
            && record.group_id.is_none()
            && record.origin_sample_id.is_none()
            && record.derived_unit_id.is_none()
            && record.component_observation_ids.is_empty()
            && record.sample_influence_weight.is_none()
            && record.quality_flag.is_none()
            && !record.is_augmented
            && !record.excluded
            && record.metadata.is_empty()
            && record.tags.is_empty()
    };
    let relation_sample_ids = relations
        .records
        .iter()
        .map(|record| record.sample_id.clone())
        .collect::<BTreeSet<_>>();
    if relations.records.len() != fold_set.sample_ids.len()
        || relation_sample_ids != fold_set.sample_ids.iter().cloned().collect()
        || relations.records.iter().any(|record| !strict_raw_relation(record))
        || training_influence
        .entries
        .iter()
        .any(|entry| !entry.group_ids.is_empty() || !entry.origin_sample_ids.is_empty())
    {
        return Err(refuse(
            "non-raw relations, groups, metadata, augmentation, exclusions, or weighted training influence",
        ));
    }

    let cohort = dag_ml_core::require_terminal_predict_cohort(predict_envelope)?;
    if cohort.role != dag_ml_core::PredictCohortRole::Inference
        || cohort.target_content_fingerprint.is_some()
        || predict_envelope.target_content_fingerprint.is_some()
        || predict_envelope.coordinator_relations.as_ref() != Some(relations)
        || cohort.target_names != output.target_names
        || predict_dataset.target_names != output.target_names
        || predict_dataset.sample_ids != cohort.physical_sample_ids
        || predict_dataset.y.is_some()
    {
        return Err(refuse("a target-bearing or non-inference PREDICT cohort, non-matching IDs, or multiple targets"));
    }
    if cohort.origin_sample_ids != cohort.physical_sample_ids
        || cohort.relations.records.len() != cohort.physical_sample_ids.len()
        || cohort
            .relations
            .records
            .iter()
            .any(|record| !strict_raw_relation(record))
    {
        return Err(refuse(
            "PREDICT cohort relation metadata, groups, origins, or augmented rows",
        ));
    }
    let actual_predict_fingerprint =
        dag_ml_core::methods_pls_predict_feature_content_fingerprint(&predict_dataset.x)?;
    if predict_envelope.data_content_fingerprint.as_deref()
        != Some(actual_predict_fingerprint.as_str())
        || cohort.data_content_fingerprint != actual_predict_fingerprint
    {
        return Err(refuse(
            "a PREDICT raw-array content fingerprint that is not exact",
        ));
    }

    dag_ml_core::validate_terminal_prediction_preflight(
        &projection.plan,
        predict_envelope,
        selector,
    )?;
    Ok(())
}

#[cfg(feature = "methods-optimizer")]
fn operator_is_strict_pls(operator: Option<&serde_json::Value>) -> bool {
    let Some(operator) = operator else {
        return false;
    };
    let label = operator.as_str().map(str::to_string).or_else(|| {
        operator
            .as_object()
            .and_then(|value| value.get("type").or_else(|| value.get("class")))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    });
    matches!(label.as_deref(), Some("PLS") | Some("PLSRegression"))
}

#[cfg(feature = "methods-optimizer")]
fn methods_predict_dataset_from_json(
    input: MethodsPredictInputJson,
) -> dag_ml_core::Result<MethodsPlsDataset> {
    methods_dataset_from_json(
        MethodsTrainingInputJson {
            sample_ids: input.sample_ids,
            x: input.x,
            y: None,
            target_names: input.target_names,
        },
        false,
    )
}

/// Reuse the live native controller registry retained by [`TrainingResult`].
/// No model, node, or artifact callback crosses Python here: the Package V2's
/// captured N4MM bytes are hydrated into an invocation-local native handle
/// before `execute_terminal_prediction` invokes the native controller.
#[cfg(feature = "methods-optimizer")]
fn execute_attached_methods_terminal_prediction(
    training_result: &TrainingResult,
    package: &PortablePredictorPackage,
    predict_envelope: &ExternalDataPlanEnvelope,
    predict_dataset: MethodsPlsDataset,
    selector: &dag_ml_core::TerminalPredictionSelector,
    terminal_run_id: &RunId,
) -> PyResult<dag_ml_core::TerminalPredictionExecution> {
    let bindings = package
        .effective_plan
        .node_plans
        .values()
        .flat_map(|node_plan| node_plan.data_bindings.iter().cloned())
        .collect::<Vec<_>>();
    let [binding] = bindings.as_slice() else {
        return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "strict Methods terminal PREDICT requires exactly one effective data binding"
                .to_string(),
        )));
    };
    let binding_key =
        dag_ml_core::data_binding_requirement_key(&binding.node_id, &binding.input_name);
    let data_content_fingerprint =
        dag_ml_core::methods_pls_predict_feature_content_fingerprint(&predict_dataset.x)
            .map_err(py_core_error)?;
    let provider = dag_ml_core::MethodsPlsPredictDataProvider::new(
        provider_controller_id().map_err(py_core_error)?,
        bindings,
        BTreeMap::from([(binding_key.clone(), predict_envelope.clone())]),
        BTreeMap::from([(
            binding_key,
            dag_ml_core::MethodsPlsPredictInput {
                data_content_profile: dag_ml_core::METHODS_PLS_PREDICT_CONTENT_PROFILE.to_string(),
                data_content_fingerprint,
                dataset: predict_dataset,
            },
        )]),
    )
    .map_err(py_core_error)?;
    let mut context = dag_ml_core::RunContext::new(
        terminal_run_id.clone(),
        package.effective_plan.campaign.root_seed,
    );

    let guard = training_result.lock_resources()?;
    let Some(resources) = guard.as_ref() else {
        return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "strict Methods terminal PREDICT requires an attached TrainingResult".to_string(),
        )));
    };
    match &resources.data_provider {
        TrainingDataProvider::Methods(_) => {}
        TrainingDataProvider::Host(_) => {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "strict Methods terminal PREDICT requires a native Methods TrainingResult"
                    .to_string(),
            )));
        }
    }
    let artifact_store = MethodsBundlePayloadArtifactStore::new(
        &package.execution_bundle,
        &resources.controllers,
    );
    let execution = dag_ml_core::execute_terminal_prediction(
        dag_ml_core::TerminalPredictionReplay {
            plan: &package.effective_plan,
            bundle: &package.execution_bundle,
            envelope: predict_envelope,
            selector,
            controllers: &resources.controllers,
            data_provider: &provider,
            artifact_store: &artifact_store,
        },
        &mut context,
    );
    finish_methods_bundle_payload_terminal(execution, &artifact_store).map_err(py_core_error)
}

/// Native-only adapter for the terminal replay.  `TrainingResources` still
/// owns ordinary in-memory REFIT handles for generic attached replay, but a
/// portable Methods PREDICT must never consume those handles.  It instead
/// hydrates the exact N4MM bytes captured in the Package V2 bundle into a
/// fresh, one-shot handle owned by the registered native controller.
#[cfg(feature = "methods-optimizer")]
struct MethodsBundlePayloadArtifactStore<'a> {
    bundle: &'a ExecutionBundle,
    controllers: &'a RuntimeControllerRegistry,
    hydrated_handles: Mutex<Vec<(dag_ml_core::ControllerId, HandleRef)>>,
}

#[cfg(feature = "methods-optimizer")]
impl<'a> MethodsBundlePayloadArtifactStore<'a> {
    fn new(
        bundle: &'a ExecutionBundle,
        controllers: &'a RuntimeControllerRegistry,
    ) -> Self {
        Self {
            bundle,
            controllers,
            hydrated_handles: Mutex::new(Vec::new()),
        }
    }

    fn release_hydrated_handles(&self) -> dag_ml_core::Result<()> {
        let handles = {
            let mut handles = self.hydrated_handles.lock().map_err(|_| {
                dag_ml_core::DagMlError::RuntimeValidation(
                    "strict Methods terminal hydrated-handle registry lock poisoned".to_string(),
                )
            })?;
            std::mem::take(&mut *handles)
        };
        let mut failures = Vec::new();
        for (controller_id, handle) in handles.into_iter().rev() {
            let release = self
                .controllers
                .get(&controller_id)
                .ok_or_else(|| {
                    dag_ml_core::DagMlError::RuntimeValidation(format!(
                        "strict Methods terminal hydrated artifact owner `{controller_id}` is no longer registered"
                    ))
                })
                .and_then(|controller| controller.release_hydrated_artifact_payload(&handle));
            if let Err(error) = release {
                failures.push(error.to_string());
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "strict Methods terminal failed to release hydrated artifact handles: {}",
                failures.join("; ")
            )))
        }
    }
}

#[cfg(feature = "methods-optimizer")]
impl RuntimeArtifactStore for MethodsBundlePayloadArtifactStore<'_> {
    fn materialize(
        &self,
        request: &ArtifactMaterializationRequest,
    ) -> dag_ml_core::Result<HandleRef> {
        let Some(payload) = self.bundle.raw_artifact_payloads.get(&request.artifact.id) else {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "strict Methods terminal Package V2 bundle `{}` has no raw N4MM payload for refit artifact `{}`",
                self.bundle.bundle_id, request.artifact.id
            )));
        };
        let controller = self.controllers.get(&request.controller_id).ok_or_else(|| {
            dag_ml_core::DagMlError::RuntimeValidation(format!(
                "strict Methods terminal bundle `{}` has no registered controller `{}` to hydrate raw artifact `{}`",
                self.bundle.bundle_id, request.controller_id, request.artifact.id
            ))
        })?;
        let handle = controller.hydrate_artifact_payload(request, payload)?;
        self.hydrated_handles
            .lock()
            .map_err(|_| {
                dag_ml_core::DagMlError::RuntimeValidation(
                    "strict Methods terminal hydrated-handle registry lock poisoned".to_string(),
                )
            })?
            .push((request.controller_id.clone(), handle.clone()));
        Ok(handle)
    }
}

#[cfg(feature = "methods-optimizer")]
fn finish_methods_bundle_payload_terminal<T>(
    execution: dag_ml_core::Result<T>,
    artifact_store: &MethodsBundlePayloadArtifactStore<'_>,
) -> dag_ml_core::Result<T> {
    let cleanup = artifact_store.release_hydrated_handles();
    match (execution, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => {
            Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "{error}; strict Methods terminal hydration cleanup also failed: {cleanup_error}"
            )))
        }
    }
}

/// Execute one fresh, target-bound full refit from a portable Methods Package
/// V2 and return its durable Package V3 child.
///
/// This is deliberately not a replay: the caller supplies a newly signed
/// training request, its relation/influence evidence, and full target cohort
/// inputs.  The core derives the only accepted recipe from the V2 parent,
/// runs exactly `REFIT` for its selected variant, and writes the V3 child from
/// the fresh execution evidence.  No source score, cache, artifact handle, or
/// training cohort is reused.
#[pyfunction]
#[pyo3(signature = (
    source_package_json,
    target_request_json,
    data_envelopes_json,
    relations_json,
    training_influence_json,
    methods_inputs_json,
    methods_library_path,
    recipe_id,
    package_id,
    outcome_id,
    run_id,
    bundle_id
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_methods_portable_full_refit_json(
    py: Python<'_>,
    source_package_json: &str,
    target_request_json: &str,
    data_envelopes_json: &str,
    relations_json: &str,
    training_influence_json: &str,
    methods_inputs_json: &str,
    methods_library_path: &str,
    recipe_id: &str,
    package_id: &str,
    outcome_id: &str,
    run_id: &str,
    bundle_id: &str,
) -> PyResult<String> {
    #[cfg(not(feature = "methods-optimizer"))]
    {
        let _ = (
            py,
            source_package_json,
            target_request_json,
            data_envelopes_json,
            relations_json,
            training_influence_json,
            methods_inputs_json,
            methods_library_path,
            recipe_id,
            package_id,
            outcome_id,
            run_id,
            bundle_id,
        );
        Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "Methods full refit support is absent from this dag-ml binding; install a wheel rebuilt with the `methods-optimizer` feature".to_string(),
        )))
    }
    #[cfg(feature = "methods-optimizer")]
    {
        let source_package =
            PortablePredictorPackage::from_json(source_package_json).map_err(py_core_error)?;
        let recipe = PortableRefitRecipe::derive_from_package(&source_package, recipe_id)
            .map_err(py_core_error)?;
        let target_request =
            TrainingRequest::from_json(target_request_json).map_err(py_core_error)?;
        let projection = target_request.project().map_err(py_core_error)?;
        let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
            data_envelopes_json,
            "native Methods full refit data envelope map",
        )?;
        for envelope in envelopes.values() {
            envelope.validate().map_err(py_core_error)?;
        }
        let relations = parse_strict_json::<SampleRelationSet>(
            relations_json,
            "native Methods full refit sample relations",
        )?;
        relations.validate().map_err(py_core_error)?;
        let training_influence = parse_strict_json::<TrainingInfluenceManifest>(
            training_influence_json,
            "native Methods full refit influence manifest",
        )?;
        training_influence
            .validate_for_projection(&projection, &target_request, &relations)
            .map_err(py_core_error)?;
        // The parent package owns the selected topology and parameter values;
        // the fresh request contributes only its independently attested data
        // bindings/fold universe.  Core derives the exact combined plan before
        // any provider object is constructed.
        let target_plan =
            derive_portable_full_refit_target_plan(&recipe, &source_package, &target_request)
                .map_err(py_core_error)?;
        let bindings = target_plan
            .node_plans
            .values()
            .flat_map(|node_plan| node_plan.data_bindings.iter().cloned())
            .collect::<Vec<DataBinding>>();
        if source_package
            .effective_plan
            .node_plans
            .values()
            .any(|node_plan| !is_native_methods_model_controller(&node_plan.controller_id))
        {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods full refit requires every executable node to use controller:methods.pls or controller:methods.ridge; host controller fallback is forbidden".to_string(),
            )));
        }
        let raw_inputs = parse_strict_json::<BTreeMap<String, MethodsTrainingInputJson>>(
            methods_inputs_json,
            "native Methods full refit input map",
        )?;
        let inputs = raw_inputs
            .into_iter()
            .map(|(key, input)| Ok((key, methods_dataset_from_json(input, true)?)))
            .collect::<dag_ml_core::Result<BTreeMap<_, _>>>()
            .map_err(py_core_error)?;
        let data_provider = TrainingDataProvider::Methods(
            PyMethodsPlsTrainingProvider::new(bindings, envelopes, inputs)
                .map_err(py_core_error)?,
        );
        let runtime =
            dag_ml_core::MethodsRuntime::configure(methods_library_path).map_err(|error| {
                py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                    error.to_string(),
                ))
            })?;
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(dag_ml_core::MethodsPlsController::new(
                runtime.clone(),
            )))
            .map_err(py_core_error)?;
        controllers
            .register(Box::new(dag_ml_core::MethodsRidgeController::new(runtime)))
            .map_err(py_core_error)?;
        let run_id = RunId::new(run_id).map_err(py_core_error)?;
        let bundle_id = BundleId::new(bundle_id).map_err(py_core_error)?;
        let package_id = package_id.to_string();
        let outcome_id = outcome_id.to_string();

        let package = py
            .detach(move || {
                let execution = execute_portable_full_refit(PortableFullRefitExecutionInput {
                    recipe: &recipe,
                    source_package: &source_package,
                    target_plan: &target_plan,
                    target_training_request: &target_request,
                    target_training_request_fingerprint: target_request.request_fingerprint.clone(),
                    target_data_identities: &target_request.data_identities,
                    target_training_influence: &training_influence,
                    run_id,
                    controllers: &controllers,
                    data_provider: &data_provider,
                })?;
                build_portable_refit_package_v3(PortableRefitPackageV3BuildInput {
                    package_id,
                    outcome_id,
                    bundle_id,
                    recipe: &recipe,
                    source_package: &source_package,
                    target_plan: &target_plan,
                    target_training_request: &target_request,
                    target_data_identities: &target_request.data_identities,
                    target_training_influence: &training_influence,
                    execution: &execution,
                })
            })
            .map_err(py_core_error)?;
        serialize_json(&package)
    }
}

/// Register the native PLS/Ridge controllers and, when attested by the campaign,
/// its controller-owned Methods HPO companion.  The scheduler creates the
/// thread-affine optimizer session later from the complete training context;
/// this binding only establishes the exact controller identities before any
/// operation can reach the provider.
#[cfg(feature = "methods-optimizer")]
fn register_methods_training_controllers(
    projection: &dag_ml_core::TrainingContractProjection,
    runtime: dag_ml_core::MethodsRuntime,
    controllers: &mut RuntimeControllerRegistry,
) -> dag_ml_core::Result<()> {
    let Some(hpo_controller_id) = methods_hpo_controller_id(&projection.plan.campaign.metadata)?
    else {
        controllers.register(Box::new(dag_ml_core::MethodsPlsController::new(
            runtime.clone(),
        )))?;
        controllers.register(Box::new(dag_ml_core::MethodsRidgeController::new(runtime)))?;
        return Ok(());
    };
    dag_ml_core::register_methods_runtime_controllers(controllers, hpo_controller_id, runtime)
}

/// Return whether an executable node is owned by one of the two native Methods
/// model controllers that the portable training/refit lane registers locally.
#[cfg(feature = "methods-optimizer")]
fn is_native_methods_model_controller(controller_id: &dag_ml_core::ControllerId) -> bool {
    matches!(
        controller_id.as_str(),
        dag_ml_core::METHODS_PLS_CONTROLLER_ID | dag_ml_core::METHODS_RIDGE_CONTROLLER_ID
    )
}

/// Extract only the controller identity that must be registered locally.
/// Descriptor semantics themselves remain owned by `dag-ml-core` training
/// preflight; accepting a partial descriptor here would otherwise turn a
/// malformed request into an unrelated missing-controller error.
#[cfg(feature = "methods-optimizer")]
fn methods_hpo_controller_id(
    metadata: &BTreeMap<String, serde_json::Value>,
) -> dag_ml_core::Result<Option<dag_ml_core::ControllerId>> {
    let Some(value) = metadata.get("methods_hpo_operation") else {
        return Ok(None);
    };
    let controller_id = value
        .as_object()
        .and_then(|descriptor| descriptor.get("study"))
        .and_then(serde_json::Value::as_object)
        .and_then(|study| study.get("controller_id"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods HPO descriptor must declare study.controller_id before controller registration"
                    .to_string(),
            )
        })?;
    dag_ml_core::ControllerId::new(controller_id)
        .map(Some)
        .map_err(|error| {
            dag_ml_core::DagMlError::RuntimeValidation(format!(
                "native Methods HPO descriptor controller id is invalid: {error}"
            ))
        })
}

#[cfg(feature = "methods-optimizer")]
fn methods_dataset_from_json(
    input: MethodsTrainingInputJson,
    require_targets: bool,
) -> dag_ml_core::Result<MethodsPlsDataset> {
    let rows_to_matrix =
        |rows: Vec<Vec<f64>>, label: &str| -> dag_ml_core::Result<MethodsPlsMatrix> {
            let row_count = rows.len();
            let columns = rows.first().map(Vec::len).unwrap_or(0);
            if row_count == 0 || columns == 0 || rows.iter().any(|row| row.len() != columns) {
                return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                    "native Methods training input `{label}` is not a non-empty rectangular matrix"
                )));
            }
            let values = rows.into_iter().flatten().collect::<Vec<_>>();
            Ok(MethodsPlsMatrix {
                values,
                rows: row_count,
                cols: columns,
            })
        };
    let sample_ids = input
        .sample_ids
        .into_iter()
        .map(SampleId::new)
        .collect::<dag_ml_core::Result<Vec<_>>>()?;
    let dataset = MethodsPlsDataset {
        sample_ids,
        x: rows_to_matrix(input.x, "x")?,
        y: input.y.map(|rows| rows_to_matrix(rows, "y")).transpose()?,
        target_names: input.target_names,
    };
    dataset.validate("native Methods input", require_targets)?;
    Ok(dataset)
}

/// Replay a raw N4MM Package V2 through the registered native Methods PLS controller.
///
/// This is deliberately distinct from the generic callback replay API.  It
/// accepts only typed numeric PREDICT views and never constructs a Python
/// operator or host-side artifact callback.
#[pyfunction]
#[pyo3(signature = (
    package_json,
    request_json,
    data_envelopes_json,
    methods_inputs_json,
    methods_library_path,
    outcome_id,
    run_id,
    warnings_json = "[]",
    diagnostics_json = "{}"
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_loaded_methods_predictor_replay_json(
    py: Python<'_>,
    package_json: &str,
    request_json: &str,
    data_envelopes_json: &str,
    methods_inputs_json: &str,
    methods_library_path: &str,
    outcome_id: &str,
    run_id: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<String> {
    #[cfg(not(feature = "methods-optimizer"))]
    {
        let _ = (
            py,
            package_json,
            request_json,
            data_envelopes_json,
            methods_inputs_json,
            methods_library_path,
            outcome_id,
            run_id,
            warnings_json,
            diagnostics_json,
        );
        Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "Methods replay support is absent from this dag-ml binding; install a wheel rebuilt with the `methods-optimizer` feature".to_string(),
        )))
    }
    #[cfg(feature = "methods-optimizer")]
    {
        let package = PortablePredictorPackage::from_json(package_json).map_err(py_core_error)?;
        let request = TrainingReplayRequest::from_json(request_json).map_err(py_core_error)?;
        let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
            data_envelopes_json,
            "native Methods replay data envelope map",
        )?;
        for envelope in envelopes.values() {
            envelope.validate().map_err(py_core_error)?;
        }
        let raw_inputs = parse_strict_json::<BTreeMap<String, MethodsTrainingInputJson>>(
            methods_inputs_json,
            "native Methods replay input map",
        )?;
        let inputs = raw_inputs
            .into_iter()
            .map(|(key, input)| Ok((key, methods_dataset_from_json(input, false)?)))
            .collect::<dag_ml_core::Result<BTreeMap<_, _>>>()
            .map_err(py_core_error)?;
        let runtime =
            dag_ml_core::MethodsRuntime::configure(methods_library_path).map_err(|error| {
                py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                    error.to_string(),
                ))
            })?;
        let warnings =
            parse_strict_json::<Vec<String>>(warnings_json, "native Methods replay warnings")?;
        let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
            diagnostics_json,
            "native Methods replay diagnostics",
        )?;
        let outcome_id = outcome_id.to_string();
        let run_id = RunId::new(run_id).map_err(py_core_error)?;
        let outcome = py
            .detach(move || {
                dag_ml_core::execute_loaded_methods_predictor_replay(
                    MethodsPortablePredictorReplayInput {
                        package: &package,
                        request: &request,
                        data_envelopes: &envelopes,
                        methods_inputs: &inputs,
                        runtime,
                        outcome_id,
                        run_id,
                        warnings,
                        diagnostics,
                    },
                )
            })
            .map_err(py_core_error)?;
        serialize_json(&outcome)
    }
}

/// Replay a detached Package V3 full-refit child through the native Methods
/// controller registered for this invocation.
///
/// V3 deliberately has a separate wire family from Package V2: it represents
/// a fresh target-cohort refit, not the original CV/SELECT outcome.  The
/// binding therefore parses the strict child package itself and delegates only
/// to Core's V3 scheduler-owned replay entry point.  Its raw N4MM artifact is
/// hydrated and released within this call; Python never supplies a handle or
/// callback.
#[pyfunction]
#[pyo3(signature = (
    package_json,
    request_json,
    data_envelopes_json,
    methods_inputs_json,
    methods_library_path,
    outcome_id,
    run_id,
    warnings_json = "[]",
    diagnostics_json = "{}"
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_loaded_methods_portable_refit_replay_v3_json(
    py: Python<'_>,
    package_json: &str,
    request_json: &str,
    data_envelopes_json: &str,
    methods_inputs_json: &str,
    methods_library_path: &str,
    outcome_id: &str,
    run_id: &str,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<String> {
    #[cfg(not(feature = "methods-optimizer"))]
    {
        let _ = (
            py,
            package_json,
            request_json,
            data_envelopes_json,
            methods_inputs_json,
            methods_library_path,
            outcome_id,
            run_id,
            warnings_json,
            diagnostics_json,
        );
        Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "Methods V3 refit replay support is absent from this dag-ml binding; install a wheel rebuilt with the `methods-optimizer` feature".to_string(),
        )))
    }
    #[cfg(feature = "methods-optimizer")]
    {
        let package = PortableRefitPackageV3::from_json(package_json).map_err(py_core_error)?;
        let request = TrainingReplayRequest::from_json(request_json).map_err(py_core_error)?;
        let envelopes = parse_strict_json::<BTreeMap<String, ExternalDataPlanEnvelope>>(
            data_envelopes_json,
            "native Methods V3 refit replay data envelope map",
        )?;
        for envelope in envelopes.values() {
            envelope.validate().map_err(py_core_error)?;
        }
        let raw_inputs = parse_strict_json::<BTreeMap<String, MethodsTrainingInputJson>>(
            methods_inputs_json,
            "native Methods V3 refit replay input map",
        )?;
        let inputs = raw_inputs
            .into_iter()
            .map(|(key, input)| Ok((key, methods_dataset_from_json(input, false)?)))
            .collect::<dag_ml_core::Result<BTreeMap<_, _>>>()
            .map_err(py_core_error)?;
        let runtime =
            dag_ml_core::MethodsRuntime::configure(methods_library_path).map_err(|error| {
                py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                    error.to_string(),
                ))
            })?;
        let warnings = parse_strict_json::<Vec<String>>(
            warnings_json,
            "native Methods V3 refit replay warnings",
        )?;
        let diagnostics = parse_strict_json::<BTreeMap<String, serde_json::Value>>(
            diagnostics_json,
            "native Methods V3 refit replay diagnostics",
        )?;
        let outcome_id = outcome_id.to_string();
        let run_id = RunId::new(run_id).map_err(py_core_error)?;
        let outcome = py
            .detach(move || {
                execute_loaded_methods_portable_refit_replay_v3(MethodsPortableRefitReplayInputV3 {
                    package: &package,
                    request: &request,
                    data_envelopes: &envelopes,
                    methods_inputs: &inputs,
                    runtime,
                    supplemental_controllers: RuntimeControllerRegistry::new(),
                    outcome_id,
                    run_id,
                    warnings,
                    diagnostics,
                })
            })
            .map_err(py_core_error)?;
        serialize_json(&outcome)
    }
}

/// Execute stateless PREDICT/EXPLAIN replay from a loaded portable predictor package.
///
/// `artifact_handles_json` is a host-side sidecar map keyed by artifact id. The
/// portable package remains handle-free; this binding only joins the package to
/// the explicit handles supplied by the host for this process.  A package with
/// durable raw artifacts may instead provide `artifact_callback`: it receives
/// strict `{operation, request, payload}` hydration calls and
/// `{operation, handle}` release calls. The callback is opt-in, so a raw
/// portable package fails closed rather than being silently treated as a host
/// sidecar package.
#[pyfunction]
#[pyo3(signature = (
    package_json,
    request_json,
    data_envelopes_json,
    artifact_handles_json,
    op_callback,
    outcome_id,
    run_id,
    artifact_callback = None,
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
    artifact_callback: Option<Py<PyAny>>,
    warnings_json: &str,
    diagnostics_json: &str,
) -> PyResult<String> {
    if !op_callback.bind(py).is_callable() {
        return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
            "loaded predictor replay op_callback must be callable".to_string(),
        )));
    }
    if let Some(callback) = artifact_callback.as_ref() {
        if !callback.bind(py).is_callable() {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "loaded predictor replay artifact_callback must be callable".to_string(),
            )));
        }
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
    let controllers = build_runtime_controllers_with_artifact_callback(
        py,
        &predictor.package().effective_plan,
        &op_callback,
        artifact_callback.as_ref(),
    )
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

    #[cfg(feature = "methods-optimizer")]
    use dag_ml_core::{
        methods_pls_predict_feature_content_fingerprint, MethodsPlsDataset, MethodsPlsMatrix,
        NodeId, PortablePredictorPackage, PredictCohort, PredictCohortRole,
        TerminalPredictionSelector, TrainingReplayRequest,
        EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2, PORTABLE_PREDICTOR_PACKAGE_SCHEMA_VERSION,
        TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,
    };
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

    #[cfg(feature = "methods-optimizer")]
    #[test]
    fn portable_methods_model_lane_allows_only_pls_and_ridge_controllers() {
        for controller in [
            dag_ml_core::METHODS_PLS_CONTROLLER_ID,
            dag_ml_core::METHODS_RIDGE_CONTROLLER_ID,
        ] {
            assert!(is_native_methods_model_controller(
                &dag_ml_core::ControllerId::new(controller).unwrap()
            ));
        }
        assert!(!is_native_methods_model_controller(
            &dag_ml_core::ControllerId::new("controller:test.host").unwrap()
        ));
    }

    #[cfg(not(feature = "methods-optimizer"))]
    #[test]
    fn methods_portable_full_refit_fails_closed_without_native_methods_support() {
        Python::initialize();
        Python::attach(|py| {
            let error = execute_methods_portable_full_refit_json(
                py,
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "/missing/libn4m.so",
                "recipe:refit",
                "package:refit",
                "outcome:refit",
                "run:refit",
                "bundle:refit",
            )
            .expect_err("portable builds must not emulate a native full refit");
            assert!(error
                .to_string()
                .contains("Methods full refit support is absent"));
        });
    }

    #[cfg(not(feature = "methods-optimizer"))]
    #[test]
    fn methods_portable_refit_replay_fails_closed_without_native_methods_support() {
        Python::initialize();
        Python::attach(|py| {
            let error = execute_loaded_methods_portable_refit_replay_v3_json(
                py,
                "{}",
                "{}",
                "{}",
                "{}",
                "/missing/libn4m.so",
                "outcome:refit.replay",
                "run:refit.replay",
                "[]",
                "{}",
            )
            .expect_err("portable builds must not emulate a native V3 replay");
            assert!(error
                .to_string()
                .contains("Methods V3 refit replay support is absent"));
        });
    }

    #[cfg(feature = "methods-optimizer")]
    #[test]
    fn methods_training_views_follow_scheduler_identities_not_host_positions() {
        let dataset = MethodsPlsDataset {
            sample_ids: vec![
                SampleId::new("sample:a").unwrap(),
                SampleId::new("sample:b").unwrap(),
                SampleId::new("sample:c").unwrap(),
            ],
            x: MethodsPlsMatrix {
                values: vec![1.0, 2.0, 3.0],
                rows: 3,
                cols: 1,
            },
            y: Some(MethodsPlsMatrix {
                values: vec![10.0, 20.0, 30.0],
                rows: 3,
                cols: 1,
            }),
            target_names: vec!["protein".to_string()],
        };
        let selected = PyMethodsPlsTrainingProvider::dataset_for_view(
            &dataset,
            &[
                SampleId::new("sample:c").unwrap(),
                SampleId::new("sample:a").unwrap(),
            ],
            true,
            "test",
        )
        .unwrap();
        assert_eq!(
            selected.sample_ids,
            vec![
                SampleId::new("sample:c").unwrap(),
                SampleId::new("sample:a").unwrap(),
            ]
        );
        assert_eq!(selected.x.values, vec![3.0, 1.0]);
        assert_eq!(selected.y.unwrap().values, vec![30.0, 10.0]);

        let error = PyMethodsPlsTrainingProvider::dataset_for_view(
            &dataset,
            &[SampleId::new("sample:absent").unwrap()],
            true,
            "test",
        )
        .unwrap_err();
        assert!(error.to_string().contains("absent from its attested input"));
    }

    #[cfg(not(feature = "methods-optimizer"))]
    #[test]
    fn strict_methods_terminal_facade_fails_closed_without_native_support() {
        Python::initialize();
        Python::attach(|py| {
            let error = execute_methods_cv_refit_terminal_predict_json(
                py,
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "/missing/libn4m.so",
                "outcome:strict.methods",
                "run:strict.methods",
                "bundle:strict.methods",
                "package:strict.methods",
                "{}",
                "[]",
                "{}",
            )
            .expect_err("a non-Methods build must not fall back to a Python callback path");
            assert!(error
                .to_string()
                .contains("strict Methods terminal prediction support is absent"));
        });
    }

    #[test]
    fn strict_methods_terminal_public_native_types_reject_object_new_and_rebinding() {
        Python::initialize();
        Python::attach(|py| {
            let object = py.import("builtins").unwrap().getattr("object").unwrap();
            let object_new = object.getattr("__new__").unwrap();
            assert!(
                object_new
                    .call1((py.get_type::<MethodsTerminalPredictionReceipt>(),))
                    .is_err(),
                "object.__new__ must not construct a public terminal receipt"
            );
            assert!(
                object_new
                    .call1((py.get_type::<MethodsTerminalPredictionResult>(),))
                    .is_err(),
                "object.__new__ must not construct a public terminal result"
            );

            let receipt = Py::new(
                py,
                MethodsTerminalPredictionReceipt {
                    json: r#"{"schema_version":1}"#.to_string(),
                    terminal_run_id: "run:strict.methods:methods-terminal-predict".to_string(),
                    receipt_fingerprint: "f".repeat(64),
                },
            )
            .unwrap();
            assert!(
                object
                    .getattr("__setattr__")
                    .unwrap()
                    .call1((
                        receipt.bind(py),
                        "terminal_run_id",
                        "run:forged:methods-terminal-predict",
                    ))
                    .is_err(),
                "object.__setattr__ must not rebind a public terminal receipt"
            );
        });
    }

    #[cfg(feature = "methods-optimizer")]
    #[test]
    fn strict_methods_terminal_preflight_rejects_v2_ids_port_and_labels_before_runtime() {
        let fixture = strict_methods_terminal_fixture();
        Python::initialize();
        Python::attach(|_| {
            parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect("the bounded one-node PLS fixture preflights without libn4m");

            let mut bad_ids: serde_json::Value =
                serde_json::from_str(&fixture.predict_input_json).unwrap();
            bad_ids["sample_ids"][0] = serde_json::json!("sample:substituted");
            let error = parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &serde_json::to_string(&bad_ids).unwrap(),
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect_err("PREDICT rows must carry the exact V2 cohort IDs");
            assert!(error.to_string().contains("non-matching IDs"));

            let error = parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
                r#"{"node_id":"model:base","port":"probability"}"#,
                "[]",
                "{}",
            )
            .expect_err("only the native PLS oof terminal port is accepted");
            assert!(error.to_string().contains("terminal port"));

            let mut v1: serde_json::Value =
                serde_json::from_str(&fixture.predict_envelope_json).unwrap();
            v1["schema_version"] = serde_json::json!(1);
            v1.as_object_mut().unwrap().remove("predict_cohort");
            let error = parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &serde_json::to_string(&v1).unwrap(),
                &fixture.predict_input_json,
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect_err("terminal PREDICT requires the separately attested V2 cohort");
            assert!(error
                .to_string()
                .contains("requires external data-plan envelope V2"));

            let mut labels: serde_json::Value =
                serde_json::from_str(&fixture.predict_input_json).unwrap();
            labels["y"] = serde_json::json!([[1.0], [2.0]]);
            let error = parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &serde_json::to_string(&labels).unwrap(),
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect_err("the X-only boundary must reject a y payload");
            assert!(error.to_string().contains("unknown field `y`"));

            let error = parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
                &fixture.selector_json,
                r#"["caller metadata is outside the strict facade"]"#,
                "{}",
            )
            .expect_err("caller diagnostics must not widen the closed facade");
            assert!(error.to_string().contains("warnings or diagnostics metadata"));

            let mut external_oof: TrainingRequest =
                serde_json::from_str(&fixture.request_json).unwrap();
            external_oof.options.selection.stacking_fit_contract =
                Some(dag_ml_core::StackingFitContract {
                    meta_training_features: dag_ml_core::MetaTrainingFeatures::Oof,
                    inference_features: dag_ml_core::InferenceFeatures::RefitBasePredictions,
                    selection_protocol: dag_ml_core::SelectionProtocol::Nested,
                    meta_row_domain: dag_ml_core::MetaRowDomain::Sample,
                    final_reduction_id: None,
                    unsafe_allow_reuse_oof: false,
                });
            external_oof.request_fingerprint = "0".repeat(64);
            external_oof.request_fingerprint = external_oof.compute_fingerprint().unwrap();
            let error = parse_strict_methods_terminal_facade_inputs(
                &serde_json::to_string(&external_oof).unwrap(),
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect_err("external OOF stacking requirements must not enter native CV");
            assert!(error.to_string().contains("non-internal-OOF selection"));

            let mut callback_capability: TrainingRequest =
                serde_json::from_str(&fixture.request_json).unwrap();
            callback_capability.controller_manifests[0]
                .capabilities
                .insert(ControllerCapability::NeedsPythonGil);
            callback_capability.request_fingerprint = "0".repeat(64);
            callback_capability.request_fingerprint =
                callback_capability.compute_fingerprint().unwrap();
            let error = parse_strict_methods_terminal_facade_inputs(
                &serde_json::to_string(&callback_capability).unwrap(),
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect_err("a callback/GIL controller capability must not enter native CV");
            assert!(error.to_string().contains("callback/GIL"));

            let mut cohort_metadata = fixture.predict_envelope.clone();
            let cohort = cohort_metadata.predict_cohort.as_mut().unwrap();
            cohort.relations.records[0]
                .metadata
                .insert("site".to_string(), serde_json::json!("lab-a"));
            cohort.relation_fingerprint = cohort.relations.fingerprint().unwrap();
            cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
            cohort_metadata.validate().unwrap();
            let error = parse_strict_methods_terminal_facade_inputs(
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &serde_json::to_string(&cohort_metadata).unwrap(),
                &fixture.predict_input_json,
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect_err("a valid V2 cohort must still reject metadata before native execution");
            assert!(error.to_string().contains("PREDICT cohort relation metadata"));
        });
    }

    #[cfg(feature = "methods-optimizer")]
    #[test]
    fn strict_methods_terminal_identity_and_width_refuse_before_native_runtime() {
        let fixture = strict_methods_terminal_fixture();
        let missing_library = "/strict-methods-sentinel/no-libn4m.so";
        Python::initialize();
        Python::attach(|py| {
            let invoke = |outcome_id: &str,
                          run_id: &str,
                          bundle_id: &str,
                          package_id: &str,
                          predict_envelope_json: &str,
                          predict_input_json: &str| {
                execute_methods_cv_refit_terminal_predict_json(
                    py,
                    &fixture.request_json,
                    &fixture.training_envelopes_json,
                    &fixture.relations_json,
                    &fixture.influence_json,
                    &fixture.methods_inputs_json,
                    predict_envelope_json,
                    predict_input_json,
                    missing_library,
                    outcome_id,
                    run_id,
                    bundle_id,
                    package_id,
                    &fixture.selector_json,
                    "[]",
                    "{}",
                )
                .expect_err("hostile preflight input must stop before libn4m configuration")
                .to_string()
            };
            let assert_no_native = |error: &str| {
                assert!(
                    !error.contains("libn4m") && !error.contains("strict-methods-sentinel"),
                    "preflight must fail before native runtime configuration, got: {error}"
                );
            };

            let error = invoke(
                "outcome:strict.methods",
                "run/invalid",
                "bundle:strict.methods",
                "package:strict.methods",
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
            );
            assert!(error.contains("identifier"));
            assert_no_native(&error);

            let error = invoke(
                "outcome:strict.methods",
                "run:strict.methods",
                "bundle/invalid",
                "package:strict.methods",
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
            );
            assert!(error.contains("identifier"));
            assert_no_native(&error);

            let error = invoke(
                "outcome/invalid",
                "run:strict.methods",
                "bundle:strict.methods",
                "package:strict.methods",
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
            );
            assert!(error.contains("outcome_id"));
            assert_no_native(&error);

            let error = invoke(
                "outcome:strict.methods",
                "run:strict.methods",
                "bundle:strict.methods",
                "package/invalid",
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
            );
            assert!(error.contains("package_id"));
            assert_no_native(&error);

            let derived_too_long_run_id = "r".repeat(128);
            let error = invoke(
                "outcome:strict.methods",
                &derived_too_long_run_id,
                "bundle:strict.methods",
                "package:strict.methods",
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
            );
            assert!(error.contains("longer than 128 bytes"));
            assert_no_native(&error);

            let mut wide_predict_input = fixture.predict_input.clone();
            wide_predict_input["x"] = serde_json::json!([[5.0, 0.0, 9.0], [6.0, 1.0, 8.0]]);
            let wide_matrix = MethodsPlsMatrix {
                values: vec![5.0, 0.0, 9.0, 6.0, 1.0, 8.0],
                rows: 2,
                cols: 3,
            };
            let wide_fingerprint =
                methods_pls_predict_feature_content_fingerprint(&wide_matrix).unwrap();
            let mut wide_predict_envelope = fixture.predict_envelope.clone();
            wide_predict_envelope.data_content_fingerprint = Some(wide_fingerprint.clone());
            let cohort = wide_predict_envelope.predict_cohort.as_mut().unwrap();
            cohort.data_content_fingerprint = wide_fingerprint;
            cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
            wide_predict_envelope.validate().unwrap();
            let error = invoke(
                "outcome:strict.methods",
                "run:strict.methods",
                "bundle:strict.methods",
                "package:strict.methods",
                &serde_json::to_string(&wide_predict_envelope).unwrap(),
                &serde_json::to_string(&wide_predict_input).unwrap(),
            );
            assert!(error.contains("feature-width compatibility"));
            assert_no_native(&error);
        });
    }

    #[cfg(feature = "methods-optimizer")]
    #[test]
    fn strict_methods_terminal_facade_reuses_refit_artifact_and_exports_replayable_v2() {
        let library_path = match std::env::var_os("N4M_LIBRARY_PATH") {
            Some(library_path) => library_path,
            None if std::env::var_os("DAG_ML_REQUIRE_N4M_TEST").is_some() => {
                panic!(
                    "DAG_ML_REQUIRE_N4M_TEST=1 requires an explicit N4M_LIBRARY_PATH for the strict Methods terminal facade gate"
                );
            }
            // Ordinary local feature builds retain the contract test but do
            // not pretend to qualify native execution without a libn4m file.
            None => return,
        };
        let fixture = strict_methods_terminal_fixture();
        Python::initialize();
        Python::attach(|py| {
            let native_result = execute_methods_cv_refit_terminal_predict_json(
                py,
                &fixture.request_json,
                &fixture.training_envelopes_json,
                &fixture.relations_json,
                &fixture.influence_json,
                &fixture.methods_inputs_json,
                &fixture.predict_envelope_json,
                &fixture.predict_input_json,
                &library_path.to_string_lossy(),
                "outcome:strict.methods",
                "run:strict.methods",
                "bundle:strict.methods",
                "package:strict.methods",
                &fixture.selector_json,
                "[]",
                "{}",
            )
            .expect("native PLS CV/REFIT/terminal-PREDICT succeeds without a Python callback");
            let (terminal_prediction_json, sealed_receipt, package_json, training_result) = {
                let result = native_result.bind(py).borrow();
                (
                    result.terminal_prediction_json.clone(),
                    result.terminal_receipt.clone_ref(py),
                    result.portable_predictor_package_json.clone(),
                    result.training_result.clone_ref(py),
                )
            };
            let prediction: serde_json::Value =
                serde_json::from_str(&terminal_prediction_json).unwrap();
            let receipt_json = sealed_receipt.bind(py).borrow().json.clone();
            let receipt: serde_json::Value = serde_json::from_str(&receipt_json).unwrap();
            assert_eq!(prediction["partition"], "final");
            assert_eq!(
                prediction["sample_ids"],
                serde_json::json!(["sample:predict:1", "sample:predict:2"])
            );
            assert_eq!(prediction["target_names"], serde_json::json!(["protein"]));
            assert_eq!(receipt["terminal_node_id"], "model:base");
            assert_eq!(receipt["terminal_port"], "oof");
            assert_eq!(
                receipt["terminal_run_id"],
                "run:strict.methods:methods-terminal-predict"
            );
            assert_eq!(
                sealed_receipt.bind(py).borrow().terminal_run_id,
                "run:strict.methods:methods-terminal-predict"
            );
            assert_eq!(
                sealed_receipt.bind(py).borrow().receipt_fingerprint,
                receipt["receipt_fingerprint"].as_str().unwrap()
            );
            assert_eq!(receipt["refit_artifacts"].as_array().unwrap().len(), 1);

            let receipt_snapshot_value = sealed_receipt.bind(py).call_method0("to_dict").unwrap();
            let receipt_snapshot = receipt_snapshot_value
                .cast::<pyo3::types::PyDict>()
                .unwrap();
            receipt_snapshot
                .set_item("terminal_run_id", "run:forged:methods-terminal-predict")
                .unwrap();
            assert_eq!(
                sealed_receipt.bind(py).borrow().terminal_run_id,
                "run:strict.methods:methods-terminal-predict",
                "a mutable receipt snapshot is explicitly non-attesting"
            );

            let mut mutated_receipt = receipt.clone();
            mutated_receipt["terminal_run_id"] =
                serde_json::json!("run:forged:methods-terminal-predict");
            let error = parse_strict_methods_terminal_receipt_json(
                &serde_json::to_string(&mutated_receipt).unwrap(),
            )
            .expect_err("a changed terminal RunId must invalidate the sealed receipt");
            assert!(error.to_string().contains("fingerprint"));
            let mut extended_receipt = receipt.clone();
            extended_receipt["unattested"] = serde_json::json!(true);
            let error = parse_strict_methods_terminal_receipt_json(
                &serde_json::to_string(&extended_receipt).unwrap(),
            )
            .expect_err("a receipt with an extra field cannot enter the closed schema");
            assert!(error.to_string().contains("closed schema"));

            let object = py.import("builtins").unwrap().getattr("object").unwrap();
            let set_attribute = object.getattr("__setattr__").unwrap();
            let forged_receipt = Py::new(
                py,
                MethodsTerminalPredictionReceipt {
                    json: receipt_json.clone(),
                    terminal_run_id: "run:forged:methods-terminal-predict".to_string(),
                    receipt_fingerprint: "0".repeat(64),
                },
            )
            .unwrap();
            assert!(
                set_attribute
                    .call1((
                        native_result.bind(py),
                        "terminal_receipt",
                        forged_receipt.bind(py),
                    ))
                    .is_err(),
                "object.__setattr__ must not rebind a frozen native terminal result"
            );
            assert!(
                set_attribute
                    .call1((
                        sealed_receipt.bind(py),
                        "terminal_run_id",
                        "run:forged:methods-terminal-predict",
                    ))
                    .is_err(),
                "object.__setattr__ must not mutate a frozen native receipt"
            );

            let package = PortablePredictorPackage::from_json(&package_json).unwrap();
            assert_eq!(
                package.schema_version,
                PORTABLE_PREDICTOR_PACKAGE_SCHEMA_VERSION
            );
            assert!(package.conformal_calibration.is_none());
            assert!(package.conformal_calibration_replay.is_none());
            assert!(package.artifact_bindings.iter().all(|binding| {
                binding.load_mode == dag_ml_core::ArtifactLoadMode::NativePortable
            }));
            let refit_artifact_ids = package
                .execution_bundle
                .refit_artifacts
                .iter()
                .map(|record| record.artifact.id.clone())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                package
                    .execution_bundle
                    .raw_artifact_payloads
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>(),
                refit_artifact_ids,
                "the terminal facade may hydrate only the Package V2 refit artifact"
            );
            assert!(package
                .execution_bundle
                .raw_artifact_payloads
                .values()
                .all(|payload| !payload.is_empty()));

            let source_outcome_fingerprint = training_result
                .bind(py)
                .borrow()
                .outcome
                .outcome_fingerprint
                .clone();
            assert!(training_result.bind(py).borrow().is_attached().unwrap());
            assert!(training_result.bind(py).borrow().detach().unwrap());
            assert!(!training_result.bind(py).borrow().is_attached().unwrap());

            let mut replay_request = TrainingReplayRequest {
                schema_version: TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,
                request_id: "replay:strict.methods".to_string(),
                source_outcome_fingerprint,
                phase: Phase::Predict,
                data_envelope_keys: vec!["model:base.x".to_string()],
                output_binding_ids: package
                    .output_bindings
                    .iter()
                    .map(|binding| binding.binding_id.clone())
                    .collect(),
                request_fingerprint: "0".repeat(64),
            };
            replay_request.request_fingerprint = replay_request.compute_fingerprint().unwrap();
            let replay = execute_loaded_methods_predictor_replay_json(
                py,
                &package_json,
                &serde_json::to_string(&replay_request).unwrap(),
                &serde_json::to_string(&BTreeMap::from([(
                    "model:base.x".to_string(),
                    fixture.predict_envelope,
                )]))
                .unwrap(),
                &serde_json::to_string(&serde_json::json!({
                    "model:base.x": fixture.predict_input,
                }))
                .unwrap(),
                &library_path.to_string_lossy(),
                "outcome:strict.methods.replay",
                "run:strict.methods.replay",
                "[]",
                "{}",
            )
            .expect("the exported native Package V2 replays after TrainingResult cleanup");
            let replay =
                serde_json::from_str::<dag_ml_core::TrainingReplayOutcome>(&replay).unwrap();
            assert_eq!(replay.phase, Phase::Predict);
            assert!(!replay.outputs.is_empty());
        });
    }

    #[cfg(feature = "methods-optimizer")]
    #[test]
    fn methods_hpo_controller_identity_is_required_and_canonical_before_registration() {
        let absent = BTreeMap::new();
        assert!(methods_hpo_controller_id(&absent).unwrap().is_none());

        let valid = BTreeMap::from([(
            "methods_hpo_operation".to_string(),
            serde_json::json!({"study": {"controller_id": "controller:methods.hpo"}}),
        )]);
        assert_eq!(
            methods_hpo_controller_id(&valid).unwrap().unwrap().as_str(),
            "controller:methods.hpo"
        );

        for descriptor in [
            serde_json::json!({}),
            serde_json::json!({"study": {}}),
            serde_json::json!({"study": {"controller_id": "not canonical"}}),
        ] {
            let metadata = BTreeMap::from([("methods_hpo_operation".to_string(), descriptor)]);
            assert!(
                methods_hpo_controller_id(&metadata).is_err(),
                "partial or non-canonical Methods HPO descriptor must fail before registration"
            );
        }
    }

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

    #[cfg(feature = "methods-optimizer")]
    struct StrictMethodsTerminalFixture {
        request_json: String,
        training_envelopes_json: String,
        relations_json: String,
        influence_json: String,
        methods_inputs_json: String,
        predict_envelope_json: String,
        predict_input_json: String,
        selector_json: String,
        predict_envelope: ExternalDataPlanEnvelope,
        predict_input: serde_json::Value,
    }

    /// Build the smallest contract accepted by the strict facade.  The helper
    /// deliberately starts from the public W1 fixture so request, identity and
    /// influence signing stays exercised; it then removes every optional lane
    /// the facade refuses (transform, group, generator, metadata and OOF
    /// cache) rather than relying on an ad-hoc test-only schema.
    #[cfg(feature = "methods-optimizer")]
    fn strict_methods_terminal_fixture() -> StrictMethodsTerminalFixture {
        let mut request: TrainingRequest = serde_json::from_str(REQUEST_FIXTURE).unwrap();
        request
            .graph
            .nodes
            .retain(|node| node.id.as_str() == "model:base");
        request.graph.edges.clear();
        request.graph.metadata.clear();
        let graph_node = request.graph.nodes.first_mut().unwrap();
        graph_node.operator = Some(serde_json::json!({"type": "PLSRegression"}));
        graph_node.params = BTreeMap::from([("n_components".to_string(), serde_json::json!(1))]);
        graph_node.metadata.clear();
        graph_node.seed_label = None;

        request.campaign.generation = GenerationSpec::default();
        request
            .campaign
            .shape_plans
            .retain(|node_id, _| node_id.as_str() == "model:base");
        let shape = request.campaign.shape_plans.values_mut().next().unwrap();
        shape.augmentation_policy.sample_scope = dag_ml_core::AugmentationScope::None;
        shape.augmentation_policy.feature_scope = dag_ml_core::AugmentationScope::None;
        shape.augmentation_policy.unsafe_flags.clear();
        shape.selection_policy.scope = dag_ml_core::FeatureSelectionScope::None;
        shape.selection_policy.allow_schema_mismatch_on_join = false;
        request
            .campaign
            .data_bindings
            .retain(|node_id, _| node_id.as_str() == "model:base");
        let binding = request
            .campaign
            .data_bindings
            .values_mut()
            .next()
            .unwrap()
            .first_mut()
            .unwrap();
        binding.metadata.clear();
        binding.view_policy.include_augmented_train = false;
        binding.view_policy.include_augmented_validation = false;
        binding.view_policy.include_excluded = false;
        binding.view_policy.unsafe_flags.clear();
        request.campaign.branch_view_plans.clear();
        request.campaign.inner_cv = None;
        request.campaign.metadata.clear();
        request.campaign.leakage_policy.split_unit = dag_ml_core::SplitUnit::Sample;
        request.campaign.leakage_policy.require_group_ids = false;
        request.campaign.leakage_policy.unsafe_flags.clear();
        let split = request.campaign.split_invocation.as_mut().unwrap();
        split.controller_id = None;
        split.leakage_policy.split_unit = dag_ml_core::SplitUnit::Sample;
        split.leakage_policy.require_group_ids = false;
        split.leakage_policy.unsafe_flags.clear();
        split.params = BTreeMap::from([
            ("kind".to_string(), serde_json::json!("kfold")),
            ("n_splits".to_string(), serde_json::json!(2)),
            ("shuffle".to_string(), serde_json::json!(false)),
        ]);
        split.fold_set.as_mut().unwrap().sample_groups.clear();

        request
            .controller_manifests
            .retain(|manifest| manifest.operator_kind == NodeKind::Model);
        let manifest = request.controller_manifests.first_mut().unwrap();
        manifest.controller_id =
            dag_ml_core::ControllerId::new(dag_ml_core::METHODS_PLS_CONTROLLER_ID).unwrap();
        manifest.controller_version = "libn4m:strict-test".to_string();
        manifest.capabilities = BTreeSet::from([
            ControllerCapability::Deterministic,
            ControllerCapability::ThreadSafe,
            ControllerCapability::ProcessSafe,
            ControllerCapability::UsesCoreRng,
            ControllerCapability::EmitsPredictions,
            ControllerCapability::EmitsArtifacts,
            ControllerCapability::Stateful,
        ]);

        request.options.scheduler.kind = dag_ml_core::TrainingSchedulerKind::Sequential;
        request.options.scheduler.backend = None;
        request.options.scheduler.workers = 1;
        request.options.selection.required_metric_level = Some(PredictionLevel::Sample);
        request.options.selection.evaluation_scope = Some(EvaluationScope::Oof);
        request.options.selection.require_finite = true;
        request.options.resources.cpu_threads = 1;
        request.options.resources.gpu_devices.clear();
        request.options.resources.memory_bytes = None;
        request.options.resources.wall_time_ms = None;
        request.options.artifacts.cv_artifacts = CvArtifactRetention::Discard;
        request.options.artifacts.prediction_caches =
            dag_ml_core::PredictionCacheRetention::Discard;
        request.options.artifacts.fitted_artifacts = FittedArtifactMode::PortableRequired;

        let relations = SampleRelationSet {
            records: (1..=4)
                .map(|index| {
                    SampleRelation::new(
                        ObservationId::new(format!("observation:{index}")).unwrap(),
                        sample(&format!("sample:{index}")),
                    )
                })
                .collect(),
        };
        let relation_fingerprint = relations.fingerprint().unwrap();
        let binding = request
            .campaign
            .data_bindings
            .values_mut()
            .next()
            .unwrap()
            .first_mut()
            .unwrap();
        binding.relation_fingerprint = Some(relation_fingerprint);
        let envelope = envelope(binding, &request.data_identities[0], relations.clone());
        request.data_identities =
            vec![TrainingDataIdentity::from_binding_envelope(binding, &envelope).unwrap()];
        request.request_fingerprint = "0".repeat(64);
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        let projection = request.project().unwrap();
        let influence = influence_manifest(&request, &projection, &relations);

        let methods_inputs = serde_json::json!({
            "model:base.x": {
                "sample_ids": ["sample:1", "sample:2", "sample:3", "sample:4"],
                "x": [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]],
                "y": [[1.0], [2.0], [3.0], [4.0]],
                "target_names": ["protein"]
            }
        });
        let predict_dataset = MethodsPlsDataset {
            sample_ids: vec![
                SampleId::new("sample:predict:1").unwrap(),
                SampleId::new("sample:predict:2").unwrap(),
            ],
            x: MethodsPlsMatrix {
                values: vec![5.0, 0.0, 6.0, 1.0],
                rows: 2,
                cols: 2,
            },
            y: None,
            target_names: vec!["protein".to_string()],
        };
        let predict_fingerprint =
            methods_pls_predict_feature_content_fingerprint(&predict_dataset.x).unwrap();
        let predict_relations = SampleRelationSet {
            records: vec![
                SampleRelation::new(
                    ObservationId::new("observation:predict:1").unwrap(),
                    SampleId::new("sample:predict:1").unwrap(),
                ),
                SampleRelation::new(
                    ObservationId::new("observation:predict:2").unwrap(),
                    SampleId::new("sample:predict:2").unwrap(),
                ),
            ],
        };
        let mut predict_envelope = envelope.clone();
        predict_envelope.schema_version = EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2;
        predict_envelope.data_content_fingerprint = Some(predict_fingerprint.clone());
        predict_envelope.target_content_fingerprint = None;
        predict_envelope.predict_cohort = Some(
            PredictCohort::from_relations(
                PredictCohortRole::Inference,
                predict_relations,
                vec!["protein".to_string()],
                predict_fingerprint,
                None,
            )
            .unwrap(),
        );
        predict_envelope.validate().unwrap();
        let predict_input = serde_json::json!({
            "sample_ids": ["sample:predict:1", "sample:predict:2"],
            "x": [[5.0, 0.0], [6.0, 1.0]],
            "target_names": ["protein"]
        });

        StrictMethodsTerminalFixture {
            request_json: serde_json::to_string(&request).unwrap(),
            training_envelopes_json: serde_json::to_string(&BTreeMap::from([(
                "model:base.x".to_string(),
                envelope,
            )]))
            .unwrap(),
            relations_json: serde_json::to_string(&relations).unwrap(),
            influence_json: serde_json::to_string(&influence).unwrap(),
            methods_inputs_json: serde_json::to_string(&methods_inputs).unwrap(),
            predict_envelope_json: serde_json::to_string(&predict_envelope).unwrap(),
            predict_input_json: serde_json::to_string(&predict_input).unwrap(),
            selector_json: serde_json::to_string(
                &TerminalPredictionSelector::new(NodeId::new("model:base").unwrap(), "oof")
                    .unwrap(),
            )
            .unwrap(),
            predict_envelope,
            predict_input,
        }
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
            predict_cohort: None,
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
