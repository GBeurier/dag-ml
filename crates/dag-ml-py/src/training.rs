//! Owning PyO3 surface for the native W1 training operation.
//!
//! The binding only translates strict JSON contracts and Python controller
//! callbacks. Compile/plan/FIT_CV/SELECT/REFIT, scoring, output binding,
//! lineage and artifact capture remain implemented once in `dag-ml-core`.

use std::collections::BTreeMap;
#[cfg(feature = "methods-optimizer")]
use std::collections::BTreeSet;
use std::sync::{Mutex, MutexGuard};

use dag_ml_core::{
    calibrate_attached_training_replay_with_derived_context, execute_attached_training_replay,
    execute_loaded_predictor_replay, execute_training, parse_typed_json, ArtifactId,
    ArtifactLoadMode, AttachedTrainingReplayInput, BundleId,
    ConformalCalibrationTruth, ConformalMultiTargetPolicy, ConformalSmallSamplePolicy,
    DataBinding, DataMaterializationRequest, DataViewRequest,
    EnvelopeAttestedRuntimeDataProvider, ExternalDataPlanEnvelope, FittedArtifactMode,
    HandleKind, HandleRef, InMemoryArtifactStore, InMemoryDataProvider, LoadedPredictor,
    LoadedPredictorReplayInput, MethodsPlsData, MethodsPlsDataRequest,
    PortablePredictorPackage, RunId, RuntimeControllerRegistry, RuntimeDataProvider,
    SampleRelationSet, TrainingExecutionInput, TrainingInfluenceManifest, TrainingOutcome,
    TrainingReplayOutcome, TrainingReplayRequest, TrainingRequest,
};
#[cfg(feature = "methods-optimizer")]
use dag_ml_core::{
    LoadedPortableRefitReplayInputV3, MethodsPlsDataset, MethodsPlsMatrix,
    MethodsPortablePredictorReplayInput,
    PortableFullRefitExecutionInput, PortableRefitPackageV3,
    PortableRefitPackageV3BuildInput, PortableRefitRecipe, SampleId,
    build_portable_refit_package_v3, derive_portable_full_refit_target_plan,
    execute_loaded_portable_refit_replay_v3, execute_portable_full_refit,
};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
#[cfg(feature = "methods-optimizer")]
use serde::Deserialize;

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

/// PREDICT-only counterpart which deliberately does not re-attest the
/// training relation against a new replay cohort. Replay's signed envelope is
/// validated by the scheduler; this provider only serves its identity-keyed
/// numeric rows.
#[cfg(feature = "methods-optimizer")]
struct PyMethodsPlsReplayProvider {
    inner: InMemoryDataProvider,
    inputs: BTreeMap<String, MethodsPlsDataset>,
}

enum TrainingDataProvider {
    Host(EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>),
    #[cfg(feature = "methods-optimizer")]
    Methods(PyMethodsPlsTrainingProvider),
    #[cfg(feature = "methods-optimizer")]
    MethodsReplay(PyMethodsPlsReplayProvider),
}

impl TrainingDataProvider {
    fn data_handle_count(&self) -> usize {
        match self {
            Self::Host(provider) => provider.inner().handle_records().len(),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.inner.inner().handle_records().len(),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.inner.handle_records().len(),
        }
    }

    fn data_view_count(&self) -> usize {
        match self {
            Self::Host(provider) => provider.inner().view_records().len(),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.inner.inner().view_records().len(),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.inner.view_records().len(),
        }
    }
}

impl RuntimeDataProvider for TrainingDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> dag_ml_core::Result<HandleRef> {
        match self {
            Self::Host(provider) => provider.materialize(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.materialize(request),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.materialize(request),
        }
    }

    fn make_view(&self, request: &DataViewRequest) -> dag_ml_core::Result<HandleRef> {
        match self {
            Self::Host(provider) => provider.make_view(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.make_view(request),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.make_view(request),
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
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.training_data_identity(binding),
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
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.coordinator_relations(binding),
        }
    }

    fn methods_pls_capability(&self) -> dag_ml_core::Result<()> {
        match self {
            Self::Host(provider) => provider.methods_pls_capability(),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.methods_pls_capability(),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.methods_pls_capability(),
        }
    }

    fn preflight_methods_pls(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<()> {
        match self {
            Self::Host(provider) => provider.preflight_methods_pls(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.preflight_methods_pls(request),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.preflight_methods_pls(request),
        }
    }

    fn methods_pls_data(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<MethodsPlsData> {
        match self {
            Self::Host(provider) => provider.methods_pls_data(request),
            #[cfg(feature = "methods-optimizer")]
            Self::Methods(provider) => provider.methods_pls_data(request),
            #[cfg(feature = "methods-optimizer")]
            Self::MethodsReplay(provider) => provider.methods_pls_data(request),
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

        let mut raw = InMemoryDataProvider::new(
            dag_ml_core::ControllerId::new(PY_DATA_PROVIDER_CONTROLLER_ID)?,
        );
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
                "native Methods training fit view lacks scheduler-selected sample identities".to_string(),
            )
        })?;
        let requires_targets = matches!(request.phase, dag_ml_core::Phase::FitCv | dag_ml_core::Phase::Refit);
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

    fn methods_pls_data(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<MethodsPlsData> {
        self.data_for(request)
    }
}

#[cfg(feature = "methods-optimizer")]
impl PyMethodsPlsReplayProvider {
    fn new(
        envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
        inputs: BTreeMap<String, MethodsPlsDataset>,
    ) -> dag_ml_core::Result<Self> {
        let mut inner = InMemoryDataProvider::new(
            dag_ml_core::ControllerId::new(PY_DATA_PROVIDER_CONTROLLER_ID)?,
        );
        for envelope in envelopes.into_values() {
            inner.register_envelope(envelope)?;
        }
        for (key, dataset) in &inputs {
            dataset.validate(&format!("native Methods replay input `{key}`"), false)?;
        }
        Ok(Self { inner, inputs })
    }

    fn data_for(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<MethodsPlsData> {
        request.validate()?;
        if request.phase != dag_ml_core::Phase::Predict {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods replay provider supports PREDICT only".to_string(),
            ));
        }
        let key = dag_ml_core::data_binding_requirement_key(
            &request.binding.node_id,
            &request.binding.input_name,
        );
        let dataset = self.inputs.get(&key).ok_or_else(|| {
            dag_ml_core::DagMlError::RuntimeValidation(format!(
                "native Methods replay has no input for `{key}`"
            ))
        })?;
        // PREDICT's scheduler view intentionally has no fold subset. The
        // caller's strictly keyed input is therefore the complete requested
        // cohort; retain its explicit sample-id order rather than inventing
        // positional IDs or borrowing a training fold.
        let ids = request
            .fit_view
            .sample_ids
            .as_deref()
            .unwrap_or(&dataset.sample_ids);
        Ok(MethodsPlsData {
            fit: PyMethodsPlsTrainingProvider::dataset_for_view(dataset, ids, false, "replay")?,
            prediction: None,
        })
    }
}

#[cfg(feature = "methods-optimizer")]
impl RuntimeDataProvider for PyMethodsPlsReplayProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> dag_ml_core::Result<HandleRef> {
        self.inner.materialize(request)
    }

    fn make_view(&self, request: &DataViewRequest) -> dag_ml_core::Result<HandleRef> {
        self.inner.make_view(request)
    }

    fn methods_pls_capability(&self) -> dag_ml_core::Result<()> {
        Ok(())
    }

    fn preflight_methods_pls(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<()> {
        self.data_for(request).map(|_| ())
    }

    fn methods_pls_data(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<MethodsPlsData> {
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
        let coverages = parse_strict_json::<Vec<f64>>(
            coverages_json,
            "conformal calibration coverages",
        )?;
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

/// Execute the narrow portable Methods PLS training lane.
///
/// Unlike [`execute_training_json`], this entry point does not accept a Python
/// operator callback. Every executable node must be the registered native
/// Methods PLS controller, and numeric rows are supplied through the typed
/// `methods_inputs_json` provider. This is the public bridge for hosts that
/// want a durable N4MM Package V2 rather than a process-local sidecar.
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
        let warnings = parse_strict_json::<Vec<String>>(warnings_json, "native Methods training warnings")?;
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
        let methods_controller = dag_ml_core::ControllerId::new(
            dag_ml_core::METHODS_PLS_CONTROLLER_ID,
        )
        .map_err(py_core_error)?;
        if projection
            .plan
            .node_plans
            .values()
            .any(|node_plan| node_plan.controller_id != methods_controller)
        {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods training requires every executable node to use controller:methods.pls; host controller fallback is forbidden".to_string(),
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
        let runtime = dag_ml_core::MethodsRuntime::configure(methods_library_path)
            .map_err(|error| py_core_error(dag_ml_core::DagMlError::RuntimeValidation(error.to_string())))?;
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
        let target_request = TrainingRequest::from_json(target_request_json).map_err(py_core_error)?;
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
        let target_plan = derive_portable_full_refit_target_plan(
            &recipe,
            &source_package,
            &target_request,
        )
        .map_err(py_core_error)?;
        let bindings = target_plan
            .node_plans
            .values()
            .flat_map(|node_plan| node_plan.data_bindings.iter().cloned())
            .collect::<Vec<DataBinding>>();
        let methods_controller = dag_ml_core::ControllerId::new(dag_ml_core::METHODS_PLS_CONTROLLER_ID)
            .map_err(py_core_error)?;
        if source_package
            .effective_plan
            .node_plans
            .values()
            .any(|node_plan| node_plan.controller_id != methods_controller)
        {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods full refit requires every executable node to use controller:methods.pls; host controller fallback is forbidden".to_string(),
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
        let runtime = dag_ml_core::MethodsRuntime::configure(methods_library_path).map_err(|error| {
            py_core_error(dag_ml_core::DagMlError::RuntimeValidation(error.to_string()))
        })?;
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(dag_ml_core::MethodsPlsController::new(runtime)))
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

/// Register the native PLS controller and, when attested by the campaign,
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
    let Some(hpo_controller_id) = methods_hpo_controller_id(&projection.plan.campaign.metadata)? else {
        controllers.register(Box::new(dag_ml_core::MethodsPlsController::new(runtime)))?;
        return Ok(());
    };
    dag_ml_core::register_methods_runtime_controllers(controllers, hpo_controller_id, runtime)
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
    let rows_to_matrix = |rows: Vec<Vec<f64>>, label: &str| -> dag_ml_core::Result<MethodsPlsMatrix> {
        let row_count = rows.len();
        let columns = rows.first().map(Vec::len).unwrap_or(0);
        if row_count == 0 || columns == 0 || rows.iter().any(|row| row.len() != columns) {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "native Methods training input `{label}` is not a non-empty rectangular matrix"
            )));
        }
        let values = rows.into_iter().flatten().collect::<Vec<_>>();
        Ok(MethodsPlsMatrix { values, rows: row_count, cols: columns })
    };
    let sample_ids = input
        .sample_ids
        .into_iter()
        .map(SampleId::new)
        .collect::<dag_ml_core::Result<Vec<_>>>()?;
    let dataset = MethodsPlsDataset {
        sample_ids,
        x: rows_to_matrix(input.x, "x")?,
        y: input
            .y
            .map(|rows| rows_to_matrix(rows, "y"))
            .transpose()?,
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
        let runtime = dag_ml_core::MethodsRuntime::configure(methods_library_path)
            .map_err(|error| py_core_error(dag_ml_core::DagMlError::RuntimeValidation(error.to_string())))?;
        let warnings = parse_strict_json::<Vec<String>>(warnings_json, "native Methods replay warnings")?;
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
        let methods_controller = dag_ml_core::ControllerId::new(dag_ml_core::METHODS_PLS_CONTROLLER_ID)
            .map_err(py_core_error)?;
        if package
            .outcome
            .effective_plan
            .node_plans
            .values()
            .any(|node_plan| node_plan.controller_id != methods_controller)
        {
            return Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation(
                "native Methods V3 refit replay requires every executable node to use controller:methods.pls".to_string(),
            )));
        }
        let data_provider = TrainingDataProvider::MethodsReplay(
            PyMethodsPlsReplayProvider::new(envelopes.clone(), inputs)
                .map_err(py_core_error)?,
        );
        let runtime = dag_ml_core::MethodsRuntime::configure(methods_library_path)
            .map_err(|error| py_core_error(dag_ml_core::DagMlError::RuntimeValidation(error.to_string())))?;
        let mut controllers = RuntimeControllerRegistry::new();
        controllers
            .register(Box::new(dag_ml_core::MethodsPlsController::new(runtime)))
            .map_err(py_core_error)?;
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
                execute_loaded_portable_refit_replay_v3(LoadedPortableRefitReplayInputV3 {
                    package: &package,
                    request: &request,
                    outcome_id,
                    run_id,
                    controllers: &controllers,
                    data_provider: &data_provider,
                    data_envelopes: &envelopes,
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
    let warnings = parse_strict_json::<Vec<String>>(
        warnings_json,
        "loaded predictor replay warnings",
    )?;
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
                format!("loaded predictor sidecar handle references unknown artifact `{artifact_id}`"),
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
            let mut predictions = if is_model && matches!(task.phase, Phase::FitCv | Phase::Refit)
            {
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
