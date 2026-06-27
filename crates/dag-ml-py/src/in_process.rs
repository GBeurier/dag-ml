//! In-process campaign execution (Mechanism B).
//!
//! Unlike the rest of this crate (control-plane only: validate / compile / plan
//! JSON in, JSON out), this module *executes* a campaign through the existing
//! `dag-ml-core` runtime while calling BACK into Python for operator execution
//! and data materialization — no subprocess. The host wires two Python
//! callbacks:
//!
//! * `op_callback(task_dict) -> result_dict` runs one [`NodeTask`] and returns a
//!   [`NodeResult`] (the same JSON contract the JSONL subprocess adapter uses).
//! * `data_callback(request_dict) -> handle_dict` materializes data / builds a
//!   view and returns an opaque [`HandleRef`]. The actual feature buffers stay
//!   on the Python side keyed by `handle.handle`; Rust only ever holds the
//!   opaque handle, never touches Arrow.
//!
//! Every bridge crossing is a JSON round-trip (Rust value -> `serde_json`
//! string -> `json.loads` -> Python dict; and back). A Python callback panic is
//! contained with [`std::panic::catch_unwind`] so it surfaces as a structured
//! [`DagMlError`] instead of unwinding across the FFI boundary; a Python
//! *exception* is propagated as [`DagMlError::RuntimeValidation`] carrying the
//! message text.

use std::panic::AssertUnwindSafe;

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

use dag_ml_core::{
    build_execution_plan, compile_pipeline_dsl_with_generation_and_controller_registry,
    parse_pipeline_dsl_json, AggregationControllerResult, AggregationControllerTask, CampaignSpec,
    ControllerId, ControllerRegistry, DagMlError as CoreDagMlError, DataBinding,
    DataMaterializationRequest, DataViewRequest, HandleRef, NodeResult, NodeTask, Phase,
    RunContext, RunId, RuntimeController, RuntimeControllerRegistry, RuntimeDataProvider,
    SampleRelationSet, SequentialScheduler,
};

use crate::{py_core_error, py_serde_error};

/// Serialize a `dag-ml-core` value to a Python object via a JSON round-trip
/// (`serde_json` -> `json.loads`). The dict the callback receives is therefore
/// byte-identical to the JSONL frame the subprocess adapter would see.
fn to_py_object<T: serde::Serialize>(
    py: Python<'_>,
    value: &T,
) -> Result<Py<PyAny>, CoreDagMlError> {
    let json = serde_json::to_string(value).map_err(CoreDagMlError::Serialization)?;
    let loads = py
        .import("json")
        .and_then(|module| module.getattr("loads"))
        .map_err(core_error_from_py)?;
    let obj = loads.call1((json,)).map_err(core_error_from_py)?;
    Ok(obj.unbind())
}

/// Deserialize a Python object returned by a callback into a `dag-ml-core` value
/// via a JSON round-trip (`json.dumps` -> `serde_json`).
fn from_py_object<T: serde::de::DeserializeOwned>(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> Result<T, CoreDagMlError> {
    let dumps = py
        .import("json")
        .and_then(|module| module.getattr("dumps"))
        .map_err(core_error_from_py)?;
    let json: String = dumps
        .call1((obj,))
        .and_then(|value| value.extract())
        .map_err(core_error_from_py)?;
    serde_json::from_str(&json).map_err(CoreDagMlError::Serialization)
}

/// Convert a Python exception into a structured core error carrying its message.
fn core_error_from_py(err: PyErr) -> CoreDagMlError {
    Python::attach(|py| {
        CoreDagMlError::RuntimeValidation(format!(
            "python callback raised an exception: {}",
            err.value(py)
        ))
    })
}

/// Convert a payload captured from a `catch_unwind` panic into a message.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Invoke a Python callback with a single serialized request and deserialize the
/// returned object. The whole GIL section is wrapped in `catch_unwind` so a
/// panic inside PyO3 (or a panicking `__del__`, etc.) becomes a structured error
/// rather than an unwind across the FFI boundary.
fn call_py_bridge<Req, Resp>(
    callback: &Py<PyAny>,
    request: &Req,
    bridge: &str,
) -> Result<Resp, CoreDagMlError>
where
    Req: serde::Serialize,
    Resp: serde::de::DeserializeOwned,
{
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        Python::attach(|py| -> Result<Resp, CoreDagMlError> {
            let payload = to_py_object(py, request)?;
            let returned = callback
                .bind(py)
                .call1((payload,))
                .map_err(core_error_from_py)?;
            from_py_object::<Resp>(py, &returned)
        })
    }));
    match result {
        Ok(value) => value,
        Err(payload) => Err(CoreDagMlError::RuntimeValidation(format!(
            "{bridge} python callback panicked: {}",
            panic_message(payload.as_ref())
        ))),
    }
}

/// A [`RuntimeController`] backed by a Python callback. The scheduler hands it a
/// [`NodeTask`]; it round-trips the task to a Python dict, calls the host's node
/// runner, and round-trips the returned dict back into a [`NodeResult`].
struct PyOperatorController {
    controller_id: ControllerId,
    op_callback: Py<PyAny>,
}

impl RuntimeController for PyOperatorController {
    fn controller_id(&self) -> &ControllerId {
        &self.controller_id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult, CoreDagMlError> {
        call_py_bridge::<NodeTask, NodeResult>(&self.op_callback, task, "operator")
    }

    fn invoke_aggregation(
        &self,
        task: &AggregationControllerTask,
    ) -> Result<AggregationControllerResult, CoreDagMlError> {
        call_py_bridge::<AggregationControllerTask, AggregationControllerResult>(
            &self.op_callback,
            task,
            "aggregation",
        )
    }
}

/// A [`RuntimeDataProvider`] backed by a Python callback. `materialize` and
/// `make_view` round-trip their request to the host resolver and expect an
/// opaque [`HandleRef`] back; the feature buffers stay Python-side keyed by the
/// returned handle, so Rust never inspects them.
struct PyDataProvider {
    data_callback: Py<PyAny>,
}

impl RuntimeDataProvider for PyDataProvider {
    fn materialize(
        &self,
        request: &DataMaterializationRequest,
    ) -> Result<HandleRef, CoreDagMlError> {
        call_py_bridge::<DataMaterializationRequest, HandleRef>(
            &self.data_callback,
            request,
            "data-materialize",
        )
    }

    fn make_view(&self, request: &DataViewRequest) -> Result<HandleRef, CoreDagMlError> {
        call_py_bridge::<DataViewRequest, HandleRef>(&self.data_callback, request, "data-view")
    }

    fn coordinator_relations(
        &self,
        binding: &DataBinding,
    ) -> Result<Option<SampleRelationSet>, CoreDagMlError> {
        call_py_bridge::<DataBinding, Option<SampleRelationSet>>(
            &self.data_callback,
            binding,
            "coordinator-relations",
        )
    }
}

/// Parse the scope phase identifier the host requests.
fn parse_scope(scope: &str) -> Result<Phase, CoreDagMlError> {
    match scope {
        "fit_cv" => Ok(Phase::FitCv),
        "refit" => Ok(Phase::Refit),
        "predict" => Ok(Phase::Predict),
        other => Err(CoreDagMlError::RuntimeValidation(format!(
            "unsupported in-process scope `{other}`; expected one of fit_cv, refit, predict"
        ))),
    }
}

/// Run a CV / refit / predict campaign IN-PROCESS through the existing runtime,
/// dispatching operator execution and data materialization back to Python.
///
/// `dsl_json` is the pipeline DSL (compiled to a graph with the supplied
/// controller manifests), `campaign_json` the campaign spec, and
/// `controller_manifests_json` the controller manifest list. `op_callback` and
/// `data_callback` are the host bridges. `scope` selects the phase to run
/// (`fit_cv` / `refit` / `predict`). Returns a JSON object with the per-node
/// `node_results` and the native `scores` (or `null` when scoring is off).
#[pyfunction]
pub fn run_cv_refit_in_process(
    py: Python<'_>,
    dsl_json: &str,
    campaign_json: &str,
    controller_manifests_json: &str,
    op_callback: Py<PyAny>,
    data_callback: Py<PyAny>,
    scope: &str,
) -> PyResult<String> {
    let phase = parse_scope(scope).map_err(py_core_error)?;

    let dsl_spec = parse_pipeline_dsl_json(dsl_json.as_bytes()).map_err(py_core_error)?;
    let manifests =
        serde_json::from_str::<Vec<dag_ml_core::ControllerManifest>>(controller_manifests_json)
            .map_err(py_serde_error)?;
    let mut controller_registry = ControllerRegistry::new();
    for manifest in &manifests {
        controller_registry
            .register(manifest.clone())
            .map_err(py_core_error)?;
    }
    let compiled = compile_pipeline_dsl_with_generation_and_controller_registry(
        &dsl_spec,
        &controller_registry,
    )
    .map_err(py_core_error)?;
    let campaign: CampaignSpec = serde_json::from_str(campaign_json).map_err(py_serde_error)?;

    let plan = build_execution_plan(
        format!("plan:{}", dsl_spec.id),
        compiled.graph,
        campaign,
        &controller_registry,
    )
    .map_err(py_core_error)?;

    let mut runtime_controllers = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        runtime_controllers
            .register(Box::new(PyOperatorController {
                controller_id: controller_id.clone(),
                op_callback: op_callback.clone_ref(py),
            }))
            .map_err(py_core_error)?;
    }
    let data_provider = PyDataProvider {
        data_callback: data_callback.clone_ref(py),
    };

    let mut ctx = RunContext::new(
        RunId::new(format!("run:{}:{scope}", dsl_spec.id)).map_err(py_core_error)?,
        Some(0),
    );

    let results = match phase {
        Phase::FitCv => SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &runtime_controllers,
                &data_provider,
                &mut ctx,
                Phase::FitCv,
            )
            .map_err(py_core_error)?,
        Phase::Refit => {
            let mut artifact_store = dag_ml_core::InMemoryArtifactStore::new();
            SequentialScheduler
                .execute_campaign_phase_with_data_provider_and_artifact_store(
                    &plan,
                    &runtime_controllers,
                    &data_provider,
                    &mut artifact_store,
                    &mut ctx,
                    Phase::Refit,
                )
                .map_err(py_core_error)?
        }
        phase => SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                &plan,
                &runtime_controllers,
                &data_provider,
                &mut ctx,
                phase,
            )
            .map_err(py_core_error)?,
    };

    if phase == Phase::FitCv {
        ctx.collect_cross_fold_validation_scores(dag_ml_core::plan_oof_partition_mode(&plan))
            .map_err(py_core_error)?;
    }
    let scores = ctx.build_score_set(plan.id.clone(), None);

    let payload = serde_json::json!({
        "node_results": results,
        "scores": scores,
    });
    serde_json::to_string(&payload).map_err(py_serde_error)
}
