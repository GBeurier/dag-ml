//! In-process campaign execution (Mechanism B).
//!
//! Unlike the rest of this crate (control-plane only: validate / compile / plan
//! JSON in, JSON out), this module *executes* a campaign through the existing
//! `dag-ml-core` runtime while calling BACK into Python for operator execution —
//! no subprocess. The data path is **identical** to the subprocess CLI
//! (`dag-ml-cli run-process-dsl-cv-refit-bundle`): the same envelope-based
//! [`InMemoryDataProvider`] produces the per-fold / per-partition `data_views`
//! (the `sample_ids`), so the host adapter's `run_node` self-fetches the same
//! X / y and the scores cannot diverge. Only the OPERATOR execution crosses to
//! Python — there is no Python data callback.
//!
//! The host wires ONE Python callback:
//!
//! * `op_callback(task_dict) -> result_dict` runs one [`NodeTask`] and returns a
//!   [`NodeResult`] (the same JSON contract the JSONL subprocess adapter uses).
//!
//! Every bridge crossing is a DIRECT serde<->`PyObject` conversion via the
//! `pythonize` crate (Rust value -> `pythonize` -> Python dict; and the returned
//! dict -> `depythonize` -> Rust value) — no intermediate JSON *string* step
//! (`json.dumps` / `json.loads`). `pythonize` walks the same serde data model as
//! `serde_json`, so the dict the callback receives is structurally identical to
//! the JSONL frame the subprocess adapter would see; only the per-NodeTask string
//! serialization overhead is removed. A Python callback panic is contained with
//! [`std::panic::catch_unwind`] so it surfaces as a structured [`DagMlError`]
//! instead of unwinding across the FFI boundary; a Python *exception* is
//! propagated as [`DagMlError::RuntimeValidation`] carrying the message text.
//!
//! The phase sequence mirrors the CLI's `build_bundle_from_cv_then_captured_refit`
//! exactly: native variant SELECT (when the plan is multi-variant and unpinned)
//! -> FIT_CV -> REFIT in ONE [`RunContext`] -> cross-fold OOF scoring -> the
//! native [`ScoreSet`]. That is what the host maps into a `RunResult`, so the
//! returned `scores` is byte-identical to the bundle's `scores` the subprocess
//! path reads back.

use std::panic::AssertUnwindSafe;

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pythonize::{depythonize, pythonize};

use dag_ml_core::{
    build_execution_plan, compile_pipeline_dsl_with_generation_and_controller_registry,
    fan_out_data_aware_branches, parse_pipeline_dsl_json, plan_oof_partition_mode,
    select_best_variant_by_cv, AggregationControllerResult, AggregationControllerTask,
    ControllerId, ControllerRegistry, DagMlError as CoreDagMlError, ExecutionPlan,
    ExternalDataPlanEnvelope, InMemoryArtifactStore, InMemoryDataProvider, NodeResult, NodeTask,
    Phase, RegressionMetricKind, RegressionMetricReport, RunContext, RunId, RuntimeController,
    RuntimeControllerRegistry, ScoreSet, SequentialScheduler, VariantId,
    SCORE_SET_SCHEMA_VERSION,
};

use crate::{py_core_error, py_serde_error};

/// Serialize a `dag-ml-core` value directly to a Python object with `pythonize`
/// (serde data model -> `PyObject`), skipping any JSON *string* step. The dict
/// the callback receives is structurally identical to `json.loads` of the value's
/// JSON, so the host `op_callback` (`node_runner.run_node`) sees the same shape it
/// always did.
fn to_py_object<T: serde::Serialize>(
    py: Python<'_>,
    value: &T,
) -> Result<Py<PyAny>, CoreDagMlError> {
    pythonize(py, value)
        .map(|bound| bound.unbind())
        .map_err(pythonize_error)
}

/// Deserialize a Python object returned by a callback directly into a
/// `dag-ml-core` value with `depythonize` (`PyObject` -> serde data model),
/// skipping any JSON *string* step.
fn from_py_object<T: serde::de::DeserializeOwned>(
    obj: &Bound<'_, PyAny>,
) -> Result<T, CoreDagMlError> {
    depythonize(obj).map_err(pythonize_error)
}

/// Map a `pythonize`/`depythonize` conversion failure to a structured core error.
fn pythonize_error(err: pythonize::PythonizeError) -> CoreDagMlError {
    CoreDagMlError::RuntimeValidation(format!("in-process bridge conversion failed: {err}"))
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
            from_py_object::<Resp>(&returned)
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

/// Map the host's selection-metric string to the core metric kind. Mirrors the
/// CLI's `--selection-metric` (`rmse` | `accuracy`); anything else defaults to
/// RMSE, exactly like the CLI's clap default.
fn parse_selection_metric(metric: &str) -> RegressionMetricKind {
    match metric {
        "accuracy" => RegressionMetricKind::Accuracy,
        _ => RegressionMetricKind::Rmse,
    }
}

/// Build the in-process runtime controller registry: one [`PyOperatorController`]
/// per controller the plan references, all backed by the SAME host `op_callback`
/// (the host dispatches by node kind inside `run_node`, exactly as the subprocess
/// adapter does).
fn build_runtime_controllers(
    py: Python<'_>,
    plan: &ExecutionPlan,
    op_callback: &Py<PyAny>,
) -> Result<RuntimeControllerRegistry, CoreDagMlError> {
    let mut runtime_controllers = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        runtime_controllers.register(Box::new(PyOperatorController {
            controller_id: controller_id.clone(),
            op_callback: op_callback.clone_ref(py),
        }))?;
    }
    Ok(runtime_controllers)
}

/// The resolved REFIT target plus the non-selected variants' VALIDATION (OOF) reports, so the host
/// bundle can surface ALL variants' CV scores — not just the winner's. The winner's validation
/// reports come fresh from the real FIT_CV run, so only the LOSER variants' reports are carried here
/// (avoiding a duplicate `(node, variant, partition, fold, level)` key in the final `ScoreSet`).
struct ResolvedRefitVariant {
    variant_id: VariantId,
    loser_validation_reports: Vec<RegressionMetricReport>,
}

/// Resolve the variant REFIT targets, mirroring the CLI's `resolve_refit_variant`:
/// a single-variant plan refits that variant; a multi-variant plan runs one
/// single-variant FIT_CV per variant and refits the best by `selection_metric`
/// (or the default variant when native scoring is off — no host targets).
fn resolve_refit_variant(
    plan: &ExecutionPlan,
    run_id: &RunId,
    root_seed: u64,
    selection_metric: RegressionMetricKind,
    runtime_controllers: &RuntimeControllerRegistry,
    data_provider: &InMemoryDataProvider,
) -> Result<ResolvedRefitVariant, CoreDagMlError> {
    if plan.variants.len() > 1 {
        let selected = select_best_variant_by_cv(
            plan,
            run_id,
            Some(root_seed),
            selection_metric,
            |variant_plan, ctx| {
                SequentialScheduler
                    .execute_campaign_phase_with_data_provider(
                        variant_plan,
                        runtime_controllers,
                        data_provider,
                        ctx,
                        Phase::FitCv,
                    )
                    .map(|_results| ())
            },
        )?;
        if let Some(selection) = selected {
            let variant_id = selection.selected_variant_id.clone();
            // Keep only the LOSER variants' reports — the winner's come from the real FIT_CV run.
            let loser_validation_reports = selection
                .validation_reports
                .into_iter()
                .filter(|report| report.variant_id.as_ref() != Some(&variant_id))
                .collect();
            return Ok(ResolvedRefitVariant {
                variant_id,
                loser_validation_reports,
            });
        }
    }
    let variant_id = plan
        .variants
        .first()
        .map(|variant| variant.variant_id.clone())
        .ok_or_else(|| {
            CoreDagMlError::RuntimeValidation("execution plan has no variants to refit".to_string())
        })?;
    Ok(ResolvedRefitVariant {
        variant_id,
        loser_validation_reports: Vec::new(),
    })
}

/// Merge the non-selected variants' VALIDATION (OOF) reports into the run's `ScoreSet` so the bundle
/// carries every variant's CV score, not just the winner's. ADDITIVE only: each loser report is
/// already tagged with its own `variant_id`, so it cannot collide with the winner's reports on the
/// `(node, variant, partition, fold, level)` key. A no-op when there are no losers (single-variant
/// runs, or native scoring off). If scoring produced no winner `ScoreSet` but losers exist, a new
/// `ScoreSet` is created to hold them.
fn merge_loser_validation_reports(
    scores: &mut Option<ScoreSet>,
    plan_id: &str,
    loser_validation_reports: Vec<RegressionMetricReport>,
) {
    if loser_validation_reports.is_empty() {
        return;
    }
    match scores {
        Some(score_set) => score_set.reports.extend(loser_validation_reports),
        None => {
            *scores = Some(ScoreSet {
                schema_version: SCORE_SET_SCHEMA_VERSION,
                plan_id: plan_id.to_string(),
                selection_metric: None,
                reports: loser_validation_reports,
            });
        }
    }
}

/// Run a CV + refit campaign IN-PROCESS through the existing runtime, dispatching
/// operator execution back to Python while the data path stays entirely in Rust
/// via the SAME envelope-based [`InMemoryDataProvider`] the subprocess CLI uses.
///
/// * `dsl_json` — the executable compat DSL (carries the embedded `fold_set` in
///   `split_invocation` and the model `data_bindings`, exactly as the host
///   `assemble_cv_refit_dsl` builds for the subprocess path).
/// * `envelope_json` — the [`ExternalDataPlanEnvelope`]; its coordinator
///   relations + the DSL fold set drive `data_views` production.
/// * `controller_manifests_json` — the controller manifest list.
/// * `op_callback` — the host bridge running one [`NodeTask`] (`run_node`).
/// * `selection_metric` — `rmse` | `accuracy`, used only for native multi-variant
///   SELECT (ignored for single-variant plans).
///
/// Returns a JSON object `{ "node_results": [...], "scores": <ScoreSet|null> }`.
/// `scores` is byte-identical to the subprocess bundle's `scores`, so the host
/// maps it into the same `RunResult`.
#[pyfunction]
pub fn run_cv_refit_in_process(
    py: Python<'_>,
    dsl_json: &str,
    envelope_json: &str,
    controller_manifests_json: &str,
    op_callback: Py<PyAny>,
    selection_metric: &str,
) -> PyResult<String> {
    let metric = parse_selection_metric(selection_metric);

    // 1. Read the envelope first (the CLI reads it before the plan so data-aware
    //    branch fan-out can discover partition values from coordinator relations).
    let envelope: ExternalDataPlanEnvelope =
        serde_json::from_str(envelope_json).map_err(py_serde_error)?;

    // 2. Build the plan exactly as `build_plan_from_dsl_path_with_envelope`:
    //    fan out data-aware branches against the envelope, compile with the
    //    controller registry, then build the execution plan from the compiled
    //    graph + campaign template.
    let dsl_spec = parse_pipeline_dsl_json(dsl_json.as_bytes()).map_err(py_core_error)?;
    let dsl_spec = fan_out_data_aware_branches(&dsl_spec, &envelope).map_err(py_core_error)?;
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
    let plan = build_execution_plan(
        format!("plan:{}", dsl_spec.id),
        compiled.graph,
        compiled.campaign_template,
        &controller_registry,
    )
    .map_err(py_core_error)?;

    // 3. Build the SAME Rust data provider the CLI uses
    //    (`data_provider_for_training_envelope`): validate the envelope relations
    //    against the campaign folds, then register the envelope. data_views /
    //    sample_ids are produced identically to the subprocess path.
    plan.campaign
        .validate_data_envelope_relations(&envelope)
        .map_err(py_core_error)?;
    let data_provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data.provider").map_err(py_core_error)?,
        envelope,
    )
    .map_err(py_core_error)?;

    let runtime_controllers =
        build_runtime_controllers(py, &plan, &op_callback).map_err(py_core_error)?;

    let run_id = RunId::new(format!("run:{}:in-process", dsl_spec.id)).map_err(py_core_error)?;
    let root_seed: u64 = 0;

    // 4. Mirror `build_bundle_from_cv_then_captured_refit`: native variant SELECT
    //    (when multi-variant), then FIT_CV + REFIT in ONE RunContext, then score.
    let resolved = resolve_refit_variant(
        &plan,
        &run_id,
        root_seed,
        metric,
        &runtime_controllers,
        &data_provider,
    )
    .map_err(py_core_error)?;
    let selected_variant_id = resolved.variant_id;
    let loser_validation_reports = resolved.loser_validation_reports;

    let mut artifact_store = InMemoryArtifactStore::new();
    let mut ctx = RunContext::new(run_id, Some(root_seed));
    ctx.variant_id = Some(selected_variant_id);

    let fit_cv_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &runtime_controllers,
            &data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .map_err(py_core_error)?;

    let refit_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider_and_artifact_store(
            &plan,
            &runtime_controllers,
            &data_provider,
            &mut artifact_store,
            &mut ctx,
            Phase::Refit,
        )
        .map_err(py_core_error)?;

    // 5. Score: collect the cross-fold OOF average (cv_best_score) + the REFIT
    //    final/test reports. The loser variants' VALIDATION (OOF) reports captured
    //    during native SELECT are merged in FIRST (each tagged its own variant_id,
    //    REPORT-ONLY — they carry no predictions/handles), so the bundle surfaces
    //    every variant's CV score, not just the winner's. Then build the native
    //    ScoreSet the host maps to a RunResult — identical to the CLI's
    //    `bundle.scores`.
    ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(&plan))
        .map_err(py_core_error)?;
    let mut scores = ctx.build_score_set(plan.id.clone(), None);
    merge_loser_validation_reports(&mut scores, &plan.id, loser_validation_reports);

    let mut node_results = fit_cv_results;
    node_results.extend(refit_results);

    let payload = serde_json::json!({
        "node_results": node_results,
        "scores": scores,
    });
    serde_json::to_string(&payload).map_err(py_serde_error)
}
