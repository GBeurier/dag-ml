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
    build_execution_plan, compile_operator_variant_models,
    compile_pipeline_dsl_with_generation_and_controller_registry, enumerate_variants,
    fan_out_data_aware_branches, parse_pipeline_dsl_json, plan_oof_partition_mode,
    prune_plan_to_active, select_best_operator_variant_from_models, select_best_variant_by_cv,
    AggregationControllerResult, AggregationControllerTask, ControllerId, ControllerRegistry,
    DagMlError as CoreDagMlError, ExecutionPlan, ExternalDataPlanEnvelope, InMemoryArtifactStore,
    InMemoryDataProvider, NodeResult, NodeTask, OperatorVariantModel, Phase, RegressionMetricKind,
    RegressionMetricReport, RunContext, RunId, RuntimeController, RuntimeControllerRegistry,
    ScoreSet, SequentialScheduler, VariantId, VariantValidationPredictions,
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
fn from_py_object<T>(obj: &Bound<'_, PyAny>) -> Result<T, CoreDagMlError>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let raw: serde_json::Value = depythonize(obj).map_err(pythonize_error)?;
    dag_ml_core::deserialize_external_value(
        raw,
        "in-process callback result",
        CoreDagMlError::RuntimeValidation,
    )
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
    Resp: serde::de::DeserializeOwned + serde::Serialize,
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
/// CLI's `--selection-metric` (`rmse` | `accuracy` | `balanced_accuracy`); anything
/// else defaults to RMSE, exactly like the CLI's clap default. `balanced_accuracy`
/// matches nirs4all's default classification ranking metric.
fn parse_selection_metric(metric: &str) -> RegressionMetricKind {
    match metric {
        "accuracy" => RegressionMetricKind::Accuracy,
        "balanced_accuracy" => RegressionMetricKind::BalancedAccuracy,
        _ => RegressionMetricKind::Rmse,
    }
}

/// Build the in-process runtime controller registry: one [`PyOperatorController`]
/// per controller the plan references, all backed by the SAME host `op_callback`
/// (the host dispatches by node kind inside `run_node`, exactly as the subprocess
/// adapter does).
pub(crate) fn build_runtime_controllers(
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
#[derive(Debug)]
struct ResolvedRefitVariant {
    variant_id: VariantId,
    loser_validation_reports: Vec<RegressionMetricReport>,
    /// The non-selected variants' VALIDATION (OOF) PREDICTIONS, each re-tagged with its own variant
    /// id + content fingerprint, so the host can fill a LOSER variant's per-sample prediction rows
    /// (not just its scalar CV score). The WINNER's predictions come fresh from the real FIT_CV pass,
    /// so only the LOSERS' are carried here. Empty for param-variant / pinned / single-variant runs.
    loser_validation_predictions: Vec<VariantValidationPredictions>,
    /// `Some` only for operator-SELECT: the union plan PRUNED to the winning operator choice
    /// (merge, meta-model and inactive choices elided), on which the winner FIT_CV + REFIT must run
    /// instead of the stacking union. `None` for param-variant / pinned / single-variant runs (the
    /// union plan is the refit plan) — mirrors the CLI `ResolvedRefitVariant::pruned_plan`.
    pruned_plan: Option<ExecutionPlan>,
    /// The WINNER's operator-variant content fingerprint (Phase 5 `variant_label`): `Some(<sha256>)`
    /// for an operator-SELECT winner, `None` otherwise. Stamped onto the winner's fresh FIT_CV/REFIT
    /// reports so the WINNER report carries `variant_label`, not just the losers.
    winner_variant_label: Option<String>,
}

/// Run native OPERATOR-SELECT off the lowered operator-variant models, mirroring the CLI's
/// `resolve_operator_select`: score each choice on its PRUNED plan, return the winner together with
/// its pruned plan, the losers' OOF reports, and the winner's content fingerprint. Returns
/// `Ok(None)` when scoring is off (no host targets) so the caller falls back to the default. Keeps
/// winner-ONLY refit (the multi-model 32-not-34 contract).
fn resolve_operator_select(
    plan: &ExecutionPlan,
    operator_variant_models: &[OperatorVariantModel],
    run_id: &RunId,
    root_seed: u64,
    selection_metric: RegressionMetricKind,
    runtime_controllers: &RuntimeControllerRegistry,
    data_provider: &InMemoryDataProvider,
) -> Result<Option<ResolvedRefitVariant>, CoreDagMlError> {
    let selected = select_best_operator_variant_from_models(
        plan,
        operator_variant_models,
        run_id,
        Some(root_seed),
        selection_metric,
        |pruned_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase_with_data_provider(
                    pruned_plan,
                    runtime_controllers,
                    data_provider,
                    ctx,
                    Phase::FitCv,
                )
                .map(|_results| ())
        },
    )?;
    let Some(selection) = selected else {
        return Ok(None);
    };
    let variant_id = selection.selected_variant_id.clone();
    // The winner's content fingerprint (Phase 5) — recovered from its OWN report in the selection
    // loop (already stamped there) so the fresh winner FIT_CV/REFIT reports get the SAME label.
    let winner_variant_label = selection
        .validation_reports
        .iter()
        .find(|report| report.variant_id.as_ref() == Some(&variant_id))
        .and_then(|report| report.variant_label.clone());
    let loser_validation_reports = selection
        .validation_reports
        .into_iter()
        .filter(|report| report.variant_id.as_ref() != Some(&variant_id))
        .collect();
    // Keep only the LOSER variants' captured VALIDATION (OOF) predictions — the winner's come fresh
    // from the real FIT_CV pass below, so re-surfacing the transient ones would duplicate them.
    let loser_validation_predictions = selection
        .variant_validation_predictions
        .into_iter()
        .filter(|captured| captured.variant_id != variant_id)
        .collect();
    // Recompute the WINNER's pruned plan so FIT_CV + REFIT run on it (not the union). The single
    // operator model is guaranteed by `select_best_operator_variant_from_models`.
    let model = &operator_variant_models[0];
    let pruned_plan = pruned_plan_for_operator_variant(plan, model, &variant_id, root_seed)?;
    Ok(Some(ResolvedRefitVariant {
        variant_id,
        loser_validation_reports,
        loser_validation_predictions,
        pruned_plan: Some(pruned_plan),
        winner_variant_label,
    }))
}

/// Rebuild the PRUNED plan for a chosen operator variant id by re-enumerating the model's variants
/// (deterministic), matching the winner, and pruning the union to its active choice. Mirrors the
/// CLI's `pruned_plan_for_operator_variant` so the in-process winner refits on the pruned candidate
/// rather than the stacking union.
fn pruned_plan_for_operator_variant(
    union_plan: &ExecutionPlan,
    model: &OperatorVariantModel,
    variant_id: &VariantId,
    root_seed: u64,
) -> Result<ExecutionPlan, CoreDagMlError> {
    let variants = enumerate_variants(&model.generation_spec(), Some(root_seed))?;
    let variant = variants
        .iter()
        .find(|variant| &variant.variant_id == variant_id)
        .ok_or_else(|| {
            CoreDagMlError::RuntimeValidation(format!(
                "operator-SELECT winner `{variant_id}` not found in enumerated variants"
            ))
        })?;
    let choice = variant.choices.get(&model.dimension.name).ok_or_else(|| {
        CoreDagMlError::RuntimeValidation(format!(
            "operator winner `{variant_id}` missing operator dimension"
        ))
    })?;
    let active_subsequence = choice.active_subsequence.as_ref().ok_or_else(|| {
        CoreDagMlError::RuntimeValidation(format!(
            "operator winner `{variant_id}` choice has no active_subsequence"
        ))
    })?;
    let active_nodes = model.active_nodes.get(active_subsequence).ok_or_else(|| {
        CoreDagMlError::RuntimeValidation(format!(
            "operator model has no active-node set for `{active_subsequence}`"
        ))
    })?;
    let all_choice_nodes = model
        .active_nodes
        .values()
        .flatten()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    prune_plan_to_active(union_plan, active_nodes, &all_choice_nodes, variant)
}

/// Resolve the variant REFIT targets, mirroring the CLI's `resolve_refit_variant`:
///
/// * when the spec carries operator-variant models (a DSL operator generator), run native
///   OPERATOR-SELECT — each choice scored on its PRUNED plan; the winner FIT_CV + REFITs on its
///   pruned plan (winner-ONLY refit), or
/// * otherwise, a multi-variant plan runs one single-variant FIT_CV per variant and refits the best
///   by `selection_metric` (Mechanism A), or
/// * a single-variant plan refits that variant (or the default when native scoring is off).
fn resolve_refit_variant(
    plan: &ExecutionPlan,
    operator_variant_models: &[OperatorVariantModel],
    run_id: &RunId,
    root_seed: u64,
    selection_metric: RegressionMetricKind,
    runtime_controllers: &RuntimeControllerRegistry,
    data_provider: &InMemoryDataProvider,
) -> Result<ResolvedRefitVariant, CoreDagMlError> {
    if !operator_variant_models.is_empty() {
        if let Some(resolved) = resolve_operator_select(
            plan,
            operator_variant_models,
            run_id,
            root_seed,
            selection_metric,
            runtime_controllers,
            data_provider,
        )? {
            return Ok(resolved);
        }
        // Operator scoring was off (no host targets): fall back to the union plan's default variant,
        // exactly today's behavior for unscored runs.
    }
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
            let loser_validation_predictions = selection
                .variant_validation_predictions
                .into_iter()
                .filter(|captured| captured.variant_id != variant_id)
                .collect();
            return Ok(ResolvedRefitVariant {
                variant_id,
                loser_validation_reports,
                loser_validation_predictions,
                pruned_plan: None,
                winner_variant_label: None,
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
        loser_validation_predictions: Vec::new(),
        pruned_plan: None,
        winner_variant_label: None,
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

/// Stamp the operator-SELECT winner's content fingerprint (Phase 5 `variant_label`) onto EVERY report
/// in the winner's freshly-scored `ScoreSet`. Called BEFORE the loser reports (which already carry
/// their own labels) are merged, so it only touches the winner's reports. A no-op for param-variant /
/// pinned / single-variant refit (`label` is `None`), keeping those paths byte-identical. Mirrors the
/// CLI's `stamp_winner_variant_label`.
fn stamp_winner_variant_label(scores: &mut Option<ScoreSet>, label: Option<String>) {
    let Some(label) = label else {
        return;
    };
    if let Some(score_set) = scores {
        for report in &mut score_set.reports {
            report.variant_label = Some(label.clone());
        }
    }
}

/// Build the additive synthetic frames that surface one LOSER variant's captured VALIDATION (OOF)
/// predictions, each TAGGED with the loser's `variant_id` + `variant_label` so the host routes them
/// to ITS OWN variant (no cross-variant mixing). One frame per per-fold prediction block (paired
/// POSITION-FOR-POSITION with its id-matched y_true), plus one frame for the cross-fold OOF AVERAGE
/// block — exactly the shape the host's `_index_sample_blocks` reads for the winner (per-fold
/// `predictions` + `regression_targets`; the avg as a sample-level `aggregated_predictions` block).
/// The blocks are the loser's OWN validation predictions, surfaced for host display only — they never
/// feed a training/feature path.
fn surface_loser_validation_frames(
    captured: &VariantValidationPredictions,
) -> Vec<serde_json::Value> {
    let mut frames = Vec::new();
    let variant_id = captured.variant_id.as_str();
    for (block, target) in captured
        .predictions
        .iter()
        .zip(captured.regression_targets.iter())
    {
        frames.push(serde_json::json!({
            "node_id": block.producer_node,
            "variant_id": variant_id,
            "variant_label": captured.variant_label,
            "predictions": [block],
            "regression_targets": [target],
        }));
    }
    if let Some(oof) = &captured.oof_average {
        frames.push(serde_json::json!({
            "node_id": oof.predictions.producer_node,
            "variant_id": variant_id,
            "variant_label": captured.variant_label,
            "aggregated_predictions": [oof.predictions],
            "regression_targets": [oof.y_true],
        }));
    }
    frames
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
    // Lower the spec's operator-level generators (Mechanism B) into operator-variant models — the
    // SAME additive derivation the CLI runs (it does not touch the compiled graph / OOF lanes /
    // fingerprints). Empty when the spec has no operator generator. This is what enables the
    // default in-process binding to native operator-SELECT, mirroring CLI Mechanism A.
    let operator_variant_models =
        compile_operator_variant_models(&dsl_spec).map_err(py_core_error)?;
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

    // 4. Mirror `build_bundle_from_cv_then_captured_refit`: native operator-SELECT (when the spec
    //    carries an operator generator) or param-variant SELECT (when multi-variant), then the
    //    winner FIT_CV + REFIT in ONE RunContext, then score. For operator-SELECT the winner runs on
    //    its PRUNED plan (merge + meta-model + inactive choices elided), not the stacking union.
    let resolved = resolve_refit_variant(
        &plan,
        &operator_variant_models,
        &run_id,
        root_seed,
        metric,
        &runtime_controllers,
        &data_provider,
    )
    .map_err(py_core_error)?;
    let selected_variant_id = resolved.variant_id;
    let loser_validation_reports = resolved.loser_validation_reports;
    let loser_validation_predictions = resolved.loser_validation_predictions;
    let winner_variant_label = resolved.winner_variant_label;
    // For operator-SELECT the winner FIT_CV + REFIT run on the WINNER's PRUNED plan; for all other
    // paths the union plan IS the refit plan.
    let refit_plan = resolved.pruned_plan.as_ref().unwrap_or(&plan);

    let mut artifact_store = InMemoryArtifactStore::new();
    let mut ctx = RunContext::new(run_id, Some(root_seed));
    ctx.variant_id = Some(selected_variant_id);

    let fit_cv_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            refit_plan,
            &runtime_controllers,
            &data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .map_err(py_core_error)?;

    let refit_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider_and_artifact_store(
            refit_plan,
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
    ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(refit_plan))
        .map_err(py_core_error)?;
    let mut scores = ctx.build_score_set(refit_plan.id.clone(), None);
    // Phase 5: the winner reports come from the REAL winner FIT_CV/REFIT pass above (not the
    // transient selection loop), so stamp the winner's operator-variant content fingerprint on them
    // BEFORE merging the (already-labeled) loser reports — so the WINNER report carries
    // `variant_label`, not just the losers.
    stamp_winner_variant_label(&mut scores, winner_variant_label);
    merge_loser_validation_reports(&mut scores, &refit_plan.id, loser_validation_reports);

    let mut node_results = fit_cv_results;
    node_results.extend(refit_results);

    // 6. ADDITIVELY surface the per-sample cross-fold OOF AVERAGE so the host fills the
    //    `(validation, avg)` row's y_pred (it had only the scalar OOF report before). Each
    //    `OofAverageBlock` becomes a synthetic NodeResult frame carrying the SAMPLE-level
    //    `aggregated_predictions` block (producer / validation / `avg`) + its id-matched sample-level
    //    `regression_targets` y_true — the exact shape `result._index_sample_blocks` reads. The block
    //    holds the SAME averaged values the scalar was computed from (purely additive; no score,
    //    `num_predictions` or existing block changes), and never feeds a training/feature path.
    let node_results = serde_json::to_value(&node_results).map_err(py_serde_error)?;
    let mut node_results = match node_results {
        serde_json::Value::Array(frames) => frames,
        other => vec![other],
    };
    for oof in &ctx.oof_average_blocks {
        node_results.push(serde_json::json!({
            "node_id": oof.predictions.producer_node,
            "aggregated_predictions": [oof.predictions],
            "regression_targets": [oof.y_true],
        }));
    }

    // 7. ADDITIVELY surface each LOSER variant's per-fold VALIDATION (OOF) predictions so the host can
    //    fill that variant's per-sample prediction rows (it had only the loser's scalar OOF report
    //    before). Each loser's captured per-fold `PredictionBlock` (paired POSITION-FOR-POSITION with
    //    its id-matched y_true) + cross-fold OOF AVERAGE block becomes a synthetic frame TAGGED with
    //    the loser's `variant_id` + `variant_label`, so the host routes a loser's frames to ITS OWN
    //    variant (NO cross-variant mixing). These are the loser's OWN validation (OOF) predictions —
    //    for host persistence/display only, never fed as a training feature / across a `requires_oof`
    //    edge (strictly additive, analogous to the OOF-average block above).
    for captured in &loser_validation_predictions {
        node_results.extend(surface_loser_validation_frames(captured));
    }

    let payload = serde_json::json!({
        "node_results": node_results,
        "scores": scores,
    });
    serde_json::to_string(&payload).map_err(py_serde_error)
}

#[cfg(test)]
mod tests {
    //! Phase 7 (dag-ml part) — operator-SELECT through the DEFAULT in-process binding.
    //!
    //! These mirror the CLI's Phase-4 operator-SELECT tests for the in-process resolution path:
    //! `resolve_refit_variant` runs native operator-SELECT off the lowered models, returns the
    //! winner + its PRUNED plan + the losers' OOF reports (each carrying the Phase-5 `variant_label`),
    //! keeps winner-ONLY refit, and rejects multiple operator generators. The data path uses an empty
    //! `InMemoryDataProvider` (no envelope) over a hand-built union plan + model + mock controllers —
    //! identical in shape to the CLI Phase-4 fixtures — so the test is self-contained and needs no
    //! Python callback (the Python op bridge is exercised by the existing JSON-contract tests).

    use std::collections::{BTreeMap, BTreeSet};

    use dag_ml_core::{
        ArtifactId, ArtifactRef, ControllerManifest, ExecutionPlan, HandleKind, HandleRef,
        LineageId, LineageRecord, NodeId, NodeKind, NodeResult, NodeTask, OperatorVariantModel,
        Phase, PredictionBlock, PredictionLevel, PredictionPartition, PredictionUnitId,
        RegressionMetricKind, RegressionTargetBlock, RunId, RuntimeController,
        RuntimeControllerRegistry, SampleId,
    };

    use super::*;

    // Two distinct, pinned 64-hex `variant_labels` (the cross-language content fingerprints). The
    // derivation is contract-tested in dag-ml-core; here we only assert the labels PROPAGATE from the
    // model onto the winner + loser reports through the in-process resolution path.
    const CHOICE0_LABEL: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const CHOICE1_LABEL: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    #[test]
    fn in_process_callback_result_preserves_closed_object_shapes() {
        Python::initialize();
        Python::attach(|py| {
            let valid = serde_json::json!({
                "handle": 7,
                "kind": "data",
                "owner_controller": "controller:python.strict"
            });
            let valid_object = pythonize(py, &valid).unwrap();
            from_py_object::<HandleRef>(&valid_object).expect("object-form handle is accepted");

            let positional = serde_json::json!([7, "data", "controller:python.strict"]);
            let positional_object = pythonize(py, &positional).unwrap();
            let error = from_py_object::<HandleRef>(&positional_object)
                .expect_err("serde's positional struct form must stay internal");
            assert!(
                error
                    .to_string()
                    .contains("must use a JSON object at the external contract boundary"),
                "{error}"
            );

            let mut unknown = valid;
            unknown.as_object_mut().unwrap().insert(
                "unexpected_contract_field".to_string(),
                serde_json::json!(true),
            );
            let unknown_object = pythonize(py, &unknown).unwrap();
            let error = from_py_object::<HandleRef>(&unknown_object)
                .expect_err("schema-closed callback fields must be rejected");
            assert!(
                error.to_string().contains("unexpected_contract_field"),
                "{error}"
            );

            let mut colliding_keys = serde_json::Map::new();
            colliding_keys.insert("é".to_string(), serde_json::json!(1));
            colliding_keys.insert("e\u{301}".to_string(), serde_json::json!(2));
            let colliding_object =
                pythonize(py, &serde_json::Value::Object(colliding_keys)).unwrap();
            let error = from_py_object::<BTreeMap<String, serde_json::Value>>(&colliding_object)
                .expect_err("NFC-colliding callback map keys must be rejected");
            assert!(error.to_string().contains("NFC-colliding"), "{error}");
        });
    }

    fn stable_handle(node_id: &str) -> u64 {
        let mut hash = 1469598103934665603u64;
        for byte in node_id.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }

    /// A mock model/transform/filter/merge controller mirroring the CLI's `OperatorScoringCliController`:
    /// a model node scores per fold (`model:choice0__pls` predicts y_true exactly -> RMSE 0, the
    /// winner; `model:choice1__ridge` predicts y_true + 1 -> RMSE 1), emits a refit artifact, and the
    /// non-model nodes pass through.
    struct MockController {
        id: ControllerId,
        offsets: BTreeMap<NodeId, f64>,
    }

    impl MockController {
        fn fold_sample(task: &NodeTask) -> Option<(SampleId, f64)> {
            match task.fold_id.as_ref()?.as_str() {
                "fold:0" => Some((SampleId::new("s1").unwrap(), 1.0)),
                "fold:1" => Some((SampleId::new("s2").unwrap(), 2.0)),
                _ => None,
            }
        }
    }

    impl RuntimeController for MockController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult, CoreDagMlError> {
            let is_model = matches!(task.node_plan.kind, NodeKind::Model);
            let data_output = HandleRef {
                handle: stable_handle(task.node_plan.node_id.as_str()),
                kind: HandleKind::Data,
                owner_controller: self.id.clone(),
            };
            let prediction_output = HandleRef {
                handle: stable_handle(task.node_plan.node_id.as_str()),
                kind: HandleKind::Prediction,
                owner_controller: self.id.clone(),
            };
            let mut predictions = Vec::new();
            let mut regression_targets = Vec::new();
            if is_model {
                if task.phase == Phase::FitCv {
                    if let Some((sample_id, y_true)) = Self::fold_sample(task) {
                        let offset = self
                            .offsets
                            .get(&task.node_plan.node_id)
                            .copied()
                            .unwrap_or(0.0);
                        predictions.push(PredictionBlock {
                            prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                            producer_node: task.node_plan.node_id.clone(),
                            producer_port: None,
                            partition: PredictionPartition::Validation,
                            fold_id: task.fold_id.clone(),
                            sample_ids: vec![sample_id.clone()],
                            values: vec![vec![y_true + offset]],
                            target_names: vec!["y".to_string()],
                        });
                        regression_targets.push(RegressionTargetBlock {
                            level: PredictionLevel::Sample,
                            unit_ids: vec![PredictionUnitId::Sample(sample_id)],
                            values: vec![vec![y_true]],
                            target_names: vec!["y".to_string()],
                        });
                    }
                } else {
                    predictions.push(PredictionBlock {
                        prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                        producer_node: task.node_plan.node_id.clone(),
                        producer_port: None,
                        partition: PredictionPartition::Final,
                        fold_id: None,
                        sample_ids: vec![SampleId::new("s1").unwrap()],
                        values: vec![vec![1.0]],
                        target_names: vec!["y".to_string()],
                    });
                }
            }
            let artifacts = if task.phase == Phase::Refit && is_model {
                vec![ArtifactRef {
                    id: ArtifactId::new(format!("artifact:{}:refit", task.node_plan.node_id))
                        .unwrap(),
                    kind: "mock_model".to_string(),
                    controller_id: self.id.clone(),
                    backend: None,
                    uri: None,
                    content_fingerprint: None,
                    size_bytes: Some(128),
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
                            handle: stable_handle(artifact.id.as_str()),
                            kind: HandleKind::Model,
                            owner_controller: self.id.clone(),
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>();
            Ok(NodeResult {
                schema_version: None,
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([
                    ("x".to_string(), data_output.clone()),
                    ("out".to_string(), data_output),
                    ("oof".to_string(), prediction_output),
                ]),
                predictions,
                observation_predictions: Vec::new(),
                aggregated_predictions: Vec::new(),
                explanations: Vec::new(),
                shape_deltas: Vec::new(),
                fit_influence_diagnostics: Vec::new(),
                artifacts: artifacts.clone(),
                artifact_handles,
                regression_targets,
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:{}:{:?}:{}:{}",
                        task.node_plan.node_id,
                        task.phase,
                        task.variant_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "base".to_string()),
                        task.fold_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "nofold".to_string())
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
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
            })
        }
    }

    /// The operator-SELECT UNION plan: a STACKING graph `filter -> choice_i(transform -> model) ->
    /// merge:gen (oof) -> model:meta`, identical in shape to the CLI Phase-4 fixture.
    fn operator_select_union_plan() -> ExecutionPlan {
        let graph: dag_ml_core::GraphSpec = serde_json::from_str(
            r#"{
  "id": "graph:in_process.operator.select",
  "interface": {"inputs": [], "outputs": []},
  "nodes": [
    {"id": "filter:y_outlier", "kind": "exclude", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null},
    {"id": "transform:choice0__snv", "kind": "transform", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null},
    {"id": "model:choice0__pls", "kind": "model", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "oof", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null},
    {"id": "transform:choice1__msc", "kind": "transform", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null},
    {"id": "model:choice1__ridge", "kind": "model", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "oof", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null},
    {"id": "merge:gen", "kind": "prediction_join", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "c0", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""},
                           {"name": "c1", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null},
    {"id": "model:meta", "kind": "model", "operator": null, "params": {},
     "ports": {"inputs": [{"name": "x", "kind": "data", "representation": null, "cardinality": "one", "description": ""}],
               "outputs": [{"name": "oof", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}]},
     "metadata": {}, "seed_label": null}
  ],
  "edges": [
    {"source": {"node_id": "filter:y_outlier", "port_name": "x"}, "target": {"node_id": "transform:choice0__snv", "port_name": "x"},
     "contract": {"kind": "data", "representation": null, "requires_oof": false, "requires_fold_alignment": false, "propagates_lineage": true}},
    {"source": {"node_id": "transform:choice0__snv", "port_name": "x"}, "target": {"node_id": "model:choice0__pls", "port_name": "x"},
     "contract": {"kind": "data", "representation": null, "requires_oof": false, "requires_fold_alignment": false, "propagates_lineage": true}},
    {"source": {"node_id": "filter:y_outlier", "port_name": "x"}, "target": {"node_id": "transform:choice1__msc", "port_name": "x"},
     "contract": {"kind": "data", "representation": null, "requires_oof": false, "requires_fold_alignment": false, "propagates_lineage": true}},
    {"source": {"node_id": "transform:choice1__msc", "port_name": "x"}, "target": {"node_id": "model:choice1__ridge", "port_name": "x"},
     "contract": {"kind": "data", "representation": null, "requires_oof": false, "requires_fold_alignment": false, "propagates_lineage": true}},
    {"source": {"node_id": "model:choice0__pls", "port_name": "oof"}, "target": {"node_id": "merge:gen", "port_name": "c0"},
     "contract": {"kind": "prediction", "representation": null, "requires_oof": true, "requires_fold_alignment": false, "propagates_lineage": true}},
    {"source": {"node_id": "model:choice1__ridge", "port_name": "oof"}, "target": {"node_id": "merge:gen", "port_name": "c1"},
     "contract": {"kind": "prediction", "representation": null, "requires_oof": true, "requires_fold_alignment": false, "propagates_lineage": true}},
    {"source": {"node_id": "merge:gen", "port_name": "x"}, "target": {"node_id": "model:meta", "port_name": "x"},
     "contract": {"kind": "data", "representation": null, "requires_oof": false, "requires_fold_alignment": false, "propagates_lineage": true}}
  ],
  "search_space_fingerprint": null,
  "metadata": {}
}"#,
        )
        .unwrap();
        let campaign: dag_ml_core::CampaignSpec = serde_json::from_str(
            r#"{
  "id": "campaign:in_process.operator.select",
  "root_seed": 7,
  "leakage_policy": {"split_unit": "sample", "forbid_origin_cross_fold": true,
    "allow_observation_split_with_shared_target": false, "require_group_ids": false, "unsafe_flags": []},
  "aggregation_policy": {"aggregation_level": "sample", "method": "mean", "weights": "none",
    "emit_parallel_metrics": true, "selection_metric_level": "sample",
    "store_raw_predictions": true, "store_aggregated_predictions": true},
  "split_invocation": {
    "id": "split:in_process.operator.select", "controller_id": null,
    "leakage_policy": {"split_unit": "sample", "forbid_origin_cross_fold": true,
      "allow_observation_split_with_shared_target": false, "require_group_ids": false, "unsafe_flags": []},
    "params": {},
    "fold_set": {
      "id": "folds:in_process.operator.select",
      "sample_ids": ["s1", "s2"],
      "folds": [
        {"fold_id": "fold:0", "train_sample_ids": ["s2"], "validation_sample_ids": ["s1"], "metadata": {}},
        {"fold_id": "fold:1", "train_sample_ids": ["s1"], "validation_sample_ids": ["s2"], "metadata": {}}
      ],
      "sample_groups": {}
    }
  },
  "generation": {"strategy": "none", "dimensions": [], "max_variants": 1},
  "shape_plans": {},
  "data_bindings": {},
  "metadata": {}
}"#,
        )
        .unwrap();
        let mut manifests = ControllerRegistry::new();
        for json in [
            r#"{"controller_id": "controller:filter", "controller_version": "0.1.0", "operator_kind": "exclude",
               "priority": 0, "supported_phases": ["FIT_CV", "REFIT", "PREDICT"], "input_ports": [], "output_ports": [],
               "data_requirements": null, "capabilities": ["deterministic", "thread_safe", "process_safe"],
               "fit_scope": "fold_train", "rng_policy": "uses_core_seed", "artifact_policy": "serializable"}"#,
            r#"{"controller_id": "controller:transform", "controller_version": "0.1.0", "operator_kind": "transform",
               "priority": 0, "supported_phases": ["FIT_CV", "REFIT", "PREDICT"], "input_ports": [], "output_ports": [],
               "data_requirements": null, "capabilities": ["deterministic", "thread_safe", "process_safe"],
               "fit_scope": "fold_train", "rng_policy": "uses_core_seed", "artifact_policy": "serializable"}"#,
            r#"{"controller_id": "controller:model", "controller_version": "0.1.0", "operator_kind": "model",
               "priority": 0, "supported_phases": ["FIT_CV", "REFIT", "PREDICT"], "input_ports": [], "output_ports": [],
               "data_requirements": null, "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions", "consumes_oof_predictions", "emits_artifacts", "stateful"],
               "fit_scope": "fold_train", "rng_policy": "uses_core_seed", "artifact_policy": "serializable"}"#,
            r#"{"controller_id": "controller:merge", "controller_version": "0.1.0", "operator_kind": "prediction_join",
               "priority": 0, "supported_phases": ["FIT_CV", "REFIT", "PREDICT"], "input_ports": [], "output_ports": [],
               "data_requirements": null, "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions", "consumes_oof_predictions"],
               "fit_scope": "fold_train", "rng_policy": "uses_core_seed", "artifact_policy": "serializable"}"#,
        ] {
            manifests
                .register(serde_json::from_str::<ControllerManifest>(json).unwrap())
                .unwrap();
        }
        build_execution_plan(
            "plan:in_process.operator.select",
            graph,
            campaign,
            &manifests,
        )
        .unwrap()
    }

    /// The operator-variant model with Phase-5 `variant_labels` populated (pinned valid 64-hex
    /// fingerprints), one per choice.
    fn operator_select_model() -> OperatorVariantModel {
        let model: OperatorVariantModel = serde_json::from_str(&format!(
            r#"{{
              "generator_id": "generator:preproc_model",
              "dimension": {{
                "name": "generator:preproc_model.operators",
                "choices": [
                  {{"label": "choice0", "value": "choice0", "active_subsequence": "choice0"}},
                  {{"label": "choice1", "value": "choice1", "active_subsequence": "choice1"}}
                ]
              }},
              "active_nodes": {{
                "choice0": ["transform:choice0__snv", "model:choice0__pls"],
                "choice1": ["transform:choice1__msc", "model:choice1__ridge"]
              }},
              "variant_labels": {{
                "choice0": "{CHOICE0_LABEL}",
                "choice1": "{CHOICE1_LABEL}"
              }}
            }}"#
        ))
        .unwrap();
        model.validate().unwrap();
        model
    }

    fn operator_select_controllers() -> RuntimeControllerRegistry {
        let mut registry = RuntimeControllerRegistry::new();
        for id in [
            "controller:filter",
            "controller:transform",
            "controller:merge",
        ] {
            registry
                .register(Box::new(MockController {
                    id: ControllerId::new(id).unwrap(),
                    offsets: BTreeMap::new(),
                }))
                .unwrap();
        }
        registry
            .register(Box::new(MockController {
                id: ControllerId::new("controller:model").unwrap(),
                offsets: BTreeMap::from([
                    (NodeId::new("model:choice0__pls").unwrap(), 0.0),
                    (NodeId::new("model:choice1__ridge").unwrap(), 1.0),
                ]),
            }))
            .unwrap();
        registry
    }

    fn empty_provider() -> InMemoryDataProvider {
        InMemoryDataProvider::new(ControllerId::new("controller:data.provider").unwrap())
    }

    #[test]
    fn in_process_resolve_refit_variant_runs_operator_select_and_labels_reports() {
        // Mirrors the CLI Phase-4 operator-SELECT path through the IN-PROCESS resolver: the winner is
        // the lower-RMSE choice (choice0, offset 0); the loser report carries its Phase-5
        // variant_label; the winner runs FIT_CV + REFIT on its PRUNED plan (merge + meta + sibling
        // elided); and the WINNER's content fingerprint comes back so it can be stamped on the winner
        // reports.
        let union_plan = operator_select_union_plan();
        let model = operator_select_model();
        let controllers = operator_select_controllers();
        let provider = empty_provider();
        let run_id = RunId::new("run:in_process.operator.select").unwrap();

        let resolved = resolve_refit_variant(
            &union_plan,
            std::slice::from_ref(&model),
            &run_id,
            7,
            RegressionMetricKind::Rmse,
            &controllers,
            &provider,
        )
        .expect("in-process operator-SELECT must succeed");

        // (a) The winner runs on the PRUNED plan — merge + meta + the losing sibling are elided.
        let pruned = resolved
            .pruned_plan
            .as_ref()
            .expect("operator-SELECT must thread out the pruned winner plan");
        assert!(pruned
            .node_plans
            .contains_key(&NodeId::new("model:choice0__pls").unwrap()));
        for elided in ["merge:gen", "model:meta", "model:choice1__ridge"] {
            assert!(
                !pruned
                    .node_plans
                    .contains_key(&NodeId::new(elided).unwrap()),
                "`{elided}` must be elided from the pruned winner plan"
            );
        }

        // (b) The winner's content fingerprint is choice0's label (so it can stamp the winner reports).
        assert_eq!(
            resolved.winner_variant_label.as_deref(),
            Some(CHOICE0_LABEL),
            "the winner's variant_label must be the choice0 content fingerprint"
        );

        // (c) Loser reports each carry their OWN variant_label (the losing choice1's fingerprint).
        assert!(
            !resolved.loser_validation_reports.is_empty(),
            "the losing choice must surface its OOF reports"
        );
        for report in &resolved.loser_validation_reports {
            assert_eq!(
                report.partition,
                PredictionPartition::Validation,
                "operator-SELECT retains only Validation (OOF) reports"
            );
            assert_eq!(
                report.variant_label.as_deref(),
                Some(CHOICE1_LABEL),
                "every loser report must carry the loser choice's variant_label"
            );
        }
    }

    #[test]
    fn in_process_surfaces_loser_validation_predictions_tagged_by_variant() {
        // The in-process resolver carries the LOSER variant's per-fold VALIDATION (OOF) predictions
        // (not just its scalar reports), and `surface_loser_validation_frames` turns them into
        // synthetic frames TAGGED with the loser's variant_id + variant_label — what lets the host fill
        // the loser's per-fold val rows. The winner's predictions are NOT carried here (they come fresh
        // from the real FIT_CV pass); only the loser's Validation (OOF) predictions are surfaced.
        let union_plan = operator_select_union_plan();
        let model = operator_select_model();
        let controllers = operator_select_controllers();
        let provider = empty_provider();
        let run_id = RunId::new("run:in_process.operator.loser.predictions").unwrap();

        let resolved = resolve_refit_variant(
            &union_plan,
            std::slice::from_ref(&model),
            &run_id,
            7,
            RegressionMetricKind::Rmse,
            &controllers,
            &provider,
        )
        .unwrap();

        // Only the LOSER (choice1) surfaces captured predictions — the winner's come fresh below.
        assert_eq!(
            resolved.loser_validation_predictions.len(),
            1,
            "exactly the single loser variant surfaces captured predictions"
        );
        let loser = &resolved.loser_validation_predictions[0];
        assert_ne!(
            loser.variant_id, resolved.variant_id,
            "the carried predictions belong to the LOSER, not the winner"
        );
        assert_eq!(
            loser.variant_label.as_deref(),
            Some(CHOICE1_LABEL),
            "the loser's captured predictions carry the loser choice's content fingerprint"
        );
        assert!(
            !loser.predictions.is_empty()
                && loser
                    .predictions
                    .iter()
                    .all(|block| block.partition == PredictionPartition::Validation),
            "only the loser's Validation (OOF) predictions are surfaced — no refit/train/test"
        );

        // The synthetic frames carry the loser's variant tag (so the host routes them to ITS variant)
        // and exactly the captured per-fold prediction + paired y_true (id-matched).
        let frames = surface_loser_validation_frames(loser);
        assert!(
            !frames.is_empty(),
            "the loser surfaces at least one validation frame"
        );
        for frame in &frames {
            assert_eq!(
                frame.get("variant_id").and_then(|value| value.as_str()),
                Some(loser.variant_id.as_str()),
                "every surfaced loser frame is tagged with the loser's variant_id"
            );
            assert_eq!(
                frame.get("variant_label").and_then(|value| value.as_str()),
                Some(CHOICE1_LABEL),
                "every surfaced loser frame is tagged with the loser's variant_label"
            );
        }
        // A per-fold frame pairs `predictions[0]` with `regression_targets[0]` (the host reads them
        // position-for-position), covering the SAME samples (id-matched).
        let fold_frame = frames
            .iter()
            .find(|frame| frame.get("predictions").is_some())
            .expect("a per-fold prediction frame is surfaced");
        let pred = &fold_frame["predictions"][0];
        let target = &fold_frame["regression_targets"][0];
        assert_eq!(
            pred["partition"].as_str(),
            Some("validation"),
            "the surfaced per-fold block is a Validation (OOF) block"
        );
        let pred_ids: BTreeSet<String> = pred["sample_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();
        let target_ids: BTreeSet<String> = target["unit_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|unit| unit["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            pred_ids, target_ids,
            "the per-fold y_true covers exactly the prediction block's samples (id-matched)"
        );
    }

    #[test]
    fn in_process_winner_report_carries_variant_label() {
        // (3) The WINNER report (not just losers) carries variant_label: build the winner's freshly
        // scored ScoreSet (real FIT_CV + REFIT on the pruned plan) and stamp it with the resolved
        // winner label — exactly what `run_cv_refit_in_process` does before merging the losers.
        let union_plan = operator_select_union_plan();
        let model = operator_select_model();
        let controllers = operator_select_controllers();
        let provider = empty_provider();
        let run_id = RunId::new("run:in_process.operator.winner").unwrap();

        let resolved = resolve_refit_variant(
            &union_plan,
            std::slice::from_ref(&model),
            &run_id,
            7,
            RegressionMetricKind::Rmse,
            &controllers,
            &provider,
        )
        .unwrap();
        let refit_plan = resolved.pruned_plan.as_ref().unwrap();

        let mut ctx = RunContext::new(run_id, Some(7));
        ctx.variant_id = Some(resolved.variant_id.clone());
        SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                refit_plan,
                &controllers,
                &provider,
                &mut ctx,
                Phase::FitCv,
            )
            .unwrap();
        let mut artifact_store = InMemoryArtifactStore::new();
        SequentialScheduler
            .execute_campaign_phase_with_data_provider_and_artifact_store(
                refit_plan,
                &controllers,
                &provider,
                &mut artifact_store,
                &mut ctx,
                Phase::Refit,
            )
            .unwrap();
        ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(refit_plan))
            .unwrap();
        let mut scores = ctx.build_score_set(refit_plan.id.clone(), None);
        stamp_winner_variant_label(&mut scores, resolved.winner_variant_label.clone());

        let score_set = scores.expect("the winner must produce a cross-fold OOF score");
        assert!(
            score_set
                .reports
                .iter()
                .any(|report| report.variant_label.as_deref() == Some(CHOICE0_LABEL)),
            "the WINNER report must carry the winner's variant_label, not just the losers"
        );
        assert!(
            score_set
                .reports
                .iter()
                .all(|report| report.variant_label.as_deref() == Some(CHOICE0_LABEL)),
            "every winner report carries the winner's variant_label"
        );
    }

    #[test]
    fn in_process_resolve_refit_variant_rejects_multiple_operator_generators() {
        // (6) Multiple operator generators are rejected for this phase (flat single operator generator
        // scope), exactly as the CLI / core do.
        let union_plan = operator_select_union_plan();
        let model = operator_select_model();
        let mut second = model.clone();
        second.generator_id = NodeId::new("generator:other").unwrap();
        second.dimension.name = "generator:other.operators".to_string();
        let models = vec![model, second];
        let controllers = operator_select_controllers();
        let provider = empty_provider();
        let run_id = RunId::new("run:in_process.operator.multi").unwrap();

        let error = resolve_refit_variant(
            &union_plan,
            &models,
            &run_id,
            7,
            RegressionMetricKind::Rmse,
            &controllers,
            &provider,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("does not support 2 operator generators"),
            "multiple operator generators must be rejected: {error}"
        );
    }
}
