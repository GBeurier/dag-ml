//! C ABI for a no-splitter initial REFIT package and its independent PREDICT replay.

use super::*;
use dag_ml_core::{
    execute_initial_full_refit, execute_initial_full_refit_prediction, ArtifactId,
    ExplicitPhaseDataProvider, InitialFullRefitExecutionInput, InitialRefitReplayInput,
    InitialRefitScheduler,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlInitialFullRefitExecuteRequest {
    pub plan_json: DagMlBytesView,
    pub envelope_json: DagMlBytesView,
    pub trusted_controllers_json: DagMlBytesView,
    pub training_sample_ids_json: DagMlBytesView,
    pub package_id: DagMlBytesView,
    pub run_id: DagMlBytesView,
    pub root_seed: u64,
    pub controller_bindings: *const DagMlControllerBinding,
    pub controller_binding_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlInitialFullRefitPredictRequest {
    pub package_json: DagMlBytesView,
    pub envelope_json: DagMlBytesView,
    pub output_ids_json: DagMlBytesView,
    pub artifact_handles_json: DagMlBytesView,
    pub run_id: DagMlBytesView,
    pub controller_bindings: *const DagMlControllerBinding,
    pub controller_binding_count: usize,
}

/// Execute REFIT once with host controllers, returning a closed initial package.
/// Host-sidecar model bytes and handles remain owned by the caller's controllers.
///
/// # Safety
/// `request` and its views must remain valid for the call; release returned
/// `out_json` with `dagml_owned_bytes_free` and errors with `dagml_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_initial_full_refit_execute_json(
    request: *const DagMlInitialFullRefitExecuteRequest,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        initial_full_refit_execute_json_impl(request, out_json, error_out)
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(
                error_out,
                "panic while executing initial full-refit through C ABI",
            );
            DagMlStatusCode::PANIC
        }
    }
}

unsafe fn initial_full_refit_execute_json_impl(
    request: *const DagMlInitialFullRefitExecuteRequest,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let Some(request) = request.as_ref() else {
        set_error(error_out, "initial full-refit execute request is null");
        return DagMlStatusCode::INVALID_ARGUMENT;
    };
    let plan = match parse_external_contract_ptr(
        request.plan_json.ptr,
        request.plan_json.len,
        error_out,
        "initial full-refit execution plan",
        ExecutionPlan::from_json,
    ) {
        Ok(plan) => plan,
        Err(status) => return status,
    };
    let envelope: ExternalDataPlanEnvelope = match parse_json_ptr(
        request.envelope_json.ptr,
        request.envelope_json.len,
        error_out,
        "initial full-refit data envelope",
    ) {
        Ok(envelope) => envelope,
        Err(status) => return status,
    };
    let manifests: Vec<ControllerManifest> = match parse_json_ptr(
        request.trusted_controllers_json.ptr,
        request.trusted_controllers_json.len,
        error_out,
        "trusted initial full-refit controllers",
    ) {
        Ok(manifests) => manifests,
        Err(status) => return status,
    };
    let trusted = match controller_registry_from_manifests(manifests) {
        Ok(registry) => registry,
        Err(error) => return validation_error(error_out, error),
    };
    if let Err(error) = validate_execution_plan_controller_manifests(&plan, &trusted) {
        return validation_error(error_out, error);
    }
    let training_sample_ids: Vec<SampleId> = match parse_json_ptr(
        request.training_sample_ids_json.ptr,
        request.training_sample_ids_json.len,
        error_out,
        "initial full-refit training sample IDs",
    ) {
        Ok(ids) => ids,
        Err(status) => return status,
    };
    let package_id = match parse_utf8_view(
        request.package_id,
        error_out,
        "initial full-refit package id",
    ) {
        Ok(id) => id,
        Err(status) => return status,
    };
    let run_id = match parse_run_id_view(request.run_id, error_out, "initial full-refit run id") {
        Ok(id) => id,
        Err(status) => return status,
    };
    let provider = match ExplicitPhaseDataProvider::new(
        ControllerId::new("controller:data.provider").expect("static id"),
        envelope.clone(),
        Some(training_sample_ids.clone()),
    ) {
        Ok(provider) => provider,
        Err(error) => return validation_error(error_out, error),
    };
    let controllers = match build_controller_registry(
        request.controller_bindings,
        request.controller_binding_count,
        error_out,
    ) {
        Ok(registry) => registry,
        Err(status) => return status,
    };
    match execute_initial_full_refit(InitialFullRefitExecutionInput {
        package_id,
        run_id,
        plan: &plan,
        training_envelope: &envelope,
        training_sample_ids: &training_sample_ids,
        controllers: &controllers,
        data_provider: &provider,
        root_seed: Some(request.root_seed),
        resource_limits: None,
        scheduler: InitialRefitScheduler::Sequential,
    }) {
        Ok(execution) => write_owned_json(
            out_json,
            error_out,
            &serde_json::json!({
                "initial_full_refit_package": execution.package,
                "node_results": execution.results, "scores": execution.scores,
            }),
        ),
        Err(error) => validation_error(error_out, error),
    }
}

/// Replay explicit package outputs for a fresh V2 cohort with host sidecars.
/// Raw native payloads are hydrated by the owning controller from the package.
///
/// # Safety
/// Same ownership and pointer rules as `dagml_initial_full_refit_execute_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_initial_full_refit_predict_json(
    request: *const DagMlInitialFullRefitPredictRequest,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        initial_full_refit_predict_json_impl(request, out_json, error_out)
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(
                error_out,
                "panic while replaying initial full-refit through C ABI",
            );
            DagMlStatusCode::PANIC
        }
    }
}

unsafe fn initial_full_refit_predict_json_impl(
    request: *const DagMlInitialFullRefitPredictRequest,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let Some(request) = request.as_ref() else {
        set_error(error_out, "initial full-refit predict request is null");
        return DagMlStatusCode::INVALID_ARGUMENT;
    };
    let package = match parse_external_contract_ptr(
        request.package_json.ptr,
        request.package_json.len,
        error_out,
        "initial full-refit package",
        InitialFullRefitPackage::from_json,
    ) {
        Ok(package) => package,
        Err(status) => return status,
    };
    let envelope: ExternalDataPlanEnvelope = match parse_json_ptr(
        request.envelope_json.ptr,
        request.envelope_json.len,
        error_out,
        "initial full-refit predict envelope",
    ) {
        Ok(envelope) => envelope,
        Err(status) => return status,
    };
    let output_ids: Vec<String> = match parse_json_ptr(
        request.output_ids_json.ptr,
        request.output_ids_json.len,
        error_out,
        "initial full-refit output IDs",
    ) {
        Ok(ids) => ids,
        Err(status) => return status,
    };
    let handles: BTreeMap<ArtifactId, HandleRef> = match parse_json_ptr(
        request.artifact_handles_json.ptr,
        request.artifact_handles_json.len,
        error_out,
        "initial full-refit sidecar handles",
    ) {
        Ok(handles) => handles,
        Err(status) => return status,
    };
    let expected = package
        .artifacts
        .iter()
        .filter(|artifact| artifact.load_mode == dag_ml_core::ArtifactLoadMode::HostSidecar)
        .map(|artifact| &artifact.record.artifact.id)
        .collect::<BTreeSet<_>>();
    if handles.keys().collect::<BTreeSet<_>>() != expected {
        return validation_error(
            error_out,
            DagMlError::RuntimeValidation(
                "initial full-refit handles must exactly cover host-sidecar artifacts".into(),
            ),
        );
    }
    let run_id = match parse_run_id_view(
        request.run_id,
        error_out,
        "initial full-refit predict run id",
    ) {
        Ok(id) => id,
        Err(status) => return status,
    };
    let provider = match ExplicitPhaseDataProvider::new(
        ControllerId::new("controller:data.provider").expect("static id"),
        envelope.clone(),
        None,
    ) {
        Ok(provider) => provider,
        Err(error) => return validation_error(error_out, error),
    };
    let controllers = match build_controller_registry(
        request.controller_bindings,
        request.controller_binding_count,
        error_out,
    ) {
        Ok(registry) => registry,
        Err(status) => return status,
    };
    let mut store = InMemoryArtifactStore::new();
    for artifact in &package.artifacts {
        if artifact.load_mode != dag_ml_core::ArtifactLoadMode::HostSidecar {
            continue;
        }
        if let Err(error) = store.register(
            &artifact.record,
            handles[&artifact.record.artifact.id].clone(),
        ) {
            return validation_error(error_out, error);
        }
    }
    match execute_initial_full_refit_prediction(InitialRefitReplayInput {
        package: &package,
        envelope: &envelope,
        output_ids: &output_ids,
        run_id,
        controllers: &controllers,
        data_provider: &provider,
        artifact_store: &store,
    }) {
        Ok(replay) => write_owned_json(
            out_json,
            error_out,
            &serde_json::json!({
                "replay_outcome": replay.outcome, "node_results": replay.results,
            }),
        ),
        Err(error) => validation_error(error_out, error),
    }
}
