//! Process-local C callback registry for loss and metric implementations.

use std::ffi::c_void;
use std::slice;
use std::sync::{Arc, Mutex, MutexGuard};

use dag_ml_core::{
    deserialize_external_contract, DagMlError, ImplementationDescriptor,
    LocalImplementationRegistry as CoreLocalImplementationRegistry, LossExecutionAttestation,
    LossReference, MetricReference, NodeTask, Phase, PortabilityClass, TrainingLossRoleReference,
};

use super::{
    clear_error, clear_owned_bytes, parse_typed_json, parse_utf8_view, set_error, validation_error,
    write_owned_vec, DagMlBytesView, DagMlOwnedBytes, DagMlStatusCode, DagMlString,
};

pub const DAG_ML_LOCAL_IMPLEMENTATION_VTABLE_ABI_VERSION: u32 = 1;

/// Host callback retained by a process-local loss or metric registry.
///
/// `retain` and `release` are optional but must either both be present or both
/// be absent. When present, a successful registration retains `user_data` once
/// and unregister, clear, or registry destruction releases it once.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlLocalImplementationVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub invoke: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            request_json: DagMlBytesView,
            out_result_json: *mut DagMlOwnedBytes,
        ) -> DagMlStatusCode,
    >,
    pub release_bytes: Option<unsafe extern "C" fn(user_data: *mut c_void, bytes: DagMlOwnedBytes)>,
    pub retain: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

struct LocalCallback {
    vtable: DagMlLocalImplementationVTable,
    retained: bool,
}

// The registry only copies the opaque pointer and forwards immutable JSON to
// the supplied function table. As with controller vtables, the host must make
// `user_data` safe for the scheduler modes/capabilities it advertises.
unsafe impl Send for LocalCallback {}
unsafe impl Sync for LocalCallback {}

impl LocalCallback {
    fn new(vtable: DagMlLocalImplementationVTable) -> dag_ml_core::Result<Self> {
        if vtable.abi_version != DAG_ML_LOCAL_IMPLEMENTATION_VTABLE_ABI_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "local implementation callback ABI version {} is unsupported; expected {}",
                vtable.abi_version, DAG_ML_LOCAL_IMPLEMENTATION_VTABLE_ABI_VERSION
            )));
        }
        if vtable.invoke.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "local implementation vtable is missing invoke".to_string(),
            ));
        }
        if vtable.release_bytes.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "local implementation vtable is missing release_bytes".to_string(),
            ));
        }
        if vtable.retain.is_some() != vtable.release.is_some() {
            return Err(DagMlError::RuntimeValidation(
                "local implementation vtable retain and release callbacks must be provided together"
                    .to_string(),
            ));
        }
        let retained = !vtable.user_data.is_null() && vtable.retain.is_some();
        if !vtable.user_data.is_null() {
            if let Some(retain) = vtable.retain {
                unsafe { retain(vtable.user_data) };
            }
        }
        Ok(Self { vtable, retained })
    }

    fn invoke(&self, request_json: DagMlBytesView) -> dag_ml_core::Result<Vec<u8>> {
        let invoke = self.vtable.invoke.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "local implementation vtable is missing invoke".to_string(),
            )
        })?;
        let release_bytes = self.vtable.release_bytes.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "local implementation vtable is missing release_bytes".to_string(),
            )
        })?;
        let mut output = DagMlOwnedBytes::default();
        let status = unsafe { invoke(self.vtable.user_data, request_json, &mut output) };
        let data = if output.ptr.is_null() {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(output.ptr, output.len) }.to_vec()
        };
        if !output.ptr.is_null() {
            unsafe { release_bytes(self.vtable.user_data, output) };
        }

        if status != DagMlStatusCode::OK {
            let detail = std::str::from_utf8(&data)
                .ok()
                .filter(|value| !value.is_empty())
                .map(|value| format!(": {value}"))
                .unwrap_or_default();
            let label = match status {
                DagMlStatusCode::INVALID_ARGUMENT => "invalid argument",
                DagMlStatusCode::VALIDATION_ERROR => "validation error",
                DagMlStatusCode::PANIC => "host exception or panic",
                _ => "unknown status",
            };
            return Err(DagMlError::RuntimeValidation(format!(
                "local implementation callback returned {label} ({}){detail}",
                status.0
            )));
        }
        if data.is_empty() {
            return Err(DagMlError::RuntimeValidation(
                "local implementation callback returned empty result JSON".to_string(),
            ));
        }
        let raw = std::str::from_utf8(&data).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "local implementation callback returned invalid UTF-8: {error}"
            ))
        })?;
        parse_typed_json(raw).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "local implementation callback returned invalid result JSON: {error}"
            ))
        })?;
        Ok(data)
    }
}

impl Drop for LocalCallback {
    fn drop(&mut self) {
        if self.retained && !self.vtable.user_data.is_null() {
            if let Some(release) = self.vtable.release {
                unsafe { release(self.vtable.user_data) };
            }
        }
    }
}

/// Opaque process-local registry. Executable callbacks never enter JSON.
pub struct DagMlLocalImplementationRegistry {
    binding_id: String,
    registry: Mutex<CoreLocalImplementationRegistry<Arc<LocalCallback>>>,
}

/// Creates a registry scoped to one host binding identity.
///
/// # Safety
///
/// `binding_id` must be a valid UTF-8 view. `out_registry` must point to
/// writable memory and must later be released with
/// [`dagml_local_implementation_registry_free`].
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_create(
    binding_id: DagMlBytesView,
    out_registry: *mut *mut DagMlLocalImplementationRegistry,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    if !out_registry.is_null() {
        *out_registry = std::ptr::null_mut();
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out_registry.is_null() {
            set_error(
                error_out,
                "local implementation registry output pointer is null",
            );
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let binding_id = match parse_utf8_view(binding_id, error_out, "binding id") {
            Ok(binding_id) => binding_id,
            Err(status) => return status,
        };
        if !valid_binding_id(&binding_id) {
            set_error(
                error_out,
                "binding id must be canonical non-blank text beginning with `binding:`",
            );
            return DagMlStatusCode::VALIDATION_ERROR;
        }
        *out_registry = Box::into_raw(Box::new(DagMlLocalImplementationRegistry {
            binding_id,
            registry: Mutex::new(CoreLocalImplementationRegistry::new()),
        }));
        DagMlStatusCode::OK
    })) {
        Ok(status) => status,
        Err(_) => panic_status(
            error_out,
            "panic while creating local implementation registry through C ABI",
        ),
    }
}

/// Registers a loss callback under an exact `LossReference` descriptor.
///
/// # Safety
///
/// `registry` must be a live registry pointer created by DAG-ML. JSON views and
/// callback pointers must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_register_loss(
    registry: *mut DagMlLocalImplementationRegistry,
    loss_reference_json: DagMlBytesView,
    implementation: DagMlLocalImplementationVTable,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    registration_boundary(error_out, "loss", || {
        let registry = registry_ref(registry)?;
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_local_descriptor(registry, &loss.implementation)?;
        let implementation = Arc::new(LocalCallback::new(implementation)?);
        registry_lock(registry)?.register_loss(&loss, implementation)
    })
}

/// Registers a metric callback under an exact `MetricReference` descriptor.
///
/// # Safety
///
/// The pointer and view requirements match
/// [`dagml_local_implementation_registry_register_loss`].
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_register_metric(
    registry: *mut DagMlLocalImplementationRegistry,
    metric_reference_json: DagMlBytesView,
    implementation: DagMlLocalImplementationVTable,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    registration_boundary(error_out, "metric", || {
        let registry = registry_ref(registry)?;
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_local_descriptor(registry, &metric.implementation)?;
        let implementation = Arc::new(LocalCallback::new(implementation)?);
        registry_lock(registry)?.register_metric(&metric, implementation)
    })
}

/// Invokes a registered local loss with an opaque, strict JSON request.
///
/// # Safety
///
/// `out_result_json` must point to writable memory. Returned bytes are owned by
/// DAG-ML and must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_invoke_loss(
    registry: *mut DagMlLocalImplementationRegistry,
    loss_reference_json: DagMlBytesView,
    request_json: DagMlBytesView,
    out_result_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    invocation_boundary(out_result_json, error_out, "loss", || {
        let registry = registry_ref(registry)?;
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_local_descriptor(registry, &loss.implementation)?;
        validate_invocation_json(request_json)?;
        let callback = registry_lock(registry)?.resolve_loss(&loss)?.clone();
        callback.invoke(request_json)
    })
}

/// Resolves and invokes a training loss for `FIT_CV` or `REFIT`, returning the
/// common execution attestation only after the callback succeeds.
///
/// # Safety
///
/// Both output pointers must be writable. Both outputs are DAG-ML-owned bytes.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_invoke_training_loss(
    registry: *mut DagMlLocalImplementationRegistry,
    training_loss_role_json: DagMlBytesView,
    phase: DagMlBytesView,
    request_json: DagMlBytesView,
    out_result_json: *mut DagMlOwnedBytes,
    out_attestation_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    training_invocation_boundary(
        out_result_json,
        out_attestation_json,
        error_out,
        "training loss invocation",
        || {
            let registry = registry_ref(registry)?;
            let role = parse_training_loss_role(training_loss_role_json)?;
            validate_local_descriptor(registry, &role.loss.implementation)?;
            let phase = parse_phase_for_registry(phase)?;
            let attestation = LossExecutionAttestation::for_role(&role, phase)?;
            validate_invocation_json(request_json)?;
            let callback = registry_lock(registry)?.resolve_loss(&role.loss)?.clone();
            let result = callback.invoke(request_json)?;
            let attestation = serde_json::to_vec(&attestation).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to serialize loss execution attestation: {error}"
                ))
            })?;
            Ok((result, attestation))
        },
    )
}

/// Selects one active loss directly from an exact native `NodeTask`.
///
/// This validation-only boundary is intended for host runtimes such as R and
/// MATLAB that keep numerical arrays and executable functions in the host. It
/// returns the native role and task-owned attestation without invoking or
/// serializing the host function. `role_index` is zero-based within the task's
/// phase-filtered loss roles.
///
/// # Safety
///
/// Both output pointers must be writable. Both outputs are DAG-ML-owned bytes.
#[no_mangle]
pub unsafe extern "C" fn dagml_node_task_training_loss_binding(
    node_task_json: DagMlBytesView,
    role_index: usize,
    out_training_loss_role_json: *mut DagMlOwnedBytes,
    out_attestation_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    training_invocation_boundary(
        out_training_loss_role_json,
        out_attestation_json,
        error_out,
        "training loss binding",
        || {
            let task = parse_node_task(node_task_json)?;
            let (role, attestation) = task.training_loss_binding(role_index)?;
            let role = serde_json::to_vec(role).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to serialize task training loss role: {error}"
                ))
            })?;
            let attestation = serde_json::to_vec(attestation).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to serialize task loss execution attestation: {error}"
                ))
            })?;
            Ok((role, attestation))
        },
    )
}

/// Resolves and invokes one active loss directly from an exact native
/// `NodeTask`, returning the task-owned attestation only after callback success.
///
/// `role_index` is zero-based within the task's phase-filtered loss roles.
///
/// # Safety
///
/// Both output pointers must be writable. Both outputs are DAG-ML-owned bytes.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_invoke_task_training_loss(
    registry: *mut DagMlLocalImplementationRegistry,
    node_task_json: DagMlBytesView,
    role_index: usize,
    request_json: DagMlBytesView,
    out_result_json: *mut DagMlOwnedBytes,
    out_attestation_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    training_invocation_boundary(
        out_result_json,
        out_attestation_json,
        error_out,
        "training loss invocation",
        || {
            let registry = registry_ref(registry)?;
            let task = parse_node_task(node_task_json)?;
            let (role, attestation) = task.training_loss_binding(role_index)?;
            validate_local_descriptor(registry, &role.loss.implementation)?;
            validate_invocation_json(request_json)?;
            let callback = registry_lock(registry)?.resolve_loss(&role.loss)?.clone();
            let result = callback.invoke(request_json)?;
            let attestation = serde_json::to_vec(attestation).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to serialize task loss execution attestation: {error}"
                ))
            })?;
            Ok((result, attestation))
        },
    )
}

/// Invokes a registered local metric with an opaque, strict JSON request.
///
/// # Safety
///
/// Pointer and ownership requirements match
/// [`dagml_local_implementation_registry_invoke_loss`].
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_invoke_metric(
    registry: *mut DagMlLocalImplementationRegistry,
    metric_reference_json: DagMlBytesView,
    request_json: DagMlBytesView,
    out_result_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    invocation_boundary(out_result_json, error_out, "metric", || {
        let registry = registry_ref(registry)?;
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_local_descriptor(registry, &metric.implementation)?;
        validate_invocation_json(request_json)?;
        let callback = registry_lock(registry)?.resolve_metric(&metric)?.clone();
        callback.invoke(request_json)
    })
}

/// Unregisters a loss and releases its retained callback state.
///
/// # Safety
///
/// `registry` and the JSON view must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_unregister_loss(
    registry: *mut DagMlLocalImplementationRegistry,
    loss_reference_json: DagMlBytesView,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    registration_boundary(error_out, "loss unregistration", || {
        let registry = registry_ref(registry)?;
        let loss = parse_loss_reference(loss_reference_json)?;
        validate_local_descriptor(registry, &loss.implementation)?;
        let removed = registry_lock(registry)?.unregister(&loss.implementation)?;
        drop(removed);
        Ok(())
    })
}

/// Unregisters a metric and releases its retained callback state.
///
/// # Safety
///
/// Pointer requirements match
/// [`dagml_local_implementation_registry_unregister_loss`].
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_unregister_metric(
    registry: *mut DagMlLocalImplementationRegistry,
    metric_reference_json: DagMlBytesView,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    registration_boundary(error_out, "metric unregistration", || {
        let registry = registry_ref(registry)?;
        let metric = parse_metric_reference(metric_reference_json)?;
        validate_local_descriptor(registry, &metric.implementation)?;
        let removed = registry_lock(registry)?.unregister(&metric.implementation)?;
        drop(removed);
        Ok(())
    })
}

/// Returns the exact registered implementation descriptors as a JSON array.
///
/// # Safety
///
/// `out_json` must be writable and its returned bytes must be released with
/// `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_descriptors_json(
    registry: *mut DagMlLocalImplementationRegistry,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    invocation_boundary(out_json, error_out, "descriptor inspection", || {
        let registry = registry_ref(registry)?;
        let guard = registry_lock(registry)?;
        serde_json::to_vec(&guard.descriptors().collect::<Vec<_>>()).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize local implementation descriptors: {error}"
            ))
        })
    })
}

/// Clears all entries and releases retained callback state.
///
/// # Safety
///
/// `registry` must be a live registry pointer created by DAG-ML.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_clear(
    registry: *mut DagMlLocalImplementationRegistry,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    registration_boundary(error_out, "registry clear", || {
        let registry = registry_ref(registry)?;
        let old = {
            let mut guard = registry_lock(registry)?;
            std::mem::take(&mut *guard)
        };
        drop(old);
        Ok(())
    })
}

/// Releases a process-local implementation registry and all retained callbacks.
///
/// # Safety
///
/// `registry` must be null or a pointer returned by
/// [`dagml_local_implementation_registry_create`] that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn dagml_local_implementation_registry_free(
    registry: *mut DagMlLocalImplementationRegistry,
) {
    if registry.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(Box::from_raw(registry));
    }));
}

fn valid_binding_id(binding_id: &str) -> bool {
    binding_id
        .strip_prefix("binding:")
        .is_some_and(|suffix| !suffix.is_empty())
        && binding_id.trim() == binding_id
        && !binding_id.chars().any(char::is_whitespace)
        && !binding_id.chars().any(char::is_control)
}

unsafe fn registry_ref<'a>(
    registry: *mut DagMlLocalImplementationRegistry,
) -> dag_ml_core::Result<&'a DagMlLocalImplementationRegistry> {
    registry.as_ref().ok_or_else(|| {
        DagMlError::RuntimeValidation("local implementation registry pointer is null".to_string())
    })
}

fn registry_lock(
    registry: &DagMlLocalImplementationRegistry,
) -> dag_ml_core::Result<MutexGuard<'_, CoreLocalImplementationRegistry<Arc<LocalCallback>>>> {
    registry.registry.lock().map_err(|_| {
        DagMlError::RuntimeValidation("local implementation registry lock is poisoned".to_string())
    })
}

fn validate_local_descriptor(
    registry: &DagMlLocalImplementationRegistry,
    descriptor: &ImplementationDescriptor,
) -> dag_ml_core::Result<()> {
    descriptor.validate()?;
    if descriptor.binding_id != registry.binding_id {
        return Err(DagMlError::CampaignValidation(format!(
            "local implementation registry for `{}` rejects descriptor binding `{}`",
            registry.binding_id, descriptor.binding_id
        )));
    }
    if descriptor.portability == PortabilityClass::PortableBuiltIn {
        return Err(DagMlError::CampaignValidation(
            "local implementation registry rejects portable_builtin descriptors".to_string(),
        ));
    }
    Ok(())
}

fn parse_loss_reference(view: DagMlBytesView) -> dag_ml_core::Result<LossReference> {
    let json = unsafe { view_json(view, "loss reference") }?;
    let loss: LossReference =
        deserialize_external_contract(json, "loss reference", DagMlError::CampaignValidation)?;
    loss.validate()?;
    Ok(loss)
}

fn parse_metric_reference(view: DagMlBytesView) -> dag_ml_core::Result<MetricReference> {
    let json = unsafe { view_json(view, "metric reference") }?;
    let metric: MetricReference =
        deserialize_external_contract(json, "metric reference", DagMlError::CampaignValidation)?;
    metric.validate()?;
    Ok(metric)
}

fn parse_training_loss_role(
    view: DagMlBytesView,
) -> dag_ml_core::Result<TrainingLossRoleReference> {
    let json = unsafe { view_json(view, "training loss role") }?;
    TrainingLossRoleReference::from_json(json)
}

fn parse_node_task(view: DagMlBytesView) -> dag_ml_core::Result<NodeTask> {
    let json = unsafe { view_json(view, "node task") }?;
    deserialize_external_contract(json, "node task", DagMlError::RuntimeValidation)
}

unsafe fn view_json<'a>(view: DagMlBytesView, label: &str) -> dag_ml_core::Result<&'a str> {
    if view.ptr.is_null() {
        return Err(DagMlError::RuntimeValidation(format!(
            "{label} JSON pointer is null"
        )));
    }
    let bytes = slice::from_raw_parts(view.ptr, view.len);
    std::str::from_utf8(bytes).map_err(|error| {
        DagMlError::RuntimeValidation(format!("failed to parse {label} UTF-8: {error}"))
    })
}

fn validate_invocation_json(view: DagMlBytesView) -> dag_ml_core::Result<()> {
    let json = unsafe { view_json(view, "local implementation request") }?;
    parse_typed_json(json).map(|_| ()).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "invalid local implementation request JSON: {error}"
        ))
    })
}

fn parse_phase_for_registry(view: DagMlBytesView) -> dag_ml_core::Result<Phase> {
    let raw = unsafe { view_json(view, "training loss phase") }?;
    match raw {
        "FIT_CV" => Ok(Phase::FitCv),
        "REFIT" => Ok(Phase::Refit),
        _ => Err(DagMlError::CampaignValidation(format!(
            "unsupported training loss phase `{raw}`; expected FIT_CV or REFIT"
        ))),
    }
}

unsafe fn registration_boundary(
    error_out: *mut DagMlString,
    label: &str,
    operation: impl FnOnce() -> dag_ml_core::Result<()>,
) -> DagMlStatusCode {
    clear_error(error_out);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(Ok(())) => DagMlStatusCode::OK,
        Ok(Err(error)) => validation_error(error_out, error),
        Err(_) => panic_status(
            error_out,
            &format!("panic during local implementation {label} through C ABI"),
        ),
    }
}

unsafe fn invocation_boundary(
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
    label: &str,
    operation: impl FnOnce() -> dag_ml_core::Result<Vec<u8>>,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out_json.is_null() {
            set_error(
                error_out,
                format!("local implementation {label} output pointer is null"),
            );
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        match operation() {
            Ok(data) => {
                write_owned_vec(out_json, data);
                DagMlStatusCode::OK
            }
            Err(error) => validation_error(error_out, error),
        }
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_owned_bytes(out_json);
            panic_status(
                error_out,
                &format!("panic during local implementation {label} through C ABI"),
            )
        }
    }
}

unsafe fn training_invocation_boundary(
    out_result_json: *mut DagMlOwnedBytes,
    out_attestation_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
    label: &str,
    operation: impl FnOnce() -> dag_ml_core::Result<(Vec<u8>, Vec<u8>)>,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_result_json);
    clear_owned_bytes(out_attestation_json);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out_result_json.is_null() || out_attestation_json.is_null() {
            set_error(error_out, format!("{label} output pointer is null"));
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        match operation() {
            Ok((result, attestation)) => {
                write_owned_vec(out_result_json, result);
                write_owned_vec(out_attestation_json, attestation);
                DagMlStatusCode::OK
            }
            Err(error) => validation_error(error_out, error),
        }
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_owned_bytes(out_result_json);
            clear_owned_bytes(out_attestation_json);
            panic_status(error_out, &format!("panic during {label} through C ABI"))
        }
    }
}

unsafe fn panic_status(error_out: *mut DagMlString, message: &str) -> DagMlStatusCode {
    clear_error(error_out);
    set_error(error_out, message);
    DagMlStatusCode::PANIC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_ids_are_explicit_and_canonical() {
        assert!(valid_binding_id("binding:c"));
        assert!(valid_binding_id("binding:r"));
        assert!(valid_binding_id("binding:matlab"));
        assert!(!valid_binding_id("c"));
        assert!(!valid_binding_id("binding:"));
        assert!(!valid_binding_id("binding: c"));
    }
}
