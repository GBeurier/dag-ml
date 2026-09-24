//! Host-owned proposal and operator callbacks for non-durable native HPO.

use super::*;
use std::sync::Arc;

use dag_ml_core::{
    HostHpoCandidateControllerFactory, HostHpoCandidateProviderFactory, HostHpoCheckpoint,
    HostHpoInterruptedTrial, HostHpoProgress, HostHpoProposalSource, HostHpoResumeOptions,
    HostHpoSearchRequest, HostHpoSearchStatus,
};

pub const DAG_ML_HOST_HPO_CALLBACKS_ABI_VERSION: u32 = 1;
pub const DAG_ML_HOST_HPO_FEEDBACK_ABI_VERSION: u32 = 1;

/// Coordinator-owned feedback and durable-progress channel for ABI v2. The
/// callback receives one JSON event and returns one owned JSON reply. Every
/// returned buffer is released on all success/error paths.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlHostHpoFeedbackCallbacks {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub invoke: Option<
        unsafe extern "C" fn(*mut c_void, DagMlBytesView, *mut DagMlOwnedBytes) -> DagMlStatusCode,
    >,
    pub release_bytes: Option<unsafe extern "C" fn(*mut c_void, DagMlOwnedBytes)>,
}

impl DagMlHostHpoFeedbackCallbacks {
    fn validate(&self) -> dag_ml_core::Result<()> {
        if self.abi_version != DAG_ML_HOST_HPO_FEEDBACK_ABI_VERSION
            || self.invoke.is_none()
            || self.release_bytes.is_none()
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO C feedback table has unsupported ABI or missing callbacks".into(),
            ));
        }
        Ok(())
    }

    fn call(&self, event: serde_json::Value) -> dag_ml_core::Result<serde_json::Value> {
        let encoded = serde_json::to_vec(&event)?;
        let mut bytes = DagMlOwnedBytes::default();
        let status = unsafe {
            self.invoke.expect("validated")(self.user_data, bytes_view(&encoded), &mut bytes)
        };
        if status != DagMlStatusCode::OK {
            if !bytes.ptr.is_null() {
                unsafe { self.release_bytes.expect("validated")(self.user_data, bytes) };
            }
            callback_status(status, "feedback")?;
        }
        if bytes.ptr.is_null() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO C feedback callback returned null reply".into(),
            ));
        }
        let payload = unsafe { slice::from_raw_parts(bytes.ptr, bytes.len) }.to_vec();
        unsafe { self.release_bytes.expect("validated")(self.user_data, bytes) };
        serde_json::from_slice(&payload).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "host HPO C feedback returned invalid JSON: {error}"
            ))
        })
    }
}

/// All proposal transitions occur on the calling thread. `create_candidate`
/// returns a fresh opaque state for one trial; only that trial's worker uses
/// it. `destroy_candidate` runs once after every controller using that state
/// has released its handles. The caller must keep `user_data` alive until this
/// function returns. Callback implementations must not unwind across the ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlHostHpoCallbacks {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub ask: Option<
        unsafe extern "C" fn(*mut c_void, u32, i32, *mut DagMlOwnedBytes) -> DagMlStatusCode,
    >,
    pub tell: Option<unsafe extern "C" fn(*mut c_void, u32, f64) -> DagMlStatusCode>,
    pub fail: Option<unsafe extern "C" fn(*mut c_void, u32, DagMlBytesView) -> DagMlStatusCode>,
    pub release_proposal_bytes: Option<unsafe extern "C" fn(*mut c_void, DagMlOwnedBytes)>,
    pub create_candidate:
        Option<unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> DagMlStatusCode>,
    pub invoke_candidate: Option<
        unsafe extern "C" fn(*mut c_void, DagMlBytesView, *mut DagMlOwnedBytes) -> DagMlStatusCode,
    >,
    pub release_candidate_bytes: Option<unsafe extern "C" fn(*mut c_void, DagMlOwnedBytes)>,
    pub destroy_candidate: Option<unsafe extern "C" fn(*mut c_void)>,
}

impl DagMlHostHpoCallbacks {
    fn validate(&self) -> dag_ml_core::Result<()> {
        if self.abi_version != DAG_ML_HOST_HPO_CALLBACKS_ABI_VERSION
            || self.ask.is_none()
            || self.tell.is_none()
            || self.fail.is_none()
            || self.release_proposal_bytes.is_none()
            || self.create_candidate.is_none()
            || self.invoke_candidate.is_none()
            || self.release_candidate_bytes.is_none()
            || self.destroy_candidate.is_none()
        {
            return Err(DagMlError::RuntimeValidation(
                "host HPO C callback table has unsupported ABI or missing callbacks".into(),
            ));
        }
        Ok(())
    }
}

fn callback_status(status: DagMlStatusCode, operation: &str) -> dag_ml_core::Result<()> {
    if status == DagMlStatusCode::OK {
        Ok(())
    } else {
        Err(DagMlError::RuntimeValidation(format!(
            "host HPO C {operation} callback returned {status:?}"
        )))
    }
}

struct CProposals {
    callbacks: DagMlHostHpoCallbacks,
    feedback: Option<DagMlHostHpoFeedbackCallbacks>,
}

impl HostHpoProposalSource for CProposals {
    fn ask(
        &mut self,
        trial_index: u32,
    ) -> dag_ml_core::Result<Option<BTreeMap<String, serde_json::Value>>> {
        self.ask_in_phase(trial_index, None)
    }

    fn ask_in_phase(
        &mut self,
        trial_index: u32,
        phase_index: Option<u32>,
    ) -> dag_ml_core::Result<Option<BTreeMap<String, serde_json::Value>>> {
        let mut bytes = DagMlOwnedBytes::default();
        let phase = phase_index.map_or(-1, |index| index as i32);
        let status = unsafe {
            self.callbacks.ask.expect("validated")(
                self.callbacks.user_data,
                trial_index,
                phase,
                &mut bytes,
            )
        };
        if status != DagMlStatusCode::OK {
            if !bytes.ptr.is_null() {
                unsafe {
                    self.callbacks.release_proposal_bytes.expect("validated")(
                        self.callbacks.user_data,
                        bytes,
                    )
                };
            }
            callback_status(status, "ask")?;
        }
        if bytes.ptr.is_null() {
            if bytes.len != 0 {
                return Err(DagMlError::RuntimeValidation(
                    "host HPO C ask returned null bytes with nonzero length".into(),
                ));
            }
            return Ok(None);
        }
        let value = unsafe { slice::from_raw_parts(bytes.ptr, bytes.len) }.to_vec();
        unsafe {
            self.callbacks.release_proposal_bytes.expect("validated")(
                self.callbacks.user_data,
                bytes,
            )
        };
        let params = serde_json::from_slice(&value).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "host HPO C ask returned invalid parameter JSON: {error}"
            ))
        })?;
        Ok(Some(params))
    }

    fn tell(&mut self, trial_index: u32, score: f64) -> dag_ml_core::Result<()> {
        callback_status(
            unsafe {
                self.callbacks.tell.expect("validated")(
                    self.callbacks.user_data,
                    trial_index,
                    score,
                )
            },
            "tell",
        )
    }

    fn fail(&mut self, trial_index: u32, error: &str) -> dag_ml_core::Result<()> {
        callback_status(
            unsafe {
                self.callbacks.fail.expect("validated")(
                    self.callbacks.user_data,
                    trial_index,
                    bytes_view(error.as_bytes()),
                )
            },
            "fail",
        )
    }

    fn report_intermediate(
        &mut self,
        trial_index: u32,
        step: u32,
        score: f64,
    ) -> dag_ml_core::Result<bool> {
        let feedback = self.feedback.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "host HPO C ABI v1 cannot request progressive pruning".into(),
            )
        })?;
        feedback
            .call(serde_json::json!({
                "operation": "report_intermediate", "trial_index": trial_index,
                "step": step, "score": score,
            }))?
            .get("prune")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "host HPO C pruning reply lacks boolean `prune`".into(),
                )
            })
    }

    fn pruned(&mut self, trial_index: u32) -> dag_ml_core::Result<()> {
        let feedback = self.feedback.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "host HPO C ABI v1 cannot terminalize a pruned trial".into(),
            )
        })?;
        let reply = feedback.call(serde_json::json!({
            "operation": "pruned", "trial_index": trial_index,
        }))?;
        if reply.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(DagMlError::RuntimeValidation(
                "host HPO C pruned reply lacks true `ok`".into(),
            ))
        }
    }
}

struct CProgress(DagMlHostHpoFeedbackCallbacks);

impl HostHpoProgress for CProgress {
    fn prepare_terminal(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> dag_ml_core::Result<()> {
        let reply = self.0.call(serde_json::json!({
            "operation": "prepare_terminal", "checkpoint": checkpoint, "status": status,
        }))?;
        if reply.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(DagMlError::RuntimeValidation(
                "host HPO C prepare reply lacks true `ok`".into(),
            ))
        }
    }

    fn checkpoint(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> dag_ml_core::Result<bool> {
        let reply = self.0.call(serde_json::json!({
            "operation": "checkpoint", "checkpoint": checkpoint, "status": status,
        }))?;
        if reply.get("ok").and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(DagMlError::RuntimeValidation(
                "host HPO C checkpoint reply lacks true `ok`".into(),
            ));
        }
        Ok(reply
            .get("continue")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true))
    }
}

struct CandidateState {
    pointer: *mut c_void,
    destroy: unsafe extern "C" fn(*mut c_void),
}

// Each CandidateState comes from create_candidate for exactly one worker.
// The caller promises its state remains valid for that worker until destroy.
unsafe impl Send for CandidateState {}
unsafe impl Sync for CandidateState {}

impl Drop for CandidateState {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pointer) };
    }
}

struct CandidateController {
    inner: CAbiRuntimeController,
    _state: Arc<CandidateState>,
}

impl RuntimeController for CandidateController {
    fn controller_id(&self) -> &ControllerId {
        self.inner.controller_id()
    }

    fn invoke(&self, task: &NodeTask) -> dag_ml_core::Result<NodeResult> {
        self.inner.invoke(task)
    }

    fn invoke_aggregation(
        &self,
        task: &AggregationControllerTask,
    ) -> dag_ml_core::Result<AggregationControllerResult> {
        self.inner.invoke_aggregation(task)
    }
}

struct CControllerFactory {
    callbacks: DagMlHostHpoCallbacks,
    controller_ids: Vec<ControllerId>,
}

// Factory calls happen on the coordinator thread; only candidate-local state
// returned by create_candidate moves to a worker.
unsafe impl Send for CControllerFactory {}
unsafe impl Sync for CControllerFactory {}

impl HostHpoCandidateControllerFactory for CControllerFactory {
    fn create(&self, trial_index: u32) -> dag_ml_core::Result<RuntimeControllerRegistry> {
        let mut pointer = std::ptr::null_mut();
        callback_status(
            unsafe {
                self.callbacks.create_candidate.expect("validated")(
                    self.callbacks.user_data,
                    trial_index,
                    &mut pointer,
                )
            },
            "create_candidate",
        )?;
        if pointer.is_null() {
            return Err(DagMlError::RuntimeValidation(
                "host HPO C candidate factory returned null state".into(),
            ));
        }
        let state = Arc::new(CandidateState {
            pointer,
            destroy: self.callbacks.destroy_candidate.expect("validated"),
        });
        let mut registry = RuntimeControllerRegistry::new();
        for id in &self.controller_ids {
            let vtable = DagMlControllerVTable {
                abi_version: DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION,
                user_data: pointer,
                clone_with: None,
                describe: None,
                fit: None,
                predict: None,
                invoke: self.callbacks.invoke_candidate,
                release_bytes: self.callbacks.release_candidate_bytes,
                release: None,
                destroy: None,
            };
            registry.register(Box::new(CandidateController {
                inner: CAbiRuntimeController::new(id.clone(), vtable)?,
                _state: state.clone(),
            }))?;
        }
        Ok(registry)
    }
}

struct CProviderFactory {
    envelope: ExternalDataPlanEnvelope,
    controller_id: ControllerId,
}

impl HostHpoCandidateProviderFactory for CProviderFactory {
    fn create(
        &self,
        _trial_index: u32,
    ) -> dag_ml_core::Result<Box<dyn RuntimeDataProvider + Send>> {
        Ok(Box::new(InMemoryDataProvider::with_envelope(
            self.controller_id.clone(),
            self.envelope.clone(),
        )?))
    }
}

/// Execute a non-durable host HPO request through the same Rust scheduler as
/// PyO3. `max_parallel_trials` is a real worker bound; no pruning or resume is
/// accepted by this ABI version. All buffers follow the ordinary JSON C ABI
/// ownership rules and callbacks must remain valid until this call returns.
///
/// # Safety
/// All input pointers must reference readable bytes for their declared lengths.
/// `out_json` and `error_out` must be writable. Callback state ownership follows
/// `DagMlHostHpoCallbacks` above.
#[no_mangle]
pub unsafe extern "C" fn dagml_host_hpo_search_json(
    plan_ptr: *const u8,
    plan_len: usize,
    trusted_controllers_ptr: *const u8,
    trusted_controllers_len: usize,
    envelope_ptr: *const u8,
    envelope_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    callbacks: DagMlHostHpoCallbacks,
    max_parallel_trials: u32,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dagml_host_hpo_search_json_impl(
            plan_ptr,
            plan_len,
            trusted_controllers_ptr,
            trusted_controllers_len,
            envelope_ptr,
            envelope_len,
            request_ptr,
            request_len,
            callbacks,
            max_parallel_trials,
            out_json,
            error_out,
        )
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(error_out, "panic while executing host HPO through C ABI");
            DagMlStatusCode::PANIC
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dagml_host_hpo_search_json_impl(
    plan_ptr: *const u8,
    plan_len: usize,
    trusted_controllers_ptr: *const u8,
    trusted_controllers_len: usize,
    envelope_ptr: *const u8,
    envelope_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    callbacks: DagMlHostHpoCallbacks,
    max_parallel_trials: u32,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    if max_parallel_trials == 0 {
        set_error(error_out, "host HPO max_parallel_trials must be positive");
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    if let Err(error) = callbacks.validate() {
        return validation_error(error_out, error);
    }
    let plan = match parse_external_contract_ptr(
        plan_ptr,
        plan_len,
        error_out,
        "execution plan",
        ExecutionPlan::from_json,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let trusted: Vec<ControllerManifest> = match parse_json_ptr(
        trusted_controllers_ptr,
        trusted_controllers_len,
        error_out,
        "trusted controller manifests",
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let registry = match controller_registry_from_manifests(trusted) {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    if let Err(error) = validate_execution_plan_controller_manifests(&plan, &registry) {
        return validation_error(error_out, error);
    }
    let envelope: ExternalDataPlanEnvelope =
        match parse_json_ptr(envelope_ptr, envelope_len, error_out, "host HPO envelope") {
            Ok(value) => value,
            Err(status) => return status,
        };
    if let Err(error) = envelope
        .validate()
        .and_then(|()| plan.campaign.validate_data_envelope_relations(&envelope))
    {
        return validation_error(error_out, error);
    }
    let request: HostHpoSearchRequest =
        match parse_json_ptr(request_ptr, request_len, error_out, "host HPO request") {
            Ok(value) => value,
            Err(status) => return status,
        };
    if request.progressive_pruning {
        return validation_error(
            error_out,
            DagMlError::RuntimeValidation(
                "host HPO C ABI does not yet support progressive pruning".into(),
            ),
        );
    }
    let controller_id = match ControllerId::new("controller:data.provider") {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    let provider_factory = CProviderFactory {
        envelope: envelope.clone(),
        controller_id: controller_id.clone(),
    };
    let provider = match InMemoryDataProvider::with_envelope(controller_id, envelope) {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    let controller_factory = CControllerFactory {
        callbacks,
        controller_ids: plan.controller_manifests.keys().cloned().collect(),
    };
    let mut proposals = CProposals {
        callbacks,
        feedback: None,
    };
    let scheduler = SequentialScheduler;
    let result = if max_parallel_trials > 1 {
        scheduler.execute_parallel_host_hpo_search_with_candidate_factories(
            &plan,
            &provider_factory,
            &controller_factory,
            &request,
            &mut proposals,
            max_parallel_trials as usize,
        )
    } else {
        scheduler.execute_host_hpo_search_with_candidate_factories(
            &plan,
            &RuntimeControllerRegistry::new(),
            &provider,
            &provider_factory,
            &controller_factory,
            &request,
            &mut proposals,
        )
    };
    match result {
        Ok(value) => write_owned_json(out_json, error_out, &value),
        Err(error) => validation_error(error_out, error),
    }
}

/// ABI v2 adds coordinator-mediated pruning and durable native checkpoints.
/// The host must pair each `prepare_terminal`/`checkpoint` callback with its
/// own optimizer state. A null/zero resume pointer starts a fresh campaign.
///
/// # Safety
/// Input pointers must remain readable for their lengths; output pointers
/// must be writable. Callback state must remain alive until this call returns.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dagml_host_hpo_search_json_v2(
    plan_ptr: *const u8,
    plan_len: usize,
    trusted_controllers_ptr: *const u8,
    trusted_controllers_len: usize,
    envelope_ptr: *const u8,
    envelope_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    resume_checkpoint_ptr: *const u8,
    resume_checkpoint_len: usize,
    callbacks: DagMlHostHpoCallbacks,
    feedback: DagMlHostHpoFeedbackCallbacks,
    max_parallel_trials: u32,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dagml_host_hpo_search_json_v2_impl(
            plan_ptr,
            plan_len,
            trusted_controllers_ptr,
            trusted_controllers_len,
            envelope_ptr,
            envelope_len,
            request_ptr,
            request_len,
            resume_checkpoint_ptr,
            resume_checkpoint_len,
            callbacks,
            feedback,
            max_parallel_trials,
            out_json,
            error_out,
        )
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(error_out, "panic while executing host HPO v2 through C ABI");
            DagMlStatusCode::PANIC
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dagml_host_hpo_search_json_v2_impl(
    plan_ptr: *const u8,
    plan_len: usize,
    trusted_controllers_ptr: *const u8,
    trusted_controllers_len: usize,
    envelope_ptr: *const u8,
    envelope_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    resume_checkpoint_ptr: *const u8,
    resume_checkpoint_len: usize,
    callbacks: DagMlHostHpoCallbacks,
    feedback: DagMlHostHpoFeedbackCallbacks,
    max_parallel_trials: u32,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    if max_parallel_trials == 0 {
        set_error(error_out, "host HPO max_parallel_trials must be positive");
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    if let Err(error) = callbacks.validate().and_then(|()| feedback.validate()) {
        return validation_error(error_out, error);
    }
    let plan = match parse_external_contract_ptr(
        plan_ptr,
        plan_len,
        error_out,
        "execution plan",
        ExecutionPlan::from_json,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let trusted: Vec<ControllerManifest> = match parse_json_ptr(
        trusted_controllers_ptr,
        trusted_controllers_len,
        error_out,
        "trusted controller manifests",
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let registry = match controller_registry_from_manifests(trusted) {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    if let Err(error) = validate_execution_plan_controller_manifests(&plan, &registry) {
        return validation_error(error_out, error);
    }
    let envelope: ExternalDataPlanEnvelope =
        match parse_json_ptr(envelope_ptr, envelope_len, error_out, "host HPO envelope") {
            Ok(value) => value,
            Err(status) => return status,
        };
    if let Err(error) = envelope
        .validate()
        .and_then(|()| plan.campaign.validate_data_envelope_relations(&envelope))
    {
        return validation_error(error_out, error);
    }
    let request: HostHpoSearchRequest =
        match parse_json_ptr(request_ptr, request_len, error_out, "host HPO request") {
            Ok(value) => value,
            Err(status) => return status,
        };
    let checkpoint: Option<HostHpoCheckpoint> = if resume_checkpoint_len == 0 {
        None
    } else {
        match parse_json_ptr(
            resume_checkpoint_ptr,
            resume_checkpoint_len,
            error_out,
            "host HPO resume checkpoint",
        ) {
            Ok(value) => Some(value),
            Err(status) => return status,
        }
    };
    let options = match HostHpoResumeOptions::from_envelope(&envelope, checkpoint) {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    let controller_id = match ControllerId::new("controller:data.provider") {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    let provider_factory = CProviderFactory {
        envelope: envelope.clone(),
        controller_id: controller_id.clone(),
    };
    let provider = match InMemoryDataProvider::with_envelope(controller_id, envelope) {
        Ok(value) => value,
        Err(error) => return validation_error(error_out, error),
    };
    let controller_factory = CControllerFactory {
        callbacks,
        controller_ids: plan.controller_manifests.keys().cloned().collect(),
    };
    let mut proposals = CProposals {
        callbacks,
        feedback: Some(feedback),
    };
    let mut progress = CProgress(feedback);
    let scheduler = SequentialScheduler;
    let result = if max_parallel_trials > 1 {
        scheduler.execute_resumable_parallel_host_hpo_search_with_candidate_factories(
            &plan,
            &provider_factory,
            &controller_factory,
            &request,
            &mut proposals,
            max_parallel_trials as usize,
            &options,
            &mut progress,
        )
    } else {
        scheduler.execute_resumable_host_hpo_search_with_candidate_factories(
            &plan,
            &RuntimeControllerRegistry::new(),
            &provider,
            &provider_factory,
            &controller_factory,
            &request,
            &mut proposals,
            &options,
            &mut progress,
        )
    };
    match result {
        Ok(value) => write_owned_json(out_json, error_out, &value),
        Err(error) => validation_error(error_out, error),
    }
}

/// Validate a paired native checkpoint and recover a prepared terminal plus
/// candidate proposals interrupted before native evaluation.
///
/// # Safety
/// Every non-null input pointer must reference readable bytes for its length;
/// output pointers must be writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dagml_host_hpo_checkpoint_recover_json(
    checkpoint_ptr: *const u8,
    checkpoint_len: usize,
    prepared_ptr: *const u8,
    prepared_len: usize,
    interrupted_ptr: *const u8,
    interrupted_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clear_error(error_out);
        clear_owned_bytes(out_json);
        let checkpoint: HostHpoCheckpoint = match parse_json_ptr(
            checkpoint_ptr,
            checkpoint_len,
            error_out,
            "host HPO checkpoint",
        ) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let prepared: Option<HostHpoCheckpoint> = if prepared_len == 0 {
            None
        } else {
            match parse_json_ptr(
                prepared_ptr,
                prepared_len,
                error_out,
                "prepared host HPO terminal",
            ) {
                Ok(value) => Some(value),
                Err(status) => return status,
            }
        };
        let interrupted: Vec<HostHpoInterruptedTrial> = match parse_json_ptr(
            interrupted_ptr,
            interrupted_len,
            error_out,
            "interrupted host HPO trials",
        ) {
            Ok(value) => value,
            Err(status) => return status,
        };
        match checkpoint.recover_interrupted_trials(prepared, interrupted) {
            Ok(value) => write_owned_json(out_json, error_out, &value),
            Err(error) => validation_error(error_out, error),
        }
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(
                error_out,
                "panic while recovering host HPO checkpoint through C ABI",
            );
            DagMlStatusCode::PANIC
        }
    }
}
