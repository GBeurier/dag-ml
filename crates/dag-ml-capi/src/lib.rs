use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

use dag_ml_core::{
    select_candidate, select_candidate_groups, BundlePredictionCachePayloadSet,
    BundleReplayExecution, CandidateScore, ControllerId, DagMlError, DataMaterializationRequest,
    DataRequestPartition, DataViewRequest, ExecutionBundle, ExecutionPlan,
    ExternalDataPlanEnvelope, GraphSpec, HandleKind, HandleRef, InMemoryArtifactStore,
    InMemoryDataProvider, LineageId, LineageRecord, NodeResult, NodeTask, Phase, PredictionBlock,
    PredictionCacheMaterializationRequest, PredictionPartition, ReplayPhaseRequest, RunContext,
    RunId, RuntimeController, RuntimeControllerRegistry, RuntimeDataProvider,
    RuntimePredictionCacheStore, SampleId, SelectionDecision, SelectionPolicy, SequentialScheduler,
};
use serde::{de::DeserializeOwned, Serialize};

pub type DagMlHandle = u64;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DagMlStatusCode(pub u32);

impl DagMlStatusCode {
    pub const OK: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const VALIDATION_ERROR: Self = Self(2);
    pub const PANIC: Self = Self(255);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DagMlVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlString {
    pub ptr: *mut c_char,
    pub len: usize,
}

impl Default for DagMlString {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlBytesView {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlOwnedBytes {
    pub ptr: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl Default for DagMlOwnedBytes {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(array: *mut ArrowArray)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(schema: *mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlControllerVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub clone_with: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            op: DagMlHandle,
            params_json: DagMlBytesView,
            out_op: *mut DagMlHandle,
        ) -> DagMlStatusCode,
    >,
    pub describe: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            op: DagMlHandle,
            out_json: *mut DagMlOwnedBytes,
        ) -> DagMlStatusCode,
    >,
    pub fit: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            op: DagMlHandle,
            data: DagMlHandle,
            context_json: DagMlBytesView,
            out_fitted: *mut DagMlHandle,
        ) -> DagMlStatusCode,
    >,
    pub predict: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            fitted: DagMlHandle,
            data: DagMlHandle,
            out_arrow_array: *mut *mut ArrowArray,
            out_arrow_schema: *mut *mut ArrowSchema,
        ) -> DagMlStatusCode,
    >,
    pub invoke: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            task_json: DagMlBytesView,
            out_result_json: *mut DagMlOwnedBytes,
        ) -> DagMlStatusCode,
    >,
    pub release_bytes: Option<unsafe extern "C" fn(user_data: *mut c_void, bytes: DagMlOwnedBytes)>,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void, handle: DagMlHandle)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlDataVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub materialize: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            dataset: DagMlHandle,
            request_json: DagMlBytesView,
            out_handle: *mut DagMlHandle,
        ) -> DagMlStatusCode,
    >,
    pub make_view: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            data: DagMlHandle,
            selector_json: DagMlBytesView,
            out_view: *mut DagMlHandle,
        ) -> DagMlStatusCode,
    >,
    pub view_identity: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlHandle,
            out_arrow_array: *mut *mut ArrowArray,
            out_arrow_schema: *mut *mut ArrowSchema,
        ) -> DagMlStatusCode,
    >,
    pub target_arrow: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlHandle,
            target_name: DagMlBytesView,
            out_arrow_array: *mut *mut ArrowArray,
            out_arrow_schema: *mut *mut ArrowSchema,
        ) -> DagMlStatusCode,
    >,
    pub feature_arrow: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlHandle,
            feature_set_name: DagMlBytesView,
            out_arrow_array: *mut *mut ArrowArray,
            out_arrow_schema: *mut *mut ArrowSchema,
        ) -> DagMlStatusCode,
    >,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void, handle: DagMlHandle)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlPredictionCacheVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub load_blocks: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            requirement_key: DagMlBytesView,
            out_json: *mut DagMlOwnedBytes,
        ) -> DagMlStatusCode,
    >,
    pub materialize: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            request_json: DagMlBytesView,
            out_handle: *mut DagMlHandle,
        ) -> DagMlStatusCode,
    >,
    pub release_bytes: Option<unsafe extern "C" fn(user_data: *mut c_void, bytes: DagMlOwnedBytes)>,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void, handle: DagMlHandle)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

#[no_mangle]
pub extern "C" fn dagml_version() -> DagMlVersion {
    DagMlVersion {
        major: 0,
        minor: 1,
        patch: 0,
    }
}

/// Releases a string allocated by DAG-ML.
///
/// # Safety
///
/// `value.ptr` must either be null or a pointer previously returned by a
/// DAG-ML C ABI function in a `DagMlString`. Passing any other pointer, or
/// freeing the same string twice, is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagml_string_free(value: DagMlString) {
    if !value.ptr.is_null() {
        drop(CString::from_raw(value.ptr));
    }
}

/// Releases bytes allocated by DAG-ML.
///
/// # Safety
///
/// `value.ptr` must either be null or a pointer previously returned by a
/// DAG-ML C ABI function in a `DagMlOwnedBytes`. Passing any other pointer, or
/// freeing the same byte buffer twice, is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagml_owned_bytes_free(value: DagMlOwnedBytes) {
    if !value.ptr.is_null() {
        drop(Vec::from_raw_parts(value.ptr, value.len, value.capacity));
    }
}

/// Validates a canonical JSON `GraphSpec`.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `error_out` may be null; when non-null it must
/// point to writable memory for one `DagMlString`. Any returned string must be
/// released with `dagml_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_graph_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    validate_json::<GraphSpec>(json_ptr, json_len, error_out, "graph", GraphSpec::validate)
}

/// Validates a canonical JSON `SelectionPolicy`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_selection_policy_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    validate_json::<SelectionPolicy>(
        json_ptr,
        json_len,
        error_out,
        "selection policy",
        SelectionPolicy::validate,
    )
}

/// Validates a canonical JSON `SelectionDecision`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_selection_decision_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    validate_json::<SelectionDecision>(
        json_ptr,
        json_len,
        error_out,
        "selection decision",
        SelectionDecision::validate,
    )
}

/// Selects one candidate from JSON `SelectionPolicy` and `CandidateScore[]`.
///
/// # Safety
///
/// Input pointers follow the same rules as `dagml_graph_validate_json`.
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`; returned
/// bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_select_candidate_json(
    policy_ptr: *const u8,
    policy_len: usize,
    candidates_ptr: *const u8,
    candidates_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let policy = match parse_json_ptr::<SelectionPolicy>(
        policy_ptr,
        policy_len,
        error_out,
        "selection policy",
    ) {
        Ok(policy) => policy,
        Err(status) => return status,
    };
    let candidates = match parse_json_ptr::<Vec<CandidateScore>>(
        candidates_ptr,
        candidates_len,
        error_out,
        "candidate scores",
    ) {
        Ok(candidates) => candidates,
        Err(status) => return status,
    };
    match select_candidate(&policy, &candidates) {
        Ok(decision) => write_owned_json(out_json, error_out, &decision),
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

/// Selects candidates per group from JSON policy, candidates and group map.
///
/// # Safety
///
/// Input pointers follow the same rules as `dagml_graph_validate_json`.
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`; returned
/// bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_select_candidate_groups_json(
    policy_ptr: *const u8,
    policy_len: usize,
    candidates_ptr: *const u8,
    candidates_len: usize,
    groups_ptr: *const u8,
    groups_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let policy = match parse_json_ptr::<SelectionPolicy>(
        policy_ptr,
        policy_len,
        error_out,
        "selection policy",
    ) {
        Ok(policy) => policy,
        Err(status) => return status,
    };
    let candidates = match parse_json_ptr::<Vec<CandidateScore>>(
        candidates_ptr,
        candidates_len,
        error_out,
        "candidate scores",
    ) {
        Ok(candidates) => candidates,
        Err(status) => return status,
    };
    let groups = match parse_json_ptr::<BTreeMap<String, Vec<String>>>(
        groups_ptr,
        groups_len,
        error_out,
        "candidate groups",
    ) {
        Ok(groups) => groups,
        Err(status) => return status,
    };
    match select_candidate_groups(&policy, &candidates, &groups) {
        Ok(decisions) => write_owned_json(out_json, error_out, &decisions),
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

/// Validates a canonical JSON `ExecutionBundle`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_execution_bundle_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    validate_json::<ExecutionBundle>(
        json_ptr,
        json_len,
        error_out,
        "execution bundle",
        ExecutionBundle::validate,
    )
}

/// Validates replay data envelopes against an `ExecutionBundle`.
///
/// `envelopes_json` must be a JSON object keyed by bundle data requirement key,
/// for example `{ "model:base.x": { ... ExternalDataPlanEnvelope ... } }`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_execution_bundle_validate_replay_envelopes_json(
    bundle_ptr: *const u8,
    bundle_len: usize,
    envelopes_ptr: *const u8,
    envelopes_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    let bundle =
        match parse_json_ptr::<ExecutionBundle>(bundle_ptr, bundle_len, error_out, "bundle") {
            Ok(bundle) => bundle,
            Err(status) => return status,
        };
    let envelopes = match parse_json_ptr::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        envelopes_ptr,
        envelopes_len,
        error_out,
        "replay envelopes",
    ) {
        Ok(envelopes) => envelopes,
        Err(status) => return status,
    };
    match bundle.validate_replay_envelopes(&envelopes) {
        Ok(()) => DagMlStatusCode::OK,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

/// Validates a replay request against an `ExecutionBundle`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_replay_request_validate_for_bundle_json(
    bundle_ptr: *const u8,
    bundle_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    let bundle =
        match parse_json_ptr::<ExecutionBundle>(bundle_ptr, bundle_len, error_out, "bundle") {
            Ok(bundle) => bundle,
            Err(status) => return status,
        };
    let request = match parse_json_ptr::<ReplayPhaseRequest>(
        request_ptr,
        request_len,
        error_out,
        "replay request",
    ) {
        Ok(request) => request,
        Err(status) => return status,
    };
    match request.validate_for_bundle(&bundle) {
        Ok(()) => DagMlStatusCode::OK,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

/// Validates a prediction-cache payload set against an `ExecutionBundle`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_prediction_cache_payload_validate_for_bundle_json(
    bundle_ptr: *const u8,
    bundle_len: usize,
    payload_ptr: *const u8,
    payload_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    let bundle =
        match parse_json_ptr::<ExecutionBundle>(bundle_ptr, bundle_len, error_out, "bundle") {
            Ok(bundle) => bundle,
            Err(status) => return status,
        };
    let payload = match parse_json_ptr::<BundlePredictionCachePayloadSet>(
        payload_ptr,
        payload_len,
        error_out,
        "prediction cache payload set",
    ) {
        Ok(payload) => payload,
        Err(status) => return status,
    };
    match payload.validate_against_bundle(&bundle) {
        Ok(()) => DagMlStatusCode::OK,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

/// Validates a replay request against an `ExecutionBundle` plus OOF cache payloads.
///
/// This variant is required for OOF-dependent `REFIT` replay; the manifest-only
/// validator keeps refusing that case.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_replay_request_validate_for_bundle_with_prediction_cache_payload_json(
    bundle_ptr: *const u8,
    bundle_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    payload_ptr: *const u8,
    payload_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    let bundle =
        match parse_json_ptr::<ExecutionBundle>(bundle_ptr, bundle_len, error_out, "bundle") {
            Ok(bundle) => bundle,
            Err(status) => return status,
        };
    let request = match parse_json_ptr::<ReplayPhaseRequest>(
        request_ptr,
        request_len,
        error_out,
        "replay request",
    ) {
        Ok(request) => request,
        Err(status) => return status,
    };
    let payload = match parse_json_ptr::<BundlePredictionCachePayloadSet>(
        payload_ptr,
        payload_len,
        error_out,
        "prediction cache payload set",
    ) {
        Ok(payload) => payload,
        Err(status) => return status,
    };
    match request.validate_for_bundle_with_prediction_cache_payloads(&bundle, Some(&payload)) {
        Ok(()) => DagMlStatusCode::OK,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

/// Executes a deterministic Rust-side mock replay from JSON contracts.
///
/// This is a conformance smoke for bindings: it validates the replay bundle,
/// materializes data and refit artifact handles, invokes mock controllers, and
/// returns a small JSON summary. Real host controller execution is handled by
/// the vtable roadmap and is intentionally not implemented by this helper.
///
/// # Safety
///
/// Input pointers follow the same rules as `dagml_graph_validate_json`.
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`; returned
/// bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_mock_replay_execute_json(
    plan_ptr: *const u8,
    plan_len: usize,
    bundle_ptr: *const u8,
    bundle_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    envelopes_ptr: *const u8,
    envelopes_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let plan =
        match parse_json_ptr::<ExecutionPlan>(plan_ptr, plan_len, error_out, "execution plan") {
            Ok(plan) => plan,
            Err(status) => return status,
        };
    let bundle =
        match parse_json_ptr::<ExecutionBundle>(bundle_ptr, bundle_len, error_out, "bundle") {
            Ok(bundle) => bundle,
            Err(status) => return status,
        };
    let request = match parse_json_ptr::<ReplayPhaseRequest>(
        request_ptr,
        request_len,
        error_out,
        "replay request",
    ) {
        Ok(request) => request,
        Err(status) => return status,
    };
    let envelopes = match parse_json_ptr::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        envelopes_ptr,
        envelopes_len,
        error_out,
        "replay envelopes",
    ) {
        Ok(envelopes) => envelopes,
        Err(status) => return status,
    };

    match execute_mock_replay(&plan, &bundle, &request, &envelopes) {
        Ok(summary) => write_owned_json(out_json, error_out, &summary),
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

unsafe fn clear_error(error_out: *mut DagMlString) {
    if !error_out.is_null() {
        *error_out = DagMlString::default();
    }
}

unsafe fn clear_owned_bytes(out_json: *mut DagMlOwnedBytes) {
    if !out_json.is_null() {
        *out_json = DagMlOwnedBytes::default();
    }
}

unsafe fn set_error(error_out: *mut DagMlString, message: impl Into<String>) {
    if error_out.is_null() {
        return;
    }
    let sanitized = message.into().replace('\0', "\\0");
    let c_string = CString::new(sanitized).expect("nul bytes were sanitized");
    let len = c_string.as_bytes().len();
    *error_out = DagMlString {
        ptr: c_string.into_raw(),
        len,
    };
}

unsafe fn validate_json<T>(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
    label: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> DagMlStatusCode
where
    T: DeserializeOwned,
{
    clear_error(error_out);
    let value = match parse_json_ptr::<T>(json_ptr, json_len, error_out, label) {
        Ok(value) => value,
        Err(status) => return status,
    };
    match validate(&value) {
        Ok(()) => DagMlStatusCode::OK,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

unsafe fn parse_json_ptr<T>(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
    label: &str,
) -> Result<T, DagMlStatusCode>
where
    T: DeserializeOwned,
{
    if json_ptr.is_null() {
        set_error(error_out, format!("{label} json pointer is null"));
        return Err(DagMlStatusCode::INVALID_ARGUMENT);
    }
    let json = slice::from_raw_parts(json_ptr, json_len);
    serde_json::from_slice::<T>(json).map_err(|error| {
        set_error(error_out, format!("failed to parse {label} JSON: {error}"));
        DagMlStatusCode::VALIDATION_ERROR
    })
}

unsafe fn write_owned_json<T>(
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
    value: &T,
) -> DagMlStatusCode
where
    T: Serialize,
{
    if out_json.is_null() {
        set_error(error_out, "output JSON pointer is null");
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    match serde_json::to_vec(value) {
        Ok(mut data) => {
            let owned = DagMlOwnedBytes {
                ptr: data.as_mut_ptr(),
                len: data.len(),
                capacity: data.capacity(),
            };
            std::mem::forget(data);
            *out_json = owned;
            DagMlStatusCode::OK
        }
        Err(error) => {
            set_error(
                error_out,
                format!("failed to serialize output JSON: {error}"),
            );
            DagMlStatusCode::VALIDATION_ERROR
        }
    }
}

#[derive(Clone)]
pub struct CAbiRuntimeDataProvider {
    vtable: DagMlDataVTable,
    dataset: DagMlHandle,
    owner_controller: ControllerId,
}

impl CAbiRuntimeDataProvider {
    pub fn new(
        owner_controller: ControllerId,
        dataset: DagMlHandle,
        vtable: DagMlDataVTable,
    ) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < 2 {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider ABI version {} is unsupported",
                vtable.abi_version
            )));
        }
        if vtable.materialize.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "data provider vtable is missing materialize".to_string(),
            ));
        }
        if vtable.make_view.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "data provider vtable is missing make_view".to_string(),
            ));
        }
        Ok(Self {
            vtable,
            dataset,
            owner_controller,
        })
    }
}

impl RuntimeDataProvider for CAbiRuntimeDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> dag_ml_core::Result<HandleRef> {
        let materialize = self.vtable.materialize.ok_or_else(|| {
            DagMlError::RuntimeValidation("data provider vtable is missing materialize".to_string())
        })?;
        let request_json = serde_json::to_vec(&CAbiDataMaterializationJson::from(request))
            .map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "failed to serialize data materialization request: {error}"
                ))
            })?;
        let mut out_handle = 0;
        let status = unsafe {
            materialize(
                self.vtable.user_data,
                self.dataset,
                bytes_view(&request_json),
                &mut out_handle,
            )
        };
        data_provider_status(status, "materialize")?;
        if out_handle == 0 {
            return Err(DagMlError::RuntimeValidation(
                "data provider materialize returned empty handle".to_string(),
            ));
        }
        Ok(HandleRef {
            handle: out_handle,
            kind: HandleKind::Data,
            owner_controller: self.owner_controller.clone(),
        })
    }

    fn make_view(&self, request: &DataViewRequest) -> dag_ml_core::Result<HandleRef> {
        let make_view = self.vtable.make_view.ok_or_else(|| {
            DagMlError::RuntimeValidation("data provider vtable is missing make_view".to_string())
        })?;
        if request.data_handle.kind != HandleKind::Data {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider make_view received non-data parent handle for `{}` on `{}`",
                request.input_name, request.node_id
            )));
        }
        let selector_json = serde_json::to_vec(&request.view).map_err(|error| {
            DagMlError::RuntimeValidation(format!("failed to serialize data view request: {error}"))
        })?;
        let mut out_view = 0;
        let status = unsafe {
            make_view(
                self.vtable.user_data,
                request.data_handle.handle,
                bytes_view(&selector_json),
                &mut out_view,
            )
        };
        data_provider_status(status, "make_view")?;
        if out_view == 0 {
            return Err(DagMlError::RuntimeValidation(
                "data provider make_view returned empty handle".to_string(),
            ));
        }
        Ok(HandleRef {
            handle: out_view,
            kind: HandleKind::DataView,
            owner_controller: self.owner_controller.clone(),
        })
    }
}

#[derive(Clone)]
pub struct CAbiRuntimeController {
    id: ControllerId,
    vtable: DagMlControllerVTable,
}

impl CAbiRuntimeController {
    pub fn new(id: ControllerId, vtable: DagMlControllerVTable) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < 2 {
            return Err(DagMlError::RuntimeValidation(format!(
                "controller ABI version {} is unsupported for generic invoke",
                vtable.abi_version
            )));
        }
        if vtable.invoke.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "controller vtable is missing invoke".to_string(),
            ));
        }
        if vtable.release_bytes.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "controller vtable is missing release_bytes".to_string(),
            ));
        }
        Ok(Self { id, vtable })
    }
}

impl RuntimeController for CAbiRuntimeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> dag_ml_core::Result<NodeResult> {
        let invoke = self.vtable.invoke.ok_or_else(|| {
            DagMlError::RuntimeValidation("controller vtable is missing invoke".to_string())
        })?;
        let release_bytes = self.vtable.release_bytes.ok_or_else(|| {
            DagMlError::RuntimeValidation("controller vtable is missing release_bytes".to_string())
        })?;
        let task_json = serde_json::to_vec(task).map_err(|error| {
            DagMlError::RuntimeValidation(format!("failed to serialize node task: {error}"))
        })?;
        let mut out_json = DagMlOwnedBytes::default();
        let status =
            unsafe { invoke(self.vtable.user_data, bytes_view(&task_json), &mut out_json) };
        if status != DagMlStatusCode::OK {
            if !out_json.ptr.is_null() {
                unsafe { release_bytes(self.vtable.user_data, out_json) };
            }
            controller_status(status, "invoke")?;
        }
        if out_json.ptr.is_null() {
            return Err(DagMlError::RuntimeValidation(
                "controller invoke returned null result JSON".to_string(),
            ));
        }
        let data = unsafe { slice::from_raw_parts(out_json.ptr, out_json.len) }.to_vec();
        unsafe { release_bytes(self.vtable.user_data, out_json) };
        serde_json::from_slice::<NodeResult>(&data).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "controller invoke returned invalid node result JSON: {error}"
            ))
        })
    }
}

fn controller_status(status: DagMlStatusCode, action: &str) -> dag_ml_core::Result<()> {
    if status == DagMlStatusCode::OK {
        Ok(())
    } else if status == DagMlStatusCode::INVALID_ARGUMENT {
        Err(DagMlError::RuntimeValidation(format!(
            "controller {action} rejected invalid arguments"
        )))
    } else if status == DagMlStatusCode::VALIDATION_ERROR {
        Err(DagMlError::RuntimeValidation(format!(
            "controller {action} rejected request"
        )))
    } else if status == DagMlStatusCode::PANIC {
        Err(DagMlError::RuntimeValidation(format!(
            "controller {action} reported panic"
        )))
    } else {
        Err(DagMlError::RuntimeValidation(format!(
            "controller {action} returned unknown status code {}",
            status.0
        )))
    }
}

#[derive(Clone)]
pub struct CAbiRuntimePredictionCacheStore {
    vtable: DagMlPredictionCacheVTable,
}

impl CAbiRuntimePredictionCacheStore {
    pub fn new(vtable: DagMlPredictionCacheVTable) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < 1 {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache ABI version {} is unsupported",
                vtable.abi_version
            )));
        }
        if vtable.load_blocks.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "prediction cache vtable is missing load_blocks".to_string(),
            ));
        }
        if vtable.materialize.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "prediction cache vtable is missing materialize".to_string(),
            ));
        }
        if vtable.release_bytes.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "prediction cache vtable is missing release_bytes".to_string(),
            ));
        }
        Ok(Self { vtable })
    }
}

impl RuntimePredictionCacheStore for CAbiRuntimePredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> dag_ml_core::Result<Vec<PredictionBlock>> {
        let load_blocks = self.vtable.load_blocks.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "prediction cache vtable is missing load_blocks".to_string(),
            )
        })?;
        let release_bytes = self.vtable.release_bytes.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "prediction cache vtable is missing release_bytes".to_string(),
            )
        })?;
        let mut out_json = DagMlOwnedBytes::default();
        let status = unsafe {
            load_blocks(
                self.vtable.user_data,
                bytes_view(requirement_key.as_bytes()),
                &mut out_json,
            )
        };
        if status != DagMlStatusCode::OK {
            if !out_json.ptr.is_null() {
                unsafe { release_bytes(self.vtable.user_data, out_json) };
            }
            prediction_cache_status(status, "load_blocks")?;
        }
        if out_json.ptr.is_null() {
            return Err(DagMlError::RuntimeValidation(
                "prediction cache load_blocks returned null JSON".to_string(),
            ));
        }
        let data = unsafe { slice::from_raw_parts(out_json.ptr, out_json.len) }.to_vec();
        unsafe { release_bytes(self.vtable.user_data, out_json) };
        serde_json::from_slice::<Vec<PredictionBlock>>(&data).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache load_blocks returned invalid prediction block JSON: {error}"
            ))
        })
    }

    fn materialize(
        &self,
        request: &PredictionCacheMaterializationRequest,
    ) -> dag_ml_core::Result<HandleRef> {
        let materialize = self.vtable.materialize.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "prediction cache vtable is missing materialize".to_string(),
            )
        })?;
        let request_json = serde_json::to_vec(request).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize prediction cache materialization request: {error}"
            ))
        })?;
        let mut out_handle = 0;
        let status = unsafe {
            materialize(
                self.vtable.user_data,
                bytes_view(&request_json),
                &mut out_handle,
            )
        };
        prediction_cache_status(status, "materialize")?;
        if out_handle == 0 {
            return Err(DagMlError::RuntimeValidation(
                "prediction cache materialize returned empty handle".to_string(),
            ));
        }
        Ok(HandleRef {
            handle: out_handle,
            kind: HandleKind::Prediction,
            owner_controller: request.producer_controller_id.clone(),
        })
    }
}

fn prediction_cache_status(status: DagMlStatusCode, action: &str) -> dag_ml_core::Result<()> {
    if status == DagMlStatusCode::OK {
        Ok(())
    } else if status == DagMlStatusCode::INVALID_ARGUMENT {
        Err(DagMlError::RuntimeValidation(format!(
            "prediction cache {action} rejected invalid arguments"
        )))
    } else if status == DagMlStatusCode::VALIDATION_ERROR {
        Err(DagMlError::RuntimeValidation(format!(
            "prediction cache {action} rejected request"
        )))
    } else if status == DagMlStatusCode::PANIC {
        Err(DagMlError::RuntimeValidation(format!(
            "prediction cache {action} reported panic"
        )))
    } else {
        Err(DagMlError::RuntimeValidation(format!(
            "prediction cache {action} returned unknown status code {}",
            status.0
        )))
    }
}

#[derive(Debug, Serialize)]
struct CAbiDataMaterializationJson {
    run_id: String,
    node_id: String,
    input_name: String,
    phase: &'static str,
    variant_id: Option<String>,
    fold_id: Option<String>,
    request_id: String,
    schema_fingerprint: String,
    plan_fingerprint: String,
    relation_fingerprint: Option<String>,
    output_representation: String,
    source_ids: Vec<String>,
    require_relations: bool,
}

impl From<&DataMaterializationRequest> for CAbiDataMaterializationJson {
    fn from(request: &DataMaterializationRequest) -> Self {
        Self {
            run_id: request.run_id.to_string(),
            node_id: request.node_id.to_string(),
            input_name: request.input_name.clone(),
            phase: phase_abi_name(request.phase),
            variant_id: request.variant_id.as_ref().map(ToString::to_string),
            fold_id: request.fold_id.as_ref().map(ToString::to_string),
            request_id: request.binding.request_id.clone(),
            schema_fingerprint: request.binding.schema_fingerprint.clone(),
            plan_fingerprint: request.binding.plan_fingerprint.clone(),
            relation_fingerprint: request.binding.relation_fingerprint.clone(),
            output_representation: request.binding.output_representation.clone(),
            source_ids: request.binding.source_ids.clone(),
            require_relations: request.binding.require_relations,
        }
    }
}

fn phase_abi_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Compile => "COMPILE",
        Phase::Plan => "PLAN",
        Phase::FitCv => "FIT_CV",
        Phase::Select => "SELECT",
        Phase::Refit => "REFIT",
        Phase::Predict => "PREDICT",
        Phase::Explain => "EXPLAIN",
    }
}

fn bytes_view(bytes: &[u8]) -> DagMlBytesView {
    DagMlBytesView {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}

fn data_provider_status(status: DagMlStatusCode, action: &str) -> dag_ml_core::Result<()> {
    if status == DagMlStatusCode::OK {
        Ok(())
    } else if status == DagMlStatusCode::INVALID_ARGUMENT {
        Err(DagMlError::RuntimeValidation(format!(
            "data provider {action} returned invalid argument"
        )))
    } else if status == DagMlStatusCode::VALIDATION_ERROR {
        Err(DagMlError::RuntimeValidation(format!(
            "data provider {action} returned validation error"
        )))
    } else if status == DagMlStatusCode::PANIC {
        Err(DagMlError::RuntimeValidation(format!(
            "data provider {action} panicked"
        )))
    } else {
        Err(DagMlError::RuntimeValidation(format!(
            "data provider {action} returned unknown status code {}",
            status.0
        )))
    }
}

#[derive(Debug, Serialize)]
struct MockReplaySummary {
    bundle_id: String,
    phase: Phase,
    result_count: usize,
    lineage_record_count: usize,
    prediction_block_count: usize,
    data_handle_count: usize,
    data_view_count: usize,
    artifact_handle_count: usize,
}

fn execute_mock_replay(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    request: &ReplayPhaseRequest,
    envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
) -> dag_ml_core::Result<MockReplaySummary> {
    if envelopes.is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "mock replay requires at least one replay envelope".to_string(),
        ));
    }
    let mut data_provider =
        InMemoryDataProvider::new(ControllerId::new("controller:data.provider")?);
    for envelope in envelopes.values() {
        data_provider.register_envelope(envelope.clone())?;
    }
    let artifact_store = mock_artifact_store(plan, bundle)?;
    let controllers = mock_runtime_controllers(plan)?;
    let mut ctx = RunContext::new(RunId::new("run:capi.mock.replay")?, None);
    let results = SequentialScheduler.execute_bundle_replay(
        BundleReplayExecution {
            plan,
            bundle,
            replay_request: request,
            prediction_cache_store: None,
            controllers: &controllers,
            data_provider: &data_provider,
            artifact_store: &artifact_store,
            data_envelopes: envelopes,
        },
        &mut ctx,
    )?;
    Ok(MockReplaySummary {
        bundle_id: bundle.bundle_id.to_string(),
        phase: request.phase,
        result_count: results.len(),
        lineage_record_count: ctx.lineage.len(),
        prediction_block_count: ctx.prediction_store.blocks().len(),
        data_handle_count: data_provider.handle_records().len(),
        data_view_count: data_provider.view_records().len(),
        artifact_handle_count: artifact_store.len(),
    })
}

fn mock_artifact_store(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
) -> dag_ml_core::Result<InMemoryArtifactStore> {
    bundle.validate_against_plan(plan)?;
    let mut store = InMemoryArtifactStore::new();
    for artifact in &bundle.refit_artifacts {
        let node_plan = plan.node_plans.get(&artifact.node_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "bundle artifact `{}` references unknown node `{}`",
                artifact.artifact.id, artifact.node_id
            ))
        })?;
        let kind = if matches!(node_plan.kind, dag_ml_core::NodeKind::Model) {
            HandleKind::Model
        } else {
            HandleKind::Artifact
        };
        store.register(
            artifact,
            HandleRef {
                handle: stable_handle(artifact.artifact.id.as_str()),
                kind,
                owner_controller: artifact.controller_id.clone(),
            },
        )?;
    }
    Ok(store)
}

fn mock_runtime_controllers(
    plan: &ExecutionPlan,
) -> dag_ml_core::Result<RuntimeControllerRegistry> {
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        registry.register(Box::new(CapiMockController {
            id: controller_id.clone(),
        }))?;
    }
    Ok(registry)
}

struct CapiMockController {
    id: ControllerId,
}

impl RuntimeController for CapiMockController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> dag_ml_core::Result<NodeResult> {
        for binding in &task.node_plan.data_bindings {
            let key = format!("data:{}", binding.input_name);
            let handle = task.input_handles.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data handle `{key}`",
                    task.node_plan.node_id
                ))
            })?;
            if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received non-data/data-view handle for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if !task.data_views.contains_key(&key) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data view spec for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if task.phase == Phase::FitCv && task.fold_id.is_some() {
                let validation_key = format!("{key}:validation");
                let validation_view = task.data_views.get(&validation_key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive validation data view spec for `{validation_key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if validation_view.partition != DataRequestPartition::FoldValidation {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-validation data view for `{validation_key}`",
                        task.node_plan.node_id
                    )));
                }
            }
        }

        if task.phase == Phase::Predict
            && matches!(task.node_plan.kind, dag_ml_core::NodeKind::Model)
        {
            let artifact_handles = task
                .input_handles
                .iter()
                .filter(|(key, _)| key.starts_with("artifact:"))
                .collect::<Vec<_>>();
            if artifact_handles.is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive a replay artifact handle",
                    task.node_plan.node_id
                )));
            }
            for (key, handle) in artifact_handles {
                if !matches!(handle.kind, HandleKind::Model | HandleKind::Artifact) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received invalid replay artifact handle `{key}`",
                        task.node_plan.node_id
                    )));
                }
            }
        }

        let predictions = if matches!(task.node_plan.kind, dag_ml_core::NodeKind::Model) {
            let sample_ids = prediction_sample_ids_for_task(task)?;
            vec![PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: prediction_partition_for_phase(task.phase),
                fold_id: if task.phase == Phase::FitCv {
                    task.fold_id.clone()
                } else {
                    None
                },
                sample_ids: sample_ids.clone(),
                values: vec![
                    vec![stable_handle(task.node_plan.node_id.as_str()) as f64];
                    sample_ids.len()
                ],
                target_names: vec!["y".to_string()],
            }]
        } else {
            Vec::new()
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "out".to_string(),
                HandleRef {
                    handle: stable_handle(task.node_plan.node_id.as_str()),
                    kind: HandleKind::Data,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions,
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
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
                ))?,
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

fn prediction_partition_for_phase(phase: Phase) -> PredictionPartition {
    match phase {
        Phase::FitCv => PredictionPartition::Validation,
        Phase::Refit | Phase::Predict | Phase::Explain => PredictionPartition::Final,
        Phase::Compile | Phase::Plan | Phase::Select => PredictionPartition::Test,
    }
}

fn prediction_sample_ids_for_task(task: &NodeTask) -> dag_ml_core::Result<Vec<SampleId>> {
    if task.phase == Phase::FitCv {
        if let Some(view) = task
            .data_views
            .values()
            .find(|view| view.partition == DataRequestPartition::FoldValidation)
        {
            if let Some(sample_ids) = &view.sample_ids {
                if !sample_ids.is_empty() {
                    return Ok(sample_ids.clone());
                }
            }
        }
    }
    Ok(vec![SampleId::new("sample:mock")?])
}

fn stable_handle(value: &str) -> u64 {
    value.bytes().fold(17u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dag_ml_core::{
        build_execution_plan, build_prediction_cache_record, BundleId, BundlePredictionRequirement,
        CampaignSpec, ControllerManifest, ControllerRegistry, DataBinding, DataProviderViewSpec,
        DataRequestPartition, DataViewPolicy, FoldId, NodeId, NodeKind, NodePlan, VariantId,
    };
    use std::ffi::CStr;

    #[derive(Default)]
    struct DataProviderStub {
        materialize_dataset: DagMlHandle,
        materialize_json: Vec<u8>,
        make_view_parent: DagMlHandle,
        make_view_json: Vec<u8>,
    }

    #[derive(Default)]
    struct ControllerStub {
        task_json: Vec<u8>,
        result_json: Vec<u8>,
        release_count: usize,
    }

    #[derive(Default)]
    struct PredictionCacheStub {
        load_key: Vec<u8>,
        blocks_json: Vec<u8>,
        release_count: usize,
        materialize_json: Vec<u8>,
    }

    unsafe extern "C" fn materialize_stub(
        user_data: *mut c_void,
        dataset: DagMlHandle,
        request_json: DagMlBytesView,
        out_handle: *mut DagMlHandle,
    ) -> DagMlStatusCode {
        if user_data.is_null() || request_json.ptr.is_null() || out_handle.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<DataProviderStub>());
        state.materialize_dataset = dataset;
        state.materialize_json = slice::from_raw_parts(request_json.ptr, request_json.len).to_vec();
        *out_handle = 41;
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn make_view_stub(
        user_data: *mut c_void,
        data: DagMlHandle,
        selector_json: DagMlBytesView,
        out_view: *mut DagMlHandle,
    ) -> DagMlStatusCode {
        if user_data.is_null() || selector_json.ptr.is_null() || out_view.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<DataProviderStub>());
        state.make_view_parent = data;
        state.make_view_json = slice::from_raw_parts(selector_json.ptr, selector_json.len).to_vec();
        *out_view = 42;
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn feature_arrow_stub(
        _user_data: *mut c_void,
        _view: DagMlHandle,
        _feature_set_name: DagMlBytesView,
        out_arrow_array: *mut *mut ArrowArray,
        out_arrow_schema: *mut *mut ArrowSchema,
    ) -> DagMlStatusCode {
        if !out_arrow_array.is_null() {
            *out_arrow_array = std::ptr::null_mut();
        }
        if !out_arrow_schema.is_null() {
            *out_arrow_schema = std::ptr::null_mut();
        }
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn controller_invoke_stub(
        user_data: *mut c_void,
        task_json: DagMlBytesView,
        out_result_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || task_json.ptr.is_null() || out_result_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.task_json = slice::from_raw_parts(task_json.ptr, task_json.len).to_vec();
        let mut data = state.result_json.clone();
        *out_result_json = DagMlOwnedBytes {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            capacity: data.capacity(),
        };
        std::mem::forget(data);
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn controller_invoke_error_stub(
        user_data: *mut c_void,
        task_json: DagMlBytesView,
        out_result_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || task_json.ptr.is_null() || out_result_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.task_json = slice::from_raw_parts(task_json.ptr, task_json.len).to_vec();
        let mut data = b"{}".to_vec();
        *out_result_json = DagMlOwnedBytes {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            capacity: data.capacity(),
        };
        std::mem::forget(data);
        DagMlStatusCode::VALIDATION_ERROR
    }

    unsafe extern "C" fn controller_invoke_unknown_status_stub(
        user_data: *mut c_void,
        task_json: DagMlBytesView,
        out_result_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || task_json.ptr.is_null() || out_result_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.task_json = slice::from_raw_parts(task_json.ptr, task_json.len).to_vec();
        DagMlStatusCode(998)
    }

    unsafe extern "C" fn controller_release_bytes_stub(
        user_data: *mut c_void,
        bytes: DagMlOwnedBytes,
    ) {
        if user_data.is_null() || bytes.ptr.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.release_count += 1;
        drop(Vec::from_raw_parts(bytes.ptr, bytes.len, bytes.capacity));
    }

    unsafe extern "C" fn prediction_cache_load_blocks_stub(
        user_data: *mut c_void,
        requirement_key: DagMlBytesView,
        out_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || requirement_key.ptr.is_null() || out_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.load_key = slice::from_raw_parts(requirement_key.ptr, requirement_key.len).to_vec();
        let mut data = state.blocks_json.clone();
        *out_json = DagMlOwnedBytes {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            capacity: data.capacity(),
        };
        std::mem::forget(data);
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn prediction_cache_load_blocks_error_stub(
        user_data: *mut c_void,
        requirement_key: DagMlBytesView,
        out_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || requirement_key.ptr.is_null() || out_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.load_key = slice::from_raw_parts(requirement_key.ptr, requirement_key.len).to_vec();
        let mut data = b"[]".to_vec();
        *out_json = DagMlOwnedBytes {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            capacity: data.capacity(),
        };
        std::mem::forget(data);
        DagMlStatusCode::VALIDATION_ERROR
    }

    unsafe extern "C" fn prediction_cache_load_blocks_unknown_status_stub(
        user_data: *mut c_void,
        requirement_key: DagMlBytesView,
        out_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || requirement_key.ptr.is_null() || out_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.load_key = slice::from_raw_parts(requirement_key.ptr, requirement_key.len).to_vec();
        DagMlStatusCode(999)
    }

    unsafe extern "C" fn prediction_cache_materialize_stub(
        user_data: *mut c_void,
        request_json: DagMlBytesView,
        out_handle: *mut DagMlHandle,
    ) -> DagMlStatusCode {
        if user_data.is_null() || request_json.ptr.is_null() || out_handle.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.materialize_json = slice::from_raw_parts(request_json.ptr, request_json.len).to_vec();
        *out_handle = 77;
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn prediction_cache_release_bytes_stub(
        user_data: *mut c_void,
        bytes: DagMlOwnedBytes,
    ) {
        if user_data.is_null() || bytes.ptr.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.release_count += 1;
        drop(Vec::from_raw_parts(bytes.ptr, bytes.len, bytes.capacity));
    }

    fn controller_task_result_fixture() -> (ControllerId, NodeTask, NodeResult) {
        let controller_id = ControllerId::new("controller:transform").unwrap();
        let node_id = NodeId::new("transform:scale").unwrap();
        let task = NodeTask {
            run_id: RunId::new("run:cabi.controller").unwrap(),
            node_plan: NodePlan {
                node_id: node_id.clone(),
                kind: NodeKind::Transform,
                controller_id: controller_id.clone(),
                controller_version: "0.1.0".to_string(),
                supported_phases: BTreeSet::from([Phase::FitCv]),
                input_nodes: Vec::new(),
                output_nodes: Vec::new(),
                shape_plan: None,
                data_bindings: Vec::new(),
                params_fingerprint: "params:controller-fixture".to_string(),
            },
            phase: Phase::FitCv,
            variant_id: Some(VariantId::new("variant:controller").unwrap()),
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
            prediction_inputs: BTreeMap::new(),
            seed: Some(42),
        };
        let result = NodeResult {
            node_id: node_id.clone(),
            outputs: BTreeMap::from([(
                "out".to_string(),
                HandleRef {
                    handle: 88,
                    kind: HandleKind::Data,
                    owner_controller: controller_id.clone(),
                },
            )]),
            predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new("lineage:cabi.controller").unwrap(),
                run_id: task.run_id.clone(),
                node_id,
                phase: task.phase,
                controller_id: controller_id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        };
        result.validate_for_task(&task).unwrap();
        (controller_id, task, result)
    }

    #[test]
    fn data_vtable_exposes_feature_arrow_slot() {
        let table = DagMlDataVTable {
            abi_version: 2,
            user_data: std::ptr::null_mut(),
            materialize: Some(materialize_stub),
            make_view: None,
            view_identity: None,
            target_arrow: None,
            feature_arrow: Some(feature_arrow_stub),
            release: None,
            destroy: None,
        };

        assert_eq!(table.abi_version, 2);
        assert!(table.materialize.is_some());
        assert!(table.feature_arrow.is_some());
    }

    #[test]
    fn c_abi_runtime_data_provider_routes_materialize_and_view_requests() {
        let mut state = DataProviderStub::default();
        let table = DagMlDataVTable {
            abi_version: 2,
            user_data: (&mut state as *mut DataProviderStub).cast::<c_void>(),
            materialize: Some(materialize_stub),
            make_view: Some(make_view_stub),
            view_identity: None,
            target_arrow: None,
            feature_arrow: None,
            release: None,
            destroy: None,
        };
        let provider = CAbiRuntimeDataProvider::new(
            ControllerId::new("controller:data.provider").unwrap(),
            7,
            table,
        )
        .unwrap();
        let binding = DataBinding {
            node_id: NodeId::new("model:base").unwrap(),
            input_name: "x".to_string(),
            request_id: "nir-to-tabular".to_string(),
            schema_fingerprint: "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b"
                .to_string(),
            plan_fingerprint: "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d"
                .to_string(),
            relation_fingerprint: Some(
                "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10".to_string(),
            ),
            output_representation: "tabular_numeric".to_string(),
            feature_set_id: Some("x".to_string()),
            source_ids: vec!["nir".to_string()],
            require_relations: true,
            view_policy: DataViewPolicy::default(),
            metadata: BTreeMap::new(),
        };
        let data = provider
            .materialize(&DataMaterializationRequest {
                run_id: RunId::new("run:cabi.data").unwrap(),
                node_id: binding.node_id.clone(),
                input_name: binding.input_name.clone(),
                phase: Phase::FitCv,
                variant_id: Some(VariantId::new("variant:base").unwrap()),
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                binding: binding.clone(),
            })
            .unwrap();

        assert_eq!(data.handle, 41);
        assert_eq!(data.kind, HandleKind::Data);
        assert_eq!(state.materialize_dataset, 7);
        let materialize_json: serde_json::Value =
            serde_json::from_slice(&state.materialize_json).unwrap();
        assert_eq!(materialize_json["phase"], "FIT_CV");
        assert_eq!(materialize_json["request_id"], "nir-to-tabular");
        assert_eq!(materialize_json["source_ids"][0], "nir");

        let view = provider
            .make_view(&DataViewRequest {
                run_id: RunId::new("run:cabi.data").unwrap(),
                node_id: binding.node_id.clone(),
                input_name: binding.input_name.clone(),
                phase: Phase::FitCv,
                variant_id: Some(VariantId::new("variant:base").unwrap()),
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                binding,
                data_handle: data,
                view: DataProviderViewSpec {
                    sample_ids: Some(vec![SampleId::new("s1").unwrap()]),
                    partition: DataRequestPartition::FoldTrain,
                    fold_id: Some(FoldId::new("fold:0").unwrap()),
                    source_ids: Some(vec!["nir".to_string()]),
                    columns: Some(vec!["abs_1000".to_string()]),
                    include_augmented: true,
                    include_excluded: false,
                    extra: BTreeMap::new(),
                },
            })
            .unwrap();

        assert_eq!(view.handle, 42);
        assert_eq!(view.kind, HandleKind::DataView);
        assert_eq!(state.make_view_parent, 41);
        let view_json: serde_json::Value = serde_json::from_slice(&state.make_view_json).unwrap();
        assert_eq!(view_json["partition"], "fold_train");
        assert_eq!(view_json["fold_id"], "fold:0");
        assert_eq!(view_json["sample_ids"][0], "s1");
        assert_eq!(view_json["columns"][0], "abs_1000");
    }

    #[test]
    fn c_abi_runtime_controller_routes_node_task_and_result_json() {
        let (controller_id, task, expected) = controller_task_result_fixture();
        let mut state = ControllerStub {
            result_json: serde_json::to_vec(&expected).unwrap(),
            ..Default::default()
        };
        let table = DagMlControllerVTable {
            abi_version: 2,
            user_data: (&mut state as *mut ControllerStub).cast::<c_void>(),
            clone_with: None,
            describe: None,
            fit: None,
            predict: None,
            invoke: Some(controller_invoke_stub),
            release_bytes: Some(controller_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let controller = CAbiRuntimeController::new(controller_id.clone(), table).unwrap();
        let actual = controller.invoke(&task).unwrap();
        assert_eq!(controller.controller_id(), &controller_id);
        assert_eq!(actual, expected);
        assert_eq!(state.release_count, 1);

        let task_json: serde_json::Value = serde_json::from_slice(&state.task_json).unwrap();
        assert_eq!(task_json["node_plan"]["node_id"], "transform:scale");
        assert_eq!(task_json["phase"], "FIT_CV");
        assert_eq!(task_json["seed"], 42);
    }

    #[test]
    fn c_abi_runtime_controller_releases_error_buffers() {
        let (controller_id, task, _) = controller_task_result_fixture();
        let mut state = ControllerStub::default();
        let table = DagMlControllerVTable {
            abi_version: 2,
            user_data: (&mut state as *mut ControllerStub).cast::<c_void>(),
            clone_with: None,
            describe: None,
            fit: None,
            predict: None,
            invoke: Some(controller_invoke_error_stub),
            release_bytes: Some(controller_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let controller = CAbiRuntimeController::new(controller_id, table).unwrap();
        let error = controller.invoke(&task).unwrap_err();
        assert!(format!("{error}").contains("controller invoke rejected request"));
        assert_eq!(state.release_count, 1);

        let task_json: serde_json::Value = serde_json::from_slice(&state.task_json).unwrap();
        assert_eq!(task_json["node_plan"]["node_id"], "transform:scale");
    }

    #[test]
    fn c_abi_runtime_controller_rejects_unknown_status_codes() {
        let (controller_id, task, _) = controller_task_result_fixture();
        let mut state = ControllerStub::default();
        let table = DagMlControllerVTable {
            abi_version: 2,
            user_data: (&mut state as *mut ControllerStub).cast::<c_void>(),
            clone_with: None,
            describe: None,
            fit: None,
            predict: None,
            invoke: Some(controller_invoke_unknown_status_stub),
            release_bytes: Some(controller_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let controller = CAbiRuntimeController::new(controller_id, table).unwrap();
        let error = controller.invoke(&task).unwrap_err();
        assert!(format!("{error}").contains("unknown status code 998"));
        assert_eq!(state.release_count, 0);

        let task_json: serde_json::Value = serde_json::from_slice(&state.task_json).unwrap();
        assert_eq!(task_json["node_plan"]["node_id"], "transform:scale");
    }

    #[test]
    fn c_abi_prediction_cache_store_loads_blocks_and_materializes_handles() {
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            fold_ids: vec![FoldId::new("fold:0").unwrap()],
            sample_ids: vec![SampleId::new("sample:1").unwrap()],
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let blocks = vec![PredictionBlock {
            prediction_id: Some("prediction:model:base.fold0".to_string()),
            producer_node: requirement.producer_node.clone(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: requirement.sample_ids.clone(),
            values: vec![vec![0.42]],
            target_names: vec!["y".to_string()],
        }];
        let cache = build_prediction_cache_record(&requirement, &blocks).unwrap();
        let mut state = PredictionCacheStub {
            blocks_json: serde_json::to_vec(&blocks).unwrap(),
            ..Default::default()
        };
        let table = DagMlPredictionCacheVTable {
            abi_version: 1,
            user_data: (&mut state as *mut PredictionCacheStub).cast::<c_void>(),
            load_blocks: Some(prediction_cache_load_blocks_stub),
            materialize: Some(prediction_cache_materialize_stub),
            release_bytes: Some(prediction_cache_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let store = CAbiRuntimePredictionCacheStore::new(table).unwrap();
        let loaded = store.load_blocks(&requirement.key()).unwrap();
        assert_eq!(loaded, blocks);
        assert_eq!(
            String::from_utf8(state.load_key.clone()).unwrap(),
            requirement.key()
        );
        assert_eq!(state.release_count, 1);

        let handle = store
            .materialize(&PredictionCacheMaterializationRequest {
                run_id: RunId::new("run:prediction.cache.abi").unwrap(),
                bundle_id: BundleId::new("bundle:prediction.cache.abi").unwrap(),
                phase: Phase::Refit,
                variant_id: None,
                requirement: requirement.clone(),
                cache,
                producer_controller_id: ControllerId::new("controller:model").unwrap(),
            })
            .unwrap();
        assert_eq!(handle.handle, 77);
        assert_eq!(handle.kind, HandleKind::Prediction);
        assert_eq!(
            handle.owner_controller,
            ControllerId::new("controller:model").unwrap()
        );
        let request_json: serde_json::Value =
            serde_json::from_slice(&state.materialize_json).unwrap();
        assert_eq!(request_json["requirement"]["producer_node"], "model:base");
        assert_eq!(request_json["cache"]["requirement_key"], requirement.key());
    }

    #[test]
    fn c_abi_prediction_cache_store_releases_error_buffers() {
        let mut state = PredictionCacheStub::default();
        let table = DagMlPredictionCacheVTable {
            abi_version: 1,
            user_data: (&mut state as *mut PredictionCacheStub).cast::<c_void>(),
            load_blocks: Some(prediction_cache_load_blocks_error_stub),
            materialize: Some(prediction_cache_materialize_stub),
            release_bytes: Some(prediction_cache_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let store = CAbiRuntimePredictionCacheStore::new(table).unwrap();
        let error = store.load_blocks("requirement:error").unwrap_err();
        assert!(format!("{error}").contains("prediction cache load_blocks rejected request"));
        assert_eq!(
            String::from_utf8(state.load_key.clone()).unwrap(),
            "requirement:error"
        );
        assert_eq!(state.release_count, 1);
    }

    #[test]
    fn c_abi_prediction_cache_store_rejects_unknown_status_codes() {
        let mut state = PredictionCacheStub::default();
        let table = DagMlPredictionCacheVTable {
            abi_version: 1,
            user_data: (&mut state as *mut PredictionCacheStub).cast::<c_void>(),
            load_blocks: Some(prediction_cache_load_blocks_unknown_status_stub),
            materialize: Some(prediction_cache_materialize_stub),
            release_bytes: Some(prediction_cache_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let store = CAbiRuntimePredictionCacheStore::new(table).unwrap();
        let error = store.load_blocks("requirement:unknown").unwrap_err();
        assert!(format!("{error}").contains("unknown status code 999"));
        assert_eq!(
            String::from_utf8(state.load_key.clone()).unwrap(),
            "requirement:unknown"
        );
        assert_eq!(state.release_count, 0);
    }

    #[test]
    fn validates_graph_json_over_abi() {
        let graph = include_bytes!("../../../examples/minimal_graph.json");
        let mut error = DagMlString::default();

        let status = unsafe { dagml_graph_validate_json(graph.as_ptr(), graph.len(), &mut error) };

        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());
    }

    #[test]
    fn selects_grouped_candidates_over_abi() {
        let policy = include_bytes!("../../../examples/fixtures/bundle/selection_policy_rmse.json");
        let candidates =
            include_bytes!("../../../examples/fixtures/bundle/candidate_scores_demo.json");
        let groups = include_bytes!("../../../examples/fixtures/bundle/candidate_groups_demo.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_select_candidate_groups_json(
                policy.as_ptr(),
                policy.len(),
                candidates.as_ptr(),
                candidates.len(),
                groups.as_ptr(),
                groups.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let decisions: BTreeMap<String, SelectionDecision> = serde_json::from_slice(json).unwrap();
        assert_eq!(
            decisions["merge"].selected_candidate_id,
            "merge:m1.predictions_plus_original"
        );
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn validates_bundle_replay_contracts_over_abi() {
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let request =
            include_bytes!("../../../examples/fixtures/bundle/replay_request_predict.json");
        let envelope =
            include_str!("../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json");
        let envelopes = format!(r#"{{"model:base.x":{envelope}}}"#);
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_execution_bundle_validate_json(bundle.as_ptr(), bundle.len(), &mut error)
        };
        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());

        let status = unsafe {
            dagml_execution_bundle_validate_replay_envelopes_json(
                bundle.as_ptr(),
                bundle.len(),
                envelopes.as_ptr(),
                envelopes.len(),
                &mut error,
            )
        };
        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());

        let status = unsafe {
            dagml_replay_request_validate_for_bundle_json(
                bundle.as_ptr(),
                bundle.len(),
                request.as_ptr(),
                request.len(),
                &mut error,
            )
        };
        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());
    }

    #[test]
    fn validates_prediction_cache_payload_over_abi() {
        let bundle = include_bytes!(
            "../../../examples/generated/execution_bundle_branch_merge_cv_refit.json"
        );
        let payload = include_bytes!(
            "../../../examples/generated/prediction_cache_branch_merge_cv_refit.json"
        );
        let refit_request = include_bytes!(
            "../../../examples/fixtures/bundle/replay_request_branch_merge_refit.json"
        );
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_prediction_cache_payload_validate_for_bundle_json(
                bundle.as_ptr(),
                bundle.len(),
                payload.as_ptr(),
                payload.len(),
                &mut error,
            )
        };
        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());

        let status = unsafe {
            dagml_replay_request_validate_for_bundle_json(
                bundle.as_ptr(),
                bundle.len(),
                refit_request.as_ptr(),
                refit_request.len(),
                &mut error,
            )
        };
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(!error.ptr.is_null());
        unsafe { dagml_string_free(error) };
        error = DagMlString::default();

        let status = unsafe {
            dagml_replay_request_validate_for_bundle_with_prediction_cache_payload_json(
                bundle.as_ptr(),
                bundle.len(),
                refit_request.as_ptr(),
                refit_request.len(),
                payload.as_ptr(),
                payload.len(),
                &mut error,
            )
        };
        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());
    }

    #[test]
    fn executes_mock_replay_over_abi() {
        let plan = fixture_plan_json();
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let request =
            include_bytes!("../../../examples/fixtures/bundle/replay_request_predict.json");
        let envelope =
            include_str!("../../../examples/fixtures/data/coordinator_data_plan_envelope_nir.json");
        let envelopes = format!(r#"{{"model:base.x":{envelope}}}"#);
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_mock_replay_execute_json(
                plan.as_ptr(),
                plan.len(),
                bundle.as_ptr(),
                bundle.len(),
                request.as_ptr(),
                request.len(),
                envelopes.as_ptr(),
                envelopes.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK);
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let summary: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(summary["bundle_id"], "bundle:cli.demo");
        assert_eq!(summary["result_count"], 2);
        assert_eq!(summary["prediction_block_count"], 1);
        assert_eq!(summary["data_handle_count"], 1);
        assert_eq!(summary["data_view_count"], 1);
        assert_eq!(summary["artifact_handle_count"], 1);
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn invalid_bundle_returns_error_string() {
        let invalid = br#"{"bundle_id":"bundle:bad"}"#;
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_execution_bundle_validate_json(invalid.as_ptr(), invalid.len(), &mut error)
        };

        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(!error.ptr.is_null());
        let message = unsafe { CStr::from_ptr(error.ptr) }
            .to_string_lossy()
            .into_owned();
        assert!(message.contains("failed to parse execution bundle JSON"));
        unsafe { dagml_string_free(error) };
    }

    #[test]
    fn null_json_pointer_is_invalid_argument() {
        let mut error = DagMlString::default();

        let status = unsafe { dagml_graph_validate_json(std::ptr::null(), 0, &mut error) };

        assert_eq!(status, DagMlStatusCode::INVALID_ARGUMENT);
        assert!(!error.ptr.is_null());
        unsafe { dagml_string_free(error) };
    }

    fn fixture_plan_json() -> Vec<u8> {
        let graph: GraphSpec =
            serde_json::from_str(include_str!("../../../examples/minimal_graph.json")).unwrap();
        let campaign: CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_oof_generation.json"
        ))
        .unwrap();
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(include_str!("../../../examples/controller_manifests.json"))
                .unwrap();
        let mut registry = ControllerRegistry::new();
        for manifest in manifests {
            registry.register(manifest).unwrap();
        }
        let plan = build_execution_plan("plan:cli.bundle", graph, campaign, &registry).unwrap();
        serde_json::to_vec(&plan).unwrap()
    }
}
