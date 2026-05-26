use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

use dag_ml_core::{
    select_candidate, select_candidate_groups, CandidateScore, ExecutionBundle,
    ExternalDataPlanEnvelope, GraphSpec, ReplayPhaseRequest, SelectionDecision, SelectionPolicy,
};
use serde::{de::DeserializeOwned, Serialize};

pub type DagMlHandle = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DagMlStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    ValidationError = 2,
    Panic = 255,
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
            out_arrow_array: *mut *mut c_void,
            out_arrow_schema: *mut *mut c_void,
        ) -> DagMlStatusCode,
    >,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void, handle: DagMlHandle)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlDataVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub make_view: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            data: DagMlHandle,
            sample_ids_json: DagMlBytesView,
            out_view: *mut DagMlHandle,
        ) -> DagMlStatusCode,
    >,
    pub view_identity: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlHandle,
            out_arrow_array: *mut *mut c_void,
            out_arrow_schema: *mut *mut c_void,
        ) -> DagMlStatusCode,
    >,
    pub target_arrow: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlHandle,
            target_name: DagMlBytesView,
            out_arrow_array: *mut *mut c_void,
            out_arrow_schema: *mut *mut c_void,
        ) -> DagMlStatusCode,
    >,
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
            DagMlStatusCode::ValidationError
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
            DagMlStatusCode::ValidationError
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
        Ok(()) => DagMlStatusCode::Ok,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::ValidationError
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
        Ok(()) => DagMlStatusCode::Ok,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::ValidationError
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
        Ok(()) => DagMlStatusCode::Ok,
        Err(error) => {
            set_error(error_out, error.to_string());
            DagMlStatusCode::ValidationError
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
        return Err(DagMlStatusCode::InvalidArgument);
    }
    let json = slice::from_raw_parts(json_ptr, json_len);
    serde_json::from_slice::<T>(json).map_err(|error| {
        set_error(error_out, format!("failed to parse {label} JSON: {error}"));
        DagMlStatusCode::ValidationError
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
        return DagMlStatusCode::InvalidArgument;
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
            DagMlStatusCode::Ok
        }
        Err(error) => {
            set_error(
                error_out,
                format!("failed to serialize output JSON: {error}"),
            );
            DagMlStatusCode::ValidationError
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn validates_graph_json_over_abi() {
        let graph = include_bytes!("../../../examples/minimal_graph.json");
        let mut error = DagMlString::default();

        let status = unsafe { dagml_graph_validate_json(graph.as_ptr(), graph.len(), &mut error) };

        assert_eq!(status, DagMlStatusCode::Ok);
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

        assert_eq!(status, DagMlStatusCode::Ok);
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
        assert_eq!(status, DagMlStatusCode::Ok);
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
        assert_eq!(status, DagMlStatusCode::Ok);
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
        assert_eq!(status, DagMlStatusCode::Ok);
        assert!(error.ptr.is_null());
    }

    #[test]
    fn invalid_bundle_returns_error_string() {
        let invalid = br#"{"bundle_id":"bundle:bad"}"#;
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_execution_bundle_validate_json(invalid.as_ptr(), invalid.len(), &mut error)
        };

        assert_eq!(status, DagMlStatusCode::ValidationError);
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

        assert_eq!(status, DagMlStatusCode::InvalidArgument);
        assert!(!error.ptr.is_null());
        unsafe { dagml_string_free(error) };
    }
}
