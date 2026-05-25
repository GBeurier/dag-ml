use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

use dag_ml_core::GraphSpec;

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
    clear_error(error_out);
    if json_ptr.is_null() {
        set_error(error_out, "json pointer is null");
        return DagMlStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<GraphSpec>(json) {
        Ok(graph) => match graph.validate() {
            Ok(()) => DagMlStatusCode::Ok,
            Err(error) => {
                set_error(error_out, error.to_string());
                DagMlStatusCode::ValidationError
            }
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_graph_json_over_abi() {
        let graph = include_bytes!("../../../examples/minimal_graph.json");
        let mut error = DagMlString::default();

        let status = unsafe { dagml_graph_validate_json(graph.as_ptr(), graph.len(), &mut error) };

        assert_eq!(status, DagMlStatusCode::Ok);
        assert!(error.ptr.is_null());
    }
}
