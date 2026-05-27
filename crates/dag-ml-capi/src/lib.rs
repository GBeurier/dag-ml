use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;
use std::sync::Mutex;

use dag_ml_core::{
    build_execution_plan, build_openlineage_run_event, build_research_provenance_export,
    compile_pipeline_dsl, compile_pipeline_dsl_with_generation,
    regression_report_to_candidate_score, score_regression_aggregated_block,
    score_regression_prediction_block, select_candidate, select_candidate_groups,
    AggregatedPredictionBlock, ArtifactMaterializationRequest, BundlePredictionCachePayload,
    BundlePredictionCachePayloadSet, BundleReplayExecution, CampaignSpec, CandidateScore,
    ControllerId, ControllerManifest, ControllerRegistry, DagMlError, DataMaterializationRequest,
    DataOutputProvenance, DataRequestPartition, DataViewRequest, ExecutionBundle, ExecutionPlan,
    ExternalDataPlanEnvelope, FileArtifactManifest, FilePredictionCacheManifest, GraphSpec,
    HandleKind, HandleRef, InMemoryArtifactStore, InMemoryDataProvider, LineageId, LineageRecord,
    NodeResult, NodeTask, OpenLineageRunEventOptions, Phase, PipelineDslSpec, PredictionBlock,
    PredictionCacheMaterializationRequest, PredictionLevel, PredictionPartition, PredictionUnitId,
    RegressionMetricKind, RegressionMetricReport, RegressionTargetBlock, ReplayPhaseRequest,
    RunContext, RunId, RuntimeArtifactStore, RuntimeController, RuntimeControllerRegistry,
    RuntimeDataProvider, RuntimePredictionCacheStore, SampleId, SelectionDecision, SelectionPolicy,
    SequentialScheduler, DATA_OUTPUT_PROVENANCE_KEY, DATA_OUTPUT_PROVENANCE_SCHEMA_ID,
    DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION,
};
use serde::{de::DeserializeOwned, Serialize};

pub type DagMlHandle = u64;
pub const DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION: u32 = 2;
pub const DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION: u32 = 3;
pub const DAG_ML_ARTIFACT_STORE_VTABLE_BORROWED_ABI_VERSION: u32 = 1;
pub const DAG_ML_ARTIFACT_STORE_VTABLE_OWNED_ABI_VERSION: u32 = 2;
pub const DAG_ML_PREDICTION_CACHE_VTABLE_BORROWED_ABI_VERSION: u32 = 1;
pub const DAG_ML_PREDICTION_CACHE_VTABLE_OWNED_ABI_VERSION: u32 = 2;
pub const DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION: u32 = 1;
pub const DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION: u32 = DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION;
pub const DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION: u32 = 2;
pub const DAG_ML_HANDLE_KIND_DATA: u32 = 1;
pub const DAG_ML_HANDLE_KIND_DATA_VIEW: u32 = 2;
pub const DAG_ML_HANDLE_KIND_MODEL: u32 = 3;
pub const DAG_ML_HANDLE_KIND_ARTIFACT: u32 = 4;
pub const DAG_ML_HANDLE_KIND_PREDICTION: u32 = 5;
pub const DAG_ML_HANDLE_KIND_RELATION: u32 = 6;

#[derive(Serialize)]
struct DataOutputProvenanceContractInfo {
    schema_version: u32,
    extra_key: &'static str,
    schema_id: &'static str,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DagMlStatusCode(pub u32);

impl DagMlStatusCode {
    pub const OK: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const VALIDATION_ERROR: Self = Self(2);
    pub const PANIC: Self = Self(255);
}

impl Default for DagMlStatusCode {
    fn default() -> Self {
        Self::OK
    }
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DagMlHandleRef {
    pub handle: DagMlHandle,
    pub kind: u32,
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
#[derive(Clone, Copy, Debug)]
pub struct DagMlF64Tensor {
    pub ptr: *mut f64,
    pub len: usize,
    pub capacity: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Default for DagMlF64Tensor {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
            rows: 0,
            cols: 0,
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlArtifactStoreVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub materialize: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            request_json: DagMlBytesView,
            out_handle: *mut DagMlHandleRef,
        ) -> DagMlStatusCode,
    >,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void, handle: DagMlHandle)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlControllerBinding {
    pub controller_id: DagMlBytesView,
    pub vtable: DagMlControllerVTable,
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

/// Releases an F64 tensor allocated by DAG-ML.
///
/// # Safety
///
/// `value.ptr` must either be null or a pointer previously returned by a
/// DAG-ML C ABI function in a `DagMlF64Tensor`. Passing any other pointer, or
/// freeing the same tensor twice, is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagml_f64_tensor_free(value: DagMlF64Tensor) {
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

/// Returns the public C ABI contract for propagated data-output provenance.
///
/// Host bindings should look for `extra_key` inside `DataProviderViewSpec.extra`
/// and parse that value as the schema identified by `schema_id`.
///
/// # Safety
///
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`. Any
/// returned bytes must be released with `dagml_owned_bytes_free`.
/// `error_out` follows the same ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_data_output_provenance_contract_json(
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let contract = DataOutputProvenanceContractInfo {
        schema_version: DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION,
        extra_key: DATA_OUTPUT_PROVENANCE_KEY,
        schema_id: DATA_OUTPUT_PROVENANCE_SCHEMA_ID,
    };
    write_owned_json(out_json, error_out, &contract)
}

/// Validates a `DataOutputProvenance` JSON object.
///
/// This lets non-Rust controllers reject unsupported provenance schema versions,
/// malformed fingerprints and inconsistent shape deltas before trusting a
/// propagated data view's `extra[DAG_ML_DATA_OUTPUT_PROVENANCE_EXTRA_KEY]`.
///
/// # Safety
///
/// Same pointer and error ownership rules as `dagml_graph_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_data_output_provenance_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    validate_json::<DataOutputProvenance>(
        json_ptr,
        json_len,
        error_out,
        "data output provenance",
        DataOutputProvenance::validate,
    )
}

/// Compiles a strict JSON `PipelineDslSpec` into a canonical `GraphSpec` JSON.
///
/// This compiler is pure: it lowers host-declared operator references and
/// branch/merge structure into DAG-ML graph contracts without instantiating
/// operators or touching data.
///
/// # Safety
///
/// Same pointer and output ownership rules as `dagml_graph_validate_json` and
/// other JSON-output helpers.
#[no_mangle]
pub unsafe extern "C" fn dagml_pipeline_dsl_compile_json(
    dsl_ptr: *const u8,
    dsl_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let dsl = match parse_json_ptr::<PipelineDslSpec>(dsl_ptr, dsl_len, error_out, "pipeline DSL") {
        Ok(dsl) => dsl,
        Err(status) => return status,
    };
    match compile_pipeline_dsl(&dsl) {
        Ok(graph) => write_owned_json(out_json, error_out, &graph),
        Err(error) => validation_error(error_out, error),
    }
}

/// Compiles a strict JSON `PipelineDslSpec` into `CompiledPipelineDsl` JSON.
///
/// The artifact contains the canonical graph, extracted `GenerationSpec`,
/// validated shape/data-binding fragments, a `CampaignSpec` template, and the
/// generation fingerprint copied into `graph.search_space_fingerprint` when
/// variants are present.
///
/// # Safety
///
/// Same pointer and output ownership rules as `dagml_pipeline_dsl_compile_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_pipeline_dsl_compile_artifact_json(
    dsl_ptr: *const u8,
    dsl_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let dsl = match parse_json_ptr::<PipelineDslSpec>(dsl_ptr, dsl_len, error_out, "pipeline DSL") {
        Ok(dsl) => dsl,
        Err(status) => return status,
    };
    match compile_pipeline_dsl_with_generation(&dsl) {
        Ok(compiled) => write_owned_json(out_json, error_out, &compiled),
        Err(error) => validation_error(error_out, error),
    }
}

/// Compiles a strict JSON `PipelineDslSpec` and controller manifests into
/// validated `ExecutionPlan` JSON.
///
/// This is the direct non-Rust binding path for the DSL artifact: splits,
/// generation, shape plans and data bindings are taken from the compiled
/// campaign template, while controller resolution and planner invariants still
/// run through Rust.
///
/// # Safety
///
/// Same pointer and output ownership rules as `dagml_execution_plan_build_json`.
#[no_mangle]
pub unsafe extern "C" fn dagml_pipeline_dsl_execution_plan_build_json(
    dsl_ptr: *const u8,
    dsl_len: usize,
    controllers_ptr: *const u8,
    controllers_len: usize,
    plan_id: DagMlBytesView,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let dsl = match parse_json_ptr::<PipelineDslSpec>(dsl_ptr, dsl_len, error_out, "pipeline DSL") {
        Ok(dsl) => dsl,
        Err(status) => return status,
    };
    let manifests = match parse_json_ptr::<Vec<ControllerManifest>>(
        controllers_ptr,
        controllers_len,
        error_out,
        "controller manifests",
    ) {
        Ok(manifests) => manifests,
        Err(status) => return status,
    };
    let plan_id = match parse_utf8_view(plan_id, error_out, "execution plan id") {
        Ok(plan_id) => plan_id,
        Err(status) => return status,
    };
    let registry = match controller_registry_from_manifests(manifests) {
        Ok(registry) => registry,
        Err(error) => return validation_error(error_out, error),
    };
    let compiled = match compile_pipeline_dsl_with_generation(&dsl) {
        Ok(compiled) => compiled,
        Err(error) => return validation_error(error_out, error),
    };
    match build_execution_plan(
        plan_id,
        compiled.graph,
        compiled.campaign_template,
        &registry,
    ) {
        Ok(plan) => write_owned_json(out_json, error_out, &plan),
        Err(error) => validation_error(error_out, error),
    }
}

/// Returns deterministic topological levels for parallel node scheduling.
///
/// # Safety
///
/// Same pointer and output ownership rules as `dagml_graph_validate_json` and
/// other JSON-output helpers.
#[no_mangle]
pub unsafe extern "C" fn dagml_graph_parallel_levels_json(
    json_ptr: *const u8,
    json_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let graph = match parse_json_ptr::<GraphSpec>(json_ptr, json_len, error_out, "graph") {
        Ok(graph) => graph,
        Err(status) => return status,
    };
    match graph.parallel_levels() {
        Ok(levels) => write_owned_json(out_json, error_out, &levels),
        Err(error) => validation_error(error_out, error),
    }
}

/// Builds an `ExecutionPlan` from graph, campaign and controller manifests.
///
/// # Safety
///
/// Input JSON pointers follow the same rules as `dagml_graph_validate_json`.
/// `plan_id` must point to UTF-8 bytes. `out_json` must point to writable
/// memory for one `DagMlOwnedBytes`; returned bytes must be released with
/// `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_execution_plan_build_json(
    graph_ptr: *const u8,
    graph_len: usize,
    campaign_ptr: *const u8,
    campaign_len: usize,
    controllers_ptr: *const u8,
    controllers_len: usize,
    plan_id: DagMlBytesView,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dagml_execution_plan_build_json_impl(ExecutionPlanBuildJsonArgs {
            graph_ptr,
            graph_len,
            campaign_ptr,
            campaign_len,
            controllers_ptr,
            controllers_len,
            plan_id,
            out_json,
            error_out,
        })
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(
                error_out,
                "panic while building execution plan through C ABI",
            );
            DagMlStatusCode::PANIC
        }
    }
}

/// Returns a deterministic phase execution schedule from an `ExecutionPlan`.
///
/// # Safety
///
/// Input JSON and output ownership follow `dagml_execution_plan_build_json`.
/// `phase` must point to UTF-8 bytes with one ABI phase name such as `FIT_CV`.
#[no_mangle]
pub unsafe extern "C" fn dagml_execution_plan_schedule_json(
    plan_ptr: *const u8,
    plan_len: usize,
    phase: DagMlBytesView,
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
    let phase = match parse_phase_view(phase, error_out, "phase") {
        Ok(phase) => phase,
        Err(status) => return status,
    };
    match plan.campaign_phase_schedule(phase) {
        Ok(schedule) => write_owned_json(out_json, error_out, &schedule),
        Err(error) => validation_error(error_out, error),
    }
}

struct ExecutionPlanBuildJsonArgs {
    graph_ptr: *const u8,
    graph_len: usize,
    campaign_ptr: *const u8,
    campaign_len: usize,
    controllers_ptr: *const u8,
    controllers_len: usize,
    plan_id: DagMlBytesView,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
}

unsafe fn dagml_execution_plan_build_json_impl(
    args: ExecutionPlanBuildJsonArgs,
) -> DagMlStatusCode {
    let ExecutionPlanBuildJsonArgs {
        graph_ptr,
        graph_len,
        campaign_ptr,
        campaign_len,
        controllers_ptr,
        controllers_len,
        plan_id,
        out_json,
        error_out,
    } = args;
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let graph = match parse_json_ptr::<GraphSpec>(graph_ptr, graph_len, error_out, "graph") {
        Ok(graph) => graph,
        Err(status) => return status,
    };
    let campaign =
        match parse_json_ptr::<CampaignSpec>(campaign_ptr, campaign_len, error_out, "campaign") {
            Ok(campaign) => campaign,
            Err(status) => return status,
        };
    let manifests = match parse_json_ptr::<Vec<ControllerManifest>>(
        controllers_ptr,
        controllers_len,
        error_out,
        "controller manifests",
    ) {
        Ok(manifests) => manifests,
        Err(status) => return status,
    };
    let plan_id = match parse_utf8_view(plan_id, error_out, "execution plan id") {
        Ok(plan_id) => plan_id,
        Err(status) => return status,
    };
    let registry = match controller_registry_from_manifests(manifests) {
        Ok(registry) => registry,
        Err(error) => return validation_error(error_out, error),
    };
    match build_execution_plan(plan_id, graph, campaign, &registry) {
        Ok(plan) => write_owned_json(out_json, error_out, &plan),
        Err(error) => validation_error(error_out, error),
    }
}

fn controller_registry_from_manifests(
    manifests: Vec<ControllerManifest>,
) -> dag_ml_core::Result<ControllerRegistry> {
    let mut registry = ControllerRegistry::new();
    for manifest in manifests {
        registry.register(manifest)?;
    }
    Ok(registry)
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

/// Scores a sample-level `PredictionBlock` against a `RegressionTargetBlock`.
///
/// `metrics_json` must be a JSON array of metric names, for example
/// `["rmse","mae","r2"]`.
///
/// # Safety
///
/// Input pointers follow the same rules as `dagml_graph_validate_json`.
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`; returned
/// bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_score_regression_prediction_block_json(
    predictions_ptr: *const u8,
    predictions_len: usize,
    targets_ptr: *const u8,
    targets_len: usize,
    metrics_ptr: *const u8,
    metrics_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let predictions = match parse_json_ptr::<PredictionBlock>(
        predictions_ptr,
        predictions_len,
        error_out,
        "sample prediction block",
    ) {
        Ok(predictions) => predictions,
        Err(status) => return status,
    };
    let targets = match parse_json_ptr::<RegressionTargetBlock>(
        targets_ptr,
        targets_len,
        error_out,
        "regression target block",
    ) {
        Ok(targets) => targets,
        Err(status) => return status,
    };
    let metrics = match parse_json_ptr::<Vec<RegressionMetricKind>>(
        metrics_ptr,
        metrics_len,
        error_out,
        "regression metric list",
    ) {
        Ok(metrics) => metrics,
        Err(status) => return status,
    };
    match score_regression_prediction_block(&predictions, &targets, &metrics) {
        Ok(report) => write_owned_json(out_json, error_out, &report),
        Err(error) => validation_error(error_out, error),
    }
}

/// Scores an `AggregatedPredictionBlock` against a `RegressionTargetBlock`.
///
/// `metrics_json` must be a JSON array of metric names, for example
/// `["rmse","mae","r2"]`.
///
/// # Safety
///
/// Input pointers follow the same rules as `dagml_graph_validate_json`.
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`; returned
/// bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_score_regression_aggregated_block_json(
    predictions_ptr: *const u8,
    predictions_len: usize,
    targets_ptr: *const u8,
    targets_len: usize,
    metrics_ptr: *const u8,
    metrics_len: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let predictions = match parse_json_ptr::<AggregatedPredictionBlock>(
        predictions_ptr,
        predictions_len,
        error_out,
        "aggregated prediction block",
    ) {
        Ok(predictions) => predictions,
        Err(status) => return status,
    };
    let targets = match parse_json_ptr::<RegressionTargetBlock>(
        targets_ptr,
        targets_len,
        error_out,
        "regression target block",
    ) {
        Ok(targets) => targets,
        Err(status) => return status,
    };
    let metrics = match parse_json_ptr::<Vec<RegressionMetricKind>>(
        metrics_ptr,
        metrics_len,
        error_out,
        "regression metric list",
    ) {
        Ok(metrics) => metrics,
        Err(status) => return status,
    };
    match score_regression_aggregated_block(&predictions, &targets, &metrics) {
        Ok(report) => write_owned_json(out_json, error_out, &report),
        Err(error) => validation_error(error_out, error),
    }
}

/// Converts a sample-level prediction block JSON into an owned row-major F64 tensor.
///
/// The tensor has `rows == sample_ids.len()`, `cols == prediction width`, and
/// `len == rows * cols`. The returned data must be released with
/// `dagml_f64_tensor_free`.
///
/// # Safety
///
/// `predictions_ptr` must point to `predictions_len` readable bytes.
/// `out_tensor` must point to writable memory for one `DagMlF64Tensor`.
#[no_mangle]
pub unsafe extern "C" fn dagml_prediction_block_f64_tensor_json(
    predictions_ptr: *const u8,
    predictions_len: usize,
    out_tensor: *mut DagMlF64Tensor,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_f64_tensor(out_tensor);
    let predictions = match parse_json_ptr::<PredictionBlock>(
        predictions_ptr,
        predictions_len,
        error_out,
        "sample prediction block",
    ) {
        Ok(predictions) => predictions,
        Err(status) => return status,
    };
    let width = match predictions.validate_shape() {
        Ok(width) => width,
        Err(error) => return validation_error(error_out, error),
    };
    let values = match flatten_f64_rows(
        "sample prediction block",
        predictions.producer_node.as_str(),
        &predictions.values,
        width,
    ) {
        Ok(values) => values,
        Err(error) => return validation_error(error_out, error),
    };
    write_f64_tensor(
        out_tensor,
        error_out,
        values,
        predictions.sample_ids.len(),
        width,
    )
}

/// Converts an aggregated target/group prediction block JSON into an owned
/// row-major F64 tensor.
///
/// The tensor has `rows == unit_ids.len()`, `cols == prediction width`, and
/// `len == rows * cols`. The returned data must be released with
/// `dagml_f64_tensor_free`.
///
/// # Safety
///
/// `predictions_ptr` must point to `predictions_len` readable bytes.
/// `out_tensor` must point to writable memory for one `DagMlF64Tensor`.
#[no_mangle]
pub unsafe extern "C" fn dagml_aggregated_prediction_block_f64_tensor_json(
    predictions_ptr: *const u8,
    predictions_len: usize,
    out_tensor: *mut DagMlF64Tensor,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_f64_tensor(out_tensor);
    let predictions = match parse_json_ptr::<AggregatedPredictionBlock>(
        predictions_ptr,
        predictions_len,
        error_out,
        "aggregated prediction block",
    ) {
        Ok(predictions) => predictions,
        Err(status) => return status,
    };
    let width = match predictions.validate_shape() {
        Ok(width) => width,
        Err(error) => return validation_error(error_out, error),
    };
    let values = match flatten_f64_rows(
        "aggregated prediction block",
        predictions.producer_node.as_str(),
        &predictions.values,
        width,
    ) {
        Ok(values) => values,
        Err(error) => return validation_error(error_out, error),
    };
    write_f64_tensor(
        out_tensor,
        error_out,
        values,
        predictions.unit_ids.len(),
        width,
    )
}

/// Converts a `RegressionMetricReport` to a selection `CandidateScore`.
///
/// # Safety
///
/// Input pointers follow the same rules as `dagml_graph_validate_json`.
/// `candidate_id` must point to UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn dagml_regression_report_candidate_score_json(
    report_ptr: *const u8,
    report_len: usize,
    candidate_id: DagMlBytesView,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_owned_bytes(out_json);
    let report = match parse_json_ptr::<RegressionMetricReport>(
        report_ptr,
        report_len,
        error_out,
        "regression metric report",
    ) {
        Ok(report) => report,
        Err(status) => return status,
    };
    let candidate_id = match parse_utf8_view(candidate_id, error_out, "candidate id") {
        Ok(candidate_id) => candidate_id,
        Err(status) => return status,
    };
    match regression_report_to_candidate_score(candidate_id, report) {
        Ok(score) => write_owned_json(out_json, error_out, &score),
        Err(error) => validation_error(error_out, error),
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

/// Exports one validated prediction-cache payload requirement as an owned
/// row-major F64 tensor plus JSON metadata.
///
/// The payload set is first validated against the bundle, then the requested
/// requirement's blocks are concatenated in payload order. The tensor contains
/// only prediction values; `out_metadata_json` carries block offsets, fold ids,
/// sample/unit ids and target names needed to interpret each row. Returned
/// values must be released with `dagml_f64_tensor_free` and
/// `dagml_owned_bytes_free`.
///
/// # Safety
///
/// Same pointer ownership rules as `dagml_graph_validate_json`.
/// `requirement_key` must point to UTF-8 bytes. `out_tensor` and
/// `out_metadata_json` must point to writable output structs.
#[no_mangle]
pub unsafe extern "C" fn dagml_prediction_cache_payload_f64_tensor_json(
    bundle_ptr: *const u8,
    bundle_len: usize,
    payload_ptr: *const u8,
    payload_len: usize,
    requirement_key: DagMlBytesView,
    out_tensor: *mut DagMlF64Tensor,
    out_metadata_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    clear_error(error_out);
    clear_f64_tensor(out_tensor);
    clear_owned_bytes(out_metadata_json);
    if out_tensor.is_null() {
        set_error(error_out, "output F64 tensor pointer is null");
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    if out_metadata_json.is_null() {
        set_error(error_out, "output metadata JSON pointer is null");
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    let bundle =
        match parse_json_ptr::<ExecutionBundle>(bundle_ptr, bundle_len, error_out, "bundle") {
            Ok(bundle) => bundle,
            Err(status) => return status,
        };
    let payload_set = match parse_json_ptr::<BundlePredictionCachePayloadSet>(
        payload_ptr,
        payload_len,
        error_out,
        "prediction cache payload set",
    ) {
        Ok(payload) => payload,
        Err(status) => return status,
    };
    if let Err(error) = payload_set.validate_against_bundle(&bundle) {
        return validation_error(error_out, error);
    }
    let requirement_key = match parse_utf8_view(requirement_key, error_out, "requirement key") {
        Ok(requirement_key) => requirement_key,
        Err(status) => return status,
    };
    if requirement_key.trim().is_empty() {
        set_error(error_out, "requirement key is empty");
        return DagMlStatusCode::VALIDATION_ERROR;
    }
    let payload = match payload_set
        .caches
        .iter()
        .find(|payload| payload.requirement_key == requirement_key)
    {
        Some(payload) => payload,
        None => {
            set_error(
                error_out,
                format!(
                    "prediction cache payload set for bundle `{}` has no requirement `{}`",
                    payload_set.bundle_id, requirement_key
                ),
            );
            return DagMlStatusCode::VALIDATION_ERROR;
        }
    };
    let (values, metadata) = match prediction_cache_payload_to_f64_tensor(payload) {
        Ok(output) => output,
        Err(error) => return validation_error(error_out, error),
    };
    let metadata_json = match serde_json::to_vec(&metadata) {
        Ok(metadata_json) => metadata_json,
        Err(error) => {
            set_error(
                error_out,
                format!("failed to serialize output metadata JSON: {error}"),
            );
            return DagMlStatusCode::VALIDATION_ERROR;
        }
    };
    let status = write_f64_tensor(out_tensor, error_out, values, metadata.rows, metadata.cols);
    if status != DagMlStatusCode::OK {
        return status;
    }
    write_owned_vec(out_metadata_json, metadata_json);
    DagMlStatusCode::OK
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

/// Builds a standards-facing research provenance export from validated DAG-ML
/// contracts.
///
/// The returned JSON is a serialized `ResearchProvenanceExport` containing
/// `lineage.prov.jsonld` and `ro-crate-metadata.json` payloads. `lineage`,
/// `envelopes`, `prediction_cache_manifest` and `artifact_manifest` are
/// optional: pass a null pointer with length 0 to omit them. When present,
/// `lineage` must be a JSON array of `LineageRecord`, `envelopes` a JSON object
/// keyed by bundle data requirement key, and the manifest inputs must match the
/// bundle.
///
/// # Safety
///
/// Non-null input pointers must point to readable bytes for the duration of the
/// call. `out_json` must point to writable memory for one `DagMlOwnedBytes`;
/// returned bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_research_provenance_export_json(
    plan_ptr: *const u8,
    plan_len: usize,
    bundle_ptr: *const u8,
    bundle_len: usize,
    lineage_ptr: *const u8,
    lineage_len: usize,
    envelopes_ptr: *const u8,
    envelopes_len: usize,
    prediction_cache_manifest_ptr: *const u8,
    prediction_cache_manifest_len: usize,
    artifact_manifest_ptr: *const u8,
    artifact_manifest_len: usize,
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
    let lineage = match parse_optional_json_ptr::<Vec<LineageRecord>>(
        lineage_ptr,
        lineage_len,
        error_out,
        "lineage records",
    ) {
        Ok(Some(lineage)) => lineage,
        Ok(None) => Vec::new(),
        Err(status) => return status,
    };
    let envelopes = match parse_optional_json_ptr::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        envelopes_ptr,
        envelopes_len,
        error_out,
        "replay envelopes",
    ) {
        Ok(Some(envelopes)) => envelopes,
        Ok(None) => BTreeMap::new(),
        Err(status) => return status,
    };
    let prediction_cache_manifest = match parse_optional_json_ptr::<FilePredictionCacheManifest>(
        prediction_cache_manifest_ptr,
        prediction_cache_manifest_len,
        error_out,
        "prediction cache manifest",
    ) {
        Ok(manifest) => manifest,
        Err(status) => return status,
    };
    let artifact_manifest = match parse_optional_json_ptr::<FileArtifactManifest>(
        artifact_manifest_ptr,
        artifact_manifest_len,
        error_out,
        "artifact manifest",
    ) {
        Ok(manifest) => manifest,
        Err(status) => return status,
    };

    match build_research_provenance_export(
        &plan,
        &bundle,
        &lineage,
        &envelopes,
        prediction_cache_manifest.as_ref(),
        artifact_manifest.as_ref(),
    ) {
        Ok(export) => write_owned_json(out_json, error_out, &export),
        Err(error) => validation_error(error_out, error),
    }
}

/// Builds an OpenLineage RunEvent from validated DAG-ML provenance contracts.
///
/// This function mirrors `dagml_research_provenance_export_json` inputs, but
/// returns a single OpenLineage-compatible JSON event. The event is a
/// publication view only: DAG-ML fingerprints and OOF evidence remain available
/// through custom `dagml_*` facets.
///
/// # Safety
///
/// Non-null input pointers must point to readable bytes for the duration of the
/// call. `namespace` and `event_time` must be valid UTF-8 byte views.
/// `out_json` must point to writable memory for one `DagMlOwnedBytes`; returned
/// bytes must be released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_openlineage_run_event_json(
    plan_ptr: *const u8,
    plan_len: usize,
    bundle_ptr: *const u8,
    bundle_len: usize,
    lineage_ptr: *const u8,
    lineage_len: usize,
    envelopes_ptr: *const u8,
    envelopes_len: usize,
    prediction_cache_manifest_ptr: *const u8,
    prediction_cache_manifest_len: usize,
    artifact_manifest_ptr: *const u8,
    artifact_manifest_len: usize,
    namespace: DagMlBytesView,
    event_time: DagMlBytesView,
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
    let lineage = match parse_optional_json_ptr::<Vec<LineageRecord>>(
        lineage_ptr,
        lineage_len,
        error_out,
        "lineage records",
    ) {
        Ok(Some(lineage)) => lineage,
        Ok(None) => Vec::new(),
        Err(status) => return status,
    };
    let envelopes = match parse_optional_json_ptr::<BTreeMap<String, ExternalDataPlanEnvelope>>(
        envelopes_ptr,
        envelopes_len,
        error_out,
        "replay envelopes",
    ) {
        Ok(Some(envelopes)) => envelopes,
        Ok(None) => BTreeMap::new(),
        Err(status) => return status,
    };
    let prediction_cache_manifest = match parse_optional_json_ptr::<FilePredictionCacheManifest>(
        prediction_cache_manifest_ptr,
        prediction_cache_manifest_len,
        error_out,
        "prediction cache manifest",
    ) {
        Ok(manifest) => manifest,
        Err(status) => return status,
    };
    let artifact_manifest = match parse_optional_json_ptr::<FileArtifactManifest>(
        artifact_manifest_ptr,
        artifact_manifest_len,
        error_out,
        "artifact manifest",
    ) {
        Ok(manifest) => manifest,
        Err(status) => return status,
    };
    let namespace = match parse_utf8_view(namespace, error_out, "OpenLineage namespace") {
        Ok(namespace) => namespace,
        Err(status) => return status,
    };
    let event_time = match parse_utf8_view(event_time, error_out, "OpenLineage event_time") {
        Ok(event_time) => event_time,
        Err(status) => return status,
    };
    let options = OpenLineageRunEventOptions {
        namespace,
        event_time,
    };

    match build_openlineage_run_event(
        &plan,
        &bundle,
        &lineage,
        &envelopes,
        prediction_cache_manifest.as_ref(),
        artifact_manifest.as_ref(),
        &options,
    ) {
        Ok(event) => write_owned_json(out_json, error_out, &event),
        Err(error) => validation_error(error_out, error),
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

/// Executes replay from JSON contracts through host-provided runtime vtables.
///
/// Rust keeps control over bundle/replay validation, data-view construction,
/// OOF cache preloading, artifact handle materialization, scheduler ordering
/// and `NodeResult` conformance. Host bindings provide the operator,
/// data-provider, artifact-store and optional prediction-cache implementations.
///
/// # Safety
///
/// Input JSON pointers follow the same rules as `dagml_graph_validate_json`.
/// `controller_bindings` must point to `controller_binding_count` readable
/// entries when the count is non-zero. `prediction_cache_store` may be null
/// when the bundle does not require OOF cache replay. Returned bytes must be
/// released with `dagml_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn dagml_replay_execute_json(
    plan_ptr: *const u8,
    plan_len: usize,
    bundle_ptr: *const u8,
    bundle_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    envelopes_ptr: *const u8,
    envelopes_len: usize,
    data_owner_controller_id: DagMlBytesView,
    dataset: DagMlHandle,
    data_provider: DagMlDataVTable,
    artifact_store: DagMlArtifactStoreVTable,
    prediction_cache_store: *const DagMlPredictionCacheVTable,
    controller_bindings: *const DagMlControllerBinding,
    controller_binding_count: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
) -> DagMlStatusCode {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dagml_replay_execute_json_impl(CAbiReplayExecuteArgs {
            plan_ptr,
            plan_len,
            bundle_ptr,
            bundle_len,
            request_ptr,
            request_len,
            envelopes_ptr,
            envelopes_len,
            data_owner_controller_id,
            dataset,
            data_provider,
            artifact_store,
            prediction_cache_store,
            controller_bindings,
            controller_binding_count,
            out_json,
            error_out,
        })
    })) {
        Ok(status) => status,
        Err(_) => {
            clear_error(error_out);
            clear_owned_bytes(out_json);
            set_error(error_out, "panic while executing replay through C ABI");
            DagMlStatusCode::PANIC
        }
    }
}

struct CAbiReplayExecuteArgs {
    plan_ptr: *const u8,
    plan_len: usize,
    bundle_ptr: *const u8,
    bundle_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    envelopes_ptr: *const u8,
    envelopes_len: usize,
    data_owner_controller_id: DagMlBytesView,
    dataset: DagMlHandle,
    data_provider: DagMlDataVTable,
    artifact_store: DagMlArtifactStoreVTable,
    prediction_cache_store: *const DagMlPredictionCacheVTable,
    controller_bindings: *const DagMlControllerBinding,
    controller_binding_count: usize,
    out_json: *mut DagMlOwnedBytes,
    error_out: *mut DagMlString,
}

unsafe fn dagml_replay_execute_json_impl(args: CAbiReplayExecuteArgs) -> DagMlStatusCode {
    let CAbiReplayExecuteArgs {
        plan_ptr,
        plan_len,
        bundle_ptr,
        bundle_len,
        request_ptr,
        request_len,
        envelopes_ptr,
        envelopes_len,
        data_owner_controller_id,
        dataset,
        data_provider,
        artifact_store,
        prediction_cache_store,
        controller_bindings,
        controller_binding_count,
        out_json,
        error_out,
    } = args;
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
    let data_owner = match parse_controller_id_view(
        data_owner_controller_id,
        error_out,
        "data owner controller id",
    ) {
        Ok(controller_id) => controller_id,
        Err(status) => return status,
    };
    let data_provider = match CAbiRuntimeDataProvider::new(data_owner, dataset, data_provider) {
        Ok(provider) => provider,
        Err(error) => return validation_error(error_out, error),
    };
    let artifact_store = match CAbiRuntimeArtifactStore::new(artifact_store) {
        Ok(store) => store,
        Err(error) => return validation_error(error_out, error),
    };
    let prediction_cache = if prediction_cache_store.is_null() {
        None
    } else {
        match CAbiRuntimePredictionCacheStore::new(*prediction_cache_store) {
            Ok(store) => Some(store),
            Err(error) => return validation_error(error_out, error),
        }
    };
    let controllers =
        match build_controller_registry(controller_bindings, controller_binding_count, error_out) {
            Ok(controllers) => controllers,
            Err(status) => return status,
        };
    let prediction_cache_ref = prediction_cache
        .as_ref()
        .map(|store| store as &dyn RuntimePredictionCacheStore);
    match execute_vtable_replay(
        &plan,
        &bundle,
        &request,
        &envelopes,
        VtableReplayRuntime {
            data_provider: &data_provider,
            artifact_store: &artifact_store,
            prediction_cache_store: prediction_cache_ref,
            controllers: &controllers,
        },
    ) {
        Ok(summary) => write_owned_json(out_json, error_out, &summary),
        Err(error) => validation_error(error_out, error),
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

unsafe fn clear_f64_tensor(out_tensor: *mut DagMlF64Tensor) {
    if !out_tensor.is_null() {
        *out_tensor = DagMlF64Tensor::default();
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

unsafe fn parse_optional_json_ptr<T>(
    json_ptr: *const u8,
    json_len: usize,
    error_out: *mut DagMlString,
    label: &str,
) -> Result<Option<T>, DagMlStatusCode>
where
    T: DeserializeOwned,
{
    if json_ptr.is_null() && json_len == 0 {
        return Ok(None);
    }
    parse_json_ptr::<T>(json_ptr, json_len, error_out, label).map(Some)
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

unsafe fn write_owned_vec(out_json: *mut DagMlOwnedBytes, mut data: Vec<u8>) {
    let owned = DagMlOwnedBytes {
        ptr: data.as_mut_ptr(),
        len: data.len(),
        capacity: data.capacity(),
    };
    std::mem::forget(data);
    *out_json = owned;
}

unsafe fn write_f64_tensor(
    out_tensor: *mut DagMlF64Tensor,
    error_out: *mut DagMlString,
    mut values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> DagMlStatusCode {
    if out_tensor.is_null() {
        set_error(error_out, "output F64 tensor pointer is null");
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    let expected_len = match rows.checked_mul(cols) {
        Some(expected_len) => expected_len,
        None => {
            set_error(error_out, "F64 tensor shape overflows usize");
            return DagMlStatusCode::VALIDATION_ERROR;
        }
    };
    if values.len() != expected_len {
        set_error(
            error_out,
            format!(
                "F64 tensor has {} value(s), expected {} for shape {}x{}",
                values.len(),
                expected_len,
                rows,
                cols
            ),
        );
        return DagMlStatusCode::VALIDATION_ERROR;
    }
    let tensor = DagMlF64Tensor {
        ptr: values.as_mut_ptr(),
        len: values.len(),
        capacity: values.capacity(),
        rows,
        cols,
    };
    std::mem::forget(values);
    *out_tensor = tensor;
    DagMlStatusCode::OK
}

fn flatten_f64_rows(
    label: &str,
    producer_node: &str,
    rows: &[Vec<f64>],
    width: usize,
) -> dag_ml_core::Result<Vec<f64>> {
    let expected_len = rows.len().checked_mul(width).ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "{label} for producer `{producer_node}` shape overflows usize"
        ))
    })?;
    let mut values = Vec::with_capacity(expected_len);
    for row in rows {
        if row.len() != width {
            return Err(DagMlError::RuntimeValidation(format!(
                "{label} for producer `{producer_node}` has ragged rows"
            )));
        }
        for value in row {
            if !value.is_finite() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "{label} for producer `{producer_node}` contains non-finite value"
                )));
            }
            values.push(*value);
        }
    }
    Ok(values)
}

#[derive(Serialize)]
struct PredictionCacheTensorMetadata {
    schema_version: u32,
    requirement_key: String,
    cache_id: String,
    prediction_level: PredictionLevel,
    block_count: usize,
    row_count: usize,
    rows: usize,
    cols: usize,
    target_names: Vec<String>,
    blocks: Vec<PredictionCacheTensorBlockMetadata>,
}

#[derive(Serialize)]
struct PredictionCacheTensorBlockMetadata {
    block_index: usize,
    prediction_id: Option<String>,
    fold_id: Option<String>,
    row_offset: usize,
    row_count: usize,
    sample_ids: Vec<SampleId>,
    unit_ids: Vec<PredictionUnitId>,
}

fn prediction_cache_payload_to_f64_tensor(
    payload: &BundlePredictionCachePayload,
) -> dag_ml_core::Result<(Vec<f64>, PredictionCacheTensorMetadata)> {
    payload.validate()?;
    match payload.prediction_level {
        PredictionLevel::Observation => Err(DagMlError::RuntimeValidation(format!(
            "prediction cache payload `{}` cannot export observation-level tensors",
            payload.cache_id
        ))),
        PredictionLevel::Sample => sample_prediction_cache_payload_to_f64_tensor(payload),
        PredictionLevel::Target | PredictionLevel::Group => {
            aggregated_prediction_cache_payload_to_f64_tensor(payload)
        }
    }
}

fn sample_prediction_cache_payload_to_f64_tensor(
    payload: &BundlePredictionCachePayload,
) -> dag_ml_core::Result<(Vec<f64>, PredictionCacheTensorMetadata)> {
    let mut values = Vec::new();
    let mut blocks = Vec::with_capacity(payload.blocks.len());
    let mut width = None;
    let mut target_names = None::<Vec<String>>;
    let mut row_offset = 0usize;
    for (block_index, block) in payload.blocks.iter().enumerate() {
        let block_width = block.validate_shape()?;
        ensure_prediction_tensor_width(
            payload,
            &mut width,
            &mut target_names,
            block_width,
            &block.target_names,
        )?;
        let flattened = flatten_f64_rows(
            "prediction cache sample block",
            block.producer_node.as_str(),
            &block.values,
            block_width,
        )?;
        values.extend(flattened);
        let row_count = block.sample_ids.len();
        blocks.push(PredictionCacheTensorBlockMetadata {
            block_index,
            prediction_id: block.prediction_id.clone(),
            fold_id: block.fold_id.as_ref().map(ToString::to_string),
            row_offset,
            row_count,
            sample_ids: block.sample_ids.clone(),
            unit_ids: Vec::new(),
        });
        row_offset += row_count;
    }
    let cols = width.unwrap_or(0);
    build_prediction_cache_tensor_metadata(payload, row_offset, cols, target_names, blocks)
        .map(|metadata| (values, metadata))
}

fn aggregated_prediction_cache_payload_to_f64_tensor(
    payload: &BundlePredictionCachePayload,
) -> dag_ml_core::Result<(Vec<f64>, PredictionCacheTensorMetadata)> {
    let mut values = Vec::new();
    let mut blocks = Vec::with_capacity(payload.aggregated_blocks.len());
    let mut width = None;
    let mut target_names = None::<Vec<String>>;
    let mut row_offset = 0usize;
    for (block_index, block) in payload.aggregated_blocks.iter().enumerate() {
        let block_width = block.validate_shape()?;
        ensure_prediction_tensor_width(
            payload,
            &mut width,
            &mut target_names,
            block_width,
            &block.target_names,
        )?;
        let flattened = flatten_f64_rows(
            "prediction cache aggregated block",
            block.producer_node.as_str(),
            &block.values,
            block_width,
        )?;
        values.extend(flattened);
        let row_count = block.unit_ids.len();
        blocks.push(PredictionCacheTensorBlockMetadata {
            block_index,
            prediction_id: block.prediction_id.clone(),
            fold_id: block.fold_id.as_ref().map(ToString::to_string),
            row_offset,
            row_count,
            sample_ids: Vec::new(),
            unit_ids: block.unit_ids.clone(),
        });
        row_offset += row_count;
    }
    let cols = width.unwrap_or(0);
    build_prediction_cache_tensor_metadata(payload, row_offset, cols, target_names, blocks)
        .map(|metadata| (values, metadata))
}

fn ensure_prediction_tensor_width(
    payload: &BundlePredictionCachePayload,
    width: &mut Option<usize>,
    target_names: &mut Option<Vec<String>>,
    block_width: usize,
    block_target_names: &[String],
) -> dag_ml_core::Result<()> {
    match width {
        Some(width) if *width != block_width => {
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache payload `{}` has mixed prediction widths",
                payload.cache_id
            )));
        }
        Some(_) => {}
        None => {
            *width = Some(block_width);
            *target_names = Some(block_target_names.to_vec());
            return Ok(());
        }
    }
    if target_names.as_deref().unwrap_or(&[]) != block_target_names {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache payload `{}` has mixed target names",
            payload.cache_id
        )));
    }
    Ok(())
}

fn build_prediction_cache_tensor_metadata(
    payload: &BundlePredictionCachePayload,
    rows: usize,
    cols: usize,
    target_names: Option<Vec<String>>,
    blocks: Vec<PredictionCacheTensorBlockMetadata>,
) -> dag_ml_core::Result<PredictionCacheTensorMetadata> {
    if rows != payload.row_count {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache payload `{}` tensor row count {} does not match payload row count {}",
            payload.cache_id, rows, payload.row_count
        )));
    }
    if cols == 0 {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache payload `{}` tensor has zero width",
            payload.cache_id
        )));
    }
    Ok(PredictionCacheTensorMetadata {
        schema_version: DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION,
        requirement_key: payload.requirement_key.clone(),
        cache_id: payload.cache_id.clone(),
        prediction_level: payload.prediction_level,
        block_count: payload.block_count,
        row_count: payload.row_count,
        rows,
        cols,
        target_names: target_names.unwrap_or_default(),
        blocks,
    })
}

unsafe fn validation_error(error_out: *mut DagMlString, error: DagMlError) -> DagMlStatusCode {
    set_error(error_out, error.to_string());
    DagMlStatusCode::VALIDATION_ERROR
}

unsafe fn parse_controller_id_view(
    view: DagMlBytesView,
    error_out: *mut DagMlString,
    label: &str,
) -> Result<ControllerId, DagMlStatusCode> {
    let raw = parse_utf8_view(view, error_out, label)?;
    ControllerId::new(raw).map_err(|error| {
        set_error(error_out, error.to_string());
        DagMlStatusCode::VALIDATION_ERROR
    })
}

unsafe fn parse_utf8_view(
    view: DagMlBytesView,
    error_out: *mut DagMlString,
    label: &str,
) -> Result<String, DagMlStatusCode> {
    if view.ptr.is_null() {
        set_error(error_out, format!("{label} pointer is null"));
        return Err(DagMlStatusCode::INVALID_ARGUMENT);
    }
    let bytes = slice::from_raw_parts(view.ptr, view.len);
    let raw = std::str::from_utf8(bytes).map_err(|error| {
        set_error(error_out, format!("{label} is not valid UTF-8: {error}"));
        DagMlStatusCode::VALIDATION_ERROR
    })?;
    Ok(raw.to_string())
}

unsafe fn build_controller_registry(
    controller_bindings: *const DagMlControllerBinding,
    controller_binding_count: usize,
    error_out: *mut DagMlString,
) -> Result<RuntimeControllerRegistry, DagMlStatusCode> {
    if controller_binding_count > 0 && controller_bindings.is_null() {
        set_error(error_out, "controller bindings pointer is null");
        return Err(DagMlStatusCode::INVALID_ARGUMENT);
    }
    let mut registry = RuntimeControllerRegistry::new();
    let bindings = if controller_binding_count == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(controller_bindings, controller_binding_count)
    };
    for binding in bindings {
        let controller_id =
            parse_controller_id_view(binding.controller_id, error_out, "controller id")?;
        let controller =
            CAbiRuntimeController::new(controller_id, binding.vtable).map_err(|error| {
                set_error(error_out, error.to_string());
                DagMlStatusCode::VALIDATION_ERROR
            })?;
        registry.register(Box::new(controller)).map_err(|error| {
            set_error(error_out, error.to_string());
            DagMlStatusCode::VALIDATION_ERROR
        })?;
    }
    Ok(registry)
}

pub struct CAbiRuntimeDataProvider {
    vtable: DagMlDataVTable,
    dataset: DagMlHandle,
    owner_controller: ControllerId,
    live_handles: Mutex<Vec<DagMlHandle>>,
}

impl CAbiRuntimeDataProvider {
    pub fn new(
        owner_controller: ControllerId,
        dataset: DagMlHandle,
        vtable: DagMlDataVTable,
    ) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION {
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
            live_handles: Mutex::new(Vec::new()),
        })
    }
}

impl Drop for CAbiRuntimeDataProvider {
    fn drop(&mut self) {
        if let Some(release) = self.vtable.release {
            let handles = self
                .live_handles
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for handle in handles.iter().rev() {
                unsafe { release(self.vtable.user_data, *handle) };
            }
            handles.clear();
        }
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
        self.live_handles
            .lock()
            .map_err(|_| {
                DagMlError::RuntimeValidation(
                    "data provider handle registry is poisoned".to_string(),
                )
            })?
            .push(out_handle);
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
        self.live_handles
            .lock()
            .map_err(|_| {
                DagMlError::RuntimeValidation(
                    "data provider handle registry is poisoned".to_string(),
                )
            })?
            .push(out_view);
        Ok(HandleRef {
            handle: out_view,
            kind: HandleKind::DataView,
            owner_controller: self.owner_controller.clone(),
        })
    }
}

pub struct CAbiRuntimeController {
    id: ControllerId,
    vtable: DagMlControllerVTable,
    live_handles: Mutex<Vec<DagMlHandle>>,
}

// The C ABI controller is an opaque host callback table. dag-ml only forwards
// immutable task JSON and release calls through the supplied function pointers;
// host implementations are responsible for making `user_data` safe for any
// scheduler mode they opt into.
unsafe impl Send for CAbiRuntimeController {}
unsafe impl Sync for CAbiRuntimeController {}

impl CAbiRuntimeController {
    pub fn new(id: ControllerId, vtable: DagMlControllerVTable) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION {
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
        Ok(Self {
            id,
            vtable,
            live_handles: Mutex::new(Vec::new()),
        })
    }

    fn track_result_handles(&self, result: &NodeResult) -> dag_ml_core::Result<()> {
        if self.vtable.release.is_none() {
            return Ok(());
        }
        let mut handles = self.live_handles.lock().map_err(|_| {
            DagMlError::RuntimeValidation("controller handle registry is poisoned".to_string())
        })?;
        for handle in result
            .outputs
            .values()
            .chain(result.artifact_handles.values())
        {
            if handle.owner_controller == self.id
                && handle.handle != 0
                && !handles.contains(&handle.handle)
            {
                handles.push(handle.handle);
            }
        }
        Ok(())
    }

    fn release_result_handles_immediately(&self, result: &NodeResult) {
        if let Some(release) = self.vtable.release {
            let mut released = BTreeSet::new();
            for handle in result
                .outputs
                .values()
                .chain(result.artifact_handles.values())
            {
                if handle.owner_controller == self.id
                    && handle.handle != 0
                    && released.insert(handle.handle)
                {
                    unsafe { release(self.vtable.user_data, handle.handle) };
                }
            }
        }
    }
}

impl Drop for CAbiRuntimeController {
    fn drop(&mut self) {
        if let Some(release) = self.vtable.release {
            let handles = self
                .live_handles
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for handle in handles.iter().rev() {
                unsafe { release(self.vtable.user_data, *handle) };
            }
            handles.clear();
        }
        if self.vtable.abi_version >= DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION {
            if let Some(destroy) = self.vtable.destroy {
                unsafe { destroy(self.vtable.user_data) };
            }
        }
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
        let result = serde_json::from_slice::<NodeResult>(&data).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "controller invoke returned invalid node result JSON: {error}"
            ))
        })?;
        if let Err(error) = result.validate_for_task(task) {
            self.release_result_handles_immediately(&result);
            return Err(error);
        }
        self.track_result_handles(&result)?;
        Ok(result)
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

pub struct CAbiRuntimeArtifactStore {
    vtable: DagMlArtifactStoreVTable,
    live_handles: Mutex<Vec<DagMlHandle>>,
}

impl CAbiRuntimeArtifactStore {
    pub fn new(vtable: DagMlArtifactStoreVTable) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < DAG_ML_ARTIFACT_STORE_VTABLE_BORROWED_ABI_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact store ABI version {} is unsupported",
                vtable.abi_version
            )));
        }
        if vtable.materialize.is_none() {
            return Err(DagMlError::RuntimeValidation(
                "artifact store vtable is missing materialize".to_string(),
            ));
        }
        Ok(Self {
            vtable,
            live_handles: Mutex::new(Vec::new()),
        })
    }
}

impl Drop for CAbiRuntimeArtifactStore {
    fn drop(&mut self) {
        if let Some(release) = self.vtable.release {
            let handles = self
                .live_handles
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for handle in handles.iter().rev() {
                unsafe { release(self.vtable.user_data, *handle) };
            }
            handles.clear();
        }
        if self.vtable.abi_version >= DAG_ML_ARTIFACT_STORE_VTABLE_OWNED_ABI_VERSION {
            if let Some(destroy) = self.vtable.destroy {
                unsafe { destroy(self.vtable.user_data) };
            }
        }
    }
}

impl RuntimeArtifactStore for CAbiRuntimeArtifactStore {
    fn materialize(
        &self,
        request: &ArtifactMaterializationRequest,
    ) -> dag_ml_core::Result<HandleRef> {
        let materialize = self.vtable.materialize.ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "artifact store vtable is missing materialize".to_string(),
            )
        })?;
        let request_json = serde_json::to_vec(request).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "failed to serialize artifact materialization request: {error}"
            ))
        })?;
        let mut out_handle = DagMlHandleRef::default();
        let status = unsafe {
            materialize(
                self.vtable.user_data,
                bytes_view(&request_json),
                &mut out_handle,
            )
        };
        artifact_store_status(status, "materialize")?;
        if out_handle.handle == 0 {
            return Err(DagMlError::RuntimeValidation(
                "artifact store materialize returned empty handle".to_string(),
            ));
        }
        let kind = match handle_kind_from_abi(out_handle.kind) {
            Ok(kind) => kind,
            Err(error) => {
                if let Some(release) = self.vtable.release {
                    unsafe { release(self.vtable.user_data, out_handle.handle) };
                }
                return Err(error);
            }
        };
        self.live_handles
            .lock()
            .map_err(|_| {
                DagMlError::RuntimeValidation(
                    "artifact store handle registry is poisoned".to_string(),
                )
            })?
            .push(out_handle.handle);
        Ok(HandleRef {
            handle: out_handle.handle,
            kind,
            owner_controller: request.controller_id.clone(),
        })
    }
}

fn handle_kind_from_abi(kind: u32) -> dag_ml_core::Result<HandleKind> {
    match kind {
        DAG_ML_HANDLE_KIND_DATA => Ok(HandleKind::Data),
        DAG_ML_HANDLE_KIND_DATA_VIEW => Ok(HandleKind::DataView),
        DAG_ML_HANDLE_KIND_MODEL => Ok(HandleKind::Model),
        DAG_ML_HANDLE_KIND_ARTIFACT => Ok(HandleKind::Artifact),
        DAG_ML_HANDLE_KIND_PREDICTION => Ok(HandleKind::Prediction),
        DAG_ML_HANDLE_KIND_RELATION => Ok(HandleKind::Relation),
        _ => Err(DagMlError::RuntimeValidation(format!(
            "unknown ABI handle kind {kind}"
        ))),
    }
}

fn artifact_store_status(status: DagMlStatusCode, action: &str) -> dag_ml_core::Result<()> {
    if status == DagMlStatusCode::OK {
        Ok(())
    } else if status == DagMlStatusCode::INVALID_ARGUMENT {
        Err(DagMlError::RuntimeValidation(format!(
            "artifact store {action} rejected invalid arguments"
        )))
    } else if status == DagMlStatusCode::VALIDATION_ERROR {
        Err(DagMlError::RuntimeValidation(format!(
            "artifact store {action} rejected request"
        )))
    } else if status == DagMlStatusCode::PANIC {
        Err(DagMlError::RuntimeValidation(format!(
            "artifact store {action} reported panic"
        )))
    } else {
        Err(DagMlError::RuntimeValidation(format!(
            "artifact store {action} returned unknown status code {}",
            status.0
        )))
    }
}

pub struct CAbiRuntimePredictionCacheStore {
    vtable: DagMlPredictionCacheVTable,
    live_handles: Mutex<Vec<DagMlHandle>>,
}

impl CAbiRuntimePredictionCacheStore {
    pub fn new(vtable: DagMlPredictionCacheVTable) -> dag_ml_core::Result<Self> {
        if vtable.abi_version < DAG_ML_PREDICTION_CACHE_VTABLE_BORROWED_ABI_VERSION {
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
        Ok(Self {
            vtable,
            live_handles: Mutex::new(Vec::new()),
        })
    }

    fn load_prediction_json(&self, requirement_key: &str) -> dag_ml_core::Result<Vec<u8>> {
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
        Ok(data)
    }
}

impl RuntimePredictionCacheStore for CAbiRuntimePredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> dag_ml_core::Result<Vec<PredictionBlock>> {
        let data = self.load_prediction_json(requirement_key)?;
        serde_json::from_slice::<Vec<PredictionBlock>>(&data).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache load_blocks returned invalid prediction block JSON: {error}"
            ))
        })
    }

    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> dag_ml_core::Result<Vec<AggregatedPredictionBlock>> {
        let data = self.load_prediction_json(requirement_key)?;
        serde_json::from_slice::<Vec<AggregatedPredictionBlock>>(&data).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache load_blocks returned invalid aggregated prediction block JSON: {error}"
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
        self.live_handles
            .lock()
            .map_err(|_| {
                DagMlError::RuntimeValidation(
                    "prediction cache handle registry is poisoned".to_string(),
                )
            })?
            .push(out_handle);
        Ok(HandleRef {
            handle: out_handle,
            kind: HandleKind::Prediction,
            owner_controller: request.producer_controller_id.clone(),
        })
    }
}

impl Drop for CAbiRuntimePredictionCacheStore {
    fn drop(&mut self) {
        if let Some(release) = self.vtable.release {
            let handles = self
                .live_handles
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for handle in handles.iter().rev() {
                unsafe { release(self.vtable.user_data, *handle) };
            }
            handles.clear();
        }
        if self.vtable.abi_version >= DAG_ML_PREDICTION_CACHE_VTABLE_OWNED_ABI_VERSION {
            if let Some(destroy) = self.vtable.destroy {
                unsafe { destroy(self.vtable.user_data) };
            }
        }
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

unsafe fn parse_phase_view(
    view: DagMlBytesView,
    error_out: *mut DagMlString,
    label: &str,
) -> Result<Phase, DagMlStatusCode> {
    let raw = parse_utf8_view(view, error_out, label)?;
    match raw.as_str() {
        "COMPILE" => Ok(Phase::Compile),
        "PLAN" => Ok(Phase::Plan),
        "FIT_CV" => Ok(Phase::FitCv),
        "SELECT" => Ok(Phase::Select),
        "REFIT" => Ok(Phase::Refit),
        "PREDICT" => Ok(Phase::Predict),
        "EXPLAIN" => Ok(Phase::Explain),
        _ => {
            set_error(error_out, format!("unsupported {label} `{raw}`"));
            Err(DagMlStatusCode::VALIDATION_ERROR)
        }
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

#[derive(Debug, Serialize)]
struct ReplayExecutionSummary {
    bundle_id: String,
    phase: Phase,
    result_count: usize,
    lineage_record_count: usize,
    prediction_block_count: usize,
    controller_count: usize,
    prediction_cache_store: bool,
}

struct VtableReplayRuntime<'a> {
    data_provider: &'a dyn RuntimeDataProvider,
    artifact_store: &'a dyn RuntimeArtifactStore,
    prediction_cache_store: Option<&'a dyn RuntimePredictionCacheStore>,
    controllers: &'a RuntimeControllerRegistry,
}

fn execute_vtable_replay(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    request: &ReplayPhaseRequest,
    envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    runtime: VtableReplayRuntime<'_>,
) -> dag_ml_core::Result<ReplayExecutionSummary> {
    let prediction_cache_store = runtime.prediction_cache_store;
    let mut ctx = RunContext::new(RunId::new("run:capi.replay")?, None);
    let results = SequentialScheduler.execute_bundle_replay(
        BundleReplayExecution {
            plan,
            bundle,
            replay_request: request,
            prediction_cache_store,
            controllers: runtime.controllers,
            data_provider: runtime.data_provider,
            artifact_store: runtime.artifact_store,
            data_envelopes: envelopes,
        },
        &mut ctx,
    )?;
    Ok(ReplayExecutionSummary {
        bundle_id: bundle.bundle_id.to_string(),
        phase: request.phase,
        result_count: results.len(),
        lineage_record_count: ctx.lineage.len(),
        prediction_block_count: ctx.prediction_store.blocks().len(),
        controller_count: plan.controller_manifests.len(),
        prediction_cache_store: prediction_cache_store.is_some(),
    })
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
        build_aggregated_prediction_cache_record, build_prediction_cache_record, ArtifactId,
        ArtifactPolicy, ArtifactRef, BundleId, BundlePredictionRequirement, ControllerCapability,
        ControllerFitScope, DataBinding, DataProviderViewSpec, DataRequestPartition,
        DataViewPolicy, FoldId, NodeId, NodeKind, NodePlan, PredictionLevel, PredictionUnitId,
        RngPolicy, TargetId, VariantId,
    };
    use std::ffi::CStr;

    #[derive(Default)]
    struct DataProviderStub {
        materialize_dataset: DagMlHandle,
        materialize_json: Vec<u8>,
        make_view_parent: DagMlHandle,
        make_view_json: Vec<u8>,
        release_handles: Vec<DagMlHandle>,
    }

    #[derive(Default)]
    struct ControllerStub {
        task_json: Vec<u8>,
        task_node_ids: Vec<String>,
        result_json: Vec<u8>,
        release_count: usize,
        release_handles: Vec<DagMlHandle>,
        invocation_count: usize,
        destroy_count: usize,
    }

    #[derive(Default)]
    struct ArtifactStoreStub {
        materialize_json: Vec<u8>,
        handle: DagMlHandleRef,
        status: DagMlStatusCode,
        release_handles: Vec<DagMlHandle>,
        destroy_count: usize,
    }

    #[derive(Default)]
    struct PredictionCacheStub {
        load_key: Vec<u8>,
        load_keys: Vec<String>,
        blocks_json: Vec<u8>,
        blocks_by_key: BTreeMap<String, Vec<u8>>,
        release_count: usize,
        release_handles: Vec<DagMlHandle>,
        materialize_json: Vec<u8>,
        materialize_count: usize,
        destroy_count: usize,
    }

    fn error_message(error: &DagMlString) -> String {
        if error.ptr.is_null() {
            return "<no error>".to_string();
        }
        unsafe { CStr::from_ptr(error.ptr) }
            .to_string_lossy()
            .into_owned()
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

    unsafe extern "C" fn data_release_stub(user_data: *mut c_void, handle: DagMlHandle) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<DataProviderStub>());
        state.release_handles.push(handle);
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
        state.invocation_count += 1;
        let mut data = state.result_json.clone();
        *out_result_json = DagMlOwnedBytes {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            capacity: data.capacity(),
        };
        std::mem::forget(data);
        DagMlStatusCode::OK
    }

    unsafe extern "C" fn replay_controller_invoke_stub(
        user_data: *mut c_void,
        task_json: DagMlBytesView,
        out_result_json: *mut DagMlOwnedBytes,
    ) -> DagMlStatusCode {
        if user_data.is_null() || task_json.ptr.is_null() || out_result_json.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.task_json = slice::from_raw_parts(task_json.ptr, task_json.len).to_vec();
        let task = match serde_json::from_slice::<NodeTask>(&state.task_json) {
            Ok(task) => task,
            Err(_) => return DagMlStatusCode::VALIDATION_ERROR,
        };
        state.task_node_ids.push(task.node_plan.node_id.to_string());
        state.invocation_count += 1;
        let sample_ids = task
            .data_views
            .values()
            .find_map(|view| view.sample_ids.clone())
            .unwrap_or_else(|| vec![SampleId::new("sample:cabi.replay").unwrap()]);
        let predictions = if matches!(task.node_plan.kind, NodeKind::Model) {
            vec![PredictionBlock {
                prediction_id: Some(format!("prediction:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Final,
                fold_id: None,
                sample_ids: sample_ids.clone(),
                values: vec![vec![0.7]; sample_ids.len()],
                target_names: vec!["y".to_string()],
            }]
        } else {
            Vec::new()
        };
        let result = NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "out".to_string(),
                HandleRef {
                    handle: stable_handle(task.node_plan.node_id.as_str()),
                    kind: HandleKind::Data,
                    owner_controller: task.node_plan.controller_id.clone(),
                },
            )]),
            predictions,
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:cabi.replay:{}",
                    task.node_plan.node_id
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
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        };
        if result.validate_for_task(&task).is_err() {
            return DagMlStatusCode::VALIDATION_ERROR;
        }
        let mut data = match serde_json::to_vec(&result) {
            Ok(data) => data,
            Err(_) => return DagMlStatusCode::VALIDATION_ERROR,
        };
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

    unsafe extern "C" fn controller_release_stub(user_data: *mut c_void, handle: DagMlHandle) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.release_handles.push(handle);
    }

    unsafe extern "C" fn controller_destroy_stub(user_data: *mut c_void) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<ControllerStub>());
        state.destroy_count += 1;
    }

    unsafe extern "C" fn artifact_store_materialize_stub(
        user_data: *mut c_void,
        request_json: DagMlBytesView,
        out_handle: *mut DagMlHandleRef,
    ) -> DagMlStatusCode {
        if user_data.is_null() || request_json.ptr.is_null() || out_handle.is_null() {
            return DagMlStatusCode::INVALID_ARGUMENT;
        }
        let state = &mut *(user_data.cast::<ArtifactStoreStub>());
        state.materialize_json = slice::from_raw_parts(request_json.ptr, request_json.len).to_vec();
        *out_handle = state.handle;
        state.status
    }

    unsafe extern "C" fn artifact_store_release_stub(user_data: *mut c_void, handle: DagMlHandle) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<ArtifactStoreStub>());
        state.release_handles.push(handle);
    }

    unsafe extern "C" fn artifact_store_destroy_stub(user_data: *mut c_void) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<ArtifactStoreStub>());
        state.destroy_count += 1;
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
        let key = String::from_utf8_lossy(&state.load_key).to_string();
        state.load_keys.push(key.clone());
        let mut data = state
            .blocks_by_key
            .get(&key)
            .cloned()
            .unwrap_or_else(|| state.blocks_json.clone());
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
        state.materialize_count += 1;
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

    unsafe extern "C" fn prediction_cache_release_stub(
        user_data: *mut c_void,
        handle: DagMlHandle,
    ) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.release_handles.push(handle);
    }

    unsafe extern "C" fn prediction_cache_destroy_stub(user_data: *mut c_void) {
        if user_data.is_null() {
            return;
        }
        let state = &mut *(user_data.cast::<PredictionCacheStub>());
        state.destroy_count += 1;
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
                controller_capabilities: BTreeSet::from([
                    ControllerCapability::Deterministic,
                    ControllerCapability::ThreadSafe,
                ]),
                fit_scope: ControllerFitScope::FoldTrain,
                rng_policy: RngPolicy::UsesCoreSeed,
                artifact_policy: ArtifactPolicy::Serializable,
                input_nodes: Vec::new(),
                output_nodes: Vec::new(),
                shape_plan: None,
                data_bindings: Vec::new(),
                params: BTreeMap::new(),
                params_fingerprint: "params:controller-fixture".to_string(),
            },
            phase: Phase::FitCv,
            variant_id: Some(VariantId::new("variant:controller").unwrap()),
            variant: None,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            branch_path: Vec::new(),
            input_handles: BTreeMap::new(),
            data_views: BTreeMap::new(),
            prediction_inputs: BTreeMap::new(),
            artifact_inputs: BTreeMap::new(),
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

    fn artifact_materialization_request_fixture() -> ArtifactMaterializationRequest {
        ArtifactMaterializationRequest {
            run_id: RunId::new("run:cabi.artifact").unwrap(),
            bundle_id: BundleId::new("bundle:cabi.artifact").unwrap(),
            node_id: NodeId::new("model:base").unwrap(),
            phase: Phase::Predict,
            variant_id: Some(VariantId::new("variant:selected").unwrap()),
            controller_id: ControllerId::new("controller:model").unwrap(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:model.base").unwrap(),
                kind: "sklearn_pickle".to_string(),
                controller_id: ControllerId::new("controller:model").unwrap(),
                backend: None,
                uri: None,
                content_fingerprint: None,
                size_bytes: Some(1024),
                plugin: None,
                plugin_version: None,
            },
            params_fingerprint: "params:artifact-fixture".to_string(),
        }
    }

    #[test]
    fn data_vtable_exposes_feature_arrow_slot() {
        let table = DagMlDataVTable {
            abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
            user_data: std::ptr::null_mut(),
            materialize: Some(materialize_stub),
            make_view: None,
            view_identity: None,
            target_arrow: None,
            feature_arrow: Some(feature_arrow_stub),
            release: None,
            destroy: None,
        };

        assert_eq!(table.abi_version, DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION);
        assert!(table.materialize.is_some());
        assert!(table.feature_arrow.is_some());
    }

    #[test]
    fn c_abi_runtime_data_provider_routes_materialize_and_view_requests() {
        let mut state = DataProviderStub::default();
        let table = DagMlDataVTable {
            abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
            user_data: (&mut state as *mut DataProviderStub).cast::<c_void>(),
            materialize: Some(materialize_stub),
            make_view: Some(make_view_stub),
            view_identity: None,
            target_arrow: None,
            feature_arrow: None,
            release: Some(data_release_stub),
            destroy: None,
        };
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
        {
            let provider = CAbiRuntimeDataProvider::new(
                ControllerId::new("controller:data.provider").unwrap(),
                7,
                table,
            )
            .unwrap();
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
            let view_json: serde_json::Value =
                serde_json::from_slice(&state.make_view_json).unwrap();
            assert_eq!(view_json["partition"], "fold_train");
            assert_eq!(view_json["fold_id"], "fold:0");
            assert_eq!(view_json["sample_ids"][0], "s1");
            assert_eq!(view_json["columns"][0], "abs_1000");
        }
        assert_eq!(state.release_handles, vec![42, 41]);
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
    fn c_abi_runtime_controller_releases_result_handles_on_drop() {
        let (controller_id, task, expected) = controller_task_result_fixture();
        let mut state = ControllerStub {
            result_json: serde_json::to_vec(&expected).unwrap(),
            ..Default::default()
        };
        {
            let table = DagMlControllerVTable {
                abi_version: 2,
                user_data: (&mut state as *mut ControllerStub).cast::<c_void>(),
                clone_with: None,
                describe: None,
                fit: None,
                predict: None,
                invoke: Some(controller_invoke_stub),
                release_bytes: Some(controller_release_bytes_stub),
                release: Some(controller_release_stub),
                destroy: Some(controller_destroy_stub),
            };
            let controller = CAbiRuntimeController::new(controller_id, table).unwrap();
            let actual = controller.invoke(&task).unwrap();
            assert_eq!(actual.outputs["out"].handle, 88);
            assert!(state.release_handles.is_empty());
        }
        assert_eq!(state.release_handles, vec![88]);
        assert_eq!(state.destroy_count, 0);
    }

    #[test]
    fn c_abi_runtime_controller_owned_vtable_destroys_user_data_after_handles() {
        let (controller_id, task, expected) = controller_task_result_fixture();
        let mut state = ControllerStub {
            result_json: serde_json::to_vec(&expected).unwrap(),
            ..Default::default()
        };
        {
            let table = DagMlControllerVTable {
                abi_version: DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION,
                user_data: (&mut state as *mut ControllerStub).cast::<c_void>(),
                clone_with: None,
                describe: None,
                fit: None,
                predict: None,
                invoke: Some(controller_invoke_stub),
                release_bytes: Some(controller_release_bytes_stub),
                release: Some(controller_release_stub),
                destroy: Some(controller_destroy_stub),
            };
            let controller = CAbiRuntimeController::new(controller_id, table).unwrap();
            let actual = controller.invoke(&task).unwrap();
            assert_eq!(actual.outputs["out"].handle, 88);
            assert!(state.release_handles.is_empty());
            assert_eq!(state.destroy_count, 0);
        }
        assert_eq!(state.release_handles, vec![88]);
        assert_eq!(state.destroy_count, 1);
    }

    #[test]
    fn c_abi_runtime_controller_releases_invalid_result_handles_immediately() {
        let (controller_id, task, mut result) = controller_task_result_fixture();
        result.lineage.seed = Some(999);
        let mut state = ControllerStub {
            result_json: serde_json::to_vec(&result).unwrap(),
            ..Default::default()
        };
        {
            let table = DagMlControllerVTable {
                abi_version: 2,
                user_data: (&mut state as *mut ControllerStub).cast::<c_void>(),
                clone_with: None,
                describe: None,
                fit: None,
                predict: None,
                invoke: Some(controller_invoke_stub),
                release_bytes: Some(controller_release_bytes_stub),
                release: Some(controller_release_stub),
                destroy: None,
            };
            let controller = CAbiRuntimeController::new(controller_id, table).unwrap();
            let error = controller.invoke(&task).unwrap_err();
            assert!(format!("{error}").contains("has seed"));
            assert_eq!(state.release_handles, vec![88]);
        }
        assert_eq!(state.release_handles, vec![88]);
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
    fn c_abi_artifact_store_materializes_typed_handles() {
        let request = artifact_materialization_request_fixture();
        let mut state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 99,
                kind: DAG_ML_HANDLE_KIND_MODEL,
            },
            ..Default::default()
        };
        let table = DagMlArtifactStoreVTable {
            abi_version: 1,
            user_data: (&mut state as *mut ArtifactStoreStub).cast::<c_void>(),
            materialize: Some(artifact_store_materialize_stub),
            release: None,
            destroy: None,
        };
        let store = CAbiRuntimeArtifactStore::new(table).unwrap();
        let handle = store.materialize(&request).unwrap();
        assert_eq!(handle.handle, 99);
        assert_eq!(handle.kind, HandleKind::Model);
        assert_eq!(handle.owner_controller, request.controller_id);

        let request_json: serde_json::Value =
            serde_json::from_slice(&state.materialize_json).unwrap();
        assert_eq!(request_json["artifact"]["id"], "artifact:model.base");
        assert_eq!(request_json["controller_id"], "controller:model");
        assert_eq!(request_json["phase"], "PREDICT");
    }

    #[test]
    fn c_abi_artifact_store_rejects_unknown_handle_kind() {
        let request = artifact_materialization_request_fixture();
        let mut state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 99,
                kind: 999,
            },
            ..Default::default()
        };
        let table = DagMlArtifactStoreVTable {
            abi_version: 1,
            user_data: (&mut state as *mut ArtifactStoreStub).cast::<c_void>(),
            materialize: Some(artifact_store_materialize_stub),
            release: None,
            destroy: None,
        };
        let store = CAbiRuntimeArtifactStore::new(table).unwrap();
        let error = store.materialize(&request).unwrap_err();
        assert!(format!("{error}").contains("unknown ABI handle kind 999"));
    }

    #[test]
    fn c_abi_artifact_store_releases_materialized_handles_on_drop() {
        let request = artifact_materialization_request_fixture();
        let mut state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 99,
                kind: DAG_ML_HANDLE_KIND_MODEL,
            },
            ..Default::default()
        };
        {
            let table = DagMlArtifactStoreVTable {
                abi_version: 1,
                user_data: (&mut state as *mut ArtifactStoreStub).cast::<c_void>(),
                materialize: Some(artifact_store_materialize_stub),
                release: Some(artifact_store_release_stub),
                destroy: Some(artifact_store_destroy_stub),
            };
            let store = CAbiRuntimeArtifactStore::new(table).unwrap();
            let handle = store.materialize(&request).unwrap();
            assert_eq!(handle.handle, 99);
            assert!(state.release_handles.is_empty());
        }
        assert_eq!(state.release_handles, vec![99]);
        assert_eq!(state.destroy_count, 0);
    }

    #[test]
    fn c_abi_artifact_store_owned_vtable_destroys_user_data_after_handles() {
        let request = artifact_materialization_request_fixture();
        let mut state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 99,
                kind: DAG_ML_HANDLE_KIND_MODEL,
            },
            ..Default::default()
        };
        {
            let table = DagMlArtifactStoreVTable {
                abi_version: DAG_ML_ARTIFACT_STORE_VTABLE_OWNED_ABI_VERSION,
                user_data: (&mut state as *mut ArtifactStoreStub).cast::<c_void>(),
                materialize: Some(artifact_store_materialize_stub),
                release: Some(artifact_store_release_stub),
                destroy: Some(artifact_store_destroy_stub),
            };
            let store = CAbiRuntimeArtifactStore::new(table).unwrap();
            let handle = store.materialize(&request).unwrap();
            assert_eq!(handle.handle, 99);
            assert!(state.release_handles.is_empty());
            assert_eq!(state.destroy_count, 0);
        }
        assert_eq!(state.release_handles, vec![99]);
        assert_eq!(state.destroy_count, 1);
    }

    #[test]
    fn c_abi_artifact_store_releases_unknown_kind_handles_immediately() {
        let request = artifact_materialization_request_fixture();
        let mut state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 99,
                kind: 999,
            },
            ..Default::default()
        };
        {
            let table = DagMlArtifactStoreVTable {
                abi_version: 1,
                user_data: (&mut state as *mut ArtifactStoreStub).cast::<c_void>(),
                materialize: Some(artifact_store_materialize_stub),
                release: Some(artifact_store_release_stub),
                destroy: None,
            };
            let store = CAbiRuntimeArtifactStore::new(table).unwrap();
            let error = store.materialize(&request).unwrap_err();
            assert!(format!("{error}").contains("unknown ABI handle kind 999"));
        }
        assert_eq!(state.release_handles, vec![99]);
    }

    #[test]
    fn c_abi_artifact_store_rejects_unknown_status_codes() {
        let request = artifact_materialization_request_fixture();
        let mut state = ArtifactStoreStub {
            status: DagMlStatusCode(997),
            ..Default::default()
        };
        let table = DagMlArtifactStoreVTable {
            abi_version: 1,
            user_data: (&mut state as *mut ArtifactStoreStub).cast::<c_void>(),
            materialize: Some(artifact_store_materialize_stub),
            release: None,
            destroy: None,
        };
        let store = CAbiRuntimeArtifactStore::new(table).unwrap();
        let error = store.materialize(&request).unwrap_err();
        assert!(format!("{error}").contains("unknown status code 997"));
    }

    #[test]
    fn c_abi_prediction_cache_store_loads_blocks_and_materializes_handles() {
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: dag_ml_core::PredictionLevel::Sample,
            fold_ids: vec![FoldId::new("fold:0").unwrap()],
            unit_ids: Vec::new(),
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
    fn c_abi_prediction_cache_store_loads_aggregated_blocks() {
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Target,
            fold_ids: vec![FoldId::new("fold:0").unwrap()],
            unit_ids: vec![PredictionUnitId::Target(TargetId::new("target:1").unwrap())],
            sample_ids: Vec::new(),
            prediction_width: 1,
            target_names: vec!["y".to_string()],
        };
        let blocks = vec![AggregatedPredictionBlock {
            prediction_id: Some("prediction:model:base.target.fold0".to_string()),
            producer_node: requirement.producer_node.clone(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            level: PredictionLevel::Target,
            unit_ids: requirement.unit_ids.clone(),
            values: vec![vec![0.42]],
            target_names: vec!["y".to_string()],
        }];
        let cache = build_aggregated_prediction_cache_record(&requirement, &blocks).unwrap();
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

        let loaded = store.load_aggregated_blocks(&requirement.key()).unwrap();
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
        let request_json: serde_json::Value =
            serde_json::from_slice(&state.materialize_json).unwrap();
        assert_eq!(request_json["requirement"]["prediction_level"], "target");
        assert_eq!(
            request_json["requirement"]["unit_ids"][0]["level"],
            "target"
        );
        assert_eq!(request_json["requirement"]["unit_ids"][0]["id"], "target:1");
        assert_eq!(request_json["cache"]["unit_ids"][0]["level"], "target");
    }

    #[test]
    fn c_abi_prediction_cache_store_releases_materialized_handles_on_drop() {
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Sample,
            fold_ids: vec![FoldId::new("fold:0").unwrap()],
            unit_ids: Vec::new(),
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
        {
            let table = DagMlPredictionCacheVTable {
                abi_version: 1,
                user_data: (&mut state as *mut PredictionCacheStub).cast::<c_void>(),
                load_blocks: Some(prediction_cache_load_blocks_stub),
                materialize: Some(prediction_cache_materialize_stub),
                release_bytes: Some(prediction_cache_release_bytes_stub),
                release: Some(prediction_cache_release_stub),
                destroy: Some(prediction_cache_destroy_stub),
            };
            let store = CAbiRuntimePredictionCacheStore::new(table).unwrap();
            let handle = store
                .materialize(&PredictionCacheMaterializationRequest {
                    run_id: RunId::new("run:prediction.cache.abi.release").unwrap(),
                    bundle_id: BundleId::new("bundle:prediction.cache.abi.release").unwrap(),
                    phase: Phase::Refit,
                    variant_id: None,
                    requirement: requirement.clone(),
                    cache,
                    producer_controller_id: ControllerId::new("controller:model").unwrap(),
                })
                .unwrap();
            assert_eq!(handle.handle, 77);
            assert!(state.release_handles.is_empty());
        }
        assert_eq!(state.release_handles, vec![77]);
        assert_eq!(state.destroy_count, 0);
    }

    #[test]
    fn c_abi_prediction_cache_store_owned_vtable_destroys_user_data_after_handles() {
        let requirement = BundlePredictionRequirement {
            producer_node: NodeId::new("model:base").unwrap(),
            source_port: "pred".to_string(),
            consumer_node: NodeId::new("model:meta").unwrap(),
            target_port: "pred".to_string(),
            partition: PredictionPartition::Validation,
            prediction_level: PredictionLevel::Sample,
            fold_ids: vec![FoldId::new("fold:0").unwrap()],
            unit_ids: Vec::new(),
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
        {
            let table = DagMlPredictionCacheVTable {
                abi_version: DAG_ML_PREDICTION_CACHE_VTABLE_OWNED_ABI_VERSION,
                user_data: (&mut state as *mut PredictionCacheStub).cast::<c_void>(),
                load_blocks: Some(prediction_cache_load_blocks_stub),
                materialize: Some(prediction_cache_materialize_stub),
                release_bytes: Some(prediction_cache_release_bytes_stub),
                release: Some(prediction_cache_release_stub),
                destroy: Some(prediction_cache_destroy_stub),
            };
            let store = CAbiRuntimePredictionCacheStore::new(table).unwrap();
            let handle = store
                .materialize(&PredictionCacheMaterializationRequest {
                    run_id: RunId::new("run:prediction.cache.abi.owned").unwrap(),
                    bundle_id: BundleId::new("bundle:prediction.cache.abi.owned").unwrap(),
                    phase: Phase::Refit,
                    variant_id: None,
                    requirement: requirement.clone(),
                    cache,
                    producer_controller_id: ControllerId::new("controller:model").unwrap(),
                })
                .unwrap();
            assert_eq!(handle.handle, 77);
            assert!(state.release_handles.is_empty());
            assert_eq!(state.destroy_count, 0);
        }
        assert_eq!(state.release_handles, vec![77]);
        assert_eq!(state.destroy_count, 1);
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

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
    }

    #[test]
    fn exposes_data_output_provenance_contract_over_abi() {
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe { dagml_data_output_provenance_contract_json(&mut out, &mut error) };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let contract: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(contract["schema_version"], 1);
        assert_eq!(contract["extra_key"], "dag_ml_output");
        assert_eq!(
            contract["schema_id"],
            dag_ml_core::DATA_OUTPUT_PROVENANCE_SCHEMA_ID
        );
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn validates_data_output_provenance_over_abi() {
        let provenance = include_bytes!(
            "../../../examples/fixtures/runtime/data_output_provenance_augmented_view.json"
        );
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_data_output_provenance_validate_json(
                provenance.as_ptr(),
                provenance.len(),
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());

        let invalid = br#"{
  "schema_version": 2,
  "producer_node": "branch:b1.augment:noise",
  "producer_port": "x_out",
  "producer_phase": "FIT_CV"
}"#;
        let status = unsafe {
            dagml_data_output_provenance_validate_json(invalid.as_ptr(), invalid.len(), &mut error)
        };
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(error_message(&error).contains("unsupported schema_version"));
        unsafe { dagml_string_free(error) };
    }

    #[test]
    fn compiles_pipeline_dsl_over_abi() {
        let dsl = include_bytes!("../../../examples/pipeline_dsl_branch_merge.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_pipeline_dsl_compile_json(dsl.as_ptr(), dsl.len(), &mut out, &mut error)
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let graph: GraphSpec = serde_json::from_slice(json).unwrap();
        assert_eq!(graph.id, "dsl-branch-merge-oof-smoke");
        assert_eq!(graph.nodes.len(), 4);
        assert!(graph.edges.iter().any(|edge| edge.contract.requires_oof
            && edge.target.node_id.as_str() == "merge:stack.pred_plus_original.meta:ridge"
            && edge.target.port_name == "b0_oof"));
        graph.validate().unwrap();
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn compiles_pipeline_dsl_generation_artifact_over_abi() {
        let dsl = include_bytes!("../../../examples/pipeline_dsl_generation.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_pipeline_dsl_compile_artifact_json(dsl.as_ptr(), dsl.len(), &mut out, &mut error)
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let artifact: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(artifact["graph"]["id"], "dsl-generation-smoke");
        assert_eq!(artifact["generation"]["strategy"], "cartesian");
        assert_eq!(
            artifact["generation"]["dimensions"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            artifact["generation"]["dimensions"][1]["choices"][1]["param_overrides"][0]["params"]
                ["alpha"],
            1.0
        );
        assert_eq!(
            artifact["graph"]["search_space_fingerprint"],
            artifact["generation_fingerprint"]
        );
        assert_eq!(
            artifact["campaign_template"]["generation"],
            artifact["generation"]
        );
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn compiles_pipeline_dsl_coordinated_generation_artifact_over_abi() {
        let dsl = include_bytes!("../../../examples/pipeline_dsl_coordinated_generation.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_pipeline_dsl_compile_artifact_json(dsl.as_ptr(), dsl.len(), &mut out, &mut error)
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let artifact: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(
            artifact["generation"]["dimensions"][0]["name"],
            "stack_profile"
        );
        assert_eq!(
            artifact["generation"]["dimensions"][0]["choices"][1]["param_overrides"][2]["node_id"],
            "merge:stack.pred_plus_original.meta:ridge"
        );
        assert_eq!(
            artifact["graph"]["search_space_fingerprint"],
            artifact["generation_fingerprint"]
        );
        assert_eq!(
            artifact["campaign_template"]["split_invocation"]["id"],
            "split:group-kfold"
        );
        assert_eq!(
            artifact["campaign_template"]["generation"],
            artifact["generation"]
        );
        assert_eq!(
            artifact["data_bindings"]["merge:stack.pred_plus_original.meta:ridge"][0]["input_name"],
            "x_original"
        );
        assert_eq!(
            artifact["campaign_template"]["data_bindings"],
            artifact["data_bindings"]
        );
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn builds_pipeline_dsl_execution_plan_over_abi() {
        let dsl = include_bytes!("../../../examples/pipeline_dsl_coordinated_generation.json");
        let manifests = include_bytes!("../../../examples/controller_manifests.json");
        let plan_id = b"plan:cabi.dsl.coordinated";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_pipeline_dsl_execution_plan_build_json(
                dsl.as_ptr(),
                dsl.len(),
                manifests.as_ptr(),
                manifests.len(),
                DagMlBytesView {
                    ptr: plan_id.as_ptr(),
                    len: plan_id.len(),
                },
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let plan: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(plan["id"], "plan:cabi.dsl.coordinated");
        assert_eq!(plan["variants"].as_array().unwrap().len(), 2);
        assert_eq!(
            plan["campaign"]["data_bindings"]["merge:stack.pred_plus_original.meta:ridge"][0]
                ["input_name"],
            "x_original"
        );
        assert!(plan["graph_plan"]["graph"]["search_space_fingerprint"]
            .as_str()
            .is_some());
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn compiles_pipeline_dsl_shape_plan_artifact_over_abi() {
        let dsl = include_bytes!("../../../examples/pipeline_dsl_shape_plan.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_pipeline_dsl_compile_artifact_json(dsl.as_ptr(), dsl.len(), &mut out, &mut error)
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let artifact: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(artifact["graph"]["id"], "dsl-shape-plan-smoke");
        assert_eq!(
            artifact["shape_plans"]["augment:synthetic"]["augmentation_policy"]["sample_scope"],
            "train_only"
        );
        assert_eq!(
            artifact["shape_plans"]["transform:select"]["selection_policy"]["scope"],
            "supervised_fold_train"
        );
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn returns_graph_parallel_levels_over_abi() {
        let graph = include_bytes!("../../../examples/minimal_graph.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_graph_parallel_levels_json(graph.as_ptr(), graph.len(), &mut out, &mut error)
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let levels: Vec<Vec<NodeId>> = serde_json::from_slice(json).unwrap();
        assert_eq!(
            levels,
            vec![
                vec![NodeId::new("transform:snv").unwrap()],
                vec![NodeId::new("model:base").unwrap()]
            ]
        );
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn returns_execution_schedule_over_abi() {
        let plan = fixture_plan_json();
        let phase = b"FIT_CV";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_execution_plan_schedule_json(
                plan.as_ptr(),
                plan.len(),
                bytes_view(phase),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let schedule: dag_ml_core::PhaseExecutionSchedule = serde_json::from_slice(json).unwrap();
        assert_eq!(schedule.phase, Phase::FitCv);
        assert_eq!(schedule.scopes.len(), 4);
        assert_eq!(
            schedule.scopes[0].node_levels,
            vec![
                vec![NodeId::new("transform:snv").unwrap()],
                vec![NodeId::new("model:base").unwrap()]
            ]
        );
        unsafe { dagml_owned_bytes_free(out) };
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

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
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
    fn scores_regression_predictions_over_abi() {
        let predictions = br#"{
  "prediction_id": "pred:sample",
  "producer_node": "model:pls",
  "partition": "validation",
  "sample_ids": ["sample:1", "sample:2"],
  "values": [[2.0], [4.0]],
  "target_names": ["y"]
}"#;
        let targets = br#"{
  "level": "sample",
  "unit_ids": [
    {"level": "sample", "id": "sample:2"},
    {"level": "sample", "id": "sample:1"}
  ],
  "values": [[5.0], [1.0]],
  "target_names": ["y"]
}"#;
        let metrics = br#"["rmse", "mae", "r2"]"#;
        let mut report_out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_score_regression_prediction_block_json(
                predictions.as_ptr(),
                predictions.len(),
                targets.as_ptr(),
                targets.len(),
                metrics.as_ptr(),
                metrics.len(),
                &mut report_out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!report_out.ptr.is_null());
        let report_json = unsafe { slice::from_raw_parts(report_out.ptr, report_out.len) };
        let report: RegressionMetricReport = serde_json::from_slice(report_json).unwrap();
        assert_eq!(report.producer_node, NodeId::new("model:pls").unwrap());
        assert_eq!(report.partition, PredictionPartition::Validation);
        assert_eq!(report.metrics["rmse"], 1.0);
        assert_eq!(report.metrics["r2"], 0.75);

        let candidate_id = b"model:pls";
        let mut candidate_out = DagMlOwnedBytes::default();
        let status = unsafe {
            dagml_regression_report_candidate_score_json(
                report_json.as_ptr(),
                report_json.len(),
                bytes_view(candidate_id),
                &mut candidate_out,
                &mut error,
            )
        };
        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        let candidate_json = unsafe { slice::from_raw_parts(candidate_out.ptr, candidate_out.len) };
        let candidate: CandidateScore = serde_json::from_slice(candidate_json).unwrap();
        assert_eq!(candidate.candidate_id, "model:pls");
        assert_eq!(candidate.metrics["rmse"], 1.0);
        assert_eq!(candidate.metadata["producer_node"], "model:pls");
        unsafe { dagml_owned_bytes_free(candidate_out) };
        unsafe { dagml_owned_bytes_free(report_out) };
    }

    #[test]
    fn exports_prediction_block_f64_tensor_over_abi() {
        let predictions = br#"{
  "prediction_id": "pred:sample",
  "producer_node": "model:base",
  "partition": "validation",
  "fold_id": "fold:0",
  "sample_ids": ["sample:1", "sample:2"],
  "values": [[1.0, 2.5], [3.0, 4.5]],
  "target_names": ["y1", "y2"]
}"#;
        let mut out = DagMlF64Tensor::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_prediction_block_f64_tensor_json(
                predictions.as_ptr(),
                predictions.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        assert_eq!(out.rows, 2);
        assert_eq!(out.cols, 2);
        assert_eq!(out.len, 4);
        let values = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        assert_eq!(values, &[1.0, 2.5, 3.0, 4.5]);
        unsafe { dagml_f64_tensor_free(out) };
    }

    #[test]
    fn exports_aggregated_prediction_block_f64_tensor_over_abi() {
        let predictions = br#"{
  "prediction_id": "pred:target",
  "producer_node": "model:base",
  "partition": "validation",
  "fold_id": "fold:0",
  "level": "target",
  "unit_ids": [
    {"level": "target", "id": "target:1"},
    {"level": "target", "id": "target:2"}
  ],
  "values": [[9.0], [11.0]],
  "target_names": ["y"]
}"#;
        let mut out = DagMlF64Tensor::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_aggregated_prediction_block_f64_tensor_json(
                predictions.as_ptr(),
                predictions.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        assert_eq!(out.rows, 2);
        assert_eq!(out.cols, 1);
        assert_eq!(out.len, 2);
        let values = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        assert_eq!(values, &[9.0, 11.0]);
        unsafe { dagml_f64_tensor_free(out) };
    }

    #[test]
    fn rejects_null_f64_tensor_output_pointer_over_abi() {
        let predictions = br#"{
  "prediction_id": "pred:sample",
  "producer_node": "model:base",
  "partition": "validation",
  "sample_ids": ["sample:1"],
  "values": [[1.0]],
  "target_names": ["y"]
}"#;
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_prediction_block_f64_tensor_json(
                predictions.as_ptr(),
                predictions.len(),
                std::ptr::null_mut(),
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::INVALID_ARGUMENT);
        assert_eq!(error_message(&error), "output F64 tensor pointer is null");
        unsafe { dagml_string_free(error) };
    }

    #[test]
    fn scores_aggregated_regression_predictions_over_abi() {
        let predictions = br#"{
  "prediction_id": "pred:target",
  "producer_node": "model:pls",
  "partition": "validation",
  "level": "target",
  "unit_ids": [
    {"level": "target", "id": "target:a"},
    {"level": "target", "id": "target:b"}
  ],
  "values": [[1.0, 10.0], [3.0, 30.0]],
  "target_names": ["y1", "y2"]
}"#;
        let targets = br#"{
  "level": "target",
  "unit_ids": [
    {"level": "target", "id": "target:b"},
    {"level": "target", "id": "target:a"}
  ],
  "values": [[2.0, 28.0], [2.0, 12.0]],
  "target_names": ["y1", "y2"]
}"#;
        let metrics = br#"["mse", "rmse"]"#;
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_score_regression_aggregated_block_json(
                predictions.as_ptr(),
                predictions.len(),
                targets.as_ptr(),
                targets.len(),
                metrics.as_ptr(),
                metrics.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let report: RegressionMetricReport = serde_json::from_slice(json).unwrap();
        assert_eq!(report.row_count, 2);
        assert_eq!(report.target_width, 2);
        assert_eq!(report.metrics["mse"], 2.5);
        assert_eq!(report.metrics["rmse"], 1.5);
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn validates_bundle_replay_contracts_over_abi() {
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let request =
            include_bytes!("../../../examples/fixtures/bundle/replay_request_predict.json");
        let envelope = include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        );
        let envelopes = format!(r#"{{"model:base.x":{envelope}}}"#);
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_execution_bundle_validate_json(bundle.as_ptr(), bundle.len(), &mut error)
        };
        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
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
        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
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
    fn exports_prediction_cache_payload_f64_tensor_over_abi() {
        let bundle = include_bytes!(
            "../../../examples/generated/execution_bundle_branch_merge_cv_refit.json"
        );
        let payload = include_bytes!(
            "../../../examples/generated/prediction_cache_branch_merge_cv_refit.json"
        );
        let requirement_key =
            b"branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof";
        let mut tensor = DagMlF64Tensor::default();
        let mut metadata_out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_prediction_cache_payload_f64_tensor_json(
                bundle.as_ptr(),
                bundle.len(),
                payload.as_ptr(),
                payload.len(),
                bytes_view(requirement_key),
                &mut tensor,
                &mut metadata_out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!tensor.ptr.is_null());
        assert_eq!(tensor.rows, 4);
        assert_eq!(tensor.cols, 1);
        assert_eq!(tensor.len, 4);
        let values = unsafe { slice::from_raw_parts(tensor.ptr, tensor.len) };
        assert_eq!(values, &[9931.0, 9931.0, 9932.0, 9932.0]);

        assert!(!metadata_out.ptr.is_null());
        let metadata_json = unsafe { slice::from_raw_parts(metadata_out.ptr, metadata_out.len) };
        let metadata: serde_json::Value = serde_json::from_slice(metadata_json).unwrap();
        assert_eq!(
            metadata["requirement_key"],
            "branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof"
        );
        assert_eq!(metadata["schema_version"], 1);
        assert_eq!(metadata["prediction_level"], "sample");
        assert_eq!(metadata["block_count"], 2);
        assert_eq!(metadata["row_count"], 4);
        assert_eq!(metadata["rows"], 4);
        assert_eq!(metadata["cols"], 1);
        assert_eq!(metadata["target_names"][0], "y");
        assert_eq!(metadata["blocks"][0]["fold_id"], "fold:0");
        assert_eq!(metadata["blocks"][0]["row_offset"], 0);
        assert_eq!(metadata["blocks"][0]["row_count"], 2);
        assert_eq!(metadata["blocks"][0]["sample_ids"][0], "sample:1");
        assert_eq!(metadata["blocks"][1]["fold_id"], "fold:1");
        assert_eq!(metadata["blocks"][1]["row_offset"], 2);
        unsafe { dagml_f64_tensor_free(tensor) };
        unsafe { dagml_owned_bytes_free(metadata_out) };
    }

    #[test]
    fn rejects_null_prediction_cache_payload_tensor_metadata_pointer_over_abi() {
        let bundle = include_bytes!(
            "../../../examples/generated/execution_bundle_branch_merge_cv_refit.json"
        );
        let payload = include_bytes!(
            "../../../examples/generated/prediction_cache_branch_merge_cv_refit.json"
        );
        let requirement_key =
            b"branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof";
        let mut tensor = DagMlF64Tensor::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_prediction_cache_payload_f64_tensor_json(
                bundle.as_ptr(),
                bundle.len(),
                payload.as_ptr(),
                payload.len(),
                bytes_view(requirement_key),
                &mut tensor,
                std::ptr::null_mut(),
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::INVALID_ARGUMENT);
        assert_eq!(
            error_message(&error),
            "output metadata JSON pointer is null"
        );
        assert!(tensor.ptr.is_null());
        unsafe { dagml_string_free(error) };
    }

    #[test]
    fn exports_research_provenance_over_abi() {
        let plan = fixture_plan_json();
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_research_provenance_export_json(
                plan.as_ptr(),
                plan.len(),
                bundle.as_ptr(),
                bundle.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let export: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(export["schema_version"], 1);
        assert_eq!(
            export["prov_jsonld"]["@context"]["prov"],
            "http://www.w3.org/ns/prov#"
        );
        assert!(export["prov_jsonld"]["wasDerivedFrom"]
            .to_string()
            .contains("dagml:derived:bundle-plan"));
        assert!(export["ro_crate_metadata"]["@graph"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["@type"].to_string().contains("ComputationalWorkflow")));
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn exports_openlineage_run_event_over_abi() {
        let plan = fixture_plan_json();
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let namespace = b"dag-ml-capi-test";
        let event_time = b"2026-05-27T00:00:00Z";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_openlineage_run_event_json(
                plan.as_ptr(),
                plan.len(),
                bundle.as_ptr(),
                bundle.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                bytes_view(namespace),
                bytes_view(event_time),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let event: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(event["eventType"], "COMPLETE");
        assert_eq!(event["job"]["namespace"], "dag-ml-capi-test");
        assert!(event["run"]["facets"]["dagml_reproducibility"]
            .to_string()
            .contains("graph_fingerprint"));
        assert!(event["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|output| output["namespace"] == "dagml:bundle"));
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn executes_mock_replay_over_abi() {
        let plan = fixture_plan_json();
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let request =
            include_bytes!("../../../examples/fixtures/bundle/replay_request_predict.json");
        let envelope = include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        );
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

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
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
    fn executes_vtable_replay_over_abi() {
        let plan = fixture_plan_json();
        let bundle = include_bytes!("../../../examples/generated/execution_bundle_minimal.json");
        let request =
            include_bytes!("../../../examples/fixtures/bundle/replay_request_predict.json");
        let envelope = include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        );
        let envelopes = format!(r#"{{"model:base.x":{envelope}}}"#);
        let mut data_state = DataProviderStub::default();
        let data_provider = DagMlDataVTable {
            abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
            user_data: (&mut data_state as *mut DataProviderStub).cast::<c_void>(),
            materialize: Some(materialize_stub),
            make_view: Some(make_view_stub),
            view_identity: None,
            target_arrow: None,
            feature_arrow: None,
            release: None,
            destroy: None,
        };
        let mut artifact_state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 700,
                kind: DAG_ML_HANDLE_KIND_MODEL,
            },
            ..Default::default()
        };
        let artifact_store = DagMlArtifactStoreVTable {
            abi_version: 1,
            user_data: (&mut artifact_state as *mut ArtifactStoreStub).cast::<c_void>(),
            materialize: Some(artifact_store_materialize_stub),
            release: None,
            destroy: None,
        };
        let mut transform_state = ControllerStub::default();
        let mut model_state = ControllerStub::default();
        let transform_controller_id = b"controller:transform.mock";
        let model_controller_id = b"controller:model.mock";
        let bindings = [
            DagMlControllerBinding {
                controller_id: bytes_view(transform_controller_id),
                vtable: DagMlControllerVTable {
                    abi_version: 2,
                    user_data: (&mut transform_state as *mut ControllerStub).cast::<c_void>(),
                    clone_with: None,
                    describe: None,
                    fit: None,
                    predict: None,
                    invoke: Some(replay_controller_invoke_stub),
                    release_bytes: Some(controller_release_bytes_stub),
                    release: None,
                    destroy: None,
                },
            },
            DagMlControllerBinding {
                controller_id: bytes_view(model_controller_id),
                vtable: DagMlControllerVTable {
                    abi_version: 2,
                    user_data: (&mut model_state as *mut ControllerStub).cast::<c_void>(),
                    clone_with: None,
                    describe: None,
                    fit: None,
                    predict: None,
                    invoke: Some(replay_controller_invoke_stub),
                    release_bytes: Some(controller_release_bytes_stub),
                    release: None,
                    destroy: None,
                },
            },
        ];
        let data_owner = b"controller:data.provider";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_replay_execute_json(
                plan.as_ptr(),
                plan.len(),
                bundle.as_ptr(),
                bundle.len(),
                request.as_ptr(),
                request.len(),
                envelopes.as_ptr(),
                envelopes.len(),
                bytes_view(data_owner),
                7,
                data_provider,
                artifact_store,
                std::ptr::null(),
                bindings.as_ptr(),
                bindings.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let summary: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(summary["bundle_id"], "bundle:cli.demo");
        assert_eq!(summary["result_count"], 2);
        assert_eq!(summary["prediction_block_count"], 1);
        assert_eq!(summary["controller_count"], 2);
        assert_eq!(summary["prediction_cache_store"], false);
        assert_eq!(data_state.materialize_dataset, 7);
        assert_eq!(artifact_state.handle.handle, 700);
        assert_eq!(transform_state.task_node_ids, vec!["transform:snv"]);
        assert_eq!(model_state.task_node_ids, vec!["model:base"]);
        assert_eq!(transform_state.release_count, 1);
        assert_eq!(model_state.release_count, 1);
        unsafe { dagml_owned_bytes_free(out) };
    }

    #[test]
    fn executes_vtable_refit_replay_with_prediction_cache_over_abi() {
        let plan = fixture_branch_merge_plan_json();
        let bundle = include_bytes!(
            "../../../examples/generated/execution_bundle_branch_merge_cv_refit.json"
        );
        let request = include_bytes!(
            "../../../examples/fixtures/bundle/replay_request_branch_merge_refit.json"
        );
        let envelope = include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        );
        let envelopes = format!(
            r#"{{
                "branch:b0.model:ridge.x":{envelope},
                "branch:b1.model:rf.x":{envelope},
                "merge:stack.pred_plus_original.meta:ridge.x_original":{envelope}
            }}"#
        );
        let payloads: BundlePredictionCachePayloadSet = serde_json::from_slice(include_bytes!(
            "../../../examples/generated/prediction_cache_branch_merge_cv_refit.json"
        ))
        .unwrap();
        let mut blocks_by_key = BTreeMap::new();
        for payload in payloads.caches {
            blocks_by_key.insert(
                payload.requirement_key.clone(),
                serde_json::to_vec(&payload.blocks).unwrap(),
            );
        }
        let mut prediction_cache_state = PredictionCacheStub {
            blocks_by_key,
            ..Default::default()
        };
        let prediction_cache = DagMlPredictionCacheVTable {
            abi_version: 1,
            user_data: (&mut prediction_cache_state as *mut PredictionCacheStub).cast::<c_void>(),
            load_blocks: Some(prediction_cache_load_blocks_stub),
            materialize: Some(prediction_cache_materialize_stub),
            release_bytes: Some(prediction_cache_release_bytes_stub),
            release: None,
            destroy: None,
        };
        let mut data_state = DataProviderStub::default();
        let data_provider = DagMlDataVTable {
            abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
            user_data: (&mut data_state as *mut DataProviderStub).cast::<c_void>(),
            materialize: Some(materialize_stub),
            make_view: Some(make_view_stub),
            view_identity: None,
            target_arrow: None,
            feature_arrow: None,
            release: None,
            destroy: None,
        };
        let mut artifact_state = ArtifactStoreStub {
            handle: DagMlHandleRef {
                handle: 701,
                kind: DAG_ML_HANDLE_KIND_MODEL,
            },
            ..Default::default()
        };
        let artifact_store = DagMlArtifactStoreVTable {
            abi_version: 1,
            user_data: (&mut artifact_state as *mut ArtifactStoreStub).cast::<c_void>(),
            materialize: Some(artifact_store_materialize_stub),
            release: None,
            destroy: None,
        };
        let mut model_state = ControllerStub::default();
        let model_controller_id = b"controller:model.mock";
        let bindings = [DagMlControllerBinding {
            controller_id: bytes_view(model_controller_id),
            vtable: DagMlControllerVTable {
                abi_version: 2,
                user_data: (&mut model_state as *mut ControllerStub).cast::<c_void>(),
                clone_with: None,
                describe: None,
                fit: None,
                predict: None,
                invoke: Some(replay_controller_invoke_stub),
                release_bytes: Some(controller_release_bytes_stub),
                release: None,
                destroy: None,
            },
        }];
        let data_owner = b"controller:data.provider";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_replay_execute_json(
                plan.as_ptr(),
                plan.len(),
                bundle.as_ptr(),
                bundle.len(),
                request.as_ptr(),
                request.len(),
                envelopes.as_ptr(),
                envelopes.len(),
                bytes_view(data_owner),
                7,
                data_provider,
                artifact_store,
                &prediction_cache,
                bindings.as_ptr(),
                bindings.len(),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(&error));
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let summary: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(
            summary["bundle_id"],
            "bundle:generated.branch.merge.cv.refit"
        );
        assert_eq!(summary["phase"], "REFIT");
        assert_eq!(summary["result_count"], 3);
        assert_eq!(summary["prediction_cache_store"], true);
        assert_eq!(prediction_cache_state.load_keys.len(), 2);
        assert_eq!(prediction_cache_state.materialize_count, 2);
        assert_eq!(prediction_cache_state.release_count, 2);
        assert_eq!(
            model_state.task_node_ids,
            vec![
                "branch:b0.model:ridge".to_string(),
                "branch:b1.model:rf".to_string(),
                "merge:stack.pred_plus_original.meta:ridge".to_string()
            ]
        );
        assert_eq!(model_state.release_count, 3);
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

    #[test]
    fn builds_execution_plan_over_abi() {
        let graph = include_bytes!("../../../examples/minimal_graph.json");
        let campaign = include_bytes!("../../../examples/campaign_oof_generation.json");
        let manifests = include_bytes!("../../../examples/controller_manifests.json");
        let plan_id = b"plan:cabi.build";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();

        let status = unsafe {
            dagml_execution_plan_build_json(
                graph.as_ptr(),
                graph.len(),
                campaign.as_ptr(),
                campaign.len(),
                manifests.as_ptr(),
                manifests.len(),
                bytes_view(plan_id),
                &mut out,
                &mut error,
            )
        };

        assert_eq!(
            status,
            DagMlStatusCode::OK,
            "{}",
            if error.ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(error.ptr) }
                    .to_string_lossy()
                    .into_owned()
            }
        );
        assert!(error.ptr.is_null());
        assert!(!out.ptr.is_null());
        let json = unsafe { slice::from_raw_parts(out.ptr, out.len) };
        let plan: ExecutionPlan = serde_json::from_slice(json).unwrap();
        plan.validate().unwrap();
        assert_eq!(plan.id, "plan:cabi.build");
        assert_eq!(plan.node_plans.len(), plan.graph_plan.graph.nodes.len());
        assert_eq!(plan.controller_manifests.len(), 2);
        unsafe { dagml_owned_bytes_free(out) };
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

    fn fixture_branch_merge_plan_json() -> Vec<u8> {
        let graph: GraphSpec = serde_json::from_str(include_str!(
            "../../../examples/branch_merge_oof_graph.json"
        ))
        .unwrap();
        let campaign: CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_branch_merge_oof.json"
        ))
        .unwrap();
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(include_str!("../../../examples/controller_manifests.json"))
                .unwrap();
        let mut registry = ControllerRegistry::new();
        for manifest in manifests {
            registry.register(manifest).unwrap();
        }
        let plan = build_execution_plan(
            "plan:generated.branch.merge.cv.refit",
            graph,
            campaign,
            &registry,
        )
        .unwrap();
        serde_json::to_vec(&plan).unwrap()
    }
}
