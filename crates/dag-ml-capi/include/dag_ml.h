#ifndef DAG_ML_H
#define DAG_ML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t DagMlHandle;

enum {
    DAG_ML_HANDLE_KIND_DATA = 1,
    DAG_ML_HANDLE_KIND_DATA_VIEW = 2,
    DAG_ML_HANDLE_KIND_MODEL = 3,
    DAG_ML_HANDLE_KIND_ARTIFACT = 4,
    DAG_ML_HANDLE_KIND_PREDICTION = 5,
    DAG_ML_HANDLE_KIND_RELATION = 6
};

typedef uint32_t DagMlStatusCode;

enum {
    DAG_ML_STATUS_OK = 0,
    DAG_ML_STATUS_INVALID_ARGUMENT = 1,
    DAG_ML_STATUS_VALIDATION_ERROR = 2,
    DAG_ML_STATUS_PANIC = 255
};

typedef struct DagMlVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} DagMlVersion;

typedef struct DagMlString {
    char *ptr;
    size_t len;
} DagMlString;

typedef struct DagMlBytesView {
    const uint8_t *ptr;
    size_t len;
} DagMlBytesView;

typedef struct DagMlHandleRef {
    DagMlHandle handle;
    uint32_t kind;
} DagMlHandleRef;

typedef struct DagMlOwnedBytes {
    uint8_t *ptr;
    size_t len;
    size_t capacity;
} DagMlOwnedBytes;

typedef struct DagMlF64Tensor {
    double *ptr;
    size_t len;
    size_t capacity;
    size_t rows;
    size_t cols;
} DagMlF64Tensor;

typedef struct DagMlF64ColumnarTensor {
    double *ptr;
    size_t len;
    size_t capacity;
    size_t rows;
    size_t cols;
} DagMlF64ColumnarTensor;

typedef struct DagMlF32Tensor {
    float *ptr;
    size_t len;
    size_t capacity;
    size_t rows;
    size_t cols;
} DagMlF32Tensor;

typedef struct DagMlF32ColumnarTensor {
    float *ptr;
    size_t len;
    size_t capacity;
    size_t rows;
    size_t cols;
} DagMlF32ColumnarTensor;

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

typedef struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void **buffers;
    struct ArrowArray **children;
    struct ArrowArray *dictionary;
    void (*release)(struct ArrowArray *array);
    void *private_data;
} ArrowArray;

typedef struct ArrowSchema {
    const char *format;
    const char *name;
    const char *metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema **children;
    struct ArrowSchema *dictionary;
    void (*release)(struct ArrowSchema *schema);
    void *private_data;
} ArrowSchema;

#endif

#ifndef DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION
#define DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION 2u
#endif

#ifndef DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION
#define DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION 3u
#endif

#ifndef DAG_ML_ARTIFACT_STORE_VTABLE_BORROWED_ABI_VERSION
#define DAG_ML_ARTIFACT_STORE_VTABLE_BORROWED_ABI_VERSION 1u
#endif

#ifndef DAG_ML_ARTIFACT_STORE_VTABLE_OWNED_ABI_VERSION
#define DAG_ML_ARTIFACT_STORE_VTABLE_OWNED_ABI_VERSION 2u
#endif

#ifndef DAG_ML_PREDICTION_CACHE_VTABLE_BORROWED_ABI_VERSION
#define DAG_ML_PREDICTION_CACHE_VTABLE_BORROWED_ABI_VERSION 1u
#endif

#ifndef DAG_ML_PREDICTION_CACHE_VTABLE_OWNED_ABI_VERSION
#define DAG_ML_PREDICTION_CACHE_VTABLE_OWNED_ABI_VERSION 2u
#endif

#ifndef DAG_ML_LOCAL_IMPLEMENTATION_VTABLE_ABI_VERSION
#define DAG_ML_LOCAL_IMPLEMENTATION_VTABLE_ABI_VERSION 1u
#endif

#ifndef DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION
#define DAG_ML_PREDICTION_CACHE_TENSOR_METADATA_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_VERSION
#define DAG_ML_PREDICTION_CACHE_COLUMNAR_TENSOR_METADATA_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_GRAPH_SPEC_SCHEMA_VERSION
#define DAG_ML_GRAPH_SPEC_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_CAMPAIGN_SPEC_SCHEMA_VERSION
#define DAG_ML_CAMPAIGN_SPEC_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_EXECUTION_PLAN_SCHEMA_VERSION
#define DAG_ML_EXECUTION_PLAN_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_MODEL_INPUT_SPEC_SCHEMA_VERSION
#define DAG_ML_MODEL_INPUT_SPEC_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_DATA_PLAN_SCHEMA_VERSION
#define DAG_ML_DATA_PLAN_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_CONTROLLER_MANIFEST_SCHEMA_VERSION
#define DAG_ML_CONTROLLER_MANIFEST_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION
#define DAG_ML_DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_NODE_TASK_SCHEMA_VERSION
#define DAG_ML_NODE_TASK_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_NODE_RESULT_SCHEMA_VERSION
#define DAG_ML_NODE_RESULT_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_PIPELINE_DSL_SCHEMA_VERSION
#define DAG_ML_PIPELINE_DSL_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION
#define DAG_ML_PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_PROCESS_ADAPTER_FRAME_SCHEMA_VERSION
#define DAG_ML_PROCESS_ADAPTER_FRAME_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION
#define DAG_ML_AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION
#define DAG_ML_AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_SELECTION_POLICY_SCHEMA_VERSION
#define DAG_ML_SELECTION_POLICY_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_SELECTION_DECISION_SCHEMA_VERSION
#define DAG_ML_SELECTION_DECISION_SCHEMA_VERSION 1u
#endif

#ifndef DAG_ML_DATA_OUTPUT_PROVENANCE_EXTRA_KEY
#define DAG_ML_DATA_OUTPUT_PROVENANCE_EXTRA_KEY "dag_ml_output"
#endif

typedef struct DagMlControllerVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*clone_with)(void *user_data, DagMlHandle op, DagMlBytesView params_json, DagMlHandle *out_op);
    DagMlStatusCode (*describe)(void *user_data, DagMlHandle op, DagMlOwnedBytes *out_json);
    DagMlStatusCode (*fit)(void *user_data, DagMlHandle op, DagMlHandle data, DagMlBytesView context_json, DagMlHandle *out_fitted);
    DagMlStatusCode (*predict)(void *user_data, DagMlHandle fitted, DagMlHandle data, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    DagMlStatusCode (*invoke)(void *user_data, DagMlBytesView task_json, DagMlOwnedBytes *out_result_json);
    void (*release_bytes)(void *user_data, DagMlOwnedBytes bytes);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlControllerVTable;

#ifndef DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION
#define DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION 2u
#endif

#ifndef DAG_ML_DATA_VTABLE_DEFINED
#define DAG_ML_DATA_VTABLE_DEFINED
typedef struct DagMlDataVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*materialize)(void *user_data, DagMlHandle dataset, DagMlBytesView request_json, DagMlHandle *out_handle);
    DagMlStatusCode (*make_view)(void *user_data, DagMlHandle data, DagMlBytesView selector_json, DagMlHandle *out_view);
    DagMlStatusCode (*view_identity)(void *user_data, DagMlHandle view, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    DagMlStatusCode (*target_arrow)(void *user_data, DagMlHandle view, DagMlBytesView target_name, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    DagMlStatusCode (*feature_arrow)(void *user_data, DagMlHandle view, DagMlBytesView feature_set_name, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlDataVTable;
#endif

typedef struct DagMlPredictionCacheVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*load_blocks)(void *user_data, DagMlBytesView requirement_key, DagMlOwnedBytes *out_json);
    DagMlStatusCode (*materialize)(void *user_data, DagMlBytesView request_json, DagMlHandle *out_handle);
    void (*release_bytes)(void *user_data, DagMlOwnedBytes bytes);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlPredictionCacheVTable;

typedef struct DagMlArtifactStoreVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*materialize)(void *user_data, DagMlBytesView request_json, DagMlHandleRef *out_handle);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlArtifactStoreVTable;

/* Opaque process-local loss/metric callback registry. The registry is scoped
 * to the binding id supplied at creation (for example binding:c, binding:r, or
 * binding:matlab). Callback pointers and language runtime objects never enter
 * serialized DAG-ML contracts. */
typedef struct DagMlLocalImplementationRegistry DagMlLocalImplementationRegistry;

/* Generic host callback retained under an exact implementation descriptor.
 * invoke receives binding-defined strict JSON and returns host-allocated JSON;
 * release_bytes must release every non-NULL callback output, including error
 * outputs. retain/release are optional but must be supplied together. When
 * present, successful registration retains non-NULL user_data once and
 * unregister/clear/free releases it once. A failed registration leaves no
 * retained reference. Host exceptions and unwinds must be caught by the host
 * trampoline and reported as DAG_ML_STATUS_PANIC; they must not cross the C
 * boundary. The host must synchronize user_data unless its descriptor declares
 * thread-safe execution. Lifecycle callbacks must not re-enter the same
 * registry. */
typedef struct DagMlLocalImplementationVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*invoke)(void *user_data, DagMlBytesView request_json, DagMlOwnedBytes *out_result_json);
    void (*release_bytes)(void *user_data, DagMlOwnedBytes bytes);
    void (*retain)(void *user_data);
    void (*release)(void *user_data);
} DagMlLocalImplementationVTable;

typedef struct DagMlControllerBinding {
    DagMlBytesView controller_id;
    DagMlControllerVTable vtable;
} DagMlControllerBinding;

/* Opaque, owning result of dagml_training_execute. It keeps the training
 * outcome plus the controller registry and artifact store alive so emitted
 * model/refit handles stay valid. Read it with
 * dagml_training_result_outcome_json and release it with
 * dagml_training_result_free (which releases handles, then destroys each owning
 * controller user_data exactly once). Free at most once; NULL is a no-op. */
typedef struct DagMlTrainingResult DagMlTrainingResult;

/* Stateless input for dagml_training_execute. All JSON payloads are UTF-8.
 * warnings_json (["..."]) and diagnostics_json ({"k": <json>}) are optional: a
 * NULL pointer (or zero length) is the empty default. data_provider is a
 * borrowed data vtable whose user_data is never destroyed by this crate;
 * controllers whose vtable advertises the owned ABI are consumed by the call. */
typedef struct DagMlTrainingExecuteRequest {
    DagMlBytesView request_json;
    DagMlBytesView outcome_id;
    DagMlBytesView run_id;
    DagMlBytesView bundle_id;
    DagMlBytesView relations_json;
    DagMlBytesView influence_json;
    DagMlBytesView envelopes_json;
    DagMlBytesView warnings_json;
    DagMlBytesView diagnostics_json;
    DagMlHandle dataset;
    DagMlDataVTable data_provider;
    DagMlBytesView data_owner_controller_id;
    const DagMlControllerBinding *controller_bindings;
    size_t controller_binding_count;
} DagMlTrainingExecuteRequest;

/* Stateless input for dagml_training_result_replay. All JSON payloads are
 * UTF-8. warnings_json (["..."]) and diagnostics_json ({"k": <json>}) are
 * optional: a NULL pointer (or zero length) is the empty default. data_provider
 * is borrowed for the call; controllers/artifacts are borrowed from the live
 * DagMlTrainingResult. */
typedef struct DagMlTrainingReplayRequest {
    DagMlBytesView replay_request_json;
    DagMlBytesView outcome_id;
    DagMlBytesView run_id;
    DagMlBytesView data_envelopes_json;
    DagMlBytesView warnings_json;
    DagMlBytesView diagnostics_json;
    DagMlHandle dataset;
    DagMlDataVTable data_provider;
    DagMlBytesView data_owner_controller_id;
} DagMlTrainingReplayRequest;

DagMlVersion dagml_version(void);
void dagml_string_free(DagMlString value);
/* ADR-11 thread-local last-error accessors. The buffer holds the structured
 * descriptor JSON and numeric (category << 16 | code) of the most recent failing
 * call on the calling thread. Every failing call updates it: taxonomy errors
 * carry their real category/code; boundary errors (null pointer, bad UTF-8, JSON
 * parse) use validation/c_abi_argument (0x0000FFFF). Errno-like semantics: it is
 * NOT cleared on success, so check the function's return code first, then read
 * the buffer on the same thread. */
DagMlStatusCode dagml_last_error_json(DagMlString *out);
uint32_t dagml_last_error_code(void);
/* ADR-12 minimal telemetry hook: install a process-global tracing subscriber to
 * stderr (RUST_LOG-filtered, default "info"). json_output != 0 emits JSON-logfmt.
 * Returns OK, or VALIDATION_ERROR if a subscriber is already installed. */
DagMlStatusCode dagml_init_tracing(uint8_t json_output);
void dagml_owned_bytes_free(DagMlOwnedBytes value);
void dagml_f64_tensor_free(DagMlF64Tensor value);
void dagml_f64_columnar_tensor_free(DagMlF64ColumnarTensor value);
void dagml_f32_tensor_free(DagMlF32Tensor value);
void dagml_f32_columnar_tensor_free(DagMlF32ColumnarTensor value);
/* Select one zero-based, phase-filtered training-loss role from an exact
 * NodeTask. This validation-only entry point returns the native role and the
 * task-owned attestation without invoking a host callback, so runtimes can
 * keep tensors and executable functions local. Both outputs are released with
 * dagml_owned_bytes_free. */
DagMlStatusCode dagml_node_task_training_loss_binding(
    DagMlBytesView node_task_json,
    size_t role_index,
    DagMlOwnedBytes *out_training_loss_role_json,
    DagMlOwnedBytes *out_attestation_json,
    DagMlString *error_out);
/* Local callbacks are resolved by the complete validated loss/metric
 * descriptor, not only by registry_key. invoke_training_loss accepts only
 * FIT_CV/REFIT and emits a DAG-ML-owned attestation after callback success.
 * All returned DagMlOwnedBytes values must be released with
 * dagml_owned_bytes_free. Registry pointers are process-local and cannot be
 * serialized or shared across workers; each runtime must register locally. */
DagMlStatusCode dagml_local_implementation_registry_create(
    DagMlBytesView binding_id,
    DagMlLocalImplementationRegistry **out_registry,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_register_loss(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView loss_reference_json,
    DagMlLocalImplementationVTable implementation,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_register_metric(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView metric_reference_json,
    DagMlLocalImplementationVTable implementation,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_invoke_loss(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView loss_reference_json,
    DagMlBytesView request_json,
    DagMlOwnedBytes *out_result_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_invoke_training_loss(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView training_loss_role_json,
    DagMlBytesView phase,
    DagMlBytesView request_json,
    DagMlOwnedBytes *out_result_json,
    DagMlOwnedBytes *out_attestation_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_invoke_task_training_loss(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView node_task_json,
    size_t role_index,
    DagMlBytesView request_json,
    DagMlOwnedBytes *out_result_json,
    DagMlOwnedBytes *out_attestation_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_invoke_metric(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView metric_reference_json,
    DagMlBytesView request_json,
    DagMlOwnedBytes *out_result_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_unregister_loss(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView loss_reference_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_unregister_metric(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView metric_reference_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_descriptors_json(
    DagMlLocalImplementationRegistry *registry,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
DagMlStatusCode dagml_local_implementation_registry_clear(
    DagMlLocalImplementationRegistry *registry,
    DagMlString *error_out);
void dagml_local_implementation_registry_free(DagMlLocalImplementationRegistry *registry);
DagMlStatusCode dagml_graph_spec_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_graph_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_campaign_spec_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_campaign_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_execution_plan_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_model_input_spec_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_model_input_spec_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_data_plan_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_data_plan_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_controller_manifest_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_controller_manifest_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_controller_manifest_list_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_data_output_provenance_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_data_output_provenance_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_node_task_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_node_result_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_process_adapter_description_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_process_adapter_frame_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_aggregation_controller_task_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_aggregation_controller_result_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_aggregation_controller_task_validate_json(const uint8_t *task_ptr, size_t task_len, DagMlString *error_out);
DagMlStatusCode dagml_aggregation_controller_result_validate_for_task_json(const uint8_t *task_ptr, size_t task_len, const uint8_t *result_ptr, size_t result_len, DagMlString *error_out);
DagMlStatusCode dagml_node_result_validate_for_task_json(const uint8_t *task_ptr, size_t task_len, const uint8_t *result_ptr, size_t result_len, DagMlString *error_out);
DagMlStatusCode dagml_pipeline_dsl_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_pipeline_dsl_validate_json(const uint8_t *dsl_ptr, size_t dsl_len, DagMlString *error_out);
DagMlStatusCode dagml_pipeline_dsl_compile_json(const uint8_t *dsl_ptr, size_t dsl_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_pipeline_dsl_compile_artifact_json(const uint8_t *dsl_ptr, size_t dsl_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_pipeline_dsl_execution_plan_build_json(
    const uint8_t *dsl_ptr,
    size_t dsl_len,
    const uint8_t *controllers_ptr,
    size_t controllers_len,
    DagMlBytesView plan_id,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
DagMlStatusCode dagml_graph_parallel_levels_json(
    const uint8_t *json_ptr,
    size_t json_len,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
DagMlStatusCode dagml_execution_plan_build_json(
    const uint8_t *graph_ptr,
    size_t graph_len,
    const uint8_t *campaign_ptr,
    size_t campaign_len,
    const uint8_t *controllers_ptr,
    size_t controllers_len,
    DagMlBytesView plan_id,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
DagMlStatusCode dagml_execution_plan_schedule_json(
    const uint8_t *plan_ptr,
    size_t plan_len,
    DagMlBytesView phase,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
/* Execute one phase from a previously built ExecutionPlan through native
 * SequentialScheduler. The embedded plan manifests must exactly match the
 * trusted controller manifest list before any controller callback is invoked.
 * This JSON-returning entry point is intended for local binding callbacks and
 * conformance smokes. Controller vtables must be borrowed and must not expose
 * release/destroy callbacks, because no opaque result retains the registry;
 * long-lived native handles should use the opaque training/replay APIs that
 * retain controller registries. */
DagMlStatusCode dagml_execution_plan_execute_phase_json(
    const uint8_t *plan_ptr,
    size_t plan_len,
    const uint8_t *trusted_controllers_ptr,
    size_t trusted_controllers_len,
    DagMlBytesView run_id,
    uint64_t root_seed,
    DagMlBytesView phase,
    const DagMlControllerBinding *controller_bindings,
    size_t controller_binding_count,
    DagMlOwnedBytes *out_json,
    DagMlString *error_out);
DagMlStatusCode dagml_execution_plan_validate_json(
    const uint8_t *plan_ptr,
    size_t plan_len,
    DagMlString *error_out);
DagMlStatusCode dagml_selection_policy_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_selection_policy_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_selection_decision_contract_json(DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_selection_decision_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_select_candidate_json(const uint8_t *policy_ptr, size_t policy_len, const uint8_t *candidates_ptr, size_t candidates_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_select_candidate_groups_json(const uint8_t *policy_ptr, size_t policy_len, const uint8_t *candidates_ptr, size_t candidates_len, const uint8_t *groups_ptr, size_t groups_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_score_regression_prediction_block_json(const uint8_t *predictions_ptr, size_t predictions_len, const uint8_t *targets_ptr, size_t targets_len, const uint8_t *metrics_ptr, size_t metrics_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_score_regression_aggregated_block_json(const uint8_t *predictions_ptr, size_t predictions_len, const uint8_t *targets_ptr, size_t targets_len, const uint8_t *metrics_ptr, size_t metrics_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_prediction_block_f64_tensor_json(const uint8_t *predictions_ptr, size_t predictions_len, DagMlF64Tensor *out_tensor, DagMlString *error_out);
DagMlStatusCode dagml_aggregated_prediction_block_f64_tensor_json(const uint8_t *predictions_ptr, size_t predictions_len, DagMlF64Tensor *out_tensor, DagMlString *error_out);
DagMlStatusCode dagml_prediction_block_f32_tensor_json(const uint8_t *predictions_ptr, size_t predictions_len, DagMlF32Tensor *out_tensor, DagMlString *error_out);
DagMlStatusCode dagml_aggregated_prediction_block_f32_tensor_json(const uint8_t *predictions_ptr, size_t predictions_len, DagMlF32Tensor *out_tensor, DagMlString *error_out);
DagMlStatusCode dagml_regression_report_candidate_score_json(const uint8_t *report_ptr, size_t report_len, DagMlBytesView candidate_id, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_execution_bundle_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_execution_bundle_validate_replay_envelopes_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *envelopes_ptr, size_t envelopes_len, DagMlString *error_out);
DagMlStatusCode dagml_replay_request_validate_for_bundle_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, DagMlString *error_out);
DagMlStatusCode dagml_prediction_cache_payload_validate_for_bundle_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *payload_ptr, size_t payload_len, DagMlString *error_out);
DagMlStatusCode dagml_prediction_cache_payload_f64_tensor_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *payload_ptr, size_t payload_len, DagMlBytesView requirement_key, DagMlF64Tensor *out_tensor, DagMlOwnedBytes *out_metadata_json, DagMlString *error_out);
DagMlStatusCode dagml_prediction_cache_payload_f64_columnar_tensor_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *payload_ptr, size_t payload_len, DagMlBytesView requirement_key, DagMlF64ColumnarTensor *out_tensor, DagMlOwnedBytes *out_metadata_json, DagMlString *error_out);
DagMlStatusCode dagml_prediction_cache_payload_f32_tensor_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *payload_ptr, size_t payload_len, DagMlBytesView requirement_key, DagMlF32Tensor *out_tensor, DagMlOwnedBytes *out_metadata_json, DagMlString *error_out);
DagMlStatusCode dagml_prediction_cache_payload_f32_columnar_tensor_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *payload_ptr, size_t payload_len, DagMlBytesView requirement_key, DagMlF32ColumnarTensor *out_tensor, DagMlOwnedBytes *out_metadata_json, DagMlString *error_out);
DagMlStatusCode dagml_replay_request_validate_for_bundle_with_prediction_cache_payload_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, const uint8_t *payload_ptr, size_t payload_len, DagMlString *error_out);
DagMlStatusCode dagml_research_provenance_export_json(const uint8_t *plan_ptr, size_t plan_len, const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *lineage_ptr, size_t lineage_len, const uint8_t *envelopes_ptr, size_t envelopes_len, const uint8_t *prediction_cache_manifest_ptr, size_t prediction_cache_manifest_len, const uint8_t *artifact_manifest_ptr, size_t artifact_manifest_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_openlineage_run_event_json(const uint8_t *plan_ptr, size_t plan_len, const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *lineage_ptr, size_t lineage_len, const uint8_t *envelopes_ptr, size_t envelopes_len, const uint8_t *prediction_cache_manifest_ptr, size_t prediction_cache_manifest_len, const uint8_t *artifact_manifest_ptr, size_t artifact_manifest_len, DagMlBytesView namespace, DagMlBytesView event_time, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_mock_replay_execute_json(const uint8_t *plan_ptr, size_t plan_len, const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, const uint8_t *envelopes_ptr, size_t envelopes_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_replay_execute_json(const uint8_t *plan_ptr, size_t plan_len, const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, const uint8_t *envelopes_ptr, size_t envelopes_len, DagMlBytesView data_owner_controller_id, DagMlHandle dataset, DagMlDataVTable data_provider, DagMlArtifactStoreVTable artifact_store, const DagMlPredictionCacheVTable *prediction_cache_store, const DagMlControllerBinding *controller_bindings, size_t controller_binding_count, DagMlOwnedBytes *out_json, DagMlString *error_out);
/* Execute native COMPILE/PLAN -> FIT_CV -> SELECT -> optional REFIT. After the
 * required request and out_result pointers are accepted, once the
 * controller_bindings slice is readable (count 0, or a non-null pointer), the
 * call owns every DISTINCT owning controller user_data (owned-ABI vtable with a
 * non-null destroy) and consumes each exactly once: on OK they move into
 * *out_result (free with dagml_training_result_free); on any error *out_result
 * is NULL and each distinct owning user_data is destroyed exactly once,
 * including rejected/unreached bindings. Borrowed data/controller vtables are
 * never destroyed; if controller_bindings is NULL with a non-zero count the
 * caller keeps ownership. NULL request/out_result pointers are also rejected
 * before ownership transfer. Every training controller binding must provide a
 * non-null release callback. Strict-JSON/duplicate-key, envelope
 * coverage/collision and any user_data alias involving an owning vtable are
 * refused before any callback runs. */
DagMlStatusCode dagml_training_execute(const DagMlTrainingExecuteRequest *request, DagMlTrainingResult **out_result, DagMlString *error_out);
/* Serialize the outcome owned by result into fresh bytes (release with
 * dagml_owned_bytes_free). NULL result yields INVALID_ARGUMENT. */
DagMlStatusCode dagml_training_result_outcome_json(const DagMlTrainingResult *result, DagMlOwnedBytes *out_json, DagMlString *error_out);
/* Execute attached PREDICT/EXPLAIN replay from a live result into fresh
 * TrainingReplayOutcome JSON bytes (release with dagml_owned_bytes_free). NULL
 * result/request yields INVALID_ARGUMENT. */
DagMlStatusCode dagml_training_result_replay(const DagMlTrainingResult *result, const DagMlTrainingReplayRequest *request, DagMlOwnedBytes *out_json, DagMlString *error_out);
/* Release a DagMlTrainingResult. NULL is a no-op; free at most once. */
void dagml_training_result_free(DagMlTrainingResult *result);

#ifdef __cplusplus
}
#endif

#endif
