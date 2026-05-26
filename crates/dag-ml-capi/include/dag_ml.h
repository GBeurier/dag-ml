#ifndef DAG_ML_H
#define DAG_ML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t DagMlHandle;

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

typedef struct DagMlOwnedBytes {
    uint8_t *ptr;
    size_t len;
    size_t capacity;
} DagMlOwnedBytes;

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

typedef struct DagMlControllerVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*clone_with)(void *user_data, DagMlHandle op, DagMlBytesView params_json, DagMlHandle *out_op);
    DagMlStatusCode (*describe)(void *user_data, DagMlHandle op, DagMlOwnedBytes *out_json);
    DagMlStatusCode (*fit)(void *user_data, DagMlHandle op, DagMlHandle data, DagMlBytesView context_json, DagMlHandle *out_fitted);
    DagMlStatusCode (*predict)(void *user_data, DagMlHandle fitted, DagMlHandle data, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlControllerVTable;

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

typedef struct DagMlPredictionCacheVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*load_blocks)(void *user_data, DagMlBytesView requirement_key, DagMlOwnedBytes *out_json);
    DagMlStatusCode (*materialize)(void *user_data, DagMlBytesView request_json, DagMlHandle *out_handle);
    void (*release_bytes)(void *user_data, DagMlOwnedBytes bytes);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlPredictionCacheVTable;

DagMlVersion dagml_version(void);
void dagml_string_free(DagMlString value);
void dagml_owned_bytes_free(DagMlOwnedBytes value);
DagMlStatusCode dagml_graph_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_selection_policy_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_selection_decision_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_select_candidate_json(const uint8_t *policy_ptr, size_t policy_len, const uint8_t *candidates_ptr, size_t candidates_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_select_candidate_groups_json(const uint8_t *policy_ptr, size_t policy_len, const uint8_t *candidates_ptr, size_t candidates_len, const uint8_t *groups_ptr, size_t groups_len, DagMlOwnedBytes *out_json, DagMlString *error_out);
DagMlStatusCode dagml_execution_bundle_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);
DagMlStatusCode dagml_execution_bundle_validate_replay_envelopes_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *envelopes_ptr, size_t envelopes_len, DagMlString *error_out);
DagMlStatusCode dagml_replay_request_validate_for_bundle_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, DagMlString *error_out);
DagMlStatusCode dagml_prediction_cache_payload_validate_for_bundle_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *payload_ptr, size_t payload_len, DagMlString *error_out);
DagMlStatusCode dagml_replay_request_validate_for_bundle_with_prediction_cache_payload_json(const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, const uint8_t *payload_ptr, size_t payload_len, DagMlString *error_out);
DagMlStatusCode dagml_mock_replay_execute_json(const uint8_t *plan_ptr, size_t plan_len, const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, const uint8_t *envelopes_ptr, size_t envelopes_len, DagMlOwnedBytes *out_json, DagMlString *error_out);

#ifdef __cplusplus
}
#endif

#endif
