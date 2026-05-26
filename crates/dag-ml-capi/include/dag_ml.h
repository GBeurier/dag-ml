#ifndef DAG_ML_H
#define DAG_ML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t DagMlHandle;

typedef enum DagMlStatusCode {
    DAG_ML_STATUS_OK = 0,
    DAG_ML_STATUS_INVALID_ARGUMENT = 1,
    DAG_ML_STATUS_VALIDATION_ERROR = 2,
    DAG_ML_STATUS_PANIC = 255
} DagMlStatusCode;

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

typedef struct DagMlControllerVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*clone_with)(void *user_data, DagMlHandle op, DagMlBytesView params_json, DagMlHandle *out_op);
    DagMlStatusCode (*describe)(void *user_data, DagMlHandle op, DagMlOwnedBytes *out_json);
    DagMlStatusCode (*fit)(void *user_data, DagMlHandle op, DagMlHandle data, DagMlBytesView context_json, DagMlHandle *out_fitted);
    DagMlStatusCode (*predict)(void *user_data, DagMlHandle fitted, DagMlHandle data, void **out_arrow_array, void **out_arrow_schema);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlControllerVTable;

typedef struct DagMlDataVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlStatusCode (*make_view)(void *user_data, DagMlHandle data, DagMlBytesView sample_ids_json, DagMlHandle *out_view);
    DagMlStatusCode (*view_identity)(void *user_data, DagMlHandle view, void **out_arrow_array, void **out_arrow_schema);
    DagMlStatusCode (*target_arrow)(void *user_data, DagMlHandle view, DagMlBytesView target_name, void **out_arrow_array, void **out_arrow_schema);
    void (*release)(void *user_data, DagMlHandle handle);
    void (*destroy)(void *user_data);
} DagMlDataVTable;

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
DagMlStatusCode dagml_mock_replay_execute_json(const uint8_t *plan_ptr, size_t plan_len, const uint8_t *bundle_ptr, size_t bundle_len, const uint8_t *request_ptr, size_t request_len, const uint8_t *envelopes_ptr, size_t envelopes_len, DagMlOwnedBytes *out_json, DagMlString *error_out);

#ifdef __cplusplus
}
#endif

#endif
