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
DagMlStatusCode dagml_graph_validate_json(const uint8_t *json_ptr, size_t json_len, DagMlString *error_out);

#ifdef __cplusplus
}
#endif

#endif
