use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use dag_ml_core::{
    build_execution_bundle, build_execution_plan, ArtifactId, ArtifactRef, BundleId, CampaignSpec,
    ControllerId, ControllerManifest, ControllerRegistry, GraphSpec, NodeId, RefitArtifactRecord,
};
use std::collections::BTreeMap;

const C_CONFORMANCE_SOURCE: &str = r#"
#include "dag_ml.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Buffer {
    uint8_t *ptr;
    size_t len;
} Buffer;

static Buffer read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek %s\n", path);
        exit(2);
    }
    long size = ftell(file);
    if (size < 0) {
        fprintf(stderr, "failed to tell %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "failed to rewind %s\n", path);
        exit(2);
    }
    uint8_t *data = (uint8_t *)malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "allocation failure\n");
        exit(2);
    }
    if (fread(data, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "failed to read %s\n", path);
        exit(2);
    }
    fclose(file);
    Buffer buffer = { data, (size_t)size };
    return buffer;
}

static DagMlBytesView bytes_view(const char *text) {
    DagMlBytesView view = { (const uint8_t *)text, strlen(text) };
    return view;
}

static int contains_bytes(const uint8_t *haystack, size_t haystack_len, const char *needle) {
    size_t needle_len = strlen(needle);
    if (!haystack || needle_len == 0 || needle_len > haystack_len) {
        return 0;
    }
    for (size_t i = 0; i + needle_len <= haystack_len; i++) {
        if (memcmp(haystack + i, needle, needle_len) == 0) {
            return 1;
        }
    }
    return 0;
}

static int extract_string(
    const char *json,
    size_t len,
    const char *pattern,
    char *out,
    size_t out_len
) {
    const char *end = json + len;
    const char *start = strstr(json, pattern);
    if (!start) {
        return 0;
    }
    start += strlen(pattern);
    const char *cursor = start;
    while (cursor < end && *cursor != '"') {
        cursor++;
    }
    if (cursor >= end) {
        return 0;
    }
    size_t value_len = (size_t)(cursor - start);
    if (value_len + 1 > out_len) {
        return 0;
    }
    memcpy(out, start, value_len);
    out[value_len] = '\0';
    return 1;
}

static int extract_seed(const char *json, size_t len, char *out, size_t out_len) {
    const char *end = json + len;
    const char *start = NULL;
    const char *search = json;
    while (search < end) {
        const char *candidate = strstr(search, "\"seed\":");
        if (!candidate || candidate >= end) {
            break;
        }
        start = candidate;
        search = candidate + 1;
    }
    if (!start) {
        return 0;
    }
    start += strlen("\"seed\":");
    while (start < end && (*start == ' ' || *start == '\n' || *start == '\t')) {
        start++;
    }
    const char *cursor = start;
    while (cursor < end && *cursor != ',' && *cursor != '}') {
        cursor++;
    }
    size_t value_len = (size_t)(cursor - start);
    if (value_len + 1 > out_len) {
        return 0;
    }
    memcpy(out, start, value_len);
    out[value_len] = '\0';
    return 1;
}

static DagMlStatusCode data_materialize(
    void *user_data,
    DagMlHandle dataset,
    DagMlBytesView request_json,
    DagMlHandle *out_handle
) {
    (void)user_data;
    (void)request_json;
    if (!out_handle || dataset == 0) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = 41;
    return DAG_ML_STATUS_OK;
}

static DagMlStatusCode data_make_view(
    void *user_data,
    DagMlHandle data,
    DagMlBytesView selector_json,
    DagMlHandle *out_view
) {
    (void)user_data;
    (void)selector_json;
    if (!out_view || data == 0) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    *out_view = 42;
    return DAG_ML_STATUS_OK;
}

static unsigned data_release_count = 0;
static DagMlHandle data_released[8] = {0};

static void data_release(void *user_data, DagMlHandle handle) {
    (void)user_data;
    if (data_release_count < 8) {
        data_released[data_release_count] = handle;
    }
    data_release_count++;
}

static DagMlStatusCode artifact_materialize(
    void *user_data,
    DagMlBytesView request_json,
    DagMlHandleRef *out_handle
) {
    (void)user_data;
    (void)request_json;
    if (!out_handle) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    out_handle->handle = 700;
    out_handle->kind = DAG_ML_HANDLE_KIND_MODEL;
    return DAG_ML_STATUS_OK;
}

static DagMlStatusCode controller_invoke(
    void *user_data,
    DagMlBytesView task_json,
    DagMlOwnedBytes *out_result_json
) {
    (void)user_data;
    if (!task_json.ptr || !out_result_json) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    const char *json = (const char *)task_json.ptr;
    char run_id[128];
    char node_id[128];
    char controller_id[128];
    char controller_version[64];
    char params_fingerprint[128];
    char phase[32];
    char variant_id[128];
    char variant_json[160];
    char seed[64];
    if (!extract_string(json, task_json.len, "\"run_id\":\"", run_id, sizeof(run_id)) ||
        !extract_string(json, task_json.len, "\"node_id\":\"", node_id, sizeof(node_id)) ||
        !extract_string(json, task_json.len, "\"controller_id\":\"", controller_id, sizeof(controller_id)) ||
        !extract_string(json, task_json.len, "\"controller_version\":\"", controller_version, sizeof(controller_version)) ||
        !extract_string(json, task_json.len, "\"params_fingerprint\":\"", params_fingerprint, sizeof(params_fingerprint)) ||
        !extract_string(json, task_json.len, "\"phase\":\"", phase, sizeof(phase)) ||
        !extract_seed(json, task_json.len, seed, sizeof(seed))) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    if (extract_string(json, task_json.len, "\"variant_id\":\"", variant_id, sizeof(variant_id))) {
        snprintf(variant_json, sizeof(variant_json), "\"%s\"", variant_id);
    } else {
        strcpy(variant_id, "base");
        strcpy(variant_json, "null");
    }

    int is_model = strstr(node_id, "model:base") != NULL;
    const char *prediction_json = "";
    if (is_model) {
        prediction_json =
            ",\"predictions\":[{\"prediction_id\":\"prediction:c.conformance\","
            "\"producer_node\":\"model:base\",\"partition\":\"final\",\"fold_id\":null,"
            "\"sample_ids\":[\"sample:c.conformance\"],\"values\":[[0.7]],"
            "\"target_names\":[\"y\"]}]";
    }

    int needed = snprintf(
        NULL,
        0,
        "{\"node_id\":\"%s\",\"outputs\":{\"out\":{\"handle\":88,\"kind\":\"data\","
        "\"owner_controller\":\"%s\"}}%s,\"shape_deltas\":[],\"artifacts\":[],"
        "\"artifact_handles\":{},\"lineage\":{\"record_id\":\"lineage:c.conformance.%s\","
        "\"run_id\":\"%s\",\"node_id\":\"%s\",\"phase\":\"%s\",\"controller_id\":\"%s\","
        "\"controller_version\":\"%s\",\"variant_id\":%s,\"fold_id\":null,"
        "\"branch_path\":[],\"input_lineage\":[],\"artifact_refs\":[],"
        "\"params_fingerprint\":\"%s\",\"data_model_shape_fingerprint\":null,"
        "\"aggregation_policy_fingerprint\":null,\"seed\":%s,\"unsafe_flags\":[],"
        "\"metrics\":{}}}",
        node_id,
        controller_id,
        prediction_json,
        is_model ? "model" : "transform",
        run_id,
        node_id,
        phase,
        controller_id,
        controller_version,
        variant_json,
        params_fingerprint,
        seed
    );
    if (needed < 0) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    char *result = (char *)malloc((size_t)needed + 1);
    if (!result) {
        return DAG_ML_STATUS_PANIC;
    }
    snprintf(
        result,
        (size_t)needed + 1,
        "{\"node_id\":\"%s\",\"outputs\":{\"out\":{\"handle\":88,\"kind\":\"data\","
        "\"owner_controller\":\"%s\"}}%s,\"shape_deltas\":[],\"artifacts\":[],"
        "\"artifact_handles\":{},\"lineage\":{\"record_id\":\"lineage:c.conformance.%s\","
        "\"run_id\":\"%s\",\"node_id\":\"%s\",\"phase\":\"%s\",\"controller_id\":\"%s\","
        "\"controller_version\":\"%s\",\"variant_id\":%s,\"fold_id\":null,"
        "\"branch_path\":[],\"input_lineage\":[],\"artifact_refs\":[],"
        "\"params_fingerprint\":\"%s\",\"data_model_shape_fingerprint\":null,"
        "\"aggregation_policy_fingerprint\":null,\"seed\":%s,\"unsafe_flags\":[],"
        "\"metrics\":{}}}",
        node_id,
        controller_id,
        prediction_json,
        is_model ? "model" : "transform",
        run_id,
        node_id,
        phase,
        controller_id,
        controller_version,
        variant_json,
        params_fingerprint,
        seed
    );
    out_result_json->ptr = (uint8_t *)result;
    out_result_json->len = (size_t)needed;
    out_result_json->capacity = (size_t)needed + 1;
    return DAG_ML_STATUS_OK;
}

static void release_bytes(void *user_data, DagMlOwnedBytes bytes) {
    (void)user_data;
    free(bytes.ptr);
}

static int verify_prediction_tensor_exports(void) {
    const char *sample_predictions =
        "{\"prediction_id\":\"pred:c.sample\","
        "\"producer_node\":\"model:c\","
        "\"partition\":\"validation\","
        "\"fold_id\":\"fold:0\","
        "\"sample_ids\":[\"sample:1\",\"sample:2\"],"
        "\"values\":[[1.0,2.5],[3.0,4.5]],"
        "\"target_names\":[\"y1\",\"y2\"]}";
    DagMlF64Tensor sample_tensor = {0};
    DagMlString error = {0};
    DagMlStatusCode status = dagml_prediction_block_f64_tensor_json(
        (const uint8_t *)sample_predictions,
        strlen(sample_predictions),
        &sample_tensor,
        &error
    );
    if (status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "sample tensor export failed with status %u: %.*s\n",
            status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        if (error.ptr) {
            dagml_string_free(error);
        }
        return 0;
    }
    if (!sample_tensor.ptr || sample_tensor.rows != 2 || sample_tensor.cols != 2 ||
        sample_tensor.len != 4 || sample_tensor.capacity < sample_tensor.len ||
        sample_tensor.ptr[0] != 1.0 || sample_tensor.ptr[1] != 2.5 ||
        sample_tensor.ptr[2] != 3.0 || sample_tensor.ptr[3] != 4.5) {
        fprintf(stderr, "unexpected sample tensor shape or values\n");
        dagml_f64_tensor_free(sample_tensor);
        return 0;
    }
    dagml_f64_tensor_free(sample_tensor);

    const char *aggregated_predictions =
        "{\"prediction_id\":\"pred:c.target\","
        "\"producer_node\":\"model:c\","
        "\"partition\":\"validation\","
        "\"fold_id\":\"fold:0\","
        "\"level\":\"target\","
        "\"unit_ids\":["
        "{\"level\":\"target\",\"id\":\"target:1\"},"
        "{\"level\":\"target\",\"id\":\"target:2\"}],"
        "\"values\":[[9.0],[11.0]],"
        "\"target_names\":[\"y\"]}";
    DagMlF64Tensor aggregated_tensor = {0};
    status = dagml_aggregated_prediction_block_f64_tensor_json(
        (const uint8_t *)aggregated_predictions,
        strlen(aggregated_predictions),
        &aggregated_tensor,
        &error
    );
    if (status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "aggregated tensor export failed with status %u: %.*s\n",
            status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        if (error.ptr) {
            dagml_string_free(error);
        }
        return 0;
    }
    if (!aggregated_tensor.ptr || aggregated_tensor.rows != 2 ||
        aggregated_tensor.cols != 1 || aggregated_tensor.len != 2 ||
        aggregated_tensor.capacity < aggregated_tensor.len ||
        aggregated_tensor.ptr[0] != 9.0 || aggregated_tensor.ptr[1] != 11.0) {
        fprintf(stderr, "unexpected aggregated tensor shape or values\n");
        dagml_f64_tensor_free(aggregated_tensor);
        return 0;
    }
    dagml_f64_tensor_free(aggregated_tensor);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s GRAPH CAMPAIGN CONTROLLERS BUNDLE REQUEST ENVELOPES\n", argv[0]);
        return 2;
    }
    if (!verify_prediction_tensor_exports()) {
        return 1;
    }
    Buffer graph = read_file(argv[1]);
    Buffer campaign = read_file(argv[2]);
    Buffer controllers = read_file(argv[3]);
    Buffer bundle = read_file(argv[4]);
    Buffer request = read_file(argv[5]);
    Buffer envelopes = read_file(argv[6]);

    DagMlDataVTable data_provider = {0};
    data_provider.abi_version = DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION;
    data_provider.materialize = data_materialize;
    data_provider.make_view = data_make_view;
    data_provider.release = data_release;

    DagMlArtifactStoreVTable artifact_store = {0};
    artifact_store.abi_version = 1;
    artifact_store.materialize = artifact_materialize;

    DagMlControllerBinding bindings[2];
    memset(bindings, 0, sizeof(bindings));
    bindings[0].controller_id = bytes_view("controller:transform.mock");
    bindings[0].vtable.abi_version = 2;
    bindings[0].vtable.invoke = controller_invoke;
    bindings[0].vtable.release_bytes = release_bytes;
    bindings[1].controller_id = bytes_view("controller:model.mock");
    bindings[1].vtable.abi_version = 2;
    bindings[1].vtable.invoke = controller_invoke;
    bindings[1].vtable.release_bytes = release_bytes;

    DagMlOwnedBytes plan = {0};
    DagMlOwnedBytes out = {0};
    DagMlString error = {0};
    DagMlStatusCode plan_status = dagml_execution_plan_build_json(
        graph.ptr,
        graph.len,
        campaign.ptr,
        campaign.len,
        controllers.ptr,
        controllers.len,
        bytes_view("plan:cli.bundle"),
        &plan,
        &error
    );
    free(graph.ptr);
    free(campaign.ptr);
    free(controllers.ptr);
    if (plan_status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "dagml_execution_plan_build_json failed with status %u: %.*s\n",
            plan_status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        free(bundle.ptr);
        free(request.ptr);
        free(envelopes.ptr);
        if (error.ptr) {
            dagml_string_free(error);
        }
        return 1;
    }

    DagMlStatusCode status = dagml_replay_execute_json(
        plan.ptr,
        plan.len,
        bundle.ptr,
        bundle.len,
        request.ptr,
        request.len,
        envelopes.ptr,
        envelopes.len,
        bytes_view("controller:data.provider"),
        7,
        data_provider,
        artifact_store,
        NULL,
        bindings,
        2,
        &out,
        &error
    );
    free(bundle.ptr);
    free(request.ptr);
    free(envelopes.ptr);
    dagml_owned_bytes_free(plan);
    if (status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "dagml_replay_execute_json failed with status %u: %.*s\n",
            status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        if (error.ptr) {
            dagml_string_free(error);
        }
        return 1;
    }
    if (!out.ptr || !contains_bytes(out.ptr, out.len, "\"result_count\":2") ||
        !contains_bytes(out.ptr, out.len, "\"prediction_block_count\":1")) {
        fprintf(stderr, "unexpected replay summary: %.*s\n", (int)out.len, out.ptr ? (char *)out.ptr : "");
        if (out.ptr) {
            dagml_owned_bytes_free(out);
        }
        return 1;
    }
    if (data_release_count != 2 || data_released[0] != 42 || data_released[1] != 41) {
        fprintf(stderr, "unexpected data release lifecycle: count=%u first=%llu second=%llu\n",
            data_release_count,
            (unsigned long long)data_released[0],
            (unsigned long long)data_released[1]);
        dagml_owned_bytes_free(out);
        return 1;
    }
    dagml_owned_bytes_free(out);
    return 0;
}
"#;

const C_PREDICTION_CACHE_TENSOR_SOURCE: &str = r#"
#include "dag_ml.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Buffer {
    uint8_t *ptr;
    size_t len;
} Buffer;

static Buffer read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek %s\n", path);
        exit(2);
    }
    long size = ftell(file);
    if (size < 0) {
        fprintf(stderr, "failed to tell %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "failed to rewind %s\n", path);
        exit(2);
    }
    uint8_t *data = (uint8_t *)malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "allocation failure\n");
        exit(2);
    }
    if (fread(data, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "failed to read %s\n", path);
        exit(2);
    }
    fclose(file);
    Buffer buffer = { data, (size_t)size };
    return buffer;
}

static int contains_bytes(const uint8_t *haystack, size_t haystack_len, const char *needle) {
    size_t needle_len = strlen(needle);
    if (!haystack || needle_len == 0 || needle_len > haystack_len) {
        return 0;
    }
    for (size_t i = 0; i + needle_len <= haystack_len; i++) {
        if (memcmp(haystack + i, needle, needle_len) == 0) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s BUNDLE PAYLOAD REQUIREMENT_KEY\n", argv[0]);
        return 2;
    }
    Buffer bundle = read_file(argv[1]);
    Buffer payload = read_file(argv[2]);
    DagMlBytesView requirement_key = { (const uint8_t *)argv[3], strlen(argv[3]) };
    DagMlF64Tensor tensor = {0};
    DagMlOwnedBytes metadata = {0};
    DagMlString error = {0};
    DagMlStatusCode status = dagml_prediction_cache_payload_f64_tensor_json(
        bundle.ptr,
        bundle.len,
        payload.ptr,
        payload.len,
        requirement_key,
        &tensor,
        &metadata,
        &error
    );
    free(bundle.ptr);
    free(payload.ptr);
    if (status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "prediction-cache tensor export failed with status %u: %.*s\n",
            status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        if (error.ptr) {
            dagml_string_free(error);
        }
        return 1;
    }
    if (!tensor.ptr || tensor.rows != 4 || tensor.cols != 1 || tensor.len != 4 ||
        tensor.ptr[0] != 9931.0 || tensor.ptr[1] != 9931.0 ||
        tensor.ptr[2] != 9932.0 || tensor.ptr[3] != 9932.0) {
        fprintf(stderr, "unexpected prediction-cache tensor shape or values\n");
        dagml_f64_tensor_free(tensor);
        if (metadata.ptr) {
            dagml_owned_bytes_free(metadata);
        }
        return 1;
    }
    if (!metadata.ptr ||
        !contains_bytes(metadata.ptr, metadata.len, "\"schema_version\":1") ||
        !contains_bytes(metadata.ptr, metadata.len, "\"prediction_level\":\"sample\"") ||
        !contains_bytes(metadata.ptr, metadata.len, "\"row_offset\":2") ||
        !contains_bytes(metadata.ptr, metadata.len, "\"sample:4\"")) {
        fprintf(stderr, "unexpected prediction-cache tensor metadata: %.*s\n",
            (int)metadata.len,
            metadata.ptr ? (char *)metadata.ptr : "");
        dagml_f64_tensor_free(tensor);
        if (metadata.ptr) {
            dagml_owned_bytes_free(metadata);
        }
        return 1;
    }
    dagml_f64_tensor_free(tensor);
    dagml_owned_bytes_free(metadata);
    return 0;
}
"#;

const C_DAG_ML_DATA_PROVIDER_SOURCE: &str = r#"
#include "dag_ml.h"
#include "dag_ml_data.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Buffer {
    uint8_t *ptr;
    size_t len;
} Buffer;

typedef struct ProviderBridge {
    DagMlDataVTable inner;
    DagMlHandle last_data;
    DagMlHandle last_view;
    unsigned release_count;
    DagMlHandle released[8];
} ProviderBridge;

typedef struct ControllerState {
    ProviderBridge *provider;
} ControllerState;

static Buffer read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek %s\n", path);
        exit(2);
    }
    long size = ftell(file);
    if (size < 0) {
        fprintf(stderr, "failed to tell %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "failed to rewind %s\n", path);
        exit(2);
    }
    uint8_t *data = (uint8_t *)malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "allocation failure\n");
        exit(2);
    }
    if (fread(data, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "failed to read %s\n", path);
        exit(2);
    }
    fclose(file);
    Buffer buffer = { data, (size_t)size };
    return buffer;
}

static DagMlBytesView ml_bytes_view(const char *text) {
    DagMlBytesView view = { (const uint8_t *)text, strlen(text) };
    return view;
}

static DagMlDataBytesView data_bytes_view(const char *text) {
    DagMlDataBytesView view = { (const uint8_t *)text, strlen(text) };
    return view;
}

static int contains_bytes(const uint8_t *haystack, size_t haystack_len, const char *needle) {
    size_t needle_len = strlen(needle);
    if (!haystack || needle_len == 0 || needle_len > haystack_len) {
        return 0;
    }
    for (size_t i = 0; i + needle_len <= haystack_len; i++) {
        if (memcmp(haystack + i, needle, needle_len) == 0) {
            return 1;
        }
    }
    return 0;
}

static int extract_string(
    const char *json,
    size_t len,
    const char *pattern,
    char *out,
    size_t out_len
) {
    const char *end = json + len;
    const char *start = strstr(json, pattern);
    if (!start) {
        return 0;
    }
    start += strlen(pattern);
    const char *cursor = start;
    while (cursor < end && *cursor != '"') {
        cursor++;
    }
    if (cursor >= end) {
        return 0;
    }
    size_t value_len = (size_t)(cursor - start);
    if (value_len + 1 > out_len) {
        return 0;
    }
    memcpy(out, start, value_len);
    out[value_len] = '\0';
    return 1;
}

static int extract_seed(const char *json, size_t len, char *out, size_t out_len) {
    const char *end = json + len;
    const char *start = NULL;
    const char *search = json;
    while (search < end) {
        const char *candidate = strstr(search, "\"seed\":");
        if (!candidate || candidate >= end) {
            break;
        }
        start = candidate;
        search = candidate + 1;
    }
    if (!start) {
        return 0;
    }
    start += strlen("\"seed\":");
    while (start < end && (*start == ' ' || *start == '\n' || *start == '\t')) {
        start++;
    }
    const char *cursor = start;
    while (cursor < end && *cursor != ',' && *cursor != '}') {
        cursor++;
    }
    size_t value_len = (size_t)(cursor - start);
    if (value_len + 1 > out_len) {
        return 0;
    }
    memcpy(out, start, value_len);
    out[value_len] = '\0';
    return 1;
}

static double f64_value(ArrowArray *array, int64_t row) {
    const double *values = (const double *)array->buffers[1];
    return values[row];
}

static DagMlStatusCode bridge_materialize(
    void *user_data,
    DagMlHandle dataset,
    DagMlBytesView request_json,
    DagMlHandle *out_handle
) {
    ProviderBridge *bridge = (ProviderBridge *)user_data;
    if (!bridge || !bridge->inner.materialize || !out_handle) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    DagMlStatusCode status = bridge->inner.materialize(
        bridge->inner.user_data,
        dataset,
        request_json,
        out_handle
    );
    if (status == DAG_ML_STATUS_OK) {
        bridge->last_data = *out_handle;
    }
    return status;
}

static DagMlStatusCode bridge_make_view(
    void *user_data,
    DagMlHandle data,
    DagMlBytesView selector_json,
    DagMlHandle *out_view
) {
    ProviderBridge *bridge = (ProviderBridge *)user_data;
    if (!bridge || !bridge->inner.make_view || !out_view) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    DagMlStatusCode status = bridge->inner.make_view(
        bridge->inner.user_data,
        data,
        selector_json,
        out_view
    );
    if (status == DAG_ML_STATUS_OK) {
        bridge->last_view = *out_view;
    }
    return status;
}

static DagMlStatusCode bridge_feature_arrow(
    void *user_data,
    DagMlHandle view,
    DagMlBytesView feature_set_name,
    ArrowArray **out_arrow_array,
    ArrowSchema **out_arrow_schema
) {
    ProviderBridge *bridge = (ProviderBridge *)user_data;
    if (!bridge || !bridge->inner.feature_arrow) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    return bridge->inner.feature_arrow(
        bridge->inner.user_data,
        view,
        feature_set_name,
        out_arrow_array,
        out_arrow_schema
    );
}

static DagMlStatusCode bridge_target_arrow(
    void *user_data,
    DagMlHandle view,
    DagMlBytesView target_name,
    ArrowArray **out_arrow_array,
    ArrowSchema **out_arrow_schema
) {
    ProviderBridge *bridge = (ProviderBridge *)user_data;
    if (!bridge || !bridge->inner.target_arrow) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    return bridge->inner.target_arrow(
        bridge->inner.user_data,
        view,
        target_name,
        out_arrow_array,
        out_arrow_schema
    );
}

static void bridge_release(void *user_data, DagMlHandle handle) {
    ProviderBridge *bridge = (ProviderBridge *)user_data;
    if (!bridge) {
        return;
    }
    if (bridge->release_count < 8) {
        bridge->released[bridge->release_count] = handle;
    }
    bridge->release_count++;
    if (bridge->inner.release) {
        bridge->inner.release(bridge->inner.user_data, handle);
    }
}

static DagMlStatusCode artifact_materialize(
    void *user_data,
    DagMlBytesView request_json,
    DagMlHandleRef *out_handle
) {
    (void)user_data;
    (void)request_json;
    if (!out_handle) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    out_handle->handle = 700;
    out_handle->kind = DAG_ML_HANDLE_KIND_MODEL;
    return DAG_ML_STATUS_OK;
}

static DagMlStatusCode verify_provider_values(ProviderBridge *provider, double *out_prediction) {
    if (!provider || provider->last_view == 0) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    ArrowArray *feature_array = NULL;
    ArrowSchema *feature_schema = NULL;
    ArrowArray *target_array = NULL;
    ArrowSchema *target_schema = NULL;
    DagMlStatusCode status = bridge_feature_arrow(
        provider,
        provider->last_view,
        ml_bytes_view("x"),
        &feature_array,
        &feature_schema
    );
    if (status != DAG_ML_STATUS_OK || !feature_array || feature_array->length != 3 || feature_array->n_children != 4) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    ArrowArray **feature_children = feature_array->children;
    double f0_first = f64_value(feature_children[2], 0);
    double f1_last = f64_value(feature_children[3], feature_array->length - 1);

    status = bridge_target_arrow(
        provider,
        provider->last_view,
        ml_bytes_view("y"),
        &target_array,
        &target_schema
    );
    if (status != DAG_ML_STATUS_OK || !target_array || target_array->length != 2 || target_array->n_children != 3) {
        dagmldata_arrow_array_free(feature_array);
        dagmldata_arrow_schema_free(feature_schema);
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    ArrowArray **target_children = target_array->children;
    double y_first = f64_value(target_children[2], 0);
    dagmldata_arrow_array_free(feature_array);
    dagmldata_arrow_schema_free(feature_schema);
    dagmldata_arrow_array_free(target_array);
    dagmldata_arrow_schema_free(target_schema);

    if (f0_first != 1.0 || f1_last != 40.0 || y_first != 42.0) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    *out_prediction = f0_first + f1_last + y_first;
    return DAG_ML_STATUS_OK;
}

static DagMlStatusCode controller_invoke(
    void *user_data,
    DagMlBytesView task_json,
    DagMlOwnedBytes *out_result_json
) {
    ControllerState *state = (ControllerState *)user_data;
    if (!task_json.ptr || !out_result_json || !state) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    const char *json = (const char *)task_json.ptr;
    char run_id[128];
    char node_id[128];
    char controller_id[128];
    char controller_version[64];
    char params_fingerprint[128];
    char phase[32];
    char variant_id[128];
    char variant_json[160];
    char seed[64];
    if (!extract_string(json, task_json.len, "\"run_id\":\"", run_id, sizeof(run_id)) ||
        !extract_string(json, task_json.len, "\"node_id\":\"", node_id, sizeof(node_id)) ||
        !extract_string(json, task_json.len, "\"controller_id\":\"", controller_id, sizeof(controller_id)) ||
        !extract_string(json, task_json.len, "\"controller_version\":\"", controller_version, sizeof(controller_version)) ||
        !extract_string(json, task_json.len, "\"params_fingerprint\":\"", params_fingerprint, sizeof(params_fingerprint)) ||
        !extract_string(json, task_json.len, "\"phase\":\"", phase, sizeof(phase)) ||
        !extract_seed(json, task_json.len, seed, sizeof(seed))) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    if (extract_string(json, task_json.len, "\"variant_id\":\"", variant_id, sizeof(variant_id))) {
        snprintf(variant_json, sizeof(variant_json), "\"%s\"", variant_id);
    } else {
        strcpy(variant_id, "base");
        strcpy(variant_json, "null");
    }

    int is_model = strstr(node_id, "model:base") != NULL;
    double prediction_value = 0.7;
    if (is_model) {
        DagMlStatusCode provider_status = verify_provider_values(state->provider, &prediction_value);
        if (provider_status != DAG_ML_STATUS_OK) {
            return provider_status;
        }
    }

    char prediction_json[512] = {0};
    if (is_model) {
        snprintf(
            prediction_json,
            sizeof(prediction_json),
            ",\"predictions\":[{\"prediction_id\":\"prediction:dagml.data.provider\","
            "\"producer_node\":\"model:base\",\"partition\":\"final\",\"fold_id\":null,"
            "\"sample_ids\":[\"S001\"],\"values\":[[%.1f]],"
            "\"target_names\":[\"y\"]}]",
            prediction_value
        );
    }

    int needed = snprintf(
        NULL,
        0,
        "{\"node_id\":\"%s\",\"outputs\":{\"out\":{\"handle\":88,\"kind\":\"data\","
        "\"owner_controller\":\"%s\"}}%s,\"shape_deltas\":[],\"artifacts\":[],"
        "\"artifact_handles\":{},\"lineage\":{\"record_id\":\"lineage:dagml.data.%s\","
        "\"run_id\":\"%s\",\"node_id\":\"%s\",\"phase\":\"%s\",\"controller_id\":\"%s\","
        "\"controller_version\":\"%s\",\"variant_id\":%s,\"fold_id\":null,"
        "\"branch_path\":[],\"input_lineage\":[],\"artifact_refs\":[],"
        "\"params_fingerprint\":\"%s\",\"data_model_shape_fingerprint\":null,"
        "\"aggregation_policy_fingerprint\":null,\"seed\":%s,\"unsafe_flags\":[],"
        "\"metrics\":{}}}",
        node_id,
        controller_id,
        prediction_json,
        is_model ? "model" : "transform",
        run_id,
        node_id,
        phase,
        controller_id,
        controller_version,
        variant_json,
        params_fingerprint,
        seed
    );
    if (needed < 0) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
    }
    char *result = (char *)malloc((size_t)needed + 1);
    if (!result) {
        return DAG_ML_STATUS_PANIC;
    }
    snprintf(
        result,
        (size_t)needed + 1,
        "{\"node_id\":\"%s\",\"outputs\":{\"out\":{\"handle\":88,\"kind\":\"data\","
        "\"owner_controller\":\"%s\"}}%s,\"shape_deltas\":[],\"artifacts\":[],"
        "\"artifact_handles\":{},\"lineage\":{\"record_id\":\"lineage:dagml.data.%s\","
        "\"run_id\":\"%s\",\"node_id\":\"%s\",\"phase\":\"%s\",\"controller_id\":\"%s\","
        "\"controller_version\":\"%s\",\"variant_id\":%s,\"fold_id\":null,"
        "\"branch_path\":[],\"input_lineage\":[],\"artifact_refs\":[],"
        "\"params_fingerprint\":\"%s\",\"data_model_shape_fingerprint\":null,"
        "\"aggregation_policy_fingerprint\":null,\"seed\":%s,\"unsafe_flags\":[],"
        "\"metrics\":{}}}",
        node_id,
        controller_id,
        prediction_json,
        is_model ? "model" : "transform",
        run_id,
        node_id,
        phase,
        controller_id,
        controller_version,
        variant_json,
        params_fingerprint,
        seed
    );
    out_result_json->ptr = (uint8_t *)result;
    out_result_json->len = (size_t)needed;
    out_result_json->capacity = (size_t)needed + 1;
    return DAG_ML_STATUS_OK;
}

static void release_bytes(void *user_data, DagMlOwnedBytes bytes) {
    (void)user_data;
    free(bytes.ptr);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s GRAPH CAMPAIGN CONTROLLERS BUNDLE REQUEST ENVELOPE\n", argv[0]);
        return 2;
    }
    Buffer graph = read_file(argv[1]);
    Buffer campaign = read_file(argv[2]);
    Buffer controllers = read_file(argv[3]);
    Buffer bundle = read_file(argv[4]);
    Buffer request = read_file(argv[5]);
    Buffer envelope = read_file(argv[6]);

    const uint8_t target_tables[] =
        "[{\"target_id\":\"y\",\"values\":[{\"sample_id\":\"S001\",\"value\":42.0},"
        "{\"sample_id\":\"S002\",\"value\":7.0}]}]";
    const DagMlDataBytesView feature_names[] = {
        {(const uint8_t *)"f0", sizeof("f0") - 1},
        {(const uint8_t *)"f1", sizeof("f1") - 1},
    };
    const DagMlDataBytesView observation_ids[] = {
        {(const uint8_t *)"obs.S001.aug0", sizeof("obs.S001.aug0") - 1},
        {(const uint8_t *)"obs.S001.base", sizeof("obs.S001.base") - 1},
        {(const uint8_t *)"obs.S001.rep1", sizeof("obs.S001.rep1") - 1},
        {(const uint8_t *)"obs.S002.base", sizeof("obs.S002.base") - 1},
    };
    const double feature_values[] = {3.0, 30.0, 1.0, 10.0, 2.0, 20.0, 4.0, 40.0};
    const DagMlDataFeatureMatrixF64View feature_matrices[] = {{
        {(const uint8_t *)"x", sizeof("x") - 1},
        {(const uint8_t *)"tabular_numeric", sizeof("tabular_numeric") - 1},
        feature_names,
        2,
        observation_ids,
        4,
        feature_values,
        8,
        NULL,
        0,
    }};

    DagMlDataVTable provider = {0};
    DagMlDataString data_error = {0};
    DagMlDataStatusCode provider_status = dagmldata_inmemory_provider_new_with_f64_feature_views(
        envelope.ptr,
        envelope.len,
        target_tables,
        sizeof(target_tables) - 1,
        feature_matrices,
        1,
        &provider,
        &data_error
    );
    if (provider_status != DAG_ML_DATA_STATUS_OK) {
        fprintf(stderr, "failed to create dag-ml-data provider: %.*s\n",
            (int)data_error.len,
            data_error.ptr ? data_error.ptr : "");
        if (data_error.ptr) {
            dagmldata_string_free(data_error);
        }
        return 1;
    }

    ProviderBridge bridge = {0};
    bridge.inner = provider;
    DagMlDataVTable bridged_provider = provider;
    bridged_provider.user_data = &bridge;
    bridged_provider.materialize = bridge_materialize;
    bridged_provider.make_view = bridge_make_view;
    bridged_provider.target_arrow = bridge_target_arrow;
    bridged_provider.feature_arrow = bridge_feature_arrow;
    bridged_provider.release = bridge_release;
    bridged_provider.destroy = NULL;

    DagMlArtifactStoreVTable artifact_store = {0};
    artifact_store.abi_version = 1;
    artifact_store.materialize = artifact_materialize;

    ControllerState controller_state = { &bridge };
    DagMlControllerBinding bindings[2];
    memset(bindings, 0, sizeof(bindings));
    bindings[0].controller_id = ml_bytes_view("controller:transform.mock");
    bindings[0].vtable.abi_version = 2;
    bindings[0].vtable.user_data = &controller_state;
    bindings[0].vtable.invoke = controller_invoke;
    bindings[0].vtable.release_bytes = release_bytes;
    bindings[1].controller_id = ml_bytes_view("controller:model.mock");
    bindings[1].vtable.abi_version = 2;
    bindings[1].vtable.user_data = &controller_state;
    bindings[1].vtable.invoke = controller_invoke;
    bindings[1].vtable.release_bytes = release_bytes;

    DagMlOwnedBytes plan = {0};
    DagMlOwnedBytes out = {0};
    DagMlString error = {0};
    DagMlStatusCode plan_status = dagml_execution_plan_build_json(
        graph.ptr,
        graph.len,
        campaign.ptr,
        campaign.len,
        controllers.ptr,
        controllers.len,
        ml_bytes_view("plan:dagml.data.provider.smoke"),
        &plan,
        &error
    );
    free(graph.ptr);
    free(campaign.ptr);
    free(controllers.ptr);
    if (plan_status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "dagml_execution_plan_build_json failed with status %u: %.*s\n",
            plan_status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        if (error.ptr) {
            dagml_string_free(error);
        }
        return 1;
    }

    size_t replay_envelopes_len = strlen("{\"model:base.x\":}") + envelope.len;
    char *replay_envelopes = (char *)malloc(replay_envelopes_len + 1);
    if (!replay_envelopes) {
        return 1;
    }
    memcpy(replay_envelopes, "{\"model:base.x\":", strlen("{\"model:base.x\":"));
    memcpy(replay_envelopes + strlen("{\"model:base.x\":"), envelope.ptr, envelope.len);
    replay_envelopes[replay_envelopes_len - 1] = '}';
    replay_envelopes[replay_envelopes_len] = '\0';

    DagMlStatusCode status = dagml_replay_execute_json(
        plan.ptr,
        plan.len,
        bundle.ptr,
        bundle.len,
        request.ptr,
        request.len,
        (const uint8_t *)replay_envelopes,
        replay_envelopes_len,
        ml_bytes_view("controller:data.provider"),
        7,
        bridged_provider,
        artifact_store,
        NULL,
        bindings,
        2,
        &out,
        &error
    );
    free(bundle.ptr);
    free(request.ptr);
    free(replay_envelopes);
    dagml_owned_bytes_free(plan);
    if (status != DAG_ML_STATUS_OK) {
        fprintf(stderr, "dagml_replay_execute_json failed with status %u: %.*s\n",
            status,
            (int)error.len,
            error.ptr ? error.ptr : "");
        if (error.ptr) {
            dagml_string_free(error);
        }
        dagmldata_inmemory_provider_destroy(&provider);
        free(envelope.ptr);
        return 1;
    }
    if (!out.ptr || !contains_bytes(out.ptr, out.len, "\"result_count\":2") ||
        !contains_bytes(out.ptr, out.len, "\"prediction_block_count\":1")) {
        fprintf(stderr, "unexpected replay summary: %.*s\n", (int)out.len, out.ptr ? (char *)out.ptr : "");
        if (out.ptr) {
            dagml_owned_bytes_free(out);
        }
        dagmldata_inmemory_provider_destroy(&provider);
        free(envelope.ptr);
        return 1;
    }
    if (bridge.release_count != 2 || bridge.released[0] != bridge.last_view || bridge.released[1] != bridge.last_data) {
        fprintf(stderr, "unexpected bridged provider release lifecycle: count=%u view=%llu data=%llu first=%llu second=%llu\n",
            bridge.release_count,
            (unsigned long long)bridge.last_view,
            (unsigned long long)bridge.last_data,
            (unsigned long long)bridge.released[0],
            (unsigned long long)bridge.released[1]);
        dagml_owned_bytes_free(out);
        dagmldata_inmemory_provider_destroy(&provider);
        free(envelope.ptr);
        return 1;
    }
    dagml_owned_bytes_free(out);
    dagmldata_inmemory_provider_destroy(&provider);
    free(envelope.ptr);
    return 0;
}
"#;

#[test]
fn c_headers_can_be_included_with_dag_ml_data_peer() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate lives under workspace/crates")
        .to_path_buf();
    let Some(peer) = dag_ml_data_peer_root(&workspace) else {
        eprintln!("skipping cross-header smoke; dag-ml-data checkout not found");
        return;
    };
    let peer_include = peer.join("crates/dag-ml-data-capi/include");
    assert!(
        peer_include.exists(),
        "dag-ml-data include directory not found at {}",
        peer_include.display()
    );

    let sources = [
        (
            "dag_ml_first",
            r#"
#include "dag_ml.h"
#include "dag_ml_data.h"

int main(void) {
#if DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION != 2u
#error unexpected data-provider ABI version
#endif
    DagMlDataVTable table = {0};
    DagMlDataTensorF64 tensor = {0};
    table.abi_version = DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION;
    tensor.abi_version = DAG_ML_DATA_TENSOR_F64_ABI_VERSION;
    return table.abi_version == 0 || tensor.abi_version == 0;
}
"#,
        ),
        (
            "dag_ml_data_first",
            r#"
#include "dag_ml_data.h"
#include "dag_ml.h"

int main(void) {
#if DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION != 2u
#error unexpected data-provider ABI version
#endif
    DagMlDataVTable table = {0};
    DagMlDataTensorF64 tensor = {0};
    table.abi_version = DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION;
    tensor.abi_version = DAG_ML_DATA_TENSOR_F64_ABI_VERSION;
    return table.abi_version == 0 || tensor.abi_version == 0;
}
"#,
        ),
    ];

    for (name, source) in sources {
        let path = std::env::temp_dir().join(format!(
            "dag_ml_cross_header_{name}_{}_{}.c",
            std::process::id(),
            unique_suffix()
        ));
        fs::write(&path, source).expect("write cross-header smoke source");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let output = Command::new(cc)
            .arg("-std=c11")
            .arg("-fsyntax-only")
            .arg("-I")
            .arg(manifest_dir.join("include"))
            .arg("-I")
            .arg(&peer_include)
            .arg(&path)
            .output()
            .expect("run C compiler");
        let _ = fs::remove_file(&path);
        assert!(
            output.status.success(),
            "cross-header smoke `{name}` failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn c_program_executes_vtable_replay_against_c_abi() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate lives under workspace/crates")
        .to_path_buf();
    let target_debug = std::env::current_exe()
        .expect("current test exe path")
        .parent()
        .and_then(Path::parent)
        .expect("test exe lives under target/debug/deps")
        .to_path_buf();
    let dynamic_lib = find_dynamic_library(&target_debug);
    let dynamic_lib_dir = dynamic_lib
        .parent()
        .expect("dynamic library has parent directory")
        .to_path_buf();
    let temp = std::env::temp_dir().join(format!(
        "dag_ml_c_conformance_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    fs::create_dir_all(&temp).expect("create conformance temp dir");

    let c_path = temp.join("conformance.c");
    let envelopes_path = temp.join("envelopes.json");
    let exe_path = temp.join("conformance");
    fs::write(&c_path, C_CONFORMANCE_SOURCE).expect("write C conformance source");
    let envelope = fs::read_to_string(
        workspace.join("examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"),
    )
    .expect("read data envelope fixture");
    fs::write(&envelopes_path, format!(r#"{{"model:base.x":{envelope}}}"#))
        .expect("write replay envelopes fixture");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut compile = Command::new(cc);
    compile
        .arg("-std=c11")
        .arg(&c_path)
        .arg("-I")
        .arg(manifest_dir.join("include"))
        .arg(&dynamic_lib)
        .arg("-o")
        .arg(&exe_path);
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        compile.arg(format!("-Wl,-rpath,{}", dynamic_lib_dir.display()));
    }
    let compile_output = compile.output().expect("run C compiler");
    assert!(
        compile_output.status.success(),
        "C conformance compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let run_output = Command::new(&exe_path)
        .arg(workspace.join("examples/minimal_graph.json"))
        .arg(workspace.join("examples/campaign_oof_generation.json"))
        .arg(workspace.join("examples/controller_manifests.json"))
        .arg(workspace.join("examples/generated/execution_bundle_minimal.json"))
        .arg(workspace.join("examples/fixtures/bundle/replay_request_predict.json"))
        .arg(&envelopes_path)
        .output()
        .expect("run C conformance executable");
    assert!(
        run_output.status.success(),
        "C conformance executable failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );
}

#[test]
fn c_program_exports_prediction_cache_payload_tensor_against_c_abi() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate lives under workspace/crates")
        .to_path_buf();
    let target_debug = std::env::current_exe()
        .expect("current test exe path")
        .parent()
        .and_then(Path::parent)
        .expect("test exe lives under target/debug/deps")
        .to_path_buf();
    let dynamic_lib = find_dynamic_library(&target_debug);
    let dynamic_lib_dir = dynamic_lib
        .parent()
        .expect("dynamic library has parent directory")
        .to_path_buf();
    let temp = std::env::temp_dir().join(format!(
        "dag_ml_cache_tensor_conformance_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    fs::create_dir_all(&temp).expect("create prediction-cache tensor conformance temp dir");

    let c_path = temp.join("prediction_cache_tensor.c");
    let exe_path = temp.join("prediction_cache_tensor");
    fs::write(&c_path, C_PREDICTION_CACHE_TENSOR_SOURCE)
        .expect("write prediction-cache tensor C conformance source");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut compile = Command::new(cc);
    compile
        .arg("-std=c11")
        .arg(&c_path)
        .arg("-I")
        .arg(manifest_dir.join("include"))
        .arg(&dynamic_lib)
        .arg("-o")
        .arg(&exe_path);
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        compile.arg(format!("-Wl,-rpath,{}", dynamic_lib_dir.display()));
    }
    let compile_output = compile.output().expect("run C compiler");
    assert!(
        compile_output.status.success(),
        "prediction-cache tensor C conformance compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let run_output = Command::new(&exe_path)
        .arg(workspace.join("examples/generated/execution_bundle_branch_merge_cv_refit.json"))
        .arg(workspace.join("examples/generated/prediction_cache_branch_merge_cv_refit.json"))
        .arg("branch:b0.model:ridge.oof->merge:stack.pred_plus_original.meta:ridge.b0_oof")
        .output()
        .expect("run prediction-cache tensor C conformance executable");
    assert!(
        run_output.status.success(),
        "prediction-cache tensor C conformance executable failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );
}

#[test]
fn c_program_executes_replay_with_dag_ml_data_f64_provider() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate lives under workspace/crates")
        .to_path_buf();
    let Some(peer) = dag_ml_data_peer_root(&workspace) else {
        eprintln!("skipping dag-ml-data provider conformance; sibling checkout not found");
        return;
    };
    let peer_include = peer.join("crates/dag-ml-data-capi/include");
    assert!(
        peer_include.exists(),
        "dag-ml-data include directory not found at {}",
        peer_include.display()
    );
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let build = Command::new(cargo)
        .arg("build")
        .arg("-p")
        .arg("dag-ml-data-capi")
        .arg("--lib")
        .current_dir(&peer)
        .output()
        .expect("run cargo build for dag-ml-data cdylib");
    assert!(
        build.status.success(),
        "dag-ml-data cargo build failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );

    let target_debug = std::env::current_exe()
        .expect("current test exe path")
        .parent()
        .and_then(Path::parent)
        .expect("test exe lives under target/debug/deps")
        .to_path_buf();
    let dag_ml_dynamic_lib = find_dynamic_library(&target_debug);
    let dag_ml_dynamic_lib_dir = dag_ml_dynamic_lib
        .parent()
        .expect("dynamic library has parent directory")
        .to_path_buf();
    let peer_target_debug = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| peer.join("target"))
        .join("debug");
    let dag_ml_data_dynamic_lib =
        find_named_dynamic_library(&peer_target_debug, "dag_ml_data_capi");
    let dag_ml_data_dynamic_lib_dir = dag_ml_data_dynamic_lib
        .parent()
        .expect("dag-ml-data dynamic library has parent directory")
        .to_path_buf();

    let temp = std::env::temp_dir().join(format!(
        "dag_ml_data_provider_c_conformance_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    fs::create_dir_all(&temp).expect("create dag-ml-data provider temp dir");
    let c_path = temp.join("dag_ml_data_provider_conformance.c");
    let bundle_path = temp.join("execution_bundle_dag_ml_data_provider.json");
    let exe_path = temp.join("dag_ml_data_provider_conformance");
    fs::write(&c_path, C_DAG_ML_DATA_PROVIDER_SOURCE)
        .expect("write dag-ml-data provider conformance source");
    write_dag_ml_data_provider_bundle(&workspace, &bundle_path);

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut compile = Command::new(cc);
    compile
        .arg("-std=c11")
        .arg(&c_path)
        .arg("-I")
        .arg(manifest_dir.join("include"))
        .arg("-I")
        .arg(&peer_include)
        .arg(&dag_ml_dynamic_lib)
        .arg(&dag_ml_data_dynamic_lib)
        .arg("-o")
        .arg(&exe_path);
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        compile
            .arg(format!("-Wl,-rpath,{}", dag_ml_dynamic_lib_dir.display()))
            .arg(format!(
                "-Wl,-rpath,{}",
                dag_ml_data_dynamic_lib_dir.display()
            ));
    }
    let compile_output = compile.output().expect("run C compiler");
    assert!(
        compile_output.status.success(),
        "dag-ml-data provider C conformance compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let run_output = Command::new(&exe_path)
        .arg(workspace.join("examples/minimal_graph.json"))
        .arg(workspace.join("examples/campaign_data_contract_nir_s001.json"))
        .arg(workspace.join("examples/controller_manifests.json"))
        .arg(&bundle_path)
        .arg(workspace.join("examples/fixtures/bundle/replay_request_predict.json"))
        .arg(workspace.join("examples/fixtures/data/coordinator_data_plan_envelope_nir.json"))
        .output()
        .expect("run dag-ml-data provider C conformance executable");
    let _ = fs::remove_dir_all(&temp);
    assert!(
        run_output.status.success(),
        "dag-ml-data provider C conformance executable failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );
}

fn write_dag_ml_data_provider_bundle(workspace: &Path, path: &Path) {
    let graph: GraphSpec = serde_json::from_str(
        &fs::read_to_string(workspace.join("examples/minimal_graph.json"))
            .expect("read minimal graph fixture"),
    )
    .expect("parse minimal graph fixture");
    let campaign: CampaignSpec = serde_json::from_str(
        &fs::read_to_string(workspace.join("examples/campaign_data_contract_nir_s001.json"))
            .expect("read S001 campaign fixture"),
    )
    .expect("parse S001 campaign fixture");
    let manifests: Vec<ControllerManifest> = serde_json::from_str(
        &fs::read_to_string(workspace.join("examples/controller_manifests.json"))
            .expect("read controller manifests fixture"),
    )
    .expect("parse controller manifests fixture");
    let mut controllers = ControllerRegistry::new();
    for manifest in manifests {
        controllers
            .register(manifest)
            .expect("register controller manifest");
    }
    let plan = build_execution_plan(
        "plan:dagml.data.provider.smoke",
        graph,
        campaign,
        &controllers,
    )
    .expect("build S001 data-provider execution plan");
    let model_node = NodeId::new("model:base").expect("valid model node id");
    let model_plan = plan
        .node_plans
        .get(&model_node)
        .expect("model node exists in plan");
    let controller_id = ControllerId::new("controller:model.mock").expect("valid controller id");
    let artifact = RefitArtifactRecord {
        node_id: model_node,
        controller_id: controller_id.clone(),
        artifact: ArtifactRef {
            id: ArtifactId::new("artifact:model:base:refit").expect("valid artifact id"),
            kind: "mock_model".to_string(),
            controller_id,
            backend: None,
            uri: None,
            content_fingerprint: None,
            size_bytes: Some(128),
            plugin: None,
            plugin_version: None,
        },
        params_fingerprint: model_plan.params_fingerprint.clone(),
        data_requirement_keys: vec!["model:base.x".to_string()],
        prediction_requirement_keys: Vec::new(),
    };
    let bundle = build_execution_bundle(
        BundleId::new("bundle:cli.demo").expect("valid bundle id"),
        &plan,
        None,
        BTreeMap::new(),
        vec![artifact],
    )
    .expect("build S001 data-provider execution bundle");
    fs::write(
        path,
        serde_json::to_vec_pretty(&bundle).expect("serialize S001 data-provider bundle"),
    )
    .expect("write S001 data-provider bundle");
}

fn dag_ml_data_peer_root(workspace: &Path) -> Option<PathBuf> {
    std::env::var_os("DAG_ML_DATA_REPO")
        .map(PathBuf::from)
        .into_iter()
        .chain([
            workspace.parent()?.join("dag-ml-data"),
            workspace.join("external/dag-ml-data"),
        ])
        .find(|candidate| candidate.exists())
        .map(|candidate| candidate.canonicalize().unwrap_or(candidate))
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time after unix epoch")
        .as_nanos()
}

fn find_dynamic_library(target_debug: &Path) -> PathBuf {
    find_named_dynamic_library(target_debug, "dag_ml_capi")
}

fn find_named_dynamic_library(target_debug: &Path, crate_name: &str) -> PathBuf {
    let library_name = if cfg!(target_os = "macos") {
        format!("lib{crate_name}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{crate_name}.dll")
    } else {
        format!("lib{crate_name}.so")
    };
    for directory in [target_debug.join("deps"), target_debug.to_path_buf()] {
        let candidate = directory.join(&library_name);
        if candidate.exists() {
            return candidate;
        }
    }
    panic!(
        "could not locate dynamic C ABI library `{}` under {}",
        library_name,
        target_debug.display()
    );
}
