use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    char seed[64];
    if (!extract_string(json, task_json.len, "\"run_id\":\"", run_id, sizeof(run_id)) ||
        !extract_string(json, task_json.len, "\"node_id\":\"", node_id, sizeof(node_id)) ||
        !extract_string(json, task_json.len, "\"controller_id\":\"", controller_id, sizeof(controller_id)) ||
        !extract_string(json, task_json.len, "\"controller_version\":\"", controller_version, sizeof(controller_version)) ||
        !extract_string(json, task_json.len, "\"params_fingerprint\":\"", params_fingerprint, sizeof(params_fingerprint)) ||
        !extract_string(json, task_json.len, "\"phase\":\"", phase, sizeof(phase)) ||
        !extract_string(json, task_json.len, "\"variant_id\":\"", variant_id, sizeof(variant_id)) ||
        !extract_seed(json, task_json.len, seed, sizeof(seed))) {
        return DAG_ML_STATUS_VALIDATION_ERROR;
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
        "\"controller_version\":\"%s\",\"variant_id\":\"%s\",\"fold_id\":null,"
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
        variant_id,
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
        "\"controller_version\":\"%s\",\"variant_id\":\"%s\",\"fold_id\":null,"
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
        variant_id,
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
        fprintf(stderr, "usage: %s GRAPH CAMPAIGN CONTROLLERS BUNDLE REQUEST ENVELOPES\n", argv[0]);
        return 2;
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
    dagml_owned_bytes_free(out);
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
    let library_name = if cfg!(target_os = "macos") {
        "libdag_ml_capi.dylib"
    } else if cfg!(target_os = "windows") {
        "dag_ml_capi.dll"
    } else {
        "libdag_ml_capi.so"
    };
    for directory in [target_debug.join("deps"), target_debug.to_path_buf()] {
        let candidate = directory.join(library_name);
        if candidate.exists() {
            return candidate;
        }
    }
    panic!(
        "could not locate dynamic dag-ml C ABI library `{}` under {}",
        library_name,
        target_debug.display()
    );
}
