use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const C_LOCAL_IMPLEMENTATION_SOURCE: &str = r#"
#include "dag_ml.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Buffer {
    uint8_t *ptr;
    size_t len;
} Buffer;

typedef struct CallbackState {
    unsigned calls;
    unsigned retains;
    unsigned releases;
    unsigned byte_releases;
    int fail;
    const char *result_json;
} CallbackState;

static Buffer read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(2);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        exit(2);
    }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        exit(2);
    }
    uint8_t *data = (uint8_t *)malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) {
        exit(2);
    }
    fclose(file);
    Buffer buffer = { data, (size_t)size };
    return buffer;
}

static DagMlBytesView buffer_view(Buffer buffer) {
    DagMlBytesView view = { buffer.ptr, buffer.len };
    return view;
}

static DagMlBytesView text_view(const char *text) {
    DagMlBytesView view = { (const uint8_t *)text, strlen(text) };
    return view;
}

static int contains_bytes(const uint8_t *data, size_t len, const char *needle) {
    size_t needle_len = strlen(needle);
    if (!data || needle_len > len) {
        return 0;
    }
    for (size_t index = 0; index + needle_len <= len; index++) {
        if (memcmp(data + index, needle, needle_len) == 0) {
            return 1;
        }
    }
    return 0;
}

static void callback_retain(void *user_data) {
    CallbackState *state = (CallbackState *)user_data;
    state->retains += 1;
}

static void callback_release(void *user_data) {
    CallbackState *state = (CallbackState *)user_data;
    state->releases += 1;
}

static void callback_release_bytes(void *user_data, DagMlOwnedBytes bytes) {
    CallbackState *state = (CallbackState *)user_data;
    if (state) {
        state->byte_releases += 1;
    }
    free(bytes.ptr);
}

static DagMlStatusCode callback_invoke(
    void *user_data,
    DagMlBytesView request_json,
    DagMlOwnedBytes *out_result_json
) {
    if (!user_data || !out_result_json || !request_json.ptr) {
        return DAG_ML_STATUS_INVALID_ARGUMENT;
    }
    CallbackState *state = (CallbackState *)user_data;
    state->calls += 1;
    const char *result = state->fail
        ? "{\"error\":\"local failure\"}"
        : state->result_json;
    size_t len = strlen(result);
    uint8_t *data = (uint8_t *)malloc(len);
    if (!data) {
        return DAG_ML_STATUS_PANIC;
    }
    memcpy(data, result, len);
    out_result_json->ptr = data;
    out_result_json->len = len;
    out_result_json->capacity = len;
    return state->fail ? DAG_ML_STATUS_VALIDATION_ERROR : DAG_ML_STATUS_OK;
}

static DagMlLocalImplementationVTable callback_vtable(CallbackState *state) {
    DagMlLocalImplementationVTable vtable;
    memset(&vtable, 0, sizeof(vtable));
    vtable.abi_version = DAG_ML_LOCAL_IMPLEMENTATION_VTABLE_ABI_VERSION;
    vtable.user_data = state;
    vtable.invoke = callback_invoke;
    vtable.release_bytes = callback_release_bytes;
    vtable.retain = callback_retain;
    vtable.release = callback_release;
    return vtable;
}

static void free_error(DagMlString error) {
    if (error.ptr) {
        dagml_string_free(error);
    }
}

static int expect_ok(DagMlStatusCode status, DagMlString error, const char *label) {
    if (status == DAG_ML_STATUS_OK) {
        return 1;
    }
    fprintf(stderr, "%s failed: %.*s\n", label, (int)error.len, error.ptr ? error.ptr : "");
    free_error(error);
    return 0;
}

static int expect_task_loss_rejected(
    DagMlLocalImplementationRegistry *registry,
    DagMlBytesView task,
    size_t role_index,
    DagMlBytesView request,
    const CallbackState *state,
    unsigned expected_callbacks,
    const char *label
) {
    DagMlOwnedBytes result = {0};
    DagMlOwnedBytes attestation = {0};
    DagMlString error = {0};
    DagMlStatusCode status =
        dagml_local_implementation_registry_invoke_task_training_loss(
            registry,
            task,
            role_index,
            request,
            &result,
            &attestation,
            &error);
    int rejected = status != DAG_ML_STATUS_OK &&
        !result.ptr && !attestation.ptr && error.ptr &&
        state->calls == expected_callbacks &&
        state->byte_releases == expected_callbacks;
    if (!rejected) {
        fprintf(stderr, "%s reached the callback or retained output bytes\n", label);
    }
    if (result.ptr) {
        dagml_owned_bytes_free(result);
    }
    if (attestation.ptr) {
        dagml_owned_bytes_free(attestation);
    }
    free_error(error);
    return rejected;
}

static int expect_task_binding_rejected(
    DagMlBytesView task,
    size_t role_index,
    const char *label
) {
    DagMlOwnedBytes role = {0};
    DagMlOwnedBytes attestation = {0};
    DagMlString error = {0};
    DagMlStatusCode status = dagml_node_task_training_loss_binding(
        task, role_index, &role, &attestation, &error);
    int rejected = status != DAG_ML_STATUS_OK &&
        !role.ptr && !attestation.ptr && error.ptr;
    if (!rejected) {
        fprintf(stderr, "%s retained binding output bytes\n", label);
    }
    if (role.ptr) {
        dagml_owned_bytes_free(role);
    }
    if (attestation.ptr) {
        dagml_owned_bytes_free(attestation);
    }
    free_error(error);
    return rejected;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "expected loss, role, metric, foreign loss, FIT_CV, REFIT, PREDICT, and stale task fixture paths\n");
        return 2;
    }
    Buffer loss = read_file(argv[1]);
    Buffer role = read_file(argv[2]);
    Buffer metric = read_file(argv[3]);
    Buffer foreign_loss = read_file(argv[4]);
    Buffer fit_cv_task = read_file(argv[5]);
    Buffer refit_task = read_file(argv[6]);
    Buffer predict_task = read_file(argv[7]);
    Buffer stale_task = read_file(argv[8]);

    CallbackState loss_state = {0, 0, 0, 0, 0, "{\"value\":5.0}"};
    CallbackState metric_state = {0, 0, 0, 0, 0, "{\"value\":1.5}"};
    DagMlLocalImplementationRegistry *registry = NULL;
    DagMlString error = {0};
    DagMlStatusCode status = dagml_local_implementation_registry_create(
        text_view("binding:c"), &registry, &error);
    if (!expect_ok(status, error, "create") || !registry) {
        return 2;
    }

    status = dagml_local_implementation_registry_register_loss(
        registry, buffer_view(loss), callback_vtable(&loss_state), &error);
    if (!expect_ok(status, error, "register loss") || loss_state.retains != 1) {
        return 2;
    }

    status = dagml_local_implementation_registry_register_metric(
        registry, buffer_view(metric), callback_vtable(&metric_state), &error);
    if (!expect_ok(status, error, "register metric") || metric_state.retains != 1) {
        return 2;
    }

    status = dagml_local_implementation_registry_register_loss(
        registry, buffer_view(foreign_loss), callback_vtable(&loss_state), &error);
    if (status == DAG_ML_STATUS_OK || loss_state.retains != 1) {
        fprintf(stderr, "foreign binding registration was not rejected before retain\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};

    DagMlOwnedBytes descriptors = {0};
    status = dagml_local_implementation_registry_descriptors_json(
        registry, &descriptors, &error);
    if (!expect_ok(status, error, "descriptors") ||
        !contains_bytes(descriptors.ptr, descriptors.len, "loss:c:asymmetric") ||
        !contains_bytes(descriptors.ptr, descriptors.len, "metric:c:bias")) {
        return 2;
    }
    dagml_owned_bytes_free(descriptors);

    const char *request = "{\"target\":2.0,\"prediction\":5.0}";
    const char *phases[] = { "FIT_CV", "REFIT" };
    for (size_t index = 0; index < 2; index++) {
        DagMlOwnedBytes result = {0};
        DagMlOwnedBytes attestation = {0};
        status = dagml_local_implementation_registry_invoke_training_loss(
            registry,
            buffer_view(role),
            text_view(phases[index]),
            text_view(request),
            &result,
            &attestation,
            &error);
        if (!expect_ok(status, error, phases[index]) ||
            !contains_bytes(result.ptr, result.len, "\"value\":5.0") ||
            !contains_bytes(attestation.ptr, attestation.len, phases[index])) {
            return 2;
        }
        dagml_owned_bytes_free(result);
        dagml_owned_bytes_free(attestation);
    }
    if (loss_state.calls != 2 || loss_state.byte_releases != 2) {
        fprintf(stderr, "training loss callback/release count mismatch\n");
        return 2;
    }

    Buffer tasks[] = { fit_cv_task, refit_task };
    for (size_t index = 0; index < 2; index++) {
        DagMlOwnedBytes bound_role = {0};
        DagMlOwnedBytes bound_attestation = {0};
        status = dagml_node_task_training_loss_binding(
            buffer_view(tasks[index]),
            0,
            &bound_role,
            &bound_attestation,
            &error);
        if (!expect_ok(status, error, phases[index]) ||
            !contains_bytes(bound_role.ptr, bound_role.len, "loss:c:asymmetric") ||
            !contains_bytes(bound_attestation.ptr, bound_attestation.len, phases[index])) {
            return 2;
        }
        dagml_owned_bytes_free(bound_role);
        dagml_owned_bytes_free(bound_attestation);

        DagMlOwnedBytes result = {0};
        DagMlOwnedBytes attestation = {0};
        status = dagml_local_implementation_registry_invoke_task_training_loss(
            registry,
            buffer_view(tasks[index]),
            0,
            text_view(request),
            &result,
            &attestation,
            &error);
        if (!expect_ok(status, error, phases[index]) ||
            !contains_bytes(result.ptr, result.len, "\"value\":5.0") ||
            !contains_bytes(attestation.ptr, attestation.len, phases[index])) {
            return 2;
        }
        dagml_owned_bytes_free(result);
        dagml_owned_bytes_free(attestation);
    }
    if (loss_state.calls != 4 || loss_state.byte_releases != 4) {
        fprintf(stderr, "task-bound loss callback/release count mismatch\n");
        return 2;
    }

    if (!expect_task_binding_rejected(
            buffer_view(fit_cv_task), 1, "invalid binding role index") ||
        !expect_task_binding_rejected(
            buffer_view(predict_task), 0, "PREDICT binding task") ||
        !expect_task_binding_rejected(
            buffer_view(stale_task), 0, "stale binding task") ||
        !expect_task_binding_rejected(
            text_view("{"), 0, "malformed binding task JSON")) {
        return 2;
    }

    DagMlBytesView null_binding_task = {0};
    if (!expect_task_binding_rejected(
            null_binding_task, 0, "null binding task JSON")) {
        return 2;
    }

    DagMlOwnedBytes null_binding_attestation = {0};
    status = dagml_node_task_training_loss_binding(
        buffer_view(fit_cv_task),
        0,
        NULL,
        &null_binding_attestation,
        &error);
    if (status == DAG_ML_STATUS_OK || null_binding_attestation.ptr) {
        fprintf(stderr, "null binding output pointer retained output bytes\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};

    if (!expect_task_loss_rejected(
            registry, buffer_view(fit_cv_task), 1, text_view(request),
            &loss_state, 4, "invalid task role index") ||
        !expect_task_loss_rejected(
            registry, buffer_view(predict_task), 0, text_view(request),
            &loss_state, 4, "PREDICT task") ||
        !expect_task_loss_rejected(
            registry, buffer_view(stale_task), 0, text_view(request),
            &loss_state, 4, "stale task") ||
        !expect_task_loss_rejected(
            registry, text_view("{"), 0, text_view(request),
            &loss_state, 4, "malformed task JSON")) {
        return 2;
    }

    DagMlBytesView null_task = {0};
    if (!expect_task_loss_rejected(
            registry, null_task, 0, text_view(request),
            &loss_state, 4, "null task JSON")) {
        return 2;
    }

    DagMlOwnedBytes null_output_attestation = {0};
    status = dagml_local_implementation_registry_invoke_task_training_loss(
        registry,
        buffer_view(fit_cv_task),
        0,
        text_view(request),
        NULL,
        &null_output_attestation,
        &error);
    if (status == DAG_ML_STATUS_OK || null_output_attestation.ptr ||
        loss_state.calls != 4 || loss_state.byte_releases != 4) {
        fprintf(stderr, "null task-loss output pointer reached the callback\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};

    DagMlOwnedBytes refused_result = {0};
    DagMlOwnedBytes refused_attestation = {0};
    status = dagml_local_implementation_registry_invoke_training_loss(
        registry,
        buffer_view(role),
        text_view("PREDICT"),
        text_view(request),
        &refused_result,
        &refused_attestation,
        &error);
    if (status == DAG_ML_STATUS_OK || refused_result.ptr || refused_attestation.ptr ||
        loss_state.calls != 4) {
        fprintf(stderr, "PREDICT was not refused before callback invocation\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};

    DagMlOwnedBytes metric_result = {0};
    status = dagml_local_implementation_registry_invoke_metric(
        registry,
        buffer_view(metric),
        text_view(request),
        &metric_result,
        &error);
    if (!expect_ok(status, error, "invoke metric") ||
        !contains_bytes(metric_result.ptr, metric_result.len, "\"value\":1.5") ||
        metric_state.calls != 1 || metric_state.byte_releases != 1) {
        return 2;
    }
    dagml_owned_bytes_free(metric_result);

    loss_state.fail = 1;
    DagMlOwnedBytes failed_task_result = {0};
    DagMlOwnedBytes failed_task_attestation = {0};
    status = dagml_local_implementation_registry_invoke_task_training_loss(
        registry,
        buffer_view(fit_cv_task),
        0,
        text_view(request),
        &failed_task_result,
        &failed_task_attestation,
        &error);
    if (status == DAG_ML_STATUS_OK || failed_task_result.ptr ||
        failed_task_attestation.ptr || !error.ptr ||
        !strstr(error.ptr, "local failure") ||
        loss_state.calls != 5 || loss_state.byte_releases != 5) {
        fprintf(stderr, "task loss callback error retained output or attestation bytes\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};

    DagMlOwnedBytes failed_result = {0};
    status = dagml_local_implementation_registry_invoke_loss(
        registry,
        buffer_view(loss),
        text_view(request),
        &failed_result,
        &error);
    if (status == DAG_ML_STATUS_OK || failed_result.ptr ||
        !error.ptr || !strstr(error.ptr, "local failure") ||
        loss_state.calls != 6 || loss_state.byte_releases != 6) {
        fprintf(stderr, "callback error was not propagated and released exactly once\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};
    loss_state.fail = 0;

    DagMlLocalImplementationVTable invalid = callback_vtable(&loss_state);
    invalid.invoke = NULL;
    status = dagml_local_implementation_registry_register_loss(
        registry, buffer_view(loss), invalid, &error);
    if (status == DAG_ML_STATUS_OK || loss_state.retains != 1 || loss_state.releases != 0) {
        fprintf(stderr, "null callback was not rejected without lifecycle effects\n");
        return 2;
    }
    free_error(error);
    error = (DagMlString){0};

    status = dagml_local_implementation_registry_unregister_loss(
        registry, buffer_view(loss), &error);
    if (!expect_ok(status, error, "unregister loss") || loss_state.releases != 1) {
        return 2;
    }
    status = dagml_local_implementation_registry_clear(registry, &error);
    if (!expect_ok(status, error, "clear") || metric_state.releases != 1) {
        return 2;
    }
    dagml_local_implementation_registry_free(registry);
    if (loss_state.releases != 1 || metric_state.releases != 1) {
        fprintf(stderr, "registry free released an entry more than once\n");
        return 2;
    }

    DagMlLocalImplementationRegistry *borrowed_registry = NULL;
    status = dagml_local_implementation_registry_create(
        text_view("binding:c"), &borrowed_registry, &error);
    if (!expect_ok(status, error, "create borrowed registry")) {
        return 2;
    }
    DagMlLocalImplementationVTable borrowed = callback_vtable(NULL);
    borrowed.retain = NULL;
    borrowed.release = NULL;
    status = dagml_local_implementation_registry_register_loss(
        borrowed_registry, buffer_view(loss), borrowed, &error);
    if (!expect_ok(status, error, "register borrowed loss")) {
        return 2;
    }
    DagMlOwnedBytes stale_result = {0};
    status = dagml_local_implementation_registry_invoke_loss(
        borrowed_registry, buffer_view(loss), text_view(request), &stale_result, &error);
    if (status == DAG_ML_STATUS_OK || stale_result.ptr) {
        fprintf(stderr, "callback with unavailable user_data was not rejected\n");
        return 2;
    }
    free_error(error);
    dagml_local_implementation_registry_free(borrowed_registry);

    free(loss.ptr);
    free(role.ptr);
    free(metric.ptr);
    free(foreign_loss.ptr);
    free(fit_cv_task.ptr);
    free(refit_task.ptr);
    free(predict_task.ptr);
    free(stale_task.ptr);
    return 0;
}
"#;

#[test]
fn c_program_invokes_local_loss_and_metric_with_exact_lifecycle() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("C API crate is under workspace/crates");
    let fixture_path = workspace.join("examples/fixtures/criteria/c_local_implementations.v1.json");
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(&fixture_path).expect("read generated C fixture"))
            .expect("parse generated C fixture");
    let mut predict_task = fixture["tasks"]["FIT_CV"].clone();
    predict_task["phase"] = serde_json::Value::String("PREDICT".to_string());
    predict_task["required_loss_attestations"] = serde_json::Value::Array(Vec::new());
    let mut stale_task = fixture["tasks"]["FIT_CV"].clone();
    stale_task["required_loss_attestations"] = serde_json::Value::Array(Vec::new());

    let target_debug = std::env::current_exe()
        .expect("current test exe path")
        .parent()
        .and_then(Path::parent)
        .expect("test exe lives under target/debug/deps")
        .to_path_buf();
    let dynamic_lib = find_dynamic_library(&target_debug);
    let dynamic_lib_dir = dynamic_lib
        .parent()
        .expect("dynamic library has parent directory");
    let temp = std::env::temp_dir().join(format!(
        "dag_ml_local_implementation_conformance_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    fs::create_dir_all(&temp).expect("create C local implementation temp dir");

    let documents = [
        ("loss.json", &fixture["loss_reference"]),
        ("role.json", &fixture["training_loss_role"]),
        ("metric.json", &fixture["metric_reference"]),
        ("foreign_loss.json", &fixture["foreign_loss_reference"]),
        ("fit_cv_task.json", &fixture["tasks"]["FIT_CV"]),
        ("refit_task.json", &fixture["tasks"]["REFIT"]),
        ("predict_task.json", &predict_task),
        ("stale_task.json", &stale_task),
    ];
    for (name, document) in documents {
        fs::write(temp.join(name), serde_json::to_vec(document).unwrap())
            .expect("write extracted C fixture");
    }

    let c_path = temp.join("local_implementations.c");
    let exe_path = temp.join("local_implementations");
    fs::write(&c_path, C_LOCAL_IMPLEMENTATION_SOURCE).expect("write C local implementation source");

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
        "C local implementation compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let run_output = Command::new(&exe_path)
        .arg(temp.join("loss.json"))
        .arg(temp.join("role.json"))
        .arg(temp.join("metric.json"))
        .arg(temp.join("foreign_loss.json"))
        .arg(temp.join("fit_cv_task.json"))
        .arg(temp.join("refit_task.json"))
        .arg(temp.join("predict_task.json"))
        .arg(temp.join("stale_task.json"))
        .output()
        .expect("run C local implementation executable");
    assert!(
        run_output.status.success(),
        "C local implementation executable failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );
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
        "could not locate dynamic C ABI library `{library_name}` under {}",
        target_debug.display()
    );
}
