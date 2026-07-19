#include "mex.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE DagMlLibraryHandle;
#else
#include <dlfcn.h>
typedef void *DagMlLibraryHandle;
#endif

typedef uint32_t DagMlStatusCode;
typedef uint64_t DagMlHandle;
typedef struct ArrowArray ArrowArray;
typedef struct ArrowSchema ArrowSchema;

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
    DagMlStatusCode (*clone_with)(
        void *, DagMlHandle, DagMlBytesView, DagMlHandle *);
    DagMlStatusCode (*describe)(void *, DagMlHandle, DagMlOwnedBytes *);
    DagMlStatusCode (*fit)(
        void *, DagMlHandle, DagMlHandle, DagMlBytesView, DagMlHandle *);
    DagMlStatusCode (*predict)(
        void *, DagMlHandle, DagMlHandle, ArrowArray **, ArrowSchema **);
    DagMlStatusCode (*invoke)(void *, DagMlBytesView, DagMlOwnedBytes *);
    void (*release_bytes)(void *, DagMlOwnedBytes);
    void (*release)(void *, DagMlHandle);
    void (*destroy)(void *);
} DagMlControllerVTable;

typedef struct DagMlControllerBinding {
    DagMlBytesView controller_id;
    DagMlControllerVTable vtable;
} DagMlControllerBinding;

typedef DagMlStatusCode (*DagMlExecutionPlanExecutePhaseFn)(
    DagMlBytesView,
    DagMlBytesView,
    DagMlBytesView,
    uint64_t,
    DagMlBytesView,
    const DagMlControllerBinding *,
    size_t,
    DagMlOwnedBytes *,
    DagMlString *);
typedef void (*DagMlOwnedBytesFreeFn)(DagMlOwnedBytes);
typedef void (*DagMlStringFreeFn)(DagMlString);

enum { DAG_ML_STATUS_OK = 0 };
enum { DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION = 2 };

typedef struct DagMlMatlabControllerState {
    const mxArray *callback;
    const char *controller_id;
} DagMlMatlabControllerState;

static DagMlLibraryHandle dagml_open_library(const char *path) {
#ifdef _WIN32
    return LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void dagml_close_library(DagMlLibraryHandle library) {
#ifdef _WIN32
    FreeLibrary(library);
#else
    dlclose(library);
#endif
}

static void *dagml_load_symbol(DagMlLibraryHandle library, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress(library, name);
#else
    return dlsym(library, name);
#endif
}

static const char *dagml_loader_error(void) {
#ifdef _WIN32
    return "Windows failed to load the DAG-ML library or required symbol";
#else
    const char *message = dlerror();
    return message ? message : "failed to load the DAG-ML library or required symbol";
#endif
}

static char *dagml_copy_text(const char *text, size_t len) {
    char *copy = (char *)mxMalloc(len + 1);
    if (!copy) {
        mexErrMsgIdAndTxt("dagml:NativePhase:Allocation", "Failed to allocate text.");
    }
    memcpy(copy, text, len);
    copy[len] = '\0';
    return copy;
}

static void dagml_matlab_release_bytes(void *user_data, DagMlOwnedBytes bytes) {
    (void)user_data;
    if (bytes.ptr) {
        mxFree(bytes.ptr);
    }
}

static DagMlStatusCode dagml_matlab_invoke_controller(
    void *user_data,
    DagMlBytesView task_json,
    DagMlOwnedBytes *out_result_json) {
    if (!user_data || !task_json.ptr || !out_result_json) {
        return 1;
    }
    DagMlMatlabControllerState *state =
        (DagMlMatlabControllerState *)user_data;
    if (task_json.len > (size_t)INT_MAX) {
        return 2;
    }
    mxArray *rhs[3];
    mxArray *lhs[1] = {NULL};
    rhs[0] = (mxArray *)state->callback;
    rhs[1] = mxCreateString(state->controller_id);
    char *task_copy = dagml_copy_text((const char *)task_json.ptr, task_json.len);
    rhs[2] = mxCreateString(task_copy);
    mxFree(task_copy);
    if (!rhs[1] || !rhs[2]) {
        if (rhs[1]) mxDestroyArray(rhs[1]);
        if (rhs[2]) mxDestroyArray(rhs[2]);
        return 2;
    }
    mxArray *exception = mexCallMATLABWithTrap(1, lhs, 3, rhs, "feval");
    if (exception) {
        mxDestroyArray(exception);
        mxDestroyArray(rhs[1]);
        mxDestroyArray(rhs[2]);
        if (lhs[0]) mxDestroyArray(lhs[0]);
        return 2;
    }
    mxDestroyArray(rhs[1]);
    mxDestroyArray(rhs[2]);
    if (!lhs[0] || !mxIsChar(lhs[0])) {
        if (lhs[0]) mxDestroyArray(lhs[0]);
        return 2;
    }
    char *result = mxArrayToString(lhs[0]);
    mxDestroyArray(lhs[0]);
    if (!result) {
        return 2;
    }
    size_t len = strlen(result);
    out_result_json->ptr = (uint8_t *)result;
    out_result_json->len = len;
    out_result_json->capacity = len;
    return DAG_ML_STATUS_OK;
}

static DagMlBytesView dagml_text_view(const char *text) {
    DagMlBytesView view;
    view.ptr = (const uint8_t *)text;
    view.len = strlen(text);
    return view;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 8 || nlhs != 1) {
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Arity",
            "Expected plan JSON, trusted manifests JSON, run id, root seed, phase, controller ids, callbacks, native library path.");
    }
    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1]) || !mxIsChar(prhs[2]) ||
        !mxIsDouble(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1 ||
        !mxIsChar(prhs[4]) || !mxIsCell(prhs[5]) || !mxIsCell(prhs[6]) ||
        !mxIsChar(prhs[7]) ||
        mxGetNumberOfElements(prhs[5]) != mxGetNumberOfElements(prhs[6])) {
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Arguments",
            "Invalid execution phase bridge arguments.");
    }
    double seed_value = mxGetScalar(prhs[3]);
    if (!mxIsFinite(seed_value) || seed_value < 0.0 ||
        floor(seed_value) != seed_value || seed_value > 9007199254740991.0) {
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Seed",
            "Root seed must be a non-negative safe integer.");
    }

    char *plan_json = mxArrayToString(prhs[0]);
    char *manifests_json = mxArrayToString(prhs[1]);
    char *run_id = mxArrayToString(prhs[2]);
    char *phase = mxArrayToString(prhs[4]);
    char *library_path = mxArrayToString(prhs[7]);
    if (!plan_json || !manifests_json || !run_id || !phase ||
        !library_path || !library_path[0]) {
        if (plan_json) mxFree(plan_json);
        if (manifests_json) mxFree(manifests_json);
        if (run_id) mxFree(run_id);
        if (phase) mxFree(phase);
        if (library_path) mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Library",
            "A DAG-ML native library path is required.");
    }

    DagMlLibraryHandle library = dagml_open_library(library_path);
    if (!library) {
        const char *loader_message = dagml_loader_error();
        char message[1024];
        snprintf(message, sizeof(message), "%s", loader_message);
        mxFree(plan_json);
        mxFree(manifests_json);
        mxFree(run_id);
        mxFree(phase);
        mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Library",
            "Failed to load DAG-ML native library: %s",
            message);
    }

    DagMlExecutionPlanExecutePhaseFn execute_phase =
        (DagMlExecutionPlanExecutePhaseFn)dagml_load_symbol(
            library, "dagml_execution_plan_execute_phase_json");
    DagMlOwnedBytesFreeFn free_bytes =
        (DagMlOwnedBytesFreeFn)dagml_load_symbol(library, "dagml_owned_bytes_free");
    DagMlStringFreeFn free_string =
        (DagMlStringFreeFn)dagml_load_symbol(library, "dagml_string_free");
    if (!execute_phase || !free_bytes || !free_string) {
        const char *loader_message = dagml_loader_error();
        char message[1024];
        snprintf(message, sizeof(message), "%s", loader_message);
        dagml_close_library(library);
        mxFree(plan_json);
        mxFree(manifests_json);
        mxFree(run_id);
        mxFree(phase);
        mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Abi",
            "DAG-ML native library is missing the phase execution ABI: %s",
            message);
    }

    size_t count = mxGetNumberOfElements(prhs[5]);
    DagMlMatlabControllerState *states =
        (DagMlMatlabControllerState *)mxCalloc(count, sizeof(DagMlMatlabControllerState));
    DagMlControllerBinding *bindings =
        (DagMlControllerBinding *)mxCalloc(count, sizeof(DagMlControllerBinding));
    char **controller_ids = (char **)mxCalloc(count, sizeof(char *));
    if (!states || !bindings || !controller_ids) {
        mexErrMsgIdAndTxt("dagml:NativePhase:Allocation", "Failed to allocate controller bindings.");
    }
    for (size_t index = 0; index < count; index++) {
        const mxArray *id_value = mxGetCell(prhs[5], index);
        const mxArray *callback = mxGetCell(prhs[6], index);
        if (!id_value || !mxIsChar(id_value) || !callback) {
            mexErrMsgIdAndTxt(
                "dagml:NativePhase:Controller",
                "Controller ids must be text and callbacks must be present.");
        }
        controller_ids[index] = mxArrayToString(id_value);
        if (!controller_ids[index] || !controller_ids[index][0]) {
            mexErrMsgIdAndTxt(
                "dagml:NativePhase:Controller",
                "Controller ids must be non-empty text.");
        }
        states[index].callback = callback;
        states[index].controller_id = controller_ids[index];
        bindings[index].controller_id = dagml_text_view(controller_ids[index]);
        bindings[index].vtable.abi_version =
            DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION;
        bindings[index].vtable.user_data = &states[index];
        bindings[index].vtable.clone_with = NULL;
        bindings[index].vtable.describe = NULL;
        bindings[index].vtable.fit = NULL;
        bindings[index].vtable.predict = NULL;
        bindings[index].vtable.invoke = dagml_matlab_invoke_controller;
        bindings[index].vtable.release_bytes = dagml_matlab_release_bytes;
        bindings[index].vtable.release = NULL;
        bindings[index].vtable.destroy = NULL;
    }

    DagMlOwnedBytes out = {0};
    DagMlString error = {0};
    DagMlStatusCode status = execute_phase(
        dagml_text_view(plan_json),
        dagml_text_view(manifests_json),
        dagml_text_view(run_id),
        (uint64_t)seed_value,
        dagml_text_view(phase),
        bindings,
        count,
        &out,
        &error);
    if (status != DAG_ML_STATUS_OK) {
        const char *fallback =
            "DAG-ML native execution plan phase failed without an error message";
        const char *source = error.ptr ? error.ptr : fallback;
        size_t len = error.ptr ? error.len : strlen(fallback);
        char message[1024];
        size_t copy_len = len < sizeof(message) - 1 ? len : sizeof(message) - 1;
        memcpy(message, source, copy_len);
        message[copy_len] = '\0';
        if (out.ptr) free_bytes(out);
        if (error.ptr) free_string(error);
        dagml_close_library(library);
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Validation",
            "DAG-ML native execution plan phase failed: %s",
            message);
    }
    if (!out.ptr) {
        if (error.ptr) free_string(error);
        dagml_close_library(library);
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Output",
            "DAG-ML native execution plan phase returned invalid output bytes.");
    }
    char *result = dagml_copy_text((const char *)out.ptr, out.len);
    mxArray *result_value = mxCreateString(result);
    mxFree(result);
    free_bytes(out);
    if (error.ptr) free_string(error);
    dagml_close_library(library);
    for (size_t index = 0; index < count; index++) {
        if (controller_ids[index]) mxFree(controller_ids[index]);
    }
    mxFree(controller_ids);
    mxFree(bindings);
    mxFree(states);
    mxFree(plan_json);
    mxFree(manifests_json);
    mxFree(run_id);
    mxFree(phase);
    mxFree(library_path);
    if (!result_value) {
        mexErrMsgIdAndTxt(
            "dagml:NativePhase:Output",
            "Failed to allocate MATLAB result string.");
    }
    plhs[0] = result_value;
}
