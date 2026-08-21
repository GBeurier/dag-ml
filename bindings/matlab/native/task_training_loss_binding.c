#include "mex.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE DagMlLibraryHandle;
#else
#include <dlfcn.h>
typedef void *DagMlLibraryHandle;
#endif

typedef uint32_t DagMlStatusCode;

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

typedef DagMlStatusCode (*DagMlTaskTrainingLossBindingFn)(
    DagMlBytesView,
    size_t,
    DagMlOwnedBytes *,
    DagMlOwnedBytes *,
    DagMlString *);
typedef void (*DagMlOwnedBytesFreeFn)(DagMlOwnedBytes);
typedef void (*DagMlStringFreeFn)(DagMlString);

enum { DAG_ML_STATUS_OK = 0 };

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
        mexErrMsgIdAndTxt("dagml:NativeBinding:Allocation", "Failed to allocate native binding text.");
    }
    memcpy(copy, text, len);
    copy[len] = '\0';
    return copy;
}

static void dagml_release_outputs(
    DagMlOwnedBytesFreeFn free_bytes,
    DagMlStringFreeFn free_string,
    DagMlOwnedBytes role,
    DagMlOwnedBytes attestation,
    DagMlString error) {
    if (role.ptr) free_bytes(role);
    if (attestation.ptr) free_bytes(attestation);
    if (error.ptr) free_string(error);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3 || nlhs != 2) {
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Arity",
            "Expected task JSON, zero-based role index, native library path, and two outputs.");
    }
    if (!mxIsChar(prhs[0]) || !mxIsDouble(prhs[1]) ||
        mxGetNumberOfElements(prhs[1]) != 1 || !mxIsChar(prhs[2])) {
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Arguments",
            "Task JSON and native library must be text; role index must be scalar numeric.");
    }

    double index = mxGetScalar(prhs[1]);
    if (!mxIsFinite(index) || index < 0.0 || floor(index) != index ||
        index > 9007199254740991.0 || index > (double)SIZE_MAX) {
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:RoleIndex",
            "Role index must be a non-negative safe integer.");
    }

    char *task_json = mxArrayToString(prhs[0]);
    char *library_path = mxArrayToString(prhs[2]);
    if (!task_json || !library_path || !library_path[0]) {
        if (task_json) mxFree(task_json);
        if (library_path) mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Library",
            "A DAG-ML native library path is required.");
    }

    DagMlLibraryHandle library = dagml_open_library(library_path);
    if (!library) {
        const char *loader_message = dagml_loader_error();
        char *message = dagml_copy_text(loader_message, strlen(loader_message));
        mxFree(task_json);
        mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Library",
            "Failed to load DAG-ML native library: %s",
            message);
    }

    DagMlTaskTrainingLossBindingFn bind_task =
        (DagMlTaskTrainingLossBindingFn)dagml_load_symbol(
            library, "dagml_node_task_training_loss_binding");
    DagMlOwnedBytesFreeFn free_bytes =
        (DagMlOwnedBytesFreeFn)dagml_load_symbol(library, "dagml_owned_bytes_free");
    DagMlStringFreeFn free_string =
        (DagMlStringFreeFn)dagml_load_symbol(library, "dagml_string_free");
    if (!bind_task || !free_bytes || !free_string) {
        const char *loader_message = dagml_loader_error();
        char *message = dagml_copy_text(loader_message, strlen(loader_message));
        dagml_close_library(library);
        mxFree(task_json);
        mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Abi",
            "DAG-ML native library is missing the task binding ABI: %s",
            message);
    }

    DagMlBytesView task_view = {(const uint8_t *)task_json, strlen(task_json)};
    DagMlOwnedBytes role = {0};
    DagMlOwnedBytes attestation = {0};
    DagMlString error = {0};
    DagMlStatusCode status = bind_task(
        task_view, (size_t)index, &role, &attestation, &error);

    if (status != DAG_ML_STATUS_OK) {
        const char *fallback =
            "DAG-ML native training loss binding failed without an error message";
        const char *source = error.ptr ? error.ptr : fallback;
        size_t len = error.ptr ? error.len : strlen(fallback);
        char *message = dagml_copy_text(source, len);
        dagml_release_outputs(free_bytes, free_string, role, attestation, error);
        dagml_close_library(library);
        mxFree(task_json);
        mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Validation",
            "DAG-ML native training loss binding failed: %s",
            message);
    }
    if (!role.ptr || !attestation.ptr) {
        dagml_release_outputs(free_bytes, free_string, role, attestation, error);
        dagml_close_library(library);
        mxFree(task_json);
        mxFree(library_path);
        mexErrMsgIdAndTxt(
            "dagml:NativeBinding:Output",
            "DAG-ML native training loss binding returned invalid output bytes.");
    }

    char *role_json = dagml_copy_text((const char *)role.ptr, role.len);
    char *attestation_json =
        dagml_copy_text((const char *)attestation.ptr, attestation.len);
    dagml_release_outputs(free_bytes, free_string, role, attestation, error);
    dagml_close_library(library);
    mxFree(task_json);
    mxFree(library_path);

    plhs[0] = mxCreateString(role_json);
    plhs[1] = mxCreateString(attestation_json);
    mxFree(role_json);
    mxFree(attestation_json);
}
