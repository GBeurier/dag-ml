#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
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

typedef DagMlStatusCode (*DagMlTaskTrainingLossBindingFn)(
    DagMlBytesView,
    size_t,
    DagMlOwnedBytes *,
    DagMlOwnedBytes *,
    DagMlString *);
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

typedef struct DagMlRControllerState {
    SEXP callback;
    const char *controller_id;
} DagMlRControllerState;

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

static void dagml_require_scalar_string(SEXP value, const char *label) {
    if (TYPEOF(value) != STRSXP || XLENGTH(value) != 1 ||
        STRING_ELT(value, 0) == NA_STRING) {
        Rf_error("%s must be scalar text", label);
    }
}

static char *dagml_copy_error(DagMlString native_error) {
    const char *fallback = "DAG-ML native training loss binding failed without an error message";
    size_t len = native_error.ptr ? native_error.len : strlen(fallback);
    const char *source = native_error.ptr ? native_error.ptr : fallback;
    char *message = (char *)R_alloc(len + 1, sizeof(char));
    memcpy(message, source, len);
    message[len] = '\0';
    return message;
}

static DagMlBytesView dagml_string_view(SEXP value) {
    DagMlBytesView view;
    view.ptr = (const uint8_t *)CHAR(STRING_ELT(value, 0));
    view.len = (size_t)LENGTH(STRING_ELT(value, 0));
    return view;
}

static void dagml_r_controller_release_bytes(void *user_data, DagMlOwnedBytes bytes) {
    (void)user_data;
    if (bytes.ptr) {
        free(bytes.ptr);
    }
}

static DagMlStatusCode dagml_r_controller_invoke(
    void *user_data,
    DagMlBytesView task_json,
    DagMlOwnedBytes *out_result_json) {
    if (!user_data || !task_json.ptr || !out_result_json) {
        return 1;
    }
    DagMlRControllerState *state = (DagMlRControllerState *)user_data;
    int protect_count = 0;
    SEXP controller_id = PROTECT(Rf_mkString(state->controller_id));
    protect_count++;
    SEXP task = PROTECT(Rf_allocVector(STRSXP, 1));
    protect_count++;
    if (task_json.len > (size_t)INT_MAX) {
        UNPROTECT(protect_count);
        return 2;
    }
    SET_STRING_ELT(
        task,
        0,
        Rf_mkCharLenCE((const char *)task_json.ptr, (int)task_json.len, CE_UTF8));
    SEXP call = PROTECT(Rf_lang3(state->callback, controller_id, task));
    protect_count++;
    int eval_error = 0;
    SEXP returned = R_tryEval(call, R_GlobalEnv, &eval_error);
    if (eval_error) {
        UNPROTECT(protect_count);
        return 2;
    }
    PROTECT(returned);
    protect_count++;
    if (TYPEOF(returned) != STRSXP || XLENGTH(returned) != 1 ||
        STRING_ELT(returned, 0) == NA_STRING) {
        UNPROTECT(protect_count);
        return 2;
    }
    size_t len = (size_t)LENGTH(STRING_ELT(returned, 0));
    uint8_t *copy = (uint8_t *)malloc(len == 0 ? 1 : len);
    if (!copy) {
        UNPROTECT(protect_count);
        return 2;
    }
    memcpy(copy, CHAR(STRING_ELT(returned, 0)), len);
    out_result_json->ptr = copy;
    out_result_json->len = len;
    out_result_json->capacity = len == 0 ? 1 : len;
    UNPROTECT(protect_count);
    return DAG_ML_STATUS_OK;
}

SEXP dagml_task_training_loss_binding_native(
    SEXP task_json,
    SEXP role_index,
    SEXP native_library) {
    dagml_require_scalar_string(task_json, "task_json");
    dagml_require_scalar_string(native_library, "native_library");
    if (TYPEOF(role_index) != REALSXP || XLENGTH(role_index) != 1) {
        Rf_error("role_index must be one non-negative safe integer");
    }

    const char *task = CHAR(STRING_ELT(task_json, 0));
    const char *library_path = CHAR(STRING_ELT(native_library, 0));
    double index = REAL(role_index)[0];
    if (!R_FINITE(index) || index < 0.0 || floor(index) != index ||
        index > 9007199254740991.0 || index > (double)SIZE_MAX) {
        Rf_error("role_index must be one non-negative safe integer");
    }
    if (!library_path[0]) {
        Rf_error(
            "native DAG-ML library path is required; pass native_library or set DAGML_NATIVE_LIBRARY");
    }

    DagMlLibraryHandle library = dagml_open_library(library_path);
    if (!library) {
        Rf_error("failed to load DAG-ML native library '%s': %s", library_path, dagml_loader_error());
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
        char *message = (char *)R_alloc(strlen(loader_message) + 1, sizeof(char));
        strcpy(message, loader_message);
        dagml_close_library(library);
        Rf_error("DAG-ML native library is missing the task binding ABI: %s", message);
    }

    DagMlBytesView task_view = {(const uint8_t *)task, (size_t)LENGTH(STRING_ELT(task_json, 0))};
    DagMlOwnedBytes role = {0};
    DagMlOwnedBytes attestation = {0};
    DagMlString native_error = {0};
    DagMlStatusCode status = bind_task(
        task_view, (size_t)index, &role, &attestation, &native_error);

    if (status != DAG_ML_STATUS_OK) {
        char *message = dagml_copy_error(native_error);
        if (role.ptr) free_bytes(role);
        if (attestation.ptr) free_bytes(attestation);
        if (native_error.ptr) free_string(native_error);
        dagml_close_library(library);
        Rf_error("DAG-ML native training loss binding failed: %s", message);
    }
    if (!role.ptr || !attestation.ptr || role.len > INT_MAX || attestation.len > INT_MAX) {
        if (role.ptr) free_bytes(role);
        if (attestation.ptr) free_bytes(attestation);
        if (native_error.ptr) free_string(native_error);
        dagml_close_library(library);
        Rf_error("DAG-ML native training loss binding returned invalid output bytes");
    }

    SEXP result = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(
        result,
        0,
        Rf_mkCharLenCE((const char *)role.ptr, (int)role.len, CE_UTF8));
    SET_STRING_ELT(
        result,
        1,
        Rf_mkCharLenCE((const char *)attestation.ptr, (int)attestation.len, CE_UTF8));
    free_bytes(role);
    free_bytes(attestation);
    if (native_error.ptr) free_string(native_error);
    dagml_close_library(library);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("role_json"));
    SET_STRING_ELT(names, 1, Rf_mkChar("attestation_json"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

SEXP dagml_execute_execution_plan_phase_native(
    SEXP execution_plan_json,
    SEXP trusted_controller_manifests_json,
    SEXP run_id,
    SEXP root_seed,
    SEXP phase,
    SEXP controller_ids,
    SEXP callbacks,
    SEXP native_library) {
    dagml_require_scalar_string(execution_plan_json, "execution_plan_json");
    dagml_require_scalar_string(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json");
    dagml_require_scalar_string(run_id, "run_id");
    dagml_require_scalar_string(phase, "phase");
    dagml_require_scalar_string(native_library, "native_library");
    if (TYPEOF(root_seed) != REALSXP || XLENGTH(root_seed) != 1) {
        Rf_error("root_seed must be one non-negative safe integer");
    }
    double seed = REAL(root_seed)[0];
    if (!R_FINITE(seed) || seed < 0.0 || floor(seed) != seed ||
        seed > 9007199254740991.0) {
        Rf_error("root_seed must be one non-negative safe integer");
    }
    if (TYPEOF(controller_ids) != STRSXP || !isVectorList(callbacks) ||
        XLENGTH(controller_ids) != XLENGTH(callbacks)) {
        Rf_error("controller_ids and callbacks must have the same length");
    }
    R_xlen_t count = XLENGTH(controller_ids);
    if ((uint64_t)count > (uint64_t)SIZE_MAX) {
        Rf_error("too many controller callbacks");
    }
    const char *library_path = CHAR(STRING_ELT(native_library, 0));
    if (!library_path[0]) {
        Rf_error(
            "native DAG-ML library path is required; pass native_library or set DAGML_NATIVE_LIBRARY");
    }

    DagMlLibraryHandle library = dagml_open_library(library_path);
    if (!library) {
        Rf_error("failed to load DAG-ML native library '%s': %s", library_path, dagml_loader_error());
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
        char *message = (char *)R_alloc(strlen(loader_message) + 1, sizeof(char));
        strcpy(message, loader_message);
        dagml_close_library(library);
        Rf_error("DAG-ML native library is missing the phase execution ABI: %s", message);
    }

    DagMlRControllerState *states =
        (DagMlRControllerState *)R_alloc((size_t)count, sizeof(DagMlRControllerState));
    DagMlControllerBinding *bindings =
        (DagMlControllerBinding *)R_alloc((size_t)count, sizeof(DagMlControllerBinding));
    for (R_xlen_t index = 0; index < count; index++) {
        if (STRING_ELT(controller_ids, index) == NA_STRING) {
            dagml_close_library(library);
            Rf_error("controller ids must be text");
        }
        SEXP callback = VECTOR_ELT(callbacks, index);
        if (!Rf_isFunction(callback)) {
            dagml_close_library(library);
            Rf_error("controller callbacks must be functions");
        }
        states[index].callback = callback;
        states[index].controller_id = CHAR(STRING_ELT(controller_ids, index));
        bindings[index].controller_id.ptr = (const uint8_t *)states[index].controller_id;
        bindings[index].controller_id.len = (size_t)LENGTH(STRING_ELT(controller_ids, index));
        bindings[index].vtable.abi_version = DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION;
        bindings[index].vtable.user_data = &states[index];
        bindings[index].vtable.clone_with = NULL;
        bindings[index].vtable.describe = NULL;
        bindings[index].vtable.fit = NULL;
        bindings[index].vtable.predict = NULL;
        bindings[index].vtable.invoke = dagml_r_controller_invoke;
        bindings[index].vtable.release_bytes = dagml_r_controller_release_bytes;
        bindings[index].vtable.release = NULL;
        bindings[index].vtable.destroy = NULL;
    }

    DagMlOwnedBytes out = {0};
    DagMlString native_error = {0};
    DagMlStatusCode status = execute_phase(
        dagml_string_view(execution_plan_json),
        dagml_string_view(trusted_controller_manifests_json),
        dagml_string_view(run_id),
        (uint64_t)seed,
        dagml_string_view(phase),
        bindings,
        (size_t)count,
        &out,
        &native_error);

    if (status != DAG_ML_STATUS_OK) {
        char *message = dagml_copy_error(native_error);
        if (out.ptr) free_bytes(out);
        if (native_error.ptr) free_string(native_error);
        dagml_close_library(library);
        Rf_error("DAG-ML native execution plan phase failed: %s", message);
    }
    if (!out.ptr || out.len > INT_MAX) {
        if (out.ptr) free_bytes(out);
        if (native_error.ptr) free_string(native_error);
        dagml_close_library(library);
        Rf_error("DAG-ML native execution plan phase returned invalid output bytes");
    }

    SEXP result = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(
        result,
        0,
        Rf_mkCharLenCE((const char *)out.ptr, (int)out.len, CE_UTF8));
    free_bytes(out);
    if (native_error.ptr) free_string(native_error);
    dagml_close_library(library);
    UNPROTECT(1);
    return result;
}

static const R_CallMethodDef call_methods[] = {
    {"dagml_task_training_loss_binding_native",
     (DL_FUNC)&dagml_task_training_loss_binding_native,
     3},
    {"dagml_execute_execution_plan_phase_native",
     (DL_FUNC)&dagml_execute_execution_plan_phase_native,
     8},
    {NULL, NULL, 0}
};

void attribute_visible R_init_dagml(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
