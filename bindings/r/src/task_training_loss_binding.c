#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>

#include <limits.h>
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

static void dagml_require_scalar_string(SEXP value, const char *label) {
    if (TYPEOF(value) != STRSXP || XLENGTH(value) != 1 ||
        STRING_ELT(value, 0) == NA_STRING) {
        Rf_error("%s must be scalar text", label);
    }
}

static char *dagml_copy_error(DagMlString error) {
    const char *fallback = "DAG-ML native training loss binding failed without an error message";
    size_t len = error.ptr ? error.len : strlen(fallback);
    const char *source = error.ptr ? error.ptr : fallback;
    char *message = (char *)R_alloc(len + 1, sizeof(char));
    memcpy(message, source, len);
    message[len] = '\0';
    return message;
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
    DagMlString error = {0};
    DagMlStatusCode status = bind_task(
        task_view, (size_t)index, &role, &attestation, &error);

    if (status != DAG_ML_STATUS_OK) {
        char *message = dagml_copy_error(error);
        if (role.ptr) free_bytes(role);
        if (attestation.ptr) free_bytes(attestation);
        if (error.ptr) free_string(error);
        dagml_close_library(library);
        Rf_error("DAG-ML native training loss binding failed: %s", message);
    }
    if (!role.ptr || !attestation.ptr || role.len > INT_MAX || attestation.len > INT_MAX) {
        if (role.ptr) free_bytes(role);
        if (attestation.ptr) free_bytes(attestation);
        if (error.ptr) free_string(error);
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
    if (error.ptr) free_string(error);
    dagml_close_library(library);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("role_json"));
    SET_STRING_ELT(names, 1, Rf_mkChar("attestation_json"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

static const R_CallMethodDef call_methods[] = {
    {"dagml_task_training_loss_binding_native",
     (DL_FUNC)&dagml_task_training_loss_binding_native,
     3},
    {NULL, NULL, 0}
};

void attribute_visible R_init_dagml(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
