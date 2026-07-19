.dagml_stop <- function(message) {
  stop(message, call. = FALSE)
}

.dagml_scalar_text <- function(value, label) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(trimws(value))) {
    .dagml_stop(sprintf("%s must be non-empty text", label))
  }
  value
}

.dagml_same_json_value <- function(left, right) {
  if (is.null(left) || is.null(right)) {
    return(is.null(left) && is.null(right))
  }
  if (is.list(left) || is.list(right)) {
    if (!is.list(left) || !is.list(right) || length(left) != length(right)) {
      return(FALSE)
    }
    left_names <- names(left)
    right_names <- names(right)
    if (is.null(left_names) != is.null(right_names)) {
      return(FALSE)
    }
    if (!is.null(left_names)) {
      if (any(!nzchar(left_names)) || anyDuplicated(left_names) ||
          any(!nzchar(right_names)) || anyDuplicated(right_names) ||
          !setequal(left_names, right_names)) {
        return(FALSE)
      }
      keys <- sort(left_names)
      return(all(vapply(
        keys,
        function(key) .dagml_same_json_value(left[[key]], right[[key]]),
        logical(1)
      )))
    }
    return(all(vapply(
      seq_along(left),
      function(index) .dagml_same_json_value(left[[index]], right[[index]]),
      logical(1)
    )))
  }
  if (is.numeric(left) || is.numeric(right)) {
    return(
      is.numeric(left) && is.numeric(right) &&
        length(left) == length(right) &&
        all(is.finite(left)) && all(is.finite(right)) &&
        identical(as.numeric(left), as.numeric(right))
    )
  }
  identical(left, right)
}

.dagml_descriptor <- function(reference, semantic_kind, binding_id) {
  if (!is.list(reference) || !is.list(reference$implementation)) {
    .dagml_stop("implementation reference must contain an implementation descriptor")
  }
  descriptor <- reference$implementation
  actual_kind <- .dagml_scalar_text(
    descriptor$semantic_kind,
    "implementation semantic_kind"
  )
  if (!identical(actual_kind, semantic_kind)) {
    .dagml_stop(sprintf(
      "expected a %s implementation descriptor, got %s",
      semantic_kind,
      actual_kind
    ))
  }
  actual_binding <- .dagml_scalar_text(
    descriptor$binding_id,
    "implementation binding_id"
  )
  if (!identical(actual_binding, binding_id)) {
    .dagml_stop(sprintf(
      "local implementation requires binding_id '%s', got '%s'",
      binding_id,
      actual_binding
    ))
  }
  portability <- .dagml_scalar_text(
    descriptor$portability,
    "implementation portability"
  )
  if (!portability %in% c("host_local", "portable_registered")) {
    .dagml_stop("local implementation registry rejects portable_builtin descriptors")
  }
  key <- .dagml_scalar_text(
    descriptor$registry_key,
    "implementation registry_key"
  )
  .dagml_scalar_text(
    descriptor$descriptor_fingerprint,
    "implementation descriptor_fingerprint"
  )
  list(key = key, descriptor = descriptor)
}

.dagml_role_applies <- function(role, phase) {
  if (!is.list(role)) {
    .dagml_stop("training loss role must be an object")
  }
  phases <- unlist(role$phases, use.names = FALSE)
  is.character(phases) && phase %in% phases
}

.dagml_task_training_loss_binding <- function(task, role_index, native_library) {
  if (!nzchar(native_library)) {
    .dagml_stop(paste(
      "native DAG-ML library path is required; pass native_library",
      "or set DAGML_NATIVE_LIBRARY"
    ))
  }
  task_json <- .dagml_scalar_text(task, "NodeTask JSON")
  native <- .Call(
    C_dagml_task_training_loss_binding_native,
    task_json,
    as.double(role_index - 1L),
    native_library
  )
  list(
    role = jsonlite::fromJSON(native[["role_json"]], simplifyVector = FALSE),
    attestation = jsonlite::fromJSON(
      native[["attestation_json"]],
      simplifyVector = FALSE
    )
  )
}

.dagml_json_text <- function(value, label) {
  if (is.character(value) && length(value) == 1L && !is.na(value)) {
    return(value)
  }
  tryCatch(
    jsonlite::toJSON(
      value,
      auto_unbox = TRUE,
      null = "null",
      digits = NA
    ),
    error = function(error) {
      .dagml_stop(sprintf(
        "%s must be JSON text or a JSON-serializable R value: %s",
        label,
        conditionMessage(error)
      ))
    }
  )
}

.dagml_node_result_json <- function(value, task) {
  if (is.character(value) && length(value) == 1L && !is.na(value)) {
    result <- jsonlite::fromJSON(value, simplifyVector = FALSE)
  } else {
    result <- value
  }
  if (!is.list(result)) {
    .dagml_stop("controller callback must return NodeResult JSON text or a list")
  }
  if (is.null(result$lineage) || !is.list(result$lineage)) {
    .dagml_stop("controller callback NodeResult must contain lineage")
  }
  if (is.null(result$lineage$seed)) {
    result$lineage$seed <- task$seed
  }
  .dagml_json_text(result, "NodeResult")
}

.dagml_controller_callbacks <- function(controllers) {
  if (is.environment(controllers)) {
    controllers <- as.list(controllers, all.names = TRUE)
  }
  if (!is.list(controllers)) {
    .dagml_stop("controllers must be a named list or environment of functions")
  }
  controller_ids <- names(controllers)
  if (is.null(controller_ids) || length(controller_ids) != length(controllers) ||
      any(is.na(controller_ids)) || any(!nzchar(trimws(controller_ids))) ||
      anyDuplicated(controller_ids)) {
    .dagml_stop("controllers must be named by unique controller ids")
  }
  for (index in seq_along(controllers)) {
    if (!is.function(controllers[[index]])) {
      .dagml_stop("controller callbacks must be functions")
    }
  }
  callbacks <- lapply(controllers, function(callback) {
    force(callback)
    function(controller_id, task_json) {
      task <- jsonlite::fromJSON(task_json, simplifyVector = FALSE)
      .dagml_node_result_json(callback(controller_id, task_json), task)
    }
  })
  list(ids = unname(controller_ids), callbacks = callbacks)
}

#' Execute One ExecutionPlan Phase Through Native DAG-ML
#'
#' Runs the native sequential scheduler for one phase and dispatches each
#' `NodeTask` to a process-local R callback. The callback can close over a
#' `dagml_local_implementation_registry()` and call `invoke_training_loss()`
#' with the exact `task_json` to execute host-local custom losses.
#'
#' @param execution_plan ExecutionPlan JSON text, or a JSON-serializable R value.
#' @param trusted_controller_manifests ControllerManifest list JSON text, or a
#'   JSON-serializable R value. The native core refuses execution if these do not
#'   exactly match the manifests embedded in `execution_plan`.
#' @param run_id DAG-ML run id for the phase execution.
#' @param root_seed Non-negative safe integer root seed.
#' @param phase ABI phase name such as `FIT_CV` or `REFIT`.
#' @param controllers Named list or environment of callbacks keyed by
#'   `controller_id`. Each callback receives `(controller_id, task_json)` and
#'   returns NodeResult JSON text or a JSON-serializable list.
#' @param native_library Path to the DAG-ML C ABI shared library. Defaults to
#'   `DAGML_NATIVE_LIBRARY`.
#' @return A list of decoded NodeResult objects.
#' @export
dagml_execute_execution_plan_phase <- function(
  execution_plan,
  trusted_controller_manifests,
  run_id,
  root_seed,
  phase,
  controllers,
  native_library = Sys.getenv("DAGML_NATIVE_LIBRARY", unset = "")
) {
  run_id <- .dagml_scalar_text(run_id, "run_id")
  phase <- .dagml_scalar_text(phase, "phase")
  if (!phase %in% c("FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN")) {
    .dagml_stop("phase must be FIT_CV, SELECT, REFIT, PREDICT, or EXPLAIN")
  }
  if (!is.numeric(root_seed) || length(root_seed) != 1L ||
      is.na(root_seed) || !is.finite(root_seed) ||
      root_seed != floor(root_seed) || root_seed < 0 ||
      root_seed > 9007199254740991) {
    .dagml_stop("root_seed must be a non-negative safe integer")
  }
  if (!is.character(native_library) || length(native_library) != 1L ||
      is.na(native_library)) {
    .dagml_stop("native_library must be scalar text")
  }
  if (!nzchar(native_library)) {
    .dagml_stop(paste(
      "native DAG-ML library path is required; pass native_library",
      "or set DAGML_NATIVE_LIBRARY"
    ))
  }
  resolved <- .dagml_controller_callbacks(controllers)
  result_json <- .Call(
    C_dagml_execute_execution_plan_phase_native,
    .dagml_json_text(execution_plan, "execution_plan"),
    .dagml_json_text(
      trusted_controller_manifests,
      "trusted_controller_manifests"
    ),
    run_id,
    as.double(root_seed),
    phase,
    resolved$ids,
    resolved$callbacks,
    native_library
  )
  jsonlite::fromJSON(result_json, simplifyVector = FALSE)
}

#' Create a process-local R implementation registry
#'
#' The registry retains R functions without serializing them. Losses and metrics
#' use separate resolution paths. `invoke_training_loss()` accepts a native
#' DAG-ML `NodeTask` JSON, executes the selected local loss, and returns its
#' native attestation only after the function succeeds.
#'
#' @param native_library Path to the DAG-ML C ABI shared library. Defaults to
#'   the `DAGML_NATIVE_LIBRARY` environment variable and is required only for
#'   task-bound training-loss invocation.
#' @return An environment with registration, resolution, invocation,
#'   unregistration, inspection, and lifecycle methods.
#' @export
dagml_local_implementation_registry <- function(
  native_library = Sys.getenv("DAGML_NATIVE_LIBRARY", unset = "")
) {
  binding_id <- "binding:r"
  if (!is.character(native_library) || length(native_library) != 1L ||
      is.na(native_library)) {
    .dagml_stop("native_library must be scalar text")
  }
  entries <- new.env(hash = TRUE, parent = emptyenv())
  api <- new.env(parent = emptyenv())

  register <- function(reference, implementation, semantic_kind) {
    if (!is.function(implementation)) {
      .dagml_stop("local loss and metric implementations must be R functions")
    }
    resolved <- .dagml_descriptor(reference, semantic_kind, binding_id)
    if (exists(resolved$key, envir = entries, inherits = FALSE)) {
      .dagml_stop(sprintf(
        "duplicate local implementation registry key '%s'",
        resolved$key
      ))
    }
    assign(
      resolved$key,
      list(
        descriptor = resolved$descriptor,
        implementation = implementation
      ),
      envir = entries
    )
    invisible(api)
  }

  resolve <- function(reference, semantic_kind) {
    resolved <- .dagml_descriptor(reference, semantic_kind, binding_id)
    if (!exists(resolved$key, envir = entries, inherits = FALSE)) {
      .dagml_stop(sprintf(
        "local implementation registry has no implementation for '%s'",
        resolved$key
      ))
    }
    entry <- get(resolved$key, envir = entries, inherits = FALSE)
    if (!.dagml_same_json_value(entry$descriptor, resolved$descriptor)) {
      .dagml_stop(sprintf(
        "local implementation registered for '%s' does not match the requested descriptor",
        resolved$key
      ))
    }
    entry$implementation
  }

  unregister <- function(reference, semantic_kind) {
    implementation <- resolve(reference, semantic_kind)
    key <- reference$implementation$registry_key
    rm(list = key, envir = entries)
    implementation
  }

  api$register_loss <- function(loss_reference, implementation) {
    register(loss_reference, implementation, "loss")
  }
  api$register_metric <- function(metric_reference, implementation) {
    register(metric_reference, implementation, "metric")
  }
  api$resolve_loss <- function(loss_reference) {
    resolve(loss_reference, "loss")
  }
  api$resolve_training_loss <- function(training_loss_role, phase) {
    phase <- .dagml_scalar_text(phase, "training phase")
    if (!phase %in% c("FIT_CV", "REFIT")) {
      .dagml_stop("training loss phase must be FIT_CV or REFIT")
    }
    if (!.dagml_role_applies(training_loss_role, phase)) {
      .dagml_stop(sprintf("training loss role does not apply to phase %s", phase))
    }
    resolve(training_loss_role$loss, "loss")
  }
  api$resolve_metric <- function(metric_reference) {
    resolve(metric_reference, "metric")
  }
  api$invoke_loss <- function(loss_reference, ...) {
    do.call(resolve(loss_reference, "loss"), list(...), envir = parent.frame())
  }
  api$invoke_metric <- function(metric_reference, ...) {
    do.call(resolve(metric_reference, "metric"), list(...), envir = parent.frame())
  }
  api$invoke_training_loss <- function(task, role_index = 1L, ...) {
    if (!is.numeric(role_index) || length(role_index) != 1L ||
        is.na(role_index) || !is.finite(role_index) ||
        role_index != floor(role_index) || role_index < 1 ||
        role_index > 9007199254740992) {
      .dagml_stop("role_index must be a positive integer")
    }
    binding <- .dagml_task_training_loss_binding(
      task,
      role_index,
      native_library
    )
    implementation <- resolve(binding$role$loss, "loss")
    value <- do.call(implementation, list(...), envir = parent.frame())
    list(value = value, attestation = binding$attestation)
  }
  api$unregister_loss <- function(loss_reference) {
    unregister(loss_reference, "loss")
  }
  api$unregister_metric <- function(metric_reference) {
    unregister(metric_reference, "metric")
  }
  api$descriptors <- function() {
    keys <- sort(ls(envir = entries, all.names = TRUE))
    lapply(keys, function(key) {
      get(key, envir = entries, inherits = FALSE)$descriptor
    })
  }
  api$size <- function() length(ls(envir = entries, all.names = TRUE))
  api$clear <- function() {
    rm(list = ls(envir = entries, all.names = TRUE), envir = entries)
    invisible(api)
  }
  api$to_json <- function() {
    .dagml_stop("DAG-ML local implementation registries cannot be serialized")
  }

  class(api) <- "dagml_local_implementation_registry"
  lockEnvironment(api, bindings = TRUE)
  api
}
