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

.dagml_validate_requirement <- function(task, role, requirement) {
  if (!is.list(requirement)) {
    .dagml_stop("loss execution requirement must be an object")
  }
  loss <- role$loss
  if (!is.list(loss) || !is.list(loss$spec) || !is.list(loss$implementation)) {
    .dagml_stop("training loss role contains an invalid loss reference")
  }
  if (!.dagml_same_json_value(role$node_id, task$node_plan$node_id)) {
    .dagml_stop("training loss role node_id does not match the task node")
  }
  if (!.dagml_same_json_value(requirement$schema_version, 1)) {
    .dagml_stop("loss execution requirement schema_version must be 1")
  }
  expected <- list(
    node_id = task$node_plan$node_id,
    output_id = role$output_id,
    phase = task$phase,
    loss_id = loss$spec$loss_id,
    semantic_fingerprint = loss$spec$spec_fingerprint,
    implementation_fingerprint = loss$implementation$implementation_fingerprint,
    descriptor_fingerprint = loss$implementation$descriptor_fingerprint,
    effective_parameters = loss$spec$parameters,
    reduction = loss$spec$reduction
  )
  for (field in names(expected)) {
    if (!.dagml_same_json_value(requirement[[field]], expected[[field]])) {
      .dagml_stop(sprintf(
        "loss execution requirement field '%s' does not match the training role",
        field
      ))
    }
  }
  .dagml_scalar_text(
    requirement$attestation_fingerprint,
    "loss execution requirement attestation_fingerprint"
  )
  invisible(requirement)
}

#' Create a process-local R implementation registry
#'
#' The registry retains R functions without serializing them. Losses and metrics
#' use separate resolution paths. `invoke_training_loss()` accepts a native
#' DAG-ML `NodeTask`, executes the selected local loss, and returns its native
#' attestation only after the function succeeds.
#'
#' @return An environment with registration, resolution, invocation,
#'   unregistration, inspection, and lifecycle methods.
#' @export
dagml_local_implementation_registry <- function() {
  binding_id <- "binding:r"
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
    if (!is.list(task) || !is.list(task$node_plan)) {
      .dagml_stop("training loss invocation requires a DAG-ML NodeTask")
    }
    phase <- .dagml_scalar_text(task$phase, "task phase")
    if (!phase %in% c("FIT_CV", "REFIT")) {
      .dagml_stop("training loss phase must be FIT_CV or REFIT")
    }
    roles <- task$node_plan$training_losses
    if (is.null(roles)) roles <- list()
    active_roles <- Filter(
      function(role) .dagml_role_applies(role, phase),
      roles
    )
    requirements <- task$required_loss_attestations
    if (is.null(requirements)) requirements <- list()
    if (length(active_roles) != length(requirements)) {
      .dagml_stop("task loss execution requirement count does not match active roles")
    }
    if (!is.numeric(role_index) || length(role_index) != 1L ||
        is.na(role_index) || !is.finite(role_index) ||
        role_index != as.integer(role_index) ||
        role_index < 1L || role_index > length(active_roles)) {
      .dagml_stop("role_index is outside the active training loss range")
    }
    role <- active_roles[[as.integer(role_index)]]
    requirement <- requirements[[as.integer(role_index)]]
    .dagml_validate_requirement(task, role, requirement)
    implementation <- api$resolve_training_loss(role, phase)
    value <- do.call(implementation, list(...), envir = parent.frame())
    list(value = value, attestation = requirement)
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
