library(dagml)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required for DAG-ML R binding tests", call. = FALSE)
}

fixture_path <- system.file(
  "extdata",
  "r_local_implementations.v1.json",
  package = "dagml"
)
stopifnot(nzchar(fixture_path))
fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)

calls <- 0L
asymmetric_loss <- function(target, prediction) {
  calls <<- calls + 1L
  difference <- prediction - target
  mean(ifelse(difference >= 0, difference^2, 2 * difference^2))
}
bias_metric <- function(target, prediction) mean(prediction - target)

registry <- dagml_local_implementation_registry()
stopifnot(inherits(registry, "dagml_local_implementation_registry"))
registry$register_loss(fixture$loss_reference, asymmetric_loss)
registry$register_metric(fixture$metric_reference, bias_metric)
stopifnot(registry$size() == 2L)
stopifnot(length(registry$descriptors()) == 2L)

for (phase in c("FIT_CV", "REFIT")) {
  invocation <- registry$invoke_training_loss(
    fixture$task_json[[phase]],
    target = c(2, 4),
    prediction = c(5, 3)
  )
  stopifnot(isTRUE(all.equal(invocation$value, 5.5)))
  stopifnot(identical(invocation$attestation$phase, phase))
  stopifnot(identical(
    invocation$attestation$descriptor_fingerprint,
    fixture$loss_reference$implementation$descriptor_fingerprint
  ))
  stopifnot(isTRUE(all.equal(
    invocation$attestation,
    fixture$tasks[[phase]]$required_loss_attestations[[1]]
  )))
}
stopifnot(calls == 2L)

stopifnot(isTRUE(all.equal(
  registry$invoke_metric(
    fixture$metric_reference,
    target = c(2, 4),
    prediction = c(5, 3)
  ),
  1
)))

error <- tryCatch(
  registry$invoke_training_loss(
    fixture$invalid_task_json$predict,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(inherits(error, "error"), calls == 2L)

error <- tryCatch(
  registry$register_loss(fixture$foreign_loss_reference, asymmetric_loss),
  error = identity
)
stopifnot(inherits(error, "error"))

portable_builtin <- fixture$loss_reference
portable_builtin$implementation$portability <- "portable_builtin"
portable_registry <- dagml_local_implementation_registry()
error <- tryCatch(
  portable_registry$register_loss(portable_builtin, asymmetric_loss),
  error = identity
)
stopifnot(inherits(error, "error"), grepl("portable_builtin", conditionMessage(error)))

error <- tryCatch(
  registry$register_loss(fixture$loss_reference, asymmetric_loss),
  error = identity
)
stopifnot(inherits(error, "error"), grepl("duplicate", conditionMessage(error)))

error <- tryCatch(
  registry$register_loss(fixture$loss_reference, 42),
  error = identity
)
stopifnot(inherits(error, "error"), grepl("R functions", conditionMessage(error)))

error <- tryCatch(
  registry$resolve_metric(fixture$loss_reference),
  error = identity
)
stopifnot(inherits(error, "error"), grepl("expected a metric", conditionMessage(error)))

drifted <- fixture$loss_reference
drifted$implementation$implementation_version <- "2.0.0"
error <- tryCatch(registry$resolve_loss(drifted), error = identity)
stopifnot(inherits(error, "error"))

error <- tryCatch(
  registry$invoke_training_loss(
    fixture$invalid_task_json$tampered_attestation,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("requirements that do not match", conditionMessage(error)),
  calls == 2L
)

error <- tryCatch(
  registry$invoke_training_loss(
    fixture$invalid_task_json$wrong_attestation_schema,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("requirements that do not match", conditionMessage(error)),
  calls == 2L
)

error <- tryCatch(
  registry$invoke_training_loss(
    fixture$invalid_task_json$missing_attestation,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("requirements that do not match", conditionMessage(error)),
  calls == 2L
)

error <- tryCatch(
  registry$invoke_training_loss(
    fixture$task_json$FIT_CV,
    role_index = 2L,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("outside the active", conditionMessage(error)),
  calls == 2L
)

without_native <- dagml_local_implementation_registry(native_library = "")
without_native$register_loss(fixture$loss_reference, asymmetric_loss)
error <- tryCatch(
  without_native$invoke_training_loss(
    fixture$task_json$FIT_CV,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("DAGML_NATIVE_LIBRARY", conditionMessage(error)),
  calls == 2L
)

invalid_native <- dagml_local_implementation_registry(
  native_library = "/dagml/does/not/exist"
)
invalid_native$register_loss(fixture$loss_reference, asymmetric_loss)
error <- tryCatch(
  invalid_native$invoke_training_loss(
    fixture$task_json$FIT_CV,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("failed to load DAG-ML native library", conditionMessage(error)),
  calls == 2L
)

error <- tryCatch(
  registry$invoke_training_loss(
    fixture$tasks$FIT_CV,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("NodeTask JSON", conditionMessage(error)),
  calls == 2L
)

failing <- dagml_local_implementation_registry()
failing$register_loss(fixture$loss_reference, function(...) stop("local failure"))
error <- tryCatch(
  failing$invoke_training_loss(
    fixture$task_json$FIT_CV,
    target = 2,
    prediction = 5
  ),
  error = identity
)
stopifnot(inherits(error, "error"), grepl("local failure", conditionMessage(error)))

error <- tryCatch(registry$to_json(), error = identity)
stopifnot(inherits(error, "error"), grepl("cannot be serialized", conditionMessage(error)))

removed <- registry$unregister_loss(fixture$loss_reference)
stopifnot(identical(removed, asymmetric_loss), registry$size() == 1L)
removed_metric <- registry$unregister_metric(fixture$metric_reference)
stopifnot(identical(removed_metric, bias_metric), registry$size() == 0L)
registry$register_metric(fixture$metric_reference, bias_metric)
registry$clear()
stopifnot(registry$size() == 0L)
