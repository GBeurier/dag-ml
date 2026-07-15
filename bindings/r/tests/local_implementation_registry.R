library(dagml)

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
    fixture$tasks[[phase]],
    target = c(2, 4),
    prediction = c(5, 3)
  )
  stopifnot(isTRUE(all.equal(invocation$value, 5.5)))
  stopifnot(identical(invocation$attestation$phase, phase))
  stopifnot(identical(
    invocation$attestation$descriptor_fingerprint,
    fixture$loss_reference$implementation$descriptor_fingerprint
  ))
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

predict_task <- fixture$tasks$FIT_CV
predict_task$phase <- "PREDICT"
error <- tryCatch(
  registry$invoke_training_loss(
    predict_task,
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

drifted <- fixture$loss_reference
drifted$implementation$implementation_version <- "2.0.0"
error <- tryCatch(registry$resolve_loss(drifted), error = identity)
stopifnot(inherits(error, "error"))

failing <- dagml_local_implementation_registry()
failing$register_loss(fixture$loss_reference, function(...) stop("local failure"))
error <- tryCatch(
  failing$invoke_training_loss(
    fixture$tasks$FIT_CV,
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
registry$clear()
stopifnot(registry$size() == 0L)
