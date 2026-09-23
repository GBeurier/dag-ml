#' Capture a no-CV full REFIT package through DAG-ML
#'
#' The native CLI validates the pipeline and training row order, executes REFIT
#' through the host operator adapter, and writes a signed package. Host model
#' bytes remain in host-managed sidecars, separate from the package JSON.
#'
#' @param dsl,controllers,envelope,training_sample_ids JSON input file paths.
#' @param adapter Executable operator process adapter.
#' @param package_output Required path for the captured package JSON.
#' @param cli Path to `dag-ml-cli` or a command on `PATH`.
#' @param output Optional outcome JSON path; temporary by default.
#' @param persistent Keep the operator adapter alive across tasks.
#' @param process_workers Positive number of adapter workers.
#' @param process_timeout_ms Positive adapter timeout in milliseconds.
#' @param process_retries Non-negative retry count.
#' @param package_id,plan_id,run_id Native identifiers for capture lineage.
#' @param root_seed Non-negative native seed.
#' @param scheduler Either `sequential` or `parallel`.
#' @param scheduler_workers,cpu_threads Positive resource counts.
#' @param gpu_devices Optional GPU device identifiers.
#' @return Native REFIT outcome decoded as an R list.
#' @export
dagml_initial_full_refit <- function(
    dsl, controllers, envelope, training_sample_ids, adapter, package_output,
    cli = "dag-ml-cli", output = NULL, persistent = FALSE,
    process_workers = 1L, process_timeout_ms = 30000L, process_retries = 0L,
    package_id = "package:cli.process.dsl.initial.refit",
    plan_id = "plan:cli.process.dsl.refit.phase",
    run_id = "run:cli.process.dsl.refit.phase", root_seed = 12345L,
    scheduler = "sequential", scheduler_workers = 1L, cpu_threads = 1L,
    gpu_devices = character()) {
  inputs <- c(
    "--dsl", dagml_initial_refit_path(dsl, "dsl"),
    "--controllers", dagml_initial_refit_path(controllers, "controllers"),
    "--envelope", dagml_initial_refit_path(envelope, "envelope"),
    "--training-sample-ids", dagml_initial_refit_path(training_sample_ids, "training_sample_ids"),
    "--adapter", dagml_initial_refit_path(adapter, "adapter")
  )
  package_output <- dagml_initial_refit_path(package_output, "package_output", FALSE)
  if (!is.character(scheduler) || length(scheduler) != 1L ||
      is.na(scheduler) || !(scheduler %in% c("sequential", "parallel"))) {
    stop("scheduler must be sequential or parallel", call. = FALSE)
  }
  if (!is.character(gpu_devices) || anyNA(gpu_devices)) {
    stop("gpu_devices must be character identifiers", call. = FALSE)
  }
  if (length(gpu_devices)) {
    gpu_devices <- vapply(gpu_devices, dagml_initial_refit_path,
                          character(1), label = "gpu device", must_exist = FALSE)
  }
  extras <- c(
    "--package-output", package_output,
    "--package-id", dagml_initial_refit_path(package_id, "package_id", FALSE),
    "--plan-id", dagml_initial_refit_path(plan_id, "plan_id", FALSE),
    "--run-id", dagml_initial_refit_path(run_id, "run_id", FALSE),
    "--root-seed", dagml_initial_refit_count(root_seed, "root_seed", 0L),
    "--scheduler", scheduler,
    "--scheduler-workers", dagml_initial_refit_count(scheduler_workers, "scheduler_workers", 1L),
    "--cpu-threads", dagml_initial_refit_count(cpu_threads, "cpu_threads", 1L),
    unlist(lapply(gpu_devices, function(device) c("--gpu-device", device)),
           use.names = FALSE)
  )
  result <- dagml_initial_refit_run(
    "run-process-dsl-refit-phase", inputs, extras, cli, output, persistent,
    process_workers, process_timeout_ms, process_retries
  )
  if (!file.exists(package_output)) {
    stop("dag-ml REFIT exited without writing its package", call. = FALSE)
  }
  if (is.null(result$initial_full_refit_package)) {
    stop("dag-ml REFIT outcome lacks initial_full_refit_package", call. = FALSE)
  }
  result
}

#' Replay PREDICT from a no-CV full REFIT package
#'
#' The native CLI validates the package and V2 prediction cohort, then asks the
#' host adapter to resolve exactly the package's host-sidecar artifact handles.
#'
#' @param package,envelope,adapter,artifact_handles,output_ids JSON/package or
#'   executable adapter paths required by the native replay command.
#' @param cli Path to `dag-ml-cli` or a command on `PATH`.
#' @param output Optional outcome JSON path; temporary by default.
#' @param persistent Keep the operator adapter alive across tasks.
#' @param process_workers Positive number of adapter workers.
#' @param process_timeout_ms Positive adapter timeout in milliseconds.
#' @param process_retries Non-negative retry count.
#' @param run_id Native replay run identifier.
#' @return Native PREDICT replay outcome decoded as an R list.
#' @export
dagml_initial_full_refit_predict <- function(
    package, envelope, adapter, artifact_handles, output_ids,
    cli = "dag-ml-cli", output = NULL, persistent = FALSE,
    process_workers = 1L, process_timeout_ms = 30000L, process_retries = 0L,
    run_id = "run:cli.initial.refit.predict") {
  inputs <- c(
    "--package", dagml_initial_refit_path(package, "package"),
    "--envelope", dagml_initial_refit_path(envelope, "envelope"),
    "--adapter", dagml_initial_refit_path(adapter, "adapter"),
    "--artifact-handles", dagml_initial_refit_path(artifact_handles, "artifact_handles"),
    "--output-ids", dagml_initial_refit_path(output_ids, "output_ids")
  )
  result <- dagml_initial_refit_run(
    "run-process-initial-full-refit-predict", inputs,
    c("--run-id", dagml_initial_refit_path(run_id, "run_id", FALSE)),
    cli, output, persistent, process_workers, process_timeout_ms, process_retries
  )
  if (is.null(result$replay_outcome)) {
    stop("dag-ml PREDICT outcome lacks replay_outcome", call. = FALSE)
  }
  result
}

dagml_initial_refit_path <- function(value, label, must_exist = TRUE) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(trimws(value)) || grepl("[\r\n]", value)) {
    stop(sprintf("%s must be one non-empty path without control characters", label), call. = FALSE)
  }
  if (must_exist && !file.exists(value)) {
    stop(sprintf("%s does not exist: %s", label, value), call. = FALSE)
  }
  value
}

dagml_initial_refit_count <- function(value, label, minimum) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value < minimum || value != floor(value)) {
    stop(sprintf("%s must be an integer >= %d", label, minimum), call. = FALSE)
  }
  format(value, scientific = FALSE, trim = TRUE)
}

dagml_initial_refit_run <- function(command, inputs, extras, cli, output,
                                    persistent, workers, timeout, retries) {
  cli <- dagml_initial_refit_path(cli, "cli", FALSE)
  if (!is.logical(persistent) || length(persistent) != 1L || is.na(persistent)) {
    stop("persistent must be TRUE or FALSE", call. = FALSE)
  }
  workers <- dagml_initial_refit_count(workers, "process_workers", 1L)
  timeout <- dagml_initial_refit_count(timeout, "process_timeout_ms", 1L)
  retries <- dagml_initial_refit_count(retries, "process_retries", 0L)
  temporary_output <- is.null(output)
  if (temporary_output) {
    output <- tempfile("dagml-initial-refit-", fileext = ".json")
    on.exit(unlink(output), add = TRUE)
  } else {
    output <- dagml_initial_refit_path(output, "output", FALSE)
  }
  args <- c(
    command, vapply(inputs, shQuote, character(1)),
    vapply(extras, shQuote, character(1)),
    "--process-workers", workers, "--process-timeout-ms", timeout,
    "--process-retries", retries, "--output", shQuote(output)
  )
  if (persistent) args <- c(args, "--persistent")
  messages <- suppressWarnings(system2(cli, args = args,
                                       stdout = TRUE, stderr = TRUE, wait = TRUE))
  status <- attr(messages, "status")
  if (!is.null(status) && status != 0L) {
    stop(sprintf("dag-ml %s failed (exit %d): %s", command, status,
                 paste(messages, collapse = "\n")), call. = FALSE)
  }
  if (!file.exists(output)) {
    stop(sprintf("dag-ml %s exited without writing an outcome", command), call. = FALSE)
  }
  jsonlite::fromJSON(output, simplifyVector = FALSE, bigint_as_char = TRUE)
}
