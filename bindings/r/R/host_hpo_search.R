#' Run scheduler-owned host hyperparameter search
#'
#' This calls `dag-ml-cli run-host-hpo`. DAG-ML owns candidate scheduling,
#' folds, scoring, pruning, selection, and the native checkpoint. The two
#' executable JSONL adapters own R operators and optimizer proposals. Each
#' adapter process maintains its own state; R callbacks are never invoked from
#' a native worker thread.
#'
#' @param plan,envelope,request Paths to DAG-ML JSON contracts.
#' @param operator_adapter,optimizer_adapter Executable process adapters.
#' @param cli Path to `dag-ml-cli` (or a command on `PATH`).
#' @param checkpoint Optional native checkpoint path for durable resume.
#' @param output Optional result JSON path. A temporary file is used otherwise.
#' @param operator_persistent Keep an operator process per candidate across folds.
#' @param parallel_trials Must be 1. Parallel R adapters require isolated worker
#'   processes and are not exposed through this wrapper yet.
#' @param adapter_timeout_ms Positive adapter response timeout.
#' @return The native HPO result decoded as a list, without simplifying arrays.
#' @export
dagml_host_hpo_search <- function(
    plan, envelope, request, operator_adapter, optimizer_adapter,
    cli = "dag-ml-cli", checkpoint = NULL, output = NULL,
    operator_persistent = FALSE, parallel_trials = 1L,
    adapter_timeout_ms = 30000L) {
  scalar_path <- function(value, label, must_exist = TRUE) {
    if (!is.character(value) || length(value) != 1L || is.na(value) ||
        !nzchar(trimws(value))) {
      stop(sprintf("%s must be one non-empty path", label), call. = FALSE)
    }
    if (must_exist && !file.exists(value)) {
      stop(sprintf("%s does not exist: %s", label, value), call. = FALSE)
    }
    value
  }
  plan <- scalar_path(plan, "plan")
  envelope <- scalar_path(envelope, "envelope")
  request <- scalar_path(request, "request")
  operator_adapter <- scalar_path(operator_adapter, "operator_adapter")
  optimizer_adapter <- scalar_path(optimizer_adapter, "optimizer_adapter")
  cli <- scalar_path(cli, "cli", must_exist = FALSE)
  if (!is.numeric(parallel_trials) || length(parallel_trials) != 1L ||
      is.na(parallel_trials) || parallel_trials != 1) {
    stop("R host HPO currently supports parallel_trials=1 only; use the CLI directly with isolated R adapter workers for parallel trials", call. = FALSE)
  }
  if (!is.numeric(adapter_timeout_ms) || length(adapter_timeout_ms) != 1L ||
      is.na(adapter_timeout_ms) || !is.finite(adapter_timeout_ms) ||
      adapter_timeout_ms < 1 || adapter_timeout_ms != floor(adapter_timeout_ms)) {
    stop("adapter_timeout_ms must be a positive integer", call. = FALSE)
  }
  if (!is.logical(operator_persistent) || length(operator_persistent) != 1L ||
      is.na(operator_persistent)) {
    stop("operator_persistent must be TRUE or FALSE", call. = FALSE)
  }
  temporary_output <- is.null(output)
  if (temporary_output) {
    output <- tempfile("dagml-host-hpo-", fileext = ".json")
    on.exit(unlink(output), add = TRUE)
  } else {
    output <- scalar_path(output, "output", must_exist = FALSE)
  }
  args <- c(
    "run-host-hpo", "--plan", shQuote(plan), "--envelope", shQuote(envelope),
    "--request", shQuote(request), "--operator-adapter", shQuote(operator_adapter),
    "--optimizer-adapter", shQuote(optimizer_adapter), "--parallel-trials", "1",
    "--adapter-timeout-ms", format(adapter_timeout_ms, scientific = FALSE, trim = TRUE),
    "--output", shQuote(output)
  )
  if (operator_persistent) args <- c(args, "--operator-persistent")
  if (!is.null(checkpoint)) {
    checkpoint <- scalar_path(checkpoint, "checkpoint", must_exist = FALSE)
    args <- c(args, "--checkpoint", shQuote(checkpoint))
  }
  messages <- suppressWarnings(system2(shQuote(cli), args = args, stdout = TRUE, stderr = TRUE, wait = TRUE))
  status <- attr(messages, "status")
  if (!is.null(status) && status != 0L) {
    stop(sprintf("dag-ml host HPO failed (exit %d): %s", status, paste(messages, collapse = "\n")), call. = FALSE)
  }
  if (!file.exists(output)) {
    stop("dag-ml host HPO exited without writing a result", call. = FALSE)
  }
  jsonlite::fromJSON(output, simplifyVector = FALSE, bigint_as_char = TRUE)
}
