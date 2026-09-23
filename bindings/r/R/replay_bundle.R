#' Replay a persisted CV bundle with host-owned model sidecars
#'
#' The caller must load model sidecars into its adapter's invocation-local
#' handle namespace and provide exactly one handle per bundle REFIT artifact.
#' The native CLI validates the bundle, handles and replay envelope identities.
#'
#' @param graph,campaign,controllers,bundle,replay_request,adapter,artifact_handles
#'   Required JSON or executable adapter paths.
#' @param envelopes Named character vector mapping replay data keys to V2
#'   envelope JSON paths.
#' @param cli Path to `dag-ml-cli` or a command on `PATH`.
#' @param output Optional outcome JSON path; temporary by default.
#' @param persistent Keep the adapter alive across tasks.
#' @param process_workers,process_timeout_ms Positive process settings.
#' @param process_retries Non-negative retry count.
#' @param prediction_cache_payload,prediction_cache_store Optional native
#'   prediction cache sources for OOF-dependent replay.
#' @param score_output Optional score JSON destination.
#' @param plan_id,run_id Native identifiers.
#' @param root_seed Non-negative native seed.
#' @param scheduler Either `sequential` or `parallel`.
#' @param scheduler_workers Positive scheduler worker count.
#' @return Native replay outcome decoded as an R list.
#' @export
dagml_replay_bundle <- function(
    graph, campaign, controllers, bundle, replay_request, adapter,
    artifact_handles, envelopes, cli = "dag-ml-cli", output = NULL,
    persistent = FALSE, process_workers = 1L,
    process_timeout_ms = 30000L, process_retries = 0L,
    prediction_cache_payload = NULL, prediction_cache_store = NULL,
    score_output = NULL, plan_id = "plan:cli.bundle",
    run_id = "run:cli.process.replay", root_seed = 12345L,
    scheduler = "sequential", scheduler_workers = 1L) {
  inputs <- c(
    "--graph", dagml_initial_refit_path(graph, "graph"),
    "--campaign", dagml_initial_refit_path(campaign, "campaign"),
    "--controllers", dagml_initial_refit_path(controllers, "controllers"),
    "--bundle", dagml_initial_refit_path(bundle, "bundle"),
    "--replay-request", dagml_initial_refit_path(replay_request, "replay_request"),
    "--adapter", dagml_initial_refit_path(adapter, "adapter"),
    "--artifact-handles", dagml_initial_refit_path(artifact_handles, "artifact_handles")
  )
  if (!is.character(envelopes) || !length(envelopes) ||
      is.null(names(envelopes)) || anyNA(envelopes) ||
      anyNA(names(envelopes)) || any(!nzchar(names(envelopes))) ||
      anyDuplicated(names(envelopes)) || any(grepl("=", names(envelopes), fixed = TRUE))) {
    stop("envelopes must be a non-empty named character vector with unique keys", call. = FALSE)
  }
  envelope_args <- unlist(lapply(seq_along(envelopes), function(index) {
    c("--envelope", paste0(
      names(envelopes)[index], "=",
      dagml_initial_refit_path(envelopes[[index]], "envelope")
    ))
  }), use.names = FALSE)
  if (!is.character(scheduler) || length(scheduler) != 1L ||
      is.na(scheduler) || !(scheduler %in% c("sequential", "parallel"))) {
    stop("scheduler must be sequential or parallel", call. = FALSE)
  }
  extras <- c(
    envelope_args,
    "--plan-id", dagml_initial_refit_path(plan_id, "plan_id", FALSE),
    "--run-id", dagml_initial_refit_path(run_id, "run_id", FALSE),
    "--root-seed", dagml_initial_refit_count(root_seed, "root_seed", 0L),
    "--scheduler", scheduler,
    "--scheduler-workers", dagml_initial_refit_count(scheduler_workers, "scheduler_workers", 1L)
  )
  if (!is.null(prediction_cache_payload)) {
    extras <- c(extras, "--prediction-cache-payload",
                dagml_initial_refit_path(prediction_cache_payload, "prediction_cache_payload"))
  }
  if (!is.null(prediction_cache_store)) {
    extras <- c(extras, "--prediction-cache-store",
                dagml_initial_refit_path(prediction_cache_store, "prediction_cache_store"))
  }
  if (!is.null(score_output)) {
    extras <- c(extras, "--score-output",
                dagml_initial_refit_path(score_output, "score_output", FALSE))
  }
  result <- dagml_initial_refit_run(
    "run-process-replay", inputs, extras, cli, output,
    persistent, process_workers, process_timeout_ms, process_retries
  )
  if (is.null(result$bundle_id) || is.null(result$node_results) ||
      is.null(result$prediction_blocks)) {
    stop("dag-ml bundle replay outcome lacks native prediction evidence", call. = FALSE)
  }
  result
}
