#' Run CV, REFIT and PREDICT in one native DAG-ML session
#'
#' The CLI retains host model handles across phases and writes native bundle,
#' OOF averages, replay predictions and scores as one JSON outcome. This is an
#' in-process replay; persisting a bundle alone does not persist host models.
#'
#' @param dsl,controllers,envelope JSON input file paths.
#' @param adapter Executable operator process adapter.
#' @param cli Path to `dag-ml-cli` or a command on `PATH`.
#' @param output Optional outcome JSON path; temporary by default.
#' @param process_workers,process_timeout_ms Positive process settings.
#' @param process_retries Non-negative retry count.
#' @param bundle_id,plan_id,run_id Native identifiers.
#' @param variant_id Optional concrete variant ID.
#' @param selection_metric One of `rmse`, `accuracy`, `balanced_accuracy`.
#' @param selections Optional selection-decision JSON path.
#' @param root_seed Non-negative native seed.
#' @param scheduler Either `sequential` or `parallel`.
#' @param scheduler_workers,cpu_threads Positive resource counts.
#' @param gpu_devices Optional GPU device identifiers.
#' @return Native CV, REFIT and PREDICT outcome decoded as an R list.
#' @export
dagml_cv_refit_predict <- function(
    dsl, controllers, envelope, adapter, cli = "dag-ml-cli", output = NULL,
    process_workers = 1L, process_timeout_ms = 30000L, process_retries = 0L,
    bundle_id = "bundle:cli.process.dsl.cv.refit.replay", variant_id = NULL,
    selection_metric = "rmse", selections = NULL,
    plan_id = "plan:cli.process.dsl.cv.refit.replay",
    run_id = "run:cli.process.dsl.cv.refit.replay", root_seed = 12345L,
    scheduler = "sequential", scheduler_workers = 1L, cpu_threads = 1L,
    gpu_devices = character()) {
  inputs <- c(
    "--dsl", dagml_initial_refit_path(dsl, "dsl"),
    "--controllers", dagml_initial_refit_path(controllers, "controllers"),
    "--envelope", dagml_initial_refit_path(envelope, "envelope"),
    "--adapter", dagml_initial_refit_path(adapter, "adapter")
  )
  if (!is.character(selection_metric) || length(selection_metric) != 1L ||
      is.na(selection_metric) ||
      !(selection_metric %in% c("rmse", "accuracy", "balanced_accuracy"))) {
    stop("selection_metric must be rmse, accuracy or balanced_accuracy", call. = FALSE)
  }
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
    "--bundle-id", dagml_initial_refit_path(bundle_id, "bundle_id", FALSE),
    "--plan-id", dagml_initial_refit_path(plan_id, "plan_id", FALSE),
    "--run-id", dagml_initial_refit_path(run_id, "run_id", FALSE),
    "--selection-metric", selection_metric,
    "--root-seed", dagml_initial_refit_count(root_seed, "root_seed", 0L),
    "--scheduler", scheduler,
    "--scheduler-workers", dagml_initial_refit_count(scheduler_workers, "scheduler_workers", 1L),
    "--cpu-threads", dagml_initial_refit_count(cpu_threads, "cpu_threads", 1L),
    unlist(lapply(gpu_devices, function(device) c("--gpu-device", device)),
           use.names = FALSE)
  )
  if (!is.null(variant_id)) {
    extras <- c(extras, "--variant-id",
                dagml_initial_refit_path(variant_id, "variant_id", FALSE))
  }
  if (!is.null(selections)) {
    extras <- c(extras, "--selections",
                dagml_initial_refit_path(selections, "selections"))
  }
  result <- dagml_initial_refit_run(
    "run-process-dsl-cv-refit-replay", inputs, extras,
    cli, output, FALSE, process_workers, process_timeout_ms, process_retries
  )
  if (is.null(result$bundle) || is.null(result$replay_node_results) ||
      is.null(result$replay_prediction_blocks)) {
    stop("dag-ml CV+REFIT+PREDICT outcome lacks bundle or replay evidence", call. = FALSE)
  }
  result
}
