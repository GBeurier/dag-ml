#!/usr/bin/env Rscript
# Real R Ridge sidecar for the native selected-candidate REFIT/PREDICT oracle.
suppressPackageStartupMessages(library(jsonlite))

empty_object <- structure(list(), names = character())
args <- commandArgs(trailingOnly = TRUE)
if (identical(args, "--describe")) {
  cat(toJSON(list(
    schema_version = 1L, protocol = "dag-ml-process-adapter",
    adapter_id = "dag-ml-r-ridge-refit-oracle",
    supported_modes = list("one_shot", "jsonl"),
    capabilities = list("node_task_json_v1", "node_result_json_v1",
                        "stateful_refit_artifacts")
  ), auto_unbox = TRUE), "\n", sep = "")
  quit(save = "no")
}

rows <- read.csv(Sys.getenv("DAGML_R_HPO_DATA"), stringsAsFactors = FALSE)
sidecar <- Sys.getenv("DAGML_R_RIDGE_SIDECAR")
stopifnot(nzchar(sidecar), !anyDuplicated(rows$id),
          all(c("id", "x", "y") %in% names(rows)))

view_ids <- function(task, partition) {
  views <- Filter(function(view) identical(view$partition, partition), task$data_views)
  stopifnot(length(views) == 1L)
  ids <- unlist(views[[1L]]$sample_ids, use.names = FALSE)
  stopifnot(length(ids) > 0L, !anyDuplicated(ids))
  ids
}

emit_task <- function(task, raw_line) {
  seed_tokens <- regmatches(raw_line, gregexpr('"seed"[[:space:]]*:[[:space:]]*[0-9]+', raw_line))[[1L]]
  stopifnot(length(seed_tokens) > 0L)
  seed_decimal <- sub('.*:', '', tail(seed_tokens, 1L))
  phase <- task$phase
  stopifnot(phase %in% c("REFIT", "PREDICT"))
  node <- task$node_plan$node_id
  controller <- task$node_plan$controller_id
  artifact_id <- paste0("artifact:", node, ":r-ridge:refit")
  alpha <- as.numeric(task$node_plan$params$n_components) - 1
  stopifnot(is.finite(alpha), alpha >= 0)

  if (identical(phase, "REFIT")) {
    train_ids <- view_ids(task, "full_train")
    train <- rows[match(train_ids, rows$id), , drop = FALSE]
    stopifnot(!anyNA(train$id))
    model <- list(weight = sum(train$x * train$y) /
                    (sum(train$x * train$x) + alpha), alpha = alpha,
                  train_ids = train_ids)
    dir.create(dirname(sidecar), recursive = TRUE, showWarnings = FALSE)
    saveRDS(model, sidecar)
    artifact <- list(
      id = artifact_id, kind = "r_ridge_model", controller_id = controller,
      backend = "rds", uri = "artifacts/r-ridge.rds",
      content_fingerprint = digest::digest(sidecar, algo = "sha256", file = TRUE),
      size_bytes = as.integer(file.info(sidecar)$size),
      plugin = "dagml.r_ridge_oracle", plugin_version = "1.0.0"
    )
    artifacts <- list(artifact)
    artifact_handles <- setNames(list(list(
      handle = 7001L, kind = "model", owner_controller = controller
    )), artifact_id)
    ids <- train_ids
    partition <- "final"
  } else {
    handles <- Filter(function(handle) identical(handle$kind, "model"), task$input_handles)
    stopifnot(length(handles) == 1L)
    inputs <- task$artifact_inputs
    stopifnot(length(inputs) == 1L)
    input <- inputs[[1L]]
    stopifnot(identical(input$node_id, node),
              identical(input$controller_id, controller),
              identical(input$artifact$id, artifact_id),
              identical(input$artifact$backend, "rds"),
              identical(input$artifact$uri, "artifacts/r-ridge.rds"),
              identical(input$artifact$content_fingerprint,
                        digest::digest(sidecar, algo = "sha256", file = TRUE)))
    model <- readRDS(sidecar)
    stopifnot(identical(model$alpha, alpha))
    artifacts <- list()
    artifact_handles <- empty_object
    ids <- view_ids(task, "predict")
    partition <- "final"
  }
  selected <- rows[match(ids, rows$id), , drop = FALSE]
  stopifnot(!anyNA(selected$id))
  predictions <- model$weight * selected$x
  result <- list(
    node_id = node,
    outputs = list(oof = list(handle = 7002L, kind = "prediction",
                              owner_controller = controller)),
    predictions = list(list(
      producer_node = node, partition = partition, fold_id = NULL,
      sample_ids = as.list(ids), values = lapply(predictions, function(value) list(as.numeric(value))),
      target_names = list("y")
    )),
    artifacts = artifacts, artifact_handles = artifact_handles,
    lineage = list(
      record_id = paste("lineage:r-ridge", phase, sep = ":"),
      run_id = task$run_id, node_id = node, phase = phase,
      controller_id = controller,
      controller_version = task$node_plan$controller_version,
      variant_id = task$variant_id, fold_id = task$fold_id,
      branch_path = task$branch_path, input_lineage = list(),
      artifact_refs = artifacts, params_fingerprint = task$node_plan$params_fingerprint,
      data_model_shape_fingerprint = NULL,
      aggregation_policy_fingerprint = NULL,
      seed = "__DAGML_U64_SEED__", unsafe_flags = list(), metrics = empty_object,
      loss_attestations = list(), early_stopping_records = list()
    )
  )
  if (identical(phase, "REFIT")) {
    result$regression_targets <- list(list(
      level = "sample", unit_ids = lapply(ids, function(id) list(level = "sample", id = id)),
      values = lapply(selected$y, function(value) list(as.numeric(value))),
      target_names = list("y")
    ))
  }
  encoded <- toJSON(result, auto_unbox = TRUE, null = "null", digits = 17)
  encoded <- sub('"seed":"__DAGML_U64_SEED__"', paste0('"seed":', seed_decimal),
                 encoded, fixed = TRUE)
  cat(encoded, "\n", sep = "")
  flush(stdout())
}

input <- file("stdin", open = "r")
repeat {
  line <- readLines(input, n = 1L, warn = FALSE)
  if (!length(line)) break
  emit_task(fromJSON(line, simplifyVector = FALSE), line)
  if (!identical(args, "--jsonl")) break
}
close(input)
