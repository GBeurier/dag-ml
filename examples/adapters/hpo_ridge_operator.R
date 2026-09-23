#!/usr/bin/env Rscript
# Small real R operator for the native host-HPO cross-language oracle.
# The plan and the two-row numeric table are supplied by the test harness.
suppressPackageStartupMessages(library(jsonlite))

empty_object <- structure(list(), names = character())
args <- commandArgs(trailingOnly = TRUE)
if (identical(args, "--describe")) {
  description <- list(
    schema_version = 1L,
    protocol = "dag-ml-process-adapter",
    adapter_id = "dag-ml-r-ridge-hpo-oracle",
    supported_modes = list("one_shot", "jsonl"),
    capabilities = list(
      "control_frames_v1", "node_task_json_v1", "node_result_json_v1",
      "parallel_invocation_v1", "persistent_workers", "worker_env"
    )
  )
  cat(toJSON(description, auto_unbox = TRUE), "\n", sep = "")
  quit(save = "no")
}

plan <- fromJSON(Sys.getenv("DAGML_R_HPO_PLAN"), simplifyVector = FALSE)
rows <- read.csv(Sys.getenv("DAGML_R_HPO_DATA"), stringsAsFactors = FALSE)
stopifnot(!anyDuplicated(rows$id), all(c("id", "x", "y") %in% names(rows)))

emit_task <- function(task, raw_line) {
  # jsonlite/R numeric values cannot represent every DAG-ML u64 seed exactly.
  # Retain the attested decimal token from NodeTask JSON for lineage output.
  seed_tokens <- regmatches(raw_line, gregexpr('"seed"[[:space:]]*:[[:space:]]*[0-9]+', raw_line))[[1L]]
  stopifnot(length(seed_tokens) > 0L)
  seed_decimal <- sub('.*:', '', tail(seed_tokens, 1L))
  stopifnot(identical(task$phase, "FIT_CV"))
  fold_id <- task$fold_id
  folds <- Filter(function(fold) identical(fold$fold_id, fold_id), plan$fold_set$folds)
  stopifnot(length(folds) == 1L)
  fold <- folds[[1L]]
  train_ids <- unlist(fold$train_sample_ids, use.names = FALSE)
  validation_ids <- unlist(fold$validation_sample_ids, use.names = FALSE)
  stopifnot(length(train_ids) > 0L, length(validation_ids) > 0L,
            !any(train_ids %in% validation_ids))
  train <- rows[match(train_ids, rows$id), , drop = FALSE]
  validation <- rows[match(validation_ids, rows$id), , drop = FALSE]
  stopifnot(!anyNA(train$id), !anyNA(validation$id))
  alpha <- as.numeric(task$node_plan$params$n_components) - 1
  stopifnot(is.finite(alpha), alpha >= 0)
  weight <- sum(train$x * train$y) / (sum(train$x * train$x) + alpha)
  predictions <- weight * validation$x
  evidence_dir <- Sys.getenv("DAGML_R_HPO_EVIDENCE_DIR")
  if (nzchar(evidence_dir)) {
    evidence <- list(
      train_ids = as.list(train_ids), validation_ids = as.list(validation_ids),
      predictions = as.list(predictions), targets = as.list(validation$y),
      alpha = alpha
    )
    filename <- paste0(gsub(":", "_", task$variant_id), "_",
                       gsub(":", "_", fold_id), ".json")
    writeLines(toJSON(evidence, auto_unbox = TRUE, digits = 17),
               file.path(evidence_dir, filename))
  }
  node <- task$node_plan$node_id
  variant <- task$variant_id
  result <- list(
    node_id = node, outputs = empty_object,
    predictions = list(list(
      producer_node = node, partition = "validation", fold_id = fold_id,
      sample_ids = as.list(validation_ids),
      values = lapply(predictions, function(value) list(as.numeric(value))),
      target_names = list("y")
    )),
    regression_targets = list(list(
      level = "sample",
      unit_ids = lapply(validation_ids, function(id) list(level = "sample", id = id)),
      values = lapply(validation$y, function(value) list(as.numeric(value))),
      target_names = list("y")
    )),
    lineage = list(
      record_id = paste("lineage:r-ridge", variant, fold_id, sep = ":"),
      run_id = task$run_id, node_id = node, phase = task$phase,
      controller_id = task$node_plan$controller_id,
      controller_version = task$node_plan$controller_version,
      variant_id = variant, fold_id = fold_id,
      branch_path = task$branch_path, input_lineage = list(),
      artifact_refs = list(), params_fingerprint = task$node_plan$params_fingerprint,
      data_model_shape_fingerprint = NULL,
      aggregation_policy_fingerprint = NULL,
      seed = "__DAGML_U64_SEED__", unsafe_flags = list(), metrics = empty_object,
      loss_attestations = list(), early_stopping_records = list()
    )
  )
  encoded <- toJSON(result, auto_unbox = TRUE, null = "null", digits = 17)
  encoded <- sub('"seed":"__DAGML_U64_SEED__"',
                 paste0('"seed":', seed_decimal), encoded, fixed = TRUE)
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
