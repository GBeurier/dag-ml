#!/usr/bin/env Rscript
# Process adapter for the mdatools R package, speaking the dag-ml
# coordinator's JSONL protocol.
#
# Slice H.1 covers the model-kind scaffold plus dispatch for `pls`
# (Partial Least Squares regression) with `saveRDS`/`readRDS`-backed
# artifact persistence (R analog of sklearn's joblib). Remaining
# mdatools operators (`pca`, `plsda`, `simca`, `mcrals`) and the
# ControllerManifest are delivered in Slice H.2.
#
# Reuses the JSONL framing, structured `AdapterTaskError` condition,
# leakage checks, lifecycle markers and synthetic feature pattern
# from `prospectr_process_controller.R` (G.1). The new pattern this
# slice introduces is RData-backed stateful artifact persistence
# under `$DAG_ML_PROCESS_ARTIFACT_DIR`, mirroring the basename
# confinement used by the sklearn production controller (F.1).

PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION <- 1L
PROCESS_ADAPTER_PROTOCOL <- "dag-ml-process-adapter"
PROCESS_ADAPTER_MODES <- c("one_shot", "jsonl")
PROCESS_ADAPTER_FRAME_SCHEMA_VERSION <- 1L
PROCESS_ADAPTER_CAPABILITIES <- c(
  "control_frames_v1",
  "node_task_json_v1",
  "node_result_json_v1",
  "parallel_invocation_v1",
  "persistent_workers",
  "worker_env",
  "stateful_refit_artifacts",
  "mdatools_smoke"
)
ADAPTER_ID <- "dag-ml-mdatools-process-controller"
ADAPTER_PLUGIN <- "dagml.mdatools_process"
ADAPTER_PLUGIN_VERSION <- "1.0.0"
ARTIFACT_DIR_ENV <- "DAG_ML_PROCESS_ARTIFACT_DIR"
DEFAULT_ARTIFACT_DIR <- "artifacts"

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1L && identical(args[1], "--describe")) {
  suppressPackageStartupMessages(library(jsonlite))
  description <- list(
    schema_version = PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION,
    protocol = PROCESS_ADAPTER_PROTOCOL,
    adapter_id = ADAPTER_ID,
    supported_modes = PROCESS_ADAPTER_MODES,
    capabilities = sort(unique(PROCESS_ADAPTER_CAPABILITIES))
  )
  cat(toJSON(description, auto_unbox = TRUE), "\n", sep = "")
  flush.console()
  quit(save = "no", status = 0L)
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(mdatools)
  library(digest)
})

# Operator registry. mdatools 0.15 ships `pls`, `pca`, `plsda`,
# `simca`, `mcrals` (renamed from `mcr.als` in older versions). No
# `pcr` top-level function exists in 0.15; users wanting PCR build it
# through `pca` + linear regression manually.
#
# Slice H.1 only wires `pls`. Remaining operators land in Slice H.2.
OPERATOR_SELECTORS <- list(
  pls = list(pkg = "mdatools", fn = "pls")
)

AdapterTaskError <- function(code, message, retryable = FALSE) {
  structure(
    class = c("AdapterTaskError", "error", "condition"),
    list(message = message, code = code, retryable = retryable)
  )
}

fail <- function(message, code = "adapter_fail", retryable = FALSE) {
  stop(AdapterTaskError(code, message, retryable))
}

emit_json <- function(payload) {
  cat(toJSON(payload, auto_unbox = TRUE, null = "null"), "\n", sep = "")
  flush.console()
}

emit_ack <- function(status) {
  emit_json(list(
    type = "ack",
    schema_version = PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
    status = status
  ))
}

emit_error <- function(code, message, retryable = FALSE) {
  emit_json(list(
    type = "error",
    schema_version = PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
    error = list(code = code, message = message, retryable = retryable)
  ))
}

resolve_operator <- function(name) {
  if (!is.character(name) || length(name) != 1L) {
    fail("`operator` must be a single string", code = "unknown_operator")
  }
  selector <- OPERATOR_SELECTORS[[name]]
  if (is.null(selector)) {
    fail(
      sprintf("unknown operator `%s`; not in OPERATOR_SELECTORS registry", name),
      code = "unknown_operator"
    )
  }
  ns <- tryCatch(asNamespace(selector$pkg), error = function(e) {
    fail(
      sprintf("operator `%s` namespace `%s` is not loadable: %s", name, selector$pkg, conditionMessage(e)),
      code = "unknown_operator"
    )
  })
  if (!exists(selector$fn, envir = ns, inherits = FALSE)) {
    fail(
      sprintf("operator `%s` is not exported by `%s`", name, selector$pkg),
      code = "unknown_operator"
    )
  }
  get(selector$fn, envir = ns)
}

stable_handle <- function(value) {
  bytes <- as.integer(charToRaw(value))
  acc <- 17
  modulus <- 2147483647
  for (b in bytes) {
    acc <- (acc * 31 + b) %% modulus
  }
  if (acc == 0) 1L else as.integer(acc)
}

sample_scalar <- function(sample_id) {
  (stable_handle(sample_id) %% 10000L) / 10000
}

features <- function(sample_ids) {
  x <- vapply(sample_ids, sample_scalar, numeric(1))
  cbind(x, x * x, sin(pi * x), cos(pi * x))
}

targets <- function(sample_ids) {
  x <- vapply(sample_ids, sample_scalar, numeric(1))
  1.7 * x - 0.3 * x * x + sin(pi * x) * 0.2
}

content_fingerprint <- function(value) {
  digest::digest(value, algo = "sha256", serialize = FALSE)
}

artifact_dir <- function() {
  Sys.getenv(ARTIFACT_DIR_ENV, unset = DEFAULT_ARTIFACT_DIR)
}

resolve_artifact_root <- function() {
  raw <- artifact_dir()
  # `normalizePath(..., mustWork=FALSE)` does not always return an
  # absolute path when the directory does not yet exist (the result
  # may stay relative). Anchor against `getwd()` to make the prefix
  # check below independent of whether the directory has been
  # created yet.
  if (!startsWith(raw, .Platform$file.sep)) {
    raw <- file.path(getwd(), raw)
  }
  normalizePath(raw, mustWork = FALSE)
}

artifact_path_for <- function(uri) {
  # `readRDS` deserializes arbitrary R objects; the URI is treated
  # as untrusted. Strip to basename, join under artifact_dir, then
  # assert the resolved path lives under artifact_dir. Mirrors the
  # F.1 sklearn controller's confinement.
  base <- basename(uri)
  if (!nzchar(base) || base %in% c(".", "..")) {
    fail(sprintf("refusing to resolve artifact uri `%s` — basename is empty or traversal", uri))
  }
  root <- resolve_artifact_root()
  resolved <- normalizePath(file.path(root, base), mustWork = FALSE)
  # Compare prefixes with a separator appended to avoid false matches
  # like /tmp/foo vs /tmp/foobar.
  root_with_sep <- if (endsWith(root, .Platform$file.sep)) root else paste0(root, .Platform$file.sep)
  if (!startsWith(paste0(resolved, .Platform$file.sep), root_with_sep) && resolved != root) {
    fail(sprintf("refusing to resolve artifact uri `%s` — outside artifact dir `%s`", uri, root))
  }
  resolved
}

require_data_handles <- function(task) {
  node_plan <- task$node_plan
  input_handles <- task$input_handles
  data_views <- task$data_views
  bindings <- node_plan$data_bindings
  if (is.null(bindings)) bindings <- list()
  for (binding in bindings) {
    key <- paste0("data:", binding$input_name)
    handle <- input_handles[[key]]
    if (is.null(handle)) {
      fail(sprintf("node `%s` did not receive data handle `%s`", node_plan$node_id, key))
    }
    if (!(handle$kind %in% c("data", "data_view"))) {
      fail(sprintf("node `%s` received non-data/data-view handle `%s`", node_plan$node_id, key))
    }
    view <- data_views[[key]]
    if (is.null(view)) {
      fail(sprintf("node `%s` did not receive data view spec `%s`", node_plan$node_id, key))
    }
    phase <- task$phase
    fold_id <- task$fold_id
    if (!is.null(phase) && phase == "FIT_CV" && !is.null(fold_id)) {
      if (is.null(view$partition) || view$partition != "fold_train") {
        fail(sprintf("node `%s` received non-train fold view `%s`", node_plan$node_id, key))
      }
      validation_key <- paste0(key, ":validation")
      validation_view <- data_views[[validation_key]]
      if (is.null(validation_view) || is.null(validation_view$partition) ||
          validation_view$partition != "fold_validation") {
        fail(sprintf("node `%s` did not receive validation view `%s`", node_plan$node_id, validation_key))
      }
    }
    if (!is.null(phase) && phase == "REFIT" &&
        (is.null(view$partition) || view$partition != "full_train")) {
      fail(sprintf("node `%s` received non-full-train refit view `%s`", node_plan$node_id, key))
    }
    if (!is.null(phase) && phase == "PREDICT" &&
        (is.null(view$partition) || view$partition != "predict")) {
      fail(sprintf("node `%s` received non-predict replay view `%s`", node_plan$node_id, key))
    }
  }
}

data_view <- function(task, suffix = "") {
  bindings <- task$node_plan$data_bindings
  if (is.null(bindings) || length(bindings) == 0L) {
    if (nzchar(suffix)) {
      for (key in names(task$data_views)) {
        if (endsWith(key, suffix)) {
          return(task$data_views[[key]])
        }
      }
      return(NULL)
    }
    for (view in task$data_views) {
      if (is.null(view$partition) || view$partition != "fold_validation") {
        return(view)
      }
    }
    if (length(task$data_views) > 0L) {
      return(task$data_views[[1]])
    }
    return(NULL)
  }
  input_name <- bindings[[1]]$input_name
  task$data_views[[paste0("data:", input_name, suffix)]]
}

train_sample_ids <- function(task) {
  view <- data_view(task)
  if (is.null(view) || is.null(view$sample_ids) || length(view$sample_ids) == 0L) {
    return(c("sample:train:0", "sample:train:1", "sample:train:2", "sample:train:3"))
  }
  as.character(view$sample_ids)
}

prediction_sample_ids <- function(task) {
  phase <- task$phase
  if (!is.null(phase) && phase == "FIT_CV") {
    validation <- data_view(task, ":validation")
    if (is.null(validation) || is.null(validation$sample_ids) || length(validation$sample_ids) == 0L) {
      fail(sprintf("node `%s` validation view has no sample ids", task$node_plan$node_id))
    }
    return(as.character(validation$sample_ids))
  }
  if (!is.null(phase) && phase == "REFIT") {
    return(train_sample_ids(task))
  }
  view <- data_view(task)
  if (is.null(view) || is.null(view$sample_ids) || length(view$sample_ids) == 0L) {
    return(c("sample:predict:0", "sample:predict:1"))
  }
  as.character(view$sample_ids)
}

prediction_partition <- function(phase) {
  if (!is.null(phase) && phase == "FIT_CV") return("validation")
  if (!is.null(phase) && phase %in% c("REFIT", "PREDICT", "EXPLAIN")) return("final")
  "test"
}

build_estimator_args <- function(task, X, y) {
  params <- task$node_plan$params
  if (is.null(params) || is.null(params$operator)) {
    fail("node `params` missing `operator`")
  }
  kwargs <- if (is.null(params$params)) list() else as.list(params$params)
  if (!is.list(kwargs)) {
    fail("`params` for the operator must be an object")
  }
  # mdatools' built-in cross-validation conflicts with the dag-ml
  # scheduler's CV ownership. Force-disable internal CV to avoid
  # double-counting folds.
  kwargs$cv <- NULL
  c(list(x = X, y = y), kwargs)
}

write_artifact <- function(estimator, artifact_id, variant_label) {
  target_dir <- resolve_artifact_root()
  dir.create(target_dir, showWarnings = FALSE, recursive = TRUE)
  fingerprint <- content_fingerprint(paste(artifact_id, variant_label, sep = ":"))
  path <- file.path(target_dir, paste0(fingerprint, ".rds"))
  saveRDS(estimator, path)
  size_bytes <- file.info(path)$size
  list(uri = path, fingerprint = fingerprint, size_bytes = size_bytes)
}

replay_estimator <- function(task) {
  input_handles <- task$input_handles
  if (is.null(input_handles)) input_handles <- list()
  artifact_keys <- names(input_handles)[startsWith(names(input_handles), "artifact:")]
  if (length(artifact_keys) == 0L) {
    fail(sprintf("node `%s` did not receive replay artifact handle", task$node_plan$node_id))
  }
  key <- artifact_keys[1]
  handle <- input_handles[[key]]
  if (!grepl(task$node_plan$node_id, key, fixed = TRUE)) {
    fail(sprintf("node `%s` received artifact handle for another node `%s`", task$node_plan$node_id, key))
  }
  if (is.null(handle$kind) || !(handle$kind %in% c("model", "artifact"))) {
    fail(sprintf("node `%s` received invalid artifact handle `%s`", task$node_plan$node_id, key))
  }
  artifact_input <- task$artifact_inputs[[key]]
  if (is.null(artifact_input)) {
    fail(sprintf("node `%s` did not receive artifact metadata `%s`", task$node_plan$node_id, key))
  }
  if (is.null(artifact_input$node_id) ||
      artifact_input$node_id != task$node_plan$node_id ||
      is.null(artifact_input$controller_id) ||
      artifact_input$controller_id != task$node_plan$controller_id) {
    fail(sprintf("node `%s` received mismatched artifact metadata `%s`", task$node_plan$node_id, key))
  }
  uri <- artifact_input$uri
  if (is.null(uri) || !nzchar(uri)) {
    fail(sprintf("node `%s` artifact metadata `%s` has no uri", task$node_plan$node_id, key))
  }
  path <- artifact_path_for(uri)
  if (!file.exists(path)) {
    fail(sprintf(
      "node `%s` artifact uri `%s` resolved under artifact dir to `%s` which does not exist",
      task$node_plan$node_id, uri, path
    ))
  }
  readRDS(path)
}

run_model <- function(task) {
  phase <- task$phase
  node_id <- task$node_plan$node_id
  controller_id <- task$node_plan$controller_id
  variant_label <- if (is.null(task$variant_id)) "base" else task$variant_id
  fold_label <- if (is.null(task$fold_id)) "nofold" else task$fold_id

  if (!is.null(phase) && phase == "PREDICT") {
    estimator <- replay_estimator(task)
  } else {
    fn <- resolve_operator(task$node_plan$params$operator)
    train_ids <- train_sample_ids(task)
    X <- features(train_ids)
    y <- targets(train_ids)
    call_args <- build_estimator_args(task, X, y)
    estimator <- do.call(fn, call_args)
  }

  pred_ids <- prediction_sample_ids(task)
  X_pred <- features(pred_ids)
  raw_predictions <- predict(estimator, X_pred)
  prediction_vector <- extract_prediction_vector(raw_predictions)
  if (length(prediction_vector) != length(pred_ids)) {
    fail(sprintf(
      "operator `%s` returned %d predictions for %d sample(s)",
      task$node_plan$params$operator,
      length(prediction_vector),
      length(pred_ids)
    ))
  }
  if (any(!is.finite(prediction_vector))) {
    fail(sprintf("operator `%s` returned a non-finite prediction", task$node_plan$params$operator))
  }
  values <- lapply(prediction_vector, function(v) list(as.numeric(v)))

  prediction <- list(
    prediction_id = sprintf("pred:%s:%s:%s:%s", node_id, phase, variant_label, fold_label),
    producer_node = node_id,
    partition = prediction_partition(phase),
    fold_id = if (!is.null(phase) && phase == "FIT_CV") task$fold_id else NULL,
    sample_ids = pred_ids,
    values = values,
    target_names = list("y")
  )

  artifacts <- list()
  artifact_handles <- setNames(list(), character(0))
  if (!is.null(phase) && phase == "REFIT") {
    artifact_id <- sprintf("artifact:%s:mdatools:refit", node_id)
    write <- write_artifact(estimator, artifact_id, variant_label)
    handle_value <- stable_handle(paste(artifact_id, variant_label, sep = ":"))
    artifact <- list(
      id = artifact_id,
      kind = "mdatools_model",
      controller_id = controller_id,
      backend = "rds",
      uri = write$uri,
      content_fingerprint = write$fingerprint,
      size_bytes = write$size_bytes,
      plugin = ADAPTER_PLUGIN,
      plugin_version = ADAPTER_PLUGIN_VERSION
    )
    artifacts <- list(artifact)
    artifact_handles <- list()
    artifact_handles[[artifact_id]] <- list(
      handle = handle_value,
      kind = "model",
      owner_controller = controller_id
    )
  }

  list(predictions = list(prediction), artifacts = artifacts, artifact_handles = artifact_handles)
}

extract_prediction_vector <- function(raw) {
  # mdatools' predict.pls returns a `plsres` object whose `$y.pred`
  # is a 3-D array `[sample, component, response]`. Slice H.1
  # always returns predictions at the highest trained component
  # (`ncomp = ncomp`). mdatools' own optimum-component picker lives
  # in `$ncomp.selected` and depends on mdatools' internal CV — but
  # this controller force-disables that CV (`build_estimator_args`
  # nulls `kwargs$cv`) because dag-ml owns CV, so `$ncomp.selected`
  # is not a meaningful signal here. Callers who want component
  # selection should declare a separate variant per candidate
  # `ncomp` so dag-ml's selection layer can compare them. A future
  # slice can add an explicit `predict_at_component` param.
  if (inherits(raw, "plsres") && !is.null(raw$y.pred)) {
    arr <- raw$y.pred
    dims <- dim(arr)
    if (is.null(dims) || length(dims) < 1L) {
      return(as.numeric(arr))
    }
    return(as.numeric(arr[, dims[2], 1L]))
  }
  if (is.list(raw) && !is.null(raw$y.pred)) {
    return(as.numeric(raw$y.pred))
  }
  if (is.matrix(raw)) {
    return(as.numeric(raw[, 1L]))
  }
  as.numeric(raw)
}

output_handles <- function(task, handle_value) {
  node_plan <- task$node_plan
  controller_id <- node_plan$controller_id
  node_kind <- node_plan$kind
  outputs <- list(out = list(handle = handle_value, kind = "data", owner_controller = controller_id))
  if (!is.null(node_kind) && node_kind %in% c("model", "tuner")) {
    outputs$oof <- list(handle = handle_value, kind = "prediction", owner_controller = controller_id)
  } else if (!is.null(node_kind) && node_kind == "prediction_join") {
    outputs$prediction <- list(handle = handle_value, kind = "prediction", owner_controller = controller_id)
  } else {
    outputs$x_out <- list(handle = handle_value, kind = "data", owner_controller = controller_id)
  }
  outputs
}

build_result <- function(task) {
  node_plan <- task$node_plan
  node_id <- node_plan$node_id
  phase <- task$phase
  controller_id <- node_plan$controller_id
  variant_id <- task$variant_id
  fold_id <- task$fold_id
  variant_label <- if (is.null(variant_id)) "base" else variant_id
  fold_label <- if (is.null(fold_id)) "nofold" else fold_id
  handle_value <- stable_handle(paste(node_id, phase, variant_label, fold_label, sep = ":"))

  predictions <- list()
  artifacts <- list()
  artifact_handles <- setNames(list(), character(0))
  if (!is.null(node_plan$kind) && node_plan$kind %in% c("model", "tuner")) {
    model_out <- run_model(task)
    predictions <- model_out$predictions
    artifacts <- model_out$artifacts
    artifact_handles <- model_out$artifact_handles
  }

  metrics <- list(mdatools_adapter = 1.0)
  if (length(predictions) > 0L) {
    flat <- vapply(predictions[[1]]$values, function(row) as.numeric(row[[1]]), numeric(1))
    metrics$prediction_mean <- mean(flat)
  }
  worker_index <- Sys.getenv("DAG_ML_PROCESS_WORKER_INDEX", unset = "")
  worker_count <- Sys.getenv("DAG_ML_PROCESS_WORKER_COUNT", unset = "")
  if (nzchar(worker_index)) metrics$process_worker_index <- as.numeric(worker_index)
  if (nzchar(worker_count)) metrics$process_worker_count <- as.numeric(worker_count)

  list(
    node_id = node_id,
    outputs = output_handles(task, handle_value),
    predictions = predictions,
    shape_deltas = list(),
    artifacts = artifacts,
    artifact_handles = artifact_handles,
    lineage = list(
      record_id = sprintf("lineage:%s:%s:%s:%s", node_id, phase, variant_label, fold_label),
      run_id = task$run_id,
      node_id = node_id,
      phase = phase,
      controller_id = controller_id,
      controller_version = node_plan$controller_version,
      variant_id = variant_id,
      fold_id = fold_id,
      branch_path = if (is.null(task$branch_path)) list() else task$branch_path,
      input_lineage = list(),
      artifact_refs = artifacts,
      params_fingerprint = node_plan$params_fingerprint,
      data_model_shape_fingerprint = NULL,
      aggregation_policy_fingerprint = NULL,
      seed = task$seed,
      unsafe_flags = list(),
      metrics = metrics
    )
  )
}

emit_result <- function(task) {
  require_data_handles(task)
  emit_json(build_result(task))
}

emit_result_frame <- function(task) {
  require_data_handles(task)
  emit_json(list(
    type = "result",
    schema_version = PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
    result = build_result(task)
  ))
}

is_control_frame <- function(payload) {
  is.list(payload) && !is.null(payload$type) && is.character(payload$type)
}

validate_frame_schema <- function(frame) {
  if (is.null(frame$schema_version) ||
      frame$schema_version != PROCESS_ADAPTER_FRAME_SCHEMA_VERSION) {
    emit_error(
      "unsupported_frame_schema",
      sprintf("unsupported frame schema version `%s`", frame$schema_version)
    )
    return(FALSE)
  }
  TRUE
}

write_lifecycle_marker <- function(event, frame) {
  marker_dir <- Sys.getenv("DAG_ML_PROCESS_LIFECYCLE_MARKER_DIR", unset = "")
  if (!nzchar(marker_dir)) return(invisible(NULL))
  dir.create(marker_dir, showWarnings = FALSE, recursive = TRUE)
  controller_id <- if (!is.null(frame$controller_id)) frame$controller_id
    else Sys.getenv("DAG_ML_CONTROLLER_ID", unset = "controller")
  worker_index <- if (!is.null(frame$worker_index)) as.character(frame$worker_index)
    else Sys.getenv("DAG_ML_PROCESS_WORKER_INDEX", unset = "0")
  raw <- sprintf("%s_%s_%s", event, controller_id, worker_index)
  safe_name <- gsub("[^A-Za-z0-9._-]", "_", raw)
  marker_path <- file.path(marker_dir, paste0(safe_name, ".marker"))
  cat(event, "\n", file = marker_path, append = TRUE, sep = "")
}

handle_control_frame <- function(frame) {
  if (!validate_frame_schema(frame)) return(TRUE)
  frame_type <- frame$type
  if (frame_type == "init") {
    write_lifecycle_marker("init", frame)
    emit_ack("initialized")
    return(TRUE)
  }
  if (frame_type == "task") {
    task <- frame$task
    if (!is.list(task)) {
      emit_error("invalid_task_frame", "task frame is missing object field `task`")
      return(TRUE)
    }
    emit_result_frame(task)
    return(TRUE)
  }
  if (frame_type == "close") {
    write_lifecycle_marker("close", frame)
    emit_ack("closed")
    return(FALSE)
  }
  emit_error("unsupported_frame", sprintf("unsupported frame type `%s`", frame_type))
  TRUE
}

run_jsonl <- function() {
  input <- file("stdin", open = "r")
  on.exit(close(input), add = TRUE)
  repeat {
    line <- readLines(input, n = 1L, warn = FALSE)
    if (length(line) == 0L) break
    if (!nzchar(trimws(line))) next
    payload <- tryCatch(
      fromJSON(line, simplifyVector = FALSE),
      error = function(e) e
    )
    if (inherits(payload, "error")) {
      emit_error(
        "invalid_task_json",
        sprintf("invalid NodeTask JSON line: %s", conditionMessage(payload))
      )
      next
    }
    handler_result <- tryCatch(
      {
        if (is_control_frame(payload)) {
          if (!handle_control_frame(payload)) return(invisible(NULL))
          TRUE
        } else {
          emit_result(payload)
          TRUE
        }
      },
      AdapterTaskError = function(cond) {
        emit_error(cond$code, cond$message, cond$retryable)
        TRUE
      },
      error = function(cond) {
        emit_error(
          "adapter_unexpected_error",
          sprintf("%s: %s", class(cond)[1], conditionMessage(cond))
        )
        TRUE
      }
    )
    if (!isTRUE(handler_result)) break
  }
}

read_stdin_blob <- function() {
  input <- file("stdin", open = "r")
  on.exit(close(input), add = TRUE)
  lines <- readLines(input, warn = FALSE)
  paste(lines, collapse = "\n")
}

main <- function() {
  if (length(args) >= 1L && identical(args[1], "--jsonl")) {
    run_jsonl()
    return(invisible(NULL))
  }
  blob <- read_stdin_blob()
  payload <- tryCatch(
    fromJSON(blob, simplifyVector = FALSE),
    error = function(e) e
  )
  if (inherits(payload, "error")) {
    message(sprintf("invalid NodeTask JSON: %s", conditionMessage(payload)))
    quit(save = "no", status = 2L)
  }
  tryCatch(
    emit_result(payload),
    AdapterTaskError = function(cond) {
      message(sprintf("%s: %s", cond$code, cond$message))
      quit(save = "no", status = 2L)
    }
  )
}

# Equivalent of Python's `if __name__ == "__main__":` — only invoke
# `main()` when the script is being run directly (via Rscript), not
# when it is being `source()`d by a test fixture that wants to import
# `stable_handle`, `features`, `resolve_operator`, etc. in isolation.
if (sys.nframe() == 0L) main()
