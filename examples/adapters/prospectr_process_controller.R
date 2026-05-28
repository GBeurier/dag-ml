#!/usr/bin/env Rscript
# Process adapter for the prospectr R package, speaking the dag-ml
# coordinator's JSONL protocol.
#
# Slice G.1 covers the R-side scaffolding plus dispatch for
# standardNormalVariate (SNV) and msc (Multiplicative Scatter
# Correction). Remaining prospectr operators (savitzkyGolay, gapDer,
# binning, continuumRemoval) and the matching ControllerManifest are
# delivered in Slice G.2.
#
# Side-by-side with the Python adapters in this directory. The R
# adapter advertises a distinct adapter_id and a `prospectr_smoke`
# capability.

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
  "prospectr_smoke"
)
ADAPTER_ID <- "dag-ml-prospectr-process-controller"
ADAPTER_PLUGIN <- "dagml.prospectr_process"
ADAPTER_PLUGIN_VERSION <- "1.0.0"

args <- commandArgs(trailingOnly = TRUE)

# Describe fast path: load jsonlite only, defer prospectr until a real
# task arrives. Matches the cheap-discovery pattern used by the Python
# adapters.
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
  library(prospectr)
})

# Operator registry. The aliases are the keys callers use in
# NodeTask.node_plan.params; the values describe how to resolve each
# alias to a prospectr namespace function.
OPERATOR_SELECTORS <- list(
  binning = list(pkg = "prospectr", fn = "binning"),
  continuumRemoval = list(pkg = "prospectr", fn = "continuumRemoval"),
  gapDer = list(pkg = "prospectr", fn = "gapDer"),
  savitzkyGolay = list(pkg = "prospectr", fn = "savitzkyGolay"),
  SNV = list(pkg = "prospectr", fn = "standardNormalVariate"),
  standardNormalVariate = list(pkg = "prospectr", fn = "standardNormalVariate")
)

# `msc` (Multiplicative Scatter Correction) is intentionally omitted
# from this slice: prospectr's `msc(X, ref_spectrum = NULL)` defaults
# the reference spectrum to `colMeans(X)` of the current batch. In a
# CV/REFIT/PREDICT pipeline that would leak validation/test data into
# the reference. Wiring MSC requires stateful reference-spectrum
# persistence (artifact_policy != stateless), which is a separate
# slice from the stateless dispatch this controller advertises.

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

# Match the Python smoke's deterministic feature/target synthesis so
# the smoke contract is platform-stable. R has no built-in 64-bit
# stable hash; this 31-bit polynomial hash is sufficient for
# deterministic seeding within a fold.
stable_handle <- function(value) {
  bytes <- as.integer(charToRaw(value))
  # R has no native 64-bit integer; using double-precision arithmetic
  # under modulo 2^31-1 keeps the hash stable without `bit64` and
  # avoids 32-bit overflow on long IDs.
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
        fail(sprintf(
          "node `%s` did not receive validation view `%s`",
          node_plan$node_id, validation_key
        ))
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

apply_transform <- function(task) {
  params <- task$node_plan$params
  if (is.null(params) || is.null(params$operator)) {
    fail("node `params` missing `operator`")
  }
  fn <- resolve_operator(params$operator)
  ids <- train_sample_ids(task)
  X <- features(ids)
  call_args <- list(X)
  if (!is.null(params$params)) {
    if (!is.list(params$params)) {
      fail("`params` for the operator must be an object", code = "adapter_fail")
    }
    call_args <- c(call_args, params$params)
  }
  do.call(fn, call_args)
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

  metrics <- list(prospectr_adapter = 1.0)
  worker_index <- Sys.getenv("DAG_ML_PROCESS_WORKER_INDEX", unset = "")
  worker_count <- Sys.getenv("DAG_ML_PROCESS_WORKER_COUNT", unset = "")
  if (nzchar(worker_index)) metrics$process_worker_index <- as.numeric(worker_index)
  if (nzchar(worker_count)) metrics$process_worker_count <- as.numeric(worker_count)

  # Exercise the transform so the smoke proves the prospectr dispatch
  # actually returns a finite matrix; the transformed values stay
  # inside the controller process (the host owns the data buffer
  # itself), but we record the column count as a metric for the
  # smoke test to assert on.
  if (!is.null(node_plan$kind) && node_plan$kind == "transform") {
    transformed <- apply_transform(task)
    if (!is.matrix(transformed) && !is.data.frame(transformed)) {
      transformed <- as.matrix(transformed)
    }
    if (any(!is.finite(as.matrix(transformed)))) {
      fail(sprintf("operator `%s` returned a non-finite value", task$node_plan$params$operator))
    }
    metrics$transform_columns <- ncol(transformed)
    metrics$transform_rows <- nrow(transformed)
  }

  list(
    node_id = node_id,
    outputs = output_handles(task, handle_value),
    predictions = list(),
    shape_deltas = list(),
    artifacts = list(),
    artifact_handles = setNames(list(), character(0)),
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
      artifact_refs = list(),
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
