library(dagml)

if (.Platform$OS.type != "windows") {
  work <- tempfile("dagml cv replay 'test' ")
  dir.create(work)
  input <- file.path(work, "input 'contract'.json")
  writeLines("{}", input)
  cli <- file.path(work, "fake cli")
  writeLines(c(
    "#!/bin/sh", "set -eu",
    "test \"$1\" = run-process-dsl-cv-refit-replay || exit 11; shift",
    "output=; metric=; scheduler=; gpu=0; workers=; timeout=; retries=",
    "while test \"$#\" -gt 0; do",
    "  case \"$1\" in",
    "    --dsl|--controllers|--envelope|--adapter|--selections) test -f \"$2\" || exit 12; shift 2 ;;",
    "    --output) output=$2; shift 2 ;;",
    "    --selection-metric) metric=$2; shift 2 ;;",
    "    --scheduler) scheduler=$2; shift 2 ;;",
    "    --gpu-device) gpu=$((gpu + 1)); shift 2 ;;",
    "    --process-workers) workers=$2; shift 2 ;;",
    "    --process-timeout-ms) timeout=$2; shift 2 ;;",
    "    --process-retries) retries=$2; shift 2 ;;",
    "    --bundle-id|--variant-id|--plan-id|--run-id|--root-seed|--scheduler-workers|--cpu-threads) shift 2 ;;",
    "    *) exit 13 ;;",
    "  esac",
    "done",
    "test \"$metric\" = balanced_accuracy || exit 14",
    "test \"$scheduler\" = parallel || exit 15",
    "test \"$gpu\" = 1 || exit 16",
    "test \"$workers\" = 2 || exit 17",
    "test \"$timeout\" = 9000 || exit 18",
    "test \"$retries\" = 1 || exit 19",
    "printf '{\"bundle\":{\"bundle_id\":\"bundle:test\"},\"replay_node_results\":[{\"node_id\":\"model:test\"}],\"replay_prediction_blocks\":[{\"sample_ids\":[\"s1\"]}],\"replay_scores\":null}\\n' > \"$output\""
  ), cli)
  Sys.chmod(cli, "0755")
  outcome <- dagml_cv_refit_predict(
    input, input, input, cli, cli = cli, selections = input,
    selection_metric = "balanced_accuracy", scheduler = "parallel",
    gpu_devices = "0", process_workers = 2L,
    process_timeout_ms = 9000L, process_retries = 1L
  )
  stopifnot(outcome$bundle$bundle_id == "bundle:test",
            length(outcome$replay_node_results) == 1L,
            length(outcome$replay_prediction_blocks) == 1L)
  bad <- tryCatch(
    dagml_cv_refit_predict(input, input, input, cli, cli = cli,
                           selection_metric = "invalid"),
    error = identity
  )
  stopifnot(inherits(bad, "error"), grepl("selection_metric", conditionMessage(bad)))
  unlink(work, recursive = TRUE)
}
