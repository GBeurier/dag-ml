library(dagml)

if (.Platform$OS.type != "windows") {
  work <- tempfile("dagml refit 'test' ")
  dir.create(work)
  input <- file.path(work, "input 'contract'.json")
  writeLines("{}", input)
  cli <- file.path(work, "fake cli")
  writeLines(c(
    "#!/bin/sh",
    "set -eu",
    "command=$1; shift",
    "output=; package=; workers=; timeout=; retries=; persistent=0; scheduler=; seed=; devices=0; run_id=",
    "while test \"$#\" -gt 0; do",
    "  case \"$1\" in",
    "    --output) output=$2; shift 2 ;;",
    "    --package-output) package=$2; shift 2 ;;",
    "    --process-workers) workers=$2; shift 2 ;;",
    "    --process-timeout-ms) timeout=$2; shift 2 ;;",
    "    --process-retries) retries=$2; shift 2 ;;",
    "    --persistent) persistent=1; shift ;;",
    "    --package-id|--plan-id|--scheduler-workers|--cpu-threads) shift 2 ;;",
    "    --run-id) run_id=$2; shift 2 ;;",
    "    --root-seed) seed=$2; shift 2 ;;",
    "    --scheduler) scheduler=$2; shift 2 ;;",
    "    --gpu-device) devices=$((devices + 1)); shift 2 ;;",
    "    --dsl|--controllers|--envelope|--training-sample-ids|--adapter|--package|--artifact-handles|--output-ids) test -f \"$2\" || exit 12; shift 2 ;;",
    "    *) exit 13 ;;",
    "  esac",
    "done",
    "test \"$workers\" = 2 || exit 14",
    "test \"$timeout\" = 9000 || exit 15",
    "test \"$retries\" = 1 || exit 16",
    "test \"$persistent\" = 1 || exit 17",
    "case \"$command\" in",
    "  run-process-dsl-refit-phase)",
    "    test -n \"$package\" || exit 18",
    "    test \"$scheduler\" = parallel || exit 20",
    "    test \"$seed\" = 19 || exit 21",
    "    test \"$devices\" = 1 || exit 22",
    "    test \"$run_id\" = run:refit.test || exit 23",
    "    printf '{\"package_id\":\"package:test\"}\\n' > \"$package\"",
    "    printf '{\"initial_full_refit_package\":{\"package_id\":\"package:test\"}}\\n' > \"$output\" ;;",
    "  run-process-initial-full-refit-predict)",
    "    test \"$run_id\" = run:predict.test || exit 24",
    "    printf '{\"replay_outcome\":{\"phase\":\"PREDICT\"}}\\n' > \"$output\" ;;",
    "  *) exit 19 ;;",
    "esac"
  ), cli)
  Sys.chmod(cli, "0755")
  package <- file.path(work, "signed 'package'.json")
  result <- dagml_initial_full_refit(
    input, input, input, input, cli, package, cli = cli,
    persistent = TRUE, process_workers = 2L,
    process_timeout_ms = 9000L, process_retries = 1L,
    root_seed = 19L, scheduler = "parallel", gpu_devices = "0",
    run_id = "run:refit.test"
  )
  stopifnot(result$initial_full_refit_package$package_id == "package:test",
            file.exists(package))
  replay <- dagml_initial_full_refit_predict(
    package, input, cli, input, input, cli = cli,
    persistent = TRUE, process_workers = 2L,
    process_timeout_ms = 9000L, process_retries = 1L,
    run_id = "run:predict.test"
  )
  stopifnot(replay$replay_outcome$phase == "PREDICT")
  bad <- tryCatch(
    dagml_initial_full_refit(input, input, input, input, cli, package,
                             cli = cli, process_workers = 0L),
    error = identity
  )
  stopifnot(inherits(bad, "error"), grepl("process_workers", conditionMessage(bad)))
  unlink(work, recursive = TRUE)
}
