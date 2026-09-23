library(dagml)

if (.Platform$OS.type != "windows") {
  work <- tempfile("dagml hpo 'test' ")
  dir.create(work)
  input <- file.path(work, "input.json")
  writeLines("{}", input)
  cli <- file.path(work, "fake cli")
  writeLines(c(
    "#!/bin/sh",
    "test \"$1\" = run-host-hpo || exit 11",
    "shift",
    "output=",
    "parallel=",
    "persistent=0",
    "checkpoint=",
    "while test \"$#\" -gt 0; do",
    "  case \"$1\" in",
    "    --output) output=$2; shift 2 ;;",
    "    --parallel-trials) parallel=$2; shift 2 ;;",
    "    --operator-persistent) persistent=1; shift ;;",
    "    --checkpoint) checkpoint=$2; shift 2 ;;",
    "    --plan|--envelope|--request|--operator-adapter|--optimizer-adapter|--adapter-timeout-ms) shift 2 ;;",
    "    *) exit 12 ;;",
    "  esac",
    "done",
    "test \"$parallel\" -ge 1 || exit 13",
    "test \"$persistent\" = 1 || exit 14",
    "test -n \"$checkpoint\" || exit 15",
    "printf '{\"wrapper_smoke\":true,\"parallel_trials\":%s,\"trials\":[1,2]}\\n' \"$parallel\" > \"$output\""
  ), cli)
  Sys.chmod(cli, "0755")
  result <- dagml_host_hpo_search(
    input, input, input, cli, cli, cli = cli,
    checkpoint = file.path(work, "checkpoint 'state'.json"),
    operator_persistent = TRUE, parallel_trials = 2L
  )
  stopifnot(isTRUE(result$wrapper_smoke), length(result$trials) == 2L,
            result$parallel_trials == 2L)
  sequential <- dagml_host_hpo_search(
    input, input, input, cli, cli, cli = cli,
    checkpoint = file.path(work, "checkpoint 'state'.json"),
    operator_persistent = TRUE
  )
  stopifnot(sequential$parallel_trials == 1L)
  rejected <- tryCatch(
    dagml_host_hpo_search(input, input, input, cli, cli, cli = cli,
                           parallel_trials = 0L),
    error = identity
  )
  stopifnot(inherits(rejected, "error"),
            grepl("positive integer", conditionMessage(rejected)))
  unlink(work, recursive = TRUE)
}
