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
    "operator_timeout=",
    "while test \"$#\" -gt 0; do",
    "  case \"$1\" in",
    "    --output) output=$2; shift 2 ;;",
    "    --parallel-trials) parallel=$2; shift 2 ;;",
    "    --operator-persistent) persistent=1; shift ;;",
    "    --checkpoint) checkpoint=$2; shift 2 ;;",
    "    --operator-timeout-ms) operator_timeout=$2; shift 2 ;;",
    "    --plan|--envelope|--request|--operator-adapter|--optimizer-adapter|--adapter-timeout-ms) shift 2 ;;",
    "    *) exit 12 ;;",
    "  esac",
    "done",
    "test \"$parallel\" -ge 1 || exit 13",
    "test \"$persistent\" = 1 || exit 14",
    "test -n \"$checkpoint\" || exit 15",
    "case \"$operator_timeout\" in 0|1234) ;; *) exit 16 ;; esac",
    "printf '{\"wrapper_smoke\":true,\"parallel_trials\":%s,\"operator_timeout_ms\":%s,\"trials\":[1,2]}\\n' \"$parallel\" \"$operator_timeout\" > \"$output\""
  ), cli)
  Sys.chmod(cli, "0755")
  result <- dagml_host_hpo_search(
    input, input, input, cli, cli, cli = cli,
    checkpoint = file.path(work, "checkpoint 'state'.json"),
    operator_persistent = TRUE, parallel_trials = 2L
  )
  stopifnot(isTRUE(result$wrapper_smoke), length(result$trials) == 2L,
            result$parallel_trials == 2L, result$operator_timeout_ms == 0L)
  sequential <- dagml_host_hpo_search(
    input, input, input, cli, cli, cli = cli,
    checkpoint = file.path(work, "checkpoint 'state'.json"),
    operator_persistent = TRUE
  )
  stopifnot(sequential$parallel_trials == 1L,
            sequential$operator_timeout_ms == 0L)
  timed <- dagml_host_hpo_search(
    input, input, input, cli, cli, cli = cli,
    checkpoint = file.path(work, "checkpoint 'state'.json"),
    operator_persistent = TRUE, operator_timeout_ms = 1234L
  )
  stopifnot(timed$operator_timeout_ms == 1234L)
  for (timeout in c(-1, 0.5)) {
    rejected_timeout <- tryCatch(
      dagml_host_hpo_search(input, input, input, cli, cli, cli = cli,
                           operator_timeout_ms = timeout),
      error = identity
    )
    stopifnot(inherits(rejected_timeout, "error"),
              grepl("operator_timeout_ms must be a nonnegative integer",
                    conditionMessage(rejected_timeout), fixed = TRUE))
  }
  rejected <- tryCatch(
    dagml_host_hpo_search(input, input, input, cli, cli, cli = cli,
                           parallel_trials = 0L),
    error = identity
  )
  stopifnot(inherits(rejected, "error"),
            grepl("positive integer", conditionMessage(rejected)))
  unlink(work, recursive = TRUE)
}
