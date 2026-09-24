library(dagml)

if (.Platform$OS.type != "windows") {
  work <- tempfile("dagml replay 'test' ")
  dir.create(work)
  input <- file.path(work, "input 'contract'.json")
  writeLines("{}", input)
  cli <- file.path(work, "fake cli")
  writeLines(c(
    "#!/bin/sh", "set -eu",
    "test \"$1\" = run-process-replay || exit 11; shift",
    "output=; envelopes=0; handles=0; workers=; timeout=; retries=",
    "while test \"$#\" -gt 0; do",
    "  case \"$1\" in",
    "    --graph|--campaign|--controllers|--bundle|--replay-request|--adapter|--prediction-cache-payload) test -f \"$2\" || exit 12; shift 2 ;;",
    "    --artifact-handles) test -f \"$2\" || exit 13; handles=$((handles + 1)); shift 2 ;;",
    "    --envelope) envelopes=$((envelopes + 1)); shift 2 ;;",
    "    --output) output=$2; shift 2 ;;",
    "    --process-workers) workers=$2; shift 2 ;;",
    "    --process-timeout-ms) timeout=$2; shift 2 ;;",
    "    --process-retries) retries=$2; shift 2 ;;",
    "    --plan-id|--run-id|--root-seed|--scheduler|--scheduler-workers|--score-output) shift 2 ;;",
    "    --persistent) shift ;;",
    "    *) exit 14 ;;",
    "  esac",
    "done",
    "test \"$handles\" = 1 || exit 15",
    "test \"$envelopes\" = 2 || exit 16",
    "test \"$workers\" = 2 || exit 17",
    "test \"$timeout\" = 9000 || exit 18",
    "test \"$retries\" = 1 || exit 19",
    "printf '{\"bundle_id\":\"bundle:test\",\"node_results\":[{}],\"prediction_blocks\":[{\"sample_ids\":[\"s1\"]}],\"scores\":null}\\n' > \"$output\""
  ), cli)
  Sys.chmod(cli, "0755")
  outcome <- dagml_replay_bundle(
    input, input, input, input, input, cli, input,
    envelopes = c("node:x" = input, "node:y" = input),
    cli = cli, prediction_cache_payload = input,
    process_workers = 2L, process_timeout_ms = 9000L, process_retries = 1L
  )
  stopifnot(outcome$bundle_id == "bundle:test",
            length(outcome$prediction_blocks) == 1L)
  bad <- tryCatch(
    dagml_replay_bundle(input, input, input, input, input, cli, input,
                        envelopes = c(input, input), cli = cli),
    error = identity
  )
  stopifnot(inherits(bad, "error"), grepl("envelopes", conditionMessage(bad)))
  unlink(work, recursive = TRUE)
}
