#!/usr/bin/env Rscript
# Minimal R optimizer for dag-ml-cli run-host-hpo. DAG-ML owns trial execution,
# fold scoring, pruning and native checkpoints; this host only speaks JSONL.
suppressPackageStartupMessages(library(jsonlite))

reply_for <- function(event) {
  operation <- event$operation
  if (identical(operation, "init")) {
    return(list(prepared_checkpoint = NULL, interrupted = list()))
  }
  if (identical(operation, "ask")) {
    return(list(params = list(n_components = event$trial_index + 1L)))
  }
  if (identical(operation, "report_intermediate")) {
    return(list(prune = FALSE))
  }
  if (operation %in% c("tell", "pruned", "fail", "prepare_terminal", "checkpoint")) {
    return(list(ok = TRUE))
  }
  list(error = paste0("unsupported HPO operation ", operation))
}

input <- file("stdin", open = "r")
repeat {
  line <- readLines(input, n = 1L, warn = FALSE)
  if (length(line) == 0L) break
  event <- fromJSON(line, simplifyVector = FALSE)
  cat(toJSON(reply_for(event), auto_unbox = TRUE, null = "null"), "\n", sep = "")
  flush(stdout())
}
close(input)
