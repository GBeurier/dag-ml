#!/usr/bin/env Rscript
# Exercise the public R HPO wrapper against the native CLI and real R operator.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 9L)
source(file.path(args[[1L]], "bindings/r/R/host_hpo_search.R"))
result <- dagml_host_hpo_search(
  plan = args[[2L]], envelope = args[[3L]], request = args[[4L]],
  operator_adapter = args[[5L]], optimizer_adapter = args[[6L]],
  cli = args[[7L]], checkpoint = args[[8L]], output = args[[9L]],
  parallel_trials = 2L
)
stopifnot(identical(result$status, "completed"), length(result$trials) >= 2L)
