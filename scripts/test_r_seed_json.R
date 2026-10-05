#!/usr/bin/env Rscript
# Run from the repository root; no native library is needed for this boundary.
source("bindings/r/R/local_implementation_registry.R")

for (seed in c("42", "9007199254740993", "18446744073709551615")) {
  task_json <- paste0('{"nested":{"seed":0},"seed":', seed, "}")
  controllers <- .dagml_controller_callbacks(list(
    probe = function(id, task_json) list(lineage = list())
  ))
  text <- controllers$callbacks[[1L]]("probe", task_json)
  stopifnot(grepl(paste0('"seed":', seed), text, fixed = TRUE))
  for (result in list(
    list(lineage = list(seed = seed)),
    paste0('{"other":{"seed":0},"lineage":{"seed":', seed, "}}")
  )) {
    text <- .dagml_node_result_json(result, list(seed = seed))
    stopifnot(grepl(paste0('"seed":', seed), text, fixed = TRUE))
  }
}
stopifnot(inherits(try(
  .dagml_node_result_json(list(lineage = list(seed = 2^63)), list(seed = "9223372036854775808")),
  silent = TRUE
), "try-error"))
cat("PASS exact R u64 task/result seeds and refusal of rounded doubles\n")
