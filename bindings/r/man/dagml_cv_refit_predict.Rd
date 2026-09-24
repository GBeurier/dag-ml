\name{dagml_cv_refit_predict}
\alias{dagml_cv_refit_predict}
\title{Run CV, REFIT and PREDICT in one native DAG-ML session}
\usage{
dagml_cv_refit_predict(dsl, controllers, envelope, adapter,
  cli = "dag-ml-cli", output = NULL, process_workers = 1L,
  process_timeout_ms = 30000L, process_retries = 0L,
  bundle_id = "bundle:cli.process.dsl.cv.refit.replay", variant_id = NULL,
  selection_metric = "rmse", selections = NULL,
  plan_id = "plan:cli.process.dsl.cv.refit.replay",
  run_id = "run:cli.process.dsl.cv.refit.replay", root_seed = 12345L,
  scheduler = "sequential", scheduler_workers = 1L, cpu_threads = 1L,
  gpu_devices = character())
}
\arguments{
\item{dsl, controllers, envelope}{JSON input file paths.}
\item{adapter}{Executable operator process adapter.}
\item{cli}{Path to \code{dag-ml-cli} or a command on \code{PATH}.}
\item{output}{Optional outcome JSON path; temporary by default.}
\item{process_workers, process_timeout_ms}{Positive process settings.}
\item{process_retries}{Non-negative retry count.}
\item{bundle_id, plan_id, run_id}{Native identifiers.}
\item{variant_id}{Optional concrete variant ID.}
\item{selection_metric}{One of \code{rmse}, \code{accuracy}, \code{balanced_accuracy}.}
\item{selections}{Optional selection-decision JSON path.}
\item{root_seed}{Non-negative native seed.}
\item{scheduler}{Either \code{sequential} or \code{parallel}.}
\item{scheduler_workers, cpu_threads}{Positive resource counts.}
\item{gpu_devices}{Optional GPU device identifiers.}
}
\value{Native CV, REFIT and PREDICT outcome decoded as an R list.}
\description{The CLI retains host model handles across phases and returns a native bundle, OOF averages, replay prediction blocks and scores. This is a one-session replay; the bundle JSON does not include host model sidecars.}
