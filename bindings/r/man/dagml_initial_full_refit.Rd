\name{dagml_initial_full_refit}
\alias{dagml_initial_full_refit}
\title{Capture a no-CV full REFIT package through DAG-ML}
\usage{
dagml_initial_full_refit(dsl, controllers, envelope, training_sample_ids,
  adapter, package_output, cli = "dag-ml-cli", output = NULL,
  persistent = FALSE, process_workers = 1L, process_timeout_ms = 30000L,
  process_retries = 0L,
  package_id = "package:cli.process.dsl.initial.refit",
  plan_id = "plan:cli.process.dsl.refit.phase",
  run_id = "run:cli.process.dsl.refit.phase", root_seed = 12345L,
  scheduler = "sequential", scheduler_workers = 1L, cpu_threads = 1L,
  gpu_devices = character())
}
\arguments{
\item{dsl, controllers, envelope, training_sample_ids}{JSON input file paths.}
\item{adapter}{Executable operator process adapter.}
\item{package_output}{Required destination for the signed package JSON.}
\item{cli}{Path to \code{dag-ml-cli} or a command on \code{PATH}.}
\item{output}{Optional outcome JSON path; temporary by default.}
\item{persistent}{Keep the operator adapter alive across tasks.}
\item{process_workers}{Positive number of adapter workers.}
\item{process_timeout_ms}{Positive adapter timeout in milliseconds.}
\item{process_retries}{Non-negative retry count.}
\item{package_id, plan_id, run_id}{Native identifiers for capture lineage.}
\item{root_seed}{Non-negative native seed.}
\item{scheduler}{Either \code{sequential} or \code{parallel}.}
\item{scheduler_workers, cpu_threads}{Positive resource counts.}
\item{gpu_devices}{Optional GPU device identifiers.}
}
\value{Native REFIT outcome decoded as an R list.}
\description{Runs the native no-splitter REFIT coordinator and captures its signed package. Host model bytes remain in host-managed sidecars.}
