\name{dagml_initial_full_refit_predict}
\alias{dagml_initial_full_refit_predict}
\title{Replay PREDICT from a no-CV full REFIT package}
\usage{
dagml_initial_full_refit_predict(package, envelope, adapter,
  artifact_handles, output_ids, cli = "dag-ml-cli", output = NULL,
  persistent = FALSE, process_workers = 1L, process_timeout_ms = 30000L,
  process_retries = 0L, run_id = "run:cli.initial.refit.predict")
}
\arguments{
\item{package, envelope, adapter, artifact_handles, output_ids}{Package, JSON contract, or executable adapter paths required by native replay.}
\item{cli}{Path to \code{dag-ml-cli} or a command on \code{PATH}.}
\item{output}{Optional outcome JSON path; temporary by default.}
\item{persistent}{Keep the operator adapter alive across tasks.}
\item{process_workers}{Positive number of adapter workers.}
\item{process_timeout_ms}{Positive adapter timeout in milliseconds.}
\item{process_retries}{Non-negative retry count.}
\item{run_id}{Native replay run identifier.}
}
\value{Native PREDICT replay outcome decoded as an R list.}
\description{Validates the signed package and V2 prediction cohort, then resolves exact host-sidecar artifact handles through the host adapter.}
