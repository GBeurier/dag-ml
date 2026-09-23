\name{dagml_host_hpo_search}
\alias{dagml_host_hpo_search}
\title{Run scheduler-owned host hyperparameter search}
\usage{
dagml_host_hpo_search(plan, envelope, request, operator_adapter,
  optimizer_adapter, cli = "dag-ml-cli", checkpoint = NULL,
  output = NULL, operator_persistent = FALSE, parallel_trials = 1L,
  adapter_timeout_ms = 30000L)
}
\arguments{
\item{plan, envelope, request}{Paths to DAG-ML JSON contracts.}
\item{operator_adapter, optimizer_adapter}{Executable JSONL process adapters.}
\item{cli}{Path to \code{dag-ml-cli} or a command on \code{PATH}.}
\item{checkpoint}{Optional native checkpoint file for durable resume.}
\item{output}{Optional result JSON file. Defaults to a temporary file.}
\item{operator_persistent}{Keep a candidate-local operator process across folds.}
\item{parallel_trials}{Positive number of concurrent candidate trials. DAG-ML starts isolated operator-adapter processes.}
\item{adapter_timeout_ms}{Positive adapter response timeout in milliseconds.}
}
\value{Native HPO outcome decoded as an R list without simplifying JSON arrays.}
\description{Calls the DAG-ML CLI HPO scheduler. The native core owns trial execution,
fold scoring, pruning, selection, and checkpoints; the host adapters own R operators
and optimizer proposals.}
