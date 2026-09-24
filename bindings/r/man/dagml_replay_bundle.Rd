\name{dagml_replay_bundle}
\alias{dagml_replay_bundle}
\title{Replay a persisted CV bundle with host-owned model sidecars}
\usage{
dagml_replay_bundle(graph, campaign, controllers, bundle, replay_request,
  adapter, artifact_handles, envelopes, cli = "dag-ml-cli", output = NULL,
  persistent = FALSE, process_workers = 1L,
  process_timeout_ms = 30000L, process_retries = 0L,
  prediction_cache_payload = NULL, prediction_cache_store = NULL,
  score_output = NULL, plan_id = "plan:cli.bundle",
  run_id = "run:cli.process.replay", root_seed = 12345L,
  scheduler = "sequential", scheduler_workers = 1L)
}
\arguments{
\item{graph, campaign, controllers, bundle, replay_request, adapter, artifact_handles}{Required JSON or executable adapter paths.}
\item{envelopes}{Named character vector mapping replay data keys to V2 envelope JSON paths.}
\item{cli}{Path to \code{dag-ml-cli} or a command on \code{PATH}.}
\item{output}{Optional outcome JSON path; temporary by default.}
\item{persistent}{Keep the adapter alive across tasks.}
\item{process_workers, process_timeout_ms}{Positive process settings.}
\item{process_retries}{Non-negative retry count.}
\item{prediction_cache_payload, prediction_cache_store}{Optional native prediction cache sources for OOF-dependent replay.}
\item{score_output}{Optional score JSON destination.}
\item{plan_id, run_id}{Native identifiers.}
\item{root_seed}{Non-negative native seed.}
\item{scheduler}{Either \code{sequential} or \code{parallel}.}
\item{scheduler_workers}{Positive scheduler worker count.}
}
\value{Native replay outcome decoded as an R list.}
\description{DAG-ML validates the persisted CV bundle, exact host artifact handles and separate replay envelopes. The host owns model sidecar loading into its adapter's invocation-local handle namespace.}
