\name{dagml_execute_execution_plan_phase}
\alias{dagml_execute_execution_plan_phase}
\title{Execute One ExecutionPlan Phase Through Native DAG-ML}
\usage{
dagml_execute_execution_plan_phase(
  execution_plan,
  trusted_controller_manifests,
  run_id,
  root_seed,
  phase,
  controllers,
  native_library = Sys.getenv("DAGML_NATIVE_LIBRARY", unset = "")
)
}
\arguments{
\item{execution_plan}{ExecutionPlan JSON text, or a JSON-serializable R value.}
\item{trusted_controller_manifests}{ControllerManifest list JSON text, or a
JSON-serializable R value. The native core refuses execution if these manifests
do not exactly match the manifests embedded in \code{execution_plan}.}
\item{run_id}{DAG-ML run id for the phase execution.}
\item{root_seed}{Non-negative safe integer root seed.}
\item{phase}{ABI phase name such as \code{FIT_CV} or \code{REFIT}.}
\item{controllers}{Named list or environment of callbacks keyed by
\code{controller_id}. Each callback receives \code{(controller_id, task_json)}
and returns NodeResult JSON text or a JSON-serializable list.}
\item{native_library}{Path to the DAG-ML C ABI shared library. Defaults to the
\code{DAGML_NATIVE_LIBRARY} environment variable.}
}
\value{
A list of decoded NodeResult objects produced by the native scheduler.
}
\description{
Runs one phase with DAG-ML's native sequential scheduler while keeping operator
state and executable R callbacks local to the process. A controller callback can
close over \code{dagml_local_implementation_registry()} and call
\code{invoke_training_loss()} with the exact \code{task_json} it receives; the
loss/metric registries remain separate and only loss callbacks can emit
training-loss attestations.
}
