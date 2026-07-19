\name{dagml_local_implementation_registry}
\alias{dagml_local_implementation_registry}
\title{Create a Process-Local R Implementation Registry}
\usage{
dagml_local_implementation_registry(
  native_library = Sys.getenv("DAGML_NATIVE_LIBRARY", unset = "")
)
}
\arguments{
\item{native_library}{Path to the DAG-ML C ABI shared library. Defaults to
the \code{DAGML_NATIVE_LIBRARY} environment variable and is required only for
task-bound training-loss invocation.}
}
\value{
An environment whose methods register, resolve, invoke, unregister, inspect,
and clear process-local R functions.
}
\description{
Retains R loss and metric functions under exact DAG-ML implementation
descriptors. The \code{invoke_training_loss()} method executes an active
\code{FIT_CV} or \code{REFIT} loss from the exact native \code{NodeTask} JSON
and returns the task attestation only after the function succeeds. Role
selection and attestation validation are delegated to the native DAG-ML core;
tensors and executable R functions remain in the R process and are never
written to DAG-ML JSON contracts.
}
