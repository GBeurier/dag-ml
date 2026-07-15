\name{dagml_local_implementation_registry}
\alias{dagml_local_implementation_registry}
\title{Create a Process-Local R Implementation Registry}
\usage{
dagml_local_implementation_registry()
}
\value{
An environment whose methods register, resolve, invoke, unregister, inspect,
and clear process-local R functions.
}
\description{
Retains R loss and metric functions under exact DAG-ML implementation
descriptors. The \code{invoke_training_loss()} method executes an active
\code{FIT_CV} or \code{REFIT} loss and returns the native task attestation only
after the function succeeds. Executable functions are never written to DAG-ML
JSON contracts.
}
