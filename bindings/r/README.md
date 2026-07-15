# DAG-ML R binding

The `dagml` package provides the R-owned process-local implementation registry.
It retains R loss and metric functions under exact DAG-ML descriptors; no
function, closure, environment, or import instruction is placed in a DAG JSON
contract.

```r
implementations <- dagml_local_implementation_registry()
implementations$register_loss(loss_reference, asymmetric_loss)

execution <- implementations$invoke_training_loss(
  node_task,
  target = y_true,
  prediction = y_pred
)
result$lineage$loss_attestations <- list(execution$attestation)
```

`invoke_training_loss()` accepts only `FIT_CV` and `REFIT`. It validates the
active role against `NodeTask.required_loss_attestations`, executes the R
function, and returns the native-produced attestation only after success.
Detached workers and replay processes must register their own local functions.

Build and check with:

```bash
R CMD build bindings/r
R CMD check --no-manual dagml_*.tar.gz
```
