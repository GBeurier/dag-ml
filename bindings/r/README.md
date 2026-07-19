# DAG-ML R binding

The `dagml` package provides the R-owned process-local implementation registry.
It retains R loss and metric functions under exact DAG-ML descriptors; no
function, closure, environment, or import instruction is placed in a DAG JSON
contract.

```r
implementations <- dagml_local_implementation_registry(
  native_library = Sys.getenv("DAGML_NATIVE_LIBRARY")
)
implementations$register_loss(loss_reference, asymmetric_loss)

execution <- implementations$invoke_training_loss(
  node_task_json,
  target = y_true,
  prediction = y_pred
)
loss_attestations <- list(execution$attestation)
```

`invoke_training_loss()` accepts the exact `NodeTask` JSON emitted by DAG-ML.
Keeping the native JSON avoids lossy R round-trips between JSON scalars and
single-element arrays. The C ABI selects the phase-filtered role and task-owned
attestation, then the registry executes the R function on R-owned values.
`PREDICT`, stale attestations, and invalid role indexes fail in the native core
before the R function runs. Detached workers and replay processes must load the
DAG-ML native library and register their own local functions.

Build and check with:

```bash
cargo build -p dag-ml-capi --release
export DAGML_NATIVE_LIBRARY="$PWD/target/release/libdag_ml_capi.so"
R CMD build bindings/r
R CMD check --no-manual dagml_*.tar.gz
```
