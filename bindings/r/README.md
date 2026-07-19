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

The package also exposes native phase execution for local controller callbacks:

```r
results <- dagml_execute_execution_plan_phase(
  execution_plan = execution_plan_json,
  trusted_controller_manifests = controller_manifests_json,
  run_id = "run:r-local",
  root_seed = 42,
  phase = "FIT_CV",
  controllers = list(
    "controller:r-local" = function(controller_id, task_json) {
      task <- jsonlite::fromJSON(task_json, simplifyVector = FALSE)
      loss <- implementations$invoke_training_loss(
        task_json,
        target = y_true,
        prediction = y_pred
      )
      result <- run_r_operator(task, loss$value)
      result$lineage$loss_attestations <- list(loss$attestation)
      result
    }
  )
)
```

`invoke_training_loss()` accepts the exact `NodeTask` JSON emitted by DAG-ML.
Keeping the native JSON avoids lossy R round-trips between JSON scalars and
single-element arrays. The C ABI selects the phase-filtered role and task-owned
attestation, then the registry executes the R function on R-owned values.
`PREDICT`, stale attestations, and invalid role indexes fail in the native core
before the R function runs. Detached workers and replay processes must load the
DAG-ML native library and register their own local functions.

`dagml_execute_execution_plan_phase()` runs the native DAG-ML sequential
scheduler for a validated `ExecutionPlan` and refuses execution unless the
trusted runtime manifests exactly match the manifests embedded in the plan. The
callback surface returns `NodeResult` JSON/lists for conformance and local
controller orchestration. Long-lived native handle ownership remains the job of
the opaque training/replay APIs that retain controller registries.

Build and check with:

```bash
cargo build -p dag-ml-capi --release
export DAGML_NATIVE_LIBRARY="$PWD/target/release/libdag_ml_capi.so"
R CMD build bindings/r
R CMD check --no-manual dagml_*.tar.gz
```
