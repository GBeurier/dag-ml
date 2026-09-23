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

Host hyperparameter search can use the native scheduler through executable
JSONL operator and optimizer adapters. The three JSON inputs are an
`ExecutionPlan`, `ExternalDataPlanEnvelope`, and `HostHpoSearchRequest`.
The CLI, rather than an R callback thread, owns trial scheduling, folds,
pruning, selection, and durable checkpoints:

```r
outcome <- dagml_host_hpo_search(
  plan = "plan.json", envelope = "envelope.json", request = "hpo.json",
  operator_adapter = "./r-operator-adapter",
  optimizer_adapter = "./r-optimizer-adapter",
  checkpoint = "search.checkpoint.json",
  operator_persistent = TRUE
)
```

The executables can launch `Rscript` but must implement the same JSONL
protocol as the [CLI examples](../../examples/adapters/). The wrapper accepts
`parallel_trials = N`; DAG-ML starts a separate operator-adapter process per
candidate and keeps optimizer callbacks on the coordinator thread. The R adapter
processes and their optimizer state remain host-owned. Run the wrapper smoke with
`R CMD check`; a working R installation is required.

The strict `r_hpo_ridge` integration test also calls this wrapper with a real
R Ridge operator. It reads the plan's physical FoldSet IDs, fits on each
fold-train cohort, and checks native per-fold/OOF RMSE, the two-candidate
parallel scheduler path, pruning and checkpoint resume. Run it with
`DAGML_REQUIRE_HPO_R=1 cargo test -p dag-ml-cli --test r_hpo_ridge`.
It qualifies this HPO path for one R operator; it does not test a fitted R
model's REFIT artifact or later PREDICT replay.

A no-splitter pipeline can capture an initial full REFIT package and replay
PREDICT on a separate V2 cohort through the same native CLI:

```r
refit <- dagml_initial_full_refit(
  "pipeline.json", "controllers.json", "train-envelope.json",
  "training-ids.json", "./r-operator-adapter", "full-refit.package.json"
)
prediction <- dagml_initial_full_refit_predict(
  "full-refit.package.json", "predict-envelope.json",
  "./r-operator-adapter", "artifact-handles.json", "output-ids.json"
)
```

The package contains native lineage and artifact identities, while R retains
model sidecars. The replay adapter must resolve the exact host artifact handles
listed in the package; DAG-ML rejects missing or extra handles.

For a pipeline with CV, `dagml_cv_refit_predict()` executes CV, winner REFIT
and PREDICT in one native CLI session. Its outcome includes the execution
bundle, OOF averages and replay prediction blocks. This keeps host model
handles alive for the replay in that session; persisting the bundle JSON alone
does not persist R model sidecars.

For replay in a later process, persist those sidecars in the host and call
`dagml_replay_bundle()` with the bundle JSON, V2 replay envelopes and an exact
artifact-ID-to-invocation-handle JSON map. The host adapter reloads sidecars
from its own storage and checks their fingerprints; DAG-ML validates the map
against the bundle before PREDICT. `--artifact-handles` on the CLI enables this
path, while omitting it retains mock handles for conformance examples only.
