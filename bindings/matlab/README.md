# DAG-ML MATLAB/Octave binding

`dagml.LocalImplementationRegistry` is the MATLAB-owned process-local
implementation registry. It retains loss and metric `function_handle` objects
under exact DAG-ML descriptors; executable objects are never written into DAG
JSON contracts.

```matlab
implementations = dagml.LocalImplementationRegistry( ...
    getenv('DAGML_NATIVE_LIBRARY'));
implementations.registerLoss(lossReference, @asymmetricLoss);

[value, attestation] = implementations.invokeTrainingLoss( ...
    nodeTaskJSON, 1, yTrue, yPred);
result.lineage.loss_attestations = {attestation};
```

The binding can also execute one native DAG-ML scheduler phase with
MATLAB-local controller callbacks:

```matlab
controllers = containers.Map();
controllers('controller:matlab-local') = @(controllerId, taskJson) ...
    runMatlabOperator(controllerId, taskJson, implementations);

results = dagml.executeExecutionPlanPhase( ...
    executionPlanJSON, controllerManifestsJSON, ...
    'run:matlab-local', 42, 'FIT_CV', controllers, ...
    getenv('DAGML_NATIVE_LIBRARY'));
```

`invokeTrainingLoss` accepts the exact `NodeTask` JSON emitted by DAG-ML. This
avoids ambiguous host round-trips for single-element JSON arrays. The MEX bridge
asks the DAG-ML C ABI to select the phase-filtered role and task-owned
attestation, then executes the function handle on MATLAB-owned values.
`PREDICT`, stale attestations, and invalid role indexes fail in the native core
before the function handle runs. Each MATLAB process, parallel worker, Octave
process, or replay process must load the DAG-ML native library and register its
own local functions.

`dagml.executeExecutionPlanPhase` runs the native sequential scheduler for a
validated `ExecutionPlan`, verifies trusted manifests before dispatch, and lets
callbacks return `NodeResult` structs or JSON. It is a local orchestration and
conformance bridge; long-lived native handle ownership remains with the opaque
training/replay APIs that retain controller registries.

Build the native library and MATLAB MEX bridge with:

```bash
cargo build -p dag-ml-capi --release
export DAGML_NATIVE_LIBRARY="$PWD/target/release/libdag_ml_capi.so"
```

```matlab
addpath('bindings/matlab');
buildNativeBinding
```

For GNU Octave on Linux:

```bash
mkoctfile --mex bindings/matlab/native/task_training_loss_binding.c \
  -o bindings/matlab/+dagml/taskTrainingLossBindingNative.mex
mkoctfile --mex bindings/matlab/native/execution_plan_phase.c \
  -o bindings/matlab/+dagml/executeExecutionPlanPhaseNative.mex
```

Run the binding test with GNU Octave:

```bash
octave --no-gui --quiet --eval \
  "addpath('bindings/matlab/tests'); local_implementation_registry"
```

For host hyperparameter search, `dagml.hostHpoSearch` calls the standalone
native scheduler through executable JSONL operator and optimizer adapters.
The three JSON files are an `ExecutionPlan`, `ExternalDataPlanEnvelope`, and
`HostHpoSearchRequest`:

```matlab
result = dagml.hostHpoSearch( ...
    'plan.json', 'envelope.json', 'hpo.json', ...
    './matlab-operator-adapter', './matlab-optimizer-adapter', ...
    'checkpoint', 'search.checkpoint.json', ...
    'operatorPersistent', true);
```

The executables can launch MATLAB or Octave and must implement the same
JSONL protocol as the [CLI examples](../../examples/adapters/). DAG-ML owns
trial scheduling, fold scoring, pruning, selection, and durable checkpoints.
The wrapper is POSIX-only and accepts `parallelTrials = N`. DAG-ML starts an
isolated operator-adapter process per candidate and keeps optimizer callbacks
on the coordinator thread. Run its smoke with `addpath('bindings/matlab');
addpath('bindings/matlab/tests'); host_hpo_search` in MATLAB or Octave.

For a no-splitter pipeline, capture a signed initial REFIT package and replay
PREDICT on a separate V2 cohort through the native CLI:

```matlab
refit = dagml.initialFullRefit( ...
    'pipeline.json', 'controllers.json', 'train-envelope.json', ...
    'training-ids.json', './matlab-operator-adapter', 'full-refit.package.json');
prediction = dagml.predictInitialFullRefit( ...
    'full-refit.package.json', 'predict-envelope.json', ...
    './matlab-operator-adapter', 'artifact-handles.json', 'output-ids.json');
```

MATLAB/Octave retains host model sidecars. The replay adapter must resolve
exactly the artifact handles attested in the package. Run the wrapper smoke
with `addpath('bindings/matlab'); addpath('bindings/matlab/tests');
initial_full_refit`.

`dagml.cvRefitPredict()` runs a pipeline with CV, winner REFIT and PREDICT in
one native CLI session, returning the bundle, OOF averages and replay
prediction blocks. It keeps host model handles alive for that replay only;
the bundle JSON alone does not contain MATLAB/Octave model sidecars. Its smoke
is `addpath('bindings/matlab'); addpath('bindings/matlab/tests');
cv_refit_predict`.

For replay after restarting MATLAB/Octave, persist model sidecars in the host
and pass their exact invocation-local handle JSON to `dagml.replayBundle()`.
Its `envelopes` argument is a `containers.Map` from data key to V2 JSON path.
DAG-ML validates the artifact map against the bundle; the adapter reloads and
checks its sidecar bytes. The CLI's `--artifact-handles` selects this real
replay path; omitting it still uses mock handles for conformance examples.
