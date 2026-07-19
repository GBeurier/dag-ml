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
