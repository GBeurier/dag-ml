# DAG-ML MATLAB/Octave binding

`dagml.LocalImplementationRegistry` is the MATLAB-owned process-local
implementation registry. It retains loss and metric `function_handle` objects
under exact DAG-ML descriptors; executable objects are never written into DAG
JSON contracts.

```matlab
implementations = dagml.LocalImplementationRegistry();
implementations.registerLoss(lossReference, @asymmetricLoss);

[value, attestation] = implementations.invokeTrainingLoss( ...
    nodeTask, 1, yTrue, yPred);
result.lineage.loss_attestations = {attestation};
```

`invokeTrainingLoss` accepts only `FIT_CV` and `REFIT`. It validates the active
role against `NodeTask.required_loss_attestations`, executes the local function,
and returns the native-produced attestation only after successful execution.
Each MATLAB process, parallel worker, Octave process, or replay process must
register its own local functions.

Run the binding test with GNU Octave:

```bash
octave --no-gui --quiet --eval \
  "addpath('bindings/matlab/tests'); local_implementation_registry"
```
