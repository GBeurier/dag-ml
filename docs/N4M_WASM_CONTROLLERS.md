# Methods regression controllers for JavaScript

Version `0.3.33` includes these adapters. The older `dag-ml-wasm@0.3.32`
package does not include them.

The Node and browser npm builds expose two optional subpaths:

```js
import { N4mWasmRegressionController } from "dag-ml-wasm/n4m-controller";
import { N4mWasmHostOptimizer } from "dag-ml-wasm/n4m-optimizer";
```

Install `@nirs4all/methods` separately when using these adapters. DAG-ML's base
WASM API continues to work without this optional peer. Initialize both WASM
modules before constructing a controller.

Source builds must run `node scripts/stage_wasm_host_adapters.mjs <package-dir>`
after `wasm-pack build` to add the public modules and package exports. The
non-publishing Methods candidate workflow qualifies both targets; the npm
release workflow stages the same modules in its web package. The base WASM
conformance workflow retains its pinned training-pack authority bytes.

The controller calls native Methods `RolePipeline` for fitting, preprocessing,
prediction and fitted-state export/import. DAG-ML owns fold scheduling, nested
OOF, score calculation, HPO trial state and selection. The host resolves numeric
feature and target buffers by the native task's ordered sample IDs. It does not
implement its own CV loop.

```js
await methods.loadModule();
const controller = new N4mWasmRegressionController({
  methods,
  // Bind the model operators from the compiled graph. NodePlan omits them.
  operators: { "model:ridge": { type: "n4m:models.regularized.ridge" } },
  resolveFeatures: ({ view }) => data.resolveNumericRows(view),
  resolveTargets: ({ sampleIds }) => data.resolveTargetRows(sampleIds),
  targetNames: ["y"],
  digest: sha256Sync,
});
const manifests = [controller.manifest(dagMl)];
const optimizer = new N4mWasmHostOptimizer({
  Optimizer: methods.Optimizer, dagMl,
  space: { alpha: { kind: "float", low: 0.05, high: 2 } },
  options: { sampler: "sobol", pruner: "none", metric: "rmse",
    direction: "minimize", seed: 7 },
  objective: { graphFingerprint, dataFingerprint, targetNode: "model:ridge" },
  persist: snapshot => storage.replaceAtomically(snapshot),
});
try {
  const result = JSON.parse(dagMl.host_hpo_search_json(
    planJson, JSON.stringify(manifests), envelopeJson, requestJson,
    optimizer.committed === null ? undefined : JSON.stringify(optimizer.committed),
    controller.callback, optimizer.callback,
  ));
} finally {
  optimizer.close();
  controller.close();
}
```

`resolveFeatures` and `resolveTargets` return synchronously:
`{sampleIds, matrix: {data: Float64Array, rows, cols}, featureNames?}`. Matrices
are finite and row-major. IDs must exactly match the requested order; duplicate,
missing or reordered IDs fail before fit. Targets have one column per declared
target name. Feature names, when supplied, must match the feature width.

For multiple bindings, blocks are ordered by binding key after removing the
operational phase suffix, and joined by matching
sample identities. Meta-models use the native task's `prediction_inputs`: nested
validation OOF for fitting and outer validation predictions for evaluation.
Refit separates validation OOF fit rows from `:refit` prediction rows;
prediction consumes `:predict` rows with the same feature names used at fit.
The controller rejects training/validation overlap and outer predictions used
as training features. A native `N4mRolePipeline` operator may declare `steps`
with `{class, params}` or `{methodId, params}`; searched node parameters apply
to the last step. Native Methods validates the recipe.

`FIT_CV` models are disposed after evaluation. `REFIT` retains the fitted model
and exports native N4ME states with a SHA-256-attested JSON wrapper. `PREDICT`
uses the retained handle or a freshly hydrated artifact, without fitting.
`artifactPayload(id)` returns a copy. The synchronous portable callback supports
export, hydration and release under bridge schema 1. Payload hashes, node
identity, parameter fingerprint and target names are checked on import.
Retained and hydrated handles are bound to the exact artifact ID and content hash.
Supply a synchronous lowercase SHA-256 digest function for refit/replay; the
browser's asynchronous `crypto.subtle.digest` cannot be passed directly.

Persist the complete optimizer snapshot atomically: it contains the sealed
DAG-ML checkpoint and native N4MOPT bytes. Resume with the same objective,
search space, options and warm-start contract. `persist` returning a Promise is
rejected; asynchronous storage needs an asynchronous orchestration adapter.

## Qualification and current scope

`scripts/test_wasm_n4m_controller.mjs` checks actual Methods Ridge fitting in
22 cases: adversarial IDs, early refusal, artifact/handle binding, multiple
input ordering, OOF-only meta refit and fitted-state replay with no additional fit.
`scripts/smoke_wasm_multimodal_methods_hpo.mjs` runs four independent numeric
projections (NIR, image, series and metadata) and a meta-model, three group-held
outer folds, two inner folds, native Methods Sobol proposals, and exact
resumed/continuous trial equality. It also captures the selected NIR source
estimator in a native archive and checks automatic hydration, prediction and
release in a fresh controller with fit forbidden. It then uses the generic
native training runtime to capture all four source estimators and the OOF-trained
meta-model in a complete native-portable predictor package. Fresh-controller
replay checks five hydrations, five predictions and five releases without fit,
including release after an injected failure. Tampered package/request content,
untrusted manifests and a parallel WASM scheduler are refused before callbacks.
These numerical data are test fixtures. Their relation identity is computed by
the native WASM binding and compared with the frozen native Python receipt.

`scripts/qualify_multimodal_methods_hpo_python.py` replays the same recorded
proposals through native Python callbacks and compares all trial and per-fold
scores, then independently fits the selected source estimator and checks its
prediction against native archive replay. This isolates scheduler/operator
parity; it does not qualify a new Python optimizer. It also hydrates the exact
five-model WASM package in a separate Python process, checks held-out prediction
parity and release of every native state while fit is forbidden.
`scripts/qualify_multimodal_methods_nirs4all.py` checks the
public SDK's corresponding four-source `by_source` pipeline, archive export
and replay with `RolePipeline.fit` forbidden. The existing Python API is
`from n4m.roles import RolePipeline`, usable as `{"model": pipeline}` in
`nirs4all.run(..., engine="dag-ml")`.

This regression controller consumes already resolved numeric projections.
Raw N-D encoders, classification, generated-view consumption attestations,
custom losses, residual targets, additional FIT_CV test streams and nonuniform
fit influence are not qualified
by it. Specialized task requirements are refused. The complete JSON predictor
package is distinct from Core Archive V2 `.n4a`: that closed ZIP profile accepts
only plugin-free `n4m_model` artifacts, not this controller's RolePipeline codec.
Core `.n4a` transport of this complete recipe, raw N-D encoders and corresponding
R/Octave multimodal controllers remain separate backlog work.

## Native training and detached replay

The root WASM module exposes `training_data_identity_json`,
`sample_relation_set_fingerprint_json`, `sign_training_request_json` and
`execute_training_json`. The latter consumes a signed generic `TrainingRequest`,
an envelope map keyed by `node_id.input_name`, and sample relations. The core
derives training influence, executes CV/SELECT/REFIT and captures RAW bytes.
Require `scheduler.kind="sequential"`, `cv_artifacts="discard"`,
`fitted_artifacts="portable_required"`, and null memory/time limits. Outputs
include exact `training_outcome_json` and `portable_predictor_package_json`
strings: preserve those bytes because parsing and reserializing u64 numbers
in JavaScript can invalidate their seals.

For inference, `attach_predict_cohort_to_envelope_json` derives V2 cohort
evidence from independently identified rows, `sign_training_replay_request_json`
seals the output/envelope authorization, and `replay_training_package_json`
validates the package against caller-trusted manifests before executing native
replay. No fitted handles are supplied: RAW payloads hydrate and release for
this invocation only. Host resolvers still own actual feature buffers.

Python has the same native training/replay contracts through
`dag_ml.execute_training(..., artifact_callback=...)` and
`dag_ml.replay_loaded_predictor_package_json(..., artifact_callback=...)`.
Its bridge uses `export` (return byte values), `hydrate` (return `HandleRef`),
and `release` (return `None`). Existing calls without RAW artifacts keep the
optional callback default. RAW coverage, content hashes and sizes are checked
generically in the core, even when the native Methods feature is disabled.
