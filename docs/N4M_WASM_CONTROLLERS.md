# Methods regression controllers for JavaScript

Development `main` adds these adapters after tag `v0.3.32`. They are not in
the already published `dag-ml-wasm@0.3.32`; source qualification packages use
the unchanged native version until the next coordinated release bump.

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
release in a fresh controller with fit forbidden. These numerical data are
test fixtures. Their relation identity comes from the native Python binding
and is checked against the exact relation fixture before use.

`scripts/qualify_multimodal_methods_hpo_python.py` replays the same recorded
proposals through native Python callbacks and compares all trial and per-fold
scores, then independently fits the selected source estimator and checks its
prediction against native archive replay. This isolates scheduler/operator
parity; it does not qualify a new
Python optimizer. `scripts/qualify_multimodal_methods_nirs4all.py` checks the
public SDK's corresponding four-source `by_source` pipeline, archive export
and replay with `RolePipeline.fit` forbidden. The existing Python API is
`from n4m.roles import RolePipeline`, usable as `{"model": pipeline}` in
`nirs4all.run(..., engine="dag-ml")`.

This regression controller consumes already resolved numeric projections.
Raw N-D encoders, classification, generated-view consumption attestations,
custom losses, residual targets, additional FIT_CV test streams and nonuniform
fit influence are not qualified
by it. Specialized task requirements are refused. End-to-end portable
four-source archive replay across languages and the corresponding R/Octave
multimodal controllers remain separate backlog work.
