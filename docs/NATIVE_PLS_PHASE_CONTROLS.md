# Native PLS phase controls

The additive closed profile `n4m.pls_role_pipeline.v1` is selected explicitly.
It leaves the historical PLS/N4MM/HPOv1 contracts unchanged. Qualification of
new source requires the native gates; this document makes no release claim.

## Compilation and public Python use

`dag_ml.methods_pls_role_pipeline_contract(params)` returns the canonical
operator and controller manifest without opening Methods or reading data.
The SDK exposes this profile through `nirs4all.run(..., engine="native",
native_profile="n4m.pls_role_pipeline.v1")`. The dataset is the existing raw
mapping `X`, `y`, explicit `sample_ids`, and optional target names/groups.
The profile accepts one PLS recipe or SNV → Savitzky–Golay → PLS. It does not
accept extended graphs, model families, augmentation or native nested HPO.

Parameters contain `native_profile`, positive integer `n_components` and
boolean `scale`. The optional `pipeline` block is the existing bounded
`n4m.snv_savgol_smooth.v1` constructor: odd window 3..501, degree below window.
The canonical Methods recipe uses SG mode `interp`; PLS uses `nipals`,
centering and two native flags `scale_x`/`scale_y`. Public `scale` controls
both flags atomically. Other sklearn PLS parameters retain their historical
default restrictions; `tol`, `max_iter`, `epochs` and warm-start are not
implemented controls here.

## Signed parameters by phase

Signed `ParameterPatch` entries use namespace `fit`, path `["train_params"]`
or `["refit_params"]`, and a closed object containing `n_components` and/or
`scale`. There is one owner per phase object; nested paths and duplicates
are refused. The patch policy explicitly permits `operator` and `fit`.
Operator patches for this profile are limited to scalar n_components/scale.
Other profiles continue to refuse the unexecuted fit/control namespaces.

Constructor values form the base. Train overrides that base. An optimizer
proposal may replace searched keys only when no train override owns them;
those collisions fail before provider access or optimizer creation. REFIT
explicitly overrides the selected candidate. The effective plan retains
candidate parameters and execution-derived `phase_controls`; its fingerprint
binds both recipes. CV predictions/cache evidence therefore remains attached
to the candidate that produced it, while REFIT/PREDICT recompute the distinct
final recipe. Derived campaign controls also bind checkpoint provenance.

## Native HPO version 2

Campaign metadata `methods_hpo_operation` uses the existing descriptor fields
plus `schema_version: 2` and the explicit `native_profile`. Parameter mappings
are identity mappings for a nonempty subset of n_components/scale. The
versioned profile owns the atomic scale projection; arbitrary paths are refused.

```json
{"parameters": [
  {"kind":"int","name":"n_components","low":1,"high":3,"step":1,"log":false},
  {"kind":"categorical","name":"scale","values":[false,true]}
]}
```

Either axis may be omitted. Numeric/string boolean aliases, reordered domains,
unknown axes and ambiguous owners fail closed. Normal scheduler FIT_CV,
selection, selected rerun and REFIT execute each recipe. The opaque Methods
optimizer remains controller-owned. Resume requires the complete portable
package and matching data, graph, folds, controls and study; only the total
trial budget may grow. This is global native tuning, not independent searches
inside every outer fold.

## Export, inspection and replay

The Rust controller is `controller:methods.native.regression`, version 1.0.0;
its RAW plugin is `dagml.methods.native.regression`, version 1.0.0. It never
uses the WASM or R identities. ABI 2.14 RolePipeline primitives execute the
recipe and emit one N4ME state per stateful step inside the existing bounded
`dagml.methods.regression.v1` wrapper. Archive V2 retains the existing
`methods_role_pipeline` family, exact content-addressed URI, SHA, trust and
complete binding coverage. No pickle or empty placeholder N4MM is emitted.

`dag_ml.inspect_methods_role_pipeline_params(payload_bytes,
methods_library_path)` imports the states and reads official native parameter
getters. PLS import checks saved scale flags against the actual sub-model;
these are not inferred from arrays or copied unchecked from wrapper JSON.
The result contains profile/node/fingerprint, target and feature names,
verified per-step parameters, and `model_params` with n_components/scale.
Inspection performs no fit and grants no controller trust. Validate the archive
and signed package before inspecting selected artifact bytes.

`dag_ml.execute_loaded_methods_predictor_replay` uses the existing signed
request/envelopes and named numeric-input map. It registers the native
controller, hydrates fresh states, predicts and releases them without a Python
callback or training. Width/order/identity/recipe contradictions refuse replay.
All state handles are invocation-local; error cleanup and releases are
idempotent. The public SDK must route this profile through that native replay
instead of the historical Core N4MM-only matrix predictor.
