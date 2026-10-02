# Complete Methods multimodal predictor

`dagml.methods.multimodal.v1` carries one complete early-fusion predictor for
the existing canonical SDK U07 workflow. The Python declaration is
`MultimodalRegressor(..., backend="methods")`; its default sklearn backend is
unchanged. This profile consumes raw NIR, image, series and mixed metadata.
It does not identify the older five-Ridge numerical projection/OOF fixture.

## Numerical and orchestration ownership

Methods learns each source encoder solely from the native task's fitting rows:
population StandardScaler for NIR, unwhitened TensorPCA for image and series,
and population numeric scaling plus learned dense one-hot categories for
metadata. Unknown categories produce all-zero categorical features. Methods
multiplies declared source weights, concatenates in order
`nir,image,series,metadata`, and fits one native Ridge with
`center_x=true,center_y=true,scale_x=false`. PCA uses the native dense full-SVD
kernel; a recorded seed is part of the declaration. Sign ambiguity is handled
in independent subspace/transform qualification, never by copying oracle scores.
Large dimensions do not imply parity with sklearn's approximate solver choices.

IO owns independent raw schemas, sample assembly and source identities. DAG
owns native GroupKFold, scope/row identities, scores, SELECT and REFIT. Host
controllers subset raw arrays by the exact task sample IDs. R/Octave storage
conversion and JSON tensor transport change layout only. There is no host PCA,
scaler, category vocabulary, feature projection, fold loop or score reducer.
Each controller declares the native `dict_by_source` input policy with
`alignment="sample_id"` and no rank restriction. The four bound raw sources
retain their independent tensor shapes and metadata cells through planning;
Methods performs learned encoding and weighted early fusion.
Optional external test views are predicted by the same fold/refit model and
reported as `test`; their targets never enter fitting.

The same three public tuning keys are closed: `model__alpha`,
`source_weights__image`, `transformers__image__n_components`. The canonical
qualification uses eight proposals and three group folds. Production accepts
valid positive fixed source shapes and PCA counts, and a recorded uint32 seed;
the fixture dimensions/counts/seed are not production restrictions.

## Raw input and bounded state

The explicit operator has exactly `type="N4mMultimodalPipeline"`, `recipe` and
`source_schemas`. The recipe has exactly `schema_version=1`, `fusion="early"`,
ordered `source_order`, `encoders`, `source_weights` and `model`. Metadata
positions are numeric 0/category 1 in the first closed profile.

Each source schema has exactly `representation_id,input_shape,dtype,identity`.
Representations are respectively `signal_1d,rgb_image,series_mv,tabular_mixed`.
`input_shape` excludes the sample axis. `identity` is the unchanged UTF-8 text
of the independently resolved IO descriptor, serialized with sorted keys,
compact separators and finite JSON numbers. It captures axes, units,
coordinates, feature identities and native representation. Replay requires
exact equality against current independently resolved schemas before import.

Numeric bindings receive actual float32/float64 tensors with native
rank/shape/strides; metadata cells retain original UTF-8 categories. Limits are
non-sample rank 1–7, positive shape product at most 1,048,576, at most
16,777,216 input values per source, and identity/categorical cell at most 1 MiB.
PCA count must also fit the actual fitting row count; no silent clamping occurs.

The portable wrapper has exactly seven fields:
`schema,node_id,params_fingerprint,target_names,recipe,source_schemas,state`.
There is one named numeric target and one N4MF byte state, bounded to 64 MiB.
N4MF starts with magic, LE uint32 format 1 and writer ABI major/minor/patch;
the native codec validates the complete recipe/schema, learned branch/Ridge
states and final FNV1a64 checksum. DAG validates declarative bounds, selected
parameters, owner, RAW SHA, size and content-addressed path before hydration.
An old ABI library without the new composite symbols cannot run this profile.

## Archive and cross-host replay

Archive V2 adds optional `payloads.methods.multimodal_pipelines` separately from
`role_pipelines`. Each reference is `kind="methods_multimodal_pipeline"`,
`owner="dag-ml"`, `format_version=1`,
`semantic_profile="dagml_methods_multimodal_pipeline_raw_sha256"`, at
`artifacts/{sha256}.json`; semantic SHA equals RAW SHA. The global union of
N4MM/role/composite IDs and paths must be unique and exactly cover package
artifacts. Core stores opaque payloads and verifies inventory/reference closure.

Closed producer pairs are `controller:methods.{python,wasm,r,octave}.multimodal`
and `dagml.methods.{python,wasm,r,octave}.multimodal`, version `1.0.0`.
Fitting uses the execution host's pair. Target-free replay may bind one of the
four independently trusted producer pairs in another host. It preserves the
signed producer owner, package, SELECT and learned state unchanged and reports
the real execution host separately. Cross-pair, unknown owner and untrusted
manifest failures precede native import. Close/EOF/error paths release all
invocation-local models; fitted rows are absent from the persisted wrapper.

Python uses `dag_ml.multimodal_methods.MethodsMultimodalController`; Node/browser
uses `bindings/js/n4m_multimodal_controller.mjs` and the injected actual Core
`readPortableArchiveV2` consumer in `bindings/js/multimodal_archive.mjs`.
R/Octave use the persistent JSONL adapters under `examples/adapters`. Exact
keys containing `:/.`, singleton arrays and seeds up to `2**64-1` survive
transport. Octave requires 7.1+ with actual native JSON support and the public
`n4m.n4m_multimodal_pipeline_mex` binding.

## Qualification commands

Only execute after whole-source freeze and independent reviews. Build current
Methods, DAG Python/WASM and Core bindings from their official scripts/presets.
Core's published Cargo dependencies do not prove the current DAG validator;
capture explicit local Cargo overrides and producer/library hashes. The SDK
helper `tests/integration/api/test_methods_multimodal_u07.py::prepare_qualification`
captures actual public U07 data, signed calls, native grid/HPO outcomes and a
complete archive. It reuses the existing U07 fixture rather than generating a
new scientific data product.

Run `scripts/qualify_u07_methods_multimodal.py --sdk-root ../nirs4all
--host r --output <new-audit-directory>` or provide `--capture <canonical-u07.json>`.
Set `DAG_ML_RSCRIPT`, `R_LIBS`, or `DAG_ML_METHODS_WASM_DIST`/`DAG_ML_NODE` as
appropriate. Octave uses `DAG_ML_OCTAVE`, `DAG_ML_OCTAVE_METHODS_PATH` and optional
`DAG_ML_OCTAVE_MEX_PATH`. Dedicated R/Octave entry scripts share this driver.
`scripts/smoke_wasm_u07_methods_multimodal.mjs --campaign <capture>
<methods-dist> <dag-wasm-package> <result.json>` exercises compiled WASM DAG
training/HPO plus real Core archive read; `DAG_ML_CORE_JS_ENTRY` must select
the freshly built aggregate entry.

The complete declarative gate is `tests/test_methods_multimodal_contract.py`.
Actual controller gates are `tests/test_methods_multimodal_controller.py` with
`DAG_ML_U07_CAPTURE` and `DAG_ML_REQUIRE_U07_NATIVE=1`. Set
`DAG_ML_REQUIRE_U07_R=1`, `DAG_ML_REQUIRE_U07_WASM=1` and/or
`DAG_ML_REQUIRE_U07_OCTAVE=1` to make absent required hosts fail rather than skip.
Qualification records actual counts, native scores/SELECT, raw dimensions,
artifact/library identities, unknown/schema refusals and released no-fit
lifecycles. Source-only checks or missing-runtime skips do not qualify Octave
or browser execution. Fresh installed Python replay with initial workspace
removed and FIT/HPO forbidden belongs to the SDK integration gate.
