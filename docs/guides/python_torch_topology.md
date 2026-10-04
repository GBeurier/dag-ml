# Python CPU Torch topology profile

The Python structural profile combines four complete aligned named float32/float64
2D sources and a single regression target. An early candidate concatenates its
signed ordered source subset. A learned-late candidate uses two to four distinct
single-source Torch branches and a Ridge meta-model trained on native nested
Validation OOF predictions. GroupKFold outer 3 / inner 2 and independent full-Train
REFIT OOF, HPO proposals, score reduction and SELECT remain DAG-owned.

The existing Python owners execute `DagMLTorchEstimator` with the public
`nirs4all.operators.models.pytorch.mlp.structural_mlp` factory and Ridge SVD.
Every FIT creates a new module. The closed graph `python_torch_profile` signs
source names/widths, root seed, CPU resources and training policy
`validation=none, shuffle=true, early_stopping=false`. Full `source_schemas`
retain exact dtype, axes, units and feature-label identity. The profile also signs
`target_names`, exactly one original nonempty scalar name, and preserves it
through output binding and target-free replay. Actual source and target buffers
must remain finite after their real float32 conversion before fitting; source
buffers receive the same check before PREDICT. Raw `source_selection`
signs feature concatenation order; singleton `source_index` must match its name.

Admission rejects missing/inconsistent profiles for this factory, GPU, workers
above 1, pruning, model-local HPO, preprocessing and alternate owners. The
scheduler resource declaration cpu_threads=1 serializes DAG nodes/folds; it is
not a physical intra-op Torch/BLAS thread cap. Before callbacks, native validation
checks all effective parameters and lr/alpha domains and forces. Each raw scope
has at most 16,777,216 input cells, at most 1,000,000 requested parameters and
100,000,000 declared work units (root Train rows × epochs × hidden_units ×
(selected_width+1)). These are conservative admission bounds, not execution-time
or total-memory guarantees.

Each owner task sets its derived native-task seed before module construction
and shuffled DataLoader creation within a restoring CPU RNG scope. No CUDA
state or global thread configuration is changed. Meta fitting refuses foreign
producers, Train/Test input, outer Validation used as training and incomplete
or differently ordered branch OOF rows.

`capture_torch_topology` retains the exact selected native predictor package
and every REFIT owner artifact before process-local detach. Trusted joblib
archives verify existing byte digests, native package signatures, selected
closure, owners, effective parameters, source schemas and actual fitted CPU
architecture. At the real owner REFIT emission, a provenance record binds the
artifact/node/variant/run, exact full-Train row IDs, effective seed, ordered
sources, schemas, controls, target names and SHA of learned Torch/Ridge state.
Its TCV1 digest is the native ArtifactRef.content_fingerprint; the original
record remains attached independently to the fitted object and owner bundle.
Capture checks the unique REFIT lineage emission and copies this existing
record. Capture/replay compare its native digest and current learned state;
they never create missing origins or re-attest compatible substituted objects.
The trusted joblib ZIP byte digest remains separate; this is no cryptographic
security claim against a malicious Python archive producer. Loaded-package replay uses the existing native sequential
scheduler to PREDICT every retained branch and meta-model on the new cohort;
it reads no targets and performs no FIT/HPO. This is a Python host-sidecar
profile (`allow_host_sidecar`), with no numeric Core/N4MF/N4ME/R/WASM portability
claim. Historical generic Torch defaults and prior serial/parallel Methods
contracts remain unchanged when this profile is absent.

Sources and witnesses are authored but remain unqualified until the complete
phase reviews and grouped native/SDK gates run.
