# Methods HPO adapter boundary

`dag-ml-core::hpo` is the controller contract for the native Methods optimizer.
It does not implement an optimizer, emulate ask/tell or use a Python manager.
The native coordinator owns the independent study boundaries and their
selection; the Methods adapter supplies each study's optimizer state.

DAG-ML retains ownership of fold identity, influence evidence, lineage,
selection, and refit.  The Methods adapter owns only numeric optimization state
and its trial state machine:

- `ask` and `ask_batch`;
- `tell`, intermediate reports, failed trials, and pruned trials;
- `best` and `trials`;
- in-memory `N4MOPT` save/load through the official binding (not bundle persistence).

Host ask/tell search can also select recipes compiled from the existing DSL
generators. See [structural host HPO](HOST_STRUCTURAL_HPO.md) for the immutable
catalogue, active parameter axes, checkpoint identities and winner REFIT path.

The `dag-ml-core` crate exposes the opt-in public `methods-optimizer` Cargo
feature through the published dynamic `n4m` 0.4.0 binding (Methods ABI 2.17,
generic estimator roles). Default builds leave
that feature disabled and refuse HPO before an
objective can be called, with the typed
`HpoError::MethodsOptimizerFeatureDisabled`. The integration helper enables
that published feature and supplies a compiler-only native-test selector; the
selector is not a Cargo dependency or a production route. There is no sibling
manifest or sibling source dependency, and the registered official adapter
remains mandatory.

## Native PLS fold scope

The explicit `n4m.pls_role_pipeline.v1` profile accepts an additive
`metadata.methods_hpo_operation` schema 3 descriptor with `scope: "fold"`.
Existing schema 1/2 campaign operations remain unchanged. This first fold
slice requires REFIT and partition CV. The descriptor attests one
`inner_fold_sets[outer_fold_id]` entry, with an explicit parent ID and complete
FoldSet, per outer fold; `refit_inner_fold_set` covers the full training pool.
Each inner universe must equal its parent's training identities and restricted
group map/exclusion authority. All fold IDs are globally distinct. Feature
matrices remain host-owned.

The coordinator runs a separate Methods study in each outer training pool,
chooses that pool's winner from inner OOF, then freshly fits the winner on
outer training rows and evaluates outer validation rows. Outer validation
targets never influence this inner selection. A separate REFIT study selects
the durable RAW RolePipeline recipe; explicit REFIT overrides are then applied.
No study transfers fitted weights to another. `trials` is the total budget
per study, including resumed history. Methods retains bool categorical kinds
and indices, while `winner_params` exposes actual bool/int values for searched
axes only.

Outcome V2 and Bundle V2 carry the same optional
[`methods_hpo_fold_state`](contracts/methods_hpo_fold_state.v1.schema.json).
Campaign `methods_hpo_resume_state` is absent in fold mode. The state preserves
the unexpanded `base_plan`, root provenance, coordinator `relations` and
`selected_variant_id`, plus
`outer_scopes` sorted by outer fold ID and a required `refit_scope`. Each scope
includes its inner FoldSet, plan/parameter fingerprints, complete resume
ledger and local winner. Scope `phase` is `FIT_CV`/`REFIT`, while scope IDs
are `<operation>:scope:fit_cv:<outer-fold>` and `<operation>:scope:refit`.
The root selected variant names the independent REFIT winner; outer OOF rows
retain their own fold-local effective parameter fingerprints. The root
ScoreSet evaluates honest outer OOF, not the REFIT study's inner score.

The Python JSON functions `validate_methods_hpo_fold_state_json` and
`methods_hpo_fold_state_from_package_json` delegate validation to Rust.
The typed facade supports:

```python
state = dag_ml.methods_hpo_fold_state_from_package(package)
outer = state.to_dict()["outer_scopes"]
refit = state.to_dict()["refit_scope"]
```

The reader validates the containing package before returning a
`MethodsFoldHpoState`; a missing state is an error. Its snapshots are independent
copies, and reading performs neither FIT nor HPO. Direct standalone state
validation is useful for metadata checks but cannot authorize resume. Resume
requires the complete Package V2 and unchanged data, fold, group, controller,
phase-control and selection authorities. Swapping checkpoints between scopes
is refused even when the containing package fingerprint is recomputed.
Saved RAW replay likewise uses native package/trust validation and never fits
or starts an optimizer. Warmstart and the broader canonical ND profile remain
separate work.

The tracked root workspace and standalone `dag-ml-py` maturin workspace both
lock that binding from the registry with its published checksum. A local path
patch is evidence only, never a release source.

The `nirs4all_archive_core = "=0.4.3"` development dependency is a registry
baseline used only by the Archive V2 integration tests. It is not the Core
selected by the release train and must not be presented as cross-source Core
qualification; that qualification is owned by a separate integration harness.

## Generic estimator roles

`n4m_host_controller_specs` turns the native method manifest
(`n4m_method_manifest_json`) into one `HostControllerSpec` per graph role:
`controller:n4m.{transformer,selector,regressor,classifier,sample_filter,splitter,augmenter}`.
Each spec selects its methods by the `n4m:<method_id>` operator reference; a
multi-role method is listed once per role and resolves by node kind. The same
derivation is exposed to Python (`dag_ml.n4m_host_controller_specs`) and WASM
(`n4m_host_controller_specs_json`).

With the `methods-optimizer` feature, `register_methods_estimator_controllers`
registers the natively executed roles (transformer, selector, regressor,
classifier, sample filter) as `MethodsEstimatorController`s sharing one
invocation-local feature store, so `exclude -> transform -> model` chains run
without a host callback. A node names its method with the reserved `method_id`
parameter; all other parameters are typed by the native manifest JSON (an
integral number is an exact `int`, as numeric generators emit binary64 values).
The controllers read the method catalog from that JSON contract only, so
additive manifest keys of later ABIs (for example 2.14's per-parameter
`recorded` flag and `null` defaults of optional seeds) are accepted. Sample
filters act on training rows only and pass inference rows through.

Model nodes score the surfaces of a host model controller: FIT_CV emits the
fold-validation OOF block plus the in-fold `train` and report-only
`train_pool` blocks, REFIT the `final` block (and a `test` block for an
explicit held-out prediction view), each with its identity-keyed targets.
Classifiers fit on the integral class ids of their single target column
(`labels` fit input), predict class ids (scored as class labels by the
accuracy/balanced-accuracy/F1 metrics) and, when the method defines
probabilities, attest them on the `train`/`train_pool` surfaces.

Refit states are raw `n4m_estimator` artifacts in format N4ME (ABI >= 2.13),
one per node and variant (`artifact:n4m:<node>:<variant>:refit`), with a
`dagml.native_estimator_descriptor.v1` read back from the native state.
Methods whose state retains training rows are refused unless the node sets
`unsafe_flags: ["allow_training_rows_in_artifact"]`, which is recorded in the
lineage. Splitter and augmenter roles are derived for planning but have no
native execution path yet.

Every callback-free Methods entry point registers the same native set
(`register_methods_native_controllers`: Methods PLS, Ridge and the role
controllers) and refuses a plan with any other controller:
`execute_methods_training`, the portable full refit, Package V2/V3 replay and
`run_cv_refit_methods_in_process`. The latter is the callback-free twin of
`run_cv_refit_in_process`: the same compat-DSL compilation, envelope views,
variant SELECT, FIT_CV, REFIT and scoring, with controller manifests derived
from the configured libn4m, host rows supplied once as `methods_inputs_json`
(exactly covering the plan's data bindings) and a target-bound envelope
(relation, data content and target content fingerprints). It returns the
host payload plus the REFIT records and their raw N4ME bytes.

## Training-local runtime route

The official Rust `n4m` binding exposes a `!Send + !Sync` optimizer lifecycle.
`execute_training` therefore uses an invocation-scoped
`HpoExecutionContext` to route creation/restore through the registered campaign
controller; no `Context` or `Optimizer` is stored in the `Send + Sync`
`RuntimeControllerRegistry`.  The context validates the signed
request/projection, controller registry, provider, relations, runtime-derived
training influence and selection policy before it asks Methods for a trial.

The initial executable route is intentionally strict:

- one typed `methods_hpo_operation` campaign metadata descriptor with an
  explicit operation id, target node and registered controller;
- one portable graph operator exactly named `pls` as the target;
- that target must resolve to `controller:methods.pls`; plugin/host model
  classes are refused rather than falling back to fixture scoring;
- direct native-trial parameter to model-parameter mappings;
- one unexpanded base variant, expanded only by native `ask` results.

The HPO operation is not a graph node: it never appears in `GraphSpec`,
`ExecutionPlan`, the predictor closure, node lineage or replay topology. Each
asked trial becomes an ordinary ephemeral DAG-ML variant. The normal
scheduler owns scoped data views, inner/outer fold identity, OOF predictions,
scores and lineage.  It returns only the target producer's OOF-average scalar
to Methods through `tell`; Methods supplies `best`, and its native trial
selection is checked against DAG-ML's `select_candidate` decision.  The winner
is then rerun in the retained context and refit exactly once by the ordinary
training flow using `n4m::Model::fit`/`predict`; the refit controller exports
an `N4MM` model artifact. After all terminal states the session obtains its
incumbent exclusively through Methods `best()`; DAG-ML rejects any mismatch
with `SelectionDecision` in trial/variant, score bits, metric, direction, or
any tied incumbent score. Numeric rows are available only through the explicit
`RuntimeDataProvider::methods_pls_*` capability, whose request carries the
scheduler-selected fold/sample views. A provider without that capability is
refused before attestation or materialization. The typed campaign state is a
first-class bundle artifact, while the runtime-derived `hpo_selection`
influence entry remains in the outcome.

The scheduler routes the typed `RuntimeHpoCampaignTask` through
`RuntimeController::create_tuner_session`. The registered controller is only a
`Send + Sync` configuration/factory; its returned `RuntimeTunerSession` has no
`Send`/`Sync` bound and is created, invoked and dropped inside the sequential
invocation or parallel worker thread. Generic `RuntimeController::invoke` is
not a tuner fallback. This is the only permitted future home for a native
`Context`/`Optimizer`; neither may be put behind a mutex or an `unsafe impl
Send`.

The portable training route uses that generic factory/session. Its native state
is local to `HpoExecutionContext`; an unsupported descriptor is
refused before provider attestation or data-view work. It currently runs on the
sequential scheduler because the capability deliberately does not require a
thread-safe provider. Each completed scheduler CV evaluation is reported to
Methods as an intermediate OOF-average score before a terminal result is sent.
A native pruner may terminalize the trial as `PRUNED` at that report; DAG-ML
never sends a duplicate terminal `tell`. Failed scheduler evaluation/scoring
paths are recorded as native `FAILED` terminals. The native study state is
saved as an opaque `N4MOPT` checkpoint in the bundle and tests verify that an
uninterrupted study and a restored checkpoint produce the same subsequent
proposal sequence.

Refit `N4MM` model bytes are stored in `ExecutionBundle.raw_artifact_payloads`.
Public attached and loaded replay prefer those bytes over any process-local
artifact sidecar: the current controller validates and imports the payload,
creates a fresh invocation-local handle, and then predicts. This lets a
JSON-deserialized outcome replay in a new process/controller without retaining
the refit controller's in-memory handles.

The `dag-ml-core/Cargo.toml` manifest offers an opt-in `methods-optimizer`
feature targeting the published dynamic `n4m` 0.4.0 binding. Local
qualification builds the exact Methods runtime named above without patching
the release dependency. Default builds and extracted crates do not link, load,
or require a Methods checkout. A caller must explicitly configure an absolute
`libn4m` file through `MethodsRuntime::configure` before constructing either Methods
controller; there is no `PATH`, current-directory, sibling-checkout, or
legacy fallback.

The local integration helper activates the published feature and that
compiler-only test selector. It uses the Methods v1.3.2 source commit
`dcc570b3647f77cf0428dd346078f442ed5cd032` (ABI 2.17) to build the explicit
runtime file, but resolves the Rust binding from crates.io. The generic
estimator tests read `parity/fixtures/estimator_roles_n4me.json` from that
checkout (`N4M_ESTIMATOR_ROLES_FIXTURE`, defaulting to
`external/nirs4all-methods`). Build that checkout and invoke
the helper from the workspace root:

```bash
METHODS_SHA=dcc570b3647f77cf0428dd346078f442ed5cd032
git -C /absolute/path/to/nirs4all-methods fetch --depth=1 origin "$METHODS_SHA"
git -C /absolute/path/to/nirs4all-methods checkout --detach "$METHODS_SHA"
make -C /absolute/path/to/nirs4all-methods build PRESET=dev-release

N4M_LIBRARY_PATH=/absolute/path/to/nirs4all-methods/build/dev-release/cpp/src/libn4m.so \
LD_LIBRARY_PATH=/absolute/path/to/nirs4all-methods/build/dev-release/cpp/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
N4M_ESTIMATOR_ROLES_FIXTURE=/absolute/path/to/nirs4all-methods/parity/fixtures/estimator_roles_n4me.json \
dag-ml/scripts/test_methods_optimizer_local.sh
```

The helper refuses a missing, relative, or non-file `N4M_LIBRARY_PATH`; `--probe`
checks exactly that boundary without loading native code. It runs feature-local
clippy with `-D warnings` and tests directly from the sole workspace manifest.
CI builds the same immutable Methods source commit and passes only the resulting
absolute library path.

## Checkpoint contract

Campaign provenance excludes only the requested total trial budget and opaque
resume-package transport. Study/search space, graph, controller, fold, data and
influence identities remain validated. HPO campaign provenance and the
execution-derived phase-control objects use canonical nested JSON object order
even when a downstream Rust crate enables `serde_json/preserve_order`. This
keeps their published default-build fingerprints readable across the aggregate
and standalone Python bindings. Historical graph and controller fingerprints
still bind their original serialized representation: Python dictionary inputs
are sorted before plan construction, while string, path and `JsonContract`
inputs preserve their supplied representation. Increasing a trial budget does
not authorize any change to those scientific identities.

`N4moptCheckpointArtifact` is a live-session envelope for an opaque byte payload.
It records a schema version, `n4m_optimizer_checkpoint` kind, `N4MOPT` format,
and the payload-derived minimum ABI 2.2,
the DAG-ML study/search-space binding, the Methods ABI identity, and a SHA-256
digest. DAG-ML never decodes or mutates the payload. Restore validates the
digest, study binding, search-space fingerprint, and Methods ABI before asking
Methods to replay it. There is no checkpoint migration or fallback replay in
DAG-ML. `N4moptCheckpointReference` remains a proposed archive-member
reference for future externalized checkpoint artifacts. Current bundles retain
the validated opaque checkpoint envelope and raw N4MM model members; resume is
performed by the official binding from that envelope.

## Payload ABI minima

ABI minima describe the oldest reader capable of consuming the payload; they
are not the ABI of the process that wrote it. The provenance is fixed by the
Methods history:

- PLS N4MM is ABI 2.0+: Methods commit `bcad4a682c71b23ffbf11a5d05df4dc2dae510cf`
  declares ABI 2.0, serialization format 1, and the model export/import API.
- N4MOPT is ABI 2.2+: ABI 2.1 commit `b820341483a792d7702137ea441cfa8134fbe7e9`
  only reserves save/load and returns `N4M_ERR_NOT_IMPLEMENTED`; commit
  `4d0cc0a9ac0348bca8a81db150b38417a9e410f3` first implements checkpoint
  save/load and declares ABI 2.2.
- imported-linear N4MM (the Ridge replay payload) is ABI 2.3+: Methods commit
  `2dc536115c5ec438bedb2863b9720ec45641626d` adds the verified imported-linear
  predictor and its N4MM round trip while declaring ABI 2.3.
- SNV -> Savitzky-Golay smooth -> PLS N4MM format 2 is ABI 2.5+: Methods commit
  `2452de5ae347f8a83d81197956b1d027d11594f9` exposes the safe Rust pipeline
  constructor and typed serialized-pipeline inspection.

New writers always emit `abi_major` and `abi_min_minor`. A historical PLS
reference with neither field reads as 2.0. An unversioned Ridge reference is
refused because interpreting absence as 2.0 would allow an ABI 2.2 reader to
attempt an ABI 2.3 payload. Historical N4MOPT envelopes default to 2.2.

## Native predictor descriptor

New PLS and Ridge publications attach
`dagml.native_predictor_descriptor.v1` to their artifact reference. The
descriptor records the artifact SHA-256, controller owner, N4MM format and
writer ABI, storage algorithm, native capability mask, inspected dimensions,
an optional Methods-attested embedded-pipeline description, and a
self-excluding TCV1 fingerprint. Its only metadata authority is the `n4m`
model and pipeline inspectors over the exact exported bytes.

PLS accepts storage algorithm 0 with `PREDICT`; Ridge accepts imported-linear
algorithm 11 with `PREDICT | AFFINE`. Other Methods algorithms remain known to
the native format but are not product-supported by these controllers. Replay
inspects the detached bytes again and refuses a controller/algorithm,
capability, dimension, SHA, or descriptor mismatch before model import.
Historical Archive V2 members without the additive descriptor stay readable,
but they pass the same native algorithm/capability inspection at hydration.
Pipeline N4MM format 2 always requires the descriptor and ABI minimum 2.5;
signed controller parameters are cross-checked against its typed SG values.
