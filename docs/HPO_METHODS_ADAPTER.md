# Methods HPO adapter boundary

`dag-ml-core::hpo` is the controller contract for the native Methods optimizer.
It does not implement an optimizer, emulate ask/tell, use a Python manager, or
perform nested-CV selection.

DAG-ML retains ownership of fold identity, influence evidence, lineage,
selection, and refit.  The Methods adapter owns only numeric optimization state
and its trial state machine:

- `ask` and `ask_batch`;
- `tell`, intermediate reports, failed trials, and pruned trials;
- `best` and `trials`;
- in-memory `N4MOPT` save/load through the official binding (not bundle persistence).

The default build refuses HPO before an objective can be called, with the typed
`HpoError::MethodsOptimizerFeatureDisabled`.  Until `n4m` is published there
is deliberately no public `methods-optimizer` Cargo feature: Cargo rejects it
rather than presenting an inert production surface.  The isolated local
overlay is the only opt-in integration route and it still requires the
registered official adapter.

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

The publishable `dag-ml-core/Cargo.toml` has no Methods feature or sibling path
dependency, so standalone default builds and extracted crates do not require a
Methods checkout. The isolated overlay resolves the unpublished `n4m` crate
from the reviewed Methods commit
`0ef355e6f74573ed07a6920bdeed1a052a6e8312`; it never follows a branch. Until
the official `n4m` crate is published, this is a local prerelease integration
only. Build that exact checkout, then run the overlay from the workspace root:

```bash
METHODS_SHA=0ef355e6f74573ed07a6920bdeed1a052a6e8312
git -C /absolute/path/to/nirs4all-methods fetch --depth=1 origin "$METHODS_SHA"
git -C /absolute/path/to/nirs4all-methods checkout --detach "$METHODS_SHA"
make -C /absolute/path/to/nirs4all-methods build PRESET=dev-release

N4M_BINDING_SHA="$METHODS_SHA" \
N4M_METHODS_REPO=/absolute/path/to/nirs4all-methods \
N4M_LIB_DIR=/absolute/path/to/nirs4all-methods/build/dev-release/cpp/src \
LD_LIBRARY_PATH=/absolute/path/to/nirs4all-methods/build/dev-release/cpp/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
dag-ml/scripts/test_methods_optimizer_local.sh
```

The helper canonicalizes `N4M_METHODS_REPO`, requires the checkout HEAD to equal
the reviewed SHA, and refuses a missing, malformed, overridden, non-Git,
mismatched, or dirty build/binding source. `--probe` performs those checks and
prints the canonical checkout path and SHA without requiring a native library.
The helper temporarily overlays the feature-only manifest, runs feature-local
clippy with `-D warnings`, and restores the normal manifest and lockfile
byte-for-byte. CI checks out and builds the same immutable SHA; default CI does
not link or resolve Methods.
**Do not publish a dag-ml release enabling this feature until the official
`n4m` crate is published and the Methods ABI identity can be obtained from its
public binding API.**

## Checkpoint contract

`N4moptCheckpointArtifact` is a live-session envelope for an opaque byte payload.
It records a schema version, `n4m_optimizer_checkpoint` kind, `N4MOPT` format,
the DAG-ML study/search-space binding, the Methods ABI identity, and a SHA-256
digest. DAG-ML never decodes or mutates the payload. Restore validates the
digest, study binding, search-space fingerprint, and Methods ABI before asking
Methods to replay it. There is no checkpoint migration or fallback replay in
DAG-ML. `N4moptCheckpointReference` remains a proposed archive-member
reference for future externalized checkpoint artifacts. Current bundles retain
the validated opaque checkpoint envelope and raw N4MM model members; resume is
performed by the official binding from that envelope.
