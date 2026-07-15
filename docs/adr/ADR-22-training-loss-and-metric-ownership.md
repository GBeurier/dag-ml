:orphan:

# ADR-22: Training-loss and metric contract ownership

**Status**: proposed (2026-07-15)
**Blocks**: loss/metric roadmap phases L1-L7, nirs4all DAG-ML backend migration, local custom-loss support in every official binding.

## Context

dag-ml currently owns native scoring and selection for a closed metric set, while
training losses are controller parameters with backend-specific defaults. Python
controllers can sometimes accept callable losses, but that behavior is neither a
dag-ml contract nor portable to the other bindings. nirs4all is moving toward a
fully DAG-ML-backed execution skeleton, so placing the authoritative loss or
metric registry in nirs4all would create a second control plane that must later
be removed.

The core must continue to respect its ownership boundary: host controllers own
feature buffers, model objects, training loops and autodiff tensors. Targets,
predictions, identities and canonical descriptors may cross the ABI.

## Decision

dag-ml-core owns the versioned semantic contracts for training losses and
metrics, their pipeline roles, validation, default resolution, fingerprints,
lineage, cache identity, replay requirements and execution attestations.

The semantic contract is separate from the executable implementation:

- `LossSpec` and `MetricSpec` are canonical, serializable and backend-neutral.
- An implementation descriptor binds a spec id to an installed provider,
  implementation fingerprint, capability set and replay/portability class.
- Callable source, bytecode, pickle payloads and language runtime objects never
  enter a portable contract or bundle.
- A binding registry retains process-local implementations and resolves them
  when dag-ml dispatches work to a controller.
- Controllers execute differentiable losses inside their native training loop
  and return an attestation of the effective spec, implementation, parameters
  and reduction. dag-ml rejects mismatches.
- Built-in metrics remain native in Rust when the core owns the required data.
  Custom metric execution may be delegated through the metric-provider protocol,
  but aggregation, selection and score persistence remain core-owned.

The role of a metric is not part of `MetricSpec`. Selection, reporting, early
stopping, tuning, pruning, thresholding and ensemble weighting are distinct
policies that reference a metric and declare their partition, scope and
reduction. Training loss is a separate role attached to a trainable node/output
for `FIT_CV` and `REFIT`.

Every official binding must support local custom-loss registration and
execution: Python callable, R function, JavaScript/WASM function, Rust closure,
C ABI callback plus opaque `user_data`, and MATLAB function handle. This is a
completion requirement, not an optional portability tier. A controller may
declare `not_configurable` only when its underlying algorithm has no custom
objective hook.

Portability and replayability are independent:

- `host_local`: executable in the current binding/runtime only;
- `portable_registered`: equivalent implementations are installed in multiple
  bindings and pass shared conformance vectors;
- `portable_builtin`: canonical built-in semantics are supplied by the project.

An implementation may be reproducibly replayable in one language without being
portable to another language.

## Consequences

- nirs4all adapters translate legacy arguments into dag-ml specs and register
  callables through dag-ml bindings; they do not own a competing registry.
- Loss fingerprints and attestations become part of cache and artifact identity.
- Unknown loss or metric names are errors; backend-specific silent fallbacks are
  removed.
- Local custom losses work in every official language once its binding and at
  least one configurable controller complete the conformance gate.
- Algorithms with fixed analytical objectives remain valid but must expose the
  limitation explicitly.
- Shared changes to metric providers, controller capabilities, task contracts or
  schemas require coordination with concurrent score-provider work. Neither
  implementation may introduce a duplicate provider descriptor or registry.
- The concurrent score-provider work owns `MetricSpec`, its implementation
  descriptor and the typed custom-metric evaluation task. Loss work consumes
  those contracts after their branch/commit is recorded and reviewed; it must
  not create provisional metric types while that dependency is unpublished.
- Every versioned controller/task/result/schema change follows ADR-02 in the
  same batch, including fixtures, read-window/version decision, CHANGELOG and C
  ABI impact.

## Blocks

No binding-specific public custom-loss API should be stabilized before L1 and
L2 freeze the semantic contracts and controller attestation protocol. The ADR
must be accepted before the nirs4all compatibility layer becomes authoritative.
Metric-related L1/L2 source changes are additionally blocked until the active
score-provider work has a named branch/commit/PR and its API is mapped against
this ADR. Loss-only semantic contracts may proceed independently.
