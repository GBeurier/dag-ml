:orphan:

# Native Training Loss and Metric Roadmap

## Outcome

dag-ml becomes the single control plane for training-loss and metric semantics.
Users can register a local custom loss in every official nirs4all language,
execute it through a compatible local controller, and inspect an attestation
showing the exact implementation used during cross-validation and refit.

This roadmap implements the ecosystem's approved native-first loss-management
target. It keeps five concepts independent:

1. training loss optimized by a model controller;
2. early-stopping monitor evaluated inside a fit loop;
3. selection metric evaluated on consolidated candidate predictions;
4. reporting metrics persisted without an implicit decision;
5. tuning, pruning, threshold and ensemble policies that reference metrics.

The roadmap is native-first at the contract and orchestration layers. It does
not move feature matrices, model objects or autodiff tensors into Rust.

## Non-negotiable completion criteria

The program is complete only when all of the following are evidenced in current
source, tests and published branches:

- dag-ml-core defines and validates versioned `LossSpec` and `MetricSpec`
  contracts independently of nirs4all and any single ML backend.
- Compiled plans contain fully resolved effective defaults. Runtime fallbacks
  from an unknown name to RMSE or MSE do not exist.
- `FIT_CV` and `REFIT` tasks carry a resolved loss implementation reference.
- Controller results attest the effective loss spec, implementation fingerprint,
  parameters and reduction; mismatches fail the run.
- Loss spec and implementation fingerprints participate in lineage, artifact
  identity, cache namespaces and replay validation.
- Built-in metrics remain core-owned. Custom metric providers integrate with the
  native aggregation, selection and score-persistence path.
- Python, R, JavaScript/WASM, Rust, C ABI and MATLAB each expose local loss
  registration and execute a local callback in a training conformance test.
- Each officially supported nirs4all language has at least one controller that
  consumes the local loss callback during real training.
- A local custom-loss nirs4all pipeline passes `FIT_CV` and `REFIT` in every
  official language and returns a valid attestation.
- Fixed-objective controllers declare `not_configurable` and reject a requested
  custom loss before training.
- No bundle or manifest serializes callable code, bytecode, pickle data or a
  language runtime object.
- Targeted tests pass for every batch; full repository gates pass after each
  major contract/runtime, bindings and nirs4all integration batch.
- Every implementation batch receives an independent review before merge or
  publication.

Passing Python-only tests, accepting a callable without invoking it, or storing
a custom loss as opaque metadata does not satisfy this roadmap.

### Evidence authority by repository

This file tracks the ecosystem outcome, but no dag-ml PR alone can close
cross-repository criteria. The authoritative evidence is:

| Criterion | Owning repository | Closing evidence |
| --- | --- | --- |
| semantic specs, planning, attestation, cache/replay and native metrics | `dag-ml` | full dag-ml gate listed after L2 |
| custom metric provider contract and typed evaluation task | score-provider work in `dag-ml` | provider-targeted tests plus the full dag-ml gate; branch, commit and PR recorded in L0 |
| Python registry transport | `dag-ml` (`dag-ml-py`) | targeted binding tests plus full dag-ml gate |
| Python TensorFlow/PyTorch controllers and legacy API migration | `nirs4all` | Ruff, mypy, unit/integration tests and examples gate listed after L4 |
| Rust closure and C ABI callback | `dag-ml` | core/C ABI targeted tests plus full dag-ml gate |
| aggregate Rust/Python/R/JS-WASM/MATLAB exposure | `nirs4all-core` | all binding commands listed after L6 |
| full-Python versus portable-binding parity | `nirs4all-core` with `nirs4all` fixtures | native/binding/full-Python parity suite |

The cross-language rows are tracked-but-external from dag-ml's perspective.
Their owning repository must publish its own reviewed PR and green gate before
the ecosystem criterion can be marked complete.

## Ownership and concurrent score-provider work

A concurrent agent is implementing a score provider. Until that work lands, the
following ownership boundary prevents duplicate or conflicting abstractions.

**Dependency state at roadmap authoring**: the score-provider work is active but
has no published branch or commit available in this worktree. Metric-contract
implementation is therefore **blocked**. L0 must replace this paragraph with the
exact branch/commit/PR and a checked-in API map before any metric-related L1/L2
source or shared-schema edit begins. Loss-only contracts may proceed because
they do not define metric provider execution.

| Surface | Loss roadmap ownership | Score-provider ownership | Integration rule |
| --- | --- | --- | --- |
| Training objective semantics | `LossSpec`, loss implementation descriptor, reduction and required inputs | none | loss work leads |
| Metric semantics and provider execution | metric role requirements only | `MetricSpec`, metric implementation descriptor, provider registry/dispatch, typed evaluation task and score result | consume the landed provider API; do not create parallel types |
| Built-in scoring | no semantic rewrite | native metric calculation | preserve core ownership |
| `ControllerCapability` | loss-specific capabilities | metric-provider capabilities | additive merge after provider review |
| `NodeTask` / `NodeResult` | fit loss resolution and attestation | metric task/result fields | update once, with both contracts represented |
| Contract schemas | loss-specific schemas initially | score-provider schemas | shared schemas change only after branch comparison |
| C ABI callbacks | loss callback and lifecycle | metric callback if required | share lifecycle/versioning conventions |

Before editing a shared surface, the implementer must inspect the current branch,
the score-provider branch or worktree, and the target diff. If the provider API
already supplies a generic implementation descriptor, the loss work extends it
instead of creating `LossImplementationDescriptor` in parallel.

The loss roadmap does not create `MetricSpec` or
`MetricImplementationDescriptor`. Their authoritative definitions and typed
custom-metric evaluation task must land with the score-provider work. If that
work does not provide them, L0 must record an explicit ownership amendment and
receive independent review before L1 resumes. A shared implementation descriptor
is preferred only if it can represent loss and metric semantics without moving
objective direction into the provider descriptor.

## Contract model

### Semantic specs

`LossSpec` is backend-neutral and contains only canonical data:

- schema version and stable, versioned logical id;
- built-in/custom kind;
- task and prediction kinds;
- output/head applicability;
- objective (`minimize` in the first schema version);
- reduction semantics;
- required declared inputs such as target, prediction, sample weight or mask;
- canonical JSON parameters;
- capability requirements such as differentiability and distributed reduction;
- canonical spec fingerprint.

The score-provider-owned `MetricSpec` contains the equivalent metric semantics,
plus objective direction, supported unit levels and decomposition/reduction
behavior. It does not encode whether the metric is used for selection, reporting
or another policy. This roadmap consumes that contract after its provider branch
lands; it does not define a competing wire type.

### Implementation descriptors

An implementation descriptor records:

- semantic spec id and fingerprint;
- provider and binding id;
- implementation version and fingerprint;
- supported controller/backend families;
- runtime requirements and capabilities;
- replayability and portability class;
- process-local registry key when applicable.

The registry key is an opaque lookup token, not an import instruction. Automatic
imports from untrusted manifests are forbidden.

### Pipeline roles

The compiled objective/evaluation plan contains typed references for:

| Role | Normal scope | Owner of execution | Owner of final decision |
| --- | --- | --- | --- |
| training loss | trainable node/output, variant, fit phase | host controller | host optimizer under dag-ml contract |
| early-stopping monitor | node/output and validation partition | host controller | host controller, attested to dag-ml |
| selection metric | output/target/campaign over OOF predictions | core or metric provider | dag-ml selection policy |
| reporting metric | output and requested partitions | core or metric provider | no implicit decision |
| tuning/pruning objective | search study/trial | core or provider | dag-ml tuner policy |
| ensemble weighting metric | aggregation policy | core or provider | dag-ml aggregation policy |
| threshold metric | classification output | core or provider | dag-ml threshold policy |

The same `MetricSpec` may be referenced by several policies. Each policy supplies
its concrete partition, unit level, reduction and missing-value behavior.

### Default resolution

Defaults are resolved at compile time in this order:

1. explicit node/output or variant override;
2. explicit controller/model profile;
3. explicit campaign default;
4. dag-ml task-family default;
5. validation error when no compatible value exists.

The resolved plan is fingerprinted and persisted. `FIT_CV` and `REFIT` use the
same effective loss unless an explicit, traceable phase override is present.

## Delivery batches

Each batch is independently reviewable. A batch is not accepted based only on
the next batch passing; its own contract, negative tests and diff review must be
complete.

### L0 - Baseline, ADR and safety inventory

**Repositories**: dag-ml, nirs4all, nirs4all-core.

**Deliverables**:

- accept ADR-22 and this roadmap;
- inventory every current training-loss, selection, reporting, early-stopping
  and tuning default by controller and language;
- record every unknown-name fallback and every duplicated metric implementation;
- map concurrent score-provider types and reserve shared integration points;
- add focused characterization tests for current defaults before behavior changes.

**Acceptance evidence**:

- inventory links each default to source and a characterization test;
- the score-provider branch/commit/PR and API map are recorded in this roadmap;
- no source changes overlap the active score-provider worktree;
- independent documentation review confirms ownership and terminology.

### L1 - Native loss contracts and metric integration gate

**Repository**: dag-ml.

**Deliverables**:

- Rust types for loss semantic specs, loss role references and portability/
  replayability classes;
- canonical validation and TCV1 fingerprints;
- loss JSON schemas, negative fixtures and contract conformance entries;
- versioned built-in loss catalog descriptors without importing ML frameworks;
- compile-time default-resolution contract.

Metric work in this batch is integration-only and remains blocked until the
score-provider artifact identified in L0 lands. Once unblocked, L1 consumes its
`MetricSpec`, metric implementation descriptor and typed evaluation task, adds
pipeline-role references, and verifies that native aggregation, selection and
score persistence consume provider results without duplication.

**Required negative cases**:

- empty/unversioned id;
- unknown task or prediction compatibility;
- non-canonical parameters;
- unsupported reduction/input combination;
- callable/code payload present in canonical JSON;
- mismatched embedded fingerprint.

The score-provider contract must independently test a custom metric without an
objective, non-finite provider output, wrong scope/coverage and mismatched
provider fingerprint before metric integration is accepted here.

**Targeted validation**:

- contract module unit tests;
- schema fixture validation;
- CLI validation for one valid and one invalid spec.

Any versioned schema change follows ADR-02 in the same commit: additive/optional
wire shape where possible, explicit version/read-window decision, updated schema
and fixtures, conformance-pack entry, CHANGELOG note and C ABI decision.

**Independent review focus**: backend neutrality, schema evolution, canonical
fingerprints, absence of executable code, compatibility with the score-provider
descriptor.

### L2 - Controller resolution and attestation protocol

**Repository**: dag-ml.

**Deliverables**:

- loss-specific controller capabilities and manifest declarations;
- resolved loss reference in `FIT_CV` and `REFIT` tasks;
- mandatory loss application attestation in controller results;
- mismatch rejection for spec, implementation, parameters and reduction;
- loss identity in lineage, artifacts, cache namespaces and replay checks;
- explicit `controller_internal` and `not_configurable` modes;
- early-stopping record distinct from final OOF scoring.

L2 also integrates the score-provider-owned typed custom-metric task and verifies
finite values, scope, sample coverage and implementation fingerprint before
native aggregation or selection. L2 does not implement a second provider path.

**Targeted validation**:

- mock controller applies and attests a custom loss;
- unknown registry key fails before fit;
- false or stale attestation fails after fit;
- `FIT_CV`/`REFIT` divergence fails unless explicitly configured;
- cache key changes with implementation fingerprint;
- detached replay fails when a host-local implementation is unavailable.
- custom metric provider output with invalid scope, coverage, value or
  fingerprint is rejected before selection.

Changes to `ControllerManifest`, `ControllerCapability`, `NodeTask`,
`NodeResult`, execution plans, cache contracts or ABI snapshots must apply the
ADR-02 migration checklist in the same commit and retain compatible fixtures or
an explicit versioned migration edge.

**Major-batch gate after L2**:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
python3 scripts/check_so_freshness.py
python3 scripts/validate_contracts.py
```

**Independent review focus**: lifecycle invariants, spoof-resistant attestation,
cache/replay correctness, no feature-buffer or tensor crossing, score-provider
merge quality.

### L3 - Python reference binding

**Repository**: dag-ml.

**Deliverables**:

- process-local Python loss and metric registries owned by `dag-ml-py`;
- callable lifetime tied to the owning run/registry and explicit detach behavior;
- GIL, thread and process capability validation;
- generic callback transport without TensorFlow/PyTorch dependencies in
  `dag-ml-py`;
- direct callable convenience registration producing a host-local descriptor;
- typed Python wrappers for specs, refs and attestations;
- no-pickle/no-callable serialization tests.

**Targeted validation**:

- pure-Python reference controller invokes the registered loss in CV and refit;
- counter/spy proves the callback was called with declared inputs;
- callback exception maps to the unified dag-ml error taxonomy;
- process replay requires registry reconstruction;
- detach releases the callable and preserves portable result data.

**Independent review focus**: ownership, GIL safety, process behavior, error
mapping and no backend coupling. Error mapping follows ADR-11; GIL and async
lifecycle decisions follow ADR-15.

### L4 - nirs4all Python migration

**Repository**: nirs4all.

**Deliverables**:

- legacy `loss`, `metric`, `metrics` and `direction` arguments map to dag-ml
  semantic specs and typed policy roles;
- TensorFlow and PyTorch controllers consume resolved built-in/custom losses;
- unknown loss names are errors, never MSE fallbacks;
- scikit-learn controllers declare configurable/internal objective behavior;
- Optuna references metric specs and can tune loss parameters structurally;
- nirs4all scoring, selection and persistence delegate to the native DAG-ML path;
- compatibility defaults remain visible in the effective plan.

**Targeted validation**:

- PyTorch custom differentiable loss changes gradients and is called in CV/refit;
- TensorFlow equivalent when the optional dependency is installed;
- unknown-name and unsupported-estimator tests;
- selection/reporting remain independent from training loss;
- early stopping can monitor a different metric;
- Optuna direction is explicit for custom metrics.

**Major-batch gate after L4**:

```bash
ruff check .
mypy nirs4all
pytest tests/unit/
pytest tests/integration/
```

Run the full examples gate required by the nirs4all repository after the public
API migration is stable.

**Independent review focus**: no second registry, backend adapters only in
controllers, public compatibility, actual gradient use and default parity.

### L5 - Rust and C ABI local registries

**Repositories**: dag-ml, nirs4all-core as consumer.

**Deliverables**:

- Rust traits/closures for process-local loss execution;
- C ABI loss vtable with explicit version, callback, release and `user_data`
  lifecycle;
- panic/exception containment and deterministic error mapping;
- capability and implementation-descriptor inspection APIs;
- C and Rust training conformance controllers that invoke a local loss.

**Targeted validation**:

- Rust closure invocation and drop-count test;
- C callback invocation, owned/borrowed lifecycle and ABI snapshot tests;
- null callback, stale `user_data` and callback-error negative tests;
- attestation and cache fingerprint parity with Python.

**Independent review focus**: ABI ownership, unwind safety, versioning and exact
resource release.

### L6 - R, JavaScript/WASM and MATLAB local registries

**Repository**: nirs4all-core, consuming the upstream dag-ml contracts.

**Deliverables**:

- `register_loss`/`register_metric` in R, JavaScript/WASM and MATLAB;
- local runtime-object retention and explicit release;
- JavaScript in-process callbacks and worker-local registration by id;
- one configurable training controller per language;
- spec/attestation inspection matching Python/Rust/C.

**Targeted validation**:

- one local custom loss is invoked during both CV and refit in each language;
- a callback exception/error crosses the binding correctly;
- unavailable worker registration fails before training;
- fixed-objective controllers reject custom losses;
- no function source appears in serialized output.

**Major-batch gate after L6**:

```bash
make test
cargo test --workspace
PYTHONPATH=bindings/python/src python -m unittest discover -s bindings/python/tests
npm test --prefix bindings/wasm
mkdir -p dist/r && cd dist/r && R CMD build ../../bindings/r && cd ../.. && R CMD check --no-manual dist/r/nirs4all_*.tar.gz
octave --quiet --eval "run('bindings/matlab/tests/smoke.m')"
```

Run the native-vs-binding and core-vs-full-Python parity suites after the
language package gates. Missing local R or Octave toolchains defer only those
commands to CI; their green CI jobs remain required evidence.

**Independent review focus**: language-native ergonomics, worker/process
lifecycle, callback invocation proof and conformance with the shared ABI.

### L7 - Portable registered losses and composed objectives

**Repositories**: dag-ml and binding consumers.

**Deliverables**:

- packaging descriptor for equivalent multi-binding custom implementations;
- conformance vectors covering values, reductions, weights, masks, gradients or
  finite differences where appropriate, and edge cases;
- `CompositeLossSpec` with explicit components, coefficients and schedules;
- multi-objective tuning/pruning only after single-objective contracts stabilize.

This phase is not required for local custom losses. It is required only to claim
that one custom semantic id is portable across languages.

## Independent review protocol

Every implementation batch receives a review that is independent from the
implementer. The reviewer must inspect the diff and run or inspect the batch's
targeted tests. Reviews are recorded in the PR description or a checked-in
review note with:

- reviewed commit SHA and scope;
- findings ordered by severity with file/line references;
- contract and backward-compatibility assessment;
- concurrency assessment against the score-provider branch;
- tests inspected or rerun;
- explicit `approved`, `approved_with_followups` or `changes_requested` result.

The implementation agent resolves every blocking finding and requests a second
review of the resulting commit. A self-review may supplement but never replace
the independent review.

## Testing cadence

Use targeted tests during implementation. Run full gates only after the major
batches identified above or before publication:

- L1: focused contract/schema tests;
- L2: targeted runtime/controller/cache/replay tests, then full dag-ml gate;
- L3: targeted Python binding tests;
- L4: targeted controller/API tests, then full nirs4all gate;
- L5: targeted Rust/C ABI tests;
- L6: targeted per-binding tests, then all binding and parity gates;
- L7: targeted conformance vectors, then all affected full gates.

Any contract schema change still runs its contract validator immediately; this
is a targeted correctness check, not a full repository test suite.

## Publication strategy

Use one branch and draft PR per independently reviewable repository batch. Do
not stage unrelated worktree changes. Branches use `agent/<batch-description>`
and commits use scoped Conventional Commit subjects.

Before push, record:

- exact intended files and diff summary;
- targeted and full checks run;
- independent review result;
- known dependency on another branch or PR;
- schema, ABI, cache, replay and portability impact.

Shared contract PRs land before binding consumers. nirs4all migration PRs remain
draft until their required dag-ml contract version is available. No PR may claim
cross-language custom-loss completion before every local-binding acceptance test
listed above is green.

The loss-only L1 branch may publish while the score-provider dependency is
unpublished. Metric-related contract, task, schema or provider work may not be
committed on that branch until L0 names and reviews the provider artifact.
