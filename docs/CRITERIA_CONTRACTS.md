# Loss and Metric Contracts

The native owner is `dag_ml_core::criteria`. The published v1 wire family is:

- `docs/contracts/loss_spec.schema.json`;
- `docs/contracts/metric_spec.schema.json`;
- `docs/contracts/implementation_descriptor.schema.json`;
- `docs/contracts/training_loss_role.schema.json`;
- `docs/contracts/loss_execution_attestation.schema.json`;
- `docs/contracts/metric_role.schema.json`;
- `docs/contracts/metric_evaluation_task.schema.json`;
- `docs/contracts/metric_evaluation_result.schema.json`.

The canonical positive and negative fixtures are
`examples/fixtures/criteria/criteria_contracts.v1.json` and
`examples/fixtures/criteria/metric_provider_contracts.v1.json`. The exact
artifact closure is pinned by
`docs/contracts/criteria_conformance_pack.v1.json` and is validated
independently by `parity/criteria/oracle.py`.

Metric providers receive a self-fingerprinted `MetricEvaluationTask` containing
the semantic and implementation references, exact scope, unit identities,
prediction/target matrices and declared optional inputs. They return a
self-fingerprinted `MetricEvaluationResult`. DAG-ML rejects identity,
implementation, scope, coverage and non-finite-value mismatches before applying
the declared reduction. Built-in metrics use this same registry and delegate to
the existing native kernels; binding-local metrics register an implementation
under the descriptor's opaque `registry_key`.

`LossSpec` defines optimizer-objective semantics; `MetricSpec` defines
evaluation semantics and objective direction. Pipeline roles remain separate,
so selection, reporting, early stopping and training cannot silently exchange
meanings. Both specs reference the same generic implementation-descriptor
shape, which carries only provider identity, capabilities and lifecycle. Local
callbacks are resolved through opaque registry keys; executable code and import
instructions are forbidden in canonical JSON.

`LocalImplementationRegistry<T>` is the native process-local resolution
primitive shared by loss and metric implementations. It stores an arbitrary
binding-owned executable object as `T`, but resolves it only when the complete
validated implementation descriptor matches the registered descriptor. Rust
can therefore retain closures or trait objects directly, while language
bindings retain their own callable handles without putting executable objects
into serialized contracts.

The Python binding exposes this primitive as `LocalImplementationRegistry`.
`register_loss` and `register_metric` accept Python callables but keep separate
semantic resolution paths. Python-local descriptors must use
`binding_id = "binding:python"`, declare `needs_gil`, and use either the
`host_local` or `portable_registered` lifecycle; `portable_builtin`
implementations remain native catalog entries. The registry itself is not
serializable. A detached worker or replay process must explicitly register the
same descriptor before resolving its opaque `registry_key`.

The WASM binding exposes the same primitive as a JavaScript
`LocalImplementationRegistry`. It retains `Function` objects for
`binding:javascript` descriptors, resolves loss roles only for their declared
training phases, and emits the common execution attestation. Registries reject
JSON serialization. Every browser main thread or worker must register its own
local callback before execution or replay.

The R binding exposes `dagml_local_implementation_registry()`. It retains R
functions for `binding:r` descriptors, keeps loss and metric resolution paths
separate, and invokes active training losses directly in `FIT_CV` and `REFIT`.
Its training invocation consumes the native `NodeTask` requirement and returns
that attestation only after the R function succeeds; the binding never computes
TCV1 fingerprints or writes a function into DAG JSON.

`TrainingRequest.training_losses` is the authoritative pipeline assignment.
Each role targets a node and an optional controller-local output/head and lists
the exact training phases where it applies. The resolved roles travel inside
`ExecutionPlan::NodePlan` and therefore inside `NodeTask`; no parallel
controller-only loss setting is allowed. Controllers declare
`supports_configurable_loss`, `supports_custom_loss` and, when applicable,
`supports_differentiable_loss`. A configured loss is rejected during planning
when the controller or implementation cannot satisfy its required inputs or
runtime capabilities.

For every configured role, `NodeResult.lineage.loss_attestations` must contain
one ordered `LossExecutionAttestation`. Attestations use the resolved node-plan
order: roles are strictly ordered by `(output_id, phases)` and filtering for the
current phase preserves that order. DAG-ML compares semantic,
implementation and descriptor fingerprints, effective parameters, reduction,
node, output and phase against the resolved role. `FIT_CV` cache namespaces and
`REFIT` artifact records commit to the corresponding ordered loss-role set;
artifact materialization and replay require an exact commitment match.

Every serialized `NodeTask` with active loss roles also carries
`required_loss_attestations`, generated by DAG-ML from those roles in the same
order. This is the host-language handoff: a controller resolves and executes
its local function, then copies the corresponding native-produced template to
`NodeResult.lineage.loss_attestations` only after execution succeeds. The core
recomputes and verifies both the task requirements and the returned lineage, so
R, MATLAB and other process adapters do not implement canonical fingerprinting.

The semantic contracts are standalone v1 contracts. The controller-protocol
integration adds only defaulted/optional fields to existing v1 JSON shapes, so
historical requests, plans, tasks, results, caches and bundles without explicit
losses remain readable. Future incompatible changes publish new schema ids and
Rust readers. The C ABI exposes an opaque process-local registry and a
versioned callback vtable without changing any controller vtable layout. A
registry is scoped to an explicit binding identity (`binding:c`, `binding:r`,
`binding:matlab`, or a future native host), retains callback `user_data` through
balanced optional `retain`/`release` hooks, and copies callback-owned result
JSON before invoking its required `release_bytes` hook. Training-loss
invocation is limited to `FIT_CV` and `REFIT` and returns the common attestation
only after callback success. Host exceptions must be caught by the language
trampoline and returned as `DAG_ML_STATUS_PANIC`; no unwind may cross the C
boundary. Each process or worker owns a separate registry and must register its
local runtime objects.
