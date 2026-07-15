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

The semantic contracts are standalone v1 contracts. The controller-protocol
integration adds only defaulted/optional fields to existing v1 JSON shapes, so
historical requests, plans, tasks, results, caches and bundles without explicit
losses remain readable. Future incompatible changes publish new schema ids and
Rust readers. This publication does not add or modify a C ABI symbol, macro or
struct layout; process-local callback registration is provided by binding
registries in the next implementation layer.
