# Loss and Metric Contracts

The native owner is `dag_ml_core::criteria`. The published v1 wire family is:

- `docs/contracts/loss_spec.schema.json`;
- `docs/contracts/metric_spec.schema.json`;
- `docs/contracts/implementation_descriptor.schema.json`;
- `docs/contracts/training_loss_role.schema.json`;
- `docs/contracts/metric_role.schema.json`.

The canonical positive and negative fixture is
`examples/fixtures/criteria/criteria_contracts.v1.json`. The exact artifact
closure is pinned by `docs/contracts/criteria_conformance_pack.v1.json` and is
validated independently by `parity/criteria/oracle.py`.

`LossSpec` defines optimizer-objective semantics; `MetricSpec` defines
evaluation semantics and objective direction. Pipeline roles remain separate,
so selection, reporting, early stopping and training cannot silently exchange
meanings. Both specs reference the same generic implementation-descriptor
shape, which carries only provider identity, capabilities and lifecycle. Local
callbacks are resolved through opaque registry keys; executable code and import
instructions are forbidden in canonical JSON.

These are new standalone v1 contracts, not fields added to an existing wire
shape, so there is no previous-version read window. Future incompatible changes
publish new schema ids and Rust readers. The L1 publication is additive and
does not add or modify a C ABI symbol, macro or struct layout; C callback
registration is intentionally deferred to roadmap L5.
