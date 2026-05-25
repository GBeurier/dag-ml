# Status

Current state: foundation scaffold.

Implemented:

- Rust workspace with core, facade, C ABI and CLI crates;
- graph model and validation;
- OOF prediction join that rejects non-validation partitions;
- deterministic control seed derivation;
- C ABI graph validation entry point;
- example graph fixture;
- CI workflow.

Not implemented yet:

- full DSL compiler;
- search-space enumerator;
- executor and scheduler;
- artifact/cache/lineage stores;
- Arrow prediction storage;
- host controller adapters;
- integration with `dag-ml-data` runtime plans.

Next recommended task:

Implement canonical JSON schema snapshots for `GraphSpec` and add fixture-driven
tests derived from UC6 and UC11 in `docs/design/source/dag_ml_use_cases.md`.
