# Shared Contracts

This directory contains wire-contract artifacts shared with `dag-ml-data`.
`dag-ml` remains the consumer and semantic validator: it checks fingerprints,
campaign fold membership, OOF boundaries and leakage policies before any
controller receives a handle.

## Coordinator Data Plan Envelope v1

Schema: `coordinator_data_plan_envelope.schema.json`

Runtime type consumed here: `ExternalDataPlanEnvelope`

Producer type in `dag-ml-data`: `CoordinatorDataPlanEnvelope`

The envelope binds a data plan to stable schema, plan and relation
fingerprints. It may carry coordinator relation records for sample, target,
group, origin, source and augmentation identity. The JSON Schema documents the
portable shape of that envelope; Rust validation enforces the stronger semantic
rules that depend on the active campaign.

Short-term policy: both repositories keep a copy of the v1 schema and test that
the published artifact declares the Rust-supported version. When development
moves into a monorepo, this file should become a single generated or shared
contract artifact used by both crates.
