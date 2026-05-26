# Shared Contracts

This directory contains wire-contract artifacts shared with `dag-ml-data`.
`dag-ml` remains the consumer and semantic validator: it checks fingerprints,
campaign fold membership, OOF boundaries and leakage policies before any
controller receives a handle.

## Coordinator Data Plan Envelope v1

Schema: `coordinator_data_plan_envelope.schema.json`

Canonical fixture: `examples/fixtures/data/coordinator_data_plan_envelope_nir.json`

Conformance pack: `conformance_pack.v1.json`

Runtime type consumed here: `ExternalDataPlanEnvelope`

Producer type in `dag-ml-data`: `CoordinatorDataPlanEnvelope`

The envelope binds a data plan to stable schema, plan and relation
fingerprints. It may carry coordinator relation records for sample, target,
group, origin, source and augmentation identity. The JSON Schema documents the
portable shape of that envelope; Rust validation enforces the stronger semantic
rules that depend on the active campaign.

Short-term policy: both repositories keep a JSON-identical conformance fixture
for this envelope plus a copy of the v1 schema, and test that the published
artifact declares the Rust-supported version. `scripts/validate_contracts.py`
compares the fixture and schema copies when `DAG_ML_DATA_REPO` points to a
sibling checkout, validates the shared conformance-pack digests, and CI checks
out that peer explicitly. When development moves into a monorepo, this file
should become a single generated or shared contract artifact used by both
crates.

## Feature Fusion Selector v1

Schema: `feature_fusion_selector.schema.json`

Canonical fixture: `examples/fixtures/data/feature_fusion_selector_nir_chem.json`

Runtime shape passed through data-provider `feature_arrow` when the provider
supports `dag-ml-data` multi-source fusion:
`{ schema_version, feature_set_id, sources, alignment, policy? }`, where each
source maps a `source_id` to a provider-owned `feature_set_id` and optional
column subset. This keeps `DagMlDataVTable` ABI-compatible while making feature
fusion explicit.

## Data Provider C ABI v2

The shared provider surface is `DagMlDataVTable` guarded by
`DAG_ML_DATA_VTABLE_DEFINED` and versioned by
`DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION == 2`. `scripts/validate_contracts.py`
and the C ABI tests verify that `dag_ml.h` and `dag_ml_data.h` can be included
together in either order when the sibling checkout is available.
