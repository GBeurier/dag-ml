# dag-ml

`dag-ml` is the Rust execution coordinator for leakage-safe, reproducible ML
pipelines. It owns graph compilation, phases, folds, OOF joins, controller
dispatch, lineage, artifact/cache contracts and deterministic scheduling. Data
storage and feature-buffer ownership live in the companion `dag-ml-data` repo.

This site is the contributor and integration entry point for the producer-side
contracts used by future nirs4all and nirs4all-core integrations. It does not
wire nirs4all directly.

## Start Here

| Need | Page |
|---|---|
| Build and validate locally | [Installation](installation.md) |
| Understand runtime boundaries | [Architecture](ARCHITECTURE.md) |
| Integrate over C ABI | [C ABI](ABI.md) |
| Check the supported release surface | [Supported surface](SUPPORTED.md) |
| Map aggregation across `dag-ml-data` | [Aggregation interop](AGGREGATION_INTEROP.md) |
| Run release performance probes | [Performance probes](PERFORMANCE.md) |
| Map nirs4all parity capabilities | [Capability matrix](CAPABILITY_MATRIX.md) |
| Inspect shared contracts | [Contract manifests](contracts/README.md) |
| Use native training/fine-tuning contracts | [Training contracts](TRAINING_CONTRACTS.md) |
| Review public training replay syntax and migration | [Training replay contracts](TRAINING_REPLAY_CONTRACTS.md) |
| Review conformal/robustness W0 contracts | [Conformal contract foundation](contracts/README.md#conformal-prediction-and-robustness-foundation-v1) |
| Review conformal ownership and lifecycle modes | [ADR-20](adr/ADR-20-conformal-calibration-ownership.md) |
| Review accepted decisions | [Architecture decisions](adr/README.md) |
| Pick an example by audience | `examples/README.md` |

## API References

- Rust core API: <https://docs.rs/dag-ml-core/latest/>
- Rust facade API: <https://docs.rs/dag-ml/latest/>
- C ABI source: `crates/dag-ml-capi/include/dag_ml.h`
- Python binding source: `crates/dag-ml-py`
- WASM binding source: `crates/dag-ml-wasm`

## The nirs4all ecosystem

<!-- RTD slugs are assumed equal to the repo name; edit a :link: URL if a slug differs at import. -->

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} dag-ml-data
:link: https://dag-ml-data.readthedocs.io/en/latest/
Typed sample-aligned multi-source data contracts — the data layer dag-ml consumes (shared, JSON-identical contracts).
:::
:::{grid-item-card} nirs4all
:link: https://nirs4all.readthedocs.io/en/latest/
Main Python modelling library — pipelines, SpectroDataset, predictions.
:::
:::{grid-item-card} nirs4all-methods
:link: https://nirs4all-methods.readthedocs.io/en/latest/
Portable C-ABI PLS/NIRS engine (libn4m) + bindings.
:::
:::{grid-item-card} nirs4all-formats
:link: https://nirs4all-formats.readthedocs.io/en/latest/
Rust readers for ~58 NIRS/spectroscopy file formats.
:::
:::{grid-item-card} nirs4all-io
:link: https://nirs4all-io.readthedocs.io/en/latest/
Dataset-assembly bridge → SpectroDataset.
:::
:::{grid-item-card} nirs4all-datasets
:link: https://nirs4all-datasets.readthedocs.io/en/latest/
Curated DOI-pinned NIRS dataset catalog (n4a-datasets).
:::
:::{grid-item-card} nirs4all-core
:link: https://nirs4all-core.readthedocs.io/en/latest/
Canonical portable aggregate distribution (Rust, Python, R, WASM, MATLAB/Octave).
:::
::::

```{toctree}
:maxdepth: 2
:caption: Overview
:hidden:

installation
RATIONALE
ARCHITECTURE
COORDINATOR_SPEC
ABI
CAPABILITY_MATRIX
```

```{toctree}
:maxdepth: 1
:caption: Contracts & decisions
:hidden:

contracts/README
TRAINING_CONTRACTS
TRAINING_REPLAY_CONTRACTS
adr/README
OOF_FIXTURES
OBSERVABILITY
```

```{toctree}
:maxdepth: 1
:caption: Development
:hidden:

SUPPORTED
AGGREGATION_INTEROP
PERFORMANCE
design/README
design/DSL_NIRS4ALL_PARITY
```

```{toctree}
:maxdepth: 1
:caption: nirs4all Migration
:hidden:

migration-nirs4all/README
migration-nirs4all/WORKING_STRATEGY
migration-nirs4all/PARITY_AND_PERF_HARNESS
migration-nirs4all/TARGET_RESPONSIBILITY_SPLIT
migration-nirs4all/NATIVE_PERSISTENCE_LAYER_REPORT
```
