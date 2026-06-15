# dag-ml

`dag-ml` is the Rust execution coordinator for leakage-safe, reproducible ML
pipelines. It owns graph compilation, phases, folds, OOF joins, controller
dispatch, lineage, artifact/cache contracts and deterministic scheduling. Data
storage and feature-buffer ownership live in the companion `dag-ml-data` repo.

This site is the contributor and integration entry point for the producer-side
contracts used by future nirs4all and nirs4all-lite integrations. It does not
wire nirs4all directly.

## Start Here

| Need | Page |
|---|---|
| Build and validate locally | [Installation](installation.md) |
| Understand runtime boundaries | [Architecture](ARCHITECTURE.md) |
| Integrate over C ABI | [C ABI](ABI.md) |
| Check shipped vs pending scope | [Status](STATUS.md) |
| Run the documented gates | [Test plan](TEST_PLAN.md) |
| Review roadmap and release gates | [Roadmap](ROADMAP.md) |
| Check the supported release surface | [Supported surface](SUPPORTED.md) |
| Map aggregation across `dag-ml-data` | [Aggregation interop](AGGREGATION_INTEROP.md) |
| Run release performance probes | [Performance probes](PERFORMANCE.md) |
| Audit final release readiness | [Final release audit](FINAL_RELEASE_AUDIT.md) |
| Map nirs4all parity capabilities | [Capability matrix](CAPABILITY_MATRIX.md) |
| Inspect shared contracts | [Contract manifests](contracts/README.md) |
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
:::{grid-item-card} nirs4all-lite
:link: https://nirs4all-lite.readthedocs.io/en/latest/
Portable aggregate distribution (Rust + bindings).
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
ROADMAP
TOC
```

```{toctree}
:maxdepth: 1
:caption: Contracts & decisions
:hidden:

contracts/README
adr/README
MVP_ACCEPTANCE
OOF_FIXTURES
OBSERVABILITY
```

```{toctree}
:maxdepth: 1
:caption: Development
:hidden:

STATUS
TEST_PLAN
SUPPORTED
AGGREGATION_INTEROP
PERFORMANCE
FINAL_RELEASE_AUDIT
design/README
```
