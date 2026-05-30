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

```{toctree}
:maxdepth: 2
:hidden:

installation
ARCHITECTURE
ABI
STATUS
TEST_PLAN
ROADMAP
CAPABILITY_MATRIX
contracts/README
adr/README
```
