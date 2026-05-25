# MVP Acceptance

This document locks the first implementation target for `dag-ml`. It covers the
validation work requested before starting the full executor.

## Boundary Confirmation

| Owned by `dag-ml` | Delegated to `dag-ml-data` |
|---|---|
| `GraphSpec`, nodes, ports and edge contracts | `DatasetSchema`, source descriptors and semantic axes |
| phase flow: `COMPILE -> PLAN -> FIT_CV -> SELECT -> REFIT -> PREDICT -> EXPLAIN` | representation compatibility and `DataPlan` construction |
| fold identity, splitters and group-aware split checks | sample relations, group ids, origin ids and presence masks |
| OOF prediction storage and prediction joins | materialization, adaptation, alignment, feature joins and collation |
| leakage refusal and explicit unsafe opt-in markers | adapter fit-scope declarations |
| controller ABI and opaque model/data handles | data-provider ABI and opaque data/view handles |
| deterministic control RNG | payload-specific data transformations |

The core must never inspect feature buffers. It may inspect identity, fold,
target and prediction tables.

## First Acceptance Target

The first end-to-end target is derived from `docs/design/source/dag_ml_use_cases.md`:

| Case | Expected result | What it proves |
|---|---|---|
| UC6: stacking multi-niveau | succeeds through OOF prediction join and meta-model training input construction | prediction features are aligned by `sample_id`, not by row position |
| UC11: train predictions refused by default | fails during prediction join with an `OOFLeakageError`-class error | direct train predictions cannot silently feed a downstream training node |

## Required `dag-ml` Work Before Full Executor

| Item | Acceptance check |
|---|---|
| `FoldSet` model | fold membership is expressed in stable sample ids |
| identity splitters | `KFold` and `GroupKFold` produce deterministic fold tables from identity/group inputs |
| `PredictionBlock` expansion | prediction rows carry producer node, partition, fold id, sample ids and target columns |
| OOF join | shuffled prediction rows are joined into the requested sample order |
| leakage refusal | any `partition="train"` block is rejected for training features unless an explicit unsafe policy exists |
| unsafe policy marker | if added later, the manifest and lineage must include `train_predictions_used=true` and `leakage_acknowledged=true` |
| CLI fixture | a fixture command validates UC6 success and UC11 refusal |

## Non-Target For MVP

- parallel scheduler;
- full artifact bundle format;
- R/JS bindings;
- ONNX/safetensors export;
- feature-space explainability.

## Green Gate

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-cli -- validate-graph examples/minimal_graph.json
cargo run -p dag-ml-cli -- validate-oof-campaign examples/fixtures/oof_campaign/uc6_oof_success_predictions.json
cargo run -p dag-ml-cli -- validate-oof-campaign examples/fixtures/oof_campaign/uc11_train_prediction_refusal.json --expect-leakage
cargo run -p dag-ml-cli -- fingerprint-oof-campaign examples/fixtures/oof_campaign/uc6_oof_success_predictions.json
```
