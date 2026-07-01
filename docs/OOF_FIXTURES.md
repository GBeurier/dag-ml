# OOF Campaign Fixtures

This document defines the first reproducible OOF campaign fixtures owned by
`dag-ml`. The fixtures are intentionally tiny and identity-heavy: they prove
fold and prediction semantics before any real model execution exists.

## Fixture Set

Directory: `examples/fixtures/oof_campaign/`

| Fixture | Purpose |
|---|---|
| `uc6_oof_success_predictions.json` | UC6-shaped positive stacking case |
| `uc11_train_prediction_refusal.json` | UC11-shaped leakage refusal case |

## Shared Sample Universe

Use six sample ids and three folds:

| Fold | Validation samples | Train samples |
|---|---|---|
| `fold0` | `S001`, `S004` | `S002`, `S003`, `S005`, `S006` |
| `fold1` | `S002`, `S005` | `S001`, `S003`, `S004`, `S006` |
| `fold2` | `S003`, `S006` | `S001`, `S002`, `S004`, `S005` |

Rows inside prediction blocks should be deliberately shuffled to force joins by
`sample_id`, never by input row position.

## UC6 Success

Required top-level fields:

| Field | Shape |
|---|---|
| `fold_set` | fold assignments by absolute sample ids |
| `join_policy` | `allow_train_predictions_as_features=false`, `join_on="sample_id"` |
| `requested_sample_order` | `["S001", "S002", "S003", "S004", "S005", "S006"]` |
| `prediction_blocks` | three validation-only producers |

Prediction block fields:

| Field | Requirement |
|---|---|
| `prediction_id` | stable fixture id |
| `producer_node` | e.g. `branch:b0.model:pls` |
| `partition` | exactly `validation` |
| `fold_id` | `fold0`, `fold1`, or `fold2` |
| `sample_ids` | validation samples for that fold, possibly shuffled |
| `values` | one regression column per sample |
| `target_names` | `["y"]` |

Assertions:

- join succeeds with default safe policy;
- output sample order equals `requested_sample_order`;
- output columns are producer namespaced;
- each sample has exactly one validation prediction per producer;
- no train/leakage flags are present.

## UC11 Refusal

Required difference from UC6:

- at least one prediction block has `partition="train"`;
- `join_policy.allow_train_predictions_as_features=false`;
- `join_policy.include_partitions=["train", "validation"]`.

Assertions:

- execution refuses before building meta features;
- error kind is structured OOF leakage;
- payload includes `node_id="merge:pred"`;
- payload lists every train violator with producer, partition and fold id;
- remediation text tells the user to use validation-only OOF predictions or to
  explicitly opt into the unsafe policy.

## Boundary

`dag-ml` owns folds, prediction blocks, OOF joins, leakage refusal and campaign
fingerprints. `dag-ml-data` owns schema/model-input/adapter-plan fixtures only.

## Stacking REFIT Coverage Contract

The runtime distinguishes three stacking REFIT outcomes:

| Case | Required policy | Outcome |
|---|---|---|
| full validation-OOF coverage for the refit sample universe | default `require_full_coverage` | meta-model REFIT may consume OOF |
| incomplete but otherwise well-formed validation OOF | `cv_only` or `skip_refit_on_incomplete_oof` on `metadata.stacking_oof_refit_contract.policy` | stacking REFIT is skipped |
| incomplete OOF without explicit policy, train/final/test prediction input, missing fold ids, unknown folds, fold/sample mismatch or duplicate validation rows | none | rejected with a stable cause such as `partial_oof_without_policy` |

## D8 Conformance Pack

`docs/contracts/conformance_pack.v1.json` pins canonical digests for
`uc6_oof_success_predictions.json` and
`uc11_train_prediction_refusal.json`. D8 uses them for the
`stacking_oof_contract.v1` and `row_vs_sample_selection_mismatch.v1` scenarios,
so changes to these fixtures must update the pack and rerun
`python3 scripts/validate_contracts.py`.
