# Prediction features before a residual learner

The nirs4all pipeline `branch(models) -> merge(predictions) ->
ResidualModel(base, learner)` is executable in legacy. Its safe default uses
held-out predictions as training features. DAG-ML must accept the same graph
without materializing a precomputed training matrix outside the graph.

## Native execution contract

1. The prediction-to-data join is a graph node. It joins prediction columns by
   sample identity, target/column identity, fold and variant. It rejects duplicate,
   missing, wrong-fold and in-sample training rows. It produces a scoped Data
   output whose provenance lists every source prediction block. The host may
   build its numeric matrix from that attested output; it must not choose rows
   or folds independently.
2. For each outer validation fold, every branch model supplies training
   predictions from folds strictly inside the outer training set. The join
   supplies these features to the residual base's inner OOF fits. The residual
   learner then receives the base's own inner OOF predictions and matching
   targets. At no stage may an outer-validation target affect a fitted model.
3. REFIT constructs a separate exact-partition OOF prediction-feature set for
   the residual base and a further nested OOF set for the residual learner.
   PREDICT and archive replay use refitted branch models, the same column order
   and sample-keyed join; no held-out prediction is required for new samples.
4. Multiple dependent OOF stages require per-node parent scopes. One global
   `NestedStackingCampaignPlan` cannot represent their fold lineage. Scheduling,
   prediction stores, data handles and cache keys must preserve that ancestry.

## Current boundaries

- The DSL already emits a `PredictionJoin` with a Data output for
  `output_as=features`, but `merge_mode=predictions` has no native reduction in
  `runtime/merge.rs`; it takes the ordinary controller path. The nirs4all
  residual adapter rejects model-branch prediction merges before compilation.
- `nested_stacking_campaign_plan` returns a singleton campaign and rejects a
  second marked meta node. The ignored Rust test
  `nested_stacking_then_residual_accepts_dependent_meta_models` is the minimal
  red witness; run it with `cargo test --manifest-path
  crates/dag-ml-core/Cargo.toml nested_stacking_then_residual_accepts_dependent_meta_models
  -- --ignored`. It currently fails with `nested stacking V1 supports exactly
  one declared meta node per execution plan`. That test covers the planner
  boundary, not the complete feature-join graph.

## Acceptance

Add native tests for a two-stage fold ancestry and an OOF prediction-to-data
join with a deliberately in-sample row. Add a public nirs4all PyO3 and CLI
oracle for `branch(models) -> merge(predictions) -> ResidualModel`, including
REFIT/PREDICT/archive replay, and compare selected CV scores with legacy on a
fixed small dataset. Keep the ignored planner test until it passes unchanged,
then enable it. Neither Python row reconstruction nor unsafe train-prediction
features satisfy this contract.
