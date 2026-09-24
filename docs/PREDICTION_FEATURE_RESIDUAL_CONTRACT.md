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

## Implemented scope and remaining boundaries

The native `PredictionJoin` now reduces sample-keyed OOF prediction blocks to
a Data feature handle for this graph shape. The scheduler creates a lower-level
OOF campaign inside each residual-base fold, separate REFIT OOF campaigns,
and final prediction features from refitted branch models. Bundle capture
retains only report-grade outer-fold OOF evidence. Public nirs4all PyO3 and
CLI tests cover fixed and automatic residual gates, CV, REFIT, and `.n4a`
replay, both with and without an explicit splitter. With no splitter, legacy
uses held-out test rows for validation while DAG-ML uses training-only CV; the
two CV scores therefore are not comparable. Native tests reject in-sample,
missing, and duplicate training feature rows; the join also validates finite
values before returning a matrix.

This join yields **one feature matrix**, consumed by one residual base and one
residual learner, which yield **one final prediction output**. Multiple branch
predictions are input columns, not independent public outputs. The separate
`MULTI_OUTPUT_PREDICTOR_CONTRACT.md` concerns independent named outputs such
as `by_source -> merge: auto`; this implementation does not add `predict_all`
or multiple portable output bindings to an exported residual model. The
current public `.n4a` replay reconstructs branch features with the archived
Python estimators and their declared column order. Portable cross-language
archive replay of that composite estimator remains a separate contract.

Nor does this graph enable arbitrary chains of multiple marked nested meta
models. `nested_stacking_campaign_plan` still accepts one declared meta node;
the ignored `nested_stacking_then_residual_accepts_dependent_meta_models` test
records that broader planner boundary. The implemented prediction-feature
source branches are ordinary model nodes. Current legacy `ResidualModel` also
fails on categorical targets scored with regression estimators and on two
continuous target columns (residual rows are flattened); those are documented
legacy limitations, not functional parity oracles.
