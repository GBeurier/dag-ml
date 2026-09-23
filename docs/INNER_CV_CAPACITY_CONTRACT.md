# Nested OOF fit-capacity contract (proposal)

Status: design proposal; no automatic policy is implemented yet.

## Problem and invariant

`NestedCvSpec` already lets every DAG-ML language adapter request an explicit
inner KFold, GroupKFold, or StratifiedKFold policy. A residual learner or
stacking model must train on OOF predictions produced *inside each outer
training scope*. Reusing OOF predictions from the outer splitter is unsafe:
for outer fold A, the predictor that held out an A-training row was fitted in
another outer fold B, whose training set includes A's validation labels.

A prediction-feature join upstream of a residual base adds another OOF level.
With 30 rows and two folds at each level, an operator may eventually receive
only three fit rows. A valid legacy pipeline can therefore fail in DAG-ML even
though the model fits the larger legacy scope. Raising every split count to a
fixed value merely moves the failure for other data or operators.

## Portable declarations

Add an optional `fit_capacity` object to each model node's execution contract,
attested by its controller manifest or per-operator descriptor:

```json
{
  "min_fit_samples": 4,
  "min_fit_groups": null,
  "min_validation_samples": 1
}
```

These are necessary lower bounds, not a claim that a model always converges
above them. The host adapter owns the estimate because Rust cannot inspect
feature matrices or estimator internals. A PLS adapter may compute a bound
from `n_components`; an opaque adapter may omit the declaration. The core must
validate that declared numbers are positive and include the declaration in
plan fingerprints. It must never infer capacity from an operator's language or
class name.

Add an opt-in nested policy alongside today's fixed `NestedCvSpec` forms:

```json
{
  "kind": "capacity_kfold",
  "min_splits": 2,
  "max_splits": 10,
  "shuffle": false,
  "seed": 0
}
```

The policy is usable at campaign or node level through the existing `inner_cv`
field. `max_splits` is an explicit compute budget; the core must not silently
fall back to in-sample or outer-OOF features. Existing fixed policies retain
their current semantics and JSON representation.

## Planning algorithm

Before invoking any controller, the core should walk the dependency closure
of each nested meta node and prediction-feature join. For every candidate
split count from `min_splits` to `max_splits`, it should build the actual fold
sets using the declared splitter and sample/group identities, recursively
including all nested OOF levels. The first candidate for which every model's
minimum fit and validation counts hold in every phase (outer FIT_CV, nested
FIT_CV, and REFIT OOF preparation) becomes the resolved policy. Record the
resolved counts, fold-set fingerprints, and limiting model/scope in the plan.
Do not use a closed-form estimate alone: uneven groups and stratification can
make the smallest fold much smaller than the average.

If no candidate fits, reject before training with a typed error containing
the model node, offending scope, observed and required counts, allowed split
range, and whether group/stratum constraints prevented a choice. If a host
model later reports a numerical rank or capacity failure despite sufficient
row counts, preserve that typed host error. Retrying with a different policy
would require a *new* fingerprinted campaign, never a silent mutation of an
in-progress run.

The planner must require capacity declarations for all model nodes in the
affected dependency closure before claiming automatic capacity safety. An
unknown model may run under an explicit fixed `inner_cv`, but the result is
only an attempt, not an attested capacity guarantee. This keeps the same
cross-language behavior for Python, C, Rust, R, MATLAB, and WASM controllers.

## Acceptance oracles

1. A Rust planner test checks the three-level residual + prediction-join
   graph with a declared minimum, showing that two folds violate a real
   deepest-scope count and an allowed higher count meets it. Include an
   uneven-group case that fails with a structured capacity error.
2. Rust provenance tests verify that changing a declaration, split budget,
   or resolved folds changes the plan fingerprint and that no outer-validation
   sample enters any nested training set.
3. Public nirs4all PyO3 and CLI tests run the same seeded legacy-working
   prediction-join + residual pipeline, using an adapter-declared PLS bound,
   and verify CV, REFIT, and archive replay. Repeat with a larger permitted
   `n_components`/data shape so success cannot hinge on one fixture.
4. A controller with an unknown minimum and a too-small group scope remains
   fail-loud. The existing strict xfail remains until both public mechanisms
   pass with the automatic policy.

This proposal does not promise that count hints prevent every numerical
failure: collinear features can make PLS unstable at any sample count. It
provides a portable, reviewable guarantee that DAG-ML will not create a
known-too-small fit scope, while preserving the leakage-safe OOF contract.
