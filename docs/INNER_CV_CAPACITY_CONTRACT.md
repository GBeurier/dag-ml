# Nested OOF fit-capacity contract (proposal)

Status: first executable slice implemented for residual meta nodes without
dependent meta models or grouped samples. General group-aware capacity,
per-operator validation-sample minima, and all multi-meta topologies remain
proposed work.

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

The first slice reads a `fit_capacity` object from each model node's graph
metadata. Language adapters/controllers declare it from the operator profile:

```json
{
  "min_fit_samples": 4,
  "min_fit_groups": null,
  "min_validation_samples": 1
}
```

The implemented field is `min_fit_samples`; the other two fields above are
reserved for later slices. This is a conservative structural minimum, not a
claim that a model always converges above it. The host adapter owns the bound
because Rust cannot inspect feature matrices or estimator internals. The core
requires a positive declaration for every model in this first slice's affected
closure. An opaque adapter can retain a fixed `inner_cv` until it can declare
a meaningful minimum. The declaration is part of graph metadata and therefore
of the plan fingerprint. The core never infers capacity from a language or
class name.

The first slice adds an opt-in nested policy alongside fixed `NestedCvSpec` forms:

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
field. `max_splits` is an explicit compute budget; the core never falls back to
in-sample or outer-OOF features. Existing fixed policies retain their semantics
and JSON representation.

## Planning algorithm

Before invoking a controller, the first slice walks the dependency closure
of a residual meta node and its optional prediction-feature join. For every candidate
split count from `min_splits` to `max_splits`, it should build the actual fold
sets using the declared splitter and sample/group identities, recursively
including all nested OOF levels. The first candidate for which every model's
minimum fit and validation counts hold in every phase (outer FIT_CV, nested
FIT_CV, and REFIT OOF preparation) becomes the resolved policy. Record the
resolved counts, fold-set fingerprints, and limiting model/scope in the plan.
It uses the actual sample identities and fold memberships, not a closed-form
average. The first slice rejects grouped samples, dependent meta nodes, and
non-residual stacking under this policy until those scope shapes are supported.

If no candidate fits, the core rejects before training with a runtime
validation error containing the model node, offending scope, observed and
required counts, and allowed split range. A later slice should give this error
a dedicated code and add group/stratum diagnostics. A host numerical rank or
convergence failure remains distinct from this structural refusal. Retrying
with a different policy requires a *new* fingerprinted campaign, never a
silent mutation of an in-progress run.

The planner requires capacity declarations for all model nodes in the affected
dependency closure. An unknown model may run under an explicit fixed
`inner_cv`, but that is only an attempt, not a capacity guarantee. The JSON
contract is shared by Python, C, Rust, R, MATLAB, and WASM controllers; no
language-specific scheduler policy is added.

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
