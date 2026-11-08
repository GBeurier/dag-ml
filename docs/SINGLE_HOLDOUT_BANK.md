# Single resampled holdout bank, version 1

This opt-in DAG operation owns an ordered Cartesian FIT_CV bank and one native
SELECT. It adds no Methods algorithm and changes no existing training, O1,
metric, or cross-fold-average selection default. Its qualification is limited to
the named PLS banks and fixtures measured by the integrating host; source tests
alone do not establish numerical parity.

The WASM entrypoint is:

```ts
execute_single_holdout_bank_json(
  execution_plan_json: string,
  trusted_controller_manifests_json: string,
  envelope_json: string,
  request_json: string,
  run_id: string,
  root_seed: string, // exact decimal u64
  js_invoke: Function,
): string
```

`request_json` follows `contracts/single_holdout_bank_request.v1.schema.json`:

```json
{
  "schema_version": 1,
  "profile": "single_resampled_holdout_bank.v1",
  "report_profile": "regression_report.population_rmse_ratios.v1",
  "failure_group_dimension": "chain",
  "target_node": "model:pls"
}
```

Unknown fields, unsupported versions/profiles, or invalid plan/envelope/controller
contracts are refused before execution. The FoldSet must contain exactly one
`resampled` fold. Generation must be ordinary Cartesian with exactly two axes:
first the named failure-group dimension, then model parameters. There is no
pruning. Every first-axis choice overrides only `pipeline_steps_v1` on the target
node. Every second-axis choice overrides a nonempty parameter map on that same
node, excluding the reserved recipe key. Model parameter names and values stay
with the operator/controller: `n_components` is not a global requirement; an
`alpha` axis uses the same owner API.
The target is a model with an `N4mRolePipeline` operator and controller version
`1.1.0`. Labels are opaque and never parsed. Failure-group chain labels must be
unique; the bank refuses duplicates before controller callbacks or grouping so
distinct chains cannot cross-exclude each other or merge their issues. Host
capability refusals, including unsupported MSC/Detrend chains, remain host policy
before lowering; such chains must not be submitted.

DAG uses `enumerate_variants` order, with the first axis outermost. Each native
variant ID is mapped to `ord:00000`, `ord:00001`, etc.; width is at least five
digits and is fixed for the entire bank. SELECT minimizes the actual validation
fold's native RMSE; exact score ties use the ordinal's lexical order. There is
no average synthesized for a single fold, no tolerance in selection and no
second host selection step.

Each candidate runs through `SequentialScheduler` in a fresh `RunContext`.
Incomplete/ambiguous predictions, targets or native scores, leakage errors and
operator failures invalidate the candidate and its whole chain. Successful
members of a failed chain remain diagnostic evidence only. Failed candidates
expose no partial predictions, scores or targets. If every chain fails, the
result has `status: "refused"`, a nonempty `refusal`, and no selected winner or
selection decision. No local retry or penalty score is substituted.

A nonfinite prediction or native RMSE is a candidate failure and excludes that
candidate's entire chain, including any completed finite-score components. The
bank never ignores the invalid component and selects another member of that
chain. This is an accepted difference from the historical Core sweep, which
could leave a NaN component unselected while retaining the chain. Finite
calibration inputs do not guarantee finite predictions from a degenerate fit;
the failure is returned as a chain issue rather than salvaged selection evidence.

A defined finite RMSE of zero remains eligible even when R² and both ratios are
undefined with their explicit reasons. Report completeness is evaluated through
identity/scope evidence and finite RMSE; undefined diagnostics do not signal a
missing selection score.

The result's Rust types are `SingleHoldoutBankResult`, `HoldoutVariantIdentity`,
`HoldoutVariantEvidence`, `HoldoutChainIssue` and `HoldoutBankWinner`, re-exported
through the core facade. JSON fields are:

- `schema_version`, `profile`, `status` (`selected` or `refused`);
- `ordinal_variant_mapping`: ordered identities, each with `ordinal`,
  `variant_id`, `chain_label`, effective target `params`, and effective `recipe`.
  Optional `n_components` is retained as convenience metadata when the parameter
  is an unsigned integer; it is omitted otherwise and never controls admission;
- `variants`: one evidence object per identity, with `identity`,
  `validation_predictions`, identity-aligned `validation_targets`, `report`,
  native RMSE-only `selection_report`, `failure`, and `eligible`;
- `chain_issues`: each contains `chain_label` and `failures` mapping ordinal
  to error text;
- `selected`: null, or `{identity, selected_graph, params_by_node,
  training_sample_ids, validation_sample_ids}`. `params_by_node` maps every native
  node ID to the selected closure's complete effective parameters, identical to
  the corresponding `selected_graph` node params;
- `decision`: the native ordinal-based `SelectionDecision`, or null;
- `refusal`: null on success, otherwise an explanatory string.

Profile reports now use `RegressionMetricReport`, the finite owner ScoreSet
representation. `PopulationRmseRatiosReport` is a Rust alias of that type.
Consequently `variants[i].report.metrics.rmse` is the scalar RMSE and
`report.row_count` is the count; the superseded scalar report shape is withdrawn.
`metric_profile` is `regression_report.population_rmse_ratios.v1`.

For a complete scalar target in prediction order, let SSE be the sum of squared
prediction residuals and SST the sum of squared deviations of targets from their
population mean:

- `metrics.rmse = sqrt(SSE/n)` and `metrics.bias = mean(predicted-actual)`;
- `metrics.r2 = 1-SSE/SST` when SST > 0; otherwise omit `r2` and record
  `undefined_metrics.r2 = "constant_target"`;
- `metrics.rpd = sqrt(SST/n)/rmse` when RMSE > 0; otherwise omit `rpd` and
  record `undefined_metrics.rpd = "zero_rmse"`;
- `metrics.rpiq = (Q(.75)-Q(.25))/rmse` when RMSE > 0, using sorted linear
  quantiles at position `(n-1)*q`; otherwise omit `rpiq` and record
  `undefined_metrics.rpiq = "zero_rmse"`.

Zero population SD or IQR with a positive RMSE gives a defined ratio of zero.
No undefined value is replaced by zero, one, null, NaN, Inf or an epsilon clamp.
A tiny positive RMSE retains its finite ratios. Input and defined output values
must remain finite. Identity joins preserve prediction order for every sum.
The profile RMSE must equal native validation RMSE bitwise; differing bit
patterns are a candidate failure without a tolerance.

`metric_profile: Option<String>` and
`undefined_metrics: BTreeMap<String,String>` are additive optional fields on the
existing report. Undefined keys are disjoint from `metrics` and restricted to
`r2` → `constant_target` and `rpd`/`rpiq` → `zero_rmse`. Undefined values require
the known profile; profile reports require finite RMSE. Profile-to-candidate
conversion admits only RMSE, and ScoreSet selection metadata refuses undefined
metrics and other profile objectives. No bias/ratio objective is registered.
The existing built-in R² and legacy report bytes, fingerprints and historical
stored values are unchanged. Earlier Quali zeros remain only the frozen named
test baseline and previously stored results; they are not recomputed.

See the additive fields in both existing ScoreSet schema families and
`examples/fixtures/score_set_population_rmse_ratios.v1.json` for the four literal
singularity cases. The existing fingerprinted `score_set.json` fixture remains
unchanged. Core/Quali RMSE identity and C6 still require Root's named integration
measurements; this source change makes no qualification claim.

## Selected-winner refit and replay

Use only `selected.selected_graph`. DAG has frozen every selected variant
parameter in that graph and removed its search-space fingerprint. Build a new
execution plan from this graph and a campaign with `generation` set to its
no-generation default and `split_invocation: null`, using data bindings from a
**KS-train-only envelope**. Do not reuse full-bank relation/content attestations
for that smaller cohort. Pass `selected.training_sample_ids` to
`execute_initial_full_refit_json`, unchanged and in Methods' greedy KS order.
Then use `replay_initial_full_refit_json` on the held-out validation cohort.
The initial-full-refit operation already checks the provider's exact training
identity order and train-only relations. This batch operation performs no REFIT.

Require bitwise equality of the winning FIT_CV, train-only REFIT prediction on
validation, and hydrated validation replay. That is an integration gate, never
an assumed property or a tolerance waiver. Ordinary training's all-row REFIT
is not part of this operation. Generic parameter dispatch (including `alpha`)
does not establish numerical qualification of a Ridge bank; named bank parity
still requires Root's bounded measurements.
