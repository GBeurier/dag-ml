# Shared scoring numerical contract v1

DAG-ML owns metric definitions, task context, identity joins, missingness,
reductions and selection objectives. Methods supplies numerical kernels. The
[canonical fixture](../../examples/fixtures/criteria/shared_scoring_numerical.v1.json)
freezes this bounded contract before any provider migration. The default Rust
integration test `shared_scoring_contract` evaluates public coordinator and
built-in registry interfaces, including provider decomposition and reduction.

Regression kernels consume finite, unweighted one-dimensional slices. RMSE is
the square root of mean squared residuals, MAE the mean absolute residual, and
R² is `1 - SSE/SST`. DAG and the Methods scalar ABI define constant-target R²
as 1 for perfect predictions and 0 otherwise, including one-row slices.
Multi-output scores are the unweighted mean of per-output scores: mean RMSE
is not flattened RMSE. Selection minimizes RMSE/MAE and maximizes R² and
classification scores; this slice does not change selection or dispatch.

Targets join predictions by stable identity; missing identities are refused.
Validity masks use true for observed cells. Transport includes finite sentinels,
never NaN for missingness. The coordinator operation
`coordinator_filter_observed_per_output` filters a separate row set for each
output before calling a scalar provider, then macro-averages output scores.
Built-in registry tasks reject a raw missing mask; filtering belongs to the coordinator.
A raw mask has no Methods scalar ABI representation. NaN and either infinity
in inputs are rejected, including hidden target cells. This input policy does
not forbid intentional +Inf outputs from finite degenerate Methods ratio
metrics such as RPD, RPIQ or NRMSE.

Classification uses finite integer label identities. Accuracy counts correct
labels; balanced accuracy averages recall over classes present in targets.
F1 is support-weighted per-class F1 (zero-denominator contribution zero), not
macro F1 and not sample weighting. Predicted-only classes have zero support.
The existing label kernels round numeric class encodings; the fixture uses
exact integers and makes no new promise about fractional or extreme codes.
Probability columns map through explicit class labels before scoring. Consistent
label remapping or column reordering preserves these scores; row joins remain
identity-based. Equal-probability ties retain the first column under the existing
coordinator rule; tie invariance is not claimed.

Built-ins refuse supplied sample weights for regression and classification,
including F1. Future providers must explicitly declare their weight support;
this fixture does not add support. Methods scalar APIs have no weights,
classification or identity surface. Each fixture case has an implementation
applicability matrix: `supported`, `rejects`, or `not_applicable`. The last is
never a numerical parity pass. The masked adapter has its own applicability.

Methods' `n4m_stats_regression_scores` is a separate context ABI with four
outputs (RMSE, MAE, R² significand, R² exponent). Constant targets intentionally
produce NaN R² there; the fixture records and tests this divergence. Internal
`compute_regression_metrics` and the optimization metric enum are separate
implementations; the enum has no standalone evaluation ABI and is out of scope.
No alternate implementation is moved or removed.

Fixture values are analytical examples with absolute-plus-relative tolerance
`1e-12 + 1e-12 * abs(expected)` for finite values. NaN divergence and refusal
statuses are checked explicitly. This is not an extreme-range accuracy claim
or a claim that all metrics agree universally. Full Python reference artifacts
remain independent and unchanged. Methods keeps an exact pinned copy with a
source-repository/path/creation-commit/SHA receipt and an explicit drift checker.

The crate embeds an exact pinned fixture under `crates/dag-ml-core/tests/fixtures/`
so source packages need no sibling checkout. Its default test checks SHA-256
over exact UTF-8 file bytes (including trailing newline) and compares the
canonical source when available; isolated source packages explicitly report
that comparison as `UNAVAILABLE`. The fixture's `TCV1-unicode-17.0.0` metadata
identifies the DAG semantic canonicalization family; the byte SHA is not a
TCV1 semantic fingerprint. Methods' drift checker compares all four copies.

Both Methods direct ABIs are `not_applicable` for every multi-output or masked
case. Separate scalar/stats per-output adapter applicability permits kernel
checks after column extraction/filtering, counted separately from the five
direct scalar cases and never as full-case ABI parity.
