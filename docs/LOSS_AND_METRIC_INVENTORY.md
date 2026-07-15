:orphan:

# Loss and Metric Baseline Inventory

## Purpose and scope

This is the L0 baseline for the native loss and metric roadmap. It records the
behavior that must either remain compatible or fail explicitly while dag-ml
becomes the control plane for training loss, early stopping, scoring, selection,
tuning and reporting.

The baseline revisions are:

| Repository | Revision | State inspected |
| --- | --- | --- |
| `dag-ml` | `366ca8b` | clean committed `main`, plus a separately identified active dirty worktree |
| `nirs4all` | `16c8f070` | clean committed `main` |
| `nirs4all-core` | `83c246c` | clean committed `main` |

The private `nirs4all-drafts` and `nirs4all-lab` repositories are out of scope.
Line references below are pinned to the revisions above unless a row is
explicitly labelled as uncommitted concurrent work.

## Summary

- dag-ml has native metric calculation and generic score selection, but no
  semantic training-loss contract, implementation registry or loss attestation.
- nirs4all Python has implicit local callable loss support in TensorFlow and
  PyTorch, a hard-coded JAX loss path, and estimator-owned objectives for
  scikit-learn-compatible controllers.
- nirs4all-core has no configurable loss or metric surface. Its portable PLS
  pipeline computes RMSE and selects its minimum independently in every binding.
- Training loss, early-stopping monitor, selection metric, reporting metrics and
  tuning objective are separate runtime roles even when they reference the same
  logical loss or metric.
- Two silent fallback paths can change user intent: an unknown PyTorch loss
  becomes MSE on baseline nirs4all, and an unknown in-process dag-ml Python
  selection metric becomes RMSE.

## dag-ml baseline

### Training loss and early stopping

Committed dag-ml has no `LossSpec`, loss implementation descriptor, local loss
registry, built-in loss catalog or training-loss default. `ControllerManifest`,
`NodeTask` and `NodeResult` do not attest which loss a controller applied.

The only committed use of the term loss in the generic runtime is loss
*weighting*, not loss selection:

| Surface | Current behavior | Source | Characterization |
| --- | --- | --- | --- |
| fit influence policy | supports backend loss weights | `dag-ml:crates/dag-ml-core/src/policy.rs:84` | runtime influence tests in `runtime/tests.rs` |
| task mechanism | transports `BackendLossWeights` | `dag-ml:crates/dag-ml-core/src/runtime/task.rs:95` | runtime task validation tests |
| controller capability | declares `SupportsBackendLossWeights` | `dag-ml:crates/dag-ml-core/src/controller.rs:36` | controller manifest tests |
| loss semantics | absent | no committed type or schema | missing; L1 owns this contract |
| early-stopping monitor | absent | no committed metric reference, patience or threshold contract | missing; L2 must distinguish it from final scoring |

Backend loss weights must not be interpreted as a selected loss, metric, loss
implementation or objective direction.

### Native metrics, reporting and selection

`RegressionMetricKind` is a closed native calculation set. Despite its historic
name, it also contains label-classification metrics.

| Metric | Compatible output in current native paths | Objective | Source/test |
| --- | --- | --- | --- |
| `mse` | regression point | minimize | `metrics.rs:17`, `metric_objectives_match_selection_direction` |
| `rmse` | regression point | minimize | `metrics.rs:18`, same objective test |
| `mae` | regression point | minimize | `metrics.rs:19`, same objective test |
| `r2` | regression point | maximize | `metrics.rs:20`, same objective test |
| `accuracy` | class label | maximize | `metrics.rs:25`, classification parity tests |
| `balanced_accuracy` | class label | maximize | `metrics.rs:33`, sklearn parity tests in `metrics.rs` |

The core selection type itself is more generic: `SelectionMetric` carries a
name and an explicit `MetricObjective`, while `CandidateScore.metrics` is a map
of named scalar values (`selection.rs:20-49`). `select_candidate` applies that
objective (`selection.rs:337-406`, comparison at `selection.rs:461-471`). This
allows a host-provided named score to participate in selection, but it is not a
custom metric execution protocol and it does not validate the semantic identity
of a non-built-in metric.

Committed reporting evaluates a fixed list of all six kinds in
`runtime/scoring.rs:4-11`. `ScoreSet.selection_metric` is metadata about the
selection decision, not a training loss (`metrics.rs:388-390`). The C ABI
standalone scorer accepts an explicit serialized list and rejects an empty list
(`dag-ml-capi/src/lib.rs:1735-1839`).

dag-ml-core intentionally has no implicit selection-metric default. Defaults are
currently supplied by host surfaces:

| Host surface | Default/behavior | Source | Test status |
| --- | --- | --- | --- |
| CLI | `rmse`; accepted names are clap-enumerated | `dag-ml-cli/src/main.rs:107-119`, `:506` | `dag-ml-cli/tests/cli_contracts.rs` |
| Python in-process | `accuracy` and `balanced_accuracy` map explicitly; every other value maps to `rmse` | `dag-ml-py/src/in_process.rs:167-175` | missing negative characterization; owned by score-provider integration |
| C ABI serde scorer | known enum names only; unknown values fail deserialization | `dag-ml-capi/src/lib.rs:1774-1839` | C ABI scorer tests |
| policy metric level | sample level | `dag-ml-core/src/policy.rs:280-295` | policy tests |

### Tuning, thresholds and ensembles

- Committed DAG DSL tuning fields are opaque host parameters; dag-ml does not
  yet own a tuning metric/default or pruning decision protocol.
- Refit ensembles reuse the selected candidate metric
  (`selection.rs:107-144`).
- Native mean/fusion/robust aggregation is a statistical reducer, not an
  independently selected metric (`aggregation.rs:1305-1491`).
- No committed threshold-optimization metric contract exists.

### Concurrent work boundary

The original dag-ml checkout contains a large active, unstaged training/runtime
change. At inventory time it has no published branch, commit or PR. It adds
uncommitted `training.rs`, `training_runtime.rs`, `canonical.rs`, conformal and
replay contracts, plus modifications to metrics, tasks, controllers, schemas and
bindings. In particular, its TCV1 implementation and native training request are
not available from clean `main` and must not be copied or treated as landed API.
This training/TCV1 worktree is distinct from the separately announced
score-provider effort, for which no published integration artifact is available.

No `MetricProvider`, provider registry, implementation descriptor or typed
custom-metric evaluation task was discoverable in that worktree. Consequently:

- loss-only planning may continue;
- L1 source work requiring TCV1 waits for the concurrent artifact instead of
  duplicating canonicalization;
- metric/shared-schema implementation remains blocked until a branch, commit,
  PR and API map are published;
- the loss roadmap must consume the provider descriptor if it is generic enough,
  rather than adding a parallel descriptor type.

## nirs4all Python baseline

### Shared defaults

`ModelControllerUtils` contains framework-oriented training defaults
(`controllers/models/utilities.py:24-42`) and selection defaults
(`utilities.py:134-147`). These are separate from the reporting defaults in
`core/metrics.py:637-658`.

| Role | Regression | Binary classification | Multiclass classification |
| --- | --- | --- | --- |
| framework loss | `mse` | `binary_crossentropy` | `sparse_categorical_crossentropy` |
| controller metric list | `mae`, `mse` | `balanced_accuracy`, `accuracy`, `auc` | `balanced_accuracy`, `accuracy`, `categorical_accuracy` |
| selection metric | `rmse` / minimize | `balanced_accuracy` / maximize | `balanced_accuracy` / maximize |
| reporting metrics | 13 regression metrics | 10 binary metrics | 8 multiclass metrics |

There is no public semantic object that binds these choices to a controller,
output, phase, implementation fingerprint or replay policy.

### Controller inventory

| Controller family | Effective training loss | Local custom loss today | Early stopping | Evidence |
| --- | --- | --- | --- | --- |
| TensorFlow/Keras | task default from `get_default_loss`; base is `mse` | implicit: `compile.loss` or flat `loss` is passed through to Keras | opt-in callback, `val_loss`, patience 10, restore best weights; default best-model memory also tracks validation loss | `tensorflow/config.py:42-73`, `:284-297`, `:363-405`; characterization missing |
| PyTorch | `MSELoss` regardless of task unless configured | implicit: non-string object/callable used directly | validation value of the training loss, patience 10, restore best state | baseline `torch_model.py:197-290`; PR `nirs4all#46` adds focused resolution tests and removes unknown-name fallback |
| JAX/Flax | MSE for regression; Optax softmax cross entropy for classification | no; loss is hard-coded inside jitted train/eval closures | validation value of the same hard-coded loss, patience 10 | `jax_model.py:212-332`; characterization missing |
| scikit-learn and compatible estimators | controller-internal/estimator-owned | only when the estimator's own parameter surface supports it; no generic callback | estimator-owned or absent | `sklearn_model.py`; no generic loss contract test |
| AutoGluon | framework/model-owned | no generic nirs4all callback | framework-owned | `autogluon_model.py:216-217`; no generic loss contract test |
| analytic/portable PLS families | closed-form/internal objective | not configurable | absent | algorithm-specific tests; must eventually declare `not_configurable` |

TensorFlow and PyTorch callable acceptance is process-local behavior only. The
callable is not fingerprinted, attested, serialized or reconstructable by a
detached dag-ml replay.

### Selection and reporting

`get_best_score_metric` returns `rmse`/minimize for regression and
`balanced_accuracy`/maximize for classification (`utilities.py:134-147`).
Run-level refit defaults to `rmsecv`; an empty configured metric is resolved late
from stored scores and otherwise falls back to RMSE
(`pipeline/execution/refit/config_extractor.py:158-161`, `:580-587`).

`core.metrics.eval` is string/list dispatch. Unknown single metric calculation
raises, while `eval_list` logs an individual failure and stores `None`
(`core/metrics.py:174-191`, `:590-601`). Reporting lists are static in
`get_default_metrics` (`:637-658`).

Metric direction is less strict: `is_higher_better` checks a closed set and an
unknown name silently means lower-is-better (`core/metrics.py:119-147`). That
fallback can sort a custom or misspelled metric in the wrong direction before
metric computation reports the unknown name.

### Tuning, pruning, thresholds and ensembles

- Legacy Optuna infers direction through `is_higher_better` when a metric is
  explicit. Without a metric it uses classification/maximize or
  regression/minimize (`optimization/optuna.py:582-610`).
- Optuna defaults are sampler `auto`, approach `grouped`, evaluation mode
  `best`, pruner `none`; unknown option names raise (`optuna.py:380-421`).
- `NativeTuning` independently defaults to metric `rmse`, direction `minimize`,
  50 trials and no explicit pruner (`api/tuning.py:455-474`).
- The n4m optimizer mirrors the minimize/task-resolution behavior
  (`optimization/n4m_engine.py:178-222`, `:411-412`).
- Shared model selection defaults to RMSE and delegates direction to
  `infer_ascending` (`controllers/shared/model_selector.py:145-168`).
- Fold weighting references the task's best-score metric
  (`controllers/models/base_model.py:1184-1185`, `:2430-2462`).
- AOM classification contains a private log-loss optimization implementation
  (`operators/models/_aom_nirs/pls/classification.py:288-301`).

## nirs4all-core and binding baseline

nirs4all-core currently executes one portable PLS subset. It exposes no
training-loss, early-stopping, custom-metric or metric-selection configuration.
The `dag_ml` domain is metadata-only in the aggregate capability matrix.

Every executable binding delegates PLS numerics to nirs4all-methods, but computes
RMSE and selects the minimum locally:

| Language | RMSE/selection implementation | Characterization |
| --- | --- | --- |
| Python | `bindings/python/src/nirs4all_core/_execution.py:105-110` | `bindings/python/tests/test_execution_parity.py:65-82` |
| Rust | `bindings/rust/nirs4all/src/lib.rs:719-731`, `:1085-1102` | binding parity test around `:1906-1985` |
| R | `bindings/r/R/execution.R:92-105` | `bindings/r/tests/parity.R:97-152` |
| JavaScript/WASM | `bindings/wasm/src/execution.js:78-84`, `:373-383` | `bindings/wasm/tests/execution.test.js:153-162` and parity test |
| MATLAB/Octave | `bindings/matlab/+nirs4all/runPortablePipeline.m:55-67` | `bindings/matlab/tests/parity.m:85-88` |
| Python oracle | `scripts/parity/generate_python_oracle.py:142-151` | consumed by every binding parity gate |

These tests pin equivalent RMSE values and selected components. They do not make
RMSE configurable and do not prove custom loss execution. Adding local custom
loss support to the aggregate therefore depends on the DAG-ML contract and C ABI
instead of adding six more independent semantics.

## Fallback and duplication register

| ID | Severity | Current behavior | Required disposition |
| --- | --- | --- | --- |
| `F-LOSS-001` | high | unknown nirs4all PyTorch loss silently becomes MSE | error explicitly; the reviewed draft fix in `nirs4all#46` still requires merge |
| `F-METRIC-001` | high | unknown dag-ml Python in-process selection metric silently becomes RMSE | score-provider-owned resolver must error before execution |
| `F-DIR-001` | high | unknown nirs4all metric silently means minimize/ascending | require explicit objective for custom metrics and error for unresolved built-ins |
| `F-REPORT-001` | medium | `eval_list` converts metric failures to `None` | reporting policy must make missing/error behavior explicit |
| `D-METRIC-001` | high | RMSE calculation and argmin duplicated in five bindings plus oracle | replace with native DAG-ML metric/selection contract; bindings adapt only |
| `D-DIR-001` | medium | `get_best_score_metric` duplicates direction from `HIGHER_IS_BETTER_METRICS` | migrate to `MetricSpec.objective` |
| `D-DEFAULT-001` | medium | model utility metric lists differ from reporting metric lists | preserve role distinction in the effective plan instead of merging lists |
| `D-AOM-001` | medium | AOM owns private metric implementations | classify as model-internal training criteria or route final scoring through DAG-ML |

## Characterization coverage and gaps

Existing tests that anchor behavior before migration:

- dag-ml selection and direction: inline tests in `selection.rs` and
  `metrics.rs`, CLI contract tests, and C ABI scorer tests;
- nirs4all metric lists/direction, including the unknown-name fallback:
  `tests/unit/core/test_metrics_defaults.py`;
- nirs4all scoring/refit/ranking: `tests/unit/test_scoring_invariants.py`,
  `tests/unit/data/test_prediction_ranking.py`,
  `tests/unit/data/test_predictions_scores.py` and refit selector tests;
- nirs4all tuning validation: `tests/unit/optimization/test_optuna_validation.py`;
- nirs4all-core RMSE/argmin parity: the five binding gates listed above.

Missing characterization tests required before changing each surface:

| Gap | Owning repository | Required focused test |
| --- | --- | --- |
| TensorFlow task loss and monitor defaults | `nirs4all` | backend-independent config test plus optional real-Keras smoke |
| PyTorch callable invocation in CV and refit | `nirs4all` after DAG contract | gradient/counter integration test; resolution-only tests already exist in PR #46 |
| JAX task loss and monitor defaults | `nirs4all` | extracted resolver/monitor characterization before adding registry support |
| shared selection defaults | `nirs4all` | direct tests for `get_best_score_metric` and effective refit resolution |
| Python metric unknown-name rejection | `dag-ml` score-provider branch | negative binding test replacing silent RMSE fallback |
| local custom loss lifecycle | `dag-ml` bindings | one registry/callback lifetime test per official language |

## L0 status

- ADR-22 and the delivery roadmap are published in draft PR `dag-ml#18` and
  independently approved as documentation.
- The PyTorch silent-loss fallback fix is published in draft PR `nirs4all#46`,
  independently approved twice, and covered by focused resolution tests.
- The baseline inventory is complete, but L0 remains open until the concurrent
  score-provider work publishes its branch, commit, PR and API map and the
  missing characterization tests above land in their owning repositories.
- No source or shared-schema file in the active dag-ml worktree was modified by
  this inventory effort.
