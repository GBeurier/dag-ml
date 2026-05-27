# DSL Parity With nirs4all Pipelines

Status: working contract.

Goal: the dag-ml DSL must be at least as expressive as the current nirs4all
pipeline surface, while keeping dag-ml's architecture: operators stay external,
controllers live in bindings/hosts, and Rust compiles, validates, schedules and
audits the DAG, campaign, OOF and leakage contracts.

This document tracks semantic parity first. The strict `PipelineDslSpec` is the
portable canonical form. A JSON compatibility importer now lowers serialized
nirs4all-style list/dict pipelines into that canonical form; Python object and
YAML frontends should be thin host-side serializers around the same importer.

## Non-Negotiable Design Rules

- Splits are campaign/controller invocations (`split_invocation`), not graph
  operators.
- Any data/model shape mutation must be visible in public DSL fields:
  `shape`, `aggregation_policy`, `augmentation_policy`, `selection_policy`,
  `train_params` and `tuning`.
- OOF safety is graph-visible through prediction edges with `requires_oof` and
  fold alignment; refit/replay consumes validated caches/bundles.
- Branch, merge, concat, source fusion and stacking are DAG structure, not
  Python call-order side effects.
- Runtime behavior belongs to controllers. The DSL records enough intent and
  contracts for Rust to reject unsafe execution and for bindings to route tasks.
- Minimal aliases are the preferred frontend surface when an operator is
  unambiguous. `SNV` should compile as an opaque transform payload; the Python
  or native controller registry resolves it through manifest
  `operator_selectors` to the matching controller. Verbose forms are reserved
  for parameters, explicit ids, controller hints or ambiguous routing.

## Parity Matrix

| nirs4all surface | dag-ml DSL representation | Contract status |
|---|---|---|
| plain preprocessing/transform step | `kind: "transform"` | compiled to `NodeKind::Transform` |
| model step, named model, explicit params | `kind: "model"`, `operator`, `params`, `metadata` | compiled to `NodeKind::Model` |
| target/y processing | `kind: "y_transform"` | compiled to `NodeKind::YTransform` with target ports |
| splitters (`KFold`, `GroupKFold`, SPXY, fold files) | top-level `split_invocation` in campaign template | deliberately outside graph nodes |
| sequential grouping (`[...]`) | `kind: "sequential"` | inlined during compilation while preserving child node contracts |
| sample augmentation | `kind: "sample_augmentation"` plus mandatory `shape.augmentation_policy` | compiled as augmentation with `dsl_augmentation_kind=sample`; unsafe scopes refused |
| feature augmentation | `kind: "feature_augmentation"` plus `shape` | compiled as augmentation with `dsl_augmentation_kind=feature` |
| feature selection / shape-changing processing | `shape.selection_policy`, `feature_namespace`, schema fingerprints | shape plan validated and attached to campaign |
| sample filters | `kind: "sample_filter"` or `kind: "filter"` | compiled to exclusion/filter nodes with explicit `dsl_filter_kind` metadata |
| concat transform / multi-view feature fusion | `kind: "concat_transform"` with branch transforms | compiled to `NodeKind::FeatureJoin` |
| duplicated branches | `kind: "branch"`, `mode: "duplication"` | multiple branch predictions retained |
| separation branches by source/metadata/tag/filter | `branch.mode`, `branch.selector`, per-branch selectors | graph intent compiled; controller/data-provider semantics remain host-side |
| multiple models per branch | multiple `kind: "model"` steps inside a branch | compiled into distinct OOF inputs for downstream merge |
| merge predictions/features/original data | `kind: "merge"`, `merge_mode`, `output_as`, `include_original_data`, `selectors` | compiled to join nodes with OOF prediction edges and branch data edges; `features`/`sources` consume transformed branch outputs, `all`/`mixed` can consume branch data plus OOF predictions plus original data |
| merge plus immediate meta-model | `kind: "merge_model"` convenience | compiled as model consuming OOF prediction inputs and optional original data |
| stacking, multi-model top-level stacks | repeated `model` steps then `merge`/`merge_model` | pending predictions are preserved until consumed |
| per-branch/per-model selection (`best`, `top_k`, `all`) | `merge.selectors` with branch/model/input scopes | selector targets and `top_k`/metric requirements are compile-validated; scoring remains controller policy |
| finetune / hyperparameter search | `tuning` or `finetune_params`, plus generation dimensions/variants | intent compiled into metadata; concrete tuning engine remains controller-side |
| final train params | `train_params` | preserved as `dsl_train_params` metadata |
| `_range_`, `_log_range_`, `_grid_`, param `_or_`, `pick`, `arrange`, `count` | `variants`, explicit `generation_dimensions`, or compact `generators` on DSL nodes | compiled into deterministic `GenerationSpec` dimensions |
| structural `_or_` over step chains | `kind: "generator"`, `mode: "or"`, `branches`, `pick`/`arrange`/`count` | expanded into explicit OOF-producing choices with namespaced node ids and generator metadata |
| structural `_cartesian_` over pipeline stages | `kind: "generator"`, `mode: "cartesian"`, `stages` | expanded into explicit Cartesian OOF-producing choices with namespaced node ids and fold-safe downstream merge inputs |
| serialized list/dict nirs4all surface | top-level `pipeline` array with `preprocessing`, `model`, `branch`, `merge`, `_or_`, `_cartesian_`, `_chain_`, `_grid_`, `_range_`, `_log_range_`, `_zip_`, `_sample_` | compatibility importer lowers to canonical DSL; data-only branch feature merges and merge dicts are compiled, and data-only generator stages are fused with downstream model generators so OOF choices stay complete |
| minimal aliases / plain operator refs | short strings plus `{"class": ...}`, `{"function": ...}`, `{"ref": ...}`, `{"type": ...}` and `{"name": ..., "step": ...}` wrappers | Rust infers only safe planning class: splitters become campaign split invocations, obvious estimators become model nodes, chart aliases become chart nodes, all other aliases remain external transform operators for host registry/controller resolution |
| multiple nirs4all splitter declarations | one campaign `split_invocation` with `params.compat_split_chain` | splitters remain outside graph nodes while preserving train/test + CV chains for host split controllers |
| multisource data | `data_bindings.source_ids`, branch/source selectors, source joins | contract surface present; richer materialization belongs to dag-ml-data |
| repetition/sample/group aggregation | top-level/shape `aggregation_policy`, target/group OOF cache contracts | core runtime implemented for sample/target/group OOF |
| tag/exclude filters | `kind: "tag"` and `kind: "exclude"` | compiled to graph nodes |
| charts/reports | `kind: "chart"` | compiled as a chart node; host controller decides side effects |

## Current Gaps

- Native JSON compatibility import exists for serialized nirs4all-style
  list/dict syntax, including minimal aliases and plain `class`/`function`
  descriptors. Direct Python object/YAML parsing is still a binding-layer task:
  hosts must serialize live objects and splitters into portable descriptors
  before handing the DSL to Rust.
- The new DSL node kinds compile and validate graph contracts; production
  execution still needs host controller support for each operator family.
- Separation branch materialization by source/metadata/tag/filter must be backed
  by explicit dag-ml-data view plans before it is considered runtime-complete.
  The compiler now carries transformed branch data through `merge:
  features/sources/all`, but selector-driven materialization still belongs to
  provider/controller plans.
- Merge selector scopes and basic selection contracts are compile-validated, and
  OOF edges are enforced; actual metric scoring and ranking remain the
  responsibility of selection and merge controllers.
- Synthetic generation is not settled as a separate library boundary yet. The
  DSL should represent generator nodes/controllers, but actual generators should
  stay external unless they are pure DAG/campaign coordination primitives.

## Required Regression Coverage

- Compile every nirs4all canonical sample category into strict DSL equivalents:
  linear, feature augmentation, sample augmentation, branch, stacking/merge,
  concat transform, filters/splits, finetune, multisource and all-features.
- For each shape-changing step, assert that `DataModelShapePlan` exists and
  rejects unsafe augmentation/selection scopes.
- For every stacking/merge pattern, assert that upstream prediction edges carry
  `requires_oof=true` and fold alignment.
- For repeated/grouped data, assert that target/group OOF cache requirements
  survive bundle capture and replay.
