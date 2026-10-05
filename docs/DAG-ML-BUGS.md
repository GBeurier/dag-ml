# DAG-ML bug audit

Suivi de validation et corrections : [DAG-ML-BUGS-RESOLUTION.md](DAG-ML-BUGS-RESOLUTION.md). Les constats originaux ci-dessous sont conservés pour le réaudit.

**Second passage, 2026-10-05 : les cinq défauts reproduits R1–R5 sont corrigés (deux Medium, trois Low).** Quatre complètent des correctifs A2 ; le cinquième concerne le nouveau helper multimodal. Voir le [réaudit ciblé](#réaudit-ciblé-du-2026-10-05--r1r5) et la [résolution complémentaire](DAG-ML-BUGS-RESOLUTION.md#corrections-complémentaires--r1r5). Les 117 constats et les totaux du premier passage restent historiques.

Date: 2026-10-05. Repository state: `main` at `867f357` (workspace version 0.3.34).

## Method and caveats

- Static review of the whole workspace by eight parallel read-only reviewers, one per area (sections A1-A8 below). Every file in scope was read, except test modules, which were skimmed.
- Findings are **not reproduced**. Each was checked against surrounding code and callers by its reviewer. The lead spot-checked the High findings (A1-01 confirmed in `fold.rs`; A2-02's unbounded `collect` seen at `dsl/generation.rs:1719`). Treat Low and low-confidence items as leads to verify before fixing.
- Gate state at audit time:
  - `cargo clippy --workspace --all-targets -- -D warnings`: clean.
  - `cargo test --workspace --no-fail-fast`: one failure, `cli_replays_portable_methods_pls_payload_in_fresh_process` (`crates/dag-ml-cli/tests/initial_full_refit.rs:313`). Environmental, not a dag-ml defect: the sibling checkout's `nirs4all-methods/bindings/python/src/pls4all/lib/libn4m.so` is stale and lacks `n4m_model_from_method_result`. Rebuild `libn4m` to clear it.
- Severity: High = wrong results, leakage or process abort reachable from normal or untrusted input; Medium = real defect with narrower trigger or impact; Low = hardening, drift, edge cases.

## Summary

| Area | Scope | High | Medium | Low | Total |
|---|---|---|---|---|---|
| A1 | OOF / folds / selection / stacking / merge | 1 | 3 | 10 | 14 |
| A2 | graph / plan / DSL compiler / generation | 2 | 6 | 10 | 18 |
| A3 | runtime scheduler / data views / scoring | 0 | 5 | 14 | 19 |
| A4 | training / bundle / replay / provenance | 0 | 3 | 10 | 13 |
| A5 | HPO / metrics / aggregation / methods | 0 | 4 | 11 | 15 |
| A6 | C ABI (`dag-ml-capi`) | 0 | 3 | 9 | 12 |
| A7 | py / wasm / arrow / results bindings | 1 | 6 | 8 | 15 |
| A8 | CLI / R / MATLAB / JS bindings / scripts / CI | 0 | 3 | 8 | 11 |
| **Total** | | **4** | **33** | **80** | **117** |

No Critical findings.

## Priority list (High and leakage-relevant Medium)

- **A1-01** Nested inner CV ignores groups and augmentation origins for `KFold` / `StratifiedKFold` inner policies, so leakage reaches stacking meta-features and inner HPO scores.
- **A2-01** `enumerate_variants` builds the full cartesian product before applying `max_variants`.
- **A2-02** `pick` / `arrange` size ranges (e.g. `[1, u64::MAX]`) are collected into a Vec before bounding, so untrusted JSON can panic or abort the process.
- **A7-01** Python `TrainingResult` mutex is held across Python callbacks while the GIL is not detached (deadlock risk; medium confidence).
- **A6-01** Shared `user_data` across owned ABI v3 vtables in replay / initial-refit entry points leads to a double `destroy`.
- **A3-01..A3-04** Scheduler and HPO fold-selection inconsistencies (independent-unit grouping, Parallel vs Sequential REFIT, stacking weight-score source).
- **A5-02** `HostHpoSearchRequest.direction` is never checked against the metric objective, so a mismatch silently picks the worst trial.
- **A8-02** R / MATLAB wrappers round `lineage.seed` through a double, so seed injection fails for almost every task.
- **A4-01 / A4-02** Archive V3 and research-provenance validators accept packages that are not linked or rebuilt from the plan.

---

## A1 audit: dag-ml-core oof / fold / selection / relation / policy / chain_effect / runtime {oof, stacking, merge}

Paths are relative to /home/delete/nirs4all/dag-ml/crates/dag-ml-core/src. All files in scope were read in full, tests excluded. Nothing was edited and cargo was not run.

### A1-01 | High | confidence High | fold.rs:180-188, 652-700 (callers runtime/stacking.rs:633,675; runtime/merge.rs:1443)
**Nested inner CV ignores groups and augmentation origins (KFold / StratifiedKFold inner policies).**
- `FoldSet::nested_fold_set` documents "The inner set inherits this set's groups". It only copies `train_exclusion`.
- `build_nested_fold_set` passes `outer_groups` to the splitter only for `GroupKFold`.
- `KFold` and `StratifiedKFold` return an inner FoldSet with empty `sample_groups`, built purely by sorted-index round-robin.
- `validate_for_outer` and `FoldSet::validate` on that inner set therefore cannot flag group leakage, because the group map is empty.
- `relations.validate_against_fold_set` is only called on the outer fold set (plan.rs:162, methods_fold_hpo.rs:237,501). It is never called on inner fold sets, so `origin_cross_fold` and `split_unit=Group` are not enforced inside nested CV.
- No plan-time check rejects `inner_cv: kfold` when the outer set is grouped (`NestedCvSpec::validate` only checks `n_splits >= 2`).

Failure scenario:
- The outer fold set is group-aware (GroupKFold, `sample_groups` populated).
- A stacking meta node, residual learner, or tuner/finetune node declares `inner_cv: {kind: kfold}` (campaign-global or node-local).
- Inner folds put samples of the same group, or an augmented row and its origin, in inner-train and inner-validation.
- The inner OOF base predictions fed to the meta-model, and the inner HPO scores, are optimistically biased. The meta-model is trained on leaked OOF features and the HPO winner is chosen on leaked scores.
- This is the "group / repetition / augmentation-origin leakage" the repo rules forbid.

Fix:
- When `outer_groups` is non-empty and the spec is `KFold`/`StratifiedKFold`/`CapacityKFold`, either refuse it or build the inner folds group-aware (derive `GroupKFold` over outer-train groups).
- Carry the filtered `sample_groups` into `inner_fold_set` (the doc already promises this).
- Also run origin/group `validate_against_fold_set` on every nested inner fold set. `resolve_capacity_kfold` already rejects grouped sets and shows the intended behaviour.

### A1-02 | Medium | confidence Med | runtime/oof.rs:1382-1401 (also 1683-1714, 1198-1245)
**Stacking `top_k`/`best` producer ranking ignores `fold_ids` and `variant_id`, and uses one arbitrary report.**
- `StackingProducerSelectionRequest::selected_producer_nodes` (top_k/best path) picks, per producer, the first `reports` entry that is Validation with any `fold_id` and a finite metric.
- It ignores `self.fold_ids` (the "report-grade fold scope", documented for REFIT/replay) and `self.variant_id`. It also does not average across folds.
- The sibling `fold_candidates_top_k` and `diverse_fold_candidates` paths do honour both fields.
- `stacking_model_weights` has the same first-report behaviour.

Failure scenario:
- `scores` holds reports for several variants, or for inner/nested folds in addition to outer folds.
- The ranking of producers, and therefore which base models feed the meta-learner, depends on report insertion order. It can use an inner-fold or other-variant score.
- Selection is non-reproducible between FIT_CV and REFIT/replay, and is made on the wrong evidence.

Fix: filter by `fold_ids` (when non-empty) and `variant_id`, and aggregate (mean) over the matching folds per producer, as the fold-candidate paths do.

### A1-03 | Medium | confidence Med | relation.rs:133-163 (validate), 358-383
**An augmented relation row without `origin_sample_id` silently bypasses `forbid_origin_cross_fold`.**
- `SampleRelation::validate` never requires `origin_sample_id` when `is_augmented == true`.
- `AugmentationPolicy.require_origin_id` is not enforced on relations. `is_augmented` is read only in runtime/dataview.rs:1828.
- `validate_against_fold_set` only checks `origin_sample_id` when it is `Some`.

Failure scenario:
- A host emits augmented rows (`is_augmented=true`) with `origin_sample_id=null`, whether by mistake or as adversarial JSON.
- A copy of a validation sample's spectrum then sits in the training partition and the origin-leak guard never fires.
- The policy default is `forbid_origin_cross_fold=true`, so a user would expect protection.

Fix: in `SampleRelation::validate`, require `origin_sample_id.is_some()` when `is_augmented`, unless an explicit `allow_sample_augmentation_without_origin` unsafe flag is set. Also reject `origin_sample_id == sample_id` for non-combo augmented rows.

### A1-04 | Medium | confidence High on the loop, Med on impact | runtime/stacking.rs:208-235 (and fold.rs:604-609)
**`resolve_capacity_kfold` loops over `min_splits..=max_splits` with no upper bound or early exit.**
- `CapacityKFoldSpec` is a deserialised `usize` pair. `NestedCvSpec::validate` only requires `2 <= min <= max`.
- Every `splits` above the smallest outer-train size fails fast with "n_splits exceeds sample count", but the loop still runs all the way to `max_splits`.

Failure scenario: a plan with `max_splits = 18446744073709551615` (or just 1e9) hangs the scheduler at plan time, burning CPU on a loop that cannot succeed.

Fix: clamp `max_splits` to the smallest `train_sample_ids.len()` across outer folds and the refit universe. Break the loop once the error is a sample-count failure, since larger counts only get worse.

### A1-05 | Low | confidence Med | oof.rs:632, 693, 821; runtime/merge.rs:326, 508, 954; runtime/oof.rs:1774-2166 (`prediction_input_spec`)
**Several join and reassembly paths call `validate_shape` instead of `validate_content`, so NaN/Inf pass.**
- `join_oof_features`, `join_oof_campaign_features`, `validate_prediction_blocks_against_folds`, `reassemble_off_fold_concat`, `reassemble_separation_merge`, off-fold merge, `prediction_input_spec` and the collect_* paths use `validate_shape` only.
- `validate_content` (the documented "single gate") rejects non-finite values and in-block duplicates.
- Duplicates are caught elsewhere in most of these paths. Non-finite values are not.
- The availability path does check finiteness (`expand_available_prediction_input`).

Failure scenario: a host base model emits NaN for some rows. A non-availability OOF join or concat merge forwards the NaN to the meta-model feature matrix, or into the concatenated validation block, without an error here. This is only caught if the store or scoring gate does it later.

Fix: use `validate_content` in these paths. It is a strict superset and cheap.

### A1-06 | Low | confidence Low | selection.rs:355-429, 76-84, 204-223
**`select_candidate` does not enforce `evaluation_scope`.**
- `SelectionPolicy.evaluation_scope` is only echoed into the decision.
- `CandidateScore` has no scope or partition field. Only `metadata["metric_level"]` is checked.
- A Train-, Refit- or Holdout-scored candidate can be ranked under an `oof` policy.
- Native training enforces `evaluation_scope == Oof` separately (training_runtime.rs:3153), so this affects only direct callers.

Fix: add `evaluation_scope` to `CandidateScore` (or require `metadata["evaluation_scope"]`) and refuse mismatches, as is done for `metric_level`.

### A1-07 | Low | confidence Med | policy.rs:599-607, 744-751
**Leakage-policy asymmetries.**
- `AugmentationPolicy::validate` gates `sample_scope == AllPartitions` behind an unsafe flag, but not `feature_scope == AllPartitions`. `compat_helpers.rs:424` can set `feature_scope = AllPartitions`.
- `DataModelShapePlan::validate` never rejects `fit_rows` of `FoldValidation` or `Predict` (fitting on held-out rows), except when supervised feature selection is on. No code outside policy.rs checks `fit_rows`.

Fix: apply the same unsafe-flag gate to `feature_scope`. Reject `fit_rows` in {FoldValidation, Predict} unless an unsafe flag is present.

### A1-08 | Low | confidence Med | relation.rs:358-383
**`forbid_origin_cross_fold` hard-errors for samples that are in neither the train nor the validation list of a fold.**
- Under `FoldPartitionMode::Resampled` (ShuffleSplit/bootstrap), a sample may be absent from a fold. `partitions.get(...)` then fails with "fold does not contain sample".
- This rejects legitimate Resampled fold sets whenever any relation has an `origin_sample_id`.
- A cleaner semantic is that an absent sample cannot leak.

Fix: skip the pair when either side is absent. Fail only if exactly one side is present and they differ, or if the origin is in validation and the derived row is in train.

### A1-09 | Low | confidence Med | runtime/merge.rs:1482-1499
**The `debug_assert!` in `sample_ids_for_partition(FullTrain)` can panic on valid Resampled fold sets.**
- It asserts every `fold_set.sample_ids` entry appears in some fold's train or validation. `FoldSet::validate` does not guarantee this for Resampled.
- This panics in debug and test builds, and is silent in release. The behaviour differs between profiles.

Fix: drop the assert, or restrict it to Partition mode.

### A1-10 | Low | confidence Low | runtime/oof.rs:615-626
**`validate_available_target_masks` indexes with `masks[row]` and `block.values[row][0]` without bounds checks.**
- A host-supplied `regression_targets` block with `validity_masks` shorter than `unit_ids`, or empty `values` rows, panics.
- `NodeResult` shape validation may already catch this upstream; I did not confirm it.

Fix: use `.get()` and return an error.

### A1-11 | Low | confidence Med | runtime/oof.rs:1232-1234, 1278-1286
**`StackingFoldSelectionRequest` falls back silently on missing evidence.**
- `selected_fold_id` returns `fold_ids[0]` when no validation report exists. The "best fold" estimator is then an arbitrary one, with no error.
- In `normalized_weights`, a lower-is-better metric with a negative score gets weight `|score|`, which rewards the worst value.

Fix: error out (or at least log lineage) when `ranked` is empty. Treat a negative error metric as invalid.

### A1-12 | Low | confidence Low | runtime/oof.rs:1483-1490, 1587-1589
**`fold_candidates_top_k` and `diverse_fold_candidates` apply `take(limit)` to per-(producer, fold) report rows before dedup by producer.**
- When one producer has several top-ranked fold reports, fewer than `k` distinct producers come back.
- This is probably intended for "fold candidates", but it contradicts the name `max_per_class` / `top_k`.

### A1-13 | Low | confidence Low | runtime/merge.rs:368, 740, 994
**Branch merge target reassembly silently overwrites per-sample `y_true`.**
- `by_sample_target.insert` is last-write-wins across branches, with no conflict check. The residual path in runtime/stacking.rs:977 does check.
- If branches disagree on y_true for the same sample (different cohort or filter), the merged block is scored against whichever branch came last.

Fix: reject conflicting rows, as the residual path does.

### A1-14 | Low | confidence Low | runtime/oof.rs:310-327
**`expand_available_prediction_input` zips `sample_ids` with `values`.**
- If `values` has more rows than `sample_ids`, the extras are silently dropped. The check only validates rows that are present in the zip.

Fix: assert `input.values.len() == input.sample_ids.len()`.

### Reviewed, no bug found
- chain_effect.rs: construction is fail-closed. Median and z-score edge cases are handled.
- fold.rs: GroupKFold, StratifiedKFold, KFold and the fingerprint canonicalisation.
- selection.rs ranking: NaN is rejected and ties break by id.
- oof.rs stacking coverage and refit contracts, and the policy.rs reduction/aggregation validation.

---

## A2 audit: dag-ml-core graph / plan / generation / controller / canonical / dsl

Scope read in full (tests skipped): `graph.rs`, `plan.rs`, `generation.rs`, `controller.rs`, `controller_adapter.rs`, `canonical.rs`, `lib.rs`, `dsl/{compiler,generation,compat,compat_helpers,fanout,types}.rs`.
Paths below are relative to `/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/`.

Verified-OK (no bug found): cycle detection / self-loops / dangling edges / duplicate node ids in `graph.rs` (Kahn with BTreeSet, deterministic order); TCV1 parser/encoder in `canonical.rs` (NFC, -0.0, NaN, surrogates, depth); `controller_adapter.rs` templates; constraint rule core (`constraints_satisfied`); `prune_plan_to_active` / `validate_active_inputs`.

Summary: High 2, Medium 6, Low 10.

---

### A2-01 | High | confidence high
**generation.rs:521-555 and 620-636; dsl/generation.rs:1214-1237; generation.rs:178-195**
**`max_variants` is checked only AFTER the full cartesian product is materialized (and is unbounded by default from DSL/JSON)**

`enumerate_variants` calls `cartesian_choices` which builds every variant (`Vec<BTreeMap<String, GenerationChoice>>`, each cloning every choice incl. `serde_json::Value`s) and only then compares `variants.len()` to `max_variants`. `GenerationSpec::validate` never bounds the product size. Additionally `GenerationSpec::default()` has `max_variants: Some(1)` but the serde default for a missing `max_variants` is `None` (`#[serde(default)] Option<usize>`), and `build_generation_spec` passes the DSL `max_variants` (None by default) straight through, so a DSL/campaign JSON that omits it has no cap at all.
`Vec::with_capacity(variants.len() * dimension.choices.len())` can also overflow (debug panic / release wrap).

Scenario: campaign/DSL with 12 dimensions x 10 choices (or two `range` generators of 10 000 values each, the per-generator guard) -> 10^12 / 10^8 variants allocated before the `max_variants` error can fire: OOM-abort / hang in `build_execution_plan` or `compile_pipeline_dsl_with_generation`, from untrusted JSON. The cap that exists to prevent exactly this never protects.

Fix: compute the product size with `checked_mul` in `GenerationSpec::validate`/`enumerate_variants` (before materializing) and error when it exceeds `max_variants` (when constraints are present, cap the pre-prune product with a hard ceiling or enumerate lazily and stop at max+1 survivors). Make the serde default for `max_variants` equal `Default` (Some(1)) or require an explicit cap when strategy != none.

### A2-02 | High | confidence high
**dsl/generation.rs:1700-1723 (`selection_sizes`), called from 1291-1305**
**`pick`/`arrange` range is collected into a Vec before being bounded -> panic/OOM on external input**

`PipelineDslSelectionSpec::Range([start, stop])` is expanded with `(start..=stop).collect()` and only afterwards (in `generated_pick_sequences` / `generated_arrange_sequences`) is each size compared to `options.len()`. `[usize; 2]` deserializes any u64.

Scenario: DSL JSON `"pick": [1, 18446744073709551615]` (or `[1, 1e12]`) -> `Vec::from_iter(RangeInclusive)` = `capacity overflow` panic / allocation abort inside `compile_pipeline_dsl*` / `compile_operator_variant_models`. Through the C ABI/WASM/py bindings this is a process abort on malformed input.

Fix: validate `stop <= options.len()` (pass `options.len()` into `selection_sizes`) before collecting; better iterate lazily.

### A2-03 | Medium | confidence high
**dsl/compat_helpers.rs:736/739-755, 848-853, 983-1037; dsl/generation.rs:757-782, 1016-1072, 1386-1408, 1288-1297, 1383-1385**
**Other unbounded expansions on external input (no cap, no pre-check)**

- `_log_range_` `count`/`num` and `_sample_` `num`/`count` are `as_u64() as usize` and then `(0..count).map(..).collect()`; only `count == 0` is rejected (also `compile_log_range_generator`).
- `build_compat_grid_rows` (`_grid_` lowering) has no `count` cap at all: product of all axis lengths is built eagerly.
- `build_permutations`/`build_combinations` with `count: None` enumerate n!/C(n,k) selections (e.g. `arrange` size 12 of 12 values = 479M `Vec<usize>`).
- Operator generators: when constraints are present the `count` cap is intentionally suppressed (`gen_count = None`, `build_count = None`) so the FULL pick/arrange/cartesian expansion (every `GeneratedSequence` clones all steps) is materialized and pruned afterwards.

Scenario: `{"_log_range_":{"start":1e-3,"stop":1,"count":4294967296}}` or a `_cartesian_` with 6 stages x 12 branches + `_mutex_` -> hang/OOM at compile time.

Fix: introduce one hard ceiling (e.g. same as `max_variants`/10 000) checked before allocation; for the constrained path enumerate lazily and stop after `count` survivors (legacy order is preserved by lazy generation + filter + take).

### A2-04 | Medium | confidence medium
**campaign.rs:20-24 (`stable_json_fingerprint`), used by plan.rs:362-376, 1279, 1342-1344; generation.rs:617, 656; dag-ml-py/Cargo.toml:40**
**Graph / campaign / controller / variant / generation fingerprints depend on `serde_json::Value` object key order; the Python wheel enables `preserve_order`**

`stable_json_fingerprint` is `sha256(serde_json::to_vec(value))`. Struct fields and `BTreeMap`s are ordered, but every embedded `serde_json::Value` object (`NodeSpec.operator`, `params` values, `metadata` values, `GenerationChoice.value`, `param_overrides[].params` values, `ControllerManifest.data_requirements`, `CampaignSpec.metadata`...) serializes in map order. `crates/dag-ml-py` builds serde_json with `preserve_order` (it is excluded from the workspace, so features are not unified with the CLI/WASM/C builds) and nothing there sorts (`grep sort_all_objects` in dag-ml-py/capi/wasm/cli: none; the core only sorts in a handful of places such as `operator_variant_canonical_value`, `default_data_requirements`).

Scenario: a graph built from Python dicts `{"type": "PLS", "n_components": 5}` gets `graph_fingerprint` over key order (type, n_components); the same plan exported to JSON and re-validated by the CLI/WASM/R build (sorted maps) recomputes a different hash -> `ExecutionPlan::validate` fails "graph_fingerprint does not match the embedded graph" (or the reverse), and two Python callers with different dict insertion order get different `graph_fingerprint` / `variant.fingerprint` / `variant_id` for the same logical pipeline (breaking replay/bundle identity).

Fix: make `stable_json_fingerprint` canonical: `let mut v = serde_json::to_value(value)?; v.sort_all_objects(); sha256(to_vec(&v))` (key-sorted output is identical to the current non-preserve_order bytes, so existing fingerprints do not change).

### A2-05 | Medium | confidence medium
**dsl/compat.rs:769-786 (branch dict), 862-874 (concat_transform dict); dsl/compat_helpers.rs:720-734 (`compat_grid_rows`), 875-890**
**nirs4all-compat lowering iterates `serde_json::Map` in build-dependent order and bakes the iteration index into ids/labels**

For `{"branch": {"snv": [...], "msc": [...]}}`, `concat_transform` objects and `_grid_` objects, branches/rows are produced in `Map` iteration order and numbered (`sanitize_branch_id(key, index)`, `grid{index}`, input-port names, `compat_grid_row` metadata, row order = choice order). Without `preserve_order` the order is alphabetical, with it (py wheel) it is insertion order.

Scenario: the same nirs4all JSON compiled by dag-ml-py vs the CLI/WASM produces different branch order, different merge input port order, different `grid{index}` labels and therefore different graph/search-space fingerprints and variant ids.

Fix: sort keys explicitly (or require arrays) when lowering objects to ordered structures, so the result is independent of the serde_json feature set.

### A2-06 | Medium | confidence high
**plan.rs:326-514 (`ExecutionPlan::validate`), 1335-1341 (`build_execution_plan`)**
**`validate` does not re-derive or cross-check `variants`, `fold_set`, node-plan `data_bindings`/`shape_plan` against the (fingerprinted) campaign**

The method is documented as fail-closed against tampered embedded content, but only `graph_fingerprint`, `campaign_fingerprint`, `controller_fingerprint` and `params_fingerprint` are recomputed. Not checked: `variants` equal `enumerate_variants(campaign.generation, root_seed)` (nor `variant.fingerprint`/`seed`/unique `variant_id`), `plan.fold_set == campaign.split_invocation.fold_set`, `node_plan.data_bindings`/`shape_plan` equal the campaign maps, `validate_search_space_fingerprint`, `validate_generation_override_targets`.

Scenario: a plan JSON whose `fold_set` is replaced by folds that put validation samples in train (passes `fold_set.validate()` structurally) or whose `variants` carry a swapped override/seed, with all three fingerprints untouched, is accepted by `ExecutionPlan::from_json`/`validate` and drives scheduling (`campaign_phase_schedule` reads `self.fold_set` and `self.variants`). Duplicate `variant_id`s also yield duplicate `scope_id`s.

Fix: in `validate`, recompute `enumerate_variants` and compare (including seeds), compare `fold_set` with the campaign's, check `node_plan.data_bindings == campaign.data_bindings[node]`, reject duplicate variant ids, and re-run `validate_search_space_fingerprint` / override-target checks.

### A2-07 | Medium | confidence medium
**dsl/compat.rs:594-638 (`lower_merge_followed_by_model`), called at 135-141 and 671-675**
**Any `{"merge": ...}` followed by `{"model": ...}` is lowered to a stacking `MergeModel`, ignoring `merge_mode` / `output_as`**

`compat_merge_modes` returns `(merge_mode, include_original_data, output_as)` but `output_as` is discarded (`_`) and the mode is not inspected. A feature/source merge (`{"merge": "features"}` / `{"merge": {"features": "all"}}`) followed by an object model step becomes a `MergeModel`, whose compile requires pending OOF predictions (`compile_merge_model_with_extra`: "has no pending branch predictions") and wires prediction edges with `requires_oof`.

Scenario: nirs4all pipeline `[{"branch": {"snv": ["SNV"], "msc": ["MSC"]}}, {"merge": "features"}, {"model": PLSRegression(...)}]` (preprocessing-only branches) fails to lower with a spurious error, while the same pipeline with the model written as a bare string `"PLSRegression"` (the only form covered by `parses_nirs4all_compat_feature_branch_merge_dict`) works. When branches do contain models it silently turns a feature concatenation into prediction stacking.

Fix: only fuse when the resolved `output_as`/mode consumes predictions (`predictions`, `all`, `mixed`, `predictions_plus_original`); otherwise emit a normal `Merge` + `Model`.

### A2-08 | Medium | confidence medium
**dsl/generation.rs:707-737 (`compile_range_generator`), 1126-1135 (`json_number`, `canonical_generator_number`)**
**Integer-valued `range` generator values are emitted as JSON floats (`5.0`, not `5`)**

`start/stop/step` are `f64`, values come from `Number::from_f64`, and `canonical_generator_number` re-parses `"5.0"` as a float. So `{"_range_": [5, 15, 5]}` on `n_components` yields param override `5.0`; labels/`value` fragments also read `5.0`. Hosts that forward params verbatim (scikit-learn `PLSRegression(n_components=5.0)` -> `InvalidParameterError`, int-typed Rust params) fail at fit time, far from the DSL. The existing test compares `== 5.0`, which passes for 5 as well and does not pin the int/float distinction.

Fix: when `start`, `step` and the generated value are integral (`value.fract() == 0.0` and within i64 range, inputs integral) emit `Number::from(i64)`; keep floats only for fractional ranges.

### A2-09 | Low | confidence high
**dsl/generation.rs:767-777; dsl/compat_helpers.rs:1010-1016**
**`log_range` / `_sample_` endpoints are not exact**

`start.log(base)` is `ln(start)/ln(base)` (e.g. `1000f64.log(10.0) == 2.9999999999999996`), then `base.powf(...)`; for `count > 1` the first/last values are `~1.0000000000000002e-3` / `999.9999999999998` instead of the declared `start`/`stop` (only `count == 1` returns `start` exactly). `_sample_` uniform computes `from + (to-from)*1.0`, also not guaranteed `== to`. Hyper-parameter grids silently differ from the user's bounds and from legacy `np.logspace`/`linspace` endpoints.
Fix: pin index 0 and `count-1` to `start`/`stop`, use `log10`/`log2` for those bases.

### A2-10 | Low | confidence high
**dsl/compat.rs:144-157 with 640-726 (`combine_data_generator_with_following`)**
**Fusion look-ahead performs side effects, then may abandon and the steps are lowered a second time**

While scanning ahead the function calls `consume_side_effect_step` (sets `self.split_invocation`, `self.metadata`) and the `lower_*` helpers (advance `node_counter`/`generator_counter`) and then returns `Ok(None)` (no prediction found, or a tail-bearing next generator). The caller re-lowers the same values.

Scenario: `[{"_or_": ["SNV","MSC"]}, {"split": "KFold"}, "StandardScaler"]` (or any data-only generator followed by a split and no prediction before a tail-bearing generator): the split is registered twice -> `set_split_invocation` builds a bogus `compat_split_chain` `[KFold, KFold]` (or errors on a double `fold_set`), and generated node ids are shifted.
Fix: lower into a scratch lowerer (clone counters/split/metadata) and commit only on `Some`.

### A2-11 | Low | confidence medium
**dsl/compiler.rs:653 (`branch_metadata.extend(extra_metadata.clone())`) and 735 (`choice_metadata.extend(..)`)**
**Outer metadata overwrites the inner branch/generator context in nested constructs**

For a branch nested in a branch (or generator in a branch), the inner `dsl_branch`, `dsl_branch_mode`, `dsl_branch_selector`, `dsl_branch_view_plan` keys are replaced by the OUTER values because `extend` runs last. Nodes of the inner branch then carry the outer branch id / view plan in metadata (consumed by `runtime/oof.rs:1035` and `plan.rs:968` `dsl_branch_view_plan`), while `PredictionSource.branch_id` uses the inner id: the two disagree, and an inner `by_source` view plan is lost on the node.
Fix: insert the inner context after `extend` (inner wins) or namespace the keys (`dsl_branch_path`).

### A2-12 | Low | confidence medium
**graph.rs:271-330 (`GraphSpec::validate`); `PortCardinality` is never enforced anywhere**
**Multiple/duplicate edges into a `One`/`Optional` input port and unfed required ports are accepted**

Validation checks port existence, kinds, units and acyclicity but not `PortCardinality`: two edges can feed the same `One` input port, an identical edge can be listed twice (indegree counted twice but harmless for Kahn), and a required input with no edge is allowed (legit for external inputs, indistinguishable from a bug). `validate_active_inputs` (plan.rs:1552) enforces "exactly one source" only after an operator-SELECT prune, so the base plan is less strict than the pruned one.
Scenario: hand-written/imported graph JSON with a second `x -> model.x` edge validates; downstream input resolution becomes order-dependent.
Fix: reject >1 inbound edge to `One`/`Optional` ports (and exact duplicate edges) in `validate`.

### A2-13 | Low | confidence medium
**plan.rs:1277, 440-472; controller.rs:401-416**
**Graph node ports are never reconciled with the resolved controller manifest ports**

`resolve_for_node` returns a manifest by kind/selector (or by explicit `metadata.controller_id`, which skips selector matching entirely), and `ExecutionPlan::validate` compares only kind/phases/capabilities/policies. `manifest.input_ports/output_ports` are compared with the node's ports only for named `model_input` nodes. A node declaring ports (e.g. extra `prediction_output_ports`, named `x_out`) that the controller does not implement passes planning and fails at runtime.
Fix: check each node port exists (name/kind/representation) in the manifest port lists during `build_execution_plan`/`validate`.

### A2-14 | Low | confidence medium
**dsl/generation.rs:154-250 (`canonical_operator_step`), 71 (variant_labels)**
**`variant_label` ignores structural content and everything except `kind/class/params`**

`branch`, `generator`, `sequential`, `merge`, `concat_transform` render `class:""`, `params:{}` regardless of their children/modes; `representation`, `variants`, `param_generators`, `train_params`, `tuning`, `metadata`, `shape`, merge mode/selectors are not hashed. Two operator choices that differ only there get identical labels, so a host mapping reports back by `variant_label` cannot distinguish them. Flat operator sub-sequences (the documented use) are fine.
Fix: include children/fields in the canonical value or reject non-flat structural steps in `lower_operator_variant_model`.

### A2-15 | Low | confidence medium
**dsl/compiler.rs:2482-2526 (`branch_input_prefix`, `branch_prediction_input_name`); dsl/generation.rs:1195-1213, 1459-1461, 2013-2032; compat_helpers.rs:605-612**
**Lossy id sanitization collapses distinct ids**

`branch_input_prefix` maps every non-alnum/`_` char to `_` and trims (so `a-b`, `a.b`, `a:b`, `a_b` collide; the `index` only disambiguates when the result is empty); `branch_prediction_input_name` does the same for node ids; generator constraint members use `sanitize_generation_label(branch.id)` (distinct branch ids `a:b`/`a_b` both match a `_mutex_` ref to either); generated node ids truncate the generator id to 32 and original id to 28 chars. Effects: spurious "duplicate port"/"duplicate node" graph errors for legal inputs, and constraint refs matching more than one option.
Fix: append the index (or a short hash) when the sanitized form differs from the input, and detect collisions explicitly.

### A2-16 | Low | confidence medium
**dsl/compiler.rs:2292 (`top_k as usize > matched.len()`); dsl/compat_helpers.rs:853, 988**
**`u64 -> usize` casts truncate on 32-bit/wasm32**

`top_k = 2^32+1` becomes 1 on wasm32 and passes the "exceeds matched inputs" check; `_log_range_`/`_sample_` counts silently wrap. dag-ml-wasm is a first-class target.
Fix: `usize::try_from(..)` with an error.

### A2-17 | Low | confidence low
**dsl/types.rs:341-349 (`PipelineDslGeneratorValue`, `#[serde(untagged)]`)**
**A plain object value with `label` and `value` keys is parsed as `Labeled`**

Untagged tries `Labeled{label,value}` first, so a genuine dict parameter value such as `{"label": "x", "value": 1}` in an `or`/`grid`/`pick` list is reinterpreted (value 1 instead of the dict) with no error.
Fix: use an explicit tag or require `deny_unknown_fields` plus documentation; at minimum reject extra keys.

### A2-18 | Low | confidence medium
**dsl/generation.rs:857-885 and 1546-1561 (`compile_pick_arrange_generator`, `generated_pick_sequences`)**
**Duplicate `sizes` entries produce duplicate selections / identical variants**

`sizes: [2, 2]` (explicit list for `Pick`/`Arrange` generators) runs the builder twice and appends identical selections; labels are made unique by the index prefix so `GenerationDimension::validate` passes, and the same hyper-parameter value is trained twice as distinct variants (wasted budget, skewed selection among "ties"). Same for `values` containing duplicates.
Fix: dedup sizes (and selections) before generating.

---

## A3 audit: dag-ml-core runtime scheduler / task / dataview / artifact / prediction_store / scoring / mod / data_provider_bootstrap + data.rs + terminal_prediction.rs

Scope read in full (non-test code): `runtime/{scheduler,task,dataview,artifact,prediction_store,scoring,mod,data_provider_bootstrap}.rs`,
`data.rs`, `terminal_prediction.rs`. Callers in `runtime/{oof,merge,stacking}.rs` and `metrics.rs` were consulted only to verify findings.
No code was edited, cargo was not run. All line numbers refer to the working tree at audit time.

Summary: 0 Critical, 0 High, 5 Medium, 14 Low.

Observation on handle lifecycle: the in-scope code never releases handles (`HandleRef` is plain data; `RuntimeArtifactStore` and
`RuntimeDataProvider` have no release hook; only `RuntimeController::release_hydrated_artifact_payload` exists and is called outside scope).
No double-free or use-after-release was found inside scope; the only leak-shaped item is A3-18.

---

### A3-01  Medium  (confidence: medium)
**`execute_hpo_campaign` / host HPO fold selection is ambiguous or wrong when an independent-unit (grouping_key) policy is active**
`runtime/scheduler.rs:701-717` (per-fold report filter), `:513-526` (avg report `find`), `:806-820` (host worker fold), compare with the grouping-aware filter at `:879-891`.

`execute_phase_scope` appends BOTH the sample-level per-fold Validation report (`apply_result_scoring`) and, when
`campaign.aggregation_policy.grouping_key.is_some()`, a Group-level report per block (`apply_independent_unit_scope_reports`, scheduler.rs:2771-2796). `score_independent_unit_block`
keeps `producer_node`/`producer_port`/`partition`/`fold_id` of the sample block (metrics.rs:864-870), so both reports match the filter in
`execute_hpo_candidate_fit_cv` (node + port + Validation + fold) and in `execute_host_hpo_worker_fold` (node + Validation + fold, no level/grouping filter).

Failure scenario: HPO campaign over a plan with a signed `experimental_unit` descriptor. Every candidate hits `let [fold_report] = ... else Err("must emit exactly one target validation report per fold; fold ... emitted 2")`.
In `execute_hpo_campaign` that error is swallowed as `DAGML_CV_ERROR` (see A3-06), every trial is marked failed and the campaign ends with the misleading "no completed native incumbent". Even if per-fold passed, the final `avg` lookup (`.find`, line 513) returns the first matching `avg` report, which is the sample-level average (pushed before the Group-level one in `apply_global_oof_aggregation`), so trials would be ranked on a row-weighted metric instead of the independent-unit metric that `execute_host_hpo_candidate_fit_cv` and `select_best_variant_outcome_by_cv_for_target` explicitly use.

Fix: filter on `report.level == plan.campaign.aggregation_policy.selection_metric_level && report.grouping_key == plan.campaign.aggregation_policy.grouping_key` (exactly as lines 883-887) in all three places, including the `avg` lookup and `validate_hpo_checkpoint_result`.

---

### A3-02  Medium  (confidence: medium-low)
**`ParallelScheduler` REFIT silently bypasses the nested-stacking / residual machinery that `SequentialScheduler` runs**
`runtime/scheduler.rs:2914-2976` and `:2978-3039` (no `execute_stacking_refit_oof`), `:3277-3280` (`residual_targets: None`), vs sequential `:1214-1221`, `:1298-1305`, `:2646-2652`. Also the sequential no-provider `execute_campaign_phase` (`:1062-1117`).

Parallel refuses nested FIT_CV with an explicit error (2865-2870, 2926-2931, 2988-2993) but not nested REFIT. For a plan with a declared nested stacking or residual campaign, Sequential first prepares the full-training OOF (`execute_stacking_refit_oof`) and passes `ResidualTargetSet` to the residual learner; Parallel does neither.

Failure scenario: `ParallelScheduler::execute_campaign_phase_with_data_provider(plan, ..., Phase::Refit)` for a residual-learner plan. The residual learner task is built with `residual_targets: None`. Depending on the controller this is a late "missing OOF" error, or, for a controller that falls back to the plain target, a model fitted on y instead of the base residual (silent wrong model, no leakage check involved).

Fix: in Parallel, reject `phase == Refit && !nested_stacking_campaign_plans(plan)?.is_empty()` with the same message as for FIT_CV (and the same for the sequential no-provider variant), or call `nested_residual_targets` and `execute_stacking_refit_oof` there too.

---

### A3-03  Medium  (confidence: medium-low)
**Stacking weight/selection scores flip from the bundle's attested scores to run-local scores as soon as any node emits a score**
`runtime/scheduler.rs:2573-2584` (sequential) and `:3209-3220` (parallel): `if ctx.score_collector.is_empty() { &ctx.stacking_weight_scores } else { &ctx.score_collector }`.

`stacking_weight_scores` is only populated by bundle replay (`:2331-2335`, `:3047-3051`) to give PREDICT/REFIT replay deterministic weights equal to the training run. The choice of source depends on whether an EARLIER node in the same replay emitted any regression report. In replay with an external-test cohort that carries targets, or any REFIT replay whose base models emit `y_true` for their Final/Test blocks, `score_collector` is non-empty by the time the meta node is reached, so `select_stacking_inputs` / `stacking_model_weights` (oof.rs:1683-1714, which keys on the first Validation report of the producer) receive run-local reports that contain no Validation CV scores. Result: weights silently become `None` (unweighted mean) or the selector errors, and the outcome depends on node order and on whether targets were supplied.

Fix: pick the source by intent, not emptiness: `if !ctx.stacking_weight_scores.is_empty() { &ctx.stacking_weight_scores } else { &ctx.score_collector }` (replay always owns its bundle scores).

Out-of-scope note (oof.rs:1690-1701): `stacking_model_weights` uses `scores.iter().find(...)` = first Validation report of the producer with any `fold_id` (a single fold, any variant, any port), not the cross-fold `avg`. Worth a look by the oof.rs auditor.

---

### A3-04  Medium  (confidence: low-medium)
**`collect_cross_fold_test_scores` is the only collector that does not honour `validation_scoring_fold_ids`**
`runtime/mod.rs:529-544` vs `:424-468` (validation) and `:547-592` (train).

Validation and train collectors drop blocks/records whose fold id is not in the report-grade outer fold set (nested stacking keeps inner-fold evidence in the shared context). The Test collector passes `self.prediction_store.blocks()` and `self.score_collector` unfiltered to `cross_fold_test_reports`, which averages every `Test` block with a fold id per producer (metrics.rs:1437-1445). In nested stacking, base nodes executed in inner scopes (`fold_set_override` = inner folds) still receive the `:test` companion view (`collect_input_handles` materializes `cv_test_cohort` for every FIT_CV scope, scheduler.rs:5149-5153, 5303-5348) and can emit Test blocks for the same external cohort. Those inner-fold estimators are then averaged into the `avg`/`w_avg` Test ensemble rows, and the per-fold validation reports used for weights (metrics.rs:1474-1480) include inner fold ids.

Failure scenario: nested stacking + external-test cohort: reported Test `avg` is a mean over outer AND inner fold estimators (inconsistent with the OOF side), or `scores.len() != blocks.len()` silently disables `w_avg`.

Fix: apply the same `validation_scoring_fold_ids` filter to Test blocks and target records (fold id must be in the allowed set) before calling `cross_fold_test_reports`.

---

### A3-05  Low  (confidence: medium)
**`FileArtifactPayloadStore::write_from_source` can truncate its own payload when source and output roots are the same directory spelled differently**
`runtime/artifact.rs:1012-1031`.

Guard is `if source_path != output_path { fs::copy(..) }`, a lexical `PathBuf` comparison. `std::fs::copy` documents that copying a file onto itself truncates it. Roots such as `./store` vs `store`, relative vs absolute, or a symlinked root compare unequal but resolve to the same inode. The source payload was just validated, then destroyed, then `Self::open` fails with a fingerprint mismatch: data loss of the only copy.

Fix: compare `fs::canonicalize(source_path)` with `fs::canonicalize(output_path)` (or the roots) before copying; or write to a temp file and rename.

---

### A3-06  Medium  (confidence: medium)
**`execute_hpo_campaign` converts every candidate-evaluation error into a failed trial, including infrastructure errors**
`runtime/scheduler.rs:476-492` (all `Err` from `execute_hpo_candidate_fit_cv`), `:497-511` (score collection), `:379-659`.

The `Err(error)` arm does not distinguish a model that failed on one candidate from systemic faults: unregistered controller, data provider I/O failure, plan/relation validation failure, `report_intermediate` returning a native-session error, `configure_global_oof_aggregation` failing, leakage-guard refusals (`OofValidation`, `RuntimeValidation` from the scheduler itself). All are reported to the native optimizer as `DAGML_CV_ERROR`/`retryable: false`, then `continue`. The loop burns the whole budget, the optimizer's history is polluted with failed trials that were not the candidate's fault, and the caller only sees the final, unrelated `"native HPO campaign has no completed native incumbent..."` (or, worse, a campaign that succeeds with a subset of candidates evaluated after a transient provider failure, which makes the incumbent non-reproducible).

Fix: only convert controller-originated `RuntimeValidation`/model errors; propagate leakage/OOF/plan/provider errors with `?`; at minimum attach `error.to_string()` of the first fatal failure to the final error.

---

### A3-07  Low  (confidence: medium)
**Exhausted proposal stream is handled with `break` and then rejected by the budget check, discarding completed work**
`runtime/scheduler.rs:426-429` (`let Some(proposal) = session.ask()? else { break; }`) and `:599-605` (`history_at_checkpoint != hpo.trial_budget_total` -> Err).

If the sampler legitimately stops early (finite grid, small integer range with dedup, pruner exhaustion), the loop exits with `history < budget`, then the whole call returns an error and every completed `RuntimeHpoCandidateEvaluation` is dropped. The two code paths contradict each other.

Fix: either treat `ask() == None` as terminal (accept `history <= budget` and set `trial_history_len` accordingly) or return a specific "sampler exhausted before budget" error carrying the partial result.

---

### A3-08  Low  (confidence: medium)
**Nested-stacking lineage comparison uses two different orderings**
`runtime/scheduler.rs:4870-4886`.

`inferred` comes from `actual.into_values()` on a `BTreeMap<(NodeId, FoldId), LineageId>`, i.e. ordered by (node, fold). `declared` is a `BTreeSet<LineageId>` collected into a `Vec`, ordered by lineage-id string. When a controller declares non-empty `input_lineage` (the only case that reaches the comparison), `declared_vec != inferred` is false-negative whenever the two orderings differ (e.g. nodes `model:a` and `model:a.b`, or ids not embedding node-then-fold). The non-nested twin (`attach_coordinator_input_lineage`, :4786-4799) correctly compares sorted sets.

Fix: compare as sets (`declared == inferred.iter().cloned().collect::<BTreeSet<_>>()`), then store a canonical order.

---

### A3-09  Low  (confidence: high)
**`InMemoryLineageRecorder::record` overwrites an existing record and only then reports the duplicate; `capture_refit_artifacts` is non-atomic**
`runtime/artifact.rs:1401-1413`; `:617-699`.

`self.records.insert(id, record).is_some()` replaces the stored record first, so the "duplicate lineage record id" error leaves the NEW record in place and the original lost. In the same spirit `InMemoryArtifactStore::capture_refit_artifacts` registers artifacts one by one with `?`, so a failure on artifact k leaves artifacts 0..k-1 registered in the caller-owned store (partial state after an error).

Fix: `if self.records.contains_key(..) { return Err }` before insert (`entry` API); build all records first, register after all validate.

---

### A3-10  Low  (confidence: low-medium)
**Run-context stores have no variant dimension; `execute_campaign_phase*` loops several variants into one context**
`runtime/prediction_store.rs:24-38` (`find` by node/partition/fold only), `runtime/scheduler.rs:1236-1263`, `:1320-1348`, `:1482-1532`, `:1974-2269`; `runtime/scoring.rs:912-949`.

Callers that loop `for variant in &plan.variants` with `ctx.variant_id == None` write all variants' prediction blocks, target records, probabilities and lineage into the same context. Subsequent lookups (`validate_fit_cv_oof_edge`, `nested_base_predictions_ready`, `prediction_feature_sources_ready`, `nested_residual_targets`) are variant-blind: variant 2's base scopes are considered "ready" from variant 1, or its OOF edges see duplicate blocks (surfacing as OOF/lineage errors rather than silent corruption, thanks to the variant filter in `attach_nested_stacking_input_lineage`). In `apply_independent_unit_scope_reports` (scoring.rs:912-927) the target record is chosen with `records.first()` without filtering by variant, and `report.variant_id = record.variant_id`, so with more than one variant the Group-level reports of later variants are tagged with the FIRST variant's id.

Design intent appears to be "one variant per context" (SELECT uses a fresh ctx per variant). That is not enforced.

Fix: either reject `plan.variants.len() > 1 && ctx.variant_id.is_none()` in the multi-variant campaign entry points, or carry `variant_id` in the store keys and filter records by `record.variant_id == lineage variant`.

---

### A3-11  Low  (confidence: medium)
**`ParallelScheduler::execute_campaign_phase_with_data_provider_and_artifact_store` never configures global OOF aggregation for FIT_CV**
`runtime/scheduler.rs:2978-3039` vs `:2923-2925` (data-provider variant) and `:1277-1279` (sequential artifact-store variant).

For plans whose global Target/Group aggregation comes from node `shape_plan.aggregation_policy` (not the campaign `grouping_key`, which `execute_phase_scope` configures itself, 3139-3148), `ctx.global_oof_aggregation` stays empty on this path, so `collect_cross_fold_validation_scores` silently skips the declared Group/Target OOF reports, unlike the other three campaign entry points.

Fix: call `ctx.configure_global_oof_aggregation(plan, data_provider)?` when `phase == FitCv`.

---

### A3-12  Low  (confidence: medium)
**HPO proposal trust and metric/direction consistency are not validated**
`runtime/scheduler.rs:454-456` (`candidate_plan.variants = vec![proposal.variant]` + `validate()` only); `runtime/task.rs:466-567` (`validate_for_plan`).

The scheduler accepts any `VariantPlan` from the session. Nothing checks that its `param_overrides` touch only `hpo.target_node_id` and only keys in `hpo.parameter_paths` values, so a buggy tuner could alter params of preprocessing/splitter nodes while the evidence is certified as an HPO of the target model. `validate_for_plan` also never checks `selection.metric`/`selection.direction` against `study.optimizer.metric`/`direction`; a mismatch only surfaces after the whole budget is spent (incumbent check at :630-639).

Fix: validate overrides against `parameter_paths`/`target_node_id` before running a candidate; check metric/direction consistency in `validate_for_plan`.

---

### A3-13  Low  (confidence: low-medium)
**Unchecked `param_overrides[0]` index in `execute_fold_hpo_fit_cv` (panic on plan-supplied data)**
`runtime/scheduler.rs:1152-1160`.

`choice.param_overrides[0].params = ...` indexes a vector of a plan-provided variant (`plan.variants` found by `state.selected_variant_id`). `VariantExecutionSpec::validate` rejects empty override *params* but not an empty `param_overrides` list, and `state.validate()` does not see the plan. A selected variant whose `native_methods_hpo` choice has no override (or overrides another node) panics instead of erroring; the override is also applied without checking `node_id == state.target_node_id`.

Fix: `let [override_spec] = choice.param_overrides.as_mut_slice() else { return Err(..) }` and check `override_spec.node_id == state.target_node_id`.

---

### A3-14  Low  (confidence: high for the omission, impact latent)
**Provider wrappers drop optional `RuntimeDataProvider` capabilities**
`runtime/dataview.rs:1123-1213` (`EnvelopeAttestedRuntimeDataProvider`), `terminal_prediction.rs:574-632` (`BorrowedRuntimeDataProvider`), `:649-723` (`TerminalCohortAttestedRuntimeDataProvider`), `data.rs:2169-2208` (`ExplicitPhaseDataProvider`).

None of them forwards `generated_views_enabled`, `generated_view_manifest`, `generated_view_manifest_on_failure` or `refit_sample_ids` (trait defaults: false/None); `BorrowedRuntimeDataProvider` additionally drops `cv_test_cohort`. A generated-view provider wrapped by `EnvelopeAttestedRuntimeDataProvider` is therefore seen as static; the scheduler's consistency check (`generated_view != generated_views_enabled`, scheduler.rs:5287-5294) then refuses every FIT_CV/REFIT view ("generated-view capability and receipt disagree"), and REFIT-only provider IDs vanish. Production py/capi paths currently work around this (the Python provider is outermost and answers `generated_views_enabled` itself), so this is a trap for embedders, not a live failure.

Fix: delegate the five defaulted methods in every wrapper.

---

### A3-15  Low  (confidence: medium)
**Fingerprint validators disagree on hex case**
`runtime/artifact.rs:1368-1375`, `data.rs:2402-2409`, `runtime/dataview.rs:238` accept uppercase hex (`is_ascii_hexdigit`); `runtime/task.rs:5-10`, `dataview.rs:358-361`, `artifact.rs:155` require lowercase.

An uppercase digest passes envelope/binding/artifact validation but is compared by string equality elsewhere (`DataEnvelopeKey`, `binding.relation_fingerprint != envelope.relation_fingerprint`, `artifact_sha256`), so one digest can have two non-equal identities, or fail later with a "mismatch" that is really a case difference (`validate_artifact_payload_file` alone uses `eq_ignore_ascii_case`).

Fix: one shared lowercase-only validator.

---

### A3-16  Low  (confidence: low)
**Aggregated-block scoring pairs by level only**
`runtime/scoring.rs:441-449`.

`result.regression_targets.iter().find(|targets| targets.level == block.level)` picks the first target block of that level for every aggregated prediction block, with no partition/unit-set match (the sample path uses `sample_targets_match_block`). A result carrying two controller-emitted aggregated blocks (e.g. Validation and Test) with different unit sets is scored against the wrong target block (error from `score_regression_aggregated_block`, or silently right when the unit sets happen to coincide).

Fix: match on `unit_ids` set equality like `sample_targets_match_block`.

---

### A3-17  Low  (confidence: low; Windows only)
**`validate_prediction_cache_file_name` accepts drive-prefixed names**
`runtime/prediction_store.rs:631-639`.

Rejects `.`, `..`, `/` and `\` but not `C:name`. On Windows `root.join("C:name")` replaces the root prefix, so a hostile manifest can read a payload outside the store root (the artifact side rejects this in `validate_relative_artifact_uri`, :1341-1357). No canonical-containment check exists for cache payload files either (unlike `validate_payload_path_stays_within_root`).

Fix: reuse the artifact URI validator or reject any `:`; canonicalize-and-contain like artifacts.

---

### A3-18  Low  (confidence: low)
**Host dataset handle leaks if anything fails after the source materializes**
`runtime/data_provider_bootstrap.rs:122-180`, `:278-295`.

`RuntimeDataProviderSource::materialize` hands out a host handle; between that call and `*captured = Some(materialized)` the code can still fail (`LineageId::new(...)?`, later `result.validate_for_task`, `normalize_*` in the scheduler, `results.len() != 1`). The handle is then dropped on the Rust side with no release callback (the trait has none), leaving the host-side dataset alive.

Fix: add a release hook to the source trait and call it on every post-materialize error path, or store the handle in `captured` before building the result and release it in `execute_data_provider` on `Err`.

---

### A3-19  Low  (confidence: medium)
**`InMemoryDataProvider::make_view` does not bind the view to the parent handle's node/input/phase**
`data.rs:2264-2322`.

It validates that `request.data_handle` exists and equals the stored record, but never compares `parent.node_id/input_name/phase/fold/variant` with the request. A view for node X can be cut from a data handle materialized for node Y (or another phase/fold), with `view` sample ids that were never part of that binding's envelope universe. Used as the reference/CLI/WASM provider, so tests and tools built on it cannot catch scheduler mix-ups.

Fix: compare `parent.node_id/input_name/request_id/relation_fingerprint` (and optionally phase/fold) with the request, and restrict `view.sample_ids` to the envelope's `coordinator_relations` universe when present.

---

### Items examined and dropped (not bugs / too speculative)
- `preload_replay_prediction_cache_store` aggregated branch (scheduler.rs:5443-5460) validates but does not append: the REFIT aggregated-edge path materializes from the cache store directly (oof.rs:2239-2260), so intentional.
- Parallel `invoke` vs sequential `invoke_with_data_provider_and_prediction_ports` (scheduler.rs:3348): explicit design comment.
- `primary_output_data_view` choosing the alphabetically first non-validation view: keys `data:X` always sort before `data:X:test`; only multi-input transforms could be affected (speculative).
- `equal_sample_influence_weights` picking the first non-empty training view of a multi-binding node (task.rs:2217-2234): speculative for multi-source models.
- O(n^2) `ExperimentalUnits::position` (task.rs:2338-2347): performance only.
- `collect_cross_fold_validation_scores` appending `avg`/`w_avg` without dedup if called twice (mod.rs:520-523): single call per context in all in-scope callers.

---

## A4 audit: dag-ml-core training / bundle / replay / archive / provenance

Scope: crates/dag-ml-core/src/{training.rs, training_runtime.rs, initial_refit.rs, bundle.rs, replay.rs, archive_workspace.rs, provenance.rs}
(non-test code read fully; cargo not run; nothing edited). Paths below are relative to /home/delete/nirs4all/dag-ml/crates/dag-ml-core/src.

Overall: no Critical/High defect found. None of the in-scope files write to disk, so there are no non-atomic-write or path-traversal-on-write issues here.
Path hygiene on artifact URIs is delegated to `validate_relative_artifact_uri` (runtime/prediction_store.rs:1325) and is sound (rejects absolute, drive, scheme, `..`, control chars).
Panic-on-untrusted-input scan found only internally-guarded `expect`/index uses (guards verified by caller). Findings are validation gaps and robustness issues.

Severity count: Critical 0, High 0, Medium 3, Low 10.

---

### A4-01  Medium  confidence: medium
**File:** training_runtime.rs:1311-1435, 1494-1632, 1683-1703 (also archive_workspace.rs:284-290)
**Title:** Package V3 `validate()` does not link its target request, plan topology or influence manifest, and never runs the replay-bundle projection

**Description:** `PortableRefitOutcomeV3::validate` checks
- target request self-validity (`target_training_request.validate()`),
- `request.data_identities == outcome.data_identities`,
- selected variant/params projection and recipe controllers (`validate_portable_refit_target_plan`),
- fingerprints.

It never checks that:
- the plan produced by `target_training_request.project()` equals `effective_plan` (this equality is enforced only on the build path, `build_portable_refit_package_v3` -> `derive_portable_full_refit_target_plan`, which needs the parent package);
- graph edges are pinned to the parent (the recipe pins node ids/params/controllers only);
- `training_influence` matches the target request/plan (`validate_for_projection` is not called; only fingerprint equality with provenance);
- `predictor_node_ids` equals the predictor closure of the output bindings (only "sorted + exists");
- artifact `params_fingerprint` / `training_loss_fingerprint` / `data_requirement_keys` match the plan, or that every stateful predictor node has an artifact. `PortableRefitExecutionBundleV3::validate` checks only controller ownership and payload hash; the V2 equivalents (`ExecutionBundle::validate_against_plan`, `closure_predict_replayable`) are not applied.

`PortableRefitPackageV3::from_json` -> `validate()` therefore accepts such a package. The same holds for `build_archive_v3_native_refit_payloads`, which only calls `package.validate()` and then packs N4MM/N4ME members (it re-checks params only for RolePipeline records).

**Failure scenario:** A V3 package whose artifact records carry a wrong `params_fingerprint` (or omit the artifact of a stateful predictor node), or whose `effective_plan` has an extra/rewired edge between same-parameter nodes, is accepted by `from_json` and archived by Archive V3. The refusal only appears later, inside `to_runtime_replay_bundle()` / `validate_against_plan` at replay time, or never for the topology and influence parts.

**Fix:** In `PortableRefitOutcomeV3::validate`, compare `target_training_request.project().plan` graph/campaign against `effective_plan`; call `training_influence.validate_for_projection` with the projection; check `predictor_node_ids == predictor_closure(output nodes)`. Make `PortableRefitPackageV3::validate` call `outcome.to_runtime_replay_bundle()` (or factor out its `validate_against_plan` part) so the archive builder inherits the V2-equivalent artifact/plan cross-checks.

---

### A4-02  Medium  confidence: medium-high
**File:** provenance.rs:181-249, 1008-1019, 1212-1311
**Title:** `validate_research_provenance_package_files` never re-derives PROV / RO-Crate from plan+bundle+lineage

**Description:** Validation (a) checks the RO-Crate `sha256`/`contentSize` entries against the bytes, (b) checks the PROV file has the four root keys, (c) validates plan/bundle/lineage/envelopes/manifests consistency. It does not compare the PROV JSON-LD or the RO-Crate graph with `build_prov_jsonld` / `build_ro_crate_metadata` over the validated inputs. The RO-Crate checksums are self-attested by a file inside the same package, so they provide no tamper evidence on their own.

**Failure scenario:** Someone edits `lineage.prov.jsonld` (drops `wasDerivedFrom` OOF dependencies, flips `dagml:unsafe_flags`, changes `dagml:oof_dependency`), recomputes the sha256 in `ro-crate-metadata.json`, and ships it. `validate_research_provenance_package_files` returns `Ok` and the package is advertised as validated. OpenLineage built from the package files is derived from plan/bundle (not the tampered PROV), so the two exports silently disagree.

**Fix:** Rebuild `build_research_provenance_export` from the parsed inputs and require byte/JSON equality with the packaged PROV JSON-LD; for the RO-Crate, recompute `annotate_ro_crate_package_files` output and compare.

---

### A4-03  Medium  confidence: medium
**File:** training_runtime.rs:3200-3306 (`validate_selected_rerun_reports`, `reports_match_rerun_tolerance`)
**Title:** Selected-variant FIT_CV rerun check uses a fixed absolute 1e-12 tolerance

**Description:** When the winner is not the singleton variant, `execute_training` reruns FIT_CV and requires every metric to match the SELECT-time report within `|a-b| <= 1.0e-12` absolute. The doc comment itself says native libraries "may differ by one rounding unit across a fresh process/context". One ulp of a metric of magnitude 1e6 (RMSE in large units) or 1e10 (MSE) is about 1e-10 / 2e-6, so the check is unsatisfiable for any non-bit-deterministic native kernel (threaded BLAS reductions) on large-scale targets.

**Failure scenario:** Multi-variant training on targets in large units with a multithreaded PLS/ridge kernel: SELECT picks a winner, rerun differs in the last bit, `execute_training` aborts with "selected variant FIT_CV rerun diverged from the reports that justified SELECT" even though nothing diverged.

**Fix:** Use a relative+absolute tolerance, e.g. `|a-b| <= 1e-12 + 1e-9*max(|a|,|b|)` (match whatever replay uses).

---

### A4-04  Low  confidence: high
**File:** initial_refit.rs:784-794
**Title:** Hydrated-handle release error is swallowed when PREDICT execution fails

**Description:**
```
let released = payload_store.release_hydrated();
let results = execution?;
released?;
```
If `execution` is `Err`, the function returns it and the release result (possibly `Err`) is dropped. `replay.rs::finish_bundle_payload_replay` handles the same situation correctly by combining both errors.

**Failure scenario:** PREDICT fails halfway and one controller cannot release a hydrated native model handle: the release failure (leaked native memory/handles) is never reported.

**Fix:** Reuse the `finish_bundle_payload_replay` pattern (merge both errors).

---

### A4-05  Low  confidence: medium
**File:** archive_workspace.rs:358-361 vs 977-985
**Title:** Archive V3 N4ME member path check is weaker than the N4MM one

**Description:** N4MM paths go through `safe_n4mm_path` (length <= 512, no backslash, no empty/`.`/`..` segments). N4ME paths only need `starts_with("methods/") && ends_with(".n4me")` plus the generic `validate_portable` (rejects `..`, absolute, scheme, control chars). `methods//x.n4me`, `methods/./x.n4me`, backslashes inside segments and unbounded length pass into `members` and the manifest `member_path`.

**Failure scenario:** `methods/./a.n4me` and `methods/a.n4me` are distinct BTreeMap keys but alias the same file once extracted by a writer/reader that normalizes (duplicate-member / inventory confusion, or a zip-writer rejecting an entry after the DAG-ML closure was already "validated").

**Fix:** Apply `safe_n4mm_path`-style segment/length/backslash checks (parameterized by extension) to N4ME paths.

---

### A4-06  Low  confidence: high (spec conformance)
**File:** provenance.rs:314-335, 1049-1090, 1092-1182
**Title:** OpenLineage facets omit the required `_producer`

**Description:** Every dag-ml facet (run `dagml_reproducibility`/`dagml_oof_safety`, job `dagml_plan`, dataset `dagml_contract`) has `_schemaURL` but no `_producer`. OpenLineage BaseFacet requires both. Also `validate_openlineage_event_time` only requires the string to contain a `T` (e.g. "Tuesday" passes), and `openlineage_run_id` is derived only from `plan.id` + `bundle.id`, so two executions/replays of the same bundle share one runId although `LineageRecord.run_id` exists (OL runId is per run).

**Failure scenario:** A strict OpenLineage backend/validator rejects the event; two runs of the same bundle collapse into one run in the lineage graph.

**Fix:** Add `"_producer": "https://github.com/GBeurier/dag-ml"` to each facet; parse RFC3339 properly; derive runId from the actual `run_id` when lineage is non-empty (or accept it as input).

---

### A4-07  Low  confidence: medium
**File:** replay.rs:204-322 (compare 681-689, 790-798)
**Title:** `PortableRefitReplayOutcomeV3::validate_against` does not constrain EXPLAIN outputs to the request

**Description:** V1/V2 `validate_against*` require, for EXPLAIN, that every emitted binding is in `request.output_binding_ids`. The V3 validator enforces equality only for PREDICT; for EXPLAIN any V3-package binding is accepted, and `request.output_binding_ids` entries are not checked against package bindings.

**Failure scenario:** An EXPLAIN replay outcome that emits bindings the caller never requested validates as bound to the request.

**Fix:** Add the same subset check for `Phase::Explain` and the "request references absent binding" check.

---

### A4-08  Low  confidence: medium
**File:** training_runtime.rs:4140-4288 (construction only), bundle.rs:1232-1239, 2019-2033; training.rs:1329-1464
**Title:** Prediction-cache `cache_namespace_fingerprints` are never recomputed/verified on load

**Description:** `CacheNamespace` is documented as binding every prediction-affecting field (params, data identity, fold, variant, seed). It is computed in `attach_oof_prediction_cache_namespaces` at training time. `ExecutionBundle::validate`, `TrainingOutcome::validate` and `PortablePredictorPackage::validate` only check hex format, uniqueness and count. Nothing recomputes them from plan + data identities + variant + seed.

**Failure scenario:** A cache record/payload from a different parameter set or cohort but with a consistent block fingerprint is accepted; the namespace guarantee is unenforced on the consumer side.

**Fix:** In outcome/package validation, recompute `oof_cache_namespace_fingerprints` for each cache record (inputs are all in the outcome) and compare.

---

### A4-09  Low  confidence: medium
**File:** training_runtime.rs:4869-4963, training.rs:2313-2459 vs replay.rs:1532-1541, 1597-1606
**Title:** Calibration replay target-bound identities are required at attach time but not on validation

**Description:** `calibrate_attached_training_replay` / `derive_attached_conformal_calibration_context` refuse a replay whose `input_data_identities` lack `target_content_fingerprint`. `TrainingOutcome::validate` and `PortablePredictorPackage::validate` accept any V3 replay (target-free identities allowed) as calibration evidence.

**Failure scenario:** A package carries calibration quantiles whose replay identities do not attest the calibration cohort's targets; consumers cannot detect that the truth used for the quantiles is unbound.

**Fix:** In both validators require `replay.input_data_identities.iter().all(|i| i.target_content_fingerprint.is_some())`.

---

### A4-10  Low  confidence: medium
**File:** bundle.rs:2105-2267 (and 3554-3583)
**Title:** `ExecutionBundle::validate_against_plan` does not require a requirement for every `requires_oof` plan edge

**Description:** Requirements are checked for membership in the plan, never the reverse. The archive module admits this (archive_workspace.rs:872-879) and patches only its own synthesized-empty-cache path. `ReplayPhaseRequest::validate_for_bundle_internal` decides "REFIT has no OOF dependency" from `bundle.prediction_requirements.is_empty()`.

**Failure scenario:** A re-signed bundle with `prediction_requirements`/`prediction_caches` stripped for a stacking graph passes `validate_against_plan`; a REFIT replay request validates as having no OOF dependency (the scheduler later refuses for lack of OOF, so this fails late rather than leaking).

**Fix:** In `validate_against_plan`, require each plan edge with `requires_oof` (whose consumer is refit by the bundle) to have a matching requirement.

---

### A4-11  Low  confidence: low
**File:** replay.rs:1126, 1322-1323, 1454
**Title:** Replay RunContext seed/resource limits differ from training

**Description:** Attached/loaded/V3 replays construct `RunContext::new(run_id, None)`; training uses `Some(options.seed)` and `options.resources`. `execute_initial_full_refit_prediction` correctly reuses `package.execution_root_seed`/`resource_limits`. Node tasks fall back to `plan.campaign.root_seed`, but `data_view_identity` (runtime/dataview.rs:1711-1738) derives the provider materialize seed from `ctx.root_seed`, which is `None` on replay.

**Failure scenario:** A data provider with a seeded view transform produces different PREDICT inputs than the training-time seed would, so replay output can diverge from the original run.

**Fix:** Seed the replay context from `plan.campaign.root_seed`.

---

### A4-12  Low  confidence: low-medium
**File:** provenance.rs:636-647, 689-731, 1392-1400
**Title:** PROV record keys can collide and artifact generation is attributed to the last lineage record

**Description:** Map keys are built by `format!("dagml:used:{}:{}", record_id, input_id)`, `dagml:derived:{artifact}:data:{key}` etc. Ids may contain `:`, so distinct (a,b) pairs can yield the same key and `BTreeMap::insert` silently drops one edge. `lineage_artifact_index` keeps the last record referencing an artifact; replay lineage carries `artifact_refs` for consumed artifacts, so `wasGeneratedBy` can name a later PREDICT activity instead of the REFIT producer.

**Failure scenario:** A provenance graph silently loses a `used`/`wasDerivedFrom` edge, or attributes a model to the wrong activity.

**Fix:** Use an unambiguous separator/encoding (length-prefix or hash) for composite keys; prefer the REFIT-phase record when indexing artifact generation.

---

### A4-13  Low  confidence: low
**File:** archive_workspace.rs:186-194 (and methods_multimodal.rs parse_payload)
**Title:** 128 MiB cap is applied to bytes, but TCV1 parsing builds a full typed tree first

**Description:** `states: Vec<Vec<u8>>` is stored as JSON number arrays; a 128 MiB payload of tiny numbers yields on the order of 10^8 typed nodes in `parse_typed_json` before any structural check. The provenance package serializer (`to_vec_pretty`) likewise expands `raw_artifact_payloads` (`Vec<u8>`) to one line per byte (~7-10x).

**Failure scenario:** Memory blow-up (several GB) when validating or packaging a maximal untrusted RolePipeline payload.

**Fix:** Reject by element-count bound or stream-validate; store payloads base64 or detached in the provenance package.

---

## A5 audit: dag-ml-core metrics / aggregation / criteria / conformal / HPO / methods_*

Scope: `crates/dag-ml-core/src/{hpo,metrics,aggregation,criteria,conformal,conformal_runtime,metric_provider,methods_*,n4m_roles,python_torch_profile}.rs` and `runtime/{host_hpo,host_hpo_topology,host_hpo_structural}.rs`. Read-only; no cargo run. Test modules skimmed or skipped.

Result: 0 Critical, 0 High, 4 Medium, 11 Low. Clean areas: conformal rank arithmetic (`ceil((n+1)*cov)` is exact, k-th smallest residual is correct, small-sample policy is correct); R2, RMSE, MAE, MSE, balanced accuracy and weighted F1 match sklearn semantics (including constant-target R2); `select_candidate` direction handling; HPO checkpoint sealing and identity checks.

---

### A5-01 | Medium | confidence: medium
**File:** `metrics.rs:1313-1323` (also `methods_estimator.rs:1020-1049`)
**Title:** Resampled-CV OOF average averages numeric class labels when no Validation probabilities are attested.

**Description:**
- `cross_fold_validation_reports_with_probabilities` always reduces Validation blocks with `reduce_predictions_across_folds`, a plain (weighted) numeric mean.
- Classification-aware reduction (`reduce_classification_folds`, "a numeric class label is never averaged") is used only for the Test and TrainPool ensembles.
- The native n4m estimator controller attests class probabilities only for `Train`/`TrainPool` surfaces. For `Validation` it emits none, so `average_validation_probabilities` returns `None`.
- Under `FoldPartitionMode::Resampled` (ShuffleSplit, repeated CV) a sample is validated in several folds, so its predicted class ids are averaged arithmetically.

**Scenario:** classes {0,2}; a sample is predicted 0 in one fold and 2 in another. The OOF "label" becomes 1.0, a class that may not be the truth or a prediction. `accuracy`, `balanced_accuracy` and `f1` of the `avg` report (`cv_best_score`) are then wrong, and so is selection by them.

**Fix:** In the cross-fold validation path, detect classification (metric kind or label-valued targets) and use the soft or vote reduction as for Test. Alternatively emit Validation probability blocks from the classifier controller and make `average_validation_probabilities` mandatory for classification metrics.

### A5-02 | Medium | confidence: medium
**File:** `runtime/host_hpo.rs:273-321, 1961-1975, 2465-2478` (no check anywhere in the file)
**Title:** `HostHpoSearchRequest.direction` is never validated against `metric.objective()`.

**Description:**
- The request carries both `metric: RegressionMetricKind` and `direction: MetricObjective`.
- `validate_parameter_bindings`, `prepare_host_hpo_checkpoint` and the C ABI / CLI entry points do not check `request.direction == request.metric.objective()`.
- `select_candidate` and `reduce_host_hpo_fold_scores` use `request.direction` blindly.
- The training path does enforce this (`training_runtime.rs:1164`, `3109`; `resolve_for_prediction_kind`).

**Scenario:** `{"metric":"r2","direction":"minimize"}` (or `rmse` with `maximize`) is accepted and silently selects the worst trial. With `fold_score_reduction=Best` it also takes the min/max fold the wrong way.

**Fix:** Reject `direction != metric.objective()` in `validate_parameter_bindings` (or a new `validate()` called from every entry point and from `prepare_host_hpo_checkpoint`).

### A5-03 | Medium | confidence: low (may be intended)
**File:** `runtime/host_hpo.rs:2866-2870`
**Title:** `HostHpoFoldReduction::RobustBest` is identical to `Best` (optimistic best-single-fold objective).

**Description:**
- Both variants reduce with `min`/`max` over fold scores.
- `Best` scores a trial by its luckiest fold, which is selection-biased.
- A variant named "Robust" presumably should be median, trimmed mean or worst-fold. It gets the same optimistic value.
- The two variants also yield identical `objective_fold_scores`/fingerprints apart from the enum label, so a caller choosing "robust" gets no robustness.

**Fix:** Implement the robust reduction (for example worst-fold or median according to `direction`), or remove the variant.

### A5-04 | Medium | confidence: medium
**File:** `hpo.rs:306-313, 356-361, 393`
**Title:** Unbounded `SortedTuple.length` allows multi-billion element allocation during validation.

**Description:**
- `HpoParameter::validate` checks only `length > 0` for `SortedTuple`.
- `HpoSearchSpace::validate` (and `fingerprint`) then calls `output_names()`, which builds `(0..length).map(format!(..))`.
- `length` is `i32`, up to 2^31-1, and the study config is deserialized from caller-supplied JSON (`methods_hpo_operation.study`, `MethodsHpoStudyConfig`).

**Scenario:** `{"kind":"sorted_tuple","length":2147483647,...}` makes validation allocate about 2 billion `String`s (OOM or hang) before any native bound applies.

**Fix:** Cap `length` (for example 1..=4096) in `validate`, and compute duplicate detection without materializing all names.

### A5-05 | Low | confidence: high
**File:** `methods_multimodal.rs:339-341`
**Title:** Panic (`IndexMut` on non-object JSON) from a crafted graph operator.

**Description:**
- In `validate_methods_multimodal_pipeline_recipe`, `expected` is built from the signed graph operator's `recipe` and is validated only later, at line 350.
- For effective param `transformers__image__n_components`, the code runs `expected.encoders.get_mut("image")?["n_components"] = ...`.
- `serde_json::Value`'s `IndexMut<&str>` panics for string, number, bool and array values.
- The encoder is an arbitrary `Value`, so an operator whose image encoder is not an object or null aborts the process instead of returning a refusal.

**Fix:** Check `as_object_mut()` and return a refusal, or run `validate_recipe` before the param overrides.

### A5-06 | Low | confidence: high
**File:** `methods_classification.rs:80, 82`
**Title:** Same `IndexMut` panic in `methods_classifier_recipe`.

**Description:**
- `recipe["model"]["params"][..] = value` (raw) and `recipe[0]["params"][name] = value` (meta) mutate the operator-supplied `recipe`.
- For the raw classifier, `operator["recipe"]` is only checked for presence of the key, not for being an object, before any effective param (`model__n_components`, `model__max_iter`) is applied.
- A non-object `recipe`, or a non-object `recipe["model"]`, panics.

**Fix:** Validate that `recipe`, `recipe["model"]` and `recipe["model"]["params"]` are objects before writing.

### A5-07 | Low | confidence: medium
**File:** `hpo.rs:314-365`
**Title:** Log-scale parameters are not validated for positive bounds.

**Description:**
- `Int {log: true}` and `Float {log: true}` accept `low <= 0`, and no `high >= low` or step interplay is checked.
- `Ordinal`/`Categorical` accept duplicates, and `Float` categories accept NaN.
- Validation is deferred to libn4m, which only exists behind the `methods-optimizer` feature.
- The `HpoSearchSpace::validate` fingerprint/contract is therefore weaker than documented on default builds.

**Fix:** Reject `log && low <= 0`, duplicate ordinal/categorical values and non-finite float categories.

### A5-08 | Low | confidence: medium
**File:** `runtime/host_hpo.rs:259-271, 937-957, 1005-1026`
**Title:** `phase_index` uses unchecked `u32` addition; worker paths call it without validating the budget sum.

**Description:**
- `evaluate_host_hpo_worker_task` and `evaluate_host_hpo_worker_fold` call `request.phase_index(..)` on a request that was deserialized in the worker.
- Only `prepare_host_hpo_worker_window` and the search entry points validate that `phase_trial_budgets` sums without overflow.
- `end += budget` panics in debug and wraps in release for crafted budgets (for example `[4294967295, 2]`), which can alias phase indices.

**Fix:** Use `checked_add`, or validate the request in the worker entry points.

### A5-09 | Low | confidence: low (design)
**File:** `runtime/host_hpo.rs:2315-2336`
**Title:** Parallel host HPO with progressive pruning feeds `report_intermediate` in thread-arrival order, so results are nondeterministic.

**Description:**
- Intermediate fold scores from concurrently running trials reach the single proposal source in whatever order the threads finish folds.
- A stateful pruner (median, ASHA, hyperband) therefore makes decisions that depend on scheduling.
- `validate_parallel_execution` forbids pruning only for the typed Methods parallel profile. Generic host parallelism plus `progressive_pruning` is allowed.
- Same seed and budget can prune different trials across runs.

**Fix:** Either forbid `progressive_pruning` with workers > 1, or serialize intermediates into a deterministic (step, trial_index) order.

### A5-10 | Low | confidence: medium
**File:** `metrics.rs:1762-1788` (consumers `1493-1494`, `1591-1592`, `aggregation.rs:1417`)
**Title:** `validation_score_weights` can assign exactly zero weight, or a dominating weight, to a fold.

**Description:**
- Maximize with `min < 0` uses `score - min`, so the worst fold gets weight exactly 0. R2 or accuracy of exactly 0.0 with `min >= 0` also gives 0.
- Minimize with `min <= 0` uses `score - min + 1e-8`, so the best fold gets about 1e8 times the weight of the others and the ensemble collapses onto one fold.
- `reduce_predictions_across_folds` then errors with "a sample had zero total weight" for any sample predicted only by that fold. This happens when a TrainPool block covers differing id subsets (the unit test uses overlapping subsets), and aborts the whole report.

**Fix:** Use a strictly positive floor on weights, for example `max(w, eps)`, or drop zero-weight folds before the zero-total check.

### A5-11 | Low | confidence: high
**File:** `methods_estimator.rs:271-277, 665-673`
**Title:** The shared feature store is never released (unbounded memory growth).

**Description:**
- `emit_features` inserts an `Arc<FeatureSet>` (full numeric matrices) per transform or filter invocation into `SharedState.features`.
- Nothing ever removes entries; the only `remove` calls are for `exported` and `hydrated`.
- Across folds, variants and HPO trials sharing one registration, memory grows monotonically.

**Fix:** Remove entries when the consuming task completes (or on scheduler handle release), or scope the store per run context.

### A5-12 | Low | confidence: medium
**File:** `hpo.rs:1435-1447, 1358-1397`
**Title:** Native `ask()` is performed before proposal validation, so a rejected proposal leaves a RUNNING trial.

**Description:**
- The v1 path calls `self.study.ask()` and only then verifies the study space is exactly `n_components=1..3`; the portable-profile path checks the typed axis values after `ask()` too.
- On rejection the native optimizer keeps a RUNNING trial that is never `tell`ed.
- If the caller then retries, checkpoints or reports the history, the trial ledger contains a non-terminal trial. Terminal snapshotting later errors with "non-terminal trial".

**Fix:** Validate the study and space once at session creation, before any `ask`; on a post-ask failure, `tell` it `Failed`.

### A5-13 | Low | confidence: medium
**File:** `hpo.rs:1412, 1479`; tie-break `selection.rs:493`
**Title:** Tie-break by lexicographic candidate id with unpadded trial ids picks the wrong "first" trial.

**Description:**
- Native HPO variants are named `hpo:trial:{id}` (unpadded), unlike host HPO's `host_hpo:trial:{:010}`.
- `compare_scores` breaks score ties by `candidate_id` string order, so "hpo:trial:10" sorts before "hpo:trial:2".
- With discrete or tied metrics (categorical axes, small integer ranges, accuracy), SELECT picks trial 10 over trial 2, and may disagree with libn4m `best()`, which usually favours the earliest trial.
- The fold-HPO validator compares scores, not ids, so no hard failure, but the choice is non-intuitive.

**Fix:** Zero-pad trial ids in variant ids, or tie-break by numeric trial id.

### A5-14 | Low | confidence: medium
**File:** `conformal_runtime.rs:832-834` vs `979-983`
**Title:** Calibration accepts a point block with empty `target_names`, but `apply` then rejects the same kind of block.

**Description:**
- `calibrate_with_truth` accepts `predictions.target_names.is_empty()`, because the check is `!is_empty() && != target_names`.
- `ConformalCalibration::apply` requires `predictions.target_names == self.target_names` exactly.
- A calibration produced from an unnamed prediction block can never be applied to a prediction block with the same (empty) naming.

**Fix:** Make the two checks symmetric: require non-empty equal names at calibration, or compare only when the block has names in `apply`.

### A5-15 | Low | confidence: low
**File:** `metrics.rs:1017, 1033-1034, 1073-1075`
**Title:** Inconsistent label matching between metrics (`|pred-true|<0.5` vs `round()`).

**Description:**
- Accuracy and balanced accuracy count a hit when `|pred - true| < 0.5`; weighted F1 and the balanced-accuracy class key use `round()`.
- At an exact .5 offset or for non-integer class codes the metrics disagree (for example true=1.0, pred=0.5: accuracy miss, F1 hit).
- Non-integer outputs are only expected from misconfigured hosts, hence low severity.

**Fix:** Pick one convention (round both sides, or reject non-integral label inputs).

---

### Notes (verified, not bugs)
- `weighted_f1_for_target`, `balanced_accuracy_for_target`, `r2_for_target`: formulas correct; empty inputs are rejected earlier by `validate_shape`.
- `MetricReduction::WeightedMean` zips values with per-unit weights, but `MetricSpec::validate` forces `PerUnit` decomposition for it, so the length mismatch cannot occur.
- Conformal `finite_sample_conformal_rank`: the `scale > 38 -> rank 1` shortcut is valid (`scaled < 1e37`), and `rank > n` is handled per policy. The Winkler score `width + (2/alpha)*miss` is correct.
- Parallel host HPO: no deadlock found (all senders drop on worker panic; `events_rx.recv` Err breaks the loop). Terminalization is ordered by trial index, so the ordering race is contained.
- `select_candidate` direction handling (min/max) is correct.

---

## A6 - dag-ml C ABI audit (crates/dag-ml-capi)

Scope read in full: `src/lib.rs` (non-test part, lines 1-6372), `src/host_hpo.rs`, `src/initial_refit.rs`,
`src/local_implementation.rs`, `include/dag_ml.h`, `docs/ABI.md`. Tests only skimmed. No cargo run (read-only).

Checked and found sound: header vs Rust struct layouts and all 101 exported function names (header == Rust set);
Rust-allocated output protocol (`write_owned_json`, tensor writers, `*_free` with `Vec::from_raw_parts` using the
stored len/capacity); `CString` NUL sanitisation in `write_error_string`; checked `rows*cols`; UTF-8 handling via
`from_utf8`; escrow logic of `build_training_controller_registry` (exactly-once destroy on all paths); TCV1 nesting
bounded (`MAX_NESTING_DEPTH`); thread-local last-error has no reentrancy hazard; mutex poisoning handled in Drop impls.

Counts: High 0, Medium 3, Low 8 (details below).

---

### A6-01 - Medium - confidence: medium-high
**`dagml_replay_execute_json` and `dagml_initial_full_refit_*` can double-destroy an aliased owned `user_data`**
File: `crates/dag-ml-capi/src/lib.rs:5166-5191` (`build_controller_registry`), used at `lib.rs:3614`,
`initial_refit.rs:139`, `initial_refit.rs:276`; `CAbiRuntimeController::drop` at `lib.rs:5475-5493`.

Description: the native training path wraps bindings in `OwningControllerEscrow` and explicitly refuses any
`user_data` address shared by an owning vtable ("could use-after-free or double-destroy", `lib.rs:4338-4343`).
`build_controller_registry` has no such guard, yet owned ABI v3 vtables (`destroy != NULL`) are accepted there, and
`CAbiRuntimeController::drop` calls `destroy(user_data)` unconditionally for v3. The same applies across objects:
an owned artifact store (v2) or prediction cache (v2) sharing the pointer with an owned controller.

Failure scenario: a host binding registers N controller ids over one adapter object (common) and sets
`abi_version = 3` + `destroy`. Dropping the registry calls `destroy` N times on the same pointer -> double free /
use-after-free in the host (R/Python/MATLAB finalizer).

Fix: reuse the escrow/alias check in `build_controller_registry` (or make it take the same transactional path as
training), or refuse `abi_version >= 3` with `destroy` in the non-training registries and document it.

### A6-02 - Medium - confidence: high
**Ownership of owned (v2/v3) vtables is inconsistent on failure in replay / initial-refit entry points (leaks + undefined contract)**
File: `lib.rs:3552-3618` (`dagml_replay_execute_json_impl`), `lib.rs:5166-5191`, `initial_refit.rs:75-146, 209-283`.
Header comment for `dagml_replay_execute_json` (`dag_ml.h`, ~line 667) states nothing about consumption.

Description: `CAbiRuntimeArtifactStore`/`CAbiRuntimePredictionCacheStore`/`CAbiRuntimeController` call `destroy` in
their `Drop`, but wrappers are only built after plan/bundle/request/envelope parsing succeeded. Resulting behaviour
depends on where the call fails:
- parse failure at lib.rs:3552-3597 -> nothing destroyed (caller must still free everything);
- `CAbiRuntimePredictionCacheStore::new` failure (3609) -> artifact store already built and dropped, so its owned
  `user_data` IS destroyed, but the prediction cache and every controller are NOT;
- in `build_controller_registry`, `CAbiRuntimeController::new` failing (missing `invoke`/`release_bytes`) or a later
  binding failing -> earlier wrappers destroy their `user_data`, the failing and the not-yet-processed owning
  bindings are leaked (never destroyed).
The caller cannot know which owned objects were consumed; either it leaks or it double-destroys.

Fix: adopt the all-or-nothing escrow contract of `dagml_training_execute` for every entry point that accepts
v2/v3 owned vtables (consume everything exactly once on every return), and document it in `dag_ml.h`/`ABI.md`; or
reject owned vtables in these functions as `dagml_execution_plan_execute_phase_json` does.

### A6-03 - Medium - confidence: medium (needs a panic reachable from core)
**~90 of ~100 `extern "C"` entry points have no `catch_unwind`; a panic aborts the host process**
File: `lib.rs` (only 8 `catch_unwind` sites, e.g. 1450, 1544, 3479, 3736, 3917, 3958, 4037, 4151; plus all of
`host_hpo.rs`, `initial_refit.rs` and the `local_implementation.rs` boundaries). Unprotected examples: every
`*_validate_json`, `dagml_pipeline_dsl_*`, `dagml_execution_plan_schedule_json`, `dagml_select_*`,
`dagml_score_regression_*`, `dagml_prediction_cache_payload_*`, `dagml_research_provenance_export_json`,
`dagml_openlineage_run_event_json`, `dagml_mock_replay_execute_json`, `dagml_archive_v*_payloads_json`,
`dagml_init_tracing`, `dagml_last_error_json`.

Description: the status code `DAG_ML_STATUS_PANIC` (255) and the replay/training entry points show the intent that
panics are converted to a status. With Rust >= 1.81 (`rust-version = 1.85`) a panic escaping `extern "C"` is a
process abort (not UB, but not recoverable). These entry points feed untrusted JSON into large core parsers/planners
(DSL compiler, planner, provenance builders, scoring); any `unwrap`/index/arithmetic-overflow panic in core on
adversarial or just unusual input kills an R/MATLAB/Python session instead of returning 255. No
`panic = "abort"` / `"unwind"` profile is configured in the workspace, so default unwind semantics apply.

Fix: route all exports through one small `guard(error_out, out, || ...)` helper like `registration_boundary` in
`local_implementation.rs`.

### A6-04 - Low - confidence: high
**`dagml_version()` is hard-coded to 0.1.0 while the crate is 0.3.34**
File: `lib.rs:537-543`.
Description: returns `{0,1,0}` regardless of `CARGO_PKG_VERSION` (workspace version 0.3.34). Any host that gates
compatibility on this function (or logs it for support) is told the wrong version.
Fix: derive from `env!("CARGO_PKG_VERSION_MAJOR/MINOR/PATCH")`.

### A6-05 - Low/Medium - confidence: medium
**Every intermediate result handle and data/view handle is retained until the wrapper is dropped (unbounded host-side retention)**
File: `lib.rs:5379-5405` (`track_result_handles`), `5275-5282`, `5325-5332` (data provider `live_handles`),
`5709-5716`, `5882-5889`.
Description: handles are only released in `Drop` (`5475`, `5234`). For `dagml_training_execute` the controller
registry lives inside `DagMlTrainingResult`, so all node outputs of all folds/variants (including large
transformed matrices held by the host behind the handle) stay alive until `dagml_training_result_free`. Data
view handles are held for the lifetime of the provider. Also `handles.contains()` makes tracking O(n^2).
Scenario: wide variant x fold campaign with preprocessing nodes -> host memory grows with campaign size.
Fix: release intermediate handles when the scheduler no longer needs them (or expose a release hook), keep only
model/artifact handles referenced by the outcome; use a `BTreeSet`.

### A6-06 - Low - confidence: medium
**Possible double `release` of an already-tracked handle when a result fails validation**
File: `lib.rs:5500-5511`, `5407-5423`.
Description: on `validate_for_task` failure, `release_result_handles_immediately` releases every handle in the result
owned by this controller. It does not check `live_handles`. If a controller echoes a previously tracked handle in
`outputs`/`artifact_handles` (e.g. passes through a fitted model handle in PREDICT) and the result is rejected for
another reason, that handle is released now and again in `Drop` -> host double free. Needs a controller that reuses
handle ids across results, so low confidence.
Fix: skip handles already in `live_handles` (or remove them from the list when released).

### A6-07 - Low - confidence: high
**Inconsistent "omitted optional input" protocol**
File: `lib.rs:4540-4553` (`parse_optional_json_ptr`: omitted only if ptr==NULL **and** len==0; non-NULL + len 0 -> parse
error), vs `lib.rs:4206` (`parse_optional_strict_json_view`: NULL **or** len==0 -> default), vs
`host_hpo.rs:701, 800` (len==0 -> None regardless of pointer).
Description: hosts that pass an empty buffer with a non-NULL pointer (R `raw(0)`, MATLAB empty arrays, C++
`vector::data()` on an empty vector commonly non-NULL) get a JSON parse error from the provenance/OpenLineage
exports but "omitted" from training/HPO. `docs/ABI.md:83` documents only the NULL+0 form.
Fix: pick one rule (NULL or len==0) everywhere and document it.

### A6-08 - Low - confidence: medium
**Host-produced handles/buffers leaked when the host reports an error status or returns unparsable output**
File: `lib.rs:5261-5282, 5311-5332` (`materialize`/`make_view`: status checked, handle never released if the host
wrote `out_handle != 0` and returned non-OK); `5687-5708`; `5868-5889`; `5425-5472` (`invoke_json_task`: handles
inside a result that fails UTF-8/TCV1/serde parsing can never be identified or released); `host_hpo.rs:346-361`
(`create_candidate` non-OK status with non-NULL `*out_candidate_state` is not destroyed).
Description: the callee-allocates / caller-releases rule is only applied to the byte buffers, not to handles written
alongside a failing status. Contract for "out param valid only on OK" is not documented.
Fix: document that outputs are only valid on OK, or release non-zero out handles on non-OK.

### A6-09 - Low - confidence: medium
**Prediction-cache `load_blocks` JSON is parsed with plain serde, unlike controller results**
File: `lib.rs:5833-5852` vs `5461-5471`.
Description: controller results go through `parse_typed_json` (duplicate/NFC-colliding keys rejected) and
`deserialize_external_contract` (no positional-sequence structs). Host prediction-cache blocks (OOF predictions that
feed stacking / refit leakage decisions) use `serde_json::from_slice`, so duplicate `BTreeMap` keys silently
last-win and tuple-style arrays are accepted. Same trust boundary, weaker parsing.
Fix: use the same strict path as `invoke_json_task`.

### A6-10 - Low - confidence: high
**Owning vtables with NULL `user_data` are rejected as "aliases" in training; `destroy(NULL)` otherwise called**
File: `lib.rs:4257-4286, 4338`.
Description: two owning bindings that both use `user_data == NULL` (stateless host, distinct function tables) are
treated as aliasing and the whole call is refused; a single owning NULL `user_data` gets `destroy(NULL)`.
Fix: ignore NULL addresses in the alias map (and skip `destroy` for NULL, as `local_implementation.rs:77` does for
retain/release).

### A6-11 - Low - confidence: high
**`ABI.md` / `dag_ml.h` are stale or incomplete for part of the surface**
File: `docs/ABI.md:188-205` and 14-160; `dag_ml.h` near `dagml_replay_execute_json`.
Description: ownership table has no entries for `DagMlF32Tensor`/`DagMlF32ColumnarTensor`
(`dagml_f32_*_free`), the opaque `DagMlTrainingResult` / `DagMlLocalImplementationRegistry` (free functions), the
thread-local last-error accessors (`dagml_last_error_json/_code`, errno-like, per-thread, not cleared on success) or
`dagml_init_tracing`; the f32 tensor exports, `dagml_initial_full_refit_*`, `dagml_select_stacking_*`,
`dagml_align_named_source_rows_json` are not described. The replay entry point has no statement about vtable
ownership (see A6-02), and the Arrow vtable callbacks (`predict`, `view_identity`, `target_arrow`,
`feature_arrow`, `clone_with`, `describe`, `fit`) are described as part of the ABI but are never invoked by this
crate (only `materialize`/`make_view`/`invoke` are) - hosts may implement them for nothing.
Fix: sync docs with the header; mark unused callbacks reserved.

### A6-12 - Low - confidence: low
**`phase_index as i32` / error-detail loss in host callbacks**
File: `host_hpo.rs:143` (`u32 -> i32` wraps to a negative value, which hosts read as "-1 = unphased"; unreachable for
realistic phase counts), `lib.rs:5603-5624, 5739-5760, 5918-5939, 6018-6039` and `host_hpo.rs:114` (host error text
discarded; only `local_implementation.rs:108-123` surfaces callback detail).
Fix: `i32::try_from(index).map_err(...)`; surface the host's error bytes consistently.

---

## A7 audit: dag-ml bindings (py / wasm / arrow / results)

Scope read in full: crates/dag-ml-py/src/{lib,in_process,training,local_implementation,methods_role_pipeline}.rs (non-test code; test modules skimmed), crates/dag-ml-wasm/src/*.rs, crates/dag-ml-arrow/src/lib.rs, crates/dag-ml-results/src/lib.rs, crates/dag-ml/src/lib.rs (3-line re-export). `crates/dag-ml-data-arrow` contains no source files (empty `src/`), nothing to audit. Read-only; no cargo run.

Counts: Critical 0, High 1, Medium 6, Low 8.

---

### A7-01 High (confidence: medium) - TrainingResult mutex held across Python callbacks while GIL is not detached
- File: crates/dag-ml-py/src/training.rs:588-643 (`replay_json`), 715-721 (`lock_resources`), 475-504 (getters), 2126 (`execute_attached_methods_terminal_prediction`)
- Description: `TrainingResult.resources` is a `std::sync::Mutex` documented as making the pyclass "safely shareable across Python threads". `replay_json` takes `self.lock_resources()` (blocking `lock()`), keeps the guard for the whole `execute_attached_training_replay` call, and that call invokes Python op/artifact callbacks (`PyOperatorController`). The method never detaches the GIL (`_py` unused). Every other method (`is_attached`, `process_local_*_count`, `detach`, a second `replay_json`) also does a blocking `lock()` while holding the GIL.
- Failure scenario: thread A calls `result.replay_json(...)`; it holds the mutex, and inside a callback Python releases the GIL (any numpy/BLAS call, I/O, `time.sleep`, lock wait). Thread B (holding the GIL) calls `result.is_attached` or `result.replay_json`, blocks forever in `Mutex::lock` while still holding the GIL. Thread A can never re-acquire the GIL: process deadlock. Same-thread variant: a callback that touches `result.is_attached`/`detach()` re-locks the same non-reentrant mutex (deadlock/panic).
- Fix: use `try_lock` and raise a `RuntimeError("TrainingResult busy")`, or take the mutex only inside `py.detach(...)` (i.e. `py.detach(|| self.resources.lock())`) and never hold it across callbacks; alternatively clone `Arc`s of controllers/artifact store out of the lock before executing.

### A7-02 Medium (confidence: medium-high) - score_set_hash canonicalization diverges from the Python producer for floats
- File: crates/dag-ml-results/src/lib.rs:873-914 (`score_set_hash`, `canonical_json`); producer: nirs4all/nirs4all/pipeline/dagml/native_results.py:119-126
- Description: Rust canonical_json prints numbers with `serde_json::Number::to_string` (ryu: `1e-5`, `1e16`, `1e20`), the Python producer hashes `json.dumps(..., sort_keys=True, separators=(",",":"), ensure_ascii=False)` (`1e-05`, `1e+16`, `1e+20`). The only unit test covers `0.42`, `1.0`, `-3.5`. The reader re-serializes the parsed `score_set.json` before hashing, so the text form of the producer is lost.
- Failure scenario: a ScoreSet containing a metric below 1e-4 or at/above 1e16 (e.g. RMSE `1e-15` on an exact fit, `5e-05`, large losses) is hashed by Python as `...1e-15...` / `1e-05` but by Rust as `1e-15` / `5e-5`; `read_native_results` fails with "manifest score_set_hash does not match score_set.json" for a perfectly valid results dir (and `write_native_results` rejects the Python-computed hash).
- Fix: hash the exact bytes of score_set.json (store the hash over file bytes) or implement the exact Python float repr (shortest repr with `e+XX`/two-digit exponent rules) in canonical_json, and add golden tests with 1e-05, 1e-15, 1e16, 1e22.

### A7-03 Medium (confidence: medium) - NaN/Inf in prediction rows/scores are silently turned into null on read, and NaN is rejected on write
- File: crates/dag-ml-py/src/lib.rs:114-137 (`read_native_results_v2_json`, `write_native_results_v2_json`); crates/dag-ml-results/src/lib.rs:59-115 (`NativePredictionRow` f64 fields)
- Description: Parquet stores NaN/Inf happily and the reader returns `Vec<f64>`/`Option<f64>` containing them; `serde_json::to_string(&view)` then emits `null` for every non-finite f64 (NaN and +/-Inf are indistinguishable, array elements become `null`). On the write side `serde_json::from_str` rejects the `NaN`/`Infinity` tokens that Python `json.dumps` emits by default, so a failed-model row (NaN predictions) cannot be written unless the host sanitizes.
- Failure scenario: a run with some NaN predictions or `val_score=NaN` is written (via other paths) and read back through `read_native_results_v2_json`; Python receives `y_pred: [0.3, null, ...]`, `np.asarray` gives object/None arrays and downstream scoring breaks, or NaN becomes None "missing score".
- Fix: on read, either error on non-finite values or encode them explicitly (e.g. strings "NaN"/"Infinity"/"-Infinity" or a mask field) and symmetric decode on write; document the contract.

### A7-04 Medium (confidence: high) - Python callback exceptions are flattened to a string; KeyboardInterrupt/SystemExit become ordinary trial failures
- File: crates/dag-ml-py/src/in_process.rs:384-391 (`core_error_from_py`), 408-434 (`call_py_bridge`); consumers: PyOperatorController::invoke (1627-1634), run_host_hpo_search_in_process (1268+)
- Description: the original `PyErr` is discarded; only `err.value(py)` Display (i.e. `str(exc)`, no type name, no traceback, empty for `KeyboardInterrupt()`) is kept inside `RuntimeValidation`. Because the PyErr is consumed, the interpreter's pending-interrupt is gone. In host HPO, controller errors are recorded through `fail(trial_index, error)` and the search continues.
- Failure scenario: user presses Ctrl-C while an HPO trial's `op_callback` runs; `KeyboardInterrupt` is caught, turned into "python callback raised an exception: " and the trial is marked failed; the search proceeds to the next trial (Ctrl-C swallowed). No `Python::check_signals` exists anywhere in the long Rust loops either. Also an exception with an empty message (`assert`, `KeyError()` etc.) yields an uninformative error, and the traceback is lost.
- Fix: include `type(exc).__name__` and formatted traceback in the message; detect `KeyboardInterrupt`/`SystemExit` (`err.is_instance_of::<PyKeyboardInterrupt>`) and abort the whole run by stashing the PyErr and re-raising it from the pyfunction; call `py.check_signals()` between scheduler steps where feasible.

### A7-05 Medium (confidence: high on mechanism) - non-finite floats returned by Python callbacks become `null` before finiteness validation
- File: crates/dag-ml-py/src/in_process.rs:366-376 (`from_py_object`); crates/dag-ml-py/src/local_implementation.rs:242-247
- Description: `depythonize` into `serde_json::Value` goes through `Value`'s visitor whose `visit_f64` is `Number::from_f64(v).map_or(Value::Null, ..)`, so NaN/Inf silently become `Value::Null` before `deserialize_external_value`'s `validate_typed_serde_value` (finite check) can see them.
- Failure scenario: a model controller returns predictions/metrics with NaN (diverged fit). Where the target field is `f64` the user gets a misleading "invalid type: null, expected f64" error; where it is `Option<f64>` or `serde_json::Value` (diagnostics, metrics maps, lineage metadata) the NaN is silently converted to None/null and the run proceeds with missing values instead of a finite-ness refusal. numpy `float32`/`int64` scalars (not Python float/int subclasses) fail with `unsupported type` for the same reason.
- Fix: depythonize into a custom visitor/typed target directly, or pre-walk the Python object for non-finite floats and raise an explicit error; document/convert numpy scalars via `.item()` on the Python side.

### A7-06 Medium (confidence: medium) - Python binding uses non-strict parsing for contracts the WASM binding parses strictly (dup keys / NFC-colliding keys in maps)
- File: crates/dag-ml-py/src/lib.rs:216-220, 253-272, 387-418, 509-516; in_process.rs:292-294, 945-947, 2693-2694, 2217-2219, 2685-2686; WASM counterparts: dag-ml-wasm/src/lib.rs:237-272, dag-ml-wasm/src/training.rs:107-162
- Description: `sign_training_request_json`, `sign_training_replay_request_json`, `sample_relation_set_fingerprint_json`, `select_stacking_*`, `align_named_source_rows_json`, `fan_out_data_aware_branches_json`, controller manifest lists, envelopes in `run_cv_refit_in_process`, handle maps etc. use plain `serde_json::from_str`. Struct duplicate fields are rejected, but duplicate keys inside any `BTreeMap`/`Value` field (e.g. params, metadata, per-sample maps) are silently last-wins and NFC-colliding keys are not rejected. WASM routes the same entry points through `deserialize_external_contract` (TCV1 strict). `sign_training_request_json` in Python therefore seals a fingerprint over content that differs from what the raw document says.
- Failure scenario: a request with `"params":{"alpha":1,"alpha":2}` is signed by Python with alpha=2 (fingerprint valid) but refused by the WASM/other bindings; cross-language L2 portability claims break and a signed package can disagree with its source text.
- Fix: use `training::parse_strict_json` / `deserialize_external_contract` for every JSON-in entry point in dag-ml-py (same as WASM).

### A7-07 Medium (confidence: medium) - Python loaded-predictor replay makes the trusted controller-manifest check optional
- File: crates/dag-ml-py/src/training.rs:2782-2873 (`execute_loaded_predictor_replay_json`, `trusted_controller_manifests_json = None`)
- Description: when the argument is omitted (the default) no `validate_runtime_controller_manifests` call occurs and the controllers are built straight from `package.effective_plan.controller_manifests` (the package's own claims). WASM's `replay_training_package_json` and every WASM host-HPO/phase entry point require the independent trusted registry and refuse before any callback. `execute_phase_in_process` and the in-process HPO/CV paths also never cross-check an independent registry (they compile it from the caller's manifests, which is acceptable there).
- Failure scenario: a package edited to declare different controller versions/capabilities (and re-sealed) replays with the host `op_callback` without any trust comparison; the stated "trusted runtime manifests must match the package before any callback" guarantee only holds in WASM.
- Fix: make `trusted_controller_manifests_json` mandatory (or require an explicit `allow_untrusted_manifests=True`), matching WASM.

### A7-08 Low (confidence: high) - seed width/divergence between bindings
- File: crates/dag-ml-wasm/src/lib.rs:744-752, 773-780 (`root_seed: u32`); crates/dag-ml-py/src/in_process.rs:2264 (`let root_seed: u64 = 0` in `run_cv_refit_predict_in_process`), 2882 (`0` hard-coded in `run_cv_refit_methods_in_process`)
- Description: WASM `execute_campaign_phase_json`/`execute_execution_plan_phase_json` accept only a u32 root seed while core and the WASM initial-refit/replay functions use u64 (decimal string); the Python terminal-predict path ignores `plan.campaign.root_seed` and uses 0 whereas `execute_phase_in_process` honours `plan.campaign.root_seed`.
- Failure scenario: a campaign with `root_seed = 2^40` cannot be reproduced through the WASM phase API, and a terminal-predict/methods CV run with a campaign-declared seed silently runs with seed 0, giving different fold/RNG streams than the declared campaign.
- Fix: accept a decimal string u64 in WASM phase functions; use `plan.campaign.root_seed.unwrap_or(0)` (or a parameter) consistently.

### A7-09 Low (confidence: medium) - WASM prunable-window dispatcher: inconsistent failure handling and a panic path
- File: crates/dag-ml-wasm/src/host_hpo.rs:356-501 (`dispatch_prunable_window`)
- Description: (a) `.expect("validated FoldSet")` at 367-371 panics (wasm trap) if a task's plan has no fold set; (b) a worker fold result that fails JSON parsing/`validate_host_hpo_worker_fold_result` aborts the entire search via `?` (lines 401-412) whereas the non-pruning branch (569-572) converts parse errors into `HostHpoWorkerResult::Failed` for that trial only; (c) `fold_count` is taken from the first task only; (d) surviving candidates are evaluated by the "complete" dispatch again after all folds were already run (full recomputation).
- Failure scenario: one malformed worker reply in pruning mode kills the browser HPO run (and loses already-completed trials) while the same reply without pruning only fails one trial; a plan without fold_set traps the module.
- Fix: map validation/parse errors to `Failed` for that trial, replace `expect` with an error, validate all tasks share the same fold count, and avoid the redundant complete evaluation (build `Complete` evidence from the transcript).

### A7-10 Low (confidence: medium) - handle leak when the Python view callback fails
- File: crates/dag-ml-py/src/in_process.rs:872-892 (`make_view_attested`)
- Description: `self.inner.make_view(request)?` registers a view handle in the InMemoryDataProvider before the callback; if the callback raises or the receipt fails `validate_for`, the handle is never released.
- Failure scenario: repeated failed trials in generated-view HPO accumulate orphan view records in the provider (visible in `process_local_data_view_count`).
- Fix: release/discard the handle on the error paths (or create it after a successful callback).

### A7-11 Low (confidence: high) - raw artifact payloads cross the Python bridge as a list of ints
- File: crates/dag-ml-py/src/in_process.rs:1608-1625, 1647-1702 (`PyArtifactHydrationRequest.payload: &[u8]`, `Vec<u8>` response via `from_py_object`)
- Description: serde serializes `&[u8]`/`Vec<u8>` as a sequence, so pythonize hands the callback a Python `list[int]` (8 bytes/element + int objects) and `depythonize` into `serde_json::Value` rejects a `bytes`/`bytearray` return (`visit_bytes` unsupported by `Value`). Hosts must `bytes(payload)` and return `list(bytes)`.
- Failure scenario: a 200 MB fitted model export/hydrate allocates ~1.6 GB+ of Python list and several JSON-Value copies; a host that naturally returns `bytes` from the export callback gets "invalid type: byte array" instead of working.
- Fix: serialize with `serde_bytes`/`PyBytes` for payload fields and accept `bytes`/`bytearray` in the export response without going through `serde_json::Value`.

### A7-12 Low (confidence: medium) - Arrow prediction cache store: non-atomic writes / stale files
- File: crates/dag-ml-arrow/src/lib.rs:205-251 (`write_payload_set`), 387-398 (`write_json`), 400-458 (`cache_schema`)
- Description: cache `.arrow` files and the manifest are written with plain `fs::write` (truncate-in-place), no temp+rename, no fsync, and previously written `.arrow` files that are no longer referenced are never removed. `cache_schema` also inserts `METADATA_KEY_CACHE_NAMESPACE_FINGERPRINTS` twice (harmless duplicate).
- Failure scenario: a crash while overwriting an existing store directory leaves a truncated manifest/IPC file; `open` then fails (it fails closed via fingerprint checks, so no silent corruption), and repeated writes into the same root leave orphaned files.
- Fix: write to temp files and rename (manifest last), clean up unreferenced files.

### A7-13 Low (confidence: high) - `n_jobs` of non-integer type silently means sequential
- File: crates/dag-ml-py/src/in_process.rs:1365-1385
- Description: `optimizer_descriptor["n_jobs"].as_i64().unwrap_or(1)`: `"4"`, `4.0` or `null` silently run sequentially; `-1` maps to `available_parallelism` but any other negative is rejected only after `usize::try_from`.
- Failure scenario: Python passes `n_jobs=np.int64(4)` or `4.0` through JSON-ified descriptor (float) and HPO silently runs on one worker.
- Fix: reject a present-but-non-integer `n_jobs`.

### A7-14 Low (confidence: medium) - extra-variant scores dropped when primary ScoreSet is None
- File: crates/dag-ml-py/src/in_process.rs:2594-2605
- Description: for `refit_top_k > 1`, `if let Some(primary_scores) = scores.as_mut()` silently discards the additional variants' test/train reports when no primary ScoreSet exists (e.g. scoring disabled for the winner but present for extras).
- Failure scenario: `selected_refit_variant_ids` lists extra variants but their REFIT/test scores are missing from `scores` with no error.
- Fix: create the ScoreSet (as `merge_loser_validation_reports` does) or error.

### A7-15 Low (confidence: low) - results writer depends on hard links
- File: crates/dag-ml-results/src/lib.rs:261-275 (`publish_result_files`)
- Description: publication uses `std::fs::hard_link`; filesystems without hard-link support (some network/FUSE/Windows-share/exFAT mounts) make every write fail with an I/O error, with no fallback to an exclusive-create copy+rename.
- Failure scenario: results directory on a FAT/exFAT or certain bind mounts: `write_native_results_v2_json` always fails ("Operation not permitted").
- Fix: fall back to `rename` onto a verified-absent destination or `OpenOptions::create_new` copy when `hard_link` returns `Unsupported`/`PermissionDenied`.

---
Notes (verified non-issues, dropped): the parallel host-HPO path correctly detaches the GIL and workers reattach via `Python::attach`; sequential in-process paths run callbacks on the GIL-holding thread (no deadlock); `catch_unwind` wraps `Python::attach` in `call_py_bridge`; Arrow IPC read path rejects null/invalid `block_kind` without panicking; slice offsets in `ListArray::value` are handled by arrow. WASM `unsafe impl Send/Sync` for the JS controller is sound on single-threaded wasm32 (but would be UB if the crate were ever built with wasm threads).

---

## A8 audit: dag-ml CLI, bindings (JS/R/MATLAB), scripts, workflows, manifests

Scope read: `crates/dag-ml-cli/src/main.rs` (all 8.5k lines), `bindings/js/*.mjs`, `bindings/r/{R,src}`, `bindings/matlab/{+dagml,native}`, `scripts/` (validate_release_metadata, check_so_freshness, validate_abi_snapshot, check_error_taxonomy, release/*, validate_contracts main), `.github/workflows/*`, all Cargo manifests. Nothing edited, cargo not run.

Counts: Critical 0, High 0, Medium 3, Low 7.

Areas checked with no real bug found: JS controllers (N4mWasmRegressionController, N4mWasmHostOptimizer, N4mWasmMultimodalController handle lifecycle, dispose on error paths), R/MATLAB native bridges (library handle closed on all error paths after open except noted; callback uses R_tryEval / mexCallMATLABWithTrap), validate_contracts.py main (fails closed, `--require-sibling` honoured), validate_abi_snapshot, check_error_taxonomy, publish_crates.py / check_publish_plan.py, workflow path/script references (all exist), workspace manifest inheritance and MSRV pinning.

---

### A8-01 Medium (confidence: medium-high) - HPO operator tasks inherit the 30 s optimizer timeout
- File: `crates/dag-ml-cli/src/main.rs:307-308` (`adapter_timeout_ms` default 30_000), `:4237-4242` (`CliHpoControllerFactory::create` sets `ProcessAdapterRuntimeConfig.timeout: self.timeout`), `:4476-4481`, `:4262-4263`; also `bindings/r/R/host_hpo_search.R` (`adapter_timeout_ms = 30000L`, rejects <1) and `bindings/matlab/+dagml/hostHpoSearch.m` (same).
- Description: `run-host-hpo` has a single `--adapter-timeout-ms`. It is used for the optimizer JSONL adapter and also as the per-task timeout of every operator process (one-shot `wait_with_output_timeout` or persistent `read_response_line`). The rest of the CLI deliberately defaults operator execution to unbounded (`DEFAULT_PROCESS_TIMEOUT_MS = 0`, "a model fit can legitimately take hours"). Passing 0 to disable it is refused by `CliHpoOptimizer::spawn` ("must be positive"), and the R/MATLAB wrappers reject values <1.
- Scenario: an HPO trial whose model fit/CV fold takes more than 30 s (default) is killed by the timeout; the trial fails with "timed out after 30000 ms". The only workaround is to raise the optimizer timeout to the longest fit duration, which also weakens hang detection on the optimizer.
- Fix: add a separate `--operator-timeout-ms` (default `DEFAULT_PROCESS_TIMEOUT_MS`, 0 allowed) used by `CliHpoControllerFactory`, keep `--adapter-timeout-ms` for the optimizer only, and expose it in the R/MATLAB wrappers.

### A8-02 Medium (confidence: medium-high) - R and MATLAB phase wrappers corrupt u64 task seeds when injecting `lineage.seed`
- Files: `bindings/r/R/local_implementation_registry.R:150-169` (`jsonlite::fromJSON(task_json)`, `result$lineage$seed <- task$seed`, then `jsonlite::toJSON(..., digits = NA)`); `bindings/matlab/+dagml/executeExecutionPlanPhase.m:92-105` (`jsondecode(taskJson)`, `result.lineage.seed = task.seed`, `jsonencode`).
- Description: task seeds are full 64-bit values (`derive_task_seed` -> `SeedContext::derive_u64`, first 8 bytes of a SHA-256). jsonlite and `jsondecode` parse them into doubles, losing precision above 2^53 (jsonlite needs `bigint_as_char = TRUE`; the other R wrappers use it, this one does not). The core requires `NodeResult.lineage.seed == NodeTask.seed` (`crates/dag-ml-core/src/runtime/task.rs:1456`). The JS binding documents and handles this exact problem (`n4m_multimodal_controller.mjs`, "JS Numbers cannot represent all seeds").
- Scenario: a controller callback that returns a NodeResult without `lineage.seed` (the case the wrapper exists to fill in) gets an imprecise or float-formatted seed re-serialised (e.g. `1.2345e+19`); native deserialisation fails or the seed check refuses the result. ~99.9% of tasks have seeds above 2^53, so the convenience path is effectively broken; only callbacks that echo the exact seed text work.
- Fix: do not round-trip the seed through a numeric. Extract the raw `"seed"` token from `task_json` (regex on the top-level field) and splice it as text into the result JSON, or parse with `bigint_as_char = TRUE` in R and write it back as a literal; in MATLAB do the same string splice.

### A8-03 Medium (confidence: high) - R package version drift, not covered by any gate
- Files: `bindings/r/DESCRIPTION:3` (`Version: 0.3.32`) vs `Cargo.toml` workspace `0.3.34` (tags v0.3.33 and v0.3.34 exist); `scripts/validate_release_metadata.py` (no reference to `bindings/r` or `DESCRIPTION`); `.github/workflows/version-guard.yml` (checks only the Cargo workspace manifest).
- Description: the 0.3.33 and 0.3.34 release bumps updated Cargo, pyproject and the ABI snapshot but not the R package. The release-metadata gate only verifies the Cargo workspace, the PyO3 crate and `pyproject.toml`, so the drift passes CI.
- Scenario: the R package built from a v0.3.34 tag reports `dagml 0.3.32`; users and package indexes cannot distinguish it from the previous release, and a reinstall does not upgrade.
- Fix: bump DESCRIPTION to the workspace version and add a check in `validate_release_metadata.py` (parse `Version:` and compare, also for any MATLAB/JS package metadata that carries a version).

### A8-04 Low (confidence: high) - optional outputs print whole JSON documents to stdout when the flag is absent
- File: `crates/dag-ml-cli/src/main.rs:2027-2031` (`oof_average_output`), `:2035-2039` (`lineage_output`) in `run-process-dsl-cv-refit-bundle`; `:1824-1828` (`run-process-cv-refit-bundle`), `:1686-1690` and `:1745-1749` (`run-mock-refit-bundle`, `run-process-refit-bundle`).
- Description: `emit_json(opt.as_ref(), ...)` is called unconditionally; with `None` it pretty-prints the value to stdout. `node_results_output` and `prediction_cache_output` are correctly guarded with `if let Some`. A caller that passes only `--output bundle.json` still gets the full lineage records and the OOF average results dumped to stdout after the summary line.
- Scenario: a host (or the R/MATLAB wrappers' `system2`/`system` capture, which merges stdout) runs the command with `--output` only; stdout contains megabytes of JSON, bloating captured logs and error messages; callers that parse stdout for the summary line break.
- Fix: wrap these calls in `if let Some(path) = ... { emit_json(Some(path), ...) }` like `node_results_output`.

### A8-05 Low (confidence: high) - `--score-output` silently writes nothing
- File: `crates/dag-ml-cli/src/main.rs:2914-2922`.
- Description: when the replay produced no scores (`build_score_set` returns `None`) the `--score-output` file is not written and the command exits 0. Unlike the `--output` branch (which serialises `"scores": null`), the caller gets no signal.
- Scenario: R `dagml_replay_bundle(score_output = path)` returns success; the caller then reads a missing file, or a stale file from a previous run at the same path.
- Fix: either bail with an explicit error when scores are absent, or write `null`; also use `emit_json` for consistent path/context handling.

### A8-06 Low (confidence: medium) - `check_so_freshness.py` comment heuristic and lock coverage gaps
- File: `scripts/check_so_freshness.py:150-153` (`text.startswith(("//", "/*", "*", "*/"))`), `:48-54` (`RUST_FILES`).
- Description: (a) the "comment-only diff" test treats any changed line starting with `*` as a comment, so a code change such as `*out = value;` or `* scale` continuation lines on a post-binary commit is classified as not requiring a rebuild; (b) `crates/dag-ml-py/Cargo.lock` (the lockfile actually used by `maturin build --locked`) is not in `RUST_FILES`, and `crates/dag-ml-arrow`/other path deps are not considered, so a pyo3/pythonize/n4m version bump in the standalone lock never marks the committed `.so` stale. The self-test covers neither case.
- Scenario: a commit that only dereference-assigns in core sources, or only bumps a dependency in `crates/dag-ml-py/Cargo.lock`, leaves a stale `_dag_ml.abi3.so` and the gate stays green.
- Fix: treat a leading `*` as a comment only inside a tracked `/* */` block (or only when followed by space and not part of an expression), add `crates/dag-ml-py/Cargo.lock` (and `Cargo.toml` of path deps) to the tracked inputs, add self-test cases.

### A8-07 Low (confidence: high) - CLI integration tests silently skip in CI (false green)
- Files: `crates/dag-ml-cli/tests/cli_contracts.rs:2800-2805` (`r_has_prospectr` -> `return;`), `:4040-4054` (sklearn/joblib), further `return;` guards through the file; `crates/dag-ml-cli/tests/initial_full_refit.rs:206-215` (skips when the sibling `nirs4all-methods/bindings/python/src/pls4all` is absent, which is always the case in CI because `methods-hpo-local` only checks out Methods in its own job and does not set `PLS4ALL_PYTHONPATH` for the `rust` job).
- Description: tests return `Ok` when optional toolchains or sibling checkouts are missing. Only the HPO tests have a `DAGML_REQUIRE_*` escape. Rust CI job has no R, so the Rscript/prospectr adapter tests and the concrete Methods PLS bridge replay never run, yet are reported passed.
- Scenario: a regression in the prospectr R adapter protocol or in portable Methods PLS replay merges with a green CI.
- Fix: make the skips conditional on a `DAGML_REQUIRE_*` env var that CI sets (as already done for HPO), or mark them `#[ignore]` and run them explicitly in the jobs that provide the toolchain.

### A8-08 Low (confidence: medium) - one-shot process controller leaks the child on early error
- File: `crates/dag-ml-cli/src/main.rs:5207-5226` (`ProcessRuntimeController::invoke`).
- Description: the child is spawned, then serialisation / `write_all` of the task can return early with `?`/`map_err`. `Child` has no kill-on-drop; the process is neither killed nor waited, leaving a zombie (and a running adapter if it ignores stdin EOF) for the CLI's lifetime. With long campaigns and repeated failures under retries this accumulates.
- Scenario: adapter exits immediately (bad shebang, import error) while the task JSON is larger than the pipe buffer -> EPIPE on `write_all` -> error returned, child never reaped; the stderr that explains the failure is never read, so the error message hides the real cause.
- Fix: on any pre-wait error call `child.kill()`/`child.wait()` and include captured stderr (spawn the pipe readers before writing stdin, as is already done in `wait_with_output_timeout`).

### A8-09 Low (confidence: medium) - `release-npm.yml` can publish from any ref and cancels in-flight publishes
- File: `.github/workflows/release-npm.yml:17-18` (`cancel-in-progress: true`), `:77-85` and `:36-47`.
- Description: with `workflow_dispatch` + `publish=true` the "Validate tag matches package.json version" step is skipped (it requires `refs/tags/`), so an arbitrary branch commit can be published to npm under whatever version Cargo currently has (skipped only if that version already exists). `concurrency.cancel-in-progress: true` on a publishing workflow can cancel a tag run mid-publish when a second tag/dispatch for the same ref starts.
- Scenario: an accidental dispatch from a feature branch publishes untagged code as the next release version; or a re-pushed tag aborts an in-progress publish after a partial step.
- Fix: refuse non-tag publishes (`github.ref_type == 'tag'` required for `do=true`), and set `cancel-in-progress: false` for this workflow (as `release-crates.yml` does).

### A8-10 Low (confidence: medium) - cross-language parity job pins a stale released `dag-ml`
- File: `.github/workflows/methods-wasm-hpo-candidate.yml:127` (`pip install 'dag-ml==0.3.32' ...`).
- Description: the WASM side of the comparison is built from the current checkout (0.3.34, push-triggered on core HPO/WASM changes) but the Python side comes from a PyPI `dag-ml==0.3.32`. Fold/score parity is thus qualified against a two-release-old Python build rather than the code under test, so parity regressions introduced by the changed paths can be invisible or can be spurious.
- Fix: build and install the wheel from the checkout (as the `python-bindings` job does) or derive the pin from `Cargo.toml`.

### A8-11 Low (confidence: low) - persistent adapter restart loses artifacts held by the dead worker without invalidating registries
- File: `crates/dag-ml-cli/src/main.rs:5309-5330` (restart in `PersistentProcessRuntimeController::invoke`), `refit_artifact_workers` / `hydrated_artifact_workers` maps (`:4620-4621`).
- Description: on a restartable failure the session is replaced with a fresh process. Handles/artifacts previously produced by that worker for other nodes (hash-pinned to the same `worker_index`) no longer exist in the new process, but the registries still point at that worker index, and `capture_raw_refit_payloads` / later PREDICT will ask the fresh process for them. The failure is loud (unknown artifact), not silent, but is reported as a worker error rather than "artifact lost on restart", and `--process-retries` gives a false impression of resilience across REFIT with multiple model nodes.
- Fix: on restart, drop the registry entries owned by the replaced worker and fail REFIT/PREDICT with an explicit "artifact lost on worker restart" error (or refuse restarts after the first REFIT artifact is produced).

---

## Réaudit ciblé du 2026-10-05 — R1–R5

État examiné : `main`, HEAD `867f3576592ec390e2a16ced53cc027c022cde27`, version 0.3.34, **avec les corrections et ajouts non commités présents dans le checkout**. Les SHA-256 des sources examinées sont conservés dans [audited-sources.json](../../_audits/2026-10-05-dagml-reaudit/audited-sources.json), relevés à 11:33:56 UTC. Ils ont été recoupés après les tests ; aucun de ces fichiers n'avait changé.

Ce passage utilise le rapport de résolution comme liste d'exclusion : pas de nouvelle analyse complète des 117 points, pas de réouverture des cinq faux positifs ni des compatibilités historiques A4-08/A4-11. La revue cible les changements de validation, de canonicalisation, de génération, de runtime et les nouveaux helpers publics. Les autres pistes examinées ne sont pas présentées comme des bugs sans preuve suffisante. Ce n'est pas une nouvelle couverture exhaustive de tous les bindings.

| ID | Gravité | État | Rapport initial | Défaut confirmé |
| --- | --- | --- | --- | --- |
| R1 | Medium | Corrigé | Complément A2-06 | Changer le préfixe d'une variante contournait le contrôle de son seed et de son empreinte. |
| R2 | Medium | Corrigé | Complément A2-14 | Le label d'une structure imbriquée effaçait aussi des `id` sémantiques de l'opérateur. |
| R3 | Low | Corrigé | Complément A2-15 | Deux IDs de générateurs courts et valides devenaient le même namespace après sanitization. |
| R4 | Low | Corrigé | Ajout après l'audit initial | Le helper de préparation multimodale paniquait sur une source JSON non objet. |
| R5 | Low | Corrigé | Complément A2-13 | Le plan acceptait deux edges vers un port déclaré One par le contrôleur résolu. |

Les descriptions et numéros de ligne suivants décrivent les reproductions **avant correction**. Les sources corrigées sont identifiées dans [applied-sources.json](../../_audits/2026-10-05-dagml-reaudit/fixes/applied-sources.json).

### R1 — L'exemption par préfixe laisse passer une variante incohérente

**Source :** `crates/dag-ml-core/src/plan.rs:604–628`, `generation.rs:461–473`. Confiance haute ; reproduit par `tests::r1_nonstandard_variant_prefix_bypasses_identity_validation`.

`ExecutionPlan::validate` ne re-dérive les variantes que si leur ID commence par `variant:`. Son commentaire invoque les checkpoints HPO pour les autres profils, mais aucun checkpoint n'est vérifié dans cette branche. `VariantPlan::validate` exige seulement une empreinte non vide et des choices/overrides valides.

La reproduction construit le plan officiel de la fixture `package`, puis remplace le seed d'une variante par `666` et son empreinte par 64 caractères `f`. Avec l'ID canonique, le plan est correctement refusé. En changeant uniquement l'ID en `trial:invented`, le même plan incohérent est accepté, sans checkpoint HPO. La variante conserve ses choices de génération ordinaires ; le test n'invente pas une proposition HPO valide.

**Conséquence :** la frontière de validation des plans importés ne garantit pas l'identité, l'empreinte et le seed qu'elle annonce. La reproduction couvre cette acceptation ; elle ne prétend pas démontrer un contournement de toutes les validations supplémentaires d'un bundle ou d'un TrainingOutcome.

**Correction appliquée :** tous les profils sont re-dérivés. Les variantes ordinaires restent liées aux choices/campagne ; les trials Methods sont liés à leur base canonique, trial, namespace éventuel, empreinte et seed. Les trials host HPO transportent maintenant leur `objective_fingerprint` dans le choix signé, permettant le même contrôle sans exemption de préfixe. Les sous-ensembles sélectionnés et les variantes opérateur restent acceptés. Les anciens candidats host HPO sans cette liaison explicite sont refusés ; ils doivent être régénérés, sans relâcher les contrôles de checkpoint.

### R2 — La suppression des IDs structurels déborde dans le contenu des opérateurs

**Source :** `crates/dag-ml-core/src/dsl/generation.rs:251–279`. Confiance haute ; reproduit par `tests::r2_nested_operator_semantic_id_is_removed_from_label`.

Le nouveau membre canonique `structure` est parcouru récursivement par `strip_structural_node_ids`. Cette fonction supprime `id` de **tout objet possédant une clé `kind`**, y compris les valeurs opaques de `operator`, `params`, `metadata` ou `selector`. Ces objets ne sont pas nécessairement des étapes DSL.

La reproduction appelle l'API publique de label sur un modèle dont l'opérateur contient `selector: {kind: "table", id: "catalog:A"}`, puis sur la même déclaration avec `catalog:B`. Les labels sont distincts pour les étapes plates. En enveloppant chacune dans une étape `sequential`, les deux labels deviennent identiques : `1fc61ae9b3af97dc0c26148eaa92d86288cd301abbb95d94eaa5320d12599495`.

**Conséquence :** des configurations déclarées différentes peuvent être confondues par les consommateurs de `variant_label`. Les empreintes complètes de graph/plan restent distinctes ; ce test ne démontre pas une collision de tous les mécanismes de cache ou d'archive.

**Correction appliquée :** le parcours suit exclusivement les enfants typés du DSL : branches, stages, tail, séquences et opérateurs de concat. Seuls leurs IDs de nœuds sont retirés ; operator, params, metadata et selector restent opaques et intacts. Les tests vérifient les IDs sémantiques distincts et le renommage neutre des seuls IDs de nœuds.

### R3 — La désambiguïsation ne couvre que les IDs tronqués

**Source :** `crates/dag-ml-core/src/dsl/generation.rs:1266–1285`, `2128–2140`. Confiance haute ; reproduit par `tests::r3_short_sanitized_generator_ids_collide`.

`sanitized_id_fragment` ajoute un digest uniquement si la longueur dépasse la limite. Pour un ID court dont les caractères ont été remplacés, le résultat reste ambigu : `g:a` et `g_a` donnent tous deux `g_a`.

Chacun des deux générateurs, contenant un modèle local `m`, compile séparément. Placés dans deux branches indépendantes d'un même DSL, ils échouent ensemble avec `produced duplicate node gen:g_a:c0:n0.m`. Les IDs déclarés sont distincts et aucun n'est tronqué.

**Conséquence :** rejet de pipelines valides. Le correctif des références de contraintes et celui des longues chaînes ne sont pas remis en cause.

**Correction appliquée :** un digest est ajouté lorsque la sanitization modifie l'ID, lorsque celui-ci est tronqué ou lorsque son suffixe littéral imite le format réservé du digest. Le dernier cas empêche un ID déclaré `g_a_<digest>` de réintroduire la collision avec la forme encodée de `g:a`. Les limites 32/28 caractères et le déterminisme sont conservés. Les IDs dérivés affectés changent : les graphes et archives correspondants doivent être régénérés pour la qualification finale.

### R4 — Panic lors de la préparation d'une source multimodale malformée

**Source :** `crates/dag-ml-core/src/multimodal_replay_inputs.rs:107–115`. Confiance haute ; reproduit par `tests::r4_multimodal_preparation_panics_on_nonobject_source`.

`prepare_multimodal_replay` vérifie les noms et schémas, puis écrit `sources[name]["descriptor"]` sans contrôler la forme de l'objet source. Sur un nombre, une chaîne ou un tableau, `serde_json::Value::IndexMut` panique.

Le package brut de l'archive U07 v4 passe `package.validate()`, et la préparation avec la fixture composée valide passe également. En changeant seulement `current.sources.nir` en `42`, l'appel panique avec `cannot access key "descriptor" in JSON number`, au lieu de retourner `Err`. Le witness capture ce panic pour que l'audit puisse continuer.

**Conséquence et limite :** défaut de gestion d'entrée sur cette API Rust publique retournant un `Result`. Le déclencheur est une entrée `current` malformée ; le test ne prétend pas que le producteur IO normal émet ce format ni qu'une prédiction avec des données valides panique.

**Correction appliquée :** `sources` doit être un objet avec exactement les noms déclarés ; chaque source doit être présente et objet avant insertion du descriptor. Les entrées absentes, nulles, numériques, tableaux ou noms supplémentaires retournent une erreur contextualisée sans panic. L'entrée fournie n'est pas mutée. Le raccord aux changements IO est coordonné avec l'autre root ; la comparaison de schémas est conservée par ce correctif.

### R5 — La cardinalité du port du contrôleur n'est pas recoupée

**Source :** `crates/dag-ml-core/src/plan.rs:490–518`. Confiance haute ; reproduit par `tests::r5_two_graph_inputs_pass_a_single_input_controller_manifest`.

La nouvelle réconciliation vérifie le nom, le kind et la représentation, mais pas la compatibilité des cardinalités. La reproduction ajoute une seconde source distincte vers `model:base.x` de la fixture officielle. Le graph avec un port One refuse correctement les deux edges. En déclarant ce port Many dans le graph, `build_execution_plan` et `plan.validate()` acceptent la configuration, alors que le manifest résolu `controller:model.mock` déclare toujours `x` comme One.

**Conséquence :** un plan incompatible avec son contrôleur passe la validation préalable. Le collecteur de data edges possède ensuite un refus de clés dupliquées (`runtime/scheduler.rs:5198`) ; l'audit ne revendique donc pas une perte silencieuse de données. Le test reproduit l'acceptation du plan, pas une exécution complète de ce cas.

**Correction appliquée :** la compatibilité du port inclut le nombre d'edges entrants : One/Optional refuse un fan-in supérieur à un. Un port Many de graph avec un seul edge reste compatible avec un contrôleur One. Les prototypes Many, manifests génériques et exceptions Methods/OOF sont conservés et couverts par les suites existantes.

### Preuves, validation et limites

Les sources des witnesses, fixtures et logs sont dans le [dossier de preuves local](../../_audits/2026-10-05-dagml-reaudit/README.md). Le passage de constat a fait passer **19 tests : cinq witnesses confirmant les défauts présents et les 14 régressions existantes du premier audit**, importées sans modification. Un witness passant signifiait que le bug était reproduit ; ce harness initial et son log sont conservés comme preuve historique. Aucune source de production n'avait été modifiée à ce stade.

Commande exécutée depuis `dag-ml/` :

```bash
cargo test --offline --locked \
  --manifest-path ../_audits/2026-10-05-dagml-reaudit/Cargo.toml \
  --target-dir target -- --nocapture
```

Le harness dépend directement du core courant, sans feature Methods, et utilise son propre lockfile. La tentative directe `cargo test --offline --locked -p dag-ml-core@0.3.34 --test bug_audit_regressions` a été refusée avant compilation parce que le lockfile du workspace courant nécessite une mise à jour. Il n'a pas été modifié ; les mêmes 14 tests ont été exécutés via le harness. Ce constat de qualification est distinct des cinq bugs et peut dépendre du travail de release en cours.

Après correction, une copie de vérification inverse les cinq assertions : **les 19 tests repassent**, incluant la préparation valide du package multimodal réel puis `nir=42` retournant `Err` sans panic. Neuf régressions sont ajoutées au dépôt, ainsi qu'un test local des formes de sources. Les gates complémentaires sont consignées dans la résolution. Les sources ont été intégrées après dégel et vérification de leurs SHA de base ; la reconstruction et la qualification finale communes des bindings restent au root d'intégration, conformément à l'accord de coordination.
