//! Runtime execution: schedulers, controllers, stores, OOF/merge logic.
//!
//! Split from the former monolithic `runtime.rs` into cohesive submodules
//! (pure refactor — code moved verbatim). `mod.rs` owns the run context,
//! the controller registry, the custom-aggregation dispatch entry points,
//! native variant selection, and re-exports the full runtime surface so
//! `pub use runtime::*` in `lib.rs` resolves identically.

pub(crate) use std::cell::RefCell;
pub(crate) use std::collections::{BTreeMap, BTreeSet};
pub(crate) use std::fs;
pub(crate) use std::io::Read;
pub(crate) use std::path::{Path, PathBuf};

pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use sha2::{Digest, Sha256};

pub(crate) use crate::aggregation::{
    aggregate_observation_predictions, aggregate_sample_predictions_by_unit,
    reduce_predictions_across_branches, reduce_proba_mean_across_branches,
    AggregatedPredictionBlock, AggregationControllerInput, AggregationControllerOutput,
    AggregationControllerResult, AggregationControllerTask, ObservationPredictionBlock,
    PredictionUnitId,
};
pub(crate) use crate::bundle::{
    build_aggregated_prediction_cache_payload, build_prediction_cache_payload,
    bundle_prediction_requirement_key, validate_prediction_cache_payload_matches_record,
    BundlePredictionCachePayload, BundlePredictionCachePayloadSet, BundlePredictionCacheRecord,
    BundlePredictionRequirement, ExecutionBundle, RefitArtifactRecord, ReplayPhaseRequest,
};
pub(crate) use crate::campaign::stable_json_fingerprint;
pub(crate) use crate::controller::{capabilities_support_fit_influence, ControllerCapability};
pub(crate) use crate::data::{
    DataBinding, DataRequestPartition, ExternalDataPlanEnvelope, RepresentationCompatibilityReport,
    RepresentationPlan, RepresentationReplayManifest,
};
pub(crate) use crate::error::{DagMlError, Result};
pub(crate) use crate::fold::{FoldAssignment, FoldPartitionMode, FoldSet};
pub(crate) use crate::generation::{GenerationChoice, VariantPlan};
pub(crate) use crate::graph::{EdgeSpec, PortKind};
pub(crate) use crate::ids::{
    ArtifactId, BranchId, BundleId, ControllerId, FoldId, LineageId, NodeId, RunId, SampleId,
    VariantId,
};
pub(crate) use crate::metrics::{
    cross_fold_validation_reports, reassemble_merge_targets, score_regression_aggregated_block,
    score_regression_prediction_block, OofAverageBlock, RegressionMetricKind,
    RegressionMetricReport, RegressionTargetBlock, RegressionTargetRecord, ScoreSet,
    SCORE_SET_SCHEMA_VERSION,
};
pub(crate) use crate::oof::{PredictionBlock, PredictionPartition};
pub(crate) use crate::phase::Phase;
pub(crate) use crate::plan::{CampaignSpec, ExecutionPlan, NodePlan};
pub(crate) use crate::policy::{
    AggregationPolicy, FitInfluencePolicy, PredictionLevel, ShapeDelta, ShapeDeltaKind,
};
pub(crate) use crate::relation::SampleRelationSet;
pub(crate) use crate::rng::SeedContext;
pub(crate) use crate::selection::{
    select_candidate, CandidateScore, SelectionMetric, SelectionPolicy,
};

mod artifact;
mod dataview;
mod merge;
mod oof;
mod prediction_store;
mod scheduler;
mod scoring;
mod task;

pub use artifact::*;
pub use dataview::*;
pub(crate) use merge::*;
pub use oof::*;
pub use prediction_store::*;
pub use scheduler::*;
pub(crate) use scoring::*;
pub use task::*;

pub struct BundleReplayExecution<'a> {
    pub plan: &'a ExecutionPlan,
    pub bundle: &'a ExecutionBundle,
    pub replay_request: &'a ReplayPhaseRequest,
    pub prediction_cache_store: Option<&'a dyn RuntimePredictionCacheStore>,
    pub controllers: &'a RuntimeControllerRegistry,
    pub data_provider: &'a dyn RuntimeDataProvider,
    pub artifact_store: &'a dyn RuntimeArtifactStore,
    pub data_envelopes: &'a BTreeMap<String, ExternalDataPlanEnvelope>,
}

#[derive(Default)]
pub struct RuntimeControllerRegistry {
    controllers: BTreeMap<ControllerId, Box<dyn RuntimeController>>,
}

impl RuntimeControllerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, controller: Box<dyn RuntimeController>) -> Result<()> {
        let id = controller.controller_id().clone();
        if self.controllers.insert(id.clone(), controller).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate runtime controller `{id}`"
            )));
        }
        Ok(())
    }

    pub fn get(&self, controller_id: &ControllerId) -> Option<&dyn RuntimeController> {
        self.controllers.get(controller_id).map(Box::as_ref)
    }
}

pub fn dispatch_custom_observation_aggregation(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    task_id: impl Into<String>,
    block: ObservationPredictionBlock,
    relations: SampleRelationSet,
    policy: AggregationPolicy,
    requested_sample_order: Vec<SampleId>,
) -> Result<PredictionBlock> {
    let controller_id = custom_aggregation_controller_id(&policy)?;
    ensure_aggregation_controller_capability(plan, controller_id)?;
    let task = AggregationControllerTask {
        schema_version: crate::aggregation::AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
        task_id: task_id.into(),
        controller_id: controller_id.clone(),
        policy,
        reduction_plan: None,
        input: AggregationControllerInput::ObservationToSample {
            block,
            relations,
            requested_sample_order,
        },
    };
    let result = dispatch_custom_aggregation_task(controllers, &task)?;
    match result.output {
        AggregationControllerOutput::Sample { block } => Ok(block),
        AggregationControllerOutput::Unit { .. } => Err(DagMlError::RuntimeValidation(format!(
            "aggregation controller task `{}` returned unit output for observation input",
            task.task_id
        ))),
    }
}

pub fn dispatch_custom_sample_aggregation(
    plan: &ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    task_id: impl Into<String>,
    block: PredictionBlock,
    relations: SampleRelationSet,
    policy: AggregationPolicy,
    requested_unit_order: Vec<PredictionUnitId>,
) -> Result<AggregatedPredictionBlock> {
    let controller_id = custom_aggregation_controller_id(&policy)?;
    ensure_aggregation_controller_capability(plan, controller_id)?;
    let task = AggregationControllerTask {
        schema_version: crate::aggregation::AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
        task_id: task_id.into(),
        controller_id: controller_id.clone(),
        policy,
        reduction_plan: None,
        input: AggregationControllerInput::SampleToUnit {
            block,
            relations,
            requested_unit_order,
        },
    };
    let result = dispatch_custom_aggregation_task(controllers, &task)?;
    match result.output {
        AggregationControllerOutput::Unit { block } => Ok(block),
        AggregationControllerOutput::Sample { .. } => Err(DagMlError::RuntimeValidation(format!(
            "aggregation controller task `{}` returned sample output for sample input",
            task.task_id
        ))),
    }
}

pub fn dispatch_custom_aggregation_task(
    controllers: &RuntimeControllerRegistry,
    task: &AggregationControllerTask,
) -> Result<AggregationControllerResult> {
    task.validate()?;
    let controller = controllers.get(&task.controller_id).ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "aggregation runtime controller `{}` is not registered",
            task.controller_id
        ))
    })?;
    let result = controller.invoke_aggregation(task)?;
    result.validate_for_task(task)?;
    Ok(result)
}

pub(crate) fn custom_aggregation_controller_id(
    policy: &AggregationPolicy,
) -> Result<&ControllerId> {
    policy.validate()?;
    policy
        .custom_controller
        .as_ref()
        .map(|controller| &controller.controller_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "custom aggregation dispatch requires a custom_controller policy".to_string(),
            )
        })
}

pub(crate) fn ensure_aggregation_controller_capability(
    plan: &ExecutionPlan,
    controller_id: &ControllerId,
) -> Result<()> {
    let manifest = plan
        .controller_manifests
        .get(controller_id)
        .ok_or_else(|| {
            DagMlError::Planning(format!(
                "missing aggregation controller manifest `{controller_id}`"
            ))
        })?;
    if !manifest
        .capabilities
        .contains(&ControllerCapability::AggregatesPredictions)
    {
        return Err(DagMlError::Planning(format!(
            "aggregation controller `{controller_id}` must declare aggregates_predictions"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RunContext {
    pub run_id: RunId,
    pub root_seed: Option<u64>,
    pub variant_id: Option<VariantId>,
    pub prediction_store: InMemoryPredictionStore,
    pub aggregated_prediction_store: InMemoryAggregatedPredictionStore,
    pub lineage: InMemoryLineageRecorder,
    /// Native per-fold/per-partition score reports collected during the run (when the host emits
    /// `regression_targets`).
    pub score_collector: Vec<RegressionMetricReport>,
    /// Per-fold `y_true` records, kept so cross-fold ensembles (the OOF average) can be scored.
    pub regression_target_records: Vec<RegressionTargetRecord>,
    /// The per-sample cross-fold OOF average blocks (+ `y_true`) collected alongside the scalar OOF
    /// average reports — one per scored producer. Surfaced so the host can fill the `(validation, avg)`
    /// row's per-sample y_pred; populated by [`collect_cross_fold_validation_scores`], empty otherwise.
    pub oof_average_blocks: Vec<OofAverageBlock>,
}

impl RunContext {
    pub fn new(run_id: RunId, root_seed: Option<u64>) -> Self {
        Self {
            run_id,
            root_seed,
            variant_id: None,
            prediction_store: InMemoryPredictionStore::new(),
            aggregated_prediction_store: InMemoryAggregatedPredictionStore::new(),
            lineage: InMemoryLineageRecorder::new(),
            score_collector: Vec::new(),
            regression_target_records: Vec::new(),
            oof_average_blocks: Vec::new(),
        }
    }

    /// Score the cross-fold OOF average from the collected per-fold validation predictions + targets
    /// and append the reports (one per producer, `fold_id = "avg"`) to the score collector, plus —
    /// additively — the per-sample OOF average block + `y_true` each report was computed from to
    /// [`oof_average_blocks`](Self::oof_average_blocks) (so the host can fill the `(validation, avg)`
    /// row's per-sample y_pred). Call after FIT_CV; a no-op when nothing was scored or no producer has
    /// more than one fold.
    ///
    /// `partition_mode` is the campaign's [`FoldPartitionMode`]: `Partition` (KFold) requires a unique
    /// per-producer OOF set, while `Resampled` (ShuffleSplit / repeated CV) permits a sample to be
    /// validated in multiple folds (averaged when scored). Pass the plan's
    /// [`fold_set`](ExecutionPlan::fold_set) mode (default `Partition` when there is no fold set).
    pub fn collect_cross_fold_validation_scores(
        &mut self,
        partition_mode: FoldPartitionMode,
    ) -> Result<()> {
        let outcome = cross_fold_validation_reports(
            self.prediction_store.blocks(),
            &self.regression_target_records,
            SCORE_METRICS,
            partition_mode,
        )?;
        self.score_collector.extend(outcome.reports);
        self.oof_average_blocks.extend(outcome.oof_averages);
        Ok(())
    }

    /// Build a [`ScoreSet`] from the collected reports (or `None` if scoring was off / produced
    /// nothing), e.g. to attach to the [`ExecutionBundle`](crate::bundle::ExecutionBundle).
    pub fn build_score_set(
        &self,
        plan_id: impl Into<String>,
        selection_metric: Option<String>,
    ) -> Option<ScoreSet> {
        if self.score_collector.is_empty() {
            return None;
        }
        Some(ScoreSet {
            schema_version: SCORE_SET_SCHEMA_VERSION,
            plan_id: plan_id.into(),
            selection_metric,
            reports: self.score_collector.clone(),
        })
    }
}

/// Outcome of native variant selection: the winning variant plus EVERY scored variant's
/// cross-validation reports, each tagged with its own `variant_id`.
///
/// The reports are the per-fold + cross-fold-OOF-average VALIDATION (OOF) reports collected while
/// ranking. They are emitted so a generated sweep can surface every variant's CV score — not only
/// the winner's — to match the legacy per-variant `num_predictions`. These are REPORT-ONLY
/// validation scores of non-selected models: they never feed any downstream training/feature path
/// (no prediction blocks, no `RegressionTargetRecord`s, no handles leave selection — see
/// [`select_best_variant_by_cv`]), so the OOF/leakage invariants are unaffected.
#[derive(Clone, Debug)]
pub struct VariantSelection {
    /// The winning variant, ranked by `selection_metric`. The SELECT DECISION is identical to the
    /// pre-existing behavior; `validation_reports` is purely additive context.
    pub selected_variant_id: VariantId,
    /// Per-variant VALIDATION (OOF) reports for ALL ranked variants (winner included), each tagged
    /// with its `variant_id`. The cross-fold OOF average per producer is re-tagged with the variant
    /// id (its native form has `variant_id = None`); the per-fold reports already carry it.
    pub validation_reports: Vec<RegressionMetricReport>,
}

/// Pick the best variant of a multi-variant plan by its cross-validation score, natively.
///
/// "Option A": each variant is scored with its OWN single-variant FIT_CV — the plan is cloned with
/// `variants = vec![variant]` so the existing per-producer cross-fold OOF averaging
/// ([`RunContext::collect_cross_fold_validation_scores`]) is unambiguous (one variant in scope, so a
/// validation `PredictionBlock` belongs to exactly one variant). The OOF-average report per variant
/// becomes a [`CandidateScore`], and [`select_candidate`] ranks them by `selection_metric` (the
/// metric's [`objective`](RegressionMetricKind::objective) drives the direction — RMSE minimizes,
/// accuracy maximizes). The winning candidate id maps back to its [`VariantId`].
///
/// Beyond ranking, every scored variant's VALIDATION (OOF) reports — the per-fold reports and the
/// cross-fold OOF average, each tagged with its `variant_id` — are accumulated and returned in
/// [`VariantSelection::validation_reports`] so the caller can surface ALL variants' CV scores (not
/// just the winner's) in the final bundle. This is OOF-safe: the per-variant CV runs happen in
/// transient `RunContext`s whose prediction stores and `RegressionTargetRecord`s are dropped here;
/// only the scalar score reports (derived from `y_true`) survive, so a non-selected variant's OOF
/// predictions can NEVER reach any downstream training/feature path.
///
/// Native scoring is opt-in: it only happens when the host emits `regression_targets`. So this
/// returns `Ok(None)` when NO variant produced a cross-fold OOF average (scoring is off, the normal
/// case today) — the caller should then fall back to its default variant, behaving exactly as before.
/// When EVERY variant scored, it returns `Ok(Some(best))`. A partially-scored set (some variants
/// scored, others not) is an inconsistent host and is rejected so variants are never ranked unfairly.
///
/// `run_single_variant_fit_cv` runs FIT_CV for the single-variant plan into the supplied context
/// (the caller supplies the scheduler/data-provider wiring); this keeps the selection logic free of
/// host runtime details and unit-testable with mock controllers. Cloning a one-variant plan is
/// valid: `node_plans`/`fold_set` are plan-level (not keyed per variant) and variant params are
/// applied per-node at task build time, so the per-variant CV is isolated.
pub fn select_best_variant_by_cv<F>(
    plan: &ExecutionPlan,
    run_id: &RunId,
    root_seed: Option<u64>,
    selection_metric: RegressionMetricKind,
    mut run_single_variant_fit_cv: F,
) -> Result<Option<VariantSelection>>
where
    F: FnMut(&ExecutionPlan, &mut RunContext) -> Result<()>,
{
    plan.validate()?;
    if plan.variants.is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "cannot select a variant for a plan with no variants".to_string(),
        ));
    }

    let mut candidates: Vec<CandidateScore> = Vec::with_capacity(plan.variants.len());
    // Every ranked variant's VALIDATION (OOF) reports, each tagged with its variant_id, accumulated
    // so the caller can emit ALL variants' CV scores (not just the winner's) in the bundle.
    let mut variant_validation_reports: Vec<RegressionMetricReport> = Vec::new();
    // Tracks whether ANY variant emitted scores at all (host targets present), so an empty candidate
    // set can be told apart from "scoring genuinely off" (no targets) — see the post-loop branch.
    let mut any_scores_seen = false;
    for variant in &plan.variants {
        let single_variant_plan = ExecutionPlan {
            variants: vec![variant.clone()],
            ..plan.clone()
        };
        let mut ctx = RunContext::new(run_id.clone(), root_seed);
        ctx.variant_id = Some(variant.variant_id.clone());
        run_single_variant_fit_cv(&single_variant_plan, &mut ctx)?;
        ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(plan))?;
        if !ctx.score_collector.is_empty() {
            any_scores_seen = true;
        }
        // `cross_fold_validation_reports` emits one cross-fold OOF average PER producer. Native SELECT
        // ranks a variant by a single score, so a multi-producer DAG is ambiguous and refused rather
        // than silently ranked on whichever producer happened to be first (an explicit score-target
        // producer is a future extension).
        let avg_reports = ctx
            .score_collector
            .iter()
            .filter(|report| {
                report.partition == PredictionPartition::Validation
                    && report
                        .fold_id
                        .as_ref()
                        .is_some_and(|fold| fold.as_str() == "avg")
            })
            .collect::<Vec<_>>();
        match avg_reports.as_slice() {
            [] => {}
            [report] => candidates.push(
                (*report)
                    .clone()
                    .into_candidate_score(variant.variant_id.as_str())?,
            ),
            _ => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "variant `{}` produced {} cross-fold OOF averages (multiple prediction producers); native SELECT needs a single score target",
                    variant.variant_id,
                    avg_reports.len()
                )));
            }
        }
        // Retain this variant's VALIDATION reports (per-fold + cross-fold avg) tagged with its own
        // variant_id. The avg report's native form has `variant_id = None`, so stamp it here; the
        // per-fold reports already carry it from `apply_result_scoring`. Only Validation reports are
        // kept — the transient CV runs FIT_CV only (no Final/Test), so this is OOF-only by
        // construction, but the filter makes the report-only guarantee explicit.
        for mut report in ctx.score_collector {
            if report.partition != PredictionPartition::Validation {
                continue;
            }
            report.variant_id = Some(variant.variant_id.clone());
            variant_validation_reports.push(report);
        }
    }

    if candidates.is_empty() {
        if any_scores_seen {
            // Targets WERE emitted, but no producer yielded a cross-fold average (e.g. a single fold,
            // where the average is skipped). We cannot rank — surface it instead of falling back.
            return Err(DagMlError::RuntimeValidation(
                "variants produced scores but no cross-fold OOF average; cannot rank — need >=2 folds or an explicit score target".to_string(),
            ));
        }
        // Native scoring is genuinely off (no host targets) — let the caller keep its default variant.
        return Ok(None);
    }
    if candidates.len() != plan.variants.len() {
        return Err(DagMlError::RuntimeValidation(format!(
            "native variant SELECT scored only {} of {} variants; cannot rank variants fairly",
            candidates.len(),
            plan.variants.len()
        )));
    }

    let policy = SelectionPolicy {
        id: format!("select:variant:{}", selection_metric.name()),
        metric: SelectionMetric {
            name: selection_metric.name().to_string(),
            objective: selection_metric.objective(),
        },
        required_metric_level: None,
        require_finite: true,
        evaluation_scope: None,
        refit_slot_plan: None,
        stacking_fit_contract: None,
        reduction_id: None,
    };
    let decision = select_candidate(&policy, &candidates)?;
    let selected_variant_id = VariantId::new(decision.selected_candidate_id).map_err(|error| {
        DagMlError::RuntimeValidation(format!("selected variant id is invalid: {error}"))
    })?;
    Ok(Some(VariantSelection {
        selected_variant_id,
        validation_reports: variant_validation_reports,
    }))
}

#[cfg(test)]
mod explain_contract_tests {
    use super::*;

    fn block(method: &str) -> ExplanationBlock {
        ExplanationBlock {
            producer_node: NodeId::new("model:base").unwrap(),
            method: method.to_string(),
            target_name: Some("y".to_string()),
            payload: serde_json::json!({"feature_importance": [0.5, 0.3, 0.2]}),
        }
    }

    #[test]
    fn validates_well_formed_explanation() {
        assert!(block("shap").validate().is_ok());
    }

    #[test]
    fn rejects_empty_method() {
        assert!(block("  ").validate().is_err());
    }

    #[test]
    fn rejects_empty_target_name() {
        let mut b = block("shap");
        b.target_name = Some(String::new());
        assert!(b.validate().is_err());
    }

    #[test]
    fn round_trips_through_json() {
        let b = block("permutation_importance");
        let json = serde_json::to_string(&b).expect("serialize");
        let parsed: ExplanationBlock = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, b);
        // `target_name` is omitted when absent.
        let mut without = block("shap");
        without.target_name = None;
        let json = serde_json::to_string(&without).expect("serialize");
        assert!(!json.contains("target_name"));
    }
}

#[cfg(test)]
mod tests;
