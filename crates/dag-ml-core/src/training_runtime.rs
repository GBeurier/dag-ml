//! Native training orchestration outcome and common runtime entry point.
//!
//! The portable contracts in this module use Typed Canonical Value v1 (TCV1).
//! Historical graph, plan, controller, parameter, and bundle fingerprints keep
//! their pre-existing algorithms.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::aggregation::{AggregatedPredictionBlock, ObservationPredictionBlock, PredictionUnitId};
use crate::bundle::{
    build_aggregated_prediction_cache_payload, build_aggregated_prediction_cache_record,
    build_execution_bundle_with_prediction_contracts, build_prediction_cache_payload,
    build_prediction_cache_record, validate_prediction_cache_payload_matches_record,
    BundlePredictionCachePayload, BundlePredictionCachePayloadSet, BundlePredictionCacheRecord,
    BundlePredictionRequirement, ExecutionBundle, EXECUTION_BUNDLE_SCHEMA_VERSION,
    LEGACY_EXECUTION_BUNDLE_SCHEMA_VERSION, LEGACY_PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
    PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
};
use crate::campaign::stable_json_fingerprint;
use crate::canonical::parse_typed_json;
use crate::controller::{ControllerCapability, ControllerFitScope};
use crate::data::data_binding_requirement_key;
use crate::error::{DagMlError, Result};
use crate::graph::{NodeKind, PortKind};
use crate::ids::{BundleId, FoldId, LineageId, NodeId, RunId, SampleId, VariantId};
use crate::metrics::{
    RegressionMetricKind, ScoreSet, LEGACY_SCORE_SET_SCHEMA_VERSION, SCORE_SET_SCHEMA_VERSION,
};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::phase::Phase;
use crate::plan::ExecutionPlan;
use crate::policy::PredictionLevel;
use crate::runtime::{
    plan_oof_partition_mode, select_best_variant_outcome_by_cv_for_target, InMemoryArtifactStore,
    LineageRecord, NodeResult, ParallelScheduler, RunContext, RuntimeControllerRegistry,
    RuntimeDataProvider, SequentialScheduler, VariantExecutionSpec,
};
use crate::selection::{
    select_candidate, EvaluationScope, RefitStrategy, SelectionDecision, SelectionMetric,
    SelectionPolicy,
};
use crate::training::{
    contains_runtime_handle, ArtifactLoadMode, CacheNamespace, CvArtifactRetention,
    FittedArtifactMode, OutputBinding, PackageArtifactBinding, ParameterNamespace, ParameterPatch,
    PortablePredictorPackage, PredictionCacheRetention, PredictionKind, PredictionSource,
    PredictorTemplate, ResolvedTrainingOutput, TrainingContractProjection, TrainingDataIdentity,
    TrainingInfluenceKind, TrainingInfluenceManifest, TrainingOutcomeRef, TrainingRequest,
    TrainingSchedulerBackend, TrainingSchedulerKind, OUTPUT_BINDING_SCHEMA_VERSION,
    PARAMETER_PATCH_SCHEMA_VERSION, PORTABLE_PREDICTOR_PACKAGE_SCHEMA_VERSION,
};

pub const TRAINING_OUTCOME_SCHEMA_VERSION: u32 = 2;
pub const LEGACY_TRAINING_OUTCOME_SCHEMA_VERSION: u32 = 1;
pub const MIN_READABLE_TRAINING_OUTCOME_SCHEMA_VERSION: u32 = 1;
pub const BOUND_TRAINING_OUTPUT_SCHEMA_VERSION: u32 = 2;
pub const TRAINING_OUTCOME_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/training_outcome.v2.schema.json";

/// One resolved output binding and the actual portable blocks it selected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundTrainingOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<u32>,
    pub binding: OutputBinding,
    pub predictions: Vec<PredictionBlock>,
    pub observation_predictions: Vec<ObservationPredictionBlock>,
    pub aggregated_predictions: Vec<AggregatedPredictionBlock>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingRefitStatus {
    Completed,
    Skipped,
}

/// Exact W0 refit state embedded in [`TrainingOutcome`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRefitOutcome {
    pub requested: bool,
    pub status: TrainingRefitStatus,
    pub strategy: Option<RefitStrategy>,
}

/// Portable result of COMPILE/PLAN/FIT_CV/SELECT and optional REFIT.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingOutcome {
    pub schema_version: u32,
    pub outcome_id: String,
    pub run_id: RunId,
    pub training_request_fingerprint: String,
    pub data_identities: Vec<TrainingDataIdentity>,
    pub selection_output_id: String,
    pub effective_plan: ExecutionPlan,
    pub effective_plan_fingerprint: String,
    pub selected_variant_id: VariantId,
    pub selected_variant_fingerprint: String,
    pub parameter_patches: Vec<ParameterPatch>,
    pub refit: TrainingRefitOutcome,
    pub score_set: ScoreSet,
    pub outputs: Vec<BoundTrainingOutput>,
    pub lineage: Vec<LineageRecord>,
    pub portable_prediction_caches: Option<BundlePredictionCachePayloadSet>,
    pub training_influence: TrainingInfluenceManifest,
    pub execution_bundle: ExecutionBundle,
    pub replayable_phases: Vec<Phase>,
    pub warnings: Vec<String>,
    pub diagnostics: BTreeMap<String, serde_json::Value>,
    pub outcome_fingerprint: String,
}

/// Host-owned resources and portable identifiers for one native training run.
///
/// The operation deliberately accepts controllers and data through the same
/// runtime abstractions as ordinary phase execution. It never invokes a
/// controller directly and never implements a second fold/node loop.
pub struct TrainingExecutionInput<'a> {
    pub request: &'a TrainingRequest,
    pub outcome_id: String,
    pub run_id: RunId,
    pub bundle_id: BundleId,
    pub controllers: &'a RuntimeControllerRegistry,
    pub data_provider: &'a dyn RuntimeDataProvider,
    pub relations: &'a crate::relation::SampleRelationSet,
    pub training_influence: &'a TrainingInfluenceManifest,
    pub artifact_store: &'a mut InMemoryArtifactStore,
    pub warnings: Vec<String>,
    pub diagnostics: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug)]
enum NativeTrainingScheduler {
    Sequential(SequentialScheduler),
    Parallel(ParallelScheduler),
}

impl NativeTrainingScheduler {
    fn from_request(request: &TrainingRequest) -> Result<Self> {
        let options = &request.options.scheduler;
        if options.backend == Some(TrainingSchedulerBackend::Processes) {
            return Err(DagMlError::RuntimeValidation(
                "native training does not yet implement the processes scheduler backend"
                    .to_string(),
            ));
        }
        match options.kind {
            TrainingSchedulerKind::Sequential => Ok(Self::Sequential(SequentialScheduler)),
            TrainingSchedulerKind::Parallel => Ok(Self::Parallel(ParallelScheduler::new(
                usize::try_from(options.workers).map_err(|_| {
                    DagMlError::RuntimeValidation(
                        "training scheduler worker count does not fit usize".to_string(),
                    )
                })?,
            )?)),
        }
    }

    fn fit_cv(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
    ) -> Result<Vec<NodeResult>> {
        match self {
            Self::Sequential(scheduler) => scheduler.execute_campaign_phase_with_data_provider(
                plan,
                controllers,
                data_provider,
                ctx,
                Phase::FitCv,
            ),
            Self::Parallel(scheduler) => scheduler.execute_campaign_phase_with_data_provider(
                plan,
                controllers,
                data_provider,
                ctx,
                Phase::FitCv,
            ),
        }
    }

    fn refit(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        artifact_store: &mut InMemoryArtifactStore,
        ctx: &mut RunContext,
    ) -> Result<Vec<NodeResult>> {
        match self {
            Self::Sequential(scheduler) => scheduler
                .execute_campaign_phase_with_data_provider_and_artifact_store(
                    plan,
                    controllers,
                    data_provider,
                    artifact_store,
                    ctx,
                    Phase::Refit,
                ),
            Self::Parallel(scheduler) => scheduler
                .execute_campaign_phase_with_data_provider_and_artifact_store(
                    plan,
                    controllers,
                    data_provider,
                    artifact_store,
                    ctx,
                    Phase::Refit,
                ),
        }
    }
}

/// Execute COMPILE/PLAN -> FIT_CV -> SELECT -> optional REFIT and return the
/// complete portable W0 outcome.
///
/// Variant candidates are evaluated by the existing native selection helper;
/// the winner is then rerun once in a retained context so its lineage, OOF
/// caches, bound outputs, and optional refit artifacts all originate from one
/// auditable execution. `SELECT` is called exactly once and `REFIT` at most once.
pub fn execute_training(input: TrainingExecutionInput<'_>) -> Result<TrainingOutcome> {
    if !input.artifact_store.is_empty() {
        return Err(DagMlError::RuntimeValidation(
            "native training requires an empty artifact store for an isolated outcome".to_string(),
        ));
    }
    RunId::new(input.outcome_id.clone()).map_err(|error| {
        DagMlError::RuntimeValidation(format!(
            "native training outcome_id is not a portable identifier: {error}"
        ))
    })?;
    validate_sorted_unique_text("training execution warnings", &input.warnings)?;
    if contains_runtime_handle(&serde_json::Value::Object(
        input.diagnostics.clone().into_iter().collect(),
    )) {
        return Err(DagMlError::RuntimeValidation(
            "native training diagnostics cannot contain runtime handles".to_string(),
        ));
    }

    let mut projection = input.request.project()?;
    projection.plan = materialize_request_parameter_patches(projection.plan, input.request)?;
    projection.validate()?;
    validate_native_training_options(input.request)?;
    validate_provider_attestations(
        &projection,
        input.request,
        input.data_provider,
        input.relations,
    )?;
    for node_plan in projection.plan.node_plans.values() {
        if input.controllers.get(&node_plan.controller_id).is_none() {
            return Err(DagMlError::RuntimeValidation(format!(
                "native training controller `{}` for node `{}` is not registered",
                node_plan.controller_id, node_plan.node_id
            )));
        }
    }
    if projection.predictor_node_ids
        != projection
            .plan
            .node_plans
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
    {
        return Err(DagMlError::RuntimeValidation(
            "native training currently requires the predictor closure to equal the executable plan; refusing to persist unrelated nodes"
                .to_string(),
        ));
    }
    input.training_influence.validate_for_projection(
        &projection,
        input.request,
        input.relations,
    )?;
    let runtime_training_influence = TrainingInfluenceManifest::derive_for_projection(
        &projection,
        input.request,
        input.relations,
    )?;
    if input.training_influence != &runtime_training_influence {
        return Err(DagMlError::RuntimeValidation(
            "native training influence manifest does not match runtime-derived evidence"
                .to_string(),
        ));
    }
    if projection.plan.variants.iter().any(|variant| {
        variant
            .choices
            .values()
            .any(|choice| !choice.param_overrides.is_empty())
    }) && !input
        .training_influence
        .entries
        .iter()
        .any(|entry| entry.kind == TrainingInfluenceKind::HpoSelection)
    {
        return Err(DagMlError::RuntimeValidation(
            "selectable parameter overrides require predeclared hpo_selection influence"
                .to_string(),
        ));
    }

    let scheduler = NativeTrainingScheduler::from_request(input.request)?;
    let selection_metric = parse_selection_metric(input.request)?;
    let metric_level = effective_selection_metric_level(input.request)?;
    let selection_output = projection
        .outputs
        .iter()
        .find(|output| output.output_id == input.request.options.selection_output_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "training selection output was not resolved by projection".to_string(),
            )
        })?;
    let selection_output_id = selection_output.output_id.clone();
    let selection_producer = selection_output.node_id.clone();
    let selection_producer_port = selection_output.port_name.clone();
    validate_selection_prediction_kind(selection_metric, selection_output.prediction_kind)?;
    let selection = select_best_variant_outcome_by_cv_for_target(
        &projection.plan,
        &input.run_id,
        Some(input.request.options.seed),
        selection_metric,
        (
            &selection_producer,
            Some(selection_producer_port.as_str()),
            metric_level,
        ),
        |candidate_plan, candidate_ctx| {
            scheduler
                .fit_cv(
                    candidate_plan,
                    input.controllers,
                    input.data_provider,
                    candidate_ctx,
                )
                .map(|_| ())
        },
    )?
    .ok_or_else(|| {
        DagMlError::RuntimeValidation(
            "native training SELECT received no scored candidate; controllers must emit targets"
                .to_string(),
        )
    })?;

    validate_selection_report_levels(
        &selection.selection.validation_reports,
        &selection_producer,
        &Some(selection_producer_port.clone()),
        metric_level,
    )?;
    let mut decision = selection.decision;
    bind_selection_decision(&mut decision, input.request, metric_level)?;
    let selected_variant_id = selection.selection.selected_variant_id;
    let effective_plan = materialize_selected_variant(projection.plan, &selected_variant_id)?;
    // Keep the original union variants for replay/identity while pinning every
    // retained execution through RunContext.variant_id.
    effective_plan.validate()?;
    let selected_variant = effective_plan
        .variants
        .iter()
        .find(|variant| variant.variant_id == selected_variant_id)
        .cloned()
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "selected variant disappeared while materializing the plan".to_string(),
            )
        })?;

    let mut selected_ctx = RunContext::new(input.run_id.clone(), Some(input.request.options.seed));
    selected_ctx.variant_id = Some(selected_variant_id.clone());
    let fit_cv_results = scheduler.fit_cv(
        &effective_plan,
        input.controllers,
        input.data_provider,
        &mut selected_ctx,
    )?;
    selected_ctx.collect_cross_fold_validation_scores(plan_oof_partition_mode(&effective_plan))?;
    validate_selected_rerun_reports(
        &selection.selection.validation_reports,
        &selected_ctx.score_collector,
        &selected_variant_id,
    )?;

    let score_set = ScoreSet {
        schema_version: SCORE_SET_SCHEMA_VERSION,
        plan_id: effective_plan.id.clone(),
        selection_metric: Some(selection_metric.name().to_string()),
        reports: selection.selection.validation_reports,
    };
    score_set.validate()?;

    let prediction_requirements = build_oof_prediction_requirements(
        &effective_plan,
        selected_ctx.prediction_store.blocks(),
        selected_ctx.aggregated_prediction_store.blocks(),
    )?;
    let retain_caches =
        input.request.options.artifacts.prediction_caches == PredictionCacheRetention::Retain;
    let (prediction_caches, portable_prediction_caches) = if retain_caches {
        let mut records = build_oof_prediction_cache_records(
            &prediction_requirements,
            selected_ctx.prediction_store.blocks(),
            selected_ctx.aggregated_prediction_store.blocks(),
        )?;
        let mut payloads = build_oof_prediction_cache_payloads(
            &prediction_requirements,
            selected_ctx.prediction_store.blocks(),
            selected_ctx.aggregated_prediction_store.blocks(),
        )?;
        attach_oof_prediction_cache_namespaces(
            &effective_plan,
            &input.request.data_identities,
            &selected_variant_id,
            input.request.options.seed,
            &prediction_requirements,
            &mut records,
            &mut payloads,
        )?;
        (
            records,
            Some(BundlePredictionCachePayloadSet {
                bundle_id: input.bundle_id.clone(),
                schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
                caches: payloads,
            }),
        )
    } else {
        (Vec::new(), None)
    };

    let mut staged_artifact_store = InMemoryArtifactStore::new();
    let refit_results = if input.request.options.refit {
        scheduler.refit(
            &effective_plan,
            input.controllers,
            input.data_provider,
            &mut staged_artifact_store,
            &mut selected_ctx,
        )?
    } else {
        Vec::new()
    };

    let mut execution_bundle = build_execution_bundle_with_prediction_contracts(
        input.bundle_id.clone(),
        &effective_plan,
        Some(selected_variant_id.clone()),
        BTreeMap::from([(input.request.options.selection.id.clone(), decision)]),
        staged_artifact_store.refit_artifacts(),
        prediction_requirements,
        prediction_caches,
    )?;
    execution_bundle.scores = Some(score_set.clone());
    execution_bundle.validate_against_plan(&effective_plan)?;
    if let Some(caches) = &portable_prediction_caches {
        caches.validate_against_bundle(&execution_bundle)?;
    }

    let outputs = bind_training_outputs(
        &projection.outputs,
        input.request,
        &effective_plan,
        &fit_cv_results,
        &refit_results,
        &selected_ctx,
    )?;
    let mut lineage = selected_ctx
        .lineage
        .records()
        .filter(|record| projection.predictor_node_ids.contains(&record.node_id))
        .cloned()
        .collect::<Vec<_>>();
    for record in &mut lineage {
        record.input_lineage.sort();
        record
            .artifact_refs
            .sort_by(|left, right| left.id.cmp(&right.id));
    }
    lineage.sort_by(|left, right| left.record_id.cmp(&right.record_id));

    let effective_plan_fingerprint =
        tcv1_fingerprint(&effective_plan, "training outcome effective plan")?;
    let parameter_patches =
        merge_training_parameter_patches(&input.request.parameter_patches, &selected_variant)?;
    // Derive the honest replayable phases from the *full effective predictor
    // closure* and the artifacts/caches actually retained by this run, never
    // from the refit flag alone. `derive_replayable_phases` is the single shared
    // helper that standalone validation re-runs, so construction cannot advertise
    // a capability the closure and retained state do not support.
    let predictor_closure_nodes = predictor_closure(
        &effective_plan,
        outputs.iter().map(|output| output.binding.node_id.clone()),
    )?;
    let refit_outcome = TrainingRefitOutcome {
        requested: input.request.options.refit,
        status: if input.request.options.refit {
            TrainingRefitStatus::Completed
        } else {
            TrainingRefitStatus::Skipped
        },
        strategy: input.request.options.refit_strategy,
    };
    let replayable_phases = derive_replayable_phases(
        &effective_plan,
        &predictor_closure_nodes,
        &refit_outcome,
        &execution_bundle,
        portable_prediction_caches.as_ref(),
    )?;
    let mut outcome = TrainingOutcome {
        schema_version: TRAINING_OUTCOME_SCHEMA_VERSION,
        outcome_id: input.outcome_id,
        run_id: input.run_id,
        training_request_fingerprint: projection.request_fingerprint,
        data_identities: input.request.data_identities.clone(),
        selection_output_id,
        effective_plan,
        effective_plan_fingerprint,
        selected_variant_id,
        selected_variant_fingerprint: selected_variant.fingerprint,
        parameter_patches,
        refit: refit_outcome,
        score_set,
        outputs,
        lineage,
        portable_prediction_caches,
        training_influence: runtime_training_influence,
        execution_bundle,
        replayable_phases,
        warnings: input.warnings,
        diagnostics: input.diagnostics,
        outcome_fingerprint: zero_fingerprint(),
    };
    outcome = stabilize_training_outcome_for_tcv1(outcome)?;
    outcome.validate()?;
    *input.artifact_store = staged_artifact_store;
    Ok(outcome)
}

fn stabilize_training_outcome_for_tcv1(mut outcome: TrainingOutcome) -> Result<TrainingOutcome> {
    // Some metrics originate from floating-point computations whose first JSON
    // spelling can deserialize to the same binary64 value but reserialize with a
    // shorter decimal spelling. TCV1 fingerprints must sign the portable JSON
    // shape that readers validate after deserialization, so normalize once
    // through serde before computing the self fingerprint.
    outcome.outcome_fingerprint = zero_fingerprint();
    let json = serde_json::to_string(&outcome)?;
    let mut normalized = serde_json::from_str::<TrainingOutcome>(&json)?;
    normalized.outcome_fingerprint = normalized.compute_fingerprint()?;
    Ok(normalized)
}

fn zero_fingerprint() -> String {
    "0".repeat(64)
}

fn validate_native_training_options(request: &TrainingRequest) -> Result<()> {
    let resources = &request.options.resources;
    if resources.cpu_threads != request.options.scheduler.workers
        || resources.memory_bytes.is_some()
        || !resources.gpu_devices.is_empty()
        || resources.wall_time_ms.is_some()
    {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 supports only cpu_threads=scheduler.workers with memory_bytes=null, gpu_devices=[], and wall_time_ms=null"
                .to_string(),
        ));
    }
    if request.options.artifacts.cv_artifacts != CvArtifactRetention::Discard {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 supports only artifacts.cv_artifacts=discard".to_string(),
        ));
    }
    if request.options.artifacts.fitted_artifacts != FittedArtifactMode::AllowHostSidecar {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 cannot prove portable fitted payloads and currently requires artifacts.fitted_artifacts=allow_host_sidecar"
                .to_string(),
        ));
    }
    if request.options.artifacts.prediction_caches == PredictionCacheRetention::Discard
        && request
            .graph
            .edges
            .iter()
            .any(|edge| edge.contract.requires_oof)
    {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 requires retained prediction caches for a stacking/requires_oof graph"
                .to_string(),
        ));
    }
    Ok(())
}

fn materialize_request_parameter_patches(
    mut plan: ExecutionPlan,
    request: &TrainingRequest,
) -> Result<ExecutionPlan> {
    for patch in &request.parameter_patches {
        match patch.namespace {
            ParameterNamespace::Operator => {}
            ParameterNamespace::Structural => {
                return Err(DagMlError::RuntimeValidation(
                    "native training requires recompilation for structural parameter patches; D6 runtime accepts only operator value patches"
                        .to_string(),
                ));
            }
            ParameterNamespace::Fit | ParameterNamespace::Control => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "native training does not expose {:?} parameter patches to controllers yet; refusing to ignore them",
                    patch.namespace
                )));
            }
        }
        let node_plan = plan.node_plans.get_mut(&patch.node_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "parameter patch references absent node `{}`",
                patch.node_id
            ))
        })?;
        deep_set_plan_param(
            &mut node_plan.params,
            &patch.path,
            patch.value.clone(),
            &patch.node_id,
        )?;
        node_plan.params_fingerprint = stable_json_fingerprint(&node_plan.params)?;
    }
    plan.validate()?;
    Ok(plan)
}

fn deep_set_plan_param(
    root: &mut BTreeMap<String, serde_json::Value>,
    path: &[String],
    value: serde_json::Value,
    node_id: &NodeId,
) -> Result<()> {
    if path.is_empty() {
        return contract_error("parameter patch path cannot be empty");
    }
    if path.len() == 1 {
        root.insert(path[0].clone(), value);
        return Ok(());
    }
    let first = root.get_mut(&path[0]).ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "parameter patch for `{node_id}` is missing intermediate path `{}`",
            path[0]
        ))
    })?;
    let mut cursor = first;
    for segment in &path[1..path.len() - 1] {
        let object = cursor.as_object_mut().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "parameter patch for `{node_id}` crosses a scalar or array at `{segment}`"
            ))
        })?;
        cursor = object.get_mut(segment).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "parameter patch for `{node_id}` is missing intermediate path `{segment}`"
            ))
        })?;
    }
    let object = cursor.as_object_mut().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "parameter patch for `{node_id}` crosses a scalar or array before final key"
        ))
    })?;
    object.insert(path[path.len() - 1].clone(), value);
    Ok(())
}

fn validate_provider_attestations(
    projection: &TrainingContractProjection,
    request: &TrainingRequest,
    provider: &dyn RuntimeDataProvider,
    relations: &crate::relation::SampleRelationSet,
) -> Result<()> {
    relations.validate()?;
    let relation_fingerprint = relations.fingerprint()?;
    let identities = request
        .data_identities
        .iter()
        .map(|identity| (identity.requirement_key.as_str(), identity))
        .collect::<BTreeMap<_, _>>();
    for node_plan in projection.plan.node_plans.values() {
        for binding in &node_plan.data_bindings {
            let key = data_binding_requirement_key(&binding.node_id, &binding.input_name);
            let expected = identities.get(key.as_str()).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "native training request has no data identity for `{key}`"
                ))
            })?;
            let actual = provider.training_data_identity(binding)?.ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "runtime data provider did not attest feature/target content for `{key}`"
                ))
            })?;
            actual.validate()?;
            if &actual != *expected {
                return Err(DagMlError::RuntimeValidation(format!(
                    "runtime data provider identity for `{key}` does not match signed training request"
                )));
            }
            let provider_relations = provider.coordinator_relations(binding)?;
            if binding.require_relations && provider_relations.is_none() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "runtime data provider omitted required relations for `{key}`"
                )));
            }
            if let Some(provider_relations) = provider_relations {
                provider_relations.validate()?;
                if provider_relations.fingerprint()? != relation_fingerprint
                    || actual.relation_fingerprint != relation_fingerprint
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "runtime data provider relations for `{key}` differ from training influence relations"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn parse_selection_metric(request: &TrainingRequest) -> Result<RegressionMetricKind> {
    let metric = regression_metric_by_name(&request.options.selection.metric.name)?;
    if request.options.selection.metric.objective != metric.objective() {
        return Err(DagMlError::RuntimeValidation(format!(
            "selection metric `{}` has objective {:?}, expected {:?}",
            metric.name(),
            request.options.selection.metric.objective,
            metric.objective()
        )));
    }
    Ok(metric)
}

fn regression_metric_by_name(name: &str) -> Result<RegressionMetricKind> {
    RegressionMetricKind::from_name(name).ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "native training does not support selection metric `{name}`"
        ))
    })
}

fn validate_selection_prediction_kind(
    metric: RegressionMetricKind,
    prediction_kind: PredictionKind,
) -> Result<()> {
    RegressionMetricKind::resolve_for_prediction_kind(
        metric.name(),
        metric.objective(),
        prediction_kind,
    )
    .map(|_| ())
}

fn effective_selection_metric_level(request: &TrainingRequest) -> Result<PredictionLevel> {
    let campaign_level = request.campaign.aggregation_policy.selection_metric_level;
    if request
        .options
        .selection
        .required_metric_level
        .is_some_and(|level| level != campaign_level)
    {
        return Err(DagMlError::RuntimeValidation(
            "selection required_metric_level differs from campaign selection_metric_level"
                .to_string(),
        ));
    }
    if request.options.selection.evaluation_scope != Some(EvaluationScope::Oof) {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 requires selection.evaluation_scope=oof".to_string(),
        ));
    }
    if request.options.selection.reduction_id.is_some() {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 does not execute selection reduction_id".to_string(),
        ));
    }
    if request.options.selection.stacking_fit_contract.is_some() {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 does not execute selection stacking_fit_contract".to_string(),
        ));
    }
    if !request.options.selection.require_finite {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 requires selection.require_finite=true".to_string(),
        ));
    }
    if request.options.refit_strategy == Some(RefitStrategy::RefitEnsemble) {
        return Err(DagMlError::RuntimeValidation(
            "native training V1 does not implement refit_ensemble".to_string(),
        ));
    }
    match (
        request.options.refit,
        request.options.selection.refit_slot_plan.as_ref(),
    ) {
        (false, Some(_)) => Err(DagMlError::RuntimeValidation(
            "no-refit native training forbids selection.refit_slot_plan".to_string(),
        )),
        (true, Some(slot))
            if slot.strategy != RefitStrategy::RefitOne
                || slot.member_count != 1
                || slot.selection_level != campaign_level
                || slot.selection_metric != request.options.selection.metric
                || slot.reduction_id.is_some() =>
        {
            Err(DagMlError::RuntimeValidation(
                "selection.refit_slot_plan is not the exact native refit_one slot".to_string(),
            ))
        }
        _ => Ok(campaign_level),
    }
}

fn validate_selected_rerun_reports(
    retained: &[crate::metrics::RegressionMetricReport],
    rerun: &[crate::metrics::RegressionMetricReport],
    selected_variant_id: &VariantId,
) -> Result<()> {
    let mut retained = retained
        .iter()
        .filter(|report| report.variant_id.as_ref() == Some(selected_variant_id))
        .cloned()
        .collect::<Vec<_>>();
    let mut rerun = rerun
        .iter()
        .filter(|report| report.partition == PredictionPartition::Validation)
        .cloned()
        .map(|mut report| {
            report.variant_id = Some(selected_variant_id.clone());
            report.variant_label = None;
            report
        })
        .collect::<Vec<_>>();
    let sort = |reports: &mut Vec<crate::metrics::RegressionMetricReport>| {
        reports.sort_by(|left, right| {
            (
                &left.producer_node,
                &left.producer_port,
                &left.fold_id,
                &left.prediction_id,
                &left.level,
            )
                .cmp(&(
                    &right.producer_node,
                    &right.producer_port,
                    &right.fold_id,
                    &right.prediction_id,
                    &right.level,
                ))
        });
    };
    sort(&mut retained);
    sort(&mut rerun);
    if retained.is_empty() || retained != rerun {
        return Err(DagMlError::RuntimeValidation(
            "selected variant FIT_CV rerun diverged from the reports that justified SELECT"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_selection_report_levels(
    reports: &[crate::metrics::RegressionMetricReport],
    producer: &NodeId,
    producer_port: &Option<String>,
    expected: PredictionLevel,
) -> Result<()> {
    let target_reports = reports
        .iter()
        .filter(|report| {
            &report.producer_node == producer
                && &report.producer_port == producer_port
                && report.level == expected
        })
        .collect::<Vec<_>>();
    if target_reports.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "native SELECT target `{producer}` port {producer_port:?} has no reports at required metric level {expected:?}"
        )));
    }
    Ok(())
}

fn bind_selection_decision(
    decision: &mut SelectionDecision,
    request: &TrainingRequest,
    metric_level: PredictionLevel,
) -> Result<()> {
    decision.policy_id = request.options.selection.id.clone();
    decision.metric_level = Some(metric_level);
    decision.evaluation_scope = Some(EvaluationScope::Oof);
    decision.refit_slot_plan = request.options.selection.refit_slot_plan.clone();
    decision.reduction_id = None;
    decision.validate()
}

fn materialize_selected_variant(
    mut plan: ExecutionPlan,
    selected_variant_id: &VariantId,
) -> Result<ExecutionPlan> {
    let selected = plan
        .variants
        .iter()
        .find(|variant| &variant.variant_id == selected_variant_id)
        .cloned()
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "selected variant `{selected_variant_id}` is absent from plan"
            ))
        })?;
    let variant = VariantExecutionSpec::from_plan(&selected);
    variant.validate()?;
    for (node_id, node_plan) in &mut plan.node_plans {
        node_plan.params = variant.effective_params_for_node(node_id, &node_plan.params)?;
        node_plan.params_fingerprint = stable_json_fingerprint(&node_plan.params)?;
    }
    plan.validate()?;
    Ok(plan)
}

fn is_cv_ensemble_partition(partition: &PredictionPartition) -> bool {
    match partition {
        PredictionPartition::Validation => true,
        PredictionPartition::Train | PredictionPartition::Test | PredictionPartition::Final => {
            false
        }
    }
}

fn producer_port_matches_graph_output(
    plan: &ExecutionPlan,
    node_id: &NodeId,
    port_name: &str,
    producer_port: &Option<String>,
) -> bool {
    if let Some(producer_port) = producer_port {
        return producer_port == port_name;
    }
    let Some(node) = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| &node.id == node_id)
    else {
        return false;
    };
    let prediction_ports = node
        .ports
        .outputs
        .iter()
        .filter(|port| port.kind == PortKind::Prediction)
        .collect::<Vec<_>>();
    prediction_ports.len() == 1 && prediction_ports[0].name == port_name
}

fn bind_training_outputs(
    outputs: &[ResolvedTrainingOutput],
    request: &TrainingRequest,
    plan: &ExecutionPlan,
    fit_cv_results: &[NodeResult],
    refit_results: &[NodeResult],
    ctx: &RunContext,
) -> Result<Vec<BoundTrainingOutput>> {
    let source = if request.options.refit {
        refit_results
    } else {
        fit_cv_results
    };
    let aggregation_fingerprint = tcv1_fingerprint(
        &plan.campaign.aggregation_policy,
        "training output aggregation policy",
    )?;
    let mut bound = Vec::with_capacity(outputs.len());
    for output in outputs {
        let mut binding = OutputBinding {
            schema_version: OUTPUT_BINDING_SCHEMA_VERSION,
            binding_id: output.output_id.clone(),
            node_id: output.node_id.clone(),
            port_name: output.port_name.clone(),
            prediction_level: output.prediction_level,
            unit_level: output.unit_level,
            prediction_kind: output.prediction_kind,
            prediction_source: if request.options.refit {
                PredictionSource::FinalRefit
            } else {
                PredictionSource::CvEnsemble
            },
            refit_strategy: request.options.refit_strategy,
            aggregation_fingerprint: aggregation_fingerprint.clone(),
            target_names: output.target_names.clone(),
            target_units: output.target_units.clone(),
            class_labels: output.class_labels.clone(),
            output_order: output.output_order,
            target_space: output.target_space.clone(),
            binding_fingerprint: zero_fingerprint(),
        };
        binding.binding_fingerprint = binding.compute_fingerprint()?;

        let node_results = source
            .iter()
            .filter(|result| result.node_id == output.node_id)
            .collect::<Vec<_>>();
        let mut predictions = Vec::new();
        let mut observation_predictions = Vec::new();
        let mut aggregated_predictions = Vec::new();
        match output.prediction_level {
            PredictionLevel::Observation => {
                for result in node_results {
                    observation_predictions.extend(
                        result
                            .observation_predictions
                            .iter()
                            .filter(|block| {
                                producer_port_matches_graph_output(
                                    plan,
                                    &output.node_id,
                                    &output.port_name,
                                    &block.producer_port,
                                ) && (request.options.refit
                                    || is_cv_ensemble_partition(&block.partition))
                            })
                            .cloned(),
                    );
                }
            }
            PredictionLevel::Sample => {
                for result in node_results {
                    predictions.extend(
                        result
                            .predictions
                            .iter()
                            .filter(|block| {
                                producer_port_matches_graph_output(
                                    plan,
                                    &output.node_id,
                                    &output.port_name,
                                    &block.producer_port,
                                ) && (request.options.refit
                                    || is_cv_ensemble_partition(&block.partition))
                            })
                            .cloned(),
                    );
                    aggregated_predictions.extend(
                        result
                            .aggregated_predictions
                            .iter()
                            .filter(|block| {
                                producer_port_matches_graph_output(
                                    plan,
                                    &output.node_id,
                                    &output.port_name,
                                    &block.producer_port,
                                ) && block.level == PredictionLevel::Sample
                                    && (request.options.refit
                                        || is_cv_ensemble_partition(&block.partition))
                            })
                            .cloned(),
                    );
                }
                if !request.options.refit {
                    aggregated_predictions.extend(
                        ctx.oof_average_blocks
                            .iter()
                            .filter(|average| {
                                average.predictions.producer_node == output.node_id
                                    && producer_port_matches_graph_output(
                                        plan,
                                        &output.node_id,
                                        &output.port_name,
                                        &average.predictions.producer_port,
                                    )
                                    && is_cv_ensemble_partition(&average.predictions.partition)
                            })
                            .map(|average| average.predictions.clone()),
                    );
                }
            }
            PredictionLevel::Target | PredictionLevel::Group => {
                for result in node_results {
                    aggregated_predictions.extend(
                        result
                            .aggregated_predictions
                            .iter()
                            .filter(|block| {
                                producer_port_matches_graph_output(
                                    plan,
                                    &output.node_id,
                                    &output.port_name,
                                    &block.producer_port,
                                ) && block.level == output.prediction_level
                                    && (request.options.refit
                                        || is_cv_ensemble_partition(&block.partition))
                            })
                            .cloned(),
                    );
                }
            }
        }
        predictions.sort_by(|left, right| {
            (
                &left.partition,
                &left.fold_id,
                &left.prediction_id,
                &left.sample_ids,
            )
                .cmp(&(
                    &right.partition,
                    &right.fold_id,
                    &right.prediction_id,
                    &right.sample_ids,
                ))
        });
        observation_predictions.sort_by(|left, right| {
            (
                &left.partition,
                &left.fold_id,
                &left.prediction_id,
                &left.observation_ids,
            )
                .cmp(&(
                    &right.partition,
                    &right.fold_id,
                    &right.prediction_id,
                    &right.observation_ids,
                ))
        });
        aggregated_predictions.sort_by(|left, right| {
            (
                &left.partition,
                &left.fold_id,
                &left.prediction_id,
                &left.unit_ids,
            )
                .cmp(&(
                    &right.partition,
                    &right.fold_id,
                    &right.prediction_id,
                    &right.unit_ids,
                ))
        });
        aggregated_predictions.dedup();
        let output = BoundTrainingOutput {
            schema_version: Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION),
            binding,
            predictions,
            observation_predictions,
            aggregated_predictions,
        };
        output.validate(plan)?;
        bound.push(output);
    }
    Ok(bound)
}

/// Derive portable OOF requirements from the blocks produced by an existing
/// FIT_CV execution. Shared by the training operation and host capture paths.
pub fn build_oof_prediction_requirements(
    plan: &ExecutionPlan,
    blocks: &[PredictionBlock],
    aggregated_blocks: &[AggregatedPredictionBlock],
) -> Result<Vec<BundlePredictionRequirement>> {
    let mut requirements = Vec::new();
    for edge in plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.contract.requires_oof)
    {
        let source_plan = plan.node_plans.get(&edge.source.node_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "OOF edge source `{}` has no node plan",
                edge.source.node_id
            ))
        })?;
        let prediction_level = source_plan
            .shape_plan
            .as_ref()
            .map(|shape| shape.aggregation_policy.aggregation_level)
            .unwrap_or(PredictionLevel::Sample);
        let mut fold_ids = BTreeSet::<FoldId>::new();
        let mut sample_ids = BTreeSet::<SampleId>::new();
        let mut unit_ids = BTreeSet::<PredictionUnitId>::new();
        let mut width = None;
        let mut target_names: Option<Vec<String>> = None;

        match prediction_level {
            PredictionLevel::Sample => {
                let selected = blocks
                    .iter()
                    .filter(|block| {
                        block.producer_node == edge.source.node_id
                            && producer_port_matches_graph_output(
                                plan,
                                &edge.source.node_id,
                                &edge.source.port_name,
                                &block.producer_port,
                            )
                            && block.partition == PredictionPartition::Validation
                    })
                    .collect::<Vec<_>>();
                if selected.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "OOF requirement `{}` -> `{}` has no validation sample blocks",
                        edge.source.node_id, edge.target.node_id
                    )));
                }
                for block in selected {
                    let block_width = block.validate_shape()?;
                    merge_oof_shape(
                        &edge.source.node_id,
                        &mut width,
                        &mut target_names,
                        block_width,
                        &block.target_names,
                    )?;
                    if let Some(fold_id) = &block.fold_id {
                        fold_ids.insert(fold_id.clone());
                    }
                    sample_ids.extend(block.sample_ids.iter().cloned());
                }
            }
            PredictionLevel::Target | PredictionLevel::Group => {
                let selected = aggregated_blocks
                    .iter()
                    .filter(|block| {
                        block.producer_node == edge.source.node_id
                            && producer_port_matches_graph_output(
                                plan,
                                &edge.source.node_id,
                                &edge.source.port_name,
                                &block.producer_port,
                            )
                            && block.partition == PredictionPartition::Validation
                            && block.level == prediction_level
                    })
                    .collect::<Vec<_>>();
                if selected.is_empty() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "OOF requirement `{}` -> `{}` has no validation {prediction_level:?} blocks",
                        edge.source.node_id, edge.target.node_id
                    )));
                }
                for block in selected {
                    let block_width = block.validate_shape()?;
                    merge_oof_shape(
                        &edge.source.node_id,
                        &mut width,
                        &mut target_names,
                        block_width,
                        &block.target_names,
                    )?;
                    if let Some(fold_id) = &block.fold_id {
                        fold_ids.insert(fold_id.clone());
                    }
                    unit_ids.extend(block.unit_ids.iter().cloned());
                }
            }
            PredictionLevel::Observation => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "OOF requirement `{}` -> `{}` cannot persist observation-level predictions; aggregate before refit",
                    edge.source.node_id, edge.target.node_id
                )));
            }
        }
        let requirement = BundlePredictionRequirement {
            producer_node: edge.source.node_id.clone(),
            source_port: edge.source.port_name.clone(),
            consumer_node: edge.target.node_id.clone(),
            target_port: edge.target.port_name.clone(),
            partition: PredictionPartition::Validation,
            prediction_level,
            fold_ids: fold_ids.into_iter().collect(),
            unit_ids: unit_ids.into_iter().collect(),
            sample_ids: sample_ids.into_iter().collect(),
            prediction_width: width.unwrap_or_default(),
            target_names: target_names.unwrap_or_default(),
        };
        requirement.validate()?;
        requirements.push(requirement);
    }
    requirements.sort_by_key(BundlePredictionRequirement::key);
    Ok(requirements)
}

fn merge_oof_shape(
    producer: &NodeId,
    expected_width: &mut Option<usize>,
    expected_names: &mut Option<Vec<String>>,
    width: usize,
    names: &[String],
) -> Result<()> {
    if expected_width.is_some_and(|expected| expected != width) {
        return Err(DagMlError::RuntimeValidation(format!(
            "OOF requirement for `{producer}` has inconsistent prediction width"
        )));
    }
    *expected_width = Some(width);
    let names = if names.is_empty() {
        (0..width).map(|index| format!("p{index}")).collect()
    } else {
        names.to_vec()
    };
    if expected_names
        .as_ref()
        .is_some_and(|expected| expected != &names)
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "OOF requirement for `{producer}` has inconsistent target names"
        )));
    }
    *expected_names = Some(names);
    Ok(())
}

pub fn build_oof_prediction_cache_records(
    requirements: &[BundlePredictionRequirement],
    blocks: &[PredictionBlock],
    aggregated_blocks: &[AggregatedPredictionBlock],
) -> Result<Vec<BundlePredictionCacheRecord>> {
    requirements
        .iter()
        .map(|requirement| match requirement.prediction_level {
            PredictionLevel::Sample => build_prediction_cache_record(requirement, blocks),
            PredictionLevel::Target | PredictionLevel::Group => {
                build_aggregated_prediction_cache_record(requirement, aggregated_blocks)
            }
            PredictionLevel::Observation => Err(DagMlError::RuntimeValidation(format!(
                "prediction cache requirement `{}` cannot use observation-level predictions",
                requirement.key()
            ))),
        })
        .collect()
}

pub fn build_oof_prediction_cache_payloads(
    requirements: &[BundlePredictionRequirement],
    blocks: &[PredictionBlock],
    aggregated_blocks: &[AggregatedPredictionBlock],
) -> Result<Vec<BundlePredictionCachePayload>> {
    requirements
        .iter()
        .map(|requirement| match requirement.prediction_level {
            PredictionLevel::Sample => build_prediction_cache_payload(requirement, blocks),
            PredictionLevel::Target | PredictionLevel::Group => {
                build_aggregated_prediction_cache_payload(requirement, aggregated_blocks)
            }
            PredictionLevel::Observation => Err(DagMlError::RuntimeValidation(format!(
                "prediction cache requirement `{}` cannot use observation-level predictions",
                requirement.key()
            ))),
        })
        .collect()
}

fn attach_oof_prediction_cache_namespaces(
    plan: &ExecutionPlan,
    data_identities: &[TrainingDataIdentity],
    selected_variant_id: &VariantId,
    seed: u64,
    requirements: &[BundlePredictionRequirement],
    records: &mut [BundlePredictionCacheRecord],
    payloads: &mut [BundlePredictionCachePayload],
) -> Result<()> {
    let requirements_by_key = requirements
        .iter()
        .map(|requirement| (requirement.key(), requirement))
        .collect::<BTreeMap<_, _>>();
    for record in records {
        let requirement = requirements_by_key
            .get(&record.requirement_key)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "prediction cache `{}` references unknown OOF requirement `{}`",
                    record.cache_id, record.requirement_key
                ))
            })?;
        let fingerprints = oof_cache_namespace_fingerprints(
            plan,
            data_identities,
            selected_variant_id,
            seed,
            requirement,
            record,
        )?;
        record.cache_namespace_fingerprints = fingerprints.clone();
        let payload = payloads
            .iter_mut()
            .find(|payload| payload.requirement_key == record.requirement_key)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "prediction cache `{}` has no portable payload for requirement `{}`",
                    record.cache_id, record.requirement_key
                ))
            })?;
        payload.cache_namespace_fingerprints = fingerprints;
        validate_prediction_cache_payload_matches_record(payload, record)?;
    }
    Ok(())
}

fn oof_cache_namespace_fingerprints(
    plan: &ExecutionPlan,
    data_identities: &[TrainingDataIdentity],
    selected_variant_id: &VariantId,
    seed: u64,
    requirement: &BundlePredictionRequirement,
    record: &BundlePredictionCacheRecord,
) -> Result<Vec<String>> {
    let producer_plan = plan
        .node_plans
        .get(&requirement.producer_node)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` producer node `{}` is absent from plan",
                record.cache_id, requirement.producer_node
            ))
        })?;
    let consumer_plan = plan
        .node_plans
        .get(&requirement.consumer_node)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` consumer node `{}` is absent from plan",
                record.cache_id, requirement.consumer_node
            ))
        })?;
    let identity_binding = match (
        producer_plan.data_bindings.as_slice(),
        consumer_plan.data_bindings.as_slice(),
    ) {
        ([binding], _) => binding,
        ([], [binding]) => binding,
        (producer_bindings, consumer_bindings) => {
            let producer_count = producer_bindings.len();
            let consumer_count = consumer_bindings.len();
            return Err(DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` cannot derive a unique CacheNamespace for edge `{}.{}` -> `{}.{}` with {producer_count} producer data binding(s) and {consumer_count} consumer data binding(s)",
                record.cache_id,
                requirement.producer_node,
                requirement.source_port,
                requirement.consumer_node,
                requirement.target_port
            )));
        }
    };
    if producer_plan.data_bindings.len() > 1 || consumer_plan.data_bindings.len() > 1 {
        return Err(DagMlError::RuntimeValidation(format!(
            "prediction cache `{}` cannot derive a unique CacheNamespace for edge `{}.{}` -> `{}.{}` with ambiguous data bindings",
            record.cache_id,
            requirement.producer_node,
            requirement.source_port,
            requirement.consumer_node,
            requirement.target_port
        )));
    }
    let data_requirement_key =
        data_binding_requirement_key(&identity_binding.node_id, &identity_binding.input_name);
    let identity = data_identities
        .iter()
        .find(|identity| identity.requirement_key == data_requirement_key)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` has no training data identity for `{data_requirement_key}`",
                record.cache_id
            ))
        })?;
    let mut fingerprints = Vec::with_capacity(record.blocks.len());
    for block in &record.blocks {
        let fold_id = block.fold_id.clone().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` has a cache block without fold_id",
                record.cache_id
            ))
        })?;
        let namespace = CacheNamespace::new(
            requirement.key(),
            identity.requirement_key.clone(),
            requirement.producer_node.clone(),
            requirement.source_port.clone(),
            requirement.consumer_node.clone(),
            requirement.target_port.clone(),
            producer_plan.params_fingerprint.clone(),
            producer_plan.training_loss_fingerprint(Phase::FitCv)?,
            identity.identity_fingerprint.clone(),
            fold_id,
            selected_variant_id.to_string(),
            seed,
        )?;
        namespace.validate_for_identity(identity)?;
        fingerprints.push(namespace.namespace_fingerprint);
    }
    Ok(fingerprints)
}

impl TrainingOutcome {
    /// Strictly parse a self-fingerprinted W0 outcome without losing the JSON
    /// integer-versus-binary64 token distinction before verification.
    pub fn from_json(json: &str) -> Result<Self> {
        let typed = parse_typed_json(json).map_err(|error| {
            DagMlError::CampaignValidation(format!(
                "training outcome is not strict TCV1 JSON: {error}"
            ))
        })?;
        let raw_fingerprint =
            typed
                .fingerprint_without("outcome_fingerprint")
                .map_err(|error| {
                    DagMlError::CampaignValidation(format!(
                        "training outcome fingerprint preimage is invalid: {error}"
                    ))
                })?;
        let outcome: Self = serde_json::from_str(json)?;
        if outcome.outcome_fingerprint != raw_fingerprint {
            return contract_error(
                "training outcome fingerprint does not match original TCV1 JSON",
            );
        }
        outcome.validate()?;
        Ok(outcome)
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        tcv1_fingerprint_without(self, "outcome_fingerprint", "training outcome")
    }

    pub fn data_identities_fingerprint(&self) -> Result<String> {
        tcv1_fingerprint(&self.data_identities, "training outcome data identities")
    }

    pub fn execution_bundle_fingerprint(&self) -> Result<String> {
        tcv1_fingerprint(&self.execution_bundle, "training outcome execution bundle")
    }

    /// Build the compact cross-link embedded by a portable predictor package.
    pub fn to_reference(&self) -> Result<TrainingOutcomeRef> {
        self.validate()?;
        validate_sha256(
            "training outcome request",
            &self.training_request_fingerprint,
        )?;
        Ok(TrainingOutcomeRef {
            outcome_id: self.outcome_id.clone(),
            outcome_fingerprint: self.outcome_fingerprint.clone(),
            training_request_fingerprint: self.training_request_fingerprint.clone(),
            effective_plan_fingerprint: self.effective_plan_fingerprint.clone(),
            execution_bundle_id: self.execution_bundle.bundle_id.clone(),
            execution_bundle_fingerprint: self.execution_bundle_fingerprint()?,
            data_identities_fingerprint: self.data_identities_fingerprint()?,
            output_binding_fingerprints: self
                .outputs
                .iter()
                .map(|output| output.binding.binding_fingerprint.clone())
                .collect(),
            training_influence_fingerprint: self.training_influence.manifest_fingerprint.clone(),
        })
    }

    /// Export a self-contained portable predictor package contract from this
    /// training outcome. Runtime handles are never serialized; host-sidecar
    /// artifacts are represented only by their signed artifact descriptors and
    /// must be resolved into process-local handles by `PortablePredictorPackage::load_with`.
    pub fn to_portable_predictor_package(
        &self,
        package_id: impl Into<String>,
        fitted_artifact_mode: FittedArtifactMode,
        artifact_load_mode: ArtifactLoadMode,
    ) -> Result<PortablePredictorPackage> {
        self.validate()?;
        let mut template = PredictorTemplate {
            graph: self.effective_plan.graph_plan.graph.clone(),
            campaign: self.effective_plan.campaign.clone(),
            controller_manifests: self.effective_plan.controller_manifests.clone(),
            template_fingerprint: zero_fingerprint(),
        };
        template.template_fingerprint = template.compute_fingerprint()?;

        let output_bindings = self
            .outputs
            .iter()
            .map(|output| output.binding.clone())
            .collect::<Vec<_>>();
        let predictor_node_ids = predictor_closure(
            &self.effective_plan,
            output_bindings
                .iter()
                .map(|binding| binding.node_id.clone()),
        )?
        .into_iter()
        .collect::<Vec<_>>();
        let mut artifact_bindings = self
            .execution_bundle
            .refit_artifacts
            .iter()
            .map(|record| PackageArtifactBinding {
                artifact_id: record.artifact.id.clone(),
                load_mode: artifact_load_mode,
            })
            .collect::<Vec<_>>();
        artifact_bindings.sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
        let mut package = PortablePredictorPackage {
            schema_version: PORTABLE_PREDICTOR_PACKAGE_SCHEMA_VERSION,
            package_id: package_id.into(),
            template,
            training_request_fingerprint: self.training_request_fingerprint.clone(),
            training_outcome: self.to_reference()?,
            effective_plan: self.effective_plan.clone(),
            execution_bundle: self.execution_bundle.clone(),
            output_bindings,
            predictor_node_ids,
            training_influence: self.training_influence.clone(),
            data_identities: self.data_identities.clone(),
            fitted_artifact_mode,
            artifact_bindings,
            package_fingerprint: zero_fingerprint(),
        };
        package.package_fingerprint = package.compute_fingerprint()?;
        package.validate()?;
        Ok(package)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version < MIN_READABLE_TRAINING_OUTCOME_SCHEMA_VERSION
            || self.schema_version > TRAINING_OUTCOME_SCHEMA_VERSION
        {
            return contract_error(format!(
                "training outcome schema_version {} is unsupported; maximum readable version is {}",
                self.schema_version, TRAINING_OUTCOME_SCHEMA_VERSION
            ));
        }
        RunId::new(self.outcome_id.clone()).map_err(|error| {
            DagMlError::CampaignValidation(format!(
                "training outcome_id is not a portable identifier: {error}"
            ))
        })?;
        validate_sha256(
            "training outcome request",
            &self.training_request_fingerprint,
        )?;
        validate_sha256("training outcome plan", &self.effective_plan_fingerprint)?;
        validate_sha256(
            "training outcome selected variant",
            &self.selected_variant_fingerprint,
        )?;
        validate_sha256("training outcome", &self.outcome_fingerprint)?;
        self.effective_plan.validate()?;
        if self.effective_plan_fingerprint
            != tcv1_fingerprint(&self.effective_plan, "training outcome effective plan")?
        {
            return contract_error(
                "training outcome effective_plan_fingerprint does not match TCV1 plan content",
            );
        }

        let selected = self
            .effective_plan
            .variants
            .iter()
            .filter(|variant| variant.variant_id == self.selected_variant_id)
            .collect::<Vec<_>>();
        let [selected] = selected.as_slice() else {
            return contract_error(
                "training outcome selected_variant_id is absent or duplicated in effective plan",
            );
        };
        if selected.fingerprint != self.selected_variant_fingerprint {
            return contract_error(
                "training outcome selected_variant_fingerprint does not match effective plan",
            );
        }
        let expected_patches = selected_variant_parameter_patches(selected)?;
        validate_outcome_parameter_patches(
            &self.effective_plan,
            &self.parameter_patches,
            &expected_patches,
        )?;
        if !self.parameter_patches.is_empty()
            && !self
                .training_influence
                .entries
                .iter()
                .any(|entry| entry.kind == TrainingInfluenceKind::HpoSelection)
        {
            return contract_error(
                "training outcome parameter patches require hpo_selection influence",
            );
        }

        self.validate_refit()?;
        self.score_set.validate()?;
        self.validate_version_family()?;
        if self.score_set.plan_id != self.effective_plan.id {
            return contract_error("training outcome score_set.plan_id does not match plan");
        }
        if !self
            .score_set
            .reports
            .iter()
            .any(|report| report.variant_id.as_ref() == Some(&self.selected_variant_id))
        {
            return contract_error("training outcome score_set has no report for selected variant");
        }
        self.validate_selection_decision()?;

        let closure = self.validate_outputs()?;
        // V1 standalone-validation invariant: the effective predictor closure
        // must cover every effective plan node. Until unrelated-node execution
        // and persistence have an explicit policy, this lets replayable-phase
        // derivation trust `closure` as the full predictor without reconstructing
        // a partial schedule from outcome data alone.
        if closure
            != self
                .effective_plan
                .node_plans
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>()
        {
            return contract_error(
                "training outcome predictor closure must equal all effective plan nodes in V1",
            );
        }
        self.training_influence.validate()?;
        validate_influence_against_closure(
            &self.training_influence,
            &self.effective_plan,
            &closure,
        )?;
        let base_fit_nodes = self
            .training_influence
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.kind,
                    TrainingInfluenceKind::TransformFit
                        | TrainingInfluenceKind::ModelFit
                        | TrainingInfluenceKind::TrainedMetaAggregation
                )
            })
            .filter_map(|entry| entry.node_id.clone())
            .collect::<BTreeSet<_>>();
        if self
            .outputs
            .iter()
            .any(|output| !base_fit_nodes.contains(&output.binding.node_id))
        {
            return contract_error("training outcome output node has no fitting influence");
        }

        self.execution_bundle
            .validate_against_plan(&self.effective_plan)?;
        if self.execution_bundle.selected_variant_id.as_ref() != Some(&self.selected_variant_id) {
            return contract_error(
                "training outcome execution bundle selected variant does not match outcome",
            );
        }
        if self.execution_bundle.scores.as_ref() != Some(&self.score_set) {
            return contract_error(
                "training outcome execution bundle scores do not equal score_set",
            );
        }
        self.validate_data_identities()?;
        validate_all_identity_relations(
            &self.data_identities,
            &self.training_influence.relation_fingerprint,
        )?;
        self.validate_artifacts(&closure)?;
        self.validate_lineage(&closure)?;
        match &self.portable_prediction_caches {
            Some(caches) => caches.validate_against_bundle(&self.execution_bundle)?,
            None if !self.execution_bundle.prediction_caches.is_empty() => {
                return contract_error(
                    "training outcome portable caches are null while bundle announces caches",
                );
            }
            None => {}
        }

        let expected_replay = derive_replayable_phases(
            &self.effective_plan,
            &closure,
            &self.refit,
            &self.execution_bundle,
            self.portable_prediction_caches.as_ref(),
        )?;
        if self.replayable_phases != expected_replay {
            return contract_error(
                "training outcome replayable_phases do not match the phases derivable from the full predictor closure and retained state",
            );
        }
        validate_sorted_unique_text("training outcome warnings", &self.warnings)?;
        let portable = serde_json::to_value(self)?;
        if contains_runtime_handle(&portable) {
            return contract_error("training outcome must not contain runtime handles");
        }
        if self.outcome_fingerprint != self.compute_fingerprint()? {
            return contract_error("training outcome fingerprint does not match TCV1 content");
        }
        Ok(())
    }

    fn validate_version_family(&self) -> Result<()> {
        let expected_score_version = match self.schema_version {
            LEGACY_TRAINING_OUTCOME_SCHEMA_VERSION => LEGACY_SCORE_SET_SCHEMA_VERSION,
            TRAINING_OUTCOME_SCHEMA_VERSION => SCORE_SET_SCHEMA_VERSION,
            _ => unreachable!("training outcome schema_version was range-checked"),
        };
        if self.score_set.schema_version != expected_score_version {
            return contract_error(format!(
                "training outcome schema_version {} requires score_set schema_version {}, got {}",
                self.schema_version, expected_score_version, self.score_set.schema_version
            ));
        }
        let expected_bundle_version = match self.schema_version {
            LEGACY_TRAINING_OUTCOME_SCHEMA_VERSION => LEGACY_EXECUTION_BUNDLE_SCHEMA_VERSION,
            TRAINING_OUTCOME_SCHEMA_VERSION => EXECUTION_BUNDLE_SCHEMA_VERSION,
            _ => unreachable!("training outcome schema_version was range-checked"),
        };
        if self.execution_bundle.schema_version != expected_bundle_version {
            return contract_error(format!(
                "training outcome schema_version {} requires execution_bundle schema_version {}, got {}",
                self.schema_version,
                expected_bundle_version,
                self.execution_bundle.schema_version
            ));
        }
        let expected_cache_version = match self.schema_version {
            LEGACY_TRAINING_OUTCOME_SCHEMA_VERSION => {
                LEGACY_PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION
            }
            TRAINING_OUTCOME_SCHEMA_VERSION => PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
            _ => unreachable!("training outcome schema_version was range-checked"),
        };
        if let Some(caches) = &self.portable_prediction_caches {
            if caches.schema_version != expected_cache_version {
                return contract_error(format!(
                    "training outcome schema_version {} requires prediction cache payload set schema_version {}, got {}",
                    self.schema_version, expected_cache_version, caches.schema_version
                ));
            }
        }
        for output in &self.outputs {
            match (self.schema_version, output.schema_version) {
                (LEGACY_TRAINING_OUTCOME_SCHEMA_VERSION, None) => {}
                (LEGACY_TRAINING_OUTCOME_SCHEMA_VERSION, Some(version)) => {
                    return contract_error(format!(
                        "training outcome V1 requires absent bound output schema_version, got {version}"
                    ));
                }
                (TRAINING_OUTCOME_SCHEMA_VERSION, Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION)) => {}
                (TRAINING_OUTCOME_SCHEMA_VERSION, Some(version)) => {
                    return contract_error(format!(
                        "training outcome V2 requires bound output schema_version {}, got {version}",
                        BOUND_TRAINING_OUTPUT_SCHEMA_VERSION
                    ));
                }
                (TRAINING_OUTCOME_SCHEMA_VERSION, None) => {
                    return contract_error(
                        "training outcome V2 requires bound output schema_version",
                    );
                }
                _ => unreachable!("training outcome schema_version was range-checked"),
            }
        }
        Ok(())
    }

    fn validate_data_identities(&self) -> Result<()> {
        if self.data_identities.is_empty() {
            return contract_error("training outcome requires data identities");
        }
        let mut previous: Option<&str> = None;
        for identity in &self.data_identities {
            identity.validate()?;
            if previous.is_some_and(|key| key >= identity.requirement_key.as_str()) {
                return contract_error(
                    "training outcome data identities must be sorted and unique",
                );
            }
            previous = Some(identity.requirement_key.as_str());
            let requirement = self
                .execution_bundle
                .data_requirements
                .iter()
                .find(|requirement| requirement.key() == identity.requirement_key)
                .ok_or_else(|| {
                    DagMlError::CampaignValidation(format!(
                        "training outcome data identity `{}` has no bundle requirement",
                        identity.requirement_key
                    ))
                })?;
            if requirement.schema_fingerprint != identity.schema_fingerprint
                || requirement.plan_fingerprint != identity.plan_fingerprint
                || requirement.relation_fingerprint.as_ref() != Some(&identity.relation_fingerprint)
            {
                return contract_error(
                    "training outcome data identity does not match execution bundle requirement",
                );
            }
        }
        if self.data_identities.len() != self.execution_bundle.data_requirements.len() {
            return contract_error(
                "training outcome data identities do not exactly cover bundle data requirements",
            );
        }
        Ok(())
    }

    fn validate_selection_decision(&self) -> Result<()> {
        if self.selection_output_id.trim().is_empty() {
            return contract_error("training outcome selection_output_id is empty");
        }
        let bindings = self
            .outputs
            .iter()
            .filter(|output| output.binding.binding_id == self.selection_output_id)
            .collect::<Vec<_>>();
        let [selected_output] = bindings.as_slice() else {
            return contract_error(
                "training outcome selection_output_id does not resolve exactly one output",
            );
        };
        if self.execution_bundle.selections.len() != 1 {
            return contract_error(
                "training outcome execution bundle must contain exactly one SELECT decision",
            );
        }
        let (selection_key, decision) = self
            .execution_bundle
            .selections
            .iter()
            .next()
            .expect("selection length was checked");
        if selection_key != &decision.policy_id
            || decision.selected_candidate_id != self.selected_variant_id.as_str()
            || decision.metric_level != Some(selected_output.binding.prediction_level)
            || decision.evaluation_scope != Some(EvaluationScope::Oof)
            || self.score_set.selection_metric.as_deref() != Some(decision.metric_name.as_str())
            || selected_output.binding.prediction_level
                != self
                    .effective_plan
                    .campaign
                    .aggregation_policy
                    .selection_metric_level
        {
            return contract_error(
                "training outcome SELECT decision metadata is inconsistent with selected output",
            );
        }
        RegressionMetricKind::resolve_for_prediction_kind(
            &decision.metric_name,
            decision.objective,
            selected_output.binding.prediction_kind,
        )?;
        let mut reports_by_variant = BTreeMap::<VariantId, _>::new();
        for report in self.score_set.reports.iter().filter(|report| {
            report.producer_node == selected_output.binding.node_id
                && producer_port_matches_graph_output(
                    &self.effective_plan,
                    &selected_output.binding.node_id,
                    &selected_output.binding.port_name,
                    &report.producer_port,
                )
                && report.partition == PredictionPartition::Validation
                && report.level == selected_output.binding.prediction_level
                && report
                    .fold_id
                    .as_ref()
                    .is_some_and(|fold| fold.as_str() == "avg")
        }) {
            let variant_id = report.variant_id.clone().ok_or_else(|| {
                DagMlError::CampaignValidation(
                    "selection output average report has no variant_id".to_string(),
                )
            })?;
            if reports_by_variant
                .insert(variant_id, report.clone())
                .is_some()
            {
                return contract_error(
                    "training outcome has multiple selection average reports for one variant",
                );
            }
        }
        let expected_variants = self
            .effective_plan
            .variants
            .iter()
            .map(|variant| variant.variant_id.clone())
            .collect::<BTreeSet<_>>();
        if reports_by_variant.keys().cloned().collect::<BTreeSet<_>>() != expected_variants {
            return contract_error(
                "training outcome selection reports do not exactly cover plan variants",
            );
        }
        let candidates = reports_by_variant
            .into_iter()
            .map(|(variant_id, report)| report.into_candidate_score(variant_id.as_str()))
            .collect::<Result<Vec<_>>>()?;
        let reconstructed = select_candidate(
            &SelectionPolicy {
                id: decision.policy_id.clone(),
                metric: SelectionMetric {
                    name: decision.metric_name.clone(),
                    objective: decision.objective,
                },
                required_metric_level: decision.metric_level,
                require_finite: true,
                evaluation_scope: decision.evaluation_scope,
                refit_slot_plan: decision.refit_slot_plan.clone(),
                stacking_fit_contract: None,
                reduction_id: decision.reduction_id.clone(),
            },
            &candidates,
        )?;
        if &reconstructed != decision {
            return contract_error(
                "training outcome SELECT decision does not equal ranking reconstructed from scores",
            );
        }
        Ok(())
    }

    fn validate_refit(&self) -> Result<()> {
        match (self.refit.requested, self.refit.status, self.refit.strategy) {
            (true, TrainingRefitStatus::Completed, Some(_)) => {
                if self
                    .outputs
                    .iter()
                    .any(|output| output.binding.prediction_source != PredictionSource::FinalRefit)
                {
                    return contract_error(
                        "completed refit outputs must use final_refit prediction source",
                    );
                }
            }
            (false, TrainingRefitStatus::Skipped, None) => {
                if self
                    .outputs
                    .iter()
                    .any(|output| output.binding.prediction_source == PredictionSource::FinalRefit)
                {
                    return contract_error("no-refit outputs cannot use final_refit");
                }
            }
            _ => return contract_error("training outcome refit state is inconsistent"),
        }
        Ok(())
    }

    fn validate_outputs(&self) -> Result<BTreeSet<NodeId>> {
        if self.outputs.is_empty() {
            return contract_error("training outcome requires at least one bound output");
        }
        let mut previous: Option<&str> = None;
        let mut roots = Vec::new();
        for output in &self.outputs {
            if previous.is_some_and(|value| value >= output.binding.binding_id.as_str()) {
                return contract_error(
                    "training outcome outputs must be strictly sorted by binding_id",
                );
            }
            previous = Some(output.binding.binding_id.as_str());
            output.validate(&self.effective_plan)?;
            roots.push(output.binding.node_id.clone());
        }
        predictor_closure(&self.effective_plan, roots)
    }

    fn validate_artifacts(&self, closure: &BTreeSet<NodeId>) -> Result<()> {
        if !self.refit.requested {
            if !self.execution_bundle.refit_artifacts.is_empty() {
                return contract_error("no-refit training outcome contains refit artifacts");
            }
            return Ok(());
        }
        if self.execution_bundle.refit_artifacts.is_empty() {
            return contract_error("completed refit requires at least one artifact");
        }
        let expected_artifact_nodes = closure
            .iter()
            .filter(|node_id| {
                let plan = &self.effective_plan.node_plans[*node_id];
                plan.supported_phases.contains(&Phase::Refit)
                    && plan
                        .controller_capabilities
                        .contains(&ControllerCapability::EmitsArtifacts)
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        let artifact_nodes = self
            .execution_bundle
            .refit_artifacts
            .iter()
            .map(|record| record.node_id.clone())
            .collect::<BTreeSet<_>>();
        if artifact_nodes != expected_artifact_nodes {
            return contract_error(
                "refit artifact nodes do not exactly match predictor closure REFIT artifact emitters",
            );
        }
        for output in &self.outputs {
            if !artifact_nodes.contains(&output.binding.node_id) {
                return contract_error("final output node has no refit artifact");
            }
        }
        Ok(())
    }

    fn validate_lineage(&self, closure: &BTreeSet<NodeId>) -> Result<()> {
        if self.lineage.is_empty() {
            return contract_error("training outcome requires portable lineage");
        }
        let record_ids = self
            .lineage
            .iter()
            .map(|record| record.record_id.clone())
            .collect::<Vec<_>>();
        if record_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return contract_error("training outcome lineage must be sorted by record_id");
        }
        let by_id = self
            .lineage
            .iter()
            .map(|record| (record.record_id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        if by_id.len() != self.lineage.len() {
            return contract_error("training outcome lineage contains duplicate record ids");
        }
        let mut coordinates = BTreeMap::new();
        for record in &self.lineage {
            record.validate()?;
            if record.run_id != self.run_id
                || record.variant_id.as_ref() != Some(&self.selected_variant_id)
                || !closure.contains(&record.node_id)
            {
                return contract_error(
                    "training outcome lineage run, variant, or predictor closure is inconsistent",
                );
            }
            if !matches!(record.phase, Phase::FitCv | Phase::Select | Phase::Refit) {
                return contract_error("training outcome lineage contains a non-training phase");
            }
            let plan = &self.effective_plan.node_plans[&record.node_id];
            if record.controller_id != plan.controller_id
                || record.controller_version != plan.controller_version
                || record.params_fingerprint != plan.params_fingerprint
            {
                return contract_error("training outcome lineage does not match node plan");
            }
            let expected_losses = plan
                .training_losses_for_phase(record.phase)
                .collect::<Vec<_>>();
            if record.loss_attestations.len() != expected_losses.len() {
                return contract_error(
                    "training outcome lineage loss attestations do not match node plan",
                );
            }
            for (attestation, role) in record.loss_attestations.iter().zip(expected_losses) {
                attestation.validate_against(role, &record.node_id, record.phase)?;
            }
            let key = (record.phase, record.fold_id.clone(), record.node_id.clone());
            if coordinates.insert(key, record).is_some() {
                return contract_error("training outcome lineage duplicates phase/fold/node");
            }
            if record
                .input_lineage
                .iter()
                .any(|input| !by_id.contains_key(input))
            {
                return contract_error("training outcome lineage references an unknown input");
            }
        }
        validate_lineage_coordinates(self, closure, &coordinates)
    }
}

impl BoundTrainingOutput {
    pub(crate) fn validate(&self, plan: &ExecutionPlan) -> Result<()> {
        if let Some(schema_version) = self.schema_version {
            if schema_version != BOUND_TRAINING_OUTPUT_SCHEMA_VERSION {
                return contract_error(format!(
                    "bound training output schema_version {schema_version} is unsupported; current {}",
                    BOUND_TRAINING_OUTPUT_SCHEMA_VERSION
                ));
            }
        }
        self.binding.validate(&plan.graph_plan.graph)?;
        if self.predictions.is_empty()
            && self.observation_predictions.is_empty()
            && self.aggregated_predictions.is_empty()
        {
            return contract_error("bound training output contains no prediction block");
        }
        match self.binding.prediction_level {
            PredictionLevel::Observation
                if !self.predictions.is_empty() || !self.aggregated_predictions.is_empty() =>
            {
                return contract_error(
                    "observation output binding cannot contain sample or aggregated predictions",
                );
            }
            PredictionLevel::Sample if !self.observation_predictions.is_empty() => {
                return contract_error(
                    "sample output binding cannot contain observation predictions",
                );
            }
            PredictionLevel::Target | PredictionLevel::Group
                if !self.predictions.is_empty() || !self.observation_predictions.is_empty() =>
            {
                return contract_error(
                    "target/group output binding cannot contain sample or observation predictions",
                );
            }
            _ => {}
        }
        let expected_names = expected_output_columns(&self.binding);
        for block in &self.predictions {
            block.validate_shape()?;
            validate_bound_block(
                plan,
                &self.binding,
                BoundBlockRef {
                    producer: &block.producer_node,
                    producer_port: &block.producer_port,
                    partition: &block.partition,
                    fold_id: block.fold_id.as_ref(),
                    target_names: &block.target_names,
                },
                &expected_names,
            )?;
        }
        for block in &self.observation_predictions {
            block.validate_shape()?;
            validate_bound_block(
                plan,
                &self.binding,
                BoundBlockRef {
                    producer: &block.producer_node,
                    producer_port: &block.producer_port,
                    partition: &block.partition,
                    fold_id: block.fold_id.as_ref(),
                    target_names: &block.target_names,
                },
                &expected_names,
            )?;
        }
        for block in &self.aggregated_predictions {
            block.validate_shape()?;
            if block.level != self.binding.prediction_level {
                return contract_error(
                    "bound aggregated prediction level does not match output binding",
                );
            }
            validate_bound_block(
                plan,
                &self.binding,
                BoundBlockRef {
                    producer: &block.producer_node,
                    producer_port: &block.producer_port,
                    partition: &block.partition,
                    fold_id: block.fold_id.as_ref(),
                    target_names: &block.target_names,
                },
                &expected_names,
            )?;
        }
        match self.binding.prediction_level {
            PredictionLevel::Observation if self.observation_predictions.is_empty() => {
                return contract_error(
                    "observation output binding requires observation predictions",
                );
            }
            PredictionLevel::Target | PredictionLevel::Group
                if self.aggregated_predictions.is_empty() =>
            {
                return contract_error(
                    "target/group output binding requires aggregated predictions",
                );
            }
            _ => {}
        }
        Ok(())
    }
}

struct BoundBlockRef<'a> {
    producer: &'a NodeId,
    producer_port: &'a Option<String>,
    partition: &'a PredictionPartition,
    fold_id: Option<&'a crate::ids::FoldId>,
    target_names: &'a [String],
}

fn validate_bound_block(
    plan: &ExecutionPlan,
    binding: &OutputBinding,
    block: BoundBlockRef<'_>,
    expected_names: &[String],
) -> Result<()> {
    if block.producer != &binding.node_id
        || !producer_port_matches_graph_output(
            plan,
            &binding.node_id,
            &binding.port_name,
            block.producer_port,
        )
        || block.target_names != expected_names
    {
        return contract_error(
            "bound prediction producer, producer_port or target order does not match output binding",
        );
    }
    if binding.prediction_source == PredictionSource::FinalRefit
        && (block.partition != &PredictionPartition::Final || block.fold_id.is_some())
    {
        return contract_error("final_refit output blocks must use final partition without fold");
    }
    if binding.prediction_source == PredictionSource::CvEnsemble
        && (!is_cv_ensemble_partition(block.partition) || block.fold_id.is_none())
    {
        return contract_error(
            "cv_ensemble output blocks must use validation partition with a fold id",
        );
    }
    Ok(())
}

fn expected_output_columns(binding: &OutputBinding) -> Vec<String> {
    if binding.prediction_kind == PredictionKind::ClassProbability {
        binding
            .target_names
            .iter()
            .zip(&binding.class_labels)
            .flat_map(|(target, labels)| {
                labels.iter().map(move |label| format!("{target}:{label}"))
            })
            .collect()
    } else {
        binding.target_names.clone()
    }
}

fn selected_variant_parameter_patches(
    variant: &crate::generation::VariantPlan,
) -> Result<Vec<ParameterPatch>> {
    let mut patches = Vec::new();
    for choice in variant.choices.values() {
        for override_spec in &choice.param_overrides {
            for (key, value) in &override_spec.params {
                append_parameter_leaves(
                    &override_spec.node_id,
                    vec![key.clone()],
                    value,
                    &mut patches,
                )?;
            }
        }
    }
    patches.sort_by(|left, right| {
        (&left.node_id, left.namespace, &left.path).cmp(&(
            &right.node_id,
            right.namespace,
            &right.path,
        ))
    });
    if patches.windows(2).any(|pair| {
        pair[0].node_id == pair[1].node_id
            && pair[0].namespace == pair[1].namespace
            && pair[0].path == pair[1].path
    }) {
        return contract_error("selected variant overrides contain duplicate leaf paths");
    }
    Ok(patches)
}

fn merge_training_parameter_patches(
    request_patches: &[ParameterPatch],
    selected_variant: &crate::generation::VariantPlan,
) -> Result<Vec<ParameterPatch>> {
    let mut patches = request_patches.to_vec();
    patches.extend(selected_variant_parameter_patches(selected_variant)?);
    sort_and_validate_training_parameter_patch_keys(&mut patches, false)?;
    Ok(patches)
}

fn validate_outcome_parameter_patches(
    plan: &ExecutionPlan,
    patches: &[ParameterPatch],
    selected_variant_patches: &[ParameterPatch],
) -> Result<()> {
    let mut patches = patches.to_vec();
    sort_and_validate_training_parameter_patch_keys(&mut patches, true)?;
    let keys = patches
        .iter()
        .map(parameter_patch_key)
        .collect::<BTreeSet<_>>();
    for selected in selected_variant_patches {
        if !keys.contains(&parameter_patch_key(selected)) {
            return contract_error(
                "training outcome parameter_patches are missing a selected variant override",
            );
        }
    }
    for patch in &patches {
        validate_materialized_patch(plan, patch)?;
    }
    Ok(())
}

fn sort_and_validate_training_parameter_patch_keys(
    patches: &mut [ParameterPatch],
    require_already_sorted: bool,
) -> Result<()> {
    for patch in patches.iter() {
        patch.validate()?;
        if patch.namespace != ParameterNamespace::Operator {
            return contract_error(
                "training outcome parameter_patches must use operator namespace",
            );
        }
    }
    let original = patches.to_vec();
    patches.sort_by(|left, right| parameter_patch_key(left).cmp(&parameter_patch_key(right)));
    if require_already_sorted && patches != original {
        return contract_error(
            "training outcome parameter_patches must be sorted by (node_id, namespace, path)",
        );
    }
    for pair in patches.windows(2) {
        let left = &pair[0];
        let right = &pair[1];
        if parameter_patch_key(left) == parameter_patch_key(right) {
            return contract_error(
                "training outcome parameter_patches contain duplicate leaf paths",
            );
        }
        if left.node_id == right.node_id
            && left.namespace == right.namespace
            && (right.path.starts_with(&left.path) || left.path.starts_with(&right.path))
        {
            return contract_error(
                "training outcome parameter_patches contain a conflicting parent/child path",
            );
        }
    }
    Ok(())
}

fn parameter_patch_key(patch: &ParameterPatch) -> (&NodeId, ParameterNamespace, &[String]) {
    (&patch.node_id, patch.namespace, patch.path.as_slice())
}

fn append_parameter_leaves(
    node_id: &NodeId,
    path: Vec<String>,
    value: &serde_json::Value,
    output: &mut Vec<ParameterPatch>,
) -> Result<()> {
    if let serde_json::Value::Object(object) = value {
        for (key, child) in object {
            let mut child_path = path.clone();
            child_path.push(key.clone());
            append_parameter_leaves(node_id, child_path, child, output)?;
        }
        return Ok(());
    }
    output.push(ParameterPatch {
        schema_version: PARAMETER_PATCH_SCHEMA_VERSION,
        node_id: node_id.clone(),
        namespace: ParameterNamespace::Operator,
        path,
        value: value.clone(),
    });
    Ok(())
}

fn validate_materialized_patch(plan: &ExecutionPlan, patch: &ParameterPatch) -> Result<()> {
    patch.validate()?;
    if patch.namespace != ParameterNamespace::Operator {
        return contract_error("selected variant patches must use operator namespace");
    }
    let node = plan.node_plans.get(&patch.node_id).ok_or_else(|| {
        DagMlError::CampaignValidation(format!(
            "selected parameter patch references absent node `{}`",
            patch.node_id
        ))
    })?;
    let mut current = serde_json::Value::Object(node.params.clone().into_iter().collect());
    for segment in &patch.path {
        current = current
            .as_object()
            .and_then(|object| object.get(segment))
            .cloned()
            .ok_or_else(|| {
                DagMlError::CampaignValidation(format!(
                    "selected parameter patch path for `{}` is not materialized",
                    patch.node_id
                ))
            })?;
    }
    if current != patch.value {
        return contract_error("selected parameter patch value is not materialized in plan");
    }
    Ok(())
}

fn predictor_closure(
    plan: &ExecutionPlan,
    roots: impl IntoIterator<Item = NodeId>,
) -> Result<BTreeSet<NodeId>> {
    let mut pending = roots.into_iter().collect::<Vec<_>>();
    let mut closure = BTreeSet::new();
    while let Some(node_id) = pending.pop() {
        if !closure.insert(node_id.clone()) {
            continue;
        }
        let node = plan.node_plans.get(&node_id).ok_or_else(|| {
            DagMlError::CampaignValidation(format!(
                "training outcome closure references absent node `{node_id}`"
            ))
        })?;
        pending.extend(node.input_nodes.iter().cloned());
    }
    Ok(closure)
}

/// Per-node facts the replay derivation reads for one predictor-closure node.
struct NodeReplayFacts {
    supported_phases: BTreeSet<Phase>,
    /// Node carries fitted inference state that a later PREDICT/EXPLAIN must
    /// reload: it is `stateful` or emits artifacts (capabilities
    /// `Stateful || EmitsArtifacts`). This is deliberately NOT inferred from
    /// `artifact_policy`/`ReplayRequired` or from `fit_scope`: a stateless
    /// deterministic operator — e.g. a seeded augmentation, or a
    /// `replay_required` transform that simply recomputes at inference — carries
    /// no reloadable state, needs no retained artifact, and must not block
    /// forward replay.
    requires_retained_state: bool,
    /// A retained refit artifact for this node is present in the bundle.
    has_retained_artifact: bool,
}

/// Per-edge facts for one `requires_oof` dependency wholly inside the closure.
struct OofEdgeReplayFacts {
    has_bundle_requirement: bool,
    has_cache_record: bool,
    has_portable_payload: bool,
}

/// Everything the pure replay decision needs, extracted from the plan/bundle so
/// the decision itself is unit-testable in isolation without a full plan.
struct ClosureReplayFacts {
    nodes: Vec<NodeReplayFacts>,
    oof_edges: Vec<OofEdgeReplayFacts>,
}

/// Pure replay decision over already-extracted closure facts.
///
/// Canonical order is `[REFIT, PREDICT, EXPLAIN]`. A completed refit never
/// re-advertises REFIT; it exposes forward inference only when *every* closure
/// node supports the phase and every state-retaining closure node has a retained
/// refit artifact. A skipped refit exposes REFIT only when every closure node
/// supports REFIT and every closure OOF dependency is backed by an exact bundle
/// requirement, a retained cache record and a portable payload. An empty result
/// is a valid, honest "no replay mode" answer.
fn derive_replayable_phases_from_facts(
    completed_refit: bool,
    facts: &ClosureReplayFacts,
) -> Vec<Phase> {
    let all_support = |phase: Phase| {
        facts
            .nodes
            .iter()
            .all(|node| node.supported_phases.contains(&phase))
    };
    let inference_state_present = facts
        .nodes
        .iter()
        .all(|node| !node.requires_retained_state || node.has_retained_artifact);
    let oof_self_contained = facts.oof_edges.iter().all(|edge| {
        edge.has_bundle_requirement && edge.has_cache_record && edge.has_portable_payload
    });

    let mut phases = Vec::new();
    if completed_refit {
        if all_support(Phase::Predict) && inference_state_present {
            phases.push(Phase::Predict);
        }
        if all_support(Phase::Explain) && inference_state_present {
            phases.push(Phase::Explain);
        }
    } else if all_support(Phase::Refit) && oof_self_contained {
        phases.push(Phase::Refit);
    }
    phases
}

/// Extract the minimal per-node and per-OOF-edge facts the replay decision reads
/// from the portable outcome state. Shared by both `derive_replayable_phases`
/// (full derivation) and `closure_predict_replayable` (package PREDICT gate).
/// Fallible: a closure node absent from `node_plans` is a contract error, never a
/// panic.
fn closure_replay_facts(
    plan: &ExecutionPlan,
    closure: &BTreeSet<NodeId>,
    execution_bundle: &ExecutionBundle,
    portable_prediction_caches: Option<&BundlePredictionCachePayloadSet>,
) -> Result<ClosureReplayFacts> {
    let artifact_nodes = execution_bundle
        .refit_artifacts
        .iter()
        .map(|record| record.node_id.clone())
        .collect::<BTreeSet<_>>();
    let requirement_keys = execution_bundle
        .prediction_requirements
        .iter()
        .map(|requirement| requirement.key())
        .collect::<BTreeSet<_>>();
    let cache_keys = execution_bundle
        .prediction_caches
        .iter()
        .map(|record| record.requirement_key.clone())
        .collect::<BTreeSet<_>>();
    let payload_keys = portable_prediction_caches
        .map(|set| {
            set.caches
                .iter()
                .map(|payload| payload.requirement_key.clone())
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();

    let nodes = closure
        .iter()
        .map(|node_id| {
            let node_plan = plan.node_plans.get(node_id).ok_or_else(|| {
                DagMlError::CampaignValidation(format!(
                    "replay derivation references absent node `{node_id}`"
                ))
            })?;
            // A node carries fitted state that PREDICT/EXPLAIN must reload only
            // when it is `stateful` or emits artifacts. `artifact_policy` is not
            // used: a stateless `replay_required` operator (e.g. prospectr)
            // re-runs its deterministic transform at inference with no artifact.
            let requires_retained_state = node_plan
                .controller_capabilities
                .contains(&ControllerCapability::Stateful)
                || node_plan
                    .controller_capabilities
                    .contains(&ControllerCapability::EmitsArtifacts);
            Ok(NodeReplayFacts {
                supported_phases: node_plan.supported_phases.clone(),
                requires_retained_state,
                has_retained_artifact: artifact_nodes.contains(node_id),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let oof_edges = plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| {
            edge.contract.requires_oof
                && closure.contains(&edge.source.node_id)
                && closure.contains(&edge.target.node_id)
        })
        .map(|edge| {
            let key = crate::bundle::bundle_prediction_requirement_key(
                &edge.source.node_id,
                &edge.source.port_name,
                &edge.target.node_id,
                &edge.target.port_name,
            );
            OofEdgeReplayFacts {
                has_bundle_requirement: requirement_keys.contains(&key),
                has_cache_record: cache_keys.contains(&key),
                has_portable_payload: payload_keys.contains(&key),
            }
        })
        .collect::<Vec<_>>();

    Ok(ClosureReplayFacts { nodes, oof_edges })
}

/// Deterministically derive the phases a training outcome can honestly replay.
///
/// This is the single shared helper used by both construction and standalone
/// validation. It reads only portable outcome state (the effective plan's
/// node/controller support, the predictor closure, the retained refit artifacts,
/// the OOF prediction requirements/cache records and the portable payloads), so
/// re-running it during validation reproduces the exact vector a producer must
/// have emitted and rejects any forged claim.
fn derive_replayable_phases(
    plan: &ExecutionPlan,
    closure: &BTreeSet<NodeId>,
    refit: &TrainingRefitOutcome,
    execution_bundle: &ExecutionBundle,
    portable_prediction_caches: Option<&BundlePredictionCachePayloadSet>,
) -> Result<Vec<Phase>> {
    let facts = closure_replay_facts(plan, closure, execution_bundle, portable_prediction_caches)?;
    Ok(derive_replayable_phases_from_facts(
        matches!(refit.status, TrainingRefitStatus::Completed),
        &facts,
    ))
}

/// True when the full predictor `closure` can honestly replay PREDICT given the
/// artifacts retained in `execution_bundle`: every closure node supports PREDICT
/// and every state-retaining closure node has a retained refit artifact. A
/// [`PortablePredictorPackage`](crate::training::PortablePredictorPackage) is a
/// deployable predictor, so its construction requires this independently — it
/// must not infer portability from a merely non-empty claimed phase set. PREDICT
/// replay never consumes OOF payloads, so the OOF cache facts are irrelevant.
pub(crate) fn closure_predict_replayable(
    plan: &ExecutionPlan,
    closure: &BTreeSet<NodeId>,
    execution_bundle: &ExecutionBundle,
) -> Result<bool> {
    let facts = closure_replay_facts(plan, closure, execution_bundle, None)?;
    Ok(derive_replayable_phases_from_facts(true, &facts).contains(&Phase::Predict))
}

fn expected_base_influence_kind(
    plan: &ExecutionPlan,
    node_id: &NodeId,
) -> Option<TrainingInfluenceKind> {
    let node_plan = &plan.node_plans[node_id];
    if matches!(
        node_plan.fit_scope,
        ControllerFitScope::Stateless | ControllerFitScope::InferenceOnly
    ) {
        return None;
    }
    let oof_consumer = plan
        .graph_plan
        .graph
        .edges
        .iter()
        .any(|edge| edge.contract.requires_oof && edge.target.node_id == *node_id);
    Some(
        if oof_consumer
            || node_plan
                .controller_capabilities
                .contains(&ControllerCapability::TrainsAggregation)
        {
            TrainingInfluenceKind::TrainedMetaAggregation
        } else if node_plan.kind == NodeKind::Model {
            TrainingInfluenceKind::ModelFit
        } else if node_plan.kind == NodeKind::Tuner {
            TrainingInfluenceKind::HpoSelection
        } else {
            TrainingInfluenceKind::TransformFit
        },
    )
}

fn validate_influence_against_closure(
    influence: &TrainingInfluenceManifest,
    plan: &ExecutionPlan,
    closure: &BTreeSet<NodeId>,
) -> Result<()> {
    let mut actual_base = BTreeMap::<NodeId, BTreeSet<TrainingInfluenceKind>>::new();
    for entry in &influence.entries {
        let Some(node_id) = &entry.node_id else {
            continue;
        };
        if !closure.contains(node_id) {
            return contract_error("training influence node is outside predictor closure");
        }
        if !influence_kind_allowed_by_node_role_or_capability(plan, node_id, entry.kind) {
            return contract_error(
                "training influence kind is not allowed by node role or capability",
            );
        }
        if expected_base_influence_kind(plan, node_id) == Some(entry.kind) {
            actual_base
                .entry(node_id.clone())
                .or_default()
                .insert(entry.kind);
        }
    }
    let expected = closure
        .iter()
        .filter(|node_id| {
            plan.node_plans[*node_id]
                .supported_phases
                .contains(&Phase::FitCv)
                && expected_base_influence_kind(plan, node_id).is_some()
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual_base.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return contract_error(
            "training influence fitting nodes do not exactly match predictor closure",
        );
    }
    for node_id in expected {
        if actual_base[&node_id]
            != BTreeSet::from([expected_base_influence_kind(plan, &node_id)
                .expect("expected fitting nodes have a base influence kind")])
        {
            return contract_error("training influence fitting kind does not match node role");
        }
    }
    Ok(())
}

fn influence_kind_allowed_by_node_role_or_capability(
    plan: &ExecutionPlan,
    node_id: &NodeId,
    kind: TrainingInfluenceKind,
) -> bool {
    if expected_base_influence_kind(plan, node_id) == Some(kind) {
        return true;
    }
    let capabilities = &plan.node_plans[node_id].controller_capabilities;
    match kind {
        TrainingInfluenceKind::HpoSelection => {
            capabilities.contains(&ControllerCapability::PerformsInternalTuning)
        }
        TrainingInfluenceKind::EarlyStopping => {
            capabilities.contains(&ControllerCapability::UsesEarlyStopping)
        }
        TrainingInfluenceKind::WeightingResampling => {
            capabilities.contains(&ControllerCapability::UsesTrainingWeights)
        }
        TrainingInfluenceKind::TransformFit
        | TrainingInfluenceKind::ModelFit
        | TrainingInfluenceKind::TrainedMetaAggregation => false,
    }
}

fn validate_lineage_coordinates(
    outcome: &TrainingOutcome,
    closure: &BTreeSet<NodeId>,
    coordinates: &BTreeMap<(Phase, Option<crate::ids::FoldId>, NodeId), &LineageRecord>,
) -> Result<()> {
    let fold_set = outcome.effective_plan.fold_set.as_ref().ok_or_else(|| {
        DagMlError::CampaignValidation(
            "training outcome FIT_CV lineage requires a fold_set".to_string(),
        )
    })?;
    let expected_fit = closure
        .iter()
        .filter(|node_id| {
            outcome.effective_plan.node_plans[*node_id]
                .supported_phases
                .contains(&Phase::FitCv)
        })
        .flat_map(|node_id| {
            fold_set
                .folds
                .iter()
                .map(move |fold| (Phase::FitCv, Some(fold.fold_id.clone()), node_id.clone()))
        })
        .collect::<BTreeSet<_>>();
    let actual_fit = coordinates
        .keys()
        .filter(|(phase, _, _)| *phase == Phase::FitCv)
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual_fit != expected_fit {
        return contract_error(
            "training outcome FIT_CV lineage does not exactly cover closure folds",
        );
    }
    let expected_refit = if outcome.refit.requested {
        closure
            .iter()
            .filter(|node_id| {
                outcome.effective_plan.node_plans[*node_id]
                    .supported_phases
                    .contains(&Phase::Refit)
            })
            .map(|node_id| (Phase::Refit, None, node_id.clone()))
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let actual_refit = coordinates
        .keys()
        .filter(|(phase, _, _)| *phase == Phase::Refit)
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual_refit != expected_refit {
        return contract_error("training outcome REFIT lineage does not exactly cover closure");
    }

    for ((phase, fold, node_id), record) in coordinates {
        if *phase == Phase::Select {
            continue;
        }
        let plan = &outcome.effective_plan.node_plans[node_id];
        let expected_inputs = plan
            .input_nodes
            .iter()
            .filter(|input| {
                outcome.effective_plan.node_plans[*input]
                    .supported_phases
                    .contains(phase)
            })
            .map(|input| {
                coordinates
                    .get(&(*phase, fold.clone(), input.clone()))
                    .map(|upstream| upstream.record_id.clone())
                    .ok_or_else(|| {
                        DagMlError::CampaignValidation(format!(
                            "training lineage is missing upstream `{input}`"
                        ))
                    })
            })
            .collect::<Result<Vec<LineageId>>>()?;
        let mut expected_inputs = expected_inputs;
        expected_inputs.sort();
        if record.input_lineage != expected_inputs {
            return contract_error(
                "training outcome lineage input_lineage does not exactly match plan",
            );
        }
        if *phase == Phase::FitCv && !record.artifact_refs.is_empty() {
            return contract_error("FIT_CV lineage must not retain refit artifacts");
        }
        if *phase == Phase::Refit {
            let mut expected_artifacts = outcome
                .execution_bundle
                .refit_artifacts
                .iter()
                .filter(|artifact| artifact.node_id == *node_id)
                .map(|artifact| artifact.artifact.clone())
                .collect::<Vec<_>>();
            expected_artifacts.sort_by(|left, right| left.id.cmp(&right.id));
            let mut actual_artifacts = record.artifact_refs.clone();
            actual_artifacts.sort_by(|left, right| left.id.cmp(&right.id));
            if actual_artifacts != expected_artifacts {
                return contract_error("REFIT lineage artifact_refs do not match execution bundle");
            }
        }
    }
    Ok(())
}

fn tcv1_fingerprint<T: Serialize + ?Sized>(value: &T, label: &str) -> Result<String> {
    let json = serde_json::to_string(value)?;
    parse_typed_json(&json)
        .map_err(|error| {
            DagMlError::CampaignValidation(format!("{label} is not valid TCV1: {error}"))
        })?
        .fingerprint()
        .map_err(|error| {
            DagMlError::CampaignValidation(format!("{label} TCV1 fingerprint failed: {error}"))
        })
}

fn tcv1_fingerprint_without<T: Serialize>(value: &T, field: &str, label: &str) -> Result<String> {
    let json = serde_json::to_string(value)?;
    parse_typed_json(&json)
        .map_err(|error| {
            DagMlError::CampaignValidation(format!("{label} is not valid TCV1: {error}"))
        })?
        .fingerprint_without(field)
        .map_err(|error| {
            DagMlError::CampaignValidation(format!("{label} TCV1 fingerprint failed: {error}"))
        })
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return contract_error(format!("{label} must be lowercase sha256"));
    }
    Ok(())
}

fn validate_all_identity_relations(
    identities: &[TrainingDataIdentity],
    relation_fingerprint: &str,
) -> Result<()> {
    if identities
        .iter()
        .any(|identity| identity.relation_fingerprint != relation_fingerprint)
    {
        return contract_error(
            "training outcome data identities do not all bind the influence relation",
        );
    }
    Ok(())
}

fn validate_sorted_unique_text(label: &str, values: &[String]) -> Result<()> {
    if values.iter().any(|value| value.trim().is_empty()) {
        return contract_error(format!("{label} contains an empty value"));
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return contract_error(format!("{label} must be strictly sorted and unique"));
    }
    Ok(())
}

fn contract_error<T>(message: impl Into<String>) -> Result<T> {
    Err(DagMlError::CampaignValidation(message.into()))
}

#[cfg(test)]
mod replay_phase_tests {
    use super::{
        derive_replayable_phases_from_facts, ClosureReplayFacts, NodeReplayFacts,
        OofEdgeReplayFacts,
    };
    use crate::phase::Phase;
    use std::collections::BTreeSet;

    fn node(
        supported: &[Phase],
        requires_retained_state: bool,
        has_retained_artifact: bool,
    ) -> NodeReplayFacts {
        NodeReplayFacts {
            supported_phases: supported.iter().copied().collect::<BTreeSet<_>>(),
            requires_retained_state,
            has_retained_artifact,
        }
    }

    fn oof(
        has_bundle_requirement: bool,
        has_cache_record: bool,
        has_portable_payload: bool,
    ) -> OofEdgeReplayFacts {
        OofEdgeReplayFacts {
            has_bundle_requirement,
            has_cache_record,
            has_portable_payload,
        }
    }

    // Completed refit whose full closure supports both forward phases and whose
    // state-retaining nodes (`Stateful || EmitsArtifacts`) all have a retained
    // artifact exposes PREDICT then EXPLAIN in canonical order and never
    // re-advertises REFIT.
    #[test]
    fn completed_refit_full_support_matrix_predict_then_explain() {
        let facts = ClosureReplayFacts {
            nodes: vec![
                node(
                    &[Phase::FitCv, Phase::Refit, Phase::Predict, Phase::Explain],
                    true,
                    true,
                ),
                node(
                    &[Phase::FitCv, Phase::Refit, Phase::Predict, Phase::Explain],
                    true,
                    true,
                ),
            ],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(true, &facts),
            vec![Phase::Predict, Phase::Explain]
        );
    }

    // The current completed-refit fixture: every closure node supports
    // FIT_CV/REFIT/PREDICT but not EXPLAIN, so only PREDICT is honest.
    #[test]
    fn completed_refit_predict_only_when_explain_unsupported() {
        let facts = ClosureReplayFacts {
            nodes: vec![
                node(&[Phase::FitCv, Phase::Refit, Phase::Predict], true, true),
                // A train-only augmentation node emits no artifact, so it does not
                // require retained inference state and must not block PREDICT.
                node(&[Phase::FitCv, Phase::Refit, Phase::Predict], false, false),
            ],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(true, &facts),
            vec![Phase::Predict]
        );
    }

    // A downstream node supporting PREDICT cannot rescue an upstream required
    // node that does not support it: the whole closure must support the phase.
    #[test]
    fn upstream_node_missing_phase_blocks_whole_closure() {
        let facts = ClosureReplayFacts {
            nodes: vec![
                // downstream predictor supports PREDICT and EXPLAIN
                node(
                    &[Phase::FitCv, Phase::Refit, Phase::Predict, Phase::Explain],
                    true,
                    true,
                ),
                // upstream required transform supports neither
                node(&[Phase::FitCv, Phase::Refit], false, false),
            ],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(true, &facts),
            Vec::<Phase>::new()
        );
    }

    // A completed refit whose closure supports PREDICT but is missing the
    // retained artifact of a state-retaining node (here `requires_retained_state`)
    // has no honest replay mode: [] is the correct, preferable answer.
    #[test]
    fn completed_refit_missing_artifact_yields_empty() {
        let facts = ClosureReplayFacts {
            nodes: vec![node(
                &[Phase::FitCv, Phase::Refit, Phase::Predict],
                true,
                false,
            )],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(true, &facts),
            Vec::<Phase>::new()
        );
    }

    // No-refit outcome never advertises PREDICT/EXPLAIN even when supported, and
    // advertises REFIT only when every OOF dependency is fully self-contained
    // (exact bundle requirement + cache record + portable payload).
    #[test]
    fn no_refit_refit_requires_self_contained_oof_payload() {
        let supported = [Phase::FitCv, Phase::Refit, Phase::Predict, Phase::Explain];
        let backed = ClosureReplayFacts {
            nodes: vec![node(&supported, true, false), node(&supported, true, false)],
            oof_edges: vec![oof(true, true, true)],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(false, &backed),
            vec![Phase::Refit]
        );

        // Missing portable payload -> not self-contained -> [].
        let missing_payload = ClosureReplayFacts {
            nodes: vec![node(&supported, true, false), node(&supported, true, false)],
            oof_edges: vec![oof(true, true, false)],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(false, &missing_payload),
            Vec::<Phase>::new()
        );

        // Missing cache record -> [].
        let missing_record = ClosureReplayFacts {
            nodes: vec![node(&supported, true, false)],
            oof_edges: vec![oof(true, false, true)],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(false, &missing_record),
            Vec::<Phase>::new()
        );
    }

    // A no-refit outcome with no OOF edges is vacuously self-contained: REFIT can
    // re-fit from data alone, so REFIT is honest when every node supports it.
    #[test]
    fn no_refit_without_oof_edges_is_vacuously_refit() {
        let facts = ClosureReplayFacts {
            nodes: vec![node(
                &[Phase::FitCv, Phase::Refit, Phase::Predict],
                false,
                false,
            )],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(false, &facts),
            vec![Phase::Refit]
        );
    }

    // A no-refit closure that does not fully support REFIT yields [].
    #[test]
    fn no_refit_without_refit_support_yields_empty() {
        let facts = ClosureReplayFacts {
            nodes: vec![
                node(&[Phase::FitCv, Phase::Refit], false, false),
                node(&[Phase::FitCv, Phase::Predict], false, false),
            ],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(false, &facts),
            Vec::<Phase>::new()
        );
    }

    // A stateless `replay_required` operator (e.g. prospectr): it is neither
    // `stateful` nor an artifact emitter, so `requires_retained_state` is false
    // and it stays PREDICT-replayable with no retained artifact — the operation
    // simply replays its deterministic transform at inference time.
    #[test]
    fn stateless_replay_required_operator_without_artifact_stays_predict_replayable() {
        let facts = ClosureReplayFacts {
            nodes: vec![node(
                &[Phase::FitCv, Phase::Refit, Phase::Predict],
                false,
                false,
            )],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(true, &facts),
            vec![Phase::Predict]
        );
    }

    // A `stateful` (or artifact-emitting) node that has no retained artifact
    // carries no reloadable inference state, so PREDICT must not be advertised.
    #[test]
    fn stateful_non_emitter_without_artifact_cannot_advertise_predict() {
        let facts = ClosureReplayFacts {
            nodes: vec![node(
                &[Phase::FitCv, Phase::Refit, Phase::Predict],
                true,
                false,
            )],
            oof_edges: vec![],
        };
        assert_eq!(
            derive_replayable_phases_from_facts(true, &facts),
            Vec::<Phase>::new()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFIT_FIXTURE: &str =
        include_str!("../../../examples/fixtures/estimator/training_outcome_refit.v1.json");
    const NO_REFIT_FIXTURE: &str =
        include_str!("../../../examples/fixtures/estimator/training_outcome_no_refit.v1.json");

    #[test]
    fn cv_ensemble_partition_truth_table_retains_validation_only() {
        for (partition, expected) in [
            (PredictionPartition::Validation, true),
            (PredictionPartition::Train, false),
            (PredictionPartition::Test, false),
            (PredictionPartition::Final, false),
        ] {
            assert_eq!(
                is_cv_ensemble_partition(&partition),
                expected,
                "unexpected CvEnsemble retention decision for {partition:?}"
            );
        }
    }

    #[test]
    fn independent_w0_training_outcomes_parse_and_round_trip_fingerprint() {
        for fixture in [REFIT_FIXTURE, NO_REFIT_FIXTURE] {
            let outcome = TrainingOutcome::from_json(fixture).expect("valid W0 outcome");
            assert_eq!(
                outcome.compute_fingerprint().unwrap(),
                outcome.outcome_fingerprint
            );
            let serialized = serde_json::to_string(&outcome).unwrap();
            let reparsed = TrainingOutcome::from_json(&serialized).unwrap();
            assert_eq!(reparsed, outcome);
        }
    }

    #[test]
    fn strict_parser_rejects_tamper_and_unknown_field() {
        let mut tampered: serde_json::Value = serde_json::from_str(REFIT_FIXTURE).unwrap();
        tampered["warnings"] = serde_json::json!(["tampered"]);
        assert!(TrainingOutcome::from_json(&serde_json::to_string(&tampered).unwrap()).is_err());

        let mut unknown: serde_json::Value = serde_json::from_str(REFIT_FIXTURE).unwrap();
        unknown["unknown_field"] = serde_json::json!(true);
        assert!(TrainingOutcome::from_json(&serde_json::to_string(&unknown).unwrap()).is_err());
    }

    #[test]
    fn outcome_rejects_nested_runtime_handle_keys_defense_in_depth() {
        let mut outcome = TrainingOutcome::from_json(REFIT_FIXTURE).unwrap();
        outcome.diagnostics.insert(
            "nested".to_string(),
            serde_json::json!({"runtime_handle": "process-local"}),
        );
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
        let error = outcome.validate().unwrap_err();
        assert!(error.to_string().contains("runtime handles"), "{error}");
    }

    #[test]
    fn strict_parser_rejects_future_version_even_when_resigned() {
        let mut future: serde_json::Value = serde_json::from_str(REFIT_FIXTURE).unwrap();
        future["schema_version"] = serde_json::json!(2);
        let mut provisional: TrainingOutcome = serde_json::from_value(future.clone()).unwrap();
        provisional.outcome_fingerprint = provisional.compute_fingerprint().unwrap();
        future["outcome_fingerprint"] =
            serde_json::Value::String(provisional.outcome_fingerprint.clone());
        assert!(TrainingOutcome::from_json(&serde_json::to_string(&future).unwrap()).is_err());
    }

    #[test]
    fn select_lineage_is_portable_but_foreign_phase_is_rejected() {
        let mut outcome = TrainingOutcome::from_json(REFIT_FIXTURE).unwrap();
        let mut select = outcome.lineage[0].clone();
        select.record_id = LineageId::new("lineage:select:audit").unwrap();
        select.phase = Phase::Select;
        select.fold_id = None;
        select.input_lineage.clear();
        select.artifact_refs.clear();
        outcome.lineage.push(select.clone());
        outcome
            .lineage
            .sort_by(|left, right| left.record_id.cmp(&right.record_id));
        outcome.outcome_fingerprint = zero_fingerprint();
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
        outcome.validate().unwrap();

        let added = outcome
            .lineage
            .iter_mut()
            .find(|record| record.record_id.as_str() == "lineage:select:audit")
            .unwrap();
        added.phase = Phase::Predict;
        added.record_id = LineageId::new("lineage:predict:foreign").unwrap();
        outcome
            .lineage
            .sort_by(|left, right| left.record_id.cmp(&right.record_id));
        outcome.outcome_fingerprint = zero_fingerprint();
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
        assert!(outcome.validate().is_err());
    }

    #[test]
    fn every_data_identity_must_bind_the_global_relation() {
        let relation = "a".repeat(64);
        let identity = |key: &str, relation_fingerprint: String| TrainingDataIdentity {
            requirement_key: key.to_string(),
            schema_fingerprint: "b".repeat(64),
            plan_fingerprint: "c".repeat(64),
            relation_fingerprint,
            data_content_fingerprint: "d".repeat(64),
            target_content_fingerprint: "e".repeat(64),
            identity_fingerprint: "f".repeat(64),
        };
        let identities = vec![
            identity("model:a.x", relation.clone()),
            identity("model:b.x", "9".repeat(64)),
        ];
        assert!(validate_all_identity_relations(&identities, &relation).is_err());
        let identities = vec![
            identity("model:a.x", relation.clone()),
            identity("model:b.x", relation.clone()),
        ];
        validate_all_identity_relations(&identities, &relation).unwrap();
    }

    #[test]
    fn auxiliary_report_levels_do_not_override_selection_target_level() {
        let report = |producer: &str, level| crate::metrics::RegressionMetricReport {
            prediction_id: Some(format!("prediction:{producer}")),
            producer_node: NodeId::new(producer).unwrap(),
            producer_port: None,
            variant_id: Some(VariantId::new("variant:test").unwrap()),
            variant_label: None,
            partition: PredictionPartition::Validation,
            fold_id: Some(crate::ids::FoldId::new("avg").unwrap()),
            level,
            row_count: 2,
            target_width: 1,
            target_names: vec!["y".to_string()],
            metrics: BTreeMap::from([("rmse".to_string(), 0.1)]),
        };
        let reports = vec![
            report("model:target", PredictionLevel::Sample),
            report("model:target", PredictionLevel::Group),
            report("model:aux", PredictionLevel::Group),
        ];
        validate_selection_report_levels(
            &reports,
            &NodeId::new("model:target").unwrap(),
            &None,
            PredictionLevel::Sample,
        )
        .unwrap();
        assert!(validate_selection_report_levels(
            &reports,
            &NodeId::new("model:target").unwrap(),
            &None,
            PredictionLevel::Target,
        )
        .is_err());
    }
}
