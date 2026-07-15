// Auto-split from the former monolithic `runtime.rs` (pure refactor).
use super::*;

#[derive(Clone, Debug, Default)]
pub struct SequentialScheduler;

#[derive(Clone, Debug)]
pub struct ParallelScheduler {
    max_workers: usize,
}

impl ParallelScheduler {
    pub fn new(max_workers: usize) -> Result<Self> {
        if max_workers == 0 {
            return Err(DagMlError::RuntimeValidation(
                "parallel scheduler max_workers must be at least 1".to_string(),
            ));
        }
        Ok(Self { max_workers })
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PhaseScope {
    pub(crate) phase: Phase,
    pub(crate) variant_id: Option<VariantId>,
    pub(crate) variant: Option<VariantExecutionSpec>,
    pub(crate) fold_id: Option<FoldId>,
    pub(crate) seed_root: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct ReplayPredictionCacheContract {
    pub(crate) requirement: BundlePredictionRequirement,
    pub(crate) cache: BundlePredictionCacheRecord,
}

pub(crate) struct MaterializedReplayArtifacts {
    pub(crate) handles: BTreeMap<NodeId, BTreeMap<String, HandleRef>>,
    pub(crate) inputs: BTreeMap<NodeId, BTreeMap<String, ArtifactInputSpec>>,
}

fn prediction_output_ports_for_node(plan: &ExecutionPlan, node_id: &NodeId) -> Result<Vec<String>> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == *node_id)
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "node `{node_id}` is absent from the execution graph"
            ))
        })?;
    let mut ports = node
        .ports
        .outputs
        .iter()
        .filter(|port| port.kind == PortKind::Prediction)
        .map(|port| port.name.clone())
        .collect::<Vec<_>>();
    ports.sort();
    Ok(ports)
}

fn normalize_prediction_result_port(
    node_id: &NodeId,
    block_kind: &str,
    producer_port: &mut Option<String>,
    prediction_ports: &[String],
) -> Result<()> {
    if let Some(port) = producer_port.as_ref() {
        if port.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{node_id}` emitted {block_kind} with blank producer_port"
            )));
        }
        if !prediction_ports.iter().any(|candidate| candidate == port) {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{node_id}` emitted {block_kind} for undeclared or non-prediction output port `{port}`; declared prediction ports are {:?}",
                prediction_ports
            )));
        }
        return Ok(());
    }
    match prediction_ports {
        [only] => {
            *producer_port = Some(only.clone());
            Ok(())
        }
        [] => Err(DagMlError::RuntimeValidation(format!(
            "node `{node_id}` emitted {block_kind} without producer_port but declares no prediction output port"
        ))),
        _ => Err(DagMlError::RuntimeValidation(format!(
            "node `{node_id}` emitted {block_kind} without producer_port but declares {} prediction output ports {:?}; multi-output controllers must emit producer_port explicitly",
            prediction_ports.len(),
            prediction_ports
        ))),
    }
}

pub(crate) fn normalize_result_prediction_ports(
    plan: &ExecutionPlan,
    task: &NodeTask,
    result: &mut NodeResult,
) -> Result<()> {
    if result.predictions.is_empty()
        && result.observation_predictions.is_empty()
        && result.aggregated_predictions.is_empty()
        && result.explanations.is_empty()
    {
        return Ok(());
    }
    let prediction_ports = prediction_output_ports_for_node(plan, &task.node_plan.node_id)?;
    for block in &mut result.predictions {
        normalize_prediction_result_port(
            &task.node_plan.node_id,
            "prediction block",
            &mut block.producer_port,
            &prediction_ports,
        )?;
    }
    for block in &mut result.observation_predictions {
        normalize_prediction_result_port(
            &task.node_plan.node_id,
            "observation prediction block",
            &mut block.producer_port,
            &prediction_ports,
        )?;
    }
    for block in &mut result.aggregated_predictions {
        normalize_prediction_result_port(
            &task.node_plan.node_id,
            "aggregated prediction block",
            &mut block.producer_port,
            &prediction_ports,
        )?;
    }
    for block in &mut result.explanations {
        normalize_prediction_result_port(
            &task.node_plan.node_id,
            "explanation block",
            &mut block.producer_port,
            &prediction_ports,
        )?;
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct PhaseScopeResources<'a> {
    pub(crate) data_provider: Option<&'a dyn RuntimeDataProvider>,
    pub(crate) replay_artifact_handles: Option<&'a BTreeMap<NodeId, BTreeMap<String, HandleRef>>>,
    pub(crate) replay_artifact_inputs:
        Option<&'a BTreeMap<NodeId, BTreeMap<String, ArtifactInputSpec>>>,
    pub(crate) replay_bundle_id: Option<&'a BundleId>,
    pub(crate) data_envelopes: Option<&'a BTreeMap<String, ExternalDataPlanEnvelope>>,
    pub(crate) prediction_cache_store: Option<&'a dyn RuntimePredictionCacheStore>,
    pub(crate) prediction_cache_contracts:
        Option<&'a BTreeMap<String, ReplayPredictionCacheContract>>,
    pub(crate) artifact_store: Option<&'a mut InMemoryArtifactStore>,
}

impl SequentialScheduler {
    pub fn execute_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources::default(),
        )
    }

    pub fn execute_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(data_provider),
                ..Default::default()
            },
        )
    }

    pub fn execute_campaign_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources::default(),
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider_and_artifact_store(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        artifact_store: &mut InMemoryArtifactStore,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        artifact_store: Some(&mut *artifact_store),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_bundle_replay(
        &self,
        replay: BundleReplayExecution<'_>,
        ctx: &mut RunContext,
    ) -> Result<Vec<NodeResult>> {
        replay.bundle.validate_against_plan(replay.plan)?;
        replay
            .replay_request
            .validate_for_bundle_with_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store.is_some(),
            )?;
        replay
            .bundle
            .validate_replay_envelopes(replay.data_envelopes)?;
        let prediction_cache_contracts = if replay.replay_request.phase == Phase::Refit {
            Some(replay_prediction_cache_contracts(replay.bundle)?)
        } else {
            None
        };
        if replay.replay_request.phase == Phase::Refit {
            preload_replay_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store,
                ctx,
            )?;
        }
        let replay_artifacts = materialize_replay_artifact_handles(
            replay.plan,
            replay.bundle,
            replay.replay_request,
            replay.artifact_store,
            ctx,
        )?;
        let selected_variant = replay
            .bundle
            .selected_variant_id
            .as_ref()
            .map(|selected| {
                replay
                    .plan
                    .variants
                    .iter()
                    .find(|variant| &variant.variant_id == selected)
                    .map(VariantExecutionSpec::from_plan)
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "bundle `{}` selected unknown variant `{selected}`",
                            replay.bundle.bundle_id
                        ))
                    })
            })
            .transpose()?;
        let seed_root = selected_variant
            .as_ref()
            .and_then(|variant| variant.seed)
            .or(ctx.root_seed);

        self.execute_phase_scope(
            replay.plan,
            replay.controllers,
            ctx,
            PhaseScope {
                phase: replay.replay_request.phase,
                variant_id: replay.bundle.selected_variant_id.clone(),
                variant: selected_variant,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(replay.data_provider),
                replay_artifact_handles: Some(&replay_artifacts.handles),
                replay_artifact_inputs: Some(&replay_artifacts.inputs),
                replay_bundle_id: Some(&replay.bundle.bundle_id),
                data_envelopes: Some(replay.data_envelopes),
                prediction_cache_store: replay.prediction_cache_store,
                prediction_cache_contracts: prediction_cache_contracts.as_ref(),
                ..Default::default()
            },
        )
    }

    fn execute_phase_scope(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        scope: PhaseScope,
        mut resources: PhaseScopeResources<'_>,
    ) -> Result<Vec<NodeResult>> {
        let _phase_span = crate::observability::phase_span(
            ctx.run_id.as_str(),
            plan.id.as_str(),
            scope.phase.as_str(),
            scope.variant_id.as_ref().map(VariantId::as_str),
            scope.fold_id.as_ref().map(FoldId::as_str),
        )
        .entered();
        let mut results = Vec::new();
        let mut output_handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
        let mut output_data_views =
            BTreeMap::<NodeId, BTreeMap<String, DataProviderViewSpec>>::new();
        let mut input_lineage = BTreeMap::<NodeId, LineageId>::new();

        for level in plan.node_parallel_levels_for_phase(scope.phase)? {
            for node_id in &level {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
                // Cross-branch merge reassembly (concat or late-fusion) is a
                // scheduler/runtime handler, not a controller call: it reads the
                // upstream branch OOF blocks from the prediction store and emits
                // one merged per-sample OOF block. Intercept it before the
                // controller path (and before the `requires_oof` edge collection,
                // which is a stacking contract the branch inputs do not satisfy).
                if let Some(reduction) = merge_reduction_mode(plan, node_plan) {
                    if let Some(mut result) =
                        reassemble_branch_merge(plan, node_plan, ctx, &scope, reduction)?
                    {
                        let task_node_plan = effective_node_plan_for_scope(node_plan, &scope)?;
                        let task = NodeTask {
                            inner_fold_set: None,
                            run_id: ctx.run_id.clone(),
                            node_plan: task_node_plan.clone(),
                            phase: scope.phase,
                            variant_id: scope.variant_id.clone(),
                            variant: scope.variant.clone(),
                            fold_id: scope.fold_id.clone(),
                            branch_path: Vec::new(),
                            input_handles: BTreeMap::new(),
                            data_views: BTreeMap::new(),
                            prediction_inputs: BTreeMap::new(),
                            artifact_inputs: BTreeMap::new(),
                            fit_influence: FitInfluenceTask::default(),
                            seed: None,
                        };
                        normalize_result_prediction_ports(plan, &task, &mut result)?;
                        result.validate_for_task(&task)?;
                        for prediction in &result.predictions {
                            ctx.prediction_store.append(prediction.clone())?;
                        }
                        apply_result_scoring(
                            &result,
                            &mut ctx.score_collector,
                            &mut ctx.regression_target_records,
                        )?;
                        ctx.lineage.record(result.lineage.clone())?;
                        output_handles.insert(node_id.clone(), result.outputs.clone());
                        input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
                        results.push(result);
                    }
                    continue;
                }
                let controller = controllers.get(&node_plan.controller_id).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "runtime controller `{}` is not registered",
                        node_plan.controller_id
                    ))
                })?;
                let collected_inputs = collect_input_handles(
                    plan,
                    node_plan,
                    &output_handles,
                    &output_data_views,
                    &resources,
                    ctx,
                    &scope,
                )?;
                if collected_inputs.skip_node {
                    continue;
                }
                let mut input_handles = collected_inputs.handles;
                let mut artifact_inputs = BTreeMap::new();
                if let Some(node_artifact_handles) = resources
                    .replay_artifact_handles
                    .and_then(|handles| handles.get(node_id))
                {
                    for (key, handle) in node_artifact_handles {
                        if input_handles.insert(key.clone(), handle.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact input `{key}`"
                            )));
                        }
                    }
                }
                if let Some(node_artifact_inputs) = resources
                    .replay_artifact_inputs
                    .and_then(|inputs| inputs.get(node_id))
                {
                    for (key, spec) in node_artifact_inputs {
                        if artifact_inputs.insert(key.clone(), spec.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact metadata `{key}`"
                            )));
                        }
                    }
                }
                let task_node_plan = effective_node_plan_for_scope(node_plan, &scope)?;
                let inner_fold_set = inner_fold_set_for_scope(
                    &plan.campaign,
                    plan.fold_set.as_ref(),
                    node_plan,
                    &scope,
                )?;
                let fit_influence = fit_influence_task_for_node(
                    plan,
                    &task_node_plan,
                    &collected_inputs.data_views,
                )?;
                let task = NodeTask {
                    inner_fold_set,
                    run_id: ctx.run_id.clone(),
                    node_plan: task_node_plan.clone(),
                    phase: scope.phase,
                    variant_id: scope.variant_id.clone(),
                    variant: scope.variant.clone(),
                    fold_id: scope.fold_id.clone(),
                    branch_path: Vec::new(),
                    input_handles,
                    data_views: collected_inputs.data_views,
                    prediction_inputs: collected_inputs.prediction_inputs,
                    artifact_inputs,
                    fit_influence,
                    seed: derive_task_seed(
                        scope.seed_root,
                        scope.variant_id.as_ref(),
                        scope.fold_id.as_ref(),
                        &task_node_plan,
                        scope.phase,
                    ),
                };
                let _node_span = crate::observability::node_span(
                    task.run_id.as_str(),
                    plan.id.as_str(),
                    task.phase.as_str(),
                    task.node_plan.node_id.as_str(),
                    task.node_plan.controller_id.as_str(),
                )
                .entered();
                let mut result = controller.invoke(&task)?;
                record_fit_influence_diagnostic(&task, &mut result);
                normalize_result_prediction_ports(plan, &task, &mut result)?;
                result.validate_for_task(&task)?;
                apply_result_prediction_aggregation(
                    plan,
                    controllers,
                    &task,
                    &mut result,
                    &resources,
                )?;
                attach_coordinator_input_lineage(
                    &mut result,
                    plan,
                    &task.node_plan.node_id,
                    &input_lineage,
                )?;
                if let Some(store) = resources.artifact_store.as_deref_mut() {
                    if scope.phase == Phase::Refit {
                        store.capture_refit_artifacts(&task, &result)?;
                    }
                }
                for prediction in &result.predictions {
                    ctx.prediction_store.append(prediction.clone())?;
                }
                for prediction in &result.aggregated_predictions {
                    ctx.aggregated_prediction_store.append(prediction.clone())?;
                }
                apply_result_scoring(
                    &result,
                    &mut ctx.score_collector,
                    &mut ctx.regression_target_records,
                )?;
                ctx.lineage.record(result.lineage.clone())?;
                let data_views = derive_output_data_views(plan, &task, &result)?;
                output_handles.insert(node_id.clone(), result.outputs.clone());
                output_data_views.insert(node_id.clone(), data_views);
                input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
                results.push(result);
            }
        }

        Ok(results)
    }
}

impl ParallelScheduler {
    pub fn execute_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources::default(),
        )
    }

    pub fn execute_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let variant_id = ctx.variant_id.clone();
        let seed_root = ctx.root_seed;
        self.execute_phase_scope(
            plan,
            controllers,
            ctx,
            PhaseScope {
                phase,
                variant_id,
                variant: None,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(data_provider),
                ..Default::default()
            },
        )
    }

    pub fn execute_campaign_phase(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources::default(),
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_campaign_phase_with_data_provider_and_artifact_store(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        data_provider: &dyn RuntimeDataProvider,
        artifact_store: &mut InMemoryArtifactStore,
        ctx: &mut RunContext,
        phase: Phase,
    ) -> Result<Vec<NodeResult>> {
        plan.validate()?;
        let mut results = Vec::new();
        let fold_ids = if phase == Phase::FitCv {
            plan.fold_set
                .as_ref()
                .map(|fold_set| {
                    fold_set
                        .folds
                        .iter()
                        .map(|fold| Some(fold.fold_id.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![None])
        } else {
            vec![None]
        };
        for variant in &plan.variants {
            if ctx
                .variant_id
                .as_ref()
                .is_some_and(|requested| requested != &variant.variant_id)
            {
                continue;
            }
            for fold_id in &fold_ids {
                let seed_root = variant.seed.or(ctx.root_seed);
                results.extend(self.execute_phase_scope(
                    plan,
                    controllers,
                    ctx,
                    PhaseScope {
                        phase,
                        variant_id: Some(variant.variant_id.clone()),
                        variant: Some(VariantExecutionSpec::from_plan(variant)),
                        fold_id: fold_id.clone(),
                        seed_root,
                    },
                    PhaseScopeResources {
                        data_provider: Some(data_provider),
                        artifact_store: Some(&mut *artifact_store),
                        ..Default::default()
                    },
                )?);
            }
        }
        Ok(results)
    }

    pub fn execute_bundle_replay(
        &self,
        replay: BundleReplayExecution<'_>,
        ctx: &mut RunContext,
    ) -> Result<Vec<NodeResult>> {
        replay.bundle.validate_against_plan(replay.plan)?;
        replay
            .replay_request
            .validate_for_bundle_with_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store.is_some(),
            )?;
        replay
            .bundle
            .validate_replay_envelopes(replay.data_envelopes)?;
        let prediction_cache_contracts = if replay.replay_request.phase == Phase::Refit {
            Some(replay_prediction_cache_contracts(replay.bundle)?)
        } else {
            None
        };
        if replay.replay_request.phase == Phase::Refit {
            preload_replay_prediction_cache_store(
                replay.bundle,
                replay.prediction_cache_store,
                ctx,
            )?;
        }
        let replay_artifacts = materialize_replay_artifact_handles(
            replay.plan,
            replay.bundle,
            replay.replay_request,
            replay.artifact_store,
            ctx,
        )?;
        let selected_variant = replay
            .bundle
            .selected_variant_id
            .as_ref()
            .map(|selected| {
                replay
                    .plan
                    .variants
                    .iter()
                    .find(|variant| &variant.variant_id == selected)
                    .map(VariantExecutionSpec::from_plan)
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "bundle `{}` selected unknown variant `{selected}`",
                            replay.bundle.bundle_id
                        ))
                    })
            })
            .transpose()?;
        let seed_root = selected_variant
            .as_ref()
            .and_then(|variant| variant.seed)
            .or(ctx.root_seed);

        self.execute_phase_scope(
            replay.plan,
            replay.controllers,
            ctx,
            PhaseScope {
                phase: replay.replay_request.phase,
                variant_id: replay.bundle.selected_variant_id.clone(),
                variant: selected_variant,
                fold_id: None,
                seed_root,
            },
            PhaseScopeResources {
                data_provider: Some(replay.data_provider),
                replay_artifact_handles: Some(&replay_artifacts.handles),
                replay_artifact_inputs: Some(&replay_artifacts.inputs),
                replay_bundle_id: Some(&replay.bundle.bundle_id),
                data_envelopes: Some(replay.data_envelopes),
                prediction_cache_store: replay.prediction_cache_store,
                prediction_cache_contracts: prediction_cache_contracts.as_ref(),
                ..Default::default()
            },
        )
    }

    fn execute_phase_scope(
        &self,
        plan: &ExecutionPlan,
        controllers: &RuntimeControllerRegistry,
        ctx: &mut RunContext,
        scope: PhaseScope,
        mut resources: PhaseScopeResources<'_>,
    ) -> Result<Vec<NodeResult>> {
        // Hold the phase span on the scheduler thread, and clone it into each
        // worker so worker-thread telemetry nests under the phase (tracing spans
        // are thread-local and do not auto-propagate across `thread::scope`).
        let phase_span = crate::observability::phase_span(
            ctx.run_id.as_str(),
            plan.id.as_str(),
            scope.phase.as_str(),
            scope.variant_id.as_ref().map(VariantId::as_str),
            scope.fold_id.as_ref().map(FoldId::as_str),
        );
        let _phase_entered = phase_span.clone().entered();
        // Borrowed for the `thread::scope` below; workers join before it ends.
        let plan_id = plan.id.as_str();
        plan.validate_parallel_controller_capabilities(self.max_workers, scope.phase)?;
        let mut results = Vec::new();
        let mut output_handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
        let mut output_data_views =
            BTreeMap::<NodeId, BTreeMap<String, DataProviderViewSpec>>::new();
        let mut input_lineage = BTreeMap::<NodeId, LineageId>::new();

        for level in plan.node_parallel_levels_for_phase(scope.phase)? {
            let mut prepared = Vec::<PreparedNodeTask>::new();
            // Cross-branch merge nodes (concat or late-fusion) are not controller
            // tasks: they read the upstream branch OOF blocks from the prediction
            // store and reassemble them on the scheduler thread (no worker), AFTER
            // this level's worker tasks have populated the store. They are in a
            // later level than their branches, so the store already holds the
            // branch OOF by the time we reassemble — see `reassemble_branch_merge`.
            let mut merge_nodes = Vec::<(NodeId, MergeReduction)>::new();
            for node_id in &level {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
                if let Some(reduction) = merge_reduction_mode(plan, node_plan) {
                    merge_nodes.push((node_id.clone(), reduction));
                    continue;
                }
                let collected_inputs = collect_input_handles(
                    plan,
                    node_plan,
                    &output_handles,
                    &output_data_views,
                    &resources,
                    ctx,
                    &scope,
                )?;
                if collected_inputs.skip_node {
                    continue;
                }
                let mut input_handles = collected_inputs.handles;
                let mut artifact_inputs = BTreeMap::new();
                if let Some(node_artifact_handles) = resources
                    .replay_artifact_handles
                    .and_then(|handles| handles.get(node_id))
                {
                    for (key, handle) in node_artifact_handles {
                        if input_handles.insert(key.clone(), handle.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact input `{key}`"
                            )));
                        }
                    }
                }
                if let Some(node_artifact_inputs) = resources
                    .replay_artifact_inputs
                    .and_then(|inputs| inputs.get(node_id))
                {
                    for (key, spec) in node_artifact_inputs {
                        if artifact_inputs.insert(key.clone(), spec.clone()).is_some() {
                            return Err(DagMlError::RuntimeValidation(format!(
                                "node `{node_id}` received duplicate replay artifact metadata `{key}`"
                            )));
                        }
                    }
                }
                let task_node_plan = effective_node_plan_for_scope(node_plan, &scope)?;
                let inner_fold_set = inner_fold_set_for_scope(
                    &plan.campaign,
                    plan.fold_set.as_ref(),
                    node_plan,
                    &scope,
                )?;
                let fit_influence = fit_influence_task_for_node(
                    plan,
                    &task_node_plan,
                    &collected_inputs.data_views,
                )?;
                prepared.push(PreparedNodeTask {
                    node_id: node_id.clone(),
                    task: NodeTask {
                        inner_fold_set,
                        run_id: ctx.run_id.clone(),
                        node_plan: task_node_plan.clone(),
                        phase: scope.phase,
                        variant_id: scope.variant_id.clone(),
                        variant: scope.variant.clone(),
                        fold_id: scope.fold_id.clone(),
                        branch_path: Vec::new(),
                        input_handles,
                        data_views: collected_inputs.data_views,
                        prediction_inputs: collected_inputs.prediction_inputs,
                        artifact_inputs,
                        fit_influence,
                        seed: derive_task_seed(
                            scope.seed_root,
                            scope.variant_id.as_ref(),
                            scope.fold_id.as_ref(),
                            &task_node_plan,
                            scope.phase,
                        ),
                    },
                });
            }

            for chunk in prepared.chunks(self.max_workers) {
                let chunk_results =
                    std::thread::scope(|thread_scope| -> Result<Vec<NodeResult>> {
                        let mut handles = Vec::with_capacity(chunk.len());
                        for prepared_task in chunk {
                            let controller = controllers
                                .get(&prepared_task.task.node_plan.controller_id)
                                .ok_or_else(|| {
                                    DagMlError::RuntimeValidation(format!(
                                        "runtime controller `{}` is not registered",
                                        prepared_task.task.node_plan.controller_id
                                    ))
                                })?;
                            let worker_span = phase_span.clone();
                            handles.push(thread_scope.spawn(move || {
                                let _worker_span = worker_span.entered();
                                let _node_span = crate::observability::node_span(
                                    prepared_task.task.run_id.as_str(),
                                    plan_id,
                                    prepared_task.task.phase.as_str(),
                                    prepared_task.task.node_plan.node_id.as_str(),
                                    prepared_task.task.node_plan.controller_id.as_str(),
                                )
                                .entered();
                                let mut result = controller.invoke(&prepared_task.task)?;
                                record_fit_influence_diagnostic(&prepared_task.task, &mut result);
                                normalize_result_prediction_ports(
                                    plan,
                                    &prepared_task.task,
                                    &mut result,
                                )?;
                                result.validate_for_task(&prepared_task.task)?;
                                Ok(result)
                            }));
                        }
                        handles
                            .into_iter()
                            .map(|handle| {
                                handle.join().map_err(|_| {
                                    DagMlError::RuntimeValidation(
                                        "parallel scheduler worker panicked".to_string(),
                                    )
                                })?
                            })
                            .collect()
                    })?;

                for (prepared_task, mut result) in chunk.iter().zip(chunk_results) {
                    apply_result_prediction_aggregation(
                        plan,
                        controllers,
                        &prepared_task.task,
                        &mut result,
                        &resources,
                    )?;
                    attach_coordinator_input_lineage(
                        &mut result,
                        plan,
                        &prepared_task.task.node_plan.node_id,
                        &input_lineage,
                    )?;
                    if let Some(store) = resources.artifact_store.as_deref_mut() {
                        if scope.phase == Phase::Refit {
                            store.capture_refit_artifacts(&prepared_task.task, &result)?;
                        }
                    }
                    for prediction in &result.predictions {
                        ctx.prediction_store.append(prediction.clone())?;
                    }
                    for prediction in &result.aggregated_predictions {
                        ctx.aggregated_prediction_store.append(prediction.clone())?;
                    }
                    apply_result_scoring(
                        &result,
                        &mut ctx.score_collector,
                        &mut ctx.regression_target_records,
                    )?;
                    ctx.lineage.record(result.lineage.clone())?;
                    let data_views = derive_output_data_views(plan, &prepared_task.task, &result)?;
                    output_handles.insert(prepared_task.node_id.clone(), result.outputs.clone());
                    output_data_views.insert(prepared_task.node_id.clone(), data_views);
                    input_lineage.insert(
                        prepared_task.node_id.clone(),
                        result.lineage.record_id.clone(),
                    );
                    results.push(result);
                }
            }

            // Reassemble any cross-branch merge nodes in this level now that the
            // level's worker tasks have populated the prediction store. Merge nodes
            // sit in a later level than the branches they consume, so the upstream
            // branch OOF is already present.
            for (node_id, reduction) in &merge_nodes {
                let node_plan = plan
                    .node_plans
                    .get(node_id)
                    .expect("execution plan was validated");
                if let Some(mut result) =
                    reassemble_branch_merge(plan, node_plan, ctx, &scope, *reduction)?
                {
                    let task_node_plan = effective_node_plan_for_scope(node_plan, &scope)?;
                    let task = NodeTask {
                        inner_fold_set: None,
                        run_id: ctx.run_id.clone(),
                        node_plan: task_node_plan.clone(),
                        phase: scope.phase,
                        variant_id: scope.variant_id.clone(),
                        variant: scope.variant.clone(),
                        fold_id: scope.fold_id.clone(),
                        branch_path: Vec::new(),
                        input_handles: BTreeMap::new(),
                        data_views: BTreeMap::new(),
                        prediction_inputs: BTreeMap::new(),
                        artifact_inputs: BTreeMap::new(),
                        fit_influence: FitInfluenceTask::default(),
                        seed: None,
                    };
                    normalize_result_prediction_ports(plan, &task, &mut result)?;
                    result.validate_for_task(&task)?;
                    for prediction in &result.predictions {
                        ctx.prediction_store.append(prediction.clone())?;
                    }
                    apply_result_scoring(
                        &result,
                        &mut ctx.score_collector,
                        &mut ctx.regression_target_records,
                    )?;
                    ctx.lineage.record(result.lineage.clone())?;
                    output_handles.insert(node_id.clone(), result.outputs.clone());
                    input_lineage.insert(node_id.clone(), result.lineage.record_id.clone());
                    results.push(result);
                }
            }
        }

        Ok(results)
    }
}

pub(crate) struct PreparedNodeTask {
    pub(crate) node_id: NodeId,
    pub(crate) task: NodeTask,
}

pub(crate) fn attach_coordinator_input_lineage(
    result: &mut NodeResult,
    plan: &ExecutionPlan,
    node_id: &NodeId,
    upstream_lineage: &BTreeMap<NodeId, LineageId>,
) -> Result<()> {
    let inferred = inferred_input_lineage_for_node(plan, node_id, upstream_lineage);
    if result.lineage.input_lineage.is_empty() {
        result.lineage.input_lineage = inferred;
        return Ok(());
    }

    let declared = result
        .lineage
        .input_lineage
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if declared != inferred {
        return Err(DagMlError::RuntimeValidation(format!(
            "lineage for node `{}` declared input lineage {:?}, expected {:?}",
            result.node_id, declared, inferred
        )));
    }
    result.lineage.input_lineage = declared;
    Ok(())
}

pub(crate) fn inferred_input_lineage_for_node(
    plan: &ExecutionPlan,
    node_id: &NodeId,
    upstream_lineage: &BTreeMap<NodeId, LineageId>,
) -> Vec<LineageId> {
    plan.graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| &edge.target.node_id == node_id && edge.contract.propagates_lineage)
        .filter_map(|edge| upstream_lineage.get(&edge.source.node_id).cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub(crate) fn collect_input_handles(
    plan: &ExecutionPlan,
    node_plan: &NodePlan,
    output_handles: &BTreeMap<NodeId, BTreeMap<String, HandleRef>>,
    output_data_views: &BTreeMap<NodeId, BTreeMap<String, DataProviderViewSpec>>,
    resources: &PhaseScopeResources<'_>,
    ctx: &RunContext,
    scope: &PhaseScope,
) -> Result<CollectedInputs> {
    let mut inputs = BTreeMap::new();
    let mut data_views = BTreeMap::new();
    let mut prediction_inputs = BTreeMap::new();
    let training_oof_edges = incoming_training_oof_edges(plan, node_plan, scope)?;
    // An OOF edge replaces exactly one raw producer port. Do not hide sibling
    // outputs from the same producer: a meta-node may legally consume both an
    // OOF prediction port and an auxiliary non-OOF port. PREDICT has no
    // Validation-OOF input, but its raw prediction port must still be masked so
    // only the explicit `:predict` off-fold input reaches the controller.
    let masked_oof_source_ports = if scope.phase == Phase::Predict {
        incoming_oof_edges(plan, node_plan)?
    } else {
        training_oof_edges.clone()
    }
    .into_iter()
    .map(|edge| (edge.source.node_id.clone(), edge.source.port_name.clone()))
    .collect::<BTreeSet<_>>();
    let bound_data_inputs = node_plan
        .data_bindings
        .iter()
        .map(|binding| binding.input_name.clone())
        .collect::<BTreeSet<_>>();
    // Only forward upstream handles for ports this node DECLARES an edge to.
    // A controller must never see a handle outside its declared port contract,
    // so a sibling consumer of the same producer cannot expose extra ports here.
    let declared_source_ports = plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node_plan.node_id)
        .map(|edge| (edge.source.node_id.clone(), edge.source.port_name.clone()))
        .collect::<BTreeSet<_>>();
    for upstream in &node_plan.input_nodes {
        if let Some(handles) = output_handles.get(upstream) {
            for (port, handle) in handles {
                if !declared_source_ports.contains(&(upstream.clone(), port.clone())) {
                    continue;
                }
                if masked_oof_source_ports.contains(&(upstream.clone(), port.clone())) {
                    continue;
                }
                inputs.insert(format!("{upstream}.{port}"), handle.clone());
            }
        }
    }
    for edge in plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target.node_id == node_plan.node_id)
        .filter(|edge| edge.contract.kind == PortKind::Data && !edge.contract.requires_oof)
    {
        if bound_data_inputs.contains(&edge.target.port_name) {
            continue;
        }
        let Some(handles) = output_handles.get(&edge.source.node_id) else {
            continue;
        };
        let Some(handle) = handles.get(&edge.source.port_name) else {
            continue;
        };
        let key = data_view_key(&edge.target.port_name);
        if inputs.insert(key.clone(), handle.clone()).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` received duplicate data edge input `{key}`",
                node_plan.node_id
            )));
        }
        if let Some(source_views) = output_data_views.get(&edge.source.node_id) {
            if let Some(view) = source_views.get(&edge.source.port_name) {
                if data_views.insert(key.clone(), view.clone()).is_some() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate data edge view `{key}`",
                        node_plan.node_id
                    )));
                }
            }
            let source_validation_key = validation_data_view_key(&edge.source.port_name);
            if let Some(view) = source_views.get(&source_validation_key) {
                let validation_key = format!("{key}:validation");
                if data_views
                    .insert(validation_key.clone(), view.clone())
                    .is_some()
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate data edge validation view `{validation_key}`",
                        node_plan.node_id
                    )));
                }
            }
        }
    }
    for edge in training_oof_edges {
        let key = format!("{}.{}", edge.source.node_id, edge.source.port_name);
        let Some(input) = collect_oof_prediction_input(plan, edge, ctx, scope, resources)? else {
            return Ok(CollectedInputs {
                handles: BTreeMap::new(),
                data_views: BTreeMap::new(),
                prediction_inputs: BTreeMap::new(),
                skip_node: true,
            });
        };
        if inputs.insert(key.clone(), input.handle).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` received duplicate OOF prediction input `{key}`",
                node_plan.node_id
            )));
        }
        if prediction_inputs.insert(key.clone(), input.spec).is_some() {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` received duplicate OOF prediction spec `{key}`",
                node_plan.node_id
            )));
        }
    }
    // REFIT / PREDICT: deliver each base producer's off-fold (test / predict)
    // predictions to the stacking meta-node as a SEPARATE prediction input (suffixed
    // `:test` / `:predict`) so the host meta-model predicts from them. The FIT_CV
    // Validation-OOF input above is the meta-features the meta-model trains on; this
    // off-fold input is used ONLY for REFIT/PREDICT scoring/prediction, never FIT_CV
    // training — keeping the leakage invariant intact.
    if matches!(scope.phase, Phase::Refit | Phase::Predict) {
        let off_fold_suffix = scope.phase.as_str().to_ascii_lowercase();
        for edge in incoming_oof_edges(plan, node_plan)? {
            let Some(input) = collect_off_fold_prediction_input(plan, edge, ctx, scope)? else {
                continue;
            };
            let key = format!(
                "{}.{}:{off_fold_suffix}",
                edge.source.node_id, edge.source.port_name
            );
            if inputs.insert(key.clone(), input.handle).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate off-fold prediction input `{key}`",
                    node_plan.node_id
                )));
            }
            if prediction_inputs.insert(key.clone(), input.spec).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate off-fold prediction spec `{key}`",
                    node_plan.node_id
                )));
            }
        }
    }
    if !node_plan.data_bindings.is_empty() && resources.data_provider.is_none() {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` requires {} data binding(s) but no runtime data provider is registered",
            node_plan.node_id,
            node_plan.data_bindings.len()
        )));
    }
    if let Some(data_provider) = resources.data_provider {
        // Samples excluded from training (sample-local) for this node, derived
        // from its coordinator relations. Used to filter FIT view specs so the
        // spec, the materialized view, and fit-influence row_weights agree.
        let excluded_samples = coordinator_relations_for_node(node_plan, resources)?
            .map(|relations| relations.excluded_sample_ids())
            .unwrap_or_default();
        for binding in &node_plan.data_bindings {
            let materialized = data_provider.materialize(&DataMaterializationRequest {
                run_id: ctx.run_id.clone(),
                node_id: node_plan.node_id.clone(),
                input_name: binding.input_name.clone(),
                phase: scope.phase,
                variant_id: scope.variant_id.clone(),
                fold_id: scope.fold_id.clone(),
                binding: binding.clone(),
            })?;
            let branch_view_for_node = branch_view_from_node_metadata(plan, &node_plan.node_id)?;
            let view = data_view_for_scope(
                binding,
                plan.fold_set.as_ref(),
                scope,
                branch_view_for_node.as_ref(),
                &excluded_samples,
            )?;
            let key = data_view_key(&binding.input_name);
            let view_handle = make_data_view_handle(
                data_provider,
                ctx,
                node_plan,
                scope,
                binding,
                &materialized,
                &view,
            )?;
            if data_views.insert(key.clone(), view).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate data view `{key}`",
                    node_plan.node_id
                )));
            }
            if inputs.insert(key.clone(), view_handle).is_some() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received duplicate data input `{key}`",
                    node_plan.node_id
                )));
            }

            if let Some(validation_view) = validation_data_view_for_scope(
                binding,
                plan.fold_set.as_ref(),
                scope,
                branch_view_for_node.as_ref(),
                &excluded_samples,
            )? {
                let validation_key = format!("{key}:validation");
                let validation_handle = make_data_view_handle(
                    data_provider,
                    ctx,
                    node_plan,
                    scope,
                    binding,
                    &materialized,
                    &validation_view,
                )?;
                if data_views
                    .insert(validation_key.clone(), validation_view)
                    .is_some()
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate validation data view `{validation_key}`",
                        node_plan.node_id
                    )));
                }
                if inputs
                    .insert(validation_key.clone(), validation_handle)
                    .is_some()
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received duplicate validation data input `{validation_key}`",
                        node_plan.node_id
                    )));
                }
            }
        }
    }
    Ok(CollectedInputs {
        handles: inputs,
        data_views,
        prediction_inputs,
        skip_node: false,
    })
}
pub(crate) fn preload_replay_prediction_cache_store(
    bundle: &ExecutionBundle,
    prediction_cache_store: Option<&dyn RuntimePredictionCacheStore>,
    ctx: &mut RunContext,
) -> Result<()> {
    if bundle.prediction_requirements.is_empty() {
        return Ok(());
    }
    let store = prediction_cache_store.ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "bundle `{}` cannot preload OOF prediction caches without a prediction cache store",
            bundle.bundle_id
        ))
    })?;
    if !ctx.prediction_store.blocks().is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "bundle `{}` cannot preload OOF prediction caches into a non-empty prediction store",
            bundle.bundle_id
        )));
    }
    let contracts = replay_prediction_cache_contracts(bundle)?;
    for contract in contracts.values() {
        if contract.requirement.prediction_level == PredictionLevel::Sample {
            let blocks = store.load_blocks(&contract.cache.requirement_key)?;
            if blocks.iter().any(|block| {
                block.producer_node != contract.requirement.producer_node
                    || block.partition != contract.requirement.partition
            }) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store returned blocks outside requirement `{}`",
                    contract.cache.requirement_key
                )));
            }
            let mut payload = build_prediction_cache_payload(&contract.requirement, &blocks)?;
            payload.cache_namespace_fingerprints =
                contract.cache.cache_namespace_fingerprints.clone();
            validate_prediction_cache_payload_matches_record(&payload, &contract.cache)?;
            for block in &payload.blocks {
                ctx.prediction_store.append(block.clone())?;
            }
        } else {
            let blocks = store.load_aggregated_blocks(&contract.cache.requirement_key)?;
            if blocks.iter().any(|block| {
                block.producer_node != contract.requirement.producer_node
                    || block.partition != contract.requirement.partition
                    || block.level != contract.requirement.prediction_level
            }) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "prediction cache store returned aggregated blocks outside requirement `{}`",
                    contract.cache.requirement_key
                )));
            }
            let mut payload =
                build_aggregated_prediction_cache_payload(&contract.requirement, &blocks)?;
            payload.cache_namespace_fingerprints =
                contract.cache.cache_namespace_fingerprints.clone();
            validate_prediction_cache_payload_matches_record(&payload, &contract.cache)?;
        }
    }
    Ok(())
}

pub(crate) fn replay_prediction_cache_contracts(
    bundle: &ExecutionBundle,
) -> Result<BTreeMap<String, ReplayPredictionCacheContract>> {
    bundle.validate()?;
    let requirements = bundle
        .prediction_requirements
        .iter()
        .map(|requirement| (requirement.key(), requirement))
        .collect::<BTreeMap<_, _>>();
    let mut contracts = BTreeMap::new();
    for cache in &bundle.prediction_caches {
        let requirement = requirements.get(&cache.requirement_key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "prediction cache `{}` references unknown prediction requirement `{}`",
                cache.cache_id, cache.requirement_key
            ))
        })?;
        contracts.insert(
            cache.requirement_key.clone(),
            ReplayPredictionCacheContract {
                requirement: (*requirement).clone(),
                cache: cache.clone(),
            },
        );
    }
    Ok(contracts)
}

pub(crate) fn materialize_replay_artifact_handles(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    replay_request: &ReplayPhaseRequest,
    artifact_store: &dyn RuntimeArtifactStore,
    ctx: &RunContext,
) -> Result<MaterializedReplayArtifacts> {
    let mut handles = BTreeMap::<NodeId, BTreeMap<String, HandleRef>>::new();
    let mut inputs = BTreeMap::<NodeId, BTreeMap<String, ArtifactInputSpec>>::new();
    for artifact in &bundle.refit_artifacts {
        artifact.validate()?;
        let node_plan = plan.node_plans.get(&artifact.node_id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "bundle `{}` artifact references unknown node `{}`",
                bundle.bundle_id, artifact.node_id
            ))
        })?;
        if !node_plan.supported_phases.contains(&replay_request.phase) {
            return Err(DagMlError::RuntimeValidation(format!(
                "bundle `{}` artifact node `{}` does not support replay phase {:?}",
                bundle.bundle_id, artifact.node_id, replay_request.phase
            )));
        }
        let handle = artifact_store.materialize(&ArtifactMaterializationRequest {
            run_id: ctx.run_id.clone(),
            bundle_id: bundle.bundle_id.clone(),
            node_id: artifact.node_id.clone(),
            phase: replay_request.phase,
            variant_id: bundle.selected_variant_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
            training_loss_fingerprint: artifact.training_loss_fingerprint.clone(),
        })?;
        if !matches!(handle.kind, HandleKind::Model | HandleKind::Artifact) {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` materialized as unsupported handle kind {:?}",
                artifact.artifact.id, handle.kind
            )));
        }
        if handle.owner_controller != artifact.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` handle owner `{}` does not match controller `{}`",
                artifact.artifact.id, handle.owner_controller, artifact.controller_id
            )));
        }
        let key = refit_artifact_input_key(&artifact.artifact.id);
        if handles
            .entry(artifact.node_id.clone())
            .or_default()
            .insert(key.clone(), handle)
            .is_some()
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate replay artifact input `{key}` for node `{}`",
                artifact.node_id
            )));
        }
        if inputs
            .entry(artifact.node_id.clone())
            .or_default()
            .insert(key.clone(), ArtifactInputSpec::from_refit_record(artifact)?)
            .is_some()
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate replay artifact metadata `{key}` for node `{}`",
                artifact.node_id
            )));
        }
    }
    Ok(MaterializedReplayArtifacts { handles, inputs })
}

pub(crate) fn derive_task_seed(
    root_seed: Option<u64>,
    variant_id: Option<&VariantId>,
    fold_id: Option<&FoldId>,
    node_plan: &NodePlan,
    phase: Phase,
) -> Option<u64> {
    root_seed.map(|root| {
        let mut context = SeedContext::root(root);
        if let Some(variant_id) = variant_id {
            context = context.child(format!("variant:{variant_id}"));
        }
        if let Some(fold_id) = fold_id {
            context = context.child(format!("fold:{fold_id}"));
        }
        context
            .child(format!("node:{}", node_plan.node_id))
            .child(format!("phase:{phase:?}"))
            .derive_u64("task")
    })
}
