use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_core::{
    build_execution_bundle, build_execution_plan, oof_campaign_fingerprint, select_candidate,
    select_candidate_groups, validate_oof_campaign, BundleId, CampaignSpec, CandidateScore,
    ControllerId, ControllerManifest, ControllerRegistry, DagMlError, ExecutionBundle,
    ExternalDataPlanEnvelope, GraphSpec, HandleKind, HandleRef, InMemoryArtifactStore,
    InMemoryDataProvider, LineageId, LineageRecord, NodeId, NodeResult, NodeTask, OofCampaign,
    Phase, PredictionBlock, PredictionPartition, RefitArtifactRecord, ReplayPhaseRequest,
    RunContext, RunId, RuntimeController, RuntimeControllerRegistry, SampleId, SelectionDecision,
    SelectionPolicy, SequentialScheduler, VariantId,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ValidateGraph {
        path: PathBuf,
    },
    ValidateOofCampaign {
        path: PathBuf,
        #[arg(long)]
        expect_leakage: bool,
    },
    FingerprintOofCampaign {
        path: PathBuf,
    },
    ValidateExecutionPlan {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long, default_value = "plan:cli")]
        plan_id: String,
    },
    ValidateDataBinding {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
        #[arg(long)]
        node: String,
        #[arg(long)]
        input: String,
    },
    RunMockCampaign {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
        #[arg(long, default_value = "plan:cli.mock")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.mock")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
    },
    SelectCandidates {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long)]
        groups: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    BuildBundle {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        bundle_spec: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
    },
    ValidateBundle {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long = "envelope")]
        envelopes: Vec<String>,
        #[arg(long)]
        replay_request: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
    },
    RunMockReplay {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        replay_request: PathBuf,
        #[arg(long = "envelope")]
        envelopes: Vec<String>,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.replay")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::ValidateGraph { path } => {
            let data = std::fs::read(&path)
                .with_context(|| format!("failed to read graph JSON at {}", path.display()))?;
            let graph: GraphSpec = serde_json::from_slice(&data)
                .with_context(|| format!("failed to parse graph JSON at {}", path.display()))?;
            graph
                .validate()
                .with_context(|| format!("invalid graph at {}", path.display()))?;
            println!("valid graph: {}", graph.id);
        }
        Command::ValidateOofCampaign {
            path,
            expect_leakage,
        } => {
            let campaign: OofCampaign = read_json(&path, "OOF campaign")?;
            match validate_oof_campaign(&campaign) {
                Ok(matrix) if expect_leakage => {
                    bail!(
                        "expected OOF leakage but campaign joined {} samples and {} columns",
                        matrix.sample_ids.len(),
                        matrix.columns.len()
                    );
                }
                Ok(matrix) => {
                    println!(
                        "valid oof campaign: {} samples, {} columns",
                        matrix.sample_ids.len(),
                        matrix.columns.len()
                    );
                }
                Err(DagMlError::OofLeakage(report)) if expect_leakage => {
                    println!(
                        "expected oof leakage refused at {}: {} violator(s)",
                        report.node_id,
                        report.violators.len()
                    );
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("invalid OOF campaign at {}", path.display()));
                }
            }
        }
        Command::FingerprintOofCampaign { path } => {
            let campaign: OofCampaign = read_json(&path, "OOF campaign")?;
            let fingerprint = oof_campaign_fingerprint(&campaign)
                .with_context(|| format!("invalid OOF campaign at {}", path.display()))?;
            println!("{fingerprint}");
        }
        Command::ValidateExecutionPlan {
            graph,
            campaign,
            controllers,
            plan_id,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            println!(
                "valid execution plan: {} node(s), {} controller(s), {} variant(s), fold_set={}",
                plan.node_plans.len(),
                plan.controller_manifests.len(),
                plan.variants.len(),
                plan.fold_set
                    .as_ref()
                    .map(|fold_set| fold_set.id.as_str())
                    .unwrap_or("none")
            );
        }
        Command::ValidateDataBinding {
            campaign,
            envelope,
            node,
            input,
        } => {
            let campaign_spec: CampaignSpec = read_json(&campaign, "campaign")?;
            campaign_spec.validate()?;
            let node_id = NodeId::new(node)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let binding = campaign_spec
                .data_bindings
                .get(&node_id)
                .and_then(|bindings| bindings.iter().find(|binding| binding.input_name == input))
                .with_context(|| format!("no data binding for node `{node_id}` input `{input}`"))?;
            binding.validate_envelope(&envelope)?;
            println!(
                "valid data binding: {}.{} -> {}",
                node_id, binding.input_name, binding.plan_fingerprint
            );
        }
        Command::RunMockCampaign {
            graph,
            campaign,
            controllers,
            envelope,
            plan_id,
            run_id,
            root_seed,
        } => {
            let graph_spec: GraphSpec = read_json(&graph, "graph")?;
            let campaign_spec: CampaignSpec = read_json(&campaign, "campaign")?;
            let registry = controller_registry_from_path(&controllers)?;
            let plan = build_execution_plan(plan_id, graph_spec, campaign_spec, &registry)
                .with_context(|| "failed to build execution plan")?;

            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = InMemoryDataProvider::with_envelope(
                ControllerId::new("controller:data.provider")?,
                envelope,
            )?;
            let runtime_controllers = mock_runtime_controllers(&plan)?;
            let mut ctx = RunContext::new(RunId::new(run_id)?, Some(root_seed));
            let results = SequentialScheduler
                .execute_campaign_phase_with_data_provider(
                    &plan,
                    &runtime_controllers,
                    &data_provider,
                    &mut ctx,
                    Phase::FitCv,
                )
                .with_context(|| "mock campaign execution failed")?;
            println!(
                "mock campaign run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len()
            );
        }
        Command::SelectCandidates {
            policy,
            candidates,
            groups,
            output,
        } => {
            let policy: SelectionPolicy = read_json(&policy, "selection policy")?;
            let candidates: Vec<CandidateScore> = read_json(&candidates, "candidate scores")?;
            if let Some(groups) = groups {
                let groups: BTreeMap<String, Vec<String>> = read_json(&groups, "candidate groups")?;
                let decisions = select_candidate_groups(&policy, &candidates, &groups)
                    .with_context(|| "selection failed")?;
                emit_json(output.as_ref(), &decisions, "selection decisions")?;
            } else {
                let decision =
                    select_candidate(&policy, &candidates).with_context(|| "selection failed")?;
                emit_json(output.as_ref(), &decision, "selection decision")?;
            }
        }
        Command::BuildBundle {
            graph,
            campaign,
            controllers,
            bundle_spec,
            output,
            plan_id,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let spec: BundleBuildSpec = read_json(&bundle_spec, "bundle build spec")?;
            let mut bundle = build_execution_bundle(
                spec.bundle_id,
                &plan,
                spec.selected_variant_id,
                spec.selections,
                spec.refit_artifacts,
            )
            .with_context(|| "failed to build execution bundle")?;
            bundle.metadata = spec.metadata;
            bundle.validate_against_plan(&plan)?;
            emit_json(output.as_ref(), &bundle, "execution bundle")?;
        }
        Command::ValidateBundle {
            bundle,
            graph,
            campaign,
            controllers,
            envelopes,
            replay_request,
            plan_id,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            bundle
                .validate_against_plan(&plan)
                .with_context(|| "execution bundle does not match plan")?;
            let envelope_map = read_replay_envelopes(&envelopes)?;
            if !envelope_map.is_empty() {
                bundle
                    .validate_replay_envelopes(&envelope_map)
                    .with_context(|| "replay envelopes do not match bundle")?;
            }
            if let Some(replay_request) = replay_request {
                let request: ReplayPhaseRequest =
                    read_json(&replay_request, "replay phase request")?;
                request
                    .validate_for_bundle(&bundle)
                    .with_context(|| "replay request does not match bundle")?;
            }
            println!(
                "valid bundle: {}, selection(s)={}, artifact(s)={}, data requirement(s)={}, replay envelope(s)={}",
                bundle.bundle_id,
                bundle.selections.len(),
                bundle.refit_artifacts.len(),
                bundle.data_requirements.len(),
                envelope_map.len()
            );
        }
        Command::RunMockReplay {
            graph,
            campaign,
            controllers,
            bundle,
            replay_request,
            envelopes,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let replay_request: ReplayPhaseRequest =
                read_json(&replay_request, "replay phase request")?;
            let envelope_map = read_replay_envelopes(&envelopes)?;
            if envelope_map.is_empty() {
                bail!("run-mock-replay requires at least one --envelope KEY=PATH argument");
            }

            let mut data_provider =
                InMemoryDataProvider::new(ControllerId::new("controller:data.provider")?);
            for envelope in envelope_map.values() {
                data_provider.register_envelope(envelope.clone())?;
            }
            let artifact_store = mock_artifact_store(&plan, &bundle)?;
            let runtime_controllers = mock_runtime_controllers(&plan)?;
            let mut ctx = RunContext::new(RunId::new(run_id)?, Some(root_seed));
            let results = SequentialScheduler
                .execute_bundle_replay(
                    dag_ml_core::BundleReplayExecution {
                        plan: &plan,
                        bundle: &bundle,
                        replay_request: &replay_request,
                        controllers: &runtime_controllers,
                        data_provider: &data_provider,
                        artifact_store: &artifact_store,
                        data_envelopes: &envelope_map,
                    },
                    &mut ctx,
                )
                .with_context(|| "mock replay execution failed")?;
            println!(
                "mock replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} artifact handle(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                artifact_store.len()
            );
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct BundleBuildSpec {
    pub bundle_id: BundleId,
    #[serde(default)]
    pub selected_variant_id: Option<VariantId>,
    #[serde(default)]
    pub selections: BTreeMap<String, SelectionDecision>,
    #[serde(default)]
    pub refit_artifacts: Vec<RefitArtifactRecord>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

struct CliMockController {
    id: ControllerId,
}

impl RuntimeController for CliMockController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> dag_ml_core::Result<NodeResult> {
        for binding in &task.node_plan.data_bindings {
            let key = format!("data:{}", binding.input_name);
            let handle = task.input_handles.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data handle `{key}`",
                    task.node_plan.node_id
                ))
            })?;
            if handle.kind != HandleKind::Data {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received non-data handle for `{key}`",
                    task.node_plan.node_id
                )));
            }
        }

        if task.phase == Phase::Predict
            && matches!(task.node_plan.kind, dag_ml_core::NodeKind::Model)
        {
            let artifact_handles = task
                .input_handles
                .iter()
                .filter(|(key, _)| key.starts_with("artifact:"))
                .collect::<Vec<_>>();
            if artifact_handles.is_empty() {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive a replay artifact handle",
                    task.node_plan.node_id
                )));
            }
            for (key, handle) in artifact_handles {
                if !matches!(handle.kind, HandleKind::Model | HandleKind::Artifact) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received invalid replay artifact handle `{key}`",
                        task.node_plan.node_id
                    )));
                }
            }
        }

        let output = HandleRef {
            handle: stable_handle(task.node_plan.node_id.as_str()),
            kind: HandleKind::Data,
            owner_controller: self.id.clone(),
        };
        let predictions = if matches!(task.node_plan.kind, dag_ml_core::NodeKind::Model) {
            vec![PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: prediction_partition_for_phase(task.phase),
                fold_id: task.fold_id.clone(),
                sample_ids: vec![SampleId::new("sample:mock")?],
                values: vec![vec![stable_handle(task.node_plan.node_id.as_str()) as f64]],
                target_names: vec!["y".to_string()],
            }]
        } else {
            Vec::new()
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([("out".to_string(), output)]),
            predictions,
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{:?}:{}:{}",
                    task.node_plan.node_id,
                    task.phase,
                    task.variant_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "base".to_string()),
                    task.fold_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "nofold".to_string())
                ))?,
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

fn mock_runtime_controllers(
    plan: &dag_ml_core::ExecutionPlan,
) -> Result<RuntimeControllerRegistry> {
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        registry.register(Box::new(CliMockController {
            id: controller_id.clone(),
        }))?;
    }
    Ok(registry)
}

fn mock_artifact_store(
    plan: &dag_ml_core::ExecutionPlan,
    bundle: &ExecutionBundle,
) -> Result<InMemoryArtifactStore> {
    bundle.validate_against_plan(plan)?;
    let mut store = InMemoryArtifactStore::new();
    for artifact in &bundle.refit_artifacts {
        let node_plan = plan.node_plans.get(&artifact.node_id).with_context(|| {
            format!(
                "bundle artifact `{}` references unknown node `{}`",
                artifact.artifact.id, artifact.node_id
            )
        })?;
        let kind = if matches!(node_plan.kind, dag_ml_core::NodeKind::Model) {
            HandleKind::Model
        } else {
            HandleKind::Artifact
        };
        store.register(
            artifact,
            HandleRef {
                handle: stable_handle(artifact.artifact.id.as_str()),
                kind,
                owner_controller: artifact.controller_id.clone(),
            },
        )?;
    }
    Ok(store)
}

fn prediction_partition_for_phase(phase: Phase) -> PredictionPartition {
    match phase {
        Phase::FitCv => PredictionPartition::Validation,
        Phase::Refit | Phase::Predict => PredictionPartition::Final,
        Phase::Explain => PredictionPartition::Final,
        Phase::Compile | Phase::Plan | Phase::Select => PredictionPartition::Test,
    }
}

fn stable_handle(value: &str) -> u64 {
    value.bytes().fold(17u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u64)
    })
}

fn build_plan_from_paths(
    graph: &PathBuf,
    campaign: &PathBuf,
    controllers: &PathBuf,
    plan_id: String,
) -> Result<dag_ml_core::ExecutionPlan> {
    let graph_spec: GraphSpec = read_json(graph, "graph")?;
    let campaign_spec: CampaignSpec = read_json(campaign, "campaign")?;
    let registry = controller_registry_from_path(controllers)?;
    build_execution_plan(plan_id, graph_spec, campaign_spec, &registry)
        .with_context(|| "failed to build execution plan")
}

fn controller_registry_from_path(path: &PathBuf) -> Result<ControllerRegistry> {
    let controller_manifests: Vec<ControllerManifest> =
        read_json(path, "controller manifest list")?;
    let mut registry = ControllerRegistry::new();
    for manifest in controller_manifests {
        registry.register(manifest)?;
    }
    Ok(registry)
}

fn read_replay_envelopes(args: &[String]) -> Result<BTreeMap<String, ExternalDataPlanEnvelope>> {
    let mut envelopes = BTreeMap::new();
    for arg in args {
        let (key, path) = arg
            .split_once('=')
            .with_context(|| format!("envelope `{arg}` must use KEY=PATH format"))?;
        if key.trim().is_empty() {
            bail!("envelope key is empty in `{arg}`");
        }
        if path.trim().is_empty() {
            bail!("envelope path is empty for key `{key}`");
        }
        let envelope_path = PathBuf::from(path);
        let envelope: ExternalDataPlanEnvelope =
            read_json(&envelope_path, "external data-plan envelope")?;
        if envelopes.insert(key.to_string(), envelope).is_some() {
            bail!("duplicate envelope key `{key}`");
        }
    }
    Ok(envelopes)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf, label: &str) -> Result<T> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read {label} JSON at {}", path.display()))?;
    serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse {label} JSON at {}", path.display()))
}

fn emit_json<T: Serialize>(output: Option<&PathBuf>, value: &T, label: &str) -> Result<()> {
    let mut data = serde_json::to_vec_pretty(value)?;
    data.push(b'\n');
    if let Some(output) = output {
        std::fs::write(output, &data)
            .with_context(|| format!("failed to write {label} JSON at {}", output.display()))?;
        println!("wrote {label}: {}", output.display());
    } else {
        println!("{}", String::from_utf8(data)?);
    }
    Ok(())
}
