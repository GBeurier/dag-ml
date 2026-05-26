use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command as ProcessCommand, Stdio};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_core::{
    build_execution_bundle, build_execution_plan, oof_campaign_fingerprint, select_candidate,
    select_candidate_groups, validate_oof_campaign, BundleId, CampaignSpec, CandidateScore,
    ControllerId, ControllerManifest, ControllerRegistry, DagMlError, DataRequestPartition,
    ExecutionBundle, ExternalDataPlanEnvelope, GraphSpec, HandleKind, HandleRef,
    InMemoryArtifactStore, InMemoryDataProvider, LineageId, LineageRecord, NodeId, NodeResult,
    NodeTask, OofCampaign, Phase, PredictionBlock, PredictionPartition, RefitArtifactRecord,
    ReplayPhaseRequest, RunContext, RunId, RuntimeController, RuntimeControllerRegistry, SampleId,
    SelectionDecision, SelectionPolicy, SequentialScheduler, VariantId,
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
    RunProcessCampaign {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long)]
        persistent: bool,
        #[arg(long, default_value = "plan:cli.process")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process")]
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
    RunProcessReplay {
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
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long)]
        persistent: bool,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.replay")]
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
                "mock campaign run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len()
            );
        }
        Command::RunProcessCampaign {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            persistent,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = InMemoryDataProvider::with_envelope(
                ControllerId::new("controller:data.provider")?,
                envelope,
            )?;
            let runtime_controllers = if persistent {
                persistent_process_runtime_controllers(&plan, adapter)?
            } else {
                process_runtime_controllers(&plan, adapter)?
            };
            let mut ctx = RunContext::new(RunId::new(run_id)?, Some(root_seed));
            let results = SequentialScheduler
                .execute_campaign_phase_with_data_provider(
                    &plan,
                    &runtime_controllers,
                    &data_provider,
                    &mut ctx,
                    Phase::FitCv,
                )
                .with_context(|| "process campaign execution failed")?;
            println!(
                "process campaign run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len()
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
                "mock replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), {} artifact handle(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                artifact_store.len()
            );
        }
        Command::RunProcessReplay {
            graph,
            campaign,
            controllers,
            bundle,
            replay_request,
            envelopes,
            adapter,
            persistent,
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
                bail!("run-process-replay requires at least one --envelope KEY=PATH argument");
            }

            let mut data_provider =
                InMemoryDataProvider::new(ControllerId::new("controller:data.provider")?);
            for envelope in envelope_map.values() {
                data_provider.register_envelope(envelope.clone())?;
            }
            let artifact_store = mock_artifact_store(&plan, &bundle)?;
            let runtime_controllers = if persistent {
                persistent_process_runtime_controllers(&plan, adapter)?
            } else {
                process_runtime_controllers(&plan, adapter)?
            };
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
                .with_context(|| "process replay execution failed")?;
            println!(
                "process replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), {} artifact handle(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
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

struct ProcessRuntimeController {
    id: ControllerId,
    adapter: PathBuf,
}

struct PersistentProcessRuntimeController {
    id: ControllerId,
    adapter: PathBuf,
    session: RefCell<Option<PersistentProcessSession>>,
}

struct PersistentProcessSession {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl PersistentProcessSession {
    fn spawn(controller_id: &ControllerId, adapter: &Path) -> dag_ml_core::Result<Self> {
        let mut command = process_adapter_command(adapter, ProcessAdapterMode::Jsonl);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::inherit());
        let mut child = command.spawn().map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to spawn persistent adapter `{}`: {err}",
                adapter.display()
            ))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` persistent adapter stdin was not available"
            ))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` persistent adapter stdout was not available"
            ))
        })?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn invoke(
        &mut self,
        controller_id: &ControllerId,
        adapter: &Path,
        task: &NodeTask,
    ) -> dag_ml_core::Result<NodeResult> {
        serde_json::to_writer(&mut self.stdin, task).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to serialize persistent task JSON: {err}"
            ))
        })?;
        self.stdin.write_all(b"\n").map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to write persistent task JSON: {err}"
            ))
        })?;
        self.stdin.flush().map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to flush persistent task JSON: {err}"
            ))
        })?;

        let mut line = String::new();
        let bytes = self.stdout.read_line(&mut line).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to read persistent adapter `{}`: {err}",
                adapter.display()
            ))
        })?;
        if bytes == 0 {
            let status = self
                .child
                .try_wait()
                .map(|status| status.map(|status| status.to_string()))
                .unwrap_or_else(|err| Some(format!("status unavailable: {err}")))
                .unwrap_or_else(|| "still running".to_string());
            return Err(DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` persistent adapter `{}` returned EOF ({status})",
                adapter.display()
            )));
        }
        serde_json::from_str(&line).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` persistent adapter `{}` returned invalid NodeResult JSON: {err}",
                adapter.display()
            ))
        })
    }
}

impl Drop for PersistentProcessSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl RuntimeController for ProcessRuntimeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> dag_ml_core::Result<NodeResult> {
        let mut command = process_adapter_command(&self.adapter, ProcessAdapterMode::OneShot);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = command.spawn().map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{}` failed to spawn adapter `{}`: {err}",
                self.id,
                self.adapter.display()
            ))
        })?;

        {
            let mut stdin = child.stdin.take().ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "controller `{}` adapter stdin was not available",
                    self.id
                ))
            })?;
            serde_json::to_writer(&mut stdin, task).map_err(|err| {
                DagMlError::RuntimeValidation(format!(
                    "controller `{}` failed to serialize task JSON: {err}",
                    self.id
                ))
            })?;
            stdin.write_all(b"\n").map_err(|err| {
                DagMlError::RuntimeValidation(format!(
                    "controller `{}` failed to write task JSON: {err}",
                    self.id
                ))
            })?;
        }

        let output = child.wait_with_output().map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{}` failed while waiting for adapter `{}`: {err}",
                self.id,
                self.adapter.display()
            ))
        })?;
        if !output.status.success() {
            return Err(DagMlError::RuntimeValidation(format!(
                "controller `{}` adapter `{}` exited with status {}: {}",
                self.id,
                self.adapter.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        if output.stdout.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "controller `{}` adapter `{}` returned empty stdout",
                self.id,
                self.adapter.display()
            )));
        }
        serde_json::from_slice(&output.stdout).map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{}` adapter `{}` returned invalid NodeResult JSON: {err}",
                self.id,
                self.adapter.display()
            ))
        })
    }
}

impl RuntimeController for PersistentProcessRuntimeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> dag_ml_core::Result<NodeResult> {
        let mut session = self.session.borrow_mut();
        if session.is_none() {
            *session = Some(PersistentProcessSession::spawn(&self.id, &self.adapter)?);
        }
        session
            .as_mut()
            .expect("session was initialized")
            .invoke(&self.id, &self.adapter, task)
    }
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
            if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received non-data/data-view handle for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if !task.data_views.contains_key(&key) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data view spec for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if task.phase == Phase::FitCv && task.fold_id.is_some() {
                let validation_key = format!("{key}:validation");
                let validation_view = task.data_views.get(&validation_key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive validation data view spec for `{validation_key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if validation_view.partition != DataRequestPartition::FoldValidation {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-validation data view for `{validation_key}`",
                        task.node_plan.node_id
                    )));
                }
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
            let sample_ids = prediction_sample_ids_for_task(task)?;
            vec![PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: prediction_partition_for_phase(task.phase),
                fold_id: if task.phase == Phase::FitCv {
                    task.fold_id.clone()
                } else {
                    None
                },
                sample_ids: sample_ids.clone(),
                values: vec![
                    vec![stable_handle(task.node_plan.node_id.as_str()) as f64];
                    sample_ids.len()
                ],
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

fn process_runtime_controllers(
    plan: &dag_ml_core::ExecutionPlan,
    adapter: PathBuf,
) -> Result<RuntimeControllerRegistry> {
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        registry.register(Box::new(ProcessRuntimeController {
            id: controller_id.clone(),
            adapter: adapter.clone(),
        }))?;
    }
    Ok(registry)
}

fn persistent_process_runtime_controllers(
    plan: &dag_ml_core::ExecutionPlan,
    adapter: PathBuf,
) -> Result<RuntimeControllerRegistry> {
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        registry.register(Box::new(PersistentProcessRuntimeController {
            id: controller_id.clone(),
            adapter: adapter.clone(),
            session: RefCell::new(None),
        }))?;
    }
    Ok(registry)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessAdapterMode {
    OneShot,
    Jsonl,
}

fn process_adapter_command(adapter: &Path, mode: ProcessAdapterMode) -> ProcessCommand {
    let mut command = if adapter.extension().and_then(|extension| extension.to_str()) == Some("py")
    {
        let mut command = ProcessCommand::new("python3");
        command.arg(adapter);
        command
    } else {
        ProcessCommand::new(adapter.as_os_str())
    };
    if mode == ProcessAdapterMode::Jsonl {
        command.arg("--jsonl");
    }
    command
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

fn prediction_sample_ids_for_task(task: &NodeTask) -> dag_ml_core::Result<Vec<SampleId>> {
    if task.phase == Phase::FitCv {
        if let Some(view) = task
            .data_views
            .values()
            .find(|view| view.partition == DataRequestPartition::FoldValidation)
        {
            if let Some(sample_ids) = &view.sample_ids {
                if !sample_ids.is_empty() {
                    return Ok(sample_ids.clone());
                }
            }
        }
    }
    Ok(vec![SampleId::new("sample:mock")?])
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
