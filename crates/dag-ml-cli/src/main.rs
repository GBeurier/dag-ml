use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command as ProcessCommand, Stdio};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_core::{
    build_execution_bundle, build_execution_bundle_with_prediction_contracts, build_execution_plan,
    build_prediction_cache_payload, build_prediction_cache_record, oof_campaign_fingerprint,
    select_candidate, select_candidate_groups, validate_oof_campaign, ArtifactId, BundleId,
    BundlePredictionCachePayload, BundlePredictionCachePayloadSet, BundlePredictionCacheRecord,
    BundlePredictionRequirement, CampaignSpec, CandidateScore, ColumnarPredictionCacheStore,
    ControllerId, ControllerManifest, ControllerRegistry, DagMlError, DataRequestPartition,
    ExecutionBundle, ExternalDataPlanEnvelope, FilePredictionCacheStore, GraphSpec, HandleKind,
    HandleRef, InMemoryArtifactStore, InMemoryDataProvider, LineageId, LineageRecord,
    MetricObjective, NodeId, NodeResult, NodeTask, OofCampaign, Phase, PredictionBlock,
    PredictionPartition, RefitArtifactRecord, ReplayPhaseRequest, RunContext, RunId,
    RuntimeController, RuntimeControllerRegistry, RuntimePredictionCacheStore, SampleId,
    SelectionDecision, SelectionMetric, SelectionPolicy, SequentialScheduler, VariantId,
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
    ValidateSklearnComplexDemo {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        report: PathBuf,
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
    PrintExecutionSchedule {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long, default_value = "FIT_CV")]
        phase: String,
        #[arg(long)]
        output: Option<PathBuf>,
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
    RunMockRefitBundle {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "bundle:cli.refit")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long, default_value = "plan:cli.refit")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.refit")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
    },
    RunProcessRefitBundle {
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
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "bundle:cli.process.refit")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long, default_value = "plan:cli.process.refit")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.refit")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
    },
    RunProcessCvRefitBundle {
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
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        prediction_cache_output: Option<PathBuf>,
        #[arg(long, default_value = "bundle:cli.process.cv.refit")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long)]
        selections: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.process.cv.refit")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.cv.refit")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
    },
    RunProcessCvRefitReplay {
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
        #[arg(long, default_value = "bundle:cli.process.cv.refit.replay")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long)]
        selections: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.process.cv.refit.replay")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.cv.refit.replay")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
    },
    RunProcessRefitReplay {
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
        #[arg(long, default_value = "bundle:cli.process.refit.replay")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long, default_value = "plan:cli.process.refit.replay")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.refit.replay")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
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
        #[arg(long)]
        prediction_cache_payload: Option<PathBuf>,
        #[arg(long)]
        prediction_cache_store: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
    },
    ValidatePredictionCache {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        payload: PathBuf,
    },
    ExportPredictionCacheStore {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        payload: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    ValidatePredictionCacheStore {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        store_dir: PathBuf,
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
        #[arg(long)]
        prediction_cache_payload: Option<PathBuf>,
        #[arg(long)]
        prediction_cache_store: Option<PathBuf>,
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
        #[arg(long)]
        prediction_cache_payload: Option<PathBuf>,
        #[arg(long)]
        prediction_cache_store: Option<PathBuf>,
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
        Command::ValidateSklearnComplexDemo { campaign, report } => {
            let campaign: OofCampaign = read_json(&campaign, "sklearn complex OOF campaign")?;
            let report: serde_json::Value = read_json(&report, "sklearn complex report")?;
            let summary = validate_sklearn_complex_demo(&campaign, &report)
                .with_context(|| "sklearn complex demo validation failed")?;
            println!(
                "valid sklearn complex demo: {} sample(s), {} OOF column(s), {} branch selection(s), merge={}",
                summary.sample_count,
                summary.oof_column_count,
                summary.branch_selection_count,
                summary.selected_merge_node
            );
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
        Command::PrintExecutionSchedule {
            graph,
            campaign,
            controllers,
            phase,
            output,
            plan_id,
        } => {
            let phase = parse_phase(&phase)?;
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let schedule = plan.campaign_phase_schedule(phase)?;
            emit_json(output.as_ref(), &schedule, "execution schedule")?;
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
            campaign_spec
                .validate_data_envelope_relations(&envelope)
                .with_context(|| "data envelope relations violate campaign folds")?;
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
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
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
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
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
        Command::RunMockRefitBundle {
            graph,
            campaign,
            controllers,
            envelope,
            output,
            bundle_id,
            variant_id,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let runtime_controllers = mock_runtime_controllers_with_refit_artifacts(&plan)?;
            let captured = build_bundle_from_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections: BTreeMap::new(),
                run_id,
                root_seed,
            })
            .with_context(|| "mock refit bundle capture failed")?;
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
        }
        Command::RunProcessRefitBundle {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            persistent,
            output,
            bundle_id,
            variant_id,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let runtime_controllers = if persistent {
                persistent_process_runtime_controllers(&plan, adapter)?
            } else {
                process_runtime_controllers(&plan, adapter)?
            };
            let captured = build_bundle_from_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections: BTreeMap::new(),
                run_id,
                root_seed,
            })
            .with_context(|| "process refit bundle capture failed")?;
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
        }
        Command::RunProcessCvRefitBundle {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            persistent,
            output,
            prediction_cache_output,
            bundle_id,
            variant_id,
            selections,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let runtime_controllers = if persistent {
                persistent_process_runtime_controllers(&plan, adapter)?
            } else {
                process_runtime_controllers(&plan, adapter)?
            };
            let selections = read_selection_decisions(selections.as_ref())?;
            let captured = build_bundle_from_cv_then_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections,
                run_id,
                root_seed,
            })
            .with_context(|| "process CV+refit bundle capture failed")?;
            println!(
                "process cv refit bundle run: {} fit_cv result(s), {} OOF prediction block(s), {} refit result(s), {} captured artifact handle(s), {} prediction cache(s)",
                captured.fit_cv_result_count,
                captured.fit_cv_oof_prediction_block_count,
                captured.refit_result_count,
                captured.artifact_store.len(),
                captured.bundle.prediction_caches.len()
            );
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
            if let Some(path) = prediction_cache_output.as_ref() {
                let payload_set = BundlePredictionCachePayloadSet {
                    bundle_id: captured.bundle.bundle_id.clone(),
                    schema_version: dag_ml_core::PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
                    caches: captured.prediction_cache_payloads.clone(),
                };
                payload_set
                    .validate_against_bundle(&captured.bundle)
                    .with_context(|| "prediction cache payloads do not match captured bundle")?;
                emit_json(Some(path), &payload_set, "prediction cache payload set")?;
            }
        }
        Command::RunProcessCvRefitReplay {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            bundle_id,
            variant_id,
            selections,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope.clone())?;
            let runtime_controllers = persistent_process_runtime_controllers(&plan, adapter)?;
            let selections = read_selection_decisions(selections.as_ref())?;
            let captured = build_bundle_from_cv_then_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections,
                run_id: run_id.clone(),
                root_seed,
            })
            .with_context(|| "process CV+refit capture before replay failed")?;
            let envelope_map = replay_envelope_map_for_bundle(&captured.bundle, &envelope);
            let replay_request = ReplayPhaseRequest {
                bundle_id: captured.bundle.bundle_id.clone(),
                phase: Phase::Predict,
                data_envelope_keys: envelope_map.keys().cloned().collect(),
            };
            let mut replay_ctx =
                RunContext::new(RunId::new(format!("{run_id}:predict"))?, Some(root_seed));
            let replay_results = SequentialScheduler
                .execute_bundle_replay(
                    dag_ml_core::BundleReplayExecution {
                        plan: &plan,
                        bundle: &captured.bundle,
                        replay_request: &replay_request,
                        prediction_cache_store: None,
                        controllers: &runtime_controllers,
                        data_provider: &data_provider,
                        artifact_store: &captured.artifact_store,
                        data_envelopes: &envelope_map,
                    },
                    &mut replay_ctx,
                )
                .with_context(|| "process replay after CV+refit capture failed")?;
            println!(
                "process cv refit replay run: {} fit_cv result(s), {} OOF prediction block(s), {} refit result(s), {} replay result(s), {} replay prediction block(s), {} captured artifact handle(s), {} prediction cache(s)",
                captured.fit_cv_result_count,
                captured.fit_cv_oof_prediction_block_count,
                captured.refit_result_count,
                replay_results.len(),
                replay_ctx.prediction_store.blocks().len(),
                captured.artifact_store.len(),
                captured.bundle.prediction_caches.len()
            );
        }
        Command::RunProcessRefitReplay {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            bundle_id,
            variant_id,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope.clone())?;
            let runtime_controllers = persistent_process_runtime_controllers(&plan, adapter)?;
            let captured = build_bundle_from_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections: BTreeMap::new(),
                run_id: run_id.clone(),
                root_seed,
            })
            .with_context(|| "process refit capture before replay failed")?;
            let envelope_map = replay_envelope_map_for_bundle(&captured.bundle, &envelope);
            let replay_request = ReplayPhaseRequest {
                bundle_id: captured.bundle.bundle_id.clone(),
                phase: Phase::Predict,
                data_envelope_keys: envelope_map.keys().cloned().collect(),
            };
            let mut replay_ctx =
                RunContext::new(RunId::new(format!("{run_id}:predict"))?, Some(root_seed));
            let replay_results = SequentialScheduler
                .execute_bundle_replay(
                    dag_ml_core::BundleReplayExecution {
                        plan: &plan,
                        bundle: &captured.bundle,
                        replay_request: &replay_request,
                        prediction_cache_store: None,
                        controllers: &runtime_controllers,
                        data_provider: &data_provider,
                        artifact_store: &captured.artifact_store,
                        data_envelopes: &envelope_map,
                    },
                    &mut replay_ctx,
                )
                .with_context(|| "process replay after refit capture failed")?;
            println!(
                "process refit replay run: {} refit result(s), {} replay result(s), {} replay prediction block(s), {} captured artifact handle(s)",
                captured.refit_result_count,
                replay_results.len(),
                replay_ctx.prediction_store.blocks().len(),
                captured.artifact_store.len()
            );
        }
        Command::ValidateBundle {
            bundle,
            graph,
            campaign,
            controllers,
            envelopes,
            replay_request,
            prediction_cache_payload,
            prediction_cache_store,
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
            let prediction_cache_payloads =
                read_optional_prediction_cache_payload(prediction_cache_payload.as_ref())?;
            if let Some(payloads) = prediction_cache_payloads.as_ref() {
                payloads
                    .validate_against_bundle(&bundle)
                    .with_context(|| "prediction cache payload set does not match bundle")?;
            }
            let file_prediction_cache_store = prediction_cache_store
                .as_ref()
                .map(|store_dir| validate_file_prediction_cache_store(&bundle, store_dir))
                .transpose()?;
            if prediction_cache_payloads.is_some() && file_prediction_cache_store.is_some() {
                bail!(
                    "use either --prediction-cache-payload or --prediction-cache-store, not both"
                );
            }
            if let Some(replay_request) = replay_request {
                let request: ReplayPhaseRequest =
                    read_json(&replay_request, "replay phase request")?;
                if prediction_cache_payloads.is_some() {
                    request
                        .validate_for_bundle_with_prediction_cache_payloads(
                            &bundle,
                            prediction_cache_payloads.as_ref(),
                        )
                        .with_context(|| "replay request does not match bundle")?;
                } else {
                    request
                        .validate_for_bundle_with_prediction_cache_store(
                            &bundle,
                            file_prediction_cache_store.is_some(),
                        )
                        .with_context(|| "replay request does not match bundle")?;
                }
            }
            println!(
                "valid bundle: {}, selection(s)={}, artifact(s)={}, prediction requirement(s)={}, prediction cache(s)={}, prediction cache payload(s)={}, prediction cache store cache(s)={}, data requirement(s)={}, replay envelope(s)={}",
                bundle.bundle_id,
                bundle.selections.len(),
                bundle.refit_artifacts.len(),
                bundle.prediction_requirements.len(),
                bundle.prediction_caches.len(),
                prediction_cache_payloads
                    .as_ref()
                    .map_or(0, |payloads| payloads.caches.len()),
                file_prediction_cache_store
                    .as_ref()
                    .map_or(0, |store| store.manifest().caches.len()),
                bundle.data_requirements.len(),
                envelope_map.len()
            );
        }
        Command::ValidatePredictionCache { bundle, payload } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let payload: BundlePredictionCachePayloadSet =
                read_json(&payload, "prediction cache payload set")?;
            payload
                .validate_against_bundle(&bundle)
                .with_context(|| "prediction cache payload set does not match bundle")?;
            println!(
                "valid prediction cache payload set: bundle={}, cache(s)={}",
                payload.bundle_id,
                payload.caches.len()
            );
        }
        Command::ExportPredictionCacheStore {
            bundle,
            payload,
            output_dir,
        } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let payload: BundlePredictionCachePayloadSet =
                read_json(&payload, "prediction cache payload set")?;
            let manifest =
                FilePredictionCacheStore::write_payload_set(&output_dir, &bundle, &payload)
                    .with_context(|| "failed to export prediction cache store")?;
            println!(
                "wrote prediction cache store: bundle={}, cache(s)={}, dir={}",
                manifest.bundle_id,
                manifest.caches.len(),
                output_dir.display()
            );
        }
        Command::ValidatePredictionCacheStore { bundle, store_dir } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let store = validate_file_prediction_cache_store(&bundle, &store_dir)?;
            println!(
                "valid prediction cache store: bundle={}, cache(s)={}, dir={}",
                store.manifest().bundle_id,
                store.manifest().caches.len(),
                store_dir.display()
            );
        }
        Command::RunMockReplay {
            graph,
            campaign,
            controllers,
            bundle,
            replay_request,
            prediction_cache_payload,
            prediction_cache_store,
            envelopes,
            plan_id,
            run_id,
            root_seed,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let replay_request: ReplayPhaseRequest =
                read_json(&replay_request, "replay phase request")?;
            let prediction_cache_payloads =
                read_optional_prediction_cache_payload(prediction_cache_payload.as_ref())?;
            let prediction_cache_store = optional_prediction_cache_store(
                &bundle,
                prediction_cache_payloads,
                prediction_cache_store.as_ref(),
            )?;
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
                        prediction_cache_store: prediction_cache_store
                            .as_ref()
                            .map(|store| store as &dyn RuntimePredictionCacheStore),
                        controllers: &runtime_controllers,
                        data_provider: &data_provider,
                        artifact_store: &artifact_store,
                        data_envelopes: &envelope_map,
                    },
                    &mut ctx,
                )
                .with_context(|| "mock replay execution failed")?;
            println!(
                "mock replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), {} artifact handle(s), {} prediction cache handle(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                artifact_store.len(),
                prediction_cache_store
                    .as_ref()
                    .map(CliPredictionCacheStore::materialization_record_count)
                    .unwrap_or(0)
            );
        }
        Command::RunProcessReplay {
            graph,
            campaign,
            controllers,
            bundle,
            replay_request,
            prediction_cache_payload,
            prediction_cache_store,
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
            let prediction_cache_payloads =
                read_optional_prediction_cache_payload(prediction_cache_payload.as_ref())?;
            let prediction_cache_store = optional_prediction_cache_store(
                &bundle,
                prediction_cache_payloads,
                prediction_cache_store.as_ref(),
            )?;
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
                        prediction_cache_store: prediction_cache_store
                            .as_ref()
                            .map(|store| store as &dyn RuntimePredictionCacheStore),
                        controllers: &runtime_controllers,
                        data_provider: &data_provider,
                        artifact_store: &artifact_store,
                        data_envelopes: &envelope_map,
                    },
                    &mut ctx,
                )
                .with_context(|| "process replay execution failed")?;
            println!(
                "process replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), {} artifact handle(s), {} prediction cache handle(s)",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                artifact_store.len(),
                prediction_cache_store
                    .as_ref()
                    .map(CliPredictionCacheStore::materialization_record_count)
                    .unwrap_or(0)
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

struct SklearnComplexDemoSummary {
    sample_count: usize,
    oof_column_count: usize,
    branch_selection_count: usize,
    selected_merge_node: String,
}

fn validate_sklearn_complex_demo(
    campaign: &OofCampaign,
    report: &serde_json::Value,
) -> Result<SklearnComplexDemoSummary> {
    let matrix = validate_oof_campaign(campaign)?;
    let sample_count = json_usize(report, "sample_count")?;
    if sample_count != matrix.sample_ids.len() {
        bail!(
            "report sample_count={} but OOF campaign has {} sample(s)",
            sample_count,
            matrix.sample_ids.len()
        );
    }
    if json_usize(report, "observation_count")? < sample_count {
        bail!("observation_count must be greater than or equal to sample_count");
    }

    let policy = SelectionPolicy {
        id: "select:sklearn_complex.rmse".to_string(),
        metric: SelectionMetric {
            name: "rmse".to_string(),
            objective: MetricObjective::Minimize,
        },
        require_finite: true,
    };
    let branch_candidates = metric_candidates(report, "branch_variant_metrics")?;
    let branch_groups = branch_groups_from_report(report)?;
    let branch_decisions = select_candidate_groups(&policy, &branch_candidates, &branch_groups)?;
    assert_report_branch_selections(report, &branch_decisions)?;

    let merge_candidates = metric_candidates(report, "merge_variant_metrics")?;
    let merge_decision = select_candidate(&policy, &merge_candidates)?;
    assert_report_merge_selection(report, &merge_decision)?;
    assert_report_refit_contract(report, &branch_decisions, &merge_decision)?;
    assert_report_leakage_controls(report)?;

    Ok(SklearnComplexDemoSummary {
        sample_count,
        oof_column_count: matrix.columns.len(),
        branch_selection_count: branch_decisions.len(),
        selected_merge_node: merge_decision.selected_candidate_id,
    })
}

fn metric_candidates(report: &serde_json::Value, key: &str) -> Result<Vec<CandidateScore>> {
    let metrics = report
        .get(key)
        .and_then(serde_json::Value::as_object)
        .with_context(|| format!("report `{key}` must be an object"))?;
    metrics
        .iter()
        .map(|(candidate_id, metrics)| {
            let metrics = metrics.as_object().with_context(|| {
                format!("report `{key}.{candidate_id}` must be a metric object")
            })?;
            Ok(CandidateScore {
                candidate_id: candidate_id.clone(),
                metrics: metrics
                    .iter()
                    .map(|(name, value)| {
                        let value = value.as_f64().with_context(|| {
                            format!("metric `{key}.{candidate_id}.{name}` must be numeric")
                        })?;
                        Ok((name.clone(), value))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?,
                metadata: BTreeMap::new(),
            })
        })
        .collect()
}

fn branch_groups_from_report(report: &serde_json::Value) -> Result<BTreeMap<String, Vec<String>>> {
    let groups = report
        .pointer("/sklearn_workflow/branch_variants")
        .and_then(serde_json::Value::as_object)
        .with_context(|| "report sklearn_workflow.branch_variants must be an object")?;
    groups
        .iter()
        .map(|(group_id, candidates)| {
            let candidates = candidates.as_array().with_context(|| {
                format!("report sklearn_workflow.branch_variants.{group_id} must be an array")
            })?;
            Ok((
                group_id.clone(),
                candidates
                    .iter()
                    .map(|candidate| {
                        candidate
                            .as_str()
                            .map(ToString::to_string)
                            .with_context(|| {
                                format!(
                                    "report sklearn_workflow.branch_variants.{group_id} must contain strings"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ))
        })
        .collect()
}

fn assert_report_branch_selections(
    report: &serde_json::Value,
    decisions: &BTreeMap<String, SelectionDecision>,
) -> Result<()> {
    let selected = report
        .get("selected_branch_variants")
        .and_then(serde_json::Value::as_object)
        .with_context(|| "report selected_branch_variants must be an object")?;
    if selected.len() != decisions.len() {
        bail!(
            "report has {} branch selection(s), core recomputed {}",
            selected.len(),
            decisions.len()
        );
    }
    for (branch_id, decision) in decisions {
        let reported = selected
            .get(branch_id)
            .with_context(|| format!("report missing selected branch `{branch_id}`"))?;
        let reported_node = json_string_at(reported, "producer_node")?;
        if reported_node != decision.selected_candidate_id {
            bail!(
                "branch `{branch_id}` selected `{reported_node}` in report but core selected `{}`",
                decision.selected_candidate_id
            );
        }
        let reported_score = json_f64_at(reported, "/score/rmse")?;
        assert_close(
            reported_score,
            decision.selected_score,
            &format!("branch `{branch_id}` selected rmse"),
        )?;
    }
    Ok(())
}

fn assert_report_merge_selection(
    report: &serde_json::Value,
    decision: &SelectionDecision,
) -> Result<()> {
    let selected = report
        .get("selected_merge_variant")
        .with_context(|| "report missing selected_merge_variant")?;
    let reported_node = json_string_at(selected, "producer_node")?;
    if reported_node != decision.selected_candidate_id {
        bail!(
            "report selected merge `{reported_node}` but core selected `{}`",
            decision.selected_candidate_id
        );
    }
    let reported_score = json_f64_at(selected, "/score/rmse")?;
    assert_close(
        reported_score,
        decision.selected_score,
        "merge selected rmse",
    )
}

fn assert_report_refit_contract(
    report: &serde_json::Value,
    branch_decisions: &BTreeMap<String, SelectionDecision>,
    merge_decision: &SelectionDecision,
) -> Result<()> {
    let final_refit = report
        .get("final_refit")
        .with_context(|| "report missing final_refit")?;
    let reported_base_nodes = final_refit
        .get("selected_base_nodes")
        .and_then(serde_json::Value::as_array)
        .with_context(|| "report final_refit.selected_base_nodes must be an array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .with_context(|| "final_refit.selected_base_nodes must contain strings")
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let expected_base_nodes = branch_decisions
        .values()
        .map(|decision| decision.selected_candidate_id.clone())
        .collect::<BTreeSet<_>>();
    if reported_base_nodes != expected_base_nodes {
        bail!("final_refit selected_base_nodes do not match core branch selections");
    }
    let selected_merge_node = json_string_at(final_refit, "selected_merge_node")?;
    if selected_merge_node != merge_decision.selected_candidate_id {
        bail!(
            "final_refit selected merge `{selected_merge_node}` but core selected `{}`",
            merge_decision.selected_candidate_id
        );
    }
    let sample_count = json_usize(report, "sample_count")?;
    let merge_refit_samples = json_usize_at(final_refit, "merge_refit_samples")?;
    if merge_refit_samples != sample_count {
        bail!(
            "final_refit merge_refit_samples={} but sample_count={sample_count}",
            merge_refit_samples
        );
    }

    let raw_shape = report
        .get("original_sample_feature_shape")
        .and_then(serde_json::Value::as_array)
        .with_context(|| "report original_sample_feature_shape must be an array")?;
    let raw_width = raw_shape
        .get(1)
        .and_then(serde_json::Value::as_u64)
        .with_context(|| "report original_sample_feature_shape[1] must be an integer")?
        as usize;
    let original_mode = report
        .pointer("/selected_merge_variant/original_feature_mode")
        .and_then(serde_json::Value::as_str)
        .with_context(|| "report selected_merge_variant.original_feature_mode must be a string")?;
    let expected_width = match original_mode {
        "none" => branch_decisions.len(),
        "metadata" => branch_decisions.len() + 4,
        "all" => branch_decisions.len() + raw_width,
        other => bail!("unknown selected merge original_feature_mode `{other}`"),
    };
    let reported_width = json_usize_at(final_refit, "merge_refit_features")?;
    if reported_width != expected_width {
        bail!(
            "final_refit merge_refit_features={} but expected {expected_width}",
            reported_width
        );
    }
    Ok(())
}

fn assert_report_leakage_controls(report: &serde_json::Value) -> Result<()> {
    let controls = report
        .get("leakage_controls")
        .and_then(serde_json::Value::as_object)
        .with_context(|| "report leakage_controls must be an object")?;
    for key in [
        "split_unit",
        "group_boundary",
        "validation_augmentation",
        "branch_selection",
        "merge_selection",
        "stacking_features",
        "heterogeneous_merge",
        "aggregation",
        "refit",
    ] {
        if !controls.contains_key(key) {
            bail!("report leakage_controls missing `{key}`");
        }
    }
    if controls
        .get("validation_augmentation")
        .and_then(serde_json::Value::as_str)
        != Some("disabled")
    {
        bail!("report leakage_controls.validation_augmentation must be disabled");
    }
    Ok(())
}

fn json_usize(value: &serde_json::Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("report `{key}` must be an integer"))
}

fn json_usize_at(value: &serde_json::Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("report field `{key}` must be an integer"))
}

fn json_string_at(value: &serde_json::Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
        .with_context(|| format!("report field `{key}` must be a string"))
}

fn json_f64_at(value: &serde_json::Value, pointer: &str) -> Result<f64> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_f64)
        .with_context(|| format!("report field `{pointer}` must be numeric"))
}

fn assert_close(actual: f64, expected: f64, label: &str) -> Result<()> {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
    if (actual - expected).abs() > tolerance {
        bail!("{label} mismatch: report={actual}, core={expected}");
    }
    Ok(())
}

struct CapturedRefitBundleInput<'a> {
    plan: &'a dag_ml_core::ExecutionPlan,
    data_provider: &'a InMemoryDataProvider,
    runtime_controllers: &'a RuntimeControllerRegistry,
    bundle_id: String,
    variant_id: Option<String>,
    selections: BTreeMap<String, SelectionDecision>,
    run_id: String,
    root_seed: u64,
}

struct CapturedRefitBundle {
    bundle: ExecutionBundle,
    artifact_store: InMemoryArtifactStore,
    prediction_cache_payloads: Vec<BundlePredictionCachePayload>,
    fit_cv_result_count: usize,
    fit_cv_oof_prediction_block_count: usize,
    refit_result_count: usize,
}

fn selected_refit_variant(
    plan: &dag_ml_core::ExecutionPlan,
    variant_id: Option<String>,
) -> Result<VariantId> {
    let selected_variant_id = match variant_id {
        Some(variant_id) => VariantId::new(variant_id)?,
        None => plan
            .variants
            .first()
            .map(|variant| variant.variant_id.clone())
            .with_context(|| "execution plan has no variants to refit")?,
    };
    if !plan
        .variants
        .iter()
        .any(|variant| variant.variant_id == selected_variant_id)
    {
        bail!(
            "unknown variant `{selected_variant_id}` for plan `{}`",
            plan.id
        );
    }
    Ok(selected_variant_id)
}

fn build_bundle_from_captured_refit(
    input: CapturedRefitBundleInput<'_>,
) -> Result<CapturedRefitBundle> {
    let selected_variant_id = selected_refit_variant(input.plan, input.variant_id)?;

    let mut artifact_store = InMemoryArtifactStore::new();
    let mut ctx = RunContext::new(RunId::new(input.run_id)?, Some(input.root_seed));
    ctx.variant_id = Some(selected_variant_id.clone());

    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider_and_artifact_store(
            input.plan,
            input.runtime_controllers,
            input.data_provider,
            &mut artifact_store,
            &mut ctx,
            Phase::Refit,
        )
        .with_context(|| "refit execution failed")?;
    if artifact_store.is_empty() {
        bail!("refit did not capture any refit artifacts");
    }

    let mut bundle = build_execution_bundle(
        BundleId::new(input.bundle_id)?,
        input.plan,
        Some(selected_variant_id),
        input.selections,
        artifact_store.refit_artifacts(),
    )
    .with_context(|| "failed to build execution bundle from refit artifacts")?;
    bundle.metadata.insert(
        "refit_result_count".to_string(),
        serde_json::json!(results.len()),
    );
    bundle.metadata.insert(
        "refit_lineage_count".to_string(),
        serde_json::json!(ctx.lineage.len()),
    );
    Ok(CapturedRefitBundle {
        bundle,
        artifact_store,
        prediction_cache_payloads: Vec::new(),
        fit_cv_result_count: 0,
        fit_cv_oof_prediction_block_count: 0,
        refit_result_count: results.len(),
    })
}

fn build_bundle_from_cv_then_captured_refit(
    input: CapturedRefitBundleInput<'_>,
) -> Result<CapturedRefitBundle> {
    let selected_variant_id = selected_refit_variant(input.plan, input.variant_id)?;

    let mut artifact_store = InMemoryArtifactStore::new();
    let mut ctx = RunContext::new(RunId::new(input.run_id)?, Some(input.root_seed));
    ctx.variant_id = Some(selected_variant_id.clone());

    let fit_cv_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            input.plan,
            input.runtime_controllers,
            input.data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .with_context(|| "FIT_CV execution before refit failed")?;
    let fit_cv_lineage_count = ctx.lineage.len();
    let fit_cv_oof_prediction_block_count = ctx
        .prediction_store
        .blocks()
        .iter()
        .filter(|block| block.partition == PredictionPartition::Validation)
        .count();
    if fit_cv_oof_prediction_block_count == 0 {
        bail!("FIT_CV did not produce any validation OOF prediction blocks before refit");
    }
    let prediction_requirements =
        oof_prediction_requirements(input.plan, ctx.prediction_store.blocks())?;
    let prediction_caches =
        oof_prediction_caches(&prediction_requirements, ctx.prediction_store.blocks())?;
    let prediction_cache_payloads =
        oof_prediction_cache_payloads(&prediction_requirements, ctx.prediction_store.blocks())?;
    let oof_prediction_summary = oof_prediction_summary(ctx.prediction_store.blocks())?;

    let refit_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider_and_artifact_store(
            input.plan,
            input.runtime_controllers,
            input.data_provider,
            &mut artifact_store,
            &mut ctx,
            Phase::Refit,
        )
        .with_context(|| "refit execution after FIT_CV failed")?;
    if artifact_store.is_empty() {
        bail!("refit did not capture any refit artifacts");
    }
    let refit_lineage_count = ctx.lineage.len().saturating_sub(fit_cv_lineage_count);
    let refit_prediction_block_count = ctx
        .prediction_store
        .blocks()
        .iter()
        .filter(|block| block.partition == PredictionPartition::Final)
        .count();

    let mut bundle = build_execution_bundle_with_prediction_contracts(
        BundleId::new(input.bundle_id)?,
        input.plan,
        Some(selected_variant_id),
        input.selections,
        artifact_store.refit_artifacts(),
        prediction_requirements,
        prediction_caches,
    )
    .with_context(|| "failed to build execution bundle from CV+refit artifacts")?;
    bundle.metadata.insert(
        "fit_cv_result_count".to_string(),
        serde_json::json!(fit_cv_results.len()),
    );
    bundle.metadata.insert(
        "fit_cv_lineage_count".to_string(),
        serde_json::json!(fit_cv_lineage_count),
    );
    bundle.metadata.insert(
        "fit_cv_oof_prediction_block_count".to_string(),
        serde_json::json!(fit_cv_oof_prediction_block_count),
    );
    bundle.metadata.insert(
        "oof_prediction_summary".to_string(),
        serde_json::json!(oof_prediction_summary),
    );
    bundle.metadata.insert(
        "refit_result_count".to_string(),
        serde_json::json!(refit_results.len()),
    );
    bundle.metadata.insert(
        "refit_lineage_count".to_string(),
        serde_json::json!(refit_lineage_count),
    );
    bundle.metadata.insert(
        "refit_prediction_block_count".to_string(),
        serde_json::json!(refit_prediction_block_count),
    );
    bundle.metadata.insert(
        "total_lineage_count".to_string(),
        serde_json::json!(ctx.lineage.len()),
    );
    bundle.validate_against_plan(input.plan)?;
    Ok(CapturedRefitBundle {
        bundle,
        artifact_store,
        prediction_cache_payloads,
        fit_cv_result_count: fit_cv_results.len(),
        fit_cv_oof_prediction_block_count,
        refit_result_count: refit_results.len(),
    })
}

fn replay_envelope_map_for_bundle(
    bundle: &ExecutionBundle,
    envelope: &ExternalDataPlanEnvelope,
) -> BTreeMap<String, ExternalDataPlanEnvelope> {
    bundle
        .data_requirements
        .iter()
        .map(|requirement| {
            (
                format!("{}.{}", requirement.node_id, requirement.input_name),
                envelope.clone(),
            )
        })
        .collect()
}

fn oof_prediction_requirements(
    plan: &dag_ml_core::ExecutionPlan,
    blocks: &[PredictionBlock],
) -> Result<Vec<BundlePredictionRequirement>> {
    let mut requirements = Vec::new();
    for edge in plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.contract.requires_oof)
    {
        let edge_blocks = blocks
            .iter()
            .filter(|block| {
                block.producer_node == edge.source.node_id
                    && block.partition == PredictionPartition::Validation
            })
            .collect::<Vec<_>>();
        if edge_blocks.is_empty() {
            bail!(
                "OOF prediction requirement `{}` -> `{}` has no validation blocks",
                edge.source.node_id,
                edge.target.node_id
            );
        }
        let summary = summarize_oof_blocks(&edge.source.node_id, &edge_blocks)?;
        requirements.push(BundlePredictionRequirement {
            producer_node: edge.source.node_id.clone(),
            source_port: edge.source.port_name.clone(),
            consumer_node: edge.target.node_id.clone(),
            target_port: edge.target.port_name.clone(),
            partition: PredictionPartition::Validation,
            fold_ids: summary
                .fold_ids
                .into_iter()
                .map(dag_ml_core::FoldId::new)
                .collect::<dag_ml_core::Result<Vec<_>>>()?,
            sample_ids: summary
                .sample_ids
                .into_iter()
                .map(SampleId::new)
                .collect::<dag_ml_core::Result<Vec<_>>>()?,
            prediction_width: summary.prediction_width.unwrap_or_default(),
            target_names: summary.target_names.unwrap_or_default(),
        });
    }
    requirements.sort_by_key(BundlePredictionRequirement::key);
    for requirement in &requirements {
        requirement.validate()?;
    }
    Ok(requirements)
}

fn oof_prediction_caches(
    requirements: &[BundlePredictionRequirement],
    blocks: &[PredictionBlock],
) -> dag_ml_core::Result<Vec<BundlePredictionCacheRecord>> {
    requirements
        .iter()
        .map(|requirement| build_prediction_cache_record(requirement, blocks))
        .collect()
}

fn oof_prediction_cache_payloads(
    requirements: &[BundlePredictionRequirement],
    blocks: &[PredictionBlock],
) -> dag_ml_core::Result<Vec<BundlePredictionCachePayload>> {
    requirements
        .iter()
        .map(|requirement| build_prediction_cache_payload(requirement, blocks))
        .collect()
}

#[derive(Default)]
struct OofPredictionSummary {
    block_count: usize,
    fold_ids: BTreeSet<String>,
    sample_ids: BTreeSet<String>,
    prediction_width: Option<usize>,
    target_names: Option<Vec<String>>,
}

fn oof_prediction_summary(blocks: &[PredictionBlock]) -> Result<Vec<serde_json::Value>> {
    let mut summaries = BTreeMap::<NodeId, OofPredictionSummary>::new();
    for block in blocks
        .iter()
        .filter(|block| block.partition == PredictionPartition::Validation)
    {
        let width = block.validate_shape()?;
        let entry = summaries.entry(block.producer_node.clone()).or_default();
        entry.block_count += 1;
        if let Some(fold_id) = &block.fold_id {
            entry.fold_ids.insert(fold_id.to_string());
        }
        entry
            .sample_ids
            .extend(block.sample_ids.iter().map(ToString::to_string));
        if entry
            .prediction_width
            .is_some_and(|expected| expected != width)
        {
            bail!(
                "OOF prediction summary for `{}` has inconsistent prediction width",
                block.producer_node
            );
        }
        entry.prediction_width = Some(width);
        if entry
            .target_names
            .as_ref()
            .is_some_and(|expected| expected != &block.target_names)
        {
            bail!(
                "OOF prediction summary for `{}` has inconsistent target names",
                block.producer_node
            );
        }
        entry.target_names = Some(block.target_names.clone());
    }
    Ok(summaries
        .into_iter()
        .map(|(producer_node, summary)| {
            serde_json::json!({
                "producer_node": producer_node,
                "block_count": summary.block_count,
                "fold_ids": summary.fold_ids.into_iter().collect::<Vec<_>>(),
                "sample_ids": summary.sample_ids.into_iter().collect::<Vec<_>>(),
                "prediction_width": summary.prediction_width.unwrap_or_default(),
                "target_names": summary.target_names.unwrap_or_default(),
            })
        })
        .collect())
}

fn summarize_oof_blocks(
    producer_node: &NodeId,
    blocks: &[&PredictionBlock],
) -> Result<OofPredictionSummary> {
    let mut summary = OofPredictionSummary::default();
    for block in blocks {
        let width = block.validate_shape()?;
        summary.block_count += 1;
        if let Some(fold_id) = &block.fold_id {
            summary.fold_ids.insert(fold_id.to_string());
        }
        summary
            .sample_ids
            .extend(block.sample_ids.iter().map(ToString::to_string));
        if summary
            .prediction_width
            .is_some_and(|expected| expected != width)
        {
            bail!("OOF prediction requirement for `{producer_node}` has inconsistent prediction width");
        }
        summary.prediction_width = Some(width);
        let target_names = if block.target_names.is_empty() {
            (0..width).map(|index| format!("p{index}")).collect()
        } else {
            block.target_names.clone()
        };
        if summary
            .target_names
            .as_ref()
            .is_some_and(|expected| expected != &target_names)
        {
            bail!("OOF prediction requirement for `{producer_node}` has inconsistent target names");
        }
        summary.target_names = Some(target_names);
    }
    Ok(summary)
}

struct CliMockController {
    id: ControllerId,
    emit_refit_artifact: bool,
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
                if !key.contains(task.node_plan.node_id.as_str()) {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received replay artifact handle for another node `{key}`",
                        task.node_plan.node_id
                    )));
                }
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
        let artifacts = if self.emit_refit_artifact
            && task.phase == Phase::Refit
            && matches!(task.node_plan.kind, dag_ml_core::NodeKind::Model)
        {
            vec![dag_ml_core::ArtifactRef {
                id: ArtifactId::new(format!("artifact:{}:refit", task.node_plan.node_id))?,
                kind: "mock_model".to_string(),
                controller_id: self.id.clone(),
                size_bytes: Some(128),
            }]
        } else {
            Vec::new()
        };
        let artifact_handles = artifacts
            .iter()
            .map(|artifact| {
                (
                    artifact.id.clone(),
                    HandleRef {
                        handle: stable_handle(artifact.id.as_str()),
                        kind: HandleKind::Model,
                        owner_controller: self.id.clone(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([("out".to_string(), output)]),
            predictions,
            shape_deltas: Vec::new(),
            artifacts: artifacts.clone(),
            artifact_handles,
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
                artifact_refs: artifacts,
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
    mock_runtime_controllers_with_options(plan, false)
}

fn mock_runtime_controllers_with_refit_artifacts(
    plan: &dag_ml_core::ExecutionPlan,
) -> Result<RuntimeControllerRegistry> {
    mock_runtime_controllers_with_options(plan, true)
}

fn mock_runtime_controllers_with_options(
    plan: &dag_ml_core::ExecutionPlan,
    emit_refit_artifacts: bool,
) -> Result<RuntimeControllerRegistry> {
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        registry.register(Box::new(CliMockController {
            id: controller_id.clone(),
            emit_refit_artifact: emit_refit_artifacts,
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

fn parse_phase(value: &str) -> Result<Phase> {
    match value {
        "COMPILE" => Ok(Phase::Compile),
        "PLAN" => Ok(Phase::Plan),
        "FIT_CV" => Ok(Phase::FitCv),
        "SELECT" => Ok(Phase::Select),
        "REFIT" => Ok(Phase::Refit),
        "PREDICT" => Ok(Phase::Predict),
        "EXPLAIN" => Ok(Phase::Explain),
        _ => bail!("unsupported phase `{value}`"),
    }
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

fn data_provider_for_training_envelope(
    plan: &dag_ml_core::ExecutionPlan,
    envelope: ExternalDataPlanEnvelope,
) -> Result<InMemoryDataProvider> {
    plan.campaign
        .validate_data_envelope_relations(&envelope)
        .with_context(|| "training data envelope relations violate campaign folds")?;
    InMemoryDataProvider::with_envelope(ControllerId::new("controller:data.provider")?, envelope)
        .with_context(|| "failed to register training data envelope")
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

fn read_selection_decisions(path: Option<&PathBuf>) -> Result<BTreeMap<String, SelectionDecision>> {
    let Some(path) = path else {
        return Ok(BTreeMap::new());
    };
    let decisions: BTreeMap<String, SelectionDecision> = read_json(path, "selection decisions")?;
    for (key, decision) in &decisions {
        if key.trim().is_empty() {
            bail!("selection decision map contains an empty key");
        }
        decision
            .validate()
            .with_context(|| format!("invalid selection decision `{key}`"))?;
    }
    Ok(decisions)
}

fn read_optional_prediction_cache_payload(
    path: Option<&PathBuf>,
) -> Result<Option<BundlePredictionCachePayloadSet>> {
    path.map(|path| read_json(path, "prediction cache payload set"))
        .transpose()
}

enum CliPredictionCacheStore {
    Columnar(ColumnarPredictionCacheStore),
    File(FilePredictionCacheStore),
}

impl CliPredictionCacheStore {
    fn materialization_record_count(&self) -> usize {
        match self {
            Self::Columnar(store) => store.materialization_records().len(),
            Self::File(store) => store.materialization_records().len(),
        }
    }
}

impl RuntimePredictionCacheStore for CliPredictionCacheStore {
    fn load_blocks(&self, requirement_key: &str) -> dag_ml_core::Result<Vec<PredictionBlock>> {
        match self {
            Self::Columnar(store) => store.load_blocks(requirement_key),
            Self::File(store) => store.load_blocks(requirement_key),
        }
    }

    fn materialize(
        &self,
        request: &dag_ml_core::PredictionCacheMaterializationRequest,
    ) -> dag_ml_core::Result<HandleRef> {
        match self {
            Self::Columnar(store) => store.materialize(request),
            Self::File(store) => store.materialize(request),
        }
    }
}

fn optional_prediction_cache_store(
    bundle: &ExecutionBundle,
    payloads: Option<BundlePredictionCachePayloadSet>,
    file_store_dir: Option<&PathBuf>,
) -> Result<Option<CliPredictionCacheStore>> {
    if payloads.is_some() && file_store_dir.is_some() {
        bail!("use either --prediction-cache-payload or --prediction-cache-store, not both");
    }
    if let Some(payloads) = payloads {
        return ColumnarPredictionCacheStore::from_payloads(bundle, payloads)
            .map(CliPredictionCacheStore::Columnar)
            .map(Some)
            .with_context(|| "prediction cache payload set does not match bundle");
    }
    file_store_dir
        .map(|store_dir| validate_file_prediction_cache_store(bundle, store_dir))
        .transpose()
        .map(|store| store.map(CliPredictionCacheStore::File))
}

fn validate_file_prediction_cache_store(
    bundle: &ExecutionBundle,
    store_dir: &Path,
) -> Result<FilePredictionCacheStore> {
    let store =
        FilePredictionCacheStore::open(store_dir.to_path_buf(), bundle).with_context(|| {
            format!(
                "prediction cache store is invalid at {}",
                store_dir.display()
            )
        })?;
    for entry in &store.manifest().caches {
        store.load_blocks(&entry.requirement_key).with_context(|| {
            format!(
                "prediction cache store cannot load `{}`",
                entry.requirement_key
            )
        })?;
    }
    Ok(store)
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
